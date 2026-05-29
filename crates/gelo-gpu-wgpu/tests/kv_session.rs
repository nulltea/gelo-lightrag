//! Phase-2 resident-K/V session API on `WgpuVulkanEngine`
//! (perm-attn-gpu-offload). Exercises create → append → attend →
//! refresh → drop through the `GpuOffloadEngine` trait, and checks that
//! `kv_attend` over a [prefix + appended row] cache matches the direct
//! `fused_attention_batched` over the equivalent full K/V (fp16 floor).
//!
//! Requires a Vulkan device (skips cleanly if none is available).

use ndarray::{Array3, Axis, concatenate, s};

use gelo_gpu_wgpu::WgpuVulkanEngine;
use gelo_protocol::GpuOffloadEngine;
use gelo_protocol::attention::{attention_partial, merge_attention_partials};

const H: usize = 8; // kv heads
const D: usize = 128; // head_dim

fn fill(h: usize, n: usize, d: usize, seed: usize) -> Array3<f32> {
    Array3::from_shape_fn((h, n, d), |(a, b, c)| {
        (((a * 7 + b * 13 + c * 3 + seed * 101) % 23) as f32) * 0.01 - 0.11
    })
}

fn max_abs_diff(a: &Array3<f32>, b: &Array3<f32>) -> f32 {
    (a - b).iter().fold(0.0_f32, |m, x| m.max(x.abs()))
}

#[test]
fn kv_session_attend_matches_direct_fused() {
    let engine = match WgpuVulkanEngine::new_fp16() {
        Ok(e) => e,
        Err(_) => {
            eprintln!("[skip] no Vulkan fp16 device");
            return;
        }
    };
    let scale = 1.0_f32 / (D as f32).sqrt();
    let n_prefix = 15;
    let capacity = 32;

    let k_prefix = fill(H, n_prefix, D, 1);
    let v_prefix = fill(H, n_prefix, D, 2);
    let k_row = fill(H, 1, D, 3);
    let v_row = fill(H, 1, D, 4);
    let q = fill(H, 1, D, 5);

    // Session path: create(prefix) → append(row) → attend(q).
    let id = engine
        .kv_create_session(k_prefix.view(), v_prefix.view(), capacity)
        .expect("create_session");
    engine
        .kv_append(id, k_row.view(), v_row.view())
        .expect("append");
    let out_session = engine.kv_attend(id, q.view(), scale).expect("attend");

    // Reference: the same attention over the equivalent full K/V.
    let k_full = concatenate(Axis(1), &[k_prefix.view(), k_row.view()]).unwrap();
    let v_full = concatenate(Axis(1), &[v_prefix.view(), v_row.view()]).unwrap();
    let out_ref = engine
        .fused_attention_batched(q.view(), k_full.view(), v_full.view(), scale, None)
        .expect("fused_attention_batched");

    let drift = max_abs_diff(&out_session, &out_ref);
    assert!(
        drift < 5e-2,
        "kv_attend must match direct fused attention (fp16): drift={drift}",
    );

    // refresh_block: swap the resident cache for a fresh prefix; attend
    // must then match the new K/V.
    let k2 = fill(H, 20, D, 6);
    let v2 = fill(H, 20, D, 7);
    engine.kv_refresh_block(id, k2.view(), v2.view()).expect("refresh");
    let out_after = engine.kv_attend(id, q.view(), scale).expect("attend after refresh");
    let out_ref2 = engine
        .fused_attention_batched(q.view(), k2.view(), v2.view(), scale, None)
        .expect("fused ref2");
    let drift2 = max_abs_diff(&out_after, &out_ref2);
    assert!(drift2 < 5e-2, "kv_attend after refresh must match: drift={drift2}");

    // drop frees the session; attend on a dropped id errors.
    engine.kv_drop_session(id).expect("drop");
    assert!(engine.kv_attend(id, q.view(), scale).is_err(), "attend on dropped session must error");
}

fn expand_interleave(k: &Array3<f32>, group: usize) -> Array3<f32> {
    let (h_kv, n, d) = k.dim();
    let mut out = Array3::<f32>::zeros((h_kv * group, n, d));
    for qh in 0..h_kv * group {
        out.index_axis_mut(Axis(0), qh)
            .assign(&k.index_axis(Axis(0), qh / group));
    }
    out
}

#[test]
fn kv_session_gqa_broadcast_matches_expanded() {
    // Store K/V UN-REPLICATED (8 kv heads) but attend with 32 q heads.
    // The session's on-device GQA broadcast must match a manually
    // GQA-expanded (32-head) reference through the direct fused path.
    let engine = match WgpuVulkanEngine::new_fp16() {
        Ok(e) => e,
        Err(_) => {
            eprintln!("[skip] no Vulkan fp16 device");
            return;
        }
    };
    let scale = 1.0_f32 / (D as f32).sqrt();
    let group = 4;
    let h_q = H * group; // 32
    let n_kv = 20;

    let k_unrep = fill(H, n_kv, D, 1); // (8, 20, 128) — un-replicated
    let v_unrep = fill(H, n_kv, D, 2);
    let q = fill(h_q, 1, D, 3); // (32, 1, 128)

    let id = engine
        .kv_create_session(k_unrep.view(), v_unrep.view(), n_kv + 4)
        .expect("create");
    let out = engine.kv_attend(id, q.view(), scale).expect("attend"); // broadcasts 8→32

    let k_exp = expand_interleave(&k_unrep, group); // (32, 20, 128)
    let v_exp = expand_interleave(&v_unrep, group);
    let out_ref = engine
        .fused_attention_batched(q.view(), k_exp.view(), v_exp.view(), scale, None)
        .expect("fused expanded");

    let drift = max_abs_diff(&out, &out_ref);
    assert!(
        drift < 5e-2,
        "un-replicated GQA broadcast must match expanded reference: drift={drift}",
    );
    engine.kv_drop_session(id).ok();
}

fn normalize_partial(acc: &Array3<f32>, l: &Array3<f32>) -> Array3<f32> {
    let (h, nq, d) = acc.dim();
    let mut out = Array3::<f32>::zeros((h, nq, d));
    for hi in 0..h {
        for i in 0..nq {
            let inv = 1.0 / l[(hi, i, 0)];
            for c in 0..d {
                out[(hi, i, c)] = acc[(hi, i, c)] * inv;
            }
        }
    }
    out
}

#[test]
fn kv_session_partial_normalized_matches_fused() {
    // Phase-3 partial-stats: attend_session_partial returns (acc,m,l);
    // acc/l (normalised) must match the full fused attention.
    let engine = match WgpuVulkanEngine::new_fp16() {
        Ok(e) => e,
        Err(_) => {
            eprintln!("[skip] no Vulkan fp16 device");
            return;
        }
    };
    let scale = 1.0_f32 / (D as f32).sqrt();
    let n_kv = 40;
    let k = fill(H, n_kv, D, 1);
    let v = fill(H, n_kv, D, 2);
    let q = fill(H, 1, D, 3);
    let session = engine.create_kv_session(k.view(), v.view(), n_kv).expect("create");
    let (acc, _m, l) = engine
        .attend_session_partial(q.view(), &session, scale)
        .expect("partial");
    let out = normalize_partial(&acc, &l);
    let out_ref = engine
        .fused_attention_batched(q.view(), k.view(), v.view(), scale, None)
        .expect("fused");
    let drift = max_abs_diff(&out, &out_ref);
    assert!(drift < 5e-2, "partial-stats acc/l must match fused: drift={drift}");
}

#[test]
fn kv_session_partial_tail_merge_matches_full() {
    // The full tail-in-TEE step: GPU partial over the prefix + in-TEE
    // partial over the tail + merge must equal attention over the full
    // K/V (the write-channel-closing decode path).
    let engine = match WgpuVulkanEngine::new_fp16() {
        Ok(e) => e,
        Err(_) => {
            eprintln!("[skip] no Vulkan fp16 device");
            return;
        }
    };
    let scale = 1.0_f32 / (D as f32).sqrt();
    let (n_kv, n_tail) = (40usize, 8usize);
    let n_prefix = n_kv - n_tail;
    let k = fill(H, n_kv, D, 1);
    let v = fill(H, n_kv, D, 2);
    let q = fill(H, 1, D, 3);
    let k_prefix = k.slice(s![.., 0..n_prefix, ..]).to_owned();
    let v_prefix = v.slice(s![.., 0..n_prefix, ..]).to_owned();
    let k_tail = k.slice(s![.., n_prefix..n_kv, ..]).to_owned();
    let v_tail = v.slice(s![.., n_prefix..n_kv, ..]).to_owned();

    let session = engine
        .create_kv_session(k_prefix.view(), v_prefix.view(), n_kv)
        .expect("create");
    let (acc_g, m_g, l_g) = engine
        .attend_session_partial(q.view(), &session, scale)
        .expect("gpu partial");
    let (acc_t, m_t, l_t) = attention_partial(q.view(), k_tail.view(), v_tail.view(), scale);
    let out = merge_attention_partials(
        acc_g.view(), m_g.view(), l_g.view(),
        acc_t.view(), m_t.view(), l_t.view(),
    );
    let out_ref = engine
        .fused_attention_batched(q.view(), k.view(), v.view(), scale, None)
        .expect("fused full");
    let drift = max_abs_diff(&out, &out_ref);
    assert!(drift < 5e-2, "prefix(GPU)+tail(TEE) merge must match full: drift={drift}");
}

#[test]
fn kv_session_unsupported_on_f32_engine() {
    // The session methods require the fp16 engine (resident bf16 cache).
    let engine = match WgpuVulkanEngine::new() {
        Ok(e) => e,
        Err(_) => return,
    };
    let k = fill(H, 4, D, 1);
    let v = fill(H, 4, D, 2);
    assert!(engine.kv_create_session(k.view(), v.view(), 8).is_err());
}

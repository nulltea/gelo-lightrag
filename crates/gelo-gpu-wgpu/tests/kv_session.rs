//! Phase-2 resident-K/V session API on `WgpuVulkanEngine`
//! (perm-attn-gpu-offload). Exercises create → append → attend →
//! refresh → drop through the `GpuOffloadEngine` trait, and checks that
//! `kv_attend` over a [prefix + appended row] cache matches the direct
//! `fused_attention_batched` over the equivalent full K/V (fp16 floor).
//!
//! Requires a Vulkan device (skips cleanly if none is available).

use ndarray::{Array3, Axis, concatenate};

use gelo_gpu_wgpu::WgpuVulkanEngine;
use gelo_protocol::GpuOffloadEngine;

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

//! Microbench: Amulet-style permutation-shielded attention paths at
//! decode shapes against the in-TEE baseline.
//!
//! ## Single-sequence variants (`amulet_attention/n_kv_*`)
//!
//! Three variants compared per shape:
//!
//! 1. `in_tee` — `causal_gqa_attention_cached` running entirely in the
//!    TEE on f32. This is the current production path
//!    (`tee:attn_cached` profile bucket); the baseline we're trying to
//!    beat.
//! 2. `perm_softmax_tee` — `causal_gqa_attention_permuted_cached` with
//!    `PermAttnConfig::HIDDEN_NO_MORE`. Q·Kᵀ and ·V go to the GPU;
//!    softmax stays in TEE. Two GPU dispatches per call.
//! 3. `perm_softmax_gpu` — same wrapper, `HIDDEN_NO_MORE_DECODE_GPU`.
//!    Softmax also goes to GPU (Phase 1b code path). Three GPU
//!    dispatches per call today; will collapse to one once
//!    `WgpuVulkanEngine` overrides `fused_attention_batched`.
//!
//! Decode shapes only — n_q = 1, n_kv ∈ {256, 1000, 2000}. This is the
//! hot regime: 18 468 calls per E2E bench, ~89 s of wall on the
//! baseline.
//!
//! ## M1.12 R1.4 spike (`amulet_attention_r1_4/n_kv_*`)
//!
//! Phase A go/no-go: is `engine.fused_attention_batched` at
//! `(B=8·num_heads, n_q=1, n_kv)` shape faster than today's
//! rayon-over-B in-TEE attention loop in `decoder_block_cached_batched`?
//! Three variants per shape at Qwen3-4B GQA layout
//! (`num_heads=32, num_kv_heads=8, head_dim=128`):
//!
//! - `in_tee_rayon_b8` — Production baseline. 8 rayon-parallel calls
//!   to `causal_gqa_attention_cached` (mirrors
//!   `tee:attn_cached_inplace_many` at B=8).
//! - `gpu_batched_b8_no_mask` — Pure GPU kernel cost. Stack 8 sequences
//!   into one `(B·H=256, 1, d_head)` Q tensor and call
//!   `engine.fused_attention_batched(.., None)` once. Skips permutation
//!   + noise (Q6/§3.1 of the plan) to isolate kernel cost.
//! - `gpu_batched_b8_with_mask` — Same as above with a `(B, 1, n_kv)`
//!   right-padding mask (all-zeros in this bench; the mask kernel
//!   dispatch fires regardless). Measures the marginal cost of the
//!   `+ mask` GPU kernel dispatch — informs whether
//!   burn-cubecl-fusion folds it into the softmax (Q11).
//!
//! Phase A go/no-go gate: `gpu_batched_b8_no_mask` ≥ 1.5× faster than
//! `in_tee_rayon_b8` at n_kv = 2048 → commit to Phase B engineering.
//! ≤ 1.0× → abort R1.4. See `docs/plans/m1-12-permuted-attention-batched-decode.md`.
//!
//! Run with `cargo bench -p gelo-gpu-wgpu --bench amulet_attention`.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use gelo_embedder::decoder::attention::{
    causal_gqa_attention_cached, causal_gqa_attention_permuted_cached,
};
use gelo_gpu_wgpu::WgpuVulkanEngine;
use gelo_protocol::{GpuOffloadEngine, InProcessTrustedExecutor, PermAttnConfig, rng::MaskSeed};
use ndarray::{Array2, Array3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::hint::black_box;

// Single-sequence cell uses Qwen3-1.7B-ish GQA (16/8/128).
const N_Q_HEADS: usize = 16;
const N_KV_HEADS: usize = 8;
const D_HEAD: usize = 128;

// R1.4 spike cell uses Qwen3-4B GQA layout (32/8/128) at B=8.
const N_Q_HEADS_4B: usize = 32;
const N_KV_HEADS_4B: usize = 8;
const D_HEAD_4B: usize = 128;
const BATCH_SIZE_R1_4: usize = 8;
const GROUP_4B: usize = N_Q_HEADS_4B / N_KV_HEADS_4B; // 4

// burn-cubecl is lazy until `.into_data()` syncs; the
// `WgpuVulkanEngine::fused_attention_batched` override already calls
// `.into_data()` to return an `Array3<f32>`, so each criterion
// iteration completes the GPU work it kicked off. No explicit
// `Backend::sync` needed at the bench boundary.

fn make_qkv(n_q: usize, n_kv: usize) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let mut rng = ChaCha20Rng::seed_from_u64(0xCAFE_F00D);
    let q = Array2::from_shape_fn((n_q, N_Q_HEADS * D_HEAD), |_| {
        rng.random::<f32>() * 0.1 - 0.05
    });
    let k = Array2::from_shape_fn((n_kv, N_KV_HEADS * D_HEAD), |_| {
        rng.random::<f32>() * 0.1 - 0.05
    });
    let v = Array2::from_shape_fn((n_kv, N_KV_HEADS * D_HEAD), |_| {
        rng.random::<f32>() * 0.1 - 0.05
    });
    (q, k, v)
}

fn bench(c: &mut Criterion) {
    // Decode shapes — the hot path. n_q = 1, growing K cache.
    let shapes: &[(&str, usize, usize)] = &[
        ("n_kv_256", 1, 256),
        ("n_kv_1000", 1, 1000),
        ("n_kv_2000", 1, 2000),
    ];

    // Initialise the engine once and clone-share across executors.
    // `new_fp16` matches the production embedder path.
    let base_engine = WgpuVulkanEngine::new_fp16()
        .expect("WgpuVulkanEngine::new_fp16 failed — bench requires a Vulkan device");

    for &(label, n_q, n_kv) in shapes {
        let q_pos_offset = n_kv - n_q;
        let (q, k, v) = make_qkv(n_q, n_kv);
        let mut group = c.benchmark_group(format!("amulet_attention/{label}"));
        // Three samples interleaved per shape so any thermal drift hits
        // them uniformly.
        group.sample_size(20);
        group.warm_up_time(std::time::Duration::from_millis(500));
        group.measurement_time(std::time::Duration::from_secs(3));

        group.bench_with_input(
            BenchmarkId::new("in_tee", n_kv),
            &n_kv,
            |bencher, _| {
                bencher.iter(|| {
                    let out = causal_gqa_attention_cached(
                        black_box(q.view()),
                        black_box(k.view()),
                        black_box(v.view()),
                        N_Q_HEADS,
                        N_KV_HEADS,
                        D_HEAD,
                        q_pos_offset,
                    );
                    black_box(out);
                });
            },
        );

        // Variant 2: permuted, softmax stays in TEE.
        {
            let engine = base_engine.clone_shared();
            let mut exec = InProcessTrustedExecutor::with_seed(engine, MaskSeed([7u8; 32]))
                .with_perm_attention(PermAttnConfig::HIDDEN_NO_MORE);
            // Trivial begin/end bracket per-call is too costly to
            // include in steady-state per-call measurement; use the
            // wrapper which doesn't require it (permuted attention
            // path doesn't use the per-forward mask state).
            group.bench_with_input(
                BenchmarkId::new("perm_softmax_tee", n_kv),
                &n_kv,
                |bencher, _| {
                    bencher.iter(|| {
                        let out = causal_gqa_attention_permuted_cached(
                            &mut exec,
                            black_box(q.view()),
                            black_box(k.view()),
                            black_box(v.view()),
                            N_Q_HEADS,
                            N_KV_HEADS,
                            D_HEAD,
                            q_pos_offset,
                        )
                        .expect("perm_softmax_tee call");
                        black_box(out);
                    });
                },
            );
        }

        // Variant 3: permuted, softmax on GPU (Phase 1b).
        {
            let engine = base_engine.clone_shared();
            let mut exec = InProcessTrustedExecutor::with_seed(engine, MaskSeed([11u8; 32]))
                .with_perm_attention(PermAttnConfig::HIDDEN_NO_MORE_DECODE_GPU);
            group.bench_with_input(
                BenchmarkId::new("perm_softmax_gpu", n_kv),
                &n_kv,
                |bencher, _| {
                    bencher.iter(|| {
                        let out = causal_gqa_attention_permuted_cached(
                            &mut exec,
                            black_box(q.view()),
                            black_box(k.view()),
                            black_box(v.view()),
                            N_Q_HEADS,
                            N_KV_HEADS,
                            D_HEAD,
                            q_pos_offset,
                        )
                        .expect("perm_softmax_gpu call");
                        black_box(out);
                    });
                },
            );
        }

        group.finish();
    }
}

// ─── M1.12 R1.4 spike (B=8 batched decode attention) ──────────────────

/// Build a contiguous batch of `B` (Q, K, V) decode-shape tensors at
/// Qwen3-4B GQA layout. Each `Q_b` is `(n_q, num_q_heads * head_dim)`;
/// `K_b`, `V_b` are `(n_kv, num_kv_heads * head_dim)`. Activation scale
/// matches the existing helper (uniform in `(-0.05, 0.05)`).
fn make_qkv_b8(
    batch_size: usize,
    n_q: usize,
    n_kv: usize,
) -> (Vec<Array2<f32>>, Vec<Array2<f32>>, Vec<Array2<f32>>) {
    let mut rng = ChaCha20Rng::seed_from_u64(0xB8_F00D_4B);
    let mut qs = Vec::with_capacity(batch_size);
    let mut ks = Vec::with_capacity(batch_size);
    let mut vs = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        let q = Array2::from_shape_fn((n_q, N_Q_HEADS_4B * D_HEAD_4B), |_| {
            rng.random::<f32>() * 0.1 - 0.05
        });
        let k = Array2::from_shape_fn((n_kv, N_KV_HEADS_4B * D_HEAD_4B), |_| {
            rng.random::<f32>() * 0.1 - 0.05
        });
        let v = Array2::from_shape_fn((n_kv, N_KV_HEADS_4B * D_HEAD_4B), |_| {
            rng.random::<f32>() * 0.1 - 0.05
        });
        qs.push(q);
        ks.push(k);
        vs.push(v);
    }
    (qs, ks, vs)
}

/// Stack `B` per-sequence Q/K/V tensors into the engine input shape
/// `(B·num_q_heads, n, d_head)`, performing GQA replication (each
/// kv-head row is broadcast across `group = num_q_heads / num_kv_heads`
/// q-head slots).
///
/// This is the TEE-side reshape that `permuted_attention_cached_batched`
/// will perform in Phase B, minus the permutation + noise. For the
/// Phase A go/no-go we want kernel-only cost; permute/noise add
/// ~30 µs/call per Q6 analysis and don't change the answer to
/// "burn-chain ≥ 1.5× faster than in-TEE-rayon?".
fn stack_for_engine(
    qs: &[Array2<f32>],
    ks: &[Array2<f32>],
    vs: &[Array2<f32>],
) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
    let batch_size = qs.len();
    let n_q = qs[0].nrows();
    let n_kv = ks[0].nrows();

    let q_dim = batch_size * N_Q_HEADS_4B;
    let kv_dim = batch_size * N_Q_HEADS_4B; // GQA-replicated to match Q
    let mut q_stacked = Array3::<f32>::zeros((q_dim, n_q, D_HEAD_4B));
    let mut k_stacked = Array3::<f32>::zeros((kv_dim, n_kv, D_HEAD_4B));
    let mut v_stacked = Array3::<f32>::zeros((kv_dim, n_kv, D_HEAD_4B));

    for b in 0..batch_size {
        let qb = &qs[b];
        let kb = &ks[b];
        let vb = &vs[b];
        for qh in 0..N_Q_HEADS_4B {
            let kvh = qh / GROUP_4B;
            let q_off = qh * D_HEAD_4B;
            let kv_off = kvh * D_HEAD_4B;
            let stacked_idx = b * N_Q_HEADS_4B + qh;
            q_stacked
                .index_axis_mut(ndarray::Axis(0), stacked_idx)
                .assign(&qb.slice(ndarray::s![.., q_off..q_off + D_HEAD_4B]));
            k_stacked
                .index_axis_mut(ndarray::Axis(0), stacked_idx)
                .assign(&kb.slice(ndarray::s![.., kv_off..kv_off + D_HEAD_4B]));
            v_stacked
                .index_axis_mut(ndarray::Axis(0), stacked_idx)
                .assign(&vb.slice(ndarray::s![.., kv_off..kv_off + D_HEAD_4B]));
        }
    }
    (q_stacked, k_stacked, v_stacked)
}

fn r1_4_bench(c: &mut Criterion) {
    let shapes: &[(&str, usize, usize)] = &[
        ("n_kv_256", 1, 256),
        ("n_kv_1024", 1, 1024),
        ("n_kv_2048", 1, 2048),
    ];

    // Initialise the engine once. fp16 matches the production embedder
    // path (Qwen3-4B runs under `new_fp16` in gelo-snp-runner).
    let base_engine = WgpuVulkanEngine::new_fp16().expect(
        "WgpuVulkanEngine::new_fp16 failed — R1.4 spike bench requires a Vulkan device",
    );

    for &(label, n_q, n_kv) in shapes {
        let q_pos_offset = n_kv - n_q;
        let (qs, ks, vs) = make_qkv_b8(BATCH_SIZE_R1_4, n_q, n_kv);

        let mut group = c.benchmark_group(format!("amulet_attention_r1_4/{label}"));
        // Slightly longer measurement window than the single-sequence
        // bench since each iteration is heavier (B=8 amortised).
        group.sample_size(15);
        group.warm_up_time(std::time::Duration::from_millis(800));
        group.measurement_time(std::time::Duration::from_secs(4));

        // ─── Variant 1: in-TEE rayon-over-B (production baseline) ────
        //
        // This mirrors `decoder_block_cached_batched`'s `tee:attn_cached_inplace_many`
        // bucket at B=8: 8 rayon-parallel calls to causal_gqa_attention_cached.
        // The work-stealing scheduler picks up sequences as cores become
        // free; per-sequence cost is the single-sequence `in_tee` cell
        // scaled by min(B, threads).
        group.bench_with_input(
            BenchmarkId::new("in_tee_rayon_b8", n_kv),
            &n_kv,
            |bencher, _| {
                bencher.iter(|| {
                    use ndarray::parallel::prelude::*;
                    let outs: Vec<Array2<f32>> = (0..BATCH_SIZE_R1_4)
                        .into_par_iter()
                        .map(|b| {
                            causal_gqa_attention_cached(
                                black_box(qs[b].view()),
                                black_box(ks[b].view()),
                                black_box(vs[b].view()),
                                N_Q_HEADS_4B,
                                N_KV_HEADS_4B,
                                D_HEAD_4B,
                                q_pos_offset,
                            )
                        })
                        .collect();
                    black_box(outs);
                });
            },
        );

        // ─── Variant 2: GPU batched, no mask ─────────────────────────
        //
        // Stack the B=8 sequences into (B·H=256, n_q, d_head) and call
        // engine.fused_attention_batched(.., None) once. This is the
        // best-case kernel cost — no mask kernel dispatch, no
        // permutation overhead. Phase A's headline measurement: if
        // this isn't ≥ 1.5× faster than `in_tee_rayon_b8`, kernel
        // dispatch overhead is binding and bucket 2 doesn't ship on
        // burn-chain.
        {
            let engine = base_engine.clone_shared();
            let scale = 1.0_f32 / (D_HEAD_4B as f32).sqrt();
            // Stack outside the iteration loop — bench measures the
            // engine call, not the stacking. Phase B will fold stacking
            // into `permuted_attention_cached_batched`; its cost is
            // ~30 µs/call per Q6 analysis (negligible).
            let (q_st, k_st, v_st) = stack_for_engine(&qs, &ks, &vs);

            group.bench_with_input(
                BenchmarkId::new("gpu_batched_b8_no_mask", n_kv),
                &n_kv,
                |bencher, _| {
                    bencher.iter(|| {
                        let out = engine
                            .fused_attention_batched(
                                black_box(q_st.view()),
                                black_box(k_st.view()),
                                black_box(v_st.view()),
                                scale,
                                None,
                            )
                            .expect("fused_attention_batched no_mask call");
                        // Force GPU completion before next iteration —
                        // burn-cubecl returns once .into_data() syncs,
                        // which our trait impl already does. Black-box
                        // the output to prevent dead-code elimination.
                        black_box(out);
                    });
                },
            );
        }

        // ─── Variant 3: GPU batched, with soft -30 right-padding mask ─
        //
        // Same as Variant 2 plus a `(B·H, n_q, n_kv)` additive mask
        // tensor (all zeros — no actual padding in this bench, but the
        // mask kernel dispatch fires the same way it would in
        // production). Delta vs Variant 2 = cost of the `+ mask`
        // burn-cubecl elementwise add kernel dispatch. Informs Q11:
        // if delta is ≤ 5 % of variant 2, burn-cubecl-fusion is
        // folding the mask add into adjacent kernels; if it's
        // ≥ 20 %, the kernel is materializing as a separate dispatch
        // and we file the custom WGSL FlashAttention-D follow-up.
        {
            let engine = base_engine.clone_shared();
            let scale = 1.0_f32 / (D_HEAD_4B as f32).sqrt();
            let (q_st, k_st, v_st) = stack_for_engine(&qs, &ks, &vs);
            // Mask shape matches the engine's expectation: (B·H, n_q, n_kv).
            // At decode m=1 + no padding it's all zeros. The engine
            // still uploads + dispatches the add — measurement target.
            let mask = Array3::<f32>::zeros((q_st.shape()[0], n_q, n_kv));

            group.bench_with_input(
                BenchmarkId::new("gpu_batched_b8_with_mask", n_kv),
                &n_kv,
                |bencher, _| {
                    bencher.iter(|| {
                        let out = engine
                            .fused_attention_batched(
                                black_box(q_st.view()),
                                black_box(k_st.view()),
                                black_box(v_st.view()),
                                scale,
                                Some(black_box(mask.view())),
                            )
                            .expect("fused_attention_batched with_mask call");
                        black_box(out);
                    });
                },
            );
        }

        group.finish();
    }
}

criterion_group!(benches, bench, r1_4_bench);
criterion_main!(benches);

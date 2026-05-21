//! Microbench: Amulet-style permutation-shielded attention paths at
//! decode shapes against the in-TEE baseline.
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
//! Run with `cargo bench -p gelo-gpu-wgpu --bench amulet_attention`.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use gelo_embedder::decoder::attention::{
    causal_gqa_attention_cached, causal_gqa_attention_permuted_cached,
};
use gelo_gpu_wgpu::WgpuVulkanEngine;
use gelo_protocol::{InProcessTrustedExecutor, PermAttnConfig, rng::MaskSeed};
use ndarray::Array2;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::hint::black_box;

// Qwen3-4B-ish GQA layout (simplified for the bench — exact head counts
// affect absolute numbers but not the per-dispatch cost characteristic
// we want to measure).
const N_Q_HEADS: usize = 16;
const N_KV_HEADS: usize = 8;
const D_HEAD: usize = 128;

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

criterion_group!(benches, bench);
criterion_main!(benches);

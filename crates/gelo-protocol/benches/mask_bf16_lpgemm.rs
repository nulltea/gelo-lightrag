//! Microbench: bf16 mask GEMM via AOCL LPGEMM vs f32 BLIS at
//! production GELO mask shapes.
//!
//! Two variants compared per shape:
//!
//! 1. `f32_blis_apply` / `f32_blis_unapply` — the existing
//!    `GeloMask::{apply, unapply}` path running through AOCL-BLIS
//!    `cblas_sgemm`. This is today's production cost for the
//!    `gelo:mask_apply` / `gelo:mask_unapply` profile buckets.
//! 2. `bf16_lpgemm_apply` / `bf16_lpgemm_unapply` — the M1.12
//!    bucket-3a path through AOCL LPGEMM `aocl_gemm_bf16bf16f32of32`
//!    (AVX-512_BF16 vdpbf16ps). Includes the per-call f32→bf16
//!    downcast of the input (hidden / masked_output); the mask `A`
//!    is pre-cached as bf16 at construction.
//!
//! Shapes mirror Qwen3-4B production sizes for the M1.12 bucket-3a
//! gate (`docs/plans/m1-12-bf16-activation-pipeline.md` §1.1):
//!
//! - **prefill_qkv** (s=2056, d=2560) — n=2048 + k=8 shield, mask
//!   GEMM against the hidden_size width for the QKV / O projections.
//! - **prefill_ffn_down** (s=2056, d=2560) — same `s` but the FFN
//!   down projection input is the intermediate_size (Qwen3-4B
//!   intermediate=9728, but the mask GEMM uses the hidden width on
//!   the output side; we bench the load-bearing s×d shape).
//! - **decode** (s=16, d=2560) — n=1 + k=15 shield, the M1.11
//!   shape-adaptive decode path.
//!
//! Acceptance gate from the plan §1.1: bf16 LPGEMM should beat
//! f32 BLIS at the prefill shape by enough to deliver ≥ 20 %
//! prefill-wall reduction once integrated. The mask buckets are
//! 39 % of prefill wall today → bf16 needs to be ≥ ~50 % faster
//! per call at the prefill shape.
//!
//! Run with: `cargo bench -p gelo-protocol --features blas --bench mask_bf16_lpgemm`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use gelo_protocol::mask::{GeloMask, MaskSeed};
use ndarray::Array2;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::hint::black_box;

/// Build a random `(rows, cols)` f32 matrix in the activation
/// magnitude band (~uniform[-0.05, 0.05]) that the in-TEE post-
/// RMSNorm activations land in for Qwen3-4B. Avoids the bf16
/// denormal range and stays away from the upper exponent where
/// rounding dominates.
fn rand_f32(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    Array2::from_shape_fn((rows, cols), |_| rng.random::<f32>() * 0.1 - 0.05)
}

fn bench(c: &mut Criterion) {
    let shapes: &[(&str, usize, usize)] = &[
        // Decode: small s, full hidden width. The bucket-3a win on
        // decode is bounded by the much smaller s² mask GEMM cost
        // here (~32 KB vs ~16 MB at prefill).
        ("decode", 16, 2560),
        // Prefill QKV / O / gate-up shapes at Qwen3-4B.
        // s = n + shield = 2048 + 8 = 2056, padded to 2056 by the
        // Haar family (no pow2 pad needed); d = hidden_size = 2560.
        // This is the load-bearing shape for the mask bucket
        // measurement (`gelo:mask_apply` 14.9 % + `gelo:mask_unapply`
        // 24.5 % = 39 % of prefill wall in the post-R3 baseline).
        ("prefill_2056_2560", 2056, 2560),
    ];

    for &(label, s, d) in shapes {
        let h = rand_f32(s, d, 0xCAFE_F00D ^ s as u64);
        let mask_f32 = GeloMask::from_seed(s, MaskSeed::from_bytes([7u8; 32]));
        #[cfg(feature = "blas")]
        let mask_bf16 = GeloMask::from_seed_bf16(s, MaskSeed::from_bytes([7u8; 32]));

        let mut group = c.benchmark_group(format!("mask_bf16_lpgemm/{label}"));
        // Throughput: bytes touched by one apply call = (s² + s·d) ×
        // 4 bytes (f32) or × 2 bytes (bf16). For headline numbers we
        // report element count.
        group.throughput(Throughput::Elements((s * d) as u64));
        // Long warm-up + measurement at prefill shape — each call is
        // ~tens-of-ms; we want stable steady-state numbers.
        if s >= 1024 {
            group.sample_size(20);
            group.warm_up_time(std::time::Duration::from_secs(2));
            group.measurement_time(std::time::Duration::from_secs(10));
        } else {
            group.sample_size(30);
            group.warm_up_time(std::time::Duration::from_millis(500));
            group.measurement_time(std::time::Duration::from_secs(3));
        }

        // f32 BLIS apply baseline
        group.bench_with_input(BenchmarkId::new("f32_blis_apply", s), &s, |b, _| {
            b.iter(|| {
                let out = mask_f32.apply(black_box(h.view()));
                black_box(out);
            });
        });

        // f32 BLIS unapply baseline
        let masked_f32 = mask_f32.apply(h.view());
        group.bench_with_input(BenchmarkId::new("f32_blis_unapply", s), &s, |b, _| {
            b.iter(|| {
                let out = mask_f32.unapply(black_box(masked_f32.view()));
                black_box(out);
            });
        });

        // bf16 LPGEMM apply (bucket 3a)
        #[cfg(feature = "blas")]
        {
            group.bench_with_input(BenchmarkId::new("bf16_lpgemm_apply", s), &s, |b, _| {
                b.iter(|| {
                    let out = mask_bf16.apply(black_box(h.view()));
                    black_box(out);
                });
            });

            // bf16 LPGEMM unapply — use the bf16-path's own masked
            // output so the unapply walks bf16-precision input the
            // way the production pipeline would.
            let masked_bf16 = mask_bf16.apply(h.view());
            group.bench_with_input(
                BenchmarkId::new("bf16_lpgemm_unapply", s),
                &s,
                |b, _| {
                    b.iter(|| {
                        let out = mask_bf16.unapply(black_box(masked_bf16.view()));
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

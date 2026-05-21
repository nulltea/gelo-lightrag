//! Microbench: shield-row Gaussian fill at decode and prefill shapes.
//!
//! Compares the prior per-element `rand_distr::StandardNormal::sample`
//! loop (`legacy_fill_scalar`) against the new `gaussian::fill_gaussian`
//! bulk-RNG + SIMD path.
//!
//! Shapes match the v7 baseline (handoff 2026-05-21):
//! - decode: k = 15 shield rows × d = 2560 hidden width
//! - prefill: k = 8 shield rows × d = 2560
//!
//! Run with: `cargo bench -p gelo-protocol --bench shield_gaussian`.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use gelo_protocol::gaussian::fill_gaussian;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::hint::black_box;

/// The pre-optimisation reference loop.  Mirrors the body of the
/// previous `fill_shield_rows_inline` (sim.rs:848-861).
fn legacy_fill_scalar<R: RngCore + ?Sized>(dest: &mut [f32], sigma: f32, rng: &mut R) {
    let normal = StandardNormal;
    for v in dest.iter_mut() {
        let z: f32 = normal.sample(rng);
        *v = z * sigma;
    }
}

fn bench(c: &mut Criterion) {
    // Hidden width matches Qwen3-4B (d=2560).  The d=1024 case mirrors
    // Qwen3-Embedding-0.6B for completeness — not the v7 hot path but
    // useful to confirm the SIMD win holds across widths.
    let widths = [2560usize, 1024];
    let shapes: &[(&str, usize)] = &[("decode_k15", 15), ("prefill_k8", 8)];

    for &d in &widths {
        for &(label, k) in shapes {
            let n = k * d;
            let id_legacy = format!("legacy_scalar/d{d}/{label}");
            let id_new = format!("fill_gaussian/d{d}/{label}");
            let mut group = c.benchmark_group("shield_gaussian");
            group.throughput(Throughput::Elements(n as u64));

            group.bench_with_input(
                BenchmarkId::from_parameter(&id_legacy),
                &n,
                |bencher, &n| {
                    let mut rng = ChaCha20Rng::from_seed([0xa3u8; 32]);
                    let mut buf = vec![0.0_f32; n];
                    let sigma: f32 = 0.123;
                    bencher.iter(|| {
                        legacy_fill_scalar(black_box(&mut buf), black_box(sigma), &mut rng);
                        black_box(&buf);
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::from_parameter(&id_new),
                &n,
                |bencher, &n| {
                    let mut rng = ChaCha20Rng::from_seed([0xa3u8; 32]);
                    let mut buf = vec![0.0_f32; n];
                    let sigma: f32 = 0.123;
                    bencher.iter(|| {
                        fill_gaussian(black_box(&mut buf), black_box(sigma), &mut rng);
                        black_box(&buf);
                    });
                },
            );

            // P3: same generator, fast Xoshiro256++ instead of ChaCha20.
            // Attributes the RNG-bulk-fill share of the per-call cost.
            let id_xoshiro = format!("fill_gaussian_xoshiro/d{d}/{label}");
            group.bench_with_input(
                BenchmarkId::from_parameter(&id_xoshiro),
                &n,
                |bencher, &n| {
                    let mut rng = Xoshiro256PlusPlus::from_seed([0xa3u8; 32]);
                    let mut buf = vec![0.0_f32; n];
                    let sigma: f32 = 0.123;
                    bencher.iter(|| {
                        fill_gaussian(black_box(&mut buf), black_box(sigma), &mut rng);
                        black_box(&buf);
                    });
                },
            );

            group.finish();
        }
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);

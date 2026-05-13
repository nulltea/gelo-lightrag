//! Parity + perf test for the fp16 engine variant.
//!
//! Verifies that `WgpuVulkanEngine::new_fp16()` returns matmul outputs
//! within an f16-quantization-aware tolerance of the f32 engine, and
//! measures the per-call wall-clock difference at representative
//! BGE-base shapes.
//!
//! Run: `cargo test -p gelo-gpu-wgpu --release --test fp16_parity -- --ignored --nocapture`

use std::time::Instant;

use gelo_gpu_wgpu::WgpuVulkanEngine;
use gelo_protocol::{GpuOffloadEngine, WeightHandle, WeightKind};
use ndarray::Array2;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

fn make_input(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let normal = StandardNormal;
    Array2::from_shape_fn((rows, cols), |_| normal.sample(&mut rng))
}

/// Asserts that fp16 matmul output stays within an f16-quantization-aware
/// tolerance of the f32 reference. The expected relative error per
/// element is ~ √k · ε_f16 where ε_f16 ≈ 2⁻¹⁰ ≈ 1e-3 — at k=768 that's
/// ~28 · 1e-3 ≈ 3% relative error per accumulated dot product.
#[test]
#[ignore]
fn fp16_matches_f32_within_tolerance() {
    let m = 64;
    let k = 768;
    let n = 768;
    let weight = make_input(k, n, 1);
    let input = make_input(m, k, 2);

    let mut engine_f32 = WgpuVulkanEngine::new().expect("Vulkan adapter");
    let handle = WeightHandle::new(0, WeightKind::Q);
    engine_f32
        .register_weight(handle, weight.view())
        .expect("register f32");
    let out_f32 = engine_f32
        .matmul(handle, input.view())
        .expect("matmul f32");

    let mut engine_f16 = WgpuVulkanEngine::new_fp16().expect("Vulkan adapter (f16)");
    engine_f16
        .register_weight(handle, weight.view())
        .expect("register f16");
    let out_f16 = engine_f16
        .matmul(handle, input.view())
        .expect("matmul f16");

    assert_eq!(out_f32.shape(), out_f16.shape());

    // Tolerance vs the f32 reference. Per-element relative error blows
    // up at near-zero outputs (small denominators), so we measure against
    // the OUTPUT-SCALE: the per-row L2 norm of the f32 reference. For
    // k=768 accumulated dot products from N(0,1) inputs, output L2 is
    // ~√(n · k) ≈ √(64·768) ≈ 222. Per-element f16 rounding error
    // accumulates as ~√k · ε_f16 ≈ 28 · 1e-3 ≈ 0.03 absolute, so a
    // worst-case L2(diff) / L2(ref) of ~1e-3 to 1e-2 is expected.
    let diff = &out_f32 - &out_f16;
    let l2_diff: f32 = diff.iter().map(|x| x * x).sum::<f32>().sqrt();
    let l2_ref: f32 = out_f32.iter().map(|x| x * x).sum::<f32>().sqrt();
    let rel_l2 = l2_diff / l2_ref.max(1e-9);

    let max_abs_err = diff.iter().fold(0.0_f32, |a, &x| a.max(x.abs()));
    eprintln!(
        "fp16 vs f32: ‖Δ‖₂ = {l2_diff:.4}, ‖ref‖₂ = {l2_ref:.4}, \
         rel L2 = {:.4}%, max abs = {max_abs_err:.4}",
        rel_l2 * 100.0
    );
    // 1% L2 relative error is conservative for k=768 f16 accumulation.
    // A 0.5% threshold tests that the kernel isn't running pure-f16
    // accumulation when fp32-accum should be the default; relax if
    // burn-cubecl's f16 path is documented to use f16-accum.
    assert!(
        rel_l2 < 0.01,
        "fp16 L2 relative error {:.3}% exceeds 1% — kernel may be doing fp16 accumulation when fp32 accum was expected, or shader-f16 is missing",
        rel_l2 * 100.0
    );
}

/// Micro-bench: 20 warm matmuls at BGE-base shapes, fp32 vs fp16.
/// Win condition: fp16 ≤ 0.75× the fp32 wall-clock per call (i.e.,
/// ≥1.33× speedup) on AMD/NVIDIA hardware with f16 GEMM acceleration.
#[test]
#[ignore]
fn fp16_speedup_at_bge_shapes() {
    let shapes: Vec<(usize, usize, usize, &str)> = vec![
        (128, 768, 768, "QKV-128"),
        (128, 768, 3072, "FFNup-128"),
        (128, 3072, 768, "FFNdn-128"),
        (256, 768, 3072, "FFNup-256"),
    ];

    eprintln!();
    eprintln!(
        "{:<12} {:>5} {:>5} {:>5} {:>12} {:>12} {:>12}",
        "shape", "M", "K", "N", "f32_us", "f16_us", "f16/f32"
    );
    eprintln!("{}", "-".repeat(72));
    for (m, k, n, label) in shapes {
        let weight = make_input(k, n, 1);

        let mut e32 = WgpuVulkanEngine::new().expect("vulkan");
        let h = WeightHandle::new(0, WeightKind::Q);
        e32.register_weight(h, weight.view()).unwrap();
        let mut e16 = WgpuVulkanEngine::new_fp16().expect("vulkan f16");
        e16.register_weight(h, weight.view()).unwrap();

        // Warm-up to get past autotune.
        let _ = e32.matmul(h, make_input(m, k, 999).view());
        let _ = e16.matmul(h, make_input(m, k, 999).view());

        let iters = 10;
        let mut t32 = Vec::with_capacity(iters);
        let mut t16 = Vec::with_capacity(iters);
        for i in 0..iters {
            let input = make_input(m, k, 100 + i as u64);
            let t0 = Instant::now();
            let _ = e32.matmul(h, input.view()).unwrap();
            t32.push(t0.elapsed().as_micros());
            let t0 = Instant::now();
            let _ = e16.matmul(h, input.view()).unwrap();
            t16.push(t0.elapsed().as_micros());
        }
        t32.sort_unstable();
        t16.sort_unstable();
        let med32 = t32[t32.len() / 2];
        let med16 = t16[t16.len() / 2];
        eprintln!(
            "{:<12} {:>5} {:>5} {:>5} {:>12} {:>12} {:>11.2}×",
            label,
            m,
            k,
            n,
            med32,
            med16,
            med16 as f64 / med32.max(1) as f64
        );
    }
}

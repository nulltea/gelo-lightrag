//! Path β engine parity test for the bf16 activation pipeline
//! (plan `m1-12-bf16-activation-pipeline.md` §4.2 — engine-side
//! one-conversion bf16 → device-precision upload).
//!
//! `WgpuVulkanEngine` overrides `matmul_bf16_input` and
//! `matmul_many_bf16_input` to route bf16 inputs through the existing
//! `array2_bf16_to_tensor_*` helpers — no transient host f32 buffer is
//! materialised. This test verifies the override produces output that
//! matches the f32 path within bf16-floor tolerance.
//!
//! Run: `cargo test -p gelo-gpu-wgpu --release --test path_beta_bf16_input -- --ignored --nocapture`

use gelo_gpu_wgpu::WgpuVulkanEngine;
use gelo_protocol::{GpuOffloadEngine, WeightHandle, WeightKind};
use half::bf16;
use ndarray::Array2;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

fn make_input(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let normal = StandardNormal;
    Array2::from_shape_fn((rows, cols), |_| normal.sample(&mut rng))
}

fn quantise_to_bf16(x: &Array2<f32>) -> Array2<bf16> {
    x.mapv(|v| bf16::from_f32(v))
}

fn max_abs_delta(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// bf16-input override should match the f32 path on `bf16-quantised`
/// inputs within bf16-floor tolerance. Engine internal precision is
/// f16 in both cases — the only difference between the two paths is
/// the upload conversion (`f32 → f16` vs `bf16 → f16`). The two
/// conversions share the same destination precision, so output drift
/// is bounded by the input-side rounding round-trip.
#[test]
#[ignore]
fn matmul_bf16_input_matches_matmul_at_bf16_floor() {
    let m = 64;
    let k = 768;
    let n = 768;
    let weight = make_input(k, n, 1);
    let input_f32 = make_input(m, k, 2);
    let input_bf16 = quantise_to_bf16(&input_f32);
    // f32 reference uses the same bf16-quantised values widened back
    // to f32 so the only delta being measured is the upload conversion
    // (`f32 → f16` vs `bf16 → f16`), not input quantisation noise.
    let input_f32_q = input_bf16.mapv(|v| v.to_f32());

    let mut engine = WgpuVulkanEngine::new_fp16().expect("Vulkan adapter (fp16)");
    let handle = WeightHandle::new(0, WeightKind::Q);
    engine
        .register_weight(handle, weight.view())
        .expect("register weight");

    let out_f32 = engine.matmul(handle, input_f32_q.view()).expect("matmul f32 path");
    let out_bf16 = engine
        .matmul_bf16_input(handle, input_bf16.view())
        .expect("matmul_bf16_input path");

    assert_eq!(out_f32.dim(), out_bf16.dim(), "output shape mismatch");

    let max_abs = max_abs_delta(&out_f32, &out_bf16);
    // Both paths upload to f16; the only difference is whether the
    // upload reads f32 or bf16. Output drift is bounded by the
    // round-trip rounding noise on the input — should be near zero.
    let tol = 1e-2;
    assert!(
        max_abs < tol,
        "matmul_bf16_input vs matmul max abs delta {max_abs} exceeds tolerance {tol}"
    );
    eprintln!("Path β matmul_bf16_input parity: max_abs_delta = {max_abs:.6} (tol {tol})");
}

#[test]
#[ignore]
fn matmul_many_bf16_input_matches_matmul_many_at_bf16_floor() {
    let m = 64;
    let k = 768;
    let n = 768;
    let weights = [
        (WeightHandle::new(0, WeightKind::Q), make_input(k, n, 1)),
        (WeightHandle::new(0, WeightKind::K), make_input(k, n, 2)),
        (WeightHandle::new(0, WeightKind::V), make_input(k, n, 3)),
    ];
    let input_f32 = make_input(m, k, 7);
    let input_bf16 = quantise_to_bf16(&input_f32);
    let input_f32_q = input_bf16.mapv(|v| v.to_f32());

    let mut engine = WgpuVulkanEngine::new_fp16().expect("Vulkan adapter (fp16)");
    for (h, w) in &weights {
        engine.register_weight(*h, w.view()).expect("register weight");
    }
    let handles: Vec<WeightHandle> = weights.iter().map(|(h, _)| *h).collect();

    let out_f32 = engine
        .matmul_many(&handles, input_f32_q.view())
        .expect("matmul_many f32 path");
    let out_bf16 = engine
        .matmul_many_bf16_input(&handles, input_bf16.view())
        .expect("matmul_many_bf16_input path");

    assert_eq!(out_f32.len(), out_bf16.len(), "output count mismatch");
    let tol = 1e-2;
    for (i, (a, b)) in out_f32.iter().zip(out_bf16.iter()).enumerate() {
        assert_eq!(a.dim(), b.dim(), "output {i} shape mismatch");
        let d = max_abs_delta(a, b);
        assert!(
            d < tol,
            "matmul_many_bf16_input vs matmul_many output {i}: max abs delta {d} exceeds tolerance {tol}"
        );
        eprintln!("Path β matmul_many[{i}] parity: max_abs_delta = {d:.6} (tol {tol})");
    }
}

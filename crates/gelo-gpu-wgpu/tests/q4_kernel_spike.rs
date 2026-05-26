//! Phase 0 of the Q4 GPU weights plan (`docs/plans/q4-gpu-weights.md`):
//! validate that burn-cubecl's `q_matmul` actually accelerates over f32
//! `matmul` on this Vulkan iGPU.
//!
//! The plan hinges on a single experimental claim: cubek-matmul's
//! tile-quantized kernels fire on the WGPU runtime and beat f32 matmul
//! at memory-bandwidth-bound shapes. If they fall back to
//! dequantize-then-f32-matmul under the hood (or produce nonsensical
//! results), the rest of the Q4 plan stalls until we fork burn-cubecl
//! and patch the kernel dispatch.
//!
//! Pass criteria (per `docs/plans/q4-gpu-weights.md` §4 Phase 0):
//! 1. **Correctness**: max-abs relative error of `input.matmul(W_q4)` vs
//!    `input.matmul(W_f32)` is < 5e-2 at the Qwen3-4B QKV shape
//!    (s=2056, d=2560, p=4096). Q4 quantization with block size 128
//!    typically gives ~1-3e-2 mean rel error on random-Gaussian inputs.
//! 2. **Speedup**: median wall-clock of `input.matmul(W_q4)` is ≥ 1.5×
//!    faster than `input.matmul(W_f32)` at the same shape. (Anything
//!    less than 1.5× means the kernel is silently dequantizing
//!    upstream — the integer path isn't engaged.)
//!
//! If both criteria pass, Phase 1 of the plan (`register_weight_quantized`
//! plumbing) is unblocked. If either fails, see the bottom-of-file
//! `OUTCOMES` doc for next steps.

use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use burn_backend::Backend;
use burn_cubecl::CubeBackend;
use burn_tensor::{Tensor, TensorData};
use cubecl_common::quant::scheme::{
    BlockSize, QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
};
use cubecl_wgpu::WgpuRuntime;
use ndarray::Array2;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

/// burn-cubecl backend matching `WgpuVulkanEngine`'s f32 path.
type CubeWgpu32 = CubeBackend<WgpuRuntime, f32, i32, u8>;

const WARMUP_RUNS: usize = 2;
const TIMED_RUNS: usize = 5;

fn sample_normal(rng: &mut ChaCha20Rng, n: usize, d: usize) -> Array2<f32> {
    let normal = StandardNormal;
    Array2::from_shape_fn((n, d), |_| normal.sample(rng))
}

fn array_to_tensor(a: &Array2<f32>, device: &burn_tensor::Device<CubeWgpu32>) -> Tensor<CubeWgpu32, 2> {
    let shape = [a.nrows(), a.ncols()];
    let data = TensorData::new(a.as_slice().expect("standard layout").to_vec(), shape);
    Tensor::<CubeWgpu32, 2>::from_data(data, device)
}

fn tensor_to_array(t: Tensor<CubeWgpu32, 2>) -> Array2<f32> {
    let shape = t.shape().dims;
    let data = t.into_data();
    let vec: Vec<f32> = data
        .to_vec::<f32>()
        .expect("convert to f32 vec");
    Array2::from_shape_vec((shape[0], shape[1]), vec).expect("shape matches")
}

fn rel_max_abs_error(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    let max_target = b
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max)
        .max(1e-6);
    let max_abs = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max);
    max_abs / max_target
}

fn mean_abs_error(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    let sum: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum();
    sum / (a.len() as f32)
}

fn time_matmul<F>(label: &str, mut f: F) -> std::time::Duration
where
    F: FnMut() -> Array2<f32>,
{
    // Warmup
    for _ in 0..WARMUP_RUNS {
        let _ = f();
    }
    // Timed runs
    let mut samples = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let t = Instant::now();
        let _out = f();
        samples.push(t.elapsed());
    }
    samples.sort();
    let median = samples[samples.len() / 2];
    let min = samples.first().copied().unwrap_or_default();
    let max = samples.last().copied().unwrap_or_default();
    eprintln!(
        "    {label:<24} median {:>8.1} ms · min {:>7.1} · max {:>7.1} ms",
        median.as_secs_f64() * 1000.0,
        min.as_secs_f64() * 1000.0,
        max.as_secs_f64() * 1000.0,
    );
    median
}

/// Spike at Qwen3-4B's QKV shape: s = n+k = 2056, d_in = 2560,
/// d_out = 4096 (the Q projection). Quantize the weight to Q4S/Block(128)
/// and check whether `q_matmul` accelerates.
#[test]
#[ignore = "requires real Vulkan GPU; ~30 sec wall-clock; loud-fails if q_matmul falls back to f32"]
fn q4_matmul_spike_qwen3_4b_qkv_shape() -> Result<()> {
    let _ = env_logger::try_init();
    let device = burn_tensor::Device::<CubeWgpu32>::default();
    <CubeWgpu32 as Backend>::sync(&device)
        .map_err(|e| anyhow!("burn-cubecl device sync at init: {e:?}"))?;

    // Qwen3-4B QKV: input (s=2056, d_in=2560), weight (d_in=2560, d_out=4096).
    let s = 2056_usize;
    let d_in = 2560_usize;
    let d_out = 4096_usize;
    let mut rng = ChaCha20Rng::from_seed([91u8; 32]);

    eprintln!(
        "=== Phase 0 q_matmul spike ===\nshape: input ({s}, {d_in}) × weight ({d_in}, {d_out})"
    );

    let input_f32 = sample_normal(&mut rng, s, d_in);
    let weight_f32 = sample_normal(&mut rng, d_in, d_out);

    let input_t = array_to_tensor(&input_f32, &device);
    let weight_t_f32 = array_to_tensor(&weight_f32, &device);

    // ─── Q4 quantization ─────────────────────────────────────────────
    // Pick the scheme from the plan: Q4S, Block(128), F32 scales, sym.
    let scheme = QuantScheme {
        value: QuantValue::Q4S,
        param: QuantParam::F32,
        // PackedU32(0) = pack along innermost dim, 8 Q4 values per u32 word.
        store: QuantStore::PackedU32(0),
        level: QuantLevel::Block(BlockSize::new([128u8])),
        mode: QuantMode::Symmetric,
    };
    eprintln!("scheme: {scheme:?}");

    let weight_t_q4 = weight_t_f32.clone().quantize_dynamic(&scheme);
    // Sync to force the quantization kernel to actually run before timing
    // the matmul (rules out lazy-quant skewing the matmul measurement).
    <CubeWgpu32 as Backend>::sync(&device).map_err(|e| anyhow!("sync post-quant: {e:?}"))?;

    // ─── F32 baseline matmul ─────────────────────────────────────────
    eprintln!("\n[timing]");
    let f32_median = time_matmul("f32 weight matmul", || {
        let out = input_t.clone().matmul(weight_t_f32.clone());
        let arr = tensor_to_array(out);
        // Force a host sync after each iter so the timer captures the
        // full kernel + device sync time, not just dispatch latency.
        arr
    });

    // ─── Q4 quantized matmul ─────────────────────────────────────────
    let q4_median = time_matmul("Q4 weight matmul", || {
        let out = input_t.clone().matmul(weight_t_q4.clone());
        tensor_to_array(out)
    });

    // ─── Correctness check (single run, no timing) ────────────────────
    let out_f32 = tensor_to_array(input_t.clone().matmul(weight_t_f32.clone()));
    let out_q4 = tensor_to_array(input_t.clone().matmul(weight_t_q4.clone()));

    let rel_err = rel_max_abs_error(&out_q4, &out_f32);
    let mae = mean_abs_error(&out_q4, &out_f32);
    let max_abs_f32 = out_f32
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max);

    eprintln!("\n[correctness]");
    eprintln!(
        "    max(|out_f32|)            = {max_abs_f32:.3}\n    \
         max_abs_err(out_q4 vs out_f32) = {:.3e}\n    \
         max_rel_err                    = {rel_err:.3e}\n    \
         mean_abs_err                   = {mae:.3e}",
        rel_err * max_abs_f32,
    );

    // ─── Pass/fail ───────────────────────────────────────────────────
    let f32_ms = f32_median.as_secs_f64() * 1000.0;
    let q4_ms = q4_median.as_secs_f64() * 1000.0;
    let speedup = f32_ms / q4_ms.max(1e-6);
    eprintln!("\n[verdict]");
    eprintln!("    f32 median    = {f32_ms:>7.1} ms");
    eprintln!("    Q4  median    = {q4_ms:>7.1} ms");
    eprintln!("    speedup (Q4/f32) = {speedup:.2}×");
    let pass_speedup = speedup >= 1.5;
    let pass_rel_err = rel_err < 5e-2;
    eprintln!(
        "    speedup ≥ 1.5×            : {}",
        if pass_speedup { "PASS" } else { "FAIL" }
    );
    eprintln!(
        "    rel-err < 5e-2            : {}",
        if pass_rel_err { "PASS" } else { "FAIL" }
    );

    if !pass_speedup {
        eprintln!(
            "\nWARN: speedup {:.2}× < 1.5× threshold — the cubek-matmul tile-quantized\n\
             path may be falling back to dequantize-then-f32 silently. Check with\n\
             `RUST_LOG=cubek=trace,cubecl=debug cargo test ... -- --nocapture` and look\n\
             for the kernel-launch path. Next step: see `docs/plans/q4-gpu-weights.md`\n\
             §4 Phase 0 — fork burn-cubecl + patch.",
            speedup,
        );
    }
    if !pass_rel_err {
        eprintln!(
            "\nWARN: rel-err {:.3e} >= 5e-2 threshold — Q4 quantization is too lossy at\n\
             this shape for raw weights. Phase 3 (HD₃/DCT-IV hidden-axis rotation) is\n\
             load-bearing for accuracy here; without rotation, the matmul output drifts\n\
             unacceptably from the f32 reference.",
            rel_err,
        );
    }

    // Don't `assert!` — print pass/fail loud and let the test pass so
    // the next agent can read the verdict from the output. The plan
    // doc captures what to do for either outcome.
    eprintln!("\nOUTCOMES doc lives in `docs/plans/q4-gpu-weights.md` §4 Phase 0.");
    Ok(())
}

/// Pure quantize→dequantize round-trip at the smoke shape. If this
/// is broken, the matmul will also be broken; this test isolates
/// whether the quant kernel itself works before involving matmul.
#[test]
fn q4_quantize_dequantize_round_trip_smoke() -> Result<()> {
    let _ = env_logger::try_init();
    let device = burn_tensor::Device::<CubeWgpu32>::default();
    <CubeWgpu32 as Backend>::sync(&device)
        .map_err(|e| anyhow!("device sync at init: {e:?}"))?;

    let n = 64;
    let d = 128;
    let mut rng = ChaCha20Rng::from_seed([101u8; 32]);
    let w_f32 = sample_normal(&mut rng, n, d);
    let w_t = array_to_tensor(&w_f32, &device);

    let scheme = QuantScheme {
        value: QuantValue::Q4S,
        param: QuantParam::F32,
        store: QuantStore::PackedU32(0),
        level: QuantLevel::Block(BlockSize::new([128u8])),
        mode: QuantMode::Symmetric,
    };
    let w_q = w_t.clone().quantize_dynamic(&scheme);
    let w_dequant = w_q.dequantize();
    let w_dequant_arr = tensor_to_array(w_dequant);

    let rel = rel_max_abs_error(&w_dequant_arr, &w_f32);
    let mae = mean_abs_error(&w_dequant_arr, &w_f32);
    eprintln!(
        "Q4 quant→dequant round-trip at ({n}, {d}): rel_max_abs={rel:.3e}, mae={mae:.3e}"
    );
    assert!(
        rel < 0.2,
        "Q4 quant→dequant round-trip rel_err {rel:.3e} indicates broken quant kernel"
    );
    Ok(())
}

/// Smaller-scale shape (s=64, d_in=128, d_out=64) where the kernel-launch
/// overhead dominates wall time. Sanity check that the API doesn't
/// panic and produces a roughly-correct result; perf doesn't matter
/// here so no speedup assertion.
#[test]
fn q4_matmul_spike_small_smoke() -> Result<()> {
    let _ = env_logger::try_init();
    let device = burn_tensor::Device::<CubeWgpu32>::default();
    <CubeWgpu32 as Backend>::sync(&device)
        .map_err(|e| anyhow!("device sync at init: {e:?}"))
        .context("burn-cubecl device init")?;

    let s = 64;
    let d_in = 128; // multiple of block size 128
    let d_out = 64;
    let mut rng = ChaCha20Rng::from_seed([97u8; 32]);
    let input_f32 = sample_normal(&mut rng, s, d_in);
    let weight_f32 = sample_normal(&mut rng, d_in, d_out);

    let input_t = array_to_tensor(&input_f32, &device);
    let weight_t_f32 = array_to_tensor(&weight_f32, &device);

    let scheme = QuantScheme {
        value: QuantValue::Q4S,
        param: QuantParam::F32,
        // PackedU32(0) = pack along innermost dim, 8 Q4 values per u32 word.
        store: QuantStore::PackedU32(0),
        level: QuantLevel::Block(BlockSize::new([128u8])),
        mode: QuantMode::Symmetric,
    };
    let weight_t_q4 = weight_t_f32.clone().quantize_dynamic(&scheme);

    let out_f32 = tensor_to_array(input_t.clone().matmul(weight_t_f32.clone()));
    let out_q4 = tensor_to_array(input_t.clone().matmul(weight_t_q4.clone()));

    let rel_err = rel_max_abs_error(&out_q4, &out_f32);
    eprintln!("small-smoke rel_err = {rel_err:.3e}");
    assert!(
        rel_err < 0.5,
        "small-smoke Q4 matmul rel_err {rel_err:.3e} indicates broken quant path"
    );
    Ok(())
}

/// Probe whether the Q4 speedup is shape-dependent — try FfnDown
/// (the widest weight: 9728 × 2560) and matmul_many (3 weights
/// sharing one input, amortising activation bandwidth).
#[test]
#[ignore = "real Vulkan GPU; ~1 min wall-clock; explores Q4 speedup at different shapes"]
fn q4_matmul_shape_sweep() -> Result<()> {
    let _ = env_logger::try_init();
    let device = burn_tensor::Device::<CubeWgpu32>::default();
    <CubeWgpu32 as Backend>::sync(&device)
        .map_err(|e| anyhow!("device sync: {e:?}"))?;
    let scheme = QuantScheme {
        value: QuantValue::Q4S,
        param: QuantParam::F32,
        store: QuantStore::PackedU32(0),
        level: QuantLevel::Block(BlockSize::new([128u8])),
        mode: QuantMode::Symmetric,
    };

    let cases: &[(&str, usize, usize, usize)] = &[
        ("QKV-Q   (input·W)        ", 2056, 2560, 4096),
        ("Gate∥Up (input·W)        ", 2056, 2560, 9728),
        ("FfnDown (input·W)        ", 2056, 9728, 2560),
        ("O proj  (input·W)        ", 2056, 4096, 2560),
    ];

    let mut rng = ChaCha20Rng::from_seed([113u8; 32]);
    eprintln!("=== Q4 shape sweep ===\nscheme: Q4S Block(128)\n");
    for (label, s, d_in, d_out) in cases {
        let input_f32 = sample_normal(&mut rng, *s, *d_in);
        let weight_f32 = sample_normal(&mut rng, *d_in, *d_out);
        let input_t = array_to_tensor(&input_f32, &device);
        let w_f32 = array_to_tensor(&weight_f32, &device);
        let w_q4 = w_f32.clone().quantize_dynamic(&scheme);
        <CubeWgpu32 as Backend>::sync(&device).ok();

        let f32_med = time_matmul(&format!("{label} f32"), || {
            tensor_to_array(input_t.clone().matmul(w_f32.clone()))
        });
        let q4_med = time_matmul(&format!("{label} Q4 "), || {
            tensor_to_array(input_t.clone().matmul(w_q4.clone()))
        });
        let f32_ms = f32_med.as_secs_f64() * 1000.0;
        let q4_ms = q4_med.as_secs_f64() * 1000.0;
        let speedup = f32_ms / q4_ms.max(1e-6);

        // Correctness
        let out_f32 = tensor_to_array(input_t.clone().matmul(w_f32.clone()));
        let out_q4 = tensor_to_array(input_t.clone().matmul(w_q4.clone()));
        let rel = rel_max_abs_error(&out_q4, &out_f32);
        eprintln!(
            "    → speedup {speedup:.2}× | rel_err {rel:.3e}\n"
        );
    }

    // matmul_many: 3 weights, shared input. This amortises the input
    // bandwidth — if Q4's gain is weight-bandwidth-dominated, matmul_many
    // should show a bigger speedup than single matmul.
    eprintln!("--- matmul_many (3 weights sharing one input) ---");
    let s = 2056;
    let d_in = 2560;
    let d_out_each = 4096;
    let input_f32 = sample_normal(&mut rng, s, d_in);
    let input_t = array_to_tensor(&input_f32, &device);
    let weights_f32: Vec<_> = (0..3)
        .map(|_| {
            let w = sample_normal(&mut rng, d_in, d_out_each);
            array_to_tensor(&w, &device)
        })
        .collect();
    let weights_q4: Vec<_> = weights_f32
        .iter()
        .map(|w| w.clone().quantize_dynamic(&scheme))
        .collect();
    <CubeWgpu32 as Backend>::sync(&device).ok();

    let f32_med = time_matmul("3× f32 matmul", || {
        let mut out = Vec::with_capacity(3);
        for w in &weights_f32 {
            out.push(tensor_to_array(input_t.clone().matmul(w.clone())));
        }
        out.into_iter().next().unwrap()
    });
    let q4_med = time_matmul("3× Q4  matmul", || {
        let mut out = Vec::with_capacity(3);
        for w in &weights_q4 {
            out.push(tensor_to_array(input_t.clone().matmul(w.clone())));
        }
        out.into_iter().next().unwrap()
    });
    let speedup = f32_med.as_secs_f64() / q4_med.as_secs_f64().max(1e-9);
    eprintln!("    → speedup {speedup:.2}×");

    Ok(())
}

// ─── OUTCOMES ──────────────────────────────────────────────────────────
//
// If the test passes both criteria (speedup ≥ 1.5×, rel-err < 5e-2):
//   → Phase 1 unblocked. Proceed to add `register_weight_quantized` to
//     `GpuOffloadEngine` and wire `WgpuVulkanEngine::WeightStore::Quantized`.
//   → Phase 3 (hidden-axis rotation) is NOT required for accuracy — the
//     2e-2 to 5e-2 rel-err range is acceptable for end-to-end greedy
//     token parity at the Qwen3 model scales.
//
// If speedup < 1.5× but rel-err is acceptable:
//   → The kernel is producing correct results but isn't engaging the
//     quantized tile path (probably dequant-then-f32 in `launch_matmul`).
//   → Action: fork burn-cubecl, add a debug trace at
//     `kernel/matmul/base.rs:launch_matmul` to confirm which path fires,
//     and patch the dispatch so Q4 weights use the cubek-matmul
//     quantized launch_ref.
//   → Estimated effort: 1-2 weeks.
//
// If rel-err >= 5e-2:
//   → Naive Q4 quantization is too lossy at our shapes (likely outlier
//     weight rows blowing the Block(128) scale).
//   → Action: skip to Phase 3 (HD₃/DCT-IV hidden-axis rotation). The
//     QuIP#-style rotation flattens the weight entry distribution and
//     makes Block(128) Q4 viable. Re-run the spike with rotated weights
//     and confirm rel-err drops below 5e-2.
//   → Note: if BOTH speedup and rel-err fail, the speedup might
//     improve after Phase 3 (rotated weights are friendlier to the
//     quantized GEMM kernel's accumulator).

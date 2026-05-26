//! ROCm/HIP backend Q4 matmul spike — mirror of
//! `q4_kernel_spike.rs` but using `cubecl_hip::HipRuntime` instead of
//! `cubecl_wgpu::WgpuRuntime`.
//!
//! Hypothesis (from the wgpu spike's negative result): on AMD Strix
//! Halo's RDNA 3.5 iGPU (`gfx1151`), the WGSL/Vulkan path does not
//! engage the WMMA INT4 instructions that should make Q4 matmul fast.
//! Switching to HIP exposes WMMA intrinsics directly through
//! cubek-matmul's CMMA strategy. If that's the bottleneck, the HIP
//! path should show ≥ 1.5× speedup over f32 matmul where the wgpu
//! path showed 0.78–0.95×.
//!
//! Pre-req: ROCm 7.x installed, `hipconfig --version` works, the
//! `vendor/cubecl-hip-sys-patched` workspace patch is applied so
//! cubecl-hip-sys compiles against ROCm 7.2.x.

use std::time::Instant;

use anyhow::{Result, anyhow};
use burn_backend::Backend;
use burn_cubecl::CubeBackend;
use burn_tensor::{Tensor, TensorData};
use cubecl_common::quant::scheme::{
    BlockSize, QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
};
use cubecl_hip::HipRuntime;
use ndarray::Array2;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

/// burn-cubecl backend on the ROCm/HIP runtime, f32 internal.
type CubeHip32 = CubeBackend<HipRuntime, f32, i32, u8>;

const WARMUP_RUNS: usize = 2;
const TIMED_RUNS: usize = 5;

fn sample_normal(rng: &mut ChaCha20Rng, n: usize, d: usize) -> Array2<f32> {
    let normal = StandardNormal;
    Array2::from_shape_fn((n, d), |_| normal.sample(rng))
}

fn array_to_tensor(a: &Array2<f32>, device: &burn_tensor::Device<CubeHip32>) -> Tensor<CubeHip32, 2> {
    let shape = [a.nrows(), a.ncols()];
    let data = TensorData::new(a.as_slice().expect("standard layout").to_vec(), shape);
    Tensor::<CubeHip32, 2>::from_data(data, device)
}

fn tensor_to_array(t: Tensor<CubeHip32, 2>) -> Array2<f32> {
    let shape = t.shape().dims;
    let data = t.into_data();
    let vec: Vec<f32> = data.to_vec::<f32>().expect("convert to f32 vec");
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

fn time_matmul<F>(label: &str, mut f: F) -> std::time::Duration
where
    F: FnMut() -> Array2<f32>,
{
    for _ in 0..WARMUP_RUNS {
        let _ = f();
    }
    let mut samples = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let t = Instant::now();
        let _ = f();
        samples.push(t.elapsed());
    }
    samples.sort();
    let median = samples[samples.len() / 2];
    let min = samples.first().copied().unwrap_or_default();
    let max = samples.last().copied().unwrap_or_default();
    eprintln!(
        "    {label:<28} median {:>8.1} ms · min {:>7.1} · max {:>7.1} ms",
        median.as_secs_f64() * 1000.0,
        min.as_secs_f64() * 1000.0,
        max.as_secs_f64() * 1000.0,
    );
    median
}

/// HIP spike at Qwen3-4B's QKV-Q shape: s=2056, d_in=2560, d_out=4096.
/// Same shape as the wgpu spike's worst case (0.88× speedup) — direct
/// apples-to-apples comparison.
#[test]
#[ignore = "requires ROCm 7.x + gfx1151 iGPU; ~30 sec wall-clock; loud-fails if HipRuntime can't init"]
fn q4_matmul_hip_spike_qwen3_4b_qkv_shape() -> Result<()> {
    let _ = env_logger::try_init();
    let device = burn_tensor::Device::<CubeHip32>::default();
    <CubeHip32 as Backend>::sync(&device)
        .map_err(|e| anyhow!("HipRuntime sync at init: {e:?}"))?;

    let s = 2056_usize;
    let d_in = 2560_usize;
    let d_out = 4096_usize;
    let mut rng = ChaCha20Rng::from_seed([91u8; 32]);

    eprintln!("=== Phase 0 HIP q_matmul spike ===");
    eprintln!("shape: input ({s}, {d_in}) × weight ({d_in}, {d_out})");
    eprintln!("backend: cubecl-hip/HipRuntime (ROCm 7.2.1 on gfx1151)");

    let input_f32 = sample_normal(&mut rng, s, d_in);
    let weight_f32 = sample_normal(&mut rng, d_in, d_out);
    let input_t = array_to_tensor(&input_f32, &device);
    let weight_t_f32 = array_to_tensor(&weight_f32, &device);

    let scheme = QuantScheme {
        value: QuantValue::Q4S,
        param: QuantParam::F32,
        store: QuantStore::PackedU32(0),
        level: QuantLevel::Block(BlockSize::new([128u8])),
        mode: QuantMode::Symmetric,
    };
    eprintln!("scheme: {scheme:?}");

    let weight_t_q4 = weight_t_f32.clone().quantize_dynamic(&scheme);
    <CubeHip32 as Backend>::sync(&device).map_err(|e| anyhow!("sync post-quant: {e:?}"))?;

    eprintln!("\n[timing]");
    let f32_median = time_matmul("f32 weight matmul", || {
        tensor_to_array(input_t.clone().matmul(weight_t_f32.clone()))
    });
    let q4_median = time_matmul("Q4 weight matmul", || {
        tensor_to_array(input_t.clone().matmul(weight_t_q4.clone()))
    });

    let out_f32 = tensor_to_array(input_t.clone().matmul(weight_t_f32.clone()));
    let out_q4 = tensor_to_array(input_t.clone().matmul(weight_t_q4.clone()));
    let rel_err = rel_max_abs_error(&out_q4, &out_f32);

    let f32_ms = f32_median.as_secs_f64() * 1000.0;
    let q4_ms = q4_median.as_secs_f64() * 1000.0;
    let speedup = f32_ms / q4_ms.max(1e-6);

    eprintln!("\n[verdict]");
    eprintln!("    f32 median    = {f32_ms:>7.1} ms");
    eprintln!("    Q4  median    = {q4_ms:>7.1} ms");
    eprintln!("    speedup (f32/Q4) = {speedup:.2}×");
    eprintln!("    rel-err vs f32   = {rel_err:.3e}");
    eprintln!(
        "    speedup ≥ 1.5×            : {}",
        if speedup >= 1.5 { "PASS" } else { "FAIL" }
    );
    eprintln!(
        "    rel-err < 5e-2            : {}",
        if rel_err < 5e-2 { "PASS" } else { "FAIL (expected without HD₃ rotation)" }
    );

    // Compare to wgpu baseline (from `q4_kernel_spike.rs` at same shape):
    eprintln!("\n[wgpu reference at same shape]");
    eprintln!("    Vulkan f32 median       ≈   26.8 ms");
    eprintln!("    Vulkan Q4  median       ≈   30.5 ms");
    eprintln!("    Vulkan speedup          ≈   0.88×");
    eprintln!("    -> HIP/Vulkan f32 ratio  = {:.2}×", 26.8 / f32_ms.max(1e-6));
    eprintln!("    -> HIP/Vulkan Q4 ratio   = {:.2}×", 30.5 / q4_ms.max(1e-6));

    Ok(())
}

/// HIP shape sweep across all four Qwen3-4B per-layer projection
/// shapes. Mirror of `q4_matmul_shape_sweep` in `q4_kernel_spike.rs`.
#[test]
#[ignore = "real HIP GPU; ~1 min wall-clock"]
fn q4_matmul_hip_shape_sweep() -> Result<()> {
    let _ = env_logger::try_init();
    let device = burn_tensor::Device::<CubeHip32>::default();
    <CubeHip32 as Backend>::sync(&device)
        .map_err(|e| anyhow!("HipRuntime sync at init: {e:?}"))?;

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
    eprintln!("=== HIP Q4 shape sweep ===");
    eprintln!("scheme: Q4S Block(128) on gfx1151 (RDNA 3.5 iGPU via ROCm 7.2.1)\n");

    for (label, s, d_in, d_out) in cases {
        let input_f32 = sample_normal(&mut rng, *s, *d_in);
        let weight_f32 = sample_normal(&mut rng, *d_in, *d_out);
        let input_t = array_to_tensor(&input_f32, &device);
        let w_f32 = array_to_tensor(&weight_f32, &device);
        let w_q4 = w_f32.clone().quantize_dynamic(&scheme);
        <CubeHip32 as Backend>::sync(&device).ok();

        let f32_med = time_matmul(&format!("{label} f32"), || {
            tensor_to_array(input_t.clone().matmul(w_f32.clone()))
        });
        let q4_med = time_matmul(&format!("{label} Q4 "), || {
            tensor_to_array(input_t.clone().matmul(w_q4.clone()))
        });

        let out_f32 = tensor_to_array(input_t.clone().matmul(w_f32.clone()));
        let out_q4 = tensor_to_array(input_t.clone().matmul(w_q4.clone()));
        let rel = rel_max_abs_error(&out_q4, &out_f32);
        let speedup = f32_med.as_secs_f64() / q4_med.as_secs_f64().max(1e-9);
        eprintln!("    → speedup {speedup:.2}× | rel_err {rel:.3e}\n");
    }
    Ok(())
}

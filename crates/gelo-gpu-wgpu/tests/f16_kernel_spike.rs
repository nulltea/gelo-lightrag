//! f16 matmul spike — direct A/B test of f32 vs f16 matmul on Vulkan
//! at Qwen3-4B shapes, after the Q4 kernel was shown unable to deliver
//! speedup on Strix Halo's iGPU.
//!
//! Hypothesis: f16 (with `shader-f16` extension) should reduce both
//! weight and activation bandwidth by 2×, and modern AMD adapters have
//! a fast f16 matmul path that f32 doesn't engage. If the iGPU is
//! bandwidth-bound (which the Q4 spike strongly suggested), f16 should
//! show a ~1.5–2× speedup with negligible accuracy loss on Gaussian
//! random inputs.
//!
//! Pass criteria:
//! 1. Speedup ≥ 1.5× vs f32 matmul at all four projection shapes.
//! 2. Max relative error vs f32 < 5e-3 on N(0,1) inputs (f16 has
//!    11-bit mantissa, ~3-4 decimal digits — 5e-3 is conservative).

use std::time::Instant;

use anyhow::{Result, anyhow};
use burn_backend::Backend;
use burn_cubecl::CubeBackend;
use burn_tensor::{Tensor, TensorData};
use cubecl_wgpu::WgpuRuntime;
use half::f16;
use ndarray::Array2;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

type CubeWgpu32 = CubeBackend<WgpuRuntime, f32, i32, u8>;
type CubeWgpu16 = CubeBackend<WgpuRuntime, f16, i32, u8>;

const WARMUP_RUNS: usize = 2;
const TIMED_RUNS: usize = 5;

fn sample_normal(rng: &mut ChaCha20Rng, n: usize, d: usize) -> Array2<f32> {
    let normal = StandardNormal;
    Array2::from_shape_fn((n, d), |_| normal.sample(rng))
}

fn array_to_tensor_f32(
    a: &Array2<f32>,
    device: &burn_tensor::Device<CubeWgpu32>,
) -> Tensor<CubeWgpu32, 2> {
    let shape = [a.nrows(), a.ncols()];
    let data = TensorData::new(a.as_slice().expect("standard layout").to_vec(), shape);
    Tensor::<CubeWgpu32, 2>::from_data(data, device)
}

fn array_to_tensor_f16(
    a: &Array2<f32>,
    device: &burn_tensor::Device<CubeWgpu16>,
) -> Tensor<CubeWgpu16, 2> {
    let shape = [a.nrows(), a.ncols()];
    let data_f16: Vec<f16> = a
        .as_slice()
        .expect("standard layout")
        .iter()
        .map(|&v| f16::from_f32(v))
        .collect();
    let data = TensorData::new(data_f16, shape);
    Tensor::<CubeWgpu16, 2>::from_data(data, device)
}

fn tensor_to_array_f32(t: Tensor<CubeWgpu32, 2>) -> Array2<f32> {
    let shape = t.shape().dims;
    let data = t.into_data();
    let vec: Vec<f32> = data.to_vec::<f32>().expect("convert to f32 vec");
    Array2::from_shape_vec((shape[0], shape[1]), vec).expect("shape matches")
}

fn tensor_to_array_f16(t: Tensor<CubeWgpu16, 2>) -> Array2<f32> {
    let shape = t.shape().dims;
    let data = t.into_data();
    let vec: Vec<f16> = data.to_vec::<f16>().expect("convert to f16 vec");
    let vec_f32: Vec<f32> = vec.into_iter().map(f16::to_f32).collect();
    Array2::from_shape_vec((shape[0], shape[1]), vec_f32).expect("shape matches")
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

/// f16 shape sweep across all four Qwen3-4B projection shapes.
/// Direct apples-to-apples vs `q4_kernel_spike::q4_matmul_shape_sweep`.
#[test]
#[ignore = "requires real Vulkan GPU with shader-f16; ~1 min wall-clock"]
fn f16_matmul_shape_sweep() -> Result<()> {
    let _ = env_logger::try_init();
    let device_f32 = burn_tensor::Device::<CubeWgpu32>::default();
    let device_f16 = burn_tensor::Device::<CubeWgpu16>::default();
    <CubeWgpu32 as Backend>::sync(&device_f32).map_err(|e| anyhow!("f32 init: {e:?}"))?;
    <CubeWgpu16 as Backend>::sync(&device_f16).map_err(|e| anyhow!("f16 init: {e:?}"))?;

    let cases: &[(&str, usize, usize, usize)] = &[
        ("QKV-Q   (input·W)        ", 2056, 2560, 4096),
        ("Gate∥Up (input·W)        ", 2056, 2560, 9728),
        ("FfnDown (input·W)        ", 2056, 9728, 2560),
        ("O proj  (input·W)        ", 2056, 4096, 2560),
    ];

    let mut rng = ChaCha20Rng::from_seed([127u8; 32]);
    eprintln!("=== f16 vs f32 matmul shape sweep (Vulkan iGPU, gfx1151) ===\n");

    for (label, s, d_in, d_out) in cases {
        let input_f32 = sample_normal(&mut rng, *s, *d_in);
        let weight_f32 = sample_normal(&mut rng, *d_in, *d_out);

        let input_t_f32 = array_to_tensor_f32(&input_f32, &device_f32);
        let weight_t_f32 = array_to_tensor_f32(&weight_f32, &device_f32);
        let input_t_f16 = array_to_tensor_f16(&input_f32, &device_f16);
        let weight_t_f16 = array_to_tensor_f16(&weight_f32, &device_f16);
        <CubeWgpu32 as Backend>::sync(&device_f32).ok();
        <CubeWgpu16 as Backend>::sync(&device_f16).ok();

        let f32_med = time_matmul(&format!("{label} f32"), || {
            tensor_to_array_f32(input_t_f32.clone().matmul(weight_t_f32.clone()))
        });
        let f16_med = time_matmul(&format!("{label} f16"), || {
            tensor_to_array_f16(input_t_f16.clone().matmul(weight_t_f16.clone()))
        });

        let out_f32 = tensor_to_array_f32(input_t_f32.clone().matmul(weight_t_f32.clone()));
        let out_f16 = tensor_to_array_f16(input_t_f16.clone().matmul(weight_t_f16.clone()));
        let rel = rel_max_abs_error(&out_f16, &out_f32);
        let mae = mean_abs_error(&out_f16, &out_f32);
        let speedup = f32_med.as_secs_f64() / f16_med.as_secs_f64().max(1e-9);
        eprintln!(
            "    → speedup {speedup:.2}× | rel_err_max {rel:.3e} | mae {mae:.3e}\n"
        );
    }

    Ok(())
}

//! Q#2 RADV-async spike — does iGPU GPU matmul actually overlap
//! with CPU mask cascade on Strix Halo UMA?
//!
//! The §4.D R4 async-pipelining lever requires CPU work (mask
//! cascade for layer N+1) to overlap with GPU work (matmul for
//! layer N) on the shared DDR5 bus. wgpu / burn-cubecl already
//! expose async submit via `into_data_async`, so the question
//! isn't API design — it's bus contention. This spike measures
//! end-to-end wall in three regimes and computes the overlap
//! coefficient.
//!
//! Run:
//!   cargo test -p gelo-gpu-wgpu --release \
//!       --test q2_radv_async_spike -- --ignored --nocapture
//!
//! Interpretation:
//!   speedup = (T_gpu + T_cpu) / T_concurrent
//!     ~2.0 → full overlap → R4 viable, projected ~min(T_gpu, T_cpu) wall saving
//!     ~1.0 → no overlap → R4 dead on iGPU
//!     between → partial overlap; project savings linearly

use std::sync::Arc;
use std::thread;
use std::time::Instant;

use gelo_gpu_wgpu::WgpuVulkanEngine;
use gelo_protocol::dct4::Dct4Mask;
use gelo_protocol::{GpuOffloadEngine, WeightHandle, WeightKind};
use ndarray::Array2;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

const N_WARMUP: usize = 3;
const N_ITER: usize = 15;

fn sample_normal(rng: &mut ChaCha20Rng, n: usize, d: usize) -> Array2<f32> {
    let normal = StandardNormal;
    Array2::from_shape_fn((n, d), |_| normal.sample(rng))
}

fn stats(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

fn fmt_ms(seconds: f64) -> String {
    format!("{:8.3} ms", seconds * 1000.0)
}

/// Production-shape spike: engine.matmul at (2056, 2560) × (2560, 2560)
/// (O projection shape) vs DCT-IV cascade at (2056, 2560).
#[test]
#[ignore = "Q#2 spike: takes a couple of minutes; run on a quiet box"]
fn q2_radv_async_overlap_at_production_shape() {
    let n = 2056; // n_prompt 2048 + k_shield 8
    let d = 2560; // Qwen3-4B hidden_size
    let d_out = 2560; // O projection out_features

    eprintln!("=== Q#2 RADV-async spike — n={n} d={d} d_out={d_out} ===");
    eprintln!("warmup: {N_WARMUP} iter; measure: {N_ITER} iter\n");

    let mut rng = ChaCha20Rng::from_seed([42u8; 32]);
    let input = sample_normal(&mut rng, n, d);
    let weight = sample_normal(&mut rng, d, d_out);
    let mask = Arc::new(Dct4Mask::fresh(n, &mut rng));
    let cpu_input = Arc::new(sample_normal(&mut rng, n, d));

    let engine = WgpuVulkanEngine::new_fp16().expect("Vulkan adapter (fp16)");
    let handle = WeightHandle::new(0, WeightKind::O);
    {
        let mut e = engine.clone_shared();
        e.register_weight(handle, weight.view())
            .expect("register weight");
    }

    // ── 1. T_gpu: engine.matmul alone ──
    let mut t_gpu: Vec<f64> = Vec::with_capacity(N_ITER);
    for i in 0..(N_WARMUP + N_ITER) {
        let t0 = Instant::now();
        let _out = engine.matmul(handle, input.view()).expect("matmul");
        let dt = t0.elapsed().as_secs_f64();
        if i >= N_WARMUP {
            t_gpu.push(dt);
        }
    }

    // ── 2. T_cpu: DCT-IV cascade apply+unapply alone ──
    let mut t_cpu: Vec<f64> = Vec::with_capacity(N_ITER);
    for i in 0..(N_WARMUP + N_ITER) {
        let mut buf = (*cpu_input).clone();
        let buf_slice = buf.as_slice_mut().unwrap();
        let t0 = Instant::now();
        mask.apply_in_place_slice(buf_slice, d);
        mask.unapply_in_place_slice(buf_slice, d);
        let dt = t0.elapsed().as_secs_f64();
        if i >= N_WARMUP {
            t_cpu.push(dt);
        }
    }

    // ── 3. T_concurrent: engine.matmul on main thread, DCT-IV cascade on worker thread ──
    let mut t_concurrent: Vec<f64> = Vec::with_capacity(N_ITER);
    for i in 0..(N_WARMUP + N_ITER) {
        let mask_clone = Arc::clone(&mask);
        let cpu_input_clone = Arc::clone(&cpu_input);
        let t0 = Instant::now();
        // Spawn the CPU cascade work first so it starts immediately.
        let cpu_handle = thread::spawn(move || {
            let mut buf = (*cpu_input_clone).clone();
            let buf_slice = buf.as_slice_mut().unwrap();
            mask_clone.apply_in_place_slice(buf_slice, d);
            mask_clone.unapply_in_place_slice(buf_slice, d);
        });
        // Main thread runs the GPU matmul (sync, blocks on download).
        let _out = engine.matmul(handle, input.view()).expect("matmul");
        // Wait for the CPU thread.
        cpu_handle.join().expect("cpu thread");
        let dt = t0.elapsed().as_secs_f64();
        if i >= N_WARMUP {
            t_concurrent.push(dt);
        }
    }

    let (t_gpu_mean, t_gpu_std) = stats(&t_gpu);
    let (t_cpu_mean, t_cpu_std) = stats(&t_cpu);
    let (t_conc_mean, t_conc_std) = stats(&t_concurrent);

    let serial_sum = t_gpu_mean + t_cpu_mean;
    let max_alone = t_gpu_mean.max(t_cpu_mean);
    let speedup = serial_sum / t_conc_mean;
    let savings = serial_sum - t_conc_mean;
    let overlap_pct = 100.0 * (savings / t_cpu_mean.min(t_gpu_mean));

    eprintln!(
        "  T_gpu          {} ± {}",
        fmt_ms(t_gpu_mean),
        fmt_ms(t_gpu_std)
    );
    eprintln!(
        "  T_cpu          {} ± {}",
        fmt_ms(t_cpu_mean),
        fmt_ms(t_cpu_std)
    );
    eprintln!(
        "  T_concurrent   {} ± {}",
        fmt_ms(t_conc_mean),
        fmt_ms(t_conc_std)
    );
    eprintln!();
    eprintln!(
        "  T_gpu + T_cpu  {}        (serial baseline)",
        fmt_ms(serial_sum)
    );
    eprintln!(
        "  max(T_gpu, T_cpu) {}     (full-overlap ceiling)",
        fmt_ms(max_alone)
    );
    eprintln!();
    eprintln!("  speedup vs serial         {:.3}× (1.0 = no overlap, 2.0 = full overlap)", speedup);
    eprintln!("  wall saved vs serial      {} ({:.1}% of min(T_gpu,T_cpu))", fmt_ms(savings), overlap_pct);
    eprintln!();

    // Interpretation banner
    if overlap_pct >= 70.0 {
        eprintln!("  → FULL OVERLAP: R4 viable on iGPU. Projected ~{:.0}% wall saving at the overlapped buckets.", overlap_pct.min(100.0));
    } else if overlap_pct >= 30.0 {
        eprintln!("  → PARTIAL OVERLAP: R4 delivers some win. Project saving = {:.1}% of overlapped bucket.", overlap_pct);
    } else if overlap_pct >= 10.0 {
        eprintln!("  → WEAK OVERLAP: R4 marginal at best. DDR5 contention is the gate.");
    } else {
        eprintln!("  → NO MEANINGFUL OVERLAP: R4 dead on iGPU. CPU and GPU contend for DDR5.");
    }

    eprintln!(
        "\nQ2_RESULT shape=({n},{d},{d_out}) t_gpu_ms={:.3} t_cpu_ms={:.3} t_concurrent_ms={:.3} speedup={:.3} overlap_pct={:.1}",
        t_gpu_mean * 1000.0, t_cpu_mean * 1000.0, t_conc_mean * 1000.0, speedup, overlap_pct,
    );
}

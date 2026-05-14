//! `WgpuVulkanEngine` parity vs the CPU reference engine.
//!
//! Tests skip themselves gracefully when no Vulkan-capable adapter is
//! available (no GPU, no ICD installed, headless CI box), so checking the
//! crate in is safe even in environments that can't actually run a Vulkan
//! workload.

use ndarray::Array2;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

use gelo_gpu_wgpu::WgpuVulkanEngine;
use gelo_protocol::{GpuOffloadEngine, RayonCpuEngine, WeightHandle, WeightKind};

fn random_matrix(rows: usize, cols: usize, rng: &mut impl rand::RngCore) -> Array2<f32> {
    let normal = StandardNormal;
    Array2::from_shape_fn((rows, cols), |_| {
        <StandardNormal as Distribution<f32>>::sample(&normal, rng)
    })
}

fn open_engine() -> Option<WgpuVulkanEngine> {
    match WgpuVulkanEngine::new() {
        Ok(e) => {
            let info = e.adapter_info();
            eprintln!(
                "wgpu adapter: backend={:?} name={:?} vendor=0x{:x} device=0x{:x} device_type={:?} driver={:?} driver_info={:?}",
                info.backend,
                info.name,
                info.vendor,
                info.device,
                info.device_type,
                info.driver,
                info.driver_info,
            );
            Some(e)
        }
        Err(err) => {
            eprintln!("skipping wgpu parity: no Vulkan adapter ({err})");
            None
        }
    }
}

#[test]
fn selects_real_gpu_hardware_not_software_rasterizer() {
    let Some(engine) = open_engine() else {
        return;
    };
    let info = engine.adapter_info();
    assert_eq!(
        info.backend,
        wgpu::Backend::Vulkan,
        "expected Vulkan backend, got {:?}",
        info.backend
    );
    assert!(
        engine.is_real_gpu(),
        "wgpu selected a non-GPU adapter: {info:?} — this means we're falling back to a software Vulkan ICD (e.g. lavapipe / llvmpipe). Test must run on a real GPU.",
    );
    assert!(
        !info.name.to_lowercase().contains("llvmpipe"),
        "wgpu picked llvmpipe software rasterizer: {info:?}",
    );
    assert!(
        !info.name.to_lowercase().contains("lavapipe"),
        "wgpu picked lavapipe software rasterizer: {info:?}",
    );
}

#[test]
fn matmul_matches_cpu_engine() {
    let Some(mut gpu) = open_engine() else {
        return;
    };
    let mut cpu = RayonCpuEngine::new();

    let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
    let m = 32;
    let k = 24;
    let n = 16;

    let input = random_matrix(m, k, &mut rng);
    let weight = random_matrix(k, n, &mut rng);
    let handle = WeightHandle::new(0, WeightKind::Q);

    cpu.register_weight(handle, weight.view()).unwrap();
    gpu.register_weight(handle, weight.view()).unwrap();

    let cpu_out = cpu.matmul(handle, input.view()).unwrap();
    let gpu_out = gpu.matmul(handle, input.view()).unwrap();

    assert_eq!(cpu_out.shape(), gpu_out.shape());
    let mut max_abs = 0.0_f32;
    for ((i, j), v) in cpu_out.indexed_iter() {
        let diff = (v - gpu_out[[i, j]]).abs();
        if diff > max_abs {
            max_abs = diff;
        }
    }
    assert!(
        max_abs < 5e-4,
        "GPU vs CPU max abs diff {max_abs} exceeds tolerance",
    );
}

#[test]
fn shape_dispatch_handles_non_multiples_of_workgroup() {
    let Some(mut gpu) = open_engine() else {
        return;
    };
    let mut cpu = RayonCpuEngine::new();

    // 17×11 · 11×19 — none of these dimensions are multiples of 16, which
    // exercises the workgroup-bounds-check branches in the WGSL kernel.
    let mut rng = ChaCha20Rng::from_seed([29u8; 32]);
    let input = random_matrix(17, 11, &mut rng);
    let weight = random_matrix(11, 19, &mut rng);
    let handle = WeightHandle::new(0, WeightKind::FfnUp);

    cpu.register_weight(handle, weight.view()).unwrap();
    gpu.register_weight(handle, weight.view()).unwrap();

    let cpu_out = cpu.matmul(handle, input.view()).unwrap();
    let gpu_out = gpu.matmul(handle, input.view()).unwrap();

    assert_eq!(cpu_out.shape(), gpu_out.shape());
    for ((i, j), v) in cpu_out.indexed_iter() {
        let diff = (v - gpu_out[[i, j]]).abs();
        assert!(diff < 5e-4, "({i},{j}): cpu={v} gpu={}", gpu_out[[i, j]]);
    }
}

#[test]
fn matmul_dynamic_matches_cpu_engine() {
    let Some(gpu) = open_engine() else {
        return;
    };
    let cpu = RayonCpuEngine::new();

    let mut rng = ChaCha20Rng::from_seed([57u8; 32]);
    let m = 24;
    let k = 16;
    let n = 8;

    let lhs = random_matrix(m, k, &mut rng);
    let rhs = random_matrix(k, n, &mut rng);

    let cpu_out = cpu.matmul_dynamic(lhs.view(), rhs.view()).unwrap();
    let gpu_out = gpu.matmul_dynamic(lhs.view(), rhs.view()).unwrap();

    assert_eq!(cpu_out.shape(), gpu_out.shape());
    for ((i, j), v) in cpu_out.indexed_iter() {
        assert!(
            (v - gpu_out[[i, j]]).abs() < 5e-4,
            "matmul_dynamic GPU vs CPU diverges at ({i},{j}): {v} vs {}",
            gpu_out[[i, j]]
        );
    }
}

#[test]
fn matmul_dynamic_batched_matches_cpu_engine() {
    let Some(gpu) = open_engine() else {
        return;
    };
    let cpu = RayonCpuEngine::new();

    let mut rng = ChaCha20Rng::from_seed([67u8; 32]);
    let b = 8;
    let m = 20;
    let k = 32;
    let n = 12;

    let mut lhs = ndarray::Array3::<f32>::zeros((b, m, k));
    let mut rhs = ndarray::Array3::<f32>::zeros((b, k, n));
    for bi in 0..b {
        let nl = random_matrix(m, k, &mut rng);
        let nr = random_matrix(k, n, &mut rng);
        lhs.index_axis_mut(ndarray::Axis(0), bi).assign(&nl);
        rhs.index_axis_mut(ndarray::Axis(0), bi).assign(&nr);
    }

    let cpu_out = cpu.matmul_dynamic_batched(lhs.view(), rhs.view()).unwrap();
    let gpu_out = gpu.matmul_dynamic_batched(lhs.view(), rhs.view()).unwrap();

    assert_eq!(cpu_out.shape(), gpu_out.shape());
    for ((bi, i, j), v) in cpu_out.indexed_iter() {
        assert!(
            (v - gpu_out[[bi, i, j]]).abs() < 5e-4,
            "batched matmul GPU vs CPU diverges at ({bi},{i},{j}): {v} vs {}",
            gpu_out[[bi, i, j]]
        );
    }
}

#[test]
fn weight_buffer_is_cached_across_calls() {
    // Functional check that register_weight uploads once and matmul reuses.
    let Some(mut gpu) = open_engine() else {
        return;
    };
    let mut rng = ChaCha20Rng::from_seed([37u8; 32]);
    let weight = random_matrix(8, 8, &mut rng);
    let handle = WeightHandle::new(3, WeightKind::O);
    gpu.register_weight(handle, weight.view()).unwrap();

    let mut prev = None;
    for seed in 0u8..3 {
        let mut r = ChaCha20Rng::from_seed([seed; 32]);
        let input = random_matrix(8, 8, &mut r);
        let out = gpu.matmul(handle, input.view()).unwrap();
        let expected = input.dot(&weight);
        for ((i, j), v) in expected.indexed_iter() {
            assert!(
                (v - out[[i, j]]).abs() < 5e-4,
                "seed {seed} ({i},{j}): expected={v} got={}",
                out[[i, j]]
            );
        }
        prev = Some(out);
    }
    assert!(prev.is_some());
}

#[test]
fn softmax_batched_matches_cpu_reference() {
    // Phase 3 of the Tier 1 work: the wgpu engine's softmax_batched
    // override must produce row-stochastic results that match the CPU
    // trait default to within f32 tolerance. Without this parity, the
    // permutation-shielded attention path would silently disagree with
    // its TEE-side equivalent.
    let Some(gpu) = open_engine() else {
        return;
    };
    let cpu = RayonCpuEngine::new();
    let mut rng = ChaCha20Rng::from_seed([42u8; 32]);

    let (b, m, n) = (4usize, 32, 32);
    let mut input = ndarray::Array3::<f32>::zeros((b, m, n));
    for bi in 0..b {
        let mat = random_matrix(m, n, &mut rng);
        input.index_axis_mut(ndarray::Axis(0), bi).assign(&mat);
    }

    let cpu_out = cpu.softmax_batched(input.view()).unwrap();
    let gpu_out = gpu.softmax_batched(input.view()).unwrap();

    assert_eq!(cpu_out.shape(), gpu_out.shape());

    let mut max_diff = 0.0f32;
    for ((bi, i, j), v) in cpu_out.indexed_iter() {
        max_diff = max_diff.max((v - gpu_out[[bi, i, j]]).abs());
    }
    // burn-cubecl softmax goes through max-subtract + exp + sum_dim + div.
    // f32 tolerance of 5e-5 — softmax tends to amplify small input drift.
    assert!(
        max_diff < 5e-5,
        "GPU softmax_batched diverges from CPU: max_diff={max_diff}",
    );

    // Row-stochastic check on the GPU output.
    for bi in 0..b {
        for i in 0..m {
            let row_sum: f32 = (0..n).map(|j| gpu_out[[bi, i, j]]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-4,
                "GPU softmax row sum != 1 at ({bi},{i}): {row_sum}",
            );
        }
    }
}

#[test]
fn permuted_attention_gpu_matches_cpu() {
    // Full Tier 1 chain: permuted_attention with the wgpu engine vs with
    // the CPU engine. Same RNG seed ensures the same π is sampled both
    // times, so we compare engine outputs at identical pre-softmax
    // states — the only difference is which device does matmul + softmax.
    use gelo_protocol::attention::{self, PermAttnConfig};

    let Some(gpu) = open_engine() else {
        return;
    };
    let cpu = RayonCpuEngine::new();

    let h = 4;
    let n = 16;
    let d_head = 32;
    let scale = 1.0 / (d_head as f32).sqrt();

    let mut rng_init = ChaCha20Rng::from_seed([99u8; 32]);
    let mut q = ndarray::Array3::<f32>::zeros((h, n, d_head));
    let mut k = ndarray::Array3::<f32>::zeros((h, n, d_head));
    let mut v = ndarray::Array3::<f32>::zeros((h, n, d_head));
    for hi in 0..h {
        let qm = random_matrix(n, d_head, &mut rng_init);
        let km = random_matrix(n, d_head, &mut rng_init);
        let vm = random_matrix(n, d_head, &mut rng_init);
        q.index_axis_mut(ndarray::Axis(0), hi).assign(&qm);
        k.index_axis_mut(ndarray::Axis(0), hi).assign(&km);
        v.index_axis_mut(ndarray::Axis(0), hi).assign(&vm);
    }

    let mut rng_cpu = ChaCha20Rng::from_seed([7u8; 32]);
    let cpu_out = attention::permuted_attention(
        &cpu,
        q.view(),
        k.view(),
        v.view(),
        scale,
        PermAttnConfig::DISABLED_NOISE,
        &mut rng_cpu,
    )
    .unwrap();

    let mut rng_gpu = ChaCha20Rng::from_seed([7u8; 32]);
    let gpu_out = attention::permuted_attention(
        &gpu,
        q.view(),
        k.view(),
        v.view(),
        scale,
        PermAttnConfig::DISABLED_NOISE,
        &mut rng_gpu,
    )
    .unwrap();

    let mut max_diff = 0.0f32;
    for (idx, v) in cpu_out.indexed_iter() {
        max_diff = max_diff.max((v - gpu_out[idx]).abs());
    }
    assert!(
        max_diff < 5e-4,
        "permuted_attention CPU vs GPU divergence: max_diff={max_diff}",
    );
}

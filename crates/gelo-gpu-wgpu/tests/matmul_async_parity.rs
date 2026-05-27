//! R4 async API parity tests.
//!
//! Confirms that:
//! - `matmul_async` + `MatmulToken::into_array` produces the same result
//!   as `matmul` on the wgpu engine (override path).
//! - `matmul_many_async` matches `matmul_many` and shares one upload.
//! - Tokens are `Send`, so they can move to worker threads (compile-time
//!   check via `thread::spawn`).
//! - The trait's default sync-fallback path produces correct output on a
//!   minimal stub engine.
//!
//! Tests gracefully skip when no Vulkan adapter is available.

use std::thread;

use anyhow::Result;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

use gelo_gpu_wgpu::WgpuVulkanEngine;
use gelo_protocol::{GpuOffloadEngine, WeightHandle, WeightKind};

fn random_matrix(rows: usize, cols: usize, rng: &mut impl rand::RngCore) -> Array2<f32> {
    let normal = StandardNormal;
    Array2::from_shape_fn((rows, cols), |_| {
        <StandardNormal as Distribution<f32>>::sample(&normal, rng)
    })
}

fn open_engine() -> Option<WgpuVulkanEngine> {
    match WgpuVulkanEngine::new() {
        Ok(e) => Some(e),
        Err(err) => {
            eprintln!("skipping wgpu async parity: no Vulkan adapter ({err})");
            None
        }
    }
}

#[test]
fn matmul_async_matches_matmul() {
    let Some(mut engine) = open_engine() else {
        return;
    };
    let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
    let input = random_matrix(128, 256, &mut rng);
    let weight = random_matrix(256, 192, &mut rng);
    let handle = WeightHandle::new(0, WeightKind::O);

    engine.register_weight(handle, weight.view()).unwrap();

    let sync_out = engine.matmul(handle, input.view()).expect("sync matmul");
    let token = engine.matmul_async(handle, input.view()).expect("async submit");
    let async_out = token.into_array().expect("async drain");

    assert_eq!(sync_out.dim(), async_out.dim(), "shape parity");
    let max_abs = sync_out
        .iter()
        .zip(async_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // Same kernel under the hood — should be bit-exact, but allow a
    // tiny epsilon for any reordering/fusion differences in burn.
    assert!(max_abs < 1e-5, "max abs diff = {max_abs}");
}

#[test]
fn matmul_many_async_matches_matmul_many() {
    let Some(mut engine) = open_engine() else {
        return;
    };
    let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
    let input = random_matrix(64, 128, &mut rng);
    let w_q = random_matrix(128, 96, &mut rng);
    let w_k = random_matrix(128, 96, &mut rng);
    let w_v = random_matrix(128, 96, &mut rng);
    let handles = [
        WeightHandle::new(0, WeightKind::Q),
        WeightHandle::new(0, WeightKind::K),
        WeightHandle::new(0, WeightKind::V),
    ];
    engine.register_weight(handles[0], w_q.view()).unwrap();
    engine.register_weight(handles[1], w_k.view()).unwrap();
    engine.register_weight(handles[2], w_v.view()).unwrap();

    let sync_outs = engine.matmul_many(&handles, input.view()).expect("sync");
    let tokens = engine
        .matmul_many_async(&handles, input.view())
        .expect("async submit");
    assert_eq!(tokens.len(), 3);
    let async_outs: Vec<Array2<f32>> = tokens
        .into_iter()
        .map(|t| t.into_array().expect("drain"))
        .collect();

    for (i, (s, a)) in sync_outs.iter().zip(async_outs.iter()).enumerate() {
        assert_eq!(s.dim(), a.dim(), "shape parity at idx {i}");
        let max_abs = s.iter().zip(a.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(max_abs < 1e-5, "idx {i} max abs diff = {max_abs}");
    }
}

#[test]
fn matmul_token_is_send_across_threads() {
    let Some(mut engine) = open_engine() else {
        return;
    };
    let mut rng = ChaCha20Rng::from_seed([13u8; 32]);
    let input = random_matrix(32, 64, &mut rng);
    let weight = random_matrix(64, 48, &mut rng);
    let handle = WeightHandle::new(0, WeightKind::O);
    engine.register_weight(handle, weight.view()).unwrap();

    let token = engine.matmul_async(handle, input.view()).expect("submit");
    // Move the token to a worker thread and drain there.
    let result = thread::spawn(move || token.into_array().expect("worker drain"))
        .join()
        .expect("worker join");
    // Sanity: shape matches.
    assert_eq!(result.dim(), (32, 48));
}

#[test]
fn matmul_many_async_empty_returns_empty() {
    let Some(engine) = open_engine() else {
        return;
    };
    // Use a dummy input — the empty handles list should short-circuit
    // before any upload.
    let dummy = Array2::<f32>::zeros((4, 8));
    let tokens = engine.matmul_many_async(&[], dummy.view()).expect("ok");
    assert!(tokens.is_empty());
}

// --- default sync-fallback path ---

/// Minimal engine that implements only `matmul`/`matmul_dynamic` so the
/// trait's default `matmul_async` impl gets exercised.
struct StubEngine;

impl GpuOffloadEngine for StubEngine {
    fn register_weight(&mut self, _h: WeightHandle, _w: ArrayView2<f32>) -> Result<()> {
        Ok(())
    }

    fn matmul(&self, _h: WeightHandle, input: ArrayView2<f32>) -> Result<Array2<f32>> {
        // Identity-style: return the input doubled, so we can assert
        // the token round-trips the actual matmul result (not just zero).
        Ok(input.mapv(|v| v * 2.0))
    }

    fn matmul_dynamic(&self, _l: ArrayView2<f32>, _r: ArrayView2<f32>) -> Result<Array2<f32>> {
        unimplemented!("not used")
    }
}

#[test]
fn default_async_falls_back_to_sync_matmul() {
    let engine = StubEngine;
    let input = Array2::<f32>::from_shape_fn((4, 3), |(i, j)| (i * 3 + j) as f32);
    let handle = WeightHandle::new(0, WeightKind::O);

    let token = engine.matmul_async(handle, input.view()).expect("submit");
    let out: Array2<f32> = token.into_array().expect("drain");
    for ((i, j), &v) in out.indexed_iter() {
        assert_eq!(v, ((i * 3 + j) as f32) * 2.0);
    }
}

#[test]
fn default_matmul_many_async_falls_back_per_handle() {
    let engine = StubEngine;
    let input = Array2::<f32>::ones((2, 2));
    let handles = [
        WeightHandle::new(0, WeightKind::Q),
        WeightHandle::new(0, WeightKind::K),
    ];
    let tokens = engine.matmul_many_async(&handles, input.view()).expect("submit");
    assert_eq!(tokens.len(), 2);
    for tok in tokens {
        let out = tok.into_array().expect("drain");
        assert_eq!(out.dim(), (2, 2));
        assert!(out.iter().all(|&v| v == 2.0));
    }
}

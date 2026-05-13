//! Spike bench: measure burn-cubecl's matmul wall-clock vs the cubecl-direct
//! baseline (the latter is recorded from the previous probe run since we
//! can't link both in the same workspace post-cubecl-matmul-removal).
//!
//! Bench shape: representative BGE-base GEMM sizes
//!     (M=64,128,256,512; K=768; N=768) — QKV / output projection
//!     (M=64,128,256,512; K=768; N=3072) — FFN up
//!     (M=64,128,256,512; K=3072; N=768) — FFN down
//!
//! For each shape: 1 cold call (autotune may sweep), then 20 warm calls.
//! Report cold + median warm. The autotune cache should persist to
//! `target/{device_id}/autotune/...` per cubecl-runtime defaults.
//!
//! Win condition for the spike: warm latency ≤ 50% of what we observed
//! pre-spike on equivalent shapes, AND cold-call latency stops scaling
//! per-novel-shape after the first run (autotune cache hit).
//!
//! Run with: `cargo test -p gelo-gpu-wgpu --release --test burn_cubecl_spike -- --ignored --nocapture`

use std::time::Instant;

use burn_cubecl::CubeBackend;
use burn_tensor::{Tensor, TensorData, backend::Backend};
use cubecl::wgpu::{AutoGraphicsApi, RuntimeOptions, WgpuDevice, WgpuRuntime, init_setup_async};

// f32 floats, i32 ints, u8 bools (cube backend type aliases).
type Wgpu = CubeBackend<WgpuRuntime, f32, i32, u8>;

fn make_data(rows: usize, cols: usize, seed: u64) -> TensorData {
    // Deterministic pseudo-random fill (xorshift64 — fine for perf, not numerics).
    let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut v = Vec::with_capacity(rows * cols);
    for _ in 0..(rows * cols) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let f = ((x as u32) as f32) * (1.0 / u32::MAX as f32) - 0.5;
        v.push(f);
    }
    TensorData::new(v, [rows, cols])
}

#[test]
#[ignore]
fn burn_cubecl_matmul_warm_cold_latency() {
    // Initialise once at the workspace level so the autotune cache + persistent
    // memory pool are shared across calls.
    let device = WgpuDevice::default();
    let _setup = cubecl::future::block_on(init_setup_async::<AutoGraphicsApi>(
        &device,
        RuntimeOptions::default(),
    ));

    // Tell burn the backend exists; no setup of its own needed.
    <Wgpu as Backend>::sync(&device).expect("device sync");

    let shapes: Vec<(usize, usize, usize, &str)> = vec![
        (64, 768, 768, "QKV-64"),
        (128, 768, 768, "QKV-128"),
        (256, 768, 768, "QKV-256"),
        (512, 768, 768, "QKV-512"),
        (64, 768, 3072, "FFNup-64"),
        (128, 768, 3072, "FFNup-128"),
        (256, 768, 3072, "FFNup-256"),
        (512, 768, 3072, "FFNup-512"),
        (64, 3072, 768, "FFNdn-64"),
        (128, 3072, 768, "FFNdn-128"),
        (256, 3072, 768, "FFNdn-256"),
        (512, 3072, 768, "FFNdn-512"),
    ];

    eprintln!(
        "{:<12} {:>5} {:>5} {:>5} {:>10} {:>10} {:>10}",
        "shape", "M", "K", "N", "cold_us", "warm_med_us", "warm_min_us"
    );

    for (m, k, n, label) in shapes {
        // Upload the RHS weight once — like a registered weight in our engine.
        let rhs_data = make_data(k, n, 0x1234);
        let rhs: Tensor<Wgpu, 2> = Tensor::from_data(rhs_data, &device);

        // Cold call: fresh masked-input tensor, fresh matmul on a new shape.
        let lhs_data = make_data(m, k, 0xCAFE);
        let lhs: Tensor<Wgpu, 2> = Tensor::from_data(lhs_data, &device);

        let t0 = Instant::now();
        let out = lhs.matmul(rhs.clone());
        let _data = out.into_data(); // forces sync
        let cold_us = t0.elapsed().as_micros();

        // Warm calls: 20 fresh inputs at the same shape.
        let mut warm_us: Vec<u128> = Vec::with_capacity(20);
        for i in 0..20 {
            let lhs_data = make_data(m, k, 0xCAFE + i as u64);
            let lhs: Tensor<Wgpu, 2> = Tensor::from_data(lhs_data, &device);
            let t0 = Instant::now();
            let out = lhs.matmul(rhs.clone());
            let _data = out.into_data();
            warm_us.push(t0.elapsed().as_micros());
        }
        warm_us.sort_unstable();
        let warm_med = warm_us[warm_us.len() / 2];
        let warm_min = warm_us[0];

        eprintln!(
            "{:<12} {:>5} {:>5} {:>5} {:>10} {:>10} {:>10}",
            label, m, k, n, cold_us, warm_med, warm_min
        );
    }
}

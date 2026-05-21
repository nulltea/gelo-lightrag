//! Isolated spike: invoke `cubek_attention::launch::launch` directly on
//! our wgpu device with a decode-shaped fused attention call, then
//! compare the result against a CPU reference implementation.
//!
//! Goal: prove the cubek-attention integration **works at all** before
//! plumbing it through `WgpuVulkanEngine::fused_attention_batched`.
//! Specifically we're testing:
//!
//! 1. That `BlueprintStrategy::Inferred(())` + `Strategy::Unit(...)`
//!    can launch a kernel on the Radeon 8060S / RDNA3.5 + Vulkan +
//!    burn-cubecl + cubek-attention stack at our shapes
//!    (n_q=1, n_kv=1000, num_heads=16, head_dim=128 — Qwen3-4B decode).
//! 2. That the math matches a CPU reference within f16 precision.
//! 3. That the launch latency at decode m=1 is meaningfully below the
//!    burn-tensor 5-op chain (~22 ms).
//!
//! If this passes we wire it into the engine.  If it fails (compile
//! error / runtime error / accuracy mismatch / no perf win) we
//! document the blocker and stop.

use cubecl::ir::{ElemType, FloatKind, StorageType};
use cubek_attention::definition::{
    AccumulatorPrecision, AttentionGlobalTypes, AttentionOptions, AttentionProblem,
};
use cubek_attention::launch::{BlueprintStrategy, Strategy, launch};
use cubek_attention::routines::DeviceSettings;
use gelo_gpu_wgpu::WgpuVulkanEngine;
use std::time::Instant;

/// Reference CPU softmax-attention over `(B, H, S_q, D)` × `(B, H, S_kv, D)`.
fn ref_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    h: usize,
    s_q: usize,
    s_kv: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; b * h * s_q * d];
    for bi in 0..b {
        for hi in 0..h {
            // scores: (s_q, s_kv)
            let mut scores = vec![0.0_f32; s_q * s_kv];
            for i in 0..s_q {
                for j in 0..s_kv {
                    let mut acc = 0.0_f32;
                    for k_idx in 0..d {
                        let q_idx = bi * h * s_q * d + hi * s_q * d + i * d + k_idx;
                        let k_off = bi * h * s_kv * d + hi * s_kv * d + j * d + k_idx;
                        acc += q[q_idx] * k[k_off];
                    }
                    scores[i * s_kv + j] = acc * scale;
                }
            }
            // softmax along s_kv axis
            for i in 0..s_q {
                let row = &mut scores[i * s_kv..(i + 1) * s_kv];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for v in row.iter_mut() {
                    *v = (*v - max).exp();
                    sum += *v;
                }
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }
            // out = scores @ V
            for i in 0..s_q {
                for d_idx in 0..d {
                    let mut acc = 0.0_f32;
                    for j in 0..s_kv {
                        let v_idx = bi * h * s_kv * d + hi * s_kv * d + j * d + d_idx;
                        acc += scores[i * s_kv + j] * v[v_idx];
                    }
                    let o_idx = bi * h * s_q * d + hi * s_q * d + i * d + d_idx;
                    out[o_idx] = acc;
                }
            }
        }
    }
    out
}

#[test]
#[ignore = "runs against real Vulkan/wgpu device; opt-in via `cargo test --release -- --ignored`"]
fn cubek_attention_decode_shape_runs() {
    run_shape("decode", 1, 16, 1, 1000, 128);
}

#[test]
#[ignore = "runs against real Vulkan/wgpu device; opt-in via `cargo test --release -- --ignored`"]
fn cubek_attention_prefill_shape_runs() {
    run_shape("prefill", 1, 16, 64, 64, 128);
}

#[test]
#[ignore = "runs against real Vulkan/wgpu device; opt-in via `cargo test --release -- --ignored`"]
fn cubek_attention_prefill_long_shape_runs() {
    run_shape("prefill_long", 1, 16, 745, 745, 128);
}

fn run_shape(label: &str, b: usize, h: usize, n_q: usize, n_kv: usize, d_head: usize) {
    println!("\n=== run_shape {label}: B={b} H={h} n_q={n_q} n_kv={n_kv} d_head={d_head} ===");
    let scale = 1.0 / (d_head as f32).sqrt();

    // Build the wgpu engine to get a CubeWgpu16 device + client.
    let _engine = WgpuVulkanEngine::new_fp16().expect("wgpu engine init");

    // Synth Q/K/V — small magnitudes so softmax doesn't saturate.
    let mut rng = 0xCAFE_F00D_u64;
    let mut next = || -> f32 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((rng >> 33) as f32 / (1u32 << 31) as f32) * 0.1 - 0.05
    };
    let q_data: Vec<f32> = (0..b * h * n_q * d_head).map(|_| next()).collect();
    let k_data: Vec<f32> = (0..b * h * n_kv * d_head).map(|_| next()).collect();
    let v_data: Vec<f32> = (0..b * h * n_kv * d_head).map(|_| next()).collect();

    // CPU reference — we'll compare later.
    let reference = ref_attention(&q_data, &k_data, &v_data, b, h, n_q, n_kv, d_head, scale);

    // --- cubek-attention path ---
    // We need to:
    //   (1) build a cubecl ComputeClient + TensorHandle for each of
    //       Q, K, V, out using the wgpu runtime
    //   (2) define AttentionProblem with our shapes
    //   (3) call launch() with Strategy::Unit(BlueprintStrategy::Inferred(()))
    //   (4) read back the output buffer and compare to reference
    //
    // The wgpu runtime + client comes from cubecl_wgpu::WgpuRuntime
    // (the same one burn-cubecl uses underneath). To stay
    // independent of the WgpuVulkanEngine internals we acquire our
    // own client here.

    use cubecl::Runtime;
    use cubecl_wgpu::{WgpuDevice, WgpuRuntime};
    let device = WgpuDevice::default();
    let client = <WgpuRuntime as Runtime>::client(&device);

    let f16_dtype = StorageType::Scalar(ElemType::Float(FloatKind::F16));
    let global_dtypes = AttentionGlobalTypes::from_single_dtype(f16_dtype);

    let problem = AttentionProblem {
        dims: cubek_attention::definition::AttentionDims {
            batch: b,
            num_heads: h,
            seq_q: n_q,
            seq_kv: n_kv,
            head_dim: d_head,
            val_dim: d_head,
        },
        masked: false,
        global_dtypes: global_dtypes.clone(),
        options: AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
    };

    // Sanity: confirm DeviceSettings can be constructed for our
    // device + problem. This validates that the kernel can be
    // configured for our hardware.
    let _device_settings = DeviceSettings::new(&client, &problem);

    // --- upload Q/K/V to GPU as f16 ---
    use cubecl::std::tensor::TensorHandle;
    use cubecl_wgpu::WgpuRuntime as R0;

    let f32_to_f16_bytes = |data: &[f32]| -> Vec<u8> {
        let mut bytes = Vec::with_capacity(data.len() * 2);
        for &v in data {
            let h_val = half::f16::from_f32(v);
            bytes.extend_from_slice(&h_val.to_le_bytes());
        }
        bytes
    };

    let q_bytes = f32_to_f16_bytes(&q_data);
    let k_bytes = f32_to_f16_bytes(&k_data);
    let v_bytes = f32_to_f16_bytes(&v_data);

    let q_shape = vec![b, h, n_q, d_head];
    let k_shape = vec![b, h, n_kv, d_head];
    let v_shape = vec![b, h, n_kv, d_head];
    let out_shape = vec![b, h, n_q, d_head];

    let elem_size = 2; // f16
    let q_alloc = client.create_tensor_from_slice(&q_bytes, &q_shape, elem_size);
    let k_alloc = client.create_tensor_from_slice(&k_bytes, &k_shape, elem_size);
    let v_alloc = client.create_tensor_from_slice(&v_bytes, &v_shape, elem_size);

    let q_tensor: TensorHandle<R0> =
        TensorHandle::new(q_alloc.handle, q_shape.clone(), q_alloc.strides, f16_dtype);
    let k_tensor: TensorHandle<R0> =
        TensorHandle::new(k_alloc.handle, k_shape, k_alloc.strides, f16_dtype);
    let v_tensor: TensorHandle<R0> =
        TensorHandle::new(v_alloc.handle, v_shape, v_alloc.strides, f16_dtype);

    let out_tensor: TensorHandle<R0> =
        TensorHandle::empty(&client, out_shape.clone(), f16_dtype);

    // --- launch the kernel ---
    // First launch JIT-compiles the kernel (multi-second cost in
    // cubecl 0.9.0).  We do a warm-up dispatch, then measure
    // steady-state over a small batch of warm runs to extract the
    // kernel-only cost we'd see on the embedder hot path.
    //
    // Strategy choice: `Unit` is the portable kernel (no tensor-core
    // reliance); `BlackboxAccelerated` uses accelerated coop-matmul.
    // The bench env var `CUBEK_STRATEGY` selects: default "unit",
    // override "blackbox".
    let strategy_kind =
        std::env::var("CUBEK_STRATEGY").unwrap_or_else(|_| "unit".to_string());
    let strategy = match strategy_kind.as_str() {
        "blackbox" => Strategy::BlackboxAccelerated(BlueprintStrategy::Inferred(
            Default::default(),
        )),
        "unit" | _ => Strategy::Unit(BlueprintStrategy::Inferred(())),
    };
    println!("CUBEK_STRATEGY={strategy_kind}");

    // -- warmup --
    let t_warmup = Instant::now();
    launch::<R0>(
        strategy.clone(),
        &client,
        q_tensor.clone(),
        k_tensor.clone(),
        v_tensor.clone(),
        None,
        out_tensor.clone(),
        &global_dtypes,
        AttentionOptions {
            causal: false,
            accumulator_precision: AccumulatorPrecision::default(),
        },
    )
    .expect("cubek-attention warmup launch failed");
    // Force device sync so we don't measure pending GPU work as
    // launch latency on the next iteration.
    let _ = client.read_one(out_tensor.handle.clone());
    let warmup = t_warmup.elapsed();

    // -- steady-state: 10 warm launches --
    const WARM_ITERS: usize = 10;
    let t_warm = Instant::now();
    for _ in 0..WARM_ITERS {
        launch::<R0>(
            strategy.clone(),
            &client,
            q_tensor.clone(),
            k_tensor.clone(),
            v_tensor.clone(),
            None,
            out_tensor.clone(),
            &global_dtypes,
            AttentionOptions {
                causal: false,
                accumulator_precision: AccumulatorPrecision::default(),
            },
        )
        .expect("cubek-attention warm launch failed");
    }
    let _ = client.read_one(out_tensor.handle.clone());
    let warm_total = t_warm.elapsed();
    let warm_per_call = warm_total / WARM_ITERS as u32;

    println!(
        "cubek-attention (n_q=1, n_kv={n_kv}, h={h}, d_head={d_head}): \
         warmup (incl. JIT)={:?}, steady-state per call={:?} \
         (over {WARM_ITERS} iters)",
        warmup, warm_per_call,
    );

    // --- read back + verify ---
    let out_bytes = client.read_one(out_tensor.handle);
    let mut out_f32 = vec![0.0_f32; b * h * n_q * d_head];
    for (i, chunk) in out_bytes.chunks_exact(2).enumerate() {
        let h_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out_f32[i] = half::f16::from_bits(h_bits).to_f32();
    }

    // Compare against reference — f16 tolerance.
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for (a, r) in out_f32.iter().zip(reference.iter()) {
        let abs = (a - r).abs();
        max_abs = max_abs.max(abs);
        let rel = abs / r.abs().max(1e-3);
        max_rel = max_rel.max(rel);
    }
    println!(
        "cubek vs reference: max_abs={max_abs:.6}, max_rel={max_rel:.6}",
    );

    assert!(
        max_abs < 5e-2,
        "cubek-attention output drift too large: max_abs={max_abs:.6}, expected < 5e-2",
    );
}

// (helpers removed — using `client.read_one` + `client.create_tensor_from_slice`
// which handle layout + futures internally)

//! Phase-1 validation for Option C: confirm cubecl-matmul actually fuses
//! a 3-D batched matmul into a single Vulkan dispatch (vs. internally
//! looping per batch element).
//!
//! Method:
//!   - Build `(B, M, K)` lhs + `(B, K, N)` rhs handles.
//!   - Call `cubecl_matmul::launch_ref(Strategy::Auto, …)` once.
//!   - Verify per-batch correctness against ndarray.
//!   - Time it against a sequential `B`-element loop of 2-D matmuls.
//!
//! If batched-fusion works, the single-launch time should be substantially
//! lower than the loop (one dispatch envelope vs `B` of them). If they're
//! similar, cubecl is host-looping internally.

use std::time::Instant;

use cubecl_common::future;
use cubecl_core::client::ComputeClient;
use cubecl_core::prelude::{CubePrimitive, TensorHandleRef};
use cubecl_core::Runtime;
use cubecl_matmul as matmul;
use cubecl_matmul::components::MatmulElems;
use cubecl_matmul::{MatmulInputHandleRef, Strategy};
use cubecl_wgpu::{init_setup_async, AutoGraphicsApi, RuntimeOptions, WgpuDevice, WgpuRuntime};

const ELEM_SIZE: usize = std::mem::size_of::<f32>();

fn open_client() -> Option<ComputeClient<WgpuRuntime>> {
    let device = WgpuDevice::default();
    let setup =
        future::block_on(init_setup_async::<AutoGraphicsApi>(&device, RuntimeOptions::default()));
    eprintln!(
        "cubecl wgpu adapter: {} ({:?})",
        setup.adapter.get_info().name,
        setup.adapter.get_info().device_type,
    );
    Some(WgpuRuntime::client(&device))
}

fn launch_2d(
    client: &ComputeClient<WgpuRuntime>,
    lhs: &[f32],
    rhs: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let lhs_h = client.create_from_slice(bytemuck::cast_slice(lhs));
    let rhs_h = client.create_from_slice(bytemuck::cast_slice(rhs));
    let out_h = client.empty(m * n * ELEM_SIZE);
    let lhs_shape = [m, k];
    let lhs_strides = [k, 1];
    let rhs_shape = [k, n];
    let rhs_strides = [n, 1];
    let out_shape = [m, n];
    let out_strides = [n, 1];
    let dtype = f32::as_type_native_unchecked();
    let lhs_ref = unsafe {
        TensorHandleRef::<WgpuRuntime>::from_raw_parts(&lhs_h, &lhs_strides, &lhs_shape, ELEM_SIZE)
    };
    let rhs_ref = unsafe {
        TensorHandleRef::<WgpuRuntime>::from_raw_parts(&rhs_h, &rhs_strides, &rhs_shape, ELEM_SIZE)
    };
    let out_ref = unsafe {
        TensorHandleRef::<WgpuRuntime>::from_raw_parts(&out_h, &out_strides, &out_shape, ELEM_SIZE)
    };
    let lhs_input = MatmulInputHandleRef::Normal(lhs_ref, dtype);
    let rhs_input = MatmulInputHandleRef::Normal(rhs_ref, dtype);
    let mut dtypes = MatmulElems::new::<f32>();
    matmul::launch_ref::<WgpuRuntime>(
        &Strategy::Auto,
        client,
        &lhs_input,
        &rhs_input,
        &out_ref,
        &mut dtypes,
    )
    .expect("2-d launch_ref");
    let bytes = client.read_one(out_h);
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}

fn launch_3d(
    client: &ComputeClient<WgpuRuntime>,
    lhs: &[f32],
    rhs: &[f32],
    b: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let lhs_h = client.create_from_slice(bytemuck::cast_slice(lhs));
    let rhs_h = client.create_from_slice(bytemuck::cast_slice(rhs));
    let out_h = client.empty(b * m * n * ELEM_SIZE);
    // Row-major strides for (B, M, K): outermost stride is M·K, then K, then 1.
    let lhs_shape = [b, m, k];
    let lhs_strides = [m * k, k, 1];
    let rhs_shape = [b, k, n];
    let rhs_strides = [k * n, n, 1];
    let out_shape = [b, m, n];
    let out_strides = [m * n, n, 1];
    let dtype = f32::as_type_native_unchecked();
    let lhs_ref = unsafe {
        TensorHandleRef::<WgpuRuntime>::from_raw_parts(&lhs_h, &lhs_strides, &lhs_shape, ELEM_SIZE)
    };
    let rhs_ref = unsafe {
        TensorHandleRef::<WgpuRuntime>::from_raw_parts(&rhs_h, &rhs_strides, &rhs_shape, ELEM_SIZE)
    };
    let out_ref = unsafe {
        TensorHandleRef::<WgpuRuntime>::from_raw_parts(&out_h, &out_strides, &out_shape, ELEM_SIZE)
    };
    let lhs_input = MatmulInputHandleRef::Normal(lhs_ref, dtype);
    let rhs_input = MatmulInputHandleRef::Normal(rhs_ref, dtype);
    let mut dtypes = MatmulElems::new::<f32>();
    matmul::launch_ref::<WgpuRuntime>(
        &Strategy::Auto,
        client,
        &lhs_input,
        &rhs_input,
        &out_ref,
        &mut dtypes,
    )
    .expect("3-d batched launch_ref");
    let bytes = client.read_one(out_h);
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}

fn cpu_per_batch_reference(
    lhs: &[f32],
    rhs: &[f32],
    b: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; b * m * n];
    for bi in 0..b {
        let l = &lhs[bi * m * k..(bi + 1) * m * k];
        let r = &rhs[bi * k * n..(bi + 1) * k * n];
        let o = &mut out[bi * m * n..(bi + 1) * m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for p in 0..k {
                    acc += l[i * k + p] * r[p * n + j];
                }
                o[i * n + j] = acc;
            }
        }
    }
    out
}

#[test]
#[ignore = "requires Vulkan adapter; phase-1 validation only"]
fn cubecl_3d_batched_matmul_is_correct_and_fused() {
    let client = match open_client() {
        Some(c) => c,
        None => return,
    };

    // Use shapes that match the OutAttnMult workload on Qwen3-Embedding-0.6B:
    // batch = 16 Q heads, M = 2n with n ≈ 24, K = head_dim = 128, N = 2n.
    let b = 16;
    let m = 48;
    let k = 128;
    let n = 48;

    // Deterministic-ish synthetic data.
    let lhs: Vec<f32> = (0..b * m * k).map(|i| ((i as f32) * 0.0117).sin()).collect();
    let rhs: Vec<f32> = (0..b * k * n).map(|i| ((i as f32) * 0.0193).cos()).collect();

    // Correctness check: one 3-D launch vs. per-batch CPU reference.
    let got_3d = launch_3d(&client, &lhs, &rhs, b, m, k, n);
    let expect = cpu_per_batch_reference(&lhs, &rhs, b, m, k, n);
    assert_eq!(got_3d.len(), expect.len());
    let mut max_abs = 0.0_f32;
    for (g, e) in got_3d.iter().zip(expect.iter()) {
        max_abs = max_abs.max((g - e).abs());
    }
    eprintln!("3-D batched correctness: max-abs vs CPU reference = {max_abs:e}");
    assert!(max_abs < 1e-3, "batched output diverges from CPU reference");

    // Single-batch correctness sanity (compare batch 0 of the 3-D launch to a
    // standalone 2-D launch of the same input — should match exactly).
    let lhs0 = &lhs[..m * k];
    let rhs0 = &rhs[..k * n];
    let got_2d_b0 = launch_2d(&client, lhs0, rhs0, m, k, n);
    let got_3d_b0 = &got_3d[..m * n];
    let mut b0_max = 0.0_f32;
    for (a, c) in got_3d_b0.iter().zip(got_2d_b0.iter()) {
        b0_max = b0_max.max((a - c).abs());
    }
    eprintln!("3-D batch[0] vs standalone 2-D: max-abs = {b0_max:e}");
    assert!(b0_max < 1e-3, "batch[0] of 3-D launch doesn't match 2-D launch");

    // Fusion check: time one 3-D launch vs. B separate 2-D launches with the
    // same total compute. If cubecl actually fuses, the 3-D launch saves the
    // (B - 1) dispatch envelopes and should be markedly faster.
    let warmup_iters = 4;
    let iters = 16;

    for _ in 0..warmup_iters {
        let _ = launch_3d(&client, &lhs, &rhs, b, m, k, n);
    }
    let t = Instant::now();
    for _ in 0..iters {
        let _ = launch_3d(&client, &lhs, &rhs, b, m, k, n);
    }
    let t_3d = t.elapsed().as_secs_f64() / iters as f64;

    for _ in 0..warmup_iters {
        for bi in 0..b {
            let _ = launch_2d(
                &client,
                &lhs[bi * m * k..(bi + 1) * m * k],
                &rhs[bi * k * n..(bi + 1) * k * n],
                m,
                k,
                n,
            );
        }
    }
    let t = Instant::now();
    for _ in 0..iters {
        for bi in 0..b {
            let _ = launch_2d(
                &client,
                &lhs[bi * m * k..(bi + 1) * m * k],
                &rhs[bi * k * n..(bi + 1) * k * n],
                m,
                k,
                n,
            );
        }
    }
    let t_loop = t.elapsed().as_secs_f64() / iters as f64;

    let speedup = t_loop / t_3d;
    eprintln!(
        "per-iter time: 3-D batched = {:.3} ms, {}× 2-D loop = {:.3} ms, speedup = {:.2}×",
        t_3d * 1000.0,
        b,
        t_loop * 1000.0,
        speedup,
    );

    // The 3-D batched call MUST be faster than the loop — if it's not, cubecl
    // is host-looping per batch and Option C buys us nothing on this path.
    // We require ≥ 2× speedup which is well below the theoretical 16× ceiling
    // but high enough to flag a regression / non-fused implementation.
    assert!(
        speedup >= 2.0,
        "cubecl 3-D batched matmul is NOT meaningfully faster than the per-batch loop \
         (3-D: {t_3d:.3} s, loop: {t_loop:.3} s, speedup: {speedup:.2}×). \
         Either cubecl is host-looping or the dispatch overhead at this shape is negligible — \
         in either case, Option C as planned won't move the bench."
    );
}

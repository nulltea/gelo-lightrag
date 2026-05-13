use ndarray::{Array2, ArrayView2, Axis};

/// SwiGLU activation used by Qwen3 / LLaMA FFNs.
///   out = silu(gate) ⊙ up
/// where `silu(x) = x · σ(x) = x / (1 + exp(-x))`.
///
/// Caller has already done the two matmuls `gate = H · W_gate` and
/// `up = H · W_up` and supplies them here as inputs of identical shape.
///
/// Hot inner loop tightened in Q1: pull contiguous &[f32] row slices,
/// inline the silu computation (avoid the function-call abstraction
/// barrier so LLVM can interleave the two `exp` libcalls across pairs
/// of elements), single-statement write of `silu(g) * u` so the compiler
/// can schedule the FMA. Single-threaded — per-call work is too small
/// to amortise rayon overhead at our per-layer shape (n × 3072).
pub fn swiglu(gate: ArrayView2<'_, f32>, up: ArrayView2<'_, f32>) -> Array2<f32> {
    assert_eq!(gate.shape(), up.shape(), "swiglu: gate and up shapes must match");
    let mut out = Array2::<f32>::zeros(gate.raw_dim());
    for ((mut dst, g_row), u_row) in out
        .axis_iter_mut(Axis(0))
        .zip(gate.axis_iter(Axis(0)))
        .zip(up.axis_iter(Axis(0)))
    {
        let dst_slice = dst
            .as_slice_mut()
            .expect("Array2 rows are contiguous in row-major");
        let g_slice = g_row
            .to_slice()
            .expect("Array2 rows are contiguous in row-major");
        let u_slice = u_row
            .to_slice()
            .expect("Array2 rows are contiguous in row-major");
        for ((d_v, &g), &u) in dst_slice
            .iter_mut()
            .zip(g_slice.iter())
            .zip(u_slice.iter())
        {
            // Inlined silu: g * sigmoid(g) = g / (1 + exp(-g))
            *d_v = g / (1.0 + (-g).exp()) * u;
        }
    }
    out
}

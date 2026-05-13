use ndarray::{Array2, ArrayView2, Axis};

/// SwiGLU activation used by Qwen3 / LLaMA FFNs.
///   out = silu(gate) ⊙ up
/// where `silu(x) = x · σ(x) = x / (1 + exp(-x))`.
///
/// Caller has already done the two matmuls `gate = H · W_gate` and
/// `up = H · W_up` and supplies them here as inputs of identical shape.
/// Single-threaded; per-element work auto-vectorises in LLVM and rayon
/// overhead at our per-call shape costs more than it saves.
pub fn swiglu(gate: ArrayView2<'_, f32>, up: ArrayView2<'_, f32>) -> Array2<f32> {
    assert_eq!(gate.shape(), up.shape(), "swiglu: gate and up shapes must match");
    let mut out = Array2::<f32>::zeros(gate.raw_dim());
    for ((mut dst, g_row), u_row) in out
        .axis_iter_mut(Axis(0))
        .zip(gate.axis_iter(Axis(0)))
        .zip(up.axis_iter(Axis(0)))
    {
        for ((d_v, &g), &u) in dst.iter_mut().zip(g_row.iter()).zip(u_row.iter()) {
            *d_v = silu(g) * u;
        }
    }
    out
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

use ndarray::{Array2, ArrayView2};

/// SwiGLU activation used by Qwen3 / LLaMA FFNs.
///   out = silu(gate) ⊙ up
/// where `silu(x) = x · σ(x) = x / (1 + exp(-x))`.
///
/// Caller has already done the two matmuls `gate = H · W_gate` and
/// `up = H · W_up` and supplies them here as inputs of identical shape.
pub fn swiglu(gate: ArrayView2<'_, f32>, up: ArrayView2<'_, f32>) -> Array2<f32> {
    assert_eq!(gate.shape(), up.shape(), "swiglu: gate and up shapes must match");
    let mut out = Array2::<f32>::zeros(gate.raw_dim());
    for ((i, j), g) in gate.indexed_iter() {
        let u = up[[i, j]];
        let s = silu(*g);
        out[[i, j]] = s * u;
    }
    out
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

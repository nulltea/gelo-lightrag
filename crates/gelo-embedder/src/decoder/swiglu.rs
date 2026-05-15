use ndarray::{Array2, ArrayView2};

/// SwiGLU activation used by Qwen3 / LLaMA FFNs.
///   out = silu(gate) ⊙ up
/// where `silu(x) = x · σ(x) = x / (1 + exp(-x))`.
///
/// Caller has already done the two matmuls `gate = H · W_gate` and
/// `up = H · W_up` and supplies them here as inputs of identical shape.
///
/// Total-elements threshold above which we hand the elementwise loop to
/// rayon — matches `bert::forward::PAR_THRESHOLD_ELEMS = 32 768`.
/// Embedder shape (n ≈ 30 × 3072 ≈ 92 k) already clears it; rerank
/// shape (n ≈ 400 × 3072 ≈ 1.2 M elements per call × 28 layers) is
/// the regime where the parallel speedup is biggest.
pub fn swiglu(gate: ArrayView2<'_, f32>, up: ArrayView2<'_, f32>) -> Array2<f32> {
    assert_eq!(gate.shape(), up.shape(), "swiglu: gate and up shapes must match");
    let mut out = Array2::<f32>::zeros(gate.raw_dim());

    let g_slice = gate.as_slice().expect("gate must be row-major contiguous");
    let u_slice = up.as_slice().expect("up must be row-major contiguous");
    let o_slice = out.as_slice_mut().expect("fresh Array2 is contiguous");
    // Inlined silu: g * sigmoid(g) = g / (1 + exp(-g)); the compiler
    // can schedule the FMA across the single elementwise write.
    let f = |((d_v, &g), &u): ((&mut f32, &f32), &f32)| {
        *d_v = g / (1.0 + (-g).exp()) * u;
    };
    const PAR_THRESHOLD: usize = 32_768;
    if o_slice.len() >= PAR_THRESHOLD {
        use rayon::prelude::*;
        o_slice
            .par_iter_mut()
            .zip(g_slice.par_iter())
            .zip(u_slice.par_iter())
            .for_each(f);
    } else {
        for triple in o_slice.iter_mut().zip(g_slice.iter()).zip(u_slice.iter()) {
            f(triple);
        }
    }
    out
}

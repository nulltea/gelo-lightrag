use ndarray::{Array2, ArrayViewMut2};

/// Llama / Qwen-style rotary position embedding tables.
///
/// For each position `p ∈ [0, max_pos)` and frequency index `i ∈ [0, head_dim/2)`,
/// the table stores `cos(p · θ_i)` and `sin(p · θ_i)` where
/// `θ_i = base^(-2i / head_dim)`.
///
/// Applied to per-head vectors `x ∈ R^head_dim` as the half-rotation
///   x_low_rot  = x_low  · cos − x_high · sin
///   x_high_rot = x_low  · sin + x_high · cos
/// with `x_low = x[..d/2]`, `x_high = x[d/2..]`.
pub struct RopeTables {
    head_dim: usize,
    cos: Array2<f32>, // (max_pos, head_dim/2)
    sin: Array2<f32>, // (max_pos, head_dim/2)
}

impl RopeTables {
    pub fn new(head_dim: usize, max_positions: usize, base: f32) -> Self {
        assert!(head_dim.is_multiple_of(2), "RoPE requires even head_dim");
        let half = head_dim / 2;
        let mut cos = Array2::<f32>::zeros((max_positions, half));
        let mut sin = Array2::<f32>::zeros((max_positions, half));
        for i in 0..half {
            let exponent = (2 * i) as f32 / head_dim as f32;
            let theta_i = base.powf(-exponent);
            for p in 0..max_positions {
                let angle = (p as f32) * theta_i;
                cos[[p, i]] = angle.cos();
                sin[[p, i]] = angle.sin();
            }
        }
        Self { head_dim, cos, sin }
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Apply RoPE in-place to `x` shaped `(n, num_heads * head_dim)`. Each
    /// head's `head_dim` slice is rotated by its position's angle.
    ///
    /// Row `i` is rotated as if it sits at absolute position `i`. For
    /// decode-step inputs where the new token's position is `n_cache`,
    /// use [`Self::apply_at`] with `start_pos = n_cache`.
    pub fn apply(&self, x: ArrayViewMut2<'_, f32>, num_heads: usize) {
        self.apply_at(x, num_heads, 0)
    }

    /// Apply RoPE in-place treating row `i` as absolute position
    /// `start_pos + i`. Used by the decode path: a single-row Q/K input
    /// at position `n_cache` rotates with the cos/sin entry for that
    /// absolute index, not for index 0.
    pub fn apply_at(
        &self,
        x: ArrayViewMut2<'_, f32>,
        num_heads: usize,
        start_pos: usize,
    ) {
        let head_dim = self.head_dim;
        self.apply_partial_at(x, num_heads, start_pos, head_dim)
    }

    /// Partial-rotation RoPE: rotate only the first `rotated_dim` of
    /// each head, leaving the remaining `head_dim − rotated_dim` dims
    /// untouched. Gemma 4 global layers use `rotated_dim = 0.25 *
    /// head_dim` per the p-RoPE recipe.
    ///
    /// Convention matches the GELO doc + the HF Gemma reference: the
    /// low/high halves are taken from the rotated-prefix slice, so
    /// rotation operates on `x[0..rotated_dim]` as two halves of length
    /// `rotated_dim / 2`. `rotated_dim` must be even (the config's
    /// `rotated_dim()` enforces this via snap-to-even).
    pub fn apply_partial_at(
        &self,
        mut x: ArrayViewMut2<'_, f32>,
        num_heads: usize,
        start_pos: usize,
        rotated_dim: usize,
    ) {
        let n = x.nrows();
        let head_dim = self.head_dim;
        assert_eq!(
            x.ncols(),
            num_heads * head_dim,
            "rope.apply_partial_at: ncols must equal num_heads × head_dim",
        );
        assert!(
            rotated_dim <= head_dim,
            "rope.apply_partial_at: rotated_dim {rotated_dim} > head_dim {head_dim}",
        );
        assert!(
            rotated_dim & 1 == 0,
            "rope.apply_partial_at: rotated_dim {rotated_dim} must be even",
        );
        if rotated_dim == 0 {
            return; // identity — nothing to rotate.
        }
        let half = rotated_dim / 2;
        let max_pos = self.cos.nrows();
        assert!(
            start_pos + n <= max_pos,
            "rope.apply_partial_at: start_pos {start_pos} + n {n} exceeds precomputed max_positions {max_pos}",
        );
        for i in 0..n {
            let abs_pos = start_pos + i;
            let cos_row = self.cos.row(abs_pos);
            let sin_row = self.sin.row(abs_pos);
            let mut row = x.row_mut(i);
            for h in 0..num_heads {
                let off = h * head_dim;
                for j in 0..half {
                    let lo = row[off + j];
                    let hi = row[off + half + j];
                    let c = cos_row[j];
                    let s = sin_row[j];
                    row[off + j] = lo * c - hi * s;
                    row[off + half + j] = lo * s + hi * c;
                }
                // dims [rotated_dim..head_dim] are untouched (identity
                // pass-through). Skipping the inner loop is the entire
                // semantic difference between p-RoPE and full RoPE.
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn apply_at_zero_matches_apply() {
        let rope = RopeTables::new(8, 32, 10_000.0);
        let n = 5;
        let num_heads = 2;
        let mut a = Array2::<f32>::zeros((n, num_heads * 8));
        for i in 0..n {
            for j in 0..num_heads * 8 {
                a[[i, j]] = ((i * 8 + j) as f32 * 0.1).sin();
            }
        }
        let mut b = a.clone();
        rope.apply(a.view_mut(), num_heads);
        rope.apply_at(b.view_mut(), num_heads, 0);
        assert_eq!(a, b);
    }

    #[test]
    fn apply_partial_at_full_dim_matches_apply_at() {
        let rope = RopeTables::new(8, 32, 10_000.0);
        let n = 4;
        let num_heads = 2;
        let mut a = Array2::<f32>::zeros((n, num_heads * 8));
        for i in 0..n {
            for j in 0..num_heads * 8 {
                a[[i, j]] = ((i * 8 + j) as f32 * 0.07).cos();
            }
        }
        let mut b = a.clone();
        rope.apply_at(a.view_mut(), num_heads, 3);
        rope.apply_partial_at(b.view_mut(), num_heads, 3, 8);
        assert_eq!(a, b);
    }

    #[test]
    fn apply_partial_at_zero_dim_is_identity() {
        let rope = RopeTables::new(8, 32, 10_000.0);
        let n = 3;
        let num_heads = 1;
        let mut x = Array2::<f32>::zeros((n, 8));
        for i in 0..n {
            for j in 0..8 {
                x[[i, j]] = (i * 8 + j) as f32;
            }
        }
        let before = x.clone();
        rope.apply_partial_at(x.view_mut(), num_heads, 0, 0);
        assert_eq!(before, x, "rotated_dim=0 must be identity");
    }

    #[test]
    fn apply_partial_at_passes_untouched_dims_through() {
        // Rotate only the first 4 of head_dim=8. Dims 4..8 must come
        // out byte-identical to the input.
        let rope = RopeTables::new(8, 32, 10_000.0);
        let n = 3;
        let num_heads = 1;
        let rotated = 4;
        let mut input = Array2::<f32>::zeros((n, 8));
        for i in 0..n {
            for j in 0..8 {
                input[[i, j]] = (i * 10 + j) as f32 * 0.5;
            }
        }
        let mut x = input.clone();
        rope.apply_partial_at(x.view_mut(), num_heads, 1, rotated);

        for i in 0..n {
            for j in rotated..8 {
                assert_eq!(
                    x[[i, j]],
                    input[[i, j]],
                    "dim {j} (≥ rotated_dim) must pass through unchanged at row {i}",
                );
            }
        }
        // Sanity: at least one of the rotated dims actually changed.
        let any_rotated = (0..rotated).any(|j| (x[[1, j]] - input[[1, j]]).abs() > 1e-6);
        assert!(any_rotated, "rotated prefix didn't actually rotate");
    }

    #[test]
    fn apply_at_offset_matches_full_prefill_slice() {
        // Rotating a single row at absolute position p must equal what
        // you'd get by rotating a length-(p+1) batch and taking the last
        // row. This is the property the decode path relies on.
        let rope = RopeTables::new(8, 32, 10_000.0);
        let p = 7;
        let num_heads = 3;
        let mut full = Array2::<f32>::zeros((p + 1, num_heads * 8));
        for i in 0..p + 1 {
            for j in 0..num_heads * 8 {
                full[[i, j]] = ((i * 8 + j) as f32 * 0.13).cos();
            }
        }
        let last_row_pre = full.row(p).to_owned();
        rope.apply(full.view_mut(), num_heads);
        let last_row_full = full.row(p).to_owned();

        let mut single = Array2::<f32>::zeros((1, num_heads * 8));
        single.row_mut(0).assign(&last_row_pre);
        rope.apply_at(single.view_mut(), num_heads, p);

        for j in 0..num_heads * 8 {
            assert!(
                (last_row_full[j] - single[[0, j]]).abs() < 1e-6,
                "mismatch at col {j}: full={} single_at_p={}",
                last_row_full[j],
                single[[0, j]],
            );
        }
    }
}

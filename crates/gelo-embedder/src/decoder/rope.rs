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
        mut x: ArrayViewMut2<'_, f32>,
        num_heads: usize,
        start_pos: usize,
    ) {
        let n = x.nrows();
        let head_dim = self.head_dim;
        let half = head_dim / 2;
        assert_eq!(
            x.ncols(),
            num_heads * head_dim,
            "rope.apply_at: ncols must equal num_heads × head_dim",
        );
        let max_pos = self.cos.nrows();
        assert!(
            start_pos + n <= max_pos,
            "rope.apply_at: start_pos {start_pos} + n {n} exceeds precomputed max_positions {max_pos}",
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

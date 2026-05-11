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
    pub fn apply(&self, mut x: ArrayViewMut2<'_, f32>, num_heads: usize) {
        let n = x.nrows();
        let head_dim = self.head_dim;
        let half = head_dim / 2;
        assert_eq!(
            x.ncols(),
            num_heads * head_dim,
            "rope.apply: ncols must equal num_heads × head_dim",
        );
        for pos in 0..n {
            let cos_row = self.cos.row(pos);
            let sin_row = self.sin.row(pos);
            let mut row = x.row_mut(pos);
            for h in 0..num_heads {
                let off = h * head_dim;
                for i in 0..half {
                    let lo = row[off + i];
                    let hi = row[off + half + i];
                    let c = cos_row[i];
                    let s = sin_row[i];
                    row[off + i] = lo * c - hi * s;
                    row[off + half + i] = lo * s + hi * c;
                }
            }
        }
    }
}

//! Shield-vector defence against the Gram-matrix leak of bare orthogonal
//! masks (GELO §4.2).
//!
//! Sampling: high-energy random rows are stacked onto the original hidden
//! state before the per-batch orthogonal mask is applied. After unmask, the
//! shield rows are stripped. Because the mask is fresh and the shield is
//! drawn from `N(0, σ²I)` with `σ ≈ energy_scale · mean ‖h‖`, the per-batch
//! Gram matrix `UᵀU = HᵀH + SᵀS` no longer leaks `HᵀH` directly.

use ndarray::{Array2, ArrayView2};
use rand::RngCore;

use crate::gaussian::fill_gaussian;

#[derive(Debug, Clone, Copy)]
pub struct ShieldConfig {
    /// Number of shield rows to splice in. `k = 0` disables shielding.
    pub k: usize,
    /// Multiplier vs. the mean row-norm of `H`. The paper recommends 4–8×.
    pub energy_scale: f32,
}

impl ShieldConfig {
    pub const NONE: Self = Self {
        k: 0,
        energy_scale: 0.0,
    };

    pub const fn new(k: usize, energy_scale: f32) -> Self {
        Self { k, energy_scale }
    }

    /// Effective shielding is gated on `k > 0`.
    pub const fn enabled(&self) -> bool {
        self.k > 0
    }
}

impl Default for ShieldConfig {
    fn default() -> Self {
        Self::NONE
    }
}

/// Stack `k` high-energy random rows below `hidden`, return the resulting
/// `(n + k, d)` matrix and the number of original data rows `n`.
///
/// The shield rows are sampled from `N(0, σ²I)` with per-component
/// `σ = energy_scale · mean_row_norm(H) / sqrt(d)` so that each shield row's
/// expected L2 norm is `energy_scale · mean_row_norm(H)`.
pub fn stack_shield<R: RngCore>(
    hidden: ArrayView2<'_, f32>,
    cfg: ShieldConfig,
    rng: &mut R,
) -> (Array2<f32>, usize) {
    let n = hidden.nrows();
    if !cfg.enabled() {
        return (hidden.to_owned(), n);
    }
    let d = hidden.ncols();

    let mean_norm = mean_row_norm(hidden);
    let per_component_sigma = if d > 0 {
        cfg.energy_scale * mean_norm / (d as f32).sqrt()
    } else {
        0.0
    };

    let mut stacked = Array2::<f32>::zeros((n + cfg.k, d));
    for i in 0..n {
        stacked.row_mut(i).assign(&hidden.row(i));
    }
    // Fill the trailing `k × d` shield rows with `N(0, σ²)` in one
    // bulk SIMD pass.  `Array2::zeros` is always row-major contiguous,
    // so the shield slab is a contiguous `k·d` slice.
    if cfg.k > 0 && d > 0 {
        let shield_slab = stacked
            .slice_mut(ndarray::s![n.., ..])
            .into_slice()
            .expect("Array2 row-major shield slab is contiguous");
        fill_gaussian(shield_slab, per_component_sigma, rng);
    }
    (stacked, n)
}

/// Average L2 norm of the rows of `m`.
fn mean_row_norm(m: ArrayView2<'_, f32>) -> f32 {
    let n = m.nrows();
    if n == 0 {
        return 0.0;
    }
    let mut acc = 0.0_f32;
    for row in m.rows() {
        acc += row.iter().map(|v| v * v).sum::<f32>().sqrt();
    }
    acc / (n as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn disabled_stack_is_identity() {
        let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
        let h = Array2::<f32>::from_shape_fn((4, 3), |(_i, _j)| 1.0);
        let (stacked, n) = stack_shield(h.view(), ShieldConfig::NONE, &mut rng);
        assert_eq!(n, 4);
        assert_eq!(stacked.shape(), &[4, 3]);
        assert!(stacked.iter().all(|v| (*v - 1.0).abs() < 1e-9));
    }

    #[test]
    fn shield_rows_have_expected_energy() {
        let mut rng = ChaCha20Rng::from_seed([3u8; 32]);
        let d = 64;
        let n = 8;
        let h = Array2::<f32>::from_shape_fn((n, d), |(_i, j)| if j % 3 == 0 { 1.0 } else { 0.0 });
        let scale = 4.0;
        let (stacked, n_data) = stack_shield(h.view(), ShieldConfig::new(16, scale), &mut rng);
        assert_eq!(n_data, n);

        let mean_norm_orig = mean_row_norm(h.view());
        let target = scale * mean_norm_orig;

        // Sample shield row norms — expect within ~20% of target on average.
        let shield = stacked.slice(ndarray::s![n_data.., ..]);
        let mean_shield_norm = mean_row_norm(shield);
        assert!(
            (mean_shield_norm - target).abs() / target < 0.2,
            "shield norm {mean_shield_norm} far from target {target}",
        );
    }
}

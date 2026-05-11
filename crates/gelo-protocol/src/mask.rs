use ndarray::{Array2, ArrayView2};
use rand::RngCore;
use rand_distr::{Distribution, StandardNormal};

pub use crate::rng::MaskSeed;

/// Token-axis orthogonal mask used to obfuscate hidden states before the
/// untrusted side sees them.
///
/// For hidden states `H ∈ R^(n×d)` the trusted side computes `U = A·H` and
/// ships `U` to the engine. Because `A` is orthogonal, `A⁻¹ = Aᵀ` and
/// recovery is `H·W = Aᵀ·(U·W)`. `A` is sampled fresh per batch from the
/// Haar measure on `O(n)` using the Mezzadri 2007 QR-with-sign-correction
/// trick over a standard normal seed.
#[derive(Debug, Clone)]
pub struct GeloMask {
    /// `(n, n)` orthogonal mask matrix.
    a: Array2<f32>,
}

impl GeloMask {
    /// Sample a fresh Haar-uniform orthogonal `A ∈ R^(n×n)`.
    pub fn fresh<R: RngCore>(n: usize, rng: &mut R) -> Self {
        Self {
            a: sample_haar_orthogonal(n, rng),
        }
    }

    /// Deterministic constructor for tests.
    pub fn from_seed(n: usize, seed: MaskSeed) -> Self {
        let mut rng = seed.rng();
        Self::fresh(n, &mut rng)
    }

    /// `n`, the token-axis dimension this mask operates on.
    pub fn n(&self) -> usize {
        self.a.nrows()
    }

    /// Reference to the underlying `(n, n)` orthogonal matrix.
    pub fn matrix(&self) -> ArrayView2<'_, f32> {
        self.a.view()
    }

    /// Apply the mask: `U = A · H`.
    ///
    /// `hidden` must have shape `(n, d)` with `n == self.n()`.
    pub fn apply(&self, hidden: ArrayView2<'_, f32>) -> Array2<f32> {
        assert_eq!(
            hidden.nrows(),
            self.n(),
            "hidden row count must equal mask n"
        );
        self.a.dot(&hidden)
    }

    /// Remove the mask: `H·W = Aᵀ · (U·W)`.
    ///
    /// `masked_output` must have shape `(n, p)` with `n == self.n()`.
    pub fn unapply(&self, masked_output: ArrayView2<f32>) -> Array2<f32> {
        assert_eq!(
            masked_output.nrows(),
            self.n(),
            "masked output row count must equal mask n"
        );
        self.a.t().dot(&masked_output)
    }
}

/// Haar-uniform orthogonal sampler via Householder QR with Mezzadri-2007
/// sign correction. O(n³) work, O(n²) memory, no LAPACK dep.
fn sample_haar_orthogonal<R: RngCore>(n: usize, rng: &mut R) -> Array2<f32> {
    let normal = StandardNormal;
    let mut a = Array2::<f32>::from_shape_fn((n, n), |_| normal.sample(rng));
    let mut q = Array2::<f32>::eye(n);
    let mut v = vec![0.0f32; n];

    for k in 0..n.saturating_sub(1) {
        let mut sigma_sq: f32 = 0.0;
        for i in k..n {
            sigma_sq += a[[i, k]] * a[[i, k]];
        }
        let sigma = sigma_sq.sqrt();
        if sigma < 1e-30 {
            continue;
        }

        let sign = if a[[k, k]] >= 0.0 { 1.0 } else { -1.0 };
        let alpha = -sign * sigma;

        let v0 = a[[k, k]] - alpha;
        let mut v_norm_sq: f32 = v0 * v0;
        for i in (k + 1)..n {
            v_norm_sq += a[[i, k]] * a[[i, k]];
        }
        let v_norm = v_norm_sq.sqrt();
        if v_norm < 1e-30 {
            continue;
        }

        v[k] = v0 / v_norm;
        for i in (k + 1)..n {
            v[i] = a[[i, k]] / v_norm;
        }

        // A[k:, k:] -= 2 * v[k:] * (v[k:]ᵀ · A[k:, k:])
        for j in k..n {
            let mut dot: f32 = 0.0;
            for i in k..n {
                dot += v[i] * a[[i, j]];
            }
            let dot2 = 2.0 * dot;
            for i in k..n {
                a[[i, j]] -= dot2 * v[i];
            }
        }

        // Q[:, k:] -= 2 * (Q[:, k:] · v[k:]) * v[k:]ᵀ
        for r in 0..n {
            let mut dot: f32 = 0.0;
            for c in k..n {
                dot += q[[r, c]] * v[c];
            }
            let dot2 = 2.0 * dot;
            for c in k..n {
                q[[r, c]] -= dot2 * v[c];
            }
        }
    }

    // Mezzadri 2007: normalize so diag(R) >= 0, making the orthogonal output
    // Haar-uniform.
    for i in 0..n {
        if a[[i, i]] < 0.0 {
            for r in 0..n {
                q[[r, i]] = -q[[r, i]];
            }
        }
    }

    q
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn orthogonality() {
        let mut rng = rand_chacha::ChaCha20Rng::from_seed([7u8; 32]);
        let n = 32;
        let mask = GeloMask::fresh(n, &mut rng);
        let a = mask.matrix();
        let prod = a.t().dot(&a);
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(prod[[i, j]], expected, 1e-4),
                    "AᵀA[{i},{j}] = {} expected {}",
                    prod[[i, j]],
                    expected
                );
            }
        }
    }

    #[test]
    fn mask_round_trip_preserves_matmul() {
        // Verify that for any H, W:  Aᵀ · ((A·H) · W) ≈ H · W.
        let mut rng = rand_chacha::ChaCha20Rng::from_seed([11u8; 32]);
        let n = 16;
        let d = 12;
        let p = 8;
        let mask = GeloMask::fresh(n, &mut rng);

        let normal = StandardNormal;
        let h = Array2::<f32>::from_shape_fn((n, d), |_| normal.sample(&mut rng));
        let w = Array2::<f32>::from_shape_fn((d, p), |_| normal.sample(&mut rng));

        let plaintext = h.dot(&w);
        let masked = mask.apply(h.view());
        let masked_out = masked.dot(&w);
        let recovered = mask.unapply(masked_out.view());

        for ((i, j), pt) in plaintext.indexed_iter() {
            assert!(
                approx_eq(*pt, recovered[[i, j]], 1e-3),
                "mismatch at ({i},{j}): plain={pt} recovered={}",
                recovered[[i, j]]
            );
        }
    }

    #[test]
    fn deterministic_from_seed() {
        let seed = MaskSeed::from_bytes([42u8; 32]);
        let m1 = GeloMask::from_seed(8, seed);
        let m2 = GeloMask::from_seed(8, seed);
        let diff = &m1.a - &m2.a;
        let max_abs = diff.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
        assert_eq!(max_abs, 0.0);
    }

    use rand::SeedableRng;
}

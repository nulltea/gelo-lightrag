use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut2, Axis, s};
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
///
/// **Algorithm** — identical to the textbook Householder QR; the speedup
/// over the scalar reference comes from:
///   - Outer reduction (`σ² = Σ vᵢ²`) computed via slice sum, LLVM SIMDs it.
///   - The rank-1 sub-matrix update `A[k:, k:] -= 2 v (vᵀ A[k:, k:])`
///     decomposes into a GEMV `vᵀ A` (via `ndarray::dot`, which dispatches
///     to `matrixmultiply` — SIMD-vectorised + cache-tiled) followed by an
///     outer-product subtraction expressed as row-wise slice operations
///     (LLVM auto-vectorises the inner loop because the row is a
///     contiguous `&mut [f32]`).
///   - The Q accumulation has its rank-1 update mirrored on columns (Q's
///     stride is row-major so we operate on `Q[:, k:]` column-wise).
///
/// **Correctness invariant** — the output `Q ∈ R^(n×n)` is Haar-distributed
/// on O(n). Mezzadri-2007 sign correction (after the QR factorisation) is
/// load-bearing: without it, `Q` is orthogonal but not Haar-uniform, which
/// would weaken GELO's information-theoretic privacy argument.
fn sample_haar_orthogonal<R: RngCore>(n: usize, rng: &mut R) -> Array2<f32> {
    let normal = StandardNormal;
    let mut a = Array2::<f32>::from_shape_fn((n, n), |_| normal.sample(rng));
    let mut q = Array2::<f32>::eye(n);
    // Reusable storage for the Householder vector v[k..] and the GEMV
    // result `v^T A[k:, k:]` of length (n-k).
    let mut v_buf = vec![0.0f32; n];
    let mut dot_buf = vec![0.0f32; n];

    for k in 0..n.saturating_sub(1) {
        let m = n - k;
        // σ² = Σ_{i≥k} a[i, k]². Column-stride read in row-major: not
        // ideal, but only n elements per step — O(n²) total work, well
        // below the O(n³) rank-1 update cost we care about.
        let mut sigma_sq = 0.0_f32;
        for i in k..n {
            let x = a[[i, k]];
            sigma_sq += x * x;
        }
        let sigma = sigma_sq.sqrt();
        if sigma < 1e-30 {
            continue;
        }

        let a_kk = a[[k, k]];
        let sign = if a_kk >= 0.0 { 1.0 } else { -1.0 };
        let alpha = -sign * sigma;

        let v0 = a_kk - alpha;
        let mut v_norm_sq = v0 * v0;
        for i in (k + 1)..n {
            let x = a[[i, k]];
            v_norm_sq += x * x;
        }
        let v_norm = v_norm_sq.sqrt();
        if v_norm < 1e-30 {
            continue;
        }

        v_buf[0] = v0 / v_norm;
        for (offset, dst) in v_buf[1..m].iter_mut().enumerate() {
            *dst = a[[k + 1 + offset, k]] / v_norm;
        }

        // A[k:, k:] -= 2 v vᵀ A[k:, k:]
        rank1_householder_update_rows(&mut a, k, &v_buf[..m], &mut dot_buf[..m]);

        // Q[:, k:] -= 2 (Q[:, k:] v) vᵀ. dot_buf needs length n_rows; pass
        // the full buffer (Q is the n×n accumulator so n_rows == n).
        rank1_householder_update_cols(&mut q, k, &v_buf[..m], &mut dot_buf);
    }

    // Mezzadri 2007: normalize so diag(R) ≥ 0, making the orthogonal output
    // Haar-uniform. Without this step Q is orthogonal but biased — would
    // weaken GELO's privacy guarantee. Column-wise sign flip on Q based on
    // the sign of A's diagonal.
    for i in 0..n {
        if a[[i, i]] < 0.0 {
            let mut col = q.slice_mut(s![.., i]);
            for x in col.iter_mut() {
                *x = -*x;
            }
        }
    }

    q
}

/// Rank-1 update of the bottom-right submatrix:
///   `A[k:, k:] -= 2 v vᵀ A[k:, k:]`
/// `v` has length `m = n - k`. `dot_buf[..m]` is reused scratch.
///
/// Cache-friendly via two row-major passes — never reads a column of A
/// out of stride. ndarray's `.dot()` doesn't pull BLAS unless we enable
/// the `blas` feature (we don't); the in-tree fallback has per-call
/// overhead that exceeds our SIMD-friendly hand-rolled version at the
/// shapes we hit (m ≤ 512).
fn rank1_householder_update_rows(
    a: &mut Array2<f32>,
    k: usize,
    v: &[f32],
    dot_buf: &mut [f32],
) {
    let m = v.len();
    let mut sub = a.slice_mut(s![k.., k..]);
    debug_assert_eq!(sub.shape(), &[m, m]);
    let dot = &mut dot_buf[..m];

    // Pass 1: dot[j] += v[i] * sub[i, j], iterating row-by-row so each
    // inner loop touches a contiguous &[f32] slice. LLVM auto-vectorises.
    dot.fill(0.0);
    for (row, &vi) in sub.axis_iter(Axis(0)).zip(v.iter()) {
        let row_slice = row
            .to_slice()
            .expect("row is contiguous in row-major Array2");
        for (d, &x) in dot.iter_mut().zip(row_slice.iter()) {
            *d += vi * x;
        }
    }

    // Pass 2: sub[i, j] -= 2 v[i] dot[j]. Row-by-row, contiguous writes.
    for (row_view, &vi) in sub.axis_iter_mut(Axis(0)).zip(v.iter()) {
        let coef = 2.0 * vi;
        if coef == 0.0 {
            continue;
        }
        let row_slice = row_view
            .into_slice()
            .expect("row is contiguous in row-major Array2");
        for (r, &d) in row_slice.iter_mut().zip(dot.iter()) {
            *r -= coef * d;
        }
    }
}

/// Rank-1 update of the right block of Q:
///   `Q[:, k:] -= 2 (Q[:, k:] v) vᵀ`
/// where v has length `m = n - k`. `dot_buf[..n_rows]` is scratch.
///
/// Same cache strategy: row-major sweeps over `Q[r, k..n]` slices (which
/// are contiguous in row-major). LLVM SIMD-vectorises both inner loops.
fn rank1_householder_update_cols(
    q: &mut Array2<f32>,
    k: usize,
    v: &[f32],
    dot_buf: &mut [f32],
) {
    let m = v.len();
    let n_rows = q.nrows();
    let mut sub = q.slice_mut(s![.., k..]);
    debug_assert_eq!(sub.shape(), &[n_rows, m]);
    let dot = &mut dot_buf[..n_rows];

    // Pass 1: dot[r] = Σ_c sub[r, c] * v[c]. Each row of sub is contiguous.
    for (r_idx, row) in sub.axis_iter(Axis(0)).enumerate() {
        let row_slice = row
            .to_slice()
            .expect("row is contiguous in row-major Array2");
        let mut acc = 0.0_f32;
        for (&x, &vc) in row_slice.iter().zip(v.iter()) {
            acc += x * vc;
        }
        dot[r_idx] = acc;
    }

    // Pass 2: sub[r, c] -= 2 dot[r] v[c]. Same row-major sweep.
    for (row_view, &dot_r) in sub.axis_iter_mut(Axis(0)).zip(dot.iter()) {
        let coef = 2.0 * dot_r;
        if coef == 0.0 {
            continue;
        }
        let row_slice = row_view
            .into_slice()
            .expect("row is contiguous in row-major Array2");
        for (r, &vc) in row_slice.iter_mut().zip(v.iter()) {
            *r -= coef * vc;
        }
    }
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

    /// Haar-uniformity smoke test. The Mezzadri sign correction is what
    /// distinguishes Haar-uniform sampling from "any orthogonal matrix" —
    /// without it we'd get a biased distribution and lose GELO's privacy
    /// argument. We can't directly test "uniform on O(n)" with a finite
    /// sample, but we *can* test invariants that any Haar sample must
    /// satisfy: det(A) ∈ {±1} (exactly orthogonal, no sign bias toward
    /// +1), and the mean entry across many draws should converge to 0.
    #[test]
    fn haar_uniformity_invariants() {
        let mut rng = rand_chacha::ChaCha20Rng::from_seed([99u8; 32]);
        let n = 16;
        let n_draws = 64;
        let mut dets: Vec<f32> = Vec::with_capacity(n_draws);
        let mut mean_acc = 0.0_f32;
        let mut count = 0usize;

        for _ in 0..n_draws {
            let mask = GeloMask::fresh(n, &mut rng);
            let a = mask.matrix();

            // Orthogonality: ‖AᵀA − I‖_max < 1e-4 (covered by `orthogonality`
            // test; spot-check here too).
            let ata = a.t().dot(&a);
            for i in 0..n {
                for j in 0..n {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (ata[[i, j]] - want).abs() < 1e-3,
                        "AᵀA[{i},{j}] = {} expected {}",
                        ata[[i, j]],
                        want
                    );
                }
            }

            // det(A) via product of eigenvalues = ±1 for orthogonal
            // matrices. Compute via Gauss elimination — simple O(n³).
            let det = determinant(a.to_owned().view());
            dets.push(det);
            assert!(
                (det.abs() - 1.0).abs() < 5e-3,
                "|det(A)| = {} not close to 1; sign correction likely broken",
                det.abs()
            );

            for &x in a.iter() {
                mean_acc += x;
                count += 1;
            }
        }

        // det signs should be roughly balanced; Mezzadri sign correction
        // makes Q span O(n) uniformly (both SO(n) and the reflection
        // coset). Without sign correction, det would always be +1 (since
        // Householder reflectors compose to ±1 and a particular impl
        // detail picks +1).
        let pos = dets.iter().filter(|&&d| d > 0.0).count();
        let neg = dets.iter().filter(|&&d| d < 0.0).count();
        assert!(
            pos > 0 && neg > 0,
            "Haar distribution should hit both det=+1 and det=-1 in {n_draws} draws; \
             got pos={pos} neg={neg} — sign correction likely broken"
        );

        // Mean of all entries across many draws should converge to ~0.
        // With n_draws=64 and n²=256 entries each, we have ~16k samples;
        // the per-entry mean is ~0 with stderr ~ 1/√(n_draws · n²) ≈ 0.008.
        let mean = mean_acc / count as f32;
        assert!(
            mean.abs() < 0.05,
            "mean of mask entries = {mean} — should be near zero for Haar-uniform"
        );
    }

    /// Naive determinant via row reduction. O(n³) — only used in tests.
    fn determinant(a: ArrayView2<f32>) -> f32 {
        let n = a.nrows();
        let mut m = a.to_owned();
        let mut det = 1.0_f32;
        for i in 0..n {
            // Find pivot
            let mut pivot = i;
            for r in (i + 1)..n {
                if m[[r, i]].abs() > m[[pivot, i]].abs() {
                    pivot = r;
                }
            }
            if m[[pivot, i]].abs() < 1e-9 {
                return 0.0;
            }
            if pivot != i {
                for j in 0..n {
                    let tmp = m[[i, j]];
                    m[[i, j]] = m[[pivot, j]];
                    m[[pivot, j]] = tmp;
                }
                det = -det;
            }
            det *= m[[i, i]];
            let pv = m[[i, i]];
            for r in (i + 1)..n {
                let factor = m[[r, i]] / pv;
                for c in i..n {
                    let s = m[[i, c]];
                    m[[r, c]] -= factor * s;
                }
            }
        }
        det
    }

    use rand::SeedableRng;
}

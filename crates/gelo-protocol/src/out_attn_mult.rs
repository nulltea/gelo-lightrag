//! TwinShield OutAttnMult — outsource `Q · Kᵀ` to an untrusted accelerator
//! without revealing either operand. Both `Q` and `Kᵀ` are runtime values
//! (unlike GELO's linear offload, which has a public weight on one side),
//! so the standard mask-then-unmask trick doesn't apply directly.
//!
//! The protocol (Xue et al. 2025 §V-A):
//!
//! 1. Sample independent random matrices `R_Q ∈ ℝ^(n×d)`, `R_Kt ∈ ℝ^(d×n)`
//!    and non-zero scalars `a, b ∈ ℝ`.
//! 2. Vertically stack `[Q + R_Q ; a·R_Q]` into a `(2n, d)` matrix and
//!    apply a random row permutation `λ_Q`.
//! 3. Horizontally stack `[Kᵀ + R_Kt , b·R_Kt]` into a `(d, 2n)` matrix
//!    and apply a random column permutation `λ_K`.
//! 4. Hand both stacked matrices to the engine; receive `(2n, 2n)` product.
//! 5. Invert `λ_Q` on rows and `λ_K` on columns to recover the canonical
//!    4-partition layout:
//!    `T₁ = (Q+R_Q)(Kᵀ+R_Kt)`,
//!    `T₂ = b(Q+R_Q) R_Kt`,
//!    `T₃ = a R_Q (Kᵀ+R_Kt)`,
//!    `T₄ = a·b · R_Q R_Kt`.
//! 6. Recover `Q·Kᵀ = T₁ − T₂/b − T₃/a + T₄/(a·b)`.
//!
//! Security: the engine sees a `(2n × 2n)` masked + permuted matmul. Without
//! knowing `(R_Q, R_Kt, a, b, λ_Q, λ_K)` it cannot recover `Q` or `Kᵀ`.

use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2, ArrayView2};
use rand::RngCore;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, StandardNormal, Uniform};

use crate::integrity::verify_offload;
use crate::substrate::GpuOffloadEngine;

/// Magnitude (`σ`) of the additive masks `R_Q`, `R_Kt`. Picked to match
/// typical Q/K row scales (≈ 1) so the recovered partitions don't suffer
/// catastrophic cancellation in f32. Larger σ adds more obscuration at the
/// cost of numerical conditioning.
const MASK_SIGMA: f32 = 1.0;

/// Range of the non-zero scalars `a`, `b`. Bounded away from zero so that
/// the recovery divisions `T₂/b`, `T₃/a`, `T₄/(a·b)` stay well-conditioned.
const SCALAR_LO: f32 = 0.5;
const SCALAR_HI: f32 = 1.5;

/// Run the OutAttnMult protocol against an offload engine. When
/// `n_verify_probes > 0`, a U-Verify Freivalds check is run on the engine's
/// raw `stacked_Q · stacked_Kᵀ` output before TEE-side recovery — so any
/// byzantine tampering is caught before the masks are removed.
pub fn offload_qkt<R: RngCore, E: GpuOffloadEngine + ?Sized>(
    engine: &E,
    rng: &mut R,
    q: ArrayView2<'_, f32>,
    kt: ArrayView2<'_, f32>,
    n_verify_probes: usize,
) -> Result<Array2<f32>> {
    let n = q.nrows();
    let d = q.ncols();
    if kt.nrows() != d || kt.ncols() != n {
        return Err(anyhow!(
            "OutAttnMult shape mismatch: q is ({n},{d}) but kt is ({},{})",
            kt.nrows(),
            kt.ncols(),
        ));
    }

    // 1. Sample masks + scalars.
    let r_q = sample_normal(n, d, MASK_SIGMA, rng);
    let r_kt = sample_normal(d, n, MASK_SIGMA, rng);
    let a = sample_scalar(rng);
    let b = sample_scalar(rng);

    // 2. Build stacked Q ∈ (2n, d).
    let mut stacked_q = Array2::<f32>::zeros((2 * n, d));
    stacked_q.slice_mut(ndarray::s![..n, ..]).assign(&(&q + &r_q));
    stacked_q.slice_mut(ndarray::s![n.., ..]).assign(&(&r_q * a));
    let lambda_q = random_perm(2 * n, rng);
    let stacked_q = permute_rows(stacked_q.view(), &lambda_q);

    // 3. Build stacked Kᵀ ∈ (d, 2n).
    let mut stacked_kt = Array2::<f32>::zeros((d, 2 * n));
    stacked_kt.slice_mut(ndarray::s![.., ..n]).assign(&(&kt + &r_kt));
    stacked_kt.slice_mut(ndarray::s![.., n..]).assign(&(&r_kt * b));
    let lambda_k = random_perm(2 * n, rng);
    let stacked_kt = permute_cols(stacked_kt.view(), &lambda_k);

    // 4. Offload the masked, permuted product.
    let tilde = engine.matmul_dynamic(stacked_q.view(), stacked_kt.view())?;

    // U-Verify integrity check (TwinShield §V-C). Probe on the raw masked
    // product so tampering is detected before TEE-side recovery removes the
    // masks.
    if n_verify_probes > 0 {
        verify_offload(
            n_verify_probes,
            stacked_q.view(),
            stacked_kt.view(),
            tilde.view(),
            rng,
        )?;
    }

    // 5. Invert row + column permutations.
    let permed = inverse_permute(tilde.view(), &lambda_q, &lambda_k);

    let t1 = permed.slice(ndarray::s![..n, ..n]).to_owned();
    let t2 = permed.slice(ndarray::s![..n, n..]).to_owned();
    let t3 = permed.slice(ndarray::s![n.., ..n]).to_owned();
    let t4 = permed.slice(ndarray::s![n.., n..]).to_owned();

    // 6. Algebraic recovery: Q·Kᵀ = T₁ − T₂/b − T₃/a + T₄/(a·b).
    let ab = a * b;
    let qkt = &t1 - &(&t2 / b) - &(&t3 / a) + &(&t4 / ab);
    Ok(qkt)
}

fn sample_normal<R: RngCore>(rows: usize, cols: usize, sigma: f32, rng: &mut R) -> Array2<f32> {
    let normal = StandardNormal;
    Array2::from_shape_fn((rows, cols), |_| {
        let z: f32 = normal.sample(rng);
        z * sigma
    })
}

fn sample_scalar<R: RngCore>(rng: &mut R) -> f32 {
    let pos_dist = Uniform::new(SCALAR_LO, SCALAR_HI).expect("valid uniform range");
    let v: f32 = pos_dist.sample(rng);
    // 50/50 sign flip to widen the effective range without dipping near zero.
    if (rng.next_u32() & 1) == 0 { v } else { -v }
}

fn random_perm<R: RngCore>(n: usize, rng: &mut R) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    v.shuffle(rng);
    v
}

fn permute_rows(m: ArrayView2<'_, f32>, lambda: &[usize]) -> Array2<f32> {
    let (n, d) = (m.nrows(), m.ncols());
    debug_assert_eq!(lambda.len(), n);
    let mut out = Array2::<f32>::zeros((n, d));
    for (i, &src) in lambda.iter().enumerate() {
        out.row_mut(i).assign(&m.row(src));
    }
    out
}

fn permute_cols(m: ArrayView2<'_, f32>, lambda: &[usize]) -> Array2<f32> {
    let (d, n) = (m.nrows(), m.ncols());
    debug_assert_eq!(lambda.len(), n);
    let mut out = Array2::<f32>::zeros((d, n));
    for (j, &src) in lambda.iter().enumerate() {
        out.column_mut(j).assign(&m.column(src));
    }
    out
}

fn inverse_permute(
    m: ArrayView2<'_, f32>,
    lambda_rows: &[usize],
    lambda_cols: &[usize],
) -> Array2<f32> {
    let rows = m.nrows();
    let cols = m.ncols();
    debug_assert_eq!(lambda_rows.len(), rows);
    debug_assert_eq!(lambda_cols.len(), cols);
    // Build inverse permutations.
    let mut inv_rows: Array1<usize> = Array1::zeros(rows);
    for (perm_idx, &orig_idx) in lambda_rows.iter().enumerate() {
        inv_rows[orig_idx] = perm_idx;
    }
    let mut inv_cols: Array1<usize> = Array1::zeros(cols);
    for (perm_idx, &orig_idx) in lambda_cols.iter().enumerate() {
        inv_cols[orig_idx] = perm_idx;
    }
    let mut out = Array2::<f32>::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            out[[i, j]] = m[[inv_rows[i], inv_cols[j]]];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::RayonCpuEngine;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn rand2(rows: usize, cols: usize, rng: &mut impl RngCore, scale: f32) -> Array2<f32> {
        let normal = StandardNormal;
        Array2::from_shape_fn((rows, cols), |_| {
            let z: f32 = normal.sample(rng);
            z * scale
        })
    }

    #[test]
    fn out_attn_mult_recovers_qkt() {
        let mut rng = ChaCha20Rng::from_seed([91u8; 32]);
        let n = 32;
        let d = 16;
        let q = rand2(n, d, &mut rng, 0.5);
        let kt = rand2(d, n, &mut rng, 0.5);

        let expected = q.dot(&kt);

        let engine = RayonCpuEngine::new();
        let got = offload_qkt(&engine, &mut rng, q.view(), kt.view(), 0).unwrap();

        assert_eq!(got.shape(), expected.shape());
        for ((i, j), e) in expected.indexed_iter() {
            let diff = (e - got[[i, j]]).abs();
            assert!(
                diff < 1e-3,
                "OutAttnMult diverges at ({i},{j}): expected={e} got={}",
                got[[i, j]]
            );
        }
    }

    #[test]
    fn out_attn_mult_handles_small_sequence() {
        // Edge case: very small n (smaller than head_dim).
        let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
        let n = 4;
        let d = 32;
        let q = rand2(n, d, &mut rng, 0.3);
        let kt = rand2(d, n, &mut rng, 0.3);

        let expected = q.dot(&kt);

        let engine = RayonCpuEngine::new();
        let got = offload_qkt(&engine, &mut rng, q.view(), kt.view(), 0).unwrap();

        for ((i, j), e) in expected.indexed_iter() {
            assert!(
                (e - got[[i, j]]).abs() < 1e-3,
                "small-n OutAttnMult diverges at ({i},{j}): {e} vs {}",
                got[[i, j]],
            );
        }
    }

    #[test]
    fn out_attn_mult_with_verify_accepts_honest_engine() {
        let mut rng = ChaCha20Rng::from_seed([99u8; 32]);
        let n = 24;
        let d = 16;
        let q = rand2(n, d, &mut rng, 0.5);
        let kt = rand2(d, n, &mut rng, 0.5);
        let engine = RayonCpuEngine::new();
        let got = offload_qkt(&engine, &mut rng, q.view(), kt.view(), 8).unwrap();
        let expected = q.dot(&kt);
        for ((i, j), e) in expected.indexed_iter() {
            assert!((e - got[[i, j]]).abs() < 1e-3);
        }
    }
}

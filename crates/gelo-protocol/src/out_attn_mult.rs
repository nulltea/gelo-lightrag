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
use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3, Axis};
use rand::RngCore;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, StandardNormal, Uniform};

use crate::integrity::verify_offload;
use crate::profile;
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

    // 1-3. Sample masks + scalars + permutations, build stacked operands.
    let (stacked_q, stacked_kt, lambda_q, lambda_k, a, b) =
        profile::time("outattn:setup_stack", || {
            let r_q = sample_normal(n, d, MASK_SIGMA, rng);
            let r_kt = sample_normal(d, n, MASK_SIGMA, rng);
            let a = sample_scalar(rng);
            let b = sample_scalar(rng);

            let mut sq = Array2::<f32>::zeros((2 * n, d));
            sq.slice_mut(ndarray::s![..n, ..]).assign(&(&q + &r_q));
            sq.slice_mut(ndarray::s![n.., ..]).assign(&(&r_q * a));
            let lambda_q = random_perm(2 * n, rng);
            let sq = permute_rows(sq.view(), &lambda_q);

            let mut skt = Array2::<f32>::zeros((d, 2 * n));
            skt.slice_mut(ndarray::s![.., ..n]).assign(&(&kt + &r_kt));
            skt.slice_mut(ndarray::s![.., n..]).assign(&(&r_kt * b));
            let lambda_k = random_perm(2 * n, rng);
            let skt = permute_cols(skt.view(), &lambda_k);

            (sq, skt, lambda_q, lambda_k, a, b)
        });

    // 4. Offload the masked, permuted product.
    let tilde = profile::time("engine:matmul_dynamic", || {
        engine.matmul_dynamic(stacked_q.view(), stacked_kt.view())
    })?;

    // U-Verify integrity check (TwinShield §V-C). Probe on the raw masked
    // product so tampering is detected before TEE-side recovery removes the
    // masks.
    if n_verify_probes > 0 {
        profile::time("uverify:attn_qkt", || {
            verify_offload(
                n_verify_probes,
                stacked_q.view(),
                stacked_kt.view(),
                tilde.view(),
                rng,
            )
        })?;
    }

    // 5-6. Invert permutations, extract partitions, algebraic recovery.
    let qkt = profile::time("outattn:recover", || {
        let permed = inverse_permute(tilde.view(), &lambda_q, &lambda_k);
        let t1 = permed.slice(ndarray::s![..n, ..n]).to_owned();
        let t2 = permed.slice(ndarray::s![..n, n..]).to_owned();
        let t3 = permed.slice(ndarray::s![n.., ..n]).to_owned();
        let t4 = permed.slice(ndarray::s![n.., n..]).to_owned();
        let ab = a * b;
        &t1 - &(&t2 / b) - &(&t3 / a) + &(&t4 / ab)
    });
    Ok(qkt)
}

/// Batched OutAttnMult — one engine call covering every Q head of a layer.
///
/// Each head gets independent masks `R_Q[h]`, `R_Kt[h]`, scalars `a[h]`,
/// `b[h]`, and row/column permutations `λ_Q[h]` / `λ_K[h]`. The 4-partition
/// recovery is then done per-head against the batched output, so the
/// privacy story is identical to running `offload_qkt` h times in
/// sequence — only the engine sees the single fused batched matmul.
///
/// `q` shape: `(H, n, d_head)`. `kt` shape: `(H, d_head, n)`. Output:
/// `(H, n, n)`.
pub fn offload_qkt_batched<R: RngCore, E: GpuOffloadEngine + ?Sized>(
    engine: &E,
    rng: &mut R,
    q: ArrayView3<'_, f32>,
    kt: ArrayView3<'_, f32>,
    n_verify_probes: usize,
) -> Result<Array3<f32>> {
    let h = q.shape()[0];
    let n = q.shape()[1];
    let d = q.shape()[2];
    if kt.shape() != [h, d, n] {
        return Err(anyhow!(
            "offload_qkt_batched shape mismatch: q is ({h},{n},{d}) but kt is {:?}",
            kt.shape()
        ));
    }

    // 1-3. Per-head sampling + stacking. Collected into 3-D arrays so the
    //      engine sees a single contiguous batched tensor on each side.
    let mut stacked_q = Array3::<f32>::zeros((h, 2 * n, d));
    let mut stacked_kt = Array3::<f32>::zeros((h, d, 2 * n));
    let mut lambdas_q: Vec<Vec<usize>> = Vec::with_capacity(h);
    let mut lambdas_k: Vec<Vec<usize>> = Vec::with_capacity(h);
    let mut scalars: Vec<(f32, f32)> = Vec::with_capacity(h);

    profile::time("outattn:setup_stack_batched", || {
        for hi in 0..h {
            let q_h = q.index_axis(Axis(0), hi);
            let kt_h = kt.index_axis(Axis(0), hi);

            let r_q = sample_normal(n, d, MASK_SIGMA, rng);
            let r_kt = sample_normal(d, n, MASK_SIGMA, rng);
            let a = sample_scalar(rng);
            let b = sample_scalar(rng);

            // Build the masked & filler rows for this head's stacked_Q.
            let mut sq = Array2::<f32>::zeros((2 * n, d));
            sq.slice_mut(ndarray::s![..n, ..]).assign(&(&q_h + &r_q));
            sq.slice_mut(ndarray::s![n.., ..]).assign(&(&r_q * a));
            let lambda_q = random_perm(2 * n, rng);
            let sq = permute_rows(sq.view(), &lambda_q);
            stacked_q.index_axis_mut(Axis(0), hi).assign(&sq);

            // Same for stacked_Kt.
            let mut skt = Array2::<f32>::zeros((d, 2 * n));
            skt.slice_mut(ndarray::s![.., ..n]).assign(&(&kt_h + &r_kt));
            skt.slice_mut(ndarray::s![.., n..]).assign(&(&r_kt * b));
            let lambda_k = random_perm(2 * n, rng);
            let skt = permute_cols(skt.view(), &lambda_k);
            stacked_kt.index_axis_mut(Axis(0), hi).assign(&skt);

            lambdas_q.push(lambda_q);
            lambdas_k.push(lambda_k);
            scalars.push((a, b));
        }
    });

    // 4. One fused batched dispatch.
    let tilde = profile::time("engine:matmul_dynamic_batched", || {
        engine.matmul_dynamic_batched(stacked_q.view(), stacked_kt.view())
    })?;

    // 5. Per-head U-Verify on the engine's raw output. We probe the
    //    individual 2-D batch slices rather than the full 3-D tensor so the
    //    Wilkinson-style eps in `integrity::verify_offload` stays scaled
    //    against per-head magnitudes.
    if n_verify_probes > 0 {
        profile::time("uverify:attn_qkt_batched", || -> Result<()> {
            for hi in 0..h {
                verify_offload(
                    n_verify_probes,
                    stacked_q.index_axis(Axis(0), hi),
                    stacked_kt.index_axis(Axis(0), hi),
                    tilde.index_axis(Axis(0), hi),
                    rng,
                )?;
            }
            Ok(())
        })?;
    }

    // 6. Per-head depermute + algebraic recovery.
    let qkt = profile::time("outattn:recover_batched", || {
        let mut out = Array3::<f32>::zeros((h, n, n));
        for hi in 0..h {
            let permed = inverse_permute(
                tilde.index_axis(Axis(0), hi),
                &lambdas_q[hi],
                &lambdas_k[hi],
            );
            let t1 = permed.slice(ndarray::s![..n, ..n]).to_owned();
            let t2 = permed.slice(ndarray::s![..n, n..]).to_owned();
            let t3 = permed.slice(ndarray::s![n.., ..n]).to_owned();
            let t4 = permed.slice(ndarray::s![n.., n..]).to_owned();
            let (a, b) = scalars[hi];
            let ab = a * b;
            let recovered = &t1 - &(&t2 / b) - &(&t3 / a) + &(&t4 / ab);
            out.index_axis_mut(Axis(0), hi).assign(&recovered);
        }
        out
    });
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
    use crate::sim::ReferenceCpuEngine;
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

        let engine = ReferenceCpuEngine::new();
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

        let engine = ReferenceCpuEngine::new();
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
        let engine = ReferenceCpuEngine::new();
        let got = offload_qkt(&engine, &mut rng, q.view(), kt.view(), 8).unwrap();
        let expected = q.dot(&kt);
        for ((i, j), e) in expected.indexed_iter() {
            assert!((e - got[[i, j]]).abs() < 1e-3);
        }
    }

    #[test]
    fn out_attn_mult_batched_recovers_per_head_qkt() {
        // 8 Q heads, each with independent random Q/K^T — confirm the
        // batched path returns the same per-head Q·Kᵀ products as a CPU
        // reference.
        let mut rng = ChaCha20Rng::from_seed([0xAB; 32]);
        let h = 8;
        let n = 20;
        let d = 12;

        let mut q3 = Array3::<f32>::zeros((h, n, d));
        let mut kt3 = Array3::<f32>::zeros((h, d, n));
        for hi in 0..h {
            q3.index_axis_mut(Axis(0), hi)
                .assign(&rand2(n, d, &mut rng, 0.4));
            kt3.index_axis_mut(Axis(0), hi)
                .assign(&rand2(d, n, &mut rng, 0.4));
        }

        let engine = ReferenceCpuEngine::new();
        let got = offload_qkt_batched(&engine, &mut rng, q3.view(), kt3.view(), 0).unwrap();

        for hi in 0..h {
            let expected = q3.index_axis(Axis(0), hi).dot(&kt3.index_axis(Axis(0), hi));
            for ((i, j), e) in expected.indexed_iter() {
                let g = got[[hi, i, j]];
                assert!(
                    (e - g).abs() < 1e-3,
                    "batched OutAttnMult diverges at head={hi} ({i},{j}): expected={e}, got={g}",
                );
            }
        }
    }

    #[test]
    fn out_attn_mult_batched_with_verify_accepts_honest_engine() {
        let mut rng = ChaCha20Rng::from_seed([0xCD; 32]);
        let h = 4;
        let n = 16;
        let d = 8;
        let mut q3 = Array3::<f32>::zeros((h, n, d));
        let mut kt3 = Array3::<f32>::zeros((h, d, n));
        for hi in 0..h {
            q3.index_axis_mut(Axis(0), hi)
                .assign(&rand2(n, d, &mut rng, 0.3));
            kt3.index_axis_mut(Axis(0), hi)
                .assign(&rand2(d, n, &mut rng, 0.3));
        }
        let engine = ReferenceCpuEngine::new();
        let got = offload_qkt_batched(&engine, &mut rng, q3.view(), kt3.view(), 8).unwrap();
        for hi in 0..h {
            let expected = q3.index_axis(Axis(0), hi).dot(&kt3.index_axis(Axis(0), hi));
            for ((i, j), e) in expected.indexed_iter() {
                assert!((e - got[[hi, i, j]]).abs() < 1e-3);
            }
        }
    }
}

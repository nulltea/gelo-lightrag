//! U-Verify — TwinShield's Freivalds-style integrity check (Xue et al. 2025 §V-C).
//!
//! For an outsourced matmul `Z = A · B` where the trusted side has access to
//! both `A` and `B` (and observes `Z` from the engine), a single probe is:
//!
//!   1. Pick a random vector `r ∈ {−L..L}^p`.
//!   2. Compute `Br = B · r`  (TEE-local, `O(k·p)`)
//!   3. Compute `lhs = A · Br` (TEE-local, `O(m·k)`)
//!   4. Compute `rhs = Z · r` (TEE-local, `O(m·p)`)
//!   5. Assert `lhs ≈ rhs`.
//!
//! Soundness: if `Z ≠ A·B`, the probe fires with probability `≥ 1 − 1/(2L)`.
//! Running `k` independent probes brings the undetected-tamper rate to
//! `(2L)^-k` — at `L=3, k=8` that is `≈ 2.4·10^-7`.

use anyhow::{Result, anyhow};
use ndarray::{Array1, ArrayView2};
use rand::RngCore;
use rand_distr::{Distribution, Uniform};

/// Probe-coefficient bound `L`. Bigger `L` → fewer probes for the same soundness,
/// at the cost of larger probe-vector magnitudes and thus worse f32 cancellation.
const PROBE_L: i32 = 3;

/// f32 machine epsilon (`2⁻²³`).
const F32_EPS: f32 = 1.1920929e-7;

/// Multiplier on the f32 roundoff bound. Wilkinson-style backward error
/// analysis bounds the relative error of an `n`-way fma chain by `~n·eps`,
/// but in practice we see ~`sqrt(n)·eps` due to random-sign cancellation.
/// `40` here absorbs that observation plus an extra ~10x cushion for
/// pre-/post- ops (mask round-trip, accumulators, etc.).
const ROUNDOFF_CUSHION: f32 = 40.0;

/// Floor on the absolute tolerance, used when `||A||·||B||` is tiny.
const ABS_FLOOR: f32 = 1e-4;

/// Run `n_probes` Freivalds-style checks over the asserted product `Z = A · B`.
/// Returns `Err` if any probe disagrees, indicating a tampered or buggy engine.
///
/// All three inputs must already be the *exact* operands the engine saw —
/// e.g. for GELO's linear offload, `A` is the masked activation and `B` is
/// the registered weight; for OutAttnMult, `A` is `stacked_Q` and `B` is
/// `stacked_Kᵀ`.
pub fn verify_offload<R: RngCore>(
    n_probes: usize,
    a: ArrayView2<'_, f32>,
    b: ArrayView2<'_, f32>,
    z: ArrayView2<'_, f32>,
    rng: &mut R,
) -> Result<()> {
    if n_probes == 0 {
        return Ok(());
    }
    let m = a.nrows();
    let k = a.ncols();
    let p = b.ncols();
    if b.nrows() != k {
        return Err(anyhow!(
            "U-Verify shape: A is ({m},{k}) but B is ({},{p})",
            b.nrows()
        ));
    }
    if z.shape() != [m, p] {
        return Err(anyhow!(
            "U-Verify shape: A·B should be ({m},{p}) but Z is ({},{})",
            z.nrows(),
            z.ncols()
        ));
    }

    let dist =
        Uniform::new_inclusive(-PROBE_L, PROBE_L).expect("valid uniform range for probe coefficients");

    // The Freivalds inner product chains together ~(k + p) multiply-adds per
    // output element, so its f32 roundoff is `O((k+p)·eps·max|A|·max|B|·L)`.
    // We compute that bound up front and accept anything below it as "no
    // detectable tamper". An attacker who slips a perturbation under this
    // bound is one we couldn't have distinguished from honest f32 noise
    // anyway.
    let a_inf = inf_norm(a);
    let b_inf = inf_norm(b);
    let probe_l = PROBE_L as f32;
    let chain = (k + p) as f32;
    let roundoff_bound =
        ROUNDOFF_CUSHION * chain * F32_EPS * a_inf * b_inf * probe_l;

    for probe_idx in 0..n_probes {
        let r: Array1<f32> = Array1::from_shape_fn(p, |_| dist.sample(rng) as f32);
        // br ∈ ℝ^k
        let br = b.dot(&r);
        // expected ∈ ℝ^m
        let expected = a.dot(&br);
        let observed = z.dot(&r);
        for i in 0..m {
            let diff = (expected[i] - observed[i]).abs();
            let scale = expected[i].abs().max(observed[i].abs()).max(1.0);
            let tol = roundoff_bound.max(ABS_FLOOR) + 1e-4 * scale;
            if diff > tol {
                return Err(anyhow!(
                    "U-Verify mismatch (probe {probe_idx} row {i}): expected={} observed={} diff={diff} tol={tol}",
                    expected[i],
                    observed[i]
                ));
            }
        }
    }
    Ok(())
}

fn inf_norm(m: ArrayView2<'_, f32>) -> f32 {
    m.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()))
}

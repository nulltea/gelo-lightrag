//! Analytic Matrix Gaussian Mechanism (aMGM) — Balle & Wang, ICML 2018,
//! adapted to embedding-row sensitivity by Yue et al. (DP-Forward, CCS 2023).
//!
//! Given a privacy budget `(ε, δ)` and an L2 sensitivity `Δ₂`, the mechanism
//! `f(x) + N(0, σ²I)` is `(ε, δ)`-DP iff `σ ≥ Δ₂ / R(ε)`, where `R(ε)` is the
//! reciprocal of the analytic-Gaussian scale factor. The DP-Forward repo
//! [`xiangyue9607/DP-Forward`](https://github.com/xiangyue9607/DP-Forward)
//! computes `R(ε)` by bisection over a delta-balancing equation due to
//! Balle & Wang; we reproduce that bisection below in Rust.
//!
//! Pre-mechanism, each embedding row is L2-clipped to `‖x‖₂ ≤ C`, giving
//! per-row sensitivity `Δ₂ = 2C`.

use rand::RngCore;
use rand_distr::{Distribution, Normal};
use statrs::function::erf::erf;
use std::f64::consts::SQRT_2;

/// Clip each row to L2 norm ≤ `c` in place.
pub fn clip_l2_in_place(v: &mut [f32], c: f32) {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt();
    if norm > c {
        let scale = c / norm;
        for x in v.iter_mut() {
            *x *= scale;
        }
    }
}

/// Add isotropic Gaussian noise `N(0, σ²I)` componentwise. `sigma` is the
/// noise *scale* (standard deviation), not variance — match
/// `rand_distr::Normal::new(mean, std_dev)`.
pub fn add_gaussian_noise<R: RngCore>(v: &mut [f32], sigma: f64, rng: &mut R) {
    if sigma <= 0.0 {
        return;
    }
    let normal = Normal::new(0.0, sigma).expect("sigma > 0");
    for x in v.iter_mut() {
        let n: f64 = normal.sample(rng);
        *x += n as f32;
    }
}

/// Standard-normal CDF, `Φ(t) = (1 + erf(t/√2)) / 2`.
#[inline]
fn phi(t: f64) -> f64 {
    0.5 * (1.0 + erf(t / SQRT_2))
}

/// Balle–Wang analytic-Gaussian calibration. Returns `σ` such that the
/// mechanism `f(x) + N(0, σ²I)` is `(ε, δ)`-DP for an L2 sensitivity
/// equal to `sensitivity`.
///
/// Both `epsilon > 0` and `delta ∈ (0, 1)` are required. Panics if either is
/// out of range.
pub fn calibrate_sigma(epsilon: f64, delta: f64, sensitivity: f64) -> f64 {
    assert!(epsilon > 0.0, "epsilon must be > 0 (got {epsilon})");
    assert!(
        delta > 0.0 && delta < 1.0,
        "delta must be in (0, 1) (got {delta})"
    );
    assert!(
        sensitivity > 0.0,
        "sensitivity must be > 0 (got {sensitivity})"
    );

    let delta_0 = phi(0.0) - epsilon.exp() * phi(-(2.0 * epsilon).sqrt());

    // B_plus(v) and B_minus(u) are the two branches of the Balle–Wang
    // equation. We solve `B(x) = δ` for `x` by bisection on `[0, 1e5]`,
    // then convert `x*` to the scale `R(ε)`.
    let (b, sign): (Box<dyn Fn(f64) -> f64>, f64) = if delta >= delta_0 {
        let eps = epsilon;
        let b = Box::new(move |v: f64| {
            phi((eps * v).sqrt()) - eps.exp() * phi(-(eps * (v + 2.0)).sqrt())
        });
        (b, -1.0)
    } else {
        let eps = epsilon;
        let b = Box::new(move |u: f64| {
            phi(-(eps * u).sqrt()) - eps.exp() * phi(-(eps * (u + 2.0)).sqrt())
        });
        (b, 1.0)
    };

    let x_star = bisect(|x| b(x) - delta, 0.0, 1.0e5, 200);
    let alpha = (1.0 + x_star / 2.0).sqrt() + sign * (x_star / 2.0).sqrt();
    let r = (2.0 * epsilon).sqrt() / alpha;
    sensitivity / r
}

/// Robust sign-bracket bisection. Assumes `f(lo)` and `f(hi)` have opposite
/// signs (true for both Balle–Wang branches once `δ ≠ δ₀`).
fn bisect<F: Fn(f64) -> f64>(f: F, mut lo: f64, mut hi: f64, max_iters: usize) -> f64 {
    let mut f_lo = f(lo);
    let f_hi = f(hi);
    if f_lo == 0.0 {
        return lo;
    }
    if f_hi == 0.0 {
        return hi;
    }
    // If endpoints don't bracket a sign change, the equation has no solution
    // in the interval; return the closer endpoint. Indicates a misuse.
    if f_lo.signum() == f_hi.signum() {
        return if f_lo.abs() < f_hi.abs() { lo } else { hi };
    }
    for _ in 0..max_iters {
        let mid = 0.5 * (lo + hi);
        let f_mid = f(mid);
        if f_mid.abs() < 1.0e-12 || (hi - lo) < 1.0e-12 {
            return mid;
        }
        if f_mid.signum() == f_lo.signum() {
            lo = mid;
            f_lo = f_mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha20Rng;
    use rand::SeedableRng;

    #[test]
    fn clip_no_op_when_under_bound() {
        let mut v = [0.3_f32, 0.4, 0.0]; // norm = 0.5
        clip_l2_in_place(&mut v, 1.0);
        assert!((v[0] - 0.3).abs() < 1e-6);
        assert!((v[1] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn clip_scales_to_bound() {
        let mut v = [3.0_f32, 4.0]; // norm = 5
        clip_l2_in_place(&mut v, 1.0);
        let norm: f32 = (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "post-clip norm = {norm}");
        // direction preserved: original v / 5 = (0.6, 0.8)
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn phi_known_values() {
        assert!((phi(0.0) - 0.5).abs() < 1e-12);
        // Φ(1) ≈ 0.841344746
        assert!((phi(1.0) - 0.841_344_746_068_543).abs() < 1e-9);
        // Φ(-1) ≈ 0.158655254
        assert!((phi(-1.0) - 0.158_655_253_931_457).abs() < 1e-9);
    }

    #[test]
    fn calibrate_sigma_positive_and_decreases_with_epsilon() {
        let s_eps_1 = calibrate_sigma(1.0, 1e-5, 2.0);
        let s_eps_4 = calibrate_sigma(4.0, 1e-5, 2.0);
        let s_eps_8 = calibrate_sigma(8.0, 1e-5, 2.0);
        assert!(s_eps_1 > 0.0);
        assert!(s_eps_4 > 0.0);
        assert!(s_eps_8 > 0.0);
        // tighter privacy (smaller eps) → larger noise
        assert!(s_eps_1 > s_eps_4);
        assert!(s_eps_4 > s_eps_8);
    }

    #[test]
    fn calibrate_sigma_scales_linearly_with_sensitivity() {
        let s1 = calibrate_sigma(2.0, 1e-5, 1.0);
        let s2 = calibrate_sigma(2.0, 1e-5, 2.0);
        // σ is linear in sensitivity: σ(2C) = 2·σ(C)
        assert!((s2 / s1 - 2.0).abs() < 1e-9, "s1={s1}, s2={s2}");
    }

    /// Golden value lock: at the standard `(ε=4.0, δ=1e-5, Δ=2C, C=1.0)`
    /// configuration, our bisection produces σ ≈ 2.1623, which is 2× the
    /// Balle–Wang table-1 entry for unit-sensitivity (σ_unit ≈ 1.081) — the
    /// expected linear scaling. Tolerance is loose because different
    /// bisection cutoffs differ at the 4th decimal.
    #[test]
    fn calibrate_sigma_at_ref_config() {
        let sigma = calibrate_sigma(4.0, 1e-5, 2.0);
        assert!(
            (sigma - 2.1623).abs() < 1e-3,
            "sigma at ε=4, δ=1e-5, Δ=2 = {sigma}, expected ≈ 2.1623"
        );
        // and the unit-sensitivity table-1 value
        let sigma_unit = calibrate_sigma(4.0, 1e-5, 1.0);
        assert!(
            (sigma_unit - 1.0811).abs() < 1e-3,
            "sigma at ε=4, δ=1e-5, Δ=1 = {sigma_unit}, expected ≈ 1.0811"
        );
    }

    #[test]
    fn noise_changes_vector_and_preserves_dimension() {
        let mut v = vec![0.1_f32; 32];
        let v_clean = v.clone();
        let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
        add_gaussian_noise(&mut v, 0.1, &mut rng);
        assert_eq!(v.len(), v_clean.len());
        let max_abs = v
            .iter()
            .zip(v_clean.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_abs > 1e-4, "noise must perturb at sigma=0.1");
    }

    #[test]
    fn noise_empirical_std_matches_sigma() {
        let n = 10_000;
        let mut v = vec![0.0_f32; n];
        let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
        let sigma = 0.5;
        add_gaussian_noise(&mut v, sigma, &mut rng);
        let mean: f64 = v.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let var: f64 = v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n as f64;
        let empirical = var.sqrt();
        assert!(
            (empirical - sigma).abs() < 0.02,
            "empirical std {empirical} vs nominal {sigma}"
        );
    }
}

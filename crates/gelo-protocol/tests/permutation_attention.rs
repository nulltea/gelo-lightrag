//! Phase 1 of the permutation-shielded attention work — math-only verification.
//!
//! Goal: prove the identity
//!
//!   softmax(π·Q·Kᵀ·πᵀ / √d) · π·V  =  π · softmax(Q·Kᵀ / √d) · V
//!
//! composes correctly with an additive Gaussian noise term `η ~ N(0, σ²·I)`
//! on Q and K (the Hidden No More mitigation), so that the round-trip
//! recovery `πᵀ · ((softmax((πQ+η_Q)(πK+η_K)ᵀ / √d) · πV)` is close to
//! plain attention `softmax(QKᵀ/√d) · V` within bounds determined by σ.
//!
//! This test is intentionally substrate-free: no executor, no engine, no
//! BLAS. The point is to lock the math down before Phase 2 wires it into
//! the protocol substrate.

use ndarray::{Array1, Array2, ArrayView2, Axis};
use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

/// Numerically stable softmax along the last axis (row-wise on `(n, n)`).
fn softmax_rowwise(scores: ArrayView2<'_, f32>) -> Array2<f32> {
    let (n, m) = scores.dim();
    let mut out = Array2::<f32>::zeros((n, m));
    for i in 0..n {
        let row = scores.row(i);
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (j, v) in row.iter().enumerate() {
            let e = (*v - max).exp();
            out[(i, j)] = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for j in 0..m {
            out[(i, j)] *= inv;
        }
    }
    out
}

/// Reference attention: `softmax(Q·Kᵀ / √d) · V`. Single-head.
fn attention(q: ArrayView2<'_, f32>, k: ArrayView2<'_, f32>, v: ArrayView2<'_, f32>) -> Array2<f32> {
    let (_, d) = q.dim();
    let scale = 1.0 / (d as f32).sqrt();
    let mut scores = q.dot(&k.t());
    scores.mapv_inplace(|x| x * scale);
    let probs = softmax_rowwise(scores.view());
    probs.dot(&v)
}

/// Apply a row permutation to `m`: result[i, :] = m[perm[i], :].
fn permute_rows(m: ArrayView2<'_, f32>, perm: &[usize]) -> Array2<f32> {
    let (n, d) = m.dim();
    assert_eq!(perm.len(), n);
    let mut out = Array2::<f32>::zeros((n, d));
    for (i, &p) in perm.iter().enumerate() {
        out.row_mut(i).assign(&m.row(p));
    }
    out
}

/// Invert a permutation: if perm maps i -> perm[i], inv_perm maps perm[i] -> i.
fn inverse_permutation(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}

/// Add Gaussian noise N(0, σ²·I) to `m` element-wise.
fn add_gaussian<R: rand::RngCore>(m: ArrayView2<'_, f32>, sigma: f32, rng: &mut R) -> Array2<f32> {
    let normal = StandardNormal;
    let mut out = m.to_owned();
    if sigma == 0.0 {
        return out;
    }
    for v in out.iter_mut() {
        let z: f32 = normal.sample(rng);
        *v += sigma * z;
    }
    out
}

/// Permutation-shielded attention round-trip:
///
/// 1. TEE samples π (and noise).
/// 2. TEE ships πQ + η_Q, πK + η_K, πV to GPU.
/// 3. GPU computes softmax over the masked product, multiplies by πV.
/// 4. TEE applies πᵀ to recover plain attention output.
///
/// `sigma` is the standard deviation of the Q/K noise. `sigma = 0.0`
/// disables noise (pure permutation equivariance).
fn permutation_shielded_attention<R: rand::RngCore>(
    q: ArrayView2<'_, f32>,
    k: ArrayView2<'_, f32>,
    v: ArrayView2<'_, f32>,
    sigma: f32,
    rng: &mut R,
) -> Array2<f32> {
    let n = q.nrows();

    // 1. Sample fresh row permutation π ∈ S_n.
    let mut perm: Vec<usize> = (0..n).collect();
    perm.shuffle(rng);

    // 2. Apply π to Q, K, V along the token axis.
    let q_perm = permute_rows(q, &perm);
    let k_perm = permute_rows(k, &perm);
    let v_perm = permute_rows(v, &perm);

    // 3. Add Gaussian noise to Q and K only (not V — V keeps its data).
    let q_noisy = add_gaussian(q_perm.view(), sigma, rng);
    let k_noisy = add_gaussian(k_perm.view(), sigma, rng);

    // 4. GPU side: compute attention under the obfuscation.
    let out_perm = attention(q_noisy.view(), k_noisy.view(), v_perm.view());

    // 5. TEE recovery: unpermute rows with πᵀ.
    let inv = inverse_permutation(&perm);
    permute_rows(out_perm.view(), &inv)
}

/// Random (n, d) f32 matrix with entries in [-1, 1].
fn random_matrix(n: usize, d: usize, rng: &mut ChaCha20Rng) -> Array2<f32> {
    Array2::from_shape_fn((n, d), |_| rng.random::<f32>() * 2.0 - 1.0)
}

/// Max absolute element difference.
fn max_abs_diff(a: ArrayView2<'_, f32>, b: ArrayView2<'_, f32>) -> f32 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn equivariance_holds_exactly_at_sigma_zero() {
    // For σ = 0 the protocol reduces to softmax(πQ·(πK)ᵀ/√d)·πV with πᵀ recovery,
    // which is mathematically exact (modulo f32 round-off).
    let cases = [(8, 16), (16, 32), (32, 64), (64, 128)];
    for (n, d) in cases {
        let mut rng = ChaCha20Rng::seed_from_u64(0xCAFEBABE ^ (n as u64) << 8 ^ d as u64);
        let q = random_matrix(n, d, &mut rng);
        let k = random_matrix(n, d, &mut rng);
        let v = random_matrix(n, d, &mut rng);

        let plain = attention(q.view(), k.view(), v.view());
        let recovered = permutation_shielded_attention(q.view(), k.view(), v.view(), 0.0, &mut rng);

        let drift = max_abs_diff(plain.view(), recovered.view());
        // f32 cancellation floor — softmax + matmul drift at this scale.
        assert!(
            drift < 1e-5,
            "σ=0 must be bit-equivalent up to f32 round-off: n={n}, d={d}, drift={drift}",
        );
    }
}

#[test]
fn drift_is_bounded_at_sigma_small() {
    // σ = 0.001 — the lower-noise option. Drift should be small.
    let n = 64;
    let d = 128;
    let mut rng = ChaCha20Rng::seed_from_u64(0xDEADBEEF);
    let q = random_matrix(n, d, &mut rng);
    let k = random_matrix(n, d, &mut rng);
    let v = random_matrix(n, d, &mut rng);

    let plain = attention(q.view(), k.view(), v.view());

    // Sample 8 fresh runs and check worst-case drift — the protocol is
    // randomised over π and η, so we want the worst run to still be bounded.
    let mut worst = 0.0f32;
    for _ in 0..8 {
        let recovered =
            permutation_shielded_attention(q.view(), k.view(), v.view(), 0.001, &mut rng);
        let drift = max_abs_diff(plain.view(), recovered.view());
        worst = worst.max(drift);
    }
    assert!(
        worst < 5e-3,
        "σ=0.001 drift should stay below 5e-3 elementwise; got {worst}",
    );
}

#[test]
fn drift_is_bounded_at_sigma_hidden_no_more() {
    // σ = 0.01 — the value Hidden No More (arXiv 2505.18332) reports as
    // achieving ROUGE < 0.1 against their attack. Drift should be larger
    // than σ=0.001 but still small enough that downstream cosine ranks
    // could plausibly survive (verified in later phases against the
    // accuracy bench, not here).
    let n = 64;
    let d = 128;
    let mut rng = ChaCha20Rng::seed_from_u64(0xABCDEF01);
    let q = random_matrix(n, d, &mut rng);
    let k = random_matrix(n, d, &mut rng);
    let v = random_matrix(n, d, &mut rng);

    let plain = attention(q.view(), k.view(), v.view());

    let mut worst = 0.0f32;
    for _ in 0..8 {
        let recovered =
            permutation_shielded_attention(q.view(), k.view(), v.view(), 0.01, &mut rng);
        let drift = max_abs_diff(plain.view(), recovered.view());
        worst = worst.max(drift);
    }
    // Looser bound — σ=0.01 propagates through softmax non-linearly so the
    // worst-case element drift can be larger than σ itself. We're checking
    // it doesn't blow up, not that it's negligible.
    assert!(
        worst < 5e-2,
        "σ=0.01 drift should stay below 5e-2 elementwise; got {worst}",
    );
}

#[test]
fn permutation_alone_is_exact() {
    // Sanity check: permute Q, K, V identically but apply NO noise, then
    // recover. This is the cleanest form of softmax-permutation
    // equivariance and must hold to f32 cancellation.
    let n = 32;
    let d = 64;
    let mut rng = ChaCha20Rng::seed_from_u64(0x12345678);
    let q = random_matrix(n, d, &mut rng);
    let k = random_matrix(n, d, &mut rng);
    let v = random_matrix(n, d, &mut rng);

    let plain = attention(q.view(), k.view(), v.view());

    let mut perm: Vec<usize> = (0..n).collect();
    perm.shuffle(&mut rng);

    let q_p = permute_rows(q.view(), &perm);
    let k_p = permute_rows(k.view(), &perm);
    let v_p = permute_rows(v.view(), &perm);

    let scrambled = attention(q_p.view(), k_p.view(), v_p.view());
    let inv = inverse_permutation(&perm);
    let recovered = permute_rows(scrambled.view(), &inv);

    let drift = max_abs_diff(plain.view(), recovered.view());
    assert!(
        drift < 1e-5,
        "pure permutation equivariance must hold to f32 round-off: drift={drift}",
    );
}

#[test]
fn shield_rows_corrupt_attention_output() {
    // Negative result documented as a guardrail: if we naively stack
    // shield rows into Q/K/V and run attention on the joint matrix,
    // attention scores include cross-terms with shield rows and the
    // recovered output for the data rows is NOT plain attention. This
    // proves that shield rows (which work for linear projections) do
    // NOT compose with attention — Phase 2/3 must avoid that mistake.
    let n_data = 16;
    let k_shield = 4;
    let d = 32;
    let mut rng = ChaCha20Rng::seed_from_u64(0x9999);

    let q = random_matrix(n_data, d, &mut rng);
    let k = random_matrix(n_data, d, &mut rng);
    let v = random_matrix(n_data, d, &mut rng);
    let plain = attention(q.view(), k.view(), v.view());

    // Stack k_shield fake rows (large-magnitude) into Q, K, V.
    let q_aug = {
        let mut a = Array2::<f32>::zeros((n_data + k_shield, d));
        a.slice_mut(ndarray::s![..n_data, ..]).assign(&q);
        for i in 0..k_shield {
            let mut row = a.row_mut(n_data + i);
            for v in row.iter_mut() {
                *v = 5.0 * (rng.random::<f32>() * 2.0 - 1.0);
            }
        }
        a
    };
    let k_aug = {
        let mut a = Array2::<f32>::zeros((n_data + k_shield, d));
        a.slice_mut(ndarray::s![..n_data, ..]).assign(&k);
        for i in 0..k_shield {
            let mut row = a.row_mut(n_data + i);
            for v in row.iter_mut() {
                *v = 5.0 * (rng.random::<f32>() * 2.0 - 1.0);
            }
        }
        a
    };
    let v_aug = {
        let mut a = Array2::<f32>::zeros((n_data + k_shield, d));
        a.slice_mut(ndarray::s![..n_data, ..]).assign(&v);
        for i in 0..k_shield {
            let mut row = a.row_mut(n_data + i);
            for v in row.iter_mut() {
                *v = 5.0 * (rng.random::<f32>() * 2.0 - 1.0);
            }
        }
        a
    };

    let augmented = attention(q_aug.view(), k_aug.view(), v_aug.view());
    let stripped = augmented.slice(ndarray::s![..n_data, ..]).to_owned();

    let drift = max_abs_diff(plain.view(), stripped.view());
    // We expect the drift to be SIGNIFICANT — shield rows alter softmax
    // normalization across all n+k tokens. If this assertion fails the
    // assumption "shield rows compose with attention" was wrong — which
    // it is, hence the negative result documented here.
    assert!(
        drift > 1e-2,
        "shield rows should corrupt the data-row attention output; drift={drift}",
    );
}

/// Helper used by the next test — compute row-axis Gram matrix.
fn gram(m: ArrayView2<'_, f32>) -> Array2<f32> {
    m.t().dot(&m)
}

#[test]
fn permutation_alone_preserves_gram() {
    // Document the Gram-leak property of the bare permutation construction:
    // U = π·H has UᵀU = HᵀπᵀπH = HᵀH (permutation matrices are orthogonal).
    // So the column-Gram is identical to the clear-text. This is the same
    // structural leak GELO's orthogonal A has — TwinShield's shield rows
    // close it at the LINEAR PROJECTION level, but cannot close it at the
    // attention level (see shield_rows_corrupt_attention_output above).
    //
    // Implication for Phase 2: the permutation-shielded attention block
    // does NOT remove the activation-Gram leak. That leak is handled by
    // shield rows BEFORE the linear projection produces Q/K/V; the
    // attention block inherits the post-projection state and must not
    // re-introduce the leak.
    let n = 32;
    let d = 16;
    let mut rng = ChaCha20Rng::seed_from_u64(0xCC11FF22);
    let h = random_matrix(n, d, &mut rng);

    let mut perm: Vec<usize> = (0..n).collect();
    perm.shuffle(&mut rng);
    let h_perm = permute_rows(h.view(), &perm);

    let gram_clear = gram(h.view());
    let gram_perm = gram(h_perm.view());

    let drift = max_abs_diff(gram_clear.view(), gram_perm.view());
    assert!(
        drift < 1e-5,
        "permutation alone preserves column-Gram (the leak): drift={drift}",
    );
}

#[test]
fn additive_noise_breaks_gram_identity() {
    // Counterpart to the above: adding Gaussian noise η_Q to πQ breaks the
    // exact Gram identity. Quantifies that the σ=0.01 setting we plan to
    // ship perturbs UᵀU by an amount on the order of σ² · n per entry, so
    // an attacker reading the post-noise Gram off the GPU side does NOT
    // see HᵀH cleanly. This is one strand of the "why noise helps" story.
    let n = 64;
    let d = 128;
    let mut rng = ChaCha20Rng::seed_from_u64(0xBEEFCAFE);
    let q = random_matrix(n, d, &mut rng);

    let gram_clear = gram(q.view());
    let q_noisy = add_gaussian(q.view(), 0.01, &mut rng);
    let gram_noisy = gram(q_noisy.view());

    let drift = max_abs_diff(gram_clear.view(), gram_noisy.view());
    // σ² · n ≈ 1e-4 · 64 = 6.4e-3 per entry. Expect at least this scale.
    assert!(
        drift > 1e-4,
        "noise should perturb the Gram visibly: drift={drift}",
    );
}

#[test]
fn worst_case_drift_scales_with_sigma() {
    // Confirm σ=0.01 drift > σ=0.001 drift (not bit-equal). This is the
    // expected ordering — larger noise should produce more recovery
    // drift. If this fails, something is wrong with the noise injection.
    let n = 32;
    let d = 64;
    let mut rng = ChaCha20Rng::seed_from_u64(0x42424242);
    let q = random_matrix(n, d, &mut rng);
    let k = random_matrix(n, d, &mut rng);
    let v = random_matrix(n, d, &mut rng);
    let plain = attention(q.view(), k.view(), v.view());

    let drift_low = (0..8)
        .map(|_| {
            let r = permutation_shielded_attention(q.view(), k.view(), v.view(), 0.001, &mut rng);
            max_abs_diff(plain.view(), r.view())
        })
        .fold(0.0f32, f32::max);

    let drift_hi = (0..8)
        .map(|_| {
            let r = permutation_shielded_attention(q.view(), k.view(), v.view(), 0.01, &mut rng);
            max_abs_diff(plain.view(), r.view())
        })
        .fold(0.0f32, f32::max);

    assert!(
        drift_hi > drift_low,
        "σ=0.01 drift {drift_hi} should exceed σ=0.001 drift {drift_low}",
    );
}

#[allow(dead_code)]
fn _l2_distance(a: ArrayView2<'_, f32>, b: ArrayView2<'_, f32>) -> f32 {
    let diff: Array1<f32> = (&a.to_owned() - &b.to_owned())
        .map_axis(Axis(1), |row| row.iter().map(|v| v * v).sum::<f32>().sqrt())
        .into();
    diff.iter().copied().fold(0.0f32, f32::max)
}

// ---------------------------------------------------------------------------
// Phase 2: substrate-level integration tests for offload_attention_permuted.
//
// These exercise the trait method through both InProcessTrustedExecutor
// (the protocol implementer) and PlaintextExecutor (the parity baseline)
// and assert the trait API composes the math correctly.
// ---------------------------------------------------------------------------

use gelo_protocol::rng::MaskSeed;
use gelo_protocol::{
    InProcessTrustedExecutor, PermAttnConfig, PlaintextExecutor, ReferenceCpuEngine, TrustedExecutor,
};
use ndarray::Array3;

fn random_q3(h: usize, n: usize, d: usize, rng: &mut ChaCha20Rng) -> Array3<f32> {
    Array3::from_shape_fn((h, n, d), |_| rng.random::<f32>() * 2.0 - 1.0)
}

#[test]
fn trait_method_sigma_zero_matches_plaintext_executor() {
    // InProcessTrustedExecutor with σ=0 must produce bit-exact (to f32 floor)
    // output as PlaintextExecutor — the default impl is plain multi-head
    // attention, and σ=0 in the permuted protocol is mathematically the same.
    let h = 4;
    let n = 16;
    let d_head = 32;
    let scale = 1.0 / (d_head as f32).sqrt();

    let mut rng = ChaCha20Rng::seed_from_u64(0xABCDEF);
    let q = random_q3(h, n, d_head, &mut rng);
    let k = random_q3(h, n, d_head, &mut rng);
    let v = random_q3(h, n, d_head, &mut rng);

    let mut plain_exec = PlaintextExecutor::new(ReferenceCpuEngine::new());
    let plain_out = plain_exec
        .offload_attention_permuted(q.view(), k.view(), v.view(), scale, gelo_protocol::attention::AttentionMask::None)
        .unwrap();

    let mut in_proc = InProcessTrustedExecutor::with_seed(ReferenceCpuEngine::new(), MaskSeed([7u8; 32]))
        .with_perm_attention(PermAttnConfig::DISABLED_NOISE);
    let in_proc_out = in_proc
        .offload_attention_permuted(q.view(), k.view(), v.view(), scale, gelo_protocol::attention::AttentionMask::None)
        .unwrap();

    let drift = plain_out
        .iter()
        .zip(in_proc_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        drift < 1e-5,
        "trait method @ σ=0 must match plaintext exec to f32 floor: drift={drift}",
    );
}

#[test]
fn trait_method_hnm_noise_deviates_bounded() {
    // With σ=0.01 the InProcessTrustedExecutor output differs from
    // PlaintextExecutor, but by a bounded amount.
    let h = 4;
    let n = 32;
    let d_head = 64;
    let scale = 1.0 / (d_head as f32).sqrt();

    let mut rng = ChaCha20Rng::seed_from_u64(0xC0FFEE);
    let q = random_q3(h, n, d_head, &mut rng);
    let k = random_q3(h, n, d_head, &mut rng);
    let v = random_q3(h, n, d_head, &mut rng);

    let mut plain_exec = PlaintextExecutor::new(ReferenceCpuEngine::new());
    let plain_out = plain_exec
        .offload_attention_permuted(q.view(), k.view(), v.view(), scale, gelo_protocol::attention::AttentionMask::None)
        .unwrap();

    let mut in_proc = InProcessTrustedExecutor::with_seed(ReferenceCpuEngine::new(), MaskSeed([9u8; 32]))
        .with_perm_attention(PermAttnConfig::HIDDEN_NO_MORE);
    let in_proc_out = in_proc
        .offload_attention_permuted(q.view(), k.view(), v.view(), scale, gelo_protocol::attention::AttentionMask::None)
        .unwrap();

    let drift = plain_out
        .iter()
        .zip(in_proc_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        drift > 0.0 && drift < 5e-2,
        "σ=0.01 should deviate but stay bounded: drift={drift}",
    );
}

#[test]
fn trait_method_seed_determinism() {
    // Same seed + same inputs ⇒ same output. The permutation is
    // ChaCha20-driven, so two executors with the same seed produce
    // identical π and identical noise.
    let h = 2;
    let n = 8;
    let d_head = 16;
    let scale = 1.0 / (d_head as f32).sqrt();

    let mut rng = ChaCha20Rng::seed_from_u64(0x42);
    let q = random_q3(h, n, d_head, &mut rng);
    let k = random_q3(h, n, d_head, &mut rng);
    let v = random_q3(h, n, d_head, &mut rng);

    let seed = MaskSeed([0xAAu8; 32]);
    let mut exec1 = InProcessTrustedExecutor::with_seed(ReferenceCpuEngine::new(), seed)
        .with_perm_attention(PermAttnConfig::HIDDEN_NO_MORE);
    let out1 = exec1
        .offload_attention_permuted(q.view(), k.view(), v.view(), scale, gelo_protocol::attention::AttentionMask::None)
        .unwrap();

    let mut exec2 = InProcessTrustedExecutor::with_seed(ReferenceCpuEngine::new(), seed)
        .with_perm_attention(PermAttnConfig::HIDDEN_NO_MORE);
    let out2 = exec2
        .offload_attention_permuted(q.view(), k.view(), v.view(), scale, gelo_protocol::attention::AttentionMask::None)
        .unwrap();

    let drift = out1
        .iter()
        .zip(out2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        drift < 1e-7,
        "same seed must yield bit-identical output: drift={drift}",
    );
}

#[test]
fn empirical_direction_recovery_is_bounded() {
    // Phase 5 (lighter): the engine's view of (πQ + η_Q) — which is what
    // a curious GPU side observes — must not trivially leak Q's row
    // directions via cosine-matching against the public-base / unmasked
    // checkpoint. ARROWMATCH-style attacks (Wang et al. USENIX Sec '25)
    // succeed by finding `argmax_j cos(W_obs[i], W_base[j])` to recover
    // the row permutation; the per-batch fresh π plus σ-noise should
    // prevent any single batch's observation from giving a confident
    // mapping back to the cleartext Q.
    //
    // Test setup: generate Q, k=σ=0.01, sample 16 fresh (π, η) batches,
    // compute argmax-cos mapping from observed rows back to original Q,
    // measure recovery rate.
    let n = 32;
    let d = 64;
    let mut rng = ChaCha20Rng::seed_from_u64(0xA77ACC);
    let q = Array2::<f32>::from_shape_fn((n, d), |_| rng.random::<f32>() * 2.0 - 1.0);

    // Pre-compute per-row L2 norms for the cosine denominator.
    let row_norm = |m: ArrayView2<'_, f32>, i: usize| -> f32 {
        m.row(i).iter().map(|v| v * v).sum::<f32>().sqrt()
    };
    let q_norms: Vec<f32> = (0..n).map(|i| row_norm(q.view(), i)).collect();

    let sigma = 0.01f32;
    let normal = rand_distr::StandardNormal;

    let mut total_correct = 0usize;
    let trials = 16usize;
    for _ in 0..trials {
        // Sample fresh π, apply to Q (rows), add noise η_Q.
        let mut perm: Vec<usize> = (0..n).collect();
        perm.shuffle(&mut rng);
        let mut q_obs = Array2::<f32>::zeros((n, d));
        for (i, &src) in perm.iter().enumerate() {
            q_obs.row_mut(i).assign(&q.row(src));
        }
        for v in q_obs.iter_mut() {
            let z: f32 = rand_distr::Distribution::sample(&normal, &mut rng);
            *v += sigma * z;
        }

        // Attacker: for each observed row i, find argmax_j cos(q_obs[i], q[j]).
        // If they recover perm[i] = j, that's a success.
        for i in 0..n {
            let row_obs_norm = row_norm(q_obs.view(), i);
            let mut best_j = 0usize;
            let mut best_cos = f32::NEG_INFINITY;
            for j in 0..n {
                let mut dot = 0.0f32;
                for d_i in 0..d {
                    dot += q_obs[(i, d_i)] * q[(j, d_i)];
                }
                let c = dot / (row_obs_norm * q_norms[j] + 1e-9);
                if c > best_cos {
                    best_cos = c;
                    best_j = j;
                }
            }
            if best_j == perm[i] {
                total_correct += 1;
            }
        }
    }

    // With σ=0.01 the row directions are still mostly preserved (each
    // entry perturbed by 0.01 vs row mean ≈ 0.5), so cosine matching
    // recovers the permutation with high probability. This documents the
    // single-batch direction-recovery threat: σ=0.01 alone is NOT enough
    // to defeat the ARROWMATCH-class attacker who has the cleartext Q.
    //
    // In the deployed protocol Q is not directly cleartext — it's
    // computed from H (which is masked via GELO) and the public W. The
    // ARROWMATCH-class attacker who knows W would still need to recover
    // H from U=A·H, which the orthogonal-mask security argument blocks.
    //
    // So this test ASSERTS the negative — the noise alone doesn't
    // protect Q if the attacker has cleartext Q — to keep us honest
    // about why the overall protocol's security argument relies on
    // GELO's orthogonal mask, not just the per-batch σ-noise.
    let recovery_rate = total_correct as f32 / (n * trials) as f32;
    assert!(
        recovery_rate > 0.5,
        "with cleartext Q available, σ=0.01 noise alone permits direction \
         recovery >50%: got {recovery_rate} — this is the expected negative \
         result; protocol security relies on Q not being cleartext, which is \
         the GELO orthogonal-mask invariant.",
    );
    // Document the upper bound too so a future tightening (e.g. larger σ
    // or per-row noise scaling) lowers this number visibly.
    eprintln!(
        "[security] σ=0.01 cleartext-Q direction-recovery rate: {:.1}% ({} of {} trials × {} rows)",
        100.0 * recovery_rate,
        total_correct,
        trials,
        n,
    );
}

/// Gate 2 (the `perm_kv` clock) — σ-vs-N permutation-recovery sweep for
/// the persistent-K/V design (`docs/plans/perm-attn-gpu-offload.md`).
///
/// Holding `perm_kv` fixed across N decode steps removes the
/// fresh-per-call protection: an attacker who observes the permuted+noised
/// rows under ONE fixed π can denoise by averaging N observations
/// (the canonical Hidden-No-More √N mechanism). This sweep measures the
/// ARROWMATCH cosine-recovery rate vs (σ, N) and, alongside it, the
/// attention output drift (the quality ceiling) at each σ — so the gate
/// window [security floor, quality ceiling] is visible in one table.
///
/// Two brackets on adversary power (the truth lies between):
///   * N=1  — the design as specced: K cache uploaded once, noise baked
///            in, fixed for the block → one noisy observation.
///   * N>1  — if the N decode steps leak independent noisy views of the
///            fixed-π positions, the attacker averages → noise ↓ √N.
///
/// Reference = cleartext Q (worst-case attacker, per
/// `empirical_direction_recovery_is_bounded`). NOTE: random activations —
/// representative for the *direction*-recovery channel; the *content*
/// channel (gate 3, `O_v` covariance-alignment) is activation-structure
/// dependent and needs real anisotropic K/V + ICA/JADE (Python harness).
#[test]
fn gate2_perm_recovery_vs_sigma_and_n() {
    let n = 256usize;
    let d = 128usize; // production head_dim
    let trials = 8usize;
    let sigmas = [0.0f32, 0.01, 0.04, 0.08, 0.15, 0.30, 0.60, 1.2, 2.4, 5.0, 10.0];
    let n_obs = [1usize, 8, 16, 64];
    let mut rng = ChaCha20Rng::seed_from_u64(0x6A7E2);

    let q = Array2::<f32>::from_shape_fn((n, d), |_| rng.random::<f32>() * 2.0 - 1.0);
    let q_norms: Vec<f32> = (0..n)
        .map(|i| q.row(i).iter().map(|v| v * v).sum::<f32>().sqrt())
        .collect();
    let normal = StandardNormal;

    // Recovery rate: average N noisy observations of πQ, then argmax-cos
    // each averaged row back to cleartext Q; fraction matching π.
    let recover = |sigma: f32, n_avg: usize, rng: &mut ChaCha20Rng| -> f32 {
        let mut correct = 0usize;
        for _ in 0..trials {
            let mut perm: Vec<usize> = (0..n).collect();
            perm.shuffle(rng);
            // N noisy observations of the same permuted Q, averaged.
            let mut acc = Array2::<f32>::zeros((n, d));
            for (i, &src) in perm.iter().enumerate() {
                acc.row_mut(i).assign(&q.row(src));
            }
            let mut avg = acc.clone();
            for v in avg.iter_mut() {
                let mut s = 0.0f32;
                for _ in 0..n_avg {
                    let z: f32 = normal.sample(rng);
                    s += sigma * z;
                }
                *v += s / n_avg as f32; // mean of N noise draws → std ↓ √N
            }
            for i in 0..n {
                let obs_norm = avg.row(i).iter().map(|v| v * v).sum::<f32>().sqrt();
                let (mut best_j, mut best_c) = (0usize, f32::NEG_INFINITY);
                for j in 0..n {
                    let dot: f32 = (0..d).map(|t| avg[(i, t)] * q[(j, t)]).sum();
                    let c = dot / (obs_norm * q_norms[j] + 1e-9);
                    if c > best_c {
                        best_c = c;
                        best_j = j;
                    }
                }
                if best_j == perm[i] {
                    correct += 1;
                }
            }
        }
        correct as f32 / (n * trials) as f32
    };

    // Quality ceiling: attention output drift at each σ (vs plain).
    let drift_at = |sigma: f32, rng: &mut ChaCha20Rng| -> f32 {
        let qa = random_matrix(64, d, rng);
        let ka = random_matrix(64, d, rng);
        let va = random_matrix(64, d, rng);
        let plain = attention(qa.view(), ka.view(), va.view());
        let rec = permutation_shielded_attention(qa.view(), ka.view(), va.view(), sigma, rng);
        max_abs_diff(plain.view(), rec.view())
    };

    eprintln!("[gate2] perm_kv σ-vs-N recovery (random Q, cleartext ref, worst case); chance = {:.3}", 1.0 / n as f32);
    eprintln!("  σ      N=1     N=8     N=16    N=64    | attn-drift");
    for &sigma in &sigmas {
        let r: Vec<f32> = n_obs.iter().map(|&m| recover(sigma, m, &mut rng)).collect();
        let drift = drift_at(sigma, &mut rng);
        eprintln!(
            "  {:<6.3} {:<7.3} {:<7.3} {:<7.3} {:<7.3} | {:.4}",
            sigma, r[0], r[1], r[2], r[3], drift
        );
    }

    // Sanity: σ=0 → perfect recovery at any N; recovery is monotone
    // non-increasing in σ at fixed N, and non-decreasing in N at fixed σ.
    assert!(recover(0.0, 1, &mut rng) > 0.99, "σ=0 must be fully recoverable");
}

#[test]
fn trait_method_cached_sigma_zero_matches_plaintext_executor() {
    // Asymmetric Q × KV at decode shape (n_q=1, q_pos_offset=n_kv-1)
    // and continuation-prefill shape (n_q small, q_pos_offset > 0).
    // The InProcess masked path must produce bit-exact (f32 floor)
    // output as the Plaintext baseline at σ=0.
    let h = 4;
    let d_head = 32;
    let scale = 1.0 / (d_head as f32).sqrt();

    for (n_q, n_kv) in [(1usize, 64usize), (4, 32), (8, 8)] {
        let q_pos_offset = n_kv - n_q;
        let mut rng = ChaCha20Rng::seed_from_u64(0xCACE_DECA ^ (n_q as u64 * 31 + n_kv as u64));
        let q = random_q3(h, n_q, d_head, &mut rng);
        let k = random_q3(h, n_kv, d_head, &mut rng);
        let v = random_q3(h, n_kv, d_head, &mut rng);

        let mut plain_exec = PlaintextExecutor::new(ReferenceCpuEngine::new());
        let plain_out = plain_exec
            .offload_attention_permuted_cached(
                q.view(),
                k.view(),
                v.view(),
                scale,
                q_pos_offset,
                gelo_protocol::attention::AttentionMask::Causal,
            )
            .unwrap();

        let mut in_proc = InProcessTrustedExecutor::with_seed(
            ReferenceCpuEngine::new(),
            MaskSeed([42u8 ^ n_q as u8 ^ n_kv as u8; 32]),
        )
        .with_perm_attention(PermAttnConfig::DISABLED_NOISE);
        let in_proc_out = in_proc
            .offload_attention_permuted_cached(
                q.view(),
                k.view(),
                v.view(),
                scale,
                q_pos_offset,
                gelo_protocol::attention::AttentionMask::Causal,
            )
            .unwrap();

        let drift = plain_out
            .iter()
            .zip(in_proc_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            drift < 1e-5,
            "trait-method cached parity at σ=0 must hold to f32 floor: \
             n_q={n_q} n_kv={n_kv} q_pos_offset={q_pos_offset} drift={drift}",
        );
    }
}

#[test]
fn phase_1b_decode_softmax_on_gpu_matches_in_tee() {
    // Phase 1b: at decode (n_q=1, q_pos_offset = n_kv − 1) the causal
    // mask is structurally a no-op, so softmax can run on the engine
    // without the F1+ mask-pattern leak.  This test pins the parity
    // claim: PermAttnConfig with `decode_softmax_on_gpu = true` and
    // σ=0 must produce the same output as the legacy in-TEE softmax
    // path (`DISABLED_NOISE`) to f32 floor.  The RNG state advances
    // identically on both paths because the row-sum integrity probe
    // consumes the same two `next_u32`s the in-TEE path doesn't —
    // hence we use separate executors (one config each) rather than
    // toggling at runtime on the same RNG.
    let h = 4;
    let d_head = 32;
    let scale = 1.0 / (d_head as f32).sqrt();

    for n_kv in [1usize, 8, 64, 256] {
        let n_q = 1;
        let q_pos_offset = n_kv - n_q;
        let mut rng = ChaCha20Rng::seed_from_u64(0xDECA_F00D ^ n_kv as u64);
        let q = random_q3(h, n_q, d_head, &mut rng);
        let k = random_q3(h, n_kv, d_head, &mut rng);
        let v = random_q3(h, n_kv, d_head, &mut rng);

        let mut in_tee_exec = InProcessTrustedExecutor::with_seed(
            ReferenceCpuEngine::new(),
            MaskSeed([99u8 ^ n_kv as u8; 32]),
        )
        .with_perm_attention(PermAttnConfig::DISABLED_NOISE);
        let in_tee_out = in_tee_exec
            .offload_attention_permuted_cached(
                q.view(),
                k.view(),
                v.view(),
                scale,
                q_pos_offset,
                gelo_protocol::attention::AttentionMask::Causal,
            )
            .unwrap();

        let mut gpu_exec = InProcessTrustedExecutor::with_seed(
            ReferenceCpuEngine::new(),
            MaskSeed([99u8 ^ n_kv as u8; 32]),
        )
        .with_perm_attention(PermAttnConfig {
            noise_sigma: 0.0,
            causal_mask_neg: 30.0,
            decode_softmax_on_gpu: true,
            feature_rotation: false,
        });
        let gpu_out = gpu_exec
            .offload_attention_permuted_cached(
                q.view(),
                k.view(),
                v.view(),
                scale,
                q_pos_offset,
                gelo_protocol::attention::AttentionMask::Causal,
            )
            .unwrap();

        let drift = in_tee_out
            .iter()
            .zip(gpu_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            drift < 1e-5,
            "Phase 1b decode-softmax-on-GPU parity must hold to f32 floor at \
             decode shape n_q=1 n_kv={n_kv}: drift={drift}",
        );
    }
}

#[test]
fn phase_1b_prefill_falls_back_to_in_tee_softmax() {
    // Negative guard: at prefill (n_q > 1) the causal mask is NOT a
    // no-op, so the F1+ mask-pattern leak applies and softmax MUST
    // stay in-TEE regardless of `decode_softmax_on_gpu`.  Parity
    // with `DISABLED_NOISE` (which always uses in-TEE softmax) is
    // therefore expected — if it failed, the flag would be leaking
    // into prefill, which is a security bug.
    let h = 4;
    let d_head = 32;
    let n_q = 8;
    let n_kv = 16;
    let q_pos_offset = n_kv - n_q;
    let scale = 1.0 / (d_head as f32).sqrt();

    let mut rng = ChaCha20Rng::seed_from_u64(0xFACE_F00D);
    let q = random_q3(h, n_q, d_head, &mut rng);
    let k = random_q3(h, n_kv, d_head, &mut rng);
    let v = random_q3(h, n_kv, d_head, &mut rng);

    let make_executor = |cfg: PermAttnConfig| {
        InProcessTrustedExecutor::with_seed(
            ReferenceCpuEngine::new(),
            MaskSeed([55u8; 32]),
        )
        .with_perm_attention(cfg)
    };

    let mut a = make_executor(PermAttnConfig::DISABLED_NOISE);
    let out_a = a
        .offload_attention_permuted_cached(
            q.view(),
            k.view(),
            v.view(),
            scale,
            q_pos_offset,
            gelo_protocol::attention::AttentionMask::Causal,
        )
        .unwrap();

    let mut b = make_executor(PermAttnConfig {
        noise_sigma: 0.0,
        causal_mask_neg: 30.0,
        decode_softmax_on_gpu: true,
        feature_rotation: false,
    });
    let out_b = b
        .offload_attention_permuted_cached(
            q.view(),
            k.view(),
            v.view(),
            scale,
            q_pos_offset,
            gelo_protocol::attention::AttentionMask::Causal,
        )
        .unwrap();

    let drift = out_a
        .iter()
        .zip(out_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        drift < 1e-5,
        "Phase 1b must not affect prefill (n_q={n_q}, n_kv={n_kv}): drift={drift} \
         — if non-zero, decode_softmax_on_gpu is leaking into the prefill path",
    );
}

#[test]
fn trait_method_causal_mask_matches_plaintext() {
    // With AttentionMask::Causal, the InProcessTrustedExecutor must
    // produce output equivalent to the PlaintextExecutor's causal
    // path (which the default trait impl computes via -inf upper
    // triangle). Validates the permuted-causal-mask math.
    let h = 4;
    let n = 16;
    let d_head = 32;
    let scale = 1.0 / (d_head as f32).sqrt();

    let mut rng = ChaCha20Rng::seed_from_u64(0xC0DEC0DE);
    let q = random_q3(h, n, d_head, &mut rng);
    let k = random_q3(h, n, d_head, &mut rng);
    let v = random_q3(h, n, d_head, &mut rng);

    let mut plain_exec = PlaintextExecutor::new(ReferenceCpuEngine::new());
    let plain_out = plain_exec
        .offload_attention_permuted(
            q.view(),
            k.view(),
            v.view(),
            scale,
            gelo_protocol::attention::AttentionMask::Causal,
        )
        .unwrap();

    let mut in_proc = InProcessTrustedExecutor::with_seed(ReferenceCpuEngine::new(), MaskSeed([4u8; 32]))
        .with_perm_attention(PermAttnConfig::DISABLED_NOISE);
    let in_proc_out = in_proc
        .offload_attention_permuted(
            q.view(),
            k.view(),
            v.view(),
            scale,
            gelo_protocol::attention::AttentionMask::Causal,
        )
        .unwrap();

    let drift = plain_out
        .iter()
        .zip(in_proc_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        drift < 1e-5,
        "permuted causal mask must reproduce plain causal attention at σ=0: drift={drift}",
    );
}

// ---------------------------------------------------------------------------
// M1.10.0.3 — Phase-0 regression: engine cannot recover π via the exact-zero
// count attack on the softmax output. The F1+ resolution (in-TEE softmax +
// soft causal mask -C = 30, replacing -∞) removes the trivial mask-pattern
// leak. See docs/plans/m1-10-security-review.md.
//
// What this test DEFENDS:
//   (a) Exact-zero count attack: if blocked positions softmax to exactly 0
//       (-∞ mask), the per-row count of zeros equals (n - 1 - perm[i])
//       and recovers π directly. F1+ uses -C = 30 so blocked positions
//       softmax to ~exp(-30) ≈ 9.4e-14, not exact zero. This test asserts
//       the captured tensor passed to the 2nd matmul_dynamic_batched
//       (= probs in the permuted_attention flow) contains zero exact-zero
//       entries per row.
//   (b) Softmax-on-GPU leak: even with -C, sending the scaled+masked score
//       tensor through `engine.softmax_batched` would expose the mask
//       pattern at the input layer. F1+ keeps softmax in-TEE; this test
//       asserts the engine's `softmax_batched` is NEVER invoked during a
//       Causal-masked permuted_attention call.
//
// What this test does NOT cover (documented gap; F1++ is the follow-up):
//   * Threshold-count attack — counting probs entries below some small
//     threshold T (e.g. 1e-12) still recovers per-row blocked counts and
//     hence π. The standard mitigation is to add small Gaussian noise to
//     probs in-TEE before sending to GPU (σ ≈ 1e-6, comfortably below
//     parity tolerance). Tracked as a follow-up in the security review.
// ---------------------------------------------------------------------------

use anyhow::Result;
use gelo_protocol::WeightHandle;
use gelo_protocol::attention as ga;
use gelo_protocol::substrate::GpuOffloadEngine;
use ndarray::ArrayView3;
use std::sync::Mutex;

/// Engine that wraps an honest `ReferenceCpuEngine` and records every input
/// the engine sees from a `permuted_attention` forward pass. Used to
/// run recovery attacks on the captured tensors.
struct SpyEngine {
    inner: Mutex<ReferenceCpuEngine>,
    softmax_call_count: Mutex<usize>,
    /// Captures every LHS tensor passed to `matmul_dynamic_batched`,
    /// in call order. Call 0 is `Q · Kᵀ` (input: permuted noisy Q);
    /// call 1 is `probs · V` (input: softmax output — the tensor the
    /// exact-zero attack would target).
    matmul_batched_lhs: Mutex<Vec<ndarray::Array3<f32>>>,
}

impl SpyEngine {
    fn new() -> Self {
        Self {
            inner: Mutex::new(ReferenceCpuEngine::new()),
            softmax_call_count: Mutex::new(0),
            matmul_batched_lhs: Mutex::new(Vec::new()),
        }
    }
}

impl GpuOffloadEngine for SpyEngine {
    fn register_weight(&mut self, handle: WeightHandle, weight: ArrayView2<f32>) -> Result<()> {
        self.inner.lock().unwrap().register_weight(handle, weight)
    }
    fn matmul(&self, handle: WeightHandle, input: ArrayView2<f32>) -> Result<ndarray::Array2<f32>> {
        self.inner.lock().unwrap().matmul(handle, input)
    }
    fn matmul_dynamic(
        &self,
        lhs: ArrayView2<f32>,
        rhs: ArrayView2<f32>,
    ) -> Result<ndarray::Array2<f32>> {
        self.inner.lock().unwrap().matmul_dynamic(lhs, rhs)
    }
    fn matmul_dynamic_batched(
        &self,
        lhs: ArrayView3<f32>,
        rhs: ArrayView3<f32>,
    ) -> Result<ndarray::Array3<f32>> {
        self.matmul_batched_lhs.lock().unwrap().push(lhs.to_owned());
        self.inner.lock().unwrap().matmul_dynamic_batched(lhs, rhs)
    }
    fn softmax_batched(&self, input: ArrayView3<f32>) -> Result<ndarray::Array3<f32>> {
        *self.softmax_call_count.lock().unwrap() += 1;
        self.inner.lock().unwrap().softmax_batched(input)
    }
}

#[test]
fn f1plus_softmax_runs_in_tee_not_on_engine() {
    // M1.10.0.3 part (i): under F1+, the engine's softmax_batched
    // MUST NOT be invoked by permuted_attention. Softmax runs in-TEE
    // so the mask pattern (which the score+mask tensor would carry)
    // never reaches the engine.
    let h = 4;
    let n = 32;
    let d_head = 16;
    let scale = 1.0 / (d_head as f32).sqrt();

    let spy = SpyEngine::new();
    let mut rng = ChaCha20Rng::seed_from_u64(0xF1A5_F1A5);
    let q = random_q3(h, n, d_head, &mut rng);
    let k = random_q3(h, n, d_head, &mut rng);
    let v = random_q3(h, n, d_head, &mut rng);

    let _ = ga::permuted_attention(
        &spy,
        q.view(),
        k.view(),
        v.view(),
        scale,
        ga::AttentionMask::Causal,
        PermAttnConfig::HIDDEN_NO_MORE,
        &mut rng,
    )
    .expect("permuted_attention through spy");

    let softmax_calls = *spy.softmax_call_count.lock().unwrap();
    assert_eq!(
        softmax_calls, 0,
        "F1+ resolution: engine.softmax_batched MUST NOT be invoked under \
         Causal mask (softmax must run in-TEE). Observed {softmax_calls} calls.",
    );

    // Sanity: two matmul_dynamic_batched calls (Q·Kᵀ and probs·V) are expected.
    let matmul_calls = spy.matmul_batched_lhs.lock().unwrap().len();
    assert_eq!(
        matmul_calls, 2,
        "permuted_attention should issue exactly 2 batched matmul calls (Q·Kᵀ \
         and probs·V); observed {matmul_calls}.",
    );
}

#[test]
fn f1plus_probs_have_no_exact_zeros_under_causal_mask() {
    // M1.10.0.3 part (ii): the softmax output (= LHS of the 2nd
    // batched matmul, = `probs` in permuted_attention) must contain
    // zero exact-zero entries under F1+. With a `-∞` mask, blocked
    // positions softmax to 0 exactly and trivially leak π via per-row
    // exact-zero counting. With `-C = 30`, blocked positions softmax
    // to ~exp(-30) ≈ 9.4e-14 — small but representable; the exact-zero
    // attack yields no signal.
    let h = 4;
    let d_head = 16;
    let scale = 1.0 / (d_head as f32).sqrt();

    for n in [64usize, 256, 1024] {
        let spy = SpyEngine::new();
        let mut rng = ChaCha20Rng::seed_from_u64(0xCAFE_DAD_u64.wrapping_add(n as u64));
        let q = random_q3(h, n, d_head, &mut rng);
        let k = random_q3(h, n, d_head, &mut rng);
        let v = random_q3(h, n, d_head, &mut rng);

        let _ = ga::permuted_attention(
            &spy,
            q.view(),
            k.view(),
            v.view(),
            scale,
            ga::AttentionMask::Causal,
            PermAttnConfig::HIDDEN_NO_MORE,
            &mut rng,
        )
        .expect("permuted_attention through spy");

        // 2nd matmul LHS = the softmax output (probs).
        let captured = spy.matmul_batched_lhs.lock().unwrap();
        assert_eq!(captured.len(), 2);
        let probs = &captured[1];

        // For each row across all heads, count exact zeros. Under F1+
        // every row should have zero exact-zero entries (the soft mask
        // produces ~1e-14, not 0). The original -∞ implementation
        // would produce row-i count of (n - 1 - perm[i]) and trivially
        // leak π via Spearman correlation.
        let mut max_exact_zeros_per_row = 0usize;
        for hi in 0..probs.shape()[0] {
            for i in 0..probs.shape()[1] {
                let row = probs.slice(ndarray::s![hi, i, ..]);
                let zeros = row.iter().filter(|&&x| x == 0.0_f32).count();
                if zeros > max_exact_zeros_per_row {
                    max_exact_zeros_per_row = zeros;
                }
            }
        }
        assert_eq!(
            max_exact_zeros_per_row, 0,
            "F1+ resolution: probs must have no exact-zero entries (mask uses \
             -C=30 not -∞). At n={n}, max exact zeros in any row = \
             {max_exact_zeros_per_row}; this would leak π via exact-zero counting.",
        );
    }
}

#[test]
fn f1plus_baseline_neg_inf_demonstration_only() {
    // M1.10.0.3 part (iii): demonstration that the F1+ improvement is
    // not vacuous. We simulate what would happen if we'd kept `-∞` as
    // the mask: per-row exact-zero count perfectly recovers π. This is
    // the attack F1+ is designed to defeat. The test asserts the
    // attack SUCCEEDS on the `-∞` baseline (to lock in the threat as
    // real), motivating the F1+ change above.
    //
    // We synthesise the baseline by directly building the softmax of a
    // -∞-masked score tensor and running the attack on it; we do NOT
    // exercise the production code with -∞ since the production code
    // no longer admits that path.
    let n = 64usize;
    let mut rng = ChaCha20Rng::seed_from_u64(0xDEAD_C0DE);
    let mut perm: Vec<usize> = (0..n).collect();
    perm.shuffle(&mut rng);

    // Build a single-head random score row with realistic magnitudes,
    // then add -∞ at "blocked" positions per the permuted causal mask.
    let mut scores = Array2::<f32>::from_shape_fn((n, n), |_| rng.random::<f32>() * 2.0 - 1.0);
    for i in 0..n {
        for j in 0..n {
            if perm[j] > perm[i] {
                scores[(i, j)] = f32::NEG_INFINITY;
            }
        }
    }

    let probs = softmax_rowwise(scores.view());
    let mut recovered = vec![0usize; n];
    for i in 0..n {
        let row = probs.slice(ndarray::s![i, ..]);
        let blocked = row.iter().filter(|&&x| x == 0.0_f32).count();
        // perm[i] = n - 1 - blocked.
        recovered[i] = n.saturating_sub(1).saturating_sub(blocked);
    }

    // Under -∞ this recovery is perfect.
    assert_eq!(
        recovered, perm,
        "baseline -∞ leak demonstration: per-row exact-zero count should \
         exactly recover π. This is the attack F1+'s -C mask removes.",
    );
}

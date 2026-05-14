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
    InProcessTrustedExecutor, PermAttnConfig, PlaintextExecutor, RayonCpuEngine, TrustedExecutor,
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

    let mut plain_exec = PlaintextExecutor::new(RayonCpuEngine::new());
    let plain_out = plain_exec
        .offload_attention_permuted(q.view(), k.view(), v.view(), scale, gelo_protocol::attention::AttentionMask::None)
        .unwrap();

    let mut in_proc = InProcessTrustedExecutor::with_seed(RayonCpuEngine::new(), MaskSeed([7u8; 32]))
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

    let mut plain_exec = PlaintextExecutor::new(RayonCpuEngine::new());
    let plain_out = plain_exec
        .offload_attention_permuted(q.view(), k.view(), v.view(), scale, gelo_protocol::attention::AttentionMask::None)
        .unwrap();

    let mut in_proc = InProcessTrustedExecutor::with_seed(RayonCpuEngine::new(), MaskSeed([9u8; 32]))
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
    let mut exec1 = InProcessTrustedExecutor::with_seed(RayonCpuEngine::new(), seed)
        .with_perm_attention(PermAttnConfig::HIDDEN_NO_MORE);
    let out1 = exec1
        .offload_attention_permuted(q.view(), k.view(), v.view(), scale, gelo_protocol::attention::AttentionMask::None)
        .unwrap();

    let mut exec2 = InProcessTrustedExecutor::with_seed(RayonCpuEngine::new(), seed)
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

    let mut plain_exec = PlaintextExecutor::new(RayonCpuEngine::new());
    let plain_out = plain_exec
        .offload_attention_permuted(
            q.view(),
            k.view(),
            v.view(),
            scale,
            gelo_protocol::attention::AttentionMask::Causal,
        )
        .unwrap();

    let mut in_proc = InProcessTrustedExecutor::with_seed(RayonCpuEngine::new(), MaskSeed([4u8; 32]))
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

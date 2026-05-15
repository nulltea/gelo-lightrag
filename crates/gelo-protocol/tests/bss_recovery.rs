//! Regression tests for the Gram-matrix leak addressed by shield vectors.
//!
//! These tests intercept the masked activations that the trusted executor
//! ships to its untrusted engine, then attempt the simplest attack: compute
//! `UᵀU` and compare to `HᵀH`.
//!
//! - **Bare orthogonal mask** (`ShieldConfig::NONE`): GELO §4 shows
//!   `UᵀU = HᵀAᵀAH = HᵀH`, so the attacker recovers the Gram matrix in full.
//!   This test confirms the leak so we don't accidentally "fix" it by
//!   silently changing the mask construction.
//! - **Orthogonal + shield**: `UᵀU = [H;S]ᵀ[H;S] = HᵀH + SᵀS`, which
//!   differs from `HᵀH` by a quantity proportional to the shield energy.
//!   This test asserts the Frobenius distance is large.

use std::sync::Mutex;

use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

use gelo_protocol::rng::MaskSeed;
use gelo_protocol::{
    GpuOffloadEngine, InProcessTrustedExecutor, RayonCpuEngine, ShieldConfig, TrustedExecutor,
    WeightHandle, WeightKind,
};

/// Wraps another engine, recording every input matrix it sees.
struct SnoopingEngine {
    inner: RayonCpuEngine,
    captured: Mutex<Vec<Array2<f32>>>,
}

impl SnoopingEngine {
    fn new(inner: RayonCpuEngine) -> Self {
        Self {
            inner,
            captured: Mutex::new(Vec::new()),
        }
    }

    fn last_capture(&self) -> Array2<f32> {
        let guard = self.captured.lock().unwrap();
        guard.last().expect("no captures").clone()
    }
}

impl GpuOffloadEngine for SnoopingEngine {
    fn register_weight(&mut self, handle: WeightHandle, weight: ArrayView2<'_, f32>) -> anyhow::Result<()> {
        self.inner.register_weight(handle, weight)
    }

    fn matmul(
        &self,
        handle: WeightHandle,
        input: ArrayView2<'_, f32>,
    ) -> anyhow::Result<Array2<f32>> {
        self.captured.lock().unwrap().push(input.to_owned());
        self.inner.matmul(handle, input)
    }

    fn matmul_dynamic(
        &self,
        lhs: ArrayView2<'_, f32>,
        rhs: ArrayView2<'_, f32>,
    ) -> anyhow::Result<Array2<f32>> {
        self.inner.matmul_dynamic(lhs, rhs)
    }
}

fn frobenius_distance(a: ArrayView2<'_, f32>, b: ArrayView2<'_, f32>) -> f32 {
    assert_eq!(a.shape(), b.shape());
    let mut acc = 0.0_f32;
    for ((i, j), v) in a.indexed_iter() {
        let d = v - b[[i, j]];
        acc += d * d;
    }
    acc.sqrt()
}

fn random_matrix(rows: usize, cols: usize, rng: &mut impl rand::RngCore) -> Array2<f32> {
    let normal = StandardNormal;
    Array2::from_shape_fn((rows, cols), |_| {
        <StandardNormal as Distribution<f32>>::sample(&normal, rng)
    })
}

#[test]
fn bare_orthogonal_mask_leaks_gram_matrix() {
    let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
    let n = 32;
    let d = 24;
    let p = 16;

    let hidden = random_matrix(n, d, &mut rng);
    let weight = random_matrix(d, p, &mut rng);

    let mut engine = SnoopingEngine::new(RayonCpuEngine::new());
    let handle = WeightHandle::new(0, WeightKind::Q);
    engine.register_weight(handle, weight.view()).unwrap();

    let mut exec = InProcessTrustedExecutor::with_seed(engine, MaskSeed::from_bytes([1u8; 32]))
        .with_per_offload_mask();
    exec.offload_linear(handle, hidden.view()).unwrap();

    let u = exec.engine().last_capture();
    let ut_u = u.t().dot(&u);
    let ht_h = hidden.t().dot(&hidden);
    let leak = frobenius_distance(ut_u.view(), ht_h.view());

    // With bare orthogonal A and no shield, UᵀU and HᵀH must agree to within
    // numerical roundoff — that *is* the leak.
    let baseline = ht_h.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        leak < 1e-2 * baseline.max(1.0),
        "expected UᵀU ≈ HᵀH for bare mask, got Frobenius distance {leak} (baseline {baseline})",
    );
}

#[test]
fn shielded_mask_masks_gram_matrix() {
    let mut rng = ChaCha20Rng::from_seed([19u8; 32]);
    let n = 32;
    let d = 24;
    let p = 16;
    let k = 16;
    let energy_scale = 6.0;

    let hidden = random_matrix(n, d, &mut rng);
    let weight = random_matrix(d, p, &mut rng);

    let mut engine = SnoopingEngine::new(RayonCpuEngine::new());
    let handle = WeightHandle::new(0, WeightKind::Q);
    engine.register_weight(handle, weight.view()).unwrap();

    let mut exec = InProcessTrustedExecutor::with_shield(
        engine,
        MaskSeed::from_bytes([23u8; 32]),
        ShieldConfig::new(k, energy_scale),
    );
    exec.offload_linear(handle, hidden.view()).unwrap();

    let u = exec.engine().last_capture();
    assert_eq!(u.nrows(), n + k, "shielded U should have n+k rows");

    let ut_u = u.t().dot(&u);
    let ht_h = hidden.t().dot(&hidden);
    let baseline = ht_h.iter().map(|v| v * v).sum::<f32>().sqrt();
    let leak = frobenius_distance(ut_u.view(), ht_h.view());

    // SᵀS contributes roughly k·d·σ² to the Frobenius distance. With
    // energy_scale = 6 and unit-norm H rows this dwarfs HᵀH.
    assert!(
        leak > baseline,
        "shielded mask should hide HᵀH: leak {leak} ≤ baseline {baseline}",
    );
}

#[test]
fn shielded_executor_preserves_functional_output() {
    // Functional sanity: even with shielding, the unmasked output of
    // offload_linear is exactly H·W (modulo numerical roundoff).
    let mut rng = ChaCha20Rng::from_seed([41u8; 32]);
    let n = 12;
    let d = 8;
    let p = 5;

    let hidden = random_matrix(n, d, &mut rng);
    let weight = random_matrix(d, p, &mut rng);

    let mut engine = RayonCpuEngine::new();
    let handle = WeightHandle::new(0, WeightKind::Q);
    engine.register_weight(handle, weight.view()).unwrap();

    let mut exec = InProcessTrustedExecutor::with_shield(
        engine,
        MaskSeed::from_bytes([53u8; 32]),
        ShieldConfig::new(8, 5.0),
    );
    let out = exec.offload_linear(handle, hidden.view()).unwrap();
    let expected = hidden.dot(&weight);

    assert_eq!(out.shape(), expected.shape());
    for ((i, j), e) in expected.indexed_iter() {
        let diff = (e - out[[i, j]]).abs();
        // Slightly looser tolerance than the no-shield case because the
        // mask is (n+k)×(n+k) and accumulates more roundoff.
        assert!(diff < 5e-3, "({i},{j}) plain={e} got={}", out[[i, j]]);
    }
}

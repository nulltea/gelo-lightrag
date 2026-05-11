//! U-Verify tamper-detection regression tests.
//!
//! Wraps a normal `RayonCpuEngine` with a `TamperingEngine` that adds a
//! small perturbation to every matmul output. Without verification the
//! executor accepts the corrupted result silently; with `verify_probes > 0`
//! the executor returns `Err`.

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

/// Engine that returns the correct product plus an additive tamper `ε·δ`,
/// where `δ` is a fixed unit-norm random direction so the tampering is
/// deterministic across runs but invisible to scaling.
struct TamperingEngine {
    inner: RayonCpuEngine,
    epsilon: f32,
    /// Counter so we tamper only the Nth call (helps distinguish "caught the
    /// tamper at the right time" from "caught some other call").
    call_count: Mutex<usize>,
    tamper_after: usize,
}

impl TamperingEngine {
    fn new(inner: RayonCpuEngine, epsilon: f32, tamper_after: usize) -> Self {
        Self {
            inner,
            epsilon,
            call_count: Mutex::new(0),
            tamper_after,
        }
    }
}

impl GpuOffloadEngine for TamperingEngine {
    fn register_weight(&mut self, h: WeightHandle, w: ArrayView2<'_, f32>) -> anyhow::Result<()> {
        self.inner.register_weight(h, w)
    }

    fn matmul(&self, h: WeightHandle, input: ArrayView2<'_, f32>) -> anyhow::Result<Array2<f32>> {
        let mut out = self.inner.matmul(h, input)?;
        let mut c = self.call_count.lock().unwrap();
        *c += 1;
        // Tamper a single, fixed element to make the corruption catchable but
        // not so structured that signed-cancellation hides it.
        if *c > self.tamper_after && !out.is_empty() {
            out[[0, 0]] += self.epsilon;
        }
        Ok(out)
    }

    fn matmul_dynamic(
        &self,
        lhs: ArrayView2<'_, f32>,
        rhs: ArrayView2<'_, f32>,
    ) -> anyhow::Result<Array2<f32>> {
        let mut out = self.inner.matmul_dynamic(lhs, rhs)?;
        let mut c = self.call_count.lock().unwrap();
        *c += 1;
        if *c > self.tamper_after && !out.is_empty() {
            out[[0, 0]] += self.epsilon;
        }
        Ok(out)
    }
}

fn rand_matrix(rows: usize, cols: usize, rng: &mut impl rand::RngCore, scale: f32) -> Array2<f32> {
    let normal = StandardNormal;
    Array2::from_shape_fn((rows, cols), |_| {
        <StandardNormal as Distribution<f32>>::sample(&normal, rng) * scale
    })
}

#[test]
fn u_verify_catches_tampered_offload_linear() {
    let mut rng = ChaCha20Rng::from_seed([13u8; 32]);
    let hidden = rand_matrix(16, 32, &mut rng, 1.0);
    let weight = rand_matrix(32, 24, &mut rng, 0.3);
    let handle = WeightHandle::new(0, WeightKind::Q);

    // Tamper every call, eps = 0.5 — well above f32 roundoff slack.
    let mut tampering = TamperingEngine::new(RayonCpuEngine::new(), 0.5, 0);
    tampering.register_weight(handle, weight.view()).unwrap();

    let mut exec = InProcessTrustedExecutor::with_seed(tampering, MaskSeed::from_bytes([15u8; 32]))
        .with_verify_probes(8);
    exec.provision_weight(handle, weight.view()).unwrap();

    let result = exec.offload_linear(handle, hidden.view());
    assert!(
        result.is_err(),
        "U-Verify failed to catch eps=0.5 tampering on offload_linear",
    );
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("U-Verify"),
        "expected U-Verify error message, got: {msg}",
    );
}

#[test]
fn u_verify_passes_honest_engine_offload_linear() {
    let mut rng = ChaCha20Rng::from_seed([14u8; 32]);
    let hidden = rand_matrix(16, 32, &mut rng, 1.0);
    let weight = rand_matrix(32, 24, &mut rng, 0.3);
    let handle = WeightHandle::new(0, WeightKind::Q);

    // No tampering: epsilon = 0.
    let mut honest = TamperingEngine::new(RayonCpuEngine::new(), 0.0, 100);
    honest.register_weight(handle, weight.view()).unwrap();

    let mut exec = InProcessTrustedExecutor::with_seed(honest, MaskSeed::from_bytes([15u8; 32]))
        .with_verify_probes(8);
    exec.provision_weight(handle, weight.view()).unwrap();

    let result = exec.offload_linear(handle, hidden.view());
    assert!(result.is_ok(), "U-Verify rejected an honest engine: {:?}", result.err());
}

#[test]
fn u_verify_catches_tampered_offload_attention_qkt() {
    let mut rng = ChaCha20Rng::from_seed([21u8; 32]);
    let n = 16;
    let d = 32;
    let q = rand_matrix(n, d, &mut rng, 0.5);
    let kt = rand_matrix(d, n, &mut rng, 0.5);

    let tampering = TamperingEngine::new(RayonCpuEngine::new(), 0.5, 0);
    let mut exec =
        InProcessTrustedExecutor::with_seed(tampering, MaskSeed::from_bytes([23u8; 32]))
            .with_verify_probes(8);

    let result = exec.offload_attention_qkt(q.view(), kt.view());
    assert!(
        result.is_err(),
        "U-Verify failed to catch tampering on offload_attention_qkt",
    );
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("U-Verify"));
}

#[test]
fn u_verify_with_shield_still_catches_tampering() {
    // Shield expands the stacked matrix from (n, d) to (n+k, d). The probe
    // must still operate on the expanded matrix correctly.
    let mut rng = ChaCha20Rng::from_seed([29u8; 32]);
    let hidden = rand_matrix(20, 32, &mut rng, 1.0);
    let weight = rand_matrix(32, 24, &mut rng, 0.3);
    let handle = WeightHandle::new(0, WeightKind::Q);

    let mut tampering = TamperingEngine::new(RayonCpuEngine::new(), 0.5, 0);
    tampering.register_weight(handle, weight.view()).unwrap();

    let mut exec = InProcessTrustedExecutor::with_shield(
        tampering,
        MaskSeed::from_bytes([31u8; 32]),
        ShieldConfig::new(8, 6.0),
    )
    .with_verify_probes(8);
    exec.provision_weight(handle, weight.view()).unwrap();

    let result = exec.offload_linear(handle, hidden.view());
    assert!(result.is_err(), "shield+U-Verify missed the tamper");
}

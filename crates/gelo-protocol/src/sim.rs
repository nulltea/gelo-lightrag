use std::collections::HashMap;

use anyhow::{Result, anyhow};
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::integrity::verify_offload;
use crate::mask::GeloMask;
use crate::out_attn_mult;
use crate::profile;
use crate::rng::MaskSeed;
use crate::shield::{ShieldConfig, stack_shield};
use crate::substrate::{GpuOffloadEngine, TrustedExecutor, WeightHandle, WeightKind};

/// Reference [`GpuOffloadEngine`] that performs the offloaded GEMM on the CPU.
/// Stand-in for a real Vulkan / CUDA backend.
#[derive(Default)]
pub struct RayonCpuEngine {
    weights: HashMap<WeightHandle, Array2<f32>>,
}

impl RayonCpuEngine {
    pub fn new() -> Self {
        Self::default()
    }
}

impl GpuOffloadEngine for RayonCpuEngine {
    fn register_weight(&mut self, handle: WeightHandle, weight: ArrayView2<f32>) -> Result<()> {
        self.weights.insert(handle, weight.to_owned());
        Ok(())
    }

    fn matmul(&self, handle: WeightHandle, input: ArrayView2<f32>) -> Result<Array2<f32>> {
        let w = self
            .weights
            .get(&handle)
            .ok_or_else(|| anyhow!("weight {handle:?} not registered with engine"))?;
        if input.ncols() != w.nrows() {
            return Err(anyhow!(
                "matmul shape mismatch: input cols {} != weight rows {}",
                input.ncols(),
                w.nrows()
            ));
        }
        Ok(input.dot(w))
    }

    fn matmul_dynamic(
        &self,
        lhs: ArrayView2<f32>,
        rhs: ArrayView2<f32>,
    ) -> Result<Array2<f32>> {
        if lhs.ncols() != rhs.nrows() {
            return Err(anyhow!(
                "matmul_dynamic shape mismatch: lhs cols {} != rhs rows {}",
                lhs.ncols(),
                rhs.nrows()
            ));
        }
        Ok(lhs.dot(&rhs))
    }
}

/// In-process trusted side. Owns the mask RNG, applies / removes the per-batch
/// orthogonal `A`, optionally stacks shield rows, and delegates the GEMM
/// itself to a [`GpuOffloadEngine`].
///
/// When `verify_probes > 0`, a Freivalds-style U-Verify check is run after
/// every offload to detect byzantine tampering. The executor keeps its own
/// copy of every provisioned weight so the probe can compute `B · r`
/// independently of the engine.
///
/// Useful for tests, parity benchmarks, and any deployment where the trusted
/// boundary is logical rather than hardware-attested.
pub struct InProcessTrustedExecutor<E: GpuOffloadEngine> {
    engine: E,
    rng: ChaCha20Rng,
    shield: ShieldConfig,
    verify_probes: usize,
    weights: HashMap<WeightHandle, Array2<f32>>,
}

impl<E: GpuOffloadEngine> InProcessTrustedExecutor<E> {
    /// Construct with a fresh OS-seeded mask RNG and no shield.
    pub fn new(engine: E) -> Self {
        Self::with_seed(engine, MaskSeed::from_os_rng())
    }

    /// Construct with a deterministic seed (used by parity / regression tests).
    pub fn with_seed(engine: E, seed: MaskSeed) -> Self {
        Self {
            engine,
            rng: ChaCha20Rng::from_seed(seed.0),
            shield: ShieldConfig::NONE,
            verify_probes: 0,
            weights: HashMap::new(),
        }
    }

    /// Construct with both a deterministic seed and a shield configuration.
    pub fn with_shield(engine: E, seed: MaskSeed, shield: ShieldConfig) -> Self {
        Self {
            engine,
            rng: ChaCha20Rng::from_seed(seed.0),
            shield,
            verify_probes: 0,
            weights: HashMap::new(),
        }
    }

    /// Set or update the shield configuration in place.
    pub fn set_shield(&mut self, shield: ShieldConfig) {
        self.shield = shield;
    }

    /// Enable U-Verify with `k` Freivalds-style probes per offload.
    /// `k = 0` disables the check. Soundness scales as `(2L)^-k`; `k = 8`
    /// with the default `L = 3` is `≈ 2.4·10⁻⁷` undetected-tamper rate.
    pub fn with_verify_probes(mut self, n: usize) -> Self {
        self.verify_probes = n;
        self
    }

    pub fn set_verify_probes(&mut self, n: usize) {
        self.verify_probes = n;
    }

    pub fn engine(&self) -> &E {
        &self.engine
    }

    pub fn shield_config(&self) -> ShieldConfig {
        self.shield
    }

    pub fn verify_probes(&self) -> usize {
        self.verify_probes
    }

    /// Stack shield rows (if enabled) and sample a fresh mask sized to the
    /// stacked matrix. Returns `(mask, H', n_data_rows)`.
    fn build_shielded(&mut self, hidden: ArrayView2<'_, f32>) -> (GeloMask, Array2<f32>, usize) {
        let (stacked, n_data) = profile::time("gelo:shield_stack", || {
            stack_shield(hidden, self.shield, &mut self.rng)
        });
        let mask = profile::time("gelo:mask_sample", || {
            GeloMask::fresh(stacked.nrows(), &mut self.rng)
        });
        (mask, stacked, n_data)
    }
}

impl<E: GpuOffloadEngine> TrustedExecutor for InProcessTrustedExecutor<E> {
    fn provision_weight(&mut self, handle: WeightHandle, weight: ArrayView2<f32>) -> Result<()> {
        // Keep a TEE-local copy of the weight so the integrity probe can
        // compute `B · r` independently of what the engine reports.
        if self.verify_probes > 0 {
            self.weights.insert(handle, weight.to_owned());
        }
        self.engine.register_weight(handle, weight)
    }

    fn offload_linear(
        &mut self,
        handle: WeightHandle,
        hidden: ArrayView2<f32>,
    ) -> Result<Array2<f32>> {
        let (mask, stacked, n_data) = self.build_shielded(hidden);
        let masked = profile::time("gelo:mask_apply", || mask.apply(stacked.view()));
        let masked_out =
            profile::time("engine:matmul", || self.engine.matmul(handle, masked.view()))?;
        if self.verify_probes > 0 {
            let weight = self.weights.get(&handle).ok_or_else(|| {
                anyhow!("verify_probes>0 but weight {handle:?} not cached in TEE")
            })?;
            profile::time("uverify:linear", || {
                verify_offload(
                    self.verify_probes,
                    masked.view(),
                    weight.view(),
                    masked_out.view(),
                    &mut self.rng,
                )
            })?;
        }
        let unmasked = profile::time("gelo:mask_unapply", || mask.unapply(masked_out.view()));
        let strip = profile::time("gelo:strip_shield", || {
            unmasked.slice(ndarray::s![..n_data, ..]).to_owned()
        });
        Ok(strip)
    }

    fn offload_qkv(
        &mut self,
        layer: u16,
        hidden: ArrayView2<f32>,
    ) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>)> {
        let (mask, stacked, n_data) = self.build_shielded(hidden);
        let masked = profile::time("gelo:mask_apply", || mask.apply(stacked.view()));

        let mq = profile::time("engine:matmul", || {
            self.engine.matmul(WeightHandle::new(layer, WeightKind::Q), masked.view())
        })?;
        let mk = profile::time("engine:matmul", || {
            self.engine.matmul(WeightHandle::new(layer, WeightKind::K), masked.view())
        })?;
        let mv = profile::time("engine:matmul", || {
            self.engine.matmul(WeightHandle::new(layer, WeightKind::V), masked.view())
        })?;

        if self.verify_probes > 0 {
            for (kind, observed) in
                [(WeightKind::Q, &mq), (WeightKind::K, &mk), (WeightKind::V, &mv)]
            {
                let h = WeightHandle::new(layer, kind);
                let w = self.weights.get(&h).ok_or_else(|| {
                    anyhow!("verify_probes>0 but weight {h:?} not cached in TEE")
                })?;
                profile::time("uverify:linear", || {
                    verify_offload(
                        self.verify_probes,
                        masked.view(),
                        w.view(),
                        observed.view(),
                        &mut self.rng,
                    )
                })?;
            }
        }

        let q = profile::time("gelo:mask_unapply", || mask.unapply(mq.view()));
        let k = profile::time("gelo:mask_unapply", || mask.unapply(mk.view()));
        let v = profile::time("gelo:mask_unapply", || mask.unapply(mv.view()));

        let slice_n = ndarray::s![..n_data, ..];
        let triple = profile::time("gelo:strip_shield", || {
            (
                q.slice(slice_n).to_owned(),
                k.slice(slice_n).to_owned(),
                v.slice(slice_n).to_owned(),
            )
        });
        Ok(triple)
    }

    fn offload_attention_qkt(
        &mut self,
        q: ArrayView2<f32>,
        kt: ArrayView2<f32>,
    ) -> Result<Array2<f32>> {
        out_attn_mult::offload_qkt(&self.engine, &mut self.rng, q, kt, self.verify_probes)
    }
}

/// Trusted executor that skips the mask entirely. Used as the parity baseline
/// in tests: any [`TrustedExecutor`] that returns the same `H·W` as
/// [`PlaintextExecutor`] is correct on the protocol's math, regardless of
/// what the offload engine sees.
pub struct PlaintextExecutor<E: GpuOffloadEngine> {
    engine: E,
}

impl<E: GpuOffloadEngine> PlaintextExecutor<E> {
    pub fn new(engine: E) -> Self {
        Self { engine }
    }

    pub fn engine(&self) -> &E {
        &self.engine
    }
}

impl<E: GpuOffloadEngine> TrustedExecutor for PlaintextExecutor<E> {
    fn provision_weight(&mut self, handle: WeightHandle, weight: ArrayView2<f32>) -> Result<()> {
        self.engine.register_weight(handle, weight)
    }

    fn offload_linear(
        &mut self,
        handle: WeightHandle,
        hidden: ArrayView2<f32>,
    ) -> Result<Array2<f32>> {
        profile::time("engine:matmul", || self.engine.matmul(handle, hidden))
    }

    fn offload_attention_qkt(
        &mut self,
        q: ArrayView2<f32>,
        kt: ArrayView2<f32>,
    ) -> Result<Array2<f32>> {
        // Parity baseline: no mask, just compute Q · K^T directly via the engine.
        profile::time("engine:matmul_dynamic", || self.engine.matmul_dynamic(q, kt))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use rand_distr::{Distribution, StandardNormal};

    #[test]
    fn masked_and_plaintext_executors_agree() {
        let mut rng = ChaCha20Rng::from_seed([3u8; 32]);
        let normal = StandardNormal;
        let n = 8;
        let d = 6;
        let p = 4;

        let hidden = Array2::<f32>::from_shape_fn((n, d), |_| normal.sample(&mut rng));
        let weight = Array2::<f32>::from_shape_fn((d, p), |_| normal.sample(&mut rng));
        let handle = WeightHandle::new(0, WeightKind::Q);

        let mut plain = PlaintextExecutor::new(RayonCpuEngine::new());
        plain.provision_weight(handle, weight.view()).unwrap();
        let plain_out = plain.offload_linear(handle, hidden.view()).unwrap();

        let mut masked = InProcessTrustedExecutor::with_seed(
            RayonCpuEngine::new(),
            MaskSeed::from_bytes([9u8; 32]),
        );
        masked.provision_weight(handle, weight.view()).unwrap();
        let masked_out = masked.offload_linear(handle, hidden.view()).unwrap();

        for ((i, j), p) in plain_out.indexed_iter() {
            assert!(
                (*p - masked_out[[i, j]]).abs() < 1e-3,
                "({i},{j}): plain={p} masked={}",
                masked_out[[i, j]]
            );
        }
    }

    #[test]
    fn qkv_shares_one_mask() {
        let mut rng = ChaCha20Rng::from_seed([4u8; 32]);
        let normal = StandardNormal;
        let n = 4;
        let d = 3;

        let hidden = Array2::<f32>::from_shape_fn((n, d), |_| normal.sample(&mut rng));
        let wq = Array2::<f32>::from_shape_fn((d, d), |_| normal.sample(&mut rng));
        let wk = Array2::<f32>::from_shape_fn((d, d), |_| normal.sample(&mut rng));
        let wv = Array2::<f32>::from_shape_fn((d, d), |_| normal.sample(&mut rng));

        let mut exec = InProcessTrustedExecutor::with_seed(
            RayonCpuEngine::new(),
            MaskSeed::from_bytes([5u8; 32]),
        );
        exec.provision_weight(WeightHandle::new(0, WeightKind::Q), wq.view())
            .unwrap();
        exec.provision_weight(WeightHandle::new(0, WeightKind::K), wk.view())
            .unwrap();
        exec.provision_weight(WeightHandle::new(0, WeightKind::V), wv.view())
            .unwrap();

        let (q, k, v) = exec.offload_qkv(0, hidden.view()).unwrap();

        let expected_q = hidden.dot(&wq);
        let expected_k = hidden.dot(&wk);
        let expected_v = hidden.dot(&wv);
        for ((i, j), e) in expected_q.indexed_iter() {
            assert!((q[[i, j]] - e).abs() < 1e-3);
        }
        for ((i, j), e) in expected_k.indexed_iter() {
            assert!((k[[i, j]] - e).abs() < 1e-3);
        }
        for ((i, j), e) in expected_v.indexed_iter() {
            assert!((v[[i, j]] - e).abs() < 1e-3);
        }
    }
}

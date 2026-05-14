use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::attention::{self, PermAttnConfig};
use crate::integrity::verify_offload;
use crate::mask::GeloMask;
use crate::out_attn_mult;
use crate::profile;
use crate::rng::MaskSeed;
use crate::shield::{ShieldConfig, stack_shield};
use crate::substrate::{GpuOffloadEngine, TrustedExecutor, WeightHandle, WeightKind};

/// Reference [`GpuOffloadEngine`] that performs the offloaded GEMM on the CPU.
/// Stand-in for a real Vulkan / CUDA backend.
#[derive(Default, Clone)]
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
    /// TEE-side weight cache for U-Verify probe computation. Held as
    /// `Arc<Array2<f32>>` so callers that already own the weight bytes
    /// (the embedder loads them into `Arc<DecoderWeights>` at startup) can
    /// share via `provision_weight_shared` instead of paying for a second
    /// 2.4 GB copy on Qwen3-class models. The `provision_weight` path still
    /// clones via `weight.to_owned()` for callers that don't have an Arc.
    weights: HashMap<WeightHandle, Arc<Array2<f32>>>,
    /// Whether to use the GELO paper's per-forward-pass mask (one A
    /// sampled at `begin_forward_pass`, reused across every offload
    /// until `end_forward_pass`) vs. the per-offload mode (fresh A
    /// inside every offload — strictly safer but ~48-140× more QR
    /// work per text). Toggled via [`Self::with_per_forward_mask`].
    per_forward_mask: bool,
    /// Active session mask. `Some` between `begin_forward_pass` and
    /// `end_forward_pass` when `per_forward_mask` is enabled. The mask
    /// is sized to `n + shield.k` because shield rows are part of the
    /// stacked operand the mask acts on.
    session: Option<SessionMask>,
    /// Per-session reusable (stacked_n × d) shield-scratch buffers, keyed
    /// by input width `d`. In paper-parity mode the data rows are
    /// invariant across all offloads in a forward (only the input width
    /// changes, e.g. hidden_size for QKV/O/gate/up and intermediate_size
    /// for FFN-down). Reusing the buffer skips ~140 × (n+k)·d memcpys
    /// per Qwen3 forward and the matching allocator churn; only the k
    /// shield rows are rewritten per offload.
    ///
    /// Cleared on `end_forward_pass`.
    stacked_scratch: HashMap<usize, Array2<f32>>,
    /// Configuration for the permutation-shielded attention protocol
    /// (Tier 1). Defaults to no noise (pure permutation equivariance).
    /// Set via [`Self::with_perm_attention`] /
    /// [`Self::set_perm_attention`] to opt into the Hidden No More
    /// σ-noise mitigation. Phase 2 keeps the inner ops TEE-side; Phase 3
    /// will swap softmax + matmuls onto the GPU.
    perm_attn: PermAttnConfig,
}

/// Clone the executor sharing the underlying engine (engines that opt
/// into the `clone_shared` Arc-pattern reuse their weight cache; engines
/// that don't will duplicate state per the `E: Clone` impl). The session
/// mask and scratch are NOT cloned — they only make sense inside a
/// `begin_forward_pass`/`end_forward_pass` bracket on the owning thread.
/// The RNG state IS cloned; callers that want independent streams across
/// clones should chain `.with_rng_stream(stream_id)`.
impl<E: GpuOffloadEngine + Clone> Clone for InProcessTrustedExecutor<E> {
    fn clone(&self) -> Self {
        Self {
            engine: self.engine.clone(),
            rng: self.rng.clone(),
            shield: self.shield,
            verify_probes: self.verify_probes,
            weights: self.weights.clone(),
            per_forward_mask: self.per_forward_mask,
            session: None,
            stacked_scratch: HashMap::new(),
            perm_attn: self.perm_attn,
        }
    }
}

/// Per-forward-pass mask + bookkeeping for the GELO paper's
/// "one A per batch" construction (§3.2). Constructed inside
/// `begin_forward_pass` and dropped on `end_forward_pass`.
struct SessionMask {
    /// Haar-uniform orthogonal mask of size `(stacked_n, stacked_n)`
    /// where `stacked_n = data_n + shield.k`.
    mask: GeloMask,
    /// Original data-row count (excluding shield rows).
    data_n: usize,
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
            per_forward_mask: false,
            session: None,
            stacked_scratch: HashMap::new(),
            perm_attn: PermAttnConfig::default(),
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
            per_forward_mask: false,
            session: None,
            stacked_scratch: HashMap::new(),
            perm_attn: PermAttnConfig::default(),
        }
    }

    /// Set or update the shield configuration in place.
    pub fn set_shield(&mut self, shield: ShieldConfig) {
        self.shield = shield;
    }

    /// Enable the GELO paper's "one A per forward pass" construction
    /// (§3.2). When set, every `offload_*` call reuses the session mask
    /// established at [`begin_forward_pass`] instead of sampling its
    /// own fresh A.
    ///
    /// **Privacy requirement:** the paper pairs mask reuse with
    /// shield vectors (§4.2) to defeat ICA / blind-source-separation
    /// attacks that exploit cross-offload correlation under shared A.
    /// This builder takes a `shield` argument and refuses to enable
    /// per-forward-pass mode without one. Pass [`ShieldConfig::new(8,
    /// 4.0)`] for the paper's recommended defaults.
    ///
    /// Trade-off: ~48–140× fewer Haar-QR samples per text vs the
    /// per-offload default, at the cost of relying on the shield for
    /// reuse-across-correlated-states security.
    pub fn with_per_forward_mask(mut self, shield: ShieldConfig) -> Self {
        assert!(
            shield.enabled(),
            "per-forward-pass mask requires shield to be enabled; \
             pass `ShieldConfig::new(k>0, energy_scale>0)`"
        );
        self.per_forward_mask = true;
        self.shield = shield;
        self
    }

    /// Whether this executor is in paper-parity per-forward-pass mode.
    pub fn is_per_forward_mask(&self) -> bool {
        self.per_forward_mask
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

    /// Configure the permutation-shielded attention protocol (Tier 1).
    /// Defaults to no noise; pass `PermAttnConfig::HIDDEN_NO_MORE` to
    /// enable the σ = 0.01 mitigation per arXiv 2505.18332.
    pub fn with_perm_attention(mut self, cfg: PermAttnConfig) -> Self {
        self.perm_attn = cfg;
        self
    }

    /// In-place setter for the perm-attention config.
    pub fn set_perm_attention(&mut self, cfg: PermAttnConfig) {
        self.perm_attn = cfg;
    }

    /// Read the current perm-attention config.
    pub fn perm_attention(&self) -> PermAttnConfig {
        self.perm_attn
    }

    /// Stack shield rows (if enabled), then apply the per-batch mask `A`
    /// to the stacked matrix in a single fused step. Returns `(mask,
    /// masked, n_data)` where `masked = A · [H; S]` and `n_data` is the
    /// original (pre-shield) row count.
    ///
    /// In per-forward-pass mode the mask is the session mask established
    /// at [`Self::begin_forward_pass`] (cheap to clone — internal
    /// `Array2` only). In per-offload mode a fresh Haar-uniform `A` is
    /// sampled every call.
    ///
    /// Shield rows are **always sampled fresh per offload**, even in
    /// per-forward-pass mode. The mask reuse is what saves the Haar QR
    /// cost; the shield is cheap (just k Gaussian rows) and per-offload
    /// freshness is strictly safer for the ICA / cross-offload
    /// correlation defence the paper relies on.
    ///
    /// In paper-parity mode the (stacked_n × d) scratch buffer is taken
    /// from `stacked_scratch` if one of matching width exists, else
    /// allocated once and cached for the rest of the forward. Data rows
    /// are copied in; the k shield rows are overwritten with fresh
    /// Gaussians on every call. The buffer is **not** returned to the
    /// caller — the mask is applied immediately so the scratch can stay
    /// in place across the whole forward pass without any per-offload
    /// clone. Saves ~140 mallocs + ~224 MB of memcpy per Qwen3 forward
    /// without weakening the protocol.
    fn build_shielded_and_apply(
        &mut self,
        hidden: ArrayView2<'_, f32>,
    ) -> (GeloMask, Array2<f32>, usize) {
        let n_data = hidden.nrows();
        let d = hidden.ncols();
        let k = self.shield.k;
        let stacked_n = n_data + k;

        // Resolve the mask first (cheap; just an Arc-clone of the session
        // mask in paper-parity mode, or a fresh Haar sample otherwise).
        let mask = if self.per_forward_mask {
            match &self.session {
                Some(s) if s.data_n == n_data && s.mask.n() == stacked_n => s.mask.clone(),
                Some(s) => {
                    panic!(
                        "per-forward-pass mask: offload n={n_data} (stacked {stacked_n}) \
                         doesn't match session n={} (stacked {}); did you forget to call \
                         begin_forward_pass for the new shape?",
                        s.data_n,
                        s.mask.n(),
                    );
                }
                None => panic!(
                    "per-forward-pass mode but no session mask — \
                     embedder must call begin_forward_pass(n) before any offload_*"
                ),
            }
        } else {
            profile::time("gelo:mask_sample", || {
                GeloMask::fresh(stacked_n, &mut self.rng)
            })
        };

        let masked = if self.per_forward_mask && self.shield.enabled() {
            // Scratch-reuse path: populate cached buffer in place, then
            // apply the mask. The buffer stays owned by `stacked_scratch`
            // — no per-offload clone or allocation.
            let scale = self.shield.energy_scale;
            // Compute mean_row_norm before we borrow the scratch buffer
            // (avoids overlapping borrow on hidden through the buffer slice).
            let mean_norm = mean_row_norm(hidden);
            let sigma = if d == 0 { 0.0 } else { scale * mean_norm / (d as f32).sqrt() };

            let buf = self
                .stacked_scratch
                .entry(d)
                .or_insert_with(|| Array2::<f32>::zeros((stacked_n, d)));
            debug_assert_eq!(buf.shape(), &[stacked_n, d]);
            profile::time("gelo:shield_stack", || {
                buf.slice_mut(ndarray::s![..n_data, ..]).assign(&hidden);
                fill_shield_rows_inline(
                    buf.slice_mut(ndarray::s![n_data.., ..]),
                    sigma,
                    &mut self.rng,
                );
            });
            profile::time("gelo:mask_apply", || mask.apply(buf.view()))
        } else {
            // Legacy path: allocate-each-time, used in per-offload mode
            // and whenever shield is disabled.
            let stacked = profile::time("gelo:shield_stack", || {
                let (stacked, _n) = stack_shield(hidden, self.shield, &mut self.rng);
                stacked
            });
            profile::time("gelo:mask_apply", || mask.apply(stacked.view()))
        };

        (mask, masked, n_data)
    }
}

/// Overwrite `shield_dest` (a (k × d) mutable view) with fresh Gaussian
/// shield rows scaled to per-component `sigma`. The caller computes
/// `sigma = energy_scale × mean_row_norm(hidden) / sqrt(d)` once and
/// passes it in; this lets the helper avoid re-borrowing `hidden`
/// against the scratch buffer slice in the paper-parity path.
fn fill_shield_rows_inline<R: rand::RngCore>(
    mut shield_dest: ndarray::ArrayViewMut2<'_, f32>,
    sigma: f32,
    rng: &mut R,
) {
    use rand_distr::{Distribution, StandardNormal};
    let normal = StandardNormal;
    for mut row in shield_dest.rows_mut() {
        for v in row.iter_mut() {
            let z: f32 = normal.sample(rng);
            *v = z * sigma;
        }
    }
}

/// Mean L2 norm of the rows of `m`. Mirrors `shield::mean_row_norm`,
/// kept module-local to skip the export round-trip.
fn mean_row_norm(m: ArrayView2<'_, f32>) -> f32 {
    let n = m.nrows();
    if n == 0 {
        return 0.0;
    }
    let mut acc = 0.0_f32;
    for row in m.rows() {
        acc += row.iter().map(|v| v * v).sum::<f32>().sqrt();
    }
    acc / (n as f32)
}

impl<E: GpuOffloadEngine> TrustedExecutor for InProcessTrustedExecutor<E> {
    fn begin_forward_pass(&mut self, n: usize) -> Result<()> {
        if !self.per_forward_mask {
            // Per-offload mode: nothing to do. Each offload_* will sample
            // its own fresh mask the legacy way.
            return Ok(());
        }
        let stacked_n = n + self.shield.k;
        let mask = profile::time("gelo:mask_sample", || {
            GeloMask::fresh(stacked_n, &mut self.rng)
        });
        self.session = Some(SessionMask { mask, data_n: n });
        // Stale scratches from a prior forward with a different `n` are
        // unusable now — clear to avoid silently feeding the wrong row
        // count into mask.apply().
        self.stacked_scratch.clear();
        Ok(())
    }

    fn end_forward_pass(&mut self) -> Result<()> {
        self.session = None;
        self.stacked_scratch.clear();
        Ok(())
    }

    fn set_rng_stream(&mut self, stream: u64) {
        self.rng.set_stream(stream);
    }

    fn provision_weight(&mut self, handle: WeightHandle, weight: ArrayView2<f32>) -> Result<()> {
        // Keep a TEE-local copy of the weight so the integrity probe can
        // compute `B · r` independently of what the engine reports.
        if self.verify_probes > 0 {
            self.weights.insert(handle, Arc::new(weight.to_owned()));
        }
        self.engine.register_weight(handle, weight)
    }

    fn provision_weight_shared(
        &mut self,
        handle: WeightHandle,
        weight: Arc<Array2<f32>>,
    ) -> Result<()> {
        // Same as `provision_weight` but avoids the 2.4 GB clone on
        // Qwen3-class models when the embedder already holds an Arc.
        self.engine.register_weight(handle, weight.view())?;
        if self.verify_probes > 0 {
            self.weights.insert(handle, weight);
        }
        Ok(())
    }

    fn offload_linear(
        &mut self,
        handle: WeightHandle,
        hidden: ArrayView2<f32>,
    ) -> Result<Array2<f32>> {
        let (mask, masked, n_data) = self.build_shielded_and_apply(hidden);
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
        let (mask, masked, n_data) = self.build_shielded_and_apply(hidden);

        // Batched offload: one upload of `masked`, one device sync across
        // all three matmuls (vs three of each via separate `matmul` calls).
        // For backends without a lazy-tensor path, `matmul_many` falls back
        // to looping over `matmul`, so correctness is preserved.
        let handles = [
            WeightHandle::new(layer, WeightKind::Q),
            WeightHandle::new(layer, WeightKind::K),
            WeightHandle::new(layer, WeightKind::V),
        ];
        let qkv_out = profile::time("engine:matmul_many", || {
            self.engine.matmul_many(&handles, masked.view())
        })?;
        anyhow::ensure!(
            qkv_out.len() == 3,
            "engine.matmul_many returned {} results; expected 3",
            qkv_out.len()
        );
        let mut it = qkv_out.into_iter();
        let mq = it.next().expect("len checked above");
        let mk = it.next().expect("len checked above");
        let mv = it.next().expect("len checked above");

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

        // Three separate `Aᵀ · M` GEMMs. We tried batching them into a
        // single stacked GEMM (mask::unapply_many) and it regressed: at
        // stacked_n ≈ 408 the combined (stacked_n × 2·intermediate)
        // working set thrashes L2 vs matrixmultiply's tile-tuned
        // separate (stacked_n × hidden_size) calls. FLOPs are equal;
        // cache behaviour is not.
        let q_full = profile::time("gelo:mask_unapply", || mask.unapply(mq.view()));
        let k_full = profile::time("gelo:mask_unapply", || mask.unapply(mk.view()));
        let v_full = profile::time("gelo:mask_unapply", || mask.unapply(mv.view()));

        let slice_n = ndarray::s![..n_data, ..];
        let triple = profile::time("gelo:strip_shield", || {
            (
                q_full.slice(slice_n).to_owned(),
                k_full.slice(slice_n).to_owned(),
                v_full.slice(slice_n).to_owned(),
            )
        });
        Ok(triple)
    }

    fn offload_linear_many(
        &mut self,
        handles: &[WeightHandle],
        hidden: ArrayView2<f32>,
    ) -> Result<Vec<Array2<f32>>> {
        if handles.is_empty() {
            return Ok(Vec::new());
        }
        let (mask, masked, n_data) = self.build_shielded_and_apply(hidden);

        let masked_outs = profile::time("engine:matmul_many", || {
            self.engine.matmul_many(handles, masked.view())
        })?;
        anyhow::ensure!(
            masked_outs.len() == handles.len(),
            "engine.matmul_many returned {} results; expected {}",
            masked_outs.len(),
            handles.len(),
        );

        if self.verify_probes > 0 {
            for (h, observed) in handles.iter().zip(masked_outs.iter()) {
                let w = self.weights.get(h).ok_or_else(|| {
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

        // Separate unapply per output (same reasoning as offload_qkv: at
        // our shapes a stacked Aᵀ · [V₁ | V₂ | …] GEMM thrashes L2 vs
        // matrixmultiply's tile-tuned per-call GEMMs).
        let unmasked: Vec<Array2<f32>> = masked_outs
            .iter()
            .map(|m| profile::time("gelo:mask_unapply", || mask.unapply(m.view())))
            .collect();

        let slice_n = ndarray::s![..n_data, ..];
        let stripped = profile::time("gelo:strip_shield", || {
            unmasked
                .iter()
                .map(|u| u.slice(slice_n).to_owned())
                .collect::<Vec<_>>()
        });
        Ok(stripped)
    }

    fn offload_attention_qkt(
        &mut self,
        q: ArrayView2<f32>,
        kt: ArrayView2<f32>,
    ) -> Result<Array2<f32>> {
        out_attn_mult::offload_qkt(&self.engine, &mut self.rng, q, kt, self.verify_probes)
    }

    fn offload_attention_qkt_batched(
        &mut self,
        q: ArrayView3<f32>,
        kt: ArrayView3<f32>,
    ) -> Result<Array3<f32>> {
        out_attn_mult::offload_qkt_batched(
            &self.engine,
            &mut self.rng,
            q,
            kt,
            self.verify_probes,
        )
    }

    fn offload_attention_permuted(
        &mut self,
        q: ArrayView3<f32>,
        k: ArrayView3<f32>,
        v: ArrayView3<f32>,
        scale: f32,
    ) -> Result<Array3<f32>> {
        // Phase 2: inner ops stay TEE-side. Phase 3 will swap the matmuls
        // and softmax into the engine via softmax_batched.
        profile::time("gelo:perm_attention", || {
            attention::permuted_attention(q, k, v, scale, self.perm_attn, &mut self.rng)
        })
    }
}

/// Trusted executor that skips the mask entirely. Used as the parity baseline
/// in tests: any [`TrustedExecutor`] that returns the same `H·W` as
/// [`PlaintextExecutor`] is correct on the protocol's math, regardless of
/// what the offload engine sees.
pub struct PlaintextExecutor<E: GpuOffloadEngine> {
    engine: E,
}

impl<E: GpuOffloadEngine + Clone> Clone for PlaintextExecutor<E> {
    fn clone(&self) -> Self {
        Self {
            engine: self.engine.clone(),
        }
    }
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

    fn offload_attention_qkt_batched(
        &mut self,
        q: ArrayView3<f32>,
        kt: ArrayView3<f32>,
    ) -> Result<Array3<f32>> {
        // Parity baseline: no mask, one fused batched dispatch.
        profile::time("engine:matmul_dynamic_batched", || {
            self.engine.matmul_dynamic_batched(q, kt)
        })
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

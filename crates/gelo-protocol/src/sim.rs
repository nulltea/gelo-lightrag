use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::attention::{self, PermAttnConfig};
use crate::integrity::verify_offload;
use crate::hd3::Hd3Mask;
use crate::mask::{GeloMask, MaskFamily, MaskKind};
use crate::out_attn_mult;
use crate::profile;
use crate::rng::MaskSeed;
use crate::shield::{ShieldConfig, stack_shield};
use crate::snapshot::{SnapshotCapture, SnapshotConfig};
use crate::substrate::{GpuOffloadEngine, TrustedExecutor, WeightHandle, WeightKind};

/// **DEPRECATED — DO NOT USE IN NEW CODE.**
///
/// Reference [`GpuOffloadEngine`] that performs the offloaded GEMM on the
/// CPU. Retained only so existing tests / synthetic parity harnesses keep
/// compiling while they migrate. **Every measurement, bench, example,
/// and production runtime must use `gelo_gpu_wgpu::WgpuVulkanEngine`** —
/// see `feedback_benches_use_gelo_gpu.md` and
/// `feedback_no_rayon_cpu_engine.md`.
///
/// Weights are stored as `Arc<Array2<f32>>` so the legacy path that does
/// reach this type still avoids the engine-side clone via
/// [`Self::register_weight_shared`]. New call sites must not be added.
#[deprecated(
    since = "0.1.0",
    note = "RayonCpuEngine is the reference CPU impl; use `gelo_gpu_wgpu::WgpuVulkanEngine` \
            for all measurements, benches, and production runtimes — see \
            `feedback_benches_use_gelo_gpu.md` + `feedback_no_rayon_cpu_engine.md`."
)]
#[derive(Default, Clone)]
pub struct RayonCpuEngine {
    weights: HashMap<WeightHandle, Arc<Array2<f32>>>,
}

#[allow(deprecated)]
impl RayonCpuEngine {
    pub fn new() -> Self {
        Self::default()
    }
}

#[allow(deprecated)]
impl GpuOffloadEngine for RayonCpuEngine {
    fn register_weight(&mut self, handle: WeightHandle, weight: ArrayView2<f32>) -> Result<()> {
        // Legacy path: caller doesn't own an Arc. Clone once and wrap.
        // New call sites should use `register_weight_shared` instead.
        self.weights.insert(handle, Arc::new(weight.to_owned()));
        Ok(())
    }

    fn register_weight_shared(
        &mut self,
        handle: WeightHandle,
        weight: Arc<Array2<f32>>,
    ) -> Result<()> {
        self.weights.insert(handle, weight);
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
        Ok(input.dot(w.as_ref()))
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
    /// Active shield for the current forward pass. Re-set by
    /// `begin_forward_pass(n)` from one of two configurations
    /// described below — see `shield_default` / `shield_small_n`.
    /// Per-offload mode (legacy / safety-test path) uses
    /// `shield_default` directly.
    shield: ShieldConfig,
    /// Paper-parity shield for "normal" forward shapes (k=8). Used
    /// when `n > shield_small_n_max` at `begin_forward_pass`.
    shield_default: ShieldConfig,
    /// Optional overlay shield for **small-n forward passes** —
    /// specifically m=1 decode steps where the default `k=8` gives
    /// `stacked_n = 9` and forces Auto into DCT-IV (pad 9 → 16 is
    /// 1.78× over the HD₃ threshold). Default initialisation sets
    /// this to `Some(ShieldConfig::new(15, 4.0))` so decode lands
    /// at `stacked_n = 16` exactly — HD₃ zero-pad, no waste — and
    /// the per-decode mask cost drops onto the radix-8 FWHT path.
    /// Going from k=8 to k=15 is **monotonically safer** (more
    /// shield rows = more confusion of the engine-observed
    /// subspace); paper specifies k=8 as a minimum, not a ceiling.
    /// See feedback_memory_efficiency_priority.md / threshold tuning
    /// commit 2026-05-21.
    shield_small_n: Option<ShieldConfig>,
    /// Max `n` (data rows) at which `shield_small_n` overrides
    /// `shield_default`. Default 1 (decode-only).
    shield_small_n_max: usize,
    verify_probes: usize,
    /// TEE-side weight cache for U-Verify probe computation. Held as
    /// `Arc<Array2<f32>>` so callers that already own the weight bytes
    /// (the embedder loads them into `Arc<DecoderWeights>` at startup) can
    /// share via `provision_weight_shared` instead of paying for a second
    /// 2.4 GB copy on Qwen3-class models. The `provision_weight` path still
    /// clones via `weight.to_owned()` for callers that don't have an Arc.
    weights: HashMap<WeightHandle, Arc<Array2<f32>>>,
    /// Per-Layer Embedding table for Gemma 3n / Gemma 4 models. None
    /// for non-hybrid models; `Some(Arc<_>)` after
    /// `provision_ple_table` has been called. Clones (rayon workers)
    /// share the underlying storage — the table is hundreds of MB to
    /// >1 GB and must not be copied per worker.
    ple_table: Option<Arc<crate::ple::PleTable>>,
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
    /// Optional PCIe-side snapshot capture for AloePri attack-resistance
    /// evaluation. `None` by default — zero overhead, no allocations on
    /// the hot path. `Some(_)` after `with_snapshot_capture()` /
    /// `enable_snapshot_capture()`; every `offload_*` call clones its
    /// post-mask operand (and engine output, configurable) into the
    /// buffer for later drain by the test harness.
    snapshot_capture: Option<SnapshotCapture>,
    /// Which mask family to use. Default `MaskKind::Haar` is the GELO
    /// paper's dense Householder-QR sample — paper-parity. Switching
    /// to `MaskKind::Hd3` via [`Self::with_hd3_mask`] uses the
    /// structured QuIP#/QuaRot-style cascade described in
    /// [`crate::hd3`]. **Trade**: HD₃ kills the per-forward Haar QR
    /// (~3 s wall at n=2048 prefill on Qwen3-1.7B) and drops mask
    /// GEMM cost from `O(s²·d)` to `O(s·d·log s)`, but requires
    /// power-of-two `s`. At non-pow2 shapes the executor pads the
    /// stacked-with-shield operand to `s_pad = s.next_power_of_two()`,
    /// so the GPU sees `s_pad/s ≈ 2×` more rows per call.
    mask_kind: MaskKind,
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
            shield_default: self.shield_default,
            shield_small_n: self.shield_small_n,
            shield_small_n_max: self.shield_small_n_max,
            verify_probes: self.verify_probes,
            weights: self.weights.clone(),
            per_forward_mask: self.per_forward_mask,
            session: None,
            stacked_scratch: HashMap::new(),
            perm_attn: self.perm_attn,
            // Arc-share the PLE table across clones — no buffer copy.
            ple_table: self.ple_table.clone(),
            // Don't clone the snapshot buffer — captures are per-test
            // artifacts, not shared state. The clone's config is
            // re-derived from the source's config (capture stays on
            // for parallel rayon workers if the parent had it on).
            snapshot_capture: self
                .snapshot_capture
                .as_ref()
                .map(|c| SnapshotCapture::new(c.config())),
            mask_kind: self.mask_kind,
        }
    }
}

/// Per-forward-pass mask + bookkeeping for the GELO paper's
/// "one A per batch" construction (§3.2). Constructed inside
/// `begin_forward_pass` and dropped on `end_forward_pass`.
struct SessionMask {
    /// The mask. For `MaskFamily::Haar` `mask.n() == data_n + shield.k`;
    /// for `MaskFamily::Hd3` `mask.n() == (data_n + shield.k).next_power_of_two()`.
    mask: MaskFamily,
    /// Original data-row count (excluding shield rows and any
    /// HD₃-padding rows). The pipeline strips back to this at the end
    /// of every `offload_*` call.
    data_n: usize,
}

impl<E: GpuOffloadEngine> InProcessTrustedExecutor<E> {
    /// Construct with a fresh OS-seeded mask RNG. Paper-parity defaults
    /// apply: one Haar `A` per forward pass, shield `k=8` rows at
    /// energy scale 4× — the GELO paper §3.2 / §4.2 protocol. Callers
    /// who need a per-offload Haar for parity / BSS-recovery testing
    /// should chain [`Self::with_per_offload_mask`] (clears the
    /// paper-parity flag and the shield).
    pub fn new(engine: E) -> Self {
        Self::with_seed(engine, MaskSeed::from_os_rng())
    }

    /// Construct with a deterministic seed. Same paper-parity defaults
    /// as [`Self::new`]. The seed reproducibility is preserved
    /// across the per-forward / per-offload toggle.
    pub fn with_seed(engine: E, seed: MaskSeed) -> Self {
        // Pin BLIS to single-thread BEFORE any rayon worker spawns and
        // BEFORE any mask GEMM fires. `bli_thread_set_num_threads`
        // applied lazily inside `sgemm_blis` is too late on the ingest
        // path: rayon workers have already allocated against the
        // multi-thread BLIS pool by the time the first GEMM runs.
        // Idempotent — OnceLock guards subsequent calls.
        crate::mask::ensure_blis_single_thread();
        let shield_default = ShieldConfig::new(8, 4.0);
        Self {
            engine,
            rng: ChaCha20Rng::from_seed(seed.0),
            shield: shield_default,
            shield_default,
            // 2026-05-21: at m=1 decode the default k=8 gives
            // stacked_n = 9 and Auto falls to DCT-IV (pad 16/9 = 1.78×
            // > 1.4 threshold). Overlay k=15 makes stacked_n = 16
            // exactly — HD₃ zero-pad. Decode mask bucket drops from
            // ~64 s of wall to an estimated ~20–25 s on Qwen3-4B
            // chunks. Security-wise k=15 is strictly safer than k=8
            // (paper specifies k=8 as a minimum).
            shield_small_n: Some(ShieldConfig::new(15, 4.0)),
            shield_small_n_max: 1,
            verify_probes: 0,
            weights: HashMap::new(),
            per_forward_mask: true,
            session: None,
            stacked_scratch: HashMap::new(),
            perm_attn: PermAttnConfig::default(),
            ple_table: None,
            snapshot_capture: None,
            // 2026-05-21: default switched from Haar to Auto.
            // `MaskKind::Auto` picks HD₃ when the stacked-axis size
            // fits the pow2 pad budget, DCT-IV otherwise. Cuts the
            // CPU-mask bottleneck dramatically vs the O(s³) Haar QR
            // sampler, lifting GPU utilisation out of the 14%
            // range observed at Haar. See
            // `private_llm_inference_round_3.md` §2.1,
            // `hd3_mask_landed.md`, and the
            // `perf(gelo-protocol): MaskKind::Auto (HD₃ + DCT-IV)`
            // commit (recent).
            mask_kind: MaskKind::Auto,
        }
    }

    /// Construct with both a deterministic seed and a **custom shield**
    /// configuration in **per-offload** mode (legacy / safety-test
    /// path). Used by the BSS-recovery and DP-Forward tests that
    /// intentionally exercise the without-paper-parity construction so
    /// they can prove their respective claims (correlated-cross-offload
    /// attacks need per-offload masks; DP-Forward noise injection
    /// targets a specific layer position rather than the masked
    /// product). Production code should prefer [`Self::with_seed`].
    pub fn with_shield(engine: E, seed: MaskSeed, shield: ShieldConfig) -> Self {
        Self {
            engine,
            rng: ChaCha20Rng::from_seed(seed.0),
            shield,
            shield_default: shield,
            // Per-offload legacy/safety-test path: no shape-adaptive
            // override. Whatever shield the test caller picked is
            // exactly what runs.
            shield_small_n: None,
            shield_small_n_max: 0,
            verify_probes: 0,
            weights: HashMap::new(),
            per_forward_mask: false,
            session: None,
            stacked_scratch: HashMap::new(),
            perm_attn: PermAttnConfig::default(),
            ple_table: None,
            snapshot_capture: None,
            // `with_shield` is the per-offload legacy/safety-test
            // path used by BSS-recovery and DP-Forward tests; keep
            // Haar to preserve the explicit reference behaviour
            // those tests target. Production paths use `new` /
            // `with_seed` which default to `MaskKind::Auto`.
            mask_kind: MaskKind::Haar,
        }
    }

    /// Opt into the HD₃ Hadamard-cascade mask
    /// ([`crate::hd3::Hd3Mask`]) instead of the default Haar mask.
    /// Eliminates the per-forward `O(s³)` Haar QR sampler and drops
    /// the mask apply/unapply cost from `O(s²·d)` to `O(s·d·log s)`
    /// at the cost of zero-padding the operand to the next power of
    /// two before the engine matmul (~2× GPU rows at `n=2048`).
    ///
    /// **Security gate (open)**: the HD₃ orbit is a discrete subset
    /// of `O(s)` rather than the full Haar measure. Empirical parity
    /// with Haar against the paper §4.3 attack pipeline has not yet
    /// been validated at our shapes — see the round-3 doc step B.3.
    /// **Treat as research-grade** until the attack-suite gate
    /// passes; default stays Haar (paper-parity) until then.
    pub fn with_hd3_mask(mut self) -> Self {
        self.mask_kind = MaskKind::Hd3;
        // Clear any stale Haar session — mask kind change invalidates
        // the per-forward mask. The caller must re-bracket with
        // `begin_forward_pass` after switching kinds.
        self.session = None;
        self.stacked_scratch.clear();
        self
    }

    /// Switch to the DCT-IV cascade mask `A = D₃·C·D₂·C·D₁·C` (where
    /// `C` is the orthonormal DCT-IV). Works at **arbitrary `n`** — no
    /// power-of-two padding required — so the GPU sees the same row
    /// count as Haar with no pad regression. CPU mask cost is ~3× HD₃-
    /// at-pow2 but skips the 2× GPU GEMM penalty HD₃ pays when padding
    /// `n+k` up to the next pow2.
    ///
    /// **Security caveat:** DCT-IV cascade preserves orthogonality and
    /// QuIP#-style incoherence (entry bound `√(2/n)`), but the
    /// BSS-distinguishing-game security proof for DCT-IV is not in the
    /// published literature — only HD₃ has the QuIP#/QuaRot incoherence
    /// argument. **Treat as research-grade** until the attack-suite
    /// gate at `c5_dct4` passes; see
    /// `docs/research/hd3-non-pow2-fix.md` §6.2 for the design.
    pub fn with_dct4_mask(mut self) -> Self {
        self.mask_kind = MaskKind::Dct4;
        self.session = None;
        self.stacked_scratch.clear();
        self
    }

    /// Switch to the shape-adaptive mask dispatch: picks HD₃ when the
    /// pad penalty `s_pad / s` is small (≤ 4/3 ≈ 33 % pad), DCT-IV
    /// otherwise. Resolution happens at `begin_forward_pass` (per-
    /// forward-pass mode) or per-call (per-offload mode), so the
    /// physical mask family used at each call adapts to the shape
    /// without caller intervention.
    ///
    /// Use this as the default for production workloads with mixed
    /// prompt sizes — both HD₃ at pow2-aligned shapes and DCT-IV at
    /// non-pow2 shapes beat Haar; the crossover is at ~40 % pad and
    /// the 4/3 threshold sits safely on the HD₃ side.
    ///
    /// Inherits the security caveats of both [`Self::with_hd3_mask`]
    /// and [`Self::with_dct4_mask`] — neither has a published
    /// BSS-game proof at our shapes; both clear the empirical
    /// attack-suite gate (HD₃ at `c3_hd3`, DCT-IV at `c5_dct4`
    /// pending). Default stays Haar until both gates close.
    pub fn with_auto_mask(mut self) -> Self {
        self.mask_kind = MaskKind::Auto;
        self.session = None;
        self.stacked_scratch.clear();
        self
    }

    pub fn mask_kind(&self) -> MaskKind {
        self.mask_kind
    }

    /// Override the **shape-adaptive small-n shield** (the overlay
    /// that fires at decode shapes to land `stacked_n` on a
    /// power-of-two for HD₃). Pass `None` to disable — restores
    /// strict paper-parity (k=8 at every n). Pass `Some((max_n,
    /// shield))` to override the threshold and / or the override
    /// shield itself.
    ///
    /// Default (set by `new` / `with_seed`): `Some((1, k=15))` so
    /// m=1 decode lands at `stacked_n=16`. Tests that want
    /// strict paper-parity (k=8 everywhere) should call
    /// `.with_small_n_shield(None)`.
    pub fn with_small_n_shield(mut self, cfg: Option<(usize, ShieldConfig)>) -> Self {
        match cfg {
            Some((max_n, shield)) => {
                self.shield_small_n = Some(shield);
                self.shield_small_n_max = max_n;
            }
            None => {
                self.shield_small_n = None;
                self.shield_small_n_max = 0;
            }
        }
        self
    }

    /// Set or update the shield configuration in place. Updates both
    /// `shield` (the active value for the next non-overlay forward)
    /// and `shield_default` (so subsequent `begin_forward_pass(n)`
    /// calls fall back to this value when `n > shield_small_n_max`).
    pub fn set_shield(&mut self, shield: ShieldConfig) {
        self.shield = shield;
        self.shield_default = shield;
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

    /// Opt out of the paper-parity default and run in the per-offload
    /// mode (a fresh Haar `A` and fresh shield rows per offload, or no
    /// shield at all if not separately configured). Intended for
    /// parity / BSS-recovery / DP-Forward tests that need to exercise
    /// the alternative construction directly. Also clears the shield —
    /// most opt-out callers want raw per-offload Haar without shield
    /// rows; those that want per-offload with shield should use
    /// [`Self::with_shield`] instead.
    pub fn with_per_offload_mask(mut self) -> Self {
        self.per_forward_mask = false;
        self.shield = ShieldConfig::NONE;
        self.session = None;
        self.stacked_scratch.clear();
        self
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

    /// Opt into PCIe-side snapshot capture for the AloePri attack-harness
    /// pipeline. When enabled, every `offload_linear` / `offload_qkv` /
    /// `offload_linear_many` call records the post-mask operand (and
    /// engine output, configurable via `SnapshotConfig::capture_outputs`)
    /// into an internal buffer. Drain with [`Self::drain_pcie_snapshots`]
    /// or inspect with [`Self::pcie_snapshots`] / [`Self::pcie_snapshot_capture`].
    ///
    /// Cost: one `clone()` of the masked operand (shape
    /// `(n + shield.k, d)`) and optionally the masked output per offload.
    /// At Qwen3-1.7B prefill shape (n ≈ 16, d = 2048, k = 8, 28 layers ×
    /// 7 op_kinds) that's ~196 clones × ~24 × 8192 bytes ≈ 38 MB per
    /// forward — fine for tests and the attack harness, not free.
    ///
    /// Off by default; never engaged in the production embedder / reranker
    /// paths.
    pub fn with_snapshot_capture(mut self, cfg: SnapshotConfig) -> Self {
        self.snapshot_capture = Some(SnapshotCapture::new(cfg));
        self
    }

    /// In-place form of [`Self::with_snapshot_capture`] for tests that
    /// flip the flag after construction.
    pub fn enable_snapshot_capture(&mut self, cfg: SnapshotConfig) {
        self.snapshot_capture = Some(SnapshotCapture::new(cfg));
    }

    /// Disable snapshot capture and drop the buffer. The buffer's contents
    /// are dropped — call `drain_pcie_snapshots()` first to retain them.
    pub fn disable_snapshot_capture(&mut self) {
        self.snapshot_capture = None;
    }

    /// Borrow the captured snapshots (read-only) — `None` when capture is
    /// disabled, `Some(&[])` when enabled but no offload has fired yet.
    pub fn pcie_snapshots(&self) -> Option<&[crate::snapshot::PcieSnapshot]> {
        self.snapshot_capture.as_ref().map(|c| c.snapshots())
    }

    /// Borrow the full capture aggregator (read-only) — useful for tests
    /// that need access to the dropped-count or current config.
    pub fn pcie_snapshot_capture(&self) -> Option<&SnapshotCapture> {
        self.snapshot_capture.as_ref()
    }

    /// Drain captured snapshots and return ownership. The internal buffer
    /// is cleared; the seq-idx counter continues monotonically (see
    /// [`SnapshotCapture::drain`]).
    pub fn drain_pcie_snapshots(&mut self) -> Vec<crate::snapshot::PcieSnapshot> {
        self.snapshot_capture
            .as_mut()
            .map(|c| c.drain())
            .unwrap_or_default()
    }

    /// Internal helper: record one snapshot iff capture is enabled.
    /// Cloning `operand` and `output` is acceptable because capture is
    /// off in the production path; the cost is paid only in attack-
    /// resistance benchmark runs.
    fn record_snapshot(
        &mut self,
        handle: WeightHandle,
        operand: &Array2<f32>,
        output: Option<&Array2<f32>>,
    ) {
        if let Some(cap) = self.snapshot_capture.as_mut() {
            cap.record(handle, operand, output);
        }
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
    /// In paper-parity mode the (stacked_n × d) scratch buffer comes from
    /// `stacked_scratch` keyed by width `d`. The Haar path borrows the
    /// scratch in place (mask is allocated separately by `mask.apply`).
    /// The HD₃ path **removes** the scratch from the HashMap, mutates
    /// it in place via [`Hd3Mask::apply_in_place`], and hands the now-
    /// masked buffer back to the caller; the caller MUST call
    /// [`Self::return_apply_scratch`] before returning so the buffer
    /// re-enters the cache (otherwise we'd re-allocate 32 MB on the
    /// next call at long-context shapes). Saves ~140 mallocs + ~224 MB
    /// of memcpy per Qwen3 forward without weakening the protocol.
    fn build_shielded_and_apply(
        &mut self,
        hidden: ArrayView2<'_, f32>,
    ) -> (MaskFamily, Array2<f32>, usize) {
        let n_data = hidden.nrows();
        let d = hidden.ncols();
        let k = self.shield.k;
        // The mask must be sized to `stacked_n`. For Haar that's `n_data + k`;
        // for HD₃ it must be a power of two, so we round up. The extra rows
        // (between `n_data + k` and `stacked_n`) get zero-padded — they sit
        // in the HD₃ mask's output and round-trip exactly (orthogonality).
        // Resolve `Auto` to a concrete kind based on `s = n+k`'s pad
        // ratio; non-Auto kinds pass through unchanged.
        let resolved_kind = crate::mask::resolve_mask_kind_for_shape(self.mask_kind, n_data + k);
        let stacked_n = match resolved_kind {
            MaskKind::Haar => n_data + k,
            MaskKind::Hd3 => (n_data + k).next_power_of_two().max(2),
            // DCT-IV works at arbitrary `n` so no pow2 pad — operand
            // shape stays `(n_data + k, d)`, GPU sees same row count
            // as Haar (no pad regression at non-pow2 prompts).
            MaskKind::Dct4 => n_data + k,
            // Auto was already resolved by the call above; the match
            // is exhaustive on the resolved kind.
            MaskKind::Auto => unreachable!("Auto was resolved above"),
        };

        // Resolve the mask first (cheap; clone of session mask in
        // paper-parity mode, fresh sample otherwise).
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
            profile::time("gelo:mask_sample", || match resolved_kind {
                MaskKind::Haar => MaskFamily::Haar(GeloMask::fresh(stacked_n, &mut self.rng)),
                MaskKind::Hd3 => MaskFamily::Hd3(Hd3Mask::fresh(stacked_n, &mut self.rng)),
                MaskKind::Dct4 => {
                    MaskFamily::Dct4(crate::dct4::Dct4Mask::fresh(stacked_n, &mut self.rng))
                }
                MaskKind::Auto => unreachable!("Auto was resolved above"),
            })
        };

        let masked = if self.per_forward_mask && self.shield.enabled() {
            // Scratch-reuse path: populate cached buffer in place, then
            // apply the mask. For HD₃ the buffer is `(s_pad, d)` and the
            // rows past `n_data + k` stay zero (zero-padding the operand
            // to a power of two).
            let scale = self.shield.energy_scale;
            let mean_norm = mean_row_norm(hidden);
            let sigma = if d == 0 { 0.0 } else { scale * mean_norm / (d as f32).sqrt() };

            match &mask {
                MaskFamily::Hd3(hd3) => {
                    // Take the scratch out so we can both populate it in
                    // place AND hand it back as the masked output (apply
                    // is in-place for HD₃). The caller re-inserts via
                    // `return_apply_scratch` after the engine round-trip.
                    let mut buf = self
                        .stacked_scratch
                        .remove(&d)
                        .filter(|b| b.shape() == [stacked_n, d])
                        .unwrap_or_else(|| Array2::<f32>::zeros((stacked_n, d)));
                    let shield_end = (n_data + k).min(stacked_n);
                    profile::time("gelo:shield_stack", || {
                        buf.slice_mut(ndarray::s![..n_data, ..]).assign(&hidden);
                        fill_shield_rows_inline(
                            buf.slice_mut(ndarray::s![n_data..shield_end, ..]),
                            sigma,
                            &mut self.rng,
                        );
                        if stacked_n > shield_end {
                            buf.slice_mut(ndarray::s![shield_end.., ..]).fill(0.0);
                        }
                    });
                    profile::time(
                        "gelo:mask_apply:hd3",
                        || hd3.apply_in_place(&mut buf),
                    );
                    buf
                }
                MaskFamily::Dct4(dct4) => {
                    // Same scratch-reuse pattern as HD₃; DCT-IV is also
                    // in-place capable. stacked_n == n_data + k for DCT-IV
                    // (no pow2 padding), so shield_end == stacked_n and
                    // no zero-pad section to clear.
                    let mut buf = self
                        .stacked_scratch
                        .remove(&d)
                        .filter(|b| b.shape() == [stacked_n, d])
                        .unwrap_or_else(|| Array2::<f32>::zeros((stacked_n, d)));
                    profile::time("gelo:shield_stack", || {
                        buf.slice_mut(ndarray::s![..n_data, ..]).assign(&hidden);
                        fill_shield_rows_inline(
                            buf.slice_mut(ndarray::s![n_data..stacked_n, ..]),
                            sigma,
                            &mut self.rng,
                        );
                    });
                    profile::time(
                        "gelo:mask_apply:dct4",
                        || dct4.apply_in_place(&mut buf),
                    );
                    buf
                }
                MaskFamily::Haar(_) => {
                    let buf = self
                        .stacked_scratch
                        .entry(d)
                        .or_insert_with(|| Array2::<f32>::zeros((stacked_n, d)));
                    if buf.shape() != [stacked_n, d] {
                        *buf = Array2::<f32>::zeros((stacked_n, d));
                    }
                    let shield_end = (n_data + k).min(stacked_n);
                    profile::time("gelo:shield_stack", || {
                        buf.slice_mut(ndarray::s![..n_data, ..]).assign(&hidden);
                        fill_shield_rows_inline(
                            buf.slice_mut(ndarray::s![n_data..shield_end, ..]),
                            sigma,
                            &mut self.rng,
                        );
                        if stacked_n > shield_end {
                            buf.slice_mut(ndarray::s![shield_end.., ..]).fill(0.0);
                        }
                    });
                    profile::time(
                        "gelo:mask_apply:haar",
                        || mask.apply(buf.view()),
                    )
                }
            }
        } else {
            // Legacy path: allocate-each-time, used in per-offload mode
            // and whenever shield is disabled. For HD₃ this path zero-
            // pads the stacked operand to `stacked_n` (a power of two).
            let mut stacked = profile::time("gelo:shield_stack", || {
                let (mut stacked, _n) = stack_shield(hidden, self.shield, &mut self.rng);
                // `stack_shield` returns shape `(n_data + k, d)`. For
                // HD₃ pad to `stacked_n` with zeros.
                if stacked.nrows() < stacked_n {
                    let mut padded = Array2::<f32>::zeros((stacked_n, d));
                    padded
                        .slice_mut(ndarray::s![..stacked.nrows(), ..])
                        .assign(&stacked);
                    stacked = padded;
                }
                stacked
            });
            // HD₃ and DCT-IV can both mask in place on the owned
            // `stacked` buffer (saves the fresh 32 MB allocation
            // `mask.apply` would have done). Haar still allocates a
            // separate (n+k)×d output.
            profile::time(mask.apply_profile_category(), || match &mask {
                MaskFamily::Hd3(hd3) => {
                    hd3.apply_in_place(&mut stacked);
                    stacked
                }
                MaskFamily::Dct4(dct4) => {
                    dct4.apply_in_place(&mut stacked);
                    stacked
                }
                MaskFamily::Haar(_) => mask.apply(stacked.view()),
            })
        };

        (mask, masked, n_data)
    }

    /// Re-insert a masked buffer into `stacked_scratch` after the offload
    /// completed. Called by every `offload_*` method before returning, to
    /// counterbalance the `remove(&d)` that `build_shielded_and_apply`
    /// performs on the HD₃ / DCT-IV paper-parity paths.
    ///
    /// No-op outside the paper-parity HD₃/DCT-IV regime — the Haar
    /// path keeps the borrow in place across `apply`, and per-offload
    /// mode owns its `stacked` buffer outright (dropped at end of call).
    fn return_apply_scratch(&mut self, buf: Array2<f32>) {
        // The scratch-reuse path fires for HD₃, DCT-IV, and Auto
        // (which always resolves to one of those two). Haar keeps the
        // scratch by borrow inside `build_shielded_and_apply` and per-
        // offload mode owns the buffer outright.
        if self.per_forward_mask
            && self.shield.enabled()
            && matches!(
                self.mask_kind,
                MaskKind::Hd3 | MaskKind::Dct4 | MaskKind::Auto
            )
        {
            let d = buf.ncols();
            self.stacked_scratch.insert(d, buf);
        }
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
        // Shape-adaptive shield: at small n (default: m=1 decode
        // steps), swap in the overlay shield so `stacked_n = n + k`
        // lands on a power-of-two for HD₃ zero-pad. For larger n
        // (prefill), use the paper-parity default. See the
        // `shield_small_n` field doc.
        self.shield = match self.shield_small_n {
            Some(small) if n <= self.shield_small_n_max => small,
            _ => self.shield_default,
        };
        let stacked_n = n + self.shield.k;
        let resolved_kind = crate::mask::resolve_mask_kind_for_shape(self.mask_kind, stacked_n);
        let mask = profile::time("gelo:mask_sample", || match resolved_kind {
            MaskKind::Haar => MaskFamily::Haar(GeloMask::fresh(stacked_n, &mut self.rng)),
            MaskKind::Hd3 => {
                // HD₃ requires power-of-two side length.
                let s_pad = stacked_n.next_power_of_two().max(2);
                MaskFamily::Hd3(Hd3Mask::fresh(s_pad, &mut self.rng))
            }
            MaskKind::Dct4 => {
                // DCT-IV works at any positive integer — no pad.
                MaskFamily::Dct4(crate::dct4::Dct4Mask::fresh(stacked_n, &mut self.rng))
            }
            MaskKind::Auto => unreachable!("Auto resolved above"),
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

    fn provision_weight_bf16(
        &mut self,
        handle: WeightHandle,
        weight: ArrayView2<half::bf16>,
    ) -> Result<()> {
        // U-Verify cache deliberately not populated for the bf16
        // path: the probe machinery is f32-only, and bf16 +
        // verify_probes is unsupported in v1. The engine still gets
        // the bf16 weight directly.
        self.engine.register_weight_bf16(handle, weight)
    }

    fn provision_weight_bf16_shared(
        &mut self,
        handle: WeightHandle,
        weight: Arc<Array2<half::bf16>>,
    ) -> Result<()> {
        // Hand the Arc straight to the engine. For wgpu in F16 mode
        // the upload converts bf16 → f16 device-side and the Arc
        // refcount drops once the engine returns. For the deprecated
        // RayonCpuEngine, the default impl converts bf16 → f32 via
        // mapv() inside `register_weight_bf16_shared` — never used in
        // production. See `feedback_no_rayon_cpu_engine.md`.
        self.engine.register_weight_bf16_shared(handle, weight)
    }

    fn provision_weight_shared(
        &mut self,
        handle: WeightHandle,
        weight: Arc<Array2<f32>>,
    ) -> Result<()> {
        // Same as `provision_weight` but avoids the 2.4 GB clone on
        // Qwen3-class models when the embedder already holds an Arc.
        // The engine-side clone is also eliminated via
        // `register_weight_shared` — for engines that override
        // (`RayonCpuEngine` does), the Arc is stored directly.
        self.engine.register_weight_shared(handle, Arc::clone(&weight))?;
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
        // PCIe-side snapshot: record the masked operand and engine output.
        // No-op when snapshot capture is disabled (the default).
        self.record_snapshot(handle, &masked, Some(&masked_out));
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
        let unmasked = profile::time(mask.unapply_profile_category(), || {
            mask.unapply_take(masked_out)
        });
        let strip = profile::time("gelo:strip_shield", || {
            unmasked.slice(ndarray::s![..n_data, ..]).to_owned()
        });
        self.return_apply_scratch(masked);
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

        // PCIe-side snapshot: same masked operand drives Q, K, V; record
        // one entry per kind so the harness can partition by op_kind.
        self.record_snapshot(WeightHandle::new(layer, WeightKind::Q), &masked, Some(&mq));
        self.record_snapshot(WeightHandle::new(layer, WeightKind::K), &masked, Some(&mk));
        self.record_snapshot(WeightHandle::new(layer, WeightKind::V), &masked, Some(&mv));

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
        let unapply_cat = mask.unapply_profile_category();
        let q_full = profile::time(unapply_cat, || mask.unapply_take(mq));
        let k_full = profile::time(unapply_cat, || mask.unapply_take(mk));
        let v_full = profile::time(unapply_cat, || mask.unapply_take(mv));

        let slice_n = ndarray::s![..n_data, ..];
        let triple = profile::time("gelo:strip_shield", || {
            (
                q_full.slice(slice_n).to_owned(),
                k_full.slice(slice_n).to_owned(),
                v_full.slice(slice_n).to_owned(),
            )
        });
        self.return_apply_scratch(masked);
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

        // PCIe-side snapshot: one entry per (handle, output) pair so the
        // attack harness can partition by op_kind across a SwiGLU
        // gate+up batch (or any other multi-output offload).
        for (h, out) in handles.iter().zip(masked_outs.iter()) {
            self.record_snapshot(*h, &masked, Some(out));
        }

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
        let unapply_cat = mask.unapply_profile_category();
        let unmasked: Vec<Array2<f32>> = masked_outs
            .into_iter()
            .map(|m| profile::time(unapply_cat, || mask.unapply_take(m)))
            .collect();

        let slice_n = ndarray::s![..n_data, ..];
        let stripped = profile::time("gelo:strip_shield", || {
            unmasked
                .iter()
                .map(|u| u.slice(slice_n).to_owned())
                .collect::<Vec<_>>()
        });
        self.return_apply_scratch(masked);
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
        mask: attention::AttentionMask,
    ) -> Result<Array3<f32>> {
        // Phase 3+4: matmuls + softmax delegated to the engine. The engine
        // sees only permuted (and optionally noise-perturbed) Q, K, V —
        // the secret state (π, σ) stays inside `self`. Causal mask, when
        // requested, is applied TEE-side to the score tensor between the
        // first and second engine call (shared across heads).
        profile::time("gelo:perm_attention", || {
            attention::permuted_attention(
                &self.engine,
                q,
                k,
                v,
                scale,
                mask,
                self.perm_attn,
                &mut self.rng,
            )
        })
    }

    fn offload_attention_permuted_cached(
        &mut self,
        q: ArrayView3<f32>,
        k: ArrayView3<f32>,
        v: ArrayView3<f32>,
        scale: f32,
        q_pos_offset: usize,
        mask: attention::AttentionMask,
    ) -> Result<Array3<f32>> {
        // Same protocol as the symmetric `offload_attention_permuted`
        // (Amulet equivariance + Hidden-No-More σ-noise + F1+ in-TEE
        // softmax) extended to `n_q ≤ n_kv` via two independent
        // permutations sampled from `self.rng`. Used by the cached
        // generation path (`decoder_block_cached`) on Global layers
        // past the auto-switch threshold.
        profile::time("gelo:perm_attention_cached", || {
            attention::permuted_attention_cached(
                &self.engine,
                q,
                k,
                v,
                scale,
                q_pos_offset,
                mask,
                self.perm_attn,
                &mut self.rng,
            )
        })
    }

    fn provision_ple_table(&mut self, table: crate::ple::PleTable) -> Result<()> {
        // The table is shared by `Arc` across rayon worker clones; we
        // store it inside the trusted executor's owned state and never
        // hand it to the offload engine — this is what closes the P0
        // round-2 PLE address-bus leak.
        self.ple_table = Some(Arc::new(table));
        Ok(())
    }

    fn ple_gather(
        &self,
        token_ids: &[u32],
        layer_idx: usize,
    ) -> Result<Array2<f32>> {
        let table = self
            .ple_table
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("ple_gather: no PLE table provisioned"))?;
        table.gather(token_ids, layer_idx)
    }
}

/// Trusted executor that skips the mask entirely. Used as the parity baseline
/// in tests: any [`TrustedExecutor`] that returns the same `H·W` as
/// [`PlaintextExecutor`] is correct on the protocol's math, regardless of
/// what the offload engine sees.
pub struct PlaintextExecutor<E: GpuOffloadEngine> {
    engine: E,
    /// Same TEE-resident PLE table as `InProcessTrustedExecutor`. Held
    /// here too so parity tests (PlaintextExecutor vs masked executor)
    /// can both run Gemma 4 / Gemma 3n hybrid models.
    ple_table: Option<Arc<crate::ple::PleTable>>,
}

impl<E: GpuOffloadEngine + Clone> Clone for PlaintextExecutor<E> {
    fn clone(&self) -> Self {
        Self {
            engine: self.engine.clone(),
            ple_table: self.ple_table.clone(),
        }
    }
}

impl<E: GpuOffloadEngine> PlaintextExecutor<E> {
    pub fn new(engine: E) -> Self {
        Self {
            engine,
            ple_table: None,
        }
    }

    pub fn engine(&self) -> &E {
        &self.engine
    }
}

impl<E: GpuOffloadEngine> TrustedExecutor for PlaintextExecutor<E> {
    fn provision_weight(&mut self, handle: WeightHandle, weight: ArrayView2<f32>) -> Result<()> {
        self.engine.register_weight(handle, weight)
    }

    fn provision_weight_bf16(
        &mut self,
        handle: WeightHandle,
        weight: ArrayView2<half::bf16>,
    ) -> Result<()> {
        self.engine.register_weight_bf16(handle, weight)
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

    fn provision_ple_table(&mut self, table: crate::ple::PleTable) -> Result<()> {
        // Same TEE-resident contract as InProcessTrustedExecutor —
        // never leaves the trusted side, never reaches the engine.
        self.ple_table = Some(Arc::new(table));
        Ok(())
    }

    fn ple_gather(
        &self,
        token_ids: &[u32],
        layer_idx: usize,
    ) -> Result<Array2<f32>> {
        let table = self
            .ple_table
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("ple_gather: no PLE table provisioned"))?;
        table.gather(token_ids, layer_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mask::resolve_mask_kind_for_shape;
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
        masked.begin_forward_pass(n).unwrap();
        let masked_out = masked.offload_linear(handle, hidden.view()).unwrap();
        masked.end_forward_pass().unwrap();

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

        exec.begin_forward_pass(n).unwrap();
        let (q, k, v) = exec.offload_qkv(0, hidden.view()).unwrap();
        exec.end_forward_pass().unwrap();

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

    /// `with_hd3_mask()` produces the same downstream output as
    /// `PlaintextExecutor` to f32 noise — i.e. HD₃ is a correct GELO
    /// mask alternative on the round-trip math. Uses a power-of-two
    /// `n` so the executor's pad-to-pow2 step is a no-op (well, k=8
    /// shield rows still force pow2 pad — see below).
    #[test]
    fn hd3_executor_agrees_with_plaintext() {
        let mut rng = ChaCha20Rng::from_seed([13u8; 32]);
        let normal = StandardNormal;
        let n = 16;
        let d = 12;
        let p = 8;

        let hidden = Array2::<f32>::from_shape_fn((n, d), |_| normal.sample(&mut rng));
        let weight = Array2::<f32>::from_shape_fn((d, p), |_| normal.sample(&mut rng));
        let handle = WeightHandle::new(0, WeightKind::Q);

        let mut plain = PlaintextExecutor::new(RayonCpuEngine::new());
        plain.provision_weight(handle, weight.view()).unwrap();
        let plain_out = plain.offload_linear(handle, hidden.view()).unwrap();

        let mut hd3 = InProcessTrustedExecutor::with_seed(
            RayonCpuEngine::new(),
            MaskSeed::from_bytes([19u8; 32]),
        )
        .with_hd3_mask();
        assert_eq!(hd3.mask_kind(), MaskKind::Hd3);
        hd3.provision_weight(handle, weight.view()).unwrap();
        hd3.begin_forward_pass(n).unwrap();
        let hd3_out = hd3.offload_linear(handle, hidden.view()).unwrap();
        hd3.end_forward_pass().unwrap();

        assert_eq!(hd3_out.dim(), plain_out.dim());
        for ((i, j), &e) in plain_out.indexed_iter() {
            let got = hd3_out[[i, j]];
            // HD₃ + shield + matmul + unapply accumulates noise from
            // 3·log₂(s_pad) FWHT stages plus the depth-d inner matmul.
            // Loose tolerance because the shield rows add their own
            // Gaussian energy on top of the pure round-trip noise.
            assert!(
                (got - e).abs() < 5e-3,
                "({i},{j}): plain={e} hd3={got} diff={}",
                (got - e).abs()
            );
        }
    }

    /// HD₃ executor in **per-offload** mode (shield disabled, fresh
    /// mask per call) — exercises the legacy/owned-`stacked` branch of
    /// `build_shielded_and_apply`, where `apply_in_place` runs on the
    /// caller-owned padded buffer (no scratch round-trip).
    #[test]
    fn hd3_per_offload_agrees_with_plaintext() {
        let mut rng = ChaCha20Rng::from_seed([29u8; 32]);
        let normal = StandardNormal;
        let n = 16;
        let d = 12;
        let p = 8;

        let hidden = Array2::<f32>::from_shape_fn((n, d), |_| normal.sample(&mut rng));
        let weight = Array2::<f32>::from_shape_fn((d, p), |_| normal.sample(&mut rng));
        let handle = WeightHandle::new(0, WeightKind::Q);

        let mut plain = PlaintextExecutor::new(RayonCpuEngine::new());
        plain.provision_weight(handle, weight.view()).unwrap();
        let plain_out = plain.offload_linear(handle, hidden.view()).unwrap();

        let mut hd3 = InProcessTrustedExecutor::with_seed(
            RayonCpuEngine::new(),
            MaskSeed::from_bytes([31u8; 32]),
        )
        .with_per_offload_mask()
        .with_hd3_mask();
        hd3.provision_weight(handle, weight.view()).unwrap();
        // No begin_forward_pass — per-offload mode samples its own mask.
        let hd3_out = hd3.offload_linear(handle, hidden.view()).unwrap();

        assert_eq!(hd3_out.dim(), plain_out.dim());
        for ((i, j), &e) in plain_out.indexed_iter() {
            let got = hd3_out[[i, j]];
            assert!(
                (got - e).abs() < 1e-3,
                "({i},{j}): plain={e} hd3={got} diff={}",
                (got - e).abs()
            );
        }
    }

    /// HD₃ executor at `offload_qkv`: all three projections produce
    /// the same downstream output as plaintext. Verifies the
    /// shared-mask path (3× unapply on one apply).
    #[test]
    fn hd3_qkv_agrees_with_plaintext() {
        let mut rng = ChaCha20Rng::from_seed([21u8; 32]);
        let normal = StandardNormal;
        let n = 16;
        let d = 12;

        let hidden = Array2::<f32>::from_shape_fn((n, d), |_| normal.sample(&mut rng));
        let wq = Array2::<f32>::from_shape_fn((d, d), |_| normal.sample(&mut rng));
        let wk = Array2::<f32>::from_shape_fn((d, d), |_| normal.sample(&mut rng));
        let wv = Array2::<f32>::from_shape_fn((d, d), |_| normal.sample(&mut rng));

        let mut exec = InProcessTrustedExecutor::with_seed(
            RayonCpuEngine::new(),
            MaskSeed::from_bytes([27u8; 32]),
        )
        .with_hd3_mask();
        for (kind, w) in [
            (WeightKind::Q, &wq),
            (WeightKind::K, &wk),
            (WeightKind::V, &wv),
        ] {
            exec.provision_weight(WeightHandle::new(0, kind), w.view())
                .unwrap();
        }

        exec.begin_forward_pass(n).unwrap();
        let (q, k, v) = exec.offload_qkv(0, hidden.view()).unwrap();
        exec.end_forward_pass().unwrap();

        let expected_q = hidden.dot(&wq);
        let expected_k = hidden.dot(&wk);
        let expected_v = hidden.dot(&wv);
        for ((i, j), e) in expected_q.indexed_iter() {
            assert!((q[[i, j]] - e).abs() < 5e-3);
        }
        for ((i, j), e) in expected_k.indexed_iter() {
            assert!((k[[i, j]] - e).abs() < 5e-3);
        }
        for ((i, j), e) in expected_v.indexed_iter() {
            assert!((v[[i, j]] - e).abs() < 5e-3);
        }
    }

    /// `with_dct4_mask()` produces the same downstream output as
    /// `PlaintextExecutor` to f32 noise. Uses a **non-pow2** `n` to
    /// exercise the path DCT-IV was added for (HD₃ at n=12 would pad
    /// to s_pad=16; DCT-IV operates at n=20 exactly without padding).
    #[test]
    fn dct4_executor_agrees_with_plaintext() {
        let mut rng = ChaCha20Rng::from_seed([41u8; 32]);
        let normal = StandardNormal;
        let n = 20; // non-pow2 — DCT-IV's reason for existing
        let d = 12;
        let p = 8;

        let hidden = Array2::<f32>::from_shape_fn((n, d), |_| normal.sample(&mut rng));
        let weight = Array2::<f32>::from_shape_fn((d, p), |_| normal.sample(&mut rng));
        let handle = WeightHandle::new(0, WeightKind::Q);

        let mut plain = PlaintextExecutor::new(RayonCpuEngine::new());
        plain.provision_weight(handle, weight.view()).unwrap();
        let plain_out = plain.offload_linear(handle, hidden.view()).unwrap();

        let mut dct = InProcessTrustedExecutor::with_seed(
            RayonCpuEngine::new(),
            MaskSeed::from_bytes([43u8; 32]),
        )
        .with_dct4_mask();
        assert_eq!(dct.mask_kind(), MaskKind::Dct4);
        dct.provision_weight(handle, weight.view()).unwrap();
        dct.begin_forward_pass(n).unwrap();
        let dct_out = dct.offload_linear(handle, hidden.view()).unwrap();
        dct.end_forward_pass().unwrap();

        assert_eq!(dct_out.dim(), plain_out.dim());
        for ((i, j), &e) in plain_out.indexed_iter() {
            let got = dct_out[[i, j]];
            // DCT-IV cascade + shield + matmul + unapply accumulates
            // noise comparable to HD₃: same cascade depth, similar
            // condition number. Same tolerance as the HD₃ parity test.
            assert!(
                (got - e).abs() < 5e-3,
                "({i},{j}): plain={e} dct4={got} diff={}",
                (got - e).abs()
            );
        }
    }

    /// DCT-IV executor in per-offload mode (shield disabled, fresh
    /// mask per call) — exercises the legacy/owned-`stacked` branch.
    #[test]
    fn dct4_per_offload_agrees_with_plaintext() {
        let mut rng = ChaCha20Rng::from_seed([47u8; 32]);
        let normal = StandardNormal;
        let n = 20;
        let d = 12;
        let p = 8;

        let hidden = Array2::<f32>::from_shape_fn((n, d), |_| normal.sample(&mut rng));
        let weight = Array2::<f32>::from_shape_fn((d, p), |_| normal.sample(&mut rng));
        let handle = WeightHandle::new(0, WeightKind::Q);

        let mut plain = PlaintextExecutor::new(RayonCpuEngine::new());
        plain.provision_weight(handle, weight.view()).unwrap();
        let plain_out = plain.offload_linear(handle, hidden.view()).unwrap();

        let mut dct = InProcessTrustedExecutor::with_seed(
            RayonCpuEngine::new(),
            MaskSeed::from_bytes([53u8; 32]),
        )
        .with_per_offload_mask()
        .with_dct4_mask();
        dct.provision_weight(handle, weight.view()).unwrap();
        let dct_out = dct.offload_linear(handle, hidden.view()).unwrap();

        assert_eq!(dct_out.dim(), plain_out.dim());
        for ((i, j), &e) in plain_out.indexed_iter() {
            let got = dct_out[[i, j]];
            assert!(
                (got - e).abs() < 1e-3,
                "({i},{j}): plain={e} dct4={got} diff={}",
                (got - e).abs()
            );
        }
    }

    /// `MaskKind::Auto` resolves to HD₃ at pow2-aligned and near-pow2
    /// shapes and to DCT-IV at "far-from-pow2" shapes. Verifies the
    /// pad-ratio dispatch boundary at the 4/3 ≈ 1.333 threshold.
    #[test]
    fn auto_dispatch_resolves_by_pad_ratio() {
        // Pow2 exact: s_pad/s = 1.0 → HD₃.
        assert_eq!(resolve_mask_kind_for_shape(MaskKind::Auto, 2048), MaskKind::Hd3);
        // Near pow2 (1 row of pad): s_pad/s = 2048/2047 ≈ 1.0005 → HD₃.
        assert_eq!(resolve_mask_kind_for_shape(MaskKind::Auto, 2047), MaskKind::Hd3);
        // s=2055 → s_pad=2048? no, 2055 > 2048 → s_pad=4096, ratio 1.99 → DCT-IV.
        assert_eq!(resolve_mask_kind_for_shape(MaskKind::Auto, 2055), MaskKind::Dct4);
        // s=3072 → s_pad=4096, ratio 4096/3072 ≈ 1.333 (exact threshold).
        // s_pad * 3 = 12288, s * 4 = 12288 → 12288 ≤ 12288 → HD₃ (inclusive).
        assert_eq!(resolve_mask_kind_for_shape(MaskKind::Auto, 3072), MaskKind::Hd3);
        // s=3071 → s_pad=4096, ratio 4096/3071 > 4/3 → DCT-IV.
        // Check: s_pad * 3 = 12288, s * 4 = 12284 → 12288 > 12284 → DCT-IV.
        assert_eq!(resolve_mask_kind_for_shape(MaskKind::Auto, 3071), MaskKind::Dct4);
        // Non-Auto kinds pass through.
        assert_eq!(resolve_mask_kind_for_shape(MaskKind::Haar, 2056), MaskKind::Haar);
        assert_eq!(resolve_mask_kind_for_shape(MaskKind::Hd3, 2056), MaskKind::Hd3);
        assert_eq!(resolve_mask_kind_for_shape(MaskKind::Dct4, 2056), MaskKind::Dct4);
    }

    /// `with_auto_mask()` executor agrees with plaintext at a non-pow2
    /// shape (resolves to DCT-IV path internally).
    #[test]
    fn auto_dct4_path_agrees_with_plaintext() {
        let mut rng = ChaCha20Rng::from_seed([67u8; 32]);
        let normal = StandardNormal;
        // n=20 + k=8 = 28; s_pad = 32, ratio 1.6 > 4/3 → DCT-IV.
        let n = 20;
        let d = 12;
        let p = 8;
        let hidden = Array2::<f32>::from_shape_fn((n, d), |_| normal.sample(&mut rng));
        let weight = Array2::<f32>::from_shape_fn((d, p), |_| normal.sample(&mut rng));
        let handle = WeightHandle::new(0, WeightKind::Q);

        let mut plain = PlaintextExecutor::new(RayonCpuEngine::new());
        plain.provision_weight(handle, weight.view()).unwrap();
        let plain_out = plain.offload_linear(handle, hidden.view()).unwrap();

        let mut auto = InProcessTrustedExecutor::with_seed(
            RayonCpuEngine::new(),
            MaskSeed::from_bytes([71u8; 32]),
        )
        .with_auto_mask();
        auto.provision_weight(handle, weight.view()).unwrap();
        auto.begin_forward_pass(n).unwrap();
        let auto_out = auto.offload_linear(handle, hidden.view()).unwrap();
        auto.end_forward_pass().unwrap();

        for ((i, j), &e) in plain_out.indexed_iter() {
            let got = auto_out[[i, j]];
            assert!(
                (got - e).abs() < 5e-3,
                "({i},{j}): plain={e} auto={got} diff={}",
                (got - e).abs()
            );
        }
    }

    /// `with_auto_mask()` executor agrees with plaintext at a pow2-
    /// aligned shape (resolves to HD₃ path internally).
    #[test]
    fn auto_hd3_path_agrees_with_plaintext() {
        let mut rng = ChaCha20Rng::from_seed([79u8; 32]);
        let normal = StandardNormal;
        // n=24 + k=8 = 32 (pow2 exact) → HD₃ path.
        let n = 24;
        let d = 12;
        let p = 8;
        let hidden = Array2::<f32>::from_shape_fn((n, d), |_| normal.sample(&mut rng));
        let weight = Array2::<f32>::from_shape_fn((d, p), |_| normal.sample(&mut rng));
        let handle = WeightHandle::new(0, WeightKind::Q);

        let mut plain = PlaintextExecutor::new(RayonCpuEngine::new());
        plain.provision_weight(handle, weight.view()).unwrap();
        let plain_out = plain.offload_linear(handle, hidden.view()).unwrap();

        let mut auto = InProcessTrustedExecutor::with_seed(
            RayonCpuEngine::new(),
            MaskSeed::from_bytes([83u8; 32]),
        )
        .with_auto_mask();
        auto.provision_weight(handle, weight.view()).unwrap();
        auto.begin_forward_pass(n).unwrap();
        let auto_out = auto.offload_linear(handle, hidden.view()).unwrap();
        auto.end_forward_pass().unwrap();

        for ((i, j), &e) in plain_out.indexed_iter() {
            let got = auto_out[[i, j]];
            assert!(
                (got - e).abs() < 5e-3,
                "({i},{j}): plain={e} auto={got} diff={}",
                (got - e).abs()
            );
        }
    }

    /// DCT-IV executor at `offload_qkv`: all three projections agree
    /// with plaintext. Verifies the shared-mask path (3× unapply on
    /// one apply) at non-pow2 `n`.
    #[test]
    fn dct4_qkv_agrees_with_plaintext() {
        let mut rng = ChaCha20Rng::from_seed([59u8; 32]);
        let normal = StandardNormal;
        let n = 20;
        let d = 12;

        let hidden = Array2::<f32>::from_shape_fn((n, d), |_| normal.sample(&mut rng));
        let wq = Array2::<f32>::from_shape_fn((d, d), |_| normal.sample(&mut rng));
        let wk = Array2::<f32>::from_shape_fn((d, d), |_| normal.sample(&mut rng));
        let wv = Array2::<f32>::from_shape_fn((d, d), |_| normal.sample(&mut rng));

        let mut exec = InProcessTrustedExecutor::with_seed(
            RayonCpuEngine::new(),
            MaskSeed::from_bytes([61u8; 32]),
        )
        .with_dct4_mask();
        for (kind, w) in [
            (WeightKind::Q, &wq),
            (WeightKind::K, &wk),
            (WeightKind::V, &wv),
        ] {
            exec.provision_weight(WeightHandle::new(0, kind), w.view())
                .unwrap();
        }

        exec.begin_forward_pass(n).unwrap();
        let (q, k, v) = exec.offload_qkv(0, hidden.view()).unwrap();
        exec.end_forward_pass().unwrap();

        let expected_q = hidden.dot(&wq);
        let expected_k = hidden.dot(&wk);
        let expected_v = hidden.dot(&wv);
        for ((i, j), e) in expected_q.indexed_iter() {
            assert!((q[[i, j]] - e).abs() < 5e-3);
        }
        for ((i, j), e) in expected_k.indexed_iter() {
            assert!((k[[i, j]] - e).abs() < 5e-3);
        }
        for ((i, j), e) in expected_v.indexed_iter() {
            assert!((v[[i, j]] - e).abs() < 5e-3);
        }
    }
}

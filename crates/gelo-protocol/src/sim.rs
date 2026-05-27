use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rand_xoshiro::Xoshiro256PlusPlus;

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

/// Reference [`GpuOffloadEngine`] that performs the offloaded GEMM on
/// the CPU via rayon. **Test-only.** Gated behind
/// `#[cfg(any(test, feature = "reference-engine"))]` so production
/// crates can't import it at all.
///
/// Three test-side roles depend on this adapter:
/// - `gelo-protocol`'s own unit tests, which can't reach
///   `gelo_gpu_wgpu::WgpuVulkanEngine` without a circular dependency.
/// - `gelo-gpu-wgpu/tests/parity.rs` — literal CPU-vs-Wgpu parity oracle.
/// - The byzantine-tampering attack suites
///   (`tests/ple_pcie_leak.rs`, `tests/u_verify.rs`,
///   `tests/bss_recovery.rs`) wrap it in
///   `SpyEngine` / `TamperingEngine` / `SnoopingEngine`.
///
/// Production / benches / measurements: use
/// `gelo_gpu_wgpu::WgpuVulkanEngine` instead — see
/// `feedback_benches_use_gelo_gpu.md` and
/// `feedback_no_rayon_cpu_engine.md`.
///
/// Weights are stored as `Arc<Array2<f32>>` so callers that have an
/// Arc can register without an engine-side clone via
/// [`Self::register_weight_shared`].
#[cfg(any(test, feature = "reference-engine"))]
#[derive(Default, Clone)]
pub struct ReferenceCpuEngine {
    weights: HashMap<WeightHandle, Arc<Array2<f32>>>,
}

#[cfg(any(test, feature = "reference-engine"))]
impl ReferenceCpuEngine {
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(any(test, feature = "reference-engine"))]
impl GpuOffloadEngine for ReferenceCpuEngine {
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
    /// Fast non-cryptographic RNG used **only** for shield-row noise
    /// generation in `gaussian::fill_gaussian`.  Xoshiro256++ bulk
    /// fill_bytes is ~3× faster than ChaCha20 on Zen 5; over the v7
    /// fixture the RNG fraction of `gelo:shield_stack` (~33 %) drops
    /// roughly that much.
    ///
    /// **Security**: the shield is distribution-quality material, not
    /// key material — the per-batch mask `A` is what the protocol
    /// hides; shield rows only have to look like `N(0, σ²)` to defeat
    /// the Gram-matrix leak.  Xoshiro256++ passes BigCrush and has no
    /// known attack against its output stream that matters for shield
    /// energy.  The AloePri `c2_default` attack-suite gate must be re-
    /// run before this is flipped on by default in production — see
    /// memory `aloepri_hd3_gate_phase_a_b.md`.
    ///
    /// Seeded deterministically from the executor's [`MaskSeed`] via a
    /// dedicated ChaCha20 stream (stream id `SHIELD_RNG_STREAM`) so the
    /// main `rng` stream-0 position is undisturbed and the
    /// `set_rng_stream` API still works for callers that depend on
    /// stream-keyed reproducibility.
    shield_rng: Xoshiro256PlusPlus,
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
    /// Active session mask. `Some` between `begin_forward_pass`
    /// (or `begin_prefill_pass` / `begin_decode_pass`) and
    /// `end_forward_pass` when `per_forward_mask` is enabled.
    ///
    /// **Single** holds one mask of size `n + shield.k` — non-batched
    /// callers (`begin_forward_pass`) and the M1.11 opt-in shared-A
    /// decode path land here.
    ///
    /// **PerSequence** holds `B` independent masks of size
    /// `n_max + shield.k` each — M1.11 batched-prefill default. The
    /// caller passes `(B * n_max, d_in)` activation; the substrate
    /// applies `masks[b]` to slice `[b*n_max..(b+1)*n_max, :]` per
    /// rayon worker. See [`docs/plans/m1-11-batched-decode.md`] §3.4.
    session: Option<SessionKind>,
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
    /// **M1.12+ batched scratch reuse.** Per-input-width pool of the
    /// `(batch_size * stacked_n, d_in)` concat-masked buffer built by
    /// [`Self::build_per_sequence_masked`]. Without pooling, each
    /// batched offload allocated ~335 MB at Qwen3-4B B=8 long-n and
    /// dropped it on return — ~84 GB of allocator churn per prefill.
    /// Returned via [`Self::return_per_seq_apply_scratch`] after the
    /// engine round-trip + unmask completes. Cleared on
    /// `end_forward_pass`.
    per_seq_apply_scratch: HashMap<usize, Array2<f32>>,
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
    /// **M1.12 bucket-3a opt-in.** When `true`, Haar masks are
    /// constructed via [`crate::mask::GeloMask::fresh_bf16`] (eager
    /// bf16 cache populated alongside f32 sample). Apply / unapply
    /// then route through `aocl_gemm_bf16bf16f32of32` (AVX-512_BF16)
    /// instead of f32 BLIS. Default `false`. Flip via
    /// [`Self::with_haar_mask_bf16`]; no effect on HD₃ / DCT-IV
    /// (those use structured transforms, not GEMM).
    haar_mask_bf16: bool,
}

/// Clone the executor sharing the underlying engine (engines that opt
/// into the `clone_shared` Arc-pattern reuse their weight cache; engines
/// that don't will duplicate state per the `E: Clone` impl). The session
/// mask and scratch are NOT cloned — they only make sense inside a
/// `begin_forward_pass`/`end_forward_pass` bracket on the owning thread.
/// The RNG state IS cloned; callers that want independent streams across
/// clones should chain `.with_rng_stream(stream_id)`.
/// ChaCha20 stream id used **only** for deriving the shield RNG seed
/// from the executor's [`MaskSeed`].  Picking a fixed non-zero stream
/// (a) keeps the main RNG's stream-0 bits untouched, so
/// `set_rng_stream` semantics for callers remain unchanged, and
/// (b) is deterministic across runs given the same `MaskSeed`.
///
/// The literal value has no cryptographic significance — it just
/// needs to be stable and distinct from any stream a caller might
/// pick via `set_rng_stream`.  `0xCAFE_F00D_5EED_E11D` ("cafe food
/// seed-eli(x)d") is unlikely to collide with anything in test
/// fixtures.
const SHIELD_RNG_STREAM: u64 = 0xCAFE_F00D_5EED_E11D;

/// Deterministically derive a [`Xoshiro256PlusPlus`] seed from the
/// executor's [`MaskSeed`] without disturbing the main `ChaCha20Rng`'s
/// stream-0 position.  Uses a single-shot ChaCha20 instance on a
/// dedicated stream — the 32-byte output is the Xoshiro seed.
fn derive_shield_rng(seed: &MaskSeed) -> Xoshiro256PlusPlus {
    let mut bootstrap = ChaCha20Rng::from_seed(seed.0);
    bootstrap.set_stream(SHIELD_RNG_STREAM);
    let mut shield_seed = [0u8; 32];
    bootstrap.fill_bytes(&mut shield_seed);
    Xoshiro256PlusPlus::from_seed(shield_seed)
}

impl<E: GpuOffloadEngine + Clone> Clone for InProcessTrustedExecutor<E> {
    fn clone(&self) -> Self {
        Self {
            engine: self.engine.clone(),
            rng: self.rng.clone(),
            shield_rng: self.shield_rng.clone(),
            shield: self.shield,
            shield_default: self.shield_default,
            shield_small_n: self.shield_small_n,
            shield_small_n_max: self.shield_small_n_max,
            verify_probes: self.verify_probes,
            weights: self.weights.clone(),
            per_forward_mask: self.per_forward_mask,
            session: None,
            stacked_scratch: HashMap::new(),
            per_seq_apply_scratch: HashMap::new(),
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
            haar_mask_bf16: self.haar_mask_bf16,
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

/// Session-level mask topology — either one mask covering all rows
/// (Single, today's behaviour) or B per-sequence masks for batched
/// forwards (PerSequence, M1.11).
///
/// See `docs/plans/m1-11-batched-decode.md` §3.4 for the topology
/// decision and §3.5 for the lifecycle.
enum SessionKind {
    /// Non-batched OR shared-A batched decode (feature-flagged).
    /// One mask covers all rows of the stacked operand.
    Single(SessionMask),
    /// Default batched mode at prefill (and the default at batched
    /// decode until the `BATCHED_DECODE_SHARED_A` gate clears). One
    /// mask per sequence; mask-apply rayon-parallel across `b`.
    PerSequence {
        masks: Vec<MaskFamily>,
        /// Per-sequence data-row count (excluding shield rows). All B
        /// sequences share this; right-padding to a common `n_max`
        /// happens at the caller.
        data_n: usize,
        batch_size: usize,
    },
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
        let shield_rng = derive_shield_rng(&seed);
        Self {
            engine,
            rng: ChaCha20Rng::from_seed(seed.0),
            shield_rng,
            shield: shield_default,
            shield_default,
            // 2026-05-21: at m=1 decode the default k=8 gives
            // stacked_n = 9 and Auto falls to DCT-IV (pad 16/9 = 1.78×
            // > 1.6 threshold). Overlay k=15 makes stacked_n = 16
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
            per_seq_apply_scratch: HashMap::new(),
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
            // 2026-05-22: M1.12 bucket-3a default off. Flip via
            // `with_haar_mask_bf16()` to opt into the AOCL LPGEMM
            // bf16 path. Only affects Haar masks — HD₃ / DCT-IV
            // use structured transforms, not GEMM, so the bf16
            // GEMM path doesn't apply to them.
            haar_mask_bf16: false,
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
        let shield_rng = derive_shield_rng(&seed);
        Self {
            engine,
            rng: ChaCha20Rng::from_seed(seed.0),
            shield_rng,
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
            per_seq_apply_scratch: HashMap::new(),
            perm_attn: PermAttnConfig::default(),
            ple_table: None,
            snapshot_capture: None,
            // `with_shield` is the per-offload legacy/safety-test
            // path used by BSS-recovery and DP-Forward tests; keep
            // Haar to preserve the explicit reference behaviour
            // those tests target. Production paths use `new` /
            // `with_seed` which default to `MaskKind::Auto`.
            mask_kind: MaskKind::Haar,
            haar_mask_bf16: false,
        }
    }

    /// Pin the mask family to the GELO paper's dense Householder-QR
    /// Haar sample. Reverses any prior `with_hd3_mask` /
    /// `with_dct4_mask` / `with_auto_mask` switch. Provided for
    /// symmetry with the other family setters and for tests that
    /// hard-code stacked-row counts (Haar never pads; HD₃ rounds
    /// stacked_n up to the next power of two).
    pub fn with_haar_mask(mut self) -> Self {
        self.mask_kind = MaskKind::Haar;
        self.session = None;
        self.stacked_scratch.clear();
        self.per_seq_apply_scratch.clear();
        self
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
        self.per_seq_apply_scratch.clear();
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
        self.per_seq_apply_scratch.clear();
        self
    }

    /// Switch to the shape-adaptive mask dispatch: picks HD₃ when the
    /// pad penalty `s_pad / s ≤ 8/5 = 1.6` (the
    /// [`crate::mask::HD3_AUTO_MAX_PAD_RATIO_NUM`] / `_DEN` constants),
    /// DCT-IV otherwise. Resolution happens at `begin_forward_pass`
    /// (per-forward-pass mode) or per-call (per-offload mode), so the
    /// physical mask family used at each call adapts to the shape
    /// without caller intervention.
    ///
    /// Use this as the default for production workloads with mixed
    /// prompt sizes — both HD₃ at pow2-aligned shapes and DCT-IV at
    /// far-from-pow2 shapes beat Haar; the empirical crossover is
    /// somewhere in pad ratio (1.59, 1.99) per the 2026-05-26 sweep
    /// (`docs/plans/gelo-llm-perf-roadmap.md` §1.4); 8/5 = 1.6 sits in the
    /// confirmed-HD₃-wins band.
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
        self.per_seq_apply_scratch.clear();
        self
    }

    pub fn mask_kind(&self) -> MaskKind {
        self.mask_kind
    }

    /// **M1.12 bucket-3a opt-in.** Route Haar mask apply/unapply
    /// through the AOCL-BLIS LPGEMM addon's bf16 GEMM kernel
    /// (`aocl_gemm_bf16bf16f32of32`, AVX-512_BF16 via vdpbf16ps)
    /// instead of the f32 BLIS path. Sample-side semantics are
    /// unchanged: Haar `A` is still f32-sampled in TEE with the
    /// Mezzadri sign-correction (load-bearing for Haar-uniformity);
    /// the downcast happens once at construction time
    /// ([`crate::mask::GeloMask::fresh_bf16`]) and is cached
    /// alongside the f32 source.
    ///
    /// **No effect on HD₃ / DCT-IV** — those use structured
    /// orthogonal transforms (FWHT, DCT-IV) rather than GEMM, so a
    /// bf16 GEMM kernel isn't on their critical path. If
    /// `mask_kind` is HD₃ or DCT-IV when this flag is set, the
    /// flag is silently inert at the apply/unapply boundary — only
    /// Haar consults it.
    ///
    /// **Security argument:** bf16 quantisation noise on the
    /// masked operand `A · H` is strictly larger than the f32 →
    /// f16 noise at the existing GPU upload boundary, so adversary
    /// observations are noisier. AloePri attack drivers
    /// (anchor_ica / JADE / JD / gram_error) operate on
    /// observation noise floors; a noisier observation strictly
    /// weakens recovery rates. No new AloePri gate required — see
    /// `docs/plans/m1-12-bf16-activation-pipeline.md` §5 for the
    /// math-only argument.
    ///
    /// **Requires:** `--features blas` AND a build with the AOCL
    /// LPGEMM addon enabled. The install script
    /// `scripts/install-aocl-blis.sh` passes
    /// `--enable-addon=aocl_gemm` since 2026-05-22; pre-2026-05-22
    /// builds will get auto-rebuilt on next invocation of the
    /// install script.
    ///
    /// **Default:** off. Flip after the perf gate at
    /// `docs/plans/m1-12-bf16-activation-pipeline.md` §1.1 clears
    /// (≥ 20 % prefill wall reduction at Qwen3-4B B=8 n=2048) AND
    /// real-weight parity tests on `qwen3_generation_e2e` and the
    /// v7 extraction bench preserve greedy token output.
    #[cfg(feature = "blas")]
    pub fn with_haar_mask_bf16(mut self) -> Self {
        self.haar_mask_bf16 = true;
        // Invalidate any session that was sampled at f32 — the
        // caller must re-bracket with `begin_forward_pass` after
        // toggling precision.
        self.session = None;
        self.stacked_scratch.clear();
        self.per_seq_apply_scratch.clear();
        self
    }

    /// `true` if Haar masks will be constructed via the bf16 LPGEMM
    /// path. Always `false` on non-`blas` builds (the bf16 kernel
    /// lives in AOCL-BLIS).
    pub fn is_haar_mask_bf16(&self) -> bool {
        self.haar_mask_bf16
    }

    /// Construct a Haar mask honoring the executor's bf16-mode flag.
    /// Centralised so every `MaskKind::Haar` arm picks up the bf16
    /// path the same way; flipping `with_haar_mask_bf16` once at
    /// executor construction propagates to every per-forward-pass
    /// mask sample (5 call sites today: `begin_forward_pass`,
    /// `begin_prefill_pass`, `begin_decode_pass`, plus their PerSequence
    /// variants).
    fn make_haar_mask(&mut self, stacked_n: usize) -> MaskFamily {
        #[cfg(feature = "blas")]
        {
            if self.haar_mask_bf16 {
                return MaskFamily::Haar(GeloMask::fresh_bf16(stacked_n, &mut self.rng));
            }
        }
        MaskFamily::Haar(GeloMask::fresh(stacked_n, &mut self.rng))
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
        self.per_seq_apply_scratch.clear();
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
                Some(SessionKind::Single(s))
                    if s.data_n == n_data && s.mask.n() == stacked_n =>
                {
                    s.mask.clone()
                }
                Some(SessionKind::Single(s)) => {
                    panic!(
                        "per-forward-pass mask: offload n={n_data} (stacked {stacked_n}) \
                         doesn't match session n={} (stacked {}); did you forget to call \
                         begin_forward_pass for the new shape?",
                        s.data_n,
                        s.mask.n(),
                    );
                }
                Some(SessionKind::PerSequence { .. }) => panic!(
                    "build_shielded_and_apply called under PerSequence session — \
                     batched callers must use the batched offload path (M1.11 R1.2)"
                ),
                None => panic!(
                    "per-forward-pass mode but no session mask — \
                     embedder must call begin_forward_pass(n) before any offload_*"
                ),
            }
        } else {
            profile::time("gelo:mask_sample", || match resolved_kind {
                MaskKind::Haar => self.make_haar_mask(stacked_n),
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
                            &mut self.shield_rng,
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
                            &mut self.shield_rng,
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
                            &mut self.shield_rng,
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
                let (mut stacked, _n) = stack_shield(hidden, self.shield, &mut self.shield_rng);
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

    /// **M1.11 R1.2 / R1.5** — build B shielded+masked operands and
    /// concat into a `(batch_size * stacked_n, d_in)` tensor.
    ///
    /// Step 1 (shield stack) runs serial — the shield RNG is a single
    /// `&mut self.shield_rng` borrowed sequentially per block.
    /// Step 2 (mask apply) runs **rayon-parallel** across blocks
    /// (R1.5): `MaskFamily::apply*` are `&self` methods so per-block
    /// dispatch is `Sync`. HD₃/DCT-IV `apply_in_place` need owned
    /// `Array2`, so we clone the block view, apply, assign back — same
    /// allocation cost as the serial version, just parallelised.
    fn build_per_sequence_masked(
        &mut self,
        hidden: ArrayView2<'_, f32>,
        masks: &[MaskFamily],
        batch_size: usize,
        data_n: usize,
    ) -> Array2<f32> {
        let d_in = hidden.ncols();
        let k = self.shield.k;
        let stacked_n = masks[0].n();
        let scale = self.shield.energy_scale;

        // Step 1 (D1.7): per-block parallel shield stack with
        // sub-stream RNGs. The serial path was 17.2% of batched wall
        // in the D1.6 microbench — the bottleneck was the shared
        // `&mut self.shield_rng` borrow forcing per-block fills to
        // run sequentially.
        //
        // Fix: pre-derive `batch_size` independent Xoshiro256PlusPlus
        // seeds from the parent (advances parent by B*32 bytes),
        // then rayon-iterate across blocks with each closure holding
        // its own local Xoshiro. Mirrors
        // `attention.rs::add_gaussian_3d_inplace`'s per-head
        // parallelisation pattern. Per-element distribution is
        // unchanged (independent `N(0, σ²)` everywhere); only the
        // cross-block correlation structure differs — invariant for
        // the GELO shield-energy argument since shield rows of
        // different sequences are independent by construction.
        let mut concat_masked = profile::time("gelo:shield_stack", || {
            // **M1.12+ batched scratch reuse.** Pull a pooled
            // `(B * stacked_n, d_in)` buffer if one's available at this
            // width; otherwise allocate. The returned-after-use path
            // re-inserts via [`Self::return_per_seq_apply_scratch`]
            // from each batched offload site. Saves ~335 MB alloc per
            // offload at Qwen3-4B B=8 long-n prefill.
            let total_rows = batch_size.saturating_mul(stacked_n);
            let mut buf = self
                .per_seq_apply_scratch
                .remove(&d_in)
                .filter(|b| b.shape() == [total_rows, d_in])
                .unwrap_or_else(|| Array2::<f32>::zeros((total_rows, d_in)));
            // Zero the pad region eagerly — previous call may have
            // left non-zero data there. Cheap vs the alternative of
            // tracking pad-row state per slot.
            // (Data + shield rows are overwritten below.)
            // Pre-derive B Xoshiro seeds from the parent shield RNG.
            // Single-threaded; cheap (B * 32 bytes RNG output).
            let seeds: Vec<[u8; 32]> = (0..batch_size)
                .map(|_| {
                    let mut s = [0u8; 32];
                    self.shield_rng.fill_bytes(&mut s);
                    s
                })
                .collect();

            use ndarray::parallel::prelude::*;
            buf.axis_chunks_iter_mut(ndarray::Axis(0), stacked_n)
                .into_par_iter()
                .enumerate()
                .for_each(|(b, mut block_view)| {
                    let sub_in =
                        hidden.slice(ndarray::s![b * data_n..(b + 1) * data_n, ..]);
                    let mean_norm = mean_row_norm(sub_in);
                    let sigma = if d_in == 0 {
                        0.0
                    } else {
                        scale * mean_norm / (d_in as f32).sqrt()
                    };
                    // Place data rows.
                    block_view
                        .slice_mut(ndarray::s![..data_n, ..])
                        .assign(&sub_in);
                    // Place shield rows.
                    let shield_end_local = (data_n + k).min(stacked_n);
                    if shield_end_local > data_n {
                        let mut local_rng =
                            rand_xoshiro::Xoshiro256PlusPlus::from_seed(seeds[b]);
                        fill_shield_rows_inline(
                            block_view.slice_mut(
                                ndarray::s![data_n..shield_end_local, ..],
                            ),
                            sigma,
                            &mut local_rng,
                        );
                    }
                    // Pad region (rows shield_end_local..stacked_n).
                    // **Reused scratch may hold stale data here** from
                    // a prior offload, so we always zero it. Cheap vs
                    // tracking per-block pad-row freshness.
                    if shield_end_local < stacked_n {
                        block_view
                            .slice_mut(ndarray::s![shield_end_local.., ..])
                            .fill(0.0);
                    }
                });
            buf
        });

        // Step 2 (R1.5): rayon-parallel per-block mask apply. The
        // outer profile category mirrors `MaskFamily::apply_profile_category`
        // so per-family wall time is visible on the batched path —
        // otherwise `Auto`'s runtime HD₃-vs-DCT-IV split is invisible
        // in the prefill profile dump (the single-mask path at
        // build_shielded_and_apply already splits this way).
        profile::time(masks[0].apply_profile_category(), || {
            use rayon::prelude::*;
            // **M1.12+** — drive the per-block apply on raw `&mut [f32]`
            // chunks so HD₃/DCT-IV `apply_in_place_slice` runs directly
            // on the concat buffer. The previous `block_view.to_owned()
            // + apply_in_place + block_view.assign` triple paid ~1.5 GB
            // of extra memory traffic per offload at B=8 long-n
            // prefill; the slice path is allocation-free.
            let chunk_len = stacked_n.saturating_mul(d_in);
            let slice = concat_masked
                .as_slice_mut()
                .expect("concat_masked must be row-major contiguous");
            slice
                .par_chunks_mut(chunk_len)
                .enumerate()
                .for_each(|(b, block)| match &masks[b] {
                    MaskFamily::Hd3(hd3) => hd3.apply_in_place_slice(block, d_in),
                    MaskFamily::Dct4(dct4) => dct4.apply_in_place_slice(block, d_in),
                    MaskFamily::Haar(_) => {
                        // Haar still goes through the allocating GEMM
                        // path (separate output workspace required).
                        let view = ndarray::ArrayView2::from_shape(
                            (stacked_n, d_in),
                            block,
                        )
                        .expect("chunk has the right shape");
                        let masked = masks[b].apply(view);
                        block.copy_from_slice(
                            masked
                                .as_slice()
                                .expect("fresh Array2 is contiguous"),
                        );
                    }
                });
        });
        concat_masked
    }

    /// Re-pool a concat-masked buffer returned by [`Self::build_per_sequence_masked`].
    /// Called by the batched offload sites after the engine round-trip
    /// + unmask completes; keys by input width `d_in` so the next
    /// offload at the same width picks it up.
    fn return_per_seq_apply_scratch(&mut self, buf: Array2<f32>) {
        // Only worthwhile under paper-parity + non-Haar — Haar masks
        // would still pay the to_owned in the apply loop (see above),
        // so pooling buys nothing there; and per-offload mode owns
        // its own buffers.
        if self.per_forward_mask
            && matches!(
                self.mask_kind,
                MaskKind::Hd3 | MaskKind::Dct4 | MaskKind::Auto
            )
        {
            let d_in = buf.ncols();
            self.per_seq_apply_scratch.insert(d_in, buf);
        }
    }

    /// **M1.11 R1.5 / M1.12+** — rayon-parallel per-sequence unmask + shield
    /// strip. Consumes the `(batch_size * stacked_n, d_out)` engine output
    /// by value so HD₃/DCT-IV can unmask in place on its slice; produces a
    /// `(batch_size * data_n, d_out)` user-facing tensor.
    ///
    /// The previous `view`-flavored signature forced a per-block
    /// `(stacked_n, d_out)` allocation inside the rayon closure (via
    /// `mask.unapply()`) — ~96 MB / block at Qwen3-4B d=2560 stacked=4096,
    /// times B per offload. Taking the buffer by value lets us run
    /// `_in_place_slice` on each chunk and only pay the `(data_n, d_out)`
    /// memcpy into the output buffer.
    fn unmask_per_sequence(
        &self,
        mut concat_out: Array2<f32>,
        masks: &[MaskFamily],
        batch_size: usize,
        data_n: usize,
    ) -> Array2<f32> {
        let stacked_n = masks[0].n();
        let d_out = concat_out.ncols();
        let mut output = Array2::<f32>::zeros((batch_size * data_n, d_out));
        profile::time(masks[0].unapply_profile_category(), || {
            use rayon::prelude::*;
            let in_chunk_len = stacked_n.saturating_mul(d_out);
            let out_chunk_len = data_n.saturating_mul(d_out);
            let prefix_len = out_chunk_len; // bytes-equivalent: data_n * d_out f32s
            let in_slice = concat_out
                .as_slice_mut()
                .expect("concat_out must be row-major contiguous");
            let out_slice = output
                .as_slice_mut()
                .expect("fresh Array2 is contiguous");
            in_slice
                .par_chunks_mut(in_chunk_len)
                .zip(out_slice.par_chunks_mut(out_chunk_len))
                .enumerate()
                .for_each(|(b, (in_block, out_block))| match &masks[b] {
                    MaskFamily::Hd3(hd3) => {
                        hd3.unapply_in_place_slice(in_block, d_out);
                        out_block.copy_from_slice(&in_block[..prefix_len]);
                    }
                    MaskFamily::Dct4(dct4) => {
                        dct4.unapply_in_place_slice(in_block, d_out);
                        out_block.copy_from_slice(&in_block[..prefix_len]);
                    }
                    MaskFamily::Haar(_) => {
                        // Haar still allocates — Aᵀ · M needs a separate
                        // workspace for the GEMM output.
                        let view = ndarray::ArrayView2::from_shape(
                            (stacked_n, d_out),
                            in_block,
                        )
                        .expect("chunk has the right shape");
                        let unmasked = masks[b].unapply(view);
                        out_block.copy_from_slice(
                            unmasked
                                .slice(ndarray::s![..data_n, ..])
                                .to_owned()
                                .as_slice()
                                .expect("contiguous"),
                        );
                    }
                });
        });
        output
    }

    /// **M1.11 R1.2** — batched per-sequence offload_linear path.
    ///
    /// Called from `offload_linear` when the session is
    /// `SessionKind::PerSequence`. Hidden is `(batch_size * data_n,
    /// d_in)` with contiguous B-blocks. Each block gets `masks[b]`
    /// applied independently; the B masked operands concat into one
    /// `(batch_size * stacked_n, d_in)` tensor for a single engine
    /// matmul call (preserving GPU dispatch amortisation across B).
    ///
    /// R1.5: per-block mask apply / unapply runs rayon-parallel.
    fn offload_linear_per_sequence(
        &mut self,
        handle: WeightHandle,
        hidden: ArrayView2<'_, f32>,
        batch_size: usize,
        data_n: usize,
    ) -> Result<Array2<f32>> {
        assert_eq!(
            hidden.nrows(),
            batch_size * data_n,
            "offload_linear_per_sequence: hidden has {} rows; expected B*data_n = {}*{} = {}",
            hidden.nrows(),
            batch_size,
            data_n,
            batch_size * data_n
        );
        if !self.per_forward_mask {
            return Err(anyhow!(
                "PerSequence session but per_forward_mask is false — \
                 batched offload requires paper-parity mode"
            ));
        }
        let masks: Vec<MaskFamily> = match &self.session {
            Some(SessionKind::PerSequence { masks, .. }) => masks.clone(),
            _ => unreachable!("offload_linear_per_sequence called outside PerSequence"),
        };

        let concat_masked = self.build_per_sequence_masked(hidden, &masks, batch_size, data_n);

        let concat_out = profile::time("engine:matmul", || {
            self.engine.matmul(handle, concat_masked.view())
        })?;
        self.record_snapshot(handle, &concat_masked, Some(&concat_out));
        if self.verify_probes > 0 {
            let weight = self.weights.get(&handle).ok_or_else(|| {
                anyhow!("verify_probes>0 but weight {handle:?} not cached in TEE")
            })?;
            profile::time("uverify:linear", || {
                verify_offload(
                    self.verify_probes,
                    concat_masked.view(),
                    weight.view(),
                    concat_out.view(),
                    &mut self.rng,
                )
            })?;
        }

        let result =
            self.unmask_per_sequence(concat_out, &masks, batch_size, data_n);
        self.return_per_seq_apply_scratch(concat_masked);
        Ok(result)
    }

    /// **M1.11 R1.6** — batched per-sequence offload_qkv path. Builds
    /// the masked operand **once**, dispatches Q/K/V via one
    /// `engine.matmul_many` call (3 GEMMs amortising the operand
    /// upload), then unmasks per output. Same shape contract as the
    /// non-batched override: hidden is `(B*data_n, d_in)`; returns
    /// three `(B*data_n, kv_dim_*)` tensors.
    fn offload_qkv_per_sequence(
        &mut self,
        layer: u16,
        hidden: ArrayView2<'_, f32>,
        batch_size: usize,
        data_n: usize,
    ) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>)> {
        let masks: Vec<MaskFamily> = match &self.session {
            Some(SessionKind::PerSequence { masks, .. }) => masks.clone(),
            _ => unreachable!("offload_qkv_per_sequence called outside PerSequence"),
        };
        let concat_masked = self.build_per_sequence_masked(hidden, &masks, batch_size, data_n);

        let handles = [
            WeightHandle::new(layer, WeightKind::Q),
            WeightHandle::new(layer, WeightKind::K),
            WeightHandle::new(layer, WeightKind::V),
        ];
        let qkv_out = profile::time("engine:matmul_many", || {
            self.engine.matmul_many(&handles, concat_masked.view())
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

        // Snapshot + U-Verify per output kind (same masked operand
        // drives all three).
        self.record_snapshot(WeightHandle::new(layer, WeightKind::Q), &concat_masked, Some(&mq));
        self.record_snapshot(WeightHandle::new(layer, WeightKind::K), &concat_masked, Some(&mk));
        self.record_snapshot(WeightHandle::new(layer, WeightKind::V), &concat_masked, Some(&mv));
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
                        concat_masked.view(),
                        w.view(),
                        observed.view(),
                        &mut self.rng,
                    )
                })?;
            }
        }

        let q = self.unmask_per_sequence(mq, &masks, batch_size, data_n);
        let k_out = self.unmask_per_sequence(mk, &masks, batch_size, data_n);
        let v_out = self.unmask_per_sequence(mv, &masks, batch_size, data_n);
        self.return_per_seq_apply_scratch(concat_masked);
        Ok((q, k_out, v_out))
    }

    /// **M1.11 R1.6** — generic batched per-sequence `offload_linear_many`.
    /// Same shape as `offload_qkv_per_sequence` but for an arbitrary
    /// list of weight handles (SwiGLU gate+up shares hidden, this is
    /// the canonical caller).
    fn offload_linear_many_per_sequence(
        &mut self,
        handles: &[WeightHandle],
        hidden: ArrayView2<'_, f32>,
        batch_size: usize,
        data_n: usize,
    ) -> Result<Vec<Array2<f32>>> {
        if handles.is_empty() {
            return Ok(Vec::new());
        }
        let masks: Vec<MaskFamily> = match &self.session {
            Some(SessionKind::PerSequence { masks, .. }) => masks.clone(),
            _ => unreachable!("offload_linear_many_per_sequence called outside PerSequence"),
        };
        let concat_masked = self.build_per_sequence_masked(hidden, &masks, batch_size, data_n);

        let masked_outs = profile::time("engine:matmul_many", || {
            self.engine.matmul_many(handles, concat_masked.view())
        })?;
        anyhow::ensure!(
            masked_outs.len() == handles.len(),
            "engine.matmul_many returned {} results; expected {}",
            masked_outs.len(),
            handles.len()
        );

        for (h, out) in handles.iter().zip(masked_outs.iter()) {
            self.record_snapshot(*h, &concat_masked, Some(out));
        }
        if self.verify_probes > 0 {
            for (h, observed) in handles.iter().zip(masked_outs.iter()) {
                let w = self.weights.get(h).ok_or_else(|| {
                    anyhow!("verify_probes>0 but weight {h:?} not cached in TEE")
                })?;
                profile::time("uverify:linear", || {
                    verify_offload(
                        self.verify_probes,
                        concat_masked.view(),
                        w.view(),
                        observed.view(),
                        &mut self.rng,
                    )
                })?;
            }
        }

        let outputs: Vec<Array2<f32>> = masked_outs
            .into_iter()
            .map(|m| self.unmask_per_sequence(m, &masks, batch_size, data_n))
            .collect();
        self.return_per_seq_apply_scratch(concat_masked);
        Ok(outputs)
    }
}

/// Overwrite `shield_dest` (a (k × d) mutable view) with fresh Gaussian
/// shield rows scaled to per-component `sigma`. The caller computes
/// `sigma = energy_scale × mean_row_norm(hidden) / sqrt(d)` once and
/// passes it in; this lets the helper avoid re-borrowing `hidden`
/// against the scratch buffer slice in the paper-parity path.
///
/// Delegates to [`crate::gaussian::fill_gaussian`], which uses a bulk
/// RNG draw + SIMD Box-Muller (`wide::f32x8`). At d=2560, k=15 (decode)
/// this is ~1.6× faster than the prior per-element
/// `rand_distr::StandardNormal::sample` loop — see the
/// `shield_gaussian` criterion bench.  When `shield_dest` is contiguous
/// (the always-true case in `build_shielded_and_apply`'s scratch-reuse
/// buffer) we hit the fast path with a single `fill_gaussian` call over
/// the whole `k·d` slab.  We fall back to a per-row loop on non-
/// contiguous views to keep the helper general.
fn fill_shield_rows_inline<R: rand::RngCore>(
    mut shield_dest: ndarray::ArrayViewMut2<'_, f32>,
    sigma: f32,
    rng: &mut R,
) {
    if let Some(slab) = shield_dest.as_slice_mut() {
        crate::gaussian::fill_gaussian(slab, sigma, rng);
    } else {
        for mut row in shield_dest.rows_mut() {
            if let Some(row_slice) = row.as_slice_mut() {
                crate::gaussian::fill_gaussian(row_slice, sigma, rng);
            } else {
                // Strided row: fall back to scalar Box-Muller. This
                // path is unreachable from the executor's scratch-
                // reuse paths but kept for defensive generality.
                use rand_distr::{Distribution, StandardNormal};
                let normal = StandardNormal;
                for v in row.iter_mut() {
                    let z: f32 = normal.sample(rng);
                    *v = z * sigma;
                }
            }
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
            MaskKind::Haar => self.make_haar_mask(stacked_n),
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
        self.session = Some(SessionKind::Single(SessionMask { mask, data_n: n }));
        // Stale scratches from a prior forward with a different `n` are
        // unusable now — clear to avoid silently feeding the wrong row
        // count into mask.apply().
        self.stacked_scratch.clear();
        self.per_seq_apply_scratch.clear();
        Ok(())
    }

    fn end_forward_pass(&mut self) -> Result<()> {
        self.session = None;
        self.stacked_scratch.clear();
        self.per_seq_apply_scratch.clear();
        Ok(())
    }

    /// M1.11 R1.1 — batched-prefill bracket. Samples `batch_size`
    /// per-sequence masks of size `(n_max + shield_k, n_max + shield_k)`
    /// each, stores them as `SessionKind::PerSequence`. The shape-
    /// adaptive shield overlay applies based on `n_max` (per-sequence
    /// row count), not on the batch size — at typical prefill widths
    /// (n_max ≥ 32) the overlay never fires, so we use
    /// `shield_default` (k=8).
    fn begin_prefill_pass(
        &mut self,
        batch_size: usize,
        n_max: usize,
    ) -> Result<()> {
        if !self.per_forward_mask {
            // Per-offload mode: each offload samples its own fresh
            // mask, so the bracket is a no-op (same contract as
            // begin_forward_pass).
            return Ok(());
        }
        if batch_size == 0 {
            return Err(anyhow!(
                "begin_prefill_pass: batch_size must be > 0"
            ));
        }
        if n_max == 0 {
            return Err(anyhow!(
                "begin_prefill_pass: n_max must be > 0"
            ));
        }
        // Shape-adaptive shield by per-sequence n_max — same logic as
        // begin_forward_pass. At typical rerank/extraction prefill
        // widths (n_max ≥ 200) the overlay won't fire.
        self.shield = match self.shield_small_n {
            Some(small) if n_max <= self.shield_small_n_max => small,
            _ => self.shield_default,
        };
        let stacked_n = n_max + self.shield.k;
        let resolved_kind =
            crate::mask::resolve_mask_kind_for_shape(self.mask_kind, stacked_n);

        // Sample B independent masks sequentially from the executor's
        // main RNG stream. Each mask consumes some bytes from the
        // stream, so the masks are independent. No need to set_stream
        // per-b — the RNG is a long-period CSPRNG (ChaCha20) and
        // sequential consumption gives uncorrelated samples.
        let mut masks = Vec::with_capacity(batch_size);
        for _b in 0..batch_size {
            let mask = profile::time("gelo:mask_sample", || match resolved_kind {
                MaskKind::Haar => self.make_haar_mask(stacked_n),
                MaskKind::Hd3 => {
                    let s_pad = stacked_n.next_power_of_two().max(2);
                    MaskFamily::Hd3(Hd3Mask::fresh(s_pad, &mut self.rng))
                }
                MaskKind::Dct4 => MaskFamily::Dct4(
                    crate::dct4::Dct4Mask::fresh(stacked_n, &mut self.rng),
                ),
                MaskKind::Auto => unreachable!("Auto resolved above"),
            });
            masks.push(mask);
        }
        self.session = Some(SessionKind::PerSequence {
            masks,
            data_n: n_max,
            batch_size,
        });
        // Per-sequence scratch is bespoke; clear any pre-existing
        // single-mask scratch so the per-sequence offload path
        // doesn't accidentally pick it up.
        self.stacked_scratch.clear();
        self.per_seq_apply_scratch.clear();
        Ok(())
    }

    /// M1.11 D1.1 — batched-decode bracket. At decode each sequence
    /// contributes one data row, so per-sequence A_b is sized
    /// `(1 + shield_k, 1 + shield_k)` — the shape-adaptive shield
    /// overlay (k=15 at n=1) lands stacked_n=16 HD₃-aligned per b.
    ///
    /// Opt-in shared-A path (env `BATCHED_DECODE_SHARED_A=1`, gated
    /// on AloePri `c5_batched_decode_shared_a`): one Single mask of
    /// size `(B + k, B + k)` with `k = shield_k_for_batch(B, 8)`.
    /// Mixes B current-token rows; HD₃-aligned at every B.
    fn begin_decode_pass(&mut self, batch_size: usize) -> Result<()> {
        if !self.per_forward_mask {
            return Ok(());
        }
        if batch_size == 0 {
            return Err(anyhow!("begin_decode_pass: batch_size must be > 0"));
        }

        let shared = std::env::var("BATCHED_DECODE_SHARED_A").as_deref() == Ok("1");

        if shared {
            // Shared dense A — size (B+k, B+k), HD₃ at every B.
            let k_base = self.shield_default.k.max(1);
            let k = crate::shield::shield_k_for_batch(batch_size, k_base);
            // Use the default energy scale; shape is dictated by k.
            self.shield = crate::shield::ShieldConfig::new(k, self.shield_default.energy_scale);
            let stacked_n = batch_size + k;
            let resolved_kind =
                crate::mask::resolve_mask_kind_for_shape(self.mask_kind, stacked_n);
            let mask = profile::time("gelo:mask_sample", || match resolved_kind {
                MaskKind::Haar => self.make_haar_mask(stacked_n),
                MaskKind::Hd3 => {
                    let s_pad = stacked_n.next_power_of_two().max(2);
                    MaskFamily::Hd3(Hd3Mask::fresh(s_pad, &mut self.rng))
                }
                MaskKind::Dct4 => {
                    MaskFamily::Dct4(crate::dct4::Dct4Mask::fresh(stacked_n, &mut self.rng))
                }
                MaskKind::Auto => unreachable!("Auto resolved above"),
            });
            self.session = Some(SessionKind::Single(SessionMask {
                mask,
                data_n: batch_size,
            }));
            self.stacked_scratch.clear();
        self.per_seq_apply_scratch.clear();
            return Ok(());
        }

        // Default: per-sequence A_b at n=1. Reuses the shape-
        // adaptive overlay (k=15 at n=1 by construction) so each
        // A_b lands HD₃-aligned at stacked_n=16.
        self.shield = match self.shield_small_n {
            Some(small) if 1 <= self.shield_small_n_max => small,
            _ => self.shield_default,
        };
        let stacked_n = 1 + self.shield.k;
        let resolved_kind = crate::mask::resolve_mask_kind_for_shape(self.mask_kind, stacked_n);
        let mut masks = Vec::with_capacity(batch_size);
        for _b in 0..batch_size {
            let mask = profile::time("gelo:mask_sample", || match resolved_kind {
                MaskKind::Haar => self.make_haar_mask(stacked_n),
                MaskKind::Hd3 => {
                    let s_pad = stacked_n.next_power_of_two().max(2);
                    MaskFamily::Hd3(Hd3Mask::fresh(s_pad, &mut self.rng))
                }
                MaskKind::Dct4 => {
                    MaskFamily::Dct4(crate::dct4::Dct4Mask::fresh(stacked_n, &mut self.rng))
                }
                MaskKind::Auto => unreachable!("Auto resolved above"),
            });
            masks.push(mask);
        }
        self.session = Some(SessionKind::PerSequence {
            masks,
            data_n: 1,
            batch_size,
        });
        self.stacked_scratch.clear();
        self.per_seq_apply_scratch.clear();
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
        // ReferenceCpuEngine, the default impl converts bf16 → f32 via
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
        // (`ReferenceCpuEngine` does), the Arc is stored directly.
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
        // Branch on session kind: PerSequence (batched prefill, M1.11)
        // expects hidden = (batch_size * data_n, d_in) with contiguous
        // B-blocks. Single (legacy / non-batched) uses the existing
        // single-mask path.
        let per_sequence_dims = match &self.session {
            Some(SessionKind::PerSequence {
                batch_size,
                data_n,
                ..
            }) => Some((*batch_size, *data_n)),
            _ => None,
        };
        if let Some((batch_size, data_n)) = per_sequence_dims {
            return self.offload_linear_per_sequence(handle, hidden, batch_size, data_n);
        }
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
        // M1.11 R1.6: under PerSequence session, route to the
        // batched per-sequence qkv helper which shares one masked
        // operand across Q/K/V via a single `matmul_many` dispatch.
        let per_sequence_dims = match &self.session {
            Some(SessionKind::PerSequence {
                batch_size, data_n, ..
            }) => Some((*batch_size, *data_n)),
            _ => None,
        };
        if let Some((batch_size, data_n)) = per_sequence_dims {
            return self.offload_qkv_per_sequence(layer, hidden, batch_size, data_n);
        }
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
        // M1.11 R1.6: under PerSequence session, route to the
        // batched per-sequence many helper which amortises the
        // masked operand build + dispatch across N matmuls.
        let per_sequence_dims = match &self.session {
            Some(SessionKind::PerSequence {
                batch_size, data_n, ..
            }) => Some((*batch_size, *data_n)),
            _ => None,
        };
        if let Some((batch_size, data_n)) = per_sequence_dims {
            return self.offload_linear_many_per_sequence(handles, hidden, batch_size, data_n);
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

        let mut plain = PlaintextExecutor::new(ReferenceCpuEngine::new());
        plain.provision_weight(handle, weight.view()).unwrap();
        let plain_out = plain.offload_linear(handle, hidden.view()).unwrap();

        let mut masked = InProcessTrustedExecutor::with_seed(
            ReferenceCpuEngine::new(),
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
            ReferenceCpuEngine::new(),
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

    /// **M1.12 bucket-3a** — `with_haar_mask_bf16()` produces the
    /// same downstream output as `PlaintextExecutor` to bf16-floor.
    /// Validates that the executor-level builder flag actually
    /// routes Haar mask construction through `GeloMask::fresh_bf16`
    /// at `begin_forward_pass` AND that `apply` / `unapply` route
    /// through the AOCL LPGEMM bf16 path.
    #[cfg(feature = "blas")]
    #[test]
    fn haar_bf16_executor_agrees_with_plaintext() {
        let mut rng = ChaCha20Rng::from_seed([23u8; 32]);
        let normal = StandardNormal;
        let n = 16;
        let d = 12;
        let p = 8;

        let hidden = Array2::<f32>::from_shape_fn((n, d), |_| normal.sample(&mut rng));
        let weight = Array2::<f32>::from_shape_fn((d, p), |_| normal.sample(&mut rng));
        let handle = WeightHandle::new(0, WeightKind::Q);

        let mut plain = PlaintextExecutor::new(ReferenceCpuEngine::new());
        plain.provision_weight(handle, weight.view()).unwrap();
        let plain_out = plain.offload_linear(handle, hidden.view()).unwrap();

        let mut bf16 = InProcessTrustedExecutor::with_seed(
            ReferenceCpuEngine::new(),
            MaskSeed::from_bytes([29u8; 32]),
        )
        .with_haar_mask()       // pin Haar (default is Auto = HD₃/DCT-IV)
        .with_haar_mask_bf16(); // opt into the AOCL LPGEMM bf16 path
        assert_eq!(bf16.mask_kind(), MaskKind::Haar);
        assert!(bf16.is_haar_mask_bf16());

        bf16.provision_weight(handle, weight.view()).unwrap();
        bf16.begin_forward_pass(n).unwrap();
        let bf16_out = bf16.offload_linear(handle, hidden.view()).unwrap();
        bf16.end_forward_pass().unwrap();

        assert_eq!(bf16_out.dim(), plain_out.dim());
        // Tolerance: bf16 mask round-trip + shield + matmul + unapply
        // through the full executor pipeline. bf16 has 7 mantissa
        // bits → per-element ULP ≈ 0.78 % of magnitude. The chain
        // accumulates:
        //   1. Mask apply A·H at bf16 (one matmul of depth s_pad)
        //   2. Shield stack + matmul · weight at engine f32 (depth d)
        //   3. Mask unapply Aᵀ·(·) at bf16 (one matmul of depth s_pad)
        // Per-element relative error budget ≈ √(s_pad + d) · 2¯⁷
        // For StandardNormal inputs of magnitude ~1 at s_pad ≈ 32
        // (n=16 + k=8 = 24, padded to 32), d=12:
        //   √(32+12) · 0.0078 ≈ 5.2 %  → ~0.05 absolute at unit scale
        // Empirically observed 0.030 at this shape — within budget.
        // Production-shape tolerance (n ≈ 2048, d ≈ 2560) is the
        // job of the integration parity tests at the embedder layer.
        for ((i, j), &e) in plain_out.indexed_iter() {
            let got = bf16_out[[i, j]];
            let delta = (got - e).abs();
            let scale = e.abs().max(1.0);
            let rel = delta / scale;
            // Hybrid tolerance: absolute 1e-1 covers small-value
            // elements where relative error blows up but absolute is
            // bounded; relative 8 % covers larger-magnitude elements
            // where √(s_pad+d)·2¯⁷ ≈ 5–7 % is the real bf16 floor
            // at this shape (s_pad=32, d=12, with rare ~5σ outliers).
            assert!(
                delta < 1e-1 || rel < 0.08,
                "({i},{j}): plain={e} bf16={got} diff={delta} rel={rel:.4} exceeds bf16-floor"
            );
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

        let mut plain = PlaintextExecutor::new(ReferenceCpuEngine::new());
        plain.provision_weight(handle, weight.view()).unwrap();
        let plain_out = plain.offload_linear(handle, hidden.view()).unwrap();

        let mut hd3 = InProcessTrustedExecutor::with_seed(
            ReferenceCpuEngine::new(),
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

        let mut plain = PlaintextExecutor::new(ReferenceCpuEngine::new());
        plain.provision_weight(handle, weight.view()).unwrap();
        let plain_out = plain.offload_linear(handle, hidden.view()).unwrap();

        let mut hd3 = InProcessTrustedExecutor::with_seed(
            ReferenceCpuEngine::new(),
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
            ReferenceCpuEngine::new(),
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

        let mut plain = PlaintextExecutor::new(ReferenceCpuEngine::new());
        plain.provision_weight(handle, weight.view()).unwrap();
        let plain_out = plain.offload_linear(handle, hidden.view()).unwrap();

        let mut dct = InProcessTrustedExecutor::with_seed(
            ReferenceCpuEngine::new(),
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

        let mut plain = PlaintextExecutor::new(ReferenceCpuEngine::new());
        plain.provision_weight(handle, weight.view()).unwrap();
        let plain_out = plain.offload_linear(handle, hidden.view()).unwrap();

        let mut dct = InProcessTrustedExecutor::with_seed(
            ReferenceCpuEngine::new(),
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
    /// pad-ratio dispatch boundary at the 8/5 = 1.6 threshold
    /// (relaxed from 7/5 = 1.4 on 2026-05-26 after the perf
    /// sweep showed HD₃-forced wins at pad ratios 1.56-1.59;
    /// `docs/plans/gelo-llm-perf-roadmap.md` §1.4).
    #[test]
    fn auto_dispatch_resolves_by_pad_ratio() {
        // Pow2 exact: s_pad/s = 1.0 → HD₃.
        assert_eq!(resolve_mask_kind_for_shape(MaskKind::Auto, 2048), MaskKind::Hd3);
        // Near pow2 (1 row of pad): s_pad/s = 2048/2047 ≈ 1.0005 → HD₃.
        assert_eq!(resolve_mask_kind_for_shape(MaskKind::Auto, 2047), MaskKind::Hd3);
        // s=2055 → s_pad=4096, ratio ≈ 1.99 → DCT-IV (production long-n shape).
        assert_eq!(resolve_mask_kind_for_shape(MaskKind::Auto, 2055), MaskKind::Dct4);
        // s=2561 → s_pad=4096, ratio = 4096/2561 ≈ 1.59 < 8/5 = 1.6 → HD₃.
        // (Sweep cell B=1 n=2561+k=8 confirmed HD₃ wins by 1 % here.)
        // s_pad * 5 = 20480, s * 8 = 20488 → 20480 ≤ 20488 → HD₃.
        assert_eq!(resolve_mask_kind_for_shape(MaskKind::Auto, 2561), MaskKind::Hd3);
        // s=328 → s_pad=512, ratio ≈ 1.56 < 8/5 → HD₃.
        // (Sweep cell B=8 n=320+k=8 confirmed HD₃ wins by 2 % here.)
        // s_pad * 5 = 2560, s * 8 = 2624 → 2560 ≤ 2624 → HD₃.
        assert_eq!(resolve_mask_kind_for_shape(MaskKind::Auto, 328), MaskKind::Hd3);
        // s=2560 → s_pad=4096, ratio = 1.6 exact. 4096*5 = 20480 = 2560*8 → HD₃.
        assert_eq!(resolve_mask_kind_for_shape(MaskKind::Auto, 2560), MaskKind::Hd3);
        // s=2559 → s_pad=4096, ratio just over 1.6. 4096*5 = 20480 > 2559*8 = 20472 → DCT-IV.
        assert_eq!(resolve_mask_kind_for_shape(MaskKind::Auto, 2559), MaskKind::Dct4);
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

        let mut plain = PlaintextExecutor::new(ReferenceCpuEngine::new());
        plain.provision_weight(handle, weight.view()).unwrap();
        let plain_out = plain.offload_linear(handle, hidden.view()).unwrap();

        let mut auto = InProcessTrustedExecutor::with_seed(
            ReferenceCpuEngine::new(),
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

        let mut plain = PlaintextExecutor::new(ReferenceCpuEngine::new());
        plain.provision_weight(handle, weight.view()).unwrap();
        let plain_out = plain.offload_linear(handle, hidden.view()).unwrap();

        let mut auto = InProcessTrustedExecutor::with_seed(
            ReferenceCpuEngine::new(),
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
            ReferenceCpuEngine::new(),
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

    /// **M1.11 R1.2** — batched per-sequence offload_linear round-trips
    /// correctly. Each sub-block of the output must equal the plaintext
    /// reference `hidden_b · W` to f32 floor, with B independent masks
    /// applied internally.
    #[test]
    fn per_sequence_offload_linear_round_trips_to_plaintext() {
        let mut rng = ChaCha20Rng::from_seed([17u8; 32]);
        let normal = StandardNormal;
        let batch_size = 4;
        let n_max = 6;
        let d_in = 5;
        let d_out = 7;

        // Build (B * n_max, d_in) activation by stacking B random sub-blocks.
        let hidden = Array2::<f32>::from_shape_fn(
            (batch_size * n_max, d_in),
            |_| normal.sample(&mut rng),
        );
        let weight = Array2::<f32>::from_shape_fn((d_in, d_out), |_| normal.sample(&mut rng));
        let handle = WeightHandle::new(0, WeightKind::Q);

        let mut exec = InProcessTrustedExecutor::with_seed(
            ReferenceCpuEngine::new(),
            MaskSeed::from_bytes([23u8; 32]),
        );
        exec.provision_weight(handle, weight.view()).unwrap();
        exec.begin_prefill_pass(batch_size, n_max).unwrap();
        let out = exec.offload_linear(handle, hidden.view()).unwrap();
        exec.end_forward_pass().unwrap();

        assert_eq!(out.shape(), &[batch_size * n_max, d_out]);

        // Each block must match the plaintext reference within mask
        // round-trip f32 noise (~5e-3 is the same tolerance as the
        // single-sequence Q/K/V test above).
        for b in 0..batch_size {
            let sub_in = hidden.slice(ndarray::s![b * n_max..(b + 1) * n_max, ..]);
            let expected = sub_in.dot(&weight);
            for ((i, j), e) in expected.indexed_iter() {
                let got = out[[b * n_max + i, j]];
                assert!(
                    (got - e).abs() < 5e-3,
                    "b={b} ({i},{j}): batched={got} expected={e} delta={}",
                    (got - e).abs()
                );
            }
        }
    }

    /// **M1.11 D1.1** — `begin_decode_pass(B)` default path. B per-
    /// sequence A_b each of size (1+k, 1+k) (k=15 from shape-adaptive
    /// overlay at n=1 → stacked_n=16, HD₃-aligned). Per-block round-
    /// trip must match plaintext within f32 mask floor.
    #[test]
    fn begin_decode_pass_default_per_sequence_round_trips() {
        // Ensure the shared-A env var is OFF for this test.
        unsafe {
            std::env::remove_var("BATCHED_DECODE_SHARED_A");
        }
        let mut rng = ChaCha20Rng::from_seed([91u8; 32]);
        let normal = StandardNormal;
        let batch_size = 4;
        let d_in = 6;
        let d_out = 5;

        // (B*1, d_in) — one row per sequence at decode shape.
        let hidden = Array2::<f32>::from_shape_fn(
            (batch_size, d_in),
            |_| normal.sample(&mut rng),
        );
        let weight = Array2::<f32>::from_shape_fn((d_in, d_out), |_| normal.sample(&mut rng));
        let handle = WeightHandle::new(0, WeightKind::Q);

        let mut exec = InProcessTrustedExecutor::with_seed(
            ReferenceCpuEngine::new(),
            MaskSeed::from_bytes([93u8; 32]),
        );
        exec.provision_weight(handle, weight.view()).unwrap();
        exec.begin_decode_pass(batch_size).unwrap();
        let out = exec.offload_linear(handle, hidden.view()).unwrap();
        exec.end_forward_pass().unwrap();

        assert_eq!(out.shape(), &[batch_size, d_out]);
        let expected = hidden.dot(&weight);
        for ((i, j), e) in expected.indexed_iter() {
            let got = out[[i, j]];
            assert!(
                (got - e).abs() < 5e-3,
                "decode-batched per-sequence b={i} dim {j}: got {got} want {e}",
            );
        }
    }

    /// **M1.11 D1.1** — `begin_decode_pass(B)` shared-A path (env-
    /// gated). One Single mask of size (B+k, B+k); mixes B current-
    /// token rows. Same round-trip math at f32 floor.
    #[test]
    fn begin_decode_pass_shared_a_round_trips() {
        // SAFETY: single-threaded test. We restore the env at end.
        unsafe {
            std::env::set_var("BATCHED_DECODE_SHARED_A", "1");
        }
        let mut rng = ChaCha20Rng::from_seed([95u8; 32]);
        let normal = StandardNormal;
        let batch_size = 6;
        let d_in = 4;
        let d_out = 7;

        let hidden = Array2::<f32>::from_shape_fn(
            (batch_size, d_in),
            |_| normal.sample(&mut rng),
        );
        let weight = Array2::<f32>::from_shape_fn((d_in, d_out), |_| normal.sample(&mut rng));
        let handle = WeightHandle::new(0, WeightKind::Q);

        let mut exec = InProcessTrustedExecutor::with_seed(
            ReferenceCpuEngine::new(),
            MaskSeed::from_bytes([97u8; 32]),
        );
        exec.provision_weight(handle, weight.view()).unwrap();
        exec.begin_decode_pass(batch_size).unwrap();
        // Sanity: shared-A path lands in SessionKind::Single.
        assert!(
            matches!(&exec.session, Some(SessionKind::Single(_))),
            "shared-A path must store Single, not PerSequence",
        );
        let out = exec.offload_linear(handle, hidden.view()).unwrap();
        exec.end_forward_pass().unwrap();

        assert_eq!(out.shape(), &[batch_size, d_out]);
        let expected = hidden.dot(&weight);
        for ((i, j), e) in expected.indexed_iter() {
            let got = out[[i, j]];
            assert!(
                (got - e).abs() < 5e-3,
                "decode-batched shared-A b={i} dim {j}: got {got} want {e}",
            );
        }
        // Restore.
        unsafe {
            std::env::remove_var("BATCHED_DECODE_SHARED_A");
        }
    }

    /// Round-trip parity at batch_size=1 — the degenerate case should
    /// match the single-sequence `begin_forward_pass(n)` path to mask
    /// round-trip precision (not bit-identical: different mask
    /// sampling order across the RNG).
    #[test]
    fn per_sequence_offload_linear_b1_matches_single_to_plaintext() {
        let mut rng = ChaCha20Rng::from_seed([29u8; 32]);
        let normal = StandardNormal;
        let n = 8;
        let d_in = 4;
        let d_out = 6;

        let hidden = Array2::<f32>::from_shape_fn((n, d_in), |_| normal.sample(&mut rng));
        let weight = Array2::<f32>::from_shape_fn((d_in, d_out), |_| normal.sample(&mut rng));
        let handle = WeightHandle::new(0, WeightKind::Q);

        let mut exec = InProcessTrustedExecutor::with_seed(
            ReferenceCpuEngine::new(),
            MaskSeed::from_bytes([31u8; 32]),
        );
        exec.provision_weight(handle, weight.view()).unwrap();

        exec.begin_prefill_pass(1, n).unwrap();
        let batched_out = exec.offload_linear(handle, hidden.view()).unwrap();
        exec.end_forward_pass().unwrap();

        let expected = hidden.dot(&weight);
        for ((i, j), e) in expected.indexed_iter() {
            let got = batched_out[[i, j]];
            assert!(
                (got - e).abs() < 5e-3,
                "({i},{j}): batched(B=1)={got} expected={e}",
            );
        }
    }
}

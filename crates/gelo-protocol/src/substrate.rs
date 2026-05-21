use std::sync::Arc;

use anyhow::{Result, anyhow};
use half::bf16;
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Axis};

use crate::ple::PleTable;

/// Identifies a specific projection weight inside a transformer encoder.
/// The trusted side uses this both to address weights on the GPU and to
/// decide whether a given matmul should be offloaded or run locally.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct WeightHandle {
    pub layer: u16,
    pub kind: WeightKind,
}

impl WeightHandle {
    pub const fn new(layer: u16, kind: WeightKind) -> Self {
        Self { layer, kind }
    }
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum WeightKind {
    Q,
    K,
    V,
    O,
    /// SwiGLU "gate" projection. Some BERT-class models lack a gate
    /// (they only have a single FFN up projection); for those, only
    /// `FfnUp` is provisioned and `FfnGate` is unused.
    FfnGate,
    FfnUp,
    FfnDown,
}

/// The untrusted accelerator side of the split protocol.
///
/// All implementations must accept activations through [`Self::matmul`] in
/// whatever form the trusted side supplies them — masked, plaintext, or
/// otherwise. The engine has no notion of correctness verification; integrity
/// is the [`TrustedExecutor`]'s responsibility.
pub trait GpuOffloadEngine: Send {
    /// Provision a public weight tensor to the offload engine. Called once at
    /// model load. Shape is `(in_features, out_features)`.
    fn register_weight(&mut self, handle: WeightHandle, weight: ArrayView2<f32>) -> Result<()>;

    /// Arc-shared variant of [`Self::register_weight`]. When the caller
    /// already owns the weight in an `Arc<Array2<f32>>` (e.g. the
    /// embedder's `DecoderLayerWeights` after the 2026-05-21
    /// Arc-conversion), engines that retain a host-side cache can store
    /// the Arc directly and avoid the 7.5 GB-per-Qwen3-4B clone in
    /// `register_weight`.
    ///
    /// Default impl preserves the legacy behaviour by calling
    /// `register_weight(handle, weight.view())`. Override on engines that
    /// can take Arc ownership — currently `RayonCpuEngine`. The wgpu
    /// engine does not need this override because it uploads weights to
    /// VRAM at registration and never keeps a host copy.
    fn register_weight_shared(
        &mut self,
        handle: WeightHandle,
        weight: std::sync::Arc<ndarray::Array2<f32>>,
    ) -> Result<()> {
        self.register_weight(handle, weight.view())
    }

    /// **bf16-native** weight registration. Engines that can accept
    /// bf16 directly (the wgpu engine in F16 mode, which converts
    /// bf16 → f16 at upload) override this to avoid the loader-side
    /// bf16 → f32 upcast that `register_weight` would otherwise force
    /// — see `feedback_memory_efficiency_priority.md`.
    ///
    /// Default impl converts to f32 in a transient scratch buffer and
    /// forwards to `register_weight`. This preserves correctness for
    /// engines that haven't been updated, but is the path the loader
    /// goes out of its way to avoid: a non-overridden engine on a
    /// Qwen3-1.7B run pays ~3.4 GB of host RAM in the scratch buffer
    /// during provisioning.
    fn register_weight_bf16(
        &mut self,
        handle: WeightHandle,
        weight: ArrayView2<bf16>,
    ) -> Result<()> {
        let f32_owned: Array2<f32> = weight.mapv(|v| v.to_f32());
        self.register_weight(handle, f32_owned.view())
    }

    /// bf16 + Arc-shared. Mirrors [`Self::register_weight_shared`] for
    /// the bf16 storage layout. Overriders should retain the Arc on
    /// engines that hold a host-side cache; the default impl
    /// downcasts via `register_weight_bf16` and drops the Arc.
    fn register_weight_bf16_shared(
        &mut self,
        handle: WeightHandle,
        weight: Arc<Array2<bf16>>,
    ) -> Result<()> {
        self.register_weight_bf16(handle, weight.view())
    }

    /// Compute `input · W[handle]` and return the product.
    ///
    /// `input` has shape `(n, in_features)`; the result has shape
    /// `(n, out_features)`. The engine treats `input` as opaque bytes —
    /// masking is applied by the trusted side before the call.
    fn matmul(&self, handle: WeightHandle, input: ArrayView2<f32>) -> Result<Array2<f32>>;

    /// Compute `input · W[h]` for each `h` in `handles`, sharing **one
    /// upload of `input` and one device sync** across all N matmuls.
    /// Returns the results in the same order as `handles`.
    ///
    /// Required-equivalent shape: each `W[h]` must accept `input`'s second
    /// dim; outputs can differ in their second dim.
    ///
    /// **Why this exists:** the GELO mask round-trip means the trusted side
    /// pays one upload + one sync per offloaded GEMM via `matmul()`. For
    /// `offload_qkv` (3 matmuls sharing the same masked input) that's 2
    /// redundant uploads + 2 redundant syncs per layer — ~24 wasted
    /// CPU↔GPU bounces per BGE-base forward. With lazy-tensor engines
    /// (burn-cubecl) `matmul_many` collapses the redundancy.
    ///
    /// Default impl just loops over `matmul`, so backends without a
    /// lazy-tensor path produce the right answer without the speedup.
    fn matmul_many(
        &self,
        handles: &[WeightHandle],
        input: ArrayView2<f32>,
    ) -> Result<Vec<Array2<f32>>> {
        handles
            .iter()
            .map(|h| self.matmul(*h, input))
            .collect()
    }

    /// Two-operand dynamic matmul where neither operand is a pre-registered
    /// weight. Required by OutAttnMult (TwinShield §V-A): both `Q` and `Kᵀ`
    /// are runtime values, so neither side can be uploaded ahead of time.
    ///
    /// `lhs` has shape `(m, k)`, `rhs` has shape `(k, n)`, result is `(m, n)`.
    fn matmul_dynamic(
        &self,
        lhs: ArrayView2<f32>,
        rhs: ArrayView2<f32>,
    ) -> Result<Array2<f32>>;

    /// Batched two-operand dynamic matmul. `lhs` is `(B, M, K)`, `rhs` is
    /// `(B, K, N)`, result is `(B, M, N)`. Each batch element is an
    /// independent GEMM — no cross-batch reduction. Used by OutAttnMult to
    /// fuse all Q-heads of a layer into one GPU dispatch.
    ///
    /// Default impl loops over the batch axis calling `matmul_dynamic`, so
    /// engines that haven't been upgraded still produce the right answer
    /// (just without the dispatch saving). Wgpu/cubecl backends override
    /// with one batched launch.
    fn matmul_dynamic_batched(
        &self,
        lhs: ArrayView3<f32>,
        rhs: ArrayView3<f32>,
    ) -> Result<Array3<f32>> {
        let b = lhs.shape()[0];
        let m = lhs.shape()[1];
        let k = lhs.shape()[2];
        if rhs.shape()[0] != b || rhs.shape()[1] != k {
            return Err(anyhow::anyhow!(
                "matmul_dynamic_batched shape mismatch: lhs ({b},{m},{k}) vs rhs {:?}",
                rhs.shape()
            ));
        }
        let n = rhs.shape()[2];
        let mut out = Array3::<f32>::zeros((b, m, n));
        for i in 0..b {
            let r = self.matmul_dynamic(
                lhs.index_axis(Axis(0), i),
                rhs.index_axis(Axis(0), i),
            )?;
            out.index_axis_mut(Axis(0), i).assign(&r);
        }
        Ok(out)
    }

    /// Row-wise numerically stable softmax on the last axis of a 3D
    /// tensor. `input` shape `(B, M, N)` → output shape `(B, M, N)`.
    /// Used by the permutation-shielded attention protocol (Tier 1) to
    /// offload softmax onto the engine.
    ///
    /// Default impl computes softmax in-process (on the CPU). The Wgpu
    /// backend overrides with `burn_tensor::activation::softmax` so that
    /// the GPU pipeline of `matmul_dynamic_batched + softmax_batched +
    /// matmul_dynamic_batched` runs entirely on the accelerator with one
    /// device sync at the end.
    fn softmax_batched(&self, input: ArrayView3<f32>) -> Result<Array3<f32>> {
        let (b, m, n) = input.dim();
        let mut out = Array3::<f32>::zeros((b, m, n));
        for bi in 0..b {
            for i in 0..m {
                let row = input.slice(ndarray::s![bi, i, ..]);
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for (j, v) in row.iter().enumerate() {
                    let e = (*v - max).exp();
                    out[(bi, i, j)] = e;
                    sum += e;
                }
                let inv = 1.0 / sum;
                for j in 0..n {
                    out[(bi, i, j)] *= inv;
                }
            }
        }
        Ok(out)
    }

    /// **M1.10 fused permuted attention seam.** Compute the full
    /// per-head causal-masked attention `softmax(scale · Q·Kᵀ + mask) · V`
    /// in one engine call. The `mask` is an additive `(B, n_q, n_kv)`
    /// tensor (`-inf` for blocked positions, `0` elsewhere); the
    /// TEE side bakes the permuted-causal pattern into it before
    /// invoking.
    ///
    /// Default impl composes the existing 3-dispatch path:
    /// `matmul_dynamic_batched + add-mask + softmax_batched +
    /// matmul_dynamic_batched`. Engines that ship a FlashAttention-
    /// style fused kernel (the M1.10 work — see
    /// `docs/plans/path-1-gelo-gemma.md` for the cubek/burn-cubecl
    /// option matrix) override this method so the kernel runs in one
    /// GPU dispatch with no `(B, n, n)` score-tensor materialisation.
    /// Until that override lands, callers get correct math at the
    /// 3-dispatch wall-clock — same as today.
    ///
    /// Shapes:
    ///   q: (B, n_q, d_head)
    ///   k: (B, n_kv, d_head)
    ///   v: (B, n_kv, d_head)
    ///   mask: (B, n_q, n_kv) additive
    ///   → (B, n_q, d_head)
    fn fused_attention_batched(
        &self,
        q: ArrayView3<f32>,
        k: ArrayView3<f32>,
        v: ArrayView3<f32>,
        scale: f32,
        mask: ArrayView3<f32>,
    ) -> Result<Array3<f32>> {
        // Compose: scores = Q · Kᵀ; scaled + masked; softmax; · V.
        let (b, n_q, d_head) = q.dim();
        let n_kv = k.dim().1;
        debug_assert_eq!(q.dim().0, b);
        debug_assert_eq!(k.dim(), (b, n_kv, d_head));
        debug_assert_eq!(v.dim(), (b, n_kv, d_head));
        debug_assert_eq!(mask.dim(), (b, n_q, n_kv));

        // Build (B, d_head, n_kv) for K^T per batch slot.
        let mut kt = Array3::<f32>::zeros((b, d_head, n_kv));
        for bi in 0..b {
            for i in 0..d_head {
                for j in 0..n_kv {
                    kt[(bi, i, j)] = k[(bi, j, i)];
                }
            }
        }

        let mut scores = self.matmul_dynamic_batched(q, kt.view())?;
        for bi in 0..b {
            for i in 0..n_q {
                for j in 0..n_kv {
                    scores[(bi, i, j)] = scores[(bi, i, j)] * scale + mask[(bi, i, j)];
                }
            }
        }
        let probs = self.softmax_batched(scores.view())?;
        self.matmul_dynamic_batched(probs.view(), v)
    }
}

#[cfg(test)]
mod fused_attention_tests {
    //! M1.10a regression: ensure the `fused_attention_batched` default
    //! impl produces the same answer as a hand-rolled reference. When
    //! a fused-kernel override lands (per `docs/plans/path-1-gelo-gemma.md`
    //! M1.10 option A/B/C), the same `mask` semantics must hold.

    use super::*;
    use ndarray::{Array3, ArrayView3};

    /// Minimal engine that just implements `matmul_dynamic` (and
    /// inherits the default impls for the rest, including the new
    /// `fused_attention_batched`).
    struct LocalEngine;
    impl GpuOffloadEngine for LocalEngine {
        fn register_weight(
            &mut self,
            _h: WeightHandle,
            _w: ArrayView2<f32>,
        ) -> Result<()> {
            Ok(())
        }
        fn matmul(
            &self,
            _h: WeightHandle,
            _input: ArrayView2<f32>,
        ) -> Result<Array2<f32>> {
            unimplemented!("not used in fused-attention tests")
        }
        fn matmul_dynamic(
            &self,
            lhs: ArrayView2<f32>,
            rhs: ArrayView2<f32>,
        ) -> Result<Array2<f32>> {
            Ok(lhs.dot(&rhs))
        }
    }

    fn rand_array3(b: usize, m: usize, n: usize, seed: u64) -> Array3<f32> {
        let mut state = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut a = Array3::<f32>::zeros((b, m, n));
        for v in a.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((state >> 33) as f32 / u32::MAX as f32) - 0.5;
        }
        a
    }

    fn ref_attention(
        q: ArrayView3<f32>,
        k: ArrayView3<f32>,
        v: ArrayView3<f32>,
        scale: f32,
        mask: ArrayView3<f32>,
    ) -> Array3<f32> {
        let (b, n_q, d_head) = q.dim();
        let n_kv = k.dim().1;
        let mut out = Array3::<f32>::zeros((b, n_q, d_head));
        for bi in 0..b {
            // scores: (n_q, n_kv)
            let mut scores = ndarray::Array2::<f32>::zeros((n_q, n_kv));
            for i in 0..n_q {
                for j in 0..n_kv {
                    let mut s = 0.0_f32;
                    for d in 0..d_head {
                        s += q[(bi, i, d)] * k[(bi, j, d)];
                    }
                    scores[[i, j]] = s * scale + mask[(bi, i, j)];
                }
            }
            // softmax + multiply by V
            for i in 0..n_q {
                let row = scores.row(i);
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0_f32;
                let mut exps = vec![0.0_f32; n_kv];
                for j in 0..n_kv {
                    exps[j] = (row[j] - max).exp();
                    sum += exps[j];
                }
                let inv = 1.0 / sum;
                for d in 0..d_head {
                    let mut s = 0.0_f32;
                    for j in 0..n_kv {
                        s += exps[j] * inv * v[(bi, j, d)];
                    }
                    out[(bi, i, d)] = s;
                }
            }
        }
        out
    }

    #[test]
    fn default_impl_matches_reference_with_no_mask() {
        let b = 2;
        let n_q = 4;
        let n_kv = 4;
        let d_head = 8;
        let q = rand_array3(b, n_q, d_head, 1);
        let k = rand_array3(b, n_kv, d_head, 2);
        let v = rand_array3(b, n_kv, d_head, 3);
        let mask = Array3::<f32>::zeros((b, n_q, n_kv));
        let scale = 1.0 / (d_head as f32).sqrt();

        let engine = LocalEngine;
        let got = engine
            .fused_attention_batched(q.view(), k.view(), v.view(), scale, mask.view())
            .unwrap();
        let want = ref_attention(q.view(), k.view(), v.view(), scale, mask.view());
        for (a, b) in got.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn default_impl_honours_additive_causal_mask() {
        // Build a strict lower-triangular causal mask: 0 for j ≤ i,
        // -inf for j > i. Verify the result matches the reference.
        let b = 1;
        let n = 5;
        let d_head = 4;
        let q = rand_array3(b, n, d_head, 11);
        let k = rand_array3(b, n, d_head, 12);
        let v = rand_array3(b, n, d_head, 13);
        let mut mask = Array3::<f32>::zeros((b, n, n));
        for i in 0..n {
            for j in (i + 1)..n {
                mask[(0, i, j)] = f32::NEG_INFINITY;
            }
        }
        let scale = 1.0 / (d_head as f32).sqrt();

        let engine = LocalEngine;
        let got = engine
            .fused_attention_batched(q.view(), k.view(), v.view(), scale, mask.view())
            .unwrap();
        let want = ref_attention(q.view(), k.view(), v.view(), scale, mask.view());
        for (a, b) in got.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}

/// The trusted side of the split protocol.
///
/// Implementations own the mask RNG, perform the `A·H` / `Aᵀ·(U·W)`
/// dance, and decide which projections to offload vs. run locally
/// (e.g. for the sensitive first/last layers per GELO §3.2).
pub trait TrustedExecutor {
    /// Hand a public weight to the offload engine. Called at model load.
    fn provision_weight(&mut self, handle: WeightHandle, weight: ArrayView2<f32>) -> Result<()>;

    /// Begin a forward pass for a single text of token-axis length `n`.
    ///
    /// Implementations running in **paper-parity mode** (one mask A per
    /// forward pass, per GELO §3.2) sample a fresh Haar-uniform `A` of
    /// size `(n + shield_k, n + shield_k)` here and reuse it across every
    /// subsequent `offload_*` call until the matching [`end_forward_pass`].
    ///
    /// Implementations in **per-offload mode** (sample fresh A inside
    /// every `offload_*`) treat this as a no-op — the default impl does
    /// exactly that, so `PlaintextExecutor` and other backends that don't
    /// care about session lifecycle keep working unchanged.
    ///
    /// Embedders **MUST** call this at the start of each per-text
    /// forward pass and call [`end_forward_pass`] before the next, even
    /// if the executor is in per-offload mode (the no-op default makes
    /// this cheap). This way the embedder code is engine-agnostic.
    fn begin_forward_pass(&mut self, _n: usize) -> Result<()> {
        Ok(())
    }

    /// End the current forward pass. Frees the session mask (if any)
    /// and returns the executor to idle state. Default impl is no-op.
    fn end_forward_pass(&mut self) -> Result<()> {
        Ok(())
    }

    /// Move this executor's randomness source to an independent
    /// stream. Used by the embedder's rayon-parallel `embed` path so
    /// each worker in a batch gets its own mask `A` — without this,
    /// every worker would share the cloned executor's RNG state and
    /// sample the same `A`, exposing the cross-text Gram leak (see
    /// `docs/prototype/future-rnd.md` §5 "Shared-A multi-text
    /// batching").
    ///
    /// Default impl is no-op: executors that don't sample randomness
    /// (e.g. `PlaintextExecutor`) or that derive their `A` from
    /// elsewhere just ignore the call.
    fn set_rng_stream(&mut self, _stream: u64) {}

    /// Same as [`Self::provision_weight`] but accepts an `Arc<Array2<f32>>` so
    /// the executor's TEE-side weight cache (for U-Verify probe computation)
    /// can share storage with the embedder's loaded weight bytes instead of
    /// cloning them. GELO targets openweight models — weight confidentiality
    /// is not a goal — so the only reason the executor used to keep a
    /// separate copy was to have the bytes in encrypted CVM RAM. With this
    /// API the embedder's existing `Arc<DecoderWeights>` shards are reused
    /// directly, halving the encrypted memory footprint on Qwen3-class
    /// models (−2.4 GB).
    ///
    /// Default impl falls back to the cloning path so existing callers keep
    /// working unchanged.
    fn provision_weight_shared(
        &mut self,
        handle: WeightHandle,
        weight: Arc<Array2<f32>>,
    ) -> Result<()> {
        self.provision_weight(handle, weight.view())
    }

    /// **bf16-native** weight provisioning. The trusted side does not
    /// need a host f32 copy of offloadable projection weights —
    /// activations are masked f32 but weights live on the engine
    /// (GPU) at f16. Loader stores bf16 to avoid the
    /// bf16 → f32 widening called out in
    /// `feedback_memory_efficiency_priority.md`.
    ///
    /// Default impl forwards to `register_weight_bf16` on the engine
    /// and skips the U-Verify cache (verify_probes is gated on the
    /// f32 path; bf16 + U-Verify is not supported in v1).
    fn provision_weight_bf16(
        &mut self,
        handle: WeightHandle,
        weight: ArrayView2<bf16>,
    ) -> Result<()>;

    /// bf16 + Arc-shared variant. The loader's `Arc<Array2<bf16>>`
    /// flows straight through to the engine; for the wgpu engine in
    /// F16 mode the Arc is consumed during upload and the host bytes
    /// drop when the Arc refcount hits zero.
    fn provision_weight_bf16_shared(
        &mut self,
        handle: WeightHandle,
        weight: Arc<Array2<bf16>>,
    ) -> Result<()> {
        self.provision_weight_bf16(handle, weight.view())
    }

    /// Provision a Per-Layer Embedding (PLE) table into the trusted
    /// side's encrypted memory. The table is owned by the executor
    /// (and shared via `Arc` across clones); it is **never** handed to
    /// the offload engine — that would defeat the round-2 P0 leak
    /// fix described in `docs/prototype/gelo-llm.html` §03. Gemma 3n /
    /// Gemma 4 callers invoke this once at model load alongside
    /// `provision_weight` for the standard offload weights.
    ///
    /// Default impl rejects the call — executors that don't support
    /// PLE either have no need for it (Qwen3 embed/rerank paths) or
    /// would be loading the table into the wrong memory region.
    /// Hybrid models should fail loud rather than silently fall back
    /// to a leaky path, hence the error rather than a no-op.
    fn provision_ple_table(&mut self, _table: PleTable) -> Result<()> {
        Err(anyhow!(
            "TrustedExecutor: provision_ple_table not implemented for this executor",
        ))
    }

    /// Gather `(n, d_ple)` f32 rows from the provisioned PLE table at
    /// layer `layer_idx`, one row per `token_id`. Errors when no PLE
    /// table is provisioned, when the layer index is out of range, or
    /// when any token_id exceeds the table's vocab.
    ///
    /// The gather happens entirely inside the trusted executor — no
    /// engine round-trip, no PCIe traffic. A spy engine observing the
    /// offload path sees zero PLE-keyed activity.
    ///
    /// Default impl rejects the call for the same reason as
    /// `provision_ple_table`: silent fallback would mask a leak.
    fn ple_gather(&self, _token_ids: &[u32], _layer_idx: usize) -> Result<Array2<f32>> {
        Err(anyhow!(
            "TrustedExecutor: ple_gather called without a provisioned PLE table",
        ))
    }

    /// Run a single offloaded linear: mask `hidden` on the token axis, ship
    /// to the engine, unmask, return `hidden · W[handle]`.
    ///
    /// `hidden` shape is `(n, in_features)`; result shape is
    /// `(n, out_features)`.
    fn offload_linear(
        &mut self,
        handle: WeightHandle,
        hidden: ArrayView2<f32>,
    ) -> Result<Array2<f32>>;

    /// Offload Q, K, V projections in one shot, sharing a single fresh mask
    /// across all three. This is the optimization called out in GELO §3:
    /// reusing `A` across the three reads of the same hidden state saves
    /// two mask samples per block without leaking additional information.
    fn offload_qkv(
        &mut self,
        layer: u16,
        hidden: ArrayView2<f32>,
    ) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>)> {
        let q = self.offload_linear(WeightHandle::new(layer, WeightKind::Q), hidden)?;
        let k = self.offload_linear(WeightHandle::new(layer, WeightKind::K), hidden)?;
        let v = self.offload_linear(WeightHandle::new(layer, WeightKind::V), hidden)?;
        Ok((q, k, v))
    }

    /// Offload several linear projections that all read the **same** hidden
    /// state, sharing a single mask apply + a single batched matmul + a
    /// single batched unapply. The canonical caller is the SwiGLU FFN
    /// (gate + up share `h_norm_ffn`); the QKV path is hand-written for
    /// historical reasons and uses an equivalent shape internally.
    ///
    /// Result order matches `handles` order. Each output's column dim is
    /// determined by the corresponding weight's `out_features`.
    ///
    /// Default impl loops over `offload_linear` so executors that don't
    /// override (e.g. `PlaintextExecutor`) still produce correct results.
    fn offload_linear_many(
        &mut self,
        handles: &[WeightHandle],
        hidden: ArrayView2<f32>,
    ) -> Result<Vec<Array2<f32>>> {
        handles
            .iter()
            .map(|h| self.offload_linear(*h, hidden))
            .collect()
    }

    /// Offload the attention `Q · Kᵀ` matmul via the TwinShield OutAttnMult
    /// 4-partition embedding (Xue et al. 2025 §V-A). Both `q` and `kt` are
    /// runtime values; the trick lets the untrusted engine compute the
    /// product without recovering either operand.
    ///
    /// `q` shape: `(n, d_head)`, `kt` shape: `(d_head, n)`, result `(n, n)`.
    ///
    /// Default impl just calls the engine's `matmul_dynamic` directly with
    /// no masking, suitable for `PlaintextExecutor` parity baselines.
    fn offload_attention_qkt(
        &mut self,
        _q: ArrayView2<f32>,
        _kt: ArrayView2<f32>,
    ) -> Result<Array2<f32>> {
        unimplemented!("offload_attention_qkt not implemented for this executor")
    }

    /// Batched OutAttnMult — one GPU dispatch covering every Q head in a
    /// layer. `q` is `(num_q_heads, n, d_head)`, `kt` is
    /// `(num_q_heads, d_head, n)` (with K already repeated to match Q heads
    /// for GQA), result is `(num_q_heads, n, n)`.
    ///
    /// Each head gets independent masks, scalars, and permutations — the
    /// privacy story stays identical to the per-head form. Default impl
    /// loops over the batch axis calling `offload_attention_qkt`; engines
    /// implementing the batched engine primitive will override.
    fn offload_attention_qkt_batched(
        &mut self,
        q: ArrayView3<f32>,
        kt: ArrayView3<f32>,
    ) -> Result<Array3<f32>> {
        let h = q.shape()[0];
        let n = q.shape()[1];
        let mut out = Array3::<f32>::zeros((h, n, n));
        for i in 0..h {
            let r = self.offload_attention_qkt(
                q.index_axis(Axis(0), i),
                kt.index_axis(Axis(0), i),
            )?;
            out.index_axis_mut(Axis(0), i).assign(&r);
        }
        Ok(out)
    }

    /// Compute `softmax(Q·Kᵀ / √d + mask) · V` for every head, under the
    /// permutation-shielded attention protocol (Tier 1 — Amulet's
    /// softmax-permutation equivariance, arXiv 2512.07495, combined with
    /// Hidden No More's σ-noise mitigation, arXiv 2505.18332).
    ///
    /// Unlike [`Self::offload_attention_qkt`] (which returns just the
    /// pre-softmax scores), this returns the **full attention output** —
    /// softmax and `·V` are performed under the same per-batch permutation
    /// so neither operand is observable to the untrusted side.
    ///
    /// `q`, `k`, `v` shape: `(num_heads, n, d_head)`. Result shape:
    /// `(num_heads, n, d_head)`. `scale` is typically `1 / √d_head`.
    /// `mask` selects between full bidirectional and causal attention.
    ///
    /// Default impl falls back to **plain** multi-head attention — useful
    /// only as a parity baseline (no privacy). Real implementations override.
    fn offload_attention_permuted(
        &mut self,
        q: ArrayView3<f32>,
        k: ArrayView3<f32>,
        v: ArrayView3<f32>,
        scale: f32,
        mask: crate::attention::AttentionMask,
    ) -> Result<Array3<f32>> {
        // Default: plain multi-head attention. Used by PlaintextExecutor
        // and as a fallback for executors that haven't been upgraded.
        let (h, n, _d) = q.dim();
        let mut out = Array3::<f32>::zeros((h, n, q.shape()[2]));
        for i in 0..h {
            let qh = q.index_axis(Axis(0), i);
            let kh = k.index_axis(Axis(0), i);
            let vh = v.index_axis(Axis(0), i);
            let mut scores = qh.dot(&kh.t());
            scores.mapv_inplace(|x| x * scale);
            // Causal mask: -inf on the strict upper triangle.
            if let crate::attention::AttentionMask::Causal = mask {
                for r in 0..scores.nrows() {
                    for c in (r + 1)..scores.ncols() {
                        scores[(r, c)] = f32::NEG_INFINITY;
                    }
                }
            }
            // Numerically stable softmax row-wise.
            let mut probs = Array2::<f32>::zeros(scores.dim());
            for r in 0..scores.nrows() {
                let row = scores.row(r);
                let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut s = 0.0f32;
                for (c, v) in row.iter().enumerate() {
                    let e = (*v - m).exp();
                    probs[(r, c)] = e;
                    s += e;
                }
                if s > 0.0 {
                    let inv = 1.0 / s;
                    for c in 0..probs.ncols() {
                        probs[(r, c)] *= inv;
                    }
                }
            }
            out.index_axis_mut(Axis(0), i).assign(&probs.dot(&vh));
        }
        Ok(out)
    }

    /// Cached-KV variant of [`Self::offload_attention_permuted`] for
    /// the autoregressive generation shape. Same protocol semantics
    /// (Amulet softmax-equivariance under fresh per-call permutations
    /// + Hidden-No-More σ-noise) but allows `n_q ≤ n_kv` for the
    /// decode / continuation-prefill case.
    ///
    /// `q_pos_offset` is the absolute position of Q row 0 in the
    /// full sequence. Q row `i` is at absolute position
    /// `q_pos_offset + i` and may attend to K rows `0..=(q_pos_offset
    /// + i)`. For decode (`n_q = 1`, `q_pos_offset = n_kv − 1`) the
    /// causal mask is a no-op.
    ///
    /// Shapes:
    ///   q: `(num_heads, n_q,  d_head)`
    ///   k: `(num_heads, n_kv, d_head)`
    ///   v: `(num_heads, n_kv, d_head)`
    ///   → `(num_heads, n_q,  d_head)`
    ///
    /// Default impl falls back to **plain** asymmetric multi-head
    /// attention with explicit `-inf` causal mask — no privacy.
    /// Real implementations override to call
    /// `crate::attention::permuted_attention_cached` under the
    /// executor's fresh-per-call RNG.
    fn offload_attention_permuted_cached(
        &mut self,
        q: ArrayView3<f32>,
        k: ArrayView3<f32>,
        v: ArrayView3<f32>,
        scale: f32,
        q_pos_offset: usize,
        mask: crate::attention::AttentionMask,
    ) -> Result<Array3<f32>> {
        let (h, n_q, d_head) = q.dim();
        let n_kv = k.dim().1;
        if n_q > n_kv {
            return Err(anyhow!(
                "offload_attention_permuted_cached: n_q ({n_q}) > n_kv ({n_kv})"
            ));
        }
        if q_pos_offset + n_q > n_kv {
            return Err(anyhow!(
                "offload_attention_permuted_cached: q_pos_offset ({q_pos_offset}) + \
                 n_q ({n_q}) must be ≤ n_kv ({n_kv})"
            ));
        }
        let mut out = Array3::<f32>::zeros((h, n_q, d_head));
        for hi in 0..h {
            let qh = q.index_axis(Axis(0), hi);
            let kh = k.index_axis(Axis(0), hi);
            let vh = v.index_axis(Axis(0), hi);
            let mut scores = qh.dot(&kh.t());
            scores.mapv_inplace(|x| x * scale);
            if let crate::attention::AttentionMask::Causal = mask {
                for i in 0..n_q {
                    let q_abs = q_pos_offset + i;
                    for j in 0..n_kv {
                        if j > q_abs {
                            scores[(i, j)] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
            let mut probs = Array2::<f32>::zeros(scores.dim());
            for r in 0..scores.nrows() {
                let row = scores.row(r);
                let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut s = 0.0f32;
                for (c, v) in row.iter().enumerate() {
                    let e = (*v - m).exp();
                    probs[(r, c)] = e;
                    s += e;
                }
                if s > 0.0 {
                    let inv = 1.0 / s;
                    for c in 0..probs.ncols() {
                        probs[(r, c)] *= inv;
                    }
                }
            }
            out.index_axis_mut(Axis(0), hi).assign(&probs.dot(&vh));
        }
        Ok(out)
    }
}

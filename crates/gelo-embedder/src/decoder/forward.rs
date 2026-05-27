use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2, Array3, ArrayView2};

use gelo_protocol::profile;
use gelo_protocol::tee_matmul_bf16;
use gelo_protocol::{ForwardSessionShape, TrustedExecutor, WeightHandle, WeightKind};

use super::attention::{
    causal_gqa_attention, causal_gqa_attention_cached, causal_gqa_attention_permuted,
    causal_gqa_attention_permuted_cached, causal_gqa_attention_swa_cached,
    causal_gqa_attention_with_offload,
};
use super::config::{AttentionClass, DecoderConfig};
use super::kv_cache::KvCache;
use super::rms_norm::{apply_qk_norm, rms_norm};
use super::rope::RopeTables;
use super::swiglu::swiglu;
use super::weights::{DecoderLayerWeights, DecoderWeights};

/// Run a Qwen3-style decoder embedder forward pass under the GELO protocol.
///
/// `input_ids` is a flat `[seq_len]` slice. Returns the per-token hidden
/// state matrix `(seq_len, hidden_size)` after the final RMSNorm. The caller
/// applies last-token pooling + L2 normalize.
pub fn run(
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    rope: &RopeTables,
    exec: &mut impl TrustedExecutor,
    input_ids: &[u32],
) -> Result<Array2<f32>> {
    run_with_hook(cfg, weights, rope, exec, input_ids, |_, _| {})
}

/// Same as [`run`] but invokes `after_layer(layer_idx, &mut h)` after the
/// residual stream output of each transformer block (before the next
/// layer's input). The hook is a general per-layer instrumentation point.
pub fn run_with_hook<F: FnMut(usize, &mut Array2<f32>)>(
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    rope: &RopeTables,
    exec: &mut impl TrustedExecutor,
    input_ids: &[u32],
    mut after_layer: F,
) -> Result<Array2<f32>> {
    let n = input_ids.len();
    let mut h = profile::time("tee:embed_lookup", || {
        embedding_lookup(cfg, weights, input_ids)
    });

    // GELO paper §3.2 forward-pass session: see bert/forward.rs for the
    // rationale. Paper-parity executors sample one mask here; per-offload
    // executors and PlaintextExecutor treat this as a no-op.
    with_forward_session(exec, ForwardSessionShape::Single { n }, |exec| {
        for (li, layer) in weights.layers.iter().enumerate() {
            h = decoder_block(
                cfg,
                layer,
                rope,
                exec,
                li as u16,
                h.view(),
                cfg.offload_layer(li),
            )?;
            after_layer(li, &mut h);
        }
        Ok(profile::time("tee:rmsnorm", || {
            rms_norm(
                h.view(),
                weights.final_norm.as_slice().unwrap(),
                cfg.rms_norm_eps,
            )
        }))
    })
}

fn embedding_lookup(cfg: &DecoderConfig, w: &DecoderWeights, ids: &[u32]) -> Array2<f32> {
    let n = ids.len();
    let d = cfg.hidden_size;
    let mut out = Array2::<f32>::zeros((n, d));
    for (i, &id) in ids.iter().enumerate() {
        // bf16 → f32 widening per element. No intermediate row alloc.
        let row = w.token_embedding.row(id as usize);
        for (j, &v) in row.iter().enumerate() {
            out[(i, j)] = v.to_f32();
        }
    }
    out
}

fn with_forward_session<X, T, F>(exec: &mut X, shape: ForwardSessionShape, body: F) -> Result<T>
where
    X: TrustedExecutor,
    F: FnOnce(&mut X) -> Result<T>,
{
    exec.begin_forward_session(shape)?;
    let result = body(exec);
    let end = exec.end_forward_pass();
    match (result, end) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(err), Ok(())) => Err(err),
        (Ok(_), Err(err)) => Err(err),
        (Err(err), Err(_end_err)) => Err(err),
    }
}

/// Prefill — run a full prompt forward and populate the KV cache.
///
/// Returns the per-token hidden state matrix `(n_prompt, hidden_size)`
/// after the final RMSNorm. Caller takes the last row for next-token
/// sampling and re-uses the populated `kv_cache` for subsequent
/// [`run_decode_step`] calls.
///
/// Equivalent to [`run`] for one-shot embedding, except K and V are
/// preserved in `kv_cache` for autoregressive continuation. The
/// protocol-level forward-pass bracket (one fresh Haar `A`) covers the
/// full prefill in a single call — same property as [`run`].
pub fn run_prefill(
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    rope: &RopeTables,
    exec: &mut impl TrustedExecutor,
    input_ids: &[u32],
    kv_cache: &mut KvCache,
) -> Result<Array2<f32>> {
    assert_eq!(
        kv_cache.num_layers(),
        weights.layers.len(),
        "kv_cache layer count must match model layer count",
    );
    assert_eq!(
        kv_cache.kv_dim(),
        cfg.kv_dim(),
        "kv_cache kv_dim must match cfg.kv_dim()",
    );
    let n = input_ids.len();
    let q_pos_offset = kv_cache.len();
    assert!(
        q_pos_offset + n <= kv_cache.capacity(),
        "prefill would overflow kv_cache: {} + {} > {}",
        q_pos_offset,
        n,
        kv_cache.capacity(),
    );

    let mut h = profile::time("tee:embed_lookup", || {
        embedding_lookup(cfg, weights, input_ids)
    });

    with_forward_session(exec, ForwardSessionShape::Single { n }, |exec| {
        for (li, layer) in weights.layers.iter().enumerate() {
            h = decoder_block_cached(
                cfg,
                layer,
                rope,
                exec,
                li as u16,
                h.view(),
                cfg.offload_layer(li),
                kv_cache,
                q_pos_offset,
            )?;
        }
        Ok(profile::time("tee:rmsnorm", || {
            rms_norm(
                h.view(),
                weights.final_norm.as_slice().unwrap(),
                cfg.rms_norm_eps,
            )
        }))
    })
}

/// Decode one token — append its K/V to the cache, return the
/// resulting last-layer hidden state row `(hidden_size,)`.
///
/// `token_id` is the token whose embedding becomes the single-row input
/// to this step. The caller is responsible for the prefill phase having
/// populated `kv_cache` for positions `0..kv_cache.len()`; this
/// function appends one position at `kv_cache.len()` to every layer's
/// cache. The protocol-level forward-pass bracket fires once per
/// decode step — one fresh Haar `A` per token, per the locked design
/// decision.
pub fn run_decode_step(
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    rope: &RopeTables,
    exec: &mut impl TrustedExecutor,
    token_id: u32,
    kv_cache: &mut KvCache,
) -> Result<Array1<f32>> {
    assert_eq!(
        kv_cache.num_layers(),
        weights.layers.len(),
        "kv_cache layer count must match model layer count",
    );
    assert_eq!(
        kv_cache.kv_dim(),
        cfg.kv_dim(),
        "kv_cache kv_dim must match cfg.kv_dim()",
    );
    let q_pos_offset = kv_cache.len();
    assert!(
        q_pos_offset + 1 <= kv_cache.capacity(),
        "decode would overflow kv_cache: {} + 1 > {}",
        q_pos_offset,
        kv_cache.capacity(),
    );

    let mut h = profile::time("tee:embed_lookup", || {
        embedding_lookup(cfg, weights, &[token_id])
    });

    with_forward_session(exec, ForwardSessionShape::Single { n: 1 }, |exec| {
        for (li, layer) in weights.layers.iter().enumerate() {
            h = decoder_block_cached(
                cfg,
                layer,
                rope,
                exec,
                li as u16,
                h.view(),
                cfg.offload_layer(li),
                kv_cache,
                q_pos_offset,
            )?;
        }
        let normed = profile::time("tee:rmsnorm", || {
            rms_norm(
                h.view(),
                weights.final_norm.as_slice().unwrap(),
                cfg.rms_norm_eps,
            )
        });
        Ok(normed.row(0).to_owned())
    })
}

/// **M1.11 D1.3** — Batched decode-step forward over B sequences.
///
/// `token_ids` has length B — sequence `b` contributes `token_ids[b]`
/// as its next input row. Each sequence's KV cache is appended one
/// position at the sequence's current `kv_cache.len_b(li, b)`. Returns
/// the final hidden states `(B, hidden)` — caller gathers per-sequence
/// rows for logit computation.
///
/// Prefill must already have populated `kv_cache` per-sequence via
/// `forward::run_batched` (or a future `run_prefill_batched` if we
/// split it out). Each call here advances every sequence by exactly
/// one token.
///
/// Per the M1.11 plan §3.4: under `BATCHED_DECODE_SHARED_A=1` the
/// shielded operand at each layer goes through one shared dense mask
/// `(B+k, B+k)`; default is per-sequence `A_b` of size `(1+k, 1+k)`.
/// The executor's `begin_decode_pass(B)` handles the topology switch.
pub fn run_decode_step_batched(
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    rope: &RopeTables,
    exec: &mut impl TrustedExecutor,
    token_ids: &[u32],
    kv_cache: &mut KvCache,
) -> Result<Array2<f32>> {
    let batch_size = token_ids.len();
    assert!(batch_size > 0, "run_decode_step_batched: token_ids empty");
    assert_eq!(
        batch_size,
        kv_cache.batch_size(),
        "token_ids len {} != kv_cache batch_size {}",
        batch_size,
        kv_cache.batch_size(),
    );
    assert_eq!(
        kv_cache.num_layers(),
        weights.layers.len(),
        "kv_cache layer count must match model layer count",
    );
    assert_eq!(
        kv_cache.kv_dim(),
        cfg.kv_dim(),
        "kv_cache kv_dim must match cfg.kv_dim()",
    );
    // Per-sequence position offset: each sequence's row sits at its
    // own absolute position `kv_cache.len_b(0, b)`. Layer 0 is the
    // representative — debug-asserts in `lens()` catch cross-layer
    // divergence.
    let q_pos_offsets: Vec<usize> = (0..batch_size).map(|b| kv_cache.len_b(0, b)).collect();
    for (b, off) in q_pos_offsets.iter().enumerate() {
        assert!(
            off + 1 <= kv_cache.capacity(),
            "decode would overflow kv_cache: sequence {b} cur {off} + 1 > {}",
            kv_cache.capacity(),
        );
    }

    // (B, hidden) — one embedding row per sequence.
    let mut h = profile::time("tee:embed_lookup", || {
        embedding_lookup(cfg, weights, token_ids)
    });

    with_forward_session(
        exec,
        ForwardSessionShape::DecodeBatch { batch_size },
        |exec| {
            for (li, layer) in weights.layers.iter().enumerate() {
                h = decoder_block_cached_batched(
                    cfg,
                    layer,
                    rope,
                    exec,
                    li as u16,
                    h.view(),
                    cfg.offload_layer(li),
                    kv_cache,
                    &q_pos_offsets,
                )?;
            }
            Ok(profile::time("tee:rmsnorm", || {
                rms_norm(
                    h.view(),
                    weights.final_norm.as_slice().unwrap(),
                    cfg.rms_norm_eps,
                )
            }))
        },
    )
}

/// **M1.11 D1.3** — Batched cache-aware decoder block. Mirror of
/// [`decoder_block_cached`] for `B` parallel sequences each with their
/// own KV cache prefix length. Differences from the single-sequence
/// version:
///
/// 1. `hidden` shape is `(B, hidden)` — one new token row per sequence.
/// 2. `q_pos_offsets[b]` is per-sequence; RoPE applies row `b`'s
///    rotation at its own absolute position.
/// 3. KV cache append uses `append_decode` (one row per sequence,
///    placed at each sequence's current `lens[b]`).
/// 4. Attention runs per-sequence in-TEE over each sequence's full
///    cached prefix (`tee:attn_cached_inplace_many` — same stopgap
///    pattern as `decoder_block_batched`'s prefill attention). R1.4
///    would replace this with a single batched-kernel dispatch.
#[allow(clippy::too_many_arguments)]
fn decoder_block_cached_batched(
    cfg: &DecoderConfig,
    layer: &DecoderLayerWeights,
    rope: &RopeTables,
    exec: &mut impl TrustedExecutor,
    layer_idx: u16,
    hidden: ArrayView2<'_, f32>,
    offload: bool,
    kv_cache: &mut KvCache,
    q_pos_offsets: &[usize],
) -> Result<Array2<f32>> {
    let batch_size = hidden.nrows();
    debug_assert_eq!(batch_size, q_pos_offsets.len());

    let h_norm = profile::time("tee:rmsnorm", || {
        rms_norm(
            hidden,
            layer.norm_attn.as_slice().unwrap(),
            cfg.rms_norm_eps,
        )
    });

    // Q/K/V: hidden has shape (B, hidden_size). Under
    // SessionKind::PerSequence (default decode) or Single (shared-A
    // opt-in), offload_qkv produces (B, q_dim/kv_dim) outputs.
    let (mut q, mut k, v) = if offload {
        exec.offload_qkv(layer_idx, h_norm.view())?
    } else {
        profile::time("tee:qkv_direct", || {
            (
                tee_matmul_bf16(
                    h_norm.view(),
                    layer
                        .wq
                        .as_ref()
                        .expect("offload=false requires layer.wq")
                        .view(),
                ),
                tee_matmul_bf16(
                    h_norm.view(),
                    layer
                        .wk
                        .as_ref()
                        .expect("offload=false requires layer.wk")
                        .view(),
                ),
                tee_matmul_bf16(
                    h_norm.view(),
                    layer
                        .wv
                        .as_ref()
                        .expect("offload=false requires layer.wv")
                        .view(),
                ),
            )
        })
    };

    profile::time("tee:qk_norm", || {
        if let Some(q_gamma) = layer.q_norm.as_ref() {
            apply_qk_norm(
                q.view_mut(),
                cfg.num_attention_heads,
                cfg.head_dim_value(),
                q_gamma.as_slice().expect("q_norm contiguous"),
                cfg.rms_norm_eps,
            );
        }
        if let Some(k_gamma) = layer.k_norm.as_ref() {
            apply_qk_norm(
                k.view_mut(),
                cfg.num_key_value_heads,
                cfg.head_dim_value(),
                k_gamma.as_slice().expect("k_norm contiguous"),
                cfg.rms_norm_eps,
            );
        }
    });

    // RoPE: row `b` sits at absolute position `q_pos_offsets[b]`.
    // Apply rotation per-row by slicing the single row out, calling
    // apply_partial_at with start_pos = q_pos_offsets[b], and writing
    // back.
    //
    // `rotated_dim` follows the existing decoder_block_cached logic
    // for attention class. At decode the layer class is consistent
    // across all sequences (it's a property of the layer, not the
    // sequence), so resolve once outside the per-b loop.
    let layer_class = cfg.effective_attention_class(layer_idx as usize);
    let rotated_dim = match (layer_class, cfg.partial_rope) {
        (AttentionClass::Global, Some(_)) => cfg.rotated_dim(),
        _ => cfg.head_dim_value(),
    };
    profile::time("tee:rope", || {
        for b in 0..batch_size {
            let pos = q_pos_offsets[b];
            rope.apply_partial_at(
                q.slice_mut(ndarray::s![b..b + 1, ..]),
                cfg.num_attention_heads,
                pos,
                rotated_dim,
            );
            rope.apply_partial_at(
                k.slice_mut(ndarray::s![b..b + 1, ..]),
                cfg.num_key_value_heads,
                pos,
                rotated_dim,
            );
        }
    });

    // K=V sharing (Gemma 4 global): re-derive V from K post-RoPE so
    // that the V tensor stays identical to K after rotation.
    let kv_shared = cfg.kv_shared_in_global && matches!(layer_class, AttentionClass::Global);
    let v = if kv_shared { k.clone() } else { v };

    // Append one row per sequence to the KV cache.
    kv_cache.append_decode(layer_idx as usize, k.view(), v.view())?;

    // Per-sequence in-TEE causal attention over the full cached
    // prefix. n_q = 1 per sequence, n_kv = lens[b]. R1.4 stopgap as
    // in decoder_block_batched.
    //
    // D1.8: rayon-parallel over B. At n_q = 1 the inner attention
    // kernel (`causal_gqa_attention_cached` / `_swa_cached`) runs
    // serial over heads (its internal rayon threshold is n_q ≥ 64),
    // so an outer rayon-iter over the B sequences gives clean
    // B-parallel attention with no nested-rayon contention. We
    // pre-collect `kv_cache.view_b` views outside the parallel
    // section to keep Result plumbing out of the closure.
    let q_dim = cfg.num_attention_heads * cfg.head_dim_value();
    let mut ctx = Array2::<f32>::zeros((batch_size, q_dim));
    let kv_views: Vec<(ArrayView2<'_, f32>, ArrayView2<'_, f32>)> = (0..batch_size)
        .map(|b| kv_cache.view_b(layer_idx as usize, b))
        .collect::<Result<Vec<_>>>()?;
    profile::time("tee:attn_cached_inplace_many", || {
        use ndarray::parallel::prelude::*;
        ctx.axis_chunks_iter_mut(ndarray::Axis(0), 1)
            .into_par_iter()
            .enumerate()
            .for_each(|(b, mut ctx_slice)| {
                let q_b = q.slice(ndarray::s![b..b + 1, ..]);
                let (k_cached, v_cached) = kv_views[b];
                let ctx_b = match layer_class {
                    AttentionClass::Local { window } => causal_gqa_attention_swa_cached(
                        q_b,
                        k_cached,
                        v_cached,
                        cfg.num_attention_heads,
                        cfg.num_key_value_heads,
                        cfg.head_dim_value(),
                        q_pos_offsets[b],
                        window,
                    ),
                    AttentionClass::Global => causal_gqa_attention_cached(
                        q_b,
                        k_cached,
                        v_cached,
                        cfg.num_attention_heads,
                        cfg.num_key_value_heads,
                        cfg.head_dim_value(),
                        q_pos_offsets[b],
                    ),
                };
                ctx_slice.assign(&ctx_b);
            });
    });

    let attn_out = if offload {
        exec.offload_linear(WeightHandle::new(layer_idx, WeightKind::O), ctx.view())?
    } else {
        profile::time("tee:o_direct", || {
            tee_matmul_bf16(
                ctx.view(),
                layer
                    .wo
                    .as_ref()
                    .expect("offload=false requires layer.wo")
                    .view(),
            )
        })
    };
    let h1 = profile::time("tee:residual", || &hidden + &attn_out);

    let h1_norm = profile::time("tee:rmsnorm", || {
        rms_norm(
            h1.view(),
            layer.norm_ffn.as_slice().unwrap(),
            cfg.rms_norm_eps,
        )
    });

    let (gate, up) = if offload {
        let handles = [
            WeightHandle::new(layer_idx, WeightKind::FfnGate),
            WeightHandle::new(layer_idx, WeightKind::FfnUp),
        ];
        let mut out = exec.offload_linear_many(&handles, h1_norm.view())?;
        let u = out.pop().expect("offload_linear_many returns 2 outputs");
        let g = out.pop().expect("offload_linear_many returns 2 outputs");
        (g, u)
    } else {
        profile::time("tee:swiglu_proj_direct", || {
            (
                tee_matmul_bf16(
                    h1_norm.view(),
                    layer
                        .w_gate
                        .as_ref()
                        .expect("offload=false requires layer.w_gate")
                        .view(),
                ),
                tee_matmul_bf16(
                    h1_norm.view(),
                    layer
                        .w_up
                        .as_ref()
                        .expect("offload=false requires layer.w_up")
                        .view(),
                ),
            )
        })
    };
    let activated = profile::time("tee:swiglu_activate", || swiglu(gate.view(), up.view()));
    let ffn_out = if offload {
        exec.offload_linear(
            WeightHandle::new(layer_idx, WeightKind::FfnDown),
            activated.view(),
        )?
    } else {
        profile::time("tee:swiglu_down_direct", || {
            tee_matmul_bf16(
                activated.view(),
                layer
                    .w_down
                    .as_ref()
                    .expect("offload=false requires layer.w_down")
                    .view(),
            )
        })
    };
    Ok(profile::time("tee:residual", || &h1 + &ffn_out))
}

/// Cache-aware decoder block. Same compute path as the legacy
/// [`decoder_block`] but additionally appends the post-RoPE K, V to
/// `kv_cache` for layer `layer_idx`, and routes attention through the
/// asymmetric [`causal_gqa_attention_cached`] kernel so a single-row
/// Q (decode) can attend to the full cached prefix.
///
/// At prefill shape (n_q = n_kv, q_pos_offset = 0) this matches the
/// legacy block bit-for-bit (the asymmetric mask collapses to the
/// lower-triangular causal mask). At decode shape it's the harness's
/// single-token-per-step path.
///
/// OutAttnMult / permuted attention auto-switches are intentionally
/// not wired through this path yet — those are square-only kernels and
/// the fused permuted FlashAttention path lands in M1.10. Until then,
/// the cached block uses the in-TEE attention computation. This
/// matches the locked design decision: decode global attention stays
/// in-TEE always; long-context prefill global attention will use the
/// fused permuted kernel once M1.10 ships.
#[allow(clippy::too_many_arguments)]
fn decoder_block_cached(
    cfg: &DecoderConfig,
    layer: &DecoderLayerWeights,
    rope: &RopeTables,
    exec: &mut impl TrustedExecutor,
    layer_idx: u16,
    hidden: ArrayView2<'_, f32>,
    offload: bool,
    kv_cache: &mut KvCache,
    q_pos_offset: usize,
) -> Result<Array2<f32>> {
    // Pre-attention RMSNorm.
    let h_norm = profile::time("tee:rmsnorm", || {
        rms_norm(
            hidden,
            layer.norm_attn.as_slice().unwrap(),
            cfg.rms_norm_eps,
        )
    });

    // Q/K/V projections.
    //
    // For Gemma 4 global layers with `kv_shared_in_global` true, the
    // trained model ties `W_k = W_v` ("K equals V" trick). Two
    // mathematically-identical matmuls collapse to one — we compute K
    // once and reuse the result as V. The KV cache for these layers
    // is sized half as wide (one tensor instead of two) — see
    // `KvCache::new_with_sharing`.
    let layer_class = cfg.effective_attention_class(layer_idx as usize);
    let kv_shared = cfg.kv_shared_in_global && matches!(layer_class, AttentionClass::Global);

    let (mut q_new, mut k_new, v_new) = if offload {
        if kv_shared {
            // One masked matmul for Q, one for K=V. The K and V handles
            // map to the same backing weight when the model is loaded
            // with `wk` and `wv` Arc-shared; the executor doesn't need
            // to know that. We just skip the V offload and use K.
            let q =
                exec.offload_linear(WeightHandle::new(layer_idx, WeightKind::Q), h_norm.view())?;
            let k =
                exec.offload_linear(WeightHandle::new(layer_idx, WeightKind::K), h_norm.view())?;
            let v = k.clone();
            (q, k, v)
        } else {
            exec.offload_qkv(layer_idx, h_norm.view())?
        }
    } else {
        profile::time("tee:qkv_direct", || {
            let q = tee_matmul_bf16(
                h_norm.view(),
                layer
                    .wq
                    .as_ref()
                    .expect("offload=false requires layer.wq present (skip-layers mode)")
                    .view(),
            );
            let k = tee_matmul_bf16(
                h_norm.view(),
                layer
                    .wk
                    .as_ref()
                    .expect("offload=false requires layer.wk present (skip-layers mode)")
                    .view(),
            );
            let v = if kv_shared {
                k.clone()
            } else {
                tee_matmul_bf16(
                    h_norm.view(),
                    layer
                        .wv
                        .as_ref()
                        .expect("offload=false requires layer.wv present (skip-layers mode)")
                        .view(),
                )
            };
            (q, k, v)
        })
    };

    // Qwen3 QK-norm — per-head RMSNorm on Q and K **before** RoPE.
    // No-op for older models (norms = None). Applied per-head so each
    // attention head normalises its own `head_dim` slice independently;
    // gamma has length `head_dim`.
    profile::time("tee:qk_norm", || {
        if let Some(q_gamma) = layer.q_norm.as_ref() {
            apply_qk_norm(
                q_new.view_mut(),
                cfg.num_attention_heads,
                cfg.head_dim_value(),
                q_gamma.as_slice().expect("q_norm Array1 is contiguous"),
                cfg.rms_norm_eps,
            );
        }
        if let Some(k_gamma) = layer.k_norm.as_ref() {
            apply_qk_norm(
                k_new.view_mut(),
                cfg.num_key_value_heads,
                cfg.head_dim_value(),
                k_gamma.as_slice().expect("k_norm Array1 is contiguous"),
                cfg.rms_norm_eps,
            );
        }
    });

    // RoPE — rotate Q and K at absolute positions
    // `q_pos_offset..q_pos_offset + n_q`. Per the Gemma 4 p-RoPE
    // recipe: global layers rotate only the first `rotated_dim` of
    // each head; local layers rotate the full head_dim.
    let rotated_dim = match (layer_class, cfg.partial_rope) {
        (AttentionClass::Global, Some(_)) => cfg.rotated_dim(),
        // Local layers always rotate the full head_dim (Gemma 4
        // spec). Models with `partial_rope = None` likewise use
        // full rotation everywhere.
        _ => cfg.head_dim_value(),
    };
    profile::time("tee:rope", || {
        rope.apply_partial_at(
            q_new.view_mut(),
            cfg.num_attention_heads,
            q_pos_offset,
            rotated_dim,
        );
        rope.apply_partial_at(
            k_new.view_mut(),
            cfg.num_key_value_heads,
            q_pos_offset,
            rotated_dim,
        );
    });
    // For K=V shared global layers, the V tensor must stay identical
    // to K after RoPE. The simplest correctness path: re-derive V
    // from K post-RoPE. (Earlier we cloned K into V before RoPE, so
    // the clone is now stale.) Cheap — one ndarray clone.
    let v_new = if kv_shared { k_new.clone() } else { v_new };

    // Append fresh K, V to the cache before attention so the kernel
    // sees the full prefix including the current step's contribution.
    kv_cache.append(layer_idx as usize, k_new.view(), v_new.view())?;
    let (k_cached, v_cached) = kv_cache.view(layer_idx as usize)?;

    // Per-layer hybrid attention dispatch. The class falls back to
    // `Global` for `attention_classes = None`, preserving the
    // Qwen3 / Llama behaviour byte-for-byte.
    let ctx = match layer_class {
        AttentionClass::Local { window } => profile::time("tee:attn_swa_cached", || {
            causal_gqa_attention_swa_cached(
                q_new.view(),
                k_cached,
                v_cached,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim_value(),
                q_pos_offset,
                window,
            )
        }),
        AttentionClass::Global => {
            // M1.10.1.2: route Global cached attention through the
            // permutation-shielded path when the per-batch auto-switch
            // engages. The threshold compares against `n_q` (number of
            // NEW Q rows this forward) — at decode shape (n_q=1) the
            // permuted overhead would dominate so we stay in-TEE; at
            // prefill (n_q = n_prompt ≥ threshold) the permuted path
            // engages. Falls back to in-TEE when offload=false or the
            // master switch is off — the M1.3 default behaviour.
            let n_q = q_new.shape()[0];
            if offload && cfg.perm_attention_enabled_for(n_q) {
                profile::time("tee:attn_permuted_cached", || {
                    causal_gqa_attention_permuted_cached(
                        exec,
                        q_new.view(),
                        k_cached,
                        v_cached,
                        cfg.num_attention_heads,
                        cfg.num_key_value_heads,
                        cfg.head_dim_value(),
                        q_pos_offset,
                    )
                })?
            } else {
                profile::time("tee:attn_cached", || {
                    causal_gqa_attention_cached(
                        q_new.view(),
                        k_cached,
                        v_cached,
                        cfg.num_attention_heads,
                        cfg.num_key_value_heads,
                        cfg.head_dim_value(),
                        q_pos_offset,
                    )
                })
            }
        }
    };

    // Output projection — fresh mask per the per-offload protocol.
    let attn_out = if offload {
        exec.offload_linear(WeightHandle::new(layer_idx, WeightKind::O), ctx.view())?
    } else {
        profile::time("tee:o_direct", || {
            tee_matmul_bf16(
                ctx.view(),
                layer
                    .wo
                    .as_ref()
                    .expect("offload=false requires layer.wo present (skip-layers mode)")
                    .view(),
            )
        })
    };
    let h1 = profile::time("tee:residual", || &hidden + &attn_out);

    // Pre-FFN RMSNorm.
    let h1_norm = profile::time("tee:rmsnorm", || {
        rms_norm(
            h1.view(),
            layer.norm_ffn.as_slice().unwrap(),
            cfg.rms_norm_eps,
        )
    });

    // SwiGLU FFN — same shape, same offload group as the legacy block.
    let (gate, up) = if offload {
        let handles = [
            WeightHandle::new(layer_idx, WeightKind::FfnGate),
            WeightHandle::new(layer_idx, WeightKind::FfnUp),
        ];
        let mut out = exec.offload_linear_many(&handles, h1_norm.view())?;
        let u = out.pop().expect("offload_linear_many returns 2 outputs");
        let g = out.pop().expect("offload_linear_many returns 2 outputs");
        (g, u)
    } else {
        profile::time("tee:swiglu_proj_direct", || {
            (
                tee_matmul_bf16(
                    h1_norm.view(),
                    layer
                        .w_gate
                        .as_ref()
                        .expect("offload=false requires layer.w_gate present (skip-layers mode)")
                        .view(),
                ),
                tee_matmul_bf16(
                    h1_norm.view(),
                    layer
                        .w_up
                        .as_ref()
                        .expect("offload=false requires layer.w_up present (skip-layers mode)")
                        .view(),
                ),
            )
        })
    };

    let activated = profile::time("tee:swiglu_activate", || swiglu(gate.view(), up.view()));

    let ffn_out = if offload {
        exec.offload_linear(
            WeightHandle::new(layer_idx, WeightKind::FfnDown),
            activated.view(),
        )?
    } else {
        profile::time("tee:swiglu_down_direct", || {
            tee_matmul_bf16(
                activated.view(),
                layer
                    .w_down
                    .as_ref()
                    .expect("offload=false requires layer.w_down present (skip-layers mode)")
                    .view(),
            )
        })
    };
    Ok(profile::time("tee:residual", || &h1 + &ffn_out))
}

/// **M1.11 R1.3** — Batched forward pass over `B` sequences.
///
/// `input_ids[b]` is sequence b's token IDs. Sequences may differ in
/// length; right-padded internally to `n_max = max(len)`. Returns
/// `(hidden, seq_lens)` where `hidden` has shape
/// `(B, n_max, hidden_size)` (rows past `seq_lens[b]` are valid
/// numerically but represent positions the model never trained on —
/// callers gather the last *valid* row per sequence via `seq_lens`).
///
/// One `begin_prefill_pass(B, n_max)` bracket wraps the whole call;
/// the substrate samples B per-sequence masks (see `m1-11-batched-decode.md`
/// §3.4) and reuses them across every offload in the forward.
pub fn run_batched(
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    rope: &RopeTables,
    exec: &mut impl TrustedExecutor,
    input_ids: &[Vec<u32>],
) -> Result<(Array3<f32>, Vec<usize>)> {
    if input_ids.is_empty() {
        return Err(anyhow!("run_batched: input_ids must be non-empty"));
    }
    let batch_size = input_ids.len();
    let seq_lens: Vec<usize> = input_ids.iter().map(|s| s.len()).collect();
    let n_max = seq_lens.iter().copied().max().unwrap_or(0);
    if n_max == 0 {
        return Err(anyhow!(
            "run_batched: at least one sequence must have length > 0"
        ));
    }
    let d = cfg.hidden_size;

    // (B * n_max, d) flat embedding tensor. Pad rows stay zero.
    let mut h_flat = profile::time("tee:embed_lookup", || {
        let mut h = Array2::<f32>::zeros((batch_size * n_max, d));
        for (b, ids) in input_ids.iter().enumerate() {
            for (i, &id) in ids.iter().enumerate() {
                let row = weights.token_embedding.row(id as usize);
                for (j, v) in row.iter().enumerate() {
                    h[[b * n_max + i, j]] = v.to_f32();
                }
            }
        }
        h
    });

    let h_final = with_forward_session(
        exec,
        ForwardSessionShape::PrefillBatch { batch_size, n_max },
        |exec| {
            for (li, layer) in weights.layers.iter().enumerate() {
                h_flat = decoder_block_batched(
                    cfg,
                    layer,
                    rope,
                    exec,
                    li as u16,
                    h_flat.view(),
                    batch_size,
                    n_max,
                    &seq_lens,
                    cfg.offload_layer(li),
                    None,
                )?;
            }
            Ok(profile::time("tee:rmsnorm", || {
                rms_norm(
                    h_flat.view(),
                    weights.final_norm.as_slice().unwrap(),
                    cfg.rms_norm_eps,
                )
            }))
        },
    )?;

    // Materialise (B, n_max, hidden) from the flat (B*n_max, hidden)
    // tensor.  Direct copy preserves contiguity guarantees that
    // downstream gather logic relies on.
    let mut out = Array3::<f32>::zeros((batch_size, n_max, d));
    for b in 0..batch_size {
        out.slice_mut(ndarray::s![b, .., ..])
            .assign(&h_final.slice(ndarray::s![b * n_max..(b + 1) * n_max, ..]));
    }
    Ok((out, seq_lens))
}

/// **M1.11 D1.4** — Batched prefill forward that populates the KV
/// cache for subsequent batched decode steps.
///
/// Same shape contract as [`run_batched`]: returns `(B, n_max, hidden)`
/// + per-sequence valid lengths. Additionally, each layer's
/// post-RoPE K/V valid prefix per sequence is appended to `kv_cache`
/// via `kv_cache.append_prefill(li, b, k_b, v_b)`. After this call,
/// `kv_cache.len_b(li, b) == input_ids[b].len()` for every (li, b).
///
/// The caller then runs `run_decode_step_batched` one step at a time
/// with `token_ids: &[u32]` of length B (typically the next sampled
/// token per sequence; finished sequences feed a pad token, which is
/// fine because their output rows are discarded).
pub fn run_prefill_batched(
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    rope: &RopeTables,
    exec: &mut impl TrustedExecutor,
    input_ids: &[Vec<u32>],
    kv_cache: &mut KvCache,
) -> Result<(Array3<f32>, Vec<usize>)> {
    if input_ids.is_empty() {
        return Err(anyhow!("run_prefill_batched: input_ids must be non-empty"));
    }
    let batch_size = input_ids.len();
    assert_eq!(
        batch_size,
        kv_cache.batch_size(),
        "input_ids B={} != kv_cache batch_size={}",
        batch_size,
        kv_cache.batch_size(),
    );
    assert_eq!(
        kv_cache.num_layers(),
        weights.layers.len(),
        "kv_cache layer count must match model layer count",
    );
    assert_eq!(
        kv_cache.kv_dim(),
        cfg.kv_dim(),
        "kv_cache kv_dim must match cfg.kv_dim()",
    );
    let seq_lens: Vec<usize> = input_ids.iter().map(|s| s.len()).collect();
    let n_max = seq_lens.iter().copied().max().unwrap_or(0);
    if n_max == 0 {
        return Err(anyhow!(
            "run_prefill_batched: at least one sequence must have length > 0"
        ));
    }
    let d = cfg.hidden_size;

    let mut h_flat = profile::time("tee:embed_lookup", || {
        let mut h = Array2::<f32>::zeros((batch_size * n_max, d));
        for (b, ids) in input_ids.iter().enumerate() {
            for (i, &id) in ids.iter().enumerate() {
                let row = weights.token_embedding.row(id as usize);
                for (j, v) in row.iter().enumerate() {
                    h[[b * n_max + i, j]] = v.to_f32();
                }
            }
        }
        h
    });

    let h_final = with_forward_session(
        exec,
        ForwardSessionShape::PrefillBatch { batch_size, n_max },
        |exec| {
            for (li, layer) in weights.layers.iter().enumerate() {
                h_flat = decoder_block_batched(
                    cfg,
                    layer,
                    rope,
                    exec,
                    li as u16,
                    h_flat.view(),
                    batch_size,
                    n_max,
                    &seq_lens,
                    cfg.offload_layer(li),
                    Some(kv_cache),
                )?;
            }
            Ok(profile::time("tee:rmsnorm", || {
                rms_norm(
                    h_flat.view(),
                    weights.final_norm.as_slice().unwrap(),
                    cfg.rms_norm_eps,
                )
            }))
        },
    )?;

    let mut out = Array3::<f32>::zeros((batch_size, n_max, d));
    for b in 0..batch_size {
        out.slice_mut(ndarray::s![b, .., ..])
            .assign(&h_final.slice(ndarray::s![b * n_max..(b + 1) * n_max, ..]));
    }
    Ok((out, seq_lens))
}

/// Batched-prefill decoder block over `(B * n_max, hidden)` rows with
/// per-sequence valid-lengths `seq_lens`. Mirrors [`decoder_block`]
/// except:
///
/// 1. RoPE is applied **per-sequence** (each sequence's row 0 is at
///    absolute position 0).
/// 2. Attention is computed **per-sequence** in-TEE over each
///    sequence's valid prefix `[0..seq_lens[b]]` (R1.3 stopgap;
///    R1.4 ships a batched kernel routed through `engine.fused_attention_batched`).
/// 3. Linear projections (Q/K/V, O, gate/up, down) go through the
///    substrate's `PerSequence` session — `exec.offload_*` calls
///    transparently apply the B per-sequence masks.
///
/// Pad rows (rows `seq_lens[b]..n_max` of sequence b) carry garbage
/// values through the entire forward; they're irrelevant for the
/// caller's last-token gather. We do NOT zero them out per layer — the
/// residual stream just propagates whatever the embedding lookup
/// placed there (zero, in `run_batched`'s case).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn decoder_block_batched(
    cfg: &DecoderConfig,
    layer: &DecoderLayerWeights,
    rope: &RopeTables,
    exec: &mut impl TrustedExecutor,
    layer_idx: u16,
    hidden: ArrayView2<'_, f32>,
    batch_size: usize,
    n_max: usize,
    seq_lens: &[usize],
    offload: bool,
    // M1.11 D1.4: when `Some`, append each sequence's post-RoPE K/V
    // valid prefix to the cache so subsequent `run_decode_step_batched`
    // calls can continue from this prefill state.
    kv_cache_prefill: Option<&mut KvCache>,
) -> Result<Array2<f32>> {
    debug_assert_eq!(hidden.nrows(), batch_size * n_max);

    let h_norm = profile::time("tee:rmsnorm", || {
        rms_norm(
            hidden,
            layer.norm_attn.as_slice().unwrap(),
            cfg.rms_norm_eps,
        )
    });

    // Q/K/V — under PerSequence session, `offload_qkv` falls through
    // to 3 `offload_linear` calls each running the per-sequence
    // path (substrate handles slicing).
    let (mut q, mut k, v) = if offload {
        exec.offload_qkv(layer_idx, h_norm.view())?
    } else {
        profile::time("tee:qkv_direct", || {
            (
                tee_matmul_bf16(
                    h_norm.view(),
                    layer
                        .wq
                        .as_ref()
                        .expect("offload=false requires layer.wq present")
                        .view(),
                ),
                tee_matmul_bf16(
                    h_norm.view(),
                    layer
                        .wk
                        .as_ref()
                        .expect("offload=false requires layer.wk present")
                        .view(),
                ),
                tee_matmul_bf16(
                    h_norm.view(),
                    layer
                        .wv
                        .as_ref()
                        .expect("offload=false requires layer.wv present")
                        .view(),
                ),
            )
        })
    };

    profile::time("tee:qk_norm", || {
        if let Some(q_gamma) = layer.q_norm.as_ref() {
            apply_qk_norm(
                q.view_mut(),
                cfg.num_attention_heads,
                cfg.head_dim_value(),
                q_gamma.as_slice().expect("q_norm contiguous"),
                cfg.rms_norm_eps,
            );
        }
        if let Some(k_gamma) = layer.k_norm.as_ref() {
            apply_qk_norm(
                k.view_mut(),
                cfg.num_key_value_heads,
                cfg.head_dim_value(),
                k_gamma.as_slice().expect("k_norm contiguous"),
                cfg.rms_norm_eps,
            );
        }
    });

    // Per-sequence RoPE — each sequence's row i sits at absolute
    // position i (start_pos = 0).
    profile::time("tee:rope", || {
        for b in 0..batch_size {
            let q_block = q.slice_mut(ndarray::s![b * n_max..(b + 1) * n_max, ..]);
            rope.apply(q_block, cfg.num_attention_heads);
            let k_block = k.slice_mut(ndarray::s![b * n_max..(b + 1) * n_max, ..]);
            rope.apply(k_block, cfg.num_key_value_heads);
        }
    });

    // M1.11 D1.4 — when running as part of a batched prefill that
    // initialises the decode KV cache, append each sequence's
    // post-RoPE K/V valid prefix.  Pad rows are *not* appended; the
    // KV cache stores only real-token K/V per sequence.
    if let Some(kv_cache) = kv_cache_prefill {
        for b in 0..batch_size {
            let valid_n = seq_lens[b];
            if valid_n == 0 {
                continue;
            }
            let k_b = k.slice(ndarray::s![b * n_max..b * n_max + valid_n, ..]);
            let v_b = v.slice(ndarray::s![b * n_max..b * n_max + valid_n, ..]);
            kv_cache.append_prefill(layer_idx as usize, b, k_b, v_b)?;
        }
    }

    // Per-sequence attention via B serial in-TEE `causal_gqa_attention`
    // calls, one per sequence's valid prefix. Pad rows get a zero
    // context (their residual stream is unchanged beyond the FFN
    // residual).
    //
    // TODO(R1.4): replace this loop with a single dispatch through
    // `engine.fused_attention_batched` — reshape Q/K/V to per-head-
    // batched `(B·num_heads, n_q, d_head)`, fold per-sequence right-
    // padding + causal into one additive `(B·num_heads, n_q, n_kv)`
    // mask, dispatch once. Only land this when the
    // `tee:attn_inplace_many` bucket grows past ~10% of batched wall
    // (Qwen3-Reranker-0.6B at B=8 is currently 3.4% — not worth the
    // engineering). The trigger is longer per-sequence n_max (cross-
    // encoder rerank with longer docs, batched extraction with
    // longer chunks, or D-phase generate_batched). See M1.11 plan
    // §3.2 and `docs/handoffs/2026-05-21-attn-offload-spike.md` for
    // the kernel-routing reasoning.
    let q_dim = cfg.num_attention_heads * cfg.head_dim_value();
    let mut ctx = Array2::<f32>::zeros((batch_size * n_max, q_dim));
    profile::time("tee:attn_inplace_many", || {
        for b in 0..batch_size {
            let valid_n = seq_lens[b];
            if valid_n == 0 {
                continue;
            }
            let q_b = q.slice(ndarray::s![b * n_max..b * n_max + valid_n, ..]);
            let k_b = k.slice(ndarray::s![b * n_max..b * n_max + valid_n, ..]);
            let v_b = v.slice(ndarray::s![b * n_max..b * n_max + valid_n, ..]);
            let ctx_b = causal_gqa_attention(
                q_b,
                k_b,
                v_b,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim_value(),
            );
            ctx.slice_mut(ndarray::s![b * n_max..b * n_max + valid_n, ..])
                .assign(&ctx_b);
        }
    });

    let attn_out = if offload {
        exec.offload_linear(WeightHandle::new(layer_idx, WeightKind::O), ctx.view())?
    } else {
        profile::time("tee:o_direct", || {
            tee_matmul_bf16(
                ctx.view(),
                layer
                    .wo
                    .as_ref()
                    .expect("offload=false requires layer.wo present")
                    .view(),
            )
        })
    };
    let h1 = profile::time("tee:residual", || &hidden + &attn_out);

    let h1_norm = profile::time("tee:rmsnorm", || {
        rms_norm(
            h1.view(),
            layer.norm_ffn.as_slice().unwrap(),
            cfg.rms_norm_eps,
        )
    });

    let (gate, up) = if offload {
        let handles = [
            WeightHandle::new(layer_idx, WeightKind::FfnGate),
            WeightHandle::new(layer_idx, WeightKind::FfnUp),
        ];
        let mut out = exec.offload_linear_many(&handles, h1_norm.view())?;
        let u = out.pop().expect("offload_linear_many returns 2 outputs");
        let g = out.pop().expect("offload_linear_many returns 2 outputs");
        (g, u)
    } else {
        profile::time("tee:swiglu_proj_direct", || {
            (
                tee_matmul_bf16(
                    h1_norm.view(),
                    layer
                        .w_gate
                        .as_ref()
                        .expect("offload=false requires layer.w_gate present")
                        .view(),
                ),
                tee_matmul_bf16(
                    h1_norm.view(),
                    layer
                        .w_up
                        .as_ref()
                        .expect("offload=false requires layer.w_up present")
                        .view(),
                ),
            )
        })
    };
    let activated = profile::time("tee:swiglu_activate", || swiglu(gate.view(), up.view()));
    let ffn_out = if offload {
        exec.offload_linear(
            WeightHandle::new(layer_idx, WeightKind::FfnDown),
            activated.view(),
        )?
    } else {
        profile::time("tee:swiglu_down_direct", || {
            tee_matmul_bf16(
                activated.view(),
                layer
                    .w_down
                    .as_ref()
                    .expect("offload=false requires layer.w_down present")
                    .view(),
            )
        })
    };
    Ok(profile::time("tee:residual", || &h1 + &ffn_out))
}

fn decoder_block(
    cfg: &DecoderConfig,
    layer: &DecoderLayerWeights,
    rope: &RopeTables,
    exec: &mut impl TrustedExecutor,
    layer_idx: u16,
    hidden: ArrayView2<'_, f32>,
    offload: bool,
) -> Result<Array2<f32>> {
    // Pre-attention RMSNorm.
    let h_norm = profile::time("tee:rmsnorm", || {
        rms_norm(
            hidden,
            layer.norm_attn.as_slice().unwrap(),
            cfg.rms_norm_eps,
        )
    });

    // Q/K/V projections — offloaded one mask shared across the three reads,
    // matching the BERT path.
    let (mut q, mut k, v) = if offload {
        exec.offload_qkv(layer_idx, h_norm.view())?
    } else {
        profile::time("tee:qkv_direct", || {
            (
                tee_matmul_bf16(
                    h_norm.view(),
                    layer
                        .wq
                        .as_ref()
                        .expect("offload=false requires layer.wq present (skip-layers mode)")
                        .view(),
                ),
                tee_matmul_bf16(
                    h_norm.view(),
                    layer
                        .wk
                        .as_ref()
                        .expect("offload=false requires layer.wk present (skip-layers mode)")
                        .view(),
                ),
                tee_matmul_bf16(
                    h_norm.view(),
                    layer
                        .wv
                        .as_ref()
                        .expect("offload=false requires layer.wv present (skip-layers mode)")
                        .view(),
                ),
            )
        })
    };

    // Qwen3 QK-norm — per-head RMSNorm on Q and K **before** RoPE.
    // No-op when the loaded checkpoint lacks `q_norm` / `k_norm`
    // (Qwen2 / LLaMA / Mistral).
    profile::time("tee:qk_norm", || {
        if let Some(q_gamma) = layer.q_norm.as_ref() {
            apply_qk_norm(
                q.view_mut(),
                cfg.num_attention_heads,
                cfg.head_dim_value(),
                q_gamma.as_slice().expect("q_norm Array1 is contiguous"),
                cfg.rms_norm_eps,
            );
        }
        if let Some(k_gamma) = layer.k_norm.as_ref() {
            apply_qk_norm(
                k.view_mut(),
                cfg.num_key_value_heads,
                cfg.head_dim_value(),
                k_gamma.as_slice().expect("k_norm Array1 is contiguous"),
                cfg.rms_norm_eps,
            );
        }
    });

    // RoPE rotates Q and K only (V left alone) per-head.
    profile::time("tee:rope", || {
        rope.apply(q.view_mut(), cfg.num_attention_heads);
        rope.apply(k.view_mut(), cfg.num_key_value_heads);
    });

    // GQA + causal attention. When this layer is offloaded **and** the
    // auto-switch fires (sequence length ≥ threshold; see
    // `DecoderConfig::out_attn_mult_enabled_for`), route per-head Q·Kᵀ
    // through TwinShield OutAttnMult. Otherwise compute attention inside
    // the TEE — equally confidential (Q, K never cross PCIe), and faster
    // at short n where the 4-partition scheme's 4× FLOP widening loses
    // to a plain in-TEE matmul.
    let n = q.shape()[0];
    let ctx = if offload && cfg.out_attn_mult_enabled_for(n) {
        causal_gqa_attention_with_offload(
            exec,
            q.view(),
            k.view(),
            v.view(),
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim_value(),
        )?
    } else if offload && cfg.perm_attention_enabled_for(n) {
        causal_gqa_attention_permuted(
            exec,
            q.view(),
            k.view(),
            v.view(),
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim_value(),
        )?
    } else {
        profile::time("tee:attn_inplace", || {
            causal_gqa_attention(
                q.view(),
                k.view(),
                v.view(),
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim_value(),
            )
        })
    };

    // Output projection — fresh mask.
    let attn_out = if offload {
        exec.offload_linear(WeightHandle::new(layer_idx, WeightKind::O), ctx.view())?
    } else {
        profile::time("tee:o_direct", || {
            tee_matmul_bf16(
                ctx.view(),
                layer
                    .wo
                    .as_ref()
                    .expect("offload=false requires layer.wo present (skip-layers mode)")
                    .view(),
            )
        })
    };
    let h1 = profile::time("tee:residual", || &hidden + &attn_out);

    // Pre-FFN RMSNorm.
    let h1_norm = profile::time("tee:rmsnorm", || {
        rms_norm(
            h1.view(),
            layer.norm_ffn.as_slice().unwrap(),
            cfg.rms_norm_eps,
        )
    });

    // SwiGLU FFN: gate + up share the same input `h1_norm`, so one
    // `offload_linear_many` call shares the mask apply + batches the
    // matmul + batches the unapply across both projections.
    let (gate, up) = if offload {
        let handles = [
            WeightHandle::new(layer_idx, WeightKind::FfnGate),
            WeightHandle::new(layer_idx, WeightKind::FfnUp),
        ];
        let mut out = exec.offload_linear_many(&handles, h1_norm.view())?;
        let u = out.pop().expect("offload_linear_many returns 2 outputs");
        let g = out.pop().expect("offload_linear_many returns 2 outputs");
        (g, u)
    } else {
        profile::time("tee:swiglu_proj_direct", || {
            (
                tee_matmul_bf16(
                    h1_norm.view(),
                    layer
                        .w_gate
                        .as_ref()
                        .expect("offload=false requires layer.w_gate present (skip-layers mode)")
                        .view(),
                ),
                tee_matmul_bf16(
                    h1_norm.view(),
                    layer
                        .w_up
                        .as_ref()
                        .expect("offload=false requires layer.w_up present (skip-layers mode)")
                        .view(),
                ),
            )
        })
    };

    let activated = profile::time("tee:swiglu_activate", || swiglu(gate.view(), up.view()));

    let ffn_out = if offload {
        exec.offload_linear(
            WeightHandle::new(layer_idx, WeightKind::FfnDown),
            activated.view(),
        )?
    } else {
        profile::time("tee:swiglu_down_direct", || {
            tee_matmul_bf16(
                activated.view(),
                layer
                    .w_down
                    .as_ref()
                    .expect("offload=false requires layer.w_down present (skip-layers mode)")
                    .view(),
            )
        })
    };
    Ok(profile::time("tee:residual", || &h1 + &ffn_out))
}

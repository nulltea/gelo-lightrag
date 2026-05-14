use anyhow::Result;
use ndarray::{Array2, ArrayView2};

use gelo_protocol::profile;
use gelo_protocol::{TrustedExecutor, WeightHandle, WeightKind};

use super::attention::{
    causal_gqa_attention, causal_gqa_attention_permuted, causal_gqa_attention_with_offload,
};
use super::config::DecoderConfig;
use super::rms_norm::rms_norm;
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
/// layer's input). The hook is the integration point for DP-Forward
/// intermediate-layer aMGM noise (M7.1): the embedder constructs a
/// closure that matches against `DpForwardConfig::layer_index` and applies
/// clip + Gaussian noise to each token-row of `h`.
///
/// For pre-norm decoder blocks (Qwen3-style), the layer output is the
/// final residual add at the end of the block — the analog of BERT's
/// `add_and_norm_2` position in the DP-Forward paper.
pub fn run_with_hook<F: FnMut(usize, &mut Array2<f32>)>(
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    rope: &RopeTables,
    exec: &mut impl TrustedExecutor,
    input_ids: &[u32],
    mut after_layer: F,
) -> Result<Array2<f32>> {
    let n = input_ids.len();
    let mut h = profile::time("tee:embed_lookup", || embedding_lookup(cfg, weights, input_ids));

    // GELO paper §3.2 forward-pass session: see bert/forward.rs for the
    // rationale. Paper-parity executors sample one mask here; per-offload
    // executors and PlaintextExecutor treat this as a no-op.
    exec.begin_forward_pass(n)?;
    let result = (|| -> Result<Array2<f32>> {
        for (li, layer) in weights.layers.iter().enumerate() {
            h = decoder_block(cfg, layer, rope, exec, li as u16, h.view(), cfg.offload_layer(li))?;
            after_layer(li, &mut h);
        }
        Ok(profile::time("tee:rmsnorm", || {
            rms_norm(h.view(), weights.final_norm.as_slice().unwrap(), cfg.rms_norm_eps)
        }))
    })();
    exec.end_forward_pass()?;
    result
}

fn embedding_lookup(cfg: &DecoderConfig, w: &DecoderWeights, ids: &[u32]) -> Array2<f32> {
    let n = ids.len();
    let d = cfg.hidden_size;
    let mut out = Array2::<f32>::zeros((n, d));
    for (i, &id) in ids.iter().enumerate() {
        let row = w.token_embedding.row(id as usize);
        out.row_mut(i).assign(&row);
    }
    out
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
        rms_norm(hidden, layer.norm_attn.as_slice().unwrap(), cfg.rms_norm_eps)
    });

    // Q/K/V projections — offloaded one mask shared across the three reads,
    // matching the BERT path.
    let (mut q, mut k, v) = if offload {
        exec.offload_qkv(layer_idx, h_norm.view())?
    } else {
        profile::time("tee:qkv_direct", || {
            (
                h_norm.dot(&layer.wq),
                h_norm.dot(&layer.wk),
                h_norm.dot(&layer.wv),
            )
        })
    };

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
        profile::time("tee:o_direct", || ctx.dot(&layer.wo))
    };
    let h1 = profile::time("tee:residual", || &hidden + &attn_out);

    // Pre-FFN RMSNorm.
    let h1_norm = profile::time("tee:rmsnorm", || {
        rms_norm(h1.view(), layer.norm_ffn.as_slice().unwrap(), cfg.rms_norm_eps)
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
            (h1_norm.dot(&layer.w_gate), h1_norm.dot(&layer.w_up))
        })
    };

    let activated = profile::time("tee:swiglu_activate", || swiglu(gate.view(), up.view()));

    let ffn_out = if offload {
        exec.offload_linear(
            WeightHandle::new(layer_idx, WeightKind::FfnDown),
            activated.view(),
        )?
    } else {
        profile::time("tee:swiglu_down_direct", || activated.dot(&layer.w_down))
    };
    Ok(profile::time("tee:residual", || &h1 + &ffn_out))
}

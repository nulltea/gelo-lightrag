use anyhow::Result;
use ndarray::{Array1, Array2, ArrayView2, Axis};

use gelo_protocol::profile;
use gelo_protocol::{TrustedExecutor, WeightHandle, WeightKind};

use super::attention::multi_head_attention;
use super::config::BertConfig;
use super::weights::{BertLayerWeights, BertWeights};

/// Run the BERT encoder for a single sequence under the GELO protocol.
///
/// `input_ids` is a flat `[seq_len]` slice. The result is the per-token
/// hidden state matrix `(seq_len, hidden_size)` from the final layer,
/// ready for pooling.
pub fn run(
    cfg: &BertConfig,
    weights: &BertWeights,
    exec: &mut impl TrustedExecutor,
    input_ids: &[u32],
) -> Result<Array2<f32>> {
    run_with_hook(cfg, weights, exec, input_ids, |_, _| {})
}

/// Same as [`run`] but invokes `after_layer(layer_idx, &mut h)` after the
/// post-FFN `add_and_norm_2` position of each transformer block — the
/// position used by the DP-Forward paper's released code for aMGM noise
/// injection (`xiangyue9607/DP-Forward`, `noise_layer = 10` on BERT-base).
/// The hook is the integration point for DP-Forward intermediate-layer
/// aMGM (M7.1).
pub fn run_with_hook<F: FnMut(usize, &mut Array2<f32>)>(
    cfg: &BertConfig,
    weights: &BertWeights,
    exec: &mut impl TrustedExecutor,
    input_ids: &[u32],
    mut after_layer: F,
) -> Result<Array2<f32>> {
    let n = input_ids.len();
    let mut h = profile::time("tee:embed_lookup", || build_embedding(cfg, weights, input_ids));
    h = profile::time("tee:layernorm", || {
        layer_norm(h.view(), &weights.embeddings_ln_w, &weights.embeddings_ln_b, cfg.layer_norm_eps)
    });

    // GELO paper §3.2 forward-pass session: bracket every per-text
    // forward with begin/end so executors running in paper-parity
    // (per-forward-pass A) mode sample exactly one mask. Per-offload
    // executors and `PlaintextExecutor` use the trait's no-op defaults.
    exec.begin_forward_pass(n)?;
    let result = (|| -> Result<Array2<f32>> {
        for (li, layer) in weights.layers.iter().enumerate() {
            h = encoder_block(cfg, layer, exec, li as u16, h.view(), cfg.offload_layer(li))?;
            after_layer(li, &mut h);
        }
        Ok(h)
    })();
    exec.end_forward_pass()?;
    result
}

fn build_embedding(cfg: &BertConfig, w: &BertWeights, ids: &[u32]) -> Array2<f32> {
    let n = ids.len();
    let d = cfg.hidden_size;
    let mut out = Array2::<f32>::zeros((n, d));
    let token_type_row = w.token_type_embedding.row(0);
    for (i, &id) in ids.iter().enumerate() {
        let word = w.word_embedding.row(id as usize);
        let pos = w.position_embedding.row(i);
        let mut dst = out.row_mut(i);
        for j in 0..d {
            dst[j] = word[j] + pos[j] + token_type_row[j];
        }
    }
    out
}

fn encoder_block(
    cfg: &BertConfig,
    layer: &BertLayerWeights,
    exec: &mut impl TrustedExecutor,
    layer_idx: u16,
    hidden: ArrayView2<'_, f32>,
    offload: bool,
) -> Result<Array2<f32>> {
    // Self-attention block: Q, K, V are computed from the residual input.
    let (q, k, v) = if offload {
        exec.offload_qkv(layer_idx, hidden)?
    } else {
        profile::time("tee:qkv_direct", || {
            (
                hidden.dot(&layer.wq),
                hidden.dot(&layer.wk),
                hidden.dot(&layer.wv),
            )
        })
    };
    let (q, k, v) = profile::time("tee:add_bias", || {
        (
            add_bias(q, &layer.bq),
            add_bias(k, &layer.bk),
            add_bias(v, &layer.bv),
        )
    });

    let ctx = profile::time("tee:bert_mha", || {
        multi_head_attention(q.view(), k.view(), v.view(), cfg.num_attention_heads)
    });

    let proj = if offload {
        exec.offload_linear(WeightHandle::new(layer_idx, WeightKind::O), ctx.view())?
    } else {
        profile::time("tee:o_direct", || ctx.dot(&layer.wo))
    };
    let proj = profile::time("tee:add_bias", || add_bias(proj, &layer.bo));

    // Post-LN around residual
    let h_attn = profile::time("tee:layernorm", || {
        layer_norm(
            add(hidden, proj.view()).view(),
            &layer.attn_ln_w,
            &layer.attn_ln_b,
            cfg.layer_norm_eps,
        )
    });

    // FFN
    let intermediate = if offload {
        exec.offload_linear(WeightHandle::new(layer_idx, WeightKind::FfnUp), h_attn.view())?
    } else {
        profile::time("tee:ffn_up_direct", || h_attn.dot(&layer.w_ffn_up))
    };
    let intermediate = profile::time("tee:add_bias", || add_bias(intermediate, &layer.b_ffn_up));
    let intermediate = profile::time("tee:gelu", || gelu(intermediate));

    let ffn_out = if offload {
        exec.offload_linear(
            WeightHandle::new(layer_idx, WeightKind::FfnDown),
            intermediate.view(),
        )?
    } else {
        profile::time("tee:ffn_down_direct", || intermediate.dot(&layer.w_ffn_down))
    };
    let ffn_out = profile::time("tee:add_bias", || add_bias(ffn_out, &layer.b_ffn_down));

    let h_out = profile::time("tee:layernorm", || {
        layer_norm(
            add(h_attn.view(), ffn_out.view()).view(),
            &layer.ffn_ln_w,
            &layer.ffn_ln_b,
            cfg.layer_norm_eps,
        )
    });

    Ok(h_out)
}

fn add(a: ArrayView2<'_, f32>, b: ArrayView2<'_, f32>) -> Array2<f32> {
    &a + &b
}

fn add_bias(mut m: Array2<f32>, bias: &Array1<f32>) -> Array2<f32> {
    // Single-thread; broadcast-add auto-vectorises in LLVM. Rayon overhead
    // at our per-call work size (typ. 128×768) costs more than it saves.
    for mut row in m.axis_iter_mut(Axis(0)) {
        row += bias;
    }
    m
}

fn layer_norm(x: ArrayView2<'_, f32>, gamma: &Array1<f32>, beta: &Array1<f32>, eps: f32) -> Array2<f32> {
    // Single-pass mean+variance (E[X²] − E[X]²) per row, then a single
    // multiply-add per element using a precomputed `inv_denom`. Avoids the
    // 2× pass + per-element division of the textbook formulation. Single-
    // threaded; per-row work is too small to amortise rayon overhead.
    let d = x.ncols() as f32;
    let inv_d = d.recip();
    let mut out = Array2::<f32>::zeros(x.raw_dim());
    for (mut dst, row) in out.axis_iter_mut(Axis(0)).zip(x.axis_iter(Axis(0))) {
        let mut s = 0.0_f32;
        let mut ss = 0.0_f32;
        for &v in row.iter() {
            s += v;
            ss += v * v;
        }
        let mean = s * inv_d;
        let var = ss * inv_d - mean * mean;
        let inv_denom = (var + eps).sqrt().recip();
        for ((d_v, &x_v), (&g, &b)) in
            dst.iter_mut().zip(row.iter()).zip(gamma.iter().zip(beta.iter()))
        {
            *d_v = (x_v - mean) * inv_denom * g + b;
        }
    }
    out
}

fn gelu(mut m: Array2<f32>) -> Array2<f32> {
    // erf-based GELU (matches HF BERT's "gelu" activation exactly):
    // 0.5 * x * (1 + erf(x / sqrt(2))). Single-threaded; element work
    // is too small to benefit from rayon scheduling.
    let inv_sqrt_2 = 1.0_f32 / 2.0_f32.sqrt();
    for v in m.iter_mut() {
        let y = *v * inv_sqrt_2;
        *v = 0.5 * *v * (1.0 + erf(y));
    }
    m
}

/// Abramowitz-Stegun 7.1.26 approximation; max abs error ~1.5e-7.
#[allow(clippy::excessive_precision)]
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0_f32 / (1.0_f32 + 0.3275911_f32 * x);
    let y = 1.0
        - (((((1.061405429_f32 * t - 1.453152027_f32) * t) + 1.421413741_f32) * t - 0.284496736_f32) * t
            + 0.254829592_f32)
            * t
            * (-x * x).exp();
    sign * y
}

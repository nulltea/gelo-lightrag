use anyhow::Result;
use ndarray::{Array1, Array2, ArrayView2};

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
    let mut h = build_embedding(cfg, weights, input_ids);
    h = layer_norm(h.view(), &weights.embeddings_ln_w, &weights.embeddings_ln_b, cfg.layer_norm_eps);

    for (li, layer) in weights.layers.iter().enumerate() {
        h = encoder_block(cfg, layer, exec, li as u16, h.view(), cfg.offload_layer(li))?;
    }
    Ok(h)
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
        (
            hidden.dot(&layer.wq),
            hidden.dot(&layer.wk),
            hidden.dot(&layer.wv),
        )
    };
    let q = add_bias(q, &layer.bq);
    let k = add_bias(k, &layer.bk);
    let v = add_bias(v, &layer.bv);

    let ctx = multi_head_attention(q.view(), k.view(), v.view(), cfg.num_attention_heads);

    let proj = if offload {
        exec.offload_linear(WeightHandle::new(layer_idx, WeightKind::O), ctx.view())?
    } else {
        ctx.dot(&layer.wo)
    };
    let proj = add_bias(proj, &layer.bo);

    // Post-LN around residual
    let h_attn = layer_norm(
        add(hidden, proj.view()).view(),
        &layer.attn_ln_w,
        &layer.attn_ln_b,
        cfg.layer_norm_eps,
    );

    // FFN
    let intermediate = if offload {
        exec.offload_linear(WeightHandle::new(layer_idx, WeightKind::FfnUp), h_attn.view())?
    } else {
        h_attn.dot(&layer.w_ffn_up)
    };
    let intermediate = add_bias(intermediate, &layer.b_ffn_up);
    let intermediate = gelu(intermediate);

    let ffn_out = if offload {
        exec.offload_linear(
            WeightHandle::new(layer_idx, WeightKind::FfnDown),
            intermediate.view(),
        )?
    } else {
        intermediate.dot(&layer.w_ffn_down)
    };
    let ffn_out = add_bias(ffn_out, &layer.b_ffn_down);

    let h_out = layer_norm(
        add(h_attn.view(), ffn_out.view()).view(),
        &layer.ffn_ln_w,
        &layer.ffn_ln_b,
        cfg.layer_norm_eps,
    );

    Ok(h_out)
}

fn add(a: ArrayView2<'_, f32>, b: ArrayView2<'_, f32>) -> Array2<f32> {
    &a + &b
}

fn add_bias(mut m: Array2<f32>, bias: &Array1<f32>) -> Array2<f32> {
    for mut row in m.rows_mut() {
        for (j, v) in row.iter_mut().enumerate() {
            *v += bias[j];
        }
    }
    m
}

fn layer_norm(x: ArrayView2<'_, f32>, gamma: &Array1<f32>, beta: &Array1<f32>, eps: f32) -> Array2<f32> {
    let d = x.ncols() as f32;
    let mut out = Array2::<f32>::zeros(x.raw_dim());
    for (i, row) in x.rows().into_iter().enumerate() {
        let mean = row.iter().sum::<f32>() / d;
        let var = row.iter().map(|v| (*v - mean).powi(2)).sum::<f32>() / d;
        let denom = (var + eps).sqrt();
        let mut dst = out.row_mut(i);
        for (j, v) in row.iter().enumerate() {
            dst[j] = (*v - mean) / denom * gamma[j] + beta[j];
        }
    }
    out
}

fn gelu(mut m: Array2<f32>) -> Array2<f32> {
    // erf-based GELU (matches HF BERT's "gelu" activation exactly):
    // 0.5 * x * (1 + erf(x / sqrt(2)))
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

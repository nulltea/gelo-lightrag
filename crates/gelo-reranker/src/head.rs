//! Head adapters consumed by the two service variants.
//!
//! - [`ClassifierHead`] — the two-layer
//!   `XLMRobertaForSequenceClassification` head used by
//!   `BAAI/bge-reranker-v2-m3`:
//!   `out_proj(tanh(dense(cls)))` with `dense: hidden→hidden` and
//!   `out_proj: hidden→1`.
//! - [`YesNoHead`] — gathers two pinned vocab logits from a causal-LM
//!   final hidden state and returns `softmax([no, yes])[1]`. Used by
//!   `Qwen/Qwen3-Reranker-0.6B`.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use ndarray::{Array1, Array2, ArrayView1};
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};
use sha2::{Digest, Sha256};

/// Two-layer classification head. Stored transposed (`in × out`) so the
/// projection is `cls · w + b` — matching the rest of the gelo-embedder
/// conventions.
#[derive(Debug, Clone)]
pub struct ClassifierHead {
    /// `(hidden, hidden)` — pooled-CLS → intermediate.
    pub dense_w: Array2<f32>,
    pub dense_b: Array1<f32>,
    /// `(hidden, 1)` — intermediate → relevance scalar.
    pub out_w: Array2<f32>,
    pub out_b: Array1<f32>,
    /// SHA-256 of the four head tensors as they were read from the
    /// safetensors file (concatenated in declaration order). Folded
    /// into `CrossEncoderRerankService::model_identity` so the
    /// attestation report's model binding covers the head as well as
    /// the backbone.
    pub identity: [u8; 32],
}

impl ClassifierHead {
    /// Apply the head to a single CLS row.
    pub fn score(&self, cls: ArrayView1<'_, f32>) -> f32 {
        // intermediate = tanh(cls · dense_w + dense_b)
        let mut h = cls.dot(&self.dense_w);
        h += &self.dense_b;
        for v in h.iter_mut() {
            *v = v.tanh();
        }
        // out = h · out_w + out_b (single scalar)
        let out = h.dot(&self.out_w);
        out[0] + self.out_b[0]
    }

    /// Load the head from a `model.safetensors` file. Looks for the
    /// standard `classifier.dense.{weight,bias}` +
    /// `classifier.out_proj.{weight,bias}` quartet under any optional
    /// model-name prefix (e.g. `roberta.` for XLM-R-backed rerankers).
    pub fn from_safetensors(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading safetensors from {}", path.display()))?;
        let st = SafeTensors::deserialize(&bytes)
            .with_context(|| format!("deserializing safetensors at {}", path.display()))?;

        let dense_w_view = lookup_first(&st, &["classifier.dense.weight"])
            .context("missing classifier.dense.weight — model is not a standard sequence classifier")?;
        let dense_b_view = lookup_first(&st, &["classifier.dense.bias"])
            .context("missing classifier.dense.bias")?;
        let out_w_view = lookup_first(&st, &["classifier.out_proj.weight"])
            .context("missing classifier.out_proj.weight")?;
        let out_b_view = lookup_first(&st, &["classifier.out_proj.bias"])
            .context("missing classifier.out_proj.bias")?;

        let mut hasher = Sha256::new();
        hasher.update(dense_w_view.data());
        hasher.update(dense_b_view.data());
        hasher.update(out_w_view.data());
        hasher.update(out_b_view.data());
        let identity: [u8; 32] = hasher.finalize().into();

        // HF Linear stores (out, in); we keep (in, out).
        let dense_w = tensor_to_2d(dense_w_view)?.t().to_owned();
        let dense_b = tensor_to_1d(dense_b_view)?;
        let out_w_raw = tensor_to_2d(out_w_view)?;
        let out_w = out_w_raw.t().to_owned();
        let out_b = tensor_to_1d(out_b_view)?;

        if out_w.shape()[1] != 1 {
            return Err(anyhow!(
                "expected single-label classifier (out_proj.weight shape[0] == 1), got {:?}",
                out_w_raw.shape()
            ));
        }

        Ok(Self { dense_w, dense_b, out_w, out_b, identity })
    }

    /// Build a head from already-loaded arrays. Used by the synthetic-
    /// weight parity tests in `crates/gelo-reranker/tests/`.
    pub fn from_arrays(
        dense_w: Array2<f32>,
        dense_b: Array1<f32>,
        out_w: Array2<f32>,
        out_b: Array1<f32>,
    ) -> Self {
        Self { dense_w, dense_b, out_w, out_b, identity: [0u8; 32] }
    }
}

/// Pinned vocab IDs for the yes/no discriminator scoring head. The IDs
/// are part of the attested scheme — a tokenizer JSON revision change
/// that re-numbers `yes` / `no` must trip the model-identity binding.
#[derive(Debug, Clone, Copy)]
pub struct YesNoHead {
    pub yes_token_id: u32,
    pub no_token_id: u32,
}

fn lookup_first<'a>(st: &'a SafeTensors<'a>, candidate_names: &[&str]) -> Result<TensorView<'a>> {
    // Try with empty prefix first; then `<root>.` for each detected
    // root. Standard HF AutoModelForSequenceClassification exports the
    // head at top level so the empty prefix is the common case.
    let roots = ["", "model.", "roberta.", "bert."];
    for name in candidate_names {
        for root in roots {
            let candidate = format!("{root}{name}");
            if let Ok(v) = st.tensor(&candidate) {
                return Ok(v);
            }
        }
    }
    Err(anyhow!("none of {candidate_names:?} found in safetensors"))
}

fn tensor_to_2d(view: TensorView<'_>) -> Result<Array2<f32>> {
    let shape = view.shape();
    if shape.len() != 2 {
        return Err(anyhow!("expected 2-D tensor, got shape {shape:?}"));
    }
    let data = view_to_f32(&view)?;
    Array2::from_shape_vec((shape[0], shape[1]), data)
        .map_err(|e| anyhow!("shape error building 2-D tensor: {e}"))
}

fn tensor_to_1d(view: TensorView<'_>) -> Result<Array1<f32>> {
    let shape = view.shape();
    if shape.len() != 1 {
        return Err(anyhow!("expected 1-D tensor, got shape {shape:?}"));
    }
    let data = view_to_f32(&view)?;
    Array1::from_shape_vec(shape[0], data).map_err(|e| anyhow!("shape error: {e}"))
}

fn view_to_f32(view: &TensorView<'_>) -> Result<Vec<f32>> {
    match view.dtype() {
        Dtype::F32 => {
            let raw = view.data();
            let n = raw.len() / 4;
            let mut out = Vec::with_capacity(n);
            for chunk in raw.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(out)
        }
        Dtype::F16 => {
            let raw = view.data();
            let n = raw.len() / 2;
            let mut out = Vec::with_capacity(n);
            for chunk in raw.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(f16_to_f32(bits));
            }
            Ok(out)
        }
        Dtype::BF16 => {
            let raw = view.data();
            let n = raw.len() / 2;
            let mut out = Vec::with_capacity(n);
            for chunk in raw.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(bf16_to_f32(bits));
            }
            Ok(out)
        }
        other => Err(anyhow!("unsupported dtype {other:?}")),
    }
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let mant = bits & 0x3ff;
    if exp == 0 {
        if mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        let val = (mant as f32) / 1024.0 / 16384.0;
        return if sign == 1 { -val } else { val };
    } else if exp == 0x1f {
        return if mant == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        };
    }
    let exp_f = (exp - 15) as f32;
    let val = (1.0 + (mant as f32) / 1024.0) * (2.0_f32).powf(exp_f);
    if sign == 1 { -val } else { val }
}

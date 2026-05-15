use std::path::Path;

use anyhow::{Context, Result, anyhow};
use ndarray::{Array1, Array2};
use safetensors::SafeTensors;
use safetensors::tensor::TensorView;
use sha2::{Digest, Sha256};

use super::config::BertConfig;

/// All weights needed for a BERT-class embedding forward pass.
pub struct BertWeights {
    pub word_embedding: Array2<f32>,             // (vocab, d)
    pub position_embedding: Array2<f32>,         // (max_pos, d)
    pub token_type_embedding: Array2<f32>,       // (type_vocab, d)
    pub embeddings_ln_w: Array1<f32>,            // (d,)
    pub embeddings_ln_b: Array1<f32>,            // (d,)
    pub layers: Vec<BertLayerWeights>,
    /// SHA-256 of the raw `model.safetensors` bytes. Bound into the SEV-SNP
    /// attestation report's `REPORT_DATA[0..32]` so the relying party can
    /// verify the CVM loaded these specific publicly-known weights — the
    /// openweight threat model GELO targets.
    pub model_identity: [u8; 32],
}

pub struct BertLayerWeights {
    // Q, K, V, O projections — kept as (in, out) so that offload_linear's
    // `input · W` shape works directly with `(n, in) · (in, out) → (n, out)`.
    pub wq: Array2<f32>, // (d, d)
    pub bq: Array1<f32>, // (d,)
    pub wk: Array2<f32>,
    pub bk: Array1<f32>,
    pub wv: Array2<f32>,
    pub bv: Array1<f32>,
    pub wo: Array2<f32>, // (d, d)
    pub bo: Array1<f32>, // (d,)

    pub attn_ln_w: Array1<f32>,
    pub attn_ln_b: Array1<f32>,

    pub w_ffn_up: Array2<f32>,   // (d, ffn)
    pub b_ffn_up: Array1<f32>,   // (ffn,)
    pub w_ffn_down: Array2<f32>, // (ffn, d)
    pub b_ffn_down: Array1<f32>, // (d,)

    pub ffn_ln_w: Array1<f32>,
    pub ffn_ln_b: Array1<f32>,
}

impl BertWeights {
    /// Load weights from a `model.safetensors` file. Accepts both
    /// `bert.*` and bare-prefixed naming conventions (sentence-transformers
    /// uses both depending on export script).
    pub fn from_safetensors(path: &Path, cfg: &BertConfig) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading safetensors from {}", path.display()))?;
        let model_identity: [u8; 32] = Sha256::digest(&bytes).into();
        let st = SafeTensors::deserialize(&bytes)
            .with_context(|| format!("deserializing safetensors at {}", path.display()))?;

        let prefix = detect_prefix(&st);

        let lookup =
            |name: &str| -> Result<TensorView<'_>> {
                let full = format!("{prefix}{name}");
                st.tensor(&full)
                    .with_context(|| format!("missing tensor {full}"))
            };

        let word_embedding = tensor_to_2d(lookup("embeddings.word_embeddings.weight")?)?;
        let position_embedding = tensor_to_2d(lookup("embeddings.position_embeddings.weight")?)?;
        let token_type_embedding =
            tensor_to_2d(lookup("embeddings.token_type_embeddings.weight")?)?;
        let embeddings_ln_w = tensor_to_1d(lookup("embeddings.LayerNorm.weight")?)?;
        let embeddings_ln_b = tensor_to_1d(lookup("embeddings.LayerNorm.bias")?)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for li in 0..cfg.num_hidden_layers {
            let base = format!("encoder.layer.{li}.");
            let read1 = |suffix: &str| -> Result<Array1<f32>> {
                let full = format!("{prefix}{base}{suffix}");
                tensor_to_1d(
                    st.tensor(&full)
                        .with_context(|| format!("missing tensor {full}"))?,
                )
            };
            let read2_t = |suffix: &str| -> Result<Array2<f32>> {
                // HuggingFace Linear stores weight as (out, in). For
                // `input · W` we want (in, out), so transpose on load.
                let full = format!("{prefix}{base}{suffix}");
                let view = st
                    .tensor(&full)
                    .with_context(|| format!("missing tensor {full}"))?;
                let m = tensor_to_2d(view)?;
                Ok(m.t().to_owned())
            };

            layers.push(BertLayerWeights {
                wq: read2_t("attention.self.query.weight")?,
                bq: read1("attention.self.query.bias")?,
                wk: read2_t("attention.self.key.weight")?,
                bk: read1("attention.self.key.bias")?,
                wv: read2_t("attention.self.value.weight")?,
                bv: read1("attention.self.value.bias")?,
                wo: read2_t("attention.output.dense.weight")?,
                bo: read1("attention.output.dense.bias")?,
                attn_ln_w: read1("attention.output.LayerNorm.weight")?,
                attn_ln_b: read1("attention.output.LayerNorm.bias")?,
                w_ffn_up: read2_t("intermediate.dense.weight")?,
                b_ffn_up: read1("intermediate.dense.bias")?,
                w_ffn_down: read2_t("output.dense.weight")?,
                b_ffn_down: read1("output.dense.bias")?,
                ffn_ln_w: read1("output.LayerNorm.weight")?,
                ffn_ln_b: read1("output.LayerNorm.bias")?,
            });
        }

        Ok(Self {
            word_embedding,
            position_embedding,
            token_type_embedding,
            embeddings_ln_w,
            embeddings_ln_b,
            layers,
            model_identity,
        })
    }
}

fn detect_prefix(st: &SafeTensors<'_>) -> String {
    let names: Vec<&str> = st.names().into_iter().map(String::as_str).collect();
    if names.iter().any(|n| n.starts_with("bert.")) {
        "bert.".to_string()
    } else if names.iter().any(|n| n.starts_with("roberta.")) {
        // XLM-RoBERTa-based rerankers (e.g. bge-reranker-v2-m3) export
        // their backbone under this prefix; the BERT forward shape
        // applies unchanged.
        "roberta.".to_string()
    } else {
        String::new()
    }
}

fn tensor_to_2d(view: TensorView<'_>) -> Result<Array2<f32>> {
    let shape = view.shape();
    if shape.len() != 2 {
        return Err(anyhow!(
            "expected 2-D tensor, got shape {:?}",
            shape
        ));
    }
    let (rows, cols) = (shape[0], shape[1]);
    let data = view_to_f32(&view)?;
    Array2::from_shape_vec((rows, cols), data)
        .map_err(|e| anyhow!("shape error building 2-D tensor: {e}"))
}

fn tensor_to_1d(view: TensorView<'_>) -> Result<Array1<f32>> {
    let shape = view.shape();
    if shape.len() != 1 {
        return Err(anyhow!(
            "expected 1-D tensor, got shape {:?}",
            shape
        ));
    }
    let data = view_to_f32(&view)?;
    Array1::from_shape_vec(shape[0], data)
        .map_err(|e| anyhow!("shape error building 1-D tensor: {e}"))
}

fn view_to_f32(view: &TensorView<'_>) -> Result<Vec<f32>> {
    use safetensors::tensor::Dtype;
    match view.dtype() {
        Dtype::F32 => {
            let raw = view.data();
            if raw.len() % 4 != 0 {
                return Err(anyhow!("f32 byte length not multiple of 4"));
            }
            let n = raw.len() / 4;
            let mut out = Vec::with_capacity(n);
            for chunk in raw.chunks_exact(4) {
                let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
                out.push(f32::from_le_bytes(arr));
            }
            Ok(out)
        }
        Dtype::F16 => {
            let raw = view.data();
            if raw.len() % 2 != 0 {
                return Err(anyhow!("f16 byte length not multiple of 2"));
            }
            let n = raw.len() / 2;
            let mut out = Vec::with_capacity(n);
            for chunk in raw.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(f16_to_f32(bits));
            }
            Ok(out)
        }
        other => Err(anyhow!("unsupported dtype {:?}", other)),
    }
}

/// Bit-exact IEEE-754 binary16 → binary32 conversion.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let mant = bits & 0x3ff;
    if exp == 0 {
        if mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        // Subnormal
        let val = (mant as f32) / 1024.0 / 16384.0; // /2^14
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

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use ndarray::{Array1, Array2};
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};
use sha2::{Digest, Sha256};

use super::config::DecoderConfig;

/// Weights for a decoder-LLM-as-embedder (Qwen3 / LLaMA / Mistral family).
/// No biases anywhere (Qwen3 default). Linear weights are transposed on load
/// so each `Array2<f32>` has shape `(in, out)` for `input · W`.
pub struct DecoderWeights {
    pub token_embedding: Array2<f32>, // (vocab, hidden)
    pub final_norm: Array1<f32>,      // (hidden,)
    pub layers: Vec<DecoderLayerWeights>,
    /// SHA-256 of the concatenated raw safetensors shard bytes (in the order
    /// `paths` was passed to [`Self::from_safetensors`]). Bound into the
    /// SEV-SNP attestation report's `REPORT_DATA[0..32]` so the relying party
    /// can verify the CVM loaded these specific publicly-known weights —
    /// the openweight threat model GELO targets.
    pub model_identity: [u8; 32],
}

pub struct DecoderLayerWeights {
    pub norm_attn: Array1<f32>,    // (hidden,)
    pub wq: Array2<f32>,           // (hidden, q_dim)
    pub wk: Array2<f32>,           // (hidden, kv_dim)
    pub wv: Array2<f32>,           // (hidden, kv_dim)
    pub wo: Array2<f32>,           // (q_dim, hidden)

    pub norm_ffn: Array1<f32>,     // (hidden,)
    pub w_gate: Array2<f32>,       // (hidden, intermediate)
    pub w_up: Array2<f32>,         // (hidden, intermediate)
    pub w_down: Array2<f32>,       // (intermediate, hidden)

    /// Qwen3 added per-head RMSNorm on Q and K **before** RoPE
    /// (`self_attn.q_norm.weight`, `self_attn.k_norm.weight`, each
    /// shape `(head_dim,)`). When loaded from a Qwen3 checkpoint these
    /// are `Some(_)` and the forward path applies a head-wise RMSNorm
    /// to Q / K before the rotary step. Qwen2 / LLaMA / Mistral
    /// checkpoints lack these tensors → `None`, and the forward path
    /// skips the norm step, preserving byte-for-byte parity with the
    /// pre-Qwen3 behaviour.
    pub q_norm: Option<Array1<f32>>,
    pub k_norm: Option<Array1<f32>>,
}

impl DecoderWeights {
    /// Load decoder weights from one or many safetensors files.
    pub fn from_safetensors(paths: &[&Path], cfg: &DecoderConfig) -> Result<Self> {
        // Read all shards into a single (name → bytes) map of owned byte vecs.
        // Borrow them via Vec<SafeTensors> kept alive for the whole function.
        let buffers: Vec<Vec<u8>> = paths
            .iter()
            .map(|p| {
                std::fs::read(p)
                    .with_context(|| format!("reading safetensors shard {}", p.display()))
            })
            .collect::<Result<_>>()?;
        let mut hasher = Sha256::new();
        for b in &buffers {
            hasher.update(b);
        }
        let model_identity: [u8; 32] = hasher.finalize().into();
        let shards: Vec<SafeTensors<'_>> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| {
                SafeTensors::deserialize(b).with_context(|| {
                    format!("deserializing safetensors shard {}", paths[i].display())
                })
            })
            .collect::<Result<_>>()?;

        let lookup_view = |name: &str| -> Result<TensorView<'_>> {
            for shard in &shards {
                if let Ok(v) = shard.tensor(name) {
                    return Ok(v);
                }
            }
            Err(anyhow!("missing tensor: {name}"))
        };

        let prefix = detect_prefix(&shards);
        let resolve = |name: &str| -> String {
            format!("{prefix}{name}")
        };

        let token_embedding =
            tensor_to_2d(lookup_view(&resolve("embed_tokens.weight"))?)?;
        let final_norm = tensor_to_1d(lookup_view(&resolve("norm.weight"))?)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for li in 0..cfg.num_hidden_layers {
            let base = format!("{prefix}layers.{li}.");
            let read1 = |name: &str| -> Result<Array1<f32>> {
                let full = format!("{base}{name}");
                tensor_to_1d(
                    lookup_view(&full).with_context(|| format!("missing tensor {full}"))?,
                )
            };
            let read1_opt = |name: &str| -> Result<Option<Array1<f32>>> {
                let full = format!("{base}{name}");
                match lookup_view(&full) {
                    Ok(view) => Ok(Some(tensor_to_1d(view)?)),
                    Err(_) => Ok(None),
                }
            };
            let read2_t = |name: &str| -> Result<Array2<f32>> {
                let full = format!("{base}{name}");
                let view = lookup_view(&full)
                    .with_context(|| format!("missing tensor {full}"))?;
                let m = tensor_to_2d(view)?;
                Ok(m.t().to_owned())
            };

            layers.push(DecoderLayerWeights {
                norm_attn: read1("input_layernorm.weight")?,
                wq: read2_t("self_attn.q_proj.weight")?,
                wk: read2_t("self_attn.k_proj.weight")?,
                wv: read2_t("self_attn.v_proj.weight")?,
                wo: read2_t("self_attn.o_proj.weight")?,
                norm_ffn: read1("post_attention_layernorm.weight")?,
                w_gate: read2_t("mlp.gate_proj.weight")?,
                w_up: read2_t("mlp.up_proj.weight")?,
                w_down: read2_t("mlp.down_proj.weight")?,
                // Qwen3 QK-norm — present in Qwen3-* checkpoints,
                // absent in Qwen2 / LLaMA / Mistral. Loader treats
                // the tensors as optional so back-compat is byte-
                // identical for older models.
                q_norm: read1_opt("self_attn.q_norm.weight")?,
                k_norm: read1_opt("self_attn.k_norm.weight")?,
            });
        }

        Ok(Self {
            token_embedding,
            final_norm,
            layers,
            model_identity,
        })
    }
}

fn detect_prefix(shards: &[SafeTensors<'_>]) -> String {
    for shard in shards {
        for n in shard.names() {
            if n.starts_with("model.") {
                return "model.".to_string();
            }
        }
    }
    String::new()
}

fn tensor_to_2d(view: TensorView<'_>) -> Result<Array2<f32>> {
    let shape = view.shape();
    if shape.len() != 2 {
        return Err(anyhow!("expected 2-D tensor, got shape {shape:?}"));
    }
    let data = view_to_f32(&view)?;
    Array2::from_shape_vec((shape[0], shape[1]), data)
        .map_err(|e| anyhow!("shape error: {e}"))
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
    // bf16 is just the upper 16 bits of f32.
    let extended = (bits as u32) << 16;
    f32::from_bits(extended)
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

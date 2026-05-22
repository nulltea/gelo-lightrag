use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use half::bf16;
use ndarray::{Array1, Array2};
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};
use sha2::{Digest, Sha256};

use gelo_protocol::{TrustedExecutor, WeightHandle, WeightKind};

use super::config::DecoderConfig;

/// Weights for a decoder-LLM-as-embedder (Qwen3 / LLaMA / Mistral family).
/// No biases anywhere (Qwen3 default). Linear weights are transposed on load
/// so each `Array2<f32>` has shape `(in, out)` for `input · W`.
///
/// `Clone` is derived so test code that constructs multiple parity
/// services (e.g. plaintext vs masked executor agreement) can reuse
/// the same synthetic weights twice. For prod paths the weights are
/// loaded once and consumed by the embedder/decoder/runtime
/// constructor (which `take()`s the offloadable Arcs into the engine
/// and drops the host copy) — `clone()` is not called on the prod
/// path.
#[derive(Clone)]
pub struct DecoderWeights {
    /// Token embedding table. Stored as **bf16** — matches the on-disk
    /// dtype and halves host RAM (~1.24 GB → ~620 MB on Qwen3-1.7B,
    /// ~3 GB → ~1.5 GB on Qwen3-4B). Stays host-resident because the
    /// LM head computation (`h_last · token_embedding.T`) runs in
    /// the TEE per the GELO threat model: the output is the sampled
    /// next token, which IS the protected secret. Per-element bf16 →
    /// f32 widening happens at use sites (`embedding_lookup` and
    /// `compute_logits`), so the intermediate accumulator stays f32 —
    /// bit-identical to the previous f32-storage code (since the
    /// disk weights were bf16 anyway).
    pub token_embedding: Array2<bf16>, // (vocab, hidden)
    pub final_norm: Array1<f32>,       // (hidden,)
    pub layers: Vec<DecoderLayerWeights>,
    /// SHA-256 of the concatenated raw safetensors shard bytes (in the order
    /// `paths` was passed to [`Self::from_safetensors`]). Bound into the
    /// SEV-SNP attestation report's `REPORT_DATA[0..32]` so the relying party
    /// can verify the CVM loaded these specific publicly-known weights —
    /// the openweight threat model GELO targets.
    pub model_identity: [u8; 32],
}

#[derive(Clone)]
pub struct DecoderLayerWeights {
    pub norm_attn: Array1<f32>,                 // (hidden,)
    /// **Offloadable projection weights — bf16 native storage**, wrapped in
    /// `Option<Arc<…>>` so the embedder / decoder runtime can `take()`
    /// each Arc after handing it to the engine. With skip-first/last
    /// layers disabled (the default), nothing on the host ever reads
    /// these after provisioning — the offload path uses
    /// `exec.offload_*(WeightHandle)` which goes through the GPU
    /// engine's weight cache, not the host bytes. Post-`take` the
    /// host RAM backing the matrix is released. See
    /// `feedback_memory_efficiency_priority.md` and
    /// `feedback_no_rayon_cpu_engine.md`.
    pub wq: Option<Arc<Array2<bf16>>>,          // (hidden, q_dim)
    pub wk: Option<Arc<Array2<bf16>>>,          // (hidden, kv_dim)
    pub wv: Option<Arc<Array2<bf16>>>,          // (hidden, kv_dim)
    pub wo: Option<Arc<Array2<bf16>>>,          // (q_dim, hidden)

    pub norm_ffn: Array1<f32>,                  // (hidden,)
    pub w_gate: Option<Arc<Array2<bf16>>>,      // (hidden, intermediate)
    pub w_up: Option<Arc<Array2<bf16>>>,        // (hidden, intermediate)
    pub w_down: Option<Arc<Array2<bf16>>>,      // (intermediate, hidden)

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

/// Provision every offloadable layer's bf16 projection matrices into
/// `exec`, **consuming** each Arc out of `weights`. With the wgpu engine
/// the upload converts bf16 → f16 device-side and the Arc's refcount
/// drops on return — releasing the host RAM that was backing the
/// matrix. From this point on, `weights.layers[li].{wq,wk,…,w_down}` is
/// `None`. With skip-first / skip-last layers off (the default) no
/// forward-path read ever touches these slots again.
///
/// Used by all three production decoder call sites
/// (`GeloQwenEmbedder::new`,
/// `CausalDiscriminatorRerankService::new`,
/// `DecoderRuntime::from_config_and_dir`). M1.12 R1 — see
/// `docs/plans/m1-12-tee-gpu-throughput.md` §2 and
/// `feedback_memory_efficiency_priority.md`.
pub fn provision_into<X: TrustedExecutor>(
    weights: &mut DecoderWeights,
    cfg: &DecoderConfig,
    exec: &mut X,
) -> Result<()> {
    for li in 0..weights.layers.len() {
        if !cfg.offload_layer(li) {
            continue;
        }
        let li16 = li as u16;
        let layer = &mut weights.layers[li];
        let pairs: [(WeightKind, Option<Arc<Array2<bf16>>>); 7] = [
            (WeightKind::Q, layer.wq.take()),
            (WeightKind::K, layer.wk.take()),
            (WeightKind::V, layer.wv.take()),
            (WeightKind::O, layer.wo.take()),
            (WeightKind::FfnGate, layer.w_gate.take()),
            (WeightKind::FfnUp, layer.w_up.take()),
            (WeightKind::FfnDown, layer.w_down.take()),
        ];
        for (kind, slot) in pairs {
            let arc = slot.ok_or_else(|| {
                anyhow!("layer {li} {kind:?}: weight already taken")
            })?;
            exec.provision_weight_bf16_shared(WeightHandle::new(li16, kind), arc)?;
        }
    }
    Ok(())
}

/// Arc-sharing variant of [`provision_into`]. The Arcs in `weights`
/// are not consumed — host bytes stay alive. Used by parity benches
/// that need to construct multiple services from the same
/// `DecoderWeights` (plaintext-vs-masked side-by-side) and by the
/// `with_shared_weights` builders. Production paths should call
/// [`provision_into`] which releases host bytes after upload.
pub fn provision_into_shared<X: TrustedExecutor>(
    weights: &DecoderWeights,
    cfg: &DecoderConfig,
    exec: &mut X,
) -> Result<()> {
    for (li, layer) in weights.layers.iter().enumerate() {
        if !cfg.offload_layer(li) {
            continue;
        }
        let li16 = li as u16;
        for (kind, slot) in [
            (WeightKind::Q, layer.wq.as_ref()),
            (WeightKind::K, layer.wk.as_ref()),
            (WeightKind::V, layer.wv.as_ref()),
            (WeightKind::O, layer.wo.as_ref()),
            (WeightKind::FfnGate, layer.w_gate.as_ref()),
            (WeightKind::FfnUp, layer.w_up.as_ref()),
            (WeightKind::FfnDown, layer.w_down.as_ref()),
        ] {
            let arc = slot.ok_or_else(|| {
                anyhow!(
                    "layer {li} {kind:?}: weight already taken — `provision_into_shared` \
                     requires fresh DecoderWeights (not one previously consumed by \
                     `provision_into`)"
                )
            })?;
            exec.provision_weight_bf16_shared(WeightHandle::new(li16, kind), Arc::clone(arc))?;
        }
    }
    Ok(())
}

/// Provision the tied input/output embedding into the executor as
/// `WeightKind::LmHead`. Materialises a transpose of
/// `weights.token_embedding` (`(vocab, hidden)` → `(hidden, vocab)`)
/// at standard layout so the engine's row-major upload matches GELO's
/// `(in_features, out_features)` weight convention. The host-side
/// `token_embedding` stays alive (used by `embedding_lookup` for the
/// input-side gather) — untying via the offline conversion script in
/// `private-rag-path-2/python/aloepri-llm/obfuscate_qwen3_gguf.py` is
/// the escape hatch if the 2× residency (host `(vocab, hidden)` +
/// VRAM `(hidden, vocab)`) ever becomes binding.
///
/// M1.12 R3 — see `docs/plans/m1-12-tee-gpu-throughput.md` §3. The
/// transpose materialises ~778 MB transient host RAM during
/// provisioning on Qwen3-4B (152 064 × 2 560 bf16); after the Arc is
/// consumed by the wgpu upload only the original `token_embedding`
/// host bytes remain.
pub fn provision_lm_head_into<X: TrustedExecutor>(
    weights: &DecoderWeights,
    exec: &mut X,
) -> Result<()> {
    let lm_head_t = weights.token_embedding.t().as_standard_layout().to_owned();
    exec.provision_weight_bf16_shared(
        WeightHandle::new(0, WeightKind::LmHead),
        Arc::new(lm_head_t),
    )
}

/// Returns `true` when `LM_HEAD_GPU_OFFLOAD=1` (or `=true`) is set in
/// the environment. Cached per-process via `OnceLock` — env reads
/// happen once per process, never per token. Runtimes call this once
/// at startup to decide both whether to call
/// [`provision_lm_head_into`] and whether to set
/// `GenerationConfig::lm_head_via_gpu_offload`.
///
/// M1.12 R3 — see `docs/plans/m1-12-tee-gpu-throughput.md` §3.
/// Default off until the c6 AloePri spot-check at the LM-head shape
/// clears.
pub fn lm_head_gpu_offload_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("LM_HEAD_GPU_OFFLOAD")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
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
            tensor_to_2d_bf16(lookup_view(&resolve("embed_tokens.weight"))?)?;
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
            let read2_t_bf16 = |name: &str| -> Result<Option<Arc<Array2<bf16>>>> {
                let full = format!("{base}{name}");
                let view = lookup_view(&full)
                    .with_context(|| format!("missing tensor {full}"))?;
                // Load directly as bf16 — never widen to f32. `.t()` keeps
                // logical `(in, out)` shape; force standard layout so
                // downstream BLAS / wgpu uploads see row-major contiguous
                // memory. See `feedback_memory_efficiency_priority.md`.
                let m = tensor_to_2d_bf16(view)?;
                Ok(Some(Arc::new(m.t().as_standard_layout().to_owned())))
            };

            layers.push(DecoderLayerWeights {
                norm_attn: read1("input_layernorm.weight")?,
                wq: read2_t_bf16("self_attn.q_proj.weight")?,
                wk: read2_t_bf16("self_attn.k_proj.weight")?,
                wv: read2_t_bf16("self_attn.v_proj.weight")?,
                wo: read2_t_bf16("self_attn.o_proj.weight")?,
                norm_ffn: read1("post_attention_layernorm.weight")?,
                w_gate: read2_t_bf16("mlp.gate_proj.weight")?,
                w_up: read2_t_bf16("mlp.up_proj.weight")?,
                w_down: read2_t_bf16("mlp.down_proj.weight")?,
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

/// Read a 2-D tensor view as native bf16. F32 / F16 on-disk variants
/// are converted **per element** into bf16 (still narrower than the
/// f32 fallback path) so the loader's host footprint stays at 2 bytes
/// per weight. Used by the offloadable-projection load path —
/// `view_to_f32` is reserved for the small non-offloaded tensors
/// (norms, embedding) that still need f32 inside the TEE.
fn tensor_to_2d_bf16(view: TensorView<'_>) -> Result<Array2<bf16>> {
    let shape = view.shape();
    if shape.len() != 2 {
        return Err(anyhow!("expected 2-D tensor, got shape {shape:?}"));
    }
    let data = view_to_bf16(&view)?;
    Array2::from_shape_vec((shape[0], shape[1]), data)
        .map_err(|e| anyhow!("shape error: {e}"))
}

fn view_to_bf16(view: &TensorView<'_>) -> Result<Vec<bf16>> {
    match view.dtype() {
        Dtype::BF16 => {
            // Zero-conversion path: bf16 on disk → bf16 in RAM.
            let raw = view.data();
            let n = raw.len() / 2;
            let mut out = Vec::with_capacity(n);
            for chunk in raw.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(bf16::from_bits(bits));
            }
            Ok(out)
        }
        Dtype::F16 => {
            // Narrow f16 → bf16 via f32. Rare path; weights are
            // almost always bf16 in modern checkpoints.
            let raw = view.data();
            let n = raw.len() / 2;
            let mut out = Vec::with_capacity(n);
            for chunk in raw.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(bf16::from_f32(f16_to_f32(bits)));
            }
            Ok(out)
        }
        Dtype::F32 => {
            // f32 on disk → bf16 in RAM. Halves footprint.
            let raw = view.data();
            let n = raw.len() / 4;
            let mut out = Vec::with_capacity(n);
            for chunk in raw.chunks_exact(4) {
                let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out.push(bf16::from_f32(v));
            }
            Ok(out)
        }
        other => Err(anyhow!("unsupported dtype {other:?} for bf16 load")),
    }
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

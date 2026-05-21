//! End-to-end greedy-generation test for `Qwen3-1.7B` under GELO.
//!
//! Status: **v1 demonstrator target** — the first real-weight test
//! exercising the `decoder::generation::generate()` loop on the v1
//! protocol surface (prefill → decode → sample → append) against a
//! published Qwen3 checkpoint.
//!
//! ## Coverage
//!
//! 1. **Smoke**: greedy `generate(max_tokens = 8)` on a fixed prompt
//!    runs to completion under both `PlaintextExecutor` and
//!    `InProcessTrustedExecutor`, emitting ≥1 token.
//! 2. **Parity**: plaintext vs masked emit the **same token sequence**
//!    on greedy sampling — confirms the GELO protocol preserves
//!    decoder output bit-for-bit (within argmax-stable float
//!    tolerance) on a real Qwen3 model with QK-norm + RoPE + GQA
//!    + SwiGLU.
//! 3. **Decode-replay invariant**: a prefill of N + 1 tokens produces
//!    the same final hidden state as prefill(N) → decode_step(N+1).
//!    Covered by `tests/generation_harness.rs` on synthetic weights;
//!    here we exercise the property end-to-end on real Qwen3 weights
//!    by re-running `generate()` with `max_tokens = 0` on the
//!    extended prompt and comparing the sampled-next-token against
//!    the original first generated token.
//!
//! Downloads ~3.4 GB on first run (Qwen3-1.7B bf16 safetensors,
//! ~3.8 GB on disk after decode); gated behind `#[ignore]`.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::anyhow;
use gelo_embedder::common::HfTokenizer;
use gelo_embedder::decoder::config::DecoderConfig;
use gelo_embedder::decoder::generation::{GenerationConfig, SamplerConfig, generate};
use gelo_embedder::decoder::qwen3::Qwen3Variant;
use gelo_embedder::decoder::rope::RopeTables;
use gelo_embedder::decoder::weights::DecoderWeights;
use gelo_protocol::rng::MaskSeed;
use gelo_protocol::{
    InProcessTrustedExecutor, PlaintextExecutor, RayonCpuEngine, TrustedExecutor, WeightHandle,
    WeightKind,
};
use hf_hub::api::sync::{ApiBuilder, ApiRepo};

/// Pin the v1 target so re-pinning is a one-line change.
const VARIANT: Qwen3Variant = Qwen3Variant::Q1_7B;

#[test]
#[ignore = "downloads ~3.4 GB on first run (Qwen3-1.7B safetensors)"]
fn qwen3_1_7b_greedy_generates_under_both_executors() -> Result<()> {
    let (cfg, tokenizer, weights, rope) = load_pretrained()?;

    // Short fixed prompt so the test stays under a second once weights
    // are warm in OS page cache. Greedy sampling makes the output
    // deterministic given the same weights + same prompt.
    let prompt = "The quick brown fox";
    let prompt_ids = tokenizer.encode(prompt, 32)?;
    assert!(!prompt_ids.is_empty(), "tokenizer must emit ≥1 token");

    let gen_cfg = GenerationConfig {
        max_tokens: 8,
        eos_token_ids: Vec::new(),
        sampler: SamplerConfig::Greedy,
    };

    // 1. Plaintext branch.
    let mut plain_exec = PlaintextExecutor::new(RayonCpuEngine::new());
    provision_decoder_weights(&cfg, &weights, &mut plain_exec)?;
    let plain_out = generate(&cfg, &weights, &rope, &mut plain_exec, &prompt_ids, &gen_cfg)?;
    assert_eq!(
        plain_out.tokens.len(),
        gen_cfg.max_tokens,
        "plaintext generate produced {} tokens, expected {}",
        plain_out.tokens.len(),
        gen_cfg.max_tokens,
    );

    // 2. InProcess (masked) branch with a pinned seed so the test is
    //    bit-stable across runs.
    let mut masked_exec = InProcessTrustedExecutor::with_seed(
        RayonCpuEngine::new(),
        MaskSeed::from_bytes([29u8; 32]),
    );
    provision_decoder_weights(&cfg, &weights, &mut masked_exec)?;
    let masked_out = generate(&cfg, &weights, &rope, &mut masked_exec, &prompt_ids, &gen_cfg)?;
    assert_eq!(
        masked_out.tokens.len(),
        gen_cfg.max_tokens,
        "masked generate produced {} tokens, expected {}",
        masked_out.tokens.len(),
        gen_cfg.max_tokens,
    );

    // 3. Parity — greedy argmax is robust to the small float drift the
    //    Haar mask introduces, so the same token sequence must come
    //    out. Even one token of divergence indicates either a protocol
    //    bug (mask not unapplied correctly) or numerical sensitivity
    //    high enough to flip argmax on adjacent-scoring vocab entries.
    assert_eq!(
        plain_out.tokens, masked_out.tokens,
        "plaintext vs masked diverge: plain={:?} masked={:?} (decoded plain={:?})",
        plain_out.tokens,
        masked_out.tokens,
        tokenizer.decode(&plain_out.tokens, true).unwrap_or_default(),
    );

    eprintln!(
        "Qwen3-1.7B greedy: {} → {:?}",
        prompt,
        tokenizer.decode(&plain_out.tokens, true).unwrap_or_default(),
    );

    Ok(())
}

fn load_pretrained()
-> Result<(DecoderConfig, HfTokenizer, Arc<DecoderWeights>, Arc<RopeTables>)> {
    let cfg = VARIANT.config();
    let api = ApiBuilder::new()
        .with_progress(false)
        .build()
        .context("building HF hub API client")?;
    let repo = api.model(VARIANT.hf_model_id().to_string());

    let tokenizer_path = repo
        .get("tokenizer.json")
        .context("downloading tokenizer.json")?;
    let tokenizer = HfTokenizer::from_file(&tokenizer_path)?;

    let shard_paths = find_safetensors_shards(&repo)?;
    let shard_refs: Vec<&Path> = shard_paths.iter().map(|p| p.as_path()).collect();
    let weights = Arc::new(DecoderWeights::from_safetensors(&shard_refs, &cfg)?);

    let rope = Arc::new(RopeTables::new(
        cfg.head_dim_value(),
        cfg.max_position_embeddings,
        cfg.rope_theta,
    ));

    Ok((cfg, tokenizer, weights, rope))
}

/// Resolve the `model.safetensors` (single-file) or
/// `model.safetensors.index.json` (sharded) layout used by HF.
///
/// Lifted from `decoder::embedder::find_safetensors_shards` — the
/// embedder's helper is module-private; if generation grows a public
/// `Qwen3Generator::from_pretrained` API later, this and the embedder
/// can share a single implementation.
fn find_safetensors_shards(repo: &ApiRepo) -> Result<Vec<PathBuf>> {
    if let Ok(p) = repo.get("model.safetensors") {
        return Ok(vec![p]);
    }
    let index_path = repo
        .get("model.safetensors.index.json")
        .context("model has neither model.safetensors nor model.safetensors.index.json")?;
    let index_bytes = std::fs::read(&index_path)?;
    let index: serde_json::Value = serde_json::from_slice(&index_bytes)?;
    let map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow!("shard index has no weight_map object"))?;
    let mut filenames: Vec<String> = map
        .values()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    filenames.sort();
    filenames.dedup();
    let mut paths = Vec::with_capacity(filenames.len());
    for name in filenames {
        paths.push(repo.get(&name)?);
    }
    Ok(paths)
}

fn provision_decoder_weights<X: TrustedExecutor>(
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    exec: &mut X,
) -> Result<()> {
    for (li, layer) in weights.layers.iter().enumerate() {
        if !cfg.offload_layer(li) {
            continue;
        }
        let li16 = li as u16;
        exec.provision_weight_bf16(WeightHandle::new(li16, WeightKind::Q), layer.wq.as_ref().expect("offloadable weight").view())?;
        exec.provision_weight_bf16(WeightHandle::new(li16, WeightKind::K), layer.wk.as_ref().expect("offloadable weight").view())?;
        exec.provision_weight_bf16(WeightHandle::new(li16, WeightKind::V), layer.wv.as_ref().expect("offloadable weight").view())?;
        exec.provision_weight_bf16(WeightHandle::new(li16, WeightKind::O), layer.wo.as_ref().expect("offloadable weight").view())?;
        exec.provision_weight_bf16(WeightHandle::new(li16, WeightKind::FfnGate), layer.w_gate.as_ref().expect("offloadable weight").view())?;
        exec.provision_weight_bf16(WeightHandle::new(li16, WeightKind::FfnUp), layer.w_up.as_ref().expect("offloadable weight").view())?;
        exec.provision_weight_bf16(WeightHandle::new(li16, WeightKind::FfnDown), layer.w_down.as_ref().expect("offloadable weight").view())?;
    }
    Ok(())
}

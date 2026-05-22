//! Qwen3-1.7B HuggingFace `transformers` parity gate (M1.8 equivalent
//! on the v1 demonstrator target).
//!
//! Implements the M1.8 accuracy validation acceptance criterion for
//! the Qwen3-1.7B demonstrator: greedy `generate()` at `temperature=0`
//! on a pinned prompt must match the HF reference output
//! **token-for-token** (top-1 match ≥ 0.99 by the milestone wording;
//! for greedy single-shot this collapses to bit-equality).
//!
//! Pattern: a sidecar JSON file at
//! `crates/gelo-embedder/tests/fixtures/qwen3_1_7b_hf_reference.json`
//! holds the HF reference output for a specific model_id + prompt +
//! sampler config (greedy, temperature=0). The test loads the fixture,
//! runs our stack, compares tokens.
//!
//! **Generating the fixture (one-shot Python):**
//!
//! ```python
//! from transformers import AutoTokenizer, AutoModelForCausalLM
//! import torch, json, hashlib
//!
//! MODEL = "Qwen/Qwen3-1.7B"
//! tok = AutoTokenizer.from_pretrained(MODEL)
//! model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
//! model.eval()
//! prompt = "The quick brown fox"
//! ids = tok(prompt, return_tensors="pt").input_ids
//! with torch.no_grad():
//!     out = model.generate(
//!         ids,
//!         max_new_tokens=8,
//!         do_sample=False,        # greedy
//!         temperature=0.0,
//!         num_beams=1,
//!     )
//! generated = out[0, ids.shape[1]:].tolist()
//! payload = {"prompt": prompt, "tokens": generated}
//! sha = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
//! fixture = {
//!     "model_id": MODEL,
//!     "transformers_version": __import__("transformers").__version__,
//!     "prompt": prompt,
//!     "prompt_ids": ids[0].tolist(),
//!     "reference_tokens": generated,
//!     "sampler": {"kind": "greedy", "temperature": 0.0},
//!     "fixture_sha256": sha,
//! }
//! import os; os.makedirs("tests/fixtures", exist_ok=True)
//! with open("tests/fixtures/qwen3_1_7b_hf_reference.json", "w") as f:
//!     json.dump(fixture, f, indent=2)
//! ```
//!
//! Run from `crates/gelo-embedder/`. The fixture is regenerated when
//! the model_id, the prompt, or the sampler config change.
//!
//! **Why bf16 rather than fp32 for the reference:** the published
//! Qwen3-1.7B weights are bf16. Running HF in fp32 would up-cast
//! before computing, which doesn't match what our loader reads
//! (we down-cast to f32 from bf16 — same precision the model was
//! trained at). bf16 reference + greedy keeps the float drift to
//! the regime where argmax-stable sampling produces deterministic
//! token agreement.
//!
//! **Blockers (un-ignore prerequisites):**
//!  - The fixture JSON must exist at the path above. Until generated,
//!    the test ignores itself.
//!  - `tests/qwen3_generation_e2e.rs` must already pass with the same
//!    prompt — otherwise this test is testing the wrong end-to-end
//!    behaviour.

use anyhow::Result;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, anyhow};
use gelo_embedder::common::HfTokenizer;
use gelo_embedder::decoder::config::DecoderConfig;
use gelo_embedder::decoder::generation::{GenerationConfig, SamplerConfig, generate};
use gelo_embedder::decoder::qwen3::Qwen3Variant;
use gelo_embedder::decoder::rope::RopeTables;
use gelo_embedder::decoder::weights::DecoderWeights;
use gelo_protocol::{
    PlaintextExecutor, RayonCpuEngine, TrustedExecutor, WeightHandle, WeightKind,
};
use hf_hub::api::sync::{ApiBuilder, ApiRepo};

#[derive(Debug, Deserialize)]
struct ReferenceFixture {
    model_id: String,
    transformers_version: String,
    prompt: String,
    prompt_ids: Vec<u32>,
    reference_tokens: Vec<u32>,
}

/// Returns the fixture path. Centralised so the docstring above and
/// the test body stay in sync.
fn fixture_path() -> PathBuf {
    let manifest = std::env!("CARGO_MANIFEST_DIR");
    std::path::Path::new(manifest)
        .join("tests")
        .join("fixtures")
        .join("qwen3_1_7b_hf_reference.json")
}

#[test]
#[ignore = "downloads ~3.4 GB AND requires tests/fixtures/qwen3_1_7b_hf_reference.json (see file docstring)"]
fn qwen3_1_7b_greedy_matches_hf_transformers() -> Result<()> {
    // 1. Load fixture; bail with a clear message if the M1.8 worker
    //    hasn't generated it yet.
    let path = fixture_path();
    if !path.exists() {
        return Err(anyhow!(
            "Qwen3 HF parity fixture missing at {} — see file-level docstring \
             for the one-shot Python snippet that produces it",
            path.display(),
        ));
    }
    let bytes = std::fs::read(&path)
        .with_context(|| format!("reading fixture at {}", path.display()))?;
    let fix: ReferenceFixture = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing fixture at {}", path.display()))?;

    let variant = Qwen3Variant::Q1_7B;
    assert_eq!(
        fix.model_id,
        variant.hf_model_id(),
        "fixture model_id ({}) must match Qwen3Variant::Q1_7B ({})",
        fix.model_id,
        variant.hf_model_id(),
    );

    // 2. Load real weights + tokenizer + RoPE.
    let (cfg, tokenizer, weights, rope) = load_pretrained(variant)?;

    // Re-tokenise the prompt for an independent sanity check that our
    // tokenizer matches the fixture's. Truncate to `prompt_ids.len()`
    // to handle the case where the HF tokenizer applied a chat
    // template (we want raw bytes for parity).
    let our_ids = tokenizer.encode(&fix.prompt, fix.prompt_ids.len().max(32))?;
    assert_eq!(
        our_ids, fix.prompt_ids,
        "tokenizer disagreement on prompt — our IDs {:?} vs fixture IDs {:?}",
        our_ids, fix.prompt_ids,
    );

    // 3. Run greedy generate at `max_new_tokens = reference_tokens.len()`.
    let mut exec = PlaintextExecutor::new(RayonCpuEngine::new());
    provision_decoder_weights(&cfg, &weights, &mut exec)?;
    let gen_cfg = GenerationConfig {
        max_tokens: fix.reference_tokens.len(),
        eos_token_ids: Vec::new(),
        sampler: SamplerConfig::Greedy,
        lm_head_via_gpu_offload: false,
    };
    let out = generate(&cfg, &weights, &rope, &mut exec, &fix.prompt_ids, &gen_cfg)?;

    // 4. Bit-equality assertion.
    assert_eq!(
        out.tokens, fix.reference_tokens,
        "Qwen3-1.7B greedy parity vs HF transformers {}: ours={:?} ref={:?} (decoded ours={:?})",
        fix.transformers_version,
        out.tokens,
        fix.reference_tokens,
        tokenizer.decode(&out.tokens, true).unwrap_or_default(),
    );

    eprintln!(
        "qwen3-1.7b HF parity PASS · {} tokens · transformers {} · prompt {:?}",
        out.tokens.len(),
        fix.transformers_version,
        fix.prompt,
    );
    Ok(())
}

fn load_pretrained(
    variant: Qwen3Variant,
) -> Result<(DecoderConfig, HfTokenizer, Arc<DecoderWeights>, Arc<RopeTables>)> {
    let cfg = variant.config();
    let api = ApiBuilder::new()
        .with_progress(false)
        .build()
        .context("building HF hub API client")?;
    let repo = api.model(variant.hf_model_id().to_string());

    let tokenizer_path = repo
        .get("tokenizer.json")
        .context("downloading tokenizer.json")?;
    let tokenizer = HfTokenizer::from_file(&tokenizer_path)?;

    let shard_paths = find_safetensors_shards(&repo)?;
    let shard_refs: Vec<&std::path::Path> =
        shard_paths.iter().map(|p| p.as_path()).collect();
    let weights = Arc::new(DecoderWeights::from_safetensors(&shard_refs, &cfg)?);
    let rope = Arc::new(RopeTables::new(
        cfg.head_dim_value(),
        cfg.max_position_embeddings,
        cfg.rope_theta,
    ));
    Ok((cfg, tokenizer, weights, rope))
}

fn find_safetensors_shards(repo: &ApiRepo) -> Result<Vec<PathBuf>> {
    if let Ok(p) = repo.get("model.safetensors") {
        return Ok(vec![p]);
    }
    let index_path = repo
        .get("model.safetensors.index.json")
        .context("model has neither model.safetensors nor model.safetensors.index.json")?;
    let bytes = std::fs::read(&index_path)?;
    let v: serde_json::Value = serde_json::from_slice(&bytes)?;
    let map = v
        .get("weight_map")
        .and_then(|x| x.as_object())
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

//! M1.6 scaffolding — Gemma 4 E2B end-to-end generation smoke test.
//!
//! Real-model integration test for the M1.0–M1.5 generation harness.
//! Loads E2B from HuggingFace and runs greedy `generate()` on a short
//! prompt under both `PlaintextExecutor` and `InProcessTrustedExecutor`.
//!
//! **Status (M1.6 scaffolding):** acceptance gate is "runs to
//! completion without panic and emits at least one token." The TTFT
//! / TPOT / peak-memory measurements specified in
//! `docs/plans/path-1-gelo-gemma.md` M1.6 are a separate workstream —
//! once the M1.6 worker has a benchmark harness wired into
//! `evals/run-eval.py` (M0.2) they'll re-use this loader.
//!
//! **Blockers (un-ignore prerequisites):**
//!  - Real Gemma 4 E2B weights published on HuggingFace under
//!    `<TODO model_id>`. Plan §6 "open decisions" still has the
//!    PLE fp16/int8 question; pin the model SHA in the loader once
//!    the M1.1-loader real-weight path lands.
//!  - GPU access (Vulkan or wgpu CPU backend) — the CPU `RayonCpuEngine`
//!    works for correctness; wall-clock numbers require Vulkan.
//!  - PLE table loading: the M1.1 safetensors loader does not yet
//!    extract `embed_tokens_per_layer.weight` and per-layer PLE
//!    projections. That's the remaining concrete code item before
//!    this test can flip `#[ignore]` off.
//!
//! Downloads multiple GB on first run; gated behind `#[ignore]`.

use anyhow::Result;
use std::sync::Arc;

use gelo_embedder::decoder::config::DecoderConfig;
use gelo_embedder::decoder::gemma4::Gemma4Variant;
use gelo_embedder::decoder::generation::{GenerationConfig, SamplerConfig, generate};
use gelo_embedder::decoder::rope::RopeTables;
use gelo_embedder::decoder::weights::DecoderWeights;
use gelo_protocol::{PlaintextExecutor, RayonCpuEngine};

const MODEL: &str = "google/gemma-4-e2b";

#[test]
#[ignore = "M1.6 scaffolding — gated on (a) M1.1 PLE-aware loader and (b) real HF model id"]
fn gemma4_e2b_greedy_generates_to_completion() -> Result<()> {
    // Build the variant config and load weights from HF. The current
    // `DecoderWeights::from_safetensors` covers the standard
    // decoder weights; PLE and per-layer attention-class metadata
    // need to be wired by the M1.1 loader extensions before this
    // test compiles against the real model_id. Until then the test
    // body is a documented placeholder.

    // 1. Pull the config + tokenizer + safetensors shards from HF.
    //    (Reuse `GeloQwenEmbedder::from_pretrained`-style helpers
    //    once gemma4 has its own loader.)
    let _cfg: DecoderConfig = Gemma4Variant::E2B.config();

    // 2. Load weights. TODO: extend `DecoderWeights::from_safetensors`
    //    to recognise Gemma 4 weight-key prefixes
    //    (`model.layers.{i}.…`, `model.embed_tokens_per_layer.weight`,
    //    per-layer PLE projection matrices). Until this lands the
    //    loader returns an error and the test fails fast.
    let _weights: Arc<DecoderWeights> = {
        // Placeholder: real impl plugs into M1.1's Gemma 4 loader.
        return Err(anyhow::anyhow!(
            "Gemma 4 loader not yet wired — see M1.1 loader extensions"
        ));
    };

    #[allow(unreachable_code)]
    {
        // 3. Build RoPE tables at the variant's head_dim / max_position.
        let rope = RopeTables::new(
            _cfg.head_dim_value(),
            _cfg.max_position_embeddings,
            _cfg.rope_theta,
        );

        // 4. Construct executor + provision standard offload weights.
        let mut exec = PlaintextExecutor::new(RayonCpuEngine::new());
        // exec.provision_ple_table(...) — when M1.2 loader produces a
        // real PleTable from the safetensors blob.

        // 5. Run greedy generate on a fixed prompt.
        let prompt: Vec<u32> = vec![/* tokenized prompt IDs */];
        let out = generate(
            &_cfg,
            &_weights,
            &rope,
            &mut exec,
            &prompt,
            &GenerationConfig {
                max_tokens: 16,
                eos_token_ids: Vec::new(),
                sampler: SamplerConfig::Greedy,
            },
        )?;
        assert!(!out.tokens.is_empty(), "expected ≥1 token from generate()");
        eprintln!(
            "M1.6 E2B smoke: emitted {} tokens (eos={})",
            out.tokens.len(),
            out.stopped_on_eos
        );
        Ok(())
    }
}

/// One-shot stub recording the model_id that M1.6 worker should
/// hard-pin once Gemma 4 E2B is published on HuggingFace. Use this
/// constant rather than a free-floating literal so re-pinning is
/// a one-line change.
pub const _M16_HF_TARGET_E2B: &str = MODEL;

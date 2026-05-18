//! Autoregressive generation loop for Gemma 4 / Qwen3-class decoders.
//!
//! Per `docs/prototype/gelo-llm.html` §06: this module owns the prefill
//! → decode → sample → append loop. Sampling happens entirely in the
//! TEE on plaintext logits; the GPU never sees the sample. Each decode
//! step is one forward pass and gets its own fresh Haar `A` via the
//! executor's per-forward-pass session bracket.
//!
//! M1.0 scope: greedy sampling only. Top-p / top-k / temperature land
//! alongside the bench harness in M1.6 — the protocol surface doesn't
//! change.
//!
//! M1.0 assumes tied input/output embeddings (Qwen3-Embedding-0.6B and
//! Gemma 4 E2B/E4B both set `tie_word_embeddings = true`). The LM-head
//! projection is `h_last · token_embedding.T`, computed in-TEE. M1.1's
//! Gemma 4 loader will add an explicit `lm_head` weight slot for the
//! untied case; this loop will then prefer `weights.lm_head` over
//! `weights.token_embedding` when present.

use anyhow::{Result, anyhow};
use ndarray::ArrayView1;

use gelo_protocol::TrustedExecutor;

use super::config::DecoderConfig;
use super::forward;
use super::kv_cache::KvCache;
use super::rope::RopeTables;
use super::weights::DecoderWeights;

/// What to do at each decode step to turn logits into the next token.
#[derive(Debug, Clone)]
pub enum SamplerConfig {
    /// argmax — deterministic, the only mode used by the M1.8 accept
    /// gate (HF `transformers` parity at `temperature=0`).
    Greedy,
}

/// User-facing generation configuration.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Hard upper bound on tokens emitted (excluding the prompt).
    pub max_tokens: usize,
    /// Stop tokens. The first sampled token in this list terminates the
    /// generation; the token IS included in the output.
    pub eos_token_ids: Vec<u32>,
    /// How to sample the next token from the per-step logits.
    pub sampler: SamplerConfig,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 64,
            eos_token_ids: Vec::new(),
            sampler: SamplerConfig::Greedy,
        }
    }
}

/// Result of one call to [`generate`].
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    /// Newly-generated tokens only (does NOT include the prompt).
    /// If `stopped_on_eos` is true, the final entry is one of
    /// `GenerationConfig::eos_token_ids`.
    pub tokens: Vec<u32>,
    /// Whether generation halted because a stop token was sampled
    /// (true) or because `max_tokens` was reached (false).
    pub stopped_on_eos: bool,
}

/// Greedy / top-k / top-p sampling driver. M1.0 implements greedy only.
fn sample(logits: ArrayView1<'_, f32>, sampler: &SamplerConfig) -> Result<u32> {
    match sampler {
        SamplerConfig::Greedy => Ok(argmax(logits)),
    }
}

fn argmax(logits: ArrayView1<'_, f32>) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// Compute logits = `h_last · token_embedding.T` for tied-embedding
/// models. Returns a `(vocab_size,)` 1-D vector. Stays in-TEE — the LM
/// head is the same primitive as any other masked offload but at
/// decode-step shape `(1, d_hidden) · (d_hidden, vocab)` it's smaller
/// than the dispatch overhead would amortise, so v1 keeps it on CPU.
/// M1.1 will route this through a `WeightKind::LmHead` offload when
/// the loader provides one.
fn compute_logits(weights: &DecoderWeights, h_last: ArrayView1<'_, f32>) -> ndarray::Array1<f32> {
    let vocab = weights.token_embedding.nrows();
    let mut logits = ndarray::Array1::<f32>::zeros(vocab);
    for v in 0..vocab {
        let row = weights.token_embedding.row(v);
        let dot: f32 = h_last
            .iter()
            .zip(row.iter())
            .map(|(a, b)| a * b)
            .sum();
        logits[v] = dot;
    }
    logits
}

/// Run prefill + decode loop. Returns the newly-sampled tokens (prompt
/// NOT included). The KV cache is allocated internally and dropped
/// when generation ends; multi-turn / streaming variants that own
/// the cache externally are a follow-up.
pub fn generate(
    cfg: &DecoderConfig,
    weights: &DecoderWeights,
    rope: &RopeTables,
    exec: &mut impl TrustedExecutor,
    prompt_ids: &[u32],
    gen_cfg: &GenerationConfig,
) -> Result<GenerationOutput> {
    if prompt_ids.is_empty() {
        return Err(anyhow!("prompt_ids must be non-empty"));
    }
    if gen_cfg.max_tokens == 0 {
        return Ok(GenerationOutput {
            tokens: Vec::new(),
            stopped_on_eos: false,
        });
    }

    let max_cache_len = prompt_ids.len() + gen_cfg.max_tokens;
    if max_cache_len > cfg.max_position_embeddings {
        return Err(anyhow!(
            "prompt {} + max_tokens {} > max_position_embeddings {}",
            prompt_ids.len(),
            gen_cfg.max_tokens,
            cfg.max_position_embeddings,
        ));
    }

    let mut kv_cache = KvCache::new(weights.layers.len(), max_cache_len, cfg.kv_dim());

    // Prefill — populate the cache, take the last position's hidden state.
    let hidden = forward::run_prefill(cfg, weights, rope, exec, prompt_ids, &mut kv_cache)?;
    let last_idx = hidden.nrows() - 1;
    let mut h_last = hidden.row(last_idx).to_owned();

    let mut tokens = Vec::with_capacity(gen_cfg.max_tokens);
    let mut stopped_on_eos = false;
    for _ in 0..gen_cfg.max_tokens {
        let logits = compute_logits(weights, h_last.view());
        let next_token = sample(logits.view(), &gen_cfg.sampler)?;
        tokens.push(next_token);
        if gen_cfg.eos_token_ids.contains(&next_token) {
            stopped_on_eos = true;
            break;
        }
        // Decode-step forward: append one position, get new hidden state.
        h_last = forward::run_decode_step(cfg, weights, rope, exec, next_token, &mut kv_cache)?;
    }

    Ok(GenerationOutput {
        tokens,
        stopped_on_eos,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn argmax_picks_largest_and_lowest_index_on_ties() {
        let logits = Array1::from(vec![1.0_f32, 3.0, 2.0, 3.0, 0.5]);
        // Tie at index 1 and 3; argmax returns the first occurrence (1).
        assert_eq!(argmax(logits.view()), 1);
    }

    #[test]
    fn argmax_handles_negatives() {
        let logits = Array1::from(vec![-5.0_f32, -1.0, -3.0, -2.0]);
        assert_eq!(argmax(logits.view()), 1);
    }

    #[test]
    fn argmax_with_single_element() {
        let logits = Array1::from(vec![42.0_f32]);
        assert_eq!(argmax(logits.view()), 0);
    }
}

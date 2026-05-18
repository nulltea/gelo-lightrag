//! M1.0 acceptance test for the autoregressive generation harness.
//!
//! Verifies the harness mechanics on a synthetic 2-layer decoder
//! with deterministic weights. The load-bearing invariant — the only
//! one a structural test can verify at this point — is the
//! prefill/decode equivalence: running greedy `generate(prompt, k)`
//! and then prefilling on `prompt ++ output_tokens` must produce
//! logits whose per-position argmax recovers `output_tokens` exactly.
//!
//! If this invariant holds, the KV-cache + RoPE-at-offset +
//! asymmetric-attention chain is correctly wired. HF `transformers`
//! parity is the M1.8 gate against real weights — not this test.
//!
//! Synthetic-weight construction follows `decoder_parity.rs` so the
//! two tests share their model topology.
//!
//! The synthetic decoder here intentionally has a TINY (vocab=8)
//! head so an argmax over the LM-head dot product is meaningful;
//! random Gaussian-init weights produce reasonable logit spread at
//! that scale.

use std::sync::Arc;

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

use gelo_embedder::decoder::config::DecoderConfig;
use gelo_embedder::decoder::forward;
use gelo_embedder::decoder::generation::{GenerationConfig, SamplerConfig, generate};
use gelo_embedder::decoder::kv_cache::KvCache;
use gelo_embedder::decoder::rope::RopeTables;
use gelo_embedder::decoder::weights::{DecoderLayerWeights, DecoderWeights};
use gelo_protocol::{GpuOffloadEngine, PlaintextExecutor, RayonCpuEngine, WeightHandle, WeightKind};

fn tiny_decoder_config() -> DecoderConfig {
    DecoderConfig {
        vocab_size: 8,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: Some(8),
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        hidden_act: "silu".into(),
        tie_word_embeddings: true,
        max_seq_len: 64,
        skip_first_layers: 0,
        skip_last_layer: false,
        use_out_attn_mult: false,
        out_attn_mult_min_seq_len: None,
        use_perm_attention: false,
        perm_attention_min_seq_len: None,
    }
}

fn rand2(rows: usize, cols: usize, rng: &mut impl rand::RngCore, scale: f32) -> Array2<f32> {
    let normal = StandardNormal;
    Array2::from_shape_fn((rows, cols), |_| {
        <StandardNormal as Distribution<f32>>::sample(&normal, rng) * scale
    })
}

fn synth_weights(cfg: &DecoderConfig, rng: &mut impl rand::RngCore) -> DecoderWeights {
    let d = cfg.hidden_size;
    let q = cfg.q_dim();
    let kv = cfg.kv_dim();
    let f = cfg.intermediate_size;
    let layers = (0..cfg.num_hidden_layers)
        .map(|_| DecoderLayerWeights {
            norm_attn: Array1::from_elem(d, 1.0),
            wq: rand2(d, q, rng, 0.05),
            wk: rand2(d, kv, rng, 0.05),
            wv: rand2(d, kv, rng, 0.05),
            wo: rand2(q, d, rng, 0.05),
            norm_ffn: Array1::from_elem(d, 1.0),
            w_gate: rand2(d, f, rng, 0.05),
            w_up: rand2(d, f, rng, 0.05),
            w_down: rand2(f, d, rng, 0.05),
        })
        .collect();
    DecoderWeights {
        token_embedding: rand2(cfg.vocab_size, d, rng, 0.5),
        final_norm: Array1::from_elem(d, 1.0),
        layers,
        model_identity: [0u8; 32],
    }
}

fn provision_decoder<E: GpuOffloadEngine>(
    weights: &DecoderWeights,
    cfg: &DecoderConfig,
    engine: &mut E,
) {
    for (li, layer) in weights.layers.iter().enumerate() {
        if !cfg.offload_layer(li) {
            continue;
        }
        let li16 = li as u16;
        engine
            .register_weight(WeightHandle::new(li16, WeightKind::Q), layer.wq.view())
            .unwrap();
        engine
            .register_weight(WeightHandle::new(li16, WeightKind::K), layer.wk.view())
            .unwrap();
        engine
            .register_weight(WeightHandle::new(li16, WeightKind::V), layer.wv.view())
            .unwrap();
        engine
            .register_weight(WeightHandle::new(li16, WeightKind::O), layer.wo.view())
            .unwrap();
        engine
            .register_weight(WeightHandle::new(li16, WeightKind::FfnGate), layer.w_gate.view())
            .unwrap();
        engine
            .register_weight(WeightHandle::new(li16, WeightKind::FfnUp), layer.w_up.view())
            .unwrap();
        engine
            .register_weight(WeightHandle::new(li16, WeightKind::FfnDown), layer.w_down.view())
            .unwrap();
    }
}

fn make_exec_and_weights() -> (
    PlaintextExecutor<RayonCpuEngine>,
    Arc<DecoderWeights>,
    DecoderConfig,
    RopeTables,
) {
    let cfg = tiny_decoder_config();
    let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
    let weights = Arc::new(synth_weights(&cfg, &mut rng));
    let rope = RopeTables::new(
        cfg.head_dim_value(),
        cfg.max_position_embeddings,
        cfg.rope_theta,
    );
    let mut engine = RayonCpuEngine::new();
    provision_decoder(&weights, &cfg, &mut engine);
    let exec = PlaintextExecutor::new(engine);
    (exec, weights, cfg, rope)
}

fn argmax_row(row: ndarray::ArrayView1<f32>) -> u32 {
    let mut best = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i as u32;
        }
    }
    best
}

#[test]
fn greedy_generate_is_deterministic_across_runs() {
    let prompt: Vec<u32> = vec![1, 3, 5];
    let gen_cfg = GenerationConfig {
        max_tokens: 5,
        eos_token_ids: Vec::new(),
        sampler: SamplerConfig::Greedy,
    };

    let (mut exec_a, w_a, cfg_a, rope_a) = make_exec_and_weights();
    let out_a = generate(&cfg_a, &w_a, &rope_a, &mut exec_a, &prompt, &gen_cfg).unwrap();

    let (mut exec_b, w_b, cfg_b, rope_b) = make_exec_and_weights();
    let out_b = generate(&cfg_b, &w_b, &rope_b, &mut exec_b, &prompt, &gen_cfg).unwrap();

    assert_eq!(out_a.tokens, out_b.tokens);
    assert_eq!(out_a.tokens.len(), 5);
    assert!(!out_a.stopped_on_eos);
}

#[test]
fn decode_replays_under_prefill() {
    // The load-bearing structural invariant: generate(prompt, k) →
    // tokens t1..tk. Prefilling on (prompt ++ t1..tk) and re-sampling
    // greedy at each output position must reproduce t1..tk exactly.
    //
    // This proves the KV-cache + RoPE-at-offset + asymmetric-attention
    // chain matches the "always prefill" baseline.

    let prompt: Vec<u32> = vec![2, 4, 6];
    let gen_cfg = GenerationConfig {
        max_tokens: 5,
        eos_token_ids: Vec::new(),
        sampler: SamplerConfig::Greedy,
    };

    let (mut exec, weights, cfg, rope) = make_exec_and_weights();
    let out = generate(&cfg, &weights, &rope, &mut exec, &prompt, &gen_cfg).unwrap();
    assert_eq!(out.tokens.len(), 5);

    // Concatenate prompt + generated tokens and prefill on the full
    // sequence. Use a fresh executor since the previous one's RNG/
    // session state has advanced.
    let (mut exec2, _, _, _) = make_exec_and_weights();
    let mut full = prompt.clone();
    full.extend_from_slice(&out.tokens);
    let mut cache = KvCache::new(weights.layers.len(), full.len() + 1, cfg.kv_dim());
    let hidden = forward::run_prefill(&cfg, &weights, &rope, &mut exec2, &full, &mut cache).unwrap();

    // Logits at position `prompt.len() - 1` predict t1; at
    // `prompt.len()` predict t2; … at `full.len() - 2` predict t5.
    let token_embedding = &weights.token_embedding;
    for (k, &expected) in out.tokens.iter().enumerate() {
        let pos = prompt.len() - 1 + k;
        // Compute logits = h[pos] · token_embedding.T.
        let h_row = hidden.row(pos);
        let vocab = token_embedding.nrows();
        let mut logits = ndarray::Array1::<f32>::zeros(vocab);
        for v in 0..vocab {
            logits[v] = h_row
                .iter()
                .zip(token_embedding.row(v).iter())
                .map(|(a, b)| a * b)
                .sum();
        }
        let got = argmax_row(logits.view());
        assert_eq!(
            got, expected,
            "decode replay mismatch at output step {k} (prefill pos {pos}): \
             expected {expected}, got {got}",
        );
    }
}

#[test]
fn generate_stops_on_eos_token() {
    // Force generate to halt by adding the first deterministic output
    // to eos_token_ids — must produce exactly one token and report
    // stopped_on_eos = true.
    let prompt: Vec<u32> = vec![1, 3, 5];

    let (mut exec, weights, cfg, rope) = make_exec_and_weights();
    let probe = generate(
        &cfg,
        &weights,
        &rope,
        &mut exec,
        &prompt,
        &GenerationConfig {
            max_tokens: 1,
            eos_token_ids: Vec::new(),
            sampler: SamplerConfig::Greedy,
        },
    )
    .unwrap();
    let first_token = probe.tokens[0];

    let (mut exec2, w2, c2, r2) = make_exec_and_weights();
    let out = generate(
        &c2,
        &w2,
        &r2,
        &mut exec2,
        &prompt,
        &GenerationConfig {
            max_tokens: 10,
            eos_token_ids: vec![first_token],
            sampler: SamplerConfig::Greedy,
        },
    )
    .unwrap();
    assert_eq!(out.tokens, vec![first_token]);
    assert!(out.stopped_on_eos);
}

#[test]
fn empty_prompt_rejected() {
    let (mut exec, weights, cfg, rope) = make_exec_and_weights();
    let err = generate(
        &cfg,
        &weights,
        &rope,
        &mut exec,
        &[],
        &GenerationConfig::default(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("non-empty"));
}

#[test]
fn max_tokens_zero_returns_empty() {
    let (mut exec, weights, cfg, rope) = make_exec_and_weights();
    let out = generate(
        &cfg,
        &weights,
        &rope,
        &mut exec,
        &[1u32, 2, 3],
        &GenerationConfig {
            max_tokens: 0,
            eos_token_ids: Vec::new(),
            sampler: SamplerConfig::Greedy,
        },
    )
    .unwrap();
    assert!(out.tokens.is_empty());
    assert!(!out.stopped_on_eos);
}

#[test]
fn overflow_max_position_embeddings_errors() {
    let (mut exec, weights, cfg, rope) = make_exec_and_weights();
    let huge_prompt: Vec<u32> = (0..63).map(|i| i % cfg.vocab_size as u32).collect();
    let err = generate(
        &cfg,
        &weights,
        &rope,
        &mut exec,
        &huge_prompt,
        &GenerationConfig {
            max_tokens: 10,
            eos_token_ids: Vec::new(),
            sampler: SamplerConfig::Greedy,
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("max_position_embeddings"));
}

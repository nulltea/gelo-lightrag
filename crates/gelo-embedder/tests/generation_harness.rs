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

use gelo_embedder::decoder::config::{AttentionClass, DecoderConfig};
use gelo_embedder::decoder::forward;
use gelo_embedder::decoder::gemma4::gemma4_attention_classes;
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
        attention_classes: None,
        partial_rope: None,
        kv_shared_in_global: false,
        final_logit_softcapping: None,
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
            wq: Some(std::sync::Arc::new(rand2(d, q, rng, 0.05).mapv(|v| half::bf16::from_f32(v)))),
            wk: Some(std::sync::Arc::new(rand2(d, kv, rng, 0.05).mapv(|v| half::bf16::from_f32(v)))),
            wv: Some(std::sync::Arc::new(rand2(d, kv, rng, 0.05).mapv(|v| half::bf16::from_f32(v)))),
            wo: Some(std::sync::Arc::new(rand2(q, d, rng, 0.05).mapv(|v| half::bf16::from_f32(v)))),
            norm_ffn: Array1::from_elem(d, 1.0),
            w_gate: Some(std::sync::Arc::new(rand2(d, f, rng, 0.05).mapv(|v| half::bf16::from_f32(v)))),
            w_up: Some(std::sync::Arc::new(rand2(d, f, rng, 0.05).mapv(|v| half::bf16::from_f32(v)))),
            w_down: Some(std::sync::Arc::new(rand2(f, d, rng, 0.05).mapv(|v| half::bf16::from_f32(v)))),
            q_norm: None,
            k_norm: None,
        })
        .collect();
    DecoderWeights {
        token_embedding: rand2(cfg.vocab_size, d, rng, 0.5).mapv(|v| half::bf16::from_f32(v)),
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
            .register_weight_bf16(WeightHandle::new(li16, WeightKind::Q), layer.wq.as_ref().expect("offloadable weight").view())
            .unwrap();
        engine
            .register_weight_bf16(WeightHandle::new(li16, WeightKind::K), layer.wk.as_ref().expect("offloadable weight").view())
            .unwrap();
        engine
            .register_weight_bf16(WeightHandle::new(li16, WeightKind::V), layer.wv.as_ref().expect("offloadable weight").view())
            .unwrap();
        engine
            .register_weight_bf16(WeightHandle::new(li16, WeightKind::O), layer.wo.as_ref().expect("offloadable weight").view())
            .unwrap();
        engine
            .register_weight_bf16(WeightHandle::new(li16, WeightKind::FfnGate), layer.w_gate.as_ref().expect("offloadable weight").view())
            .unwrap();
        engine
            .register_weight_bf16(WeightHandle::new(li16, WeightKind::FfnUp), layer.w_up.as_ref().expect("offloadable weight").view())
            .unwrap();
        engine
            .register_weight_bf16(WeightHandle::new(li16, WeightKind::FfnDown), layer.w_down.as_ref().expect("offloadable weight").view())
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
                .map(|(a, b)| a * b.to_f32())
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

/// Build a small Gemma 4-shaped synthetic decoder: 6 layers, 2:1
/// local:global pattern, window W=4. Same head/dim layout as the
/// tiny config so the existing `synth_weights` helper applies.
///
/// Purpose: M1.1 acceptance gate — the new `attention_classes` /
/// `partial_rope` / `kv_shared_in_global` fields propagate through
/// `DecoderConfig` without breaking the existing forward path. M1.3
/// will wire the hybrid dispatch; until then `effective_attention_class`
/// is read but not yet consulted by the attention kernel.
fn gemma4_shaped_config() -> DecoderConfig {
    let mut cfg = tiny_decoder_config();
    cfg.num_hidden_layers = 6;
    cfg.attention_classes = Some(gemma4_attention_classes(6, 2, 4));
    cfg.partial_rope = Some(0.25);
    cfg.kv_shared_in_global = true;
    cfg
}

#[test]
fn gemma4_shaped_attention_class_vector_is_valid() {
    let cfg = gemma4_shaped_config();
    let classes = cfg.attention_classes.as_ref().unwrap();
    assert_eq!(classes.len(), 6);
    // 2:1 pattern: [L, L, G, L, L, G_last_override]. Position 5 is the
    // last layer — always Global per the spec.
    assert_eq!(classes[0], AttentionClass::Local { window: 4 });
    assert_eq!(classes[1], AttentionClass::Local { window: 4 });
    assert_eq!(classes[2], AttentionClass::Global);
    assert_eq!(classes[3], AttentionClass::Local { window: 4 });
    assert_eq!(classes[4], AttentionClass::Local { window: 4 });
    assert_eq!(classes[5], AttentionClass::Global);
    assert!(cfg.is_hybrid_attention());
    // head_dim=8 · 0.25 = 2 (already even).
    assert_eq!(cfg.rotated_dim(), 2);
}

#[test]
fn gemma4_shaped_decoder_runs_generate() {
    // M1.3 wiring: the `attention_classes` vector is now consulted
    // per-layer. With the 6-layer 2:1 pattern from
    // `gemma4_shaped_config()`, four layers run sliding-window
    // (W = 4) and two run global causal. Both paths stay in-TEE.
    let cfg = gemma4_shaped_config();
    let mut rng = ChaCha20Rng::from_seed([42u8; 32]);
    let weights = Arc::new(synth_weights(&cfg, &mut rng));
    let rope = RopeTables::new(
        cfg.head_dim_value(),
        cfg.max_position_embeddings,
        cfg.rope_theta,
    );
    let mut engine = RayonCpuEngine::new();
    provision_decoder(&weights, &cfg, &mut engine);
    let mut exec = PlaintextExecutor::new(engine);

    let out = generate(
        &cfg,
        &weights,
        &rope,
        &mut exec,
        &[1u32, 3, 5],
        &GenerationConfig {
            max_tokens: 3,
            eos_token_ids: Vec::new(),
            sampler: SamplerConfig::Greedy,
        },
    )
    .unwrap();
    assert_eq!(out.tokens.len(), 3);
    assert!(!out.stopped_on_eos);
}

#[test]
fn hybrid_with_max_window_matches_all_global() {
    // M1.3 dispatch correctness: a hybrid config whose local layers
    // use `window = max_position_embeddings` must produce identical
    // output to a config with `attention_classes = None`. The SWA
    // kernel collapses to dense causal when window ≥ n_kv (see
    // `decoder::attention::tests::swa_with_window_ge_seq_matches_dense_causal`).
    let mut hybrid = tiny_decoder_config();
    hybrid.num_hidden_layers = 4;
    hybrid.attention_classes = Some(vec![
        AttentionClass::Local {
            window: hybrid.max_position_embeddings,
        },
        AttentionClass::Global,
        AttentionClass::Local {
            window: hybrid.max_position_embeddings,
        },
        AttentionClass::Global,
    ]);

    let mut all_global = hybrid.clone();
    all_global.attention_classes = None;

    let mut rng = ChaCha20Rng::from_seed([99u8; 32]);
    let weights = Arc::new(synth_weights(&hybrid, &mut rng));
    let rope = RopeTables::new(
        hybrid.head_dim_value(),
        hybrid.max_position_embeddings,
        hybrid.rope_theta,
    );

    let prompt = vec![1u32, 2, 3];
    let gen_cfg = GenerationConfig {
        max_tokens: 4,
        eos_token_ids: Vec::new(),
        sampler: SamplerConfig::Greedy,
    };

    let mut engine_a = RayonCpuEngine::new();
    provision_decoder(&weights, &hybrid, &mut engine_a);
    let mut exec_a = PlaintextExecutor::new(engine_a);
    let out_a = generate(&hybrid, &weights, &rope, &mut exec_a, &prompt, &gen_cfg).unwrap();

    let mut engine_b = RayonCpuEngine::new();
    provision_decoder(&weights, &all_global, &mut engine_b);
    let mut exec_b = PlaintextExecutor::new(engine_b);
    let out_b = generate(
        &all_global,
        &weights,
        &rope,
        &mut exec_b,
        &prompt,
        &gen_cfg,
    )
    .unwrap();

    assert_eq!(
        out_a.tokens, out_b.tokens,
        "hybrid(W=max) vs all-global diverged",
    );
}

#[test]
fn hybrid_with_tight_window_diverges_at_hidden_state() {
    // Inverse sanity check: a tight window MUST produce different
    // hidden states than dense causal on a deep-enough sequence.
    // (Sampled tokens are argmax-collapsed on the synthetic LM head,
    // so we compare hidden states directly — more sensitive and
    // doesn't depend on the random embedding being non-degenerate.)
    let mut hybrid = tiny_decoder_config();
    hybrid.num_hidden_layers = 4;
    hybrid.attention_classes = Some(vec![
        AttentionClass::Local { window: 2 }, // tight
        AttentionClass::Global,
        AttentionClass::Local { window: 2 },
        AttentionClass::Global,
    ]);

    let mut all_global = hybrid.clone();
    all_global.attention_classes = None;

    let mut rng = ChaCha20Rng::from_seed([77u8; 32]);
    let weights = Arc::new(synth_weights(&hybrid, &mut rng));
    let rope = RopeTables::new(
        hybrid.head_dim_value(),
        hybrid.max_position_embeddings,
        hybrid.rope_theta,
    );

    let prompt = vec![1u32, 2, 3, 4, 5, 6];

    let mut engine_a = RayonCpuEngine::new();
    provision_decoder(&weights, &hybrid, &mut engine_a);
    let mut exec_a = PlaintextExecutor::new(engine_a);
    let mut cache_a = KvCache::new(weights.layers.len(), prompt.len() + 4, hybrid.kv_dim());
    let h_hybrid = forward::run_prefill(&hybrid, &weights, &rope, &mut exec_a, &prompt, &mut cache_a)
        .unwrap();

    let mut engine_b = RayonCpuEngine::new();
    provision_decoder(&weights, &all_global, &mut engine_b);
    let mut exec_b = PlaintextExecutor::new(engine_b);
    let mut cache_b = KvCache::new(weights.layers.len(), prompt.len() + 4, all_global.kv_dim());
    let h_global = forward::run_prefill(
        &all_global,
        &weights,
        &rope,
        &mut exec_b,
        &prompt,
        &mut cache_b,
    )
    .unwrap();

    // The last row's hidden state must differ between the two
    // configurations — the tight window changes what the global
    // layers see (because hidden state propagates), and any divergence
    // at any layer accumulates into the final state.
    let last = prompt.len() - 1;
    let max_abs_diff: f32 = h_hybrid
        .row(last)
        .iter()
        .zip(h_global.row(last).iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs_diff > 1e-4,
        "tight-window hybrid produced identical hidden state to all-global \
         (max abs diff = {max_abs_diff}) — dispatch likely not wired",
    );
}

/// Build a Gemma 4-shaped config + synthetic weights where wk == wv
/// (Arc-shared), then verify `kv_shared_in_global = true` and
/// `kv_shared_in_global = false` produce identical hidden states.
/// This is the M1.4 correctness proof: the K=V optimisation only
/// removes redundant compute, never changes math.
#[test]
fn kv_shared_matches_separate_when_wk_equals_wv() {
    let mut cfg_shared = gemma4_shaped_config();
    cfg_shared.kv_shared_in_global = true;

    let mut cfg_separate = cfg_shared.clone();
    cfg_separate.kv_shared_in_global = false;

    let mut rng = ChaCha20Rng::from_seed([66u8; 32]);
    let raw_weights = synth_weights(&cfg_shared, &mut rng);

    // Force wk == wv on every layer by overwriting wv with wk.
    let layers = raw_weights
        .layers
        .into_iter()
        .map(|l| DecoderLayerWeights {
            norm_attn: l.norm_attn,
            wq: l.wq,
            wk: l.wk.clone(),
            wv: l.wk, // tie K = V at the weight level
            wo: l.wo,
            norm_ffn: l.norm_ffn,
            w_gate: l.w_gate,
            w_up: l.w_up,
            w_down: l.w_down,
            q_norm: l.q_norm,
            k_norm: l.k_norm,
        })
        .collect();
    let weights = Arc::new(DecoderWeights {
        token_embedding: raw_weights.token_embedding,
        final_norm: raw_weights.final_norm,
        layers,
        model_identity: raw_weights.model_identity,
    });

    let rope = RopeTables::new(
        cfg_shared.head_dim_value(),
        cfg_shared.max_position_embeddings,
        cfg_shared.rope_theta,
    );
    let prompt = vec![1u32, 2, 3, 4];

    let mut engine_a = RayonCpuEngine::new();
    provision_decoder(&weights, &cfg_shared, &mut engine_a);
    let mut exec_a = PlaintextExecutor::new(engine_a);
    let out_shared = generate(
        &cfg_shared,
        &weights,
        &rope,
        &mut exec_a,
        &prompt,
        &GenerationConfig {
            max_tokens: 3,
            eos_token_ids: Vec::new(),
            sampler: SamplerConfig::Greedy,
        },
    )
    .unwrap();

    let mut engine_b = RayonCpuEngine::new();
    provision_decoder(&weights, &cfg_separate, &mut engine_b);
    let mut exec_b = PlaintextExecutor::new(engine_b);
    let out_separate = generate(
        &cfg_separate,
        &weights,
        &rope,
        &mut exec_b,
        &prompt,
        &GenerationConfig {
            max_tokens: 3,
            eos_token_ids: Vec::new(),
            sampler: SamplerConfig::Greedy,
        },
    )
    .unwrap();

    assert_eq!(
        out_shared.tokens, out_separate.tokens,
        "K=V shared output diverged from separate path under wk == wv",
    );
}

#[test]
fn kv_shared_cache_layout_halves_global_layers() {
    // Allocate the same generation-time cache as `generate()` would
    // for the Gemma 4-shaped config (6 layers, 2:1 ratio = 4 local + 2
    // global), with `kv_shared_in_global = true`. The shared global
    // layers must be ~½ the size of separate ones.
    let cfg = gemma4_shaped_config();
    assert!(cfg.kv_shared_in_global);
    let n_layers = cfg.num_hidden_layers;
    let shared: Vec<bool> = (0..n_layers)
        .map(|li| matches!(cfg.effective_attention_class(li), AttentionClass::Global))
        .collect();
    let max_cache_len = 32;
    let cache = KvCache::new_with_sharing(n_layers, max_cache_len, cfg.kv_dim(), &shared);

    let all_sep = KvCache::new(n_layers, max_cache_len, cfg.kv_dim());

    // 2 of 6 layers are shared → 2 * half + 4 * full per the byte
    // formula in kv_cache::tests::shared_layer_halves_memory_footprint.
    // Concrete numbers: full = 32 * kv_dim * 8 (K + V * 4 bytes), half
    // = 32 * kv_dim * 4.
    let per_sep = max_cache_len * cfg.kv_dim() * 8;
    let per_shared = max_cache_len * cfg.kv_dim() * 4;
    assert_eq!(cache.bytes(), 4 * per_sep + 2 * per_shared);
    assert_eq!(all_sep.bytes(), 6 * per_sep);
    // Memory saved must be exactly the per-shared cost across the 2
    // shared layers (each saves one half = per_shared bytes).
    assert_eq!(all_sep.bytes() - cache.bytes(), 2 * per_shared);
}

#[test]
fn partial_rope_diverges_from_full_rope() {
    // p-RoPE on global layers must produce a different hidden state
    // than full rotation. With a 6-layer 2:1 hybrid config and
    // partial_rope=0.25 (rotated_dim = head_dim*0.25, snap-even), the
    // global layers (li=2, 5) skip rotation on the last 75% of each
    // head's dims. Compare vs the same weights with partial_rope=None.
    let mut p_rope = gemma4_shaped_config(); // partial_rope = Some(0.25)
    let mut full_rope = p_rope.clone();
    full_rope.partial_rope = None;

    let mut rng = ChaCha20Rng::from_seed([88u8; 32]);
    let weights = Arc::new(synth_weights(&p_rope, &mut rng));
    let rope = RopeTables::new(
        p_rope.head_dim_value(),
        p_rope.max_position_embeddings,
        p_rope.rope_theta,
    );

    let prompt = vec![1u32, 2, 3, 4];

    let mut engine_a = RayonCpuEngine::new();
    provision_decoder(&weights, &p_rope, &mut engine_a);
    let mut exec_a = PlaintextExecutor::new(engine_a);
    let mut cache_a = KvCache::new(weights.layers.len(), prompt.len() + 4, p_rope.kv_dim());
    let h_partial = forward::run_prefill(&p_rope, &weights, &rope, &mut exec_a, &prompt, &mut cache_a).unwrap();

    let mut engine_b = RayonCpuEngine::new();
    provision_decoder(&weights, &full_rope, &mut engine_b);
    let mut exec_b = PlaintextExecutor::new(engine_b);
    let mut cache_b = KvCache::new(weights.layers.len(), prompt.len() + 4, full_rope.kv_dim());
    let h_full = forward::run_prefill(&full_rope, &weights, &rope, &mut exec_b, &prompt, &mut cache_b).unwrap();

    // Used to silence the unused-variable warning if both configs ever
    // happen to produce identical output (they should not for this
    // synthetic — assertion below catches it).
    let _ = (&p_rope, &full_rope);

    let last = prompt.len() - 1;
    let max_abs_diff: f32 = h_partial
        .row(last)
        .iter()
        .zip(h_full.row(last).iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs_diff > 1e-4,
        "p-RoPE produced identical hidden state to full rotation \
         (max abs diff = {max_abs_diff}) — partial_rope dispatch likely not wired",
    );

    // Silence "mutable but never mutated" if compiler complains.
    let _ = &mut p_rope;
}

#[test]
fn hybrid_decode_replay_invariant_holds() {
    // The M1.0 decode-replay invariant must still hold under hybrid
    // attention: greedy generate(prompt, k) → tokens t1..tk, and
    // prefilling on (prompt ++ t1..tk) must recover the same sequence
    // by per-position argmax. This exercises both the local and
    // global attention paths in the cache-aware kernel.
    let cfg = gemma4_shaped_config();
    let mut rng = ChaCha20Rng::from_seed([55u8; 32]);
    let weights = Arc::new(synth_weights(&cfg, &mut rng));
    let rope = RopeTables::new(
        cfg.head_dim_value(),
        cfg.max_position_embeddings,
        cfg.rope_theta,
    );

    let prompt = vec![2u32, 4, 6];
    let gen_cfg = GenerationConfig {
        max_tokens: 4,
        eos_token_ids: Vec::new(),
        sampler: SamplerConfig::Greedy,
    };

    let mut engine_a = RayonCpuEngine::new();
    provision_decoder(&weights, &cfg, &mut engine_a);
    let mut exec_a = PlaintextExecutor::new(engine_a);
    let out = generate(&cfg, &weights, &rope, &mut exec_a, &prompt, &gen_cfg).unwrap();
    assert_eq!(out.tokens.len(), 4);

    let mut full = prompt.clone();
    full.extend_from_slice(&out.tokens);

    let mut engine_b = RayonCpuEngine::new();
    provision_decoder(&weights, &cfg, &mut engine_b);
    let mut exec_b = PlaintextExecutor::new(engine_b);
    let mut cache = KvCache::new(weights.layers.len(), full.len() + 1, cfg.kv_dim());
    let hidden = forward::run_prefill(&cfg, &weights, &rope, &mut exec_b, &full, &mut cache)
        .unwrap();

    let token_embedding = &weights.token_embedding;
    let vocab = token_embedding.nrows();
    for (k, &expected) in out.tokens.iter().enumerate() {
        let pos = prompt.len() - 1 + k;
        let h_row = hidden.row(pos);
        let mut logits = ndarray::Array1::<f32>::zeros(vocab);
        for v in 0..vocab {
            logits[v] = h_row
                .iter()
                .zip(token_embedding.row(v).iter())
                .map(|(a, b)| a * b.to_f32())
                .sum();
        }
        let got = argmax_row(logits.view());
        assert_eq!(
            got, expected,
            "hybrid replay mismatch at output step {k} (prefill pos {pos})",
        );
    }
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

/// M1.10.1.3 — Parity test for the new `decoder_block_cached`
/// permuted-attention dispatch. With `use_perm_attention = true` and
/// `perm_attention_min_seq_len = Some(0)` (force the path on at any
/// n_q ≥ 0), the cached prefill / decode loop routes Global-layer
/// attention through `causal_gqa_attention_permuted_cached`. At
/// `PermAttnConfig::DISABLED_NOISE` (σ = 0) the math is exact under
/// the Amulet equivariance identity, so generated token sequences
/// must match the in-TEE path bit-for-bit.
#[test]
fn permuted_cached_dispatch_at_sigma_zero_matches_in_tee() {
    use gelo_protocol::rng::MaskSeed;
    use gelo_protocol::{InProcessTrustedExecutor, PermAttnConfig};

    let cfg_in_tee = tiny_decoder_config();
    let mut cfg_permuted = cfg_in_tee.clone();
    cfg_permuted.use_perm_attention = true;
    cfg_permuted.perm_attention_min_seq_len = Some(0); // force permuted at every n_q
    cfg_permuted.use_out_attn_mult = false; // disable OutAttnMult so perm wins

    let mut rng = ChaCha20Rng::from_seed([23u8; 32]);
    let weights = Arc::new(synth_weights(&cfg_in_tee, &mut rng));
    let rope = RopeTables::new(
        cfg_in_tee.head_dim_value(),
        cfg_in_tee.max_position_embeddings,
        cfg_in_tee.rope_theta,
    );

    let input_ids: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 0];

    // Baseline: in-TEE attention via the existing cached path.
    let mut e_in_tee = RayonCpuEngine::new();
    provision_decoder(&weights, &cfg_in_tee, &mut e_in_tee);
    let mut exec_in_tee = InProcessTrustedExecutor::with_seed(e_in_tee, MaskSeed([5u8; 32]));
    exec_in_tee.set_perm_attention(PermAttnConfig::DISABLED_NOISE);
    let out_in_tee = generate(
        &cfg_in_tee,
        &weights,
        &rope,
        &mut exec_in_tee,
        &input_ids,
        &GenerationConfig {
            max_tokens: 6,
            eos_token_ids: Vec::new(),
            sampler: SamplerConfig::Greedy,
        },
    )
    .unwrap();

    // Permuted dispatch: same setup but `use_perm_attention = true`,
    // threshold = 0. At σ = 0 the protocol is exact equivariance.
    let mut e_permuted = RayonCpuEngine::new();
    provision_decoder(&weights, &cfg_permuted, &mut e_permuted);
    let mut exec_permuted = InProcessTrustedExecutor::with_seed(e_permuted, MaskSeed([5u8; 32]));
    exec_permuted.set_perm_attention(PermAttnConfig::DISABLED_NOISE);
    let out_permuted = generate(
        &cfg_permuted,
        &weights,
        &rope,
        &mut exec_permuted,
        &input_ids,
        &GenerationConfig {
            max_tokens: 6,
            eos_token_ids: Vec::new(),
            sampler: SamplerConfig::Greedy,
        },
    )
    .unwrap();

    assert_eq!(
        out_in_tee.tokens, out_permuted.tokens,
        "M1.10.1.3: permuted_cached dispatch at σ=0 must emit byte-identical \
         tokens to the in-TEE cached path. in_tee={:?} permuted={:?}",
        out_in_tee.tokens, out_permuted.tokens,
    );
}

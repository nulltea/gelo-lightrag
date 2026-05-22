//! Decoder-LLM parity tests: masked `InProcessTrustedExecutor` must produce
//! the same encoder outputs as a `PlaintextExecutor`. Synthetic-weights case
//! runs offline; the real Qwen3 path is gated behind `#[ignore]`.

use std::sync::Arc;

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

use gelo_embedder::decoder::config::DecoderConfig;
use gelo_embedder::decoder::forward;
use gelo_embedder::decoder::generation::{
    self, GenerationConfig, SamplerConfig,
};
use gelo_embedder::decoder::rope::RopeTables;
use gelo_embedder::decoder::weights::{DecoderLayerWeights, DecoderWeights};
use gelo_protocol::rng::MaskSeed;
use gelo_protocol::{
    GpuOffloadEngine, InProcessTrustedExecutor, PlaintextExecutor, RayonCpuEngine, TrustedExecutor,
    WeightHandle, WeightKind,
};
use ndarray::Axis;

fn tiny_decoder_config(
    num_layers: usize,
    hidden: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    intermediate: usize,
) -> DecoderConfig {
    DecoderConfig {
        vocab_size: 64,
        hidden_size: hidden,
        intermediate_size: intermediate,
        num_hidden_layers: num_layers,
        num_attention_heads: n_q_heads,
        num_key_value_heads: n_kv_heads,
        head_dim: Some(head_dim),
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        hidden_act: "silu".into(),
        tie_word_embeddings: false,
        max_seq_len: 64,
        skip_first_layers: 0,
        skip_last_layer: false,
        use_out_attn_mult: true,
        // Force OutAttnMult on at the small synthetic shapes used here,
        // overriding the `hidden_size`-based auto-switch.
        out_attn_mult_min_seq_len: Some(0),
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
        token_embedding: rand2(cfg.vocab_size, d, rng, 0.05).mapv(|v| half::bf16::from_f32(v)),
        final_norm: Array1::from_elem(d, 1.0),
        layers,
        // Synthetic weights have no on-disk hash; use a sentinel.
        model_identity: [0u8; 32],
    }
}

fn provision_decoder<E: GpuOffloadEngine>(weights: &DecoderWeights, cfg: &DecoderConfig, engine: &mut E) {
    for (li, layer) in weights.layers.iter().enumerate() {
        if !cfg.offload_layer(li) {
            continue;
        }
        let li16 = li as u16;
        engine.register_weight_bf16(WeightHandle::new(li16, WeightKind::Q), layer.wq.as_ref().expect("offloadable weight").view()).unwrap();
        engine.register_weight_bf16(WeightHandle::new(li16, WeightKind::K), layer.wk.as_ref().expect("offloadable weight").view()).unwrap();
        engine.register_weight_bf16(WeightHandle::new(li16, WeightKind::V), layer.wv.as_ref().expect("offloadable weight").view()).unwrap();
        engine.register_weight_bf16(WeightHandle::new(li16, WeightKind::O), layer.wo.as_ref().expect("offloadable weight").view()).unwrap();
        // SwiGLU: gate at FfnGate, up at FfnUp, down at FfnDown.
        engine.register_weight_bf16(WeightHandle::new(li16, WeightKind::FfnGate), layer.w_gate.as_ref().expect("offloadable weight").view()).unwrap();
        engine.register_weight_bf16(WeightHandle::new(li16, WeightKind::FfnUp), layer.w_up.as_ref().expect("offloadable weight").view()).unwrap();
        engine.register_weight_bf16(WeightHandle::new(li16, WeightKind::FfnDown), layer.w_down.as_ref().expect("offloadable weight").view()).unwrap();
    }
    // M1.12 R3 — LM head is the production default-and-only path.
    // Register the tied-embedding transpose so `generation::generate`
    // can dispatch its per-token offload. Defined below as
    // `register_lm_head` (used standalone by the single-shot mask-
    // round-trip parity test).
    register_lm_head(weights, engine);
}

#[test]
fn synthetic_decoder_parity_two_layer_gqa() {
    let cfg = tiny_decoder_config(/*L*/ 2, /*d*/ 32, /*n_q*/ 4, /*n_kv*/ 2, /*head*/ 8, /*f*/ 64);
    let mut rng = ChaCha20Rng::from_seed([21u8; 32]);
    let weights = Arc::new(synth_weights(&cfg, &mut rng));
    let rope = RopeTables::new(cfg.head_dim_value(), cfg.max_position_embeddings, cfg.rope_theta);

    let input_ids: Vec<u32> = vec![1, 5, 9, 13, 17, 21];

    let mut plain_engine = RayonCpuEngine::new();
    provision_decoder(&weights, &cfg, &mut plain_engine);
    let mut plain = PlaintextExecutor::new(plain_engine);
    let plain_out = forward::run(&cfg, &weights, &rope, &mut plain, &input_ids).unwrap();

    let mut masked_engine = RayonCpuEngine::new();
    provision_decoder(&weights, &cfg, &mut masked_engine);
    let mut masked =
        InProcessTrustedExecutor::with_seed(masked_engine, MaskSeed::from_bytes([22u8; 32]));
    let masked_out = forward::run(&cfg, &weights, &rope, &mut masked, &input_ids).unwrap();

    assert_eq!(plain_out.shape(), masked_out.shape());
    let mut max_abs = 0.0_f32;
    for ((i, j), v) in plain_out.indexed_iter() {
        let diff = (v - masked_out[[i, j]]).abs();
        if diff > max_abs {
            max_abs = diff;
        }
    }
    assert!(
        max_abs < 5e-3,
        "decoder masked vs plaintext diverges: max abs {max_abs}",
    );
}

#[test]
fn synthetic_decoder_parity_permuted_attention() {
    // 3-way autoswitch path #2: permuted attention. Configure the
    // config to engage it (perm_attention enabled, threshold below the
    // input length, OutAttnMult threshold above the input length so it
    // doesn't preempt). At σ = 0 (PermAttnConfig default) the math is
    // exact equivariance — should match the in-TEE / plaintext path to
    // f32 tolerance.
    let mut cfg = tiny_decoder_config(2, 32, 4, 2, 8, 64);
    cfg.use_perm_attention = true;
    cfg.perm_attention_min_seq_len = Some(0);
    cfg.use_out_attn_mult = false; // disable OutAttnMult so perm wins

    let mut rng = ChaCha20Rng::from_seed([91u8; 32]);
    let weights = Arc::new(synth_weights(&cfg, &mut rng));
    let rope = RopeTables::new(cfg.head_dim_value(), cfg.max_position_embeddings, cfg.rope_theta);

    let input_ids: Vec<u32> = vec![1, 5, 9, 13, 17, 21];

    let mut plain_engine = RayonCpuEngine::new();
    provision_decoder(&weights, &cfg, &mut plain_engine);
    let mut plain = PlaintextExecutor::new(plain_engine);
    let plain_out = forward::run(&cfg, &weights, &rope, &mut plain, &input_ids).unwrap();

    let mut masked_engine = RayonCpuEngine::new();
    provision_decoder(&weights, &cfg, &mut masked_engine);
    let mut masked =
        InProcessTrustedExecutor::with_seed(masked_engine, MaskSeed::from_bytes([92u8; 32]));
    let masked_out = forward::run(&cfg, &weights, &rope, &mut masked, &input_ids).unwrap();

    assert_eq!(plain_out.shape(), masked_out.shape());
    let mut max_abs = 0.0_f32;
    for ((i, j), v) in plain_out.indexed_iter() {
        let diff = (v - masked_out[[i, j]]).abs();
        if diff > max_abs {
            max_abs = diff;
        }
    }
    assert!(
        max_abs < 5e-3,
        "decoder permuted-attention path diverges from plain: max abs {max_abs}",
    );
}

#[test]
fn synthetic_decoder_parity_sensitive_layer_exclusion() {
    let mut cfg = tiny_decoder_config(3, 16, 4, 2, 4, 32);
    cfg.skip_first_layers = 1;
    cfg.skip_last_layer = true;

    let mut rng = ChaCha20Rng::from_seed([23u8; 32]);
    let weights = Arc::new(synth_weights(&cfg, &mut rng));
    let rope = RopeTables::new(cfg.head_dim_value(), cfg.max_position_embeddings, cfg.rope_theta);

    let input_ids: Vec<u32> = vec![2, 8, 14, 20];

    let mut plain_engine = RayonCpuEngine::new();
    provision_decoder(&weights, &cfg, &mut plain_engine);
    let mut plain = PlaintextExecutor::new(plain_engine);
    let plain_out = forward::run(&cfg, &weights, &rope, &mut plain, &input_ids).unwrap();

    let mut masked_engine = RayonCpuEngine::new();
    provision_decoder(&weights, &cfg, &mut masked_engine);
    let mut masked =
        InProcessTrustedExecutor::with_seed(masked_engine, MaskSeed::from_bytes([29u8; 32]));
    let masked_out = forward::run(&cfg, &weights, &rope, &mut masked, &input_ids).unwrap();

    let mut max_abs = 0.0_f32;
    for ((i, j), v) in plain_out.indexed_iter() {
        let diff = (v - masked_out[[i, j]]).abs();
        if diff > max_abs {
            max_abs = diff;
        }
    }
    assert!(
        max_abs < 5e-3,
        "decoder sensitive-layer path diverges: max abs {max_abs}",
    );
}

/// **M1.11 R1.3** — `forward::run_batched` at B = 1 matches
/// single-stream `forward::run` to f32 floor (mask topology is the
/// degenerate case of one A_b for the one sequence).
#[test]
fn synthetic_batched_forward_b1_matches_single_stream() {
    let cfg = tiny_decoder_config(2, 32, 4, 2, 8, 64);
    let mut rng = ChaCha20Rng::from_seed([41u8; 32]);
    let weights = Arc::new(synth_weights(&cfg, &mut rng));
    let rope = RopeTables::new(
        cfg.head_dim_value(),
        cfg.max_position_embeddings,
        cfg.rope_theta,
    );

    let input_ids: Vec<u32> = vec![1, 5, 9, 13, 17, 21];

    let mut plain_engine = RayonCpuEngine::new();
    provision_decoder(&weights, &cfg, &mut plain_engine);
    let mut plain = PlaintextExecutor::new(plain_engine);
    let single = forward::run(&cfg, &weights, &rope, &mut plain, &input_ids).unwrap();

    let mut batched_engine = RayonCpuEngine::new();
    provision_decoder(&weights, &cfg, &mut batched_engine);
    let mut batched_exec = InProcessTrustedExecutor::with_seed(
        batched_engine,
        MaskSeed::from_bytes([42u8; 32]),
    );
    let (batched_3d, lens) =
        forward::run_batched(&cfg, &weights, &rope, &mut batched_exec, &[input_ids.clone()])
            .unwrap();
    assert_eq!(lens, vec![input_ids.len()]);
    assert_eq!(batched_3d.shape(), &[1, input_ids.len(), cfg.hidden_size]);

    let mut max_abs = 0.0_f32;
    for ((i, j), v) in single.indexed_iter() {
        let diff = (v - batched_3d[[0, i, j]]).abs();
        if diff > max_abs {
            max_abs = diff;
        }
    }
    assert!(
        max_abs < 5e-3,
        "run_batched(B=1) vs single-stream run diverges: max abs {max_abs}",
    );
}

/// **M1.11 R1.3** — batched forward over B sequences with varying
/// lengths. Each sequence's output (rows `[..lens[b]]`) must match a
/// dedicated single-stream forward on that sequence, to mask round-
/// trip f32 floor. This is the core R3 parity contract for the
/// rerank rollout.
#[test]
fn synthetic_batched_forward_per_sequence_matches_single_stream() {
    let cfg = tiny_decoder_config(2, 32, 4, 2, 8, 64);
    let mut rng = ChaCha20Rng::from_seed([51u8; 32]);
    let weights = Arc::new(synth_weights(&cfg, &mut rng));
    let rope = RopeTables::new(
        cfg.head_dim_value(),
        cfg.max_position_embeddings,
        cfg.rope_theta,
    );

    let seqs: Vec<Vec<u32>> =
        vec![vec![1, 5, 9, 13, 17, 21], vec![2, 6, 10, 14], vec![3, 7, 11, 15, 19]];

    // Plaintext per-sequence references.
    let mut refs: Vec<ndarray::Array2<f32>> = Vec::with_capacity(seqs.len());
    for ids in &seqs {
        let mut plain_engine = RayonCpuEngine::new();
        provision_decoder(&weights, &cfg, &mut plain_engine);
        let mut plain = PlaintextExecutor::new(plain_engine);
        refs.push(forward::run(&cfg, &weights, &rope, &mut plain, ids).unwrap());
    }

    // One batched call.
    let mut batched_engine = RayonCpuEngine::new();
    provision_decoder(&weights, &cfg, &mut batched_engine);
    let mut batched_exec = InProcessTrustedExecutor::with_seed(
        batched_engine,
        MaskSeed::from_bytes([52u8; 32]),
    );
    let (batched_3d, lens) =
        forward::run_batched(&cfg, &weights, &rope, &mut batched_exec, &seqs).unwrap();
    assert_eq!(lens, vec![6, 4, 5]);
    let n_max = *lens.iter().max().unwrap();
    assert_eq!(batched_3d.shape(), &[seqs.len(), n_max, cfg.hidden_size]);

    for (b, reference) in refs.iter().enumerate() {
        let mut max_abs = 0.0_f32;
        for ((i, j), v) in reference.indexed_iter() {
            let diff = (v - batched_3d[[b, i, j]]).abs();
            if diff > max_abs {
                max_abs = diff;
            }
        }
        assert!(
            max_abs < 5e-3,
            "run_batched per-sequence b={b} diverges from single-stream: max abs {max_abs}",
        );
    }
}

/// **M1.11 D1.5** — `generate_batched(&[p])` at B=1 produces the
/// same token sequence as single-stream `generate(&p)` to greedy
/// f32-floor stability. Synthetic weights — the model emits
/// argmax-deterministic noise tokens that are stable across the two
/// mask topologies (single-mask vs PerSequence-of-1).
#[test]
fn synthetic_generate_batched_b1_matches_single_stream() {
    // Disable shared-A so we exercise the default PerSequence
    // decode path.
    unsafe {
        std::env::remove_var("BATCHED_DECODE_SHARED_A");
    }

    let cfg = tiny_decoder_config(2, 32, 4, 2, 8, 64);
    let mut rng = ChaCha20Rng::from_seed([61u8; 32]);
    let weights = Arc::new(synth_weights(&cfg, &mut rng));
    let rope = RopeTables::new(
        cfg.head_dim_value(),
        cfg.max_position_embeddings,
        cfg.rope_theta,
    );
    let prompt: Vec<u32> = vec![3, 11, 17, 22, 29];

    let gen_cfg = GenerationConfig {
        max_tokens: 6,
        eos_token_ids: Vec::new(),
        sampler: SamplerConfig::Greedy,
    };

    // Single-stream reference.
    let mut single_engine = RayonCpuEngine::new();
    provision_decoder(&weights, &cfg, &mut single_engine);
    let mut single_exec = InProcessTrustedExecutor::with_seed(
        single_engine,
        MaskSeed::from_bytes([62u8; 32]),
    );
    let single_out =
        generation::generate(&cfg, &weights, &rope, &mut single_exec, &prompt, &gen_cfg)
            .expect("single generate");

    // Batched at B=1.
    let mut batched_engine = RayonCpuEngine::new();
    provision_decoder(&weights, &cfg, &mut batched_engine);
    let mut batched_exec = InProcessTrustedExecutor::with_seed(
        batched_engine,
        MaskSeed::from_bytes([63u8; 32]),
    );
    let batched_out = generation::generate_batched(
        &cfg,
        &weights,
        &rope,
        &mut batched_exec,
        &[prompt.clone()],
        &gen_cfg,
    )
    .expect("generate_batched B=1");

    assert_eq!(batched_out.len(), 1);
    // Tokens must match to f32 floor — greedy argmax is robust
    // against ~1e-3 mask noise as long as the noise doesn't push a
    // tied or near-tied logit pair over the boundary. With this
    // synthetic config the prompt-driven argmax is well-separated
    // for the first ~6 tokens.
    assert_eq!(
        batched_out[0].tokens, single_out.tokens,
        "generate_batched(B=1) tokens diverged from single-stream: \
         batched={:?} single={:?}",
        batched_out[0].tokens, single_out.tokens,
    );
    assert_eq!(batched_out[0].stopped_on_eos, single_out.stopped_on_eos);
}

/// **M1.11 D1.5** — `generate_batched(&prompts)` at B=3 with varying
/// prompt lengths must produce per-sequence outputs matching B
/// dedicated `generate(&p_b)` calls. Greedy + synthetic weights;
/// token-level parity asserted under the same robustness argument
/// as B=1.
#[test]
fn synthetic_generate_batched_per_sequence_matches_single_stream() {
    unsafe {
        std::env::remove_var("BATCHED_DECODE_SHARED_A");
    }

    let cfg = tiny_decoder_config(2, 32, 4, 2, 8, 64);
    let mut rng = ChaCha20Rng::from_seed([71u8; 32]);
    let weights = Arc::new(synth_weights(&cfg, &mut rng));
    let rope = RopeTables::new(
        cfg.head_dim_value(),
        cfg.max_position_embeddings,
        cfg.rope_theta,
    );

    let prompts: Vec<Vec<u32>> = vec![
        vec![3, 11, 17, 22, 29],
        vec![5, 13, 19],
        vec![2, 7, 14, 21],
    ];
    let gen_cfg = GenerationConfig {
        max_tokens: 5,
        eos_token_ids: Vec::new(),
        sampler: SamplerConfig::Greedy,
    };

    // Reference: serial single-stream generates.
    let mut refs: Vec<gelo_embedder::decoder::generation::GenerationOutput> =
        Vec::with_capacity(prompts.len());
    for prompt in &prompts {
        let mut eng = RayonCpuEngine::new();
        provision_decoder(&weights, &cfg, &mut eng);
        let mut exec =
            InProcessTrustedExecutor::with_seed(eng, MaskSeed::from_bytes([72u8; 32]));
        refs.push(
            generation::generate(&cfg, &weights, &rope, &mut exec, prompt, &gen_cfg)
                .expect("ref generate"),
        );
    }

    // Batched.
    let mut batched_engine = RayonCpuEngine::new();
    provision_decoder(&weights, &cfg, &mut batched_engine);
    let mut batched_exec = InProcessTrustedExecutor::with_seed(
        batched_engine,
        MaskSeed::from_bytes([73u8; 32]),
    );
    let batched_out = generation::generate_batched(
        &cfg,
        &weights,
        &rope,
        &mut batched_exec,
        &prompts,
        &gen_cfg,
    )
    .expect("generate_batched B=3");

    assert_eq!(batched_out.len(), prompts.len());
    for (b, (got, want)) in batched_out.iter().zip(refs.iter()).enumerate() {
        assert_eq!(
            got.tokens, want.tokens,
            "b={b}: batched tokens={:?} != single tokens={:?}",
            got.tokens, want.tokens,
        );
        assert_eq!(got.stopped_on_eos, want.stopped_on_eos, "b={b}");
    }
}

/// **M1.11 D1.5** — EOS-padding loop: when one sequence emits EOS
/// early, subsequent steps must keep advancing the other sequences
/// while the EOS'd sequence's output stays frozen.
#[test]
fn synthetic_generate_batched_per_sequence_eos_freezes() {
    unsafe {
        std::env::remove_var("BATCHED_DECODE_SHARED_A");
    }
    let cfg = tiny_decoder_config(2, 32, 4, 2, 8, 64);
    let mut rng = ChaCha20Rng::from_seed([81u8; 32]);
    let weights = Arc::new(synth_weights(&cfg, &mut rng));
    let rope = RopeTables::new(
        cfg.head_dim_value(),
        cfg.max_position_embeddings,
        cfg.rope_theta,
    );

    // Get reference single-stream tokens for both prompts so we
    // know what to set as EOS for the early-stopping sequence.
    let prompts: Vec<Vec<u32>> = vec![vec![3, 11, 17], vec![5, 13, 19, 23]];
    let no_eos = GenerationConfig {
        max_tokens: 5,
        eos_token_ids: Vec::new(),
        sampler: SamplerConfig::Greedy,
    };
    let mut refs = Vec::with_capacity(prompts.len());
    for p in &prompts {
        let mut eng = RayonCpuEngine::new();
        provision_decoder(&weights, &cfg, &mut eng);
        let mut exec =
            InProcessTrustedExecutor::with_seed(eng, MaskSeed::from_bytes([82u8; 32]));
        refs.push(
            generation::generate(&cfg, &weights, &rope, &mut exec, p, &no_eos).expect("ref"),
        );
    }

    // Pick an EOS token that:
    //  (a) appears at SOME step in refs[0].tokens, and
    //  (b) does NOT appear in refs[1].tokens
    // so sequence 0 stops on EOS while sequence 1 runs to max_tokens.
    //
    // Synthetic weights frequently emit repeated argmax tokens, so
    // we walk refs[0] to find the first token whose value is not in
    // refs[1]. If no such token exists, fall back to a token from
    // refs[0] not in refs[1] (might happen — skip the test then).
    let eos_token = refs[0]
        .tokens
        .iter()
        .copied()
        .find(|t| !refs[1].tokens.contains(t));
    let eos_token = match eos_token {
        Some(t) => t,
        None => {
            // Degenerate fixture: refs[0] is a subset of refs[1].
            // Skip the cross-sequence freeze assertion; just verify
            // the no-eos parity assumptions hold (already covered by
            // the per-sequence parity test).
            eprintln!(
                "skipping EOS-freeze test: refs[0]={:?} ⊆ refs[1]={:?}",
                refs[0].tokens, refs[1].tokens
            );
            return;
        }
    };
    let with_eos = GenerationConfig {
        max_tokens: 5,
        eos_token_ids: vec![eos_token],
        sampler: SamplerConfig::Greedy,
    };

    // Find the expected stop-step for sequence 0: the first index
    // in refs[0].tokens where the EOS token appears.
    let expected_stop_idx = refs[0]
        .tokens
        .iter()
        .position(|t| *t == eos_token)
        .expect("eos_token came from refs[0]");

    let mut batched_engine = RayonCpuEngine::new();
    provision_decoder(&weights, &cfg, &mut batched_engine);
    let mut batched_exec = InProcessTrustedExecutor::with_seed(
        batched_engine,
        MaskSeed::from_bytes([83u8; 32]),
    );
    let out = generation::generate_batched(
        &cfg,
        &weights,
        &rope,
        &mut batched_exec,
        &prompts,
        &with_eos,
    )
    .expect("generate_batched");

    assert_eq!(out.len(), 2);

    // Sequence 0 stopped on EOS at the expected index.
    assert!(
        out[0].stopped_on_eos,
        "sequence 0 should have stopped on EOS (eos={eos_token}, refs={:?})",
        refs[0].tokens,
    );
    assert_eq!(out[0].tokens.len(), expected_stop_idx + 1);
    assert_eq!(*out[0].tokens.last().unwrap(), eos_token);

    // Sequence 1 must NOT see the EOS (we picked it so) and so
    // emit max_tokens worth of output.
    assert!(!out[1].stopped_on_eos);
    assert_eq!(out[1].tokens.len(), no_eos.max_tokens);
    // Sequence 1's tokens must match its no-EOS reference — the
    // padding-feed from sequence 0 must not corrupt sequence 1's
    // trajectory.
    assert_eq!(
        out[1].tokens, refs[1].tokens,
        "sequence 1 corrupted by sequence 0's EOS-padding feed",
    );
}

/// **M1.11 D1.5** — shared-A decode topology (env-gated) also
/// preserves per-sequence parity with single-stream `generate()`
/// to greedy f32 floor.  Same fixture as the per-sequence test,
/// just with `BATCHED_DECODE_SHARED_A=1` set.
#[test]
fn synthetic_generate_batched_shared_a_per_sequence_matches_single_stream() {
    // SAFETY: single-threaded test.
    unsafe {
        std::env::set_var("BATCHED_DECODE_SHARED_A", "1");
    }

    let cfg = tiny_decoder_config(2, 32, 4, 2, 8, 64);
    let mut rng = ChaCha20Rng::from_seed([91u8; 32]);
    let weights = Arc::new(synth_weights(&cfg, &mut rng));
    let rope = RopeTables::new(
        cfg.head_dim_value(),
        cfg.max_position_embeddings,
        cfg.rope_theta,
    );

    let prompts: Vec<Vec<u32>> = vec![vec![3, 11, 17, 22], vec![5, 13, 19], vec![2, 7, 14, 21]];
    let gen_cfg = GenerationConfig {
        max_tokens: 4,
        eos_token_ids: Vec::new(),
        sampler: SamplerConfig::Greedy,
    };

    // Build references with env cleared (single-stream doesn't read
    // the env var anyway, but be principled about it).
    unsafe {
        std::env::remove_var("BATCHED_DECODE_SHARED_A");
    }
    let mut refs = Vec::with_capacity(prompts.len());
    for p in &prompts {
        let mut eng = RayonCpuEngine::new();
        provision_decoder(&weights, &cfg, &mut eng);
        let mut exec =
            InProcessTrustedExecutor::with_seed(eng, MaskSeed::from_bytes([92u8; 32]));
        refs.push(
            generation::generate(&cfg, &weights, &rope, &mut exec, p, &gen_cfg).expect("ref"),
        );
    }
    // Now set the shared-A env back for the batched call.
    unsafe {
        std::env::set_var("BATCHED_DECODE_SHARED_A", "1");
    }

    let mut eng = RayonCpuEngine::new();
    provision_decoder(&weights, &cfg, &mut eng);
    let mut exec =
        InProcessTrustedExecutor::with_seed(eng, MaskSeed::from_bytes([93u8; 32]));
    let out = generation::generate_batched(&cfg, &weights, &rope, &mut exec, &prompts, &gen_cfg)
        .expect("shared-A generate_batched");

    // Restore env regardless of outcome.
    unsafe {
        std::env::remove_var("BATCHED_DECODE_SHARED_A");
    }

    assert_eq!(out.len(), 3);
    for (b, (got, want)) in out.iter().zip(refs.iter()).enumerate() {
        assert_eq!(
            got.tokens, want.tokens,
            "shared-A b={b}: batched={:?} single={:?}",
            got.tokens, want.tokens,
        );
    }
}

/// Helper: register the tied-embedding transpose at
/// `WeightKind::LmHead`. Mirrors
/// `gelo_embedder::decoder::weights::provision_lm_head_into` but at
/// the engine layer so the synthetic-weight tests below can wire it
/// independently of the executor type.
fn register_lm_head<E: GpuOffloadEngine>(weights: &DecoderWeights, engine: &mut E) {
    let lm_head_t = weights.token_embedding.t().as_standard_layout().to_owned();
    engine
        .register_weight_bf16(WeightHandle::new(0, WeightKind::LmHead), lm_head_t.view())
        .unwrap();
}

/// **M1.12 R3 acceptance — single-shot LM-head parity.** A fresh
/// hidden state vector projected through `exec.offload_linear(LmHead,
/// …)` under the substrate's per-forward mask + shield must agree
/// with the in-TEE bf16 dot-product loop to mask round-trip floor
/// (~1e-3). Greedy argmax stable.
#[test]
fn synthetic_lm_head_gpu_offload_matches_in_tee_to_mask_floor() {
    let cfg = tiny_decoder_config(/*L*/ 2, /*d*/ 32, /*n_q*/ 4, /*n_kv*/ 2, /*head*/ 8, /*f*/ 64);
    let mut rng = ChaCha20Rng::from_seed([101u8; 32]);
    let weights = synth_weights(&cfg, &mut rng);

    // Build a deterministic hidden state — use random Gaussian at
    // realistic scale (post-RMSNorm activations are O(1) per channel).
    let h_last: Array1<f32> = rand2(1, cfg.hidden_size, &mut rng, 0.4).row(0).to_owned();

    // CPU baseline: same arithmetic as `compute_logits` — bf16 row ×
    // f32 hidden, per-element widening, f32 accumulator.
    let mut logits_cpu = Array1::<f32>::zeros(cfg.vocab_size);
    for v in 0..cfg.vocab_size {
        let row = weights.token_embedding.row(v);
        logits_cpu[v] = h_last
            .iter()
            .zip(row.iter())
            .map(|(a, b)| a * b.to_f32())
            .sum();
    }

    // GPU offload: through the masked substrate.
    let mut engine = RayonCpuEngine::new();
    register_lm_head(&weights, &mut engine);
    let mut exec = InProcessTrustedExecutor::with_seed(
        engine,
        MaskSeed::from_bytes([102u8; 32]),
    );
    let h2 = h_last.view().insert_axis(Axis(0));
    exec.begin_forward_pass(1).unwrap();
    let logits_gpu_2d = exec
        .offload_linear(WeightHandle::new(0, WeightKind::LmHead), h2)
        .unwrap();
    exec.end_forward_pass().unwrap();
    assert_eq!(logits_gpu_2d.shape(), &[1, cfg.vocab_size]);
    let logits_gpu = logits_gpu_2d.row(0).to_owned();

    // Mask round-trip floor: ~1e-3 at f32, matching the existing
    // masked-vs-plaintext tolerance on synthetic weights.
    let max_diff = logits_cpu
        .iter()
        .zip(logits_gpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff < 5e-3,
        "lm-head GPU offload diverged from in-TEE compute_logits: \
         max_diff={max_diff:.3e}\n cpu={logits_cpu:?}\n gpu={logits_gpu:?}",
    );

    // Greedy argmax stable across the two paths.
    let argmax = |a: &Array1<f32>| -> usize {
        a.iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
            .unwrap()
            .0
    };
    assert_eq!(
        argmax(&logits_cpu),
        argmax(&logits_gpu),
        "greedy argmax flipped under mask round-trip noise",
    );
}

// (former dual-path parity test deleted alongside the
// `lm_head_via_gpu_offload` flag — the GPU LM-head offload is the
// production default and only path, so there's no second path to
// compare against. Single-shot mask-round-trip parity is covered by
// `synthetic_lm_head_gpu_offload_matches_in_tee_to_mask_floor` above.)

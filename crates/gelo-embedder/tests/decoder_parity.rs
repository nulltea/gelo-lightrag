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
use gelo_embedder::decoder::rope::RopeTables;
use gelo_embedder::decoder::weights::{DecoderLayerWeights, DecoderWeights};
use gelo_protocol::rng::MaskSeed;
use gelo_protocol::{
    GpuOffloadEngine, InProcessTrustedExecutor, PlaintextExecutor, RayonCpuEngine, WeightHandle,
    WeightKind,
};

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
            wq: rand2(d, q, rng, 0.05),
            wk: rand2(d, kv, rng, 0.05),
            wv: rand2(d, kv, rng, 0.05),
            wo: rand2(q, d, rng, 0.05),
            norm_ffn: Array1::from_elem(d, 1.0),
            w_gate: rand2(d, f, rng, 0.05),
            w_up: rand2(d, f, rng, 0.05),
            w_down: rand2(f, d, rng, 0.05),
            q_norm: None,
            k_norm: None,
        })
        .collect();
    DecoderWeights {
        token_embedding: rand2(cfg.vocab_size, d, rng, 0.05),
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
        engine.register_weight(WeightHandle::new(li16, WeightKind::Q), layer.wq.view()).unwrap();
        engine.register_weight(WeightHandle::new(li16, WeightKind::K), layer.wk.view()).unwrap();
        engine.register_weight(WeightHandle::new(li16, WeightKind::V), layer.wv.view()).unwrap();
        engine.register_weight(WeightHandle::new(li16, WeightKind::O), layer.wo.view()).unwrap();
        // SwiGLU: gate at FfnGate, up at FfnUp, down at FfnDown.
        engine.register_weight(WeightHandle::new(li16, WeightKind::FfnGate), layer.w_gate.view()).unwrap();
        engine.register_weight(WeightHandle::new(li16, WeightKind::FfnUp), layer.w_up.view()).unwrap();
        engine.register_weight(WeightHandle::new(li16, WeightKind::FfnDown), layer.w_down.view()).unwrap();
    }
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

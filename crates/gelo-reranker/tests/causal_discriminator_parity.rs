//! Parity test for the causal-LM-discriminator reranker.
//!
//! Synthetic DecoderWeights + a synthetic YesNoHead (two arbitrary
//! vocab IDs). The GELO-masked executor must produce the same
//! `softmax([no, yes])[1]` scalar (within numerical tolerance) as the
//! plaintext executor.
//!
//! Tokenization is bypassed — `score_input_ids` takes pre-built ids
//! so the parity check isolates the forward + LM-head gather from the
//! Qwen tokenizer dependency.

use std::sync::Arc;

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

use gelo_embedder::common::tokenizer::HfTokenizer;
use gelo_embedder::decoder::config::DecoderConfig;
use gelo_embedder::decoder::rope::RopeTables;
use gelo_embedder::decoder::weights::{DecoderLayerWeights, DecoderWeights};
use gelo_protocol::rng::MaskSeed;
use gelo_protocol::{
    GpuOffloadEngine, InProcessTrustedExecutor, PlaintextExecutor, ReferenceCpuEngine,
    TrustedExecutor, WeightHandle, WeightKind,
};

use gelo_reranker::causal_discriminator::CausalDiscriminatorRerankService;
use gelo_reranker::head::YesNoHead;

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
        token_embedding: rand2(cfg.vocab_size, d, rng, 0.1).mapv(|v| half::bf16::from_f32(v)),
        final_norm: Array1::from_elem(d, 1.0),
        layers,
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
        engine.register_weight_bf16(WeightHandle::new(li16, WeightKind::FfnGate), layer.w_gate.as_ref().expect("offloadable weight").view()).unwrap();
        engine.register_weight_bf16(WeightHandle::new(li16, WeightKind::FfnUp), layer.w_up.as_ref().expect("offloadable weight").view()).unwrap();
        engine.register_weight_bf16(WeightHandle::new(li16, WeightKind::FfnDown), layer.w_down.as_ref().expect("offloadable weight").view()).unwrap();
    }
}

const STUB_TOKENIZER_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Whitespace" },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": { "[UNK]": 0 },
    "unk_token": "[UNK]"
  }
}"#;

fn stub_tokenizer() -> HfTokenizer {
    let tmp = std::env::temp_dir().join(format!(
        "gelo-reranker-stub-tok-causal-{}-{}.json",
        std::process::id(),
        rand::random::<u32>()
    ));
    std::fs::write(&tmp, STUB_TOKENIZER_JSON).expect("write stub tokenizer");
    let tok = HfTokenizer::from_file(&tmp).expect("load stub tokenizer");
    let _ = std::fs::remove_file(&tmp);
    tok
}

fn build_service<X: TrustedExecutor>(
    cfg: DecoderConfig,
    weights: DecoderWeights,
    rope: Arc<RopeTables>,
    head: YesNoHead,
    exec: X,
) -> CausalDiscriminatorRerankService<X> {
    CausalDiscriminatorRerankService::new(cfg, stub_tokenizer(), weights, rope, head, exec)
        .expect("ctor")
}

#[test]
fn masked_and_plaintext_executors_agree_on_score() {
    let cfg = tiny_decoder_config(2, 32, 4, 2, 8, 64);
    let mut rng = ChaCha20Rng::from_seed([41u8; 32]);
    let weights = synth_weights(&cfg, &mut rng);
    let rope = Arc::new(RopeTables::new(
        cfg.head_dim_value(),
        cfg.max_position_embeddings,
        cfg.rope_theta,
    ));
    let head = YesNoHead { yes_token_id: 3, no_token_id: 9 };

    let mut plain_engine = ReferenceCpuEngine::new();
    provision_decoder(&weights, &cfg, &mut plain_engine);
    let plain = PlaintextExecutor::new(plain_engine);
    let mut plain_svc = build_service(cfg.clone(), weights.clone(), rope.clone(), head, plain);

    let mut masked_engine = ReferenceCpuEngine::new();
    provision_decoder(&weights, &cfg, &mut masked_engine);
    let masked =
        InProcessTrustedExecutor::with_seed(masked_engine, MaskSeed::from_bytes([43u8; 32]));
    let mut masked_svc = build_service(cfg, weights, rope, head, masked);

    let token_sets: &[&[u32]] = &[
        &[1, 5, 9, 13, 17, 21],
        &[2, 6, 14, 20, 25],
        &[1, 5, 9, 13, 17, 21, 33, 4, 12],
    ];

    for ids in token_sets {
        let p = plain_svc.score_input_ids(ids).expect("plain score");
        let m = masked_svc.score_input_ids(ids).expect("masked score");
        let diff = (p - m).abs();
        assert!(
            diff < 1e-3,
            "scores diverged for ids {ids:?}: plain={p:.6} masked={m:.6} diff={diff:.6e}"
        );
        // Sanity: discriminator output is a probability.
        assert!(p >= 0.0 && p <= 1.0, "plain score out of [0,1]: {p}");
        assert!(m >= 0.0 && m <= 1.0, "masked score out of [0,1]: {m}");
    }
}

#[test]
fn masked_executor_preserves_top1_rank() {
    let cfg = tiny_decoder_config(2, 32, 4, 2, 8, 64);
    let mut rng = ChaCha20Rng::from_seed([57u8; 32]);
    let weights = synth_weights(&cfg, &mut rng);
    let rope = Arc::new(RopeTables::new(
        cfg.head_dim_value(),
        cfg.max_position_embeddings,
        cfg.rope_theta,
    ));
    let head = YesNoHead { yes_token_id: 7, no_token_id: 11 };

    let mut plain_engine = ReferenceCpuEngine::new();
    provision_decoder(&weights, &cfg, &mut plain_engine);
    let plain = PlaintextExecutor::new(plain_engine);
    let mut plain_svc = build_service(cfg.clone(), weights.clone(), rope.clone(), head, plain);

    let mut masked_engine = ReferenceCpuEngine::new();
    provision_decoder(&weights, &cfg, &mut masked_engine);
    let masked =
        InProcessTrustedExecutor::with_seed(masked_engine, MaskSeed::from_bytes([60u8; 32]));
    let mut masked_svc = build_service(cfg, weights, rope, head, masked);

    let docs: &[&[u32]] = &[
        &[2, 9, 14, 21, 4],
        &[8, 11, 25, 3, 17, 30, 6],
        &[1, 13, 22, 5, 28],
    ];

    let plain_scores: Vec<f32> = docs.iter().map(|d| plain_svc.score_input_ids(d).unwrap()).collect();
    let masked_scores: Vec<f32> = docs.iter().map(|d| masked_svc.score_input_ids(d).unwrap()).collect();
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    };
    assert_eq!(argmax(&plain_scores), argmax(&masked_scores));
}

// R2 score-level parity is covered by a unit test inside
// `causal_discriminator.rs` (private helper `score_candidates_batched`
// is not reachable from this external crate). The integration-level
// `rerank()` contract is satisfied by the existing
// `masked_and_plaintext_executors_agree_on_score` test which exercises
// the same forward + yes/no gather code path.

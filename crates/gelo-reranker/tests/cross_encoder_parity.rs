//! Parity test for the BERT-class cross-encoder reranker.
//!
//! Synthetic BertWeights + a synthetic ClassifierHead. The GELO-masked
//! executor must produce the same scalar score (within numerical
//! tolerance) as the plaintext executor — same shape as
//! `gelo-embedder/tests/parity.rs::synthetic_two_layer_parity`, just
//! routed through `CrossEncoderRerankService::score_input_ids` so the
//! head is exercised alongside the encoder forward.
//!
//! Tokenization is intentionally bypassed: the cross-encoder
//! `score_input_ids` entry point takes pre-built ids, decoupling
//! tokenizer-file dependencies from the math we're checking here.

use std::sync::Arc;

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

use gelo_embedder::bert::config::BertConfig;
use gelo_embedder::bert::weights::{BertLayerWeights, BertWeights};
use gelo_embedder::common::tokenizer::HfTokenizer;
use gelo_protocol::rng::MaskSeed;
use gelo_protocol::{
    GpuOffloadEngine, InProcessTrustedExecutor, PlaintextExecutor, ReferenceCpuEngine,
    TrustedExecutor, WeightHandle, WeightKind,
};

use gelo_reranker::cross_encoder::CrossEncoderRerankService;
use gelo_reranker::head::ClassifierHead;

fn tiny_config(num_layers: usize, hidden: usize, heads: usize, ffn: usize) -> BertConfig {
    BertConfig {
        vocab_size: 64,
        hidden_size: hidden,
        num_hidden_layers: num_layers,
        num_attention_heads: heads,
        intermediate_size: ffn,
        max_position_embeddings: 32,
        type_vocab_size: 2,
        layer_norm_eps: 1e-12,
        hidden_act: "gelu".into(),
        max_seq_len: 32,
        skip_first_layers: 0,
        skip_last_layer: false,
        use_out_attn_mult: false,
        out_attn_mult_min_seq_len: None,
    }
}

fn random_array2(rows: usize, cols: usize, rng: &mut impl rand::RngCore, scale: f32) -> Array2<f32> {
    let normal = StandardNormal;
    Array2::from_shape_fn((rows, cols), |_| {
        <StandardNormal as Distribution<f32>>::sample(&normal, rng) * scale
    })
}

fn random_array1(n: usize, rng: &mut impl rand::RngCore, scale: f32) -> Array1<f32> {
    let normal = StandardNormal;
    Array1::from_shape_fn(n, |_| {
        <StandardNormal as Distribution<f32>>::sample(&normal, rng) * scale
    })
}

fn synthetic_weights(cfg: &BertConfig, rng: &mut impl rand::RngCore) -> BertWeights {
    let d = cfg.hidden_size;
    let f = cfg.intermediate_size;
    let layers = (0..cfg.num_hidden_layers)
        .map(|_| BertLayerWeights {
            wq: random_array2(d, d, rng, 0.05),
            bq: random_array1(d, rng, 0.01),
            wk: random_array2(d, d, rng, 0.05),
            bk: random_array1(d, rng, 0.01),
            wv: random_array2(d, d, rng, 0.05),
            bv: random_array1(d, rng, 0.01),
            wo: random_array2(d, d, rng, 0.05),
            bo: random_array1(d, rng, 0.01),
            attn_ln_w: Array1::from_elem(d, 1.0),
            attn_ln_b: Array1::zeros(d),
            w_ffn_up: random_array2(d, f, rng, 0.05),
            b_ffn_up: random_array1(f, rng, 0.01),
            w_ffn_down: random_array2(f, d, rng, 0.05),
            b_ffn_down: random_array1(d, rng, 0.01),
            ffn_ln_w: Array1::from_elem(d, 1.0),
            ffn_ln_b: Array1::zeros(d),
        })
        .collect();
    BertWeights {
        word_embedding: random_array2(cfg.vocab_size, d, rng, 0.05),
        position_embedding: random_array2(cfg.max_position_embeddings, d, rng, 0.05),
        token_type_embedding: random_array2(cfg.type_vocab_size, d, rng, 0.0),
        embeddings_ln_w: Array1::from_elem(d, 1.0),
        embeddings_ln_b: Array1::zeros(d),
        layers,
        model_identity: [0u8; 32],
    }
}

fn synthetic_head(d: usize, rng: &mut impl rand::RngCore) -> ClassifierHead {
    ClassifierHead::from_arrays(
        random_array2(d, d, rng, 0.05),
        random_array1(d, rng, 0.0),
        random_array2(d, 1, rng, 0.05),
        random_array1(1, rng, 0.0),
    )
}

fn provision_layers<E: GpuOffloadEngine>(weights: &BertWeights, cfg: &BertConfig, engine: &mut E) {
    for (li, layer) in weights.layers.iter().enumerate() {
        if !cfg.offload_layer(li) {
            continue;
        }
        let li16 = li as u16;
        engine.register_weight(WeightHandle::new(li16, WeightKind::Q), layer.wq.view()).unwrap();
        engine.register_weight(WeightHandle::new(li16, WeightKind::K), layer.wk.view()).unwrap();
        engine.register_weight(WeightHandle::new(li16, WeightKind::V), layer.wv.view()).unwrap();
        engine.register_weight(WeightHandle::new(li16, WeightKind::O), layer.wo.view()).unwrap();
        engine.register_weight(WeightHandle::new(li16, WeightKind::FfnUp), layer.w_ffn_up.view()).unwrap();
        engine.register_weight(WeightHandle::new(li16, WeightKind::FfnDown), layer.w_ffn_down.view()).unwrap();
    }
}

// Minimal tokenizer.json — produces a known empty-vocab loader; the
// service stores the tokenizer but the parity test calls
// `score_input_ids` directly so it never runs encode.
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
    // pid + rand because tests run in parallel within one process →
    // process id alone collides between concurrent stub_tokenizer
    // callers.
    let tmp = std::env::temp_dir().join(format!(
        "gelo-reranker-stub-tok-cross-{}-{}.json",
        std::process::id(),
        rand::random::<u32>()
    ));
    std::fs::write(&tmp, STUB_TOKENIZER_JSON).expect("write stub tokenizer");
    let tok = HfTokenizer::from_file(&tmp).expect("load stub tokenizer");
    let _ = std::fs::remove_file(&tmp);
    tok
}

fn build_service<X: TrustedExecutor>(
    cfg: BertConfig,
    weights: Arc<BertWeights>,
    head: ClassifierHead,
    exec: X,
) -> CrossEncoderRerankService<X> {
    CrossEncoderRerankService::new(cfg, stub_tokenizer(), weights, head, exec).expect("ctor")
}

#[test]
fn masked_and_plaintext_executors_agree_on_score() {
    let cfg = tiny_config(2, 32, 4, 64);
    let mut rng = ChaCha20Rng::from_seed([3u8; 32]);
    let weights = Arc::new(synthetic_weights(&cfg, &mut rng));
    let head = synthetic_head(cfg.hidden_size, &mut rng);

    let mut plain_engine = ReferenceCpuEngine::new();
    provision_layers(&weights, &cfg, &mut plain_engine);
    let plain = PlaintextExecutor::new(plain_engine);
    let mut plain_svc = build_service(cfg.clone(), weights.clone(), head.clone(), plain);

    let mut masked_engine = ReferenceCpuEngine::new();
    provision_layers(&weights, &cfg, &mut masked_engine);
    let masked =
        InProcessTrustedExecutor::with_seed(masked_engine, MaskSeed::from_bytes([19u8; 32]));
    let mut masked_svc = build_service(cfg, weights, head, masked);

    let token_sets: &[&[u32]] = &[
        &[5, 12, 30, 1, 19, 7],
        &[3, 9, 17, 22, 8],
        &[5, 12, 30, 1, 33, 41, 27, 6],
    ];

    for ids in token_sets {
        let p = plain_svc.score_input_ids(ids).expect("plain score");
        let m = masked_svc.score_input_ids(ids).expect("masked score");
        let diff = (p - m).abs();
        assert!(
            diff < 1e-3,
            "scores diverged for ids {ids:?}: plain={p:.6} masked={m:.6} diff={diff:.6e}"
        );
    }
}

#[test]
fn masked_executor_preserves_top1_rank() {
    let cfg = tiny_config(2, 32, 4, 64);
    let mut rng = ChaCha20Rng::from_seed([5u8; 32]);
    let weights = Arc::new(synthetic_weights(&cfg, &mut rng));
    let head = synthetic_head(cfg.hidden_size, &mut rng);

    let mut plain_engine = ReferenceCpuEngine::new();
    provision_layers(&weights, &cfg, &mut plain_engine);
    let plain = PlaintextExecutor::new(plain_engine);
    let mut plain_svc = build_service(cfg.clone(), weights.clone(), head.clone(), plain);

    let mut masked_engine = ReferenceCpuEngine::new();
    provision_layers(&weights, &cfg, &mut masked_engine);
    let masked =
        InProcessTrustedExecutor::with_seed(masked_engine, MaskSeed::from_bytes([27u8; 32]));
    let mut masked_svc = build_service(cfg, weights, head, masked);

    // Three candidate "documents" expressed as different id sequences.
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

//! Parity tests: a masked `InProcessTrustedExecutor` must produce the same
//! encoder outputs as a `PlaintextExecutor` to within numerical tolerance.
//!
//! Synthetic-weights tests run offline. The full bge-small round trip is
//! gated behind `#[ignore]` because it downloads ~130 MB on first run.

use std::sync::Arc;

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

use gelo_embedder::bert::config::BertConfig;
use gelo_embedder::bert::forward;
use gelo_embedder::bert::weights::{BertLayerWeights, BertWeights};
use gelo_protocol::rng::MaskSeed;
use gelo_protocol::{
    GpuOffloadEngine, InProcessTrustedExecutor, PlaintextExecutor, RayonCpuEngine, WeightHandle,
    WeightKind,
};

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
    }
}

fn random_array2(rows: usize, cols: usize, rng: &mut impl rand::RngCore, scale: f32) -> Array2<f32> {
    let normal = StandardNormal;
    Array2::from_shape_fn((rows, cols), |_| <StandardNormal as Distribution<f32>>::sample(&normal, rng) * scale)
}

fn random_array1(n: usize, rng: &mut impl rand::RngCore, scale: f32) -> Array1<f32> {
    let normal = StandardNormal;
    Array1::from_shape_fn(n, |_| <StandardNormal as Distribution<f32>>::sample(&normal, rng) * scale)
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
    }
}

fn provision_layers<E: GpuOffloadEngine>(weights: &BertWeights, cfg: &BertConfig, engine: &mut E) {
    for (li, layer) in weights.layers.iter().enumerate() {
        if !cfg.offload_layer(li) {
            continue;
        }
        let li16 = li as u16;
        engine.register_weight(WeightHandle::new(li16, WeightKind::Q), layer.wq.view())
            .unwrap();
        engine.register_weight(WeightHandle::new(li16, WeightKind::K), layer.wk.view())
            .unwrap();
        engine.register_weight(WeightHandle::new(li16, WeightKind::V), layer.wv.view())
            .unwrap();
        engine.register_weight(WeightHandle::new(li16, WeightKind::O), layer.wo.view())
            .unwrap();
        engine.register_weight(
            WeightHandle::new(li16, WeightKind::FfnUp),
            layer.w_ffn_up.view(),
        )
        .unwrap();
        engine.register_weight(
            WeightHandle::new(li16, WeightKind::FfnDown),
            layer.w_ffn_down.view(),
        )
        .unwrap();
    }
}

#[test]
fn synthetic_two_layer_parity() {
    // Tiny 2-layer config, hand-rolled random weights.
    let cfg = tiny_config(2, 32, 4, 64);
    let mut rng = ChaCha20Rng::from_seed([1u8; 32]);
    let weights = Arc::new(synthetic_weights(&cfg, &mut rng));

    let input_ids: Vec<u32> = vec![5, 12, 30, 1, 19, 7];

    // Plaintext executor (no mask)
    let mut plain_engine = RayonCpuEngine::new();
    provision_layers(&weights, &cfg, &mut plain_engine);
    let mut plain = PlaintextExecutor::new(plain_engine);
    // Register via TrustedExecutor::provision_weight is also OK; using direct
    // engine register above suffices since PlaintextExecutor::provision_weight
    // delegates to it.
    let plain_out = forward::run(&cfg, &weights, &mut plain, &input_ids).unwrap();

    // Masked executor (deterministic seed)
    let mut masked_engine = RayonCpuEngine::new();
    provision_layers(&weights, &cfg, &mut masked_engine);
    let mut masked =
        InProcessTrustedExecutor::with_seed(masked_engine, MaskSeed::from_bytes([2u8; 32]));
    let masked_out = forward::run(&cfg, &weights, &mut masked, &input_ids).unwrap();

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
        "max abs diff {max_abs} exceeds tolerance — mask round-trip is incorrect"
    );
}

#[test]
fn synthetic_executors_agree_with_sensitive_layer_exclusion() {
    let mut cfg = tiny_config(3, 16, 4, 32);
    cfg.skip_first_layers = 1;
    cfg.skip_last_layer = true;

    let mut rng = ChaCha20Rng::from_seed([5u8; 32]);
    let weights = Arc::new(synthetic_weights(&cfg, &mut rng));

    let input_ids: Vec<u32> = vec![3, 9, 17];

    let mut plain_engine = RayonCpuEngine::new();
    provision_layers(&weights, &cfg, &mut plain_engine);
    let mut plain = PlaintextExecutor::new(plain_engine);
    let plain_out = forward::run(&cfg, &weights, &mut plain, &input_ids).unwrap();

    let mut masked_engine = RayonCpuEngine::new();
    provision_layers(&weights, &cfg, &mut masked_engine);
    let mut masked =
        InProcessTrustedExecutor::with_seed(masked_engine, MaskSeed::from_bytes([13u8; 32]));
    let masked_out = forward::run(&cfg, &weights, &mut masked, &input_ids).unwrap();

    let mut max_abs = 0.0_f32;
    for ((i, j), v) in plain_out.indexed_iter() {
        let diff = (v - masked_out[[i, j]]).abs();
        if diff > max_abs {
            max_abs = diff;
        }
    }
    assert!(
        max_abs < 5e-3,
        "sensitive-layer-exclusion path diverges: max abs diff {max_abs}"
    );
}

#[test]
#[ignore = "downloads ~130 MB from HuggingFace; run with --ignored"]
fn bge_small_parity() {
    use gelo_embedder::GeloBertEmbedder;
    use rag_core::Embedder;

    let mut plain = GeloBertEmbedder::from_pretrained(
        "BAAI/bge-small-en-v1.5",
        PlaintextExecutor::new(RayonCpuEngine::new()),
    )
    .expect("download/load bge-small with PlaintextExecutor");

    let mut masked = GeloBertEmbedder::from_pretrained(
        "BAAI/bge-small-en-v1.5",
        InProcessTrustedExecutor::with_seed(RayonCpuEngine::new(), MaskSeed::from_bytes([42u8; 32])),
    )
    .expect("download/load bge-small with InProcessTrustedExecutor");

    let texts = vec![
        "The quick brown fox jumps over the lazy dog.".to_string(),
        "Confidential computing protects user data.".to_string(),
    ];
    let plain_vecs = plain.embed(&texts).unwrap();
    let masked_vecs = masked.embed(&texts).unwrap();

    assert_eq!(plain_vecs.len(), masked_vecs.len());
    for (pi, mi) in plain_vecs.iter().zip(masked_vecs.iter()) {
        assert_eq!(pi.len(), mi.len());
        let max_abs = pi
            .iter()
            .zip(mi.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs < 2e-3,
            "bge-small embedding diverges: max abs diff {max_abs}"
        );
    }
}

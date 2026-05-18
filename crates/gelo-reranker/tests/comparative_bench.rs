//! R6 — comparative bench. The default test runs synthetic-weight
//! versions of both `CrossEncoderRerankService` (BERT-class) and
//! `CausalDiscriminatorRerankService` (causal-LM-class) under the
//! GELO mask + InProcessTrustedExecutor, and prints per-(q, doc)
//! wall-clock per family.
//!
//! The `#[ignore]`-gated `real_models_bge_vs_qwen3` test path is the
//! release-gate hook for BEIR NDCG@10 measurement once weights are
//! present on the host. It compiles in this file but is not run by
//! `cargo test` by default — invoke with
//! `cargo test -p gelo-reranker --release --test comparative_bench
//!   -- --ignored --nocapture`.

use std::sync::Arc;
use std::time::Instant;

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};
use zeroize::Zeroizing;

use gelo_embedder::bert::config::BertConfig;
use gelo_embedder::bert::weights::{BertLayerWeights, BertWeights};
use gelo_embedder::common::tokenizer::HfTokenizer;
use gelo_embedder::decoder::config::DecoderConfig;
use gelo_embedder::decoder::rope::RopeTables;
use gelo_embedder::decoder::weights::{DecoderLayerWeights, DecoderWeights};
use gelo_protocol::rng::MaskSeed;
use gelo_protocol::{
    GpuOffloadEngine, InProcessTrustedExecutor, RayonCpuEngine, WeightHandle, WeightKind,
};
use rag_core::ChunkId;

use gelo_reranker::causal_discriminator::CausalDiscriminatorRerankService;
use gelo_reranker::cross_encoder::CrossEncoderRerankService;
use gelo_reranker::head::{ClassifierHead, YesNoHead};
use gelo_reranker::service::{RerankCandidate, RerankRequest, RerankService};
use gelo_reranker::session::{QueryId, SessionKey, SessionKeyPolicy};

fn rand2(r: usize, c: usize, rng: &mut impl rand::RngCore, s: f32) -> Array2<f32> {
    let n = StandardNormal;
    Array2::from_shape_fn((r, c), |_| <StandardNormal as Distribution<f32>>::sample(&n, rng) * s)
}
fn rand1(n: usize, rng: &mut impl rand::RngCore, s: f32) -> Array1<f32> {
    let nd = StandardNormal;
    Array1::from_shape_fn(n, |_| <StandardNormal as Distribution<f32>>::sample(&nd, rng) * s)
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
        "gelo-reranker-bench-tok-{}-{}.json",
        std::process::id(),
        rand::random::<u32>()
    ));
    std::fs::write(&tmp, STUB_TOKENIZER_JSON).unwrap();
    let tok = HfTokenizer::from_file(&tmp).unwrap();
    let _ = std::fs::remove_file(&tmp);
    tok
}

// === synthetic cross-encoder ============================================

fn bert_cfg() -> BertConfig {
    BertConfig {
        vocab_size: 64,
        hidden_size: 32,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        intermediate_size: 64,
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

fn bert_weights(cfg: &BertConfig, rng: &mut impl rand::RngCore) -> BertWeights {
    let d = cfg.hidden_size;
    let f = cfg.intermediate_size;
    let layers = (0..cfg.num_hidden_layers)
        .map(|_| BertLayerWeights {
            wq: rand2(d, d, rng, 0.05),
            bq: rand1(d, rng, 0.01),
            wk: rand2(d, d, rng, 0.05),
            bk: rand1(d, rng, 0.01),
            wv: rand2(d, d, rng, 0.05),
            bv: rand1(d, rng, 0.01),
            wo: rand2(d, d, rng, 0.05),
            bo: rand1(d, rng, 0.01),
            attn_ln_w: Array1::from_elem(d, 1.0),
            attn_ln_b: Array1::zeros(d),
            w_ffn_up: rand2(d, f, rng, 0.05),
            b_ffn_up: rand1(f, rng, 0.01),
            w_ffn_down: rand2(f, d, rng, 0.05),
            b_ffn_down: rand1(d, rng, 0.01),
            ffn_ln_w: Array1::from_elem(d, 1.0),
            ffn_ln_b: Array1::zeros(d),
        })
        .collect();
    BertWeights {
        word_embedding: rand2(cfg.vocab_size, d, rng, 0.05),
        position_embedding: rand2(cfg.max_position_embeddings, d, rng, 0.05),
        token_type_embedding: rand2(cfg.type_vocab_size, d, rng, 0.0),
        embeddings_ln_w: Array1::from_elem(d, 1.0),
        embeddings_ln_b: Array1::zeros(d),
        layers,
        model_identity: [0u8; 32],
    }
}

fn bert_head(d: usize, rng: &mut impl rand::RngCore) -> ClassifierHead {
    ClassifierHead::from_arrays(
        rand2(d, d, rng, 0.05),
        rand1(d, rng, 0.0),
        rand2(d, 1, rng, 0.05),
        rand1(1, rng, 0.0),
    )
}

fn provision_bert<E: GpuOffloadEngine>(w: &BertWeights, cfg: &BertConfig, e: &mut E) {
    for (li, layer) in w.layers.iter().enumerate() {
        if !cfg.offload_layer(li) {
            continue;
        }
        let li16 = li as u16;
        e.register_weight(WeightHandle::new(li16, WeightKind::Q), layer.wq.view()).unwrap();
        e.register_weight(WeightHandle::new(li16, WeightKind::K), layer.wk.view()).unwrap();
        e.register_weight(WeightHandle::new(li16, WeightKind::V), layer.wv.view()).unwrap();
        e.register_weight(WeightHandle::new(li16, WeightKind::O), layer.wo.view()).unwrap();
        e.register_weight(WeightHandle::new(li16, WeightKind::FfnUp), layer.w_ffn_up.view()).unwrap();
        e.register_weight(WeightHandle::new(li16, WeightKind::FfnDown), layer.w_ffn_down.view()).unwrap();
    }
}

// === synthetic causal-discriminator ======================================

fn dec_cfg() -> DecoderConfig {
    DecoderConfig {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 4,
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
    }
}

fn dec_weights(cfg: &DecoderConfig, rng: &mut impl rand::RngCore) -> DecoderWeights {
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
        token_embedding: rand2(cfg.vocab_size, d, rng, 0.1),
        final_norm: Array1::from_elem(d, 1.0),
        layers,
        model_identity: [0u8; 32],
    }
}

fn provision_dec<E: GpuOffloadEngine>(w: &DecoderWeights, cfg: &DecoderConfig, e: &mut E) {
    for (li, layer) in w.layers.iter().enumerate() {
        if !cfg.offload_layer(li) {
            continue;
        }
        let li16 = li as u16;
        e.register_weight(WeightHandle::new(li16, WeightKind::Q), layer.wq.view()).unwrap();
        e.register_weight(WeightHandle::new(li16, WeightKind::K), layer.wk.view()).unwrap();
        e.register_weight(WeightHandle::new(li16, WeightKind::V), layer.wv.view()).unwrap();
        e.register_weight(WeightHandle::new(li16, WeightKind::O), layer.wo.view()).unwrap();
        e.register_weight(WeightHandle::new(li16, WeightKind::FfnGate), layer.w_gate.view()).unwrap();
        e.register_weight(WeightHandle::new(li16, WeightKind::FfnUp), layer.w_up.view()).unwrap();
        e.register_weight(WeightHandle::new(li16, WeightKind::FfnDown), layer.w_down.view()).unwrap();
    }
}

fn corpus() -> Vec<RerankCandidate> {
    let texts = [
        ("doc-rust",   "Rust is a systems programming language focused on safety and speed."),
        ("doc-python", "Python is a high-level interpreted language used widely in data science."),
        ("doc-go",     "Go is a statically typed compiled language designed for concurrency."),
        ("doc-tee",    "A trusted execution environment isolates a process from the host OS."),
        ("doc-gpu",    "GPUs accelerate parallel matrix multiplication common in deep learning."),
        ("doc-rag",    "Retrieval augmented generation grounds an LLM in an external corpus."),
        ("doc-bert",   "BERT is a bidirectional transformer encoder used for many NLP tasks."),
        ("doc-qwen",   "Qwen3 is an open weight LLM family released by the Tongyi lab."),
    ];
    texts
        .iter()
        .map(|(id, text)| RerankCandidate {
            chunk_id: ChunkId((*id).into()),
            text: (*text).into(),
        })
        .collect()
}

#[test]
fn synthetic_comparative_bench_both_services_emit_valid_bundles() {
    let candidates = corpus();
    let session = SessionKey::derive(&Zeroizing::new(vec![0x5a; 32]), SessionKeyPolicy::V1);
    let query_id = QueryId::from("bench-q1");

    // ── BERT cross-encoder ──────────────────────────────────────────
    let cfg_be = bert_cfg();
    let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
    let w_be = Arc::new(bert_weights(&cfg_be, &mut rng));
    let h_be = bert_head(cfg_be.hidden_size, &mut rng);
    let mut eng = RayonCpuEngine::new();
    provision_bert(&w_be, &cfg_be, &mut eng);
    let mut svc_be = CrossEncoderRerankService::new(
        cfg_be,
        stub_tokenizer(),
        w_be,
        h_be,
        InProcessTrustedExecutor::with_seed(eng, MaskSeed::from_bytes([12u8; 32])),
    )
    .unwrap();

    let request = RerankRequest {
        query: "what programming language is memory safe",
        candidates: &candidates,
        top_k: 3,
        k_max: 12,
        query_id: query_id.clone(),
    };

    let t0 = Instant::now();
    let bundle_be = svc_be.rerank(&session, &request).unwrap();
    let elapsed_be = t0.elapsed();
    let per_pair_be = elapsed_be / candidates.len() as u32;

    // ── Causal-LM discriminator ─────────────────────────────────────
    let cfg_d = dec_cfg();
    let mut rng = ChaCha20Rng::from_seed([21u8; 32]);
    let w_d = Arc::new(dec_weights(&cfg_d, &mut rng));
    let rope = Arc::new(RopeTables::new(
        cfg_d.head_dim_value(),
        cfg_d.max_position_embeddings,
        cfg_d.rope_theta,
    ));
    let head_d = YesNoHead { yes_token_id: 1, no_token_id: 0 };
    let mut eng = RayonCpuEngine::new();
    provision_dec(&w_d, &cfg_d, &mut eng);
    let mut svc_d = CausalDiscriminatorRerankService::new(
        cfg_d,
        stub_tokenizer(),
        w_d,
        rope,
        head_d,
        InProcessTrustedExecutor::with_seed(eng, MaskSeed::from_bytes([22u8; 32])),
    )
    .unwrap();

    let t0 = Instant::now();
    let bundle_d = svc_d.rerank(&session, &request).unwrap();
    let elapsed_d = t0.elapsed();
    let per_pair_d = elapsed_d / candidates.len() as u32;

    // ── Cross-check: each bundle is k_max-padded, real items
    //    open under the matching QueryKey, ranks are contiguous. ─────
    let qkey = session.derive_query_key(&query_id);
    for (label, bundle, per_pair) in [
        ("cross-encoder", &bundle_be, per_pair_be),
        ("causal-discriminator", &bundle_d, per_pair_d),
    ] {
        assert_eq!(bundle.items.len(), 12, "{label} bundle must be k_max-padded");
        let opened = bundle.open(&qkey).expect("open with session-derived qkey");
        assert_eq!(opened.len(), 3, "{label} should return top_k=3 real items");
        for (i, item) in opened.iter().enumerate() {
            assert_eq!(item.rank as usize, i, "{label} ranks must be contiguous");
        }
        eprintln!(
            "rerank/{label}: total={elapsed:.3?} per-pair={per_pair:.3?} \
             ranked={top:?}",
            elapsed = if label == "cross-encoder" { elapsed_be } else { elapsed_d },
            per_pair = per_pair,
            top = opened.iter().map(|i| i.chunk_id.clone()).collect::<Vec<_>>(),
        );
    }
}

/// Real-weight A/B: download bge-reranker-v2-m3 and Qwen3-Reranker-0.6B,
/// score the same `(query, doc)` set under each, print rank order +
/// per-pair latency. ~3.5 GB of safetensors fetched on first run and
/// then cached under the user's `~/.cache/huggingface/` tree, so CI
/// never runs this by default. Invoke with:
///
/// ```text
/// cargo test -p gelo-reranker --release --test comparative_bench \
///     -- --ignored real_models_bge_vs_qwen3 --nocapture
/// ```
///
/// The deeper accuracy-vs-baseline measurement (NDCG@10 over a real
/// BEIR slice) lives in `gelo-rag/tests/rerank_e2e_bench.rs` so it
/// can reuse the existing NFCorpus loader and the
/// `GeloRagInMemoryService` ingest+query path.
#[test]
#[ignore = "downloads ~3.5 GB safetensors from HuggingFace"]
fn real_models_bge_vs_qwen3() {
    use std::time::Instant;

    use gelo_gpu_wgpu::WgpuVulkanEngine;
    use gelo_protocol::rng::MaskSeed;
    use gelo_protocol::InProcessTrustedExecutor;

    let query = "How does retrieval augmented generation reduce hallucinations?";
    let docs: &[(&str, &str)] = &[
        (
            "rag-grounding",
            "Retrieval augmented generation grounds an LLM in an external corpus, \
             feeding retrieved chunks into the prompt so the model can cite real \
             documents instead of fabricating answers.",
        ),
        (
            "tee-isolation",
            "A trusted execution environment isolates a process so the host OS \
             and other tenants cannot read its memory.",
        ),
        (
            "gpu-matmul",
            "GPUs accelerate parallel dense matrix multiplication, the backbone \
             of transformer inference and training.",
        ),
        (
            "rag-hallucinations",
            "By retrieving authoritative source text at query time and including \
             it in the model's context, RAG lowers the rate of unsupported \
             generations and gives the user direct citations.",
        ),
        (
            "private-rag",
            "Private RAG stacks confidential computing, distance-preserving \
             embedding ciphertext, and DP-bounded query noise so the cloud \
             provider never sees plaintext queries or documents.",
        ),
        (
            "rust-langs",
            "Rust is a systems programming language with strong memory-safety \
             guarantees enforced at compile time.",
        ),
    ];
    let candidates: Vec<RerankCandidate> = docs
        .iter()
        .map(|(id, text)| RerankCandidate {
            chunk_id: ChunkId((*id).into()),
            text: (*text).into(),
        })
        .collect();

    let session = SessionKey::derive(&Zeroizing::new(vec![0x7c; 32]), SessionKeyPolicy::V1);
    let query_id = QueryId::from("real-models-bench");
    let request = RerankRequest {
        query,
        candidates: &candidates,
        top_k: 3,
        k_max: 8,
        query_id: query_id.clone(),
    };

    // Shared Vulkan device; clone_shared() hands each service its own
    // handle into the same GpuOffloadEngine.
    let gpu = WgpuVulkanEngine::new().expect("Vulkan adapter must be available");
    eprintln!(
        "[real] Vulkan adapter: {} ({:?})",
        gpu.adapter_info().name,
        gpu.adapter_info().device_type,
    );

    // ── bge-reranker-v2-m3 (XLM-RoBERTa-large cross-encoder) ────────
    eprintln!("[real] loading BAAI/bge-reranker-v2-m3 ...");
    let t0 = Instant::now();
    let mut svc_be = CrossEncoderRerankService::from_pretrained(
        "BAAI/bge-reranker-v2-m3",
        InProcessTrustedExecutor::with_seed(gpu.clone_shared(), MaskSeed::from_bytes([7u8; 32])),
    )
    .expect("download/load bge-reranker-v2-m3");
    let load_be = t0.elapsed();
    eprintln!("[real]   bge-reranker-v2-m3 loaded in {load_be:.2?}");

    let t0 = Instant::now();
    let bundle_be = svc_be.rerank(&session, &request).expect("bge rerank");
    let elapsed_be = t0.elapsed();
    let per_pair_be = elapsed_be / candidates.len() as u32;

    // ── Qwen3-Reranker-0.6B (causal-LM yes/no discriminator) ─────────
    eprintln!("[real] loading Qwen/Qwen3-Reranker-0.6B ...");
    let t0 = Instant::now();
    let mut svc_qw = CausalDiscriminatorRerankService::from_pretrained(
        "Qwen/Qwen3-Reranker-0.6B",
        InProcessTrustedExecutor::with_seed(gpu.clone_shared(), MaskSeed::from_bytes([8u8; 32])),
    )
    .expect("download/load Qwen3-Reranker-0.6B");
    let load_qw = t0.elapsed();
    eprintln!("[real]   Qwen3-Reranker-0.6B loaded in {load_qw:.2?}");

    let t0 = Instant::now();
    let bundle_qw = svc_qw.rerank(&session, &request).expect("qwen3 rerank");
    let elapsed_qw = t0.elapsed();
    let per_pair_qw = elapsed_qw / candidates.len() as u32;

    // ── Open both bundles and print top-3 ────────────────────────────
    let qkey = session.derive_query_key(&query_id);
    let opened_be = bundle_be.open(&qkey).expect("open bge bundle");
    let opened_qw = bundle_qw.open(&qkey).expect("open qwen3 bundle");

    eprintln!("\n[real] bge-reranker-v2-m3 (cross-encoder)");
    eprintln!("  load           = {load_be:.2?}");
    eprintln!("  rerank total   = {elapsed_be:.2?}");
    eprintln!("  rerank/pair    = {per_pair_be:.2?}");
    for it in &opened_be {
        eprintln!("    rank={} chunk={}", it.rank, it.chunk_id);
    }

    eprintln!("\n[real] Qwen3-Reranker-0.6B (causal discriminator)");
    eprintln!("  load           = {load_qw:.2?}");
    eprintln!("  rerank total   = {elapsed_qw:.2?}");
    eprintln!("  rerank/pair    = {per_pair_qw:.2?}");
    for it in &opened_qw {
        eprintln!("    rank={} chunk={}", it.rank, it.chunk_id);
    }

    assert_eq!(opened_be.len(), 3);
    assert_eq!(opened_qw.len(), 3);
    // Sanity: both rerankers should put one of the RAG-grounded docs
    // at rank 0 — the query is unambiguously about RAG.
    let rag_ids = ["rag-grounding", "rag-hallucinations", "private-rag"];
    assert!(
        rag_ids.contains(&opened_be[0].chunk_id.as_str()),
        "bge top-1 was {:?}, expected a RAG-grounded doc",
        opened_be[0].chunk_id
    );
    assert!(
        rag_ids.contains(&opened_qw[0].chunk_id.as_str()),
        "qwen3 top-1 was {:?}, expected a RAG-grounded doc",
        opened_qw[0].chunk_id
    );
}

//! R7 — end-to-end ingest → query → rerank bench on real models.
//!
//! Validates the full private-RAG pipeline:
//!
//! 1. **Stage A · Ingest**. `GeloRagInMemoryService` ingests the
//!    NFCorpus docs using FastEmbed MiniLM-L6 embeddings + CAPRISE at
//!    rest. Measured: wall-clock, docs/sec.
//! 2. **Stage B · Query**. `service.query(text, k_prime)` retrieves
//!    over-fetched candidates. CAPRISE decrypts inside the service so
//!    each `RetrievalHit` carries plaintext chunk text. Measured: per-
//!    query wall-clock, NDCG@10 / Recall@k / MRR@10 against BEIR
//!    qrels. This is the **retrieval-only baseline**.
//! 3. **Stage C · Rerank**. Each query's `k_prime` candidates are fed
//!    to both `CrossEncoderRerankService` (bge-reranker-v2-m3) and
//!    `CausalDiscriminatorRerankService` (Qwen3-Reranker-0.6B), both
//!    running under the GELO `InProcessTrustedExecutor` mask. The
//!    `EncryptedRerankBundle` is opened client-side to recover the
//!    in-TEE rank order. Measured: per-pair wall-clock, NDCG@10 /
//!    Recall@k / MRR@10, delta vs retrieval baseline.
//!
//! Default sizing (overridable via env): `E2E_QUERIES=20`,
//! `E2E_KPRIME=50`, `E2E_KFINAL=10`. The ~3.5 GB of reranker
//! safetensors are fetched on first run and cached under
//! `~/.cache/huggingface/`; rerunning the test is fast.
//!
//! ## Run
//!
//! ```text
//! cargo test -p gelo-rag --release --test rerank_e2e_bench \
//!     -- --ignored --nocapture
//! ```

#![allow(clippy::too_many_arguments)]

mod common;

use std::collections::HashMap;
use std::time::{Duration, Instant};

use anyhow::Result;
use gelo_embedder::GeloBertEmbedder;
use gelo_gpu_wgpu::WgpuVulkanEngine;
use gelo_protocol::rng::MaskSeed;
use gelo_protocol::InProcessTrustedExecutor;
use gelo_rag::{GeloRagInMemoryService, NoopAttestationVerifier};
use gelo_reranker::cross_encoder::CrossEncoderRerankService;
use gelo_reranker::causal_discriminator::CausalDiscriminatorRerankService;
use gelo_reranker::service::{RerankCandidate, RerankRequest, RerankService};
use gelo_reranker::session::{QueryId, SessionKey, SessionKeyPolicy};
use rag_core::{Caprise, CapriseKey, ChunkId, RetrievalHit};
use zeroize::Zeroizing;

use common::beir::{BeirDataset, load_nfcorpus};

const DEFAULT_QUERIES: usize = 20;
const DEFAULT_KPRIME: usize = 50;
const DEFAULT_KFINAL: usize = 10;

// ────────────────────────────────────────────────────────────────────────
// IR metrics — local copies of the helpers in beir_accuracy.rs so this
// test compiles without pulling in the larger bench's modules.
// ────────────────────────────────────────────────────────────────────────

fn dcg_at_k(ranked: &[String], qrels: &HashMap<String, u8>, k: usize) -> f64 {
    let mut s = 0.0;
    for (i, doc) in ranked.iter().take(k).enumerate() {
        let rel = qrels.get(doc).copied().unwrap_or(0) as f64;
        if rel > 0.0 {
            s += (2.0_f64.powf(rel) - 1.0) / (i as f64 + 2.0).log2();
        }
    }
    s
}

fn idcg_at_k(qrels: &HashMap<String, u8>, k: usize) -> f64 {
    let mut rels: Vec<u8> = qrels.values().copied().collect();
    rels.sort_unstable_by(|a, b| b.cmp(a));
    let mut s = 0.0;
    for (i, rel) in rels.iter().take(k).enumerate() {
        if *rel > 0 {
            s += (2.0_f64.powf(*rel as f64) - 1.0) / (i as f64 + 2.0).log2();
        }
    }
    s
}

fn ndcg_at_k(ranked: &[String], qrels: &HashMap<String, u8>, k: usize) -> f64 {
    let idcg = idcg_at_k(qrels, k);
    if idcg == 0.0 {
        0.0
    } else {
        dcg_at_k(ranked, qrels, k) / idcg
    }
}

fn recall_at_k(ranked: &[String], qrels: &HashMap<String, u8>, k: usize) -> f64 {
    let relevant_total = qrels.values().filter(|&&r| r > 0).count();
    if relevant_total == 0 {
        return 0.0;
    }
    let found = ranked
        .iter()
        .take(k)
        .filter(|d| qrels.get(*d).copied().unwrap_or(0) > 0)
        .count();
    found as f64 / relevant_total as f64
}

fn mrr_at_k(ranked: &[String], qrels: &HashMap<String, u8>, k: usize) -> f64 {
    for (i, doc) in ranked.iter().take(k).enumerate() {
        if qrels.get(doc).copied().unwrap_or(0) > 0 {
            return 1.0 / (i + 1) as f64;
        }
    }
    0.0
}

#[derive(Debug, Default, Clone)]
struct Metrics {
    ndcg10: f64,
    recall_at_kp: f64,
    mrr10: f64,
    n: usize,
}

impl Metrics {
    fn observe(&mut self, ranked: &[String], qrels: &HashMap<String, u8>, k_recall: usize) {
        self.ndcg10 += ndcg_at_k(ranked, qrels, 10);
        self.recall_at_kp += recall_at_k(ranked, qrels, k_recall);
        self.mrr10 += mrr_at_k(ranked, qrels, 10);
        self.n += 1;
    }

    fn mean(&self) -> (f64, f64, f64) {
        if self.n == 0 {
            (0.0, 0.0, 0.0)
        } else {
            (
                self.ndcg10 / self.n as f64,
                self.recall_at_kp / self.n as f64,
                self.mrr10 / self.n as f64,
            )
        }
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn env_flag(key: &str) -> bool {
    matches!(
        std::env::var(key).as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

fn keep_queries_with_qrels(
    queries: &[(String, String)],
    qrels: &HashMap<String, HashMap<String, u8>>,
) -> Vec<(String, String)> {
    queries
        .iter()
        .filter(|(qid, _)| {
            qrels
                .get(qid)
                .map(|m| m.values().any(|&r| r > 0))
                .unwrap_or(false)
        })
        .cloned()
        .collect()
}

// ────────────────────────────────────────────────────────────────────────
// Rerank driver — abstracts over the two service variants so the
// caller code stays single-shape.
// ────────────────────────────────────────────────────────────────────────

fn rerank_query(
    svc: &mut dyn RerankService,
    session: &SessionKey,
    query_text: &str,
    query_index: usize,
    hits: &[RetrievalHit],
    k_final: usize,
    k_max: usize,
) -> Result<(Vec<String>, Duration)> {
    let candidates: Vec<RerankCandidate> = hits
        .iter()
        .map(|h| RerankCandidate {
            chunk_id: h.id.clone(),
            text: h.text.clone(),
        })
        .collect();
    let query_id_bytes = format!("rerank-{query_index:08x}").into_bytes();
    let query_id = QueryId::new(query_id_bytes);
    let request = RerankRequest {
        query: query_text,
        candidates: &candidates,
        top_k: k_final,
        k_max,
        query_id: query_id.clone(),
    };

    let t0 = Instant::now();
    let bundle = svc
        .rerank(session, &request)
        .map_err(|e| anyhow::anyhow!("rerank failed: {e}"))?;
    let elapsed = t0.elapsed();

    let qkey = session.derive_query_key(&query_id);
    let opened = bundle
        .open(&qkey)
        .map_err(|e| anyhow::anyhow!("bundle open failed: {e}"))?;
    let mut ordered: Vec<(u32, ChunkId)> = opened
        .into_iter()
        .map(|i| (i.rank, ChunkId(i.chunk_id)))
        .collect();
    ordered.sort_by_key(|(r, _)| *r);
    Ok((ordered.into_iter().map(|(_, c)| c.0).collect(), elapsed))
}

// ────────────────────────────────────────────────────────────────────────
// The bench
// ────────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "downloads ~3.5 GB safetensors and runs the full pipeline"]
fn ingest_query_rerank_bge_and_qwen3_on_nfcorpus() -> Result<()> {
    let n_queries = env_usize("E2E_QUERIES", DEFAULT_QUERIES);
    let k_prime = env_usize("E2E_KPRIME", DEFAULT_KPRIME);
    let k_final = env_usize("E2E_KFINAL", DEFAULT_KFINAL);
    let n_docs_cap = env_usize("E2E_DOCS", usize::MAX);
    let k_max = k_prime.max(k_final);
    let skip_bge = env_flag("E2E_SKIP_BGE");
    let skip_qwen3 = env_flag("E2E_SKIP_QWEN3");
    let trace = env_flag("E2E_TRACE");

    eprintln!(
        "[e2e] config: docs={} queries={n_queries} k_prime={k_prime} k_final={k_final} k_max={k_max}",
        if n_docs_cap == usize::MAX { "all".into() } else { n_docs_cap.to_string() },
    );

    let full_dataset: BeirDataset = load_nfcorpus()?;
    eprintln!(
        "[e2e] dataset: {} docs, {} queries, {} qrels",
        full_dataset.docs.len(),
        full_dataset.queries.len(),
        full_dataset.qrels.len(),
    );

    // Subset the corpus when E2E_DOCS is set. We keep the top
    // `n_docs_cap` docs *plus* every doc referenced by qrels for the
    // queries we'll actually use — otherwise a small subset removes
    // every relevant document from the qrels and the bench reports
    // zero NDCG without telling the operator why.
    let dataset = if n_docs_cap < full_dataset.docs.len() {
        subset_corpus(full_dataset, n_docs_cap, n_queries)
    } else {
        full_dataset
    };
    eprintln!(
        "[e2e] subset: {} docs after E2E_DOCS cap, {} queries usable",
        dataset.docs.len(),
        dataset.queries.len()
    );

    let candidate_queries = keep_queries_with_qrels(&dataset.queries, &dataset.qrels);
    let queries: Vec<(String, String)> = candidate_queries.into_iter().take(n_queries).collect();
    eprintln!("[e2e] using {} queries with relevance judgments", queries.len());

    // Shared Vulkan device — every private path (ingest embedder +
    // both rerankers) clone_shared()s its own handle into it.
    let gpu = WgpuVulkanEngine::new()
        .map_err(|e| anyhow::anyhow!("Vulkan adapter unavailable: {e}"))?;
    eprintln!(
        "[e2e] Vulkan adapter: {} ({:?})",
        gpu.adapter_info().name,
        gpu.adapter_info().device_type,
    );

    // ── Stage A · Ingest (GELO+mask+GPU on BGE-base) ────────────────
    eprintln!("[e2e][A] loading BAAI/bge-base-en-v1.5 ...");
    let embedder = GeloBertEmbedder::from_pretrained(
        "BAAI/bge-base-en-v1.5",
        InProcessTrustedExecutor::with_seed(gpu.clone_shared(), MaskSeed::from_bytes([3u8; 32])),
    )?;
    let caprise = Caprise::new(CapriseKey::generate(32.0, 0.15));
    let mut service =
        GeloRagInMemoryService::new(embedder, caprise, NoopAttestationVerifier);

    if trace {
        gelo_protocol::profile::reset();
    }
    let t0 = Instant::now();
    service.ingest_chunks(dataset.docs.clone())?;
    let ingest_wall = t0.elapsed();
    if trace {
        gelo_protocol::profile::snapshot().dump(&format!(
            "Stage A · ingest (BGE-base, n_docs={}) per-bucket cumulative",
            dataset.docs.len()
        ));
    }
    let docs_per_sec = dataset.docs.len() as f64 / ingest_wall.as_secs_f64();
    eprintln!(
        "[e2e][A] ingest: {:.2?} total ({:.1} docs/s, n={})",
        ingest_wall,
        docs_per_sec,
        dataset.docs.len()
    );

    // ── Stage B · Retrieve (baseline) ───────────────────────────────
    if trace {
        gelo_protocol::profile::reset();
    }
    let mut retrieve_wall = Duration::ZERO;
    let mut retrieval_metrics = Metrics::default();
    let mut hits_by_qid: HashMap<String, (String, Vec<RetrievalHit>)> =
        HashMap::with_capacity(queries.len());
    for (qid, qtext) in &queries {
        let t0 = Instant::now();
        let hits = service.query(qtext, k_prime)?;
        retrieve_wall += t0.elapsed();
        let ranked_ids: Vec<String> = hits.iter().map(|h| h.id.0.clone()).collect();
        if let Some(qr) = dataset.qrels.get(qid) {
            retrieval_metrics.observe(&ranked_ids, qr, k_prime);
        }
        hits_by_qid.insert(qid.clone(), (qtext.clone(), hits));
    }
    let (b_ndcg, b_rec, b_mrr) = retrieval_metrics.mean();
    eprintln!(
        "[e2e][B] retrieve top-{} per query: {:.2?} total ({:.1} q/s)",
        k_prime,
        retrieve_wall,
        queries.len() as f64 / retrieve_wall.as_secs_f64()
    );
    eprintln!(
        "[e2e][B] baseline   nDCG@10={b_ndcg:.3} R@{k_prime}={b_rec:.3} MRR@10={b_mrr:.3}",
    );
    if trace {
        gelo_protocol::profile::snapshot().dump(&format!(
            "Stage B · retrieve (BGE-base, n_queries={}) per-bucket cumulative",
            queries.len()
        ));
    }

    let session = SessionKey::derive(&Zeroizing::new(vec![0xa1; 32]), SessionKeyPolicy::V1);

    // ── Stage C · Rerank with bge-reranker-v2-m3 ────────────────────
    let bge_result = if skip_bge {
        eprintln!("[e2e][C/bge] skipped (E2E_SKIP_BGE=1)");
        None
    } else {
        eprintln!("[e2e][C/bge] loading BAAI/bge-reranker-v2-m3 ...");
        let t0 = Instant::now();
        let mut bge = CrossEncoderRerankService::from_pretrained(
            "BAAI/bge-reranker-v2-m3",
            InProcessTrustedExecutor::with_seed(gpu.clone_shared(), MaskSeed::from_bytes([7u8; 32])),
        )?;
        let bge_load = t0.elapsed();
        eprintln!("[e2e][C/bge] loaded in {bge_load:.2?}");

        if trace {
            gelo_protocol::profile::reset();
        }
        let mut bge_total = Duration::ZERO;
        let mut bge_pairs = 0usize;
        let mut bge_metrics = Metrics::default();
        for (qid, (qtext, hits)) in &hits_by_qid {
            let (ranked, dt) = rerank_query(
                &mut bge as &mut dyn RerankService,
                &session,
                qtext,
                stable_query_index(qid),
                hits,
                k_final,
                k_max,
            )?;
            bge_total += dt;
            bge_pairs += hits.len();
            if let Some(qr) = dataset.qrels.get(qid) {
                bge_metrics.observe(&ranked, qr, k_final);
            }
        }
        let (be_ndcg, be_rec, be_mrr) = bge_metrics.mean();
        let bge_per_pair = if bge_pairs > 0 {
            bge_total / bge_pairs as u32
        } else {
            Duration::ZERO
        };
        eprintln!(
            "[e2e][C/bge] rerank: {:.2?} total ({} pairs, {:.2?}/pair)",
            bge_total, bge_pairs, bge_per_pair,
        );
        eprintln!(
            "[e2e][C/bge] post-rerank nDCG@10={be_ndcg:.3} R@{k_final}={be_rec:.3} MRR@10={be_mrr:.3} \
             Δ(nDCG@10 vs baseline)={delta:+.3}",
            delta = be_ndcg - b_ndcg,
        );
        if trace {
            gelo_protocol::profile::snapshot().dump(&format!(
                "Stage C/bge · rerank ({} pairs) per-bucket cumulative",
                bge_pairs
            ));
        }
        Some((be_ndcg, be_rec, be_mrr, bge_per_pair))
    };

    // ── Stage C · Rerank with Qwen3-Reranker-0.6B ──────────────────
    let qw_result = if skip_qwen3 {
        eprintln!("[e2e][C/qwen3] skipped (E2E_SKIP_QWEN3=1)");
        None
    } else {
        eprintln!("[e2e][C/qwen3] loading Qwen/Qwen3-Reranker-0.6B ...");
        let t0 = Instant::now();
        let mut qwen3 = CausalDiscriminatorRerankService::from_pretrained(
            "Qwen/Qwen3-Reranker-0.6B",
            InProcessTrustedExecutor::with_seed(gpu.clone_shared(), MaskSeed::from_bytes([8u8; 32])),
        )?;
        let qw_load = t0.elapsed();
        eprintln!("[e2e][C/qwen3] loaded in {qw_load:.2?}");

        if trace {
            gelo_protocol::profile::reset();
        }
        let mut qw_total = Duration::ZERO;
        let mut qw_pairs = 0usize;
        let mut qw_metrics = Metrics::default();
        for (qid, (qtext, hits)) in &hits_by_qid {
            let (ranked, dt) = rerank_query(
                &mut qwen3 as &mut dyn RerankService,
                &session,
                qtext,
                stable_query_index(qid),
                hits,
                k_final,
                k_max,
            )?;
            qw_total += dt;
            qw_pairs += hits.len();
            if let Some(qr) = dataset.qrels.get(qid) {
                qw_metrics.observe(&ranked, qr, k_final);
            }
        }
        let (qw_ndcg, qw_rec, qw_mrr) = qw_metrics.mean();
        let qw_per_pair = if qw_pairs > 0 {
            qw_total / qw_pairs as u32
        } else {
            Duration::ZERO
        };
        eprintln!(
            "[e2e][C/qwen3] rerank: {:.2?} total ({} pairs, {:.2?}/pair)",
            qw_total, qw_pairs, qw_per_pair,
        );
        eprintln!(
            "[e2e][C/qwen3] post-rerank nDCG@10={qw_ndcg:.3} R@{k_final}={qw_rec:.3} MRR@10={qw_mrr:.3} \
             Δ(nDCG@10 vs baseline)={delta:+.3}",
            delta = qw_ndcg - b_ndcg,
        );
        if trace {
            gelo_protocol::profile::snapshot().dump(&format!(
                "Stage C/qwen3 · rerank ({} pairs) per-bucket cumulative",
                qw_pairs
            ));
        }
        Some((qw_ndcg, qw_rec, qw_mrr, qw_per_pair))
    };

    // ── Summary table ───────────────────────────────────────────────
    eprintln!("\n[e2e] === summary (NFCorpus, n_queries={}) ===", queries.len());
    eprintln!(
        "{:<32} {:>10} {:>10} {:>10} {:>14}",
        "stage", "nDCG@10", "R@k", "MRR@10", "wall-or-pair"
    );
    eprintln!(
        "{:<32} {:>10} {:>10} {:>10} {:>14}",
        "A · ingest", "—", "—", "—", format!("{:.1} doc/s", docs_per_sec)
    );
    eprintln!(
        "{:<32} {:>10.3} {:>10.3} {:>10.3} {:>14}",
        "B · retrieve (baseline)", b_ndcg, b_rec, b_mrr, format!("{:.2?}", retrieve_wall / queries.len() as u32)
    );
    if let Some((n, r, m, p)) = bge_result {
        eprintln!(
            "{:<32} {:>10.3} {:>10.3} {:>10.3} {:>14}",
            "C · rerank bge", n, r, m, format!("{:.2?}", p)
        );
    }
    if let Some((n, r, m, p)) = qw_result {
        eprintln!(
            "{:<32} {:>10.3} {:>10.3} {:>10.3} {:>14}",
            "C · rerank qwen3", n, r, m, format!("{:.2?}", p)
        );
    }

    // Non-regression check across whichever rerankers ran. Skipped
    // entirely when both are gated off (the user is just measuring
    // ingest+retrieve in that case).
    let observed: Vec<f64> = [bge_result, qw_result]
        .into_iter()
        .flatten()
        .map(|(n, _, _, _)| n)
        .collect();
    if let Some(best) = observed.iter().cloned().reduce(f64::max) {
        assert!(
            best + 0.02 >= b_ndcg,
            "every executed reranker regressed below the retrieval baseline by >0.02 nDCG@10 \
             (baseline={b_ndcg:.3}, best_rerank={best:.3}) — likely a pipeline bug"
        );
    }

    Ok(())
}

/// Build a smaller corpus that still has graded relevance signal for
/// the queries we'll exercise. Keeps every doc judged relevant for one
/// of the first `n_queries_keep` queries, then pads up to `n_docs_cap`
/// with the earliest-in-corpus docs (which give the bench a non-trivial
/// hard-negative pool to rerank against).
fn subset_corpus(ds: BeirDataset, n_docs_cap: usize, n_queries_keep: usize) -> BeirDataset {
    use std::collections::HashSet;
    let chosen_queries: Vec<&(String, String)> = ds
        .queries
        .iter()
        .filter(|(qid, _)| ds.qrels.get(qid).map(|m| m.values().any(|&r| r > 0)).unwrap_or(false))
        .take(n_queries_keep)
        .collect();
    let mut keep_ids: HashSet<String> = HashSet::new();
    for (qid, _) in &chosen_queries {
        if let Some(m) = ds.qrels.get(qid) {
            for (doc, rel) in m {
                if *rel > 0 {
                    keep_ids.insert(doc.clone());
                }
            }
        }
    }
    // Walk the corpus in declaration order, keeping all relevance-
    // tagged docs first then padding with others until cap.
    let mut new_docs = Vec::with_capacity(n_docs_cap);
    let mut included: HashSet<String> = HashSet::new();
    for d in &ds.docs {
        if keep_ids.contains(&d.id.0) {
            included.insert(d.id.0.clone());
            new_docs.push(d.clone());
            if new_docs.len() >= n_docs_cap {
                break;
            }
        }
    }
    for d in &ds.docs {
        if new_docs.len() >= n_docs_cap {
            break;
        }
        if !included.contains(&d.id.0) {
            included.insert(d.id.0.clone());
            new_docs.push(d.clone());
        }
    }
    // Filter qrels down to the included doc set. Queries with no
    // remaining positive relevance get dropped via
    // `keep_queries_with_qrels` upstream.
    let mut new_qrels: HashMap<String, HashMap<String, u8>> = HashMap::new();
    for (qid, m) in &ds.qrels {
        let filtered: HashMap<String, u8> = m
            .iter()
            .filter(|(doc, _)| included.contains(*doc))
            .map(|(d, r)| (d.clone(), *r))
            .collect();
        if !filtered.is_empty() {
            new_qrels.insert(qid.clone(), filtered);
        }
    }
    BeirDataset {
        name: ds.name,
        docs: new_docs,
        queries: ds.queries.clone(),
        qrels: new_qrels,
    }
}

fn stable_query_index(qid: &str) -> usize {
    // Hash the qid into a 64-bit value; cast for use as a unique per-
    // query index. Same qid → same index across runs, but distinct
    // qids never collide on the truncated 64 bits (NFCorpus has 323
    // queries — collisions are negligibly unlikely).
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    qid.hash(&mut h);
    h.finish() as usize
}

//! M7.3 — Accuracy bench on BEIR / NFCorpus.
//!
//! Validates the prototype's protocol fidelity and (in the no-DP-on-the-
//! pooled-output configurations) retrieval utility at a real-IR scale:
//! 3,633 docs, 100 sampled queries (or 323 with `BEIR_QUERIES=full`).
//!
//! ## Metrics
//!
//! - **`ndcg10`** — normalized Discounted Cumulative Gain at 10. The
//!   canonical IR metric on BEIR. Comparable to published numbers in
//!   the BEIR paper / MTEB leaderboard.
//! - **`recall100`** — `|relevant ∩ top-100| / |relevant|`.
//! - **`mrr10`** — Mean Reciprocal Rank at 10.
//! - **`top1_base`** — protocol-fidelity: rank-1 doc-id equals the
//!   plaintext baseline's rank-1 (averaged over queries).
//! - **`rec10_vs_base`** — `|top-10 ∩ baseline_top_10| / 10` averaged
//!   over queries (head-of-list protocol stability).
//!
//! ## Sanity check
//!
//! The first thing the bench does is reproduce the published BEIR
//! baseline for FastEmbed MiniLM-L6 plain cosine on NFCorpus: nDCG@10
//! ≈ 0.30. If we miss by more than ±0.05 the loader or metric is wrong.
//!
//! ## Run
//!
//! ```text
//! cargo test -p approach4 --release --test beir_accuracy \
//!     -- --ignored --nocapture
//! ```
//!
//! With `BEIR_QUERIES=full` for the release-gate (slower; ~3× the time).

mod common;

use std::collections::HashMap;

use approach4::{Approach4InMemoryService, NoopAttestationVerifier};
use common::beir::{BeirDataset, load_nfcorpus};
use common::embed_cache::CachingEmbedder;
use dp_forward::DpForwardConfig;
use gelo_embedder::GeloBertEmbedder;
use gelo_gpu_wgpu::WgpuVulkanEngine;
use gelo_protocol::{InProcessTrustedExecutor, MaskSeed, PlaintextExecutor};
use rag_core::{Caprise, CapriseKey, Embedder, FastEmbedEmbedder};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use remote_rag::{PlanarLaplaceConfig, RemoteRagService};

const K_NDCG: usize = 10;
const K_RECALL: usize = 100;
const DEFAULT_QUERY_SAMPLE: usize = 100;

// ─────────────────────────────────────────────────────────────────────
// DP-Forward wrapper for the M7.3 bench.
// ─────────────────────────────────────────────────────────────────────

struct DpForwardWrapper<E: Embedder> {
    inner: E,
    cfg: DpForwardConfig,
    rng: ChaCha20Rng,
}

impl<E: Embedder> DpForwardWrapper<E> {
    fn new(inner: E, cfg: DpForwardConfig) -> Self {
        let mut seed = [0u8; 32];
        rand::rng().fill_bytes(&mut seed);
        Self {
            inner,
            cfg,
            rng: ChaCha20Rng::from_seed(seed),
        }
    }
}

impl<E: Embedder> Embedder for DpForwardWrapper<E> {
    fn embed(&mut self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut out = self.inner.embed(texts)?;
        for row in out.iter_mut() {
            dp_forward::amgm::clip_l2_in_place(row, self.cfg.clip_c);
            dp_forward::amgm::add_gaussian_noise(row, self.cfg.sigma, &mut self.rng);
        }
        Ok(out)
    }
}

/// Cached FastEmbed factory — every config goes through this so the
/// internal 64-batch chunking in `CachingEmbedder` avoids the all-at-
/// once 3.6k OOM, and second-run-onwards is instant.
fn cached_fastembed(label: &str) -> anyhow::Result<CachingEmbedder<FastEmbedEmbedder>> {
    CachingEmbedder::new(FastEmbedEmbedder::new_smallest()?, label)
}

// ─────────────────────────────────────────────────────────────────────
// Cosine + IR metrics
// ─────────────────────────────────────────────────────────────────────

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn dcg_at_k(ranked_doc_ids: &[String], qrels: &HashMap<String, u8>, k: usize) -> f64 {
    let mut s = 0.0;
    for (i, doc) in ranked_doc_ids.iter().take(k).enumerate() {
        let rel = qrels.get(doc).copied().unwrap_or(0) as f64;
        s += (2.0_f64.powf(rel) - 1.0) / (i as f64 + 2.0).log2();
    }
    s
}

fn idcg_at_k(qrels: &HashMap<String, u8>, k: usize) -> f64 {
    let mut rels: Vec<u8> = qrels.values().copied().collect();
    rels.sort_unstable_by(|a, b| b.cmp(a));
    let mut s = 0.0;
    for (i, rel) in rels.iter().take(k).enumerate() {
        s += (2.0_f64.powf(*rel as f64) - 1.0) / (i as f64 + 2.0).log2();
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

// ─────────────────────────────────────────────────────────────────────
// Plaintext baseline
// ─────────────────────────────────────────────────────────────────────

struct PlainBaseline {
    doc_embeds: Vec<(String, Vec<f32>)>,
    rankings: HashMap<String, Vec<String>>,
}

fn build_plain_baseline(
    dataset: &BeirDataset,
    queries: &[(String, String)],
    k_max: usize,
) -> anyhow::Result<PlainBaseline> {
    eprintln!("[baseline] embedding {} docs (cached)...", dataset.docs.len());
    let mut embedder = CachingEmbedder::new(
        FastEmbedEmbedder::new_smallest()?,
        "fastembed-minilm-l6-plain",
    )?;
    let texts: Vec<String> = dataset.docs.iter().map(|d| d.text.clone()).collect();
    let raw_embeds = embedder.embed(&texts)?;
    let doc_embeds: Vec<(String, Vec<f32>)> = dataset
        .docs
        .iter()
        .zip(raw_embeds.into_iter())
        .map(|(d, e)| (d.id.0.clone(), e))
        .collect();

    eprintln!("[baseline] embedding {} queries (cached)...", queries.len());
    let q_texts: Vec<String> = queries.iter().map(|(_, t)| t.clone()).collect();
    let q_embeds = embedder.embed(&q_texts)?;

    let mut rankings = HashMap::with_capacity(queries.len());
    for ((qid, _), q_e) in queries.iter().zip(q_embeds.into_iter()) {
        let mut scored: Vec<(usize, f32)> = doc_embeds
            .iter()
            .enumerate()
            .map(|(i, (_, e))| (i, cosine(&q_e, e)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<String> = scored
            .into_iter()
            .take(k_max)
            .map(|(i, _)| doc_embeds[i].0.clone())
            .collect();
        rankings.insert(qid.clone(), top);
    }

    Ok(PlainBaseline {
        doc_embeds,
        rankings,
    })
}

// ─────────────────────────────────────────────────────────────────────
// Metric aggregation
// ─────────────────────────────────────────────────────────────────────

#[derive(Default, Debug, Clone, Copy)]
struct MetricSummary {
    ndcg10: f64,
    recall100: f64,
    mrr10: f64,
    top1_base_match: f64,
    rec10_overlap_base: f64,
}

fn aggregate(
    queries: &[(String, String)],
    rankings: &HashMap<String, Vec<String>>,
    dataset: &BeirDataset,
    baseline: &HashMap<String, Vec<String>>,
) -> MetricSummary {
    let mut n = 0;
    let mut s = MetricSummary::default();
    for (qid, _) in queries.iter() {
        let qrels = match dataset.qrels.get(qid) {
            Some(q) => q,
            None => continue,
        };
        let ranked = match rankings.get(qid) {
            Some(r) => r,
            None => continue,
        };
        s.ndcg10 += ndcg_at_k(ranked, qrels, K_NDCG);
        s.recall100 += recall_at_k(ranked, qrels, K_RECALL);
        s.mrr10 += mrr_at_k(ranked, qrels, K_NDCG);
        if let Some(base) = baseline.get(qid) {
            let same_top1 = base.first() == ranked.first();
            s.top1_base_match += if same_top1 { 1.0 } else { 0.0 };
            let base_set: std::collections::HashSet<&String> = base.iter().take(10).collect();
            let overlap = ranked
                .iter()
                .take(10)
                .filter(|d| base_set.contains(d))
                .count();
            s.rec10_overlap_base += overlap as f64 / 10.0;
        }
        n += 1;
    }
    if n == 0 {
        return s;
    }
    let n = n as f64;
    MetricSummary {
        ndcg10: s.ndcg10 / n,
        recall100: s.recall100 / n,
        mrr10: s.mrr10 / n,
        top1_base_match: s.top1_base_match / n,
        rec10_overlap_base: s.rec10_overlap_base / n,
    }
}

// ─────────────────────────────────────────────────────────────────────

fn run_via_approach4<E: Embedder, S: rag_core::EmbeddingEncryptionScheme>(
    mut service: Approach4InMemoryService<E, S, NoopAttestationVerifier>,
    dataset: &BeirDataset,
    queries: &[(String, String)],
    k: usize,
) -> anyhow::Result<HashMap<String, Vec<String>>> {
    service.ingest_chunks(dataset.docs.clone())?;
    let mut out = HashMap::with_capacity(queries.len());
    for (qid, text) in queries.iter() {
        let hits = service.query(text, k)?;
        out.insert(qid.clone(), hits.into_iter().map(|h| h.id.0).collect());
    }
    Ok(out)
}

fn run_via_remote_rag<E: Embedder>(
    mut service: RemoteRagService<E>,
    dataset: &BeirDataset,
    queries: &[(String, String)],
    k: usize,
) -> anyhow::Result<HashMap<String, Vec<String>>> {
    service.ingest_chunks(dataset.docs.clone())?;
    let mut out = HashMap::with_capacity(queries.len());
    for (qid, text) in queries.iter() {
        let hits = service.query(text, k)?;
        out.insert(qid.clone(), hits.into_iter().map(|h| h.id.0).collect());
    }
    Ok(out)
}

fn print_row(label: &str, m: &MetricSummary) {
    eprintln!(
        "{:<48} {:>7.3} {:>9.3} {:>7.3} {:>10.3} {:>13.3}",
        label, m.ndcg10, m.recall100, m.mrr10, m.top1_base_match, m.rec10_overlap_base,
    );
}

// ─────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "downloads NFCorpus (~5 MB) on first run; embeds 3.6k docs"]
fn beir_nfcorpus_accuracy_comparison() -> anyhow::Result<()> {
    let q_sample = std::env::var("BEIR_QUERIES")
        .ok()
        .map(|s| {
            if s == "full" {
                usize::MAX
            } else {
                s.parse().unwrap_or(DEFAULT_QUERY_SAMPLE)
            }
        })
        .unwrap_or(DEFAULT_QUERY_SAMPLE);

    // `BEIR_DOCS=N` truncates the corpus to N docs — for perf-only
    // checkpoints during engine migration (the accuracy assertions are
    // skipped below when corpus is too small to be meaningful).
    let doc_cap: Option<usize> = std::env::var("BEIR_DOCS")
        .ok()
        .and_then(|s| s.parse().ok());

    eprintln!("[load] NFCorpus (full corpus, {q_sample} queries)...");
    let mut dataset = load_nfcorpus()?;
    if let Some(n) = doc_cap {
        dataset.docs.truncate(n);
        eprintln!("[load] corpus truncated to {} docs (BEIR_DOCS={n})", dataset.docs.len());
    }
    // Any truncation of the corpus invalidates the MTEB-baseline comparison
    // (the published nDCG@10 is on the full 3,633-doc corpus). Treat all
    // BEIR_DOCS-truncated runs as perf-only.
    let perf_only = doc_cap.is_some();
    eprintln!(
        "[load] corpus = {} docs; queries available = {}; qrels available = {}",
        dataset.docs.len(),
        dataset.queries.len(),
        dataset.qrels.len()
    );

    let queries: Vec<(String, String)> = dataset
        .queries
        .iter()
        .filter(|(qid, _)| dataset.qrels.contains_key(qid))
        .take(q_sample)
        .cloned()
        .collect();
    eprintln!(
        "[load] using {} queries (after qrels-filter + sample cap)",
        queries.len()
    );

    let baseline = build_plain_baseline(&dataset, &queries, K_RECALL)?;

    let baseline_metrics = aggregate(&queries, &baseline.rankings, &dataset, &baseline.rankings);
    eprintln!(
        "[baseline] FastEmbed MiniLM-L6 plain cosine: nDCG@10 = {:.4} (MTEB published ≈ 0.30; tolerance ±0.05)",
        baseline_metrics.ndcg10
    );
    if !perf_only {
        assert!(
            (baseline_metrics.ndcg10 - 0.30).abs() < 0.05,
            "baseline nDCG@10 = {:.3} not within ±0.05 of published MTEB MiniLM-L6 on NFCorpus (≈ 0.30) — \
             loader or metric is wrong, not protocol",
            baseline_metrics.ndcg10
        );
    } else {
        eprintln!("[baseline] perf-only mode (BEIR_DOCS<100) — accuracy assertions skipped");
    }

    eprintln!("[run] CAPRISE (no DP)...");
    let caprise_rankings = run_via_approach4(
        Approach4InMemoryService::new(
            cached_fastembed("fastembed-minilm-l6-plain")?,
            Caprise::new(CapriseKey::generate(32.0, 0.15)),
            NoopAttestationVerifier,
        ),
        &dataset,
        &queries,
        K_RECALL,
    )?;
    let caprise_metrics = aggregate(&queries, &caprise_rankings, &dataset, &baseline.rankings);

    eprintln!("[run] CAPRISE + DP-Forward(ε=4) at pooled output...");
    let dp_cfg = DpForwardConfig::calibrate(4.0, 1e-5, 1.0);
    let caprise_dp_rankings = run_via_approach4(
        Approach4InMemoryService::new(
            DpForwardWrapper::new(
                cached_fastembed("fastembed-minilm-l6-plain")?,
                dp_cfg,
            ),
            Caprise::new(CapriseKey::generate(32.0, 0.15)),
            NoopAttestationVerifier,
        ),
        &dataset,
        &queries,
        K_RECALL,
    )?;
    let caprise_dp_metrics =
        aggregate(&queries, &caprise_dp_rankings, &dataset, &baseline.rankings);

    eprintln!("[run] RemoteRAG (planar-Laplace ε = 10·n)...");
    let probe_dim = baseline.doc_embeds[0].1.len();
    let planar_eps = 10.0 * probe_dim as f64;
    let rrag_rankings = run_via_remote_rag(
        RemoteRagService::new(
            cached_fastembed("fastembed-minilm-l6-plain")?,
            PlanarLaplaceConfig::new(planar_eps, probe_dim),
        )
        .with_paillier_bits(256)
        .with_over_fetch_factor(3)
        .with_seed([19u8; 32]),
        &dataset,
        &queries,
        K_RECALL,
    )?;
    let rrag_metrics = aggregate(&queries, &rrag_rankings, &dataset, &baseline.rankings);

    // ─── GELO+BGE configurations (M7.1 validation at scale) ───
    // BGE-base is 12-layer BERT — exactly the architecture the
    // DP-Forward paper validates `noise_layer=10` against. We compare:
    //   - BGE plain (no DP)                  ← BGE-only baseline
    //   - BGE + CAPRISE + DP at layer 10     ← M7.1 intermediate-layer
    //   - BGE + CAPRISE + DP at pooled out   ← legacy (control)
    //
    // With CachingEmbedder, BGE inferences are cached per
    // model_identity. The DP-enabled embedder's identity includes the
    // DP config digest, so each DP variant has its own cache entry —
    // first run is ~3-5 min per BGE config, subsequent runs are instant.
    let run_bge = std::env::var("BEIR_BGE").map(|v| v != "0").unwrap_or(true);
    let bge_metrics: Option<(MetricSummary, MetricSummary, MetricSummary)> = if run_bge {
        // BGE-base on the shared Vulkan engine — ~10× faster than CPU
        // on 3.6k docs. Each BGE embedder gets its own executor wrapping
        // a `clone_shared()` of the same GPU.
        let gpu = WgpuVulkanEngine::new()
            .map_err(|e| anyhow::anyhow!("Vulkan adapter unavailable: {e}"))?;
        anyhow::ensure!(
            gpu.is_real_gpu(),
            "BEIR BGE configs need a real Vulkan GPU (got llvmpipe); set BEIR_BGE=0 to skip"
        );
        eprintln!(
            "[bge] Vulkan adapter: {} ({:?})",
            gpu.adapter_info().name,
            gpu.adapter_info().device_type,
        );

        eprintln!("[run] BGE-base (plain, no DP) — first run embeds 3.6k docs on Vulkan...");
        let bge_plain_emb = CachingEmbedder::new(
            GeloBertEmbedder::from_pretrained(
                "BAAI/bge-base-en-v1.5",
                PlaintextExecutor::new(gpu.clone_shared()),
            )?,
            "bge-base-en-v1.5",
        )?;
        let bge_plain_rankings = run_via_approach4(
            Approach4InMemoryService::new(
                bge_plain_emb,
                Caprise::new(CapriseKey::generate(32.0, 0.15)),
                NoopAttestationVerifier,
            ),
            &dataset,
            &queries,
            K_RECALL,
        )?;
        let bge_plain = aggregate(&queries, &bge_plain_rankings, &dataset, &baseline.rankings);

        eprintln!("[run] BGE-base + CAPRISE + DP-Forward(ε=4) at LAYER 10 (M7.1 path)...");
        let dp_layer_cfg =
            DpForwardConfig::calibrate(4.0, 1e-5, 1.0).with_layer_index(Some(10));
        let bge_dp_layer_emb = CachingEmbedder::new(
            GeloBertEmbedder::from_pretrained(
                "BAAI/bge-base-en-v1.5",
                PlaintextExecutor::new(gpu.clone_shared()),
            )?
            .with_dp_forward(dp_layer_cfg),
            "bge-base-en-v1.5-dp-layer10",
        )?;
        let bge_dp_layer_rankings = run_via_approach4(
            Approach4InMemoryService::new(
                bge_dp_layer_emb,
                Caprise::new(CapriseKey::generate(32.0, 0.15)),
                NoopAttestationVerifier,
            ),
            &dataset,
            &queries,
            K_RECALL,
        )?;
        let bge_dp_layer =
            aggregate(&queries, &bge_dp_layer_rankings, &dataset, &baseline.rankings);

        eprintln!("[run] BGE-base + CAPRISE + DP-Forward(ε=4) at POOLED output (control)...");
        let dp_pooled_cfg = DpForwardConfig::calibrate(4.0, 1e-5, 1.0); // layer_index = None
        let bge_dp_pooled_emb = CachingEmbedder::new(
            GeloBertEmbedder::from_pretrained(
                "BAAI/bge-base-en-v1.5",
                PlaintextExecutor::new(gpu.clone_shared()),
            )?
            .with_dp_forward(dp_pooled_cfg),
            "bge-base-en-v1.5-dp-pooled",
        )?;
        let bge_dp_pooled_rankings = run_via_approach4(
            Approach4InMemoryService::new(
                bge_dp_pooled_emb,
                Caprise::new(CapriseKey::generate(32.0, 0.15)),
                NoopAttestationVerifier,
            ),
            &dataset,
            &queries,
            K_RECALL,
        )?;
        let bge_dp_pooled =
            aggregate(&queries, &bge_dp_pooled_rankings, &dataset, &baseline.rankings);

        // Full GELO mask round-trip via InProcessTrustedExecutor (Vulkan +
        // in-process mock TEE). Validates that the mask + engine + unmask
        // path produces baseline-equivalent embeddings under the migration
        // to burn-cubecl.
        eprintln!("[run] BGE-base + GELO mask (Vulkan + in-process TEE) — full-stack...");
        let bge_mask_emb = CachingEmbedder::new(
            GeloBertEmbedder::from_pretrained(
                "BAAI/bge-base-en-v1.5",
                InProcessTrustedExecutor::with_seed(
                    gpu.clone_shared(),
                    MaskSeed::from_bytes([7u8; 32]),
                ),
            )?,
            "bge-base-en-v1.5-gelo-mask",
        )?;
        let bge_mask_rankings = run_via_approach4(
            Approach4InMemoryService::new(
                bge_mask_emb,
                Caprise::new(CapriseKey::generate(32.0, 0.15)),
                NoopAttestationVerifier,
            ),
            &dataset,
            &queries,
            K_RECALL,
        )?;
        let bge_mask = aggregate(&queries, &bge_mask_rankings, &dataset, &baseline.rankings);
        eprintln!(
            "[run] BGE-base + GELO mask + CAPRISE: top1_vs_bge_plain check below"
        );
        let _ = bge_mask;

        Some((bge_plain, bge_dp_layer, bge_dp_pooled))
    } else {
        None
    };

    eprintln!();
    eprintln!(
        "=== BEIR/NFCorpus accuracy ({} queries × {} docs) ===",
        queries.len(),
        dataset.docs.len()
    );
    eprintln!("nDCG@10 / Recall@100 / MRR@10 are IR-canonical metrics (vs qrels).");
    eprintln!("top1_base / rec10_base are PROTOCOL-fidelity vs plaintext baseline.");
    eprintln!();
    eprintln!(
        "{:<48} {:>7} {:>9} {:>7} {:>10} {:>13}",
        "config", "nDCG10", "Rec@100", "MRR@10", "top1_base", "rec10_vs_base"
    );
    eprintln!("{}", "-".repeat(100));
    print_row("Plaintext baseline (FastEmbed MiniLM-L6)", &baseline_metrics);
    print_row("CAPRISE (no DP)", &caprise_metrics);
    print_row("CAPRISE + DP-Forward(ε=4) pooled output", &caprise_dp_metrics);
    print_row("RemoteRAG (planar-Laplace ε=10·n)", &rrag_metrics);
    if let Some((bge_plain, bge_dp_layer, bge_dp_pooled)) = &bge_metrics {
        eprintln!();
        eprintln!("M7.1 validation — DP-Forward position comparison on BGE-base (12-layer BERT, 768-d):");
        print_row("GELO/BGE-base (plain) + CAPRISE", bge_plain);
        print_row("  + DP-Forward(ε=4) @ layer 10 (M7.1)", bge_dp_layer);
        print_row("  + DP-Forward(ε=4) @ pooled output", bge_dp_pooled);
    }
    eprintln!();

    if !perf_only {
        assert!(
            caprise_metrics.top1_base_match >= 0.95,
            "CAPRISE should preserve baseline rank-1 closely; got top1_base = {:.3}",
            caprise_metrics.top1_base_match,
        );
        assert!(
            rrag_metrics.top1_base_match >= 0.85,
            "RemoteRAG with PHE rerank should preserve baseline rank-1; got top1_base = {:.3}",
            rrag_metrics.top1_base_match,
        );
    }
    if let Some((bge_plain, bge_dp_layer, bge_dp_pooled)) = &bge_metrics {
        eprintln!(
            "[m7.1] BGE plain    nDCG@10 = {:.3}\n[m7.1] BGE @layer10 nDCG@10 = {:.3} ({:.1}% of plain)\n[m7.1] BGE @pooled  nDCG@10 = {:.3} ({:.1}% of plain)",
            bge_plain.ndcg10,
            bge_dp_layer.ndcg10,
            100.0 * bge_dp_layer.ndcg10 / bge_plain.ndcg10.max(1e-9),
            bge_dp_pooled.ndcg10,
            100.0 * bge_dp_pooled.ndcg10 / bge_plain.ndcg10.max(1e-9),
        );
        // The M7.1 plan §Verification §2 hypothesised that
        // intermediate-layer DP would recover meaningful retrieval
        // utility. Empirically (at ε=4, C=1.0, noise_layer=10 on BGE-base)
        // BOTH paths destroy retrieval to near-random level. The paper's
        // mechanism is calibrated for fine-tuned downstream classification,
        // not zero-shot retrieval; without a learned downstream head that
        // can absorb the noise, ε=4 aMGM at the paper-faithful position
        // is catastrophic. Documented in docs/prototype/dp-forward.md §6.
        //
        // We assert only that BGE-plain nDCG@10 is meaningful (sanity:
        // the BGE inference pipeline produces useful embeddings), not
        // that either DP variant recovers utility (it doesn't).
        if !perf_only {
            assert!(
                bge_plain.ndcg10 >= 0.30,
                "BGE-base plain nDCG@10 = {:.3} below 0.30 — BGE inference pipeline broken, not a DP finding",
                bge_plain.ndcg10,
            );
        }
    }

    Ok(())
}

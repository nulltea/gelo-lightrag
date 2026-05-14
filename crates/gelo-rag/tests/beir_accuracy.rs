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
//! cargo test -p gelo-rag --release --test beir_accuracy \
//!     -- --ignored --nocapture
//! ```
//!
//! With `BEIR_QUERIES=full` for the release-gate (slower; ~3× the time).

mod common;

use std::collections::HashMap;

use gelo_rag::{GeloRagInMemoryService, NoopAttestationVerifier};
use common::beir::{BeirDataset, load_nfcorpus};
use common::embed_cache::CachingEmbedder;
use dp_forward::DpForwardConfig;
use gelo_embedder::{GeloBertEmbedder, GeloQwenEmbedder};
use gelo_gpu_wgpu::WgpuVulkanEngine;
use gelo_protocol::{InProcessTrustedExecutor, MaskSeed, PlaintextExecutor, ShieldConfig};
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

/// `BEIR_EMBED_CACHE=1` opt-in: wrap every embedder in CachingEmbedder
/// so re-runs of the bench replay previously-computed embeddings from
/// `target/embed-cache/`. The cache is keyed by `(model_label,
/// model_identity, text_hash)`.
///
/// Default OFF because the cache silently turns wall-clock measurement
/// into "what's on disk". It's the right choice for ranking-only
/// validation where determinism-given-same-embeddings is the goal
/// (M7.1/M7.3 protocol-fidelity work), the wrong choice for perf A/B.
fn embed_cache_enabled() -> bool {
    std::env::var("BEIR_EMBED_CACHE")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Wrap `inner` in CachingEmbedder if `BEIR_EMBED_CACHE=1`, else box
/// `inner` directly. Returns a `Box<dyn Embedder>` so the caller's
/// generic site doesn't need to switch between two concrete types.
fn maybe_cache<E: Embedder + 'static>(
    inner: E,
    label: &str,
) -> anyhow::Result<Box<dyn Embedder>> {
    if embed_cache_enabled() {
        Ok(Box::new(CachingEmbedder::new(inner, label)?))
    } else {
        Ok(Box::new(inner))
    }
}

/// Backwards-compat helper for the FastEmbed configs — same env-gated
/// behaviour as `maybe_cache` but specialized to FastEmbed construction.
fn cached_fastembed(label: &str) -> anyhow::Result<Box<dyn Embedder>> {
    let inner = FastEmbedEmbedder::new_smallest()?;
    maybe_cache(inner, label)
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
    eprintln!("[baseline] embedding {} docs...", dataset.docs.len());
    let mut embedder = maybe_cache(
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

    eprintln!("[baseline] embedding {} queries...", queries.len());
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

fn run_via_gelo_rag<E: Embedder, S: rag_core::EmbeddingEncryptionScheme>(
    mut service: GeloRagInMemoryService<E, S, NoopAttestationVerifier>,
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
    let caprise_rankings = run_via_gelo_rag(
        GeloRagInMemoryService::new(
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
    let caprise_dp_rankings = run_via_gelo_rag(
        GeloRagInMemoryService::new(
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
    // BGE embeddings are NOT cached on disk by this bench — earlier
    // runs used a CachingEmbedder wrapper but it silently turned every
    // post-first run into cache hits, contaminating wall-clock numbers.
    // Re-embed every run; if you need ranking-only validation use the
    // common::embed_cache helper directly.
    let run_bge = std::env::var("BEIR_BGE").map(|v| v != "0").unwrap_or(true);
    // BGE sub-config gates. Default is now: only GELO+mask runs.
    //   - BGE plain (no mask) and the two BGE+DP variants are skipped
    //     unless explicitly re-enabled; they already validated their
    //     hypotheses in M7.1, and their embeddings are cached on disk
    //     under `target/embed-cache/` — re-running them adds wall-clock
    //     without changing results during Tier 2 perf work.
    //   - GELO+mask is the only path that exercises the full mask
    //     round-trip + engine matmul stack; that's what Tier 2 is
    //     optimising.
    let run_bge_plain = std::env::var("BEIR_BGE_PLAIN").map(|v| v == "1").unwrap_or(false);
    let run_bge_dp = std::env::var("BEIR_BGE_DP").map(|v| v == "1").unwrap_or(false);
    let run_bge_mask = std::env::var("BEIR_BGE_MASK").map(|v| v != "0").unwrap_or(true);

    struct BgeRunMetrics {
        plain: Option<MetricSummary>,
        dp_layer: Option<MetricSummary>,
        dp_pooled: Option<MetricSummary>,
        mask: Option<MetricSummary>,
    }

    let bge_metrics: Option<BgeRunMetrics> = if run_bge {
        // BGE-base on the shared Vulkan engine — ~10× faster than CPU
        // on 3.6k docs. Each BGE embedder gets its own executor wrapping
        // a `clone_shared()` of the same GPU.
        // BEIR_BGE_FP16=1 → engine runs GEMMs in f16. Trade ~0.1% L2 rel
        // error per matmul for ~1.3-3× wall-clock per call (heavily
        // shape-dependent on Vulkan iGPU). U-Verify must stay off under
        // fp16 (CachingEmbedder caches separately by model_label, so
        // re-runs don't compare across precisions).
        let gpu = if std::env::var("BEIR_BGE_FP16").map(|v| v == "1").unwrap_or(false) {
            eprintln!("[bge] engine precision: f16 (BEIR_BGE_FP16=1)");
            WgpuVulkanEngine::new_fp16()
        } else {
            WgpuVulkanEngine::new()
        }
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

        let bge_plain = if run_bge_plain {
            eprintln!("[run] BGE-base (plain, no DP) — first run embeds 3.6k docs on Vulkan...");
            let bge_plain_emb = maybe_cache(
                GeloBertEmbedder::from_pretrained(
                    "BAAI/bge-base-en-v1.5",
                    PlaintextExecutor::new(gpu.clone_shared()),
                )?,
                "bge-base-en-v1.5",
            )?;
            let bge_plain_rankings = run_via_gelo_rag(
                GeloRagInMemoryService::new(
                    bge_plain_emb,
                    Caprise::new(CapriseKey::generate(32.0, 0.15)),
                    NoopAttestationVerifier,
                ),
                &dataset,
                &queries,
                K_RECALL,
            )?;
            Some(aggregate(
                &queries,
                &bge_plain_rankings,
                &dataset,
                &baseline.rankings,
            ))
        } else {
            eprintln!("[run] BGE-base plain SKIPPED (set BEIR_BGE_PLAIN=1 to enable)");
            None
        };

        let (bge_dp_layer, bge_dp_pooled) = if run_bge_dp {
            eprintln!("[run] BGE-base + CAPRISE + DP-Forward(ε=4) at LAYER 10 (M7.1 path)...");
            let dp_layer_cfg =
                DpForwardConfig::calibrate(4.0, 1e-5, 1.0).with_layer_index(Some(10));
            let bge_dp_layer_emb = maybe_cache(
                GeloBertEmbedder::from_pretrained(
                    "BAAI/bge-base-en-v1.5",
                    PlaintextExecutor::new(gpu.clone_shared()),
                )?
                .with_dp_forward(dp_layer_cfg),
                "bge-base-en-v1.5-dp-layer10",
            )?;
            let bge_dp_layer_rankings = run_via_gelo_rag(
                GeloRagInMemoryService::new(
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
            let bge_dp_pooled_emb = maybe_cache(
                GeloBertEmbedder::from_pretrained(
                    "BAAI/bge-base-en-v1.5",
                    PlaintextExecutor::new(gpu.clone_shared()),
                )?
                .with_dp_forward(dp_pooled_cfg),
                "bge-base-en-v1.5-dp-pooled",
            )?;
            let bge_dp_pooled_rankings = run_via_gelo_rag(
                GeloRagInMemoryService::new(
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
            (Some(bge_dp_layer), Some(bge_dp_pooled))
        } else {
            eprintln!("[run] BGE-base + DP-Forward configs SKIPPED (set BEIR_BGE_DP=1 to enable)");
            (None, None)
        };

        // Full GELO mask round-trip via InProcessTrustedExecutor (Vulkan +
        // in-process mock TEE). The headline Tier 2 target — this is the
        // path Tier 2 optimisations move the needle on.
        let bge_mask = if run_bge_mask {
            eprintln!("[run] BGE-base + GELO mask (Vulkan + in-process TEE) — full-stack...");
            // `BEIR_PAPER_PARITY=1` → match the GELO paper §3.2: one
            // Haar-uniform A sampled per forward pass (reused across all
            // offloads in that pass), paired with shield vectors (§4.2,
            // k=8 high-energy rows at 4× mean-row-norm energy) to defeat
            // the cross-offload ICA attack that mask reuse otherwise
            // exposes. Per-offload (default) samples a fresh A per
            // offloaded GEMM — strictly safer (no reuse) but ~48× more
            // QR work per BGE-base text (~140× for Qwen3).
            let paper_parity = std::env::var("BEIR_PAPER_PARITY")
                .map(|v| v == "1")
                .unwrap_or(false);
            let executor = if paper_parity {
                eprintln!("[bge] paper-parity mode: one A per forward + shield(k=8, e=4)");
                InProcessTrustedExecutor::with_seed(
                    gpu.clone_shared(),
                    MaskSeed::from_bytes([7u8; 32]),
                )
                .with_per_forward_mask(ShieldConfig::new(8, 4.0))
            } else {
                InProcessTrustedExecutor::with_seed(
                    gpu.clone_shared(),
                    MaskSeed::from_bytes([7u8; 32]),
                )
            };
            // Engine precision + protocol mode go into the cache key so
            // fp32/fp16 and per-offload/paper-parity runs don't share
            // embeddings (their outputs differ enough to perturb rankings).
            let cache_label = match (gpu.is_fp16(), paper_parity) {
                (false, false) => "bge-base-en-v1.5-gelo-mask",
                (true, false) => "bge-base-en-v1.5-gelo-mask-fp16",
                (false, true) => "bge-base-en-v1.5-gelo-mask-paper",
                (true, true) => "bge-base-en-v1.5-gelo-mask-fp16-paper",
            };
            let bge_mask_emb = maybe_cache(
                GeloBertEmbedder::from_pretrained(
                    "BAAI/bge-base-en-v1.5",
                    executor,
                )?,
                cache_label,
            )?;
            let bge_mask_rankings = run_via_gelo_rag(
                GeloRagInMemoryService::new(
                    bge_mask_emb,
                    Caprise::new(CapriseKey::generate(32.0, 0.15)),
                    NoopAttestationVerifier,
                ),
                &dataset,
                &queries,
                K_RECALL,
            )?;
            Some(aggregate(
                &queries,
                &bge_mask_rankings,
                &dataset,
                &baseline.rankings,
            ))
        } else {
            eprintln!("[run] BGE-base + GELO mask SKIPPED (set BEIR_BGE_MASK=1 to enable)");
            None
        };

        Some(BgeRunMetrics {
            plain: bge_plain,
            dp_layer: bge_dp_layer,
            dp_pooled: bge_dp_pooled,
            mask: bge_mask,
        })
    } else {
        None
    };

    // ─── Qwen3-Embedding-0.6B configuration ───
    // Same Vulkan engine as BGE (re-use via GpuOffloadEngine trait), but
    // a much bigger decoder-LLM-as-embedder. Gated behind BEIR_QWEN3=1
    // because it (a) downloads ~1.2 GB on first run, (b) at ~85 ms/text
    // is ~3× slower than BGE per text, and (c) the protocol fidelity
    // story is already proven on BGE — this is here to validate corpus-
    // scale retrieval correctness of the decoder path, not to gate
    // anything we're shipping.
    let run_qwen3 = std::env::var("BEIR_QWEN3").map(|v| v == "1").unwrap_or(false);
    // BEIR_QWEN3_PLAIN=1 enables a Qwen3 + PlaintextExecutor row alongside
    // the masked one. Used to isolate "Qwen3 vs MiniLM model disagreement"
    // from "GELO mask drift": top1_base of the plain row tells you the
    // model-vs-model floor, the masked row's gap below that is the mask's
    // contribution. Default on when BEIR_QWEN3=1.
    let run_qwen3_plain = std::env::var("BEIR_QWEN3_PLAIN")
        .map(|v| v != "0")
        .unwrap_or(true);
    let (qwen3_plain_metrics, qwen3_mask_metrics): (Option<MetricSummary>, Option<MetricSummary>) =
    if run_qwen3 {
        let gpu = WgpuVulkanEngine::new()
            .map_err(|e| anyhow::anyhow!("Vulkan adapter unavailable: {e}"))?;
        anyhow::ensure!(
            gpu.is_real_gpu(),
            "BEIR Qwen3 needs a real Vulkan GPU (got llvmpipe); set BEIR_QWEN3=0 to skip"
        );
        eprintln!(
            "[qwen3] Vulkan adapter: {} ({:?})",
            gpu.adapter_info().name,
            gpu.adapter_info().device_type,
        );
        let paper_parity = std::env::var("BEIR_PAPER_PARITY")
            .map(|v| v == "1")
            .unwrap_or(false);

        let qwen3_plain = if run_qwen3_plain {
            eprintln!("[run] Qwen3-Embedding-0.6B (plain, no mask) — model-vs-MiniLM floor for top1_base");
            let plain_emb = maybe_cache(
                GeloQwenEmbedder::from_pretrained(
                    "Qwen/Qwen3-Embedding-0.6B",
                    PlaintextExecutor::new(gpu.clone_shared()),
                )?,
                "qwen3-embedding-0.6b-plain",
            )?;
            let t0 = std::time::Instant::now();
            let plain_rankings = run_via_gelo_rag(
                GeloRagInMemoryService::new(
                    plain_emb,
                    Caprise::new(CapriseKey::generate(32.0, 0.15)),
                    NoopAttestationVerifier,
                ),
                &dataset,
                &queries,
                K_RECALL,
            )?;
            let elapsed = t0.elapsed();
            let n_texts = dataset.docs.len() + queries.len();
            eprintln!(
                "[qwen3-plain] {} texts in {:.1}s ⇒ {:.0} ms/text",
                n_texts,
                elapsed.as_secs_f64(),
                1000.0 * elapsed.as_secs_f64() / n_texts as f64,
            );
            Some(aggregate(
                &queries,
                &plain_rankings,
                &dataset,
                &baseline.rankings,
            ))
        } else {
            None
        };

        eprintln!("[run] Qwen3-Embedding-0.6B + GELO mask (Vulkan + in-process TEE) — first run downloads ~1.2 GB...");
        let executor = if paper_parity {
            eprintln!("[qwen3] paper-parity mode: one A per forward + shield(k=8, e=4)");
            InProcessTrustedExecutor::with_seed(
                gpu.clone_shared(),
                MaskSeed::from_bytes([11u8; 32]),
            )
            .with_per_forward_mask(ShieldConfig::new(8, 4.0))
        } else {
            InProcessTrustedExecutor::with_seed(
                gpu.clone_shared(),
                MaskSeed::from_bytes([11u8; 32]),
            )
        };
        let cache_label = if paper_parity {
            "qwen3-embedding-0.6b-gelo-mask-paper"
        } else {
            "qwen3-embedding-0.6b-gelo-mask"
        };
        let use_perm_attn = std::env::var("BEIR_PERM_ATTN")
            .map(|v| v == "1")
            .unwrap_or(false);
        if use_perm_attn {
            eprintln!("[qwen3] permutation-shielded attention enabled (BEIR_PERM_ATTN=1)");
        }
        let mut qwen3_inner = GeloQwenEmbedder::from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B",
            executor,
        )?;
        if use_perm_attn {
            qwen3_inner = qwen3_inner
                .with_perm_attention(true)
                .with_perm_attention_min_seq_len(Some(64));
        }
        let qwen3_emb = maybe_cache(qwen3_inner, cache_label)?;
        gelo_protocol::profile::reset();
        let t0 = std::time::Instant::now();
        let qwen3_rankings = run_via_gelo_rag(
            GeloRagInMemoryService::new(
                qwen3_emb,
                Caprise::new(CapriseKey::generate(32.0, 0.15)),
                NoopAttestationVerifier,
            ),
            &dataset,
            &queries,
            K_RECALL,
        )?;
        let elapsed = t0.elapsed();
        let n_texts = dataset.docs.len() + queries.len();
        eprintln!(
            "[qwen3-mask] {} texts in {:.1}s ⇒ {:.0} ms/text",
            n_texts,
            elapsed.as_secs_f64(),
            1000.0 * elapsed.as_secs_f64() / n_texts as f64,
        );
        gelo_protocol::profile::snapshot()
            .dump("qwen3-mask per-bucket cumulative (across all texts)");
        let qwen3_mask = Some(aggregate(
            &queries,
            &qwen3_rankings,
            &dataset,
            &baseline.rankings,
        ));
        (qwen3_plain, qwen3_mask)
    } else {
        (None, None)
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
    if let Some(bge) = &bge_metrics {
        eprintln!();
        if let Some(p) = &bge.plain {
            print_row("GELO/BGE-base (plain) + CAPRISE", p);
        }
        if let Some(m) = &bge.mask {
            print_row("GELO/BGE-base + GELO mask + CAPRISE", m);
        }
        if let (Some(dp_layer), Some(dp_pooled)) = (&bge.dp_layer, &bge.dp_pooled) {
            eprintln!(
                "M7.1 validation — DP-Forward position comparison on BGE-base (12-layer BERT, 768-d):"
            );
            print_row("  + DP-Forward(ε=4) @ layer 10 (M7.1)", dp_layer);
            print_row("  + DP-Forward(ε=4) @ pooled output", dp_pooled);
        }
    }
    if qwen3_plain_metrics.is_some() || qwen3_mask_metrics.is_some() {
        eprintln!();
    }
    if let Some(p) = &qwen3_plain_metrics {
        print_row("GELO/Qwen3-Embedding-0.6B (plain) + CAPRISE", p);
    }
    if let Some(qwen3) = &qwen3_mask_metrics {
        print_row("GELO/Qwen3-Embedding-0.6B + GELO mask + CAPRISE", qwen3);
    }
    if let (Some(plain), Some(mask)) = (&qwen3_plain_metrics, &qwen3_mask_metrics) {
        eprintln!(
            "[qwen3] plain   top1_base = {:.3} (model-vs-MiniLM floor)\n[qwen3] +mask  top1_base = {:.3} (gap vs floor = {:+.3})",
            plain.top1_base_match,
            mask.top1_base_match,
            mask.top1_base_match - plain.top1_base_match,
        );
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
    if let Some(bge) = &bge_metrics {
        // M7.1 detail print only when both DP variants ran alongside the
        // plain baseline.
        if let (Some(plain), Some(dp_layer), Some(dp_pooled)) =
            (&bge.plain, &bge.dp_layer, &bge.dp_pooled)
        {
            eprintln!(
                "[m7.1] BGE plain    nDCG@10 = {:.3}\n[m7.1] BGE @layer10 nDCG@10 = {:.3} ({:.1}% of plain)\n[m7.1] BGE @pooled  nDCG@10 = {:.3} ({:.1}% of plain)",
                plain.ndcg10,
                dp_layer.ndcg10,
                100.0 * dp_layer.ndcg10 / plain.ndcg10.max(1e-9),
                dp_pooled.ndcg10,
                100.0 * dp_pooled.ndcg10 / plain.ndcg10.max(1e-9),
            );
            // M7.1 plan §Verification §2 hypothesised that intermediate-
            // layer DP would recover meaningful retrieval utility.
            // Empirically (ε=4, C=1.0, noise_layer=10 on BGE-base) BOTH
            // paths destroy retrieval to near-random level. Documented in
            // docs/prototype/dp-forward.md §6. Assert only that BGE-plain
            // nDCG@10 is meaningful (sanity: BGE inference pipeline
            // produces useful embeddings).
            if !perf_only {
                assert!(
                    plain.ndcg10 >= 0.30,
                    "BGE-base plain nDCG@10 = {:.3} below 0.30 — BGE inference pipeline broken, not a DP finding",
                    plain.ndcg10,
                );
            }
        }
    }

    Ok(())
}

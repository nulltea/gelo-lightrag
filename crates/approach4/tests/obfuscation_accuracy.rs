//! Retrieval-accuracy benchmark across the privacy configurations.
//!
//! Measures **how much each privacy-preserving config distorts the
//! ranking** produced by a plaintext baseline, plus a coarse semantic
//! check (rank-1 doc comes from the expected topical group). Unlike
//! `obfuscation_bench.rs` which times wall-clock, this asks the
//! orthogonal question: *do the configs return the right answers?*
//!
//! ## Embedder choice — FastEmbed (MiniLM-L6-v2), not GELO+Qwen3
//!
//! Accuracy is the protocol's question, not the embedder's. We deliberately
//! use `FastEmbedEmbedder` (small, well-tuned bi-encoder, 384-d) rather
//! than `GeloQwenEmbedder` here, for three reasons:
//!
//! - FastEmbed is faster (~5 ms/embed vs ~150 ms for Qwen3 on Vulkan),
//!   keeping the bench under a minute end-to-end across all configs and
//!   trials.
//! - FastEmbed is the canonical Rust retrieval embedder used in
//!   approach-4's existing smoke tests; it produces clean semantic
//!   baselines on small corpora without instruction-prefix gymnastics.
//! - GELO's ~1e-2 per-dim masking drift (well below the parity threshold
//!   but enough to flip cosine ranks for close documents) conflates
//!   inference-side noise with protocol-side accuracy.
//!
//! GELO timing + accuracy composition is covered by `obfuscation_bench.rs`
//! and by the parity tests in `crates/gelo-embedder/tests/`.
//!
//! ## Corpus
//!
//! Twelve documents across four topical groups (3 docs each):
//! `rust`, `python`, `dist`, `crypto`. Four queries — one per group —
//! near-paraphrasing one document per group to ensure unambiguous matches
//! on a competent bi-encoder.
//!
//! ## Configurations measured
//!
//! 1. Plaintext baseline (no privacy) — used as the reference ranking.
//! 2. CAPRISE (no DP).
//! 3. CAPRISE + DP-Forward, sweep `ε ∈ {1, 4, 16}`.
//! 4. RemoteRAG (no doc-side DP), sweep planar-Laplace
//!    `ε ∈ {n, 10·n, 50·n}` per the paper's recommended range.
//! 5. RemoteRAG + doc-side DP-Forward at `ε_doc = 4`,
//!    planar-Laplace at `ε_q = 10·n`.
//!
//! ## Metrics
//!
//! For each query, compute against the plaintext baseline's top-K ranking:
//!
//! - `top1_grp`: rank-1 doc's topical group equals the query's expected
//!   group. (Semantic accuracy — embedder-quality check.)
//! - `top1_base`: rank-1 doc ID equals the plaintext baseline's rank-1.
//!   (**Protocol fidelity** — does this scheme preserve the ranking the
//!   underlying embedder produces?)
//! - `rec@3`: `|top-3 ∩ baseline_top_3| / 3`. (Head-of-list stability.)
//!
//! Multi-trial configs (those with DP noise) report **mean ± std** across
//! `N_TRIALS = 3` trials; non-stochastic configs report a single
//! deterministic measurement.
//!
//! Run:
//!
//! ```text
//! cargo test -p approach4 --release \
//!     --test obfuscation_accuracy -- --ignored --nocapture
//! ```

use approach4::{Approach4InMemoryService, NoopAttestationVerifier};
use dp_forward::DpForwardConfig;
use rag_core::{Caprise, CapriseKey, ChunkId, DocumentChunk, Embedder, FastEmbedEmbedder};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use remote_rag::{PlanarLaplaceConfig, RemoteRagService};

const TOP_K: usize = 3;
const N_TRIALS: usize = 3;

/// `(id, group_label, text)` rows. Groups are deliberately distinct so
/// even a noisy embedder retrieves the right one for an unambiguous query.
fn corpus_rows() -> &'static [(&'static str, &'static str, &'static str)] {
    &[
        ("rust-memory-safety", "rust", "Rust enforces memory safety through ownership and the borrow checker."),
        ("rust-cargo", "rust", "Cargo is the package manager and build system for Rust projects."),
        ("rust-tokio", "rust", "Tokio provides an async runtime for Rust with multi-threaded executors."),

        ("python-indentation", "python", "Python uses indentation to define code blocks instead of braces."),
        ("python-asyncio", "python", "Python asyncio schedules coroutines on a single-threaded event loop."),
        ("python-gil", "python", "The CPython global interpreter lock prevents two threads from executing bytecode simultaneously."),

        ("kube-pods", "dist", "Kubernetes pods are the smallest deployable unit and group containers together."),
        ("kafka-partitions", "dist", "Kafka topics are partitioned across brokers for parallel consumption."),
        ("raft-consensus", "dist", "The Raft consensus algorithm elects a single leader to coordinate log replication."),

        ("tls-attestation", "crypto", "Remote attestation can bind a TEE measurement into a TLS session."),
        ("aes-gcm", "crypto", "AES-GCM provides authenticated encryption with associated data using counter mode."),
        ("paillier-he", "crypto", "Paillier homomorphic encryption supports addition of ciphertexts and scalar multiplication."),
    ]
}

fn docs() -> Vec<DocumentChunk> {
    corpus_rows()
        .iter()
        .map(|(id, _, text)| DocumentChunk {
            id: ChunkId((*id).into()),
            text: (*text).into(),
        })
        .collect()
}

fn group_of(id: &str) -> &'static str {
    corpus_rows()
        .iter()
        .find(|(i, _, _)| *i == id)
        .map(|(_, g, _)| *g)
        .unwrap_or("?")
}

fn queries() -> Vec<(&'static str, &'static str)> {
    // Near-paraphrases of the target doc text so the semantic-accuracy
    // column reflects protocol fidelity rather than embedder quality on
    // ambiguous queries. Each query is constructed to share content words
    // with exactly one document in the corpus.
    vec![
        ("Rust ownership and borrow checker for memory safety", "rust"),
        ("Python asyncio coroutines on an event loop", "python"),
        ("Kubernetes pods as the smallest deployable unit", "dist"),
        ("Paillier homomorphic encryption adding ciphertexts", "crypto"),
    ]
}

// ─────────────────────────────────────────────────────────────────────
// DP-Forward applied to a FastEmbed embedder via a small wrapper. The
// `dp-forward` Cargo feature on `gelo-embedder` bakes DP into the
// attested GELO embedders directly; for accuracy testing with a non-GELO
// embedder we apply the same aMGM primitives externally. Same math,
// different harness.
// ─────────────────────────────────────────────────────────────────────

struct DpForwardFastEmbed {
    inner: FastEmbedEmbedder,
    cfg: DpForwardConfig,
    rng: ChaCha20Rng,
}

impl DpForwardFastEmbed {
    fn new(inner: FastEmbedEmbedder, cfg: DpForwardConfig) -> Self {
        // Per `dp-forward.md` §4.3: DP noise must not be deterministic.
        // Seed from OsRng each construction.
        let mut seed = [0u8; 32];
        rand::rng().fill_bytes(&mut seed);
        Self {
            inner,
            cfg,
            rng: ChaCha20Rng::from_seed(seed),
        }
    }
}

impl Embedder for DpForwardFastEmbed {
    fn embed(&mut self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut out = self.inner.embed(texts)?;
        for row in out.iter_mut() {
            dp_forward::amgm::clip_l2_in_place(row, self.cfg.clip_c);
            dp_forward::amgm::add_gaussian_noise(row, self.cfg.sigma, &mut self.rng);
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Plaintext baseline: cosine over un-encrypted, un-noised embeddings.
// This is the ranking every config is measured against.
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

fn plaintext_baseline_ranking(
    embedder: &mut impl Embedder,
    corpus: &[DocumentChunk],
    queries: &[(&'static str, &'static str)],
) -> Vec<Vec<String>> {
    let doc_texts: Vec<String> = corpus.iter().map(|d| d.text.clone()).collect();
    let doc_embeds = embedder.embed(&doc_texts).expect("baseline doc embed");
    let mut out = Vec::with_capacity(queries.len());
    for (q_text, _) in queries {
        let q_embed = embedder
            .embed(&[(*q_text).to_string()])
            .expect("baseline query embed")
            .remove(0);
        let mut scored: Vec<(usize, f32)> = doc_embeds
            .iter()
            .enumerate()
            .map(|(i, e)| (i, cosine(&q_embed, e)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<String> = scored
            .into_iter()
            .take(TOP_K)
            .map(|(i, _)| corpus[i].id.0.clone())
            .collect();
        out.push(top);
    }
    out
}

// ─────────────────────────────────────────────────────────────────────
// Metrics
// ─────────────────────────────────────────────────────────────────────

#[derive(Default, Clone, Copy, Debug)]
struct TrialResult {
    /// Fraction of queries where rank-1's group equals the expected group.
    top1_group_match: f64,
    /// Fraction of queries where rank-1 doc-id equals the plaintext baseline's rank-1.
    top1_baseline_match: f64,
    /// `|top-K ∩ baseline_top_K| / K`, averaged over queries.
    recall_overlap: f64,
}

fn evaluate(
    hits_per_query: &[Vec<String>],
    baseline: &[Vec<String>],
    qs: &[(&'static str, &'static str)],
) -> TrialResult {
    let n = qs.len() as f64;
    let mut g = 0.0;
    let mut b = 0.0;
    let mut r = 0.0;
    for ((hits, (_q, expected)), base) in hits_per_query.iter().zip(qs.iter()).zip(baseline.iter()) {
        if let Some(top) = hits.first() {
            if group_of(top) == *expected {
                g += 1.0;
            }
            if base.first() == Some(top) {
                b += 1.0;
            }
        }
        let base_set: std::collections::HashSet<&String> = base.iter().take(TOP_K).collect();
        let overlap = hits.iter().take(TOP_K).filter(|h| base_set.contains(h)).count();
        r += overlap as f64 / TOP_K as f64;
    }
    TrialResult {
        top1_group_match: g / n,
        top1_baseline_match: b / n,
        recall_overlap: r / n,
    }
}

fn mean_std(results: &[TrialResult]) -> (TrialResult, TrialResult) {
    let n = results.len() as f64;
    let pick = |f: fn(&TrialResult) -> f64| {
        let m = results.iter().map(f).sum::<f64>() / n;
        let v = results.iter().map(|r| (f(r) - m).powi(2)).sum::<f64>() / n;
        (m, v.sqrt())
    };
    let (g_m, g_s) = pick(|r| r.top1_group_match);
    let (b_m, b_s) = pick(|r| r.top1_baseline_match);
    let (r_m, r_s) = pick(|r| r.recall_overlap);
    (
        TrialResult {
            top1_group_match: g_m,
            top1_baseline_match: b_m,
            recall_overlap: r_m,
        },
        TrialResult {
            top1_group_match: g_s,
            top1_baseline_match: b_s,
            recall_overlap: r_s,
        },
    )
}

// ─────────────────────────────────────────────────────────────────────
// Per-config runners.
// ─────────────────────────────────────────────────────────────────────

fn run_caprise_trial<E: Embedder>(
    embedder: E,
    corpus: &[DocumentChunk],
    qs: &[(&'static str, &'static str)],
    baseline: &[Vec<String>],
) -> TrialResult {
    let scheme = Caprise::new(CapriseKey::generate(32.0, 0.15));
    let mut svc = Approach4InMemoryService::new(embedder, scheme, NoopAttestationVerifier);
    svc.ingest_chunks(corpus.to_vec()).expect("ingest");
    let mut hits_per_q = Vec::with_capacity(qs.len());
    for (q_text, _) in qs {
        let hits = svc.query(q_text, TOP_K).expect("query");
        hits_per_q.push(hits.into_iter().map(|h| h.id.0).collect());
    }
    evaluate(&hits_per_q, baseline, qs)
}

fn run_remote_rag_trial<E: Embedder>(
    embedder: E,
    corpus: &[DocumentChunk],
    qs: &[(&'static str, &'static str)],
    baseline: &[Vec<String>],
    planar_eps: f64,
    n: usize,
) -> TrialResult {
    let dp_cfg = PlanarLaplaceConfig::new(planar_eps, n);
    let mut svc = RemoteRagService::new(embedder, dp_cfg)
        .with_paillier_bits(1024)
        .with_over_fetch_factor(3);
    svc.ingest_chunks(corpus.to_vec()).expect("ingest");
    let mut hits_per_q = Vec::with_capacity(qs.len());
    for (q_text, _) in qs {
        let hits = svc.query(q_text, TOP_K).expect("query");
        hits_per_q.push(hits.into_iter().map(|h| h.id.0).collect());
    }
    evaluate(&hits_per_q, baseline, qs)
}

// ─────────────────────────────────────────────────────────────────────
// Pretty-print
// ─────────────────────────────────────────────────────────────────────

fn print_row_single(label: &str, r: &TrialResult) {
    eprintln!(
        "{:<50} {:>10.2} {:>14.2} {:>12.2}",
        label, r.top1_group_match, r.top1_baseline_match, r.recall_overlap
    );
}

fn print_row_multi(label: &str, m: &TrialResult, s: &TrialResult) {
    eprintln!(
        "{:<50} {:>4.2}±{:>4.2} {:>8.2}±{:>4.2} {:>6.2}±{:>4.2}",
        label,
        m.top1_group_match,
        s.top1_group_match,
        m.top1_baseline_match,
        s.top1_baseline_match,
        m.recall_overlap,
        s.recall_overlap,
    );
}

// ─────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "downloads MiniLM-L6 fastembed weights on first run; ~30s end-to-end"]
fn obfuscation_accuracy_comparison() {
    eprintln!("[load] initialising FastEmbed (MiniLM-L6-v2)...");
    let probe = FastEmbedEmbedder::new_smallest().expect("FastEmbed load");
    let mut dim_probe = FastEmbedEmbedder::new_smallest().expect("FastEmbed load");
    let dim = <FastEmbedEmbedder as Embedder>::embed(&mut dim_probe, &["probe".into()])
        .expect("probe embed")[0]
        .len();
    drop(probe);
    drop(dim_probe);

    let corpus = docs();
    let qs = queries();
    eprintln!(
        "[load] embed dim = {dim}; corpus = {} docs across {} groups; queries = {}",
        corpus.len(),
        {
            let mut groups: Vec<_> = corpus_rows().iter().map(|(_, g, _)| *g).collect();
            groups.sort();
            groups.dedup();
            groups.len()
        },
        qs.len(),
    );

    // ─── Plaintext baseline ───
    eprintln!("[baseline] computing plain-cosine baseline ranking...");
    let mut baseline_embedder = FastEmbedEmbedder::new_smallest().expect("FastEmbed load");
    let baseline = plaintext_baseline_ranking(&mut baseline_embedder, &corpus, &qs);
    drop(baseline_embedder);
    eprintln!("[baseline] top-{TOP_K} per query:");
    for ((q, expected), top) in qs.iter().zip(baseline.iter()) {
        let groups: Vec<_> = top.iter().map(|id| group_of(id)).collect();
        eprintln!(
            "    expected={:<6} top-{TOP_K}={:?} (groups={:?}) | query={:?}",
            expected, top, groups, q
        );
    }
    let baseline_self = evaluate(&baseline, &baseline, &qs);

    // ─── Run each config ───
    eprintln!("\n[run] CAPRISE (deterministic; 1 trial)");
    let caprise = run_caprise_trial(
        FastEmbedEmbedder::new_smallest().expect("FastEmbed load"),
        &corpus,
        &qs,
        &baseline,
    );

    eprintln!("[run] CAPRISE + DP-Forward (ε sweep × {N_TRIALS} trials)");
    let mut caprise_dp: Vec<(f64, TrialResult, TrialResult)> = Vec::new();
    for &eps in &[1.0_f64, 4.0, 16.0] {
        let dp_cfg = DpForwardConfig::calibrate(eps, 1e-5, 1.0);
        let trials: Vec<TrialResult> = (0..N_TRIALS)
            .map(|_| {
                let embedder = DpForwardFastEmbed::new(
                    FastEmbedEmbedder::new_smallest().expect("FastEmbed load"),
                    dp_cfg,
                );
                run_caprise_trial(embedder, &corpus, &qs, &baseline)
            })
            .collect();
        let (m, s) = mean_std(&trials);
        caprise_dp.push((eps, m, s));
    }

    eprintln!("[run] RemoteRAG (planar-Laplace ε sweep × {N_TRIALS} trials)");
    let mut remote_rag: Vec<(f64, TrialResult, TrialResult)> = Vec::new();
    for &mul in &[1.0_f64, 10.0, 50.0] {
        let planar_eps = mul * dim as f64;
        let trials: Vec<TrialResult> = (0..N_TRIALS)
            .map(|_| {
                let embedder = FastEmbedEmbedder::new_smallest().expect("FastEmbed load");
                run_remote_rag_trial(embedder, &corpus, &qs, &baseline, planar_eps, dim)
            })
            .collect();
        let (m, s) = mean_std(&trials);
        remote_rag.push((mul, m, s));
    }

    eprintln!("[run] RemoteRAG + doc-side DP-Forward (ε_doc = 4, ε_q = 10·n; {N_TRIALS} trials)");
    let rrag_docdp = {
        let dp_cfg = DpForwardConfig::calibrate(4.0, 1e-5, 1.0);
        let planar_eps = 10.0 * dim as f64;
        let trials: Vec<TrialResult> = (0..N_TRIALS)
            .map(|_| {
                let embedder = DpForwardFastEmbed::new(
                    FastEmbedEmbedder::new_smallest().expect("FastEmbed load"),
                    dp_cfg,
                );
                run_remote_rag_trial(embedder, &corpus, &qs, &baseline, planar_eps, dim)
            })
            .collect();
        mean_std(&trials)
    };

    // ─── Print summary ───
    eprintln!();
    eprintln!("=== Retrieval accuracy (top_k = {TOP_K}, trials = {N_TRIALS} where shown) ===");
    eprintln!("top1_grp  : rank-1's group matches expected group (semantic — embedder check)");
    eprintln!("top1_base : rank-1 doc-id matches plaintext baseline's rank-1 (PROTOCOL FIDELITY)");
    eprintln!("rec@{TOP_K}    : |top-{TOP_K} ∩ baseline_top_{TOP_K}| / {TOP_K} averaged over queries (HEAD-OF-LIST STABILITY)");
    eprintln!();
    eprintln!(
        "{:<50} {:>10} {:>14} {:>12}",
        "config", "top1_grp", "top1_base", "rec@3"
    );
    eprintln!("{}", "-".repeat(91));
    print_row_single("Plaintext baseline (reference)", &baseline_self);
    print_row_single("CAPRISE", &caprise);
    for (eps, m, s) in &caprise_dp {
        print_row_multi(&format!("CAPRISE + DP-Forward(ε={})", eps), m, s);
    }
    for (mul, m, s) in &remote_rag {
        print_row_multi(
            &format!("RemoteRAG (planar-Laplace ε={:.0}·n)", mul),
            m,
            s,
        );
    }
    print_row_multi(
        "RemoteRAG + doc-DP(ε=4, ε_q=10·n)",
        &rrag_docdp.0,
        &rrag_docdp.1,
    );
    eprintln!();
}

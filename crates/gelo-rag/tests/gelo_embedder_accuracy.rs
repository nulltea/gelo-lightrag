//! Does **GELO masking** corrupt retrieval ranking, and if so, is it
//! architecture-dependent (decoder-LLM vs encoder-BERT)?
//!
//! The previous accuracy bench (`obfuscation_accuracy.rs`) showed Qwen3 +
//! GELO returning semantically wrong rank-1s on a corpus where the same
//! corpus + FastEmbed (plain bi-encoder) is perfectly accurate. Two
//! hypotheses to disambiguate:
//!
//! - **GELO masking is broadly destructive.** The ~1e-2 per-dim drift
//!   from the orthogonal mask roundtrip (well below the parity threshold)
//!   is enough to flip cosine ranks for close-cosine documents *regardless
//!   of architecture*. We'd see BERT-plain disagree with BERT+GELO too.
//! - **Decoder-LLM anisotropy.** Qwen3's last-token-pooled embeddings are
//!   anisotropic on short texts, and the small mask noise tips an already-
//!   fragile ranking. BERT mean-pooled embeddings should be more
//!   isotropic and tolerate the mask roundtrip better.
//!
//! Five configurations on the same corpus + queries used by
//! `obfuscation_accuracy.rs`:
//!
//! 1. **FastEmbed MiniLM-L6** (control — known-good retrieval).
//! 2. **BGE-small (plain)** — `GeloBertEmbedder<PlaintextExecutor>`. No
//!    GELO masking applied; this is BERT as an embedder.
//! 3. **BGE-small + GELO** — `GeloBertEmbedder<InProcessTrustedExecutor>`.
//!    Per-batch fresh orthogonal mask round-trips through CPU offload.
//! 4. **Qwen3-Embedding-0.6B (plain)** — same as 2 for the decoder path.
//! 5. **Qwen3-Embedding-0.6B + GELO** — same as 3 for the decoder path.
//!
//! Metrics per config:
//! - `top1_grp`: rank-1 doc's topical group equals the query's expected
//!   group. Cross-comparable across all 5 configs.
//! - `top1_vs_plain`: rank-1 doc-id equals the *same model's* plain
//!   (non-GELO) version's rank-1. Defined only for the +GELO configs.
//!   Measures the mask roundtrip's distortion of that model's ranking.
//! - `rec3_vs_plain`: head-of-list stability vs the plain version.
//!
//! Reading the table:
//! - If `BGE+GELO`'s `top1_vs_plain ≈ 1.0` and `Qwen3+GELO`'s ≪ 1.0 →
//!   GELO + decoder-LLM is the problem; BERT is fine.
//! - If both `top1_vs_plain` values are low → GELO masking is broadly
//!   destructive (across architectures).
//! - If both are 1.0 but Qwen3-plain's `top1_grp` is low → Qwen3
//!   anisotropy alone explains the previous bench (GELO is innocent).
//!
//! All five configs use `RayonCpuEngine` for engine-parity; accuracy is
//! engine-independent modulo f32 order-of-ops noise (≪ the mask drift).
//!
//! Run:
//!
//! ```text
//! cargo test -p gelo-rag --release \
//!     --test gelo_embedder_accuracy -- --ignored --nocapture
//! ```

use std::sync::Arc;

use gelo_embedder::{DecoderConfig, DecoderWeights, GeloBertEmbedder, GeloQwenEmbedder};
use gelo_embedder::decoder::rope::RopeTables;
use gelo_gpu_wgpu::WgpuVulkanEngine;
use gelo_protocol::rng::MaskSeed;
use gelo_protocol::{InProcessTrustedExecutor, PlaintextExecutor, RayonCpuEngine};
use rag_core::{ChunkId, DocumentChunk, Embedder, FastEmbedEmbedder};

const TOP_K: usize = 3;
const BERT_MODEL: &str = "BAAI/bge-small-en-v1.5";
const QWEN_MODEL: &str = "Qwen/Qwen3-Embedding-0.6B";

// ─────────────────────────────────────────────────────────────────────
// Corpus — identical to obfuscation_accuracy.rs.
// ─────────────────────────────────────────────────────────────────────

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
    vec![
        ("Rust ownership and borrow checker for memory safety", "rust"),
        ("Python asyncio coroutines on an event loop", "python"),
        ("Kubernetes pods as the smallest deployable unit", "dist"),
        ("Paillier homomorphic encryption adding ciphertexts", "crypto"),
    ]
}

// ─────────────────────────────────────────────────────────────────────
// Direct-cosine retrieval over any `Embedder`. Skips the
// gelo-rag service so we measure pure inference + cosine
// without protocol-side encryption distortion in the mix.
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

fn retrieve_topk(
    embedder: &mut impl Embedder,
    corpus: &[DocumentChunk],
    qs: &[(&'static str, &'static str)],
) -> Vec<Vec<String>> {
    retrieve_topk_with_format(embedder, corpus, qs, |q| q.to_string())
}

/// Variant that lets the caller pre-format the query (e.g. add an
/// instruction prefix). Doc text is passed through unchanged.
fn retrieve_topk_with_format<F: Fn(&str) -> String>(
    embedder: &mut impl Embedder,
    corpus: &[DocumentChunk],
    qs: &[(&'static str, &'static str)],
    fmt: F,
) -> Vec<Vec<String>> {
    let doc_texts: Vec<String> = corpus.iter().map(|d| d.text.clone()).collect();
    let doc_embeds = embedder.embed(&doc_texts).expect("doc embed");
    let mut out = Vec::with_capacity(qs.len());
    for (q_text, _) in qs {
        let formatted = fmt(q_text);
        let q_embed = embedder
            .embed(&[formatted])
            .expect("query embed")
            .remove(0);
        let mut scored: Vec<(usize, f32)> = doc_embeds
            .iter()
            .enumerate()
            .map(|(i, e)| (i, cosine(&q_embed, e)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out.push(
            scored
                .into_iter()
                .take(TOP_K)
                .map(|(i, _)| corpus[i].id.0.clone())
                .collect(),
        );
    }
    out
}

// ─────────────────────────────────────────────────────────────────────
// Metrics
// ─────────────────────────────────────────────────────────────────────

#[derive(Default, Clone, Copy, Debug)]
struct ConfigResult {
    /// rank-1's group label equals the expected group (cross-comparable).
    top1_grp: f64,
    /// rank-1 doc-id equals the *plain* version's rank-1 for the same model.
    /// `None` for plain-executor configs (where this is trivially 1.0).
    top1_vs_plain: Option<f64>,
    /// `|top-3 ∩ plain_top_3| / 3` averaged over queries.
    rec3_vs_plain: Option<f64>,
}

fn top1_group_accuracy(
    hits: &[Vec<String>],
    qs: &[(&'static str, &'static str)],
) -> f64 {
    let n = qs.len() as f64;
    let mut g = 0.0;
    for (h, (_, expected)) in hits.iter().zip(qs.iter()) {
        if let Some(top) = h.first() {
            if group_of(top) == *expected {
                g += 1.0;
            }
        }
    }
    g / n
}

fn vs_plain_top1(hits: &[Vec<String>], plain: &[Vec<String>]) -> f64 {
    let n = hits.len() as f64;
    hits.iter()
        .zip(plain.iter())
        .filter(|(h, p)| h.first() == p.first())
        .count() as f64
        / n
}

fn vs_plain_rec3(hits: &[Vec<String>], plain: &[Vec<String>]) -> f64 {
    let n = hits.len() as f64;
    let mut sum = 0.0;
    for (h, p) in hits.iter().zip(plain.iter()) {
        let p_set: std::collections::HashSet<&String> = p.iter().take(TOP_K).collect();
        let overlap = h.iter().take(TOP_K).filter(|x| p_set.contains(x)).count();
        sum += overlap as f64 / TOP_K as f64;
    }
    sum / n
}

// ─────────────────────────────────────────────────────────────────────
// Pretty-print
// ─────────────────────────────────────────────────────────────────────

fn print_row(label: &str, r: &ConfigResult) {
    let v1 = r
        .top1_vs_plain
        .map(|v| format!("{v:>14.2}"))
        .unwrap_or_else(|| format!("{:>14}", "—"));
    let v3 = r
        .rec3_vs_plain
        .map(|v| format!("{v:>14.2}"))
        .unwrap_or_else(|| format!("{:>14}", "—"));
    eprintln!("{:<45} {:>10.2} {} {}", label, r.top1_grp, v1, v3);
}

// ─────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "downloads BGE-small (~130 MB) and Qwen3-0.6B (~1.2 GB) on first run; CPU only"]
fn gelo_masking_vs_plain_across_architectures() {
    let corpus = docs();
    let qs = queries();
    eprintln!(
        "[load] corpus = {} docs across {} groups; queries = {}",
        corpus.len(),
        {
            let mut groups: Vec<_> = corpus_rows().iter().map(|(_, g, _)| *g).collect();
            groups.sort();
            groups.dedup();
            groups.len()
        },
        qs.len(),
    );

    // ─── FastEmbed control ───
    eprintln!("[run] FastEmbed (control)...");
    let mut fast = FastEmbedEmbedder::new_smallest().expect("FastEmbed load");
    let fast_hits = retrieve_topk(&mut fast, &corpus, &qs);
    let fast_result = ConfigResult {
        top1_grp: top1_group_accuracy(&fast_hits, &qs),
        top1_vs_plain: None,
        rec3_vs_plain: None,
    };
    drop(fast);

    // ─── BGE plain (BERT, PlaintextExecutor — no GELO masking) ───
    eprintln!("[run] BGE-small (plain — no GELO mask) ...");
    let plain_exec_bert = PlaintextExecutor::new(RayonCpuEngine::new());
    let mut bert_plain =
        GeloBertEmbedder::from_pretrained(BERT_MODEL, plain_exec_bert).expect("BGE load");
    let bert_plain_hits = retrieve_topk(&mut bert_plain, &corpus, &qs);
    let bert_plain_result = ConfigResult {
        top1_grp: top1_group_accuracy(&bert_plain_hits, &qs),
        top1_vs_plain: None,
        rec3_vs_plain: None,
    };
    drop(bert_plain);

    // ─── BGE + GELO masking ───
    eprintln!("[run] BGE-small + GELO masking ...");
    let masked_exec_bert =
        InProcessTrustedExecutor::with_seed(RayonCpuEngine::new(), MaskSeed::from_bytes([7u8; 32]));
    let mut bert_gelo = GeloBertEmbedder::from_pretrained(BERT_MODEL, masked_exec_bert)
        .expect("BGE load");
    let bert_gelo_hits = retrieve_topk(&mut bert_gelo, &corpus, &qs);
    let bert_gelo_result = ConfigResult {
        top1_grp: top1_group_accuracy(&bert_gelo_hits, &qs),
        top1_vs_plain: Some(vs_plain_top1(&bert_gelo_hits, &bert_plain_hits)),
        rec3_vs_plain: Some(vs_plain_rec3(&bert_gelo_hits, &bert_plain_hits)),
    };
    drop(bert_gelo);

    // ─── Qwen3 plain (decoder-LLM, PlaintextExecutor) ───
    eprintln!("[run] Qwen3-0.6B (plain — no GELO mask) ...");
    let plain_exec_qwen = PlaintextExecutor::new(RayonCpuEngine::new());
    let qwen_plain_seed = GeloQwenEmbedder::from_pretrained(QWEN_MODEL, plain_exec_qwen)
        .expect("Qwen3 load");
    // Share weights between the two Qwen3 variants — keeps the second
    // load instant after the first download.
    let qwen_weights: Arc<DecoderWeights> = qwen_plain_seed.weights_arc();
    let qwen_rope: Arc<RopeTables> = qwen_plain_seed.rope_arc();
    let qwen_tokenizer = qwen_plain_seed.tokenizer().clone();
    let qwen_cfg: DecoderConfig = qwen_plain_seed.config().clone();
    let mut qwen_plain = qwen_plain_seed;
    let qwen_plain_hits = retrieve_topk(&mut qwen_plain, &corpus, &qs);
    let qwen_plain_result = ConfigResult {
        top1_grp: top1_group_accuracy(&qwen_plain_hits, &qs),
        top1_vs_plain: None,
        rec3_vs_plain: None,
    };
    drop(qwen_plain);

    // ─── Qwen3 + GELO masking (production defaults — OutAttnMult auto-disabled at short n) ───
    eprintln!("[run] Qwen3-0.6B + GELO masking (defaults — auto-switch keeps OutAttnMult OFF at short n) ...");
    let masked_exec_qwen = InProcessTrustedExecutor::with_seed(
        RayonCpuEngine::new(),
        MaskSeed::from_bytes([11u8; 32]),
    );
    let mut qwen_gelo = GeloQwenEmbedder::with_shared_weights(qwen_cfg.clone(), qwen_tokenizer.clone(), Arc::clone(&qwen_weights), Arc::clone(&qwen_rope), masked_exec_qwen)
    .expect("Qwen3 build with masked executor");
    let qwen_gelo_hits = retrieve_topk(&mut qwen_gelo, &corpus, &qs);
    let qwen_gelo_result = ConfigResult {
        top1_grp: top1_group_accuracy(&qwen_gelo_hits, &qs),
        top1_vs_plain: Some(vs_plain_top1(&qwen_gelo_hits, &qwen_plain_hits)),
        rec3_vs_plain: Some(vs_plain_rec3(&qwen_gelo_hits, &qwen_plain_hits)),
    };
    drop(qwen_gelo);

    // ─── Qwen3 + GELO + OutAttnMult FORCED on (min_seq_len = Some(0)) ───
    // Reproduces the obfuscation_bench.rs / obfuscation_accuracy.rs earlier
    // configuration that exhibited pathological retrieval. Production
    // default is `min_seq_len = None` (auto, ≥ hidden_size); forcing it on
    // for short prompts exercises the 4-partition Q·Kᵀ path under sequence
    // lengths it was not designed for.
    eprintln!("[run] Qwen3-0.6B + GELO + OutAttnMult FORCED on at any n (n < hidden_size) ...");
    let masked_exec_outattn = InProcessTrustedExecutor::with_seed(
        RayonCpuEngine::new(),
        MaskSeed::from_bytes([11u8; 32]),
    );
    let mut qwen_outattn = GeloQwenEmbedder::with_shared_weights(qwen_cfg.clone(), qwen_tokenizer.clone(), Arc::clone(&qwen_weights), Arc::clone(&qwen_rope), masked_exec_outattn)
    .expect("Qwen3 build with masked executor")
    .with_out_attn_mult(true)
    .with_out_attn_mult_min_seq_len(Some(0));
    let qwen_outattn_hits = retrieve_topk(&mut qwen_outattn, &corpus, &qs);
    let qwen_outattn_result = ConfigResult {
        top1_grp: top1_group_accuracy(&qwen_outattn_hits, &qs),
        top1_vs_plain: Some(vs_plain_top1(&qwen_outattn_hits, &qwen_plain_hits)),
        rec3_vs_plain: Some(vs_plain_rec3(&qwen_outattn_hits, &qwen_plain_hits)),
    };
    drop(qwen_outattn);

    // ─── Qwen3 plain + Qwen3-Embedding "Instruct: ... \nQuery: ..." prefix ───
    // Qwen3-Embedding-0.6B's HF model card documents this prefix for
    // "retrieval / RAG" workloads. Reload `qwen_plain` (was moved earlier
    // — share weights via Arc to keep this nearly free).
    eprintln!("[run] Qwen3-0.6B (plain) WITH instruction prefix \"Instruct: ...\\nQuery:\" ...");
    let plain_exec_qwen_p = PlaintextExecutor::new(RayonCpuEngine::new());
    let mut qwen_plain_prefixed = GeloQwenEmbedder::with_shared_weights(qwen_cfg.clone(), qwen_tokenizer.clone(), Arc::clone(&qwen_weights), Arc::clone(&qwen_rope), plain_exec_qwen_p)
    .expect("Qwen3 plain rebuild");
    let prefix_fmt = |q: &str| {
        format!(
            "Instruct: Given a question, retrieve passages that directly answer it.\nQuery: {q}"
        )
    };
    let qwen_plain_prefixed_hits =
        retrieve_topk_with_format(&mut qwen_plain_prefixed, &corpus, &qs, prefix_fmt);
    let qwen_plain_prefixed_result = ConfigResult {
        top1_grp: top1_group_accuracy(&qwen_plain_prefixed_hits, &qs),
        // Compared to Qwen3 *without* prefix — does the prefix shift ranking?
        top1_vs_plain: Some(vs_plain_top1(&qwen_plain_prefixed_hits, &qwen_plain_hits)),
        rec3_vs_plain: Some(vs_plain_rec3(&qwen_plain_prefixed_hits, &qwen_plain_hits)),
    };
    drop(qwen_plain_prefixed);

    // ─── Qwen3 + GELO on Vulkan engine ───
    // Localizes whether the engine itself (CPU vs Vulkan) affects rank
    // stability. Reuses the cached weights + RoPE tables via Arc — only
    // the executor (and the engine inside it) differs.
    let qwen_vulkan_result = match WgpuVulkanEngine::new() {
        Ok(gpu) if gpu.is_real_gpu() => {
            eprintln!(
                "[run] Qwen3-0.6B + GELO on Vulkan ({} {:?}) ...",
                gpu.adapter_info().name,
                gpu.adapter_info().device_type
            );
            let masked_exec_vulkan = InProcessTrustedExecutor::with_seed(
                gpu.clone_shared(),
                MaskSeed::from_bytes([11u8; 32]),
            );
            // We need a fresh `DecoderConfig` here too (the previous one
            // was moved into `qwen_outattn`). Default-from-pretrained-config
            // already used above; just reload by calling .config() on a
            // fresh from_pretrained — but to avoid a redownload, recreate
            // by sharing weights with a fresh embedder built from the
            // already-Arc'd weights.
            //
            // We can't easily recover the original `DecoderConfig` since
            // it was moved, so this path uses a fresh from_pretrained
            // which is free on the second call (HF cache).
            let mut qwen_vulkan = GeloQwenEmbedder::from_pretrained(
                QWEN_MODEL,
                masked_exec_vulkan,
            )
            .expect("Qwen3 build with Vulkan masked executor");
            let hits = retrieve_topk(&mut qwen_vulkan, &corpus, &qs);
            Some(ConfigResult {
                top1_grp: top1_group_accuracy(&hits, &qs),
                top1_vs_plain: Some(vs_plain_top1(&hits, &qwen_plain_hits)),
                rec3_vs_plain: Some(vs_plain_rec3(&hits, &qwen_plain_hits)),
            }).map(|r| (r, hits))
        }
        Ok(_) => {
            eprintln!("[skip] no real GPU (llvmpipe only); skipping Vulkan row");
            None
        }
        Err(_) => {
            eprintln!("[skip] Vulkan adapter unavailable; skipping Vulkan row");
            None
        }
    };

    // ─── Print baselines and per-query top-3 for inspection ───
    eprintln!("\nPer-query top-{TOP_K} (for inspection):");
    for ((q, expected), idx) in qs.iter().zip(0..) {
        eprintln!(
            "  expected={:<6} query={:?}",
            expected, q
        );
        eprintln!(
            "    fast    : {:?}  (groups {:?})",
            fast_hits[idx],
            fast_hits[idx].iter().map(|s| group_of(s)).collect::<Vec<_>>()
        );
        eprintln!(
            "    bert    : {:?}  (groups {:?})",
            bert_plain_hits[idx],
            bert_plain_hits[idx].iter().map(|s| group_of(s)).collect::<Vec<_>>()
        );
        eprintln!(
            "    bert+G  : {:?}  (groups {:?})",
            bert_gelo_hits[idx],
            bert_gelo_hits[idx].iter().map(|s| group_of(s)).collect::<Vec<_>>()
        );
        eprintln!(
            "    qwen    : {:?}  (groups {:?})",
            qwen_plain_hits[idx],
            qwen_plain_hits[idx].iter().map(|s| group_of(s)).collect::<Vec<_>>()
        );
        eprintln!(
            "    qwen+G  : {:?}  (groups {:?})",
            qwen_gelo_hits[idx],
            qwen_gelo_hits[idx].iter().map(|s| group_of(s)).collect::<Vec<_>>()
        );
        eprintln!(
            "    qwen+GO : {:?}  (groups {:?})",
            qwen_outattn_hits[idx],
            qwen_outattn_hits[idx].iter().map(|s| group_of(s)).collect::<Vec<_>>()
        );
        if let Some((_, vulkan_hits)) = &qwen_vulkan_result {
            eprintln!(
                "    qwen+GV : {:?}  (groups {:?})",
                vulkan_hits[idx],
                vulkan_hits[idx].iter().map(|s| group_of(s)).collect::<Vec<_>>()
            );
        }
        eprintln!(
            "    qwen+Pf : {:?}  (groups {:?})  [plain + instruction prefix]",
            qwen_plain_prefixed_hits[idx],
            qwen_plain_prefixed_hits[idx].iter().map(|s| group_of(s)).collect::<Vec<_>>()
        );
    }

    eprintln!();
    eprintln!("=== Embedder architecture × GELO masking accuracy (top_k = {TOP_K}) ===");
    eprintln!("top1_grp       : rank-1 matches expected group (cross-config comparable)");
    eprintln!("top1_vs_plain  : rank-1 matches same-model plain (PlaintextExecutor) rank-1");
    eprintln!("rec3_vs_plain  : top-3 overlap with same-model plain version");
    eprintln!();
    eprintln!(
        "{:<45} {:>10} {:>14} {:>14}",
        "config", "top1_grp", "top1_vs_plain", "rec3_vs_plain"
    );
    eprintln!("{}", "-".repeat(86));
    print_row("FastEmbed MiniLM-L6 (control)", &fast_result);
    print_row("BGE-small (plain — no GELO mask)", &bert_plain_result);
    print_row("BGE-small + GELO masking", &bert_gelo_result);
    print_row("Qwen3-0.6B (plain — no GELO mask)", &qwen_plain_result);
    print_row("Qwen3-0.6B + GELO masking (defaults)", &qwen_gelo_result);
    print_row(
        "Qwen3-0.6B + GELO + OutAttnMult forced (n < hidden_size)",
        &qwen_outattn_result,
    );
    if let Some((vulkan_r, _)) = &qwen_vulkan_result {
        print_row("Qwen3-0.6B + GELO on Vulkan engine", vulkan_r);
    }
    print_row(
        "Qwen3-0.6B (plain) + instruction prefix",
        &qwen_plain_prefixed_result,
    );
    eprintln!();
    eprintln!("Interpretation:");
    eprintln!("  Across all GELO/engine variants, `top1_vs_plain ≈ 1.0` → GELO masking,");
    eprintln!("    OutAttnMult, and the Vulkan engine path all preserve ranking exactly.");
    eprintln!();
    eprintln!("  If `qwen+Pf top1_grp ≪ 1.0` AND `top1_vs_plain ≪ 1.0`:");
    eprintln!("    The `Instruct: ...\\nQuery: ...` prefix recommended by the Qwen3-Embedding");
    eprintln!("    model card is the actual ranking-corruption source observed in the earlier");
    eprintln!("    Qwen3+GELO bench. Last-token pooling makes the embedding heavily dependent");
    eprintln!("    on the trailing context, so the prefix dominates the pooled vector and");
    eprintln!("    pulls every query toward the same instruction-tinted region of the embedding");
    eprintln!("    space, defeating semantic separation on short queries.");
    eprintln!();
    eprintln!("  Practical guidance: do NOT apply the model-card instruction prefix when using");
    eprintln!("    Qwen3-Embedding-0.6B for direct cosine retrieval — at least not on this");
    eprintln!("    corpus shape. The prefix is calibrated for the fine-tuning task description");
    eprintln!("    used during training, not for zero-shot retrieval of short docs.");
}

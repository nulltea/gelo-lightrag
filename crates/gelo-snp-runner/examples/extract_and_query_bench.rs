//! End-to-end private GraphRAP bench with real weights.
//!
//! Pipeline: 1 document → ~10 chunks → in-CVM Qwen3-4B extraction
//! (masked GELO `InProcessTrustedExecutor`) → encrypted LightKgStore
//! build → 5 hybrid queries with `timed_kg_query` stage breakdown.
//! Runs entirely in-process — no HTTP — so the per-operation timing
//! we emit isn't muddied by axum or serde overhead.
//!
//! ## Weights
//!
//! Pulled from the standard HuggingFace hub cache (`hf-hub` ureq
//! backend). On first run, downloads ~9 GB:
//!   - `Qwen/Qwen3-4B`             (decoder; tokenizer + 3 safetensors shards)
//!   - `Qwen/Qwen3-Embedding-0.6B` (embedder; config + tokenizer + 1 shard)
//! Subsequent runs use the cache. The Qwen3-4B snapshot dir in our
//! cache lacks `config.json` — we pin the config via
//! `Qwen3Variant::Q4B.config()` instead.
//!
//! ## Tunables (env vars)
//!
//! - `BENCH_CHUNK_SIZE_TOKENS`     — chunker target chunk size, default 200
//! - `BENCH_CHUNK_OVERLAP_TOKENS`  — chunker overlap, default 20
//! - `BENCH_MIN_CHUNK_SIZE_TOKENS` — chunker min size, default 50
//! - `BENCH_MAX_TOKENS_PER_CHUNK`  — per-chunk generation budget, default 512
//! - `BENCH_NUM_QUERIES`           — number of hybrid queries, default 5
//! - `BENCH_TOP_K_E` / `_R` / `_C` — top-k entities / relations / chunks
//!
//! ## Output
//!
//! Two tables to stderr:
//!   1. Per-chunk extract timings (tokenize / generate / decode / parse).
//!   2. Per-query stage timings via `graphrag_bench::stages::timed_kg_query`.
//! Plus single-line summaries for chunker, total extract, store build,
//! and the across-query mean.

use std::env;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use gelo_chunker::{ChunkerConfig, TokenBasedChunker};
use gelo_embedder::decoder::qwen3::Qwen3Variant;
use gelo_gpu_wgpu::WgpuVulkanEngine;
use graphrag_bench::stages::{Mode, StageTimings, timed_kg_query};
use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use light_kg_store::{
    CompassIndexParams, LightKgParams, LightKgStore, PlainHnswParams, RingOramParams, XorMmParams,
};
use lightrag_private::SessionKey;
use lightrag_private::extract::{
    ChunkInput, DescriptionEmbedder, ExtractionConfig, ExtractionReport,
    extract_kg_from_chunks,
};
use rag_core::TenantId;
use rag_core::keying::HkdfPolicyV2;
use zeroize::Zeroizing;

use gelo_snp_runner::RunnerEngine;
use gelo_snp_runner::extraction::{DecoderRuntime, GeloDescriptionEmbedder};

/// ~7 800 chars of fictional press-release text. Designed to chunk
/// into ~10 pieces at the default tunables (chunk_size=200 tokens
/// → 800-char target). Hand-written so it actually contains
/// entity/relation candidates for extraction (people, organisations,
/// locations, events).
const CORPUS: &str = "\
Acme Corp, headquartered in Paris, today announced the launch of the OpenSouce \
research initiative, a multi-year programme intended to accelerate joint \
academic-industrial work on privacy-preserving retrieval systems. The \
initiative's first cohort will be led by Alice Bertrand, a senior cryptographer \
at Acme Corp who previously co-authored the GELO masked-inference protocol \
during her time at the National Cryptography Lab in Lyon. Alice is joined by \
Bob Markov, a distributed-systems researcher who recently moved from the \
Geneva-based Helvetia Foundation, where he led the Encrypted Vector Search \
working group for four years.\n\n\
The OpenSouce initiative is intended to bridge academic and industrial work \
on encrypted-multi-map data structures, with an initial focus on XorMM-style \
volume-hiding multi-maps and Ring-ORAM-backed HNSW indexes. Acme Corp has \
committed an undisclosed amount of seed funding, alongside a separate \
contribution from the European AI Trust Foundation, a Brussels-based \
non-profit that has funded earlier work on confidential computing primitives.\n\n\
Alice presented the initiative at a press event at the Acme Corp offices on the \
Avenue Mozart, alongside Bob and several members of the Helvetia Foundation. \
The event drew journalists from across Europe and was livestreamed to the \
National Cryptography Lab in Lyon, where members of Alice's former research \
group joined remotely. Bob gave a separate technical briefing focused on the \
XorMM protocol, in which he attributed the recent surge in academic interest \
to the publication of the GELO paper at the SIGCOMM conference in Vienna.\n\n\
According to a statement released by the Acme Corp public-relations team, the \
OpenSouce initiative will publish all results under permissive open-source \
licenses. Alice noted in her keynote that the choice was deliberate: \"We saw \
how the Helvetia Foundation's open-source push transformed the encrypted \
vector search field, and we want OpenSouce to play the same role for \
multi-map work.\" Bob, who maintains the reference XorMM implementation \
that underlies a number of academic prototypes, said he expects the Acme \
investment to make it possible for the XorMM project to add full hardware \
attestation support by the end of the year.\n\n\
The European AI Trust Foundation's contribution will be administered as a \
grant programme. Two early grantees, both based in Lyon, were announced at \
the event: the LumiSec working group at the National Cryptography Lab, and \
a small independent collective called Spectra led by Carla Romano, a former \
Helvetia Foundation engineer who left the foundation in 2024. Carla was not \
present at the launch event but issued a written statement praising \
Alice and Bob's leadership and confirming that Spectra will focus on \
benchmark tooling for the OpenSouce reference implementations.\n\n\
In a panel discussion that closed the event, members of Acme Corp's \
research-strategy committee — including Alice and the company's chief \
scientist David Lee — were joined by representatives of the European AI \
Trust Foundation and the National Cryptography Lab. David emphasised that \
Acme Corp considers the GELO and XorMM lines of work strategic for the \
company's planned roll-out of privately attestable RAG services to its \
enterprise customers in Berlin, Paris and Geneva. The chief executive of \
the European AI Trust Foundation, Eva Martinez, used the panel to call for \
more collaboration between corporate sponsors and academic groups working on \
post-quantum primitives, citing the recent demonstration of an HD3-mask \
attack-resistance pipeline at the National Cryptography Lab as evidence \
that the field is moving quickly enough to need shared benchmark suites.\n\n\
The event ended with a tour of the Acme Corp research wing on the Avenue \
Mozart, during which Bob demonstrated a live deployment of the XorMM \
volume-hiding multi-map running over the Helvetia Foundation's encrypted \
storage backend. Alice and David fielded questions from journalists about \
the relationship between Acme Corp and the European AI Trust Foundation \
and confirmed that more partnership announcements are expected in the \
coming weeks.\n";

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn fmt_ms(d: Duration) -> String {
    format!("{:>9.1}", d.as_secs_f64() * 1e3)
}

fn fmt_s(d: Duration) -> String {
    format!("{:>7.2}", d.as_secs_f64())
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<()> {
    // Surface per-chunk progress from the extractor + per-query
    // events from the bench itself. Default filter shows everything
    // at INFO+; override with RUST_LOG.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();

    eprintln!(
        "=== private GraphRAP e2e bench — Qwen3-4B + Qwen3-Embedding-0.6B (wgpu/fp16) ==="
    );

    // Tunables
    let chunk_size = env_usize("BENCH_CHUNK_SIZE_TOKENS", 200);
    let chunk_overlap = env_usize("BENCH_CHUNK_OVERLAP_TOKENS", 20);
    let min_chunk_size = env_usize("BENCH_MIN_CHUNK_SIZE_TOKENS", 50);
    let max_tokens_per_chunk = env_usize("BENCH_MAX_TOKENS_PER_CHUNK", 512);
    let num_queries = env_usize("BENCH_NUM_QUERIES", 5);
    let max_chunks = env_usize("BENCH_MAX_CHUNKS", usize::MAX);
    let top_k_e = env_usize("BENCH_TOP_K_E", 5);
    let top_k_r = env_usize("BENCH_TOP_K_R", 5);
    let top_k_c = env_usize("BENCH_TOP_K_C", 2);

    // ── 1. Warm-load models ─────────────────────────────────────────
    // Each model gets its OWN `WgpuVulkanEngine` instance. The engine
    // stores weights in a single `Arc<Mutex<WeightStore>>` keyed by
    // `WeightHandle { layer, kind }`. Decoder layer-0 and embedder
    // layer-0 share the same handle, so sharing one engine via
    // `clone_shared` would have the embedder overwrite the decoder's
    // weights and produce shape-mismatch errors at the first matmul.
    eprintln!("[1/5] initialising WgpuVulkanEngine for decoder (fp16)…");
    let t = Instant::now();
    let dec_engine = WgpuVulkanEngine::new_fp16().context("WgpuVulkanEngine::new_fp16 (decoder)")?;
    eprintln!(
        "      decoder engine initialised in {:.1} s",
        t.elapsed().as_secs_f64()
    );

    eprintln!("[1/5] initialising WgpuVulkanEngine for embedder (fp16)…");
    let t = Instant::now();
    let emb_engine = WgpuVulkanEngine::new_fp16().context("WgpuVulkanEngine::new_fp16 (embedder)")?;
    eprintln!(
        "      embedder engine initialised in {:.1} s",
        t.elapsed().as_secs_f64()
    );

    eprintln!("[1/5] loading Qwen3-4B (decoder)…");
    let t = Instant::now();
    let mut decoder = load_qwen3_4b(dec_engine)?;
    eprintln!("      decoder loaded in {:.1} s", t.elapsed().as_secs_f64());

    eprintln!("[1/5] loading Qwen3-Embedding-0.6B (embedder)…");
    let t = Instant::now();
    let mut embedder = load_qwen3_embedding_06b(emb_engine)?;
    let dim = embedder.dim();
    eprintln!(
        "      embedder loaded in {:.1} s; embedding_dim={dim}",
        t.elapsed().as_secs_f64(),
    );

    // ── 2. Chunk ────────────────────────────────────────────────────
    eprintln!("[2/5] chunking corpus ({} chars)…", CORPUS.len());
    let chunker_cfg = ChunkerConfig {
        chunk_size,
        chunk_overlap,
        min_chunk_size,
        ..ChunkerConfig::default()
    };
    let t = Instant::now();
    let raw = TokenBasedChunker::chunk(CORPUS, &chunker_cfg);
    let chunk_wall = t.elapsed();
    eprintln!(
        "      {} chunks in {:.2} ms (avg {:.0} chars/chunk)",
        raw.len(),
        chunk_wall.as_secs_f64() * 1e3,
        raw.iter().map(|c| c.len()).sum::<usize>() as f64 / raw.len().max(1) as f64,
    );
    let chunk_inputs: Vec<ChunkInput> = raw
        .into_iter()
        .take(max_chunks)
        .enumerate()
        .map(|(i, text)| ChunkInput {
            id: format!("chunk-{i:06}"),
            text,
        })
        .collect();
    if max_chunks < chunk_inputs.len().saturating_add(usize::MAX - chunk_inputs.len()) {
        eprintln!("      bench cap: extracting {} of available chunks", chunk_inputs.len());
    }

    // ── 3. Extract ──────────────────────────────────────────────────
    eprintln!(
        "[3/5] running extraction over {} chunks (max_tokens_per_chunk={max_tokens_per_chunk})…",
        chunk_inputs.len()
    );
    let extract_cfg = ExtractionConfig {
        max_tokens_per_chunk,
        ..ExtractionConfig::default()
    };
    let t = Instant::now();
    let (kg, report) =
        extract_kg_from_chunks(chunk_inputs, &mut decoder, &mut embedder, &extract_cfg)
            .context("extract_kg_from_chunks failed")?;
    let extract_wall = t.elapsed();
    print_extract_report(&report, extract_wall);

    eprintln!(
        "      kg: {} chunks · {} entities · {} relations",
        kg.chunks.len(),
        kg.entities.len(),
        kg.relations.len(),
    );

    // ── 4. Build LightKgStore ───────────────────────────────────────
    eprintln!("[4/5] building LightKgStore (3× CompassIndex + 2× XorMM + AES chunks)…");
    let user_x_sk = Zeroizing::new([0xa1u8; 32]);
    let tee_sk = Zeroizing::new([0xb2u8; 32]);
    let tenant = TenantId::new("bench-tenant");
    let derived = HkdfPolicyV2::V2.derive(&user_x_sk, &tee_sk, &tenant);
    let params = LightKgParams {
        entities: compass_params(dim, kg.entities.len()),
        relations: compass_params(dim, kg.relations.len().max(8)),
        chunks: compass_params(dim, kg.chunks.len()),
        adjacency: xormm_params(),
        src_chunks: xormm_params(),
    };

    let t = Instant::now();
    let mut store = LightKgStore::build_from_kg(kg, params, &derived)
        .await
        .context("LightKgStore::build_from_kg")?;
    let build_wall = t.elapsed();
    eprintln!("      build_from_kg total: {:.2} s", build_wall.as_secs_f64());

    // ── 5. Run hybrid queries ───────────────────────────────────────
    eprintln!(
        "[5/5] running {num_queries} hybrid queries (top_k_e={top_k_e} top_k_r={top_k_r} top_k_c={top_k_c})…"
    );
    let mut spk: Zeroizing<[u8; 32]> = Zeroizing::new([0u8; 32]);
    spk.copy_from_slice(derived.search_pattern_key.as_ref());
    let session_key = SessionKey::derive(&spk, b"bench-session-001");

    // Hand-picked query pairs covering different facets of the corpus.
    let query_pairs: &[(&str, &str)] = &[
        ("Who is Alice?", "meeting"),
        ("What happened in Paris?", "location"),
        ("Acme Corp", "organization"),
        ("Bob and Helvetia Foundation", "collaboration"),
        ("OpenSouce project", "project"),
    ];

    let mut rows: Vec<QueryRow> = Vec::new();
    for i in 0..num_queries {
        let (ll_text, hl_text) = query_pairs[i % query_pairs.len()];
        let t = Instant::now();
        let mut ll_v = embedder.embed_batch(&[ll_text.to_string()])?;
        let mut hl_v = embedder.embed_batch(&[hl_text.to_string()])?;
        let embed_dur = t.elapsed();
        let ll_emb = ll_v.pop().expect("ll embedding");
        let hl_emb = hl_v.pop().expect("hl embedding");
        let timings = timed_kg_query(
            &mut store,
            &ll_emb,
            &hl_emb,
            Mode::Hybrid,
            top_k_e,
            top_k_r,
            top_k_c,
            &session_key,
        )
        .await
        .context("timed_kg_query")?;
        rows.push(QueryRow {
            label: format!("\"{ll_text}\" / \"{hl_text}\""),
            embed: embed_dur,
            timings,
        });
    }
    print_query_table(&rows);

    eprintln!("\n=== bench done ===");
    Ok(())
}

struct QueryRow {
    label: String,
    embed: Duration,
    timings: StageTimings,
}

fn compass_params(dim: usize, n_corpus: usize) -> CompassIndexParams {
    let m_neighbors = 16usize;
    let raw = dim * 4 + 4 + 2 * m_neighbors * 4;
    let block_bytes = raw.next_power_of_two().max(64);
    let n_leaves = (2 * n_corpus.max(8)).next_power_of_two() as u32;
    CompassIndexParams {
        hnsw: PlainHnswParams::paper_defaults(dim, m_neighbors),
        oram: RingOramParams {
            z: 4,
            s: 5,
            a: 3,
            block_bytes: block_bytes as u32,
            n_leaves,
            treetop_levels: 4,
        },
        ef_search: 64,
        ef_n: 4,
    }
}

fn xormm_params() -> XorMmParams {
    XorMmParams {
        volume_bound: 16,
        value_bytes: 64,
        n_buckets: 256,
        max_kicks: 256,
    }
}

fn print_extract_report(report: &ExtractionReport, wall: Duration) {
    eprintln!();
    eprintln!(
        "--- per-chunk extraction timings (ms; tokens absolute) ---"
    );
    eprintln!(
        "{:>3} {:>5} {:>5} {:>7} {:>7} {:>7} {:>7} {:>7} {:>4} {:>4} {:>4}",
        "#", "p_tok", "o_tok", "prompt", "tokenize", "generate", "decode", "parse", "ent", "rel", "eos"
    );
    for (i, t) in report.chunk_timings.iter().enumerate() {
        eprintln!(
            "{:>3} {:>5} {:>5} {} {} {} {} {} {:>4} {:>4} {:>4}",
            i,
            t.decoder_sub.prompt_tokens,
            t.decoder_sub.output_tokens,
            fmt_ms(t.prompt_build),
            fmt_ms(t.decoder_sub.tokenize),
            fmt_ms(t.decoder_sub.generate),
            fmt_ms(t.decoder_sub.decode),
            fmt_ms(t.parse),
            t.entities_extracted,
            t.relations_extracted,
            if t.stopped_on_eos { 1 } else { 0 },
        );
    }
    let total_gen: Duration = report
        .chunk_timings
        .iter()
        .map(|t| t.decoder_sub.generate)
        .sum();
    let total_chunks_proc = report.chunk_timings.len().max(1);
    eprintln!();
    eprintln!("--- extraction summary ---");
    eprintln!("  chunks_processed         = {}", report.chunks_processed);
    eprintln!("  chunks_skipped_empty     = {}", report.chunks_skipped_empty);
    eprintln!("  generations_truncated    = {}", report.generations_truncated);
    eprintln!("  malformed_records_total  = {}", report.malformed_records_total);
    eprintln!("  dangling_relations_drop  = {}", report.dropped_dangling_relations_total);
    eprintln!("  total decoder generate   = {} s", fmt_s(total_gen));
    eprintln!(
        "  avg generate per chunk   = {} s",
        fmt_s(total_gen / total_chunks_proc as u32),
    );
    eprintln!("  embed_chunks (batch)     = {} ms", fmt_ms(report.embed_chunks));
    eprintln!("  embed_entities (batch)   = {} ms", fmt_ms(report.embed_entities));
    eprintln!("  embed_relations (batch)  = {} ms", fmt_ms(report.embed_relations));
    eprintln!("  merge across chunks      = {} ms", fmt_ms(report.merge));
    eprintln!("  drop dangling relations  = {} ms", fmt_ms(report.drop_dangling));
    eprintln!("  assemble ExtractedKg     = {} ms", fmt_ms(report.assemble));
    eprintln!("  extract_kg total (lib)   = {} s", fmt_s(report.total));
    eprintln!("  extract wall (incl. all) = {} s", fmt_s(wall));
}

fn print_query_table(rows: &[QueryRow]) {
    eprintln!();
    eprintln!("--- per-query hybrid stage timings (ms) ---");
    eprintln!(
        "{:>2} {:<30} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>8} {:>4} {:>4} {:>4} {:>6}",
        "#", "query",
        "embed", "perturb", "ents", "rels", "adj", "src", "decrypt", "total", "ne", "nr", "nc", "l0_rd"
    );
    for (i, r) in rows.iter().enumerate() {
        eprintln!(
            "{:>2} {:<30} {} {} {} {} {} {} {} {} {:>4} {:>4} {:>4} {:>6}",
            i,
            r.label.chars().take(30).collect::<String>(),
            fmt_ms(r.embed),
            fmt_ms(r.timings.perturb),
            fmt_ms(r.timings.entities_search),
            fmt_ms(r.timings.relations_search),
            fmt_ms(r.timings.adjacency),
            fmt_ms(r.timings.src_chunks),
            fmt_ms(r.timings.chunk_decrypt),
            fmt_ms(r.timings.total),
            r.timings.n_entities_returned,
            r.timings.n_relations_returned,
            r.timings.n_chunks_returned,
            r.timings.layer0_reads_delta,
        );
    }
    if rows.is_empty() {
        return;
    }
    let mean = |sel: fn(&StageTimings) -> Duration| -> Duration {
        rows.iter().map(|r| sel(&r.timings)).sum::<Duration>() / rows.len() as u32
    };
    let mean_embed: Duration = rows.iter().map(|r| r.embed).sum::<Duration>() / rows.len() as u32;
    eprintln!();
    eprintln!("--- mean across {} queries ---", rows.len());
    eprintln!("  embed (LL+HL)        = {} ms", fmt_ms(mean_embed));
    eprintln!("  perturb              = {} ms", fmt_ms(mean(|t| t.perturb)));
    eprintln!("  entities_search      = {} ms", fmt_ms(mean(|t| t.entities_search)));
    eprintln!("  relations_search     = {} ms", fmt_ms(mean(|t| t.relations_search)));
    eprintln!("  adjacency            = {} ms", fmt_ms(mean(|t| t.adjacency)));
    eprintln!("  src_chunks           = {} ms", fmt_ms(mean(|t| t.src_chunks)));
    eprintln!("  chunk_decrypt        = {} ms", fmt_ms(mean(|t| t.chunk_decrypt)));
    eprintln!("  total                = {} ms", fmt_ms(mean(|t| t.total)));
}

// ─────────────────────────────────────────────────────────────────────
// Model loading via HF cache
// ─────────────────────────────────────────────────────────────────────

fn load_qwen3_4b(engine: RunnerEngine) -> Result<DecoderRuntime<RunnerEngine>> {
    let api = ApiBuilder::new()
        .with_progress(false)
        .build()
        .context("hf-hub API")?;
    let repo = api.model("Qwen/Qwen3-4B".to_string());
    let tokenizer_path = repo
        .get("tokenizer.json")
        .context("downloading tokenizer.json for Qwen3-4B")?;
    let snapshot_dir = tokenizer_path
        .parent()
        .ok_or_else(|| anyhow!("tokenizer path has no parent"))?
        .to_path_buf();
    pull_shards(&repo)?;
    // The HF snapshot may not include `config.json` — pin the
    // variant config directly. Qwen3-4B per the extraction plan.
    let cfg = Qwen3Variant::Q4B.config();
    DecoderRuntime::<RunnerEngine>::from_config_and_dir(cfg, &snapshot_dir, engine)
}

fn load_qwen3_embedding_06b(engine: RunnerEngine) -> Result<GeloDescriptionEmbedder<RunnerEngine>> {
    let api = ApiBuilder::new()
        .with_progress(false)
        .build()
        .context("hf-hub API")?;
    let repo = api.model("Qwen/Qwen3-Embedding-0.6B".to_string());
    let _config_path = repo
        .get("config.json")
        .context("downloading config.json for Qwen3-Embedding-0.6B")?;
    let tokenizer_path = repo
        .get("tokenizer.json")
        .context("downloading tokenizer.json for Qwen3-Embedding-0.6B")?;
    let snapshot_dir = tokenizer_path
        .parent()
        .ok_or_else(|| anyhow!("tokenizer path has no parent"))?
        .to_path_buf();
    pull_shards(&repo)?;
    GeloDescriptionEmbedder::<RunnerEngine>::from_dir(&snapshot_dir, engine)
}

/// Ensure every safetensors shard listed in the repo is materialised
/// in the local HF cache.
fn pull_shards(repo: &ApiRepo) -> Result<()> {
    // Single-file layout?
    if let Ok(_p) = repo.get("model.safetensors") {
        return Ok(());
    }
    let index_path = repo
        .get("model.safetensors.index.json")
        .context("model has neither model.safetensors nor index.json")?;
    let index_bytes = std::fs::read(&index_path)?;
    let index: serde_json::Value = serde_json::from_slice(&index_bytes)?;
    let map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow!("shard index has no weight_map"))?;
    let mut filenames: Vec<String> = map
        .values()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    filenames.sort();
    filenames.dedup();
    for name in &filenames {
        repo.get(name)
            .with_context(|| format!("downloading shard {name}"))?;
    }
    Ok(())
}


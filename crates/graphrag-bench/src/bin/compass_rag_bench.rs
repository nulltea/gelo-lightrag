//! `compass-rag-bench` — drives the Compass GraphRAG stack end-to-end
//! and emits per-query stage timings + summary stats.
//!
//! M-first cut: `in_memory` scenario only. `local_rest` and
//! `runner_http` scenarios will share `graphrag_bench::synth` +
//! `graphrag_bench::stages` and land in follow-ons.
//!
//! Configuration via env vars:
//!
//!   BENCH_SIZE     — entity count (default 100). Chunks scale ×1.5,
//!                    relations ×0.33.
//!   BENCH_DIM      — embedding dim (default 64).
//!   BENCH_QUERIES  — queries per (mode) (default 50).
//!   BENCH_MODE     — "local", "hybrid", or "both" (default "both").
//!   BENCH_TOP_K_E  — top_k_entities (default 5).
//!   BENCH_TOP_K_R  — top_k_relations (default 5; hybrid only).
//!   BENCH_TOP_K_C  — top_k_chunks_per_entity (default 2).
//!   BENCH_WARMUP   — warmup queries dropped from stats (default 5).
//!
//! Output: CSV rows to stdout (one per query); human-readable
//! summary tables to stderr.
//!
//! Run: `cargo run --release -p graphrag-bench --bin compass-rag-bench`

use std::env;
use std::time::Instant;

use anyhow::{Context, Result};
use graphrag_bench::stages::{timed_kg_query, Mode, StageTimings};
use graphrag_bench::summary::PerStagePcts;
use graphrag_bench::synth::{build_kg, random_unit_vec, SynthConfig};
use light_kg_store::{
    CompassIndexParams, LightKgParams, LightKgStore, PlainHnswParams, RingOramParams, XorMmParams,
};
use lightrag_private::SessionKey;
use rag_core::TenantId;
use rag_core::keying::HkdfPolicyV2;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use zeroize::Zeroizing;

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_str(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn default_compass_params(dim: usize, n_corpus: usize) -> CompassIndexParams {
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

fn default_xormm_params(n_keys: usize) -> XorMmParams {
    // Volume bound: log-ish in entity count. Each entity holds at
    // most a handful of relations / source chunks at synth scale.
    let volume_bound = 16u32;
    // Buckets: ~2.1× key count to keep cuckoo placement converging
    // at the paper's max_kicks default. Round up to next power of two.
    let n_buckets = ((n_keys as f32 * 2.1).ceil() as u32).max(64).next_power_of_two();
    XorMmParams {
        volume_bound,
        value_bytes: 64,
        n_buckets,
        max_kicks: 256,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let n_entities = env_usize("BENCH_SIZE", 100);
    let dim = env_usize("BENCH_DIM", 64);
    let n_queries = env_usize("BENCH_QUERIES", 50);
    let mode_str = env_str("BENCH_MODE", "both");
    let top_k_e = env_usize("BENCH_TOP_K_E", 5);
    let top_k_r = env_usize("BENCH_TOP_K_R", 5);
    let top_k_c = env_usize("BENCH_TOP_K_C", 2);
    let n_warmup = env_usize("BENCH_WARMUP", 5);

    let cfg = SynthConfig::new(n_entities, dim);
    eprintln!(
        "graphrag-bench in_memory scenario: \
         {n_entities} entities · {n_chunks} chunks · {n_relations} relations · \
         dim={dim} · queries={n_queries} (×modes) · warmup={n_warmup}",
        n_chunks = cfg.n_chunks(),
        n_relations = cfg.n_relations()
    );

    let kg = build_kg(&cfg);

    // V2 HKDF — same shape as the production tenant boot.
    let user_x_sk = Zeroizing::new([0xa1u8; 32]);
    let tee_user_x_sk = Zeroizing::new([0xb2u8; 32]);
    let tenant = TenantId::new("bench-tenant");
    let derived = HkdfPolicyV2::V2.derive(&user_x_sk, &tee_user_x_sk, &tenant);

    let xormm = default_xormm_params(n_entities);
    let params = LightKgParams {
        entities: default_compass_params(dim, kg.entities.len()),
        relations: default_compass_params(dim, kg.relations.len().max(8)),
        chunks: default_compass_params(dim, kg.chunks.len()),
        adjacency: xormm,
        src_chunks: xormm,
    };

    let ingest_start = Instant::now();
    let mut store = LightKgStore::build_from_kg(kg, params, &derived)
        .await
        .context("LightKgStore::build_from_kg")?;
    let ingest_dur = ingest_start.elapsed();
    eprintln!(
        "ingest (build_from_kg): {:.3} s",
        ingest_dur.as_secs_f64()
    );

    let session_key = SessionKey::derive(&derived.search_pattern_key, b"bench-session-001");

    // Query embeddings — fresh seed so they don't trivially hit
    // their own entity (we want a non-trivial corpus walk).
    let mut q_rng = ChaCha20Rng::from_seed([0xc3u8; 32]);

    // CSV header
    println!(
        "scenario,size,mode,query_idx,total_us,perturb_us,entities_search_us,\
         relations_search_us,adjacency_us,src_chunks_us,chunk_decrypt_us,\
         layer0_reads,n_entities,n_relations,n_chunks"
    );

    let modes: Vec<Mode> = match mode_str.as_str() {
        "local" => vec![Mode::Local],
        "hybrid" => vec![Mode::Hybrid],
        _ => vec![Mode::Local, Mode::Hybrid],
    };

    for mode in modes {
        let total_calls = n_warmup + n_queries;
        let mut samples: Vec<StageTimings> = Vec::with_capacity(n_queries);
        for i in 0..total_calls {
            let ll = random_unit_vec(&mut q_rng, dim);
            let hl = random_unit_vec(&mut q_rng, dim);
            let timings = timed_kg_query(
                &mut store,
                &ll,
                &hl,
                mode,
                top_k_e,
                top_k_r,
                top_k_c,
                &session_key,
            )
            .await
            .context("timed_kg_query")?;
            // Drop warmup samples.
            let is_warmup = i < n_warmup;
            if !is_warmup {
                println!(
                    "in_memory,{n_entities},{mode},{q},{total:.1},{perturb:.1},{es:.1},\
                     {rs:.1},{adj:.1},{src:.1},{cd:.1},{l0},{ne},{nr},{nc}",
                    mode = mode.as_str(),
                    q = i - n_warmup,
                    total = timings.total.as_secs_f64() * 1e6,
                    perturb = timings.perturb.as_secs_f64() * 1e6,
                    es = timings.entities_search.as_secs_f64() * 1e6,
                    rs = timings.relations_search.as_secs_f64() * 1e6,
                    adj = timings.adjacency.as_secs_f64() * 1e6,
                    src = timings.src_chunks.as_secs_f64() * 1e6,
                    cd = timings.chunk_decrypt.as_secs_f64() * 1e6,
                    l0 = timings.layer0_reads_delta,
                    ne = timings.n_entities_returned,
                    nr = timings.n_relations_returned,
                    nc = timings.n_chunks_returned
                );
                samples.push(timings);
            }
        }
        let pcts = PerStagePcts::from(&samples);
        pcts.print_human("in_memory", n_entities, mode.as_str(), n_queries);
    }

    Ok(())
}

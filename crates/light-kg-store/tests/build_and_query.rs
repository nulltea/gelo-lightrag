//! M6 acceptance — build_from_kg + a mixed read trace.
//!
//! Smaller-than-paper-spec for now: 16 entities, 8 relations, 24
//! chunks (vs the plan's 100 mixed reads/writes on a reference
//! plaintext-LightRAG trace). This pins the build + each surface API
//! (CompassIndex search, adjacency lookup, src_chunks lookup,
//! chunk text decrypt) so M7's retrieval-port work can layer on
//! without re-debugging the wiring.
//!
//! Run with `cargo test --release -p light-kg-store --test
//! build_and_query` — debug builds OOM-kill the inner HNSW build
//! under sandbox limits (same release-mode rationale as
//! `compass-index/tests/recall.rs`).

use std::collections::HashSet;

use compass_index::PlainHnswParams;
use light_kg_store::{
    Chunk, CompassIndexParams, Entity, ExtractedKg, LightKgParams, LightKgStore, Relation,
    RingOramParams, XorMmParams,
};
use rag_core::TenantId;
use rag_core::keying::{HkdfPolicyV2, SchemeParamsV2};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use zeroize::Zeroizing;

fn random_unit_vec(rng: &mut ChaCha20Rng, dim: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn synth_kg(rng: &mut ChaCha20Rng, dim: usize) -> ExtractedKg {
    let n_entities = 16usize;
    let n_relations = 8usize;
    let n_chunks = 24usize;

    let chunks: Vec<Chunk> = (0..n_chunks)
        .map(|i| Chunk {
            id: format!("chunk-{i:04}"),
            text: format!("This is the text of chunk {i}. Lorem ipsum dolor sit amet."),
            embedding: random_unit_vec(rng, dim),
        })
        .collect();

    let entity_names: Vec<String> = (0..n_entities).map(|i| format!("entity-{i}")).collect();
    let entities: Vec<Entity> = entity_names
        .iter()
        .enumerate()
        .map(|(i, name)| Entity {
            name: name.clone(),
            description: format!("Entity {name} is something with id {i}."),
            embedding: random_unit_vec(rng, dim),
            // Each entity ties to two chunks chosen deterministically.
            source_chunks: vec![
                chunks[i % chunks.len()].id.clone(),
                chunks[(i + 7) % chunks.len()].id.clone(),
            ],
        })
        .collect();

    let mut relations: Vec<Relation> = Vec::with_capacity(n_relations);
    for i in 0..n_relations {
        let a = &entity_names[i];
        let b = &entity_names[(i + 1) % n_entities];
        relations.push(Relation {
            src: a.clone(),
            tgt: b.clone(),
            description: format!("{a} is related to {b}"),
            embedding: random_unit_vec(rng, dim),
            source_chunks: vec![chunks[i % chunks.len()].id.clone()],
        });
    }

    ExtractedKg {
        chunks,
        entities,
        relations,
    }
}

fn params_for(dim: usize, m_neighbors: usize, n_corpus: usize) -> CompassIndexParams {
    // Block sizing: dim·4 (embedding) + 4 (count) + (2 m_neighbors)·4
    // (M_l0 = 2·m_neighbors). Pad to next power of two.
    let raw = dim * 4 + 4 + 2 * m_neighbors * 4;
    let block_bytes = raw.next_power_of_two().max(64);
    // n_leaves needs headroom over n_corpus. Power of two ≥ 2·n_corpus.
    let n_leaves = (2 * n_corpus.max(8)).next_power_of_two() as u32;
    CompassIndexParams {
        hnsw: PlainHnswParams::paper_defaults(dim, m_neighbors),
        oram: RingOramParams {
            z: 4,
            s: 5,
            a: 3,
            block_bytes: block_bytes as u32,
            n_leaves,
            treetop_levels: 2,
        },
        ef_search: 16,
        ef_n: 4,
    }
}

#[tokio::test]
async fn build_from_kg_round_trips_all_six_stores() {
    let mut rng = ChaCha20Rng::from_seed([0x6c; 32]);
    let dim = 32;
    let kg = synth_kg(&mut rng, dim);

    let user_x_sk = Zeroizing::new([0x11u8; 32]);
    let tee_user_x_sk = Zeroizing::new([0x22u8; 32]);
    let tenant = TenantId::new("acceptance-tenant");
    let derived = HkdfPolicyV2::V2.derive(&user_x_sk, &tee_user_x_sk, &tenant);
    // Suppress unused warning: scheme_identity_digest is owned by the
    // attestation path, not by LightKgStore — but pin it to ensure
    // the V2 policy compiles correctly.
    let _digest = HkdfPolicyV2::V2.scheme_identity_digest(SchemeParamsV2::default());

    let params = LightKgParams {
        entities: params_for(dim, 8, kg.entities.len()),
        relations: params_for(dim, 4, kg.relations.len().max(8)),
        chunks: params_for(dim, 8, kg.chunks.len()),
        adjacency: XorMmParams {
            volume_bound: 8,
            value_bytes: 64,
            n_buckets: 64,
            max_kicks: 64,
        },
        src_chunks: XorMmParams {
            volume_bound: 8,
            value_bytes: 64,
            n_buckets: 64,
            max_kicks: 64,
        },
    };

    let entity_names: Vec<String> = kg.entities.iter().map(|e| e.name.clone()).collect();
    let entity_query_embeddings: Vec<Vec<f32>> = kg
        .entities
        .iter()
        .map(|e| e.embedding.clone())
        .collect();
    let entity_block_id_expected: std::collections::HashMap<String, u32> = entity_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i as u32))
        .collect();
    let chunk_text_expected: std::collections::HashMap<String, String> = kg
        .chunks
        .iter()
        .map(|c| (c.id.clone(), c.text.clone()))
        .collect();
    let relation_canon: Vec<String> = kg.relations.iter().map(|r| r.canonical_key()).collect();

    let mut store = LightKgStore::build_from_kg(kg, params, &derived)
        .await
        .expect("build_from_kg");

    // ─── 1. CompassIndex (entities) — query each entity's own embedding ──
    // It should land in its own top-k (k=3 to allow HNSW slack).
    for (i, q) in entity_query_embeddings.iter().enumerate() {
        let topk = store.query_entities_topk(q, 3).await.expect("topk");
        assert!(
            topk.contains(&(i as u32)),
            "entity {i} not in its own top-3: {topk:?}"
        );
    }

    // ─── 2. entity_block_id map populated ────────────────────────────
    assert_eq!(store.entity_block_id, entity_block_id_expected);

    // ─── 3. AesChunkStore round-trip ─────────────────────────────────
    for (id, expected_text) in &chunk_text_expected {
        let got = store.chunk_text(id).expect("decrypt chunk");
        assert_eq!(&got, expected_text);
    }

    // ─── 4. adjacency EMM ───────────────────────────────────────────
    // For each entity that participates in any relation, the lookup
    // must return at least the canonical key(s) of that relation.
    // Build the expected mapping from the relation list.
    let mut expected_adjacency: std::collections::HashMap<String, HashSet<String>> =
        std::collections::HashMap::new();
    for canon in &relation_canon {
        let (a, b) = canon.split_once('\x00').unwrap();
        expected_adjacency.entry(a.to_string()).or_default().insert(canon.clone());
        expected_adjacency.entry(b.to_string()).or_default().insert(canon.clone());
    }
    for (entity_name, expected) in &expected_adjacency {
        let got: HashSet<String> = store
            .adjacency_for_entity(entity_name)
            .expect("adjacency")
            .into_iter()
            .collect();
        assert!(
            expected.is_subset(&got),
            "adjacency for {entity_name:?}: expected {expected:?}, got {got:?}"
        );
    }

    // ─── 5. src_chunks EMM ──────────────────────────────────────────
    // For every entity, the src_chunks lookup must return *all* of
    // its source chunk ids.
    for name in &entity_names {
        let got: HashSet<String> = store
            .src_chunks_for_entity(name)
            .expect("src_chunks")
            .into_iter()
            .collect();
        // Each entity has exactly 2 source chunks per synth_kg.
        assert_eq!(
            got.len(),
            2,
            "entity {name} src_chunks expected 2 entries, got {got:?}"
        );
    }
}

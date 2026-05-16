//! M7.1 acceptance — drive `kg_query` over a real `LightKgStore`.
//!
//! Builds a small synthetic KG (16 entities, 8 relations, 24
//! chunks), builds the encrypted store, then runs a Local-mode
//! query and verifies the orchestrator threads the embedding through
//! search_perturb → entities search → adjacency → src_chunks →
//! chunk_text → assembled context string.
//!
//! What this test does NOT pin (deferred to M7.x):
//! - Bit-for-bit `_build_context_str` format parity with upstream
//! - `_token_truncation`
//! - `_merge_all_chunks` (M7.1 uses simple dedup)
//! - Hybrid / Global / Mix / Naive modes
//!
//! Run with `cargo test --release -p lightrag-private --test
//! local_kg_query` — debug builds OOM-kill the HNSW build.

use light_kg_store::{
    Chunk, CompassIndexParams, Entity, ExtractedKg, LightKgParams, LightKgStore, PlainHnswParams,
    Relation, RingOramParams, XorMmParams,
};
use lightrag_private::{KgQueryParams, LightRagPrivateService, QueryShape, SessionKey};
use rag_core::TenantId;
use rag_core::keying::HkdfPolicyV2;
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
            text: format!("Body of chunk {i}: lorem ipsum dolor sit amet."),
            embedding: random_unit_vec(rng, dim),
        })
        .collect();

    let names: Vec<String> = (0..n_entities).map(|i| format!("entity-{i}")).collect();
    let entities: Vec<Entity> = names
        .iter()
        .enumerate()
        .map(|(i, name)| Entity {
            name: name.clone(),
            description: format!("Entity {name} is something with id {i}."),
            embedding: random_unit_vec(rng, dim),
            source_chunks: vec![
                chunks[i % chunks.len()].id.clone(),
                chunks[(i + 7) % chunks.len()].id.clone(),
            ],
        })
        .collect();

    let mut relations = Vec::with_capacity(n_relations);
    for i in 0..n_relations {
        let a = &names[i];
        let b = &names[(i + 1) % n_entities];
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
            treetop_levels: 2,
        },
        ef_search: 16,
        ef_n: 4,
    }
}

#[tokio::test]
async fn local_kg_query_threads_all_stages() {
    let mut rng = ChaCha20Rng::from_seed([0x7a; 32]);
    let dim = 32;
    let kg = synth_kg(&mut rng, dim);
    let entity_embeddings: Vec<Vec<f32>> = kg.entities.iter().map(|e| e.embedding.clone()).collect();
    let target_entity = "entity-3".to_string();
    let target_idx = kg
        .entities
        .iter()
        .position(|e| e.name == target_entity)
        .unwrap();
    let target_embedding = entity_embeddings[target_idx].clone();

    let user_x_sk = Zeroizing::new([0xa1u8; 32]);
    let tee_user_x_sk = Zeroizing::new([0xb2u8; 32]);
    let tenant = TenantId::new("m7-tenant");
    let derived = HkdfPolicyV2::V2.derive(&user_x_sk, &tee_user_x_sk, &tenant);

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

    let mut store = LightKgStore::build_from_kg(kg, params, &derived)
        .await
        .expect("build store");

    // Session secret + nonce → derived session key for perturbation.
    let session_key = SessionKey::derive(&derived.search_pattern_key, b"session-001");

    let query_params = KgQueryParams {
        top_k_entities: 3,
        top_k_chunks_per_entity: 2,
        shape: QueryShape::Local,
        top_k_relations: 3,
    };
    let mut svc = LightRagPrivateService::new(&mut store);
    let ctx = svc
        .kg_query(&target_embedding, &[], &query_params, &session_key)
        .await
        .expect("kg_query");

    // The target entity must appear in the result (search_perturb
    // doesn't break its own search at ε = 2 %).
    assert!(
        ctx.entities.contains(&target_entity),
        "target {target_entity:?} missing from result: {:?}",
        ctx.entities
    );
    // Chunks must non-empty (fan-out worked) and decrypt successfully
    // (each (id, text) pair is non-empty).
    assert!(!ctx.chunks.is_empty(), "no chunks returned");
    for (id, text) in &ctx.chunks {
        assert!(!id.is_empty());
        assert!(text.contains("lorem ipsum"), "chunk {id} text malformed");
    }

    // Sanity: the rendered context string is valid UTF-8 with the
    // expected section headers.
    let s = ctx.to_context_string();
    assert!(s.contains("# Entities"));
    assert!(s.contains("# Relations"));
    assert!(s.contains("# Source chunks"));
}

#[tokio::test]
async fn cross_session_traces_diverge() {
    // Plan §M7 acceptance: same query under two different
    // `session_nonce`s should produce different *retrieval traces*.
    // The strongest version is the round-count divergence over 10²
    // trials (M7.x); this smaller test pins the dual: perturb
    // produces distinct embeddings, so even a deterministic search
    // sees different inputs.
    use lightrag_private::{perturb, EmbeddingKind};

    let s_search = Zeroizing::new([0x42u8; 32]);
    let sk_a = SessionKey::derive(&s_search, b"session-A");
    let sk_b = SessionKey::derive(&s_search, b"session-B");

    let mut rng = ChaCha20Rng::from_seed([0xff; 32]);
    let mut divergent = 0;
    for _ in 0..100 {
        let e = random_unit_vec(&mut rng, 64);
        let pa = perturb(&sk_a, EmbeddingKind::Ll, &e);
        let pb = perturb(&sk_b, EmbeddingKind::Ll, &e);
        if pa != pb {
            divergent += 1;
        }
    }
    // 100/100 should differ (HMAC tag collision probability is 2^-256
    // ish). Use ≥ 99/100 to leave room for any future test-only
    // small-dim fixture.
    assert!(divergent >= 99, "divergence too low: {divergent}/100");
}

//! Stage-timed kg_query orchestration. Mirrors
//! [`light_kg_store::LightKgStore::query_context`] step-for-step so we
//! can attribute latency to perturb / search / adjacency / chunks /
//! decrypt.
//!
//! Re-implementing the orchestration here (instead of wrapping the
//! production `query_context`) lets us put `Instant::now()` boundaries
//! between each stage without polluting prod code with timing hooks.
//! Any change to the canonical fan-out **must** be mirrored here or the
//! perf attribution stops matching production behaviour.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use light_kg_store::{BlockBackend, ByteStoreBackend, LightKgError, LightKgStore};
use lightrag_private::{EmbeddingKind, SessionKey, perturb};

#[derive(Debug, Clone, Copy)]
pub enum Mode {
    Local,
    Hybrid,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Local => "local",
            Mode::Hybrid => "hybrid",
        }
    }
}

#[derive(Debug, Clone)]
pub struct StageTimings {
    pub perturb: Duration,
    pub entities_search: Duration,
    pub relations_search: Duration,
    pub adjacency: Duration,
    pub src_chunks: Duration,
    pub chunk_decrypt: Duration,
    pub total: Duration,
    pub layer0_reads_delta: u64,
    pub n_entities_returned: usize,
    pub n_relations_returned: usize,
    pub n_chunks_returned: usize,
}

/// Run one query through the full pipeline with per-stage timings.
/// Replicates `LightRagPrivateService::kg_query` semantics; result is
/// not returned (the bench only needs the timings + counts).
pub async fn timed_kg_query<B: BlockBackend, BS: ByteStoreBackend>(
    store: &mut LightKgStore<B, BS>,
    ll_query: &[f32],
    hl_query: &[f32],
    mode: Mode,
    top_k_entities: usize,
    top_k_relations: usize,
    top_k_chunks_per_entity: usize,
    session_key: &SessionKey,
) -> Result<StageTimings, LightKgError> {
    let t_start = Instant::now();
    let reads_before = store.entities.layer0_read_count()
        + store.relations.layer0_read_count()
        + store.chunks.layer0_read_count();

    // 1. perturb (ll)
    let t = Instant::now();
    let ll_perturbed = perturb(session_key, EmbeddingKind::Ll, ll_query);
    let mut perturb_dur = t.elapsed();

    // 2. entities search
    let t = Instant::now();
    let entity_block_ids = store.query_entities_topk(&ll_perturbed, top_k_entities).await?;
    let entities_search_dur = t.elapsed();

    let entity_id_to_name: HashMap<u32, String> = store
        .entity_block_id
        .iter()
        .map(|(name, id)| (*id, name.clone()))
        .collect();
    let mut entity_names: Vec<String> = entity_block_ids
        .iter()
        .filter_map(|id| entity_id_to_name.get(id).cloned())
        .collect();

    // 3. (hybrid) perturb hl + relations search + endpoint fan-out
    let mut relations_search_dur = Duration::ZERO;
    let mut relation_keys: Vec<String> = Vec::new();
    if matches!(mode, Mode::Hybrid) {
        let t = Instant::now();
        let hl_perturbed = perturb(session_key, EmbeddingKind::Hl, hl_query);
        perturb_dur += t.elapsed();

        let t = Instant::now();
        let relation_block_ids = store.query_relations_topk(&hl_perturbed, top_k_relations).await?;
        relations_search_dur = t.elapsed();

        let relation_id_to_canon: HashMap<u32, String> = store
            .relation_block_id
            .iter()
            .map(|(canon, id)| (*id, canon.clone()))
            .collect();
        for id in &relation_block_ids {
            if let Some(canon) = relation_id_to_canon.get(id) {
                relation_keys.push(canon.clone());
            }
            if let Some((src, tgt)) = store.relation_endpoints.get(id) {
                entity_names.push(src.clone());
                entity_names.push(tgt.clone());
            }
        }
    }

    let mut seen_e = HashSet::new();
    entity_names.retain(|n| seen_e.insert(n.clone()));

    // 4. adjacency
    let t = Instant::now();
    let mut all_relations: Vec<String> = relation_keys;
    for name in &entity_names {
        let mut rels = store.adjacency_for_entity(name)?;
        all_relations.append(&mut rels);
    }
    let mut seen_r = HashSet::new();
    all_relations.retain(|r| seen_r.insert(r.clone()));
    let adjacency_dur = t.elapsed();

    // 5. src_chunks
    let t = Instant::now();
    let mut chunk_ids: Vec<String> = Vec::new();
    for name in &entity_names {
        let mut chunks = store.src_chunks_for_entity(name)?;
        chunks.truncate(top_k_chunks_per_entity);
        chunk_ids.append(&mut chunks);
    }
    let mut seen_c = HashSet::new();
    chunk_ids.retain(|id| seen_c.insert(id.clone()));
    let src_chunks_dur = t.elapsed();

    // 6. chunk decrypt
    let t = Instant::now();
    let mut chunks_out = Vec::with_capacity(chunk_ids.len());
    for id in &chunk_ids {
        let text = store.chunk_text(id)?;
        chunks_out.push((id.clone(), text));
    }
    let chunk_decrypt_dur = t.elapsed();

    let total = t_start.elapsed();
    let reads_after = store.entities.layer0_read_count()
        + store.relations.layer0_read_count()
        + store.chunks.layer0_read_count();

    Ok(StageTimings {
        perturb: perturb_dur,
        entities_search: entities_search_dur,
        relations_search: relations_search_dur,
        adjacency: adjacency_dur,
        src_chunks: src_chunks_dur,
        chunk_decrypt: chunk_decrypt_dur,
        total,
        layer0_reads_delta: reads_after - reads_before,
        n_entities_returned: entity_names.len(),
        n_relations_returned: all_relations.len(),
        n_chunks_returned: chunks_out.len(),
    })
}

//! `LightKgStore` — composition of the six encrypted components that
//! back the LightRAG retrieval surface.
//!
//! ```text
//!   ┌─ entities  ── CompassIndex<B>       under oram_entities_key
//!   ├─ relations ── CompassIndex<B>       under oram_relations_key
//!   ├─ chunks    ── CompassIndex<B>       under oram_chunks_key
//!   ├─ adjacency ── XorMmClient<BS>       under emm_adjacency_key
//!   ├─ src_chunks── XorMmClient<BS>       under emm_src_chunks_key
//!   └─ chunk_text── AesChunkStore         under aes_chunk_key
//! ```
//!
//! The 7th HKDF v2 child (`search_pattern_key`) is consumed by
//! `lightrag-private::search_perturb` (M7). The 8th (`caprise_seed`)
//! is consumed by the embedder path inside `gelo-rag`.
//!
//! M6 ships:
//! - `build_from_kg(ExtractedKg, …)` constructor that drives the
//!   three CompassIndex builds and the two EMM builds.
//! - Per-component lookup helpers exposed for the M7 retrieval port
//!   (`query_entities_topk`, `adjacency_for_entity`,
//!   `src_chunks_for_entity`, `chunk_text`).

use std::collections::HashMap;

use compass_index::{CompassIndex, CompassIndexParams};
use rag_core::keying::DerivedKeysV2;
use ring_oram::{BackendError, BlockBackend, InMemoryBlockBackend};
use thiserror::Error;
use xormm_emm::{ByteStoreBackend, InMemoryByteStore, LogicalKey, LogicalValue, XorMmClient, XorMmError, XorMmParams};
use zeroize::Zeroizing;

use crate::aes_chunk_store::{AesChunkStore, ChunkStoreError};
use crate::keys::{derive_logical_key, label};
use crate::types::ExtractedKg;

#[derive(Debug, Error)]
pub enum LightKgError {
    #[error("compass-index error: {0}")]
    Compass(#[from] compass_index::CompassIndexError),
    #[error("xormm error: {0}")]
    XorMm(#[from] XorMmError),
    #[error("chunk store error: {0}")]
    Chunk(#[from] ChunkStoreError),
    #[error("backend error: {0}")]
    Backend(#[from] BackendError),
    #[error("inconsistent kg: {0}")]
    Inconsistent(String),
    #[error("entity name {0:?} unknown in this store")]
    UnknownEntity(String),
}

/// One-stop tuning surface for a `LightKgStore` build. The three
/// CompassIndex sizes can differ — typical LightRAG corpora have
/// many more chunks than relations, and many more entities than
/// relations.
#[derive(Debug, Clone, Copy)]
pub struct LightKgParams {
    pub entities: CompassIndexParams,
    pub relations: CompassIndexParams,
    pub chunks: CompassIndexParams,
    pub adjacency: XorMmParams,
    pub src_chunks: XorMmParams,
}

/// Composition of the six encrypted components plus the in-CVM
/// cleartext id maps used to translate plaintext names ↔ Compass
/// block ids. The id maps leak to a CVM-internal adversary in the
/// same way the upstream LightRAG entity-name maps would; design-
/// doc Risk F (entity-ID pseudonymisation) addresses this in a
/// future hardening pass.
pub struct LightKgStore<B: BlockBackend, BS: ByteStoreBackend> {
    pub entities: CompassIndex<B>,
    pub relations: CompassIndex<B>,
    pub chunks: CompassIndex<B>,
    pub adjacency: XorMmClient<BS>,
    pub src_chunks: XorMmClient<BS>,
    pub chunk_text: AesChunkStore,
    /// `entity_name → entities[].block_id`. Built at construction;
    /// queries look an entity name up here before dispatching to the
    /// CompassIndex / EMM.
    pub entity_block_id: HashMap<String, u32>,
    /// `relation.canonical_key() → relations[].block_id`.
    pub relation_block_id: HashMap<String, u32>,
    /// `relation_block_id → (src_name, tgt_name)`. Inverse-direction
    /// of `relation_block_id`. Cleartext inside the CVM; pseudonymising
    /// is M9 hardening (Risk F). Used by Hybrid-mode `kg_query` to
    /// fan from a relation hit out to its endpoint entities.
    pub relation_endpoints: HashMap<u32, (String, String)>,
    /// `chunk_id → chunks[].block_id`.
    pub chunk_block_id: HashMap<String, u32>,
    /// Master keys, kept in-RAM for sub-key derivation. Wiped on drop
    /// via [`Zeroizing`].
    pub keys: KeyBundle,
}

/// Subset of `DerivedKeysV2` that `LightKgStore` keeps after the build
/// phase: every key consumed by a per-request lookup. The construction
/// path also touches `caprise_seed` and `search_pattern_key`, but
/// those are owned by the embedder / retrieval service respectively,
/// not by the store.
pub struct KeyBundle {
    pub aes_chunk: Zeroizing<[u8; 32]>,
    pub oram_entities: Zeroizing<[u8; 32]>,
    pub oram_relations: Zeroizing<[u8; 32]>,
    pub oram_chunks: Zeroizing<[u8; 32]>,
    pub emm_adjacency: Zeroizing<[u8; 32]>,
    pub emm_src_chunks: Zeroizing<[u8; 32]>,
}

impl KeyBundle {
    pub fn from_derived(d: &DerivedKeysV2) -> Self {
        Self {
            aes_chunk: clone_zeroized(&d.aes_chunk_key),
            oram_entities: clone_zeroized(&d.oram_entities_key),
            oram_relations: clone_zeroized(&d.oram_relations_key),
            oram_chunks: clone_zeroized(&d.oram_chunks_key),
            emm_adjacency: clone_zeroized(&d.emm_adjacency_key),
            emm_src_chunks: clone_zeroized(&d.emm_src_chunks_key),
        }
    }
}

fn clone_zeroized(src: &Zeroizing<[u8; 32]>) -> Zeroizing<[u8; 32]> {
    let mut out = Zeroizing::new([0u8; 32]);
    out.copy_from_slice(src.as_ref());
    out
}

impl LightKgStore<InMemoryBlockBackend, InMemoryByteStore> {
    /// M6 driver — build all six stores over the in-memory backends.
    /// For a network-backed deployment, swap to `build_from_kg_on`
    /// (M5.0+M5.1 land the async REST backend).
    pub async fn build_from_kg(
        kg: ExtractedKg,
        params: LightKgParams,
        keys: &DerivedKeysV2,
    ) -> Result<Self, LightKgError> {
        build_inner::<InMemoryBlockBackend, InMemoryByteStore, _, _>(
            kg,
            params,
            keys,
            |p| InMemoryBlockBackend::new(p.oram.num_buckets()),
            |p| InMemoryByteStore::new(p.n_buckets),
        )
        .await
    }
}

impl<B: BlockBackend, BS: ByteStoreBackend> LightKgStore<B, BS> {
    /// Generic builder used when the caller wants to swap in
    /// REST-shaped backends. `mk_block_backend` and `mk_byte_store`
    /// are factories — one Ring-ORAM tree per CompassIndex (×3), one
    /// XorMm byte store per EMM (×2).
    pub async fn build_from_kg_on<MB, MS>(
        kg: ExtractedKg,
        params: LightKgParams,
        keys: &DerivedKeysV2,
        mk_block_backend: MB,
        mk_byte_store: MS,
    ) -> Result<Self, LightKgError>
    where
        MB: Fn(&CompassIndexParams) -> B,
        MS: Fn(&XorMmParams) -> BS,
    {
        build_inner(kg, params, keys, mk_block_backend, mk_byte_store).await
    }

    /// Look up an entity by plaintext name → returns the top-k
    /// nearest entities to the given query embedding (CompassIndex
    /// over the entities ORAM). Used by `lightrag-private::kg_query`.
    pub async fn query_entities_topk(
        &mut self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<u32>, LightKgError> {
        Ok(self.entities.search(query, k).await?)
    }

    /// Top-k relations by embedding similarity. Hybrid mode drives
    /// this with the high-level keyword embedding.
    pub async fn query_relations_topk(
        &mut self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<u32>, LightKgError> {
        Ok(self.relations.search(query, k).await?)
    }

    /// Adjacency lookup — returns the canonical-key strings of the
    /// relations adjacent to `entity_name`.
    pub fn adjacency_for_entity(
        &self,
        entity_name: &str,
    ) -> Result<Vec<String>, LightKgError> {
        let lk = derive_logical_key(&self.keys.emm_adjacency, label::ADJACENCY_ENTITY, entity_name);
        let values = self.adjacency.get(&lk)?;
        Ok(values
            .into_iter()
            .filter_map(|v| String::from_utf8(strip_trailing_nuls(v.0)).ok())
            .filter(|s| !s.is_empty())
            .collect())
    }

    /// Source-chunk lookup — returns the chunk ids tagged as source
    /// for `entity_name`.
    pub fn src_chunks_for_entity(
        &self,
        entity_name: &str,
    ) -> Result<Vec<String>, LightKgError> {
        let lk = derive_logical_key(
            &self.keys.emm_src_chunks,
            label::SRC_CHUNKS_ENTITY,
            entity_name,
        );
        let values = self.src_chunks.get(&lk)?;
        Ok(values
            .into_iter()
            .filter_map(|v| String::from_utf8(strip_trailing_nuls(v.0)).ok())
            .filter(|s| !s.is_empty())
            .collect())
    }

    /// Fetch the plaintext text of a chunk by id.
    pub fn chunk_text(&self, chunk_id: &str) -> Result<String, LightKgError> {
        Ok(self.chunk_text.get(chunk_id)?)
    }

    /// One-shot kg retrieval — the deep entry point all production
    /// callers should reach for. Owns the six-step fan-out (entities
    /// top-k → id↔name → optional relations top-k + endpoint inflation
    /// → adjacency → src_chunks → chunk_text decrypt) so the orchestrator
    /// doesn't need to know the store's internal id-map layout.
    ///
    /// The embeddings are expected to be **pre-perturbed** — the
    /// retrieval-side search-pattern perturbation
    /// ([`lightrag_private::perturb`]) is owned by the orchestrator,
    /// not the store, so the store remains agnostic to the
    /// search_pattern_key (HKDF v2 child #7).
    ///
    /// `hl_perturbed` is ignored when `shape == QueryShape::Local`; pass
    /// `&[]` in that case.
    pub async fn query_context(
        &mut self,
        ll_perturbed: &[f32],
        hl_perturbed: &[f32],
        params: &KgQueryParams,
    ) -> Result<KgContext, LightKgError> {
        // 1. Entities search
        let entity_block_ids = self
            .entities
            .search(ll_perturbed, params.top_k_entities)
            .await?;
        let entity_id_to_name: HashMap<u32, String> = self
            .entity_block_id
            .iter()
            .map(|(name, id)| (*id, name.clone()))
            .collect();
        let mut entity_names: Vec<String> = entity_block_ids
            .iter()
            .filter_map(|id| entity_id_to_name.get(id).cloned())
            .collect();

        // 2. Hybrid: relations search + endpoint inflation
        let mut relation_keys_from_search: Vec<String> = Vec::new();
        if matches!(params.shape, QueryShape::Hybrid) {
            let relation_block_ids = self
                .relations
                .search(hl_perturbed, params.top_k_relations)
                .await?;
            let relation_id_to_canon: HashMap<u32, String> = self
                .relation_block_id
                .iter()
                .map(|(canon, id)| (*id, canon.clone()))
                .collect();
            for id in &relation_block_ids {
                if let Some(canon) = relation_id_to_canon.get(id) {
                    relation_keys_from_search.push(canon.clone());
                }
                if let Some((src, tgt)) = self.relation_endpoints.get(id) {
                    entity_names.push(src.clone());
                    entity_names.push(tgt.clone());
                }
            }
        }
        let mut seen_e = std::collections::HashSet::new();
        entity_names.retain(|n| seen_e.insert(n.clone()));

        // 3. Adjacency fan-out
        let mut all_relations: Vec<String> = relation_keys_from_search;
        for name in &entity_names {
            let mut rels = self.adjacency_for_entity(name)?;
            all_relations.append(&mut rels);
        }
        let mut seen_r = std::collections::HashSet::new();
        all_relations.retain(|r| seen_r.insert(r.clone()));

        // 4. src_chunks fan-out
        let mut chunk_ids: Vec<String> = Vec::new();
        for name in &entity_names {
            let mut chunks = self.src_chunks_for_entity(name)?;
            chunks.truncate(params.top_k_chunks_per_entity);
            chunk_ids.append(&mut chunks);
        }
        let mut seen_c = std::collections::HashSet::new();
        chunk_ids.retain(|id| seen_c.insert(id.clone()));

        // 5. Decrypt chunk texts
        let mut chunks: Vec<(String, String)> = Vec::with_capacity(chunk_ids.len());
        for id in &chunk_ids {
            let text = self.chunk_text(id)?;
            chunks.push((id.clone(), text));
        }

        Ok(KgContext {
            entities: entity_names,
            relations: all_relations,
            chunks,
        })
    }
}

/// Local vs Hybrid retrieval shape. Local hits the entities index only;
/// Hybrid additionally pulls top-k relations and unions their endpoint
/// entities into the entity set before adjacency / src_chunk fan-out.
#[derive(Debug, Clone, Copy)]
pub enum QueryShape {
    Local,
    Hybrid,
}

/// Tuning surface for [`LightKgStore::query_context`].
#[derive(Debug, Clone)]
pub struct KgQueryParams {
    pub top_k_entities: usize,
    pub top_k_chunks_per_entity: usize,
    pub shape: QueryShape,
    /// Hybrid-mode only — number of relations to pull from the
    /// relations index before endpoint inflation. Ignored in Local
    /// mode.
    pub top_k_relations: usize,
}

impl Default for KgQueryParams {
    fn default() -> Self {
        Self {
            top_k_entities: 5,
            top_k_chunks_per_entity: 2,
            shape: QueryShape::Local,
            top_k_relations: 5,
        }
    }
}

/// One assembled retrieval result. Same shape upstream LightRAG's
/// `kg_query` returns to its caller (modulo `_token_truncation`).
#[derive(Debug, Default, Clone)]
pub struct KgContext {
    pub entities: Vec<String>,
    pub relations: Vec<String>,
    pub chunks: Vec<(String, String)>,
}

impl KgContext {
    /// Bare context string for the LLM. Minimal format — bit-for-bit
    /// LightRAG `_build_context_str` parity is M7.4.
    pub fn to_context_string(&self) -> String {
        let mut out = String::new();
        out.push_str("# Entities\n");
        for e in &self.entities {
            out.push_str(&format!("- {e}\n"));
        }
        out.push_str("\n# Relations\n");
        for r in &self.relations {
            out.push_str(&format!("- {r}\n"));
        }
        out.push_str("\n# Source chunks\n");
        for (id, text) in &self.chunks {
            out.push_str(&format!("[{id}] {text}\n"));
        }
        out
    }
}

/// Internal build driver. Shared between the in-memory and
/// REST-backed entry points.
async fn build_inner<B, BS, MB, MS>(
    kg: ExtractedKg,
    params: LightKgParams,
    keys: &DerivedKeysV2,
    mk_block_backend: MB,
    mk_byte_store: MS,
) -> Result<LightKgStore<B, BS>, LightKgError>
where
    B: BlockBackend,
    BS: ByteStoreBackend,
    MB: Fn(&CompassIndexParams) -> B,
    MS: Fn(&XorMmParams) -> BS,
{
    let bundle = KeyBundle::from_derived(keys);

    // ─── 1. Build the three CompassIndex instances ────────────────
    //
    // CompassIndex::from_plaintext_corpus_on encrypts blocks under a
    // key it derives from the build seed; the MASTER ORAM keys above
    // are the right per-tenant binding, but the M3/M4 builder
    // currently uses its own (deterministic) test seed. M6.x will
    // thread the master key through `CompassIndex::with_oram_key`
    // (M4 follow-on, not gating the build).
    let entity_embeddings: Vec<Vec<f32>> = kg
        .entities
        .iter()
        .map(|e| e.embedding.clone())
        .collect();
    let entities = CompassIndex::from_plaintext_corpus_on(
        entity_embeddings,
        params.entities,
        mk_block_backend(&params.entities),
    )
    .await?;

    let relation_embeddings: Vec<Vec<f32>> = kg
        .relations
        .iter()
        .map(|r| r.embedding.clone())
        .collect();
    let relations = CompassIndex::from_plaintext_corpus_on(
        relation_embeddings,
        params.relations,
        mk_block_backend(&params.relations),
    )
    .await?;

    let chunk_embeddings: Vec<Vec<f32>> = kg.chunks.iter().map(|c| c.embedding.clone()).collect();
    let chunks = CompassIndex::from_plaintext_corpus_on(
        chunk_embeddings,
        params.chunks,
        mk_block_backend(&params.chunks),
    )
    .await?;

    // ─── 2. Build the in-CVM id maps ───────────────────────────────
    let entity_block_id: HashMap<String, u32> = kg
        .entities
        .iter()
        .enumerate()
        .map(|(i, e)| (e.name.clone(), i as u32))
        .collect();
    let relation_block_id: HashMap<String, u32> = kg
        .relations
        .iter()
        .enumerate()
        .map(|(i, r)| (r.canonical_key(), i as u32))
        .collect();
    let relation_endpoints: HashMap<u32, (String, String)> = kg
        .relations
        .iter()
        .enumerate()
        .map(|(i, r)| (i as u32, (r.src.clone(), r.tgt.clone())))
        .collect();
    let chunk_block_id: HashMap<String, u32> = kg
        .chunks
        .iter()
        .enumerate()
        .map(|(i, c)| (c.id.clone(), i as u32))
        .collect();

    // ─── 3. Build the AES chunk store ─────────────────────────────
    let mut chunk_text = AesChunkStore::new(clone_zeroized(&bundle.aes_chunk));
    chunk_text.put_all(&kg.chunks)?;

    // ─── 4. Build the two EMMs ────────────────────────────────────
    //
    // Adjacency: for each entity → list of canonical relation keys it
    // participates in. Relations are undirected; both endpoints get
    // an entry.
    let mut adjacency_entries: HashMap<String, Vec<LogicalValue>> = HashMap::new();
    for r in &kg.relations {
        let canon = r.canonical_key();
        adjacency_entries
            .entry(r.src.clone())
            .or_default()
            .push(string_to_value(&canon, params.adjacency.value_bytes as usize)?);
        if r.src != r.tgt {
            adjacency_entries
                .entry(r.tgt.clone())
                .or_default()
                .push(string_to_value(&canon, params.adjacency.value_bytes as usize)?);
        }
    }
    let adjacency_emm = build_emm(
        &bundle.emm_adjacency,
        label::ADJACENCY_ENTITY,
        adjacency_entries,
        params.adjacency,
        mk_byte_store(&params.adjacency),
    )?;

    // Source chunks: for each entity → list of chunk_ids it was
    // extracted from. Also covers relations under a separate label
    // (M6.x extension; entity-side gives us the LightRAG fan-out).
    let mut src_chunks_entries: HashMap<String, Vec<LogicalValue>> = HashMap::new();
    for e in &kg.entities {
        let acc = src_chunks_entries.entry(e.name.clone()).or_default();
        for c in &e.source_chunks {
            acc.push(string_to_value(c, params.src_chunks.value_bytes as usize)?);
        }
    }
    let src_chunks_emm = build_emm(
        &bundle.emm_src_chunks,
        label::SRC_CHUNKS_ENTITY,
        src_chunks_entries,
        params.src_chunks,
        mk_byte_store(&params.src_chunks),
    )?;

    Ok(LightKgStore {
        entities,
        relations,
        chunks,
        adjacency: adjacency_emm,
        src_chunks: src_chunks_emm,
        chunk_text,
        entity_block_id,
        relation_block_id,
        relation_endpoints,
        chunk_block_id,
        keys: bundle,
    })
}

fn build_emm<BS: ByteStoreBackend>(
    master_key: &Zeroizing<[u8; 32]>,
    label: &str,
    entries: HashMap<String, Vec<LogicalValue>>,
    params: XorMmParams,
    backend: BS,
) -> Result<XorMmClient<BS>, LightKgError> {
    let mut prepared: Vec<(LogicalKey, Vec<LogicalValue>)> = Vec::with_capacity(entries.len());
    let mut seed1 = [0u8; 16];
    let mut seed2 = [0u8; 16];
    seed1.copy_from_slice(&master_key.as_ref()[..16]);
    seed2.copy_from_slice(&master_key.as_ref()[16..]);

    for (name, values) in entries {
        let lk = derive_logical_key(master_key, label, &name);
        prepared.push((lk, values));
    }

    // Use the master key as the AES-GCM key for buckets. XorMmClient
    // takes ownership of the cloned key buffer.
    let aes_key = clone_zeroized(master_key);
    let client = XorMmClient::build(prepared, params, aes_key, seed1, seed2, backend)?;
    Ok(client)
}

fn string_to_value(s: &str, value_bytes: usize) -> Result<LogicalValue, LightKgError> {
    if s.len() > value_bytes {
        return Err(LightKgError::Inconsistent(format!(
            "value {s:?} exceeds {value_bytes} bytes — bump XorMmParams.value_bytes"
        )));
    }
    Ok(LogicalValue(s.as_bytes().to_vec()))
}

fn strip_trailing_nuls(mut v: Vec<u8>) -> Vec<u8> {
    while v.last() == Some(&0u8) {
        v.pop();
    }
    v
}

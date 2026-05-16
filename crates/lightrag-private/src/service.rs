//! `LightRagPrivateService` — the retrieval orchestrator. M7.1
//! ships the Local-mode `kg_query` path: embed → search-perturb →
//! entities top-k → adjacency → src_chunks → chunk text → assembled
//! context string.
//!
//! Full LightRAG-parity (Hybrid, Global, Mix, Naive modes; bit-for-
//! bit `_build_context_str`; degree-sort with `(weight, cosine)`)
//! lands in M7.x follow-ons. The shape here is enough for M8 to
//! plumb the HTTP route and prove end-to-end functionality.

use std::collections::HashMap;

use light_kg_store::{ByteStoreBackend, LightKgError, LightKgStore};
use ring_oram::BlockBackend;

use crate::perturb::{EmbeddingKind, SessionKey, perturb};

/// Sub-set of `QueryMode` we wire in M7.1.
#[derive(Debug, Clone, Copy)]
pub enum QueryShape {
    /// LightRAG `local`: low-level keyword embedding drives the
    /// entities CompassIndex. Adjacency + src_chunks fan-out off
    /// the matched entities.
    Local,
}

/// One assembled retrieval result. Surfaces the same shape upstream
/// LightRAG's `kg_query` returns to its caller (modulo `_token
/// _truncation` which lands at M7.x). Bit-for-bit parity in the
/// upstream string format is M7.4.
#[derive(Debug, Default, Clone)]
pub struct KgContext {
    /// Entity-name ↦ description (decrypted from the in-store
    /// description blob). In M7.1 we just hold the entity name —
    /// description retrieval comes online when the encrypted
    /// node-props KV ships (M6.x).
    pub entities: Vec<String>,
    /// Canonical relation keys produced by the adjacency lookup.
    pub relations: Vec<String>,
    /// `(chunk_id, decrypted text)` pairs.
    pub chunks: Vec<(String, String)>,
}

impl KgContext {
    /// Bare context string for the LLM. Minimal format — M7.4 swaps
    /// in the upstream LightRAG `_build_context_str` template
    /// verbatim.
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

#[derive(Debug, Clone)]
pub struct KgQueryParams {
    pub top_k_entities: usize,
    pub top_k_chunks_per_entity: usize,
    pub shape: QueryShape,
}

impl Default for KgQueryParams {
    fn default() -> Self {
        Self {
            top_k_entities: 5,
            top_k_chunks_per_entity: 2,
            shape: QueryShape::Local,
        }
    }
}

/// The orchestrator. Holds a reference to the underlying
/// `LightKgStore` (one per tenant; lifetime managed by the runner).
pub struct LightRagPrivateService<'a, B: BlockBackend, BS: ByteStoreBackend> {
    pub store: &'a mut LightKgStore<B, BS>,
}

impl<'a, B: BlockBackend, BS: ByteStoreBackend> LightRagPrivateService<'a, B, BS> {
    pub fn new(store: &'a mut LightKgStore<B, BS>) -> Self {
        Self { store }
    }

    /// `kg_query` — the LightRAG retrieval entry point. M7.1: Local
    /// mode only, no `_token_truncation`, no `_merge_all_chunks`
    /// (single-entity fan-out is good enough for an MVP).
    ///
    /// `ll_query_embedding` is the *low-level keyword* embedding —
    /// upstream LightRAG calls `extract_keywords_only` to produce
    /// `(hl_keywords, ll_keywords)` from the raw query; for M7.1
    /// we accept the ll embedding directly (the in-TEE keyword LLM
    /// is M9). The hl path is wired the same way once we add the
    /// `Hybrid` shape.
    pub async fn kg_query(
        &mut self,
        ll_query_embedding: &[f32],
        params: &KgQueryParams,
        session_key: &SessionKey,
    ) -> Result<KgContext, LightKgError> {
        let _ = params.shape; // M7.1: only Local supported

        // ─── 1. Search perturbation (plan §8.6) ────────────────────
        let perturbed = perturb(session_key, EmbeddingKind::Ll, ll_query_embedding);

        // ─── 2. Entities CompassIndex search ───────────────────────
        let entity_block_ids = self
            .store
            .query_entities_topk(&perturbed, params.top_k_entities)
            .await?;

        // Map block id → name via the in-CVM cleartext map.
        let id_to_name: HashMap<u32, String> = self
            .store
            .entity_block_id
            .iter()
            .map(|(name, id)| (*id, name.clone()))
            .collect();
        let entity_names: Vec<String> = entity_block_ids
            .iter()
            .filter_map(|id| id_to_name.get(id).cloned())
            .collect();

        // ─── 3. Adjacency fan-out per entity ───────────────────────
        let mut all_relations: Vec<String> = Vec::new();
        for name in &entity_names {
            let mut rels = self.store.adjacency_for_entity(name)?;
            all_relations.append(&mut rels);
        }
        // Dedup, preserve order.
        let mut seen = std::collections::HashSet::new();
        all_relations.retain(|r| seen.insert(r.clone()));

        // ─── 4. src_chunks fan-out per entity ──────────────────────
        let mut chunk_ids: Vec<String> = Vec::new();
        for name in &entity_names {
            let mut chunks = self.store.src_chunks_for_entity(name)?;
            // Cap per-entity at the configured limit.
            chunks.truncate(params.top_k_chunks_per_entity);
            chunk_ids.append(&mut chunks);
        }
        let mut seen = std::collections::HashSet::new();
        chunk_ids.retain(|id| seen.insert(id.clone()));

        // ─── 5. Decrypt chunk texts ─────────────────────────────────
        let mut chunks: Vec<(String, String)> = Vec::with_capacity(chunk_ids.len());
        for id in &chunk_ids {
            let text = self.store.chunk_text(id)?;
            chunks.push((id.clone(), text));
        }

        Ok(KgContext {
            entities: entity_names,
            relations: all_relations,
            chunks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_string_renders_sections_in_order() {
        let ctx = KgContext {
            entities: vec!["alice".into(), "bob".into()],
            relations: vec!["alice\0bob".into()],
            chunks: vec![("c-0".into(), "hello".into())],
        };
        let s = ctx.to_context_string();
        // Sanity: each section header present, in order.
        let idx_e = s.find("# Entities").unwrap();
        let idx_r = s.find("# Relations").unwrap();
        let idx_c = s.find("# Source chunks").unwrap();
        assert!(idx_e < idx_r);
        assert!(idx_r < idx_c);
        // Entity names appear.
        assert!(s.contains("- alice"));
        assert!(s.contains("- bob"));
    }

    #[test]
    fn default_params_pick_local_shape() {
        let p = KgQueryParams::default();
        assert!(matches!(p.shape, QueryShape::Local));
    }

    // The end-to-end integration test that drives `kg_query` over a
    // real `LightKgStore` lives at `tests/local_kg_query.rs` — it
    // requires the async runtime + the synth-KG helper.
}

//! `LightRagPrivateService` — the retrieval orchestrator. M7.1
//! ships the Local-mode `kg_query` path: embed → search-perturb →
//! delegate to [`LightKgStore::query_context`].
//!
//! The store owns the actual entities → adjacency → src_chunks
//! fan-out (see [`KgContext`] / [`KgQueryParams`]); this service
//! contributes the retrieval-side search-pattern perturbation
//! ([`crate::perturb`]).

use light_kg_store::{
    ByteStoreBackend, KgContext, KgQueryParams, LightKgError, LightKgStore, QueryShape,
};
use ring_oram::BlockBackend;

use crate::perturb::{EmbeddingKind, SessionKey, perturb};

/// The orchestrator. Holds a reference to the underlying
/// `LightKgStore` (one per tenant; lifetime managed by the runner).
pub struct LightRagPrivateService<'a, B: BlockBackend, BS: ByteStoreBackend> {
    pub store: &'a mut LightKgStore<B, BS>,
}

impl<'a, B: BlockBackend, BS: ByteStoreBackend> LightRagPrivateService<'a, B, BS> {
    pub fn new(store: &'a mut LightKgStore<B, BS>) -> Self {
        Self { store }
    }

    /// `kg_query` — the LightRAG retrieval entry point. M7.1 ships
    /// Local mode; M7.2 adds Hybrid. The `hl_query_embedding` argument
    /// is ignored in Local mode (the caller may pass `&[]`).
    ///
    /// Upstream LightRAG calls `extract_keywords_only` to produce
    /// `(hl_keywords, ll_keywords)` from the raw query; the in-TEE
    /// keyword LLM is M9.
    pub async fn kg_query(
        &mut self,
        ll_query_embedding: &[f32],
        hl_query_embedding: &[f32],
        params: &KgQueryParams,
        session_key: &SessionKey,
    ) -> Result<KgContext, LightKgError> {
        let ll_perturbed = perturb(session_key, EmbeddingKind::Ll, ll_query_embedding);
        let hl_perturbed = if matches!(params.shape, QueryShape::Hybrid) {
            perturb(session_key, EmbeddingKind::Hl, hl_query_embedding)
        } else {
            Vec::new()
        };
        self.store
            .query_context(&ll_perturbed, &hl_perturbed, params)
            .await
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

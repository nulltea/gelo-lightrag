//! `/lightrag/*` HTTP routes — M8.0.
//!
//! Mirrors the shape of the existing `/ingest`, `/query`, `/attest`
//! routes but binds to a `LightRagTwoPartyService` and the
//! `LightKgStore`-backed retrieval surface. Wire types are JSON;
//! ciphertext fields stay base64. The KG payload itself is plaintext
//! over RATLS (the client-side extraction LLM produces it; see OQ#5).

use std::sync::Arc;

use axum::{
    Json,
    extract::State,
    response::IntoResponse,
};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use light_kg_store::{
    Chunk, CompassIndexParams, Entity, ExtractedKg, LightKgParams, PlainHnswParams, Relation,
    RingOramParams, XorMmParams,
};
use lightrag_private::{KgQueryParams, LightRagTwoPartyService, QueryShape};
use rag_core::TenantId;
use serde::{Deserialize, Serialize};
use tracing::info;
use zeroize::Zeroizing;

use crate::AppError;

/// Shared lightrag-service handle. Held alongside the existing
/// `AppState` rather than embedded so the runner's CAPRISE service
/// stays untouched.
pub type LightRagServiceHandle = Arc<LightRagTwoPartyService>;

/// Per-tenant build defaults. The plan's M0+ `CompassParams` carries
/// the production HNSW + ORAM tuning; the simplest M8 client picks
/// an embedding dim and lets the server fill in everything else.
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

fn default_xormm_params() -> XorMmParams {
    XorMmParams {
        volume_bound: 16,
        value_bytes: 64,
        n_buckets: 256,
        max_kicks: 256,
    }
}

#[derive(Deserialize, Debug)]
pub struct ChunkJson {
    pub id: String,
    pub text: String,
    pub embedding: Vec<f32>,
}

#[derive(Deserialize, Debug)]
pub struct EntityJson {
    pub name: String,
    pub description: String,
    pub embedding: Vec<f32>,
    pub source_chunks: Vec<String>,
}

#[derive(Deserialize, Debug)]
pub struct RelationJson {
    pub src: String,
    pub tgt: String,
    pub description: String,
    pub embedding: Vec<f32>,
    pub source_chunks: Vec<String>,
}

#[derive(Deserialize, Debug)]
pub struct ExtractedKgJson {
    pub chunks: Vec<ChunkJson>,
    pub entities: Vec<EntityJson>,
    pub relations: Vec<RelationJson>,
}

impl From<ExtractedKgJson> for ExtractedKg {
    fn from(j: ExtractedKgJson) -> Self {
        ExtractedKg {
            chunks: j
                .chunks
                .into_iter()
                .map(|c| Chunk {
                    id: c.id,
                    text: c.text,
                    embedding: c.embedding,
                })
                .collect(),
            entities: j
                .entities
                .into_iter()
                .map(|e| Entity {
                    name: e.name,
                    description: e.description,
                    embedding: e.embedding,
                    source_chunks: e.source_chunks,
                })
                .collect(),
            relations: j
                .relations
                .into_iter()
                .map(|r| Relation {
                    src: r.src,
                    tgt: r.tgt,
                    description: r.description,
                    embedding: r.embedding,
                    source_chunks: r.source_chunks,
                })
                .collect(),
        }
    }
}

/// Reuse the same `UserXskB64` shape from the embedder routes —
/// inlined here to keep the lightrag module independently
/// readable.
#[derive(Deserialize)]
#[serde(transparent)]
pub struct UserXskB64(String);

impl std::fmt::Debug for UserXskB64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<redacted user_x_sk : base64-32B>")
    }
}

impl UserXskB64 {
    fn decode(&self) -> anyhow::Result<Zeroizing<[u8; 32]>> {
        let bytes = Zeroizing::new(
            B64.decode(self.0.as_bytes())
                .map_err(|e| anyhow::anyhow!("user_x_sk: base64 decode: {e}"))?,
        );
        if bytes.len() != 32 {
            anyhow::bail!(
                "user_x_sk must be exactly 32 bytes after base64 decode (got {})",
                bytes.len()
            );
        }
        let mut out: Zeroizing<[u8; 32]> = Zeroizing::new([0u8; 32]);
        out.copy_from_slice(&bytes);
        Ok(out)
    }
}

#[derive(Deserialize, Debug)]
pub struct LightRagIngestRequest {
    pub tenant_id: String,
    pub user_x_sk: UserXskB64,
    pub extracted_kg: ExtractedKgJson,
    /// Embedding dimension — must match the dim of every entity /
    /// relation / chunk embedding. Sized into the CompassIndex
    /// build at server-default HNSW params.
    pub dim: usize,
}

#[derive(Serialize)]
pub struct LightRagIngestResponse {
    pub ingested: IngestStats,
}

#[derive(Serialize, Default, Debug)]
pub struct IngestStats {
    pub chunks: usize,
    pub entities: usize,
    pub relations: usize,
}

pub async fn ingest(
    State(lightrag): State<LightRagServiceHandle>,
    Json(req): Json<LightRagIngestRequest>,
) -> Result<Json<LightRagIngestResponse>, AppError> {
    let tenant = TenantId::new(req.tenant_id);
    let user_x_sk = req.user_x_sk.decode()?;
    let kg: ExtractedKg = req.extracted_kg.into();
    let stats = IngestStats {
        chunks: kg.chunks.len(),
        entities: kg.entities.len(),
        relations: kg.relations.len(),
    };

    let params = LightKgParams {
        entities: default_compass_params(req.dim, stats.entities),
        relations: default_compass_params(req.dim, stats.relations.max(8)),
        chunks: default_compass_params(req.dim, stats.chunks),
        adjacency: default_xormm_params(),
        src_chunks: default_xormm_params(),
    };

    lightrag
        .ingest_kg_for(&tenant, user_x_sk, kg, params)
        .await?;
    info!(tenant = %tenant, ?stats, "lightrag ingest complete");
    Ok(Json(LightRagIngestResponse { ingested: stats }))
}

#[derive(Deserialize, Debug)]
pub struct LightRagQueryRequest {
    pub tenant_id: String,
    /// Low-level keyword embedding. Local + Hybrid modes use it.
    pub ll_query_embedding: Vec<f32>,
    /// High-level keyword embedding. Hybrid mode only — pass `[]`
    /// or omit for Local mode.
    #[serde(default)]
    pub hl_query_embedding: Vec<f32>,
    /// Retrieval mode. Defaults to "local". M7.2 ships "local" and
    /// "hybrid". M7.x will add the remaining modes.
    #[serde(default = "default_mode")]
    pub mode: String,
    /// 16-byte session nonce. Same nonce within a session ensures
    /// `search_perturb` produces the same output each call.
    pub session_nonce_b64: String,
    #[serde(default = "default_top_k_entities")]
    pub top_k_entities: usize,
    #[serde(default = "default_top_k_relations")]
    pub top_k_relations: usize,
    #[serde(default = "default_top_k_chunks")]
    pub top_k_chunks_per_entity: usize,
}

fn default_top_k_entities() -> usize {
    5
}
fn default_top_k_relations() -> usize {
    5
}
fn default_top_k_chunks() -> usize {
    2
}
fn default_mode() -> String {
    "local".to_string()
}

#[derive(Serialize)]
pub struct LightRagQueryResponse {
    pub entities: Vec<String>,
    pub relations: Vec<String>,
    pub chunks: Vec<ChunkResp>,
    pub context_string: String,
}

#[derive(Serialize)]
pub struct ChunkResp {
    pub id: String,
    pub text: String,
}

pub async fn query(
    State(lightrag): State<LightRagServiceHandle>,
    Json(req): Json<LightRagQueryRequest>,
) -> Result<Json<LightRagQueryResponse>, AppError> {
    let tenant = TenantId::new(req.tenant_id);
    let session_nonce = B64
        .decode(req.session_nonce_b64.as_bytes())
        .map_err(|e| anyhow::anyhow!("session_nonce_b64: base64 decode: {e}"))?;
    let shape = match req.mode.as_str() {
        "local" => QueryShape::Local,
        "hybrid" => QueryShape::Hybrid,
        other => {
            return Err(AppError::from(anyhow::anyhow!(
                "unsupported mode {other:?} — M7.2 supports 'local' and 'hybrid'"
            )));
        }
    };
    let params = KgQueryParams {
        top_k_entities: req.top_k_entities,
        top_k_chunks_per_entity: req.top_k_chunks_per_entity,
        shape,
        top_k_relations: req.top_k_relations,
    };
    let ctx = lightrag
        .query_for(
            &tenant,
            &req.ll_query_embedding,
            &req.hl_query_embedding,
            &params,
            &session_nonce,
        )
        .await?;
    let context_string = ctx.to_context_string();
    Ok(Json(LightRagQueryResponse {
        entities: ctx.entities,
        relations: ctx.relations,
        chunks: ctx
            .chunks
            .into_iter()
            .map(|(id, text)| ChunkResp { id, text })
            .collect(),
        context_string,
    }))
}

/// `/lightrag/attest` — separate route name so a relying party can
/// pin a different scheme_identity for the LightRAG path vs the
/// embedder path. M8.0 just echoes the runner's `scheme_identity`
/// (the existing `/attest` does the heavy lifting). M8.x will add
/// the LightRAG-specific `scheme_identity_digest` from V2 HKDF.
pub async fn attest() -> impl IntoResponse {
    // Sentinel — the existing /attest endpoint produces the actual
    // SEV-SNP report. M8.0 ships this stub so clients can probe for
    // LightRAG-route support without parsing 404.
    "lightrag attest: use the /attest route for full evidence — \
     scheme_identity already composes the V2 HKDF digest."
}

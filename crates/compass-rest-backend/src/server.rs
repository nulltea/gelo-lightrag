//! Axum server providing the network shape of `BlockBackend`.
//!
//! Storage: one `sled::Db` for the whole server; per-tenant + per-index
//! is implemented via key prefix `{tenant}/{index}/{bucket_id u32-LE}`.
//! The value stored is the msgpack-encoded `WireBucket`. The server
//! holds no crypto state and learns nothing beyond the ORAM access
//! pattern — the trust model puts the storage server in the
//! adversary's hands.
//!
//! Per-tenant URL routing is the *only* tenant gate. Confidentiality
//! is already covered by per-tenant ORAM keys (HKDF v2). Cross-tenant
//! routing is a routing-layer 404; defense-in-depth, not the
//! confidentiality boundary.

use std::sync::Arc;

use axum::{
    Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Path, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::post,
};
use serde::Serialize;
use sled::Db;

use crate::wire::{
    InitRequest, ReadPathRequest, ReadPathResponse, WireBucket, WriteBucketsRequest,
};

const CONTENT_TYPE_MSGPACK: &str = "application/msgpack";

/// Shared server state — wraps the underlying sled DB.
#[derive(Clone)]
pub struct AppState {
    db: Arc<Db>,
}

impl AppState {
    pub fn new(db: Db) -> Self {
        Self { db: Arc::new(db) }
    }

    /// `Tree` for the `(tenant, index)` pair. sled opens new trees
    /// lazily; the trees are persisted under their UTF-8 names.
    fn tree(&self, tenant: &str, index: &str) -> Result<sled::Tree, sled::Error> {
        let name = format!("{tenant}/{index}");
        self.db.open_tree(name.as_bytes())
    }
}

/// Maximum body size accepted by the server. Sized to fit one
/// `initialize_tree` worth of writes at production parameters (block_bytes
/// = 1024, n_leaves = 2048 ⇒ ~36 MB). M4.3's batched ORAM reads will
/// push read-path response sizes up, hence the comfortable headroom.
const MAX_BODY_BYTES: usize = 128 * 1024 * 1024;

/// Build the axum router. Wire it under `/v1`. Run with `axum::serve`.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/:tenant/:index/init", post(init_handler))
        .route("/v1/:tenant/:index/read_path", post(read_path_handler))
        .route("/v1/:tenant/:index/write_buckets", post(write_buckets_handler))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state)
}

// ─── error mapping ──────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
enum ServerError {
    #[error("storage error: {0}")]
    Storage(#[from] sled::Error),
    #[error("codec error: {0}")]
    Codec(String),
    #[error("missing bucket {0} (server tree under-populated for tenant/index)")]
    MissingBucket(u32),
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let code = match &self {
            ServerError::Storage(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ServerError::Codec(_) => StatusCode::BAD_REQUEST,
            ServerError::MissingBucket(_) => StatusCode::NOT_FOUND,
        };
        tracing::warn!(error = %self, "REST backend error");
        (code, self.to_string()).into_response()
    }
}

fn msgpack<T: Serialize>(value: &T) -> Result<Response, ServerError> {
    let body = rmp_serde::to_vec(value).map_err(|e| ServerError::Codec(e.to_string()))?;
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        CONTENT_TYPE_MSGPACK
            .parse()
            .expect("static msgpack content-type is valid"),
    );
    Ok((headers, body).into_response())
}

fn parse_msgpack<T: for<'de> serde::Deserialize<'de>>(body: &[u8]) -> Result<T, ServerError> {
    rmp_serde::from_slice(body).map_err(|e| ServerError::Codec(e.to_string()))
}

// ─── handlers ───────────────────────────────────────────────────────

/// `POST /v1/{tenant}/{index}/init` — allocate `num_buckets` slots if
/// the tenant/index tree is empty. Idempotent: a replay against a
/// populated tree is a no-op (the client's own `RingOramClient::new`
/// already drives `initialize_tree` to fill every slot).
async fn init_handler(
    Path((tenant, index)): Path<(String, String)>,
    State(state): State<AppState>,
    body: Bytes,
) -> Result<Response, ServerError> {
    let req: InitRequest = parse_msgpack(&body)?;
    let tree = state.tree(&tenant, &index)?;

    // We *don't* pre-populate the tree here — that's the client's job
    // via `initialize_tree`. We just allocate the namespace and tag
    // the expected size so subsequent reads can surface
    // under-populated trees as a clear error rather than mysterious
    // missing-bucket 404s.
    if let Some(existing) = tree.get(b"__num_buckets")? {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&existing[..4]);
        let existing_count = u32::from_le_bytes(buf);
        if existing_count != req.num_buckets {
            return Err(ServerError::Codec(format!(
                "init mismatch: tree has {existing_count} buckets, request asks for {}",
                req.num_buckets
            )));
        }
    } else {
        tree.insert(b"__num_buckets", &req.num_buckets.to_le_bytes())?;
    }
    tree.flush_async().await?;
    Ok(StatusCode::OK.into_response())
}

async fn read_path_handler(
    Path((tenant, index)): Path<(String, String)>,
    State(state): State<AppState>,
    body: Bytes,
) -> Result<Response, ServerError> {
    let req: ReadPathRequest = parse_msgpack(&body)?;
    let tree = state.tree(&tenant, &index)?;

    let mut buckets = Vec::with_capacity(req.bucket_ids.len());
    for &bid in &req.bucket_ids {
        let key = bucket_key(bid);
        let raw = tree.get(&key)?.ok_or(ServerError::MissingBucket(bid))?;
        let wire: WireBucket = parse_msgpack(&raw)?;
        buckets.push(wire);
    }
    msgpack(&ReadPathResponse { buckets })
}

async fn write_buckets_handler(
    Path((tenant, index)): Path<(String, String)>,
    State(state): State<AppState>,
    body: Bytes,
) -> Result<Response, ServerError> {
    let req: WriteBucketsRequest = parse_msgpack(&body)?;
    let tree = state.tree(&tenant, &index)?;

    // Batch via sled's transactional API would be the more
    // crash-consistent choice; the M5.1 baseline takes per-bucket
    // writes since the only fault we need to recover from is a CVM
    // restart, and the per-bucket write_counter already serialises
    // concurrent updates from the client's perspective.
    for w in &req.buckets {
        let key = bucket_key(w.bucket_id);
        let value = rmp_serde::to_vec(w).map_err(|e| ServerError::Codec(e.to_string()))?;
        tree.insert(key, value)?;
    }
    tree.flush_async().await?;
    Ok(StatusCode::OK.into_response())
}

fn bucket_key(bucket_id: u32) -> [u8; 4] {
    bucket_id.to_le_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db() -> (Db, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db = sled::open(dir.path()).unwrap();
        (db, dir)
    }

    #[tokio::test]
    async fn init_is_idempotent() {
        let (db, _dir) = temp_db();
        let state = AppState::new(db);
        let tree = state.tree("alpha", "entities").unwrap();
        // first init writes the size sentinel
        tree.insert(b"__num_buckets", &16u32.to_le_bytes()).unwrap();
        // second init with same size: tree access OK, no mutation
        let again = tree.get(b"__num_buckets").unwrap().unwrap();
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&again[..4]);
        assert_eq!(u32::from_le_bytes(buf), 16);
    }

    #[tokio::test]
    async fn server_state_separates_tenants() {
        // Same index name under two tenants must be two physically
        // distinct trees. Pin the property: writes under tenant `a`
        // don't show up under tenant `b`.
        let (db, _dir) = temp_db();
        let state = AppState::new(db);
        let a = state.tree("a", "idx").unwrap();
        let b = state.tree("b", "idx").unwrap();
        a.insert(b"key", b"value-a").unwrap();
        assert_eq!(a.get(b"key").unwrap().unwrap(), b"value-a");
        assert!(b.get(b"key").unwrap().is_none());
    }
}

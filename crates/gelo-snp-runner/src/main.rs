//! `gelo-snp-runner` — production-style service binary for the GELO+SEV-SNP
//! deployment.
//!
//! Boots once at process start, parses [`SNP_MODE`] (production or mock),
//! wires up an [`Approach4InMemoryService`] with the SEV-SNP attestation
//! backend, and serves a minimal HTTP API.
//!
//! Designed to be the **same binary** at every simulation tier:
//! - **T1**: invoked via `cargo run`; useful for hand-driven smoke testing.
//! - **T2**: launched by systemd inside a regular QEMU/KVM VM that boots
//!   the CVM image with `SNP_MODE=mock` and a shim `/dev/sev-guest`.
//! - **T3**: same systemd unit, same binary, same image — but on a real
//!   SEV-SNP CVM with `SNP_MODE=production` so `HardwareReportIssuer` opens
//!   the real `/dev/sev-guest` device.
//!
//! The HTTP surface is intentionally tiny — this isn't a feature-complete
//! RAG server, it's the attestable embedder behind one. Endpoints:
//!
//! - `GET  /health`  → 200 OK
//! - `GET  /attest`  → fresh attestation evidence (report + VCEK + identities)
//! - `POST /ingest`  → `{ "chunks": [{"id": ..., "text": ...}] }`
//! - `POST /query`   → `{ "text": ..., "top_k": N }` → ranked hits

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use approach4::{Approach4InMemoryService, NoopAttestationVerifier};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use rag_core::{ChunkId, DocumentChunk, Embedder, EmbeddingEncryptionScheme, EncryptedEmbedding};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

mod config;
mod evidence;
mod issuer;

use config::RunnerConfig;
use evidence::{IssuerHandle, build_evidence};

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let mode = gelo_tee_sev_snp::runtime_mode::from_env()
        .context("parsing SNP_MODE from environment")?;
    info!("gelo-snp-runner starting; mode = {mode}");

    let cfg = RunnerConfig::load()
        .context("loading runner config (/etc/gelo-snp/runner.toml or $GELO_SNP_RUNNER_CONFIG)")?;
    info!(
        "loaded config: listen={} scheme_identity={:?} embedder={:?}",
        cfg.listen, cfg.scheme_identity, cfg.embedder
    );

    let embedder = StubEmbedder::from_model_id(&cfg.model_identity);
    let model_identity_b = cfg.model_identity.clone().into_bytes();
    let scheme = IdentityScheme;

    let issuer = IssuerHandle::for_mode(mode)?;

    let service = Approach4InMemoryService::new(embedder, scheme, NoopAttestationVerifier);

    let state = AppState {
        service: Arc::new(Mutex::new(service)),
        issuer: Arc::new(issuer),
        model_identity: model_identity_b,
        scheme_identity: cfg.scheme_identity.into_bytes(),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/attest", get(attest))
        .route("/ingest", post(ingest))
        .route("/query", post(query))
        .with_state(state);

    let addr: SocketAddr = cfg.listen.parse().context("parsing listen address")?;
    info!("listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .context("binding listen address")?;

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("axum serve loop")?;

    info!("gelo-snp-runner shut down cleanly");
    Ok(())
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,gelo_snp_runner=debug"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("install SIGTERM handler");
    #[cfg(unix)]
    tokio::select! {
        _ = ctrl_c => info!("received SIGINT"),
        _ = term.recv() => info!("received SIGTERM"),
    }
    #[cfg(not(unix))]
    {
        let _ = ctrl_c.await;
        info!("received Ctrl-C");
    }
}

#[derive(Clone)]
struct AppState {
    service: Arc<Mutex<Approach4InMemoryService<StubEmbedder, IdentityScheme, NoopAttestationVerifier>>>,
    issuer: Arc<IssuerHandle>,
    model_identity: Vec<u8>,
    scheme_identity: Vec<u8>,
}

async fn health() -> &'static str {
    "ok"
}

#[derive(Serialize)]
struct AttestResponse {
    model_identity: String,
    scheme_identity: String,
    /// Base64-encoded 1184-byte SEV-SNP attestation report.
    report_b64: String,
    /// Base64-encoded VCEK certificate (PEM bytes).
    vcek_cert_b64: String,
}

async fn attest(State(state): State<AppState>) -> Result<Json<AttestResponse>, AppError> {
    let evidence = build_evidence(&state.issuer, &state.model_identity, &state.scheme_identity)?;
    let report_b64 = B64.encode(evidence.report.as_deref().unwrap_or(&[]));
    let vcek_cert_b64 = B64.encode(evidence.vcek_cert.as_deref().unwrap_or(&[]));
    Ok(Json(AttestResponse {
        model_identity: evidence.model_identity,
        scheme_identity: evidence.scheme_identity,
        report_b64,
        vcek_cert_b64,
    }))
}

#[derive(Deserialize)]
struct IngestRequest {
    chunks: Vec<IngestChunk>,
}

#[derive(Deserialize)]
struct IngestChunk {
    id: String,
    text: String,
}

#[derive(Serialize)]
struct IngestResponse {
    ingested: usize,
}

async fn ingest(
    State(state): State<AppState>,
    Json(req): Json<IngestRequest>,
) -> Result<Json<IngestResponse>, AppError> {
    let n = req.chunks.len();
    let docs: Vec<DocumentChunk> = req
        .chunks
        .into_iter()
        .map(|c| DocumentChunk {
            id: ChunkId(c.id),
            text: c.text,
        })
        .collect();
    state.service.lock().await.ingest_chunks(docs)?;
    Ok(Json(IngestResponse { ingested: n }))
}

#[derive(Deserialize)]
struct QueryRequest {
    text: String,
    top_k: Option<usize>,
}

#[derive(Serialize)]
struct QueryHit {
    id: String,
    score: f32,
    text: String,
}

#[derive(Serialize)]
struct QueryResponse {
    hits: Vec<QueryHit>,
    attestation: AttestResponse,
}

async fn query(
    State(state): State<AppState>,
    Json(req): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, AppError> {
    let top_k = req.top_k.unwrap_or(5);
    let hits = state
        .service
        .lock()
        .await
        .query(&req.text, top_k)?
        .into_iter()
        .map(|h| QueryHit {
            id: h.id.0,
            score: h.score,
            text: h.text,
        })
        .collect();
    let evidence = build_evidence(&state.issuer, &state.model_identity, &state.scheme_identity)?;
    let attestation = AttestResponse {
        model_identity: evidence.model_identity,
        scheme_identity: evidence.scheme_identity,
        report_b64: B64.encode(evidence.report.as_deref().unwrap_or(&[])),
        vcek_cert_b64: B64.encode(evidence.vcek_cert.as_deref().unwrap_or(&[])),
    };
    Ok(Json(QueryResponse { hits, attestation }))
}

struct AppError(anyhow::Error);
impl<E: Into<anyhow::Error>> From<E> for AppError {
    fn from(e: E) -> Self {
        Self(e.into())
    }
}
impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        error!("request failed: {:#}", self.0);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("internal error: {:#}", self.0),
        )
            .into_response()
    }
}

/// A tiny deterministic embedder that hashes the input text into a unit
/// vector. Enough to demonstrate ingest+query end-to-end without depending
/// on safetensors or HF Hub. Real deployments swap in `GeloBertEmbedder` /
/// `GeloQwenEmbedder` once weights are present.
struct StubEmbedder {
    /// Free-form model identifier string from the runner config (e.g.
    /// `stub-model@v1`). Returned through `Embedder::model_identity` so
    /// the attestation evidence carries the same string a relying party
    /// would pin.
    model_identity: String,
}

impl StubEmbedder {
    fn from_model_id(model_id: &str) -> Self {
        Self {
            model_identity: model_id.to_string(),
        }
    }
}

impl Embedder for StubEmbedder {
    fn embed(&mut self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| text_to_vec(t)).collect())
    }
    fn model_identity(&self) -> &[u8] {
        self.model_identity.as_bytes()
    }
}

/// Hash text → 16-d unit vector. Same text ⇒ same vector ⇒ identical
/// queries land their corresponding chunk.
fn text_to_vec(text: &str) -> Vec<f32> {
    let mut h = Sha256::new();
    h.update(text.as_bytes());
    let digest = h.finalize();
    let mut v = vec![0f32; 16];
    for (i, chunk) in digest.chunks(2).take(16).enumerate() {
        let n = u16::from_le_bytes([chunk[0], chunk[1]]);
        v[i] = (n as f32) / 65535.0 - 0.5;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

#[derive(Clone)]
struct IdentityScheme;

impl EmbeddingEncryptionScheme for IdentityScheme {
    fn scheme_name(&self) -> &'static str {
        "identity"
    }
    fn encrypt_document(&mut self, embedding: &[f32]) -> anyhow::Result<EncryptedEmbedding> {
        Ok(EncryptedEmbedding {
            scheme: "identity",
            vector: embedding.to_vec(),
            nonce: vec![],
            original_dimension: embedding.len(),
        })
    }
    fn encrypt_query(&mut self, embedding: &[f32]) -> anyhow::Result<EncryptedEmbedding> {
        self.encrypt_document(embedding)
    }
    fn decrypt_document(&mut self, ciphertext: &EncryptedEmbedding) -> anyhow::Result<Vec<f32>> {
        Ok(ciphertext.vector.clone())
    }
}


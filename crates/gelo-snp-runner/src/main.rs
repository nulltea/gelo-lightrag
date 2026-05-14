//! `gelo-snp-runner` — production-style service binary for the GELO+SEV-SNP
//! deployment.
//!
//! Boots once at process start, parses [`SNP_MODE`] (production or mock),
//! wires up a [`GeloRagTwoPartyService`] with the SEV-SNP attestation
//! backend, and serves a minimal HTTP API. CAPRISE encryption happens
//! inside the CVM, with the key derived per-request from a two-party
//! HKDF (`user_x_sk` from the client + `tee_user_x_sk` held by the CVM)
//! — see `docs/prototype/caprise-two-party-kdf.md`.
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
//! - `POST /ingest`  → `{ tenant_id, user_x_sk, chunks: [{id, text}, …] }`
//! - `POST /query`   → `{ tenant_id, user_x_sk, text, top_k }` → ranked hits
//! - `POST /rotate`  → stub (501 Not Implemented), milestone M8

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use gelo_rag::{GeloRagTwoPartyService, NoopAttestationVerifier, TwoPartyError};
use rag_core::{ChunkId, DocumentChunk, Embedder, TenantId};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;
use zeroize::Zeroizing;

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

    let issuer = IssuerHandle::for_mode(mode)?;
    let service = GeloRagTwoPartyService::new(embedder, NoopAttestationVerifier);

    // M5: scheme_identity reported in REPORT_DATA composes the
    // runner-config string with the canonical KDF + CAPRISE digest from
    // the service. A relying party verifying the report can reproduce
    // this composition byte-for-byte.
    let scheme_identity_b = compose_scheme_identity(
        cfg.scheme_identity.as_bytes(),
        &service.scheme_identity(),
    );

    let state = AppState {
        service: Arc::new(Mutex::new(service)),
        issuer: Arc::new(issuer),
        model_identity: model_identity_b,
        scheme_identity: scheme_identity_b,
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/attest", get(attest))
        .route("/ingest", post(ingest))
        .route("/query", post(query))
        .route("/rotate", post(rotate))
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

/// Compose the runner-config `scheme_identity` string with the
/// canonical KDF + CAPRISE digest from the service:
///
/// ```text
/// REPORT_DATA[32..64] ← sha256(cfg_scheme_identity ‖ 0x00 ‖ service.scheme_identity())
/// ```
///
/// The single null separator prevents an attacker from producing two
/// distinct `(cfg, service_digest)` pairs with the same composition by
/// shifting bytes across the boundary.
fn compose_scheme_identity(cfg_bytes: &[u8], service_digest: &[u8; 32]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(cfg_bytes);
    hasher.update([0u8]);
    hasher.update(service_digest);
    hasher.finalize().to_vec()
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
    service: Arc<Mutex<GeloRagTwoPartyService<StubEmbedder, NoopAttestationVerifier>>>,
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

/// Base64-encoded 32-byte secret. Carries a manual `Debug` impl so a
/// panic backtrace or stray `tracing::debug!("{:?}", req)` cannot leak
/// `user_x_sk` to logs. The actual decoded bytes never reside in this
/// type — see [`UserXskB64::decode`].
#[derive(Deserialize)]
#[serde(transparent)]
struct UserXskB64(String);

impl std::fmt::Debug for UserXskB64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<redacted user_x_sk : base64-32B>")
    }
}

impl UserXskB64 {
    fn decode(&self) -> anyhow::Result<Zeroizing<[u8; 32]>> {
        // The raw decoded buffer is heap memory we don't control — wrap
        // it in `Zeroizing` so it wipes on drop even on the error path.
        let bytes = Zeroizing::new(
            B64.decode(self.0.as_bytes())
                .context("user_x_sk: base64 decode failed")?,
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
struct IngestRequest {
    tenant_id: String,
    user_x_sk: UserXskB64,
    chunks: Vec<IngestChunk>,
}

#[derive(Deserialize, Debug)]
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
    let tenant = TenantId::new(req.tenant_id);
    let user_x_sk = req.user_x_sk.decode()?;
    let n = req.chunks.len();
    let docs: Vec<DocumentChunk> = req
        .chunks
        .into_iter()
        .map(|c| DocumentChunk {
            id: ChunkId(c.id),
            text: c.text,
        })
        .collect();
    state
        .service
        .lock()
        .await
        .ingest_chunks_for(&tenant, user_x_sk, docs)?;
    Ok(Json(IngestResponse { ingested: n }))
}

#[derive(Deserialize, Debug)]
struct QueryRequest {
    tenant_id: String,
    user_x_sk: UserXskB64,
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
    let tenant = TenantId::new(req.tenant_id);
    let user_x_sk = req.user_x_sk.decode()?;
    let top_k = req.top_k.unwrap_or(5);
    let hits = state
        .service
        .lock()
        .await
        .query_for(&tenant, user_x_sk, &req.text, top_k)?
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

#[derive(Deserialize, Debug)]
struct RotateRequest {
    tenant_id: String,
    old_user_x_sk: UserXskB64,
    new_user_x_sk: UserXskB64,
}

/// M8 — stub. Returns 501 Not Implemented; the runner exposes the
/// route so clients can probe whether their CVM revision supports
/// rotation without parsing 404s.
async fn rotate(
    State(state): State<AppState>,
    Json(req): Json<RotateRequest>,
) -> Result<axum::response::Response, AppError> {
    let tenant = TenantId::new(req.tenant_id);
    let old = req.old_user_x_sk.decode()?;
    let new = req.new_user_x_sk.decode()?;
    let res = state
        .service
        .lock()
        .await
        .rotate_tenant(&tenant, old, new);
    // The service today returns `TwoPartyError::Inner("not implemented…")`
    // for any successful entry path. Map that to 501 explicitly so a
    // client doesn't get a 500 for "expected behaviour".
    match res {
        Err(TwoPartyError::UnknownTenant(t)) => Ok((
            StatusCode::GONE,
            format!("tenant {t} unknown — re-bootstrap"),
        )
            .into_response()),
        Err(TwoPartyError::Inner(e)) => Ok((
            StatusCode::NOT_IMPLEMENTED,
            format!("rotation not implemented: {e:#}"),
        )
            .into_response()),
        Ok(()) => Ok(StatusCode::NO_CONTENT.into_response()),
    }
}

/// Error type for HTTP handlers — maps every error variant to the
/// correct status code. `UnknownTenant` is 410 Gone per spec §12; the
/// "loud failure" contract that lets the client detect a CVM restart
/// without quietly re-encrypting under a fresh `tee_user_x_sk`.
struct AppError(AppErrorKind);

enum AppErrorKind {
    UnknownTenant(TenantId),
    Other(anyhow::Error),
}

impl From<TwoPartyError> for AppError {
    fn from(e: TwoPartyError) -> Self {
        match e {
            TwoPartyError::UnknownTenant(t) => Self(AppErrorKind::UnknownTenant(t)),
            TwoPartyError::Inner(inner) => Self(AppErrorKind::Other(inner)),
        }
    }
}

impl From<anyhow::Error> for AppError {
    fn from(e: anyhow::Error) -> Self {
        Self(AppErrorKind::Other(e))
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        match self.0 {
            AppErrorKind::UnknownTenant(t) => {
                info!(tenant = %t, "unknown tenant — returning 410 Gone");
                (
                    StatusCode::GONE,
                    format!(
                        "tenant {t} unknown — re-bootstrap the tenant \
                         (CVM may have restarted)"
                    ),
                )
                    .into_response()
            }
            AppErrorKind::Other(e) => {
                error!("request failed: {:#}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("internal error: {:#}", e),
                )
                    .into_response()
            }
        }
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

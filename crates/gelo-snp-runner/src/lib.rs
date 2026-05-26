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
use gelo_reranker::output::EncryptedRerankBundle;
use gelo_reranker::service::{RerankCandidate, RerankError, RerankRequest, RerankService};
use gelo_reranker::session::{QueryId, SessionKey, SessionKeyPolicy};
use rag_core::{ChunkId, DocumentChunk, Embedder, TenantId};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;
use zeroize::Zeroizing;

mod compass;
mod config;
mod evidence;
pub mod extraction;
mod issuer;
pub mod lightrag_routes;

use config::RunnerConfig;
use evidence::{IssuerHandle, build_evidence};
use extraction::ExtractionHandles;
use gelo_gpu_wgpu::WgpuVulkanEngine;
use lightrag_routes::{ExtractAndBuildState, LightRagServiceHandle};

/// The concrete offload engine used by the runner. GPU only —
/// `RayonCpuEngine` is deprecated and never instantiated here. See
/// `feedback_no_rayon_cpu_engine.md`.
pub type RunnerEngine = WgpuVulkanEngine;

/// Boot the runner: load config, initialise tracing, warm-load any
/// configured extraction models, then serve HTTP until shutdown.
///
/// Factored out of the `main` entrypoint so integration tests under
/// `tests/` can exercise the router (via the public surface of this
/// crate) without spawning a process.
pub async fn run() -> Result<()> {
    init_tracing();
    let mode = gelo_tee_sev_snp::runtime_mode::from_env()
        .context("parsing SNP_MODE from environment")?;
    info!("gelo-snp-runner starting; mode = {mode}");

    let cfg = RunnerConfig::load()
        .context("loading runner config (/etc/gelo-snp/runner.toml or $GELO_SNP_RUNNER_CONFIG)")?;
    info!(
        "loaded config: listen={} scheme_identity={:?} embedder={:?} compass_backend={:?}",
        cfg.listen, cfg.scheme_identity, cfg.embedder, cfg.compass_backend_url
    );
    if cfg.compass_backend_url.is_none() {
        info!(
            "compass_backend_url is unset — Compass-backed indices will be \
             refused. Set [compass_backend_url] in runner.toml or env to a \
             reachable compass-storage-server URL before M6 ships."
        );
    }

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

    // Optional warm-load of the extraction LLM + description embedder.
    // Both paths must be set to enable `/lightrag/extract_and_build`;
    // otherwise the route returns 503. Both models share one
    // `WgpuVulkanEngine` (cloned via `clone_shared` so each runtime
    // gets its own handle into the same Arc-backed weight cache).
    let extraction = match (
        cfg.extraction_decoder_path.as_deref(),
        cfg.extraction_embedder_path.as_deref(),
    ) {
        (Some(dec_path), Some(emb_path)) => {
            info!(
                "warm-loading extraction models on Wgpu engine: decoder={} embedder={}",
                dec_path.display(),
                emb_path.display()
            );
            // Each model needs its OWN engine instance — the engine's
            // `WeightHandle` namespace is per-instance, so sharing one
            // engine across two distinct models would have the embedder
            // overwrite the decoder's layer-0 Q/K/V/… cache and
            // produce shape-mismatch errors at the first matmul.
            let dec_engine = WgpuVulkanEngine::new_fp16()
                .context("initialising WgpuVulkanEngine (fp16, decoder)")?;
            let emb_engine = WgpuVulkanEngine::new_fp16()
                .context("initialising WgpuVulkanEngine (fp16, embedder)")?;
            let decoder = extraction::DecoderRuntime::<RunnerEngine>::from_dir(
                dec_path,
                dec_engine,
            )
            .context("loading extraction decoder")?;
            let embedder = extraction::GeloDescriptionEmbedder::<RunnerEngine>::from_dir(
                emb_path,
                emb_engine,
            )
            .context("loading description embedder")?;
            info!(
                "extraction models loaded; embedding_dim={}",
                embedder.dim()
            );
            Some(ExtractionHandles {
                decoder: Arc::new(std::sync::Mutex::new(decoder)),
                embedder: Arc::new(std::sync::Mutex::new(embedder)),
            })
        }
        (None, None) => {
            info!("extraction paths unset — /lightrag/extract_and_build will return 503");
            None
        }
        _ => {
            info!(
                "only one of extraction_decoder_path / extraction_embedder_path set; \
                 BOTH are required to enable extraction — route will return 503"
            );
            None
        }
    };

    let state = AppState {
        service: Arc::new(Mutex::new(service)),
        issuer: Arc::new(issuer),
        model_identity: model_identity_b,
        scheme_identity: scheme_identity_b,
        // M5 ships without a loaded reranker model; the /rerank route
        // surfaces 501 until R6/M8 wires a concrete model loader. The
        // route is registered so clients can probe for support.
        reranker: None,
        lightrag: Arc::new(lightrag_private::LightRagTwoPartyService::new()),
        extraction,
    };

    let app = build_router(state);

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

/// Box-erased reranker capability. The concrete service is one of
/// `gelo_reranker::CrossEncoderRerankService` or
/// `gelo_reranker::CausalDiscriminatorRerankService` constructed at
/// boot time. `None` means the runner was started without a reranker
/// model — `/rerank` returns 501.
type RerankerHandle = Option<Arc<Mutex<Box<dyn RerankService + Send>>>>;

#[derive(Clone)]
struct AppState {
    service: Arc<Mutex<GeloRagTwoPartyService<StubEmbedder, NoopAttestationVerifier>>>,
    issuer: Arc<IssuerHandle>,
    model_identity: Vec<u8>,
    scheme_identity: Vec<u8>,
    reranker: RerankerHandle,
    /// M8.0 — per-tenant LightRAG service. Held alongside the
    /// existing CAPRISE service; the `/lightrag/*` routes consume it.
    lightrag: LightRagServiceHandle,
    /// M8.x — warm-loaded extraction decoder + description embedder,
    /// both bound to a shared `WgpuVulkanEngine` for masked-GPU
    /// inference. `None` ⇒ `/lightrag/extract_and_build` returns 503.
    extraction: Option<ExtractionHandles<RunnerEngine>>,
}

/// Build the HTTP router for a given `AppState`. Lifted out of `run`
/// so inline tests can construct a state with a real reranker without
/// spawning a full process. Integration tests under `tests/` build a
/// narrower router directly from the public route handlers in
/// [`lightrag_routes`].
fn build_router(state: AppState) -> Router {
    // The lightrag routes consume a different state slice
    // (`LightRagServiceHandle`) than the existing embedder routes.
    // Build them as a sub-router with their own state, then merge.
    let lightrag_sub = Router::new()
        .route("/lightrag/ingest", post(lightrag_routes::ingest))
        .route("/lightrag/query", post(lightrag_routes::query))
        .route("/lightrag/attest", get(lightrag_routes::attest))
        .with_state(state.lightrag.clone());

    // The extract route consumes a third state slice: it needs both
    // the extraction handles AND the lightrag service so it can run
    // extraction then immediately ingest the resulting KG.
    let extract_sub = Router::new()
        .route(
            "/lightrag/extract_and_build",
            post(lightrag_routes::extract_and_build),
        )
        .with_state(ExtractAndBuildState {
            extraction: state.extraction.clone(),
            lightrag: state.lightrag.clone(),
        });

    Router::new()
        .route("/health", get(health))
        .route("/attest", get(attest))
        .route("/ingest", post(ingest))
        .route("/query", post(query))
        .route("/rotate", post(rotate))
        .route("/rerank", post(rerank))
        .with_state(state)
        .merge(lightrag_sub)
        .merge(extract_sub)
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

/// Per-request session secret used to derive the rerank-output
/// AES-GCM key. For M5 the client supplies a 32-byte secret directly
/// (paralleling `user_x_sk` on the embedding path). The full
/// attestation-bound ECDH key agreement lands in M5.9.
#[derive(Deserialize)]
#[serde(transparent)]
struct SessionSecretB64(String);

impl std::fmt::Debug for SessionSecretB64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<redacted session_secret : base64-32B>")
    }
}

impl SessionSecretB64 {
    fn decode(&self) -> anyhow::Result<Zeroizing<Vec<u8>>> {
        let bytes = Zeroizing::new(
            B64.decode(self.0.as_bytes())
                .context("session_secret: base64 decode failed")?,
        );
        if bytes.len() < 16 {
            anyhow::bail!(
                "session_secret must be ≥ 16 bytes after base64 decode (got {})",
                bytes.len()
            );
        }
        Ok(bytes)
    }
}

#[derive(Deserialize, Debug)]
struct RerankCandidateJson {
    id: String,
    text: String,
}

#[derive(Deserialize, Debug)]
struct RerankRequestJson {
    /// 32-byte HKDF root secret. Stand-in until M5.9's ECDH handshake.
    session_secret: SessionSecretB64,
    /// Caller-supplied unique tag, replays produce reused AES-GCM
    /// nonces under the same key — guard against this on the client.
    query_id_b64: String,
    query: String,
    candidates: Vec<RerankCandidateJson>,
    top_k: usize,
    k_max: usize,
}

#[derive(Serialize)]
struct RerankBundleItemJson {
    nonce_b64: String,
    ciphertext_b64: String,
}

#[derive(Serialize)]
struct RerankResponse {
    /// Always `"aes-256-gcm.v1"` (matches `EncryptedRerankBundle::scheme`).
    scheme: String,
    items: Vec<RerankBundleItemJson>,
    family: String,
    model_identity_b64: String,
}

async fn rerank(
    State(state): State<AppState>,
    Json(req): Json<RerankRequestJson>,
) -> Result<axum::response::Response, AppError> {
    let Some(handle) = state.reranker.clone() else {
        return Ok((
            StatusCode::NOT_IMPLEMENTED,
            "rerank not configured on this CVM — load a reranker model and re-attest",
        )
            .into_response());
    };
    let session_secret = req.session_secret.decode()?;
    let session = SessionKey::derive(&session_secret, SessionKeyPolicy::V1);
    let query_id_bytes = B64
        .decode(req.query_id_b64.as_bytes())
        .context("query_id_b64: base64 decode failed")?;
    let query_id = QueryId::new(query_id_bytes);
    let candidates: Vec<RerankCandidate> = req
        .candidates
        .into_iter()
        .map(|c| RerankCandidate {
            chunk_id: ChunkId(c.id),
            text: c.text,
        })
        .collect();
    let request = RerankRequest {
        query: &req.query,
        candidates: &candidates,
        top_k: req.top_k,
        k_max: req.k_max,
        query_id,
    };
    let bundle = {
        let mut svc = handle.lock().await;
        svc.rerank(&session, &request).map_err(|e| match e {
            RerankError::InvalidRequest(msg) => AppError(AppErrorKind::Other(anyhow::anyhow!(msg))),
            other => AppError(AppErrorKind::Other(anyhow::anyhow!(other))),
        })?
    };
    let (family, model_identity_b64) = {
        let svc = handle.lock().await;
        (svc.family().to_string(), B64.encode(svc.model_identity()))
    };
    let resp = RerankResponse {
        scheme: bundle.scheme.to_string(),
        items: bundle
            .items
            .into_iter()
            .map(|i| RerankBundleItemJson {
                nonce_b64: B64.encode(&i.nonce),
                ciphertext_b64: B64.encode(&i.ciphertext),
            })
            .collect(),
        family,
        model_identity_b64,
    };
    Ok(Json(resp).into_response())
}

// Suppress dead-code warning when an integration test exercises
// `EncryptedRerankBundle::open` on the response — the production
// runner emits but never consumes a bundle.
#[allow(dead_code)]
fn _bundle_type_witness() -> EncryptedRerankBundle {
    unreachable!()
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
pub struct AppError(AppErrorKind);

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

impl From<lightrag_private::LightRagServiceError> for AppError {
    fn from(e: lightrag_private::LightRagServiceError) -> Self {
        use lightrag_private::LightRagServiceError as L;
        match e {
            L::UnknownTenant(t) => Self(AppErrorKind::UnknownTenant(t)),
            L::Store(inner) => Self(AppErrorKind::Other(anyhow::anyhow!(inner))),
            L::Inner(inner) => Self(AppErrorKind::Other(inner)),
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

#[cfg(all(test, feature = "mock"))]
mod tests {
    use super::*;
    use axum::body::{Body, to_bytes};
    use axum::http::{Method, Request};
    use ndarray::{Array1, Array2};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};
    use tower::ServiceExt;

    use gelo_embedder::bert::config::BertConfig;
    use gelo_embedder::bert::weights::{BertLayerWeights, BertWeights};
    use gelo_embedder::common::tokenizer::HfTokenizer;
    use gelo_protocol::rng::MaskSeed;
    use gelo_protocol::{
        GpuOffloadEngine, InProcessTrustedExecutor, RayonCpuEngine, WeightHandle, WeightKind,
    };
    use gelo_reranker::cross_encoder::CrossEncoderRerankService;
    use gelo_reranker::head::ClassifierHead;
    use gelo_reranker::session::{QueryId, SessionKey, SessionKeyPolicy};

    fn tiny_cfg() -> BertConfig {
        BertConfig {
            vocab_size: 64,
            hidden_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            intermediate_size: 64,
            max_position_embeddings: 32,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            hidden_act: "gelu".into(),
            max_seq_len: 32,
            skip_first_layers: 0,
            skip_last_layer: false,
            use_out_attn_mult: false,
            out_attn_mult_min_seq_len: None,
        }
    }

    fn rand2(rows: usize, cols: usize, rng: &mut impl rand::RngCore, scale: f32) -> Array2<f32> {
        let normal = StandardNormal;
        Array2::from_shape_fn((rows, cols), |_| {
            <StandardNormal as Distribution<f32>>::sample(&normal, rng) * scale
        })
    }
    fn rand1(n: usize, rng: &mut impl rand::RngCore, scale: f32) -> Array1<f32> {
        let normal = StandardNormal;
        Array1::from_shape_fn(n, |_| {
            <StandardNormal as Distribution<f32>>::sample(&normal, rng) * scale
        })
    }

    fn synth_weights(cfg: &BertConfig, rng: &mut impl rand::RngCore) -> BertWeights {
        let d = cfg.hidden_size;
        let f = cfg.intermediate_size;
        let layers = (0..cfg.num_hidden_layers)
            .map(|_| BertLayerWeights {
                wq: rand2(d, d, rng, 0.05),
                bq: rand1(d, rng, 0.01),
                wk: rand2(d, d, rng, 0.05),
                bk: rand1(d, rng, 0.01),
                wv: rand2(d, d, rng, 0.05),
                bv: rand1(d, rng, 0.01),
                wo: rand2(d, d, rng, 0.05),
                bo: rand1(d, rng, 0.01),
                attn_ln_w: Array1::from_elem(d, 1.0),
                attn_ln_b: Array1::zeros(d),
                w_ffn_up: rand2(d, f, rng, 0.05),
                b_ffn_up: rand1(f, rng, 0.01),
                w_ffn_down: rand2(f, d, rng, 0.05),
                b_ffn_down: rand1(d, rng, 0.01),
                ffn_ln_w: Array1::from_elem(d, 1.0),
                ffn_ln_b: Array1::zeros(d),
            })
            .collect();
        BertWeights {
            word_embedding: rand2(cfg.vocab_size, d, rng, 0.05),
            position_embedding: rand2(cfg.max_position_embeddings, d, rng, 0.05),
            token_type_embedding: rand2(cfg.type_vocab_size, d, rng, 0.0),
            embeddings_ln_w: Array1::from_elem(d, 1.0),
            embeddings_ln_b: Array1::zeros(d),
            layers,
            model_identity: [0u8; 32],
        }
    }

    fn provision<E: GpuOffloadEngine>(w: &BertWeights, cfg: &BertConfig, e: &mut E) {
        for (li, layer) in w.layers.iter().enumerate() {
            if !cfg.offload_layer(li) {
                continue;
            }
            let li16 = li as u16;
            e.register_weight(WeightHandle::new(li16, WeightKind::Q), layer.wq.view()).unwrap();
            e.register_weight(WeightHandle::new(li16, WeightKind::K), layer.wk.view()).unwrap();
            e.register_weight(WeightHandle::new(li16, WeightKind::V), layer.wv.view()).unwrap();
            e.register_weight(WeightHandle::new(li16, WeightKind::O), layer.wo.view()).unwrap();
            e.register_weight(WeightHandle::new(li16, WeightKind::FfnUp), layer.w_ffn_up.view()).unwrap();
            e.register_weight(WeightHandle::new(li16, WeightKind::FfnDown), layer.w_ffn_down.view()).unwrap();
        }
    }

    const STUB_TOKENIZER_JSON: &str = r#"{
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [],
      "normalizer": null,
      "pre_tokenizer": { "type": "Whitespace" },
      "post_processor": null,
      "decoder": null,
      "model": {
        "type": "WordLevel",
        "vocab": { "[UNK]": 0 },
        "unk_token": "[UNK]"
      }
    }"#;

    fn stub_tokenizer() -> HfTokenizer {
        let tmp = std::env::temp_dir().join(format!(
            "gelo-snp-runner-test-tok-{}-{}.json",
            std::process::id(),
            rand::random::<u32>()
        ));
        std::fs::write(&tmp, STUB_TOKENIZER_JSON).unwrap();
        let tok = HfTokenizer::from_file(&tmp).unwrap();
        let _ = std::fs::remove_file(&tmp);
        tok
    }

    fn build_test_state() -> AppState {
        // Embedder side — reuse the StubEmbedder.
        let embedder = StubEmbedder::from_model_id("test-model@v1");
        let service = GeloRagTwoPartyService::new(embedder, NoopAttestationVerifier);

        // Mock issuer (the runner's `mock` feature gates this module).
        let issuer = IssuerHandle::for_mode(gelo_tee_sev_snp::runtime_mode::RuntimeMode::Mock)
            .expect("mock issuer should construct");

        // Synthetic reranker.
        let cfg = tiny_cfg();
        let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
        let weights = std::sync::Arc::new(synth_weights(&cfg, &mut rng));
        let head = ClassifierHead::from_arrays(
            rand2(cfg.hidden_size, cfg.hidden_size, &mut rng, 0.05),
            rand1(cfg.hidden_size, &mut rng, 0.0),
            rand2(cfg.hidden_size, 1, &mut rng, 0.05),
            rand1(1, &mut rng, 0.0),
        );
        let mut engine = RayonCpuEngine::new();
        provision(&weights, &cfg, &mut engine);
        let exec = InProcessTrustedExecutor::with_seed(engine, MaskSeed::from_bytes([19u8; 32]));
        let reranker_svc =
            CrossEncoderRerankService::new(cfg, stub_tokenizer(), weights, head, exec).unwrap();
        let reranker_box: Box<dyn RerankService + Send> = Box::new(reranker_svc);

        AppState {
            service: Arc::new(Mutex::new(service)),
            issuer: Arc::new(issuer),
            model_identity: b"test-model-identity".to_vec(),
            scheme_identity: b"test-scheme-identity".to_vec(),
            reranker: Some(Arc::new(Mutex::new(reranker_box))),
            lightrag: Arc::new(lightrag_private::LightRagTwoPartyService::new()),
            extraction: None,
        }
    }

    #[tokio::test]
    async fn rerank_returns_501_when_unconfigured() {
        let mut state = build_test_state();
        state.reranker = None;
        let app = build_router(state);
        let body = serde_json::json!({
            "session_secret": B64.encode([1u8; 32]),
            "query_id_b64": B64.encode(b"qid"),
            "query": "q",
            "candidates": [{"id": "a", "text": "alpha"}],
            "top_k": 1,
            "k_max": 4,
        });
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/rerank")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
    }

    #[tokio::test]
    async fn rerank_emits_padded_bundle_when_configured() {
        let state = build_test_state();
        let app = build_router(state);

        let session_secret = [0xab_u8; 32];
        let query_id = b"qid-001".to_vec();
        let body = serde_json::json!({
            "session_secret": B64.encode(session_secret),
            "query_id_b64": B64.encode(&query_id),
            "query": "what is rust",
            "candidates": [
                {"id": "c-alpha", "text": "Rust is a systems language"},
                {"id": "c-beta",  "text": "Memory safety without GC"},
                {"id": "c-gamma", "text": "Borrow checker"},
            ],
            "top_k": 2,
            "k_max": 6,
        });
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/rerank")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed["scheme"], "aes-256-gcm.v1");
        assert_eq!(parsed["family"], "cross-encoder");
        let items = parsed["items"].as_array().unwrap();
        assert_eq!(items.len(), 6, "must always emit exactly k_max items");

        // Reconstruct the bundle client-side and decrypt to confirm
        // the response is structurally correct.
        let session = SessionKey::derive(
            &zeroize::Zeroizing::new(session_secret.to_vec()),
            SessionKeyPolicy::V1,
        );
        let qkey = session.derive_query_key(&QueryId::new(query_id));
        let recon = gelo_reranker::output::EncryptedRerankBundle {
            scheme: "aes-256-gcm.v1",
            items: items
                .iter()
                .map(|i| gelo_reranker::output::EncryptedRerankItem {
                    nonce: B64.decode(i["nonce_b64"].as_str().unwrap()).unwrap(),
                    ciphertext: B64.decode(i["ciphertext_b64"].as_str().unwrap()).unwrap(),
                })
                .collect(),
        };
        let opened = recon.open(&qkey).expect("client can open with derived qkey");
        assert_eq!(opened.len(), 2);
        let known: std::collections::HashSet<String> =
            ["c-alpha", "c-beta", "c-gamma"].iter().map(|s| s.to_string()).collect();
        for it in &opened {
            assert!(known.contains(&it.chunk_id));
        }
    }

    /// M8.0 acceptance — POST /lightrag/ingest, then POST
    /// /lightrag/query, end-to-end through the runner's HTTP surface.
    /// Pins the wiring; the underlying retrieval behaviour is already
    /// pinned by `lightrag-private/tests/local_kg_query.rs`.
    #[tokio::test]
    async fn lightrag_ingest_then_query_round_trips_via_http() {
        let state = build_test_state();
        let app = build_router(state);

        // Build a small synthetic KG (4 entities, 2 relations, 4 chunks).
        // Embeddings are 16-d deterministic basis vectors so a query
        // for entity-0 reliably lands entity-0 in the top-k.
        fn one_hot(i: usize, dim: usize) -> Vec<f32> {
            let mut v = vec![0.0f32; dim];
            v[i % dim] = 1.0;
            v
        }
        let dim = 16usize;

        let entity_payload = serde_json::json!({
            "tenant_id": "m8-tenant",
            "user_x_sk": B64.encode([0xAA_u8; 32]),
            "dim": dim,
            "extracted_kg": {
                "chunks": (0..4).map(|i| serde_json::json!({
                    "id": format!("chunk-{i}"),
                    "text": format!("body of chunk {i}"),
                    "embedding": one_hot(i, dim),
                })).collect::<Vec<_>>(),
                "entities": (0..4).map(|i| serde_json::json!({
                    "name": format!("entity-{i}"),
                    "description": format!("desc {i}"),
                    "embedding": one_hot(i, dim),
                    "source_chunks": [format!("chunk-{i}")],
                })).collect::<Vec<_>>(),
                "relations": [
                    {
                        "src": "entity-0",
                        "tgt": "entity-1",
                        "description": "e0—e1",
                        "embedding": one_hot(0, dim),
                        "source_chunks": ["chunk-0"]
                    },
                    {
                        "src": "entity-2",
                        "tgt": "entity-3",
                        "description": "e2—e3",
                        "embedding": one_hot(2, dim),
                        "source_chunks": ["chunk-2"]
                    }
                ]
            }
        });
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/lightrag/ingest")
                    .header("content-type", "application/json")
                    .body(Body::from(entity_payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "ingest failed");
        let body = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["ingested"]["entities"], 4);
        assert_eq!(v["ingested"]["relations"], 2);
        assert_eq!(v["ingested"]["chunks"], 4);

        // Query for entity-0 by passing its own embedding.
        let q_payload = serde_json::json!({
            "tenant_id": "m8-tenant",
            "ll_query_embedding": one_hot(0, dim),
            "session_nonce_b64": B64.encode(b"nonce-001"),
            "top_k_entities": 3,
            "top_k_chunks_per_entity": 1,
        });
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/lightrag/query")
                    .header("content-type", "application/json")
                    .body(Body::from(q_payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "query failed");
        let body = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let entities = v["entities"].as_array().unwrap();
        let entity_names: Vec<String> = entities
            .iter()
            .map(|e| e.as_str().unwrap().to_string())
            .collect();
        assert!(
            entity_names.contains(&"entity-0".to_string()),
            "entity-0 missing from query result: {entity_names:?}"
        );
        let chunks = v["chunks"].as_array().unwrap();
        assert!(!chunks.is_empty(), "no chunks returned");
        for c in chunks {
            assert!(c["text"].as_str().unwrap().contains("body of chunk"));
        }
        let ctx_string = v["context_string"].as_str().unwrap();
        assert!(ctx_string.contains("# Entities"));
        assert!(ctx_string.contains("# Source chunks"));
    }

    /// M7.2 — hybrid mode threads through both embedding axes.
    #[tokio::test]
    async fn lightrag_hybrid_mode_threads_hl_and_ll() {
        let state = build_test_state();
        let app = build_router(state);

        fn one_hot(i: usize, dim: usize) -> Vec<f32> {
            let mut v = vec![0.0f32; dim];
            v[i % dim] = 1.0;
            v
        }
        let dim = 16usize;

        // Ingest the same 4/2/4 KG as the Local-mode test.
        let kg = serde_json::json!({
            "tenant_id": "hybrid-tenant",
            "user_x_sk": B64.encode([0xBB_u8; 32]),
            "dim": dim,
            "extracted_kg": {
                "chunks": (0..4).map(|i| serde_json::json!({
                    "id": format!("chunk-{i}"),
                    "text": format!("body of chunk {i}"),
                    "embedding": one_hot(i, dim),
                })).collect::<Vec<_>>(),
                "entities": (0..4).map(|i| serde_json::json!({
                    "name": format!("entity-{i}"),
                    "description": format!("desc {i}"),
                    "embedding": one_hot(i, dim),
                    "source_chunks": [format!("chunk-{i}")],
                })).collect::<Vec<_>>(),
                "relations": [
                    {
                        "src": "entity-0",
                        "tgt": "entity-1",
                        "description": "e0—e1",
                        "embedding": one_hot(0, dim),
                        "source_chunks": ["chunk-0"]
                    },
                    {
                        "src": "entity-2",
                        "tgt": "entity-3",
                        "description": "e2—e3",
                        "embedding": one_hot(2, dim),
                        "source_chunks": ["chunk-2"]
                    }
                ]
            }
        });
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/lightrag/ingest")
                    .header("content-type", "application/json")
                    .body(Body::from(kg.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Hybrid query: ll points to entity-3, hl points to the
        // entity-0—entity-1 relation. The merged result should
        // include entity-3 (from ll search) AND entity-0/entity-1
        // (from the hl relation's endpoint fan-out).
        let q = serde_json::json!({
            "tenant_id": "hybrid-tenant",
            "ll_query_embedding": one_hot(3, dim),
            "hl_query_embedding": one_hot(0, dim),
            "mode": "hybrid",
            "session_nonce_b64": B64.encode(b"nonce-hybrid"),
            "top_k_entities": 2,
            "top_k_relations": 1,
            "top_k_chunks_per_entity": 1,
        });
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/lightrag/query")
                    .header("content-type", "application/json")
                    .body(Body::from(q.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let entity_names: Vec<String> = v["entities"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e.as_str().unwrap().to_string())
            .collect();
        // ll path: entity-3 must be present (its own embedding query).
        assert!(
            entity_names.contains(&"entity-3".to_string()),
            "ll hit missing: {entity_names:?}"
        );
        // hl path: the entity-0—entity-1 relation's endpoints must
        // surface even though we only ll-queried for entity-3.
        assert!(
            entity_names.contains(&"entity-0".to_string())
                || entity_names.contains(&"entity-1".to_string()),
            "hl-relation endpoints missing: {entity_names:?}"
        );
    }

    /// The `/lightrag/extract_and_build` route is registered
    /// unconditionally — when no extraction models were warm-loaded,
    /// the handler returns 503 so clients can probe for support.
    #[tokio::test]
    async fn lightrag_extract_and_build_returns_503_when_unconfigured() {
        // `build_test_state` already sets `extraction: None`.
        let state = build_test_state();
        let app = build_router(state);
        let body = serde_json::json!({
            "tenant_id": "probe-tenant",
            "user_x_sk": B64.encode([0x11u8; 32]),
            "document_text": "Alice met Bob in Paris.",
        });
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/lightrag/extract_and_build")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn lightrag_query_unknown_tenant_returns_410_gone() {
        // 410 Gone matches the existing /query route's loud-failure
        // contract: lets the client detect a CVM restart and re-
        // bootstrap rather than silently re-encrypting under a fresh
        // tee_user_x_sk.
        let state = build_test_state();
        let app = build_router(state);
        let q_payload = serde_json::json!({
            "tenant_id": "no-such-tenant",
            "ll_query_embedding": vec![0.0f32; 16],
            "session_nonce_b64": B64.encode(b"n"),
            "top_k_entities": 1,
            "top_k_chunks_per_entity": 1,
        });
        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/lightrag/query")
                    .header("content-type", "application/json")
                    .body(Body::from(q_payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::GONE);
    }
}

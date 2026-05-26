//! End-to-end integration test for `POST /lightrag/extract_and_build`.
//!
//! Real-weights round-trip: a document text is chunked + extracted +
//! ingested inside the CVM, then queried via `/lightrag/query`. The
//! assertion is wiring-only: a query for "Alice" must surface the
//! entity named Alice in the response. Extraction quality bounds are
//! out of scope for this test.
//!
//! Gated on `PRIVATE_RAG_MODEL_DIR` containing two subdirs:
//! - `qwen3-4b-decoder/` — `config.json` + `tokenizer.json` + safetensors
//! - `qwen3-emb-0.6b/`   — same shape
//!
//! Marked `#[ignore]` because each chunk takes ~3-7 min on the dev box
//! (TTFT ~10 s + ~0.5 tok/s, ~100-200 output tokens before
//! `<|COMPLETE|>` on a 1-sentence fixture). Run with:
//!
//! ```text
//! PRIVATE_RAG_MODEL_DIR=/path/to/models \
//!     cargo test -p gelo-snp-runner --test lightrag_extract_and_build \
//!         --release -- --ignored
//! ```

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Method, Request, StatusCode},
    routing::post,
};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use lightrag_private::LightRagTwoPartyService;
use lightrag_private::extract::DescriptionEmbedder;
use serde_json::{Value, json};
use tower::ServiceExt;

use gelo_gpu_wgpu::WgpuVulkanEngine;
use gelo_snp_runner::extraction::{DecoderRuntime, ExtractionHandles, GeloDescriptionEmbedder};
use gelo_snp_runner::lightrag_routes::{self, ExtractAndBuildState};
use gelo_snp_runner::RunnerEngine;

#[tokio::test]
#[ignore = "requires PRIVATE_RAG_MODEL_DIR with Qwen3-4B + Qwen3-Embedding-0.6B weights"]
async fn extract_and_build_round_trips_alice() {
    let model_dir = std::env::var("PRIVATE_RAG_MODEL_DIR").expect(
        "set PRIVATE_RAG_MODEL_DIR to a dir with qwen3-4b-decoder/ and qwen3-emb-0.6b/",
    );
    let model_dir = PathBuf::from(model_dir);
    let decoder_dir = model_dir.join("qwen3-4b-decoder");
    let embedder_dir = model_dir.join("qwen3-emb-0.6b");

    // Each model gets its own engine instance — the engine's weight
    // namespace is per-instance, sharing one engine across decoder +
    // embedder would have layer-0 handles collide.
    let dec_engine = WgpuVulkanEngine::new_fp16().expect("init wgpu engine (decoder)");
    let emb_engine = WgpuVulkanEngine::new_fp16().expect("init wgpu engine (embedder)");
    let decoder = DecoderRuntime::<RunnerEngine>::from_dir(&decoder_dir, dec_engine)
        .expect("load extraction decoder");
    let embedder = GeloDescriptionEmbedder::<RunnerEngine>::from_dir(&embedder_dir, emb_engine)
        .expect("load embedder");
    let dim = embedder.dim();

    let extraction_handles = ExtractionHandles {
        decoder: Arc::new(Mutex::new(decoder)),
        embedder: Arc::new(Mutex::new(embedder)),
    };
    let lightrag: Arc<LightRagTwoPartyService> = Arc::new(LightRagTwoPartyService::new());

    // Minimal two-route router — no need to spin up the full
    // AppState. The extract route owns its own state slice; the
    // query route uses the LightRAG service handle directly.
    let app: Router = Router::new()
        .route(
            "/lightrag/extract_and_build",
            post(lightrag_routes::extract_and_build),
        )
        .with_state(ExtractAndBuildState {
            extraction: Some(extraction_handles.clone()),
            lightrag: lightrag.clone(),
        })
        .merge(
            Router::new()
                .route("/lightrag/query", post(lightrag_routes::query))
                .with_state(lightrag.clone()),
        );

    let tenant = "extract-tenant";
    let user_x_sk = B64.encode([0xCC_u8; 32]);
    let body = json!({
        "tenant_id": tenant,
        "user_x_sk": user_x_sk,
        "document_text": "Alice met Bob in Paris to discuss the new project at Acme Corp.",
        "max_tokens_per_chunk": 512,
    });
    let resp = app
        .clone()
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
    assert_eq!(resp.status(), StatusCode::OK, "extract_and_build failed");
    let body_bytes = to_bytes(resp.into_body(), 1 << 22).await.unwrap();
    let v: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert!(
        v["ingested"]["entities"].as_u64().unwrap_or(0) >= 1,
        "no entities extracted: {v}"
    );
    assert_eq!(v["extraction"]["embedding_dim"], dim as u64);

    // Compute the query embedding via the same embedder we loaded at
    // setup. Lock briefly.
    let q_emb: Vec<f32> = {
        let mut g = extraction_handles
            .embedder
            .lock()
            .expect("embedder mutex");
        let mut v = g.embed_batch(&["Alice".to_string()]).expect("embed query");
        v.pop().expect("embedder returned no rows")
    };

    let q_body = json!({
        "tenant_id": tenant,
        "ll_query_embedding": q_emb,
        "session_nonce_b64": B64.encode(b"nonce-extract"),
        "top_k_entities": 5,
        "top_k_chunks_per_entity": 1,
    });
    let resp = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/lightrag/query")
                .header("content-type", "application/json")
                .body(Body::from(q_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "query failed");
    let body_bytes = to_bytes(resp.into_body(), 1 << 22).await.unwrap();
    let v: Value = serde_json::from_slice(&body_bytes).unwrap();
    let names: Vec<String> = v["entities"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e.as_str().unwrap().to_string())
        .collect();
    assert!(
        names.iter().any(|n| n.eq_ignore_ascii_case("alice")),
        "Alice missing from entity results: {names:?}"
    );
}

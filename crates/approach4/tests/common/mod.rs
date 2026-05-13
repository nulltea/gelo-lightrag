//! Shared helpers for the Approach 4 integration/perf tests.
//!
//! * `OllamaEmbedder` — HTTP client against a locally running Ollama instance.
//! * `TokenBasedChunker` — simplified port of edgequake-pipeline's
//!   `TokenBasedChunking` (char-count-based splitting with overlap and
//!   separator-aware split-point search).
//! * `fetch_document_markdown` — chains the edgequake `GET /documents/{id}` and
//!   `GET /documents/pdf/{pdf_id}/content` endpoints and caches the markdown
//!   next to the `target/` directory so repeated runs don't re-hit the API.
//! * `beir` — loader for BEIR IR benchmark datasets (M7.3).
//! * `embed_cache::CachingEmbedder` — disk-cached Embedder wrapper (M7.3).

#![allow(dead_code)]

pub mod beir;
pub mod embed_cache;

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use rag_core::Embedder;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ─────────────────────────────────────────────────────────────────────────────
// Ollama embedder
// ─────────────────────────────────────────────────────────────────────────────

pub struct OllamaEmbedder {
    client: Client,
    base_url: String,
    model: String,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

impl OllamaEmbedder {
    pub fn new(model: impl Into<String>) -> Result<Self> {
        let base_url =
            env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()?;
        Ok(Self {
            client,
            base_url,
            model: model.into(),
        })
    }
}

impl Embedder for OllamaEmbedder {
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/api/embed", self.base_url.trim_end_matches('/'));
        let body = EmbedRequest {
            model: &self.model,
            input: texts,
        };

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .with_context(|| format!("POST {url}"))?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().unwrap_or_default();
            return Err(anyhow!(
                "ollama embed returned {status}: {}",
                text.chars().take(512).collect::<String>()
            ));
        }

        let parsed: EmbedResponse = response.json().context("parse ollama embed response")?;
        if parsed.embeddings.len() != texts.len() {
            return Err(anyhow!(
                "ollama returned {} embeddings for {} inputs",
                parsed.embeddings.len(),
                texts.len()
            ));
        }
        Ok(parsed.embeddings)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Edgequake markdown fetch (with on-disk cache)
// ─────────────────────────────────────────────────────────────────────────────

pub fn fetch_document_markdown(document_id: &str) -> Result<String> {
    let cache_path = cache_path_for(document_id);
    if let Ok(cached) = fs::read_to_string(&cache_path) {
        if !cached.trim().is_empty() {
            return Ok(cached);
        }
    }

    let base =
        env::var("EDGEQUAKE_BASE_URL").unwrap_or_else(|_| "http://localhost:8080".to_string());
    let base = base.trim_end_matches('/');
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;

    let detail_url = format!("{base}/api/v1/documents/{document_id}");
    let detail: Value = client
        .get(&detail_url)
        .send()
        .with_context(|| format!("GET {detail_url}"))?
        .error_for_status()
        .with_context(|| format!("GET {detail_url}"))?
        .json()
        .with_context(|| format!("parse {detail_url}"))?;

    let pdf_id = detail
        .get("pdf_id")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("document {document_id} has no pdf_id; not a PDF document?"))?;

    let content_url = format!("{base}/api/v1/documents/pdf/{pdf_id}/content");
    let content: Value = client
        .get(&content_url)
        .send()
        .with_context(|| format!("GET {content_url}"))?
        .error_for_status()
        .with_context(|| format!("GET {content_url}"))?
        .json()
        .with_context(|| format!("parse {content_url}"))?;

    let markdown = content
        .get("markdown_content")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("pdf {pdf_id} has no markdown_content"))?
        .to_string();

    if let Some(parent) = cache_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::write(&cache_path, &markdown);
    Ok(markdown)
}

fn cache_path_for(document_id: &str) -> PathBuf {
    let mut dir =
        env::var_os("CARGO_TARGET_DIR").map_or_else(|| PathBuf::from("target"), PathBuf::from);
    dir.push("private-rag-fixtures");
    dir.push(format!("{document_id}.md"));
    dir
}

// ─────────────────────────────────────────────────────────────────────────────
// TokenBasedChunking — simplified port from edgequake-pipeline
// ─────────────────────────────────────────────────────────────────────────────

pub struct ChunkerConfig {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub min_chunk_size: usize,
    pub separators: Vec<String>,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1600,
            chunk_overlap: 100,
            min_chunk_size: 600,
            separators: vec![
                "\n\n".to_string(),
                "\n".to_string(),
                ". ".to_string(),
                "! ".to_string(),
                "? ".to_string(),
                "; ".to_string(),
                ", ".to_string(),
                " ".to_string(),
            ],
        }
    }
}

pub struct TokenBasedChunker;

impl TokenBasedChunker {
    pub fn chunk(content: &str, config: &ChunkerConfig) -> Vec<String> {
        if content.trim().is_empty() {
            return Vec::new();
        }

        // The upstream strategy treats token-sized numbers as roughly 4 chars
        // per token, then performs char-boundary-aware splitting with overlap.
        let target_chars = config.chunk_size * 4;
        let overlap_chars = config.chunk_overlap * 4;
        let min_chars = config.min_chunk_size * 4;

        split_text_internal(
            content,
            target_chars,
            overlap_chars,
            min_chars,
            &config.separators,
        )
    }
}

fn split_text_internal(
    text: &str,
    target_size: usize,
    overlap: usize,
    min_size: usize,
    separators: &[String],
) -> Vec<String> {
    if text.len() <= target_size {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut current_pos = 0;

    while current_pos < text.len() {
        current_pos = ceil_char_boundary(text, current_pos);

        let remaining = &text[current_pos..];
        if remaining.len() <= target_size {
            chunks.push(remaining.to_string());
            break;
        }

        let end_pos = floor_char_boundary(text, current_pos + target_size);
        let chunk_text = &text[current_pos..end_pos.min(text.len())];

        let split_point = find_split_point_internal(chunk_text, target_size, separators);
        let actual_end = floor_char_boundary(text, current_pos + split_point);

        let chunk_content = text[current_pos..actual_end].to_string();
        if chunk_content.len() >= min_size {
            chunks.push(chunk_content);
        }

        let overlap_pos = actual_end.saturating_sub(overlap);
        current_pos = ceil_char_boundary(text, overlap_pos);

        if current_pos >= actual_end {
            current_pos = actual_end;
        }
    }

    chunks
}

fn find_split_point_internal(text: &str, target: usize, separators: &[String]) -> usize {
    let search_start = floor_char_boundary(text, target.saturating_sub(target / 4));
    let search_end = floor_char_boundary(text, target.min(text.len()));
    if search_start >= search_end {
        return floor_char_boundary(text, target.min(text.len()));
    }
    for separator in separators {
        if let Some(pos) = text[search_start..search_end].rfind(separator.as_str()) {
            return search_start + pos + separator.len();
        }
    }
    floor_char_boundary(text, target.min(text.len()))
}

fn floor_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut i = index;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ceil_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut i = index;
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

//! Disk-cached embeddings for the M7.3 BEIR bench. Keyed by
//! `(model_identity, doc_text_hash)`; cache files are raw little-endian
//! f32 bytes under `target/embed-cache/`. Re-runs of the bench skip the
//! embedding step entirely after the first run — critical for the
//! Qwen3-on-Vulkan path where 3.6k embeds × 150 ms = ~9 minutes uncached.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use rag_core::Embedder;
use sha2::{Digest, Sha256};

/// Wrapper around an `Embedder` that consults a disk cache before
/// invoking the underlying model. Cache key:
/// `sha256(model_identity || dim_marker || text)`. Cache miss → embed
/// + persist.
pub struct CachingEmbedder<E: Embedder> {
    inner: E,
    cache_dir: PathBuf,
    /// Cached model_identity bytes — must be stable across calls or the
    /// cache would silently mix outputs from different models.
    model_id: Vec<u8>,
}

impl<E: Embedder> CachingEmbedder<E> {
    /// `model_label` is a stable string used to namespace cache entries
    /// (e.g. "fastembed-minilm-l6", "qwen3-embedding-0.6b"). It does NOT
    /// have to be the model's full HF id, but it must be unique per
    /// embedder configuration — two different embedders sharing a label
    /// will silently collide.
    pub fn new(inner: E, model_label: &str) -> Result<Self> {
        let cache_dir = cache_root().join(model_label);
        fs::create_dir_all(&cache_dir).with_context(|| {
            format!("creating embed cache dir {}", cache_dir.display())
        })?;
        // Fold the embedder's own `model_identity()` (sha256 of weights,
        // possibly with DP config digest mixed in) into the namespace.
        // That way enabling DP-Forward on the same model gets a separate
        // cache file — we never return non-DP embeddings to a DP caller.
        let inner_id = inner.model_identity().to_vec();
        let mut id_hasher = Sha256::new();
        id_hasher.update(model_label.as_bytes());
        id_hasher.update(b"|");
        id_hasher.update(&inner_id);
        let model_id = id_hasher.finalize().to_vec();
        Ok(Self {
            inner,
            cache_dir,
            model_id,
        })
    }

    fn cache_path(&self, text: &str) -> PathBuf {
        let mut hasher = Sha256::new();
        hasher.update(&self.model_id);
        hasher.update(b"|");
        hasher.update(text.as_bytes());
        let digest = hasher.finalize();
        let hex = hex::encode(&digest[..16]);
        self.cache_dir.join(format!("{hex}.f32"))
    }

    fn read_cached(&self, path: &PathBuf) -> Result<Option<Vec<f32>>> {
        if !path.exists() {
            return Ok(None);
        }
        let mut bytes = Vec::new();
        File::open(path)?.read_to_end(&mut bytes)?;
        if bytes.len() % 4 != 0 {
            return Ok(None); // corrupt; force re-embed
        }
        let mut out = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(Some(out))
    }

    fn write_cached(&self, path: &PathBuf, v: &[f32]) -> Result<()> {
        let mut bytes = Vec::with_capacity(v.len() * 4);
        for &x in v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        let mut f = File::create(path)?;
        f.write_all(&bytes)?;
        Ok(())
    }
}

impl<E: Embedder> Embedder for CachingEmbedder<E> {
    fn embed(&mut self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        // Pass 1: collect cache state for each text — read hits, mark misses.
        let mut out: Vec<Option<Vec<f32>>> = Vec::with_capacity(texts.len());
        let mut miss_idx: Vec<usize> = Vec::new();
        let mut miss_texts: Vec<String> = Vec::new();
        for (i, t) in texts.iter().enumerate() {
            let path = self.cache_path(t);
            match self.read_cached(&path)? {
                Some(v) => out.push(Some(v)),
                None => {
                    out.push(None);
                    miss_idx.push(i);
                    miss_texts.push(t.clone());
                }
            }
        }

        // Pass 2: embed the misses in chunks (FastEmbed's internal
        // batching can OOM on very large all-at-once inputs; 64 is small
        // enough to fit comfortably while still amortizing per-call cost).
        const BATCH: usize = 64;
        if !miss_texts.is_empty() {
            for chunk_start in (0..miss_texts.len()).step_by(BATCH) {
                let chunk_end = (chunk_start + BATCH).min(miss_texts.len());
                let chunk = &miss_texts[chunk_start..chunk_end];
                let computed = self.inner.embed(&chunk.to_vec())?;
                for (offset, v) in computed.into_iter().enumerate() {
                    let slot = miss_idx[chunk_start + offset];
                    let text = &texts[slot];
                    self.write_cached(&self.cache_path(text), &v)?;
                    out[slot] = Some(v);
                }
            }
        }

        Ok(out.into_iter().map(|o| o.expect("filled in pass 1 or 2")).collect())
    }

    fn model_identity(&self) -> &[u8] {
        self.inner.model_identity()
    }
}

fn cache_root() -> PathBuf {
    // Find workspace `target/` from the running test binary.
    let mut p = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("."));
    while let Some(parent) = p.parent() {
        if parent.file_name().and_then(|s| s.to_str()) == Some("target") {
            return parent.join("embed-cache");
        }
        p = parent.to_path_buf();
    }
    PathBuf::from("target/embed-cache")
}

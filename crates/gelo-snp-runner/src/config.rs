//! Runner configuration loaded from a TOML file.

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::Deserialize;

/// Where to look for the runner config if `GELO_SNP_RUNNER_CONFIG` is unset.
pub const DEFAULT_CONFIG_PATH: &str = "/etc/gelo-snp/runner.toml";

#[derive(Debug, Deserialize)]
pub struct RunnerConfig {
    /// `host:port` to bind the HTTP listener.
    #[serde(default = "default_listen")]
    pub listen: String,
    /// Public model identifier the relying party expects. The CVM hashes
    /// this string and bakes the hash into `REPORT_DATA[0..32]`.
    #[serde(default = "default_model_identity")]
    pub model_identity: String,
    /// Protocol-scheme identifier (mask version, shield config, etc).
    #[serde(default = "default_scheme_identity")]
    pub scheme_identity: String,
    /// Embedder selector. `"stub"` is fast for boot-smoke tests;
    /// `"bge-small"` / `"qwen3-0.6b"` (future) load real weights.
    #[serde(default = "default_embedder")]
    pub embedder: String,
    /// Optional path on the rootfs where weights should be cached / loaded
    /// from. Required for non-stub embedders (consumed once those
    /// embedder backends are wired up in a follow-up commit).
    #[allow(dead_code)]
    pub weights_path: Option<PathBuf>,
    /// Compass storage-backend root URL — points at a
    /// `compass-storage-server` instance. Per-tenant URLs are formed
    /// by `compass::backend_url(root, tenant, index)`. `None` means
    /// the runner won't construct any REST backends (M5.3 wiring
    /// gate); attempting to ingest/query a Compass-backed tenant in
    /// that state surfaces 503. Wired in earnest at M6
    /// (`light-kg-store` integration).
    #[serde(default)]
    pub compass_backend_url: Option<String>,
    /// Directory holding the Qwen3-4B (or compatible) extraction
    /// decoder weights: `config.json`, `tokenizer.json`, and one or
    /// more `*.safetensors` shards. `None` → the
    /// `/lightrag/extract_and_build` route returns 503.
    #[serde(default)]
    pub extraction_decoder_path: Option<PathBuf>,
    /// Directory holding the Qwen3-Embedding-0.6B (or compatible)
    /// embedder used to embed chunks + entity/relation descriptions
    /// inside `/lightrag/extract_and_build`. `None` → the route
    /// returns 503.
    #[serde(default)]
    pub extraction_embedder_path: Option<PathBuf>,
}

fn default_listen() -> String {
    "0.0.0.0:7878".to_string()
}
fn default_model_identity() -> String {
    "stub-model@v1".to_string()
}
fn default_scheme_identity() -> String {
    "gelo+twinshield@v1".to_string()
}
fn default_embedder() -> String {
    "stub".to_string()
}

impl RunnerConfig {
    /// Resolve the config path: `GELO_SNP_RUNNER_CONFIG` if set, otherwise
    /// `/etc/gelo-snp/runner.toml`. If neither exists, returns default
    /// values so the binary still runs (e.g. when started by `cargo run`
    /// during local iteration).
    pub fn load() -> Result<Self> {
        let path = std::env::var("GELO_SNP_RUNNER_CONFIG")
            .unwrap_or_else(|_| DEFAULT_CONFIG_PATH.to_string());
        if !std::path::Path::new(&path).exists() {
            tracing::warn!(
                "config file at {path} is missing; falling back to compiled-in defaults"
            );
            return Ok(Self {
                listen: default_listen(),
                model_identity: default_model_identity(),
                scheme_identity: default_scheme_identity(),
                embedder: default_embedder(),
                weights_path: None,
                compass_backend_url: None,
                extraction_decoder_path: None,
                extraction_embedder_path: None,
            });
        }
        let txt = std::fs::read_to_string(&path)
            .with_context(|| format!("reading config from {path}"))?;
        let cfg: RunnerConfig =
            toml::from_str(&txt).with_context(|| format!("parsing config from {path}"))?;
        Ok(cfg)
    }
}

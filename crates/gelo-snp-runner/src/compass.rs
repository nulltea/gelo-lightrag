//! Per-tenant URL gate for the Compass storage backend (M5.3).
//!
//! The runner holds the configured `compass_backend_url` root and
//! composes it with `(tenant_id, index_kind)` to produce the
//! concrete URL passed to `RestBlockBackend::connect`. This module
//! enforces:
//!
//! 1. URL hygiene — tenant and index segments are validated against
//!    a safe charset before they reach the URL builder, since they
//!    flow from request payloads.
//! 2. Naming consistency — index kinds are fixed per the LightRAG
//!    surface (`entities`, `relations`, `chunks`, `adjacency`,
//!    `src_chunks`, `node_props`, `edge_props`). Anything else is a
//!    programming error caught at build time via the enum.
//!
//! M5.3 is plumbing only. M6 (`light-kg-store`) calls these helpers
//! when materialising a tenant's three `CompassIndex` + two
//! `XorMmClient` + encrypted-KV instances. M8 surfaces them through
//! `/lightrag/*` HTTP routes.

// M5.3 ships the plumbing; M6 wires the first caller. Until then the
// items below would warn unused.
#![allow(dead_code)]

use std::fmt;

/// Kind of index within a tenant's `LightKgStore`. Maps 1:1 onto the
/// HKDF v2 child keys (`oram_entities_key`, `oram_relations_key`,
/// `oram_chunks_key`, `emm_adjacency_key`, `emm_src_chunks_key`,
/// `oram_node_props_key`, `oram_edge_props_key`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexKind {
    Entities,
    Relations,
    Chunks,
    Adjacency,
    SrcChunks,
    NodeProps,
    EdgeProps,
}

impl IndexKind {
    pub fn as_url_segment(self) -> &'static str {
        match self {
            IndexKind::Entities => "entities",
            IndexKind::Relations => "relations",
            IndexKind::Chunks => "chunks",
            IndexKind::Adjacency => "adjacency",
            IndexKind::SrcChunks => "src_chunks",
            IndexKind::NodeProps => "node_props",
            IndexKind::EdgeProps => "edge_props",
        }
    }
}

impl fmt::Display for IndexKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_url_segment())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CompassUrlError {
    #[error("no compass_backend_url configured — refusing to construct a tenant URL")]
    Unconfigured,
    #[error(
        "tenant id {0:?} contains a character outside [A-Za-z0-9_-]; \
         path-traversal or URL-injection guard"
    )]
    InvalidTenantId(String),
    #[error("backend root URL invalid: {0}")]
    InvalidRoot(String),
}

/// Build the full per-tenant + per-index backend URL.
///
/// `root` should be the runner-config root (e.g.
/// `http://compass-storage:8080`). The result has the shape
/// `{root}/v1/{tenant}/{index}` with no trailing slash —
/// `RestBlockBackend::connect` normalises that.
pub fn backend_url(
    root: Option<&str>,
    tenant: &str,
    index: IndexKind,
) -> Result<String, CompassUrlError> {
    let root = root.ok_or(CompassUrlError::Unconfigured)?;
    if !is_safe_id(tenant) {
        return Err(CompassUrlError::InvalidTenantId(tenant.to_string()));
    }
    // Light validation — reject anything that doesn't parse as URL.
    // We don't dereference the parsed Url; we just want to fail loud
    // if the operator misconfigured `compass_backend_url`.
    let _ = url::Url::parse(root).map_err(|e| CompassUrlError::InvalidRoot(e.to_string()))?;

    let trimmed = root.trim_end_matches('/');
    Ok(format!("{trimmed}/v1/{tenant}/{}", index.as_url_segment()))
}

fn is_safe_id(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_url_composes_root_tenant_index() {
        let url = backend_url(
            Some("http://compass-storage:8080"),
            "alpha-tenant",
            IndexKind::Entities,
        )
        .unwrap();
        assert_eq!(url, "http://compass-storage:8080/v1/alpha-tenant/entities");
    }

    #[test]
    fn backend_url_strips_trailing_slash_on_root() {
        let url = backend_url(
            Some("http://compass-storage:8080/"),
            "alpha",
            IndexKind::Chunks,
        )
        .unwrap();
        assert_eq!(url, "http://compass-storage:8080/v1/alpha/chunks");
    }

    #[test]
    fn backend_url_errors_on_missing_root() {
        let err = backend_url(None, "alpha", IndexKind::Entities).unwrap_err();
        assert!(matches!(err, CompassUrlError::Unconfigured));
    }

    #[test]
    fn backend_url_rejects_path_traversal_in_tenant() {
        for bad in &["..", "alpha/", "../etc", "x y", "x?y", "x%2f"] {
            let err = backend_url(
                Some("http://compass-storage:8080"),
                bad,
                IndexKind::Entities,
            )
            .unwrap_err();
            assert!(
                matches!(err, CompassUrlError::InvalidTenantId(_)),
                "tenant {bad:?} should fail validation"
            );
        }
    }

    #[test]
    fn index_kind_url_segments_are_stable() {
        // Pin segment values — the URL gate is part of the runner's
        // contract with the storage server, so changes here are
        // breaking.
        assert_eq!(IndexKind::Entities.as_url_segment(), "entities");
        assert_eq!(IndexKind::Relations.as_url_segment(), "relations");
        assert_eq!(IndexKind::Chunks.as_url_segment(), "chunks");
        assert_eq!(IndexKind::Adjacency.as_url_segment(), "adjacency");
        assert_eq!(IndexKind::SrcChunks.as_url_segment(), "src_chunks");
        assert_eq!(IndexKind::NodeProps.as_url_segment(), "node_props");
        assert_eq!(IndexKind::EdgeProps.as_url_segment(), "edge_props");
    }
}

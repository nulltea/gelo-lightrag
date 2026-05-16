//! Shared helpers for the e2e Compass GraphRAG bench.
//!
//! Synth-KG generator + stage-timing primitives. Used by the
//! `compass-rag-bench` binary; library crate so future scenarios
//! (local_rest, runner_http) can share the same fixture builder.

pub mod stages;
pub mod summary;
pub mod synth;

//! Network-shaped `BlockBackend` for `ring-oram`. Plan §9 OQ#4 sub-plan
//! M5.1.
//!
//! Two halves:
//!
//! - [`server`] — axum router over a `sled::Db`. Spawned inside
//!   `gelo-snp-runner` (M5.3) or stood up standalone in tests.
//! - [`client::RestBlockBackend`] — implements the async
//!   [`ring_oram::BlockBackend`] trait via `reqwest`. Held by the
//!   `RingOramClient` inside the CVM.
//!
//! Wire format: msgpack via `rmp-serde`. URL shape
//! `/v1/{tenant}/{index}/{init|read_path|write_buckets}`. The
//! per-tenant URL gate is defense-in-depth — confidentiality is
//! already covered by per-tenant ORAM keys.

pub mod client;
pub mod server;
pub mod wire;

pub use client::{RestBackendError, RestBlockBackend};
pub use server::{AppState, router};
pub use wire::{
    InitRequest, ReadPathRequest, ReadPathResponse, WireBucket, WriteBucketsRequest,
};

//! Ring-ORAM client + server protocol — skeleton (M0).
//!
//! Spec: `docs/prototype/private-graph-rag-variant-a.md` §4.1.
//! Reference: Ren et al., USENIX Security 2015; port from the C++
//! reference in [`Clive2312/compass`](https://github.com/Clive2312/compass).
//!
//! Surface to be implemented in M1: `RingOramClient` + `BlockBackend`
//! trait, semi-honest and malicious-server (Merkle) modes, lazy
//! eviction, the XOR-trick for constant online bandwidth.

/// Ring-ORAM bucket parameters. Locked into [`RingOramParams`] at
/// `CompassIndex` construction time so the [`crate::keying::CompassParams`]
/// canonical encoding can pin them via attestation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingOramParams {
    /// Real-block slots per bucket.
    pub z: u32,
    /// Dummy slots per bucket.
    pub s: u32,
    /// Eviction rate — `EvictPath` runs every `a` `ReadPath` ops.
    pub a: u32,
    /// Block payload size (bytes).
    pub block_bytes: u32,
}

impl Default for RingOramParams {
    /// Paper-aligned defaults: see Compass paper §6 / Tab. 3.
    fn default() -> Self {
        Self {
            z: 4,
            s: 5,
            a: 3,
            block_bytes: 2048,
        }
    }
}

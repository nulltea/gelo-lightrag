//! XorMM parameters. Mirrors `rag_core::keying::XorMmParams` for the
//! subset the EMM layer reads.

/// XorMM construction parameters. Pinned in the V2 attestation
/// `scheme_identity` via `rag_core::keying::XorMmParams` so a relying
/// party can detect a change to the volume-hiding budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XorMmParams {
    /// Per-key padded value count. All `get(k)` responses contain
    /// exactly this many `LogicalValue` slots, with `DUMMY` entries
    /// padding short lists and entries beyond the bound truncated.
    pub volume_bound: u32,
    /// Per-`LogicalValue` byte budget. Entries shorter than this are
    /// zero-padded; longer entries are an error (callers must
    /// re-bucket).
    pub value_bytes: u32,
    /// Number of buckets in the EMM. Chosen so cuckoo placement
    /// succeeds with high probability for the input cardinality.
    /// Rule of thumb: `n_buckets ≥ 1.3 · n_keys` (paper §4 + folklore).
    pub n_buckets: u32,
    /// Maximum number of cuckoo displacements before a key falls back
    /// to the stash. Paper recommends ~log(n_keys).
    pub max_kicks: u32,
}

impl Default for XorMmParams {
    /// M2 defaults — tuned for LightRAG enterprise scale (10⁴-10⁶
    /// keys, 95th-percentile degree ≤ 64).
    fn default() -> Self {
        Self {
            volume_bound: 64,
            value_bytes: 32,
            n_buckets: 1024,
            max_kicks: 16,
        }
    }
}

impl XorMmParams {
    /// AES-GCM plaintext size of one bucket. Layout:
    ///
    /// ```text
    /// [dummy_flag u8] ‖ [fingerprint 32B] ‖ [value_count u32-LE] ‖
    /// [values: volume_bound × value_bytes]
    /// ```
    ///
    /// Bucket ciphertext on the wire is this + 16-byte AES-GCM tag.
    pub fn bucket_plaintext_size(&self) -> usize {
        1 + 32 + 4 + (self.volume_bound as usize) * (self.value_bytes as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_params_match_lightrag_scale() {
        let p = XorMmParams::default();
        assert_eq!(p.volume_bound, 64);
        // dummy_flag(1) + fingerprint(32) + value_count(4) +
        // volume_bound·value_bytes = 37 + 64·32 = 2085.
        assert_eq!(p.bucket_plaintext_size(), 37 + 64 * 32);
    }
}

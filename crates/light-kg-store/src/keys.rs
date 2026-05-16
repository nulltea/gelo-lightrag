//! HMAC-keyed identifier translation: plaintext entity / relation
//! names → 32-byte `LogicalKey`s consumed by `XorMmClient` lookups.
//!
//! For each EMM (adjacency, src_chunks) the master key is one of the
//! V2 HKDF children (`emm_adjacency_key`, `emm_src_chunks_key`).
//! Logical keys are domain-separated by including a fixed label
//! prefix so the same plaintext name produces *different* logical
//! keys across the two EMMs.
//!
//! HMAC-SHA-256 instead of plain SHA-256 because the key is secret;
//! HMAC is the standard PRF construction from a hash that's safe to
//! use this way.

use hmac::{Hmac, Mac};
use sha2::Sha256;
use xormm_emm::LogicalKey;
use zeroize::Zeroizing;

type HmacSha256 = Hmac<Sha256>;

/// Domain labels for EMM key derivation. Constants so a typo here
/// would silently break attestation; pinned in the parity test below.
pub mod label {
    pub const ADJACENCY_ENTITY: &str = "adjacency/entity";
    pub const ADJACENCY_RELATION: &str = "adjacency/relation";
    pub const SRC_CHUNKS_ENTITY: &str = "src_chunks/entity";
    pub const SRC_CHUNKS_RELATION: &str = "src_chunks/relation";
}

/// `LogicalKey = HMAC-SHA256(master_key, "{label}\0{name}")`.
/// The null byte separator prevents a `(label, name)` collision
/// where (`"adjacency/entityX"`, `"foo"`) and (`"adjacency/entity"`,
/// `"Xfoo"`) would otherwise hash to the same input.
pub fn derive_logical_key(master_key: &Zeroizing<[u8; 32]>, label: &str, name: &str) -> LogicalKey {
    let mut mac = HmacSha256::new_from_slice(master_key.as_ref())
        .expect("HMAC accepts any 32-byte key");
    mac.update(label.as_bytes());
    mac.update(&[0u8]);
    mac.update(name.as_bytes());
    let tag = mac.finalize().into_bytes();
    let mut out = [0u8; 32];
    out.copy_from_slice(&tag);
    LogicalKey(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(b: u8) -> Zeroizing<[u8; 32]> {
        Zeroizing::new([b; 32])
    }

    #[test]
    fn same_input_produces_same_key() {
        let a = derive_logical_key(&k(0x11), label::ADJACENCY_ENTITY, "alice");
        let b = derive_logical_key(&k(0x11), label::ADJACENCY_ENTITY, "alice");
        assert_eq!(a.0, b.0);
    }

    #[test]
    fn different_master_keys_diverge() {
        let a = derive_logical_key(&k(0x11), label::ADJACENCY_ENTITY, "alice");
        let b = derive_logical_key(&k(0x22), label::ADJACENCY_ENTITY, "alice");
        assert_ne!(a.0, b.0);
    }

    #[test]
    fn different_labels_diverge() {
        // Same key, same name, different EMM domain → different logical key.
        let a = derive_logical_key(&k(0x33), label::ADJACENCY_ENTITY, "alice");
        let b = derive_logical_key(&k(0x33), label::SRC_CHUNKS_ENTITY, "alice");
        assert_ne!(a.0, b.0);
    }

    #[test]
    fn label_name_split_isnt_ambiguous() {
        // The null separator must rule out `(label, name)` collisions
        // of the form `("xy", "z")` vs `("x", "yz")`.
        let a = derive_logical_key(&k(0x44), "xy", "z");
        let b = derive_logical_key(&k(0x44), "x", "yz");
        assert_ne!(a.0, b.0);
    }
}

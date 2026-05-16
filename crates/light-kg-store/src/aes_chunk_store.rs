//! AES-GCM-encrypted chunk-text store.
//!
//! Holds one ciphertext blob per chunk, keyed by the plaintext chunk
//! id. Threat model: the CVM's RAM is trusted; the persistent storage
//! (if any) is not. The store keeps the encrypted map in RAM today —
//! M5.4's `object_store` adapter (deferred) will spill to S3/GCS.
//!
//! Nonce derivation: 12-byte construction `chunk_id_hash[0..8] ‖ 0u32`.
//! Each chunk gets a unique nonce; updates to a chunk would require a
//! generation counter (deferred — chunks are immutable in the
//! ingest-only v1 path).

use std::collections::HashMap;

use aes_gcm::{
    Aes256Gcm, Nonce,
    aead::{Aead, KeyInit},
};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use crate::types::Chunk;

#[derive(Debug, thiserror::Error)]
pub enum ChunkStoreError {
    #[error("chunk {0:?} not found")]
    NotFound(String),
    #[error("AES-GCM error: {0}")]
    Aead(String),
}

/// Per-chunk AES-GCM record. The nonce is derivable from the chunk id
/// (which lives plaintext inside the CVM); we store it for clarity.
#[derive(Debug, Clone)]
struct Encrypted {
    nonce: [u8; 12],
    ciphertext: Vec<u8>,
}

/// AES-GCM-encrypted chunk-text store. Holds `HashMap<chunk_id,
/// Encrypted>`. The chunk_id keys are in cleartext inside the CVM —
/// LightRAG-style retrieval looks chunks up by id after the
/// CompassIndex/EMM steps select which ones to fetch. Pseudonymising
/// the id space is M9 hardening (Risk F).
pub struct AesChunkStore {
    key: Zeroizing<[u8; 32]>,
    entries: HashMap<String, Encrypted>,
}

impl AesChunkStore {
    pub fn new(key: Zeroizing<[u8; 32]>) -> Self {
        Self {
            key,
            entries: HashMap::new(),
        }
    }

    /// Bulk-insert chunks from the build path. Replaces any existing
    /// entry for the same id (last-write-wins).
    pub fn put_all(&mut self, chunks: &[Chunk]) -> Result<(), ChunkStoreError> {
        let cipher = Aes256Gcm::new(self.key.as_ref().into());
        for c in chunks {
            let nonce = derive_nonce(&c.id);
            let ct = cipher
                .encrypt(Nonce::from_slice(&nonce), c.text.as_bytes())
                .map_err(|e| ChunkStoreError::Aead(format!("encrypt {}: {e}", c.id)))?;
            self.entries.insert(
                c.id.clone(),
                Encrypted {
                    nonce,
                    ciphertext: ct,
                },
            );
        }
        Ok(())
    }

    /// Decrypt one chunk by id. Errors if the id was never inserted.
    pub fn get(&self, id: &str) -> Result<String, ChunkStoreError> {
        let rec = self
            .entries
            .get(id)
            .ok_or_else(|| ChunkStoreError::NotFound(id.to_string()))?;
        let cipher = Aes256Gcm::new(self.key.as_ref().into());
        let pt = cipher
            .decrypt(Nonce::from_slice(&rec.nonce), rec.ciphertext.as_ref())
            .map_err(|e| ChunkStoreError::Aead(format!("decrypt {id}: {e}")))?;
        String::from_utf8(pt).map_err(|e| ChunkStoreError::Aead(format!("utf8 {id}: {e}")))
    }

    /// Number of chunks held — convenience for tests.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Derive a 12-byte AES-GCM nonce from a chunk id.
/// `nonce = sha256(chunk_id)[..12]`. Chunk ids are unique in the
/// upstream LightRAG ingest path; the hash is to fold arbitrary id
/// strings into the fixed 12-byte slot. Same id ⇒ same nonce — we
/// rely on chunks being immutable per generation.
fn derive_nonce(chunk_id: &str) -> [u8; 12] {
    let digest = Sha256::digest(chunk_id.as_bytes());
    let mut out = [0u8; 12];
    out.copy_from_slice(&digest[..12]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(key: u8) -> AesChunkStore {
        let mut k: [u8; 32] = [0; 32];
        k.fill(key);
        AesChunkStore::new(Zeroizing::new(k))
    }

    fn chunk(id: &str, text: &str) -> Chunk {
        Chunk {
            id: id.to_string(),
            text: text.to_string(),
            embedding: vec![],
        }
    }

    #[test]
    fn put_then_get_round_trips() {
        let mut s = mk(0x42);
        s.put_all(&[chunk("c-0", "hello world"), chunk("c-1", "lorem ipsum")])
            .unwrap();
        assert_eq!(s.get("c-0").unwrap(), "hello world");
        assert_eq!(s.get("c-1").unwrap(), "lorem ipsum");
    }

    #[test]
    fn get_unknown_errors() {
        let s = mk(0x99);
        let err = s.get("missing").unwrap_err();
        assert!(matches!(err, ChunkStoreError::NotFound(_)));
    }

    #[test]
    fn different_keys_make_ciphertexts_undecryptable() {
        let mut a = mk(0x11);
        a.put_all(&[chunk("c-0", "secret payload")]).unwrap();

        // Swap key under same entries map — should fail to decrypt.
        let mut b = mk(0xff);
        b.entries = a.entries.clone();
        let err = b.get("c-0").unwrap_err();
        assert!(matches!(err, ChunkStoreError::Aead(_)));
    }
}

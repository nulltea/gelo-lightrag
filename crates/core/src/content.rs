use aes_gcm::{
    Aes256Gcm, Nonce,
    aead::{Aead, KeyInit},
};
use anyhow::{Context, Result, anyhow, bail};
use rand::Rng;
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::{ChunkCiphertext, DocumentChunk};

/// AES-256-GCM chunk-text cipher. The 32-byte key is zeroized on drop
/// and redacted from `Debug` output.
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct AesChunkCipher {
    key: [u8; 32],
}

impl std::fmt::Debug for AesChunkCipher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AesChunkCipher")
            .field("key", &"<redacted 32B>")
            .finish()
    }
}

impl AesChunkCipher {
    /// Generate a fresh AES-256 key from `OsRng`. Non-deterministic;
    /// use [`Self::from_key`] from inside the CVM where the key is
    /// HKDF-derived.
    pub fn generate() -> Self {
        let mut key = [0_u8; 32];
        rand::rng().fill(&mut key);
        Self { key }
    }

    /// Build a cipher from a caller-provided 32-byte key. The intended
    /// caller is [`crate::HkdfPolicy::derive`] (see the two-party-KDF
    /// design in `docs/prototype/caprise-two-party-kdf.md`).
    pub fn from_key(key: [u8; 32]) -> Self {
        Self { key }
    }

    pub fn encrypt_chunk(&self, chunk: &DocumentChunk) -> Result<ChunkCiphertext> {
        let cipher = Aes256Gcm::new_from_slice(&self.key).expect("32-byte AES-256 key");
        let mut nonce_bytes = [0_u8; 12];
        rand::rng().fill(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);
        let plaintext = chunk.text.as_bytes();
        let ciphertext = cipher
            .encrypt(nonce, plaintext)
            .map_err(|_| anyhow!("failed to AES-encrypt chunk text"))?;

        Ok(ChunkCiphertext {
            chunk_id: chunk.id.clone(),
            scheme: "aes-256-gcm",
            nonce: nonce_bytes.to_vec(),
            ciphertext,
        })
    }

    pub fn decrypt_chunk(&self, encrypted: &ChunkCiphertext) -> Result<DocumentChunk> {
        if encrypted.scheme != "aes-256-gcm" {
            bail!("unsupported chunk encryption scheme: {}", encrypted.scheme);
        }

        let cipher = Aes256Gcm::new_from_slice(&self.key).expect("32-byte AES-256 key");
        let nonce = Nonce::from_slice(&encrypted.nonce);
        let plaintext = cipher
            .decrypt(nonce, encrypted.ciphertext.as_ref())
            .map_err(|_| anyhow!("failed to AES-decrypt chunk text"))?;
        let text = String::from_utf8(plaintext).context("chunk plaintext is not valid utf-8")?;

        Ok(DocumentChunk {
            id: encrypted.chunk_id.clone(),
            text,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{AesChunkCipher, ChunkId, DocumentChunk};

    #[test]
    fn aes_chunk_cipher_round_trip_recovers_text() {
        let cipher = AesChunkCipher::generate();
        let chunk = DocumentChunk {
            id: ChunkId("alpha".into()),
            text: "secret text".into(),
        };

        let encrypted = cipher.encrypt_chunk(&chunk).unwrap();
        let decrypted = cipher.decrypt_chunk(&encrypted).unwrap();

        assert_eq!(decrypted.id, chunk.id);
        assert_eq!(decrypted.text, chunk.text);
    }
}

//! Bucket codec: serialise `Z + S` block slots to a flat byte buffer
//! the backend AES-GCM-encrypts, and the reverse.
//!
//! Each slot is `4 bytes (BlockId u32-LE) ‖ block_bytes payload`.
//! Total serialised plaintext = `(Z + S) · (4 + block_bytes)`.

use aes_gcm::{
    Aes256Gcm, Nonce,
    aead::{Aead, KeyInit},
};

use crate::block::{Block, BlockId, BlockPayload};
use crate::params::RingOramParams;

/// 12-byte AES-GCM nonce. `nonce[0..8] = bucket_id u64-LE; nonce[8..12]
/// = write_counter u32-LE` — no nonce is reused across rewrites of the
/// same bucket because `write_counter` monotonically increments.
pub(crate) fn make_nonce(bucket_id: u32, write_counter: u32) -> [u8; 12] {
    let mut n = [0u8; 12];
    n[..8].copy_from_slice(&(bucket_id as u64).to_le_bytes());
    n[8..12].copy_from_slice(&write_counter.to_le_bytes());
    n
}

fn slot_size(params: &RingOramParams) -> usize {
    4 + params.block_bytes as usize
}

#[cfg(test)]
fn bucket_plaintext_size(params: &RingOramParams) -> usize {
    params.bucket_capacity() as usize * slot_size(params)
}

/// Serialise a full bucket's `Z + S` slots into plaintext bytes. Dummy
/// slots use `BlockId::DUMMY`'s u32 representation; missing slots (if
/// the caller passes fewer than `Z + S` blocks) are padded with
/// dummies. The order is preserved.
pub(crate) fn serialise_bucket(blocks: &[Block], params: &RingOramParams) -> Vec<u8> {
    let cap = params.bucket_capacity() as usize;
    assert!(
        blocks.len() <= cap,
        "serialise_bucket: too many blocks ({} > {})",
        blocks.len(),
        cap
    );
    let slot = slot_size(params);
    let mut out = vec![0u8; cap * slot];
    for (i, block) in blocks.iter().enumerate() {
        let off = i * slot;
        out[off..off + 4].copy_from_slice(&block.id.0.to_le_bytes());
        let payload = block.payload.as_bytes();
        assert_eq!(
            payload.len(),
            params.block_bytes as usize,
            "payload size mismatch"
        );
        out[off + 4..off + 4 + payload.len()].copy_from_slice(payload);
    }
    // Pad remaining slots with dummies (already zero-payload, just set ids).
    for i in blocks.len()..cap {
        let off = i * slot;
        out[off..off + 4].copy_from_slice(&BlockId::DUMMY.0.to_le_bytes());
    }
    out
}

/// Inverse of [`serialise_bucket`].
pub(crate) fn deserialise_bucket(bytes: &[u8], params: &RingOramParams) -> Vec<Block> {
    let cap = params.bucket_capacity() as usize;
    let slot = slot_size(params);
    assert_eq!(bytes.len(), cap * slot, "bucket size mismatch");
    let mut out = Vec::with_capacity(cap);
    for i in 0..cap {
        let off = i * slot;
        let id = BlockId(u32::from_le_bytes(
            bytes[off..off + 4].try_into().expect("4-byte slice"),
        ));
        let payload = bytes[off + 4..off + 4 + params.block_bytes as usize].to_vec();
        out.push(Block {
            id,
            payload: BlockPayload::from_exact(payload, params.block_bytes as usize),
        });
    }
    out
}

/// Encrypt a serialised bucket. Wraps AES-GCM-256 with the
/// (bucket_id, write_counter)-derived nonce. Returns ciphertext +
/// 16-byte authentication tag, ready for `EncryptedBucket::ciphertext`.
pub(crate) fn aes_encrypt(
    key: &[u8; 32],
    bucket_id: u32,
    write_counter: u32,
    plaintext: &[u8],
) -> Vec<u8> {
    let cipher = Aes256Gcm::new_from_slice(key).expect("32-byte key");
    let nonce = make_nonce(bucket_id, write_counter);
    cipher
        .encrypt(Nonce::from_slice(&nonce), plaintext)
        .expect("AES-GCM encrypt cannot fail for valid inputs")
}

/// Inverse of [`aes_encrypt`]. Returns an error if the tag fails to
/// verify — caller maps it to a protocol-level corruption error.
pub(crate) fn aes_decrypt(
    key: &[u8; 32],
    bucket_id: u32,
    write_counter: u32,
    ciphertext: &[u8],
) -> Result<Vec<u8>, AesError> {
    let cipher = Aes256Gcm::new_from_slice(key).expect("32-byte key");
    let nonce = make_nonce(bucket_id, write_counter);
    cipher
        .decrypt(Nonce::from_slice(&nonce), ciphertext)
        .map_err(|_| AesError::AuthenticationFailed)
}

#[derive(Debug, thiserror::Error)]
pub enum AesError {
    #[error("AES-GCM authentication failed (corrupted bucket or wrong key)")]
    AuthenticationFailed,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_params() -> RingOramParams {
        RingOramParams {
            z: 2,
            s: 1,
            a: 3,
            block_bytes: 8,
            n_leaves: 4,
        }
    }

    fn mk_block(id: u32, byte: u8) -> Block {
        Block::new(
            BlockId(id),
            BlockPayload::from_exact(vec![byte; 8], 8),
        )
    }

    #[test]
    fn bucket_round_trips() {
        let params = small_params();
        let blocks = vec![mk_block(1, 0xaa), mk_block(2, 0xbb)];
        let pt = serialise_bucket(&blocks, &params);
        assert_eq!(pt.len(), bucket_plaintext_size(&params));
        let back = deserialise_bucket(&pt, &params);
        assert_eq!(back.len(), 3); // Z+S = 3, padded with one dummy.
        assert_eq!(back[0].id, BlockId(1));
        assert_eq!(back[0].payload.as_bytes(), &[0xaa; 8]);
        assert_eq!(back[1].id, BlockId(2));
        assert!(back[2].is_dummy());
    }

    #[test]
    fn aes_round_trips() {
        let key = [0x11u8; 32];
        let pt = b"hello-world-1234";
        let ct = aes_encrypt(&key, 5, 9, pt);
        let back = aes_decrypt(&key, 5, 9, &ct).expect("auth ok");
        assert_eq!(back, pt);
    }

    #[test]
    fn nonce_changes_under_counter_bump() {
        assert_ne!(make_nonce(5, 9), make_nonce(5, 10));
        assert_ne!(make_nonce(5, 9), make_nonce(6, 9));
    }

    #[test]
    fn aes_rejects_wrong_counter() {
        let key = [0x22u8; 32];
        let ct = aes_encrypt(&key, 3, 1, b"abcd-efgh");
        assert!(aes_decrypt(&key, 3, 2, &ct).is_err()); // wrong counter
        assert!(aes_decrypt(&key, 4, 1, &ct).is_err()); // wrong bucket
    }
}

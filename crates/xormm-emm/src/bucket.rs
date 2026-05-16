//! Bucket plaintext codec + AES-GCM frame.

use aes_gcm::{
    Aes256Gcm, Nonce,
    aead::{Aead, KeyInit},
};

use crate::LogicalKey;
use crate::params::XorMmParams;

/// Per-key fingerprint = the 32-byte LogicalKey as-is. The fingerprint
/// is held in the bucket so `get` can identify which of the two
/// candidate buckets actually holds the key (the other one is a dummy
/// or holds a different key under the same hash slot).
pub(crate) type KeyFingerprint = [u8; 32];

/// 32-byte sentinel for a dummy bucket.
pub(crate) const DUMMY_FINGERPRINT: KeyFingerprint = [0xffu8; 32];

/// Bucket payload before AES-GCM. Layout:
/// `[dummy_flag u8] ‖ [fingerprint 32B] ‖ [value_count u32-LE] ‖
/// [values: volume_bound × value_bytes]`.
/// All fields zero-padded; dummy_flag = 1 marks the bucket as a dummy
/// (no real key landed here at build time).
#[derive(Debug, Clone)]
pub(crate) struct BucketPlain {
    pub dummy_flag: u8,
    pub fingerprint: KeyFingerprint,
    pub value_count: u32,
    /// Length = `volume_bound`. Each entry is exactly `value_bytes`
    /// long (zero-padded on the short side).
    pub values: Vec<Vec<u8>>,
}

impl BucketPlain {
    pub fn dummy(params: &XorMmParams) -> Self {
        Self {
            dummy_flag: 1,
            fingerprint: DUMMY_FINGERPRINT,
            value_count: 0,
            values: vec![vec![0u8; params.value_bytes as usize]; params.volume_bound as usize],
        }
    }

    pub fn serialise(&self, params: &XorMmParams) -> Vec<u8> {
        let header = 1 + 32 + 4;
        let vb = params.value_bytes as usize;
        let body = (params.volume_bound as usize) * vb;
        debug_assert_eq!(
            params.bucket_plaintext_size(),
            header + body,
            "size formula mismatch — bucket_plaintext_size out of sync with serialise"
        );
        let mut buf = vec![0u8; header + body];
        buf[0] = self.dummy_flag;
        buf[1..33].copy_from_slice(&self.fingerprint);
        buf[33..37].copy_from_slice(&self.value_count.to_le_bytes());
        for (i, v) in self.values.iter().enumerate() {
            assert_eq!(v.len(), vb, "value not pre-padded");
            let off = header + i * vb;
            buf[off..off + vb].copy_from_slice(v);
        }
        buf
    }

    pub fn deserialise(bytes: &[u8], params: &XorMmParams) -> Self {
        let header = 1 + 32 + 4;
        let vb = params.value_bytes as usize;
        let vol = params.volume_bound as usize;
        assert!(bytes.len() >= header + vol * vb, "bucket too small");

        let dummy_flag = bytes[0];
        let mut fingerprint = [0u8; 32];
        fingerprint.copy_from_slice(&bytes[1..33]);
        let value_count = u32::from_le_bytes(bytes[33..37].try_into().expect("4-byte"));

        let mut values = Vec::with_capacity(vol);
        for i in 0..vol {
            let off = header + i * vb;
            values.push(bytes[off..off + vb].to_vec());
        }
        Self {
            dummy_flag,
            fingerprint,
            value_count,
            values,
        }
    }
}

/// Make 12-byte AES-GCM nonce from `(bucket_id, generation)`. Same
/// nonce-uniqueness argument as `ring-oram::codec::make_nonce`.
pub(crate) fn make_nonce(bucket_id: u32, generation: u32) -> [u8; 12] {
    let mut n = [0u8; 12];
    n[..4].copy_from_slice(&bucket_id.to_le_bytes());
    n[4..8].copy_from_slice(&generation.to_le_bytes());
    // Last 4 bytes left zero — gives 2^32 generations per bucket
    // before nonce reuse, which is far past any LightRAG ingest
    // lifetime.
    n
}

pub(crate) fn aes_encrypt(key: &[u8; 32], bucket_id: u32, generation: u32, pt: &[u8]) -> Vec<u8> {
    let cipher = Aes256Gcm::new_from_slice(key).expect("32B key");
    let nonce = make_nonce(bucket_id, generation);
    cipher
        .encrypt(Nonce::from_slice(&nonce), pt)
        .expect("AES-GCM encrypt cannot fail for valid inputs")
}

pub(crate) fn aes_decrypt(
    key: &[u8; 32],
    bucket_id: u32,
    generation: u32,
    ct: &[u8],
) -> Result<Vec<u8>, AesError> {
    let cipher = Aes256Gcm::new_from_slice(key).expect("32B key");
    let nonce = make_nonce(bucket_id, generation);
    cipher
        .decrypt(Nonce::from_slice(&nonce), ct)
        .map_err(|_| AesError::AuthenticationFailed)
}

#[derive(Debug, thiserror::Error)]
pub enum AesError {
    #[error("XorMM bucket AES-GCM authentication failed")]
    AuthenticationFailed,
}

/// Match a logical key against a fingerprint. Used by `get` to pick
/// the real bucket from the two candidates.
pub(crate) fn fingerprint_matches(key: &LogicalKey, fingerprint: &KeyFingerprint) -> bool {
    &key.0 == fingerprint
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dummy_round_trips() {
        let p = XorMmParams {
            volume_bound: 3,
            value_bytes: 4,
            n_buckets: 8,
            max_kicks: 4,
        };
        let d = BucketPlain::dummy(&p);
        let bytes = d.serialise(&p);
        let back = BucketPlain::deserialise(&bytes, &p);
        assert_eq!(back.dummy_flag, 1);
        assert_eq!(back.fingerprint, DUMMY_FINGERPRINT);
        assert_eq!(back.value_count, 0);
    }

    #[test]
    fn real_bucket_round_trips() {
        let p = XorMmParams {
            volume_bound: 2,
            value_bytes: 4,
            n_buckets: 8,
            max_kicks: 4,
        };
        let mut values = vec![vec![0u8; 4]; 2];
        values[0] = vec![0x11, 0x22, 0x33, 0x44];
        values[1] = vec![0x55, 0x66, 0x77, 0x88];
        let bp = BucketPlain {
            dummy_flag: 0,
            fingerprint: [0xab; 32],
            value_count: 2,
            values: values.clone(),
        };
        let bytes = bp.serialise(&p);
        let back = BucketPlain::deserialise(&bytes, &p);
        assert_eq!(back.fingerprint, [0xab; 32]);
        assert_eq!(back.value_count, 2);
        assert_eq!(back.values[0], values[0]);
        assert_eq!(back.values[1], values[1]);
    }

    #[test]
    fn aes_round_trips_with_nonce_uniqueness() {
        let key = [0x77u8; 32];
        let pt = vec![0xaa; 16];
        let c1 = aes_encrypt(&key, 1, 5, &pt);
        let c2 = aes_encrypt(&key, 1, 6, &pt);
        // Different generations ⇒ different ciphertexts (different nonce).
        assert_ne!(c1, c2);
        let back = aes_decrypt(&key, 1, 5, &c1).expect("auth ok");
        assert_eq!(back, pt);
        // Wrong generation ⇒ auth fail.
        assert!(aes_decrypt(&key, 1, 6, &c1).is_err());
    }
}

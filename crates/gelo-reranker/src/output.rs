//! [`EncryptedRerankBundle`] — the only thing that leaves a
//! [`RerankService::rerank`] call.
//!
//! Each item is `AES-256-GCM(QueryKey, nonce, payload)` where the
//! payload encodes `(rank, chunk_id, chunk_text)`. The list is shuffled
//! and padded to a fixed `k_max`. A host observing the wire learns:
//! `k_max` (fixed per deployment), per-item ciphertext size (fixed if
//! chunks are length-padded at ingest), and the fact that a query
//! happened. They do not learn which chunks, in what order, with what
//! scores.

use aes_gcm::{
    Aes256Gcm, Nonce,
    aead::{Aead, KeyInit},
};
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::score::RankedItem;
use crate::service::RerankError;
use crate::session::QueryKey;

/// Fixed-size component of the wire format. `(nonce, ciphertext)`
/// pairs — the rank lives inside the encrypted payload, not in the
/// list index. The list itself is shuffled before emission so position
/// also carries no information.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EncryptedRerankItem {
    /// AES-GCM nonce (12 bytes for AES-256-GCM with the 96-bit IV
    /// profile). Returned as `Vec<u8>` to keep the on-wire JSON shape
    /// stable across nonce-size variants.
    pub nonce: Vec<u8>,
    pub ciphertext: Vec<u8>,
}

/// The complete rerank response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EncryptedRerankBundle {
    /// AEAD scheme tag — pinned to `"aes-256-gcm.v1"`. Always present
    /// so a future scheme migration can fail closed on the client.
    pub scheme: &'static str,
    /// Fixed-shape list of `k_max` items. Order conveys no rank
    /// information — the client decrypts every item and sorts by the
    /// embedded `rank` field.
    pub items: Vec<EncryptedRerankItem>,
}

/// The cleartext that lives inside one [`EncryptedRerankItem`]. Two
/// variants because we pad with decoy items: real ranks carry a chunk;
/// decoys carry a tag the client uses to drop them after decryption.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RerankPayload {
    Real {
        rank: u32,
        chunk_id: String,
        text: String,
    },
    Decoy,
}

impl EncryptedRerankBundle {
    /// Serialize each ranked item plus `k_max - top_k.len()` decoys,
    /// shuffle, AEAD-encrypt under `key`. RNG is used for nonces, decoy
    /// payloads, and the final shuffle — pass a [`rand_chacha::ChaCha20Rng`]
    /// seeded from the [`QueryKey`] in production paths for
    /// reproducible-yet-unpredictable nonces.
    pub fn seal<R: Rng + ?Sized>(
        key: &QueryKey,
        ranked: &[RankedItem],
        k_max: usize,
        rng: &mut R,
        decoy_text_len: usize,
    ) -> Result<Self, RerankError> {
        if ranked.len() > k_max {
            return Err(RerankError::InvalidRequest(format!(
                "ranked.len()={} exceeds k_max={}",
                ranked.len(),
                k_max
            )));
        }

        let cipher = Aes256Gcm::new_from_slice(key.as_bytes())
            .expect("32-byte AES-256-GCM key");

        let mut items: Vec<EncryptedRerankItem> = Vec::with_capacity(k_max);

        for item in ranked {
            let payload = RerankPayload::Real {
                rank: item.rank,
                chunk_id: item.chunk_id.0.clone(),
                text: item.text.clone(),
            };
            items.push(encrypt_payload(&cipher, &payload, rng)?);
        }

        // Pad with decoys whose plaintext size matches a real item's
        // ballpark so per-item length doesn't betray decoy positions
        // after a length-padded ingest. The caller picks
        // `decoy_text_len` to match the corpus's per-chunk padded
        // length.
        for _ in ranked.len()..k_max {
            let mut filler = vec![0u8; decoy_text_len];
            rng.fill_bytes(&mut filler);
            let decoy = RerankPayload::Decoy;
            items.push(encrypt_payload_with_aad(&cipher, &decoy, &filler, rng)?);
        }

        items.shuffle(rng);

        Ok(Self {
            scheme: "aes-256-gcm.v1",
            items,
        })
    }

    /// Client-side decrypt. Returns the real items sorted by embedded
    /// `rank` ascending, with decoys filtered out. Used in tests; the
    /// production client is in a separate workspace consumer.
    pub fn open(&self, key: &QueryKey) -> Result<Vec<DecryptedRerankItem>, RerankError> {
        if self.scheme != "aes-256-gcm.v1" {
            return Err(RerankError::InvalidRequest(format!(
                "unsupported bundle scheme {}",
                self.scheme
            )));
        }
        let cipher = Aes256Gcm::new_from_slice(key.as_bytes())
            .expect("32-byte AES-256-GCM key");

        let mut reals: Vec<DecryptedRerankItem> = Vec::new();
        for item in &self.items {
            let nonce = Nonce::from_slice(&item.nonce);
            let plain = cipher
                .decrypt(nonce, item.ciphertext.as_ref())
                .map_err(|_| RerankError::Aead("decrypt"))?;
            let payload: RerankPayload = serde_json::from_slice(&plain)
                .map_err(|_| RerankError::Aead("payload decode"))?;
            match payload {
                RerankPayload::Real { rank, chunk_id, text } => {
                    reals.push(DecryptedRerankItem { rank, chunk_id, text });
                }
                RerankPayload::Decoy => {}
            }
        }
        reals.sort_by_key(|r| r.rank);
        Ok(reals)
    }
}

/// Plain-text view recovered by [`EncryptedRerankBundle::open`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecryptedRerankItem {
    pub rank: u32,
    pub chunk_id: String,
    pub text: String,
}

fn encrypt_payload<R: Rng + ?Sized>(
    cipher: &Aes256Gcm,
    payload: &RerankPayload,
    rng: &mut R,
) -> Result<EncryptedRerankItem, RerankError> {
    let mut nonce = [0u8; 12];
    rng.fill_bytes(&mut nonce);
    let plain = serde_json::to_vec(payload)
        .map_err(|_| RerankError::Aead("payload encode"))?;
    let ct = cipher
        .encrypt(Nonce::from_slice(&nonce), plain.as_ref())
        .map_err(|_| RerankError::Aead("encrypt"))?;
    Ok(EncryptedRerankItem {
        nonce: nonce.to_vec(),
        ciphertext: ct,
    })
}

fn encrypt_payload_with_aad<R: Rng + ?Sized>(
    cipher: &Aes256Gcm,
    payload: &RerankPayload,
    filler: &[u8],
    rng: &mut R,
) -> Result<EncryptedRerankItem, RerankError> {
    // Decoy items must look the same size as a real ranked chunk on
    // the wire. We achieve that by appending `filler` to the JSON
    // payload before encryption — the client checks `kind == decoy`
    // and discards. Using AAD instead would leave a stable AEAD-tag
    // boundary the host could exploit; padding the plaintext is
    // simpler and constant-shape.
    let _ = filler; // length absorbed below
    let mut nonce = [0u8; 12];
    rng.fill_bytes(&mut nonce);
    let mut plain = serde_json::to_vec(payload)
        .map_err(|_| RerankError::Aead("payload encode"))?;
    let pad_len = filler.len().saturating_sub(plain.len());
    plain.extend(std::iter::repeat(b' ').take(pad_len));
    let ct = cipher
        .encrypt(Nonce::from_slice(&nonce), plain.as_ref())
        .map_err(|_| RerankError::Aead("encrypt"))?;
    Ok(EncryptedRerankItem {
        nonce: nonce.to_vec(),
        ciphertext: ct,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::RankedItem;
    use crate::session::{QueryId, SessionKey, SessionKeyPolicy};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rag_core::ChunkId;
    use zeroize::Zeroizing;

    fn ranked(items: &[(u32, &str, &str)]) -> Vec<RankedItem> {
        items
            .iter()
            .map(|(r, id, text)| RankedItem {
                rank: *r,
                chunk_id: ChunkId((*id).into()),
                text: (*text).into(),
            })
            .collect()
    }

    fn fixed_session() -> SessionKey {
        let secret = Zeroizing::new(vec![0xab; 32]);
        SessionKey::derive(&secret, SessionKeyPolicy::V1)
    }

    #[test]
    fn seal_then_open_round_trip_preserves_rank_and_text() {
        let session = fixed_session();
        let qkey = session.derive_query_key(&QueryId::from("q-1"));
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let r = ranked(&[
            (0, "chunk-a", "alpha text"),
            (1, "chunk-b", "beta text"),
            (2, "chunk-c", "gamma text"),
        ]);
        let bundle = EncryptedRerankBundle::seal(&qkey, &r, 8, &mut rng, 32).unwrap();
        assert_eq!(bundle.items.len(), 8);

        let opened = bundle.open(&qkey).unwrap();
        assert_eq!(opened.len(), 3);
        assert_eq!(opened[0].rank, 0);
        assert_eq!(opened[0].chunk_id, "chunk-a");
        assert_eq!(opened[1].chunk_id, "chunk-b");
        assert_eq!(opened[2].chunk_id, "chunk-c");
    }

    #[test]
    fn bundle_is_padded_to_k_max() {
        let session = fixed_session();
        let qkey = session.derive_query_key(&QueryId::from("q-pad"));
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let r = ranked(&[(0, "only", "text")]);
        let bundle = EncryptedRerankBundle::seal(&qkey, &r, 16, &mut rng, 32).unwrap();
        assert_eq!(bundle.items.len(), 16);
        let opened = bundle.open(&qkey).unwrap();
        assert_eq!(opened.len(), 1);
    }

    #[test]
    fn wrong_query_key_fails_to_open() {
        let session = fixed_session();
        let qkey = session.derive_query_key(&QueryId::from("q-real"));
        let other = session.derive_query_key(&QueryId::from("q-other"));
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let r = ranked(&[(0, "a", "x")]);
        let bundle = EncryptedRerankBundle::seal(&qkey, &r, 4, &mut rng, 16).unwrap();
        let err = bundle.open(&other);
        assert!(err.is_err(), "decryption with wrong query key must fail");
    }

    #[test]
    fn rejects_ranked_longer_than_k_max() {
        let session = fixed_session();
        let qkey = session.derive_query_key(&QueryId::from("q-overflow"));
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let r = ranked(&[
            (0, "a", ""),
            (1, "b", ""),
            (2, "c", ""),
        ]);
        let err = EncryptedRerankBundle::seal(&qkey, &r, 2, &mut rng, 0);
        assert!(err.is_err());
    }
}

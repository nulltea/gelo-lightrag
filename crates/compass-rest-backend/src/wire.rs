//! On-the-wire types for the REST `BlockBackend` protocol.
//!
//! Body encoding: msgpack via `rmp-serde`. Ciphertext fields use
//! `serde_bytes::ByteBuf` so they encode as msgpack `bin` (one length
//! prefix + raw bytes) instead of the default `array` (one byte per
//! element). This keeps the wire size flat — each `EncryptedBucket`
//! roughly `block_bytes + ~30 B` overhead vs `block_bytes * 2` under
//! the array encoding.

use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;

use ring_oram::EncryptedBucket;

/// One bucket as it travels on the wire. Bidirectional — both
/// `read_path` responses and `write_buckets` requests carry these.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WireBucket {
    pub bucket_id: u32,
    pub write_counter: u32,
    pub ciphertext: ByteBuf,
}

impl From<&EncryptedBucket> for WireBucket {
    fn from(eb: &EncryptedBucket) -> Self {
        Self {
            bucket_id: eb.bucket_id,
            write_counter: eb.write_counter,
            ciphertext: ByteBuf::from(eb.ciphertext.clone()),
        }
    }
}

impl From<WireBucket> for EncryptedBucket {
    fn from(w: WireBucket) -> Self {
        Self {
            bucket_id: w.bucket_id,
            write_counter: w.write_counter,
            ciphertext: w.ciphertext.into_vec(),
        }
    }
}

/// `POST /v1/{tenant}/{index}/read_path` request body.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReadPathRequest {
    pub bucket_ids: Vec<u32>,
}

/// `POST /v1/{tenant}/{index}/read_path` response body.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReadPathResponse {
    pub buckets: Vec<WireBucket>,
}

/// `POST /v1/{tenant}/{index}/write_buckets` request body.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WriteBucketsRequest {
    pub buckets: Vec<WireBucket>,
}

/// `POST /v1/{tenant}/{index}/init` request body. Used at first-time
/// setup to allocate `num_buckets` slots; the server returns 200 OK
/// once the tenant/index pair is materialised. Idempotent — replays
/// on an existing pair are a no-op.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InitRequest {
    pub num_buckets: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_bucket_round_trips_through_msgpack() {
        let original = EncryptedBucket {
            bucket_id: 17,
            write_counter: 42,
            ciphertext: vec![0xde, 0xad, 0xbe, 0xef, 0xfe, 0xed],
        };
        let wire: WireBucket = (&original).into();
        let bytes = rmp_serde::to_vec(&wire).unwrap();
        let back: WireBucket = rmp_serde::from_slice(&bytes).unwrap();
        let restored: EncryptedBucket = back.into();
        assert_eq!(restored.bucket_id, original.bucket_id);
        assert_eq!(restored.write_counter, original.write_counter);
        assert_eq!(restored.ciphertext, original.ciphertext);
    }

    #[test]
    fn read_path_request_round_trips() {
        let req = ReadPathRequest {
            bucket_ids: vec![0, 1, 3, 7, 15, 31],
        };
        let bytes = rmp_serde::to_vec(&req).unwrap();
        let back: ReadPathRequest = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back.bucket_ids, req.bucket_ids);
    }
}

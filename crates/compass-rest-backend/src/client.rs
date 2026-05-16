//! `RestBlockBackend` — the network-shaped `BlockBackend` impl. Lives
//! inside the CVM; talks to a `compass-rest-backend` server (or an
//! equivalent runner-collocated process) over `reqwest`.
//!
//! One `reqwest::Client` is reused across calls for connection
//! pooling. `BlockBackend::num_buckets` is cached once at construction
//! by querying `/init` — this matches the trait's sync return type
//! and avoids a round-trip on every ORAM read.

use anyhow::Context;
use async_trait::async_trait;
use reqwest::Url;
use thiserror::Error;

use ring_oram::{BackendError, BlockBackend, EncryptedBucket};

use crate::wire::{
    InitRequest, ReadPathRequest, ReadPathResponse, WireBucket, WriteBucketsRequest,
};

const CONTENT_TYPE_MSGPACK: &str = "application/msgpack";

#[derive(Debug, Error)]
pub enum RestBackendError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("URL build error: {0}")]
    Url(#[from] url::ParseError),
    #[error("codec error: {0}")]
    Codec(String),
    #[error("server returned status {status} for {url}: {body}")]
    Status {
        status: u16,
        url: String,
        body: String,
    },
}

/// Client-side `BlockBackend` over the REST wire protocol defined in
/// `wire.rs`. Holds one `reqwest::Client` plus the resolved tenant +
/// index URL prefix.
#[derive(Debug, Clone)]
pub struct RestBlockBackend {
    http: reqwest::Client,
    base: Url,
    num_buckets: u32,
}

impl RestBlockBackend {
    /// Construct, sanity-check the server, and cache `num_buckets`.
    /// `base_url` is e.g. `https://runner.example/v1/alpha/entities`;
    /// the trailing slash is normalised.
    pub async fn connect(base_url: &str, num_buckets: u32) -> Result<Self, BackendError> {
        let base = normalise_base(base_url).context("parse REST backend base URL")?;
        let http = reqwest::Client::builder()
            .build()
            .context("build reqwest client")?;

        let this = Self {
            http,
            base,
            num_buckets,
        };
        this.init().await?;
        Ok(this)
    }

    async fn init(&self) -> Result<(), BackendError> {
        let url = self.base.join("init").context("join init URL")?;
        let body = rmp_serde::to_vec(&InitRequest {
            num_buckets: self.num_buckets,
        })
        .context("encode InitRequest")?;
        let resp = self
            .http
            .post(url.clone())
            .header(reqwest::header::CONTENT_TYPE, CONTENT_TYPE_MSGPACK)
            .body(body)
            .send()
            .await
            .context("send init")?;
        check_status(resp, &url).await.map(|_| ())
    }
}

#[async_trait]
impl BlockBackend for RestBlockBackend {
    async fn read_path(&self, bucket_ids: &[u32]) -> Result<Vec<EncryptedBucket>, BackendError> {
        let url = self.base.join("read_path").context("join read_path URL")?;
        let body = rmp_serde::to_vec(&ReadPathRequest {
            bucket_ids: bucket_ids.to_vec(),
        })
        .context("encode ReadPathRequest")?;

        let resp = self
            .http
            .post(url.clone())
            .header(reqwest::header::CONTENT_TYPE, CONTENT_TYPE_MSGPACK)
            .body(body)
            .send()
            .await
            .context("send read_path")?;
        let bytes = check_status(resp, &url).await?;
        let response: ReadPathResponse =
            rmp_serde::from_slice(&bytes).context("decode read_path response")?;

        Ok(response
            .buckets
            .into_iter()
            .map(EncryptedBucket::from)
            .collect())
    }

    async fn write_buckets(&mut self, buckets: &[EncryptedBucket]) -> Result<(), BackendError> {
        let url = self
            .base
            .join("write_buckets")
            .context("join write_buckets URL")?;
        let wire_buckets: Vec<WireBucket> = buckets.iter().map(WireBucket::from).collect();
        let body = rmp_serde::to_vec(&WriteBucketsRequest {
            buckets: wire_buckets,
        })
        .context("encode WriteBucketsRequest")?;
        let resp = self
            .http
            .post(url.clone())
            .header(reqwest::header::CONTENT_TYPE, CONTENT_TYPE_MSGPACK)
            .body(body)
            .send()
            .await
            .context("send write_buckets")?;
        check_status(resp, &url).await?;
        Ok(())
    }

    fn num_buckets(&self) -> u32 {
        self.num_buckets
    }
}

/// Normalise so `base_url` always ends with `/`. `Url::join` resolves
/// the segment relative to the path component — without the trailing
/// slash, `.join("init")` would replace the *last segment* of the base
/// instead of appending.
fn normalise_base(s: &str) -> Result<Url, url::ParseError> {
    let trimmed = s.trim_end_matches('/');
    Url::parse(&format!("{trimmed}/"))
}

async fn check_status(resp: reqwest::Response, url: &Url) -> Result<Vec<u8>, BackendError> {
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(RestBackendError::Status {
            status: status.as_u16(),
            url: url.to_string(),
            body,
        }
        .into());
    }
    Ok(resp.bytes().await.context("read response body")?.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalise_base_appends_trailing_slash() {
        let u = normalise_base("https://runner.example/v1/alpha/entities").unwrap();
        assert_eq!(u.as_str(), "https://runner.example/v1/alpha/entities/");
        // Idempotent.
        let again = normalise_base("https://runner.example/v1/alpha/entities/").unwrap();
        assert_eq!(again.as_str(), u.as_str());
    }
}

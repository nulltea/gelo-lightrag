//! `LightRagTwoPartyService` — multi-tenant LightRAG service.
//!
//! Mirrors `gelo-rag::GeloRagTwoPartyService` (CAPRISE in-TEE) but for
//! the LightRAG store + retrieval surface. Per-tenant state:
//!
//! - `tee_user_x_sk` — Variant-A: in-memory only, lost on CVM
//!   restart. First-touch creates one via OS RNG; on a known tenant
//!   we require it (else 410 Gone like the gelo-rag service).
//! - `LightKgStore<InMemoryBlockBackend, InMemoryByteStore>` — built
//!   once at `ingest_kg_for`, queried thereafter. The store *holds*
//!   the derived OramKey/EmmKey/aes_chunk_key directly (different
//!   pattern from the CAPRISE service: rebuilding the HNSW per
//!   request is not viable). `user_x_sk` is consumed and zeroized
//!   at ingest time; subsequent queries can run without it because
//!   the store already has the derived keys in CVM RAM.
//! - `search_pattern_key` derived once at ingest from
//!   (user_x_sk, tee_sk, tenant). M8.x will re-derive per-request
//!   to keep this consistent with the CAPRISE pattern; M8.0 holds
//!   it inside the store.

use std::collections::HashMap;
use std::sync::Arc;

use light_kg_store::{
    ExtractedKg, InMemoryBlockBackend, InMemoryByteStore, LightKgError, LightKgParams,
    LightKgStore,
};
use rag_core::TenantId;
use rag_core::keying::{HkdfPolicyV2, SchemeParamsV2};
use rand::RngCore;
use tokio::sync::Mutex;
use zeroize::Zeroizing;

use crate::perturb::SessionKey;
use crate::service::{KgContext, KgQueryParams, LightRagPrivateService};

#[derive(thiserror::Error, Debug)]
pub enum LightRagServiceError {
    #[error("tenant {0} unknown — re-bootstrap the tenant (CVM may have restarted)")]
    UnknownTenant(TenantId),
    #[error("light-kg-store error: {0}")]
    Store(#[from] LightKgError),
    #[error(transparent)]
    Inner(#[from] anyhow::Error),
}

/// Per-tenant state. Held behind a `tokio::Mutex` so concurrent
/// requests serialize at the tenant boundary — the underlying
/// `LightKgStore` is single-threaded (RingOramClient mutates its
/// stash per read, so reads aren't `&self`).
struct Tenant {
    tee_user_x_sk: Zeroizing<[u8; 32]>,
    /// The store + the V2 HKDF children kept alongside (the store
    /// itself holds some derived keys; the leftover `search_pattern
    /// _key` lives here too — `KgQuery` needs it to mint
    /// `SessionKey`s).
    store: LightKgStore<InMemoryBlockBackend, InMemoryByteStore>,
    search_pattern_key: Zeroizing<[u8; 32]>,
}

pub struct LightRagTwoPartyService {
    tenants: Mutex<HashMap<TenantId, Arc<Mutex<Tenant>>>>,
    hkdf_policy: HkdfPolicyV2,
    scheme_params: SchemeParamsV2,
}

impl LightRagTwoPartyService {
    pub fn new() -> Self {
        Self::with_params(HkdfPolicyV2::V2, SchemeParamsV2::default())
    }

    pub fn with_params(hkdf_policy: HkdfPolicyV2, scheme_params: SchemeParamsV2) -> Self {
        Self {
            tenants: Mutex::new(HashMap::new()),
            hkdf_policy,
            scheme_params,
        }
    }

    pub fn scheme_identity(&self) -> [u8; 32] {
        self.hkdf_policy.scheme_identity_digest(self.scheme_params)
    }

    pub async fn tenant_known(&self, tenant_id: &TenantId) -> bool {
        self.tenants.lock().await.contains_key(tenant_id)
    }

    pub async fn forget_tenant(&self, tenant_id: &TenantId) {
        self.tenants.lock().await.remove(tenant_id);
    }

    /// First-touch ingest: generates `tee_user_x_sk`, derives V2
    /// children, builds the store, holds onto it. Subsequent
    /// `ingest_kg_for` calls on a known tenant REPLACE the store —
    /// the M8.0 contract is "ingest = full rebuild". Incremental
    /// ingest is M8.x (fresh-tier in XorMM, M2.1).
    pub async fn ingest_kg_for(
        &self,
        tenant_id: &TenantId,
        user_x_sk: Zeroizing<[u8; 32]>,
        kg: ExtractedKg,
        params: LightKgParams,
    ) -> Result<(), LightRagServiceError> {
        let tee_sk = self.or_create_tee_secret(tenant_id).await;
        let derived = self
            .hkdf_policy
            .derive(&user_x_sk, &tee_sk, tenant_id);

        let store = LightKgStore::build_from_kg(kg, params, &derived).await?;

        // Clone the search_pattern_key out so we can hand it to the
        // query path. `DerivedKeysV2` is dropped at function end
        // (Zeroizing) — the surviving Zeroizing<[u8; 32]> in `tenant`
        // is the only copy.
        let mut spk: Zeroizing<[u8; 32]> = Zeroizing::new([0u8; 32]);
        spk.copy_from_slice(derived.search_pattern_key.as_ref());

        let tenant = Tenant {
            tee_user_x_sk: tee_sk,
            store,
            search_pattern_key: spk,
        };
        self.tenants
            .lock()
            .await
            .insert(tenant_id.clone(), Arc::new(Mutex::new(tenant)));
        Ok(())
    }

    /// Query path. Returns a `KgContext` ready to render with
    /// `.to_context_string()`. Mode is Local-only in M8.0; the
    /// other modes light up as M7.x adds their shapes.
    pub async fn query_for(
        &self,
        tenant_id: &TenantId,
        ll_query_embedding: &[f32],
        params: &KgQueryParams,
        session_nonce: &[u8],
    ) -> Result<KgContext, LightRagServiceError> {
        let arc = self
            .tenants
            .lock()
            .await
            .get(tenant_id)
            .cloned()
            .ok_or_else(|| LightRagServiceError::UnknownTenant(tenant_id.clone()))?;
        let mut tenant = arc.lock().await;

        let session_key = SessionKey::derive(&tenant.search_pattern_key, session_nonce);
        let mut svc = LightRagPrivateService::new(&mut tenant.store);
        Ok(svc
            .kg_query(ll_query_embedding, params, &session_key)
            .await?)
    }

    async fn or_create_tee_secret(&self, tenant_id: &TenantId) -> Zeroizing<[u8; 32]> {
        let tenants = self.tenants.lock().await;
        if let Some(existing) = tenants.get(tenant_id) {
            let tenant = existing.lock().await;
            return clone_secret(&tenant.tee_user_x_sk);
        }
        // Not yet ingested — fresh tee_sk; the *store* will be built
        // by the ingest path right after we return.
        let mut buf: Zeroizing<[u8; 32]> = Zeroizing::new([0u8; 32]);
        rand::rng().fill_bytes(buf.as_mut());
        // Don't insert a placeholder Tenant — the caller (ingest_kg
        // _for) does the full insert with the built store right
        // after. Drop the lock and return the secret.
        drop(tenants);
        buf
    }
}

impl Default for LightRagTwoPartyService {
    fn default() -> Self {
        Self::new()
    }
}

fn clone_secret(src: &Zeroizing<[u8; 32]>) -> Zeroizing<[u8; 32]> {
    let mut dst: Zeroizing<[u8; 32]> = Zeroizing::new([0u8; 32]);
    dst.copy_from_slice(src.as_ref());
    dst
}

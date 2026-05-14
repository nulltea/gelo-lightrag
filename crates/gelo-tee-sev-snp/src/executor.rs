//! `SnpTrustedExecutor` — wraps [`gelo_protocol::sim::InProcessTrustedExecutor`]
//! with SEV-SNP attestation-evidence assembly.
//!
//! The wrapper is intentionally thin: every `TrustedExecutor` method forwards
//! to `self.inner`. SEV-SNP is the **boundary**, not a change to the protocol
//! math — the mask/shield/U-Verify logic is identical in the in-process
//! simulator and on real EPYC silicon. What this wrapper adds is:
//!
//! 1. A `model_identity` / `scheme_identity` pair that gets hashed into
//!    `REPORT_DATA[0..32]` and `REPORT_DATA[32..64]` of the attestation
//!    report (see [`crate::report_data::ReportData`]).
//! 2. An issuer trait — [`AttestationIssuer`] — abstracting over how the
//!    1184-byte SEV-SNP report and VCEK certificate are produced. The mock
//!    impl ([`crate::mock::MockReportIssuer`]) signs with the bundled test
//!    PKI; the real hardware impl (`crate::hardware`, behind the `sev-snp`
//!    feature in M5.6) opens `/dev/sev-guest`.
//! 3. An `evidence()` method that the higher-level service calls to attach
//!    attestation evidence to an outgoing response.
//!
//! The wrapper deliberately re-implements `TrustedExecutor` rather than
//! exposing the inner executor directly so the embedder type-erases the
//! attestation backend (`GeloRagInMemoryService<… , SnpTrustedExecutor<…>>`
//! is the deployment target).

use std::sync::Arc;

use anyhow::Result;
use gelo_protocol::sim::InProcessTrustedExecutor;
use gelo_protocol::substrate::{GpuOffloadEngine, TrustedExecutor, WeightHandle};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};

use crate::report_data::ReportData;

/// Bytes carried back to a relying party. Mirrors the optional fields added
/// to `gelo_rag::attestation::AttestationEvidence` in M5.5; kept here as a
/// crate-local type so `gelo-tee-sev-snp` doesn't depend on `gelo-rag`.
#[derive(Clone, Debug)]
pub struct SnpEvidence {
    /// 1184-byte SEV-SNP attestation report.
    pub report_bytes: Vec<u8>,
    /// PEM-encoded VCEK certificate the report was signed with.
    pub vcek_cert_pem: Vec<u8>,
}

/// Abstraction over "where do report bytes come from".
///
/// Production impl opens `/dev/sev-guest` and issues `SNP_GET_REPORT`
/// (M5.6). Mock impl in `crate::mock::MockReportIssuer` fake-signs against
/// the bundled test PKI.
pub trait AttestationIssuer: Send + Sync {
    fn issue(&self, report_data: ReportData) -> Result<SnpEvidence>;
}

#[cfg(feature = "mock")]
impl AttestationIssuer for crate::mock::MockReportIssuer {
    fn issue(&self, report_data: ReportData) -> Result<SnpEvidence> {
        let issued = crate::mock::MockReportIssuer::issue(self, report_data)?;
        Ok(SnpEvidence {
            report_bytes: issued.report_bytes,
            vcek_cert_pem: issued.vcek_cert_pem,
        })
    }
}

/// SEV-SNP-attested trusted executor.
///
/// Holds the inner `InProcessTrustedExecutor` (the actual protocol engine)
/// plus the identity pair that gets baked into every attestation report this
/// executor issues. `model_identity` is **publicly** known (sha256 of the
/// weights manifest); `scheme_identity` covers protocol-secret state
/// (`MaskSeed` + `ShieldConfig`).
pub struct SnpTrustedExecutor<E: GpuOffloadEngine, I: AttestationIssuer> {
    inner: InProcessTrustedExecutor<E>,
    issuer: I,
    model_identity: Vec<u8>,
    scheme_identity: Vec<u8>,
}

/// Clone for the SNP wrapper. Delegates to the inner executor's Clone
/// (which Arc-shares the engine's weight cache) and clones the issuer +
/// identity bytes. Same threading semantics as the inner executor: the
/// session mask and stacked scratch are NOT carried across clones — each
/// clone starts a fresh forward-pass scope.
impl<E, I> Clone for SnpTrustedExecutor<E, I>
where
    E: GpuOffloadEngine + Clone,
    I: AttestationIssuer + Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            issuer: self.issuer.clone(),
            model_identity: self.model_identity.clone(),
            scheme_identity: self.scheme_identity.clone(),
        }
    }
}

impl<E: GpuOffloadEngine, I: AttestationIssuer> SnpTrustedExecutor<E, I> {
    pub fn new(
        inner: InProcessTrustedExecutor<E>,
        issuer: I,
        model_identity: impl Into<Vec<u8>>,
        scheme_identity: impl Into<Vec<u8>>,
    ) -> Self {
        Self {
            inner,
            issuer,
            model_identity: model_identity.into(),
            scheme_identity: scheme_identity.into(),
        }
    }

    pub fn inner(&self) -> &InProcessTrustedExecutor<E> {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut InProcessTrustedExecutor<E> {
        &mut self.inner
    }

    pub fn model_identity(&self) -> &[u8] {
        &self.model_identity
    }

    pub fn scheme_identity(&self) -> &[u8] {
        &self.scheme_identity
    }

    /// Assemble fresh attestation evidence binding the executor's identity
    /// pair (and an optional caller-supplied session nonce) into
    /// `REPORT_DATA`. Each call issues a *new* report — the SEV-SNP
    /// `SNP_GET_REPORT` ioctl is cheap (~ms) and per-session freshness lets
    /// the relying party guard against replay.
    pub fn evidence(&self, nonce: Option<&[u8]>) -> Result<SnpEvidence> {
        let rd = ReportData::build(&self.model_identity, &self.scheme_identity, nonce);
        self.issuer.issue(rd)
    }
}

impl<E: GpuOffloadEngine, I: AttestationIssuer> TrustedExecutor
    for SnpTrustedExecutor<E, I>
{
    fn begin_forward_pass(&mut self, n: usize) -> Result<()> {
        self.inner.begin_forward_pass(n)
    }

    fn end_forward_pass(&mut self) -> Result<()> {
        self.inner.end_forward_pass()
    }

    fn provision_weight(
        &mut self,
        handle: WeightHandle,
        weight: ArrayView2<f32>,
    ) -> Result<()> {
        self.inner.provision_weight(handle, weight)
    }

    fn provision_weight_shared(
        &mut self,
        handle: WeightHandle,
        weight: Arc<Array2<f32>>,
    ) -> Result<()> {
        self.inner.provision_weight_shared(handle, weight)
    }

    fn offload_linear(
        &mut self,
        handle: WeightHandle,
        hidden: ArrayView2<f32>,
    ) -> Result<Array2<f32>> {
        self.inner.offload_linear(handle, hidden)
    }

    fn offload_qkv(
        &mut self,
        layer: u16,
        hidden: ArrayView2<f32>,
    ) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>)> {
        self.inner.offload_qkv(layer, hidden)
    }

    fn offload_linear_many(
        &mut self,
        handles: &[gelo_protocol::substrate::WeightHandle],
        hidden: ArrayView2<f32>,
    ) -> Result<Vec<Array2<f32>>> {
        self.inner.offload_linear_many(handles, hidden)
    }

    fn set_rng_stream(&mut self, stream: u64) {
        self.inner.set_rng_stream(stream);
    }

    fn offload_attention_qkt(
        &mut self,
        q: ArrayView2<f32>,
        kt: ArrayView2<f32>,
    ) -> Result<Array2<f32>> {
        self.inner.offload_attention_qkt(q, kt)
    }

    fn offload_attention_qkt_batched(
        &mut self,
        q: ArrayView3<f32>,
        kt: ArrayView3<f32>,
    ) -> Result<Array3<f32>> {
        self.inner.offload_attention_qkt_batched(q, kt)
    }
}

#[cfg(all(test, feature = "mock"))]
mod tests {
    use super::*;
    use crate::mock::MockReportIssuer;
    use crate::report::parse_report;
    use crate::verify::{AttestedBinding, SnpAttestationVerifier, SnpRootTrust};
    use gelo_protocol::rng::MaskSeed;
    use gelo_protocol::sim::RayonCpuEngine;

    fn mk_executor()
    -> SnpTrustedExecutor<RayonCpuEngine, MockReportIssuer> {
        let inner = InProcessTrustedExecutor::with_seed(
            RayonCpuEngine::new(),
            MaskSeed::from_bytes([7u8; 32]),
        );
        let issuer = MockReportIssuer::from_bundled().unwrap();
        SnpTrustedExecutor::new(inner, issuer, b"model-id".to_vec(), b"scheme-id".to_vec())
    }

    #[test]
    fn evidence_round_trip_through_verifier() {
        let exec = mk_executor();
        let evidence = exec.evidence(Some(b"nonce-1")).unwrap();

        let verifier = SnpAttestationVerifier::new(SnpRootTrust::with_mock_root());
        verifier
            .verify(
                &evidence.report_bytes,
                &evidence.vcek_cert_pem,
                AttestedBinding {
                    model_identity: b"model-id",
                    scheme_identity: b"scheme-id",
                    nonce: Some(b"nonce-1"),
                },
            )
            .expect("evidence issued by executor must verify with matching binding");
    }

    #[test]
    fn evidence_carries_correct_report_data() {
        let exec = mk_executor();
        let evidence = exec.evidence(None).unwrap();
        let parsed = parse_report(&evidence.report_bytes).unwrap();
        let expected = ReportData::build(b"model-id", b"scheme-id", None);
        assert_eq!(&parsed.report_data[..], expected.as_bytes());
    }

    /// Wrong binding ⇒ verifier rejects the report-data check. Sanity that
    /// the executor's `model_identity`/`scheme_identity` are actually load-
    /// bearing.
    #[test]
    fn wrong_binding_rejected() {
        let exec = mk_executor();
        let evidence = exec.evidence(None).unwrap();
        let verifier = SnpAttestationVerifier::new(SnpRootTrust::with_mock_root());
        let err = verifier
            .verify(
                &evidence.report_bytes,
                &evidence.vcek_cert_pem,
                AttestedBinding {
                    model_identity: b"different-model",
                    scheme_identity: b"scheme-id",
                    nonce: None,
                },
            )
            .unwrap_err();
        assert!(matches!(
            err,
            crate::verify::SnpVerifyError::ReportDataMismatch { .. }
        ));
    }

    /// Provisioning a weight via the Arc-share path costs no extra memory
    /// and still gives a working matmul through the wrapper.
    #[test]
    fn provision_weight_shared_through_wrapper() {
        use gelo_protocol::substrate::{WeightHandle, WeightKind};
        use ndarray::Array2;

        let mut exec = mk_executor();
        let weight = Arc::new(Array2::<f32>::from_shape_fn((4, 3), |(i, j)| {
            (i * 3 + j) as f32
        }));
        let handle = WeightHandle::new(0, WeightKind::Q);
        exec.provision_weight_shared(handle, Arc::clone(&weight))
            .unwrap();

        let hidden = Array2::<f32>::from_shape_fn((2, 4), |(i, j)| (i + j) as f32);
        let out = exec.offload_linear(handle, hidden.view()).unwrap();
        let expected = hidden.dot(weight.as_ref());
        for ((i, j), e) in expected.indexed_iter() {
            assert!((out[[i, j]] - e).abs() < 1e-3, "({i},{j}) got {} want {e}", out[[i, j]]);
        }
    }
}

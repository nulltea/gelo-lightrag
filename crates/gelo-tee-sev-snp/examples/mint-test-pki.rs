//! One-shot generator for the bundled mock SEV-SNP test PKI.
//!
//! Produces a three-level chain mirroring AMD's production attestation PKI:
//!
//!   `Mock ARK` (P-384 ECDSA, self-signed, 20-year validity)
//!     └─ `Mock ASK` (P-384, ARK-signed)
//!          └─ `Mock VCEK` (P-384, ASK-signed) — used to sign attestation reports
//!
//! Run once and commit the output under `tests/fixtures/`. Re-running
//! regenerates fresh keys and invalidates anything previously committed.
//!
//! ```ignore
//! cargo run -p gelo-tee-sev-snp --example mint-test-pki --features mock
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use rcgen::{
    CertificateParams, DistinguishedName, DnType, KeyPair, KeyUsagePurpose,
    PKCS_ECDSA_P384_SHA384,
};
use time::{Duration, OffsetDateTime};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn params(common_name: &str) -> CertificateParams {
    let mut p = CertificateParams::default();
    let mut dn = DistinguishedName::new();
    dn.push(DnType::CommonName, common_name);
    dn.push(DnType::OrganizationName, "GELO Mock SEV-SNP PKI");
    p.distinguished_name = dn;
    // Validity: 20-year window starting ~now (rounded to start of UTC day for
    // determinism across re-runs).
    let now_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let start = OffsetDateTime::from_unix_timestamp(now_unix - (now_unix % 86_400))
        .unwrap_or(OffsetDateTime::UNIX_EPOCH);
    p.not_before = start;
    p.not_after = start + Duration::days(365 * 20);
    p.key_usages = vec![KeyUsagePurpose::DigitalSignature, KeyUsagePurpose::KeyCertSign];
    p
}

fn write(path: &Path, content: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, content).with_context(|| format!("writing {}", path.display()))?;
    println!("wrote {}", path.display());
    Ok(())
}

fn main() -> Result<()> {
    let out = fixtures_dir();

    // ARK (self-signed root).
    let ark_key = KeyPair::generate_for(&PKCS_ECDSA_P384_SHA384)?;
    let mut ark_params = params("Mock AMD ARK");
    ark_params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
    let ark_cert = ark_params.self_signed(&ark_key)?;
    write(&out.join("mock_ark.pem"), &ark_cert.pem())?;
    write(&out.join("mock_ark.key.pem"), &ark_key.serialize_pem())?;

    // ASK (signed by ARK).
    let ask_key = KeyPair::generate_for(&PKCS_ECDSA_P384_SHA384)?;
    let mut ask_params = params("Mock AMD ASK");
    ask_params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Constrained(0));
    let ask_cert = ask_params.signed_by(&ask_key, &ark_cert, &ark_key)?;
    write(&out.join("mock_ask.pem"), &ask_cert.pem())?;
    write(&out.join("mock_ask.key.pem"), &ask_key.serialize_pem())?;

    // VCEK (signed by ASK, leaf — signs the attestation report).
    let vcek_key = KeyPair::generate_for(&PKCS_ECDSA_P384_SHA384)?;
    let mut vcek_params = params("Mock AMD VCEK");
    vcek_params.is_ca = rcgen::IsCa::ExplicitNoCa;
    let vcek_cert = vcek_params.signed_by(&vcek_key, &ask_cert, &ask_key)?;
    write(&out.join("mock_vcek.pem"), &vcek_cert.pem())?;
    write(&out.join("mock_vcek.key.pem"), &vcek_key.serialize_pem())?;

    println!(
        "\nMock PKI minted under {}\n\
         Commit the .pem files; do NOT commit the .key.pem files to a public repo\n\
         (it's fine here because this is a TEST PKI with explicit mock semantics)",
        out.display()
    );
    Ok(())
}

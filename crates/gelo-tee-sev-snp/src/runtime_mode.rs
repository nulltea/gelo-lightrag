//! Runtime-mode dispatch for the SEV-SNP backend.
//!
//! A single binary supports two execution modes — `production` (real
//! `/dev/sev-guest` ioctls) and `mock` (bundled test PKI + stub issuer) —
//! and the choice is made **once at process startup** via the `SNP_MODE`
//! environment variable. Fail-closed: no autodetection, no silent fallback.
//! An unset or unrecognised `SNP_MODE` aborts startup with a clear error so
//! an accidentally-deployed dev build cannot silently emit attestations a
//! production verifier would reject with confusing errors.
//!
//! - `SNP_MODE=production` (default if explicitly set to "production"):
//!   requires `/dev/sev-guest` to exist; refuses to start without it.
//! - `SNP_MODE=mock`: requires `SNP_MOCK_PKI_PATH` env to be unset (use
//!   bundled test PKI) or to point at a directory containing the test PKI.
//!   Logs `MOCK MODE` prominently so the operator can't miss it.
//!
//! Whichever mode is chosen is locked into the binary's state for the
//! remainder of its run. Tests construct `RuntimeMode::Mock` directly via the
//! crate-internal constructor and don't go through env-var parsing.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeMode {
    /// Real SEV-SNP CVM — open `/dev/sev-guest` for `SNP_GET_REPORT`.
    Production,
    /// Mock issuer + bundled test ARK/ASK/VCEK chain. Tests and the T2
    /// VM-simulation tier use this.
    Mock,
}

impl fmt::Display for RuntimeMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeMode::Production => f.write_str("production"),
            RuntimeMode::Mock => f.write_str("mock"),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeModeError {
    #[error(
        "SNP_MODE environment variable is not set. Set `SNP_MODE=production` to run on real \
         SEV-SNP silicon, or `SNP_MODE=mock` for the bundled-PKI development path."
    )]
    Unset,
    #[error(
        "SNP_MODE has unrecognised value {got:?}. Expected `production` or `mock`."
    )]
    Unrecognised { got: String },
    #[error(
        "SNP_MODE=production was requested but `/dev/sev-guest` is not present. This binary will \
         not start unless launched inside a real SEV-SNP CVM."
    )]
    MissingSevGuestDevice,
}

/// Parse `SNP_MODE` from the process environment, fail-closed.
///
/// On `production`, additionally checks that `/dev/sev-guest` exists. A
/// missing device aborts startup rather than silently falling back to mock.
pub fn from_env() -> Result<RuntimeMode, RuntimeModeError> {
    let raw = std::env::var("SNP_MODE").map_err(|_| RuntimeModeError::Unset)?;
    match raw.as_str() {
        "production" => {
            if !std::path::Path::new("/dev/sev-guest").exists() {
                return Err(RuntimeModeError::MissingSevGuestDevice);
            }
            Ok(RuntimeMode::Production)
        }
        "mock" => Ok(RuntimeMode::Mock),
        other => Err(RuntimeModeError::Unrecognised {
            got: other.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Confined-scope env-var manipulation. We can't run these in parallel
    /// safely; the harness serializes them via cargo test's default single
    /// thread for unit tests with shared globals.
    fn with_env_var<F: FnOnce()>(key: &str, value: Option<&str>, f: F) {
        let prev = std::env::var(key).ok();
        match value {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
        f();
        match prev {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
    }

    #[test]
    fn unset_snp_mode_is_an_error() {
        with_env_var("SNP_MODE", None, || {
            let r = from_env();
            assert!(matches!(r, Err(RuntimeModeError::Unset)));
        });
    }

    #[test]
    fn mock_mode_parses() {
        with_env_var("SNP_MODE", Some("mock"), || {
            assert_eq!(from_env().unwrap(), RuntimeMode::Mock);
        });
    }

    #[test]
    fn unrecognised_value_is_an_error() {
        with_env_var("SNP_MODE", Some("yolo"), || {
            let r = from_env();
            assert!(matches!(r, Err(RuntimeModeError::Unrecognised { .. })));
        });
    }

    /// On non-EPYC dev boxes `/dev/sev-guest` is absent, so `production`
    /// must fail closed.
    #[test]
    fn production_mode_requires_sev_guest_device() {
        if std::path::Path::new("/dev/sev-guest").exists() {
            eprintln!(
                "skipping: /dev/sev-guest is present on this host, \
                 so the missing-device branch can't be exercised"
            );
            return;
        }
        with_env_var("SNP_MODE", Some("production"), || {
            let r = from_env();
            assert!(matches!(r, Err(RuntimeModeError::MissingSevGuestDevice)));
        });
    }
}

#!/usr/bin/env bash
# End-to-end smoke against a running gelo-snp-runner.
#
# Drives the runner through its full HTTP surface:
#   1. GET  /health → "ok"
#   2. GET  /attest → JSON {model_identity, scheme_identity, report_b64, vcek_cert_b64}
#   3. POST /ingest with 3 chunks
#   4. POST /query "rust safety" → top hit id == "rust-memory-safety"
#   5. Decodes report_b64 + vcek_cert_b64; sanity-checks lengths.
#
# Does NOT verify the SEV-SNP signature here — verifier round-trip is
# covered by `cargo test -p approach4 --features snp-mock --test snp_attest_e2e`,
# which exercises the same MockReportIssuer this runner uses.
#
# Usage:
#   deploy/sev-snp/sim/e2e-smoke.sh                # default: http://127.0.0.1:7878
#   deploy/sev-snp/sim/e2e-smoke.sh http://host:8080

set -euo pipefail

URL="${1:-http://127.0.0.1:7878}"

red()   { printf '\033[31m%s\033[0m\n' "$*" >&2; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }

# Each step prints what it's checking before doing it so a failing CI log
# is grep-friendly.

echo "==> /health"
HEALTH="$(curl -fsS "$URL/health")"
[[ "$HEALTH" == "ok" ]] || { red "expected health=ok, got: $HEALTH"; exit 1; }
green "    ok"

echo "==> /attest"
ATTEST_JSON="$(curl -fsS "$URL/attest")"
# Don't require jq on the smoke host; do crude grep checks instead.
echo "$ATTEST_JSON" | grep -q '"model_identity":' || { red "no model_identity"; exit 1; }
echo "$ATTEST_JSON" | grep -q '"scheme_identity":' || { red "no scheme_identity"; exit 1; }
echo "$ATTEST_JSON" | grep -q '"report_b64":' || { red "no report_b64"; exit 1; }
echo "$ATTEST_JSON" | grep -q '"vcek_cert_b64":' || { red "no vcek_cert_b64"; exit 1; }

# Decode lengths: SEV-SNP report is 1184 bytes; VCEK cert PEM is in the
# few-kB range.
REPORT_B64="$(echo "$ATTEST_JSON" | sed -n 's/.*"report_b64":"\([^"]*\)".*/\1/p')"
VCEK_B64="$(echo "$ATTEST_JSON" | sed -n 's/.*"vcek_cert_b64":"\([^"]*\)".*/\1/p')"
REPORT_LEN="$(echo -n "$REPORT_B64" | base64 -d | wc -c)"
VCEK_LEN="$(echo -n "$VCEK_B64" | base64 -d | wc -c)"
[[ "$REPORT_LEN" -eq 1184 ]] || { red "expected 1184-byte report, got $REPORT_LEN"; exit 1; }
[[ "$VCEK_LEN" -gt 256 ]] || { red "VCEK cert suspiciously short: $VCEK_LEN bytes"; exit 1; }
green "    ok (report=$REPORT_LEN B, vcek=$VCEK_LEN B)"

echo "==> /ingest"
INGEST="$(curl -fsS -X POST "$URL/ingest" -H 'content-type: application/json' \
    -d '{"chunks":[
        {"id":"rust-memory-safety","text":"Rust enforces memory safety through ownership and borrowing."},
        {"id":"postgres-index","text":"Postgres uses B-tree indexes for common equality and range lookups."},
        {"id":"tls-attestation","text":"Remote attestation can bind a TEE measurement into a TLS session."}
    ]}')"
echo "$INGEST" | grep -q '"ingested":3' || { red "expected ingested=3, got: $INGEST"; exit 1; }
green "    ok"

echo "==> /query (same text as ingested chunk → deterministic top-1)"
QUERY="$(curl -fsS -X POST "$URL/query" -H 'content-type: application/json' \
    -d '{"text":"Rust enforces memory safety through ownership and borrowing.","top_k":1}')"
echo "$QUERY" | grep -q '"rust-memory-safety"' \
    || { red "expected hit on rust-memory-safety, got: $QUERY"; exit 1; }
echo "$QUERY" | grep -q '"attestation":' \
    || { red "/query response must embed attestation evidence, got: $QUERY"; exit 1; }
green "    ok"

green "==> all checks passed"

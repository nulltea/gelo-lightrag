# T2 — VM-simulated CVM

This directory's scripts boot the production CVM image in a regular
QEMU/KVM guest (no SEV-SNP host needed) and drive it through its full
HTTP surface. The binary inside the image detects `SNP_MODE=mock` and
substitutes the bundled mock issuer for `/dev/sev-guest`; the rest of
the OS — systemd, networking, file I/O, GPU stack — runs for real.

The point: catch OS-boundary and lifecycle bugs *before* spending money
on the T3 EPYC dedicated server.

## Scripts

- `run-cvm-sim.sh` — boots `target/cvm-image/gelo-cvm-image.qcow2` in
  QEMU/KVM, injects `SNP_MODE` via cloud-init, exposes the runner port
  (default 7878) on the host. Build the image first with
  `deploy/sev-snp/cvm-image/setup-cvm-image.sh`.
- `e2e-smoke.sh [URL]` — drives any running `gelo-snp-runner` (default
  `http://127.0.0.1:7878`) through `/health → /attest → /ingest →
  /query`, asserts response shapes, decodes the SEV-SNP report bytes
  and checks the 1184-byte length.
- `smoke-local.sh` — fast CI path: `cargo build` the runner, launch it
  in-process under `SNP_MODE=mock`, run `e2e-smoke.sh` against it.
  Skips the VM entirely. Use this for PR-gating; use `run-cvm-sim.sh`
  for release-gating.

## What's *not* covered here

The smoke script intentionally does **not** verify the SEV-SNP
signature — that's already covered by
`cargo test -p approach4 --features snp-mock --test snp_attest_e2e`,
which exercises the same `MockReportIssuer` path through
`SnpAttestationVerifier`. Duplicating that check at the HTTP layer
would just slow down CI.

Hardware-specific behaviour (real PSP, real ARK chain, real CVM memory
encryption + RMP, SWIOTLB DMA cost, consumer-GPU passthrough) is a T3
concern — see `deploy/sev-snp/hetzner/` (M5.9).

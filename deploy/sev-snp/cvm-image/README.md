# gelo-snp-runner CVM image

This directory builds the **thin** CVM-bootable image that runs
`gelo-snp-runner`. It is the *same* image at simulation tiers T2 and T3
— only the host environment and `SNP_MODE` env var differ.

## Layout

- `setup-cvm-image.sh` — entrypoint. Builds the release binary,
  downloads the Ubuntu 24.04 base image, and injects the overlay +
  systemd units via `virt-customize`. Output: a ~50 MB qcow2 with no
  weights baked in.
- `overlay/` — files dropped on top of the Ubuntu base.
  - `etc/systemd/system/gelo-snp-runner.service` — main service unit.
  - `etc/systemd/system/gelo-fetch-weights.service` — one-shot first-boot
    weight fetch + SHA-256 verification.
  - `etc/gelo-snp/runner.toml` — runtime config.
  - `etc/gelo-snp/runner.env` — per-host environment (mode, model coord).
  - `etc/modprobe.d/vfio.conf` — documents consumer-GPU passthrough
    binding on the *host*.

## Why thin?

GELO targets openweight models. Baking ~1.2 GB of weight bytes into the
image would push it to >1.5 GB and force a rebuild on every model
revision. Instead, a one-shot `gelo-fetch-weights` systemd unit
downloads the weights at first boot from a configured source
(HuggingFace or an offline mirror) and validates SHA-256 against the
expected hash baked into `/etc/gelo-snp/runner.env`. Subsequent boots
hit the cache on `/var/lib/gelo-snp`.

Trade-off: first-boot needs network. For air-gapped deployment, mount
weights from a host directory and skip the fetch; see the Hetzner
runbook for that variant.

## Determinism

`setup-cvm-image.sh` should produce a bit-reproducible image given a
fixed Ubuntu base image and a fixed runner binary. CI gates on the
image SHA-256 to catch silent non-determinism.

## Mode selection

| Tier | Host | `SNP_MODE` | `/dev/sev-guest` |
|------|------|-----------|------------------|
| T2 — sim | regular QEMU/KVM (no SEV-SNP) | `mock` | absent (binary uses bundled mock PKI) |
| T3 — real | EPYC + SEV-SNP CVM | `production` | present (binary opens it) |

Wrong-mode launches fail closed at process start:
- `SNP_MODE=production` without `/dev/sev-guest` → exit 1 with
  `RuntimeModeError::MissingSevGuestDevice`.
- `SNP_MODE=mock` in a production build without the `mock` feature →
  exit 1 with `IssuerHandle::for_mode` error.

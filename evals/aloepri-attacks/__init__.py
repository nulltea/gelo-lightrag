"""AloePri attack-resistance harness (Phase 2).

The harness consumes PCIe-side snapshots captured by the
`aloepri-attack-snapshot-runner` Rust crate (`bin/capture_snapshots`),
runs the AloePri attack suite against them under the three control
conditions C0/C1/C2, and emits a results JSON per condition.

See `README.md` for the operator runbook and
`docs/prototype/aloepri-attack-harness.md` for the full Phase-2 spec.
"""

"""Per-attack drivers for the AloePri attack-resistance harness.

Each `run_<attack>.py` exposes a `run(snapshots: SnapshotSet, ...) -> dict`
entry point that computes the attack-specific TTRSR and returns a
result JSON ready to be merged into `results/path-1-attacks.json`.

The drivers import the AloePri attack primitives from
`vendor/aloepri-py/src/security_qwen/`; the vendored repo is added to
`sys.path` via `attack_drivers.common.install_aloepri_path()`.
"""

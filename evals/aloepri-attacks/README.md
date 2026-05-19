# AloePri attack-resistance harness

Phase 2 of the AloePri attack-resistance integration. Reads PCIe-side
snapshots captured from the GELO `InProcessTrustedExecutor` and runs the
AloePri attack suite (VMA / IMA / ISA / TFMA / SDA / IA) against them
under three control conditions:

| Slug             | Executor                                                                 | Shield  | Expected TTRSR (IMA + ISA) |
|------------------|--------------------------------------------------------------------------|---------|----------------------------|
| `c0_plain`       | `CapturingPlaintextExecutor` (this crate) wrapping `PlaintextExecutor`   | n/a     | ≥ 95% (control)            |
| `c1_mask_only`   | `InProcessTrustedExecutor::with_per_offload_mask()` + `ShieldConfig::NONE` | off   | < C0, > C2                 |
| `c2_default`     | `InProcessTrustedExecutor::with_seed(...)` (defaults: per-forward mask + `ShieldConfig::new(8, 4.0)`) | k=8 @ 4.0 | < 10%               |

Full spec: `docs/prototype/aloepri-attack-harness.md`. Rationale and
the 3-phase plan: `docs/research/aloepri-vs-gelo.md` §4.

The Phase 1 GELO-side snapshot-capture API (`crates/gelo-protocol/src/snapshot.rs`,
`InProcessTrustedExecutor::with_snapshot_capture`) is frozen for this
work — see `docs/plans/handoff-aloepri-attack-resistance.md` §"What's done".

## Layout

```
evals/aloepri-attacks/
├── Cargo.toml                            # snapshot-runner Rust crate
├── src/
│   ├── lib.rs                            # safetensors exporter + CapturingPlaintextExecutor
│   └── bin/capture_snapshots.rs          # Qwen3-1.7B → C0/C1/C2 snapshot writer
├── pyproject.toml                        # Python harness dep set
├── snapshots_loader.py                   # safetensors + meta.json reader
├── attack_drivers/
│   ├── common.py                         # AttackResult / TTRSR / sys.path helper
│   ├── run_vma.py, run_ima.py, run_isa.py    # load-bearing attacks
│   └── run_tfma.py, run_sda.py, run_ia.py    # template/not-applicable rows
├── run_all.py                            # 3-condition × 6-attack driver
├── conftest.py                           # pytest fixtures
└── tests/                                # smoke pytest suite
```

## Operator runbook

### 0. Prerequisites

* Rust workspace builds (`cargo check` from repo root).
* Python ≥ 3.10. `pip install -e .[test]` from `evals/aloepri-attacks/`.
* HuggingFace credentials cached if Qwen3-1.7B isn't already in
  `~/.cache/huggingface/hub/` — the snapshot runner downloads
  ~3.4 GB on first use.

### 1. Capture snapshots (Rust)

```
cargo run --release -p aloepri-attack-snapshot-runner --bin capture_snapshots -- \
    --condition all \
    --max-prompts 64 \
    --output snapshots/qwen3-1.7b
```

That produces three `(condition).safetensors` + `(condition).meta.json`
pairs under `snapshots/qwen3-1.7b/`. For a faster dev loop pass
`--max-prompts 8` (smoke corpus) and `--max-tokens 0` (prefill only).

| Flag | Default | Notes |
|---|---|---|
| `--condition` | `all` | `c0` / `c1` / `c2` / `all` |
| `--prompts` | (built-in smoke list) | one prompt per line file |
| `--max-prompts` | 64 | §2.5 cap is 256 for the gate run |
| `--max-tokens` | 0 | 0 = prefill only; >0 adds decode steps |
| `--max-prompt-tokens` | 32 | per-prompt truncation cap |
| `--seed-byte` | 29 | bit-stable mask seed for InProcess executors |

### 2. Run all attacks (Python)

```
python run_all.py \
    --snapshot-root snapshots/qwen3-1.7b \
    --output results/path-1-attacks.json
```

Output JSON shape:

```
{
  "format": "aloepri_attack_results_v1",
  "conditions": {
    "c0_plain":      { "vma": {...}, "ima": {...}, ... },
    "c1_mask_only":  { ... },
    "c2_default":    { ... }
  },
  "acceptance_gate": {
    "ima_c2_below_10pct": true|false,
    "isa_c2_below_10pct": true|false,
    "c0_ima_at_least_95pct": true|false,
    ...
  }
}
```

The Phase 3 CI wrapper (`.github/workflows/aloepri-gate.yml`, pending)
will read this file and exit non-zero if either of `ima_c2_below_10pct`
or `isa_c2_below_10pct` is false.

### 3. Run a single attack

```
python -m attack_drivers.run_ima \
    --snapshot-basename c2_default \
    --snapshot-root snapshots/qwen3-1.7b \
    --output results/c2_default.ima.json
```

Same flags for `run_isa.py`, `run_vma.py`, `run_tfma.py`, `run_sda.py`,
`run_ia.py`.

## Smoke pytest

Doesn't need Rust output or HF downloads — uses an in-memory
synthetic fixture:

```
cd evals/aloepri-attacks
pytest -ra
```

## Why VMA / TFMA / SDA / IA report `not_applicable`

VMA, TFMA, SDA, and IA in AloePri's original framing assume a *static*
token-permutation `τ` baked into shipped weights, so the attacker
sees an obfuscated token-id sequence on the wire. GELO doesn't ship
permuted weights and doesn't put token ids on the wire — the
embedding lookup is TEE-internal, and the PCIe attacker observes
masked activations only. We still emit a row per attack to keep the
6 × 3 result table square; the row carries `ttrsr_top1: null,
risk_level: "not_applicable"` and a one-line rationale in `extra.note`.

VMA gets a "zero-shot cosine match against the embedding table"
implementation so the row carries a real number on C0 (the activation
input *is* the embedding row, so cosine match is trivial) — see
`run_vma.py` for the deviation from AloePri's static-weight VMA.

IMA / ISA carry the load-bearing release-gate numbers. Both reuse
AloePri's ridge primitives unchanged (`_fit_ridge_regressor`,
`_predict_ridge`, `_evaluate_inversion_predictions` from
`vendor/aloepri-py/src/security_qwen/ima.py`); only the snapshot →
training-pair adapter differs.

## Frozen API contract

Per the Phase 2 handoff:

* Do **not** modify `crates/gelo-protocol/src/*`, `crates/gelo-embedder/src/*`,
  or any other `gelo-*` crate source.
* If an attack needs a tensor type we don't currently capture (e.g.
  KV-cache contents, attention scores) — file a Phase 2 follow-up
  rather than reaching back into the protocol crate.
* The Rust crate here adds `CapturingPlaintextExecutor` as a
  `TrustedExecutor` wrapper around `PlaintextExecutor` so the C0
  control path doesn't need a GELO-side change.

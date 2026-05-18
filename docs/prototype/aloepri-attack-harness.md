# AloePri attack-resistance harness — Phase 2/3 handoff

> **Sibling docs:**
> [`../research/aloepri-vs-gelo.md`](../research/aloepri-vs-gelo.md) §4
> "The One Win — Empirical Attack Suite" (rationale + 3-phase plan).
>
> **Status:** Phase 1 (Rust-side snapshot capture) ✅ done 2026-05-18.
> Phase 2 (Python attack harness) and Phase 3 (CI release-gate)
> pending.

## Definitions

- **AloePri**: ByteDance's private LLM inference paper (arXiv
  2603.01499, March 2026) plus accompanying open-source repo
  `github.com/sheng1feng/Aloepri`. The repo ships a model-agnostic
  attack suite in `src/security_qwen/` that this harness reuses.
- **GELO**: this project's TEE+GPU split-inference protocol (per-batch
  Haar mask + TwinShield + U-Verify), defined in
  `docs/prototype/gelo.md`.
- **PCIe-side attacker**: an adversary co-located with the GPU, on the
  PCIe bus, or possessing a sub-protocol TEE breach. Sees the masked
  operands and engine outputs but never the unmasked TEE-internal
  state. Phase 1 snapshot capture targets exactly this threat surface.
- **TTRSR**: token-level reconstruction success rate — fraction of
  prompt tokens an attack recovers. AloePri reports 5-15% under their
  attacks; GELO's release-gate threshold (proposed) is < 10% for
  ISA/IMA at the default mask + shield config.
- **VMA/IMA/ISA/TFMA/SDA/IA**: AloePri's attack taxonomy — see
  §1 below.

## 0. What Phase 1 left in place

| Component | Path | Purpose |
|---|---|---|
| `PcieSnapshot` struct | `crates/gelo-protocol/src/snapshot.rs` | One record per PCIe-crossing matmul: `(seq_idx, layer, kind, masked_operand, masked_output)` |
| `SnapshotCapture` aggregator | same file | Configurable buffer (`capture_outputs`, `max_snapshots`); supports `drain`, `reset`, `snapshots`, `dropped` |
| `InProcessTrustedExecutor::with_snapshot_capture(cfg)` | `crates/gelo-protocol/src/sim.rs` | Builder method to opt in |
| `InProcessTrustedExecutor::{enable,disable}_snapshot_capture` | same | In-place toggles for test setups |
| `InProcessTrustedExecutor::{pcie_snapshots, drain_pcie_snapshots}` | same | Read-only inspect + ownership-transfer drain |
| Hooks in `offload_linear`, `offload_qkv`, `offload_linear_many` | same | Records the post-mask operand and engine output to the active buffer (no-op when capture is disabled) |
| 11 tests | `crates/gelo-protocol/tests/snapshot_capture.rs` + `src/snapshot.rs` | Default-off invariant, opt-in capture, seq-idx ordering, drain semantics, multi-output batching |

Default behaviour is **capture disabled** so the production embedder /
reranker code path has zero overhead and zero allocations. Capture is
strictly opt-in and only engaged inside the attack-harness test
binaries.

## 1. AloePri attack taxonomy (what we are evaluating against)

From `path-2-aloepri/vendor/aloepri-py/src/security_qwen/matrix.py` and
`docs/research/aloepri-vs-gelo.md` §4:

| Attack | File | What it tries to recover | Inputs it needs |
|---|---|---|---|
| **VMA** Vocabulary-Matching | `vma.py` (30 KB) | Per-token vocab id from obfuscated embedding row | embedding-table-shape activations |
| **IMA** Inversion Model Attack | `ima.py` (34 KB) | Token ids from hidden states (trained inverter) | (token_ids, hidden_state) pairs across many prompts; targets layer 0, mid, last |
| **ISA** Internal State Attack | `isa.py` (20 KB) | Prompt from layer-i hidden states (zero-shot inversion) | hidden state, attention_score, layer_output, kv_cache |
| **TFMA** Token Frequency Matching | `tfma.py` (14 KB) | Per-position token from corpus frequency leakage | sequence-length traces |
| **SDA** Substitution Deciphering | `sda.py` (11 KB) | n-gram-based token-substitution recovery | sequence-length traces |
| **IA** Invariant Attack | `ia.py` (1 KB) | Token id from weight-invariants under obfuscation | static-weight observation only |

Of these, **IMA** and **ISA** are the load-bearing attacks for GELO's
threat-model claim (per-batch Haar mask makes hidden-state inversion
infeasible). VMA / IA are less load-bearing for GELO because we don't
obfuscate static weights at all — they're public under the openweight
threat model. TFMA / SDA target decode-time frequency leakage,
relevant to the long-form-generation surface but not the embedding /
reranking paths.

## 2. Phase 2 — Python attack harness

### 2.1 Directory layout

```
evals/aloepri-attacks/
├── README.md                  — operator runbook
├── pyproject.toml             — pinned to AloePri commit + transformers + torch
├── conftest.py                — pytest fixtures (snapshot loader, three-condition matrix)
├── snapshots_loader.py        — read safetensors snapshots → AloePri-shaped tensors
├── run_vma.py / run_ima.py / run_isa.py / run_tfma.py / run_sda.py / run_ia.py
│                              — one driver per attack, pulls AloePri's reference impl
├── run_all.py                 — single-shot runner producing one row per attack
├── results/                   — JSON outputs keyed by (model, config, attack)
└── tests/test_smoke_*.py      — pytest wrappers for CI consumption
```

### 2.2 Snapshot serialisation contract

The Rust side does not currently write snapshots to disk. Phase 2's
first task is a `gelo-embedder` test utility (NOT a feature on
`gelo-protocol` — keep that crate I/O-free) that takes the
`Vec<PcieSnapshot>` returned by `drain_pcie_snapshots` and writes a
single `.safetensors` file plus a sidecar `.json` metadata file:

**safetensors keys (one per snapshot, per tensor):**

```
snap{seq_idx:05}.{layer:03}.{kind}.operand        — Array2<f32>
snap{seq_idx:05}.{layer:03}.{kind}.output         — Array2<f32>  (optional)
```

Example: `snap00042.027.q_proj.operand`. Use lowercase `kind` names
matching AloePri's tensor conventions (`q_proj`, `k_proj`, `v_proj`,
`o_proj`, `gate_proj`, `up_proj`, `down_proj`).

**Sidecar JSON `<basename>.meta.json`:**

```json
{
  "schema_version": "1",
  "model_id": "Qwen/Qwen3-1.7B",
  "config": {
    "shield_k": 8,
    "shield_energy_scale": 4.0,
    "per_forward_mask": true,
    "verify_probes": 0,
    "prompt_token_ids": [...]
  },
  "snapshots": [
    {
      "seq_idx": 0,
      "layer": 0,
      "kind": "q_proj",
      "operand_shape": [12, 2048],
      "output_shape": [12, 2048],
      "n_data": 4,
      "shield_k": 8
    },
    ...
  ]
}
```

The `n_data` field lets the Python harness strip the shield rows
(`shield_k` last rows of every operand) before running attacks —
attackers don't see strip vs data rows separately in the wild (both
travel across PCIe together), but reproducing AloePri's attacks
against the data-only slice gives a more direct comparison to their
published numbers.

The serialisation utility should live at
`crates/gelo-embedder/src/attack_export.rs` (or `tests/common/`) so
the dependency on `safetensors` writes stays in the embedder crate,
not the protocol crate.

### 2.3 AloePri commit pin

Vendored at `/home/timo/repos/private-rag-path-2/vendor/aloepri-py/`
(commit hash documented in `path-2-aloepri-gemma.md` M2.1). The
attacks live at:

```
vendor/aloepri-py/src/security_qwen/
  ├── vma.py
  ├── ima.py
  ├── isa.py
  ├── tfma.py
  ├── sda.py
  ├── ia.py
  ├── artifacts.py / metrics.py / schema.py / summary.py  (result aggregation)
  └── matrix.py                                            (attack/target matrix definition)
```

Phase 2's `run_*.py` files should `import sys; sys.path.insert(0,
"../../private-rag-path-2/vendor/aloepri-py")` (or use a pyproject
`[tool.uv.sources]` git-pin for reproducibility) and call the
attacks' public entry points with shapes adapted via
`snapshots_loader.py`.

**Shape adaptation:** AloePri's attacks expect tensors keyed by
HuggingFace transformers conventions (layer index `0..N-1`, hidden
size matching the model). Our snapshots have shape `(n_data +
shield_k, d_hidden)`; pre-stripping the shield rows gives `(n_data,
d_hidden)` which matches what AloePri sees in their pipeline.

### 2.4 Three-condition control matrix

Per the §4.1 plan, every attack runs against three configurations of
the same model + prompts:

| Condition | Executor | Shield | Expected TTRSR |
|---|---|---|---|
| **C0 plain** | `PlaintextExecutor` | n/a | ~100% (control: attacks must succeed when there's no obfuscation) |
| **C1 mask-only** | `InProcessTrustedExecutor::with_per_offload_mask` | `ShieldConfig::disabled()` | "most" — Gram leak unprotected; baseline for "what does the mask alone buy?" |
| **C2 default** | `InProcessTrustedExecutor::new` (defaults: per-forward mask + shield k=8 @ 4.0) | k=8 @ 4.0 | < 5% target; release-gate threshold < 10% |

Each condition exports its own snapshot file. The Python driver
imports all three, runs all six attacks, and writes a results table.

### 2.5 Prompt corpus

For the AloePri-vs-GELO numbers to be comparable, use the same
prompt corpus AloePri uses for their security evaluations.
`vendor/aloepri-py/src/defaults.py::DEFAULT_PROMPTS` is the canonical
list (a small sample of natural-language sentences). Cap at 256
prompts per condition to keep the harness under 30 minutes.

### 2.6 Acceptance gate (Phase 2 done when)

1. `run_all.py --condition c2 --model qwen3-1.7b` produces a results
   JSON with all six attacks reporting TTRSR.
2. C0 plain reports TTRSR ≥ 95% on at least IMA, ISA, VMA (sanity
   check: attacks themselves work).
3. C2 default reports TTRSR < 10% on IMA and ISA (the load-bearing
   attacks for GELO's threat-model claim).
4. C1 mask-only shows a gap between C0 and C2 (proves shield rows
   add measurable defence on top of the mask alone).
5. Results JSON committed to `results/path-1-attacks.json` per
   plan §M1.9 acceptance criteria.

## 3. Phase 3 — CI release-gate integration

Once Phase 2 lands a passing C2 result:

1. Add a fast variant of `run_all.py` capped at ~64 prompts (under 5
   minutes) and wire it into CI via a release-gate workflow at
   `.github/workflows/aloepri-gate.yml`.
2. Threshold: **fail the gate if IMA or ISA C2 TTRSR ≥ 10%**.
3. Tag the gate `aloepri-attacks` and require it for any PR that
   modifies `crates/gelo-protocol/src/{mask,shield,sim,snapshot}.rs`
   or `crates/gelo-embedder/src/decoder/`.
4. Result archive: each gated PR writes its `results/<sha>.json` to
   an S3 bucket (or GitHub Releases) so we have a longitudinal record
   of TTRSR evolution as the protocol changes.

## 4. Open questions for Phase 2

These are deliberately *not* answered in Phase 1 because they need
the Python harness to be live before they can be measured:

1. **Per-forward vs per-offload mask under attack.** Spec says
   per-forward + shield is paper-parity; per-offload is "strictly
   safer" but ~140× slower. Phase 2 should run C2 (per-forward +
   shield) and a fourth condition C3 (per-offload, shield off) so
   we have empirical evidence on which is actually more attack-
   resistant on which attack — the paper's argument is qualitative
   only.
2. **Snapshot stripping policy.** Should the Python harness see the
   shield rows or only the data rows? Argument for "data only":
   matches AloePri's pipeline shape, gives directly comparable
   numbers. Argument for "include shield": that's literally what the
   PCIe attacker sees. Recommendation: run both, report both.
3. **Long-context regime.** Snapshots at decode shape (n=1) look
   very different from prefill (n=16+). IMA and ISA are trained on
   prefill; their behaviour at decode-step snapshots is an open
   question. Out of scope for Phase 2 minimum acceptance; surface as
   a Phase 3 follow-up.

## 5. Touch points for Phase 2 worker

When the Phase 2 worker (or a future session) picks this up:

1. Read `docs/research/aloepri-vs-gelo.md` §4 end to end first.
2. Verify the AloePri vendored commit is still at the pin documented
   in `path-2-aloepri-gemma.md` M2.1.
3. Land the snapshot serialiser in `crates/gelo-embedder/` (§2.2
   above) before touching any Python.
4. The Phase 1 snapshot-capture API is **frozen** as far as Phase 2
   is concerned. If a missing capability surfaces (e.g. need to
   capture KV cache contents for ISA's `kv_cache` observable type),
   file a Phase 2 follow-up rather than reaching back into
   `gelo-protocol` — that crate's API surface is hot path-adjacent.

## 6. References

- [`../research/aloepri-vs-gelo.md`](../research/aloepri-vs-gelo.md)
  §4 — the source-of-truth rationale + 3-phase plan
- [`../plans/path-1-gelo-gemma.md`](../plans/path-1-gelo-gemma.md)
  §M1.9 — Path 1 milestone this work delivers
- [`../plans/path-2-aloepri-gemma.md`](../plans/path-2-aloepri-gemma.md)
  M2.7 — Path 2's analogous attack-resistance milestone (different
  obfuscation scheme, same attack suite)
- AloePri reference code:
  `~/repos/private-rag-path-2/vendor/aloepri-py/src/security_qwen/`
- AloePri paper: arXiv 2603.01499 — *Towards Privacy-Preserving LLM
  Inference via Collaborative Obfuscation*

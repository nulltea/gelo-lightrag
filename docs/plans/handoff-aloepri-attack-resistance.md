# Handoff — AloePri attack-resistance integration (Phase 2 + Phase 3)

## What this is

AloePri (arXiv 2603.01499, ByteDance, March 2026; reference repo
`github.com/sheng1feng/Aloepri`) is the closest published baseline to
this project's GELO TEE+GPU split-inference protocol. AloePri ships
a model-agnostic empirical attack suite (`src/security_qwen/`) — six
attacks against transformer-class private inference: **VMA, IMA, ISA,
TFMA, SDA, IA**. The "AloePri attack-resistance integration" plan
*does not* port AloePri's obfuscation protocol; it ports only the
**attack suite**, runs it against GELO's PCIe-side observable surface,
and uses the resulting TTRSR (token reconstruction success rate)
numbers as a release gate.

The full rationale, attack taxonomy, and 3-phase plan are in
**`docs/research/aloepri-vs-gelo.md` §4 "The One Win — Empirical
Attack Suite"** (read this first). Threat-model alignment, the
load-bearing attacks for our claim (IMA + ISA), and the
anti-recommendations (do *not* port AloePri's static-weight
obfuscation) are all there.

## What's done — Phase 1 (Rust-side snapshot capture)

Landed this session, 2026-05-18. Builds clean, 11 new tests pass,
zero regressions across `gelo-protocol` / `gelo-embedder` /
`gelo-reranker`.

| Component | Path |
|---|---|
| `PcieSnapshot`, `SnapshotCapture`, `SnapshotConfig` | `crates/gelo-protocol/src/snapshot.rs` |
| `InProcessTrustedExecutor::{with,enable,disable}_snapshot_capture`, `pcie_snapshots`, `pcie_snapshot_capture`, `drain_pcie_snapshots` | `crates/gelo-protocol/src/sim.rs` |
| Capture hooks (no-op when disabled) | `offload_linear`, `offload_qkv`, `offload_linear_many` in same file |
| Unit tests (5) + integration tests (6) | `crates/gelo-protocol/src/snapshot.rs` + `tests/snapshot_capture.rs` |

Capture sits **between mask-apply and engine-matmul** — exactly the
PCIe-side adversary's view. Unmasked outputs never leave the
executor. Default is off — production embedder / reranker paths pay
zero overhead.

Plan tracking: `docs/plans/path-1-gelo-gemma.md` §M1.9 has been
restructured into the 3-phase shape; Phase 1 is checked off there
and in `docs/research/aloepri-vs-gelo.md` §4.1.

## What's left — Phase 2 + Phase 3

**The detailed handoff is `docs/prototype/aloepri-attack-harness.md`**
— read this end-to-end before doing anything. It contains:

- §0 Phase 1 deliverables (already done; don't re-touch
  `gelo-protocol` API — it's frozen for Phase 2)
- §1 Attack taxonomy table (VMA/IMA/ISA/TFMA/SDA/IA — files, sizes,
  what each tries to recover, which observables it needs)
- §2 Phase 2 detail:
  - §2.1 `evals/aloepri-attacks/` directory layout
  - §2.2 safetensors serialisation contract (key format
    `snap{seq_idx:05}.{layer:03}.{kind}.{operand|output}` +
    sidecar `<basename>.meta.json`) — **the serialiser belongs in
    `crates/gelo-embedder/src/attack_export.rs`**, NOT in
    `gelo-protocol` (keep that crate I/O-free)
  - §2.3 AloePri commit pin location:
    `~/repos/private-rag-path-2/vendor/aloepri-py/`; attack imports
    documented
  - §2.4 three-condition control matrix (C0 plain / C1 mask-only /
    C2 mask+shield; expected C2 TTRSR < 10% on IMA + ISA)
  - §2.5 prompt corpus (`vendor/aloepri-py/src/defaults.py::DEFAULT_PROMPTS`,
    cap 256 prompts per condition)
  - §2.6 acceptance gate (4 numbered criteria)
- §3 Phase 3 detail (CI release-gate wiring + 5-minute fast variant,
  fail on IMA/ISA TTRSR ≥ 10%, gate tag `aloepri-attacks`)
- §4 Three open questions Phase 2 should answer empirically
  (per-forward vs per-offload under attack; shield-stripping
  policy; decode-shape regime)
- §5 Touch points for the worker (frozen API contract, vendor
  commit pin, do-not-extend-protocol-crate rule)

## Suggested starting moves for the Phase 2 worker

1. **Verify Phase 1 still builds + passes**:
   `cargo test -p gelo-protocol --test snapshot_capture`
2. **Read** `docs/research/aloepri-vs-gelo.md` §4 then
   `docs/prototype/aloepri-attack-harness.md` end to end.
3. **First Rust deliverable**: `crates/gelo-embedder/src/attack_export.rs`
   implementing the §2.2 serialisation contract. Use the existing
   `safetensors` 0.4.5 dep (already pulled in by the workspace,
   `serialize_to_file` is the API). Add a unit test that round-trips
   a small `Vec<PcieSnapshot>` through serialise → re-parse →
   shape-and-name assertion.
4. **First Python deliverable**: `evals/aloepri-attacks/snapshots_loader.py`
   that reads the safetensors + meta.json into AloePri-compatible
   tensor shapes (data-only rows by default, full operands behind
   a flag — per §4 question 2).
5. **Drive Phase 2 acceptance on Qwen3-1.7B**, not Gemma 4 — the
   v1 demonstrator landed in this session, Gemma 4 real weights
   are blocked on Phase 1.5 (see
   `docs/prototype/gemma4-architecture-roadmap.md`).
6. **Three-condition runner first** (C0/C1/C2), then per-attack
   drivers — establishes the result-table shape early.

## Skills the next session may want

- **`diagnose`** if Phase 2 C2 IMA/ISA TTRSR comes in above 10% —
  it's a disciplined reproduce → minimise → hypothesise loop
  appropriate for an "unexpected privacy regression" finding. The
  research doc §4.1 calls this out as the risk path.
- **`grill-me`** if you want to stress-test the §4 design decisions
  (snapshot stripping policy, per-forward vs per-offload, prompt
  corpus choice) before committing to a Phase 2 implementation.

(`improve-codebase-architecture` is **not** the right skill here —
the API contract is deliberately frozen at the Phase 1 boundary.
`update-config` / `simplify` / `init` are unrelated.)

## Pointers in case the worker hits a wall

| Issue | Where to look |
|---|---|
| AloePri attack expects a tensor shape we don't capture | `docs/prototype/aloepri-attack-harness.md` §5 "frozen API" rule — file a Phase 2 follow-up, don't reach back into `gelo-protocol` |
| TTRSR ≥ 10% on C2 (defaults) | `docs/prototype/gelo.md` §3.3 (shield rows) and §3.2 (per-forward vs per-offload mask); consider running C3 = per-offload mask, shield off to isolate which protocol piece is doing the work |
| AloePri vendored repo missing / wrong commit | Path-2 plan `docs/plans/path-2-aloepri-gemma.md` M2.1 documents the pin |
| Python harness needs intermediate state we didn't capture (e.g. KV cache) | Out of scope for Phase 1; surface as Phase 2 follow-up per §5 |
| Snapshot serialiser doesn't fit in `gelo-embedder` | OK to put it in a new crate `gelo-attack-export` — just keep it out of `gelo-protocol` |

## Out of scope for this handoff

The conversation that produced Phase 1 also landed the **Qwen3-1.7B
v1 demonstrator pivot** (commit pending) — `Qwen3Variant`, QK-norm
wiring, real-weight greedy generation under masked executor. That
work is referenced because Phase 2 runs against Qwen3-1.7B, but it's
already-landed prior work, not part of the AloePri integration. See
`docs/plans/path-1-gelo-gemma.md` §10 if you need context on it.

## File-level diff for the Phase 1 work

`git status` at handoff time shows these uncommitted Phase 1 files
on top of the Qwen3 pivot diff:

- `crates/gelo-protocol/src/snapshot.rs` (new, ~250 lines)
- `crates/gelo-protocol/src/lib.rs` (mod export)
- `crates/gelo-protocol/src/sim.rs` (field, builders, 3 hooks)
- `crates/gelo-protocol/tests/snapshot_capture.rs` (new)
- `docs/prototype/aloepri-attack-harness.md` (new — the Phase 2/3
  handoff)
- `docs/research/aloepri-vs-gelo.md` (§4.1 status updates)
- `docs/plans/path-1-gelo-gemma.md` (§M1.9 restructure)

Both Phase 1 *and* the Qwen3 pivot work are uncommitted as of
handoff. The user asked to commit selectively when ready —
recommend two separate commits to keep the Qwen3 pivot and the
AloePri Phase 1 work auditable independently.

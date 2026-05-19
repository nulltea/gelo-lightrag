"""VMA — Vocabulary-Matching Attack driver.

Paper §F.1 + Table 8: AloePri's VMA recovers the secret token
permutation Π from `(W_plain, W_obfuscated)` weight-matrix
pairs. For each pair (e.g. `W_embed · W_gate` vs
`Π·W*_embed·W_gate·Ẑ_ffn`), the attack applies RowSort and
neighbor-matching to recover Π.

**This attack is structurally not applicable to GELO under the
openweight threat model.** GELO does not obfuscate model
weights — they are public by construction (per
`docs/prototype/gelo.md` §2). There is no `W_obfuscated` for
the attack to compare against `W_plain`, and there is no Π to
recover. The closest meaningful threat is the **NN
(Nearest Neighbor)** attack on activations — implemented in
:mod:`run_nn`, with its own column in AloePri Table 1.

We emit a `not_applicable` row to keep the per-condition table
square and to document the threat-model mismatch. Same
template-only decision AloePri's own `ia.py` made (the paper
specifies Gate-IA + Attn-IA in detail but ships a 30-line stub).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import AttackResult


def run(snapshots, **_kwargs) -> AttackResult:
    return AttackResult(
        attack="vma",
        condition=snapshots.condition,
        model_id=snapshots.model_id,
        n_prompts=snapshots.n_prompts(),
        n_train=0,
        n_test=0,
        ttrsr_top1=None,
        ttrsr_top10=None,
        risk_level="not_applicable",
        extra={
            "note": (
                "AloePri's VMA recovers Π from (W_plain, W_obfuscated) weight "
                "pairs. GELO does not obfuscate weights (openweight threat "
                "model). The closest meaningful attack on GELO observables is "
                "NN — see attack_drivers.run_nn (AloePri Table 1 reports "
                "AloePri NN = 0.0% and VMA = 13.51%; the GELO analog is NN, "
                "not VMA)."
            ),
            "phase": "phase0_template_only",
        },
    )


def _cli() -> None:
    import sys

    p = argparse.ArgumentParser(description="Run VMA against a snapshot set")
    p.add_argument("--snapshot-basename", required=True)
    p.add_argument("--snapshot-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from snapshots_loader import SnapshotSet  # type: ignore

    snaps = SnapshotSet.open(args.snapshot_basename, root=args.snapshot_root)
    result = run(snaps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2))
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    _cli()

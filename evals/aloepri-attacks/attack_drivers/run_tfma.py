"""TFMA — Token-Frequency Matching Attack driver.

AloePri's TFMA targets the **token-permutation leak** in static-key
obfuscation: an adversary collecting many obfuscated requests learns
the marginal frequency of each `τ(t)` and matches it against the
unconditional token-frequency prior of a public corpus.

This attack is structurally *not applicable* to GELO under our
threat model — token IDs never cross PCIe. The TEE owns the
embedding lookup; the GPU only sees the masked activation
trajectory, which carries no token-id information.

We still emit a result row so the §2.6 acceptance matrix is square,
but the row's TTRSR is `None` and the status is `not_applicable`
with a short rationale. C0 plain (where token-frequency *would* show
up as embedding-row frequency in the activation traffic) is reported
the same way; running TFMA on the activation rows directly is the
job of VMA, not TFMA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import AttackResult


def run(snapshots, **_kwargs) -> AttackResult:
    return AttackResult(
        attack="tfma",
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
                "Token-frequency leak does not apply to GELO under the openweight "
                "threat model — token ids are TEE-internal and never cross PCIe. "
                "See docs/research/aloepri-vs-gelo.md §3 table row #1."
            ),
            "phase": "phase0_template_only",
        },
    )


def _cli() -> None:
    import sys

    p = argparse.ArgumentParser(description="Run TFMA against a snapshot set")
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

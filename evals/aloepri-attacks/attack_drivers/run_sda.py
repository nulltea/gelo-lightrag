"""SDA — Substitution Deciphering Attack driver.

AloePri's SDA is a bigram-based decipherment of an obfuscated
*token-id sequence*. It composes with TFMA — recovers token ids
that survive frequency matching by exploiting n-gram regularities.

Like TFMA, SDA does not apply to GELO under the openweight threat
model: there is no token-id sequence crossing PCIe to decipher.
The masked activation traffic has no symbolic substructure for an
n-gram model to fit.

We emit a `not_applicable` row to keep the per-condition table
square — see :mod:`run_tfma` for the matching rationale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import AttackResult


def run(snapshots, **_kwargs) -> AttackResult:
    return AttackResult(
        attack="sda",
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
                "Substitution decipherment operates on an obfuscated token-id "
                "sequence — GELO's threat model has no such sequence visible to "
                "the PCIe attacker (token ids never cross the trust boundary). "
                "See docs/research/aloepri-vs-gelo.md §3 table row #1."
            ),
            "phase": "phase0_template_only",
        },
    )


def _cli() -> None:
    import sys

    p = argparse.ArgumentParser(description="Run SDA against a snapshot set")
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

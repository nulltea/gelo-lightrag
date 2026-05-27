"""ISA-AttnScore — ISA against attention-score observables.

Ported from `m2_7/run_hidden_state_attacks.py::_isa_attn_score`. The
M2.7 version is run against the obfuscated llama-server's
`attn.safetensors` dump (per-prompt `kq-{layer}` tensors of shape
``(n_heads, n_q, n_kv)``).

In our (path-1 / round-3 B.3) threat model, GELO keeps attention
compute in-TEE per the M1.3 cached-attention design lock, so the
PCIe-side attacker NEVER sees attention scores. This driver therefore
emits a ``not_applicable`` row by default — it's a placeholder for
the M1.10 fused-permuted-attention path, where attention compute
DOES move to the GPU and per-token attention activations cross the
trust boundary under the permuted protocol.

When (and if) the protocol grows an attention-score snapshot kind
(``WeightKind::AttnScore`` or similar), the driver becomes runnable:

* Snapshot operand shape: ``(n_heads, n_q, n_kv)`` per (layer,
  prompt). The driver flattens to ``(n_q, n_heads · n_kv)`` so the
  same ridge primitive used by ``run_isa.run`` can attack the
  attention-score features.
* Token-id alignment: row ``r`` of the flattened tensor corresponds
  to position ``r`` in the prompt — same as HiddenState ISA.

Primary metric: same TTRSR-top1 as ``run_isa`` (the attacker wins
when they can recover the plaintext token id from the masked
attention activations of that token's row).

Threat-model alignment (when capture lands):

* C0 plain — attention scores leak token identity through value-
  projection structure; baseline recovery should approach 100 %.
* C1/C2/C3 — depending on the permuted-attention protocol's noise
  parameters, recovery should drop below 10 % (analog to the
  hidden-state ISA gate).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

from .common import AttackResult


def _is_runnable(snapshots) -> bool:
    """Return True iff the snapshot set contains an attention-score
    kind we can attack (e.g. ``kq``). False otherwise — emits the
    ``not_applicable`` row.
    """
    return any(
        getattr(s, "kind", "") in {"kq", "attn_score"}
        for s in getattr(snapshots, "snapshots", [])
    )


def run(
    snapshots,
    *,
    embed_table: torch.Tensor | None = None,
    layer: int = 23,
    kind: str = "kq",
    seed: int = 20260518,
    split_mode: str = "vocab",
    strip_shield: bool = True,
    **_kwargs,
) -> AttackResult:
    """ISA-AttnScore entry point.

    If the snapshot set has no attention-score-kind tensors, returns
    a ``not_applicable`` row — current GELO captures don't include
    them. Otherwise routes to ``run_isa`` with a flattened view.
    """
    if not _is_runnable(snapshots):
        return AttackResult(
            attack="isa_attn_score",
            condition=snapshots.condition,
            model_id=snapshots.model_id,
            n_prompts=snapshots.n_prompts(),
            n_train=0,
            n_test=0,
            ttrsr_top1=None,
            ttrsr_top10=None,
            risk_level="not_applicable",
            primary_metric_name="token_top1_recovery_rate",
            extra={
                "note": (
                    "Attention-score snapshots not captured by the GELO "
                    "protocol: attention compute stays in-TEE per the "
                    "M1.3 cached-attention design lock. This driver is a "
                    "placeholder for the M1.10 permuted-attention path "
                    "(when attention crosses PCIe under the permuted "
                    "protocol). Capture-side prerequisite: a "
                    "WeightKind::AttnScore variant in gelo-protocol that "
                    "emits per-head attention activations."
                ),
                "phase": "deferred_pending_m1_10",
                "expected_kinds": ["kq", "attn_score"],
                "available_kinds": sorted(set(s.kind for s in snapshots.snapshots)),
            },
        )

    # Runnable path: flatten attention scores per prompt + delegate
    # to run_isa.run. Mirrors the M2.7 adapter logic exactly.
    from . import run_isa as _run_isa

    flat_pairs: list[tuple[int, torch.Tensor]] = []
    for s in snapshots.select(kind=kind, layer=layer):
        op = snapshots.get_operand(s, strip_shield=strip_shield)
        if op.ndim == 3:
            n_heads, n_q, n_kv = op.shape
            op = op.permute(1, 0, 2).reshape(n_q, n_heads * n_kv)
        elif op.ndim != 2:
            continue
        flat_pairs.append((s.prompt_idx, op))

    if not flat_pairs:
        return AttackResult(
            attack="isa_attn_score",
            condition=snapshots.condition,
            model_id=snapshots.model_id,
            n_prompts=snapshots.n_prompts(),
            n_train=0,
            n_test=0,
            ttrsr_top1=None,
            ttrsr_top10=None,
            risk_level="unknown",
            extra={
                "note": (
                    f"no snapshots matched (layer={layer}, kind={kind!r})"
                ),
            },
        )

    shim = copy.copy(snapshots)
    shim.per_prompt_layer_kind_tensors = lambda **kw: flat_pairs  # type: ignore[method-assign]
    return _run_isa.run(
        shim,
        embed_table=embed_table,
        layer=layer,
        kind=kind,
        seed=seed,
        split_mode=split_mode,
        strip_shield=strip_shield,
    )


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--snapshot-root", required=True, type=Path)
    p.add_argument("--snapshot-basename", required=True)
    p.add_argument("--layer", type=int, default=23)
    p.add_argument("--kind", default="kq")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from snapshots_loader import SnapshotSet  # type: ignore  # noqa: E402

    snaps = SnapshotSet.open(args.snapshot_basename, root=args.snapshot_root)
    result = run(snaps, layer=args.layer, kind=args.kind)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2))
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    _cli()

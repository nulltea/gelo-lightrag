"""NN — Nearest Neighbor attack driver.

Paper §F.1 / Table 1: the attacker takes the (obfuscated) hidden
state, looks up each row's nearest token-embedding row by cosine
similarity, and reports Top-K recovery. No training required —
this is the cheapest training-based-inversion attack the paper
evaluates.

Why this driver exists separately from `run_vma.py`:

* AloePri's true VMA (§F.1, Table 8) recovers the **secret token
  permutation Π** from `(W_plain, W_obfuscated)` weight-matrix
  pairs via RowSort + neighbor-matching. **GELO does not
  obfuscate weights** under the openweight threat model, so the
  attack target Π doesn't exist for us — `run_vma.py` is a
  no-op stub mirroring AloePri's own `ia.py` template-only
  decision.
* The previous `run_vma.py` body was a cosine-NN match against
  the embedding table, which is exactly what the paper calls
  **NN**. We re-host that body here under the correct name. The
  paper's Table 1 reports `AloePri NN = 0.0%` and we now
  measure the same metric.

Observable: any (layer, op_kind) operand whose `in_features`
equals the embedding dim. Paper defaults to a deep layer for
ISA-style comparison; we keep layer 0 as the default here so the
NN column lines up with AloePri Table 1's "first-layer
observable" baseline, but the CLI exposes `--layer` for sweeps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .common import (
    AttackResult,
    classify_risk_level,
    stack_prompt_observations,
)
from .run_ima import load_qwen3_embedding_table


def _cosine_topk(obs: torch.Tensor, embed_table: torch.Tensor, topk: int) -> torch.Tensor:
    """Return `(n_obs, topk)` token-id matrix of nearest embedding
    rows by cosine similarity. Single matmul so it scales to
    vocab-size 150k+ at hidden-dim 2048.
    """
    obs_n = obs / obs.norm(dim=1, keepdim=True).clamp_min(1e-8)
    emb_n = embed_table / embed_table.norm(dim=1, keepdim=True).clamp_min(1e-8)
    scores = obs_n @ emb_n.T
    return torch.topk(scores, k=min(topk, scores.shape[1]), dim=1).indices


def run(
    snapshots,
    *,
    embed_table: torch.Tensor,
    layer: int = 0,
    kind: str = "q_proj",
    topk: int = 10,
    strip_shield: bool = True,
    eval_size: int | None = None,
    seed: int = 20260518,
) -> AttackResult:
    X, y, _ = stack_prompt_observations(
        snapshots, layer=layer, kind=kind, strip_shield=strip_shield
    )
    if X.shape[0] == 0:
        return AttackResult(
            attack="nn",
            condition=snapshots.condition,
            model_id=snapshots.model_id,
            n_prompts=snapshots.n_prompts(),
            n_train=0,
            n_test=0,
            ttrsr_top1=None,
            ttrsr_top10=None,
            risk_level="unknown",
            extra={"note": "no compatible snapshots", "layer": layer, "kind": kind},
        )
    # NN is pure cosine-NN with NO training, so a dim mismatch
    # (obs in obfuscated d_eff vs plain embed d) means the attack
    # can't construct a sensible match. Truncate obs to embed_dim
    # as a best-effort: an attacker without keymat keys would try
    # this naively. The number this produces is a CEILING on what
    # untrained NN can do against the obfuscated artifact.
    if X.shape[1] != embed_table.shape[1]:
        if X.shape[1] > embed_table.shape[1]:
            print(
                f"  [nn] obs_dim={X.shape[1]} > embed_dim={embed_table.shape[1]} — "
                f"truncating obs to first {embed_table.shape[1]} dims "
                f"(naive-attacker upper bound)"
            )
            X = X[:, : embed_table.shape[1]]
        else:
            print(
                f"  [nn] obs_dim={X.shape[1]} < embed_dim={embed_table.shape[1]} — "
                f"cosine-NN can't pad; skipping"
            )
            return AttackResult(
                attack="nn", condition=snapshots.condition,
                model_id=snapshots.model_id, n_prompts=snapshots.n_prompts(),
                n_train=0, n_test=0, ttrsr_top1=None, ttrsr_top10=None,
                risk_level="unknown",
                extra={"note": "obs_dim < embed_dim; NN inapplicable"},
            )

    rng = np.random.default_rng(seed)
    n_total = X.shape[0]
    n_eval = n_total if eval_size is None else min(eval_size, n_total)
    eval_idx = rng.choice(n_total, size=n_eval, replace=False)
    Xe = torch.from_numpy(X[eval_idx]).to(torch.float32)
    yids = torch.from_numpy(y[eval_idx])

    topk_ids = _cosine_topk(Xe, embed_table, topk=topk)
    hits = topk_ids.eq(yids.unsqueeze(1))
    top1 = float(hits[:, 0].to(torch.float32).mean().item())
    top10 = float(
        hits[:, : min(10, hits.shape[1])].any(dim=1).to(torch.float32).mean().item()
    )

    return AttackResult(
        attack="nn",
        condition=snapshots.condition,
        model_id=snapshots.model_id,
        n_prompts=snapshots.n_prompts(),
        n_train=0,
        n_test=int(Xe.shape[0]),
        ttrsr_top1=top1,
        ttrsr_top10=top10,
        risk_level=classify_risk_level(top1),
        extra={
            "layer": layer,
            "kind": kind,
            "strip_shield": strip_shield,
            "matching_strategy": "cosine_topk_no_train",
        },
    )


def _cli() -> None:
    p = argparse.ArgumentParser(description="Run NN against a snapshot set")
    p.add_argument("--snapshot-basename", required=True)
    p.add_argument("--snapshot-root", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--kind", default="q_proj")
    p.add_argument("--strip-shield", action="store_true", default=True)
    p.add_argument("--no-strip-shield", dest="strip_shield", action="store_false")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from snapshots_loader import SnapshotSet  # type: ignore

    snaps = SnapshotSet.open(args.snapshot_basename, root=args.snapshot_root)
    embed_table = load_qwen3_embedding_table(args.model_id)
    result = run(
        snaps,
        embed_table=embed_table,
        layer=args.layer,
        kind=args.kind,
        strip_shield=args.strip_shield,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2))
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    _cli()

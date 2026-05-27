"""ISA — Internal State Attack driver.

Same ridge inverter as :mod:`run_ima` but targets a *deep* layer's
hidden state instead of layer 0 — AloePri's `ISABaselineConfig`
defaults to `observable_layer = 23`. For Qwen3-1.7B (28 layers) we
target layer 23 as well to keep numbers aligned.

ISA is paired with IMA as the load-bearing privacy claim:

* C0 plain on layer 23 should still recover ≥ 80% (the deep
  hidden state of a small LM stays close to the embedding under
  cosine match).
* C2 default must report < 10% TTRSR (release-gate threshold).
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
    load_aloepri_module,
    stack_prompt_observations,
    train_val_test_split,
    vocab_disjoint_train_val_test_split,
)
from .run_ima import load_qwen3_embedding_table


_ima = load_aloepri_module("src/security_qwen/ima.py")
_evaluate_inversion_predictions = _ima._evaluate_inversion_predictions
_fit_ridge_regressor = _ima._fit_ridge_regressor
_predict_ridge = _ima._predict_ridge


def run(
    snapshots,
    *,
    embed_table: torch.Tensor,
    layer: int = 23,
    kind: str = "q_proj",
    n_train: int = 256,
    n_val: int = 64,
    n_test: int = 64,
    topk: int = 10,
    ridge_alphas: tuple[float, ...] = (1e-4, 1e-2, 1.0),
    strip_shield: bool = True,
    candidate_pool_size: int = 2048,
    seed: int = 20260518,
    split_mode: str = "vocab",
) -> AttackResult:
    """Run ISA against a single condition's snapshots. The defaults
    mirror `ISABaselineConfig` (sequence_length=8, train_sequences=64
    → ~256 rows at the smaller corpus we run).
    """
    X, y, _ = stack_prompt_observations(
        snapshots, layer=layer, kind=kind, strip_shield=strip_shield
    )
    if X.shape[0] == 0:
        return AttackResult(
            attack="isa",
            condition=snapshots.condition,
            model_id=snapshots.model_id,
            n_prompts=snapshots.n_prompts(),
            n_train=0,
            n_test=0,
            ttrsr_top1=None,
            ttrsr_top10=None,
            risk_level="unknown",
            extra={
                "note": "no snapshots at the requested (layer, kind)",
                "layer": layer,
                "kind": kind,
            },
        )
    # obs_dim may differ from embed_dim under static-weight obfuscation
    # (AloePri keymat h=128 expansion gives obs=2304 vs plain embed=2048).
    # The ridge primitive handles non-square X→Y; just log and continue.
    if X.shape[1] != embed_table.shape[1]:
        print(
            f"  [isa] obs_dim={X.shape[1]} != embed_dim={embed_table.shape[1]} — "
            f"ridge will fit a dim-bridging inverter"
        )

    # split_mode='row' = legacy row-shuffle; 'vocab' = paper-faithful
    # vocab-disjoint (generalising attacker).
    if split_mode == "vocab":
        Xtr_np, ytr_np, Xva_np, yva_np, Xte_np, yte_np = (
            vocab_disjoint_train_val_test_split(
                X, y, n_train=n_train, n_val=n_val, n_test=n_test, seed=seed,
            )
        )
    else:
        Xtr_np, ytr_np, Xva_np, yva_np, Xte_np, yte_np = train_val_test_split(
            X, y, n_train=n_train, n_val=n_val, n_test=n_test, seed=seed
        )
    if Xtr_np.shape[0] == 0 or Xte_np.shape[0] == 0:
        return AttackResult(
            attack="isa",
            condition=snapshots.condition,
            model_id=snapshots.model_id,
            n_prompts=snapshots.n_prompts(),
            n_train=int(Xtr_np.shape[0]),
            n_test=int(Xte_np.shape[0]),
            ttrsr_top1=None,
            ttrsr_top10=None,
            risk_level="unknown",
            extra={"note": "not enough rows for train + val + test split"},
        )

    Xtr = torch.from_numpy(Xtr_np).to(torch.float32)
    Xva = torch.from_numpy(Xva_np).to(torch.float32)
    Xte = torch.from_numpy(Xte_np).to(torch.float32)
    ytr = embed_table[torch.from_numpy(ytr_np)].to(torch.float32)
    yva_ids = torch.from_numpy(yva_np)
    yte_ids = torch.from_numpy(yte_np)

    rng = np.random.default_rng(seed + 2)
    vocab_size = embed_table.shape[0]
    candidate_pool = torch.from_numpy(
        rng.choice(vocab_size, size=min(candidate_pool_size, vocab_size), replace=False)
    ).to(torch.long)
    candidate_pool = torch.unique(torch.cat([candidate_pool, yva_ids, yte_ids]))

    # Multi-alpha selection — AloePri reference pattern (ima.py:506-518).
    alpha_scores: list[dict[str, float]] = []
    best_alpha = None
    best_val_top1 = -1.0
    best_model = None
    for alpha in ridge_alphas:
        model = _fit_ridge_regressor(Xtr, ytr, ridge_alpha=float(alpha))
        val_pred = _predict_ridge(model, Xva)
        val_metrics = _evaluate_inversion_predictions(
            predicted_embeddings=val_pred,
            true_plain_ids=yva_ids,
            candidate_plain_ids=candidate_pool,
            baseline_embed=embed_table,
            topk=topk,
        )
        val_top1 = float(val_metrics["token_top1_recovery_rate"])
        alpha_scores.append({"ridge_alpha": float(alpha), "val_top1": val_top1})
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_alpha = float(alpha)
            best_model = model

    predicted = _predict_ridge(best_model, Xte)
    metrics = _evaluate_inversion_predictions(
        predicted_embeddings=predicted,
        true_plain_ids=yte_ids,
        candidate_plain_ids=candidate_pool,
        baseline_embed=embed_table,
        topk=topk,
    )
    top1 = float(metrics["token_top1_recovery_rate"])
    top10 = float(metrics["token_top10_recovery_rate"])
    return AttackResult(
        attack="isa",
        condition=snapshots.condition,
        model_id=snapshots.model_id,
        n_prompts=snapshots.n_prompts(),
        n_train=int(Xtr.shape[0]),
        n_test=int(Xte.shape[0]),
        ttrsr_top1=top1,
        ttrsr_top10=top10,
        risk_level=classify_risk_level(top1),
        extra={
            "layer": layer,
            "kind": kind,
            "split_mode": split_mode,
            "best_ridge_alpha": best_alpha,
            "ridge_alpha_val_scan": alpha_scores,
            "strip_shield": strip_shield,
            "candidate_pool_size": int(candidate_pool.shape[0]),
            "embedding_cosine_similarity": float(metrics["embedding_cosine_similarity"]),
            "observable_type": "hidden_state",
        },
    )


def _cli() -> None:
    p = argparse.ArgumentParser(description="Run ISA against a snapshot set")
    p.add_argument("--snapshot-basename", required=True)
    p.add_argument("--snapshot-root", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--layer", type=int, default=23)
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

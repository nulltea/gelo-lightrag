"""IMA — Inversion Model Attack driver.

Trains AloePri's ridge inverter on `(hidden_state_layer_L, token_embedding)`
pairs harvested from a held-out subset of the prompt corpus, then
evaluates token-id recovery on the held-out test split. The math
primitives come from `vendor/aloepri-py/src/security_qwen/ima.py`
unchanged — only the *data assembly* changes for GELO.

Threat-model alignment:

* C0 (plain) — should land near 100% TTRSR. The hidden state is the
  unmasked Qwen3 activation; ridge can invert it to the embedding
  row directly.
* C1 (mask only) — should land in the 30–70% range. The per-offload
  Haar mask scrambles each operand independently; without paired
  training data the ridge can't fit.
* C2 (mask + shield) — must land below 10%. The per-forward-pass
  fresh mask + shield rows reduce the inverter's signal to noise.
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


_ima = load_aloepri_module("src/security_qwen/ima.py")
_evaluate_inversion_predictions = _ima._evaluate_inversion_predictions
_fit_ridge_regressor = _ima._fit_ridge_regressor
_predict_ridge = _ima._predict_ridge


def load_qwen3_embedding_table(model_id: str) -> torch.Tensor:
    """Load the Qwen3 input-embedding table from the HF safetensors
    cache directly — bypasses the `transformers` dependency.

    Reads the per-shard index, finds the shard holding
    `model.embed_tokens.weight`, and pulls just that tensor via
    `safetensors.safe_open`. Returns a `(vocab_size, hidden_size)`
    float32 tensor; Qwen3-1.7B is ~610 MB at f32.

    Falls back to `transformers.AutoModelForCausalLM` only when the
    cache layout differs from what we expect (e.g. single-file
    safetensors with no index).
    """
    import glob
    import json
    import os

    # The Rust runner already downloaded the model via hf-hub, so the
    # safetensors live under the standard HF cache layout.
    cache_root = os.environ.get(
        "HF_HOME",
        os.path.expanduser("~/.cache/huggingface"),
    )
    repo_dir_name = "models--" + model_id.replace("/", "--")
    candidates = sorted(
        glob.glob(os.path.join(cache_root, "hub", repo_dir_name, "snapshots", "*"))
    )
    if not candidates:
        # Fallback path — let transformers download if the cache is missing.
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="float32")
        emb = model.get_input_embeddings().weight.detach().to(torch.float32).clone()
        del model
        return emb

    snapshot_dir = candidates[-1]
    index_path = os.path.join(snapshot_dir, "model.safetensors.index.json")
    embed_key = "model.embed_tokens.weight"
    if os.path.exists(index_path):
        with open(index_path) as fh:
            weight_map = json.load(fh)["weight_map"]
        shard = weight_map[embed_key]
        shard_path = os.path.join(snapshot_dir, shard)
    else:
        # Single-file layout.
        shard_path = os.path.join(snapshot_dir, "model.safetensors")
        if not os.path.exists(shard_path):
            raise FileNotFoundError(
                f"Couldn't find Qwen3-1.7B safetensors under {snapshot_dir}; "
                "expected an `model.safetensors.index.json` shard map or a "
                "single `model.safetensors`."
            )

    from safetensors import safe_open

    with safe_open(shard_path, framework="pt", device="cpu") as fh:
        return fh.get_tensor(embed_key).to(torch.float32).clone()


def run(
    snapshots,
    *,
    embed_table: torch.Tensor,
    layer: int = 0,
    kind: str = "q_proj",
    n_train: int = 1024,
    n_val: int = 128,
    n_test: int = 128,
    topk: int = 10,
    ridge_alphas: tuple[float, ...] = (1e-4, 1e-2, 1.0),
    strip_shield: bool = True,
    candidate_pool_size: int = 2048,
    seed: int = 20260518,
    split_mode: str = "row",
) -> AttackResult:
    """Run IMA against a single condition's snapshots. Returns one
    :class:`AttackResult` ready for the per-condition results JSON.

    The "kind" defaults to `q_proj` at layer 0 — the input to the
    first attention block, which is the closest GELO analog to
    AloePri's "post-embedding observable" (token embed → masked
    activation). Override `kind` / `layer` to sweep over capture
    points for the §4.3 long-context regime study.
    """
    X, y, _ = stack_prompt_observations(
        snapshots, layer=layer, kind=kind, strip_shield=strip_shield
    )
    if X.shape[0] == 0:
        return AttackResult(
            attack="ima",
            condition=snapshots.condition,
            model_id=snapshots.model_id,
            n_prompts=snapshots.n_prompts(),
            n_train=0,
            n_test=0,
            ttrsr_top1=None,
            ttrsr_top10=None,
            risk_level="unknown",
            extra={"note": "no snapshots at the requested (layer, kind)"},
        )

    embed_dim = embed_table.shape[1]
    # Note: obs_dim may differ from embed_dim under static-weight
    # obfuscation that pads the residual (e.g. AloePri keymat h=128
    # expansion: obs is 2304 vs plain embed 2048). The ridge primitive
    # handles non-square X→Y so we just let it through.
    if X.shape[1] != embed_dim:
        print(
            f"  [ima] obs_dim={X.shape[1]} != embed_dim={embed_dim} — "
            f"ridge will fit a dim-bridging inverter"
        )

    # split_mode='row' = legacy row-shuffle (memorising attacker);
    # 'vocab' = paper-faithful vocab-disjoint (generalising attacker).
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
            attack="ima",
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

    rng = np.random.default_rng(seed + 1)
    vocab_size = embed_table.shape[0]
    candidate_pool = torch.from_numpy(
        rng.choice(vocab_size, size=min(candidate_pool_size, vocab_size), replace=False)
    ).to(torch.long)
    candidate_pool = torch.unique(torch.cat([candidate_pool, yva_ids, yte_ids]))

    # Multi-alpha selection per AloePri reference: fit on train, score
    # on val, pick the alpha with the best val top-1. Avoids the
    # under-fit single-alpha=1.0 trap that gave a 28% C0 ISA in the
    # 64-prompt run.
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
        attack="ima",
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
        },
    )


def _cli() -> None:
    p = argparse.ArgumentParser(description="Run IMA against a snapshot set")
    p.add_argument("--snapshot-basename", required=True)
    p.add_argument("--snapshot-root", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--kind", default="q_proj")
    p.add_argument("--strip-shield", action="store_true", default=True)
    p.add_argument("--no-strip-shield", dest="strip_shield", action="store_false")
    p.add_argument("--n-train", type=int, default=1024)
    p.add_argument("--n-test", type=int, default=128)
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
        n_train=args.n_train,
        n_test=args.n_test,
        strip_shield=args.strip_shield,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2))
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    _cli()

"""Paper-faithful labelled-ridge ISA on hidden states, with multi-key attacker synthesis.

The standard ridge ISA driver
(`evals/aloepri-attacks/attack_drivers/run_isa.py`) trains a ridge
inverter `hidden_state[L] → W_e[plain_id]` on labelled pairs where
the labels come from running known prompts through the *deployment*
with the deployment's secret τ applied client-side. Under paper §3.2
the attacker does not have τ, so they cannot label deployment-side
hidden-state captures with plain_ids. That makes the standard driver
non-paper-faithful (it has the same τ-leak we just fixed in IMA).

Reconstructing the attack within paper §3.2's threat model, the attacker:

1. Has Kerckhoffs access to the obfuscation algorithm (h, λ, α_e, α_h,
   Algorithm 1, Algorithm 2) and to the public plaintext model.
2. Runs the *plain* model on prompts whose plain_ids they know — no
   τ knowledge needed (the plain model has identity τ). This produces
   `State_plain[L][position]` paired with `plain_id[position]`.
3. Generates their own attacker-side keymats K_a^k via Algorithm 1.
   They cannot run the obfuscated model end-to-end (would need to
   apply Algorithm 2 transforms to every weight and rebuild the
   network), but under AloePri's covariance claim the obfuscated
   residual at any layer L equals K_e · State_plain[L]. So the
   attacker can *synthesise* what their own obfuscated model's
   hidden state would have been:
       `State_a^k[L][i] = State_plain[L][i] @ K_a^k`
4. Trains a ridge inverter on the concatenation across K keys:
       `(State_a^k[L][i], W_e[plain_id_i])` for k=1..K, i=1..n_pairs
   forcing key-invariant inversion (same multi-key trick as
   `run_ima_embedrow_attacks_multikey.py`).
5. At inference time, captures deployment-side hidden states
   `State_d[L]` (server-side observable, no τ needed — these are
   actual obfuscated runtime states). Applies the trained inverter,
   does cosine-NN against the public W_e to recover the top-1
   plain_id.

`--identity-tau` runs the calibration probe: train + test directly
on `State_plain` with no synthesis and no K. Tests the ridge
attacker's ceiling on a no-defense task.

The covariance approximation skips per-layer additive noise (path-2
only adds noise at the embedding and the head, not at every
intermediate layer — see `python/aloepri-llm/obfuscate_qwen3_gguf.py:332-352`).
Algorithm 2's intra-head transforms also preserve the residual basis
by design, so synthesis matches the deployment up to higher-order
Algorithm-2 perturbations on the post-attention residual.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Existing reusable building blocks.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snapshots_loader import SnapshotSet  # type: ignore  # noqa: E402
from attack_drivers import run_ima  # type: ignore  # noqa: E402
from attack_drivers.common import (  # type: ignore  # noqa: E402
    AttackResult,
    classify_risk_level,
    stack_prompt_observations,
)


# ───── Multi-key attacker construction (mirrors IMA multi-key driver) ──────


def _build_attacker_keymat_pool(
    *,
    d: int,
    expansion: int,
    lam: float,
    num_keys: int,
    attacker_seed: int,
) -> torch.Tensor:
    """Pre-generate K independent attacker keymats K_a^k via the public
    Algorithm 1. Returns (K, d, d + 2h) on CPU.
    """
    sys.path.insert(0, "/home/timo/repos/private-rag-path-2/vendor/aloepri-py")
    sys.path.insert(0, "/home/timo/repos/private-rag-path-2/vendor/aloepri-py/src")
    from keymat import build_keymat_transform  # type: ignore  # noqa: E402

    d_obs = d + 2 * expansion
    pool = torch.empty((num_keys, d, d_obs), dtype=torch.float32)
    for k in range(num_keys):
        transform = build_keymat_transform(
            d=d, h=expansion, lam=lam, init_seed=attacker_seed + 1 + 10_000 * k,
        )
        pool[k] = transform.key.to(torch.float32)
    return pool


# ───── Ridge solver (multi-α, val-selected) ────────────────────────────────


def _fit_ridge(
    X: torch.Tensor, Y: torch.Tensor, *, ridge_alpha: float,
) -> dict[str, torch.Tensor]:
    """Standard closed-form ridge — same primitive as
    `attack_drivers/run_isa.py` and `vendor/aloepri-py` reference.
    """
    x_mean = X.mean(dim=0, keepdim=True)
    x_std = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_mean = Y.mean(dim=0, keepdim=True)
    y_std = Y.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xn = (X - x_mean) / x_std
    Yn = (Y - y_mean) / y_std
    ones = torch.ones((Xn.shape[0], 1), dtype=Xn.dtype)
    Xa = torch.cat([Xn, ones], dim=1)
    n = Xa.shape[1]
    I = torch.eye(n, dtype=Xn.dtype)
    I[-1, -1] = 0.0
    lhs = Xa.T @ Xa + ridge_alpha * I
    rhs = Xa.T @ Yn
    W = torch.linalg.solve(lhs, rhs)
    return {"weight": W, "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


def _predict_ridge(model: dict[str, torch.Tensor], X: torch.Tensor) -> torch.Tensor:
    Xn = (X - model["x_mean"]) / model["x_std"]
    ones = torch.ones((Xn.shape[0], 1), dtype=Xn.dtype)
    Xa = torch.cat([Xn, ones], dim=1)
    Yn = Xa @ model["weight"]
    return Yn * model["y_std"] + model["y_mean"]


# ───── Cosine-NN top-1/top-10 against full vocab ───────────────────────────


def _cosine_topk(
    pred: torch.Tensor, embed_table: torch.Tensor, true_ids: torch.Tensor, topk: int = 10,
    chunk: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (top1_hits, topk_hits) as bool tensors over the test rows."""
    pn = pred / pred.norm(dim=1, keepdim=True).clamp_min(1e-8)
    best_scores = torch.full((pn.shape[0], 0), float("-inf"), dtype=pn.dtype)
    best_ids = torch.empty((pn.shape[0], 0), dtype=torch.long)
    vocab = embed_table.shape[0]
    k_eff = min(topk, vocab)
    for s in range(0, vocab, chunk):
        e = min(s + chunk, vocab)
        cn = embed_table[s:e]
        cn = cn / cn.norm(dim=1, keepdim=True).clamp_min(1e-8)
        sc = pn @ cn.T  # (N_test, e-s)
        c_scores, c_local = torch.topk(sc, k=min(k_eff, sc.shape[1]), dim=1)
        c_ids = c_local + s
        merged_scores = torch.cat([best_scores, c_scores], dim=1)
        merged_ids = torch.cat([best_ids, c_ids], dim=1)
        new_scores, new_idx = torch.topk(merged_scores, k=k_eff, dim=1)
        best_scores = new_scores
        best_ids = merged_ids.gather(1, new_idx)
    hits = best_ids.eq(true_ids.unsqueeze(1))
    top1 = hits[:, 0]
    topk = hits[:, :k_eff].any(dim=1)
    return top1, topk


# ───── Main entrypoint ────────────────────────────────────────────────────


def run_isa_multikey(
    *,
    plain_snapshots: SnapshotSet,
    obf_snapshots: SnapshotSet | None,
    embed_table: torch.Tensor,
    layer: int,
    kind: str = "attn_norm",
    attacker_expansion: int = 128,
    attacker_lam: float = 0.3,
    attacker_num_keys: int = 64,
    attacker_seed: int = 20260521,
    ridge_alphas: tuple[float, ...] = (1e-4, 1e-2, 1.0),
    train_frac: float = 0.5,
    val_frac: float = 0.25,
    identity_tau: bool = False,
    topk: int = 10,
) -> AttackResult:
    """Run paper-faithful multi-key labelled-ridge ISA at one (layer, kind).

    plain_snapshots provides the State_plain inputs for training-data
    synthesis; obf_snapshots provides the deployment-side State_d for
    test eval. In identity_tau mode obf_snapshots is unused — train +
    test both come from plain captures (calibration probe).
    """
    t0 = time.perf_counter()

    # 1) Load plain hidden states (and their plain_id labels)
    X_plain, y_ids, _ = stack_prompt_observations(
        plain_snapshots, layer=layer, kind=kind, strip_shield=True,
    )
    if X_plain.shape[0] == 0:
        raise RuntimeError(f"no plain snapshots at layer={layer} kind={kind}")
    X_plain_t = torch.from_numpy(X_plain).to(torch.float32)
    y_ids_t = torch.from_numpy(y_ids).to(torch.long)
    d_plain = int(X_plain_t.shape[1])
    n_total = int(X_plain_t.shape[0])

    print(f"  plain captures: {n_total} (state, plain_id) rows at "
          f"layer={layer} kind={kind} d_plain={d_plain}")
    if d_plain != int(embed_table.shape[1]):
        raise RuntimeError(
            f"plain state dim {d_plain} != embed_table dim {embed_table.shape[1]} — "
            f"unexpected; the plain model's residual stream should match W_e dim"
        )

    # 2) Vocab-disjoint split on plain_ids
    unique_ids = torch.unique(y_ids_t).tolist()
    rng = np.random.default_rng(attacker_seed + 17)
    shuffled = rng.permutation(unique_ids).tolist()
    n_train_ids = int(len(shuffled) * train_frac)
    n_val_ids = int(len(shuffled) * val_frac)
    train_ids = set(shuffled[:n_train_ids])
    val_ids = set(shuffled[n_train_ids : n_train_ids + n_val_ids])
    test_ids = set(shuffled[n_train_ids + n_val_ids :])

    def _mask(ids_set):
        return torch.tensor([int(i) in ids_set for i in y_ids_t.tolist()], dtype=torch.bool)

    tr_mask = _mask(train_ids)
    va_mask = _mask(val_ids)
    te_mask = _mask(test_ids)
    print(f"  vocab-disjoint split: train={int(tr_mask.sum())} val={int(va_mask.sum())} "
          f"test={int(te_mask.sum())} rows (over {len(unique_ids)} unique plain_ids)")

    # 3) Build training inputs
    if identity_tau:
        # Calibration probe: train + test on plain states directly.
        print("  identity-τ calibration: ridge on (State_plain, W_e[plain_id]) — no K_a")
        X_train = X_plain_t[tr_mask]
        y_train = embed_table[y_ids_t[tr_mask]]
        X_val = X_plain_t[va_mask]
        y_val_ids = y_ids_t[va_mask]
        X_test = X_plain_t[te_mask]
        y_test_ids = y_ids_t[te_mask]
    else:
        # Paper-faithful: synthesise K attacker-keymat-transformed inputs
        # per training row. Test inputs come from obf captures.
        if obf_snapshots is None:
            raise RuntimeError("obf_snapshots is required in non-identity_tau mode")
        print(f"  pre-generating multi-key pool: K={attacker_num_keys} keymats "
              f"(h={attacker_expansion}, λ={attacker_lam}, seed={attacker_seed})")
        keymat_pool = _build_attacker_keymat_pool(
            d=d_plain, expansion=int(attacker_expansion), lam=float(attacker_lam),
            num_keys=int(attacker_num_keys), attacker_seed=int(attacker_seed),
        )

        # Synthesise: X_a^k[i] = X_plain[i] @ K_a^k. Stack across k.
        X_plain_train = X_plain_t[tr_mask]
        X_plain_val = X_plain_t[va_mask]
        synth_train_chunks: list[torch.Tensor] = []
        synth_train_y_chunks: list[torch.Tensor] = []
        synth_val_chunks: list[torch.Tensor] = []
        for k in range(int(attacker_num_keys)):
            K_k = keymat_pool[k]  # (d_plain, d_obs)
            X_a_train = X_plain_train @ K_k  # (n_train, d_obs)
            X_a_val = X_plain_val @ K_k
            synth_train_chunks.append(X_a_train)
            synth_train_y_chunks.append(embed_table[y_ids_t[tr_mask]])
            synth_val_chunks.append(X_a_val)
        X_train = torch.cat(synth_train_chunks, dim=0)
        y_train = torch.cat(synth_train_y_chunks, dim=0)
        X_val = torch.cat(synth_val_chunks, dim=0)
        # Val labels repeat across the K synth copies of the val set
        y_val_ids_single = y_ids_t[va_mask]
        y_val_ids = y_val_ids_single.repeat(int(attacker_num_keys))
        print(f"  synthesised training tensor: X_train {tuple(X_train.shape)} "
              f"y_train {tuple(y_train.shape)} (K × n_train)")

        # Test inputs: deployment's obf captures at the same layer, but
        # filtered to test plain_ids only (vocab-disjoint from train).
        X_obf_full, y_obf_ids, _ = stack_prompt_observations(
            obf_snapshots, layer=layer, kind=kind, strip_shield=True,
        )
        X_obf_t = torch.from_numpy(X_obf_full).to(torch.float32)
        y_obf_t = torch.from_numpy(y_obf_ids).to(torch.long)
        te_obf_mask = torch.tensor([int(i) in test_ids for i in y_obf_t.tolist()], dtype=torch.bool)
        X_test = X_obf_t[te_obf_mask]
        y_test_ids = y_obf_t[te_obf_mask]
        print(f"  obf test rows: {int(te_obf_mask.sum())} (filtered to test plain_ids)")

    # 4) Multi-α ridge selection on val
    best_alpha: float | None = None
    best_val_top1: float = -1.0
    best_model: dict[str, torch.Tensor] | None = None
    alpha_scores: list[dict[str, Any]] = []
    for alpha in ridge_alphas:
        model = _fit_ridge(X_train, y_train, ridge_alpha=float(alpha))
        val_pred = _predict_ridge(model, X_val)
        v_top1, v_topk = _cosine_topk(val_pred, embed_table, y_val_ids, topk=topk)
        v_top1_f = float(v_top1.float().mean().item())
        v_topk_f = float(v_topk.float().mean().item())
        alpha_scores.append({"ridge_alpha": float(alpha), "val_top1": v_top1_f, "val_topk": v_topk_f})
        print(f"    α={alpha:.0e}: val_top1={v_top1_f:.4f} val_top{topk}={v_topk_f:.4f}")
        if v_top1_f > best_val_top1:
            best_val_top1 = v_top1_f
            best_alpha = float(alpha)
            best_model = model
    assert best_model is not None

    # 5) Final test eval
    test_pred = _predict_ridge(best_model, X_test)
    te_top1, te_topk = _cosine_topk(test_pred, embed_table, y_test_ids, topk=topk)
    top1 = float(te_top1.float().mean().item())
    top10 = float(te_topk.float().mean().item())

    return AttackResult(
        attack="isa_multikey",
        condition="obfuscated" if not identity_tau else "plain_identity_tau",
        model_id=plain_snapshots.model_id,
        n_prompts=plain_snapshots.n_prompts() if hasattr(plain_snapshots, "n_prompts") else 0,
        n_train=int(X_train.shape[0]),
        n_test=int(X_test.shape[0]),
        ttrsr_top1=top1,
        ttrsr_top10=top10,
        risk_level=classify_risk_level(top1),
        extra={
            "layer": int(layer),
            "kind": str(kind),
            "threat_model_regime": "multikey_covariant_synthesis_paperfaithful",
            "attacker_identity_probe": bool(identity_tau),
            "attacker_expansion": int(attacker_expansion),
            "attacker_lam": float(attacker_lam),
            "attacker_num_keys": int(attacker_num_keys) if not identity_tau else 0,
            "attacker_seed": int(attacker_seed),
            "best_ridge_alpha": best_alpha,
            "alpha_scan": alpha_scores,
            "n_unique_plain_ids": len(unique_ids),
            "n_train_ids": len(train_ids),
            "n_val_ids": len(val_ids),
            "n_test_ids": len(test_ids),
            "runtime_seconds": round(time.perf_counter() - t0, 2),
        },
    )


# ───── CLI ─────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description="Paper-faithful labelled-ridge ISA with multi-key attacker synthesis"
    )
    p.add_argument("--plain-captures", type=Path, required=True,
                   help="Directory containing hidden.{safetensors,meta.json} from plain-model capture.")
    p.add_argument("--obf-captures", type=Path, default=None,
                   help="Directory containing hidden.{safetensors,meta.json} from obfuscated-model capture. "
                        "Required unless --identity-tau is set.")
    p.add_argument("--plain-model-id", type=str, default="Qwen/Qwen3-4B",
                   help="HF model id whose W_e is loaded as the candidate pool and inversion target.")
    p.add_argument("--layer", type=int, default=17,
                   help="Hidden-state capture layer to attack. Default 17 ≈ 48 percent depth on 36-layer Q3.")
    p.add_argument("--kind", type=str, default="attn_norm")
    p.add_argument("--identity-tau", action="store_true",
                   help="Calibration probe: ridge on plain captures only, no synthesis, no K_a.")
    p.add_argument("--attacker-expansion", type=int, default=128)
    p.add_argument("--attacker-lambda", type=float, default=0.3)
    p.add_argument("--attacker-num-keys", type=int, default=64)
    p.add_argument("--attacker-seed", type=int, default=20260521)
    p.add_argument("--ridge-alpha", type=float, action="append", default=None,
                   help="Override the default multi-α grid. Can be passed multiple times.")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    if not args.identity_tau and args.obf_captures is None:
        raise SystemExit("--obf-captures is required unless --identity-tau is set")

    print(f"[ISA-multikey] plain captures: {args.plain_captures}")
    plain_snap = SnapshotSet.open(args.plain_captures / "hidden")
    print(f"  {plain_snap.n_prompts()} prompt(s), layers={plain_snap.captured_layers}, "
          f"kinds={plain_snap.captured_kinds}")
    if args.identity_tau:
        obf_snap = None
    else:
        print(f"[ISA-multikey] obf captures: {args.obf_captures}")
        obf_snap = SnapshotSet.open(args.obf_captures / "hidden")
        print(f"  {obf_snap.n_prompts()} prompt(s), layers={obf_snap.captured_layers}, "
              f"kinds={obf_snap.captured_kinds}")

    print(f"[ISA-multikey] loading embed table for {args.plain_model_id}")
    embed_table = run_ima.load_qwen3_embedding_table(args.plain_model_id).to(torch.float32)
    print(f"  W_e shape = {tuple(embed_table.shape)}")

    ridge_alphas = tuple(args.ridge_alpha) if args.ridge_alpha else (1e-4, 1e-2, 1.0)

    result = run_isa_multikey(
        plain_snapshots=plain_snap,
        obf_snapshots=obf_snap,
        embed_table=embed_table,
        layer=int(args.layer),
        kind=str(args.kind),
        attacker_expansion=int(args.attacker_expansion),
        attacker_lam=float(args.attacker_lambda),
        attacker_num_keys=int(args.attacker_num_keys),
        attacker_seed=int(args.attacker_seed),
        ridge_alphas=ridge_alphas,
        identity_tau=bool(args.identity_tau),
    )

    print(f"[ISA-multikey] top1={result.ttrsr_top1:.4f} top10={result.ttrsr_top10:.4f} "
          f"risk={result.risk_level} α*={result.extra.get('best_ridge_alpha')}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "format": "aloepri_m2_7_isa_multikey_v1",
        "plain_captures": str(args.plain_captures),
        "obf_captures": str(args.obf_captures) if args.obf_captures else None,
        "plain_model_id": args.plain_model_id,
        "attack": result.to_dict(),
    }, indent=2))
    print(f"[ISA-multikey] wrote → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""H2: truncate per-prompt KV columns to actual prompt length.

Current flow:    (32 heads, n_q, 256 KV-slots) -> (n_q, 32*256=8192)
This flow:       (32 heads, n_q, n_q causal) -> (n_q, 32*n_q)
                 OR stack per-prompt with column-wise truncation
                 then zero-pad to max(n_q) so ridge dims agree.

We try two ablations:
  A. truncate-then-stack-with-zero-pad-to-max-q
  B. truncate to first valid KV cols AND first valid query rows; then
     compute diagonal-only feature (self-attention weight per position)
"""
from __future__ import annotations
import argparse, copy, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path("/home/timo/repos/private-rag-path-2")
sys.path.insert(0, str(REPO / "evals" / "aloepri-attacks"))
from attack_drivers import run_isa  # type: ignore
from attack_drivers.run_ima import load_qwen3_embedding_table  # type: ignore
from snapshots_loader import SnapshotSet  # type: ignore


def make_shim(attn_set, layer, kind, transform):
    pairs = []
    for s in attn_set.select(kind=kind, layer=layer):
        op = attn_set.get_operand(s, strip_shield=False)  # (32, n_q, 256)
        if op.ndim != 3:
            continue
        n_heads, n_q, n_kv = op.shape
        ids = attn_set.prompt_token_ids[s.prompt_idx]
        n_valid = min(n_q, len(ids))
        # Truncate to valid query rows AND valid KV columns
        op_trunc = op[:, :n_valid, :n_valid]  # (32, n_valid, n_valid)
        feat = transform(op_trunc, n_valid)
        pairs.append((s.prompt_idx, feat))
    shim = copy.copy(attn_set)
    shim.per_prompt_layer_kind_tensors = lambda **kw: pairs
    return shim, pairs


def t_pad_to_max(op_trunc, n_valid):
    """(32, n_valid, n_valid) -> (n_valid, 32*max_n_q) padded with zeros."""
    n_heads = op_trunc.shape[0]
    # Don't actually pad here — let stack_prompt_observations handle it.
    # But ridge needs uniform dim. Pad to 32 (max prompt length in batch).
    MAX = 32
    flat = op_trunc.permute(1, 0, 2).reshape(n_valid, n_heads * n_valid)
    if flat.shape[1] < n_heads * MAX:
        padded = torch.zeros(n_valid, n_heads * MAX, dtype=flat.dtype)
        padded[:, :flat.shape[1]] = flat
        flat = padded
    return flat


def t_diagonal(op_trunc, n_valid):
    """Take just the diagonal: (32, n_valid, n_valid) -> (n_valid, 32).
    The diagonal[i] is the self-attention weight from position i to itself.
    """
    diag = op_trunc.diagonal(dim1=1, dim2=2)  # (32, n_valid)
    return diag.T.contiguous()  # (n_valid, 32)


def t_row_means(op_trunc, n_valid):
    """Per-head row stats: (32, n_valid, n_valid) -> (n_valid, 32*4)
    = mean, std, max, sum-of-top3 per row per head."""
    feats = []
    for stat in ["mean", "std", "max"]:
        if stat == "mean":
            f = op_trunc.mean(dim=2)
        elif stat == "std":
            f = op_trunc.std(dim=2)
        elif stat == "max":
            f = op_trunc.max(dim=2).values
        feats.append(f.T)  # (n_valid, 32)
    return torch.cat(feats, dim=1)  # (n_valid, 32*3 = 96)


def t_default_no_truncation(op_trunc, n_valid):
    """Baseline: don't truncate, return the original (32, n_q, 256) flattened
    to (n_q, 32*256). This mirrors the current driver."""
    n_heads = op_trunc.shape[0]
    return op_trunc.permute(1, 0, 2).reshape(n_valid, n_heads * n_valid).contiguous()


TRANSFORMS = {
    "current": None,  # special-cased to mean "no transform — use original 256-col path"
    "truncate_pad": t_pad_to_max,
    "diagonal": t_diagonal,
    "row_stats": t_row_means,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("captures_dir", type=Path)
    p.add_argument("--model-id", default="Qwen/Qwen3-4B")
    p.add_argument("--layer", type=int, default=17)
    p.add_argument("--kind", default="kq")
    p.add_argument("--split", default="row")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=20260518)
    p.add_argument("--transform", choices=list(TRANSFORMS.keys()), default="truncate_pad")
    p.add_argument("--n-train", type=int, default=256)
    p.add_argument("--n-val", type=int, default=64)
    p.add_argument("--n-test", type=int, default=64)
    args = p.parse_args()

    attn = SnapshotSet.open("attn", root=args.captures_dir)
    embed = load_qwen3_embedding_table(args.model_id).to(torch.float32)

    if args.transform == "current":
        # mirror existing _isa_attn_score
        pairs = []
        for s in attn.select(kind=args.kind, layer=args.layer):
            op = attn.get_operand(s, strip_shield=False)
            if op.ndim == 3:
                n_heads, n_q, n_kv = op.shape
                op = op.permute(1, 0, 2).reshape(n_q, n_heads * n_kv)
            pairs.append((s.prompt_idx, op))
        shim = copy.copy(attn)
        shim.per_prompt_layer_kind_tensors = lambda **kw: pairs
    else:
        shim, pairs = make_shim(attn, args.layer, args.kind, TRANSFORMS[args.transform])

    sample_shape = pairs[0][1].shape
    total_rows = sum(p.shape[0] for _, p in pairs)
    print(f"[probe1] transform={args.transform}  feature_shape_sample={tuple(sample_shape)}  "
          f"total_rows={total_rows}")

    top1s, top10s = [], []
    t0 = time.perf_counter()
    for k in range(args.seeds):
        seed = args.seed_base + k
        res = run_isa.run(
            shim, embed_table=embed,
            layer=args.layer, kind=args.kind,
            n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
            ridge_alphas=(1e-4, 1e-2, 1.0),
            strip_shield=False, seed=seed, split_mode=args.split,
        )
        top1s.append(res.ttrsr_top1 or 0.0)
        top10s.append(res.ttrsr_top10 or 0.0)

    a1 = np.array(top1s) * 100
    a10 = np.array(top10s) * 100
    print(f"[probe1] {args.transform} top1: mean={a1.mean():.2f}% std={a1.std():.2f}% "
          f"min={a1.min():.2f}% max={a1.max():.2f}%")
    print(f"[probe1] {args.transform} top10: mean={a10.mean():.2f}% std={a10.std():.2f}%")
    print(f"[probe1] wall={time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()

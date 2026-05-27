#!/usr/bin/env python3
"""Diagnose harness for ISA-AttnScore plain ceiling = 11.87% mystery.

Goal: a deterministic, fast feedback loop. Single seed should take <10s.
10-seed mean + std reproduces the handoff's 11.87 ± 3.44 claim.

Usage:
    python diagnose_isa_attn_score_loop.py <captures_dir> [--layer 17]
        [--kind kq] [--split row|vocab] [--strip-shield] [--seeds 10]
        [--ridge-alpha 1e-4 1e-2 1.0] [--n-train ...] [--head-slice ...]
        [--probe row|col|topk|cosine-noise|...]
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


def flatten_attn_view(attn_set: SnapshotSet, layer: int, kind: str, strip_shield: bool):
    """Mirror m2_7.run_hidden_state_attacks._isa_attn_score: build a shim
    SnapshotSet whose per_prompt_layer_kind_tensors yields (n_q, *) per prompt.
    """
    flat_pairs: list[tuple[int, torch.Tensor]] = []
    for s in attn_set.select(kind=kind, layer=layer):
        op = attn_set.get_operand(s, strip_shield=strip_shield)
        if op.ndim == 3:
            n_heads, n_q, n_kv = op.shape
            op = op.permute(1, 0, 2).reshape(n_q, n_heads * n_kv)
        elif op.ndim != 2:
            continue
        flat_pairs.append((s.prompt_idx, op))
    shim = copy.copy(attn_set)
    shim.per_prompt_layer_kind_tensors = lambda **kw: flat_pairs  # type: ignore
    return shim, flat_pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("captures_dir", type=Path)
    p.add_argument("--model-id", default="Qwen/Qwen3-4B")
    p.add_argument("--layer", type=int, default=17)
    p.add_argument("--kind", default="kq")
    p.add_argument("--split", choices=("row", "vocab"), default="row")
    p.add_argument("--strip-shield", action="store_true")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=20260518)
    p.add_argument("--n-train", type=int, default=256)
    p.add_argument("--n-val", type=int, default=64)
    p.add_argument("--n-test", type=int, default=64)
    p.add_argument("--ridge-alpha", type=float, nargs="+",
                   default=[1e-4, 1e-2, 1.0])
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--inspect", action="store_true",
                   help="Print captured tensor stats and exit (no attack run).")
    args = p.parse_args()

    print(f"[diag] opening attn captures: {args.captures_dir}")
    attn = SnapshotSet.open("attn", root=args.captures_dir)
    print(f"  n_prompts={attn.n_prompts()} layers={attn.captured_layers} "
          f"kinds={attn.captured_kinds}")

    embed_table = load_qwen3_embedding_table(args.model_id).to(torch.float32)
    print(f"  W_e shape={tuple(embed_table.shape)}  d_e={embed_table.shape[1]}")

    shim, flat_pairs = flatten_attn_view(attn, args.layer, args.kind, args.strip_shield)
    if not flat_pairs:
        print(f"!! no flat pairs at layer={args.layer} kind={args.kind}")
        return 1
    sample_op = flat_pairs[0][1]
    print(f"  flat sample: prompt={flat_pairs[0][0]}  shape={tuple(sample_op.shape)}")
    print(f"  total prompts in flat={len(flat_pairs)}  "
          f"shape0 range=({min(p.shape[0] for _,p in flat_pairs)},"
          f"{max(p.shape[0] for _,p in flat_pairs)})  "
          f"shape1 range=({min(p.shape[1] for _,p in flat_pairs)},"
          f"{max(p.shape[1] for _,p in flat_pairs)})")
    total_rows = sum(p.shape[0] for _, p in flat_pairs)
    print(f"  total ridge rows = {total_rows}")
    if args.inspect:
        # Print per-prompt token count vs operand row count
        for prompt_idx, op in flat_pairs[:5]:
            ids = attn.prompt_token_ids[prompt_idx]
            print(f"   prompt {prompt_idx}: n_rows={op.shape[0]}, n_tokens={len(ids)}, "
                  f"row_mean={float(op.mean()):.4f}, row_std={float(op.std()):.4f}")
        return 0

    top1s = []
    top10s = []
    alphas_used = []
    t0 = time.perf_counter()
    for k in range(args.seeds):
        seed = args.seed_base + k
        result = run_isa.run(
            shim, embed_table=embed_table,
            layer=args.layer, kind=args.kind,
            n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
            ridge_alphas=tuple(args.ridge_alpha),
            strip_shield=args.strip_shield,
            seed=seed, split_mode=args.split,
        )
        t1 = result.ttrsr_top1 if result.ttrsr_top1 is not None else float("nan")
        t10 = result.ttrsr_top10 if result.ttrsr_top10 is not None else float("nan")
        top1s.append(t1)
        top10s.append(t10)
        alphas_used.append(result.extra.get("best_ridge_alpha"))
        if args.verbose:
            print(f"  seed {seed}: top1={t1*100:.2f}% top10={t10*100:.2f}% "
                  f"alpha*={result.extra.get('best_ridge_alpha')} "
                  f"n_test={result.n_test} pool={result.extra.get('candidate_pool_size')}")

    arr1 = np.array(top1s) * 100
    arr10 = np.array(top10s) * 100
    print(f"[diag] {args.seeds}-seed @ layer={args.layer} kind={args.kind} "
          f"split={args.split} strip_shield={args.strip_shield}:")
    print(f"  top1  mean={arr1.mean():.2f}%  std={arr1.std():.2f}%  "
          f"min={arr1.min():.2f}%  max={arr1.max():.2f}%")
    print(f"  top10 mean={arr10.mean():.2f}%  std={arr10.std():.2f}%")
    print(f"  alphas selected: {alphas_used}")
    print(f"  wall: {time.perf_counter()-t0:.1f}s ({(time.perf_counter()-t0)/args.seeds:.1f}s/seed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

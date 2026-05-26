#!/usr/bin/env python3
"""Probe 3: per-head ridge — 32 separate fits, pick best.

If ridge over the 8192-col flatten is overcomplete and dilutes signal,
per-head ridges should expose heads that carry stronger token-identity
signal. We report:
  - per-head top-1 distribution (mean / max / 95th percentile)
  - best-head top-1 vs current flattened top-1
  - top-1 of an ensemble (mean of per-head predictions)
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


def per_head_view(attn_set, layer, kind, head_idx):
    """Return a shim that yields (n_q, n_kv) tensors per prompt — only head `head_idx`."""
    pairs = []
    for s in attn_set.select(kind=kind, layer=layer):
        op = attn_set.get_operand(s, strip_shield=False)  # (n_heads, n_q, n_kv)
        if op.ndim != 3:
            continue
        pairs.append((s.prompt_idx, op[head_idx].contiguous()))  # (n_q, n_kv)
    shim = copy.copy(attn_set)
    shim.per_prompt_layer_kind_tensors = lambda **kw: pairs
    return shim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("captures_dir", type=Path)
    p.add_argument("--model-id", default="Qwen/Qwen3-4B")
    p.add_argument("--layer", type=int, default=17)
    p.add_argument("--kind", default="kq")
    p.add_argument("--split", default="row")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=32)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--n-train", type=int, default=256)
    p.add_argument("--n-val", type=int, default=64)
    p.add_argument("--n-test", type=int, default=64)
    args = p.parse_args()

    attn = SnapshotSet.open("attn", root=args.captures_dir)
    embed = load_qwen3_embedding_table(args.model_id).to(torch.float32)

    # Sanity: peek at sample shape
    s0 = next(iter(attn.select(kind=args.kind, layer=args.layer)))
    sample = attn.get_operand(s0, strip_shield=False)
    print(f"[probe3] sample shape={tuple(sample.shape)}  layer={args.layer} kind={args.kind}")

    per_head_top1 = np.zeros((args.n_heads, args.seeds))
    per_head_top10 = np.zeros((args.n_heads, args.seeds))
    t0 = time.perf_counter()
    for h in range(args.n_heads):
        shim = per_head_view(attn, args.layer, args.kind, h)
        for k in range(args.seeds):
            seed = 20260518 + k
            res = run_isa.run(
                shim, embed_table=embed,
                layer=args.layer, kind=args.kind,
                n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
                ridge_alphas=(args.alpha,),
                strip_shield=False, seed=seed, split_mode=args.split,
            )
            per_head_top1[h, k] = res.ttrsr_top1 or 0.0
            per_head_top10[h, k] = res.ttrsr_top10 or 0.0
        print(f"  head {h:2d}: top1 mean={per_head_top1[h].mean()*100:.2f}% std={per_head_top1[h].std()*100:.2f}% "
              f"top10 mean={per_head_top10[h].mean()*100:.2f}%", flush=True)

    means_t1 = per_head_top1.mean(axis=1) * 100  # (n_heads,)
    means_t10 = per_head_top10.mean(axis=1) * 100
    print(f"\n[probe3] summary (layer={args.layer}):")
    print(f"  per-head top-1: min={means_t1.min():.2f}  median={np.median(means_t1):.2f}  "
          f"mean={means_t1.mean():.2f}  p90={np.percentile(means_t1, 90):.2f}  max={means_t1.max():.2f}")
    print(f"  per-head top-10: min={means_t10.min():.2f}  median={np.median(means_t10):.2f}  "
          f"mean={means_t10.mean():.2f}  max={means_t10.max():.2f}")
    print(f"  best head idx: {int(np.argmax(means_t1))} (top1={means_t1.max():.2f}%)")
    print(f"  wall: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()

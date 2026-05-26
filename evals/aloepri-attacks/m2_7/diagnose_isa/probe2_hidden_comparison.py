#!/usr/bin/env python3
"""H1: ISA HiddenState @ same layer for direct comparison.

If attention-score is intrinsically weaker than hidden-state, expect
HiddenState top-1 ≫ AttnScore top-1 on the same prompts + same split
+ same attacker. If both hit ~12%, the bottleneck is corpus/attacker,
not surface.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path("/home/timo/repos/private-rag-path-2")
sys.path.insert(0, str(REPO / "evals" / "aloepri-attacks"))
from attack_drivers import run_isa  # type: ignore
from attack_drivers.run_ima import load_qwen3_embedding_table  # type: ignore
from snapshots_loader import SnapshotSet  # type: ignore


def main():
    p = argparse.ArgumentParser()
    p.add_argument("captures_dir", type=Path)
    p.add_argument("--model-id", default="Qwen/Qwen3-4B")
    p.add_argument("--layer", type=int, default=17)
    p.add_argument("--split", default="row")
    p.add_argument("--seeds", type=int, default=10)
    args = p.parse_args()

    hidden = SnapshotSet.open("hidden", root=args.captures_dir)
    embed = load_qwen3_embedding_table(args.model_id).to(torch.float32)
    print(f"[probe2] HiddenState @ L={args.layer} (attn_norm) split={args.split}")

    top1s, top10s = [], []
    t0 = time.perf_counter()
    for k in range(args.seeds):
        seed = 20260518 + k
        res = run_isa.run(
            hidden, embed_table=embed,
            layer=args.layer, kind="attn_norm",
            n_train=256, n_val=64, n_test=64,
            ridge_alphas=(1e-4, 1e-2, 1.0),
            strip_shield=False, seed=seed, split_mode=args.split,
        )
        top1s.append(res.ttrsr_top1 or 0.0)
        top10s.append(res.ttrsr_top10 or 0.0)

    a1 = np.array(top1s) * 100
    a10 = np.array(top10s) * 100
    print(f"  top1:  mean={a1.mean():.2f}% std={a1.std():.2f}% min={a1.min():.2f}% max={a1.max():.2f}%")
    print(f"  top10: mean={a10.mean():.2f}% std={a10.std():.2f}%")
    print(f"  wall: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()

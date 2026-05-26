#!/usr/bin/env python3
"""Probe 5: row-split train/test vocab overlap + per-prompt repetition.

Asks: if row-split memorization is the upper bound, what fraction of test
plain_ids are in train vocab? And how repeated are individual tokens
across positions (high repetition = ridge can memorize position-vocab map)?
"""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path("/home/timo/repos/private-rag-path-2")
sys.path.insert(0, str(REPO / "evals" / "aloepri-attacks"))
from snapshots_loader import SnapshotSet  # type: ignore


def main():
    import sys as _sys
    if len(_sys.argv) > 1:
        cells = [Path(p) for p in _sys.argv[1:]]
    else:
        cells = [
            REPO / "evals/aloepri-attacks/results/sweep/cell-qwen3-4b-plain-attn-multilayer-20260525/captures",
            REPO / "evals/aloepri-attacks/results/sweep/cell-qwen3-4b-uvo-attn-multilayer-20260525/captures",
        ]
    for cell in cells:
        attn = SnapshotSet.open("attn", root=cell)
        # Gather all (prompt_idx, token_id, position) rows
        all_ids = []
        for prompt_idx in range(attn.n_prompts()):
            ids = attn.prompt_token_ids[prompt_idx]
            all_ids.extend(ids)
        all_ids = np.array(all_ids)
        unique, counts = np.unique(all_ids, return_counts=True)
        print(f"\n=== {cell.parent.name} ===")
        print(f"  total tokens: {len(all_ids)}")
        print(f"  unique plain_ids: {len(unique)}")
        print(f"  distribution: top 10 repeats = {sorted(counts, reverse=True)[:10]}")
        print(f"  median count = {np.median(counts):.0f}, mean = {counts.mean():.2f}, max = {counts.max()}")
        # Simulate row-split: random 50/25/25 over positions
        for split_seed in [20260518, 20260519, 20260520]:
            rng = np.random.default_rng(split_seed + 17)
            n = len(all_ids)
            perm = rng.permutation(n)
            n_tr, n_va = int(0.5 * n), int(0.25 * n)
            tr_ids = set(all_ids[perm[:n_tr]].tolist())
            te_ids_arr = all_ids[perm[n_tr + n_va:]]
            te_in_tr = sum(int(i) in tr_ids for i in te_ids_arr)
            print(f"  split_seed={split_seed}: train_unique={len(tr_ids)}, "
                  f"test_rows={len(te_ids_arr)}, test_in_train_vocab={te_in_tr} "
                  f"({100*te_in_tr/len(te_ids_arr):.1f}%)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare plain vs obf kq captures element-wise.

If element-wise identical -> capture bug (patched llama-server emits
something pre-obf or shared). If statistically similar but element-wise
different -> true K_a-invariant surface.

Usage:
    python compare_plain_obf.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/timo/repos/private-rag-path-2")
sys.path.insert(0, str(REPO / "evals" / "aloepri-attacks"))
from snapshots_loader import SnapshotSet  # type: ignore

PLAIN = REPO / "evals/aloepri-attacks/results/sweep/cell-qwen3-4b-plain-attn-512-20260526/captures"
OBF = REPO / "evals/aloepri-attacks/results/sweep/cell-qwen3-4b-uvo-attn-512-20260526/captures"


def gather(root, layer, kind):
    s = SnapshotSet.open("attn", root=root)
    pairs = []
    for snap in s.select(kind=kind, layer=layer):
        op = s.get_operand(snap, strip_shield=False)
        pairs.append((snap.prompt_idx, op.detach().cpu().numpy()))
    pairs.sort(key=lambda x: x[0])
    return pairs, s


for L in [0, 17]:
    print(f"\n===== layer {L}, kind=kq =====")
    p_pairs, p_set = gather(PLAIN, L, "kq")
    o_pairs, o_set = gather(OBF, L, "kq")
    print(f"plain n={len(p_pairs)}  obf n={len(o_pairs)}")
    # Compare first few prompts
    for i in range(min(3, len(p_pairs), len(o_pairs))):
        pi, pa = p_pairs[i]
        oi, oa = o_pairs[i]
        assert pi == oi, (pi, oi)
        # Stats
        same_shape = pa.shape == oa.shape
        identical = same_shape and np.array_equal(pa, oa)
        close = same_shape and np.allclose(pa, oa, atol=1e-4)
        diff = pa - oa if same_shape else None
        print(f" prompt {pi}: plain_shape={pa.shape} obf_shape={oa.shape}")
        print(f"   identical={identical}  close(atol=1e-4)={close}")
        if same_shape:
            print(f"   plain stats: mean={pa.mean():.4f} std={pa.std():.4f} min={pa.min():.4f} max={pa.max():.4f}")
            print(f"   obf   stats: mean={oa.mean():.4f} std={oa.std():.4f} min={oa.min():.4f} max={oa.max():.4f}")
            print(f"   diff  stats: mean={diff.mean():.4f} std={diff.std():.4f} "
                  f"max_abs={np.abs(diff).max():.4f} median_abs={np.median(np.abs(diff)):.4f}")
            print(f"   correlation(flat) = {np.corrcoef(pa.flatten(), oa.flatten())[0,1]:.6f}")

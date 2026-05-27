#!/usr/bin/env python3
"""Run ISA-AttnScore on POST-SOFTMAX attention probabilities (paper's surface).

Our kq capture is pre-softmax Q·K^T. To simulate paper's `outputs.attentions[L]`:
  1. Divide by sqrt(d_head) — typical attention pre-scale
  2. Apply causal mask (set positions j > i to -inf)
  3. Softmax over the KV axis
  4. Flatten + ridge (same as before)

If recovery DROPS significantly vs pre-softmax, paper's 0% claim is mostly
"post-softmax compresses range + causal mask zeros future" rather than
defense. If recovery stays at ~48%, defense is genuinely zero on both
surfaces under our threat model.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path("/home/timo/repos/private-rag-path-2")
sys.path.insert(0, str(REPO / "evals" / "aloepri-attacks"))
sys.path.insert(0, str(REPO / "evals" / "aloepri-attacks" / "m2_7" / "diagnose_isa"))
from snapshots_loader import SnapshotSet  # type: ignore
from attack_drivers.run_ima import load_qwen3_embedding_table  # type: ignore
from gpu_sweep import fit_ridge_gpu, predict_ridge, cosine_topk, run_one  # type: ignore


def load_post_softmax(captures_dir: Path, layer: int, d_head: int = 128):
    """Load kq, apply scale + causal mask + softmax, flatten."""
    attn = SnapshotSet.open("attn", root=captures_dir)
    Xs, ys = [], []
    scale = 1.0 / (d_head ** 0.5)
    for s in attn.select(kind="kq", layer=layer):
        op = attn.get_operand(s, strip_shield=False).to(torch.float32)  # (n_heads, n_q, n_kv)
        if op.ndim != 3:
            continue
        n_heads, n_q, n_kv = op.shape
        ids = attn.prompt_token_ids[s.prompt_idx]
        n_valid = min(n_q, len(ids))
        op = op[:, :n_valid, :]  # (n_heads, n_valid, n_kv)
        # Scale
        op = op * scale
        # Causal mask: position i can only attend to positions 0..i. n_kv >= n_valid,
        # so mask out columns j > i AND columns past n_valid.
        mask = torch.full((n_valid, n_kv), float("-inf"))
        for i in range(n_valid):
            mask[i, :i + 1] = 0.0
        op = op + mask.unsqueeze(0)  # broadcast over heads
        # Softmax over kv axis
        op = torch.softmax(op, dim=-1)  # (n_heads, n_valid, n_kv)
        # Flatten: (n_valid, n_heads * n_kv)
        feat = op.permute(1, 0, 2).reshape(n_valid, -1)
        Xs.append(feat)
        ys.append(torch.tensor(ids[:n_valid], dtype=torch.long))
    if not Xs:
        return None, None
    return torch.cat(Xs, dim=0).to(torch.float32), torch.cat(ys, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plain", type=Path, required=True)
    p.add_argument("--obf", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3-4B")
    p.add_argument("--layers", type=int, nargs="+", default=[0, 17])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--d-head", type=int, default=128)
    args = p.parse_args()

    assert torch.cuda.is_available()
    dev = "cuda"
    print(f"[postsoftmax] device={dev} name={torch.cuda.get_device_name(0)}")
    embed = load_qwen3_embedding_table(args.model_id).to(torch.float32).to(dev)
    print(f"[postsoftmax] embed shape={tuple(embed.shape)}")

    print(f"\n===== POST-SOFTMAX attention probabilities (paper's surface) =====")
    print(f"{'layer':>5} | {'cell':>5} | {'top1 mean ± std':>20} | {'top10':>10}")
    for L in args.layers:
        for label, cap in [("PLAIN", args.plain), ("OBF", args.obf)]:
            X, y = load_post_softmax(cap, L, d_head=args.d_head)
            if X is None:
                print(f"  L={L} {label}: no captures"); continue
            t0 = time.perf_counter()
            t1s, t10s = [], []
            for k in range(args.seeds):
                r = run_one(X, y, embed, seed=20260518 + k, split="row")
                if r[0] is None: continue
                t1s.append(r[0]); t10s.append(r[1])
            a1 = np.array(t1s) * 100; a10 = np.array(t10s) * 100
            print(f"  L={L:>3} | {label:>5} | {a1.mean():>6.2f}% ± {a1.std():>5.2f}  "
                  f"(n={len(t1s)}, X={tuple(X.shape)}) | {a10.mean():>5.2f}%  "
                  f"wall {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()

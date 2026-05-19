"""Sweep `run_ima_paper_like` over increasing prompt counts to find the
minimal viable corpus size — the smallest N where a 2-layer transformer
inverter trained on a plain-text capture starts producing meaningful
TTRSR (well above 15 % paper gate, ideally ≥ 50 %).

Usage:

  python sweep_ima_paper_like.py \\
      --captures-dir evals/aloepri-attacks/snapshots/m2_7-plain-512 \\
      --output evals/aloepri-attacks/results/m2_7-ima-paper-like-sweep-plain.json \\
      --ns 64,128,192,256,384,512 \\
      --epochs 4
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attack_drivers import run_ima_paper_like  # type: ignore  # noqa: E402
from attack_drivers.run_ima import load_qwen3_embedding_table  # type: ignore  # noqa: E402
from snapshots_loader import SnapshotSet  # type: ignore  # noqa: E402


def truncate_snapshots(snaps: SnapshotSet, n: int) -> SnapshotSet:
    """Return a shallow copy of `snaps` that only exposes the first
    `n` prompts."""
    shim = copy.copy(snaps)
    shim.prompt_token_ids = snaps.prompt_token_ids[:n]
    shim.snapshots = [s for s in snaps.snapshots if s.prompt_idx < n]

    full_pairs = snaps.per_prompt_layer_kind_tensors

    def filtered(*, layer: int, kind: str, strip_shield: bool = True):
        return [
            (p, t) for p, t in full_pairs(
                layer=layer, kind=kind, strip_shield=strip_shield
            )
            if p < n
        ]

    shim.per_prompt_layer_kind_tensors = filtered  # type: ignore[method-assign]
    shim.n_prompts = lambda: n  # type: ignore[method-assign]
    return shim


def main() -> int:
    p = argparse.ArgumentParser(description="IMA paper-like prompt-count sweep")
    p.add_argument("--captures-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--kind", default="attn_norm")
    p.add_argument("--ns", default="64,128,192,256,384,512",
                   help="Comma-separated list of corpus sizes to sweep")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--inverter-hidden", type=int, default=256)
    p.add_argument("--seed", type=int, default=20260518)
    args = p.parse_args()

    ns = [int(x) for x in args.ns.split(",") if x.strip()]

    print(f"[sweep] loading snapshots from {args.captures_dir}")
    snaps_full = SnapshotSet.open("hidden", root=args.captures_dir)
    print(f"[sweep] loaded {snaps_full.n_prompts()} prompts; running sweep over {ns}")

    embed_table = load_qwen3_embedding_table(args.model_id)

    rows = []
    for n in ns:
        if n > snaps_full.n_prompts():
            print(f"[sweep] skipping N={n} — only {snaps_full.n_prompts()} prompts captured")
            continue
        view = truncate_snapshots(snaps_full, n)
        t0 = time.perf_counter()
        torch.manual_seed(args.seed)
        result = run_ima_paper_like.run(
            view,
            embed_table=embed_table,
            layer=args.layer,
            kind=args.kind,
            strip_shield=False,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            inverter_hidden=args.inverter_hidden,
            seed=args.seed,
        )
        dt = time.perf_counter() - t0
        rec = {
            "n_prompts": n,
            "top1": result.ttrsr_top1,
            "top10": result.ttrsr_top10,
            "n_train_rows": result.n_train,
            "n_test_rows": result.n_test,
            "final_train_loss": result.extra.get("final_train_loss"),
            "wall_s": round(dt, 2),
            "risk_level": result.risk_level,
        }
        rows.append(rec)
        print(
            f"  N={n:4d} | top1={rec['top1']:.4f} | top10={rec['top10']:.4f} "
            f"| loss={rec['final_train_loss']:.3f} "
            f"| rows train={rec['n_train_rows']}, test={rec['n_test_rows']} "
            f"| {dt:.1f}s"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "format": "aloepri_m2_7_ima_paper_like_sweep_v1",
        "captures_dir": str(args.captures_dir),
        "model_id": args.model_id,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "inverter_hidden": args.inverter_hidden,
        "seed": args.seed,
        "sweep": rows,
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"[sweep] wrote → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

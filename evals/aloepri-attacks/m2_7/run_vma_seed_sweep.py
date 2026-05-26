"""VMA 3-seed sweep — mirrors the Q3-4B 3-seed measurement methodology.

Calls `run_vma` (from `run_static_attacks`) with seeds {20260518,
20260519, 20260520} (sequential from the project default 20260518) and
emits per-seed top-1/top-10 + mean/std summary JSON.

Same defaults as `run_static_attacks.py`: eval_size=256, candidate_pool_size=4096, bins=64.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from extract_gguf_weights import load_model  # type: ignore  # noqa: E402
    from run_static_attacks import run_vma  # type: ignore  # noqa: E402

    p = argparse.ArgumentParser(description="VMA 3-seed sweep on a §05 obfuscated GGUF")
    p.add_argument("--plain", type=Path, required=True)
    p.add_argument("--obfuscated", type=Path, required=True)
    p.add_argument(
        "--key", type=Path,
        help=".key.npz with τ. Required unless --identity-tau.",
    )
    p.add_argument("--identity-tau", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[20260518, 20260519, 20260520])
    p.add_argument("--eval-size", type=int, default=256)
    p.add_argument("--candidate-pool-size", type=int, default=4096)
    p.add_argument("--bins", type=int, default=64)
    args = p.parse_args()

    if args.identity_tau and args.key is not None:
        raise SystemExit("pass --key or --identity-tau, not both")
    if not args.identity_tau and args.key is None:
        raise SystemExit("--key required unless --identity-tau")

    print(f"[VMA-sweep] loading plaintext {args.plain}")
    t0 = time.perf_counter()
    plain = load_model(args.plain, "plaintext")
    print(f"  loaded in {time.perf_counter() - t0:.1f} s")

    print(f"[VMA-sweep] loading obfuscated {args.obfuscated}")
    t0 = time.perf_counter()
    obfuscated = load_model(args.obfuscated, "obfuscated")
    print(f"  loaded in {time.perf_counter() - t0:.1f} s")

    tau: Optional[np.ndarray]
    if args.identity_tau:
        tau = None
    else:
        z = np.load(args.key, allow_pickle=False)
        tau = z["tau"].astype(np.int64)

    per_seed = []
    for seed in args.seeds:
        t0 = time.perf_counter()
        r = run_vma(
            plain, obfuscated, tau=tau,
            eval_size=args.eval_size,
            candidate_pool_size=args.candidate_pool_size,
            bins=args.bins,
            seed=int(seed),
        )
        dt = time.perf_counter() - t0
        print(f"  seed={seed:>10d}  t1={r.ttrsr_top1:.4f}  t10={r.ttrsr_top10:.4f}  risk={r.risk_level}  ({dt:.1f}s)")
        per_seed.append({
            "seed": int(seed),
            "ttrsr_top1": float(r.ttrsr_top1),
            "ttrsr_top10": float(r.ttrsr_top10),
            "risk_level": r.risk_level,
            "elapsed_s": float(dt),
        })

    t1 = np.array([s["ttrsr_top1"] for s in per_seed])
    t10 = np.array([s["ttrsr_top10"] for s in per_seed])
    summary = {
        "attack": "vma_seed_sweep",
        "plain": str(args.plain),
        "obfuscated": str(args.obfuscated),
        "key": (str(args.key) if args.key else "identity_tau"),
        "n_seeds": len(args.seeds),
        "eval_size": args.eval_size,
        "candidate_pool_size": args.candidate_pool_size,
        "bins": args.bins,
        "per_seed": per_seed,
        "top1_mean": float(t1.mean()),
        "top1_std": float(t1.std(ddof=1) if len(t1) > 1 else 0.0),
        "top10_mean": float(t10.mean()),
        "top10_std": float(t10.std(ddof=1) if len(t10) > 1 else 0.0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"[VMA-sweep] top1 = {summary['top1_mean']*100:.2f} ± {summary['top1_std']*100:.2f} % "
          f"(n={summary['n_seeds']})")
    print(f"[VMA-sweep] wrote → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Adversarial K_d selection — picks an Algorithm-1 init_seed that yields
a K_d expected to be unlucky for a paper-faithful K=64 multi-key ridge
attacker.

Background
==========
c1 luckiness-signature probe (2026-05-22, N=10 pool seeds at d=2560, h=128,
K=64) found that the per-pool feature

    top_sv_overlap_r{H}.mean(K_a_pool, K_d) =
        mean_k  Σ σ²( Vh_a_top[k] · Vh_d_top.T )

— the sum of squared singular values of the cross-matrix between K_a^k's
top-H right-singular subspace and K_d's top-H right-singular subspace —
correlates with attacker TTRSR at Spearman ρ = −0.78 (Pearson r = −0.74).

Sign is negative: **higher overlap ⇒ lower TTRSR**. The c1 mechanism reading
is that lucky attackers come from K_a^k draws whose top-h subspace is
mis-aligned with K_d's, leaving the ridge regression more room to find
near-null directions. Picking K_d at the HIGH end of this score forces the
attacker to operate in a sub-optimally-aligned subspace.

Recipe (~5 min for N_kd = 64 candidates at d=2560, h=256, K=64)
================================================================
1. Build a deterministic *reference K_a pool* of K matrices via the
   vendor's paper-faithful builder (`build_keymat_transform`). The pool
   is fixed across all candidate evaluations (cached). The cache key is
   (d, h, λ, ref_pool_seed_base, K).

2. SVD the stacked reference pool once; cache the top-H right-singular-
   vector slabs `Vh_a_top ∈ ℝ^{K × H × (d+2h)}`.

3. For each candidate K_d seed in {candidate_seed_base + i}_{i=0..N-1}:
   - Build K_d via `build_keymat_transform(d, h, lam, init_seed=seed)`.
   - SVD K_d, take `Vh_d_top ∈ ℝ^{H × (d+2h)}`.
   - Compute the K cross-products `Vh_a_top[k] @ Vh_d_top.T ∈ ℝ^{H × H}`.
   - Score per k: sum of squared singular values of that cross-matrix
     (∈ [0, H], higher = more aligned subspaces).
   - Aggregate per candidate: pool mean across k.

4. Pick `argmax` score among candidates — this is the K_d an attacker is
   expected to be least lucky against.

5. Print the winning init_seed to stdout (single int, no extra noise) so
   the result can be piped:
       SEED=$(python select_adversarial_kd.py --h 256 ...)
       python obfuscate_qwen3_gguf.py --seed $SEED ...

Outputs
=======
- stdout: a single integer — the winning K_d init_seed.
- stderr: human-readable log + per-candidate score table.
- --report <path>: optional JSON dump of {seed: score} for all candidates.

Statistical effectiveness (point estimate)
==========================================
For N_kd = 64 candidates, scoring with ρ = −0.78 against the reference
attacker, the expected attacker TTRSR at the picked K_d is approximately

    E[TTRSR | adversarial] ≈ μ_TTRSR - |ρ| · σ_TTRSR · E[max of 64 std-normals]
                          ≈ 3.19 % - 0.78 · 2.12 pp · 1.85
                          ≈ 0.13 % at h=128 baseline
                          ≈ −0.51 % at h=256 (clamped at 0 %)

i.e., picking the best of 64 candidates is expected to drive the *mean*
attacker TTRSR essentially to zero, leaving only σ_pool (the 1.37-pp std
at h=256) as the residual variance source. Adversarial selection does
NOT reduce σ_pool — only the mean. To compress σ_pool, combine with
the h-bump (already in the default of obfuscate_qwen3_gguf.py at h=256).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vendor" / "aloepri-py"))
sys.path.insert(0, str(REPO / "vendor" / "aloepri-py" / "src"))


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _cache_key(d: int, h: int, lam: float, ref_pool_seed_base: int, K: int) -> str:
    raw = f"d={d}|h={h}|lam={lam}|base={ref_pool_seed_base}|K={K}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def build_reference_pool(d: int, h: int, lam: float, ref_pool_seed_base: int,
                         K: int, device: str) -> torch.Tensor:
    """Vendor-paper-faithful K-pool builder. Uses the same seed schedule
    as run_isa_multikey.py's _build_attacker_keymat_pool_vendor:

        K_a^k init_seed = ref_pool_seed_base + 1 + 10_000 · k
    """
    from keymat import build_keymat_transform  # type: ignore  # noqa: E402

    d_obs = d + 2 * h
    pool = torch.empty((K, d, d_obs), dtype=torch.float32, device=device)
    t0 = time.perf_counter()
    for k in range(K):
        init_seed = ref_pool_seed_base + 1 + 10_000 * k
        transform = build_keymat_transform(d=d, h=h, lam=lam, init_seed=init_seed)
        pool[k] = transform.key.to(torch.float32).to(device)
    _log(f"[ref-pool] built K={K} K_a^k (shape {tuple(pool.shape)}) in "
         f"{time.perf_counter() - t0:.1f}s on {device}")
    return pool


def compute_vh_a_top(pool: torch.Tensor, H: int) -> torch.Tensor:
    """Batched SVD on the reference pool → top-H right-singular vectors.
    Returns Vh_a_top ∈ ℝ^{K, H, d_obs} on the same device."""
    t0 = time.perf_counter()
    _, _, Vh_a = torch.linalg.svd(pool, full_matrices=False)  # (K, d, d_obs)
    Vh_a_top = Vh_a[:, :H, :].contiguous()  # (K, H, d_obs)
    if pool.device.type == "cuda":
        torch.cuda.synchronize()
    _log(f"[ref-pool] batched SVD + slice Vh_a_top (shape "
         f"{tuple(Vh_a_top.shape)}) in {time.perf_counter() - t0:.1f}s")
    return Vh_a_top


def score_candidate(d: int, h: int, lam: float, seed: int,
                    Vh_a_top: torch.Tensor) -> float:
    """Build K_d at the candidate seed and compute pool-mean
    top_sv_overlap_r{H} score. Returns a scalar in [0, H]."""
    from keymat import build_keymat_transform  # type: ignore  # noqa: E402

    device = Vh_a_top.device
    transform = build_keymat_transform(d=d, h=h, lam=lam, init_seed=seed)
    K_d = transform.key.to(torch.float32).to(device)  # (d, d_obs)
    _, _, Vh_d = torch.linalg.svd(K_d, full_matrices=False)  # (d, d_obs)
    Vh_d_top = Vh_d[:h, :]  # (H, d_obs)
    # Cross-matrices: (K, H, d_obs) @ (d_obs, H) → (K, H, H)
    cross = torch.einsum("khj,gj->khg", Vh_a_top, Vh_d_top)  # (K, H, H)
    sv = torch.linalg.svdvals(cross)  # (K, H)
    per_k_score = (sv ** 2).sum(dim=1)  # (K,) — ∈ [0, H] per k
    return float(per_k_score.mean().item())


def main() -> int:
    p = argparse.ArgumentParser(
        description="Pick a K_d init_seed adversarially against a "
                    "paper-faithful K-pool attacker.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--d", type=int, required=True,
                   help="model hidden size (e.g. 2560 for Qwen3-4B, 4096 for Qwen3-8B)")
    p.add_argument("--h", type=int, default=256,
                   help="expansion size; defaults to 256 (deployment default 2026-05-22)")
    p.add_argument("--lam", type=float, default=0.3,
                   help="Algorithm 1 V-noise weight (default 0.3)")
    p.add_argument("--num-candidates", type=int, default=64,
                   help="number of K_d candidates to score (default 64)")
    p.add_argument("--candidate-seed-base", type=int, default=1_000_000,
                   help="base seed for candidate K_d's; candidate i uses "
                        "seed = base + i (default 1_000_000)")
    p.add_argument("--num-ref-keys", type=int, default=64,
                   help="K — size of reference attacker pool (default 64, "
                        "matches paper-faithful multi-key ISA driver)")
    p.add_argument("--ref-pool-seed-base", type=int, default=0,
                   help="base seed for reference K_a pool; K_a^k uses "
                        "seed = base + 1 + 10_000·k (default 0 → seeds 1, 10001, "
                        "20001, ...; matches run_isa_multikey.py convention)")
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"),
                   help="device for batched SVD (default auto: cuda if available)")
    p.add_argument("--report", type=Path, default=None,
                   help="optional path to dump per-candidate scores as JSON")
    args = p.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    _log(f"[config] d={args.d} h={args.h} lam={args.lam} "
         f"K={args.num_ref_keys} N_kd={args.num_candidates} device={device}")

    t_total = time.perf_counter()
    # Step 1+2: reference pool + Vh_a_top (one-time)
    pool = build_reference_pool(
        d=args.d, h=args.h, lam=args.lam,
        ref_pool_seed_base=args.ref_pool_seed_base,
        K=args.num_ref_keys, device=device,
    )
    Vh_a_top = compute_vh_a_top(pool, H=args.h)
    del pool  # release the big stack
    if device == "cuda":
        torch.cuda.empty_cache()

    # Step 3+4: score each candidate
    _log(f"[score] iterating {args.num_candidates} K_d candidates...")
    scores: dict[int, float] = {}
    t_score = time.perf_counter()
    for i in range(args.num_candidates):
        seed = args.candidate_seed_base + i
        s = score_candidate(d=args.d, h=args.h, lam=args.lam,
                             seed=seed, Vh_a_top=Vh_a_top)
        scores[seed] = s
        if (i + 1) % 8 == 0 or i == args.num_candidates - 1:
            _log(f"[score]   {i+1:>3}/{args.num_candidates} scored "
                 f"(last seed={seed} score={s:.4f}; elapsed "
                 f"{time.perf_counter() - t_score:.1f}s)")

    # Pick argmax — high score ⇒ K_d less-lucky for attacker
    winning_seed = max(scores, key=scores.get)
    winning_score = scores[winning_seed]
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    _log(f"\n[result] winning seed = {winning_seed}  score = {winning_score:.4f}")
    _log(f"[result] top-5 candidates by score:")
    for seed, score in sorted_scores[:5]:
        _log(f"[result]   seed={seed:<10}  score={score:.4f}")
    _log(f"[result] bottom-5 (would be lucky attackers):")
    for seed, score in sorted_scores[-5:]:
        _log(f"[result]   seed={seed:<10}  score={score:.4f}")

    score_arr = np.array(list(scores.values()))
    _log(f"[result] score distribution: mean={score_arr.mean():.4f} "
         f"std={score_arr.std():.4f} min={score_arr.min():.4f} "
         f"max={score_arr.max():.4f} max-mean={score_arr.max() - score_arr.mean():+.4f} "
         f"({(score_arr.max() - score_arr.mean()) / score_arr.std():+.2f}σ)")

    _log(f"\n[total] {time.perf_counter() - t_total:.1f}s")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "config": {
                "d": args.d, "h": args.h, "lam": args.lam,
                "num_ref_keys": args.num_ref_keys,
                "ref_pool_seed_base": args.ref_pool_seed_base,
                "num_candidates": args.num_candidates,
                "candidate_seed_base": args.candidate_seed_base,
                "ref_cache_key": _cache_key(
                    args.d, args.h, args.lam, args.ref_pool_seed_base, args.num_ref_keys),
                "device": device,
            },
            "winning_seed": int(winning_seed),
            "winning_score": float(winning_score),
            "all_scores": {str(s): float(v) for s, v in scores.items()},
            "score_distribution": {
                "mean": float(score_arr.mean()),
                "std": float(score_arr.std()),
                "min": float(score_arr.min()),
                "max": float(score_arr.max()),
            },
        }, indent=2))
        _log(f"[result] wrote report → {args.report}")

    # Stdout-clean: single integer, suitable for shell piping.
    print(winning_seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())

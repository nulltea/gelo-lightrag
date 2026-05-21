#!/usr/bin/env python3
"""
Option B pre-flight (see docs/handoffs/2026-05-19-alg2-qwen3-shape-analysis.md):

For each Qwen3 attention block, the QK-norm site sits between W_q/W_k and
RoPE+dot-product. Algorithm 2's intra-head transforms
    M_q = R̂_qk · Ĥ_qk · Ẑ_block
need to commute with Diag(γ_q) (and similarly γ_k).

  - Ĥ_qk  (diagonal scaling)        : commutes with any Diag, always free.
  - R̂_qk  (2-D rotation per NEOX
            pair (i, i+d_h/2))      : commutes iff |γ[i] − γ[i+d_h/2]| < ε.
  - Ẑ_block (permutation among pairs): commutes iff all permuted pairs lie
                                       in the same γ-band of width ε.

This script measures the γ-iso-tonic structure across all 28 layers × {q, k}
of Qwen3-1.7B and reports:

  (a) Pair-symmetry fraction per layer  : # pairs with residual < ε.
  (b) Pair-mean clustering at ε         : ε-banding the sorted pair means.
      Reports # positions of head_dim in clusters of size ≥ 8.

Decision rule from the handoff:
  "If clusters of size ≥ 8 cover ≥ 50 % of head_dim positions across all
   28 × 16 (read: 28 layers × {q, k}) vectors, Option B is viable."

Usage:
  python measure_gamma_qk_clusters.py [--gguf PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import gguf

REPO_ROOT = Path(__file__).resolve().parents[2].parent  # private-rag-path-2/
sys.path.insert(0, str(REPO_ROOT / "python" / "path-2"))

# Reuse the obfuscator's dequant helper so we read fp32 even from Q8_0 norms.
from obfuscate_qwen3_gguf import to_float_array  # noqa: E402

DEFAULT_GGUF = (
    Path.home()
    / ".cache/huggingface/hub/models--bartowski--Qwen_Qwen3-1.7B-GGUF"
    / "snapshots/dcb19155b962dbb6389f4691a982043a8e651022"
    / "Qwen_Qwen3-1.7B-Q8_0.gguf"
)
EPSILONS = (0.01, 0.05, 0.10, 0.25)
MIN_USEFUL_CLUSTER = 8  # paper-relevant Ẑ_block group size


def cluster_by_eps(values: np.ndarray, eps: float) -> list[int]:
    """1-D agglomerative ε-banding. Returns list of cluster sizes.

    Sort values, start a new cluster whenever the gap to the previous
    sorted value exceeds ε. This gives the *coarsest* partition for which
    every cluster has max-min ≤ k·ε; for the Ẑ_block check we'd ideally
    want max-min ≤ ε, but ε-gap clustering is the natural single-pass
    approximation. Tightened version below filters by intra-cluster span.
    """
    if values.size == 0:
        return []
    s = np.sort(values)
    sizes: list[int] = []
    start = 0
    for i in range(1, s.size):
        if s[i] - s[i - 1] > eps:
            sizes.append(i - start)
            start = i
    sizes.append(s.size - start)
    return sizes


def cluster_by_eps_strict(values: np.ndarray, eps: float) -> list[int]:
    """Strict ε-banding: every cluster has span ≤ ε.

    Walk sorted values; close a cluster when adding the next value would
    blow span past ε. This is the right notion for "all permuted pairs
    lie in a γ-band of width ε".
    """
    if values.size == 0:
        return []
    s = np.sort(values)
    sizes: list[int] = []
    start = 0
    for i in range(1, s.size):
        if s[i] - s[start] > eps:
            sizes.append(i - start)
            start = i
    sizes.append(s.size - start)
    return sizes


def analyse_vector(gamma: np.ndarray, head_dim: int) -> dict:
    """Per-(layer, q|k) analysis.

    Returns dict with:
      - pair_resid_max, pair_resid_mean
      - pair_symm_pct[eps]              : % of (d_h/2) pairs symmetric < eps
      - cov_pct[eps]                    : % of head_dim positions inside a
                                          γ-band-of-width-eps cluster of
                                          size ≥ MIN_USEFUL_CLUSTER
                                          (counted in PAIRS first, then
                                           ×2 because each pair = 2 positions)
    """
    assert gamma.shape == (head_dim,), f"expected ({head_dim},) got {gamma.shape}"
    half = head_dim // 2
    pair_resid = np.abs(gamma[:half] - gamma[half:])
    pair_mean = 0.5 * (gamma[:half] + gamma[half:])

    out: dict = {
        "gamma_min": float(gamma.min()),
        "gamma_max": float(gamma.max()),
        "gamma_median": float(np.median(gamma)),
        "gamma_std": float(gamma.std()),
        "pair_resid_max": float(pair_resid.max()),
        "pair_resid_mean": float(pair_resid.mean()),
        "pair_symm_pct": {},
        "cov_pct": {},
    }

    for eps in EPSILONS:
        n_symm = int((pair_resid < eps).sum())
        out["pair_symm_pct"][eps] = 100.0 * n_symm / half

        # Only γ-symmetric pairs are eligible for Ẑ_block permutation.
        # Cluster their pair-means.
        eligible_means = pair_mean[pair_resid < eps]
        sizes_strict = cluster_by_eps_strict(eligible_means, eps)
        big_cluster_pairs = sum(sz for sz in sizes_strict if sz >= MIN_USEFUL_CLUSTER)
        # Each big-cluster pair contributes 2 head_dim positions.
        out["cov_pct"][eps] = 100.0 * (2 * big_cluster_pairs) / head_dim

    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    args = p.parse_args()

    if not args.gguf.exists():
        print(f"GGUF not found: {args.gguf}", file=sys.stderr)
        return 2

    print(f"loading {args.gguf}")
    r = gguf.GGUFReader(str(args.gguf))
    arch = r.fields["general.architecture"].contents()
    if arch != "qwen3":
        print(f"unsupported architecture: {arch}", file=sys.stderr)
        return 2

    n_layer = int(r.fields["qwen3.block_count"].contents())
    head_dim = int(r.fields["qwen3.attention.key_length"].contents())
    n_q_heads = int(r.fields["qwen3.attention.head_count"].contents())
    n_kv_heads = int(r.fields["qwen3.attention.head_count_kv"].contents())
    print(f"arch=qwen3 n_layer={n_layer} head_dim={head_dim} "
          f"n_q={n_q_heads} n_kv={n_kv_heads}")

    by_name = {t.name: t for t in r.tensors}

    rows: list[tuple[int, str, dict]] = []
    for il in range(n_layer):
        for site in ("attn_q_norm", "attn_k_norm"):
            tname = f"blk.{il}.{site}.weight"
            if tname not in by_name:
                print(f"missing: {tname}", file=sys.stderr)
                continue
            gamma = to_float_array(by_name[tname]).astype(np.float64)
            stats = analyse_vector(gamma, head_dim)
            rows.append((il, site, stats))

    # ---- per-layer table ----
    print()
    print("Per-layer γ statistics + pair symmetry + coverage at ε")
    print("=" * 96)
    header = (
        f"{'layer':>5} {'site':>11} "
        f"{'γmin':>6} {'γmed':>6} {'γmax':>6} {'γstd':>6} "
        f"{'rmax':>6} {'rmean':>6} "
    )
    for eps in EPSILONS:
        header += f"sym/cov@{eps:<4} "
    print(header)
    print("-" * len(header))
    for il, site, s in rows:
        line = (
            f"{il:>5} {site:>11} "
            f"{s['gamma_min']:6.3f} {s['gamma_median']:6.3f} "
            f"{s['gamma_max']:6.3f} {s['gamma_std']:6.3f} "
            f"{s['pair_resid_max']:6.3f} {s['pair_resid_mean']:6.3f} "
        )
        for eps in EPSILONS:
            line += f"{s['pair_symm_pct'][eps]:3.0f}/{s['cov_pct'][eps]:<3.0f}    "
        print(line)

    # ---- aggregate ----
    print()
    print("Aggregate across all (layer, q|k) vectors")
    print("=" * 60)
    print(f"{'ε':>6}  {'mean pair-symm %':>18}  {'mean coverage %':>17}  "
          f"{'verdict':>10}")
    print("-" * 60)
    for eps in EPSILONS:
        sym = np.mean([s["pair_symm_pct"][eps] for _, _, s in rows])
        cov = np.mean([s["cov_pct"][eps] for _, _, s in rows])
        verdict = "VIABLE" if cov >= 50.0 else "marginal" if cov >= 25.0 else "dead"
        print(f"{eps:>6.2f}  {sym:>18.1f}  {cov:>17.1f}  {verdict:>10}")
    print()
    print("Decision (handoff §3 Option B): VIABLE iff coverage ≥ 50% of")
    print("head_dim positions at some operational ε. 'cov' = positions in")
    print(f"strict γ-band clusters of ≥ {MIN_USEFUL_CLUSTER} pairs, restricted to pairs that")
    print("are γ-symmetric at the same ε (so R̂_qk is also enabled).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

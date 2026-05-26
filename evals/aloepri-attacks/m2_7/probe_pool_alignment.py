"""Phase 1.1 — Structural probe of attacker-pool alignment with K_d.

Computes several candidate alignment metrics for each K_a^k in pool seeds
{1, 2, 3, 4, 5} (K=64 each) against the deployment's K_d (regenerated from
the GGUF metadata `aloepri.seed = 42`). Correlates metrics with the 5-pool
TTRSR data we measured in the disentangle sweep.

Goal: find a single scalar `alignment(K_a^k, K_d) → ℝ` whose pool-aggregate
ranks pools in the same order as observed TTRSR. That metric becomes the
defender's lever for adversarial K_d selection (Phase 2.1).

Metrics computed per K_a^k (each is a (d, d+2h) matrix):
  - frobenius_alignment: ‖K_a^k · pinv(K_d)‖_F. Magnitude of mapping K_a^k
    through K_d's inverse direction. Higher → K_a^k aligned with K_d's
    column space.
  - top_svd_overlap: project K_a^k's top-r right singular vectors onto K_d's
    top-r right singular vectors, sum overlaps. Higher → top-rank subspaces
    align.
  - c_block_subspace_angle: paper §5.2's C-block is `coeffs · basis_F^T` where
    basis_F is the nullspace of F^T. Compute principal angles between the
    C-block subspaces of K_a^k and K_d. Smaller → C-blocks share orientation
    (most theoretically motivated alignment site).
  - mean_singular_overlap: integrate singular-value-weighted overlap across
    all rank-1 components.

Pool aggregates: mean, max, std of each metric over the 64 keymats.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/timo/repos/private-rag-path-2")
sys.path.insert(0, str(REPO / "vendor" / "aloepri-py"))
sys.path.insert(0, str(REPO / "vendor" / "aloepri-py" / "src"))
from keymat import build_keymat_transform  # type: ignore

# Reference TTRSR data from disentangle sweep at split-101 (and pool means
# across all 5 splits) — used to validate the alignment metric's predictive
# power. From `variance-disent-p{P}-s101.json` and the pool means in
# `docs/research/aloepri-keymat-variance.md`.
TTRSR_POOL = {
    1: {"split_101_top1": 0.007, "pool_mean_top1": 0.005},
    2: {"split_101_top1": 0.138, "pool_mean_top1": 0.129},
    3: {"split_101_top1": 0.007, "pool_mean_top1": 0.026},
    4: {"split_101_top1": 0.030, "pool_mean_top1": 0.037},
    5: {"split_101_top1": 0.126, "pool_mean_top1": 0.122},
}

D = 2560
H = 128
LAM = 0.3
K = 64
KD_SEED = 42
POOL_SEEDS = [1, 2, 3, 4, 5]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[probe] device = {DEVICE}, d={D}, h={H}, lam={LAM}, K={K}, K_d seed={KD_SEED}")


def build_kd() -> torch.Tensor:
    print(f"  building K_d at seed={KD_SEED}...")
    t0 = time.perf_counter()
    kd = build_keymat_transform(d=D, h=H, lam=LAM, init_seed=KD_SEED)
    out = torch.from_numpy(kd.key.numpy().astype(np.float64)).to(DEVICE)
    print(f"    K_d shape {tuple(out.shape)}, |.| max {out.abs().max().item():.4f} ({time.perf_counter() - t0:.1f}s)")
    return out


def build_pool(pool_seed: int) -> torch.Tensor:
    """Reconstruct K=64 attacker keymats for one pool seed, matching the
    vendor-CPU path used in `run_isa_multikey._build_attacker_keymat_pool_vendor`."""
    t0 = time.perf_counter()
    d_obs = D + 2 * H
    out = torch.empty((K, D, d_obs), dtype=torch.float64, device=DEVICE)
    for k in range(K):
        init_seed = pool_seed + 1 + 10_000 * k
        kt = build_keymat_transform(d=D, h=H, lam=LAM, init_seed=init_seed)
        out[k] = torch.from_numpy(kt.key.numpy().astype(np.float64)).to(DEVICE)
    print(f"    pool {pool_seed}: K={K} keymats in {time.perf_counter() - t0:.1f}s")
    return out


def frobenius_alignment(Ka: torch.Tensor, Kd_pinv: torch.Tensor) -> float:
    """‖K_a · pinv(K_d)‖_F. Ka shape (d, d+2h), Kd_pinv shape (d+2h, d)."""
    product = Ka @ Kd_pinv  # (d, d)
    return float(product.norm().item())


def top_svd_overlap(Ka: torch.Tensor, Vd_top: torch.Tensor, r: int) -> float:
    """Overlap of top-r right singular subspaces of Ka vs Kd. Vd_top is
    (d+2h, r) from Kd. Returns sum of squared cosines between top-r
    subspaces (between 0 and r)."""
    _, _, Vh = torch.linalg.svd(Ka, full_matrices=False)
    Va_top = Vh[:r].T  # (d+2h, r)
    # principal angles via SVD of cross-correlation
    cross = Va_top.T @ Vd_top  # (r, r)
    s = torch.linalg.svdvals(cross)
    return float((s ** 2).sum().item())


def c_block_basis(Ka: torch.Tensor, d: int, h: int) -> torch.Tensor:
    """Estimate the C-block of K_a — the projection of K_a's column space
    onto its middle block (cols d_obs/3 to 2*d_obs/3, approximately). For
    practical alignment we just use the top-(d_obs - h) right-singular
    vectors weighted by their singular values, which captures the
    high-information directions including the C-contribution."""
    _, _, Vh = torch.linalg.svd(Ka, full_matrices=True)
    # Right-singular vectors corresponding to the C-block range:
    # paper's K_a = [B|C|E]Z has rank ≤ d. Vh[:d] spans the row space.
    # The bottom (d_obs - d) Vh-rows are in the nullspace of K_a's transpose.
    return Vh[:d].contiguous()


def principal_angles(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Principal angles between subspaces spanned by columns of A and B.
    Both A, B should have orthonormal columns; returns angles in radians,
    sorted ascending."""
    # A: (d_obs, ra), B: (d_obs, rb), both orthonormal columns
    cross = A.T @ B  # (ra, rb)
    sv = torch.linalg.svdvals(cross)
    sv = sv.clamp(-1.0, 1.0)
    return torch.acos(sv)


def main():
    Kd = build_kd()
    # pinv(K_d) via SVD for stability
    print("  computing pinv(K_d)...")
    U_d, S_d, Vh_d = torch.linalg.svd(Kd, full_matrices=False)
    Kd_pinv = Vh_d.T @ torch.diag(1.0 / S_d) @ U_d.T  # (d+2h, d)
    print(f"    K_d top-5 sv: {S_d[:5].tolist()}")
    Vd_top_r = Vh_d[:D].T.contiguous()  # (d+2h, d) — full rank subspace

    # Per-K_a^k metrics
    results = {"pool_seeds": {}, "config": {
        "d": D, "h": H, "lam": LAM, "K": K, "kd_seed": KD_SEED,
        "pool_seeds": POOL_SEEDS,
    }}
    for pool_seed in POOL_SEEDS:
        print(f"\n[probe] pool seed = {pool_seed}")
        pool = build_pool(pool_seed)  # (K, d, d+2h)
        per_k_metrics = {"frobenius_alignment": [], "top_svd_overlap_r128": [],
                         "principal_angle_mean": []}
        for k in range(K):
            Ka = pool[k]
            # Metric 1: Frobenius alignment with K_d^+
            fa = frobenius_alignment(Ka, Kd_pinv)
            per_k_metrics["frobenius_alignment"].append(fa)
            # Metric 2: Top-r right-singular subspace overlap (r=h=128)
            tso = top_svd_overlap(Ka, Vh_d[:H].T.contiguous(), r=H)
            per_k_metrics["top_svd_overlap_r128"].append(tso)
            # Metric 3: Principal angles between row-spaces of K_a and K_d
            # (cheaper proxy than C-block extraction)
            _, _, Vh_a = torch.linalg.svd(Ka, full_matrices=False)
            angles = principal_angles(Vh_a.T.contiguous(), Vh_d.T.contiguous())
            per_k_metrics["principal_angle_mean"].append(float(angles.mean().item()))

        # Aggregate per pool
        agg = {}
        for name, vals in per_k_metrics.items():
            arr = np.array(vals)
            agg[name] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "median": float(np.median(arr)),
                "top5_mean": float(np.sort(arr)[-5:].mean()),  # top-5 highest values
            }
        results["pool_seeds"][pool_seed] = {
            "ttrsr": TTRSR_POOL[pool_seed],
            "aggregates": agg,
            "per_k": per_k_metrics,
        }
        # Free memory before next pool
        del pool
        if DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()
        # Print quick summary
        print(f"    fa mean={agg['frobenius_alignment']['mean']:.3f} max={agg['frobenius_alignment']['max']:.3f} "
              f"top5={agg['frobenius_alignment']['top5_mean']:.3f}")
        print(f"    tso mean={agg['top_svd_overlap_r128']['mean']:.3f} max={agg['top_svd_overlap_r128']['max']:.3f} "
              f"top5={agg['top_svd_overlap_r128']['top5_mean']:.3f}")
        print(f"    pa  mean={agg['principal_angle_mean']['mean']:.4f} (rad)")

    # Correlation analysis
    print("\n[probe] correlation analysis: predict TTRSR from alignment aggregates")
    ttrsr_vec = np.array([TTRSR_POOL[p]["pool_mean_top1"] for p in POOL_SEEDS])
    corrs = {}
    for metric in ["frobenius_alignment", "top_svd_overlap_r128", "principal_angle_mean"]:
        for stat in ["mean", "max", "median", "top5_mean"]:
            vals = np.array([results["pool_seeds"][p]["aggregates"][metric][stat]
                             for p in POOL_SEEDS])
            r = float(np.corrcoef(vals, ttrsr_vec)[0, 1])
            corrs[f"{metric}.{stat}"] = {
                "pearson_r_vs_ttrsr_pool_mean": r,
                "vals_per_pool": vals.tolist(),
            }
            print(f"  {metric}.{stat:<10}  r={r:+.3f}  vals={[f'{v:.3f}' for v in vals]}")
    results["correlations"] = corrs
    results["ttrsr_pool_mean_per_pool"] = ttrsr_vec.tolist()

    outpath = Path("/tmp/aloepri-gpu-validation/probe_pool_alignment_result.json")
    outpath.write_text(json.dumps(results, indent=2))
    print(f"\n[probe] wrote → {outpath}")


if __name__ == "__main__":
    main()

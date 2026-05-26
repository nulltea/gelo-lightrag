"""c1 — Luckiness-signature probe (Algorithm-1-only / synthetic mode).

Goal
====
Find a scalar feature `f(K_a^k)` or `f(K_a^k, K_d)` that predicts the
synthetic-mode multi-key ridge-ISA TTRSR for a given attacker pool. The
feature category that wins decides the next defender lever:

  - Category 1 — INTRINSIC: `f(K_a^k)` alone. If wins, "lucky" samples are
    universal; defender attacks the K_a distribution itself (Phase b).
  - Category 2 — ALIGNMENT: `f(K_a^k, K_d)`. If wins, K_d choice is the
    lever; adversarial K_d selection uses this feature as scoring.
  - Category 3 — COMPONENT: a property of a specific Algorithm-1 block
    (U / V / E / F / Z / C). Confirms which component carries the
    variance — directly targets Phase b modifications.

Design choices
==============
- Synthetic mode only (kd_test_seed). Algorithm 2 is absent → signal is clean
  ~5 pp. Real-deployment correlation matters only after we understand the
  pure-Alg-1 mechanism.
- Fixed K_d (seed=42, deployment value). Vary attacker pool seed only.
- N=10 pool seeds × K=64 keymats = 640 K_a^k matrices.
- Vendor-paper-faithful keymat builder (matches run_isa_multikey.py's
  --keymat-impl vendor_cpu default). The TTRSR sweep this script
  correlates against MUST also use vendor_cpu — gpu_native uses a
  single-generator sequential draw schedule that produces different
  K_a^k for the same attacker_seed, which would break the correlation
  premise.
- Batched GPU SVD per pool (one pool at a time). SVD on the stack is
  computed once and shared between Cat 1 (needs S) and Cat 2 (needs Vh).
- JIT logging: every print(..., flush=True) so progress is visible from
  the start.
- Per-pool feature aggregates (mean, max, std, top5_mean) get correlated
  with the pool's synthetic TTRSR. The TTRSR comes from a parallel
  sweep that writes per-pool JSONs to /tmp/aloepri-gpu-validation/.

Runtime budget
==============
- Build pools (CPU, vendor MT19937 path, no unused b_inv): ~1 min × 10 = ~10 min.
- Shared batched SVD + feature computation per pool on iGPU: ~30 s × 10 = ~5 min.
- TTRSR sweep (run in parallel with --keymat-impl vendor_cpu, 10 pool seeds
  at K=64): ~90 s × 10 = ~15 min wall time.

Total ~30 min for c1 features + ~15 min for the parallel TTRSR sweep.

Patches vs initial draft (2026-05-22)
=====================================
1. Replaced `init_keymat_bases` + `bases.u` reference (AttributeError —
   KeyMatBases doesn't store u/v separately) with a local
   `init_bases_with_uv()` that returns u, v alongside b/e/f/z and skips
   the unused `b_inv = inv(b)` cost. Identical seed schedule.
2. Stack SVD computed once per pool and shared between Cat 1 + Cat 2
   (was running the same 64×2560×2816 SVD twice).
3. F-nullspace SVDs switched to `full_matrices=False` (Vh shape (h, h)
   vs (d, d) — ~400× memory cut on the discarded U).
4. `principal_angle_mean` dropped: its (K, d, d) cross-tensor allocates
   ~2.7 GB float64 per pool on top of the already-resident stack +
   Vh_a — risky on the 16 GiB shared-memory iGPU. The spec explicitly
   calls it "coarse subspace overlap" (non-load-bearing); `top_sv_overlap_r128`
   covers the alignment signal.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/timo/repos/private-rag-path-2")
sys.path.insert(0, str(REPO / "vendor" / "aloepri-py"))
sys.path.insert(0, str(REPO / "vendor" / "aloepri-py" / "src"))
from keymat import (  # type: ignore  # noqa: E402
    _sample_gaussian,
    _sample_orthogonal,
    sample_null_columns,
)

OUTDIR = Path("/tmp/aloepri-gpu-validation")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Config — matches the deployment + the universality probe.
# H + output-suffix overridable via env vars for the h-sweep (Phase a2).
D = 2560
H = int(os.environ.get("C1_H_OVERRIDE", "128"))
LAM = 0.3
K = 64
KD_SEED = 42
POOL_SEEDS = list(range(1, 11))  # 10 seeds: 1..10 — fast-iteration default
                                  # (|r|≥0.6 → p<0.07 at N=10; expand to 1..20 if borderline)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Output filename auto-suffixed with h when H != 128, so the h-sweep does
# not clobber the deployment-h=128 run.
OUT_TAG = os.environ.get("C1_OUT_TAG", "" if H == 128 else f"_h{H}")


def log(msg: str) -> None:
    print(msg, flush=True)


def now_s(t0: float) -> str:
    return f"{time.perf_counter() - t0:6.1f}s"


def init_bases_with_uv(d: int, h: int, lam: float, seed: int) -> dict:
    """Paper-faithful Algorithm-1 base builder that ALSO returns u and v
    (which `keymat.init_keymat_bases` collapses into b = u + λv) so we
    can compute per-component features. Skips the unused
    `b_inv = torch.linalg.inv(b)` cost.

    Identical draw order and seeds to `keymat.init_keymat_bases`:
      u  ← _sample_orthogonal(d, seed+1)
      v  ← _sample_gaussian((d, d), seed+2, scale=d**-0.5)
      e1 ← _sample_gaussian((d, h/2), seed+3, scale=d**-0.5)
      e2 ← _sample_gaussian((h/2, h), seed+4, scale=d**-0.5)
      f1 ← _sample_gaussian((h, h/2), seed+5, scale=d**-0.5)
      f2 ← _sample_gaussian((h/2, d), seed+6, scale=d**-0.5)
      z  ← _sample_orthogonal(d+2h, seed+7)
    """
    if h <= 0 or h % 2 != 0:
        raise ValueError(f"expansion size h must be a positive even integer, got {h}")
    if lam < 0:
        raise ValueError(f"lam must be non-negative, got {lam}")
    half_h = h // 2
    u = _sample_orthogonal(d, seed=seed + 1)
    v = _sample_gaussian((d, d), seed=seed + 2, scale=d ** -0.5)
    b = u + lam * v
    e1 = _sample_gaussian((d, half_h), seed=seed + 3, scale=d ** -0.5)
    e2 = _sample_gaussian((half_h, h), seed=seed + 4, scale=d ** -0.5)
    e = e1 @ e2
    f1 = _sample_gaussian((h, half_h), seed=seed + 5, scale=d ** -0.5)
    f2 = _sample_gaussian((half_h, d), seed=seed + 6, scale=d ** -0.5)
    f = f1 @ f2
    z = _sample_orthogonal(d + 2 * h, seed=seed + 7)
    return {"u": u, "v": v, "b": b, "e": e, "f": f, "z": z, "lam": float(lam)}


def generate_key_from_bases(bases: dict, seed: int) -> torch.Tensor:
    """Mirror vendor `keymat.generate_keymat`:
        c = sample_null_columns(f.T, out_rows=d, seed=seed+11)
        K = [b, c, e] @ z
    """
    d = bases["b"].shape[0]
    c = sample_null_columns(bases["f"].T, out_rows=d, seed=seed + 11)
    left = torch.cat([bases["b"], c, bases["e"]], dim=1)
    return left @ bases["z"]


def build_kd_with_bases(seed: int):
    """Build K_d and keep the intermediate bases (including u and v)
    for component-level analysis."""
    bases = init_bases_with_uv(d=D, h=H, lam=LAM, seed=seed)
    key = generate_key_from_bases(bases, seed=seed + 1000)
    Kt = torch.from_numpy(key.numpy().astype(np.float64))
    return Kt.to(DEVICE), bases


def build_pool_stack(pool_seed: int):
    """Build all K=64 K_a^k for one pool seed using the
    vendor-paper-faithful seed schedule. Returns:
      - stack: (K, d, d+2h) torch.float32 on DEVICE — float32 because the
        downstream batched SVD on this shape is 3.4× faster in float32
        on Strix Halo rocSOLVER (bench 2026-05-22: 90 s vs 26 s for K=8).
        Per-aggregate feature precision impact is ≤3e-4 relative error
        (verified vs float64 reference on the same RNG).
      - bases_list: list of K dicts (u/v/b/e/f/z, on CPU, float64) so we
        can analyse components per k for Cat 3. Bases keep float64 —
        Cat 3 is CPU/numpy and the per-K SVDs are tiny (d×h).
    """
    d_obs = D + 2 * H
    stack = torch.empty((K, D, d_obs), dtype=torch.float32, device=DEVICE)
    bases_list = []
    for k in range(K):
        init_seed = pool_seed + 1 + 10_000 * k
        bases = init_bases_with_uv(d=D, h=H, lam=LAM, seed=init_seed)
        bases_list.append(bases)
        key = generate_key_from_bases(bases, seed=init_seed + 1000)
        stack[k] = torch.from_numpy(key.numpy().astype(np.float32)).to(DEVICE)
    return stack, bases_list


def compute_intrinsic(S: torch.Tensor, stack: torch.Tensor) -> dict:
    """Category 1 — per-K_a^k features without K_d. Uses the precomputed
    singular values from the shared stack SVD. S shape (K, d)."""
    frob = (stack ** 2).sum(dim=(1, 2)).sqrt().cpu().numpy()
    S_np = S.cpu().numpy()
    sigma_1 = S_np[:, 0]
    sigma_min = S_np[:, -1]
    cond = np.clip(sigma_1 / np.clip(sigma_min, 1e-12, None), None, 1e12)
    spec_conc = sigma_1 / S_np.sum(axis=1)
    s2 = S_np ** 2
    spec_kurt = (S_np ** 4).sum(axis=1) / s2.sum(axis=1) ** 2
    return {
        "frobenius_norm": frob,
        "sigma_1": sigma_1,
        "sigma_min": sigma_min,
        "condition_number": cond,
        "spectral_concentration_top1": spec_conc,
        "spectral_kurtosis": spec_kurt,
    }


def compute_alignment(stack: torch.Tensor, Kd_pinv: torch.Tensor,
                      Vd_top: torch.Tensor, Vh_a: torch.Tensor) -> dict:
    """Category 2 — alignment features against K_d. Inputs:
      stack    (K, d, d+2h)  — for Frobenius-alignment via K_d's pinv
      Kd_pinv  (d+2h, d)     — pseudo-inverse of K_d
      Vd_top   (d+2h, h)     — top-h right-singular vectors of K_d
                              (precomputed once outside the pool loop)
      Vh_a     (K, d, d+2h)  — right-singular vectors of K_a^k from
                              the SHARED stack SVD (computed once,
                              consumed by both Cat 1 and Cat 2)

    `principal_angle_mean` from the spec is intentionally omitted; the
    (K, d, d) cross-tensor allocates ~2.7 GB float64 on top of the
    already-resident stack (~3 GB) and Vh_a (~3 GB) — risky on a 16 GiB
    shared-memory iGPU. The spec calls it a "coarse subspace overlap"
    (non-load-bearing); `top_sv_overlap_r128` captures the same signal
    with bounded memory.
    """
    K_, d_, d_obs = stack.shape
    # 1) Frobenius alignment: ‖K_a · pinv(K_d)‖_F per k
    fa = (stack @ Kd_pinv).reshape(K_, -1).norm(dim=1).cpu().numpy()
    # 2) Top-h right-SV subspace overlap. Va_top from shared SVD.
    r = H
    Va_top = Vh_a[:, :r, :].transpose(1, 2).contiguous()  # (K, d+2h, r)
    cross = torch.einsum("kij,jl->kil", Va_top.transpose(1, 2), Vd_top)  # (K, r, r)
    sv = torch.linalg.svdvals(cross)  # (K, r)
    top_sv_overlap = (sv ** 2).sum(dim=1).cpu().numpy()  # ∈ [0, r]
    return {
        "frobenius_alignment": fa,
        # Feature name reflects the actual H used (h=128 → top_sv_overlap_r128,
        # h=256 → top_sv_overlap_r256) so cross-h analysis stays unambiguous.
        f"top_sv_overlap_r{r}": top_sv_overlap,
    }


def compute_component(bases_list: list, Kd_bases: dict) -> dict:
    """Category 3 — per-Algorithm-1-component features. For each K_a^k
    pool, decompose into U, V, B=U+λV, E, F, Z; compute properties of
    each block and its alignment with K_d's corresponding block.

    Uses `full_matrices=False` for F-nullspace SVDs: F.T is (d, h) with
    d>h, so Vh has shape (h, h) instead of (d, d) — ~400× memory cut on
    the discarded U.
    """
    n = len(bases_list)
    feats = {
        "U_diag_overlap_with_kd": np.zeros(n),  # tr(U_a U_d^T) / d
        "V_norm": np.zeros(n),                  # ‖V‖_F
        "E_norm": np.zeros(n),                  # ‖E‖_F
        "F_rank_actual": np.zeros(n),           # rank(F)
        "F_top_sv": np.zeros(n),                # σ_1(F)
        "Z_diag_overlap_with_kd": np.zeros(n),  # tr(Z_a Z_d^T) / (d+2h)
        "C_nullspace_angle_with_kd": np.zeros(n),
    }
    Ud_np = Kd_bases["u"].numpy()
    Zd_np = Kd_bases["z"].numpy()
    Fd_np = Kd_bases["f"].numpy()
    # F_d.T is (d, h) — full_matrices=False gives Vh shape (h, h)
    _, sv_d, Vh_d_Ft = np.linalg.svd(Fd_np.T, full_matrices=False)
    cutoff_d = max(1e-10, 1e-10 * (float(sv_d.max()) if sv_d.size > 0 else 0.0))
    rank_d = int((sv_d > cutoff_d).sum())
    null_basis_d = Vh_d_Ft[rank_d:].T  # (h, h - rank_d)
    for i, bases in enumerate(bases_list):
        Ua_np = bases["u"].numpy()
        Va_np = bases["v"].numpy()
        Za_np = bases["z"].numpy()
        Fa_np = bases["f"].numpy()
        feats["U_diag_overlap_with_kd"][i] = np.trace(Ua_np @ Ud_np.T) / D
        feats["V_norm"][i] = float(np.linalg.norm(Va_np))
        feats["E_norm"][i] = float(bases["e"].norm().item())
        feats["Z_diag_overlap_with_kd"][i] = np.trace(Za_np @ Zd_np.T) / (D + 2 * H)
        _, sv_a, Vh_a_Ft = np.linalg.svd(Fa_np.T, full_matrices=False)
        cutoff_a = max(1e-10, 1e-10 * (float(sv_a.max()) if sv_a.size > 0 else 0.0))
        rank_a = int((sv_a > cutoff_a).sum())
        feats["F_rank_actual"][i] = rank_a
        feats["F_top_sv"][i] = float(sv_a[0]) if sv_a.size > 0 else 0.0
        null_basis_a = Vh_a_Ft[rank_a:].T  # (h, h - rank_a)
        m = min(null_basis_a.shape[1], null_basis_d.shape[1])
        cross = null_basis_a[:, :m].T @ null_basis_d[:, :m]
        sv_cross = np.linalg.svd(cross, compute_uv=False)
        feats["C_nullspace_angle_with_kd"][i] = float(
            np.arccos(np.clip(sv_cross, -1.0, 1.0)).mean()
        )
    return feats


def aggregate(per_k: np.ndarray) -> dict:
    arr = np.asarray(per_k)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "top5_mean": float(np.sort(arr)[-5:].mean()),
    }


def main():
    t0 = time.perf_counter()
    log(f"[c1] device={DEVICE}  d={D}  h={H}  λ={LAM}  K={K}")
    log(f"[c1] K_d seed={KD_SEED}, pool seeds = {POOL_SEEDS}")
    log(f"[c1] keymat_impl=vendor_cpu (paper-faithful; matches "
        f"run_isa_multikey.py default)")

    log(f"[{now_s(t0)}] building K_d + bases...")
    Kd, Kd_bases = build_kd_with_bases(KD_SEED)
    log(f"[{now_s(t0)}]   K_d shape {tuple(Kd.shape)}")
    # K_d SVD stays float64 (computed once, cost ~13 s — not in the hot
    # path). Cast pinv + Vd_top to float32 to match the per-pool stack
    # dtype downstream.
    U_d, S_d, Vh_d = torch.linalg.svd(Kd, full_matrices=False)  # Vh_d (d, d+2h)
    Kd_pinv = (Vh_d.T @ torch.diag(1.0 / S_d) @ U_d.T).to(torch.float32)  # (d+2h, d)
    Vd_top = Vh_d[:H].T.contiguous().to(torch.float32)  # (d+2h, h) — reused across all pools
    log(f"[{now_s(t0)}]   pinv(K_d) computed (top-5 sv {S_d[:5].cpu().tolist()})")

    out = OUTDIR / f"c1_luckiness_features{OUT_TAG}.json"
    results = {
        "config": {
            "d": D, "h": H, "lam": LAM, "K": K, "kd_seed": KD_SEED,
            "pool_seeds": POOL_SEEDS, "device": DEVICE,
            "keymat_impl": "vendor_cpu",
        },
        "pool_features": {},
    }
    # Resume: if a previous run wrote pools we can reuse, load them and skip.
    if out.exists():
        try:
            prev = json.loads(out.read_text())
            prev_cfg = prev.get("config", {})
            # Only reuse if the load-bearing config matches.
            reuse_ok = all(
                prev_cfg.get(k) == results["config"][k]
                for k in ("d", "h", "lam", "K", "kd_seed", "keymat_impl")
            )
            if reuse_ok and prev.get("pool_features"):
                results["pool_features"] = dict(prev["pool_features"])
                done = sorted(int(s) for s in results["pool_features"].keys())
                log(f"[{now_s(t0)}] resume: loaded {len(done)} cached pools "
                    f"from {out.name} (seeds={done})")
            else:
                log(f"[{now_s(t0)}] resume: skipping cache (config mismatch)")
        except Exception as exc:
            log(f"[{now_s(t0)}] resume: cache load failed ({exc!r}); starting fresh")

    for pool_seed in POOL_SEEDS:
        if str(pool_seed) in results["pool_features"] or pool_seed in results["pool_features"]:
            log(f"\n[{now_s(t0)}] === pool {pool_seed} === SKIP (cached)")
            continue
        log(f"\n[{now_s(t0)}] === pool {pool_seed} ===")
        log(f"[{now_s(t0)}]   building K=64 K_a^k stack...")
        stack, bases_list = build_pool_stack(pool_seed)
        log(f"[{now_s(t0)}]     stack shape {tuple(stack.shape)} "
            f"({stack.element_size() * stack.numel() / 1e9:.2f} GB)")

        # SHARED SVD — consumed by both Cat 1 (S) and Cat 2 (Vh_a).
        log(f"[{now_s(t0)}]   batched SVD on stack (shared Cat 1 + Cat 2)...")
        _, S, Vh_a = torch.linalg.svd(stack, full_matrices=False)
        log(f"[{now_s(t0)}]     S shape {tuple(S.shape)} "
            f"Vh_a shape {tuple(Vh_a.shape)}")

        log(f"[{now_s(t0)}]   computing intrinsic features (Cat 1)...")
        intr = compute_intrinsic(S, stack)
        log(f"[{now_s(t0)}]     σ_1 mean={intr['sigma_1'].mean():.4f} "
            f"max={intr['sigma_1'].max():.4f}")
        log(f"[{now_s(t0)}]     cond mean={intr['condition_number'].mean():.2e}")

        log(f"[{now_s(t0)}]   computing alignment features (Cat 2)...")
        alig = compute_alignment(stack, Kd_pinv, Vd_top, Vh_a)
        log(f"[{now_s(t0)}]     fa mean={alig['frobenius_alignment'].mean():.3f} "
            f"max={alig['frobenius_alignment'].max():.3f} "
            f"top5={np.sort(alig['frobenius_alignment'])[-5:].mean():.3f}")
        tso_key = f"top_sv_overlap_r{H}"
        log(f"[{now_s(t0)}]     tso mean={alig[tso_key].mean():.3f} "
            f"max={alig[tso_key].max():.3f}")

        # Free SVD outputs before Cat 3 (CPU/numpy work).
        del S, Vh_a
        if DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()

        log(f"[{now_s(t0)}]   computing component features (Cat 3)...")
        comp = compute_component(bases_list, Kd_bases)
        log(f"[{now_s(t0)}]     U_overlap_with_Kd "
            f"mean={comp['U_diag_overlap_with_kd'].mean():+.4f}")
        log(f"[{now_s(t0)}]     C_nullspace_angle_with_Kd "
            f"mean={comp['C_nullspace_angle_with_kd'].mean():.4f} rad")

        per_k_features = {**intr, **alig, **comp}
        results["pool_features"][pool_seed] = {
            "per_k": {name: arr.tolist() for name, arr in per_k_features.items()},
            "aggregates": {name: aggregate(arr) for name, arr in per_k_features.items()},
        }
        # Free remaining GPU buffers for this pool.
        del stack
        if DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()

        out.write_text(json.dumps(results, indent=2))
        log(f"[{now_s(t0)}]   wrote intermediate → {out}")

    log(f"\n[{now_s(t0)}] complete. All {len(POOL_SEEDS)} pools done.")
    log(f"[{now_s(t0)}] features per pool: {len(per_k_features)} "
        f"(counts: Cat1={len(intr)} Cat2={len(alig)} Cat3={len(comp)})")
    log(f"[{now_s(t0)}] result file: {out}")


if __name__ == "__main__":
    main()

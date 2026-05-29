#!/usr/bin/env python3
"""Persistent-attention security gate — attacks the attention-cover
adversary view produced by `crates/gelo-embedder/tests/attn_cover_capture.rs`
(see `docs/plans/perm-attn-gpu-offload.md`).

Three measurements per captured layer (kv-heads concatenated where the
attacker would use them — `perm_kv` is shared across heads):

  baseline   — direct Hungarian-cosine match of the cover-applied rows
               (`v_sent`) to the clean rows (`v_clean`). The cover should
               make this ~chance.
  gate 2     — `perm_kv` recovery from the O_v-INVARIANT geometry: row
               norms (and the row-Gram) survive any feature rotation, so
               position leaks regardless of `O_v`. This is the documented
               residual + the reference-free position attack.
  gate 3     — FastICA coordinate recovery: does blind ICA on `v_sent`
               recover the V feature coordinates `O_v` is meant to hide?
               Scored as mean matched |correlation| of recovered
               components to the true (perm-aligned) V columns — the
               standard ICA source-recovery metric, scale-invariant.

Attacks are reference-free (use only `v_sent` + the known `perm_kv` for
SCORING, which gives the attacker maximum benefit-of-the-doubt per the
harness convention in run_jade.py). Run in the container:

  evals/aloepri-attacks/run-in-container.sh \
      python3 evals/aloepri-attacks/persistent_attn_gate.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from safetensors import safe_open
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA, FastICA


def hungarian_cosine(recovered: np.ndarray, reference: np.ndarray):
    """Max-cosine bipartite match recovered rows → reference rows.
    Returns (per-row cosine of matched pairs, assignment col[i])."""
    R = recovered / (np.linalg.norm(recovered, axis=1, keepdims=True) + 1e-9)
    F = reference / (np.linalg.norm(reference, axis=1, keepdims=True) + 1e-9)
    C = R @ F.T
    row, col = linear_sum_assignment(-C)
    return C[row, col], col


def perm_recovery_rate(assignment: np.ndarray, true_perm: np.ndarray) -> float:
    """assignment[i] = reference row matched to observed row i; the cover
    placed observed row i = clean row true_perm[i]. Recovery = fraction
    matched correctly."""
    return float(np.mean(assignment == true_perm))


def gate2_geometry(v_sent: np.ndarray, v_clean: np.ndarray, true_perm: np.ndarray) -> dict:
    """perm_kv recovery from O_v-invariant geometry. Row norms are
    exactly preserved by an orthogonal feature rotation, so matching
    observed-row norms to clean-row norms recovers the permutation when
    norms are distinct. The full row-Gram is a stronger (here: reference)
    signal; norm-matching is the cheap demonstration."""
    sent_norm = np.linalg.norm(v_sent, axis=1)
    clean_norm = np.linalg.norm(v_clean, axis=1)
    # Hungarian on |norm_i - norm_j| (minimise) → assignment.
    cost = np.abs(sent_norm[:, None] - clean_norm[None, :])
    row, col = linear_sum_assignment(cost)
    assign = np.empty(len(row), dtype=int)
    assign[row] = col
    return {
        "norm_perm_recovery": perm_recovery_rate(assign, true_perm),
        "norm_preserved_max_err": float(np.abs(np.sort(sent_norm) - np.sort(clean_norm)).max()),
    }


def _best_match_corr(components: np.ndarray, reference: np.ndarray) -> float:
    """Mean Hungarian-matched |correlation| between recovered `components`
    columns and `reference` columns (row-aligned). Scale-invariant — the
    standard ICA source-recovery metric."""
    n = components.shape[0]
    Cc = components - components.mean(0)
    Ct = reference - reference.mean(0)
    Cc /= (Cc.std(0) + 1e-9)
    Ct /= (Ct.std(0) + 1e-9)
    corr = np.abs(Cc.T @ Ct) / n
    r, c = linear_sum_assignment(-corr)
    return float(np.mean(corr[r, c]))


def _ica_head(v_sent: np.ndarray, v_true: np.ndarray, var_keep: float, max_iter: int):
    """One head: baseline (no-attack) corr + FastICA-recovered corr.
    PCA-reduce v_sent to the effective rank (var_keep) before ICA — the
    activation cloud's significant rank is far below d_head, so asking for
    d_head independent components is what fails to converge."""
    n, d = v_sent.shape
    base = _best_match_corr(v_sent, v_true)
    if n < 20:
        return base, float("nan"), -1
    try:
        evr = PCA(n_components=min(d, n - 1)).fit(v_sent).explained_variance_ratio_
        k = int(np.searchsorted(np.cumsum(evr), var_keep)) + 1
        k = max(2, min(k, d, n - 1))
        ica = FastICA(n_components=k, whiten="unit-variance",
                      max_iter=max_iter, tol=1e-2, random_state=0)
        S = ica.fit_transform(v_sent)
        return base, _best_match_corr(S, v_true), k
    except Exception:
        return base, float("nan"), -1


def gate3_ica(v_sent_heads: list[np.ndarray], v_clean_heads: list[np.ndarray],
              true_perm: np.ndarray) -> dict:
    """FastICA coordinate recovery, per head, parallelised across cores.
    Recovered components are matched to the true (perm-aligned) V columns;
    high |corr| ⇒ O_v's coordinate-hiding is broken. Baseline = same on
    v_sent directly (the rotation should have mixed the coordinates)."""
    tasks = [(vs, vc[true_perm]) for vs, vc in zip(v_sent_heads, v_clean_heads)]
    out = Parallel(n_jobs=-1)(
        delayed(_ica_head)(vs, vt, 0.99, 200) for vs, vt in tasks
    )
    bases = [o[0] for o in out]
    recs = [o[1] for o in out]
    ks = [o[2] for o in out if o[2] > 0]
    return {
        "ica_recovered_corr": float(np.nanmean(recs)),
        "baseline_mixed_corr": float(np.mean(bases)),
        "ica_n_components_mean": float(np.mean(ks)) if ks else float("nan"),
        "heads_scored": len(bases),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", default="evals/aloepri-attacks/captures")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cdir = Path(args.capture_dir)
    meta = json.loads((cdir / "attn_cover.meta.json").read_text())
    n_kv_heads, d_head = meta["n_kv_heads"], meta["d_head"]
    layers = meta["layers"]
    f = safe_open(str(cdir / "attn_cover.safetensors"), framework="numpy")

    print(f"# persistent-attention gate — {meta['model_id']}  "
          f"n_kv={meta['n_kv']} heads={n_kv_heads} d_head={d_head} σ={meta['sigma']}")
    print(f"# cover: {meta['cover']}\n")
    print(f"{'layer':>5} | {'baseline_cos':>12} {'base_perm':>9} | "
          f"{'gate2_norm_perm':>15} | {'gate3_ica_corr':>14} {'g3_base_corr':>12}")

    results = {"meta": meta, "layers": {}}
    for li in layers:
        tag = f"layer{li:03d}"
        perm = f.get_tensor(f"{tag}.perm_kv")
        v_clean = f.get_tensor(f"{tag}.v_clean")  # (n_kv, kv_dim)
        v_sent = f.get_tensor(f"{tag}.v_sent")
        # Per-head split for ICA (each head rotated by its own O_v).
        vs_heads = [v_sent[:, h * d_head:(h + 1) * d_head] for h in range(n_kv_heads)]
        vc_heads = [v_clean[:, h * d_head:(h + 1) * d_head] for h in range(n_kv_heads)]

        base_cos, base_assign = hungarian_cosine(v_sent, v_clean)
        baseline = {
            "direct_cos_p50": float(np.median(base_cos)),
            "direct_perm_recovery": perm_recovery_rate(base_assign, perm),
        }
        g2 = gate2_geometry(v_sent, v_clean, perm)
        g3 = gate3_ica(vs_heads, vc_heads, perm)
        results["layers"][li] = {"baseline": baseline, "gate2": g2, "gate3": g3}

        print(f"{li:>5} | {baseline['direct_cos_p50']:>12.3f} "
              f"{baseline['direct_perm_recovery']:>9.3f} | "
              f"{g2['norm_perm_recovery']:>15.3f} | "
              f"{g3['ica_recovered_corr']:>14.3f} {g3['baseline_mixed_corr']:>12.3f}")

    # Verdict summary.
    g2r = np.mean([r["gate2"]["norm_perm_recovery"] for r in results["layers"].values()])
    g3i = np.nanmean([r["gate3"]["ica_recovered_corr"] for r in results["layers"].values()])
    g3b = np.nanmean([r["gate3"]["baseline_mixed_corr"] for r in results["layers"].values()])
    print(f"\n# VERDICT (chance perm recovery = {1.0/meta['n_kv']:.3f})")
    print(f"#  gate2 position: perm recovery from O_v-invariant geometry = {g2r:.3f}")
    print(f"#  gate3 content : ICA component corr = {g3i:.3f}  vs no-attack {g3b:.3f}")
    results["summary"] = {"gate2_norm_perm_recovery": float(g2r),
                          "gate3_ica_corr": float(g3i), "gate3_baseline_corr": float(g3b)}
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def gate2_seriation(k_sent: np.ndarray, true_perm: np.ndarray) -> dict:
    """REFERENCE-FREE position recovery from the K-Gram via spectral
    seriation. K is cached post-RoPE; the full-row inner products
    <k_sent_i, k_sent_j> are O_qk-INVARIANT (each head's rotation cancels
    in its own block of the dot product) and carry RoPE's relative-position
    structure. If similarity decays with |position distance| (a public
    RoPE property), the K-Gram is a permuted Robinson matrix and the
    Fiedler vector of its Laplacian recovers the linear position order —
    using only observed data + a public assumption (no private reference).

    Scored (benefit-of-doubt) by |Kendall τ| between the recovered Fiedler
    ordering and the true positions. |τ|→1 ⇒ position leaks; |τ|→0 ⇒ held.
    """
    from scipy.stats import kendalltau

    K = k_sent / (np.linalg.norm(k_sent, axis=1, keepdims=True) + 1e-9)
    S = K @ K.T  # cosine similarity (O_qk-invariant)
    W = S - S.min()
    np.fill_diagonal(W, 0.0)
    L = np.diag(W.sum(1)) - W
    evals, evecs = np.linalg.eigh(L)
    fiedler = evecs[:, 1]  # 2nd-smallest eigenvector → seriation order
    tau, _ = kendalltau(fiedler, true_perm)
    # Also: nearest-neighbour-in-position agreement of the recovered order.
    order = np.argsort(fiedler)
    pos_of_order = true_perm[order]
    adj = float(np.mean(np.abs(np.diff(pos_of_order)) == 1))
    return {
        "seriation_kendall_tau_abs": abs(float(tau)),
        "seriation_adjacent_frac": adj,  # frac of recovered-consecutive that are position-adjacent
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


# ── JADE (4th-order ICA) — gate-3 escalation ───────────────────────
# _whiten + _build_cumulants are copied from attack_drivers/run_jade.py
# (the tested cumulant math); the Jacobi joint-diagonaliser is local.

def _jade_whiten(x: np.ndarray, m: int):
    """Center + PCA-whiten X (features × samples) to Y (m × samples)."""
    xc = x - x.mean(axis=1, keepdims=True)
    cov = (xc @ xc.T) / max(x.shape[1], 1)
    ev, evec = np.linalg.eigh(cov)
    order = np.argsort(-ev)
    ev = np.maximum(ev[order][:m], 1e-12)
    evec = evec[:, order][:, :m]
    w = (evec / np.sqrt(ev)[None, :]).T
    return w @ xc


def _jade_cumulants(y: np.ndarray) -> np.ndarray:
    """JADE cumulant stack (nbcm, m, m) for whitened y (m × T)."""
    m, T = y.shape
    mom4 = np.einsum("it,jt,kt,lt->ijkl", y, y, y, y, optimize=True) / T
    eye = np.eye(m, dtype=y.dtype)
    cum = (mom4
           - eye[:, :, None, None] * eye[None, None, :, :]
           - eye[:, None, :, None] * eye[None, :, None, :]
           - eye[:, None, None, :] * eye[None, :, :, None])
    triu = [(i, j) for i in range(m) for j in range(i, m)]
    return np.stack([cum[:, :, i, j] for i, j in triu], axis=0)


def _joint_diag(Q: np.ndarray, max_sweeps: int = 80, tol: float = 1e-8) -> np.ndarray:
    """Cardoso–Souloumiac Jacobi joint-diagonalisation of the symmetric
    matrix stack Q (K, m, m). Returns orthogonal V minimising the summed
    off-diagonal Frobenius norm."""
    Q = Q.copy().astype(np.float64)
    K, m, _ = Q.shape
    V = np.eye(m)
    for _ in range(max_sweeps):
        moved = False
        for p in range(m - 1):
            for q in range(p + 1, m):
                h1 = Q[:, p, p] - Q[:, q, q]
                h2 = Q[:, p, q] + Q[:, q, p]
                ton = h1 @ h1 - h2 @ h2
                toff = 2.0 * (h1 @ h2)
                theta = 0.5 * np.arctan2(toff, ton + np.sqrt(ton * ton + toff * toff) + 1e-30)
                c, s = np.cos(theta), np.sin(theta)
                if abs(s) > tol:
                    moved = True
                    cp, cq = Q[:, :, p].copy(), Q[:, :, q].copy()
                    Q[:, :, p] = c * cp + s * cq
                    Q[:, :, q] = -s * cp + c * cq
                    rp, rq = Q[:, p, :].copy(), Q[:, q, :].copy()
                    Q[:, p, :] = c * rp + s * rq
                    Q[:, q, :] = -s * rp + c * rq
                    vp, vq = V[:, p].copy(), V[:, q].copy()
                    V[:, p] = c * vp + s * vq
                    V[:, q] = -s * vp + c * vq
        if not moved:
            break
    return V


def _jade_sources(X: np.ndarray, m: int) -> np.ndarray:
    """JADE-recovered sources (n_samples × m) from X (n_samples × d)."""
    y = _jade_whiten(X.T, m)          # (m, n)
    V = _joint_diag(_jade_cumulants(y))
    return (V.T @ y).T                 # (n, m)


def _jade_selftest() -> float:
    """Recover a known orthogonal mixture of non-Gaussian (Laplace)
    sources; returns mean matched |corr| (should be ≈ 1 if JADE works)."""
    rng = np.random.default_rng(0)
    n, m = 4000, 6
    S = rng.laplace(size=(n, m))
    A, _ = np.linalg.qr(rng.standard_normal((m, m)))
    X = S @ A
    rec = _jade_sources(X, m)
    return _best_match_corr(rec, S)


def gate3_jade(v_sent_heads, v_clean_heads, true_perm) -> dict:
    """JADE coordinate recovery, per head (PCA-reduced to effective rank,
    capped for tractable cumulants). Same metric as gate3_ica."""
    def one(v_sent, v_true):
        n, d = v_sent.shape
        if n < 40:
            return float("nan")
        evr = PCA(n_components=min(d, n - 1)).fit(v_sent).explained_variance_ratio_
        k = int(np.searchsorted(np.cumsum(evr), 0.99)) + 1
        k = max(2, min(k, 40, d, n - 1))   # cap m for the m^4 cumulant tensor
        try:
            S = _jade_sources(v_sent, k)
            return _best_match_corr(S, v_true)
        except Exception:
            return float("nan")
    tasks = [(vs, vc[true_perm]) for vs, vc in zip(v_sent_heads, v_clean_heads)]
    out = Parallel(n_jobs=-1)(delayed(one)(vs, vt) for vs, vt in tasks)
    return {"jade_recovered_corr": float(np.nanmean(out)), "heads_scored": len(out)}


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
    ap.add_argument("--self-test", action="store_true",
                    help="validate the JADE implementation on a known mixture and exit")
    args = ap.parse_args()

    if args.self_test:
        c = _jade_selftest()
        print(f"# JADE self-test (Laplace sources, orthogonal mix): matched |corr| = {c:.3f}")
        print("#   ≈1.0 ⇒ JADE recovers known sources (impl valid)")
        return 0 if c > 0.9 else 1

    cdir = Path(args.capture_dir)
    meta = json.loads((cdir / "attn_cover.meta.json").read_text())
    n_kv_heads, d_head = meta["n_kv_heads"], meta["d_head"]
    layers = meta["layers"]
    f = safe_open(str(cdir / "attn_cover.safetensors"), framework="numpy")

    print(f"# persistent-attention gate — {meta['model_id']}  "
          f"n_kv={meta['n_kv']} heads={n_kv_heads} d_head={d_head} σ={meta['sigma']}")
    print(f"# cover: {meta['cover']}\n")
    print(f"# all attacks reference-free; ground truth used for scoring only.\n")
    print(f"{'layer':>5} | {'base_cos':>8} | {'g2_seriation_τ':>14} {'g2_adj':>7} | "
          f"{'g3_ica_corr':>11} {'g3_jade_corr':>12} {'g3_base':>8}")

    results = {"meta": meta, "layers": {}}
    for li in layers:
        tag = f"layer{li:03d}"
        perm = f.get_tensor(f"{tag}.perm_kv")
        v_clean = f.get_tensor(f"{tag}.v_clean")
        v_sent = f.get_tensor(f"{tag}.v_sent")
        k_sent = f.get_tensor(f"{tag}.k_sent")  # post-RoPE, for the position attack
        vs_heads = [v_sent[:, h * d_head:(h + 1) * d_head] for h in range(n_kv_heads)]
        vc_heads = [v_clean[:, h * d_head:(h + 1) * d_head] for h in range(n_kv_heads)]

        base_cos, base_assign = hungarian_cosine(v_sent, v_clean)
        baseline = {"direct_cos_p50": float(np.median(base_cos)),
                    "direct_perm_recovery": perm_recovery_rate(base_assign, perm)}
        g2 = gate2_seriation(k_sent, perm)
        g3 = gate3_ica(vs_heads, vc_heads, perm)
        g3j = gate3_jade(vs_heads, vc_heads, perm)
        results["layers"][li] = {"baseline": baseline, "gate2": g2, "gate3_ica": g3, "gate3_jade": g3j}

        print(f"{li:>5} | {baseline['direct_cos_p50']:>8.3f} | "
              f"{g2['seriation_kendall_tau_abs']:>14.3f} {g2['seriation_adjacent_frac']:>7.3f} | "
              f"{g3['ica_recovered_corr']:>11.3f} {g3j['jade_recovered_corr']:>12.3f} "
              f"{g3['baseline_mixed_corr']:>8.3f}")

    # Verdict summary.
    g2t = np.mean([r["gate2"]["seriation_kendall_tau_abs"] for r in results["layers"].values()])
    g3i = np.nanmean([r["gate3_ica"]["ica_recovered_corr"] for r in results["layers"].values()])
    g3j = np.nanmean([r["gate3_jade"]["jade_recovered_corr"] for r in results["layers"].values()])
    g3b = np.nanmean([r["gate3_ica"]["baseline_mixed_corr"] for r in results["layers"].values()])
    print(f"\n# VERDICT")
    print(f"#  gate2 position (reference-free): K-Gram seriation |Kendall τ| = {g2t:.3f}  (0=held, 1=leaks)")
    print(f"#  gate3 content : ICA corr {g3i:.3f} / JADE corr {g3j:.3f}  vs no-attack {g3b:.3f}")
    results["summary"] = {"gate2_seriation_tau": float(g2t),
                          "gate3_ica_corr": float(g3i), "gate3_jade_corr": float(g3j),
                          "gate3_baseline_corr": float(g3b)}
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

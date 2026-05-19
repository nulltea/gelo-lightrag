"""JADE — Joint Approximate Diagonalisation of Eigenmatrices.

Classical BSS / ICA algorithm from Cardoso & Souloumiac (1993),
*Blind beamforming for non-Gaussian signals*. Recovers source
signals from a linear mixture by jointly diagonalising the
fourth-order cumulant matrices of the whitened observations.

For GELO's threat model: the attacker observes ``U = A·H`` and
runs JADE to recover sources ``Ŝ`` up to a per-row permutation
and sign ambiguity. The permutation/sign is then fixed by
Hungarian assignment to the plaintext H ground truth (this gives
the attacker maximum benefit-of-the-doubt — a real-world attacker
without access to H would have less recovery). The reported
metric is the p95 cosine similarity between aligned Ŝ rows and
true H rows.

Threat-model alignment:

* C0 plain — ``U == H``; JADE recovers a permutation of H rows;
  Hungarian-aligned cosine ≈ 1.
* C1 mask only / C2 / C3 — JADE attempts BSS demixing on the
  masked operand. If the mask is orthogonal and the hidden-state
  rows are non-Gaussian, JADE *can* in principle recover up to
  permutation/sign (that's exactly why ICA works on orthogonal
  mixtures). The shield's job is to defeat this by adding
  Gaussian rows that JADE's cumulant statistics can't separate
  from the data rows. The metric reads how much survives.

For the round-3 B.3 gate: HD₃ passes if its JADE-p95-cosine
matches Haar's within ±0.05. Higher cosine = better recovery =
worse defence; lower is better for the protocol.

Algorithm details:

1. **Centre + whiten** the (s, d) observation matrix on the
   feature axis. Whitening uses PCA via the symmetric
   eigendecomposition of the d × d covariance (treating each
   row of U as a sample).
2. **Estimate fourth-order cumulants**:
   ``Cum(Y_a, Y_b, Y_i, Y_j) = E[Y_a Y_b Y_i Y_j] − δ_{ab}δ_{ij}
       − δ_{ai}δ_{bj} − δ_{aj}δ_{bi}``
   on whitened Y, yielding ``s × s`` cumulant matrices indexed
   by ``(i, j)``. We retain the upper triangle ``i ≤ j``.
3. **Joint diagonalisation** via Cardoso's Givens-rotation
   sweep — for each (p, q) pair, find the 2 × 2 rotation that
   minimises the off-diagonal Frobenius norm summed over the
   cumulant stack. Iterate until per-sweep angle change drops
   below ``tol``.
4. **Demixing matrix** ``B = R · W`` where W whitens and R is
   the joint-diag rotation. Apply: ``Ŝ = B · U`` (sources up to
   permutation/sign).

Cardoso's algorithm is ``O(m⁴ · T + m³ · n_sweeps)`` where m is
the whitened dimension (≤ s) and T = d. We cap m at ``max_dim``
so wall-clock stays bounded at long-context Qwen3 shapes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import AttackResult, classify_risk_level


@dataclass
class _OpAccumulator:
    layer: int
    kind: str
    p95_cosines: list[float]

    @classmethod
    def empty(cls, layer: int, kind: str) -> "_OpAccumulator":
        return cls(layer=layer, kind=kind, p95_cosines=[])


def _whiten(x: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Center + PCA-whiten X (s × d) to Y (m × d), m = min(s, d).

    Returns ``(Y, W)`` where ``Y = W · X_centered`` has ``Y · Yᵀ = I·d``
    up to numerical noise. The whitening matrix W is (m × s).
    """
    s = x.shape[0]
    x_centered = x - x.mean(axis=1, keepdims=True)
    # Sample covariance on rows: (1/d) X X^T → eigen-decomp.
    cov = (x_centered @ x_centered.T) / max(x.shape[1], 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending; clip negatives from numerical jitter.
    order = np.argsort(-eigvals)
    eigvals = np.maximum(eigvals[order][:m], 1e-12)
    eigvecs = eigvecs[:, order][:, :m]
    w = (eigvecs / np.sqrt(eigvals)[None, :]).T  # (m, s)
    y = w @ x_centered
    return y, w


def _build_cumulants(y: np.ndarray) -> np.ndarray:
    """Build the JADE cumulant matrix stack.

    Args:
      y: whitened observations (m, T) — m sources, T samples.
    Returns:
      Cumulant stack (nbcm, m, m), where each slice is one of the
      ``m*(m+1)/2`` upper-triangular cumulant matrices indexed by
      ``(i, j)`` with ``i ≤ j``.
    """
    m, T = y.shape
    # Estimate the fourth-moment tensor M[a,b,i,j] = E[Y_a Y_b Y_i Y_j].
    # For m=8 this is 4 KB; for m=64 it's 16 MB; we cap m via max_dim.
    mom4 = np.einsum("it,jt,kt,lt->ijkl", y, y, y, y, optimize=True) / T
    eye = np.eye(m, dtype=y.dtype)
    # Cum(a,b,i,j) = M_abij − δ_ab δ_ij − δ_ai δ_bj − δ_aj δ_bi
    # (for whitened Y so E[Y_a Y_b] = δ_ab).
    cum = (
        mom4
        - eye[:, :, None, None] * eye[None, None, :, :]
        - eye[:, None, :, None] * eye[None, :, None, :]
        - eye[:, None, None, :] * eye[None, :, :, None]
    )
    triu_ij = [(i, j) for i in range(m) for j in range(i, m)]
    q = np.stack([cum[:, :, i, j] for i, j in triu_ij], axis=0)
    return q


def _joint_diag(q: np.ndarray, max_sweeps: int = 50, tol: float = 1e-6) -> np.ndarray:
    """Cardoso's joint-Givens diagonalisation.

    Args:
      q: cumulant stack (nbcm, m, m).
      max_sweeps: cap on Jacobi sweeps.
      tol: per-sweep convergence threshold on max rotation magnitude.
    Returns:
      Rotation matrix U (m × m) such that ``Uᵀ · Q_k · U`` are
      jointly as diagonal as possible across k.
    """
    nbcm, m, _ = q.shape
    u = np.eye(m, dtype=q.dtype)
    for _sweep in range(max_sweeps):
        max_angle = 0.0
        for p in range(m - 1):
            for r in range(p + 1, m):
                # 2x2 sub-problem: find c, s such that the rotation
                # G(p, r, θ) minimises the sum over k of off-diagonal
                # contributions from Q_k[p, r] and Q_k[r, p]. This is
                # eq. (8) of Cardoso–Souloumiac '93.
                a_pp = q[:, p, p]
                a_pr = q[:, p, r]
                a_rp = q[:, r, p]
                a_rr = q[:, r, r]
                # Build the 2-vector u_k = (a_pp − a_rr, a_pr + a_rp)
                # and form the 2x2 G = sum_k u_k u_k^T.
                v1 = a_pp - a_rr
                v2 = a_pr + a_rp
                gpp = float(np.dot(v1, v1))
                grr = float(np.dot(v2, v2))
                gpr = float(np.dot(v1, v2))
                # The optimal rotation is the eigenvector of G
                # corresponding to the larger eigenvalue.
                # 2x2 eigenproblem closed form:
                tau = (grr - gpp) / 2.0
                if abs(gpr) < 1e-30 and abs(tau) < 1e-30:
                    continue
                # Rotation angle theta solving tan(2θ) = 2*gpr / (grr - gpp)
                # The canonical Jacobi form:
                t = (gpr) / (tau + np.sign(tau if tau != 0 else 1.0) * np.sqrt(tau * tau + gpr * gpr + 1e-30))
                cos_t = 1.0 / np.sqrt(1.0 + t * t)
                sin_t = t * cos_t
                angle = abs(sin_t)
                if angle < tol:
                    continue
                max_angle = max(max_angle, angle)
                # Apply rotation to U columns p, r (right-multiply by G).
                u_p = u[:, p].copy()
                u_r = u[:, r].copy()
                u[:, p] = cos_t * u_p - sin_t * u_r
                u[:, r] = sin_t * u_p + cos_t * u_r
                # Apply rotation to each Q_k from both sides:
                #   Q ← G^T · Q · G   (rows p, r mixed; cols p, r mixed)
                qp = q[:, :, p].copy()
                qr = q[:, :, r].copy()
                q[:, :, p] = cos_t * qp - sin_t * qr
                q[:, :, r] = sin_t * qp + cos_t * qr
                qp = q[:, p, :].copy()
                qr = q[:, r, :].copy()
                q[:, p, :] = cos_t * qp - sin_t * qr
                q[:, r, :] = sin_t * qp + cos_t * qr
        if max_angle < tol:
            break
    return u


def _jade_demix(u_obs: np.ndarray, m: int) -> np.ndarray | None:
    """Run JADE on observations U (s × d). Returns demixing B (m × s).

    The "samples" axis here is d (the feature width) — each column of
    U is one observation of an s-dimensional mixed vector.
    """
    if m < 2 or u_obs.shape[1] < 2 * m:
        # Need enough samples to estimate 4th moments meaningfully.
        return None
    try:
        y, w = _whiten(u_obs, m)
        q = _build_cumulants(y)
        rot = _joint_diag(q)
        return rot.T @ w
    except np.linalg.LinAlgError:
        return None


def _p95_cosine_with_hungarian(s_hat: np.ndarray, h: np.ndarray) -> float:
    """Match Ŝ rows to H rows by Hungarian on |cosine|, return p95.

    JADE's source-recovery has a permutation + sign ambiguity per row.
    The maximum-benefit-of-the-doubt evaluation: assign sources to H
    rows minimising the p95 cosine cost (i.e. picking the assignment
    that *most favours* the attacker), then sign-flip per row.
    """
    from scipy.optimize import linear_sum_assignment

    eps = 1e-12
    n_s, d_s = s_hat.shape
    n_h, d_h = h.shape
    if d_s != d_h or n_s == 0 or n_h == 0:
        return float("nan")
    s_norm = s_hat / (np.linalg.norm(s_hat, axis=1, keepdims=True) + eps)
    h_norm = h / (np.linalg.norm(h, axis=1, keepdims=True) + eps)
    cos_abs = np.abs(s_norm @ h_norm.T)  # (n_s, n_h)
    # Maximise total |cosine|: solve linear-sum-assignment on -cost
    row_idx, col_idx = linear_sum_assignment(-cos_abs)
    matched = cos_abs[row_idx, col_idx]
    if matched.size == 0:
        return float("nan")
    return float(np.quantile(matched, 0.95))


def run(
    snapshots,
    *,
    plain_snaps=None,
    max_dim: int = 64,
    max_features: int = 256,
    **_kwargs,
) -> AttackResult:
    """JADE-recovery driver.

    Args:
      snapshots: protected condition.
      plain_snaps: paired C0 plain for ground-truth H.
      max_dim: cap on whitened dimension m. JADE's cumulant tensor
        is O(m⁴ · T) where T is the sample count (= feature axis);
        ``max_dim = 64`` keeps the m⁴ part well under 1 GB at f32
        (16 MB tensor) while still exercising a meaningful chunk
        of the row axis. Set higher in offline runs.
      max_features: cap on T (feature axis = samples per source).
        At Qwen3's d = 2 048 the cumulant einsum is the dominant
        cost; ``max_features = 256`` knocks it ~8× without
        changing what the metric measures.
    """
    if plain_snaps is None and snapshots.condition == "c0_plain":
        plain_snaps = snapshots

    if plain_snaps is None:
        return AttackResult(
            attack="jade",
            condition=snapshots.condition,
            model_id=snapshots.model_id,
            n_prompts=snapshots.n_prompts(),
            n_train=0,
            n_test=0,
            ttrsr_top1=None,
            ttrsr_top10=None,
            risk_level="unknown",
            primary_metric_name="jade_p95_cosine",
            extra={"note": "jade needs paired plaintext snapshots."},
        )

    plain_index: dict[tuple[int, int, str], Any] = {}
    for meta in plain_snaps.snapshots:
        plain_index[(meta.prompt_idx, meta.layer, meta.kind)] = meta

    accumulators: dict[tuple[int, str], _OpAccumulator] = {}
    n_paired = 0
    n_skipped = 0
    rng = np.random.default_rng(0)

    for meta in snapshots.snapshots:
        key = (meta.prompt_idx, meta.layer, meta.kind)
        plain_meta = plain_index.get(key)
        if plain_meta is None:
            n_skipped += 1
            continue
        try:
            u = snapshots.get_operand(meta, strip_shield=True).detach().cpu().numpy().astype(np.float32)
            h = plain_snaps.get_operand(plain_meta, strip_shield=True).detach().cpu().numpy().astype(np.float32)
        except Exception:
            n_skipped += 1
            continue
        if u.shape != h.shape or u.size == 0:
            n_skipped += 1
            continue
        s = u.shape[0]
        if s > max_dim:
            sel = rng.choice(s, size=max_dim, replace=False)
            sel.sort()
            u = u[sel]
            h = h[sel]
            s = max_dim
        if s < 4:
            n_skipped += 1
            continue
        if max_features is not None and u.shape[1] > max_features:
            feat_sel = rng.choice(u.shape[1], size=max_features, replace=False)
            feat_sel.sort()
            u = u[:, feat_sel]
            h = h[:, feat_sel]

        b = _jade_demix(u, m=s)
        if b is None:
            n_skipped += 1
            continue
        s_hat = b @ u           # (m, d)
        p95 = _p95_cosine_with_hungarian(s_hat, h)
        op_key = (meta.layer, meta.kind)
        acc = accumulators.setdefault(op_key, _OpAccumulator.empty(meta.layer, meta.kind))
        acc.p95_cosines.append(p95)
        n_paired += 1

    if n_paired == 0:
        return AttackResult(
            attack="jade",
            condition=snapshots.condition,
            model_id=snapshots.model_id,
            n_prompts=snapshots.n_prompts(),
            n_train=0,
            n_test=0,
            ttrsr_top1=None,
            ttrsr_top10=None,
            risk_level="unknown",
            primary_metric_name="jade_p95_cosine",
            extra={"note": "no paired snapshots / JADE failed everywhere",
                   "n_skipped": n_skipped},
        )

    all_p95 = np.array(
        [v for acc in accumulators.values() for v in acc.p95_cosines if v == v],
        dtype=np.float64,
    )
    median_p95 = float(np.nanmedian(all_p95))
    per_op = {
        f"layer{layer:03d}.{kind}": {
            "n_snapshots": len(acc.p95_cosines),
            "jade_p95_cosine_median": float(np.nanmedian(acc.p95_cosines)) if acc.p95_cosines else None,
        }
        for (layer, kind), acc in sorted(accumulators.items())
    }
    return AttackResult(
        attack="jade",
        condition=snapshots.condition,
        model_id=snapshots.model_id,
        n_prompts=snapshots.n_prompts(),
        n_train=0,
        n_test=n_paired,
        ttrsr_top1=median_p95,
        ttrsr_top10=None,
        risk_level=classify_risk_level(median_p95),
        primary_metric_name="jade_p95_cosine",
        extra={
            "jade_p95_cosine_median": median_p95,
            "max_dim": max_dim,
            "n_paired": n_paired,
            "n_skipped": n_skipped,
            "per_op": per_op,
            "note": (
                "primary metric = median over snapshots of the p95 Hungarian-"
                "aligned cosine between JADE-recovered sources and plaintext H "
                "rows. Higher cosine = more recovery. HD₃ default-flip requires "
                "C3 within ±0.05 of C2."
            ),
        },
    )


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--snapshot-root", required=True, type=Path)
    p.add_argument("--snapshot-basename", required=True)
    p.add_argument("--plain-basename", default="c0_plain")
    p.add_argument("--max-dim", type=int, default=64)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from snapshots_loader import SnapshotSet  # type: ignore  # noqa: E402

    snaps = SnapshotSet.open(args.snapshot_basename, root=args.snapshot_root)
    plain = SnapshotSet.open(args.plain_basename, root=args.snapshot_root)
    result = run(snaps, plain_snaps=plain, max_dim=args.max_dim)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2))
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    _cli()

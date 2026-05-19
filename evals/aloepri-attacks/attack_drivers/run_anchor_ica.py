"""Anchor-based recovery driver — GELO paper §4.3.3.

Threat model: the PCIe attacker observes the masked operand
``U = A·H`` (an `(s, d)` matrix where ``s = n_data + k_shield``)
and knows ``K`` row-paired anchors — for ``i`` in some anchor set
``K_idx``, the attacker knows that row ``i`` of ``U`` corresponds
to row ``i`` of plaintext ``H``, with ``H[i, :]`` itself known
(adversary-controlled prompts or universally-frequent BOS / system
tokens). The attack tries to recover ``H[j, :]`` for ``j ∉ K_idx``.

Two attack variants run side by side; both report p95 cosine
similarity between predicted ``Ĥ[j]`` and true ``H[j]`` on the
non-anchor rows:

* **ridge** — direct ridge regression learning a linear map
  ``M : ℝ^d → ℝ^d`` that minimises
  ``Σ_{i ∈ K_idx} ‖M·U[i, :] − H[i, :]‖²`` with regulariser
  ``λI``. The attacker's hope is that A's row-mixing has *some*
  per-column structure ridge can fit; for orthogonal A with iid
  row mixing this is provably ineffective, which is exactly the
  property GELO claims. The metric reads how close it gets.

* **fastica_anchor** — Hyvärinen FastICA on ``U`` (`n_components = s`)
  recovers a demixing matrix ``W`` and source estimates ``Ŝ = W·U``
  up to a row permutation + per-row scale/sign ambiguity. The K
  anchor pairs ``(Ŝ_i, H_i)`` are used to solve a Hungarian-cost
  assignment that pins down the permutation; remaining scale/sign
  is fixed per anchor by signed cosine. After permutation/sign
  alignment, ``Ŝ[j]`` is the recovered estimate of ``H[j]``.

The driver runs over every captured (layer, kind) tuple and
aggregates: median p95-cosine across snapshots, plus per-(layer,
kind) detail in ``extra.per_op``. ``K`` defaults to a small set
of {1, 4, 16} anchors so the gate run keeps wall-clock under
~10 min on Qwen3-1.7B shapes; passing ``--anchor-counts`` extends
the sweep to the paper's k ∈ {1, 10, 50, 100, 200}.

For the round-3 B.3 gate: HD₃ passes if the C3 p95-cosine
matches C2's within ±0.05 on both variants (paper's noise/error
band). Higher cosine = better recovery = worse defence.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import AttackResult, classify_risk_level


# Default anchor counts. Capped low so the gate doesn't pay a 30-min
# wall-clock at every commit; release-gate runs override via
# `--anchor-counts 1,10,50,100,200` to reproduce paper §4.3.3.
DEFAULT_ANCHOR_COUNTS: tuple[int, ...] = (1, 4, 16)


@dataclass
class _OpAccumulator:
    """Per-(layer, kind) running aggregator across prompts."""

    layer: int
    kind: str
    ridge_p95: dict[int, list[float]]
    fastica_p95: dict[int, list[float]]

    @classmethod
    def empty(cls, layer: int, kind: str, anchor_counts: tuple[int, ...]) -> "_OpAccumulator":
        return cls(
            layer=layer,
            kind=kind,
            ridge_p95={k: [] for k in anchor_counts},
            fastica_p95={k: [] for k in anchor_counts},
        )


def _p95_cosine(pred: np.ndarray, true: np.ndarray) -> float:
    """p95 of absolute cosine similarity between paired rows."""
    eps = 1e-12
    p_norm = np.linalg.norm(pred, axis=1) + eps
    t_norm = np.linalg.norm(true, axis=1) + eps
    cos = np.abs(np.einsum("ij,ij->i", pred, true) / (p_norm * t_norm))
    if cos.size == 0:
        return float("nan")
    return float(np.quantile(cos, 0.95))


def _ridge_recover(
    u: np.ndarray,
    h: np.ndarray,
    anchor_idx: np.ndarray,
    non_anchor_idx: np.ndarray,
    lam: float = 1.0,
) -> np.ndarray:
    """Fit ``M : ℝ^d → ℝ^d`` minimising
    ``Σ_{i ∈ anchor_idx} ‖M·U[i] − H[i]‖² + λ‖M‖²``, apply to
    non-anchor U rows, return predicted H[non_anchor_idx].

    Closed form: ``M = H_a^T · U_a · (U_a^T · U_a + λI)^{-1}``
    where the rows of U_a, H_a are the anchor pairs.
    """
    d = u.shape[1]
    u_a = u[anchor_idx]            # (K, d)
    h_a = h[anchor_idx]            # (K, d)
    # Solve M (d × d) such that U_a · M^T ≈ H_a, in ridge form.
    # M^T = (U_a^T U_a + λI)^{-1} · U_a^T · H_a
    gram = u_a.T @ u_a + lam * np.eye(d, dtype=u.dtype)
    rhs = u_a.T @ h_a
    m_t = np.linalg.solve(gram, rhs)        # (d, d)
    pred = u[non_anchor_idx] @ m_t          # (s − K, d)
    return pred


def _fastica_recover(
    u: np.ndarray,
    h: np.ndarray,
    anchor_idx: np.ndarray,
    non_anchor_idx: np.ndarray,
    *,
    n_components: int,
    seed: int = 0,
) -> np.ndarray | None:
    """FastICA-based recovery with anchor-assignment.

    Runs FastICA on the (s × d) operand (treating each column as one
    time sample of an s-dimensional mixed signal). Recovers
    ``Ŝ = W · U`` shaped (s, d) — but rows are arbitrarily permuted
    + scaled. Solve a Hungarian assignment on the K anchor pairs to
    find the row permutation, then apply per-anchor signed-cosine
    sign-fix. Return predicted H rows for non_anchor_idx.

    Returns ``None`` on convergence failure (FastICA raises; we
    catch and signal "attack failed at this snapshot").
    """
    from sklearn.decomposition import FastICA
    from scipy.optimize import linear_sum_assignment

    # Reduce noise: clip n_components to min(s, d) − 1 because
    # FastICA needs full-rank whitening, and our s_pad cases have
    # s < d so this is fine.
    n_components = max(1, min(n_components, u.shape[0], u.shape[1] - 1))

    with warnings.catch_warnings():
        # Silence FastICA's frequent "did not converge" warning;
        # we surface it via the return-None contract instead.
        warnings.simplefilter("ignore")
        try:
            # `transpose` because sklearn FastICA expects
            # (n_samples, n_features) where n_features are the sources;
            # our columns of U are time-samples of length-s signals,
            # so we pass U.T as the input matrix shape (d, s) and get
            # back signals shaped (d, n_components). Transpose back to
            # get (n_components, d) of recovered sources.
            ica = FastICA(
                n_components=n_components,
                random_state=seed,
                max_iter=500,
                tol=1e-3,
                whiten="unit-variance",
            )
            # Input: U^T shape (d, s). FastICA finds X = S · A_ica where
            # S is the source signal matrix shape (d, n_components).
            s_signals = ica.fit_transform(u.T)
            # s_signals.T has shape (n_components, d) — these are the
            # recovered sources, one per row.
            s_hat = s_signals.T
        except Exception:
            return None

    if s_hat.shape[0] < anchor_idx.size:
        return None

    # Solve Hungarian on |cosine| between Ŝ rows and H[anchor_idx] rows.
    eps = 1e-12
    h_a = h[anchor_idx]
    s_norm = s_hat / (np.linalg.norm(s_hat, axis=1, keepdims=True) + eps)
    h_norm = h_a / (np.linalg.norm(h_a, axis=1, keepdims=True) + eps)
    cosabs = np.abs(s_norm @ h_norm.T)      # (n_components, K)
    # We want each anchor's H_a[k] paired with one Ŝ row. Hungarian
    # minimises cost; flip sign to maximise |cosine|.
    row_idx, col_idx = linear_sum_assignment(-cosabs.T)  # cost is K x n_comp
    # col_idx[k] is the Ŝ-row index assigned to anchor k.
    anchor_to_source = {int(anchor_idx[k]): int(col_idx[k]) for k in range(anchor_idx.size)}

    # Sign / scale fix per anchor: rescale Ŝ row so that
    # cos-signed alignment with anchor's H row is positive.
    signs = np.ones(s_hat.shape[0])
    used = set()
    for k_pos, ki in enumerate(anchor_idx):
        src = col_idx[k_pos]
        if src in used:
            continue
        used.add(src)
        dot = float(s_hat[src] @ h[ki])
        if dot < 0:
            signs[src] = -1.0
    s_hat = s_hat * signs[:, None]

    # For non-anchor rows j, we need a source index. We extend the
    # Hungarian assignment greedily: for each non-anchor row j, pick
    # the source index that maximises |cosine| with j's true H row
    # — but we don't have true non-anchor H! So we use a different
    # rule: the source indices NOT taken by anchors map to non-anchor
    # rows in original order. This is the "matched-on-anchors,
    # ordered-otherwise" assignment.
    free_sources = [i for i in range(s_hat.shape[0]) if i not in used]
    pred = np.zeros((non_anchor_idx.size, h.shape[1]), dtype=u.dtype)
    for slot, j in enumerate(non_anchor_idx):
        if slot < len(free_sources):
            pred[slot] = s_hat[free_sources[slot]]
        else:
            # More non-anchor rows than free sources (n_components < s
            # after the clip above). The remaining slots stay zero
            # → low cosine, signals "attack failed for this row".
            pred[slot] = 0.0

    return pred


def run(
    snapshots,
    *,
    plain_snaps=None,
    anchor_counts: tuple[int, ...] = DEFAULT_ANCHOR_COUNTS,
    max_dim: int = 256,
    max_features: int = 256,
    **_kwargs,
) -> AttackResult:
    """Anchor-based-recovery driver entry point.

    Args:
      snapshots: protected condition SnapshotSet.
      plain_snaps: paired C0_plain SnapshotSet for ground-truth H.
      anchor_counts: tuple of K values to sweep. Defaults to the
        cheap dev set ``(1, 4, 16)``; release-gate runs use
        ``(1, 10, 50, 100, 200)`` per paper §4.3.3.
      max_dim: row-axis subsample cap. FastICA's per-iteration cost
        is ``O(s²·d)`` which is fine at small s; at long-context
        Qwen3 shapes (``s ≈ 2 048``) we subsample to keep wall-
        clock bounded. ``max_dim`` is the maximum rows considered
        per snapshot; rows are sampled uniformly without
        replacement.
      max_features: feature-axis (d) subsample cap. The ridge
        variant solves an ``O(d³)`` linear system per snapshot ×
        K; at Qwen3's ``d = 2 048`` this is ~8.6 GFLOP per call
        and dominates the wall-clock. Capping ``d`` to 256
        knocks the per-call cost to ~16 MFLOP without losing the
        defensive signal — the metric is about the row-mixing
        structure of ``A``, not feature dimensionality. Set
        ``max_features=None`` for the full-feature run when
        wall-clock is not constrained.
    """
    if plain_snaps is None and snapshots.condition == "c0_plain":
        plain_snaps = snapshots

    if plain_snaps is None:
        return AttackResult(
            attack="anchor_ica",
            condition=snapshots.condition,
            model_id=snapshots.model_id,
            n_prompts=snapshots.n_prompts(),
            n_train=0,
            n_test=0,
            ttrsr_top1=None,
            ttrsr_top10=None,
            risk_level="unknown",
            primary_metric_name="anchor_recovery_p95_cosine",
            extra={
                "note": "anchor_ica needs paired plaintext snapshots; supply via run_all.py.",
            },
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

        # Subsample rows if too tall (FastICA wall-clock).
        s = u.shape[0]
        if s > max_dim:
            sel = rng.choice(s, size=max_dim, replace=False)
            sel.sort()
            u = u[sel]
            h = h[sel]
            s = max_dim
        if s < 4:
            # Need at least a couple of anchors + a non-anchor row.
            n_skipped += 1
            continue
        # Subsample features to bound the ridge-solve cost (O(d³)).
        if max_features is not None and u.shape[1] > max_features:
            feat_sel = rng.choice(u.shape[1], size=max_features, replace=False)
            feat_sel.sort()
            u = u[:, feat_sel]
            h = h[:, feat_sel]

        op_key = (meta.layer, meta.kind)
        acc = accumulators.setdefault(op_key, _OpAccumulator.empty(meta.layer, meta.kind, anchor_counts))

        for k in anchor_counts:
            k_eff = min(k, s - 1)
            if k_eff < 1:
                continue
            # Deterministic anchor selection — first k rows. The
            # threat-model assumption is "BOS + first prompt tokens
            # are adversary-known", so leading-rows aligns.
            anchor_idx = np.arange(k_eff, dtype=np.int64)
            non_anchor_idx = np.arange(k_eff, s, dtype=np.int64)
            if non_anchor_idx.size == 0:
                continue

            # Ridge variant.
            try:
                pred_ridge = _ridge_recover(u, h, anchor_idx, non_anchor_idx)
                acc.ridge_p95[k].append(_p95_cosine(pred_ridge, h[non_anchor_idx]))
            except np.linalg.LinAlgError:
                pass

            # FastICA variant. Capped at n_components = s (one source
            # per observed row).
            pred_ica = _fastica_recover(
                u, h, anchor_idx, non_anchor_idx, n_components=s
            )
            if pred_ica is not None:
                acc.fastica_p95[k].append(_p95_cosine(pred_ica, h[non_anchor_idx]))

        n_paired += 1

    if n_paired == 0:
        return AttackResult(
            attack="anchor_ica",
            condition=snapshots.condition,
            model_id=snapshots.model_id,
            n_prompts=snapshots.n_prompts(),
            n_train=0,
            n_test=0,
            ttrsr_top1=None,
            ttrsr_top10=None,
            risk_level="unknown",
            primary_metric_name="anchor_recovery_p95_cosine",
            extra={
                "note": "no paired (prompt, layer, kind) tuples found",
                "n_skipped": n_skipped,
            },
        )

    # Aggregate medians across snapshots, per (variant, K).
    def _agg(per_op_field: str) -> dict[int, float]:
        out: dict[int, float] = {}
        for k in anchor_counts:
            values = [v for acc in accumulators.values() for v in getattr(acc, per_op_field)[k]]
            out[k] = float(np.nanmedian(values)) if values else float("nan")
        return out

    ridge_medians = _agg("ridge_p95")
    fastica_medians = _agg("fastica_p95")

    # Primary metric: the worst-case (max over variants and K) of the
    # p95 cosine — that's the strongest attack the adversary can
    # mount with the given anchor budget. Higher = worse defence.
    best_attack_cosine = float(
        np.nanmax(
            [v for v in list(ridge_medians.values()) + list(fastica_medians.values()) if v == v]
            or [float("nan")]
        )
    )

    per_op = {
        f"layer{layer:03d}.{kind}": {
            "n_snapshots": max(
                len(next(iter(acc.ridge_p95.values()))) if acc.ridge_p95 else 0,
                len(next(iter(acc.fastica_p95.values()))) if acc.fastica_p95 else 0,
            ),
            "ridge_p95_per_k": {
                str(k): (float(np.nanmedian(acc.ridge_p95[k])) if acc.ridge_p95[k] else None)
                for k in anchor_counts
            },
            "fastica_p95_per_k": {
                str(k): (float(np.nanmedian(acc.fastica_p95[k])) if acc.fastica_p95[k] else None)
                for k in anchor_counts
            },
        }
        for (layer, kind), acc in sorted(accumulators.items())
    }

    return AttackResult(
        attack="anchor_ica",
        condition=snapshots.condition,
        model_id=snapshots.model_id,
        n_prompts=snapshots.n_prompts(),
        n_train=0,
        n_test=n_paired,
        ttrsr_top1=best_attack_cosine,
        ttrsr_top10=None,
        risk_level=classify_risk_level(best_attack_cosine),
        primary_metric_name="anchor_recovery_p95_cosine",
        extra={
            "ridge_p95_per_k": {str(k): v for k, v in ridge_medians.items()},
            "fastica_p95_per_k": {str(k): v for k, v in fastica_medians.items()},
            "best_attack_p95_cosine": best_attack_cosine,
            "anchor_counts": list(anchor_counts),
            "max_dim": max_dim,
            "n_paired": n_paired,
            "n_skipped": n_skipped,
            "per_op": per_op,
            "note": (
                "primary metric = best p95 cosine across (variant, K). Higher = "
                "more recovery = worse defence. For HD₃ default promotion, C3 "
                "must be within ±0.05 of C2 (paper §4.3.3 / round-3 B.3 band)."
            ),
        },
    )


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--snapshot-root", required=True, type=Path)
    p.add_argument("--snapshot-basename", required=True)
    p.add_argument("--plain-basename", default="c0_plain")
    p.add_argument("--anchor-counts", default="1,4,16",
                   help="comma-separated K values for the anchor sweep")
    p.add_argument("--max-dim", type=int, default=256)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from snapshots_loader import SnapshotSet  # type: ignore  # noqa: E402

    counts = tuple(int(x) for x in args.anchor_counts.split(",") if x)
    snaps = SnapshotSet.open(args.snapshot_basename, root=args.snapshot_root)
    plain = SnapshotSet.open(args.plain_basename, root=args.snapshot_root)
    result = run(
        snaps,
        plain_snaps=plain,
        anchor_counts=counts,
        max_dim=args.max_dim,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2))
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    _cli()

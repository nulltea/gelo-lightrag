"""Gram-error driver — GELO paper §4.3.4 leakage metric.

A *metric*, not an attack. Quantifies how much row-Gram structure
of the plaintext hidden state ``H`` survives the mask round-trip
``U = A·H``. For an exactly-orthogonal ``A``:

* **Column Gram is preserved exactly**: ``Uᵀ·U = Hᵀ·A·Aᵀ·H = Hᵀ·H``.
  Orthogonality sanity check — must read ~0 for any orthogonal
  mask, modulo f32 noise.
* **Row Gram is similarity-transformed**:
  ``U·Uᵀ = A·(H·Hᵀ)·Aᵀ``. Eigenvalues are preserved, off-diagonal
  entries are scrambled by ``A``. The Hungarian-matched row-Gram
  error tests whether the attacker can recover the row-permutation
  cheat — if low, the mask is leaking row-identity structure.

For the round-3 B.3 gate, the question is whether HD₃'s discrete
``2^{3·s}``-element orbit leaks more row structure than Haar's
continuous measure. Three sub-metrics get reported:

1. ``col_gram_error`` — ``‖Uᵀ·U − Hᵀ·H‖_F / ‖Hᵀ·H‖_F``. Should
   be ~0 for both Haar and HD₃ (orthogonality holds in both
   cases). A non-zero value indicates a protocol bug, not a
   security gap.
2. ``row_gram_spectrum_error`` —
   ``‖sort(eig(U·Uᵀ)) − sort(eig(H·Hᵀ))‖₂ / ‖sort(eig(H·Hᵀ))‖₂``.
   Similarity-invariance sanity. Must be ~0 for either mask.
3. ``hungarian_row_gram_error`` — solve the row-assignment
   ``π : [s] → [s]`` minimising the off-diagonal Frobenius
   distance, then report
   ``‖G_U[π,π] − G_H‖_F / ‖G_H‖_F``. Approximates the cheat the
   attacker can mount with side-channel knowledge of an oracle
   ranking of plaintext row norms. **This is the leakage metric**
   for the gate.

The driver runs over every captured (layer, kind) tuple and
aggregates: median across snapshots, plus per-(layer, kind) detail
in ``extra.per_op``.

Threat-model alignment:

* C0 plain — ``U = H``; all three metrics are 0 by construction.
* C1 mask-only / C2 / C3 — col_gram_error and
  row_gram_spectrum_error must stay near 0 (mask is orthogonal);
  hungarian_row_gram_error reads the leakage level. Lower is
  worse (mask leaks more). HD₃ ships parity-with-Haar if
  ``hungarian_row_gram_error(C3) ≥ hungarian_row_gram_error(C2) − 0.20``
  (paper's noise/error reporting tolerance).
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
    """Per-(layer, kind) running aggregator across prompts."""

    layer: int
    kind: str
    col_gram_errors: list[float]
    spectrum_errors: list[float]
    hungarian_errors: list[float]

    @classmethod
    def empty(cls, layer: int, kind: str) -> "_OpAccumulator":
        return cls(layer=layer, kind=kind, col_gram_errors=[], spectrum_errors=[], hungarian_errors=[])


def _col_gram_error(u: np.ndarray, h: np.ndarray) -> float:
    """``‖Uᵀ·U − Hᵀ·H‖_F / ‖Hᵀ·H‖_F``.

    Both ``u`` and ``h`` are shaped ``(s, d)``. With strip_shield=True
    they're trimmed to the data rows (``n_data``), so a *shield-only*
    contribution to the column Gram is removed before the comparison.
    """
    g_u = u.T @ u
    g_h = h.T @ h
    num = np.linalg.norm(g_u - g_h)
    den = np.linalg.norm(g_h)
    if den == 0:
        return float("nan")
    return float(num / den)


def _row_gram_spectrum_error(u: np.ndarray, h: np.ndarray) -> float:
    """``‖sort(eig(U·Uᵀ)) − sort(eig(H·Hᵀ))‖₂ / ‖sort(eig(H·Hᵀ))‖₂``.

    Eigenvalues are similarity-invariant, so this should be ~0 for
    any orthogonal mask. Computed via the singular values of ``U``
    and ``H`` (squared SVs are the eigenvalues of the row Gram —
    avoids forming the full ``s × s`` Gram for tall matrices).
    """
    sv_u = np.linalg.svd(u, compute_uv=False)
    sv_h = np.linalg.svd(h, compute_uv=False)
    eig_u = np.sort(sv_u**2)
    eig_h = np.sort(sv_h**2)
    # Length-pad to common size (one matrix may have a different rank).
    n = max(eig_u.size, eig_h.size)
    eig_u = np.pad(eig_u, (n - eig_u.size, 0))
    eig_h = np.pad(eig_h, (n - eig_h.size, 0))
    num = np.linalg.norm(eig_u - eig_h)
    den = np.linalg.norm(eig_h)
    if den == 0:
        return float("nan")
    return float(num / den)


def _hungarian_row_gram_error(u: np.ndarray, h: np.ndarray) -> float:
    """Row-assignment π minimising ``‖G_U[π,π] − G_H‖_F``, normalised.

    This is the §4.3.4 attacker-cheat metric: even though A scrambles
    the row Gram, if the attacker can find a row permutation that
    aligns ``U·Uᵀ`` with ``H·Hᵀ``, the row-identity has leaked.

    Optimal joint row permutation is NP-hard in general (a 2D
    quadratic assignment problem). We use the standard relaxation:
    solve a linear assignment on the cost matrix
    ``C[i, j] = ‖row_i(G_U) − row_j(G_H)‖_2`` via the Hungarian
    algorithm. That gives a row → row mapping that the joint
    assignment lower-bounds; cheap and reproducible.
    """
    from scipy.optimize import linear_sum_assignment

    s = u.shape[0]
    if s == 0 or h.shape[0] == 0:
        return float("nan")
    if u.shape[0] != h.shape[0]:
        # Shapes diverge if shield rows differ between conditions —
        # we strip_shield upstream, so this should not fire.
        return float("nan")

    g_u = u @ u.T
    g_h = h @ h.T

    # Sort each row of G_U / G_H by absolute magnitude so the cost is
    # permutation-invariant in the *column* axis — the assignment is
    # then exclusively over rows. This gives a tighter lower bound
    # than naive row-distance on the unsorted matrices.
    g_u_sorted = np.sort(np.abs(g_u), axis=1)
    g_h_sorted = np.sort(np.abs(g_h), axis=1)

    # Cost matrix: pairwise row-distance in sorted-row-Gram space.
    diff = g_u_sorted[:, None, :] - g_h_sorted[None, :, :]
    cost = np.linalg.norm(diff, axis=2)
    row_idx, col_idx = linear_sum_assignment(cost)

    # Apply the permutation to G_U and report the off-diagonal-aware
    # Frobenius error against G_H (the 2D quadratic-assignment
    # objective, evaluated under the linear-assignment relaxation).
    perm = np.empty(s, dtype=np.int64)
    perm[row_idx] = col_idx  # row i of G_U should map to row perm[i] of G_H
    # Re-order rows AND columns of G_U so it matches G_H.
    g_u_aligned = g_u[perm[:, None], perm[None, :]]
    num = np.linalg.norm(g_u_aligned - g_h)
    den = np.linalg.norm(g_h)
    if den == 0:
        return float("nan")
    return float(num / den)


def run(
    snapshots,
    *,
    plain_snaps=None,
    max_features: int = 256,
    **_kwargs,
) -> AttackResult:
    """Gram-error driver entry point.

    Args:
      snapshots: the per-condition `SnapshotSet` being evaluated.
      plain_snaps: the C0 plain `SnapshotSet`, used as the
        ground-truth source for ``H``. Required for any condition
        other than C0; when ``snapshots is plain_snaps`` (or plain
        is None and the input IS C0) we compute the trivial
        identity case and report zeros.
      max_features: feature-axis (d) subsample cap. The column Gram
        ``Uᵀ·U`` is shape ``(d, d)`` so the comparison is
        ``O(d²)`` memory + traffic per snapshot. Caps at 256 by
        default; the metric is dimension-invariant in
        expectation, so subsampling doesn't change what we test.
    """
    if plain_snaps is None and snapshots.condition == "c0_plain":
        plain_snaps = snapshots

    if plain_snaps is None:
        return AttackResult(
            attack="gram_error",
            condition=snapshots.condition,
            model_id=snapshots.model_id,
            n_prompts=snapshots.n_prompts(),
            n_train=0,
            n_test=0,
            ttrsr_top1=None,
            ttrsr_top10=None,
            risk_level="unknown",
            primary_metric_name="hungarian_row_gram_error",
            extra={
                "note": (
                    "gram_error needs paired plaintext snapshots to compute "
                    "leakage; run_all.py supplies `plain_snaps=c0_plain` "
                    "when running the matrix. Standalone invocations on a "
                    "single non-C0 snapshot are not meaningful."
                ),
            },
        )

    # Build (prompt_idx, layer, kind) → operand index for the plain
    # snapshots so we can pair efficiently. The plain snapshots have
    # the same prompts in the same order (capture binary invariant),
    # so the natural index is the (prompt_idx, layer, kind) triple.
    plain_index: dict[tuple[int, int, str], Any] = {}
    for meta in plain_snaps.snapshots:
        plain_index[(meta.prompt_idx, meta.layer, meta.kind)] = meta

    accumulators: dict[tuple[int, str], _OpAccumulator] = {}
    n_paired = 0
    n_skipped_unpaired = 0
    rng = np.random.default_rng(0)

    for meta in snapshots.snapshots:
        key = (meta.prompt_idx, meta.layer, meta.kind)
        plain_meta = plain_index.get(key)
        if plain_meta is None:
            n_skipped_unpaired += 1
            continue
        try:
            u = snapshots.get_operand(meta, strip_shield=True).detach().cpu().numpy().astype(np.float32)
            h = plain_snaps.get_operand(plain_meta, strip_shield=True).detach().cpu().numpy().astype(np.float32)
        except Exception:
            n_skipped_unpaired += 1
            continue
        if u.size == 0 or h.size == 0:
            n_skipped_unpaired += 1
            continue
        # Operand widths must match for the column-Gram comparison;
        # for QKV/FFN paths H and U share `d` by construction.
        if u.shape != h.shape:
            n_skipped_unpaired += 1
            continue
        # Feature-axis subsample to bound the col-Gram O(d²) cost.
        if max_features is not None and u.shape[1] > max_features:
            feat_sel = rng.choice(u.shape[1], size=max_features, replace=False)
            feat_sel.sort()
            u = u[:, feat_sel]
            h = h[:, feat_sel]
        op_key = (meta.layer, meta.kind)
        acc = accumulators.setdefault(op_key, _OpAccumulator.empty(meta.layer, meta.kind))
        acc.col_gram_errors.append(_col_gram_error(u, h))
        acc.spectrum_errors.append(_row_gram_spectrum_error(u, h))
        acc.hungarian_errors.append(_hungarian_row_gram_error(u, h))
        n_paired += 1

    if n_paired == 0:
        return AttackResult(
            attack="gram_error",
            condition=snapshots.condition,
            model_id=snapshots.model_id,
            n_prompts=snapshots.n_prompts(),
            n_train=0,
            n_test=0,
            ttrsr_top1=None,
            ttrsr_top10=None,
            risk_level="unknown",
            primary_metric_name="hungarian_row_gram_error",
            extra={
                "note": "no paired (prompt, layer, kind) tuples found between protected and plain snapshots",
                "n_skipped_unpaired": n_skipped_unpaired,
            },
        )

    # Aggregate: median across all paired snapshots.
    all_col = np.array([v for acc in accumulators.values() for v in acc.col_gram_errors], dtype=np.float64)
    all_spec = np.array([v for acc in accumulators.values() for v in acc.spectrum_errors], dtype=np.float64)
    all_hung = np.array([v for acc in accumulators.values() for v in acc.hungarian_errors], dtype=np.float64)

    median_hung = float(np.nanmedian(all_hung))
    per_op = {
        f"layer{layer:03d}.{kind}": {
            "n": len(acc.col_gram_errors),
            "col_gram_error_median": float(np.nanmedian(acc.col_gram_errors)),
            "row_gram_spectrum_error_median": float(np.nanmedian(acc.spectrum_errors)),
            "hungarian_row_gram_error_median": float(np.nanmedian(acc.hungarian_errors)),
        }
        for (layer, kind), acc in sorted(accumulators.items())
    }

    return AttackResult(
        attack="gram_error",
        condition=snapshots.condition,
        model_id=snapshots.model_id,
        n_prompts=snapshots.n_prompts(),
        n_train=0,
        n_test=n_paired,
        # We repurpose ttrsr_top1 as the primary numeric so existing
        # consumers (acceptance-gate threshold checks) can read it
        # without a schema change. The semantic name is in
        # `primary_metric_name`.
        ttrsr_top1=median_hung,
        ttrsr_top10=None,
        risk_level=classify_risk_level(1.0 - median_hung),
        primary_metric_name="hungarian_row_gram_error",
        extra={
            "col_gram_error_median": float(np.nanmedian(all_col)),
            "row_gram_spectrum_error_median": float(np.nanmedian(all_spec)),
            "hungarian_row_gram_error_median": median_hung,
            "n_paired": n_paired,
            "n_skipped_unpaired": n_skipped_unpaired,
            "per_op": per_op,
            "note": (
                "primary metric = hungarian_row_gram_error_median; lower means "
                "more leakage. C0 plain ≈ 0 by definition. For HD₃ default "
                "promotion, C3 must be within 20 % of C2 on this metric (round-3 "
                "B.3 gate tolerance)."
            ),
        },
    )


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--snapshot-root", required=True, type=Path)
    p.add_argument("--snapshot-basename", required=True)
    p.add_argument("--plain-basename", default="c0_plain")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    # Local import so the module imports cleanly without snapshots_loader
    # on the import path (smoke pytest does that).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from snapshots_loader import SnapshotSet  # type: ignore  # noqa: E402

    snaps = SnapshotSet.open(args.snapshot_basename, root=args.snapshot_root)
    plain = SnapshotSet.open(args.plain_basename, root=args.snapshot_root)
    result = run(snaps, plain_snaps=plain)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2))
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    _cli()

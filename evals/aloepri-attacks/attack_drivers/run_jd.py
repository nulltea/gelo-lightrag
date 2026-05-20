"""JD — Joint Diagonalisation across multiple masked observations.

Adapted from Belouchrani et al. (1997), *A blind source separation
technique using second-order statistics*. The original SOBI uses
time-lagged covariance matrices of a single observation sequence;
for the GELO threat model we adapt it to **observation-stack** JD:
across T snapshots ``U_t = A_t · H_t`` (different prompts, same
layer/kind), jointly diagonalise the stack of covariance matrices
``C_t = (1/d) · U_t · U_tᵀ`` to recover a single "averaged"
demixing matrix.

What this tests in GELO's threat model:

* If the per-forward-pass mask ``A_t`` is **fresh** (independent
  across t), the stack of ``C_t`` matrices does **not** share a
  common eigen-structure — joint diagonalisation has no shared
  rotation to find, and the attack fails for all T.
* If ``A_t`` is **correlated** across t (mask reuse, partial
  rotation, or any leakage that ties mask materials between
  forward passes), the stack does share structure and JD's
  recovery improves with T.

The reported curve ``p95_cosine(T)`` for T ∈ {1, 2, 4, 8, 16}
shows how recovery scales with observation count. For a
correctly-implemented per-forward-pass mask the curve should be
**flat at the per-T baseline** (= JADE-quality recovery on a
single observation). Curve climbing with T = leakage from mask
reuse.

For the round-3 B.3 gate: HD₃ passes if its JD curve matches
Haar's within ±0.05 at every T, including T = 16. A divergence
at large T would indicate that HD₃'s discrete orbit has lower
effective entropy than Haar's continuous measure (i.e. the
``2^{3·s}`` mask space is concentrated in directions that JD can
exploit).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import AttackResult, classify_risk_level

# Reuse the joint-diag primitive from JADE — same Jacobi sweep,
# different matrices going in.
from .run_jade import _joint_diag, _p95_cosine_with_hungarian


# Default observation counts to sweep. T = 1 reads as the single-
# observation baseline (≈ JADE-quality); growing T tests for mask
# correlation leakage.
DEFAULT_T_VALUES: tuple[int, ...] = (1, 2, 4, 8, 16)


@dataclass
class _OpAccumulator:
    layer: int
    kind: str
    p95_per_t: dict[int, list[float]]

    @classmethod
    def empty(cls, layer: int, kind: str, t_values: tuple[int, ...]) -> "_OpAccumulator":
        return cls(layer=layer, kind=kind, p95_per_t={t: [] for t in t_values})


def _whiten_stack(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Whiten a stack of (T, s, d) observations to (T, m, d) using
    the SUMMED covariance ``Σ_t U_t U_tᵀ / (T · d)`` so the same
    whitener applies to every observation.

    Returns ``(Y_stack, W)`` where ``Y_t = W · U_t`` has approximately
    identity sum-covariance.
    """
    t_count, s, d = stack.shape
    # Mean-center each observation along the feature axis.
    centered = stack - stack.mean(axis=2, keepdims=True)
    # Stack-summed covariance on rows.
    cov = np.zeros((s, s), dtype=np.float64)
    for t in range(t_count):
        cov += centered[t] @ centered[t].T
    cov /= max(t_count * d, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(-eigvals)
    eigvals = np.maximum(eigvals[order][:s], 1e-12)
    eigvecs = eigvecs[:, order][:, :s]
    w = (eigvecs / np.sqrt(eigvals)[None, :]).T  # (s, s) → identity after whitening
    y_stack = np.stack([(w @ centered[t]).astype(np.float32) for t in range(t_count)], axis=0)
    return y_stack, w


def _stack_covariances(y_stack: np.ndarray) -> np.ndarray:
    """Build the JD matrix stack from whitened observations.

    For each t, take ``Q_t = (Y_t · Y_tᵀ) / d``. The stack
    ``(Q_1, ..., Q_T)`` is what we jointly diagonalise.
    """
    t_count, s, d = y_stack.shape
    q = np.zeros((t_count, s, s), dtype=np.float64)
    for t in range(t_count):
        q[t] = (y_stack[t] @ y_stack[t].T) / max(d, 1)
    # Symmetrise for numerical safety (covariance matrices are
    # symmetric by construction, but f32 noise can drift).
    q = 0.5 * (q + q.transpose(0, 2, 1))
    return q


def _jd_demix(stack: np.ndarray) -> np.ndarray | None:
    """Joint-Diagonalisation demixing across a (T, s, d) stack.

    Returns demixing matrix B (s × s) such that for each t,
    ``B · U_t`` is as close as possible to a permuted version of
    ``H_t``. Returns None if eigendecomposition fails.
    """
    t_count, s, d = stack.shape
    if t_count == 0 or s < 2 or d < 2 * s:
        return None
    try:
        y_stack, w = _whiten_stack(stack)
        q = _stack_covariances(y_stack)
        rot = _joint_diag(q, max_sweeps=50, tol=1e-7)
        return rot.T @ w
    except np.linalg.LinAlgError:
        return None


def run(
    snapshots,
    *,
    plain_snaps=None,
    t_values: tuple[int, ...] = DEFAULT_T_VALUES,
    max_dim: int = 64,
    max_features: int = 256,
    **_kwargs,
) -> AttackResult:
    """Joint-diagonalisation-across-observations driver.

    Args:
      snapshots: protected condition SnapshotSet.
      plain_snaps: paired C0 plain SnapshotSet.
      t_values: tuple of observation-stack sizes to sweep.
      max_dim: row subsample cap (matches run_jade's bound).
      max_features: feature-axis (d) cap — per-stack covariance is
        ``O(T · s · d²)`` so subsampling d directly bounds runtime.
    """
    if plain_snaps is None and snapshots.condition == "c0_plain":
        plain_snaps = snapshots

    if plain_snaps is None:
        return AttackResult(
            attack="jd",
            condition=snapshots.condition,
            model_id=snapshots.model_id,
            n_prompts=snapshots.n_prompts(),
            n_train=0,
            n_test=0,
            ttrsr_top1=None,
            ttrsr_top10=None,
            risk_level="unknown",
            primary_metric_name="jd_p95_cosine_at_t_max",
            extra={"note": "jd needs paired plaintext snapshots."},
        )

    plain_index: dict[tuple[int, int, str], Any] = {}
    for meta in plain_snaps.snapshots:
        plain_index[(meta.prompt_idx, meta.layer, meta.kind)] = meta

    # Group snapshots by (layer, kind). Within each group, observations
    # share the same `s` and `d`, so they can be stacked and jointly
    # diagonalised. Prompts within a (layer, kind) bucket use different
    # mask realisations (fresh per-forward A) — exactly the BSS multi-
    # observation regime.
    buckets: dict[tuple[int, str], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    n_unpaired = 0
    rng = np.random.default_rng(0)

    for meta in snapshots.snapshots:
        key = (meta.prompt_idx, meta.layer, meta.kind)
        plain_meta = plain_index.get(key)
        if plain_meta is None:
            n_unpaired += 1
            continue
        try:
            u = snapshots.get_operand(meta, strip_shield=True).detach().cpu().numpy().astype(np.float32)
            h = plain_snaps.get_operand(plain_meta, strip_shield=True).detach().cpu().numpy().astype(np.float32)
        except Exception:
            n_unpaired += 1
            continue
        if u.shape != h.shape or u.size == 0:
            n_unpaired += 1
            continue
        s = u.shape[0]
        if s > max_dim:
            sel = rng.choice(s, size=max_dim, replace=False)
            sel.sort()
            u = u[sel]
            h = h[sel]
        if u.shape[0] < 4:
            n_unpaired += 1
            continue
        if max_features is not None and u.shape[1] > max_features:
            feat_sel = rng.choice(u.shape[1], size=max_features, replace=False)
            feat_sel.sort()
            u = u[:, feat_sel]
            h = h[:, feat_sel]
        buckets[(meta.layer, meta.kind)].append((u, h))

    accumulators: dict[tuple[int, str], _OpAccumulator] = {}
    n_evaluated = 0

    for (layer, kind), pairs in buckets.items():
        # Filter to a common (s, d) so we can stack. Hidden-state shapes
        # for the same (layer, kind) should already match across prompts
        # because we strip_shield to n_data and use the same prompt
        # truncation cap — but check defensively.
        if not pairs:
            continue
        ref_shape = pairs[0][0].shape
        pairs = [p for p in pairs if p[0].shape == ref_shape]
        if not pairs:
            continue

        acc = _OpAccumulator.empty(layer, kind, t_values)
        for t_target in t_values:
            # Build as many non-overlapping length-T stacks as the
            # bucket allows; report per-snapshot p95 cosine over the
            # union of all stacks at this T.
            for stack_start in range(0, len(pairs) - t_target + 1, t_target):
                slice_pairs = pairs[stack_start : stack_start + t_target]
                u_stack = np.stack([p[0] for p in slice_pairs], axis=0)
                h_stack = np.stack([p[1] for p in slice_pairs], axis=0)
                b = _jd_demix(u_stack)
                if b is None:
                    continue
                # Apply the SAME demixing B to every observation in the
                # stack and average the Hungarian-aligned cosine.
                for t in range(u_stack.shape[0]):
                    s_hat = b @ u_stack[t]
                    p95 = _p95_cosine_with_hungarian(s_hat, h_stack[t])
                    if p95 == p95:  # not NaN
                        acc.p95_per_t[t_target].append(p95)
                n_evaluated += t_target

        accumulators[(layer, kind)] = acc

    if not accumulators:
        return AttackResult(
            attack="jd",
            condition=snapshots.condition,
            model_id=snapshots.model_id,
            n_prompts=snapshots.n_prompts(),
            n_train=0,
            n_test=0,
            ttrsr_top1=None,
            ttrsr_top10=None,
            risk_level="unknown",
            primary_metric_name="jd_p95_cosine_at_t_max",
            extra={"note": "no usable (layer, kind) buckets",
                   "n_unpaired": n_unpaired},
        )

    # Aggregate per-T medians across (layer, kind).
    per_t_medians: dict[int, float] = {}
    for t in t_values:
        all_vals = [v for acc in accumulators.values() for v in acc.p95_per_t[t]]
        per_t_medians[t] = float(np.nanmedian(all_vals)) if all_vals else float("nan")

    # Primary scalar: the recovery at the largest T (the most adversary-
    # favourable). A flat curve = good defence; a climbing curve = leakage.
    t_max = max(t_values)
    primary = per_t_medians.get(t_max, float("nan"))

    per_op = {
        f"layer{layer:03d}.{kind}": {
            "p95_per_t": {
                str(t): (float(np.nanmedian(acc.p95_per_t[t])) if acc.p95_per_t[t] else None)
                for t in t_values
            },
            "n_samples_per_t": {
                str(t): len(acc.p95_per_t[t]) for t in t_values
            },
        }
        for (layer, kind), acc in sorted(accumulators.items())
    }

    return AttackResult(
        attack="jd",
        condition=snapshots.condition,
        model_id=snapshots.model_id,
        n_prompts=snapshots.n_prompts(),
        n_train=0,
        n_test=n_evaluated,
        ttrsr_top1=primary,
        ttrsr_top10=None,
        risk_level=classify_risk_level(primary),
        primary_metric_name="jd_p95_cosine_at_t_max",
        extra={
            "t_values": list(t_values),
            "p95_cosine_per_t_median": {str(t): v for t, v in per_t_medians.items()},
            "primary_t_max": t_max,
            "primary_p95_cosine": primary,
            "max_dim": max_dim,
            "n_evaluated": n_evaluated,
            "n_unpaired": n_unpaired,
            "per_op": per_op,
            "note": (
                "primary = median p95 cosine at T = "
                f"{t_max} observations. Curve `p95_cosine_per_t_median` shows "
                "how recovery scales with stack size — flat = good (mask is "
                "fresh per forward); climbing = leakage from mask reuse. HD₃ "
                "default-flip requires C3 within ±0.05 of C2 at EVERY T."
            ),
        },
    )


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--snapshot-root", required=True, type=Path)
    p.add_argument("--snapshot-basename", required=True)
    p.add_argument("--plain-basename", default="c0_plain")
    p.add_argument("--t-values", default="1,2,4,8,16",
                   help="comma-separated stack sizes for the JD sweep")
    p.add_argument("--max-dim", type=int, default=64)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from snapshots_loader import SnapshotSet  # type: ignore  # noqa: E402

    t_values = tuple(int(x) for x in args.t_values.split(",") if x)
    snaps = SnapshotSet.open(args.snapshot_basename, root=args.snapshot_root)
    plain = SnapshotSet.open(args.plain_basename, root=args.snapshot_root)
    result = run(snaps, plain_snaps=plain, t_values=t_values, max_dim=args.max_dim)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2))
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    _cli()

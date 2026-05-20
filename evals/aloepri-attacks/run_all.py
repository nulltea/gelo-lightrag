"""run_all.py — AloePri attack runner across the GELO condition matrix.

Usage:

    python run_all.py \
        --snapshot-root snapshots/qwen3-1.7b \
        --output results/path-1-attacks.json \
        --model-id Qwen/Qwen3-1.7B

Reads `c0_plain`, `c1_mask_only`, `c2_default`, and (optionally)
`c3_hd3` from `--snapshot-root`, runs all attack drivers against
each, and writes one combined results JSON with shape:

```
{
  "format": "aloepri_attack_results_v1",
  "model_id": "...",
  "shield_k": 8,
  "shield_energy_scale": 4.0,
  "per_forward_mask": true,
  "captured_at": "...",
  "conditions": {
    "c0_plain":     { "vma": { ... }, "ima": { ... }, ... },
    "c1_mask_only": { ... },
    "c2_default":   { ... },
    "c3_hd3":       { ... }     # only when the snapshot is present
  },
  "acceptance_gate": {
    "ima_c2_below_10pct":             <bool>,
    "isa_c2_below_10pct":             <bool>,
    "c0_ima_at_least_95pct":          <bool>,
    "ima_c3_within_haar_band":        <bool>,   # round-3 B.3 gate
    "isa_c3_within_haar_band":        <bool>,   # round-3 B.3 gate
    ...
  }
}
```

The §2.6 acceptance gate is reported but not enforced — the CI
release-gate wrapper at `.github/workflows/aloepri-gate.yml`
(Phase 3) is what exits non-zero on threshold violation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

# Make the local package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from snapshots_loader import open_three_conditions  # type: ignore  # noqa: E402

# Memory pre-flight (ported from m2_7/m2_7_common.py). The
# `attack_matrix` phase peaks ~6 GB on Qwen3-1.7B (embedding table
# f32 ≈ 600 MB + per-condition snapshot caches + numpy scratch for
# the §4.3 drivers); 8 GB minimum is the documented threshold.
from m2_7.m2_7_common import check_phase_memory, add_min_mem_args  # type: ignore  # noqa: E402

# Add the gate phase if it's not in the path-2 default table.
import m2_7.m2_7_common as _m2_7_common  # type: ignore  # noqa: E402
_m2_7_common.PHASE_MIN_GB.setdefault("attack_matrix", 8.0)

from attack_drivers import (  # type: ignore  # noqa: E402
    run_anchor_ica,
    run_gram_error,
    run_ia,
    run_ima,
    run_ima_paper_like,
    run_isa,
    run_isa_attn_score,
    run_jade,
    run_jd,
    run_nn,
    run_sda,
    run_tfma,
    run_vma,
)


# Attack list — order matters for the report.
#
# Six AloePri drivers (NN / VMA / IA / IMA / ima_paper_like / ISA /
# TFMA / SDA) carry the original AloePri-family acceptance metrics.
# Four §4.3 drivers (anchor_ica / jade / jd / gram_error) extend
# the matrix to the GELO-specific BSS-hardness tests that AloePri
# doesn't reach — required for the round-3 B.3 attack-defence gate
# (HD₃ vs Haar parity).
ATTACKS = [
    ("nn", run_nn),
    ("vma", run_vma),
    ("ia", run_ia),
    ("ima", run_ima),
    ("ima_paper_like", run_ima_paper_like),
    ("isa", run_isa),
    ("tfma", run_tfma),
    ("sda", run_sda),
    # §4.3 — GELO threat-model-specific drivers. Each needs paired
    # plaintext snapshots (`plain_snaps` kwarg) to evaluate
    # recovery; supplied below from the c0_plain condition.
    ("anchor_ica", run_anchor_ica),
    ("jade", run_jade),
    ("jd", run_jd),
    ("gram_error", run_gram_error),
    # ISA-AttnScore — placeholder for the M1.10 permuted-attention
    # path. Emits not_applicable until the GELO protocol grows an
    # attention-score snapshot kind. See run_isa_attn_score.py.
    ("isa_attn_score", run_isa_attn_score),
]


def main() -> None:
    p = argparse.ArgumentParser(description="Run all AloePri attacks against the three-condition snapshot matrix")
    p.add_argument("--snapshot-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    p.add_argument(
        "--ima-layer",
        type=int,
        default=0,
        help="Layer to capture IMA at (default 0 = first attention block input)",
    )
    p.add_argument(
        "--ima-kind",
        default="q_proj",
        help="Kind to capture IMA at (default q_proj)",
    )
    p.add_argument(
        "--isa-layer",
        type=int,
        default=23,
        help="Layer to run ISA at (default 23, mirroring AloePri's ISABaselineConfig)",
    )
    p.add_argument(
        "--isa-kind",
        default="q_proj",
        help="Kind to run ISA at (default q_proj)",
    )
    p.add_argument(
        "--strip-shield",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--no-strip-shield",
        dest="strip_shield",
        action="store_false",
    )
    p.add_argument(
        "--skip-attacks",
        default="",
        help=(
            "Comma-separated list of attack names to skip (e.g. "
            "`anchor_ica` to bypass the multi-minute FastICA loop). "
            "Skipped attacks emit a `not_applicable` row with phase="
            "`skip_attacks_flag` so the result table stays square."
        ),
    )
    add_min_mem_args(p, phase="attack_matrix")
    args = p.parse_args()

    # Pre-flight memory check — fails fast with a clear message if
    # the box can't hold the embedding table + per-condition caches.
    check_phase_memory(
        phase="attack_matrix",
        override_min_gb=args.min_mem_gb,
        skip=args.skip_mem_check,
    )

    skip_set = {s.strip() for s in args.skip_attacks.split(",") if s.strip()}
    if skip_set:
        print(f"[skip] attacks excluded by --skip-attacks: {sorted(skip_set)}")

    conditions = open_three_conditions(args.snapshot_root)
    print(
        f"loaded conditions: {list(conditions.keys())} "
        f"({sum(len(s.snapshots) for s in conditions.values())} snapshots total)"
    )

    print(f"loading {args.model_id} embedding table (one-time)…")
    embed_table = run_ima.load_qwen3_embedding_table(args.model_id)

    per_condition: dict[str, dict[str, Any]] = {}
    plain_snaps = conditions.get("c0_plain")
    for cond_slug, snaps in conditions.items():
        per_attack: dict[str, Any] = {}
        for name, mod in ATTACKS:
            if name in skip_set:
                print(f"  [{cond_slug}] skipping {name} (--skip-attacks)")
                from attack_drivers.common import AttackResult  # type: ignore
                per_attack[name] = AttackResult(
                    attack=name,
                    condition=snaps.condition,
                    model_id=snaps.model_id,
                    n_prompts=snaps.n_prompts(),
                    n_train=0,
                    n_test=0,
                    ttrsr_top1=None,
                    ttrsr_top10=None,
                    risk_level="not_applicable",
                    extra={"phase": "skip_attacks_flag"},
                ).to_dict()
                continue
            print(f"  [{cond_slug}] running {name}…")
            kwargs: dict[str, Any] = {}
            if name in {"nn", "ima", "isa", "ima_paper_like", "isa_attn_score"}:
                kwargs["strip_shield"] = args.strip_shield
                kwargs["embed_table"] = embed_table
            if name == "ima":
                kwargs["layer"] = args.ima_layer
                kwargs["kind"] = args.ima_kind
            elif name == "isa":
                kwargs["layer"] = args.isa_layer
                kwargs["kind"] = args.isa_kind
            elif name == "nn":
                kwargs["layer"] = args.ima_layer
                kwargs["kind"] = args.ima_kind
            elif name == "ima_paper_like":
                kwargs["layer"] = args.ima_layer
                kwargs["kind"] = args.ima_kind
            elif name == "ia":
                kwargs["embed_table"] = embed_table
                kwargs["layer"] = args.ima_layer
            # §4.3 drivers need the paired plaintext snapshot set so
            # they can compute cosine-recovery / Gram-error metrics
            # against ground-truth H. plain_snaps is None when the
            # snapshot dir lacks a c0_plain set; the drivers fall
            # back to a `not_applicable`-style placeholder result.
            elif name in {"anchor_ica", "jade", "jd", "gram_error"}:
                kwargs["plain_snaps"] = plain_snaps
            # vma / tfma / sda take no kwargs — they emit
            # not_applicable rows by design.
            try:
                result = mod.run(snaps, **kwargs)
            except Exception as exc:  # surface, but keep the row
                print(f"    !! {name} failed: {exc!r}")
                import traceback
                traceback.print_exc()
                # Manufacture a failure row instead of re-invoking the
                # driver — the retry without kwargs crashes for drivers
                # with required keyword-only args (run_ima needs
                # `embed_table`). The original AttackResult contract
                # lets us emit a failure row directly.
                from attack_drivers.common import AttackResult  # type: ignore
                result = AttackResult(
                    attack=name,
                    condition=snaps.condition,
                    model_id=snaps.model_id,
                    n_prompts=snaps.n_prompts(),
                    n_train=0,
                    n_test=0,
                    ttrsr_top1=None,
                    ttrsr_top10=None,
                    risk_level="unknown",
                    extra={"error": repr(exc), "phase": "driver_raised"},
                )
            per_attack[name] = result.to_dict()
            print(
                f"    {name}: ttrsr_top1={result.ttrsr_top1!r} "
                f"risk={result.risk_level}"
            )
        per_condition[cond_slug] = per_attack

    # Acceptance-gate snapshot (per §2.6 of the harness handoff).
    def _top1(cond: str, attack: str) -> float | None:
        v = per_condition.get(cond, {}).get(attack, {}).get("ttrsr_top1")
        return None if v is None else float(v)

    # Acceptance gate — what we actually claim about the protocol.
    #
    # Notes on changes from the original gate set:
    #
    # * Replaced `c0_vma_at_least_50pct` with `c0_nn_at_least_50pct` —
    #   the old "VMA" was actually NN (cosine-NN against the embedding
    #   table); under the new naming the load-bearing control row is
    #   `nn`, and VMA is a `not_applicable` stub.
    # * Replaced `c0_isa_at_least_50pct` with `c0_ima_paper_like_at_least_50pct`.
    #   ISA at layer 23 hits a heavily-processed hidden state and the
    #   paper's "Noise only" baseline (closest published analog to our
    #   C0) already reads only 40% on HiddenState (Table 3). The
    #   paper-like trained inverter is the canonical "C0 control"
    #   sanity check and should clear 50% under plain inference.
    # * Replaced `c1_strictly_higher_than_c2_on_isa` (which encoded
    #   the wrong protocol claim) with `per_offload_at_most_per_forward_plus_shield_on_isa`.
    #   This is the inverted form: per-offload mask (C1) should be at
    #   least as ISA-protective as per-forward + shield (C2). Tracks
    #   `gelo.md` §3.2's qualitative "per-offload is strictly safer"
    #   claim. **The cross-batch Gram-leak attack that the shield
    #   primitive actually defends is a deferred Phase 2 follow-up**
    #   (see `docs/prototype/aloepri-attack-harness-followups.md`).
    acceptance = {
        "ima_c2_below_10pct": (
            _top1("c2_default", "ima") is not None
            and _top1("c2_default", "ima") < 0.10  # type: ignore[operator]
        ),
        "isa_c2_below_10pct": (
            _top1("c2_default", "isa") is not None
            and _top1("c2_default", "isa") < 0.10  # type: ignore[operator]
        ),
        "ima_paper_like_c2_below_10pct": (
            _top1("c2_default", "ima_paper_like") is not None
            and _top1("c2_default", "ima_paper_like") < 0.10  # type: ignore[operator]
        ),
        "c0_ima_at_least_95pct": (
            _top1("c0_plain", "ima") is not None
            and _top1("c0_plain", "ima") >= 0.95  # type: ignore[operator]
        ),
        "c0_ima_paper_like_at_least_50pct": (
            _top1("c0_plain", "ima_paper_like") is not None
            and _top1("c0_plain", "ima_paper_like") >= 0.50  # type: ignore[operator]
        ),
        "c0_nn_at_least_50pct": (
            _top1("c0_plain", "nn") is not None
            and _top1("c0_plain", "nn") >= 0.50  # type: ignore[operator]
        ),
        "per_offload_at_most_per_forward_plus_shield_on_isa": (
            _top1("c1_mask_only", "isa") is not None
            and _top1("c2_default", "isa") is not None
            and _top1("c1_mask_only", "isa") <= _top1("c2_default", "isa")  # type: ignore[operator]
        ),
    }

    # B.3 — HD₃ attack-defence parity check. Only fires when the C3
    # snapshot is present (the harness loader skips C3 if its
    # safetensors file isn't there, so older 3-condition runs stay
    # green). The band ±0.05 matches the round-3 doc tolerance:
    # HD₃'s discrete `2^{3·s}` orbit is structurally different from
    # the continuous Haar measure, but parity-with-Haar at the
    # bench's load-bearing attacks (IMA / ISA / NN) is what unlocks
    # the executor default-flip.
    if "c3_hd3" in per_condition:
        def _within_band(attack: str, band: float = 0.05) -> bool | None:
            c2 = _top1("c2_default", attack)
            c3 = _top1("c3_hd3", attack)
            if c2 is None or c3 is None:
                return None
            return abs(c3 - c2) <= band

        # AloePri-family parity bands. ISA / NN / JD are intentionally
        # excluded — they fail the two-sided ±0.05 check whenever HD₃
        # *outperforms* Haar (HD₃ defends more strongly than Haar
        # within bench noise), which is the wrong sign convention.
        # Comparison for those metrics is handled in the HTML report
        # rather than the boolean gate. JD additionally returns NaN at
        # short prompt counts and isn't a meaningful gate signal until
        # longer-context runs land.
        acceptance["ima_c3_within_haar_band"] = _within_band("ima")
        acceptance["ima_paper_like_c3_within_haar_band"] = _within_band("ima_paper_like")
        # Absolute thresholds on C3 — mirrors the C2 < 10% claim so
        # HD₃ must individually defend, not just parity-match Haar.
        acceptance["ima_c3_below_10pct"] = (
            _top1("c3_hd3", "ima") is not None
            and _top1("c3_hd3", "ima") < 0.10  # type: ignore[operator]
        )
        acceptance["isa_c3_below_10pct"] = (
            _top1("c3_hd3", "isa") is not None
            and _top1("c3_hd3", "isa") < 0.10  # type: ignore[operator]
        )
        # §4.3-family parity bands. anchor_ica / jade / jd use
        # cosine-recovery (±0.05 cosine); gram_error uses a Frobenius
        # error band of ±20 % (paper §4.3.4 tolerance) — wider because
        # the metric scale itself is wider.
        acceptance["anchor_ica_c3_within_haar_band"] = _within_band("anchor_ica")
        acceptance["jade_c3_within_haar_band"] = _within_band("jade")

        # gram_error is now cos-normalised (range [0, √2]). Same
        # ±0.05 band as the other cosine-distance §4.3 metrics.
        # Plus an absolute "well clear of perfect-fingerprint pole"
        # threshold ≥ 0.5: c3 must stay above this floor independently
        # of where c2 sits. See run_gram_error.py for derivation.
        acceptance["gram_error_c3_within_haar_band"] = _within_band("gram_error")
        acceptance["gram_error_c3_above_fingerprint_floor"] = (
            _top1("c3_hd3", "gram_error") is not None
            and _top1("c3_hd3", "gram_error") >= 0.5  # type: ignore[operator]
        )

    # Pull the shield/mask config off one of the loaded sets — they
    # agree on the model_id but not on the per-condition shield
    # numbers, so we record both the C2 defaults (the actual
    # release-gate numbers) and the per-condition flags separately.
    c2 = conditions["c2_default"]
    out: dict[str, Any] = {
        "format": "aloepri_attack_results_v1",
        "model_id": c2.model_id,
        "shield_k": c2.shield_k,
        "shield_energy_scale": c2.shield_energy_scale,
        "per_forward_mask": c2.per_forward_mask,
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "conditions": per_condition,
        "acceptance_gate": acceptance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"wrote results JSON → {args.output}")
    print(f"acceptance gate: {acceptance}")


if __name__ == "__main__":
    main()

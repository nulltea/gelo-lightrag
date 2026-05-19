"""Smoke tests for the per-attack drivers.

Runs IMA / ISA / VMA against the synthetic fixture (4 prompts × 6
tokens × 2 layers) and the three template-only drivers (TFMA / SDA
/ IA) to check the result-row shape. Doesn't assert *numerical*
TTRSR values — the synthetic fixture is too small to be meaningful;
the numerical claim lives in the §2.6 release-gate run against the
real Qwen3-1.7B snapshots.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attack_drivers import run_ia, run_ima, run_isa, run_sda, run_tfma, run_vma  # type: ignore  # noqa: E402


def test_ima_driver_returns_attack_result(synthetic_snapshot_set, synthetic_embed_table):
    r = run_ima.run(
        synthetic_snapshot_set,
        embed_table=synthetic_embed_table,
        layer=0,
        kind="q_proj",
        n_train=12,
        n_test=4,
        candidate_pool_size=32,
    )
    assert r.attack == "ima"
    assert r.condition == "c0_plain"
    # Either we got a metric, or we got a "not enough data" note.
    assert r.ttrsr_top1 is None or 0.0 <= r.ttrsr_top1 <= 1.0


def test_isa_driver_returns_attack_result(synthetic_snapshot_set, synthetic_embed_table):
    r = run_isa.run(
        synthetic_snapshot_set,
        embed_table=synthetic_embed_table,
        layer=1,
        kind="q_proj",
        n_train=12,
        n_test=4,
        candidate_pool_size=32,
    )
    assert r.attack == "isa"
    assert r.ttrsr_top1 is None or 0.0 <= r.ttrsr_top1 <= 1.0


def test_vma_driver_returns_attack_result(synthetic_snapshot_set, synthetic_embed_table):
    r = run_vma.run(
        synthetic_snapshot_set,
        embed_table=synthetic_embed_table,
        layer=0,
        kind="q_proj",
        eval_size=8,
    )
    assert r.attack == "vma"
    assert r.ttrsr_top1 is None or 0.0 <= r.ttrsr_top1 <= 1.0


@pytest.mark.parametrize(
    "mod, name",
    [(run_tfma, "tfma"), (run_sda, "sda"), (run_ia, "ia")],
)
def test_template_drivers_emit_not_applicable(synthetic_snapshot_set, mod, name):
    r = mod.run(synthetic_snapshot_set)
    assert r.attack == name
    assert r.ttrsr_top1 is None
    assert r.risk_level == "not_applicable"

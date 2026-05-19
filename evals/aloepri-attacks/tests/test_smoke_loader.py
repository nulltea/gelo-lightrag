"""Smoke tests for the Python snapshot loader.

Avoids needing the Rust capture binary or Qwen3 weights — the
synthetic fixture in `conftest.py` builds a 16-snapshot fixture on
disk and validates that every accessor returns the expected shape
+ value pattern.
"""

from __future__ import annotations

import torch


def test_loader_reads_synthetic_set(synthetic_snapshot_set):
    s = synthetic_snapshot_set
    assert s.condition == "c0_plain"
    assert s.shield_k == 0
    assert s.n_prompts() == 4
    assert s.captured_layers == [0, 1]
    assert s.captured_kinds == ["k_proj", "q_proj"] or s.captured_kinds == ["q_proj", "k_proj"]
    assert len(s.snapshots) == 16


def test_per_prompt_layer_kind_returns_one_per_prompt(synthetic_snapshot_set):
    s = synthetic_snapshot_set
    pairs = s.per_prompt_layer_kind_tensors(layer=0, kind="q_proj")
    assert len(pairs) == 4
    seen_prompts = set()
    for prompt_idx, op in pairs:
        seen_prompts.add(prompt_idx)
        assert op.shape == (6, 8)
        # The Rust fixture seeded operand[p, l, k] = base + 100*p + 10*l + (k==k_proj)
        assert op[0, 0].item() == 100 * prompt_idx
    assert seen_prompts == {0, 1, 2, 3}


def test_select_filters_by_dimension(synthetic_snapshot_set):
    s = synthetic_snapshot_set
    only_layer1 = s.select(layer=1)
    assert all(snap.layer == 1 for snap in only_layer1)
    assert len(only_layer1) == 8  # 4 prompts × 2 kinds

    only_q_prompt2 = s.select(layer=0, kind="q_proj", prompt_idx=2)
    assert len(only_q_prompt2) == 1
    assert only_q_prompt2[0].prompt_idx == 2


def test_get_operand_strips_shield_correctly(synthetic_snapshot_set):
    # Synthetic fixture has shield_k=0 so strip_shield is a no-op,
    # but the API contract is exercised here so a future fixture
    # with shield_k>0 catches regressions.
    s = synthetic_snapshot_set
    snap = s.snapshots[0]
    full = s.get_operand(snap, strip_shield=False)
    stripped = s.get_operand(snap, strip_shield=True)
    assert torch.equal(full, stripped)
    assert full.shape == (6, 8)

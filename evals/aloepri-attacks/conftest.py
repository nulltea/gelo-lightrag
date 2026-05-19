"""Pytest fixtures for the AloePri attack-resistance harness.

The fixtures here keep the smoke tests fast: a synthetic snapshot
set lets the loader, the AloePri ridge primitives, and the run_all
pipeline be exercised end-to-end without needing a real Qwen3
checkpoint or the Rust capture binary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


# Make local modules importable when pytest is run from any cwd.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def synthetic_snapshot_set(tmp_path):
    """A 4-prompt × 2-layer × 2-kind synthetic snapshot set on disk."""
    from snapshots_loader import synthetic_set_for_tests  # type: ignore

    return synthetic_set_for_tests(tmp_path)


@pytest.fixture
def synthetic_embed_table():
    """A tiny embedding table compatible with the synthetic snapshot
    set's hidden dimension. Used by the attack-driver smoke tests so
    we don't need to download Qwen3 weights for CI.
    """
    import torch

    vocab = 64
    hidden = 8
    return torch.randn(vocab, hidden, generator=torch.Generator().manual_seed(0))

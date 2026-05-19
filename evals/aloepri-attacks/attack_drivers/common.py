"""Shared helpers for the AloePri attack drivers.

Centralises sys.path management for the vendored AloePri primitives,
the TTRSR metric definition, and the tokenizer-loading boilerplate
that every attack driver needs.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


# Anchor everything to the repo root so the path injection works
# regardless of cwd. The eval harness lives at
# `<repo>/evals/aloepri-attacks/`; vendored AloePri at
# `<repo>/vendor/aloepri-py/`.
REPO_ROOT = Path(__file__).resolve().parents[3]
ALOEPRI_VENDORED_ROOT = REPO_ROOT / "vendor" / "aloepri-py"


def install_aloepri_path() -> None:
    """Insert the vendored AloePri repo into `sys.path` so its
    `src.security_qwen.*` imports resolve. Safe to call repeatedly.
    """
    aloepri_str = str(ALOEPRI_VENDORED_ROOT)
    if aloepri_str not in sys.path:
        sys.path.insert(0, aloepri_str)


# Cached single-module loads for the AloePri attack files. Going
# through `from src.security_qwen.<name> import …` triggers the
# package `__init__.py`, which eagerly imports every attack file
# (including ones that need `transformers`, `safetensors`, and the
# Qwen tokenizer). Most of our drivers only need three pure-torch
# helper functions from `ima.py`; loading the file directly via
# importlib.util skips the package init and keeps the harness
# importable on a minimal-deps machine.
_LOADED_VENDOR: dict[str, types.ModuleType] = {}


class _SimpleNamespace:
    """Fallback class for `from src.security_qwen.schema import X` style
    imports that we don't actually call."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def _unsupported_stub(name: str):
    """Build a callable that raises a clear error if the AloePri
    advanced entry point is reached. The harness only uses the pure
    PyTorch primitives (`_fit_ridge_regressor`, `_predict_ridge`,
    `_evaluate_inversion_predictions`); anything else would mean a
    driver outgrew the minimal stub set."""

    def _stub(*_args: Any, **_kwargs: Any):
        raise RuntimeError(
            f"AloePri helper {name!r} is stubbed in the harness — "
            "if your driver needs it, install the full vendored "
            "dependency set (transformers, tokenizers, etc.) and "
            "drop the stub registration in common.py."
        )

    return _stub


def _ensure_module_stub(
    name: str,
    attrs_iter: list[str] | None = None,
    *,
    attrs: dict[str, Any] | None = None,
) -> None:
    """Register a stub module under `name` in `sys.modules` so
    `from name import X` succeeds without the real dep installed.
    """
    # Don't clobber a real install.
    if name in sys.modules:
        try:
            real = sys.modules[name]
            if isinstance(real, types.ModuleType) and getattr(real, "__file__", None):
                return
        except Exception:
            return
    stub = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(stub, k, v)
    if attrs_iter:
        for k in attrs_iter:
            setattr(stub, k, _unsupported_stub(f"{name}.{k}"))
    # Mark as a package if dotted, so `from name import sub` works.
    if "." in name:
        stub.__path__ = []  # type: ignore[attr-defined]
    sys.modules[name] = stub


def load_aloepri_module(module_relpath: str) -> types.ModuleType:
    """Load one attack file from the vendored AloePri repo without
    running `src/security_qwen/__init__.py`.

    `module_relpath` is the file's path under `vendor/aloepri-py/`,
    e.g. `"src/security_qwen/ima.py"`. Cached: repeat calls return
    the same module object.
    """
    if module_relpath in _LOADED_VENDOR:
        return _LOADED_VENDOR[module_relpath]

    install_aloepri_path()
    path = ALOEPRI_VENDORED_ROOT / module_relpath
    if not path.exists():
        raise FileNotFoundError(f"vendored AloePri module not found: {path}")

    # Pre-register stub packages so `from src.security_qwen.x import y`
    # inside the file resolves without running the package __init__.
    # The file `ima.py` doesn't actually depend on its sibling attack
    # files — we just need `src` and `src.security_qwen` to be valid
    # importable names.
    for pkg_name, pkg_path in [
        ("src", ALOEPRI_VENDORED_ROOT / "src"),
        ("src.security_qwen", ALOEPRI_VENDORED_ROOT / "src" / "security_qwen"),
    ]:
        if pkg_name not in sys.modules:
            stub = types.ModuleType(pkg_name)
            stub.__path__ = [str(pkg_path)]  # type: ignore[attr-defined]
            sys.modules[pkg_name] = stub

    # Stub out heavy optional deps. The ridge primitives we actually
    # call don't use `transformers` / `tokenizers` / the AloePri
    # repo's own `model_loader.py` artefact-loading helpers — but
    # `ima.py` imports them at module scope. We provide minimal stubs
    # so the file evaluates; any later call into the stubbed module
    # raises AttributeError, which is the desired behaviour (caller
    # of those advanced AloePri entry points must install transformers
    # themselves).
    _ensure_module_stub("transformers", ["AutoConfig", "AutoModel", "AutoTokenizer"])
    _ensure_module_stub("tokenizers")
    _ensure_module_stub("src.defaults", attrs={
        "DEFAULT_MODEL_DIR": "model/Qwen2.5-0.5B-Instruct",
        "DEFAULT_PROMPTS": [],
        "DEFAULT_SEED": 20260323,
    })
    _ensure_module_stub("src.key_manager", attrs={
        "ordinary_token_ids": _unsupported_stub("ordinary_token_ids"),
    })
    _ensure_module_stub("src.model_loader", attrs={
        "resolve_torch_dtype": _unsupported_stub("resolve_torch_dtype"),
        "set_global_seed": _unsupported_stub("set_global_seed"),
    })
    _ensure_module_stub("src.security_qwen.artifacts", attrs={
        "resolve_security_target": _unsupported_stub("resolve_security_target"),
    })
    _ensure_module_stub("src.security_qwen.metrics", attrs={
        "classify_risk_level": _unsupported_stub("classify_risk_level"),
    })
    _ensure_module_stub("src.security_qwen.schema", attrs={
        "SecurityEvalTarget": _SimpleNamespace,
        "build_security_eval_payload": _unsupported_stub("build_security_eval_payload"),
    })
    _ensure_module_stub("src.stage_h_artifact", attrs={
        "load_stage_h_artifact": _unsupported_stub("load_stage_h_artifact"),
    })
    _ensure_module_stub("src.stage_h_pretrained", attrs={
        "load_stage_h_pretrained": _unsupported_stub("load_stage_h_pretrained"),
    })

    name = (
        module_relpath.replace("/", ".").removesuffix(".py")
    )  # "src.security_qwen.ima"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _LOADED_VENDOR[module_relpath] = module
    return module


def ttrsr(predicted_ids: np.ndarray, ground_truth_ids: np.ndarray) -> float:
    """Token-level Recovery Success Rate — the fraction of positions
    where the top-1 predicted id matches the ground-truth id.

    Both arrays must have the same length. Used uniformly across all
    six attack drivers so the numbers are directly comparable.
    """
    if predicted_ids.shape != ground_truth_ids.shape:
        raise ValueError(
            f"ttrsr: shape mismatch — predicted {predicted_ids.shape} vs ground "
            f"truth {ground_truth_ids.shape}"
        )
    if predicted_ids.size == 0:
        return 0.0
    return float((predicted_ids == ground_truth_ids).mean())


def topk_recovery(
    predicted_ids_topk: np.ndarray, ground_truth_ids: np.ndarray
) -> float:
    """Fraction of positions where the ground-truth id appears in the
    predicted top-k set. `predicted_ids_topk` has shape `(n, k)`,
    `ground_truth_ids` has shape `(n,)`.
    """
    if predicted_ids_topk.shape[0] != ground_truth_ids.shape[0]:
        raise ValueError("topk_recovery: row mismatch")
    if predicted_ids_topk.size == 0:
        return 0.0
    hits = (predicted_ids_topk == ground_truth_ids[:, None]).any(axis=1)
    return float(hits.mean())


@dataclass
class AttackResult:
    """One row in the per-condition results JSON.

    Mirrors the AloePri `qwen_security_eval_v1` schema (see
    `vendor/aloepri-py/src/security_qwen/schema.py`) so the result
    tables stay directly comparable to AloePri's published numbers.
    """

    attack: str
    condition: str
    model_id: str
    n_prompts: int
    n_train: int
    n_test: int
    ttrsr_top1: float | None
    ttrsr_top10: float | None
    risk_level: str
    primary_metric_name: str = "token_top1_recovery_rate"
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items()}


def classify_risk_level(primary_metric_value: float | None) -> str:
    """Match the AloePri thresholds in `metrics.py`:

    * ≥ 0.30 → "high"
    * ≥ 0.10 → "medium"
    * <  0.10 → "low"
    * None    → "unknown"
    """
    if primary_metric_value is None:
        return "unknown"
    if primary_metric_value >= 0.30:
        return "high"
    if primary_metric_value >= 0.10:
        return "medium"
    return "low"


def stack_prompt_observations(
    snapshots,  # SnapshotSet, not annotated to avoid circular import
    *,
    layer: int,
    kind: str,
    strip_shield: bool,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return `(X, y, prompt_lengths)` where:

    * `X` is a `(total_tokens, hidden_size)` float32 array — every
      data row from every prompt's (layer, kind) snapshot, stacked.
    * `y` is a `(total_tokens,)` int64 array — the corresponding
      ground-truth token ids (rebuilt from `prompt_token_ids`).
    * `prompt_lengths` is the per-prompt row count so callers can
      slice X/y by prompt without re-querying snapshots.

    Length mismatches between snapshot rows and tokenised prompt ids
    are tolerated by truncating to the shorter of the two — Qwen3's
    tokeniser can emit slightly different sequence lengths than the
    n_data the snapshot recorded under some BOS-handling paths, and
    failing loud here would mask the more interesting attack signal.
    """
    pairs = snapshots.per_prompt_layer_kind_tensors(
        layer=layer, kind=kind, strip_shield=strip_shield
    )
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    lengths: list[int] = []
    for prompt_idx, op in pairs:
        op_np = op.detach().cpu().numpy().astype(np.float32, copy=False)
        ids = np.asarray(snapshots.prompt_token_ids[prompt_idx], dtype=np.int64)
        n = min(op_np.shape[0], ids.shape[0])
        if n == 0:
            continue
        Xs.append(op_np[:n])
        ys.append(ids[:n])
        lengths.append(n)
    if not Xs:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            [],
        )
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0), lengths


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 0,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Deterministic train / val / test split.

    Auto-scales when the corpus is smaller than `n_train + n_val + n_test`:
    falls back to 70 / 15 / 15 of whatever's available with a minimum of
    4 / 1 / 1 rows. The val split is used by the multi-alpha ridge
    selection in `run_ima.py` / `run_isa.py` — AloePri reference
    pattern (`vendor/aloepri-py/src/security_qwen/ima.py:506-518`).
    """
    rng = np.random.default_rng(seed)
    n_total = X.shape[0]
    if n_total == 0:
        empty_x = np.zeros((0, X.shape[1] if X.ndim == 2 else 0), dtype=X.dtype)
        empty_y = np.zeros((0,), dtype=y.dtype)
        return empty_x, empty_y, empty_x, empty_y, empty_x, empty_y
    perm = rng.permutation(n_total)
    if n_train + n_val + n_test > n_total:
        n_train_eff = max(int(n_total * 0.7), min(4, n_total - 2))
        n_val_eff = max(int(n_total * 0.15), 1)
        n_test_eff = max(n_total - n_train_eff - n_val_eff, 1)
    else:
        n_train_eff = n_train
        n_val_eff = n_val
        n_test_eff = n_test
    tr = perm[:n_train_eff]
    va = perm[n_train_eff : n_train_eff + n_val_eff]
    te = perm[n_train_eff + n_val_eff : n_train_eff + n_val_eff + n_test_eff]
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def vocab_disjoint_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 0,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Vocab-disjoint train / val / test split — paper-faithful methodology.

    Unlike `train_val_test_split` (row-shuffle), this partitions the
    **distinct token ids** in `y` into three disjoint sets and assigns
    every (X[i], y[i]) pair to whichever set contains `y[i]`. Train
    and test never share a token id, so the ridge / inverter has to
    GENERALIZE rather than memorize per-token bias.

    Caps:
      * If the corpus has fewer distinct token ids than
        `n_train + n_val + n_test`, falls back to a 70/15/15 vocab
        split.
      * `n_train`/`n_val`/`n_test` are interpreted as **number of
        distinct token ids** per split, not rows. The returned arrays
        have *all* rows for those ids, so n_train_rows >> n_train_ids.

    Returns `(X_train, y_train, X_val, y_val, X_test, y_test)` —
    same shape as `train_val_test_split`.
    """
    rng = np.random.default_rng(seed)
    if X.shape[0] == 0:
        empty_x = np.zeros((0, X.shape[1] if X.ndim == 2 else 0), dtype=X.dtype)
        empty_y = np.zeros((0,), dtype=y.dtype)
        return empty_x, empty_y, empty_x, empty_y, empty_x, empty_y

    unique_ids, _ = np.unique(y, return_inverse=False), None
    if unique_ids.size == 0:
        empty_x = np.zeros((0, X.shape[1] if X.ndim == 2 else 0), dtype=X.dtype)
        empty_y = np.zeros((0,), dtype=y.dtype)
        return empty_x, empty_y, empty_x, empty_y, empty_x, empty_y

    n_total_ids = unique_ids.size
    if n_train + n_val + n_test > n_total_ids:
        # Auto-scale: 70/15/15 fallback. Reserve at least 2/1/1 ids.
        n_train_ids = max(int(n_total_ids * 0.70), min(2, n_total_ids - 2))
        n_val_ids = max(int(n_total_ids * 0.15), 1)
        n_test_ids = max(n_total_ids - n_train_ids - n_val_ids, 1)
    else:
        n_train_ids = n_train
        n_val_ids = n_val
        n_test_ids = n_test

    perm = rng.permutation(n_total_ids)
    train_ids = set(unique_ids[perm[:n_train_ids]].tolist())
    val_ids = set(
        unique_ids[perm[n_train_ids : n_train_ids + n_val_ids]].tolist()
    )
    test_ids = set(
        unique_ids[
            perm[n_train_ids + n_val_ids : n_train_ids + n_val_ids + n_test_ids]
        ].tolist()
    )

    # Boolean masks over rows.
    mask_train = np.isin(y, list(train_ids))
    mask_val = np.isin(y, list(val_ids))
    mask_test = np.isin(y, list(test_ids))
    return (
        X[mask_train], y[mask_train],
        X[mask_val], y[mask_val],
        X[mask_test], y[mask_test],
    )


def train_test_split(
    X: np.ndarray, y: np.ndarray, *, n_train: int, n_test: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic split. Returns `(X_train, y_train, X_test, y_test)`.

    Auto-scales when the corpus is smaller than `n_train + n_test`:
    falls back to an 80 / 20 split of whatever's available, with a
    minimum of 4 train and 1 test rows. Lets the fast variant
    (8 prompts × ~10 tokens ≈ 80 rows) exercise IMA/ISA against
    the same defaults that the release-gate run uses at 64+ prompts.
    """
    rng = np.random.default_rng(seed)
    n_total = X.shape[0]
    if n_total == 0:
        return (
            np.zeros((0, X.shape[1] if X.ndim == 2 else 0), dtype=X.dtype),
            np.zeros((0,), dtype=y.dtype),
            np.zeros((0, X.shape[1] if X.ndim == 2 else 0), dtype=X.dtype),
            np.zeros((0,), dtype=y.dtype),
        )
    perm = rng.permutation(n_total)
    if n_train + n_test > n_total:
        # 80/20 fallback with floors so we don't end up with empty splits.
        n_train_eff = max(int(n_total * 0.8), min(4, n_total - 1))
        n_test_eff = max(n_total - n_train_eff, 1)
    else:
        n_train_eff = n_train
        n_test_eff = n_test
    train_idx = perm[:n_train_eff]
    test_idx = perm[n_train_eff : n_train_eff + n_test_eff]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

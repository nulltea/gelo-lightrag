"""Read GELO PCIe-side snapshots written by `capture_snapshots`.

The Rust binary produces, per condition:

* `<condition>.safetensors` — every snapshot's masked operand (and
  optionally its masked output) as keys
  `snap{seq_idx:05}.{layer:03}.{kind}.{operand|output}`.
* `<condition>.meta.json` — provenance: prompt token ids, executor
  config (shield_k, per_forward_mask), and per-snapshot metadata
  (layer, kind, prompt_idx, shape, n_data).

The format contract is pinned by `docs/prototype/aloepri-attack-harness.md` §2.2.

This module exposes a single :class:`SnapshotSet` that loads both
files into memory and lets the Python attack drivers index snapshots
either by (layer, kind) or by (prompt_idx, layer, kind), with an
optional "strip shield rows" view that matches AloePri's pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from safetensors import safe_open


@dataclass(frozen=True)
class SnapshotMeta:
    seq_idx: int
    prompt_idx: int
    layer: int
    kind: str
    operand_shape: tuple[int, int]
    output_shape: tuple[int, int] | None
    n_data: int
    shield_k: int


@dataclass
class SnapshotSet:
    """In-memory view over one condition's snapshots."""

    safetensors_path: Path
    meta_path: Path
    model_id: str
    condition: str
    shield_k: int
    shield_energy_scale: float
    per_forward_mask: bool
    prompt_token_ids: list[list[int]]
    captured_layers: list[int]
    captured_kinds: list[str]
    snapshots: list[SnapshotMeta]
    # Lazy tensor handle — the safetensors file stays mmap-backed until
    # a specific tensor is requested. Set in `open()`.
    _file: Any = field(default=None, repr=False)
    _path: Path = field(default_factory=Path, repr=False)

    @classmethod
    def open(cls, basename_or_path: str | Path, root: Path | None = None) -> SnapshotSet:
        """Open the snapshot set whose basename (or full safetensors path) is
        given. If `root` is supplied, `basename_or_path` is interpreted relative
        to it.
        """
        path = Path(basename_or_path)
        if root is not None:
            path = root / path
        if path.suffix == "":
            safetensors_path = path.with_suffix(".safetensors")
            meta_path = path.with_suffix(".meta.json")
        else:
            safetensors_path = path
            base = path.with_suffix("")
            meta_path = base.with_suffix(".meta.json")

        if not safetensors_path.exists():
            raise FileNotFoundError(safetensors_path)
        if not meta_path.exists():
            raise FileNotFoundError(meta_path)

        with meta_path.open() as fh:
            meta = json.load(fh)

        if meta.get("schema_version") != "1":
            raise ValueError(
                f"unsupported snapshot schema_version {meta.get('schema_version')!r}; "
                f"loader pinned to v1"
            )

        cfg = meta["config"]
        snapshots = [
            SnapshotMeta(
                seq_idx=int(s["seq_idx"]),
                prompt_idx=int(s["prompt_idx"]),
                layer=int(s["layer"]),
                kind=str(s["kind"]),
                operand_shape=tuple(s["operand_shape"]),
                output_shape=tuple(s["output_shape"]) if s["output_shape"] is not None else None,
                n_data=int(s["n_data"]),
                shield_k=int(s["shield_k"]),
            )
            for s in meta["snapshots"]
        ]
        st_handle = safe_open(str(safetensors_path), framework="pt", device="cpu")
        return cls(
            safetensors_path=safetensors_path,
            meta_path=meta_path,
            model_id=str(meta["model_id"]),
            condition=str(meta["condition"]),
            shield_k=int(cfg["shield_k"]),
            shield_energy_scale=float(cfg["shield_energy_scale"]),
            per_forward_mask=bool(cfg["per_forward_mask"]),
            prompt_token_ids=[list(map(int, ids)) for ids in cfg["prompt_token_ids"]],
            captured_layers=list(map(int, cfg["captured_layers"])),
            captured_kinds=list(map(str, cfg["captured_kinds"])),
            snapshots=snapshots,
            _file=st_handle,
            _path=safetensors_path,
        )

    # ─── Indexing ─────────────────────────────────────────────────

    def by_prompt(self) -> dict[int, list[SnapshotMeta]]:
        """Group snapshots by prompt_idx, preserving capture order."""
        out: dict[int, list[SnapshotMeta]] = {}
        for s in self.snapshots:
            out.setdefault(s.prompt_idx, []).append(s)
        return out

    def select(
        self,
        *,
        layer: int | None = None,
        kind: str | None = None,
        prompt_idx: int | None = None,
    ) -> list[SnapshotMeta]:
        out = []
        for s in self.snapshots:
            if layer is not None and s.layer != layer:
                continue
            if kind is not None and s.kind != kind:
                continue
            if prompt_idx is not None and s.prompt_idx != prompt_idx:
                continue
            out.append(s)
        return out

    # ─── Tensor reads ─────────────────────────────────────────────

    def operand_key(self, s: SnapshotMeta) -> str:
        return f"snap{s.seq_idx:05d}.{s.layer:03d}.{s.kind}.operand"

    def output_key(self, s: SnapshotMeta) -> str:
        return f"snap{s.seq_idx:05d}.{s.layer:03d}.{s.kind}.output"

    def get_operand(self, s: SnapshotMeta, *, strip_shield: bool = True) -> torch.Tensor:
        t = self._file.get_tensor(self.operand_key(s)).to(torch.float32)
        if strip_shield and s.shield_k > 0:
            return t[: s.n_data]
        return t

    def get_output(self, s: SnapshotMeta, *, strip_shield: bool = True) -> torch.Tensor | None:
        if s.output_shape is None:
            return None
        t = self._file.get_tensor(self.output_key(s)).to(torch.float32)
        if strip_shield and s.shield_k > 0:
            return t[: s.n_data]
        return t

    # ─── AloePri-shape adapters ───────────────────────────────────
    #
    # AloePri's attacks expect tensors keyed by HuggingFace transformer
    # convention: per-layer (layer_idx, op_kind) → 2-D activation
    # (n_tokens, hidden_size). The methods below assemble those views
    # from our snapshot stream. We always operate prompt-by-prompt so
    # the per-prompt token alignment ISA/IMA need is preserved.

    def per_prompt_layer_kind_tensors(
        self,
        *,
        layer: int,
        kind: str,
        strip_shield: bool = True,
    ) -> list[tuple[int, torch.Tensor]]:
        """Return one (prompt_idx, tensor) pair per prompt for which we
        captured the given (layer, kind). Tensors are
        `(n_data [+ shield_k], in_features)` per the strip_shield flag.
        """
        out: list[tuple[int, torch.Tensor]] = []
        for s in self.select(layer=layer, kind=kind):
            out.append((s.prompt_idx, self.get_operand(s, strip_shield=strip_shield)))
        return out

    def per_prompt_outputs(
        self,
        *,
        layer: int,
        kind: str,
        strip_shield: bool = True,
    ) -> list[tuple[int, torch.Tensor | None]]:
        out: list[tuple[int, torch.Tensor | None]] = []
        for s in self.select(layer=layer, kind=kind):
            out.append((s.prompt_idx, self.get_output(s, strip_shield=strip_shield)))
        return out

    def n_prompts(self) -> int:
        return len(self.prompt_token_ids)

    def n_layers(self) -> int:
        return max(self.captured_layers) + 1 if self.captured_layers else 0


def open_three_conditions(
    root: Path,
    *,
    c0_basename: str = "c0_plain",
    c1_basename: str = "c1_mask_only",
    c2_basename: str = "c2_default",
) -> dict[str, SnapshotSet]:
    """Open the three-condition matrix as a dict slug → SnapshotSet.

    Used by `run_all.py` and by the per-attack drivers when they need
    paired observations across conditions (e.g. VMA's source attribution
    runs all three).
    """
    return {
        "c0_plain": SnapshotSet.open(c0_basename, root=root),
        "c1_mask_only": SnapshotSet.open(c1_basename, root=root),
        "c2_default": SnapshotSet.open(c2_basename, root=root),
    }


def synthetic_set_for_tests(tmp_path: Path) -> SnapshotSet:
    """Build a tiny synthetic snapshot set on disk for unit tests.

    Writes a 4-prompt × 2-layer × {q_proj, k_proj} fixture (16 snapshots
    total), each operand a deterministic `(n_data, d)` matrix derived
    from (prompt_idx, layer, kind, row). Lets the loader and the
    attack drivers be unit-tested without needing the Rust binary.
    """
    from safetensors.torch import save_file

    n_data = 6
    d_in = 8
    d_out = 12
    snapshots: list[dict[str, Any]] = []
    tensors: dict[str, torch.Tensor] = {}
    seq_idx = 0
    for prompt_idx in range(4):
        for layer in range(2):
            for kind in ("q_proj", "k_proj"):
                op = torch.arange(n_data * d_in, dtype=torch.float32).reshape(n_data, d_in)
                op = op + 100 * prompt_idx + 10 * layer + (1 if kind == "k_proj" else 0)
                out = torch.arange(n_data * d_out, dtype=torch.float32).reshape(n_data, d_out)
                tensors[f"snap{seq_idx:05d}.{layer:03d}.{kind}.operand"] = op
                tensors[f"snap{seq_idx:05d}.{layer:03d}.{kind}.output"] = out
                snapshots.append({
                    "seq_idx": seq_idx,
                    "prompt_idx": prompt_idx,
                    "layer": layer,
                    "kind": kind,
                    "operand_shape": [n_data, d_in],
                    "output_shape": [n_data, d_out],
                    "n_data": n_data,
                    "shield_k": 0,
                })
                seq_idx += 1

    safetensors_path = tmp_path / "synthetic.safetensors"
    save_file(tensors, str(safetensors_path), metadata={"schema_version": "1"})

    meta = {
        "schema_version": "1",
        "model_id": "test/synthetic",
        "condition": "c0_plain",
        "config": {
            "shield_k": 0,
            "shield_energy_scale": 0.0,
            "per_forward_mask": False,
            "verify_probes": 0,
            "prompt_token_ids": [[i, i + 1, i + 2, i + 3, i + 4, i + 5] for i in range(4)],
            "captured_layers": [0, 1],
            "captured_kinds": ["q_proj", "k_proj"],
        },
        "snapshots": snapshots,
    }
    meta_path = tmp_path / "synthetic.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    return SnapshotSet.open("synthetic", root=tmp_path)

"""Extract the weight tensors AloePri's static-weight attacks need.

The attack drivers (VMA RowSort + IA Gate-IA/Attn-IA) operate on
`(W_plain, W_obfuscated)` pairs. AloePri's reference impl loads
these from PyTorch / safetensors checkpoints; under §05 our
obfuscated artifact is a GGUF, so we go through the `gguf` library
to read tensors and dequantize when needed.

Plaintext Qwen3 1.7B (`bartowski/Qwen_Qwen3-1.7B-GGUF`) is stored as
Q8_0; the obfuscated artifact `keymat-h128-pi-noise-alg2-fp32.gguf`
is stored as fp32 (Q8_0 quantisation breaks under heavy-tailed
obfuscated weights, see §07).

For each model we expose:
  • `token_embd.weight`              — vocab × d_eff
  • `blk.{i}.attn_q.weight`          — d_eff × q_dim
  • `blk.{i}.attn_k.weight`          — d_eff × kv_dim
  • `blk.{i}.attn_v.weight`          — d_eff × kv_dim
  • `blk.{i}.attn_output.weight`     — q_dim × d_eff
  • `blk.{i}.ffn_gate.weight`        — d_eff × intermediate_eff
  • `blk.{i}.ffn_up.weight`          — d_eff × intermediate_eff
  • `blk.{i}.ffn_down.weight`        — intermediate_eff × d_eff
  • `output.weight`                  — d_eff × vocab

`d_eff` is 2048 for plaintext and 2048 + 2·h = 2304 for the keymat
artifact at h=128.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gguf
import numpy as np


@dataclass
class ModelWeights:
    """In-memory view of one Qwen3 1.7B GGUF's load-bearing weights."""

    label: str
    path: Path
    d_eff: int
    intermediate_eff: int
    n_layers: int
    vocab_size: int
    token_embd: np.ndarray
    output: np.ndarray
    per_layer: list[dict[str, np.ndarray]]


def _dequantize_to_native(tensor: Any) -> np.ndarray:
    """Return the tensor's natural-precision view, decompressing block-quant
    formats on the fly. Output dtype:

    - F32 input: zero-copy fp32 view (no upcast needed).
    - BF16 input: bf16 view (via ml_dtypes.bfloat16) — half the memory of an
      fp32 expansion, faithful to what the server actually stores. Attack
      drivers `.astype(np.float32)` each tensor in their own working window
      so the bf16 dict stays compact across the full model.
    - F16 input: native fp16 view.
    - Q8_0 / other quants: `gguf.quants.dequantize()` gives fp32, then
      truncate to bf16 — sort/ridge/cosine attacks don't need fp32
      precision and an 8B Q8_0+bf16 model pair fits in 33 GB instead of
      66 GB.
    """
    natural_shape = tuple(int(s) for s in reversed(list(tensor.shape)))
    arr = tensor.data
    if tensor.tensor_type == gguf.GGMLQuantizationType.F32:
        return np.frombuffer(arr.tobytes(), dtype=np.float32).reshape(natural_shape) \
            if arr.dtype != np.float32 else arr.reshape(natural_shape)
    if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
        # bf16 is stored as uint16 bit patterns; reinterpret with ml_dtypes.
        import ml_dtypes  # local import — only needed for bf16 inputs
        u16 = np.frombuffer(arr.tobytes(), dtype=np.uint16)
        return u16.view(ml_dtypes.bfloat16).reshape(natural_shape)
    if tensor.tensor_type == gguf.GGMLQuantizationType.F16:
        u16 = np.frombuffer(arr.tobytes(), dtype=np.uint16)
        return u16.view(np.float16).reshape(natural_shape)
    # Block-quant path: dequantise to fp32 (only output gguf supports),
    # then truncate to bf16 to halve memory peak.
    try:
        deq = gguf.quants.dequantize(arr, tensor.tensor_type)
    except Exception as exc:
        raise RuntimeError(
            f"failed to dequantize tensor {tensor.name!r} of type "
            f"{tensor.tensor_type}: {exc}"
        ) from exc
    import ml_dtypes  # noqa
    return deq.reshape(natural_shape).astype(ml_dtypes.bfloat16, copy=False)


# Back-compat alias — older callers used the fp32 name.
_dequantize_to_f32 = _dequantize_to_native


def load_model(path: str | Path, label: str, *, embed_only: bool = False) -> ModelWeights:
    """Read a Qwen3 GGUF and return its load-bearing weights.

    With ``embed_only=True``, only ``token_embd.weight`` and ``output.weight``
    are dequantised; per-layer attention/FFN weights are returned as empty
    dicts. Used by IMA-EmbedRow attacks (which only need W_e pairs) to avoid
    holding the full ~32 GB fp32 expansion of an 8B+ GGUF in memory.
    """
    p = Path(path)
    reader = gguf.GGUFReader(str(p), "r")

    # Build name → tensor index.
    by_name: dict[str, Any] = {t.name: t for t in reader.tensors}

    if "token_embd.weight" not in by_name:
        raise KeyError(f"GGUF {p} lacks token_embd.weight")
    token_embd = _dequantize_to_f32(by_name["token_embd.weight"])
    vocab_size, d_eff = token_embd.shape
    # Qwen3-4B (and other tied-embedding variants) has no separate
    # output.weight — the LM head reuses token_embd. Fall back so attacks
    # that only need W_e (e.g. IMA-EmbedRow-ridge) work on both untied
    # (1.7B) and tied (4B) backbones.
    if "output.weight" in by_name:
        output = _dequantize_to_f32(by_name["output.weight"])
    else:
        output = token_embd

    # Discover layer count by counting blk.{i}.attn_q.weight entries.
    layer_indices: list[int] = []
    for name in by_name:
        if name.startswith("blk.") and name.endswith(".attn_q.weight"):
            idx = int(name.split(".")[1])
            layer_indices.append(idx)
    layer_indices.sort()
    if not layer_indices or layer_indices != list(range(layer_indices[-1] + 1)):
        raise RuntimeError(f"GGUF {p} has non-contiguous attn_q layers: {layer_indices}")
    n_layers = len(layer_indices)

    per_layer: list[dict[str, np.ndarray]] = []
    intermediate_eff = None
    if embed_only:
        # Skip per-layer dequantisation entirely. Read just the FFN-intermediate
        # dim from the metadata cheaply so the dataclass invariant holds.
        intermediate_eff = 0  # unused; per-layer attacks should not use embed_only
        per_layer = [{} for _ in layer_indices]
    else:
        for li in layer_indices:
            layer = {}
            for kind in ("attn_q", "attn_k", "attn_v", "attn_output",
                         "ffn_gate", "ffn_up", "ffn_down"):
                key = f"blk.{li}.{kind}.weight"
                if key not in by_name:
                    raise KeyError(f"GGUF {p} missing {key}")
                layer[kind] = _dequantize_to_f32(by_name[key])
            # ffn_gate is stored (out=intermediate, in=d_eff) in nn.Linear order;
            # the FFN intermediate dim lives on axis 0.
            if intermediate_eff is None:
                intermediate_eff = layer["ffn_gate"].shape[0]
            per_layer.append(layer)

    return ModelWeights(
        label=label,
        path=p,
        d_eff=int(d_eff),
        intermediate_eff=int(intermediate_eff or 0),
        n_layers=n_layers,
        vocab_size=int(vocab_size),
        token_embd=token_embd,
        output=output,
        per_layer=per_layer,
    )


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Probe a GGUF and print its weight shapes")
    p.add_argument("path", type=Path)
    p.add_argument("--label", default="model")
    args = p.parse_args()

    m = load_model(args.path, args.label)
    print(f"label              : {m.label}")
    print(f"path               : {m.path}")
    print(f"vocab_size         : {m.vocab_size}")
    print(f"d_eff              : {m.d_eff}")
    print(f"intermediate_eff   : {m.intermediate_eff}")
    print(f"n_layers           : {m.n_layers}")
    print(f"token_embd shape   : {m.token_embd.shape}")
    print(f"output shape       : {m.output.shape}")
    print(f"layer 0 sizes      : {[(k, v.shape) for k, v in m.per_layer[0].items()]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

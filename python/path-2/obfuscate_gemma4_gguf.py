#!/usr/bin/env python3
"""
AloePri offline obfuscation rewriter for Gemma 4 GGUF artifacts.

v1 — identity-padding obfuscation only. Sets P̂_R = [I_d | 0_{d×2h}] and
Q̂_R = [[I_d]; [0_{2h×d}]]. Every weight tensor that touches the
residual stream is expanded along the residual axis by appending 2h
zero columns/rows. RMSNorm γ on the residual stream is similarly
expanded with zeros (those output dims are then zeroed by the
multiplication — math identical to plaintext through the padded
artifact).

This v1 rewrite is mathematically equivalent to plaintext inference;
it exists solely to validate that llama.cpp's gemma4 forward pass
accepts the expanded `gemma4.embedding_length = d + 2h` metadata
without asserting on tensor shapes. Once the M2.3 gate is cleared,
subsequent revisions of this script add real Algorithm 1 / Algorithm
2 obfuscation on top of the same scaffolding.

Usage:
  python obfuscate_gemma4_gguf.py \\
    --in /path/to/plaintext-gemma-4-E2B-Q8_0.gguf \\
    --out /path/to/identity-padded-E2B.gguf \\
    --expansion-size 128
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import gguf
import gguf.quants as gquants

log = logging.getLogger("obfuscate_gemma4_gguf")


# ────────────────────────────────────────────────────────────────────
# tensor classification
# ────────────────────────────────────────────────────────────────────
#
# Each tensor either:
#   - is_residual_in  : its inner (ggml-ne[0]) axis reads from the residual
#                       stream — expand ne[0] from d to d+2h.
#   - is_residual_out : its outer (ggml-ne[1]) axis writes to the residual
#                       stream — expand ne[1] from d to d+2h.
#   - is_residual_norm: 1-D γ vector that scales the residual stream —
#                       expand ne[0] from d to d+2h.
#   - is_unchanged    : no residual-stream axis — copy as-is.
#
# For per-block tensors the prefix `blk.<i>.` is stripped before matching.

# Tensors whose residual-stream axis is ne[0] (the "inner" / fastest ggml dim).
# In natural numpy shape (reversed of ggml shape), this is axis -1.
RESIDUAL_NE0_TENSORS = {
    # per-block
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "inp_gate.weight",
    "per_layer_model_proj.weight",
    # global
    "token_embd.weight",
    "output.weight",
}

# Tensors whose residual-stream axis is ne[1] (the "outer" ggml dim).
# In natural numpy shape, this is axis 0.
RESIDUAL_NE1_TENSORS = {
    "attn_output.weight",
    "ffn_down.weight",
    "proj.weight",
}

# 1-D γ vectors on the residual stream. ne[0] = d, natural axis is the only axis.
RESIDUAL_NORM_TENSORS = {
    "attn_norm.weight",
    "post_attention_norm.weight",
    "post_norm.weight",
    "post_ffw_norm.weight",
    "ffn_norm.weight",
    "per_layer_post_norm.weight",
    "output_norm.weight",
}

# Unchanged: attn_q_norm, attn_k_norm (head_dim space), rope_freqs,
# layer_output_scale, per_layer_token_embd (vocab × all-layers concat —
# unchanged in identity-padding v1), per_layer_proj_norm.


def stripped_tensor_name(full_name: str) -> str:
    """blk.0.attn_q.weight -> attn_q.weight ; output_norm.weight -> output_norm.weight"""
    if full_name.startswith("blk."):
        # blk.<idx>.<rest>
        parts = full_name.split(".", 2)
        if len(parts) == 3:
            return parts[2]
    return full_name


def classify(full_name: str) -> str:
    stripped = stripped_tensor_name(full_name)
    if stripped in RESIDUAL_NE0_TENSORS:
        return "ne0"  # expand natural axis -1
    if stripped in RESIDUAL_NE1_TENSORS:
        return "ne1"  # expand natural axis 0
    if stripped in RESIDUAL_NORM_TENSORS:
        return "norm"  # 1-D γ, expand its only axis
    return "unchanged"


# ────────────────────────────────────────────────────────────────────
# dequant / requant
# ────────────────────────────────────────────────────────────────────


def to_float_array(t: gguf.ReaderTensor) -> np.ndarray:
    """Returns the tensor as a float32 numpy array in **natural** numpy shape
    (reversed of ggml-shape: outermost dim first, innermost last).

    This is the standard pytorch / numpy "(rows, cols)" matrix layout, which
    makes the residual axis (ne[0]) be axis -1.
    """
    qtype = t.tensor_type
    natural_shape = tuple(int(s) for s in reversed(t.shape))
    if qtype == gguf.GGMLQuantizationType.F32:
        return np.frombuffer(t.data, dtype=np.float32).reshape(natural_shape)
    if qtype == gguf.GGMLQuantizationType.F16:
        return np.frombuffer(t.data, dtype=np.float16).astype(np.float32).reshape(natural_shape)
    if qtype == gguf.GGMLQuantizationType.BF16:
        u16 = np.frombuffer(t.data, dtype=np.uint16)
        f32 = np.zeros(u16.shape, dtype=np.uint32)
        f32 |= u16.astype(np.uint32) << 16
        return f32.view(np.float32).reshape(natural_shape)
    # quantised — dequant via gguf.quants
    raw = np.array(t.data)
    arr = gquants.dequantize(raw, qtype)
    return arr.reshape(natural_shape)


# NOTE: gguf-py's GGUFWriter already reverses the shape internally when packing
# to ggml format. So when calling add_tensor with raw_shape, pass the **natural**
# (numpy) shape — the writer handles the flip.


# ────────────────────────────────────────────────────────────────────
# expansion
# ────────────────────────────────────────────────────────────────────


def expand_residual_axis(arr_natural: np.ndarray, axis: int, d: int, expansion: int) -> np.ndarray:
    """Expand `arr_natural` by appending `expansion` zeros along `axis`.

    `arr_natural.shape[axis]` must equal `d`. Returns array of shape with
    that axis size `d + expansion`."""
    if arr_natural.shape[axis] != d:
        raise ValueError(
            f"axis {axis} of shape {arr_natural.shape} is {arr_natural.shape[axis]}, expected {d}"
        )
    pad_shape = list(arr_natural.shape)
    pad_shape[axis] = expansion
    zeros = np.zeros(tuple(pad_shape), dtype=arr_natural.dtype)
    return np.concatenate([arr_natural, zeros], axis=axis)


def transform_tensor(
    t: gguf.ReaderTensor, d: int, h: int
) -> tuple[np.ndarray, gguf.GGMLQuantizationType, bool]:
    """Returns (new_natural_array, new_qtype, was_changed). `h` is the
    Algorithm-1 expansion parameter — residual axis grows from d to d+2h."""
    cls = classify(t.name)
    arr_natural = to_float_array(t)
    pad = 2 * h
    if cls == "unchanged":
        return arr_natural, t.tensor_type, False
    if cls == "ne0" or cls == "norm":
        out = expand_residual_axis(arr_natural, axis=-1, d=d, expansion=pad)
    elif cls == "ne1":
        out = expand_residual_axis(arr_natural, axis=0, d=d, expansion=pad)
    else:
        raise AssertionError(cls)
    if cls == "norm":
        return out.astype(np.float32), gguf.GGMLQuantizationType.F32, True
    return out.astype(np.float16), gguf.GGMLQuantizationType.F16, True


# ────────────────────────────────────────────────────────────────────
# main pipeline
# ────────────────────────────────────────────────────────────────────


def rewrite_gguf(in_path: Path, out_path: Path, expansion: int) -> dict:
    log.info("opening %s", in_path)
    r = gguf.GGUFReader(str(in_path))

    # Identify architecture and hidden_size
    arch = r.fields["general.architecture"].contents()
    if arch != "gemma4":
        raise SystemExit(f"unsupported architecture: {arch} (expected gemma4)")
    d = int(r.fields["gemma4.embedding_length"].contents())
    log.info("architecture=%s hidden_size(d)=%d expansion(2h)=%d new_hidden=%d",
             arch, d, 2 * expansion, d + 2 * expansion)

    # ---- writer ----
    writer = gguf.GGUFWriter(str(out_path), arch=arch)
    skipped_keys = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
    for key in sorted(r.fields.keys()):
        if key in skipped_keys:
            continue
        field = r.fields[key]
        value = field.contents()
        if key == "gemma4.embedding_length":
            value = d + 2 * expansion
            log.info("rewriting metadata: %s = %d (was %d)", key, value, d)

        # Pick the writer API based on the field's value type
        # (best-effort: GGUFWriter has add_string / add_uint32 / add_array / etc.,
        # or the generic add_value path.)
        _write_field(writer, key, value, field)

    # ---- tensors ----
    # Strategy: every tensor (changed or unchanged) gets dequantised + written
    # as F16 (weights) or F32 (norms / scales). This avoids Q8_0 byte-shape
    # bookkeeping in the writer and makes the artifact uniform fp16/fp32. The
    # ~10 GB output is fine for the M2.3 gate; size optimisation (re-quantise
    # the obfuscated artifact via llama-quantize) is a later step.
    n_changed = 0
    n_unchanged = 0
    for t in r.tensors:
        cls = classify(t.name)
        if cls == "unchanged":
            arr_natural = to_float_array(t)
            if arr_natural.ndim <= 1:
                out_arr = arr_natural.astype(np.float32)
                out_qtype = gguf.GGMLQuantizationType.F32
            else:
                out_arr = arr_natural.astype(np.float16)
                out_qtype = gguf.GGMLQuantizationType.F16
            writer.add_tensor(t.name, out_arr, raw_dtype=out_qtype)
            n_unchanged += 1
            continue
        new_arr, new_qtype, _ = transform_tensor(t, d=d, h=expansion)
        writer.add_tensor(t.name, new_arr, raw_dtype=new_qtype)
        n_changed += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    log.info("wrote %s (%d tensors changed, %d unchanged)", out_path, n_changed, n_unchanged)
    return {"changed": n_changed, "unchanged": n_unchanged, "d": d, "expansion": expansion}


def _write_field(writer: gguf.GGUFWriter, key: str, value, field: gguf.ReaderField) -> None:
    """Best-effort metadata pass-through. Falls back to add_array / add_value."""
    # field.types is a list of GGUFValueType
    types = field.types
    primary = types[0]
    if primary == gguf.GGUFValueType.STRING:
        writer.add_string(key, value)
    elif primary == gguf.GGUFValueType.BOOL:
        writer.add_bool(key, bool(value))
    elif primary == gguf.GGUFValueType.UINT8:
        writer.add_uint8(key, int(value))
    elif primary == gguf.GGUFValueType.INT8:
        writer.add_int8(key, int(value))
    elif primary == gguf.GGUFValueType.UINT16:
        writer.add_uint16(key, int(value))
    elif primary == gguf.GGUFValueType.INT16:
        writer.add_int16(key, int(value))
    elif primary == gguf.GGUFValueType.UINT32:
        writer.add_uint32(key, int(value))
    elif primary == gguf.GGUFValueType.INT32:
        writer.add_int32(key, int(value))
    elif primary == gguf.GGUFValueType.UINT64:
        writer.add_uint64(key, int(value))
    elif primary == gguf.GGUFValueType.INT64:
        writer.add_int64(key, int(value))
    elif primary == gguf.GGUFValueType.FLOAT32:
        writer.add_float32(key, float(value))
    elif primary == gguf.GGUFValueType.FLOAT64:
        writer.add_float64(key, float(value))
    elif primary == gguf.GGUFValueType.ARRAY:
        # Use the inner type to pick element kind
        inner = types[1] if len(types) > 1 else gguf.GGUFValueType.STRING
        writer.add_array(key, list(value))
    else:
        raise ValueError(f"unsupported gguf value type {primary} for key {key}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, required=True)
    parser.add_argument("--out", dest="out_path", type=Path, required=True)
    parser.add_argument("--expansion-size", type=int, default=128, help="h: half of the dim expansion (default 128 → d → d+256)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    info = rewrite_gguf(args.in_path, args.out_path, args.expansion_size)
    log.info("done: %s", info)
    return 0


if __name__ == "__main__":
    sys.exit(main())

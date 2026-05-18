#!/usr/bin/env python3
"""
AloePri offline obfuscation rewriter for Gemma 4 GGUF artifacts.

Two modes:

  --mode identity-pad   (default)
    Sets P̂_R = [I_d | 0_{d×2h}] and Q̂_R = [[I_d]; [0_{2h×d}]].
    Every residual-stream-touching tensor is expanded along its
    residual axis by appending 2h zeros. Mathematically a no-op
    (identical output to plaintext); exists to validate that
    llama.cpp's gemma4 forward graph accepts `embedding_length =
    d + 2h` without asserting on tensor shapes. M2.3 baseline.

  --mode keymat
    Real Algorithm 1 obfuscation with paper §5.2.5 RMSNorm fusion:
      1. Sample one global pair (P̂_R, Q̂_R) via Algorithm 1.
      2. Compute κ = E[‖x P̂‖/‖x‖] over Gaussian samples.
      3. Per-layer, fuse each γ vector into its adjacent linear
         weight(s) (pre-norms → left-mult into following linear,
         post-norms → right-mult into preceding linear).
      4. Replace γ tensors with constant κ vectors of length d+2h.
      5. Apply Algorithm 1 obfuscation to the fused linears:
         · ne0 (residual on ggml axis 0): W_new = W_old @ Q̂_R^T
         · ne1 (residual on ggml axis 1): W_new = P̂_R^T @ W_old
      6. Untie output.weight from token_embd.weight so output_norm
         γ can fuse into the LM head without contaminating the
         input-embedding lookup.

    The artifact runs on stock llama.cpp via Vulkan unchanged.

Usage:
  python obfuscate_gemma4_gguf.py \\
    --in /path/to/plaintext.gguf \\
    --out /path/to/obfuscated.gguf \\
    --mode {identity-pad|keymat} \\
    --expansion-size 128 \\
    --seed 42
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
# Each per-block tensor name (stripped of `blk.<i>.`) falls into one of:
#
#   - "ne0"  : residual-stream axis is ggml ne[0] → expand natural axis -1.
#              These are the "input-from-residual" weights and embeddings.
#   - "ne1"  : residual-stream axis is ggml ne[1] → expand natural axis 0.
#              These are the "output-to-residual" weights.
#   - "norm" : 1-D γ vector for an RMSNorm on the residual stream.
#              In keymat mode, fuses into an adjacent linear before being
#              replaced with constant κ × (d+2h).
#   - "unchanged" : no residual-stream axis.
#
# Norm-to-linear fusion map (Gemma 4, dense path; per `src/models/gemma4.cpp`):
#
#   attn_norm.weight           → pre-norm  → fuse INTO attn_q, attn_k, attn_v
#   post_attention_norm.weight → post-norm → fuse INTO attn_output
#   ffn_norm.weight            → pre-norm  → fuse INTO ffn_gate, ffn_up
#   post_ffw_norm.weight       → post-norm → fuse INTO ffn_down
#   post_norm.weight           → post-norm → fuse INTO proj  (per_layer_proj)
#   output_norm.weight (global)→ pre-norm  → fuse INTO output (separated from token_embd)

# ne[0] = d, READS from residual stream (residual is the *input* axis).
# Transform: W_new = W_old @ Q̂_R^T (i.e. arr_old @ Q_R.T).
RESIDUAL_NE0_READ_TENSORS = {
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "inp_gate.weight",
    "per_layer_model_proj.weight",
    "output.weight",                  # LM head: reads residual, outputs logits
}

# ne[0] = d, WRITES to residual stream (residual is the *output* axis).
# Embedding lookup case: each row IS a residual-stream entry.
# Transform: W_new = W_old @ P̂_R (i.e. arr_old @ P_R).
RESIDUAL_NE0_WRITE_TENSORS = {
    "token_embd.weight",              # input embedding
}

# ne[1] = d, WRITES to residual stream (residual is the *output* axis).
# Transform: W_new = P̂_R^T @ W_old (numpy natural arr_new = P_R.T @ arr_old).
RESIDUAL_NE1_TENSORS = {
    "attn_output.weight",
    "ffn_down.weight",
    "proj.weight",
}

RESIDUAL_NORM_TENSORS = {
    "attn_norm.weight",
    "post_attention_norm.weight",
    "post_norm.weight",
    "post_ffw_norm.weight",
    "ffn_norm.weight",
    "per_layer_post_norm.weight",
    "output_norm.weight",
}

# §5.2.5 norm-to-linear fusion. Only PRE-norms are fused.
#
# Pre-norm fusion is mathematically exact in plaintext:
#   (x · γ / RMS(x)) @ W  ==  (x / RMS(x)) @ (Diag(γ) · W)
# i.e. γ commutes from "after the norm scaling" into "before the next linear".
#
# Post-norm fusion (γ row-scales into the *previous* linear's output dim) is
# NOT exact — it would require RMS(out · γ) == RMS(out), which only holds
# when γ is constant. The paper's §5.2.5 construction implicitly assumed
# pre-norm architectures (Llama / Qwen); Gemma 4's per-block post-norms
# (post_attention_norm, post_ffw_norm, post_norm) can't be fused this way.
#
# For post-norms we therefore LEAVE γ per-dim at the norm site. In keymat
# mode the γ vector is extended from d to d+2h with κ in the padded dims —
# approximate but keeps the obfuscation chain magnitude-consistent.
PER_BLOCK_FUSION_MAP: dict[str, tuple[str, list[str]]] = {
    "attn_norm.weight":           ("pre",  ["attn_q.weight", "attn_k.weight", "attn_v.weight"]),
    "ffn_norm.weight":            ("pre",  ["ffn_gate.weight", "ffn_up.weight"]),
}

POST_NORM_NAMES = {
    "post_attention_norm.weight",
    "post_ffw_norm.weight",
    "post_norm.weight",  # per_layer_post_norm
}

GLOBAL_FUSION_MAP: dict[str, tuple[str, list[str]]] = {
    "output_norm.weight": ("pre", ["output.weight"]),
}


def stripped_tensor_name(full_name: str) -> str:
    if full_name.startswith("blk."):
        parts = full_name.split(".", 2)
        if len(parts) == 3:
            return parts[2]
    return full_name


def block_index(full_name: str) -> int | None:
    if not full_name.startswith("blk."):
        return None
    parts = full_name.split(".", 2)
    return int(parts[1])


def classify(full_name: str) -> str:
    stripped = stripped_tensor_name(full_name)
    if stripped in RESIDUAL_NE0_READ_TENSORS:
        return "ne0_read"
    if stripped in RESIDUAL_NE0_WRITE_TENSORS:
        return "ne0_write"
    if stripped in RESIDUAL_NE1_TENSORS:
        return "ne1"
    if stripped in RESIDUAL_NORM_TENSORS:
        return "norm"
    return "unchanged"


# ────────────────────────────────────────────────────────────────────
# dequant / shape helpers
# ────────────────────────────────────────────────────────────────────


def to_float_array(t: gguf.ReaderTensor) -> np.ndarray:
    """Tensor as float32 numpy in **natural** numpy shape (reversed of ggml-shape:
    outermost first, innermost last). Residual axis (ne[0]) is numpy axis -1."""
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
    raw = np.array(t.data)
    arr = gquants.dequantize(raw, qtype)
    return arr.reshape(natural_shape)


# ────────────────────────────────────────────────────────────────────
# identity-pad transform (mode=identity-pad)
# ────────────────────────────────────────────────────────────────────


def expand_axis_with_zeros(arr: np.ndarray, axis: int, d: int, expansion: int) -> np.ndarray:
    if arr.shape[axis] != d:
        raise ValueError(f"axis {axis} of shape {arr.shape} is {arr.shape[axis]}, expected {d}")
    pad_shape = list(arr.shape)
    pad_shape[axis] = expansion
    return np.concatenate([arr, np.zeros(tuple(pad_shape), dtype=arr.dtype)], axis=axis)


def identity_pad_tensor(arr_natural: np.ndarray, cls: str, d: int, pad: int) -> np.ndarray:
    # For identity-pad mode the read vs write direction doesn't matter:
    # both Q̂_R^T = [I_d | 0] and P̂_R = [I_d | 0] in that special case, so
    # appending zeros along axis -1 is correct for both ne0_read and ne0_write.
    if cls in ("ne0_read", "ne0_write", "norm"):
        return expand_axis_with_zeros(arr_natural, axis=-1, d=d, expansion=pad)
    if cls == "ne1":
        return expand_axis_with_zeros(arr_natural, axis=0, d=d, expansion=pad)
    raise AssertionError(cls)


# ────────────────────────────────────────────────────────────────────
# keymat helpers (mode=keymat)
# ────────────────────────────────────────────────────────────────────


def fuse_gamma_pre(linear_arr: np.ndarray, gamma: np.ndarray, kappa: float) -> np.ndarray:
    """Pre-norm fusion: γ row-scales the *following* linear's d-axis (natural axis -1).

    The new norm sits at γ' = κ·1; the per-dim γ_plain is absorbed into the
    weight as `linear_new[m, i] = (γ_plain[i] / κ) * linear_old[m, i]`."""
    scale = (gamma / kappa).astype(linear_arr.dtype)
    return linear_arr * scale  # broadcasts along axis 0


def fuse_gamma_post(linear_arr: np.ndarray, gamma: np.ndarray, kappa: float) -> np.ndarray:
    """Post-norm fusion: γ column-scales the *previous* linear's d-axis (natural axis 0).

    For natural shape (d, M_in): `linear_new[i, m] = (γ_plain[i] / κ) * linear_old[i, m]`."""
    scale = (gamma / kappa).astype(linear_arr.dtype)
    return linear_arr * scale[:, None]  # broadcasts along axis 1


def apply_algorithm1_ne0_read(arr: np.ndarray, q_r_t: np.ndarray) -> np.ndarray:
    """For ne0 read-from-residual tensors (natural shape (..., d)):
    output = x_obf @ W_new where W_new = Q̂_R @ W_paper.
    In numpy: arr_new = arr_old @ Q̂_R^T. Result shape (..., d+2h)."""
    return arr @ q_r_t


def apply_algorithm1_ne0_write(arr: np.ndarray, p_r: np.ndarray) -> np.ndarray:
    """For ne0 write-to-residual tensors (e.g. token_embd):
    The lookup output should be x_plain @ P̂_R (lands in obfuscated residual space).
    In numpy: arr_new = arr_old @ P̂_R. Result shape (..., d+2h)."""
    return arr @ p_r


def apply_algorithm1_ne1(arr: np.ndarray, p_r_t: np.ndarray) -> np.ndarray:
    """For ne1 write-to-residual tensors (natural shape (d, ...)):
    W_new = W_paper @ P̂_R, so in numpy: arr_new = P̂_R^T @ arr_old. Shape (d+2h, ...)."""
    return p_r_t @ arr


def estimate_kappa(p_r: np.ndarray, d: int, num_samples: int = 1000, seed: int = 0) -> float:
    """κ = E[‖x P̂‖/‖x‖] over Gaussian-like samples. Reference §5.2.5."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, d)).astype(np.float64)
    y = x @ p_r.astype(np.float64)
    ratio = np.linalg.norm(y, axis=-1) / np.linalg.norm(x, axis=-1)
    return float(ratio.mean())


# ────────────────────────────────────────────────────────────────────
# main pipeline
# ────────────────────────────────────────────────────────────────────


def rewrite_gguf(
    in_path: Path,
    out_path: Path,
    mode: str,
    expansion: int,
    seed: int,
    lam: float = 0.3,
) -> dict:
    log.info("opening %s (mode=%s)", in_path, mode)
    r = gguf.GGUFReader(str(in_path))

    arch = r.fields["general.architecture"].contents()
    if arch != "gemma4":
        raise SystemExit(f"unsupported architecture: {arch} (expected gemma4)")
    d = int(r.fields["gemma4.embedding_length"].contents())
    n_layer = int(r.fields["gemma4.block_count"].contents())
    # In gamma-only mode there's no dim expansion; in identity-pad and keymat
    # the residual axis grows from d to d+2h.
    if mode == "gamma-only":
        pad = 0
    else:
        pad = 2 * expansion
    new_d = d + pad
    log.info("arch=%s d=%d 2h=%d new_d=%d n_layer=%d", arch, d, pad, new_d, n_layer)

    # ---- key material (only used in keymat mode) ----
    p_r = None
    q_r = None
    kappa = 1.0
    if mode == "keymat":
        # Lazy import — vendored Algorithm 1 implementation.
        sys.path.insert(0, "/home/timo/repos/private-rag-path-2/vendor/aloepri-py")
        sys.path.insert(0, "/home/timo/repos/private-rag-path-2/vendor/aloepri-py/src")
        from keymat import build_keymat_transform  # type: ignore

        log.info("sampling Algorithm 1 (P̂_R, Q̂_R) at d=%d h=%d λ=%.2f seed=%d", d, expansion, lam, seed)
        transform = build_keymat_transform(d=d, h=expansion, lam=lam, init_seed=seed)
        p_r = transform.key.numpy().astype(np.float64)        # shape (d, d+2h)
        q_r = transform.inverse.numpy().astype(np.float64)    # shape (d+2h, d)
        # Verify P̂·Q̂ ≈ I_d
        prod = p_r @ q_r
        identity_err = float(np.max(np.abs(prod - np.eye(d))))
        log.info("‖P̂·Q̂ - I_d‖_max = %.3e", identity_err)
        if identity_err > 1e-3:
            log.warning("key matrices have large identity error — proceeding anyway")
        kappa_e = estimate_kappa(p_r, d=d, num_samples=2000, seed=seed + 100)
        # κ_correct = κ_E · √(d/(d+2h)) accounts for the change in RMSNorm
        # denominator dim (d → d+2h). With κ_correct in γ' = κ_correct·1,
        # the obfuscated RMSNorm covariantly matches the plaintext RMSNorm
        # composed with P̂_R (paper §5.2.5 "Putting Together" derivation).
        kappa = kappa_e * float(np.sqrt(d / float(d + 2 * expansion)))
        log.info("κ_E = E[‖xP̂‖/‖x‖] ≈ %.5f (over 2000 Gaussian samples)", kappa_e)
        log.info("κ_correct = κ_E · √(d/(d+2h)) = %.5f · %.5f = %.5f",
                 kappa_e, float(np.sqrt(d / float(d + 2 * expansion))), kappa)

    # ---- pass 1: load every tensor into a name→numpy dict so we can do fusion ----
    log.info("loading + dequantising tensors...")
    arrays: dict[str, np.ndarray] = {}
    for t in r.tensors:
        arrays[t.name] = to_float_array(t)
    log.info("loaded %d tensors", len(arrays))

    # ---- pass 2 (keymat / gamma-only): γ → adjacent-linear fusion ----
    if mode in ("keymat", "gamma-only"):
        log.info("§5.2.5 fusion: γ → adjacent linears")
        # First: untie output from token_embd so output_norm γ has a target.
        if "output.weight" not in arrays:
            log.info("untying output.weight from token_embd.weight")
            arrays["output.weight"] = arrays["token_embd.weight"].copy()

        # Per-block fusion
        for il in range(n_layer):
            for norm_name, (direction, targets) in PER_BLOCK_FUSION_MAP.items():
                gamma_key = f"blk.{il}.{norm_name}"
                if gamma_key not in arrays:
                    continue
                gamma = arrays[gamma_key]
                for tgt_name in targets:
                    tgt_key = f"blk.{il}.{tgt_name}"
                    if tgt_key not in arrays:
                        continue
                    if direction == "pre":
                        arrays[tgt_key] = fuse_gamma_pre(arrays[tgt_key], gamma, kappa)
                    elif direction == "post":
                        arrays[tgt_key] = fuse_gamma_post(arrays[tgt_key], gamma, kappa)

        # Global output_norm fusion (pre-norm into output.weight)
        for norm_name, (direction, targets) in GLOBAL_FUSION_MAP.items():
            if norm_name not in arrays:
                continue
            gamma = arrays[norm_name]
            for tgt_name in targets:
                if tgt_name not in arrays:
                    continue
                if direction == "pre":
                    arrays[tgt_name] = fuse_gamma_pre(arrays[tgt_name], gamma, kappa)
                else:
                    arrays[tgt_name] = fuse_gamma_post(arrays[tgt_name], gamma, kappa)
        log.info("γ fusion complete")

    # ---- pass 3: apply Algorithm 1 obfuscation (or identity padding) + replace γ ----
    log.info("applying %s transform to residual-stream tensors", mode)
    q_r_t = p_r_t = None
    if mode == "keymat":
        q_r_t = q_r.T  # shape (d, d+2h)
        p_r_t = p_r.T  # shape (d+2h, d)

    n_changed = 0
    n_unchanged = 0
    n_added = 0

    # Collect tensors to write. For unchanged ones, keep as-is. For norm tensors,
    # we may replace with constant κ. For ne0/ne1 tensors, we may transform.
    # Output tensors written in input order, with output.weight added if it was untied.
    input_order = [t.name for t in r.tensors]
    if mode in ("keymat", "gamma-only") and "output.weight" not in input_order:
        # Place output.weight immediately after token_embd.weight in the file.
        if "token_embd.weight" in input_order:
            idx = input_order.index("token_embd.weight") + 1
            input_order.insert(idx, "output.weight")
            n_added += 1
        else:
            input_order.append("output.weight")
            n_added += 1

    # ---- writer setup + metadata pass-through ----
    writer = gguf.GGUFWriter(str(out_path), arch=arch)
    skipped_keys = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
    for key in sorted(r.fields.keys()):
        if key in skipped_keys:
            continue
        field = r.fields[key]
        value = field.contents()
        if key == "gemma4.embedding_length" and new_d != d:
            value = new_d
            log.info("metadata: %s = %d (was %d)", key, value, d)
        _write_field(writer, key, value, field)
    # Tag the artifact so we can identify obfuscated models later.
    writer.add_string("aloepri.mode", mode)
    writer.add_uint32("aloepri.expansion_size", expansion)
    writer.add_uint32("aloepri.seed", seed)
    if mode == "keymat":
        writer.add_float32("aloepri.kappa", float(kappa))
        writer.add_float32("aloepri.lambda", float(lam))

    for name in input_order:
        if name not in arrays:
            log.warning("declared name %s missing from arrays — skipping", name)
            continue
        arr = arrays[name]
        cls = classify(name)
        if cls == "unchanged":
            out_arr = arr.astype(np.float32) if arr.ndim <= 1 else arr.astype(np.float16)
            out_qtype = (gguf.GGMLQuantizationType.F32 if arr.ndim <= 1
                         else gguf.GGMLQuantizationType.F16)
            writer.add_tensor(name, out_arr, raw_dtype=out_qtype)
            n_unchanged += 1
            continue

        if cls == "norm":
            stripped = stripped_tensor_name(name)
            is_post_norm = stripped in POST_NORM_NAMES
            if mode == "identity-pad":
                out_arr = expand_axis_with_zeros(arr, axis=-1, d=d, expansion=pad).astype(np.float32)
            elif mode == "keymat":
                if is_post_norm:
                    # post-norm γ stays per-dim; padded dims get κ for magnitude consistency
                    out_arr = np.concatenate(
                        [arr.astype(np.float32),
                         np.full((pad,), float(kappa), dtype=np.float32)]
                    )
                else:
                    out_arr = np.full((new_d,), float(kappa), dtype=np.float32)
            else:  # gamma-only
                if is_post_norm:
                    # don't touch — post-norm γ stays per-dim, unfused
                    out_arr = arr.astype(np.float32)
                else:
                    out_arr = np.full((d,), float(kappa), dtype=np.float32)
            writer.add_tensor(name, out_arr, raw_dtype=gguf.GGMLQuantizationType.F32)
            n_changed += 1
            continue

        # ne0_read / ne0_write / ne1
        if mode == "identity-pad":
            out_arr = identity_pad_tensor(arr, cls, d=d, pad=pad).astype(np.float16)
        elif mode == "keymat":
            arr_f64 = arr.astype(np.float64)
            if cls == "ne0_read":
                out_arr = apply_algorithm1_ne0_read(arr_f64, q_r_t).astype(np.float16)
            elif cls == "ne0_write":
                out_arr = apply_algorithm1_ne0_write(arr_f64, p_r).astype(np.float16)
            elif cls == "ne1":
                out_arr = apply_algorithm1_ne1(arr_f64, p_r_t).astype(np.float16)
            else:
                raise AssertionError(cls)
        else:  # gamma-only — no dim expansion, no Algorithm 1; just write γ-fused linears as-is
            out_arr = arr.astype(np.float16)
        writer.add_tensor(name, out_arr, raw_dtype=gguf.GGMLQuantizationType.F16)
        n_changed += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    log.info("wrote %s (changed=%d unchanged=%d added=%d)", out_path, n_changed, n_unchanged, n_added)
    return {
        "mode": mode,
        "d": d,
        "new_d": new_d,
        "expansion": expansion,
        "n_changed": n_changed,
        "n_unchanged": n_unchanged,
        "n_added": n_added,
        "kappa": kappa,
    }


def _write_field(writer: gguf.GGUFWriter, key: str, value, field: gguf.ReaderField) -> None:
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
        writer.add_array(key, list(value))
    else:
        raise ValueError(f"unsupported gguf value type {primary} for key {key}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, required=True)
    parser.add_argument("--out", dest="out_path", type=Path, required=True)
    parser.add_argument("--mode", choices=["identity-pad", "keymat", "gamma-only"], default="identity-pad",
                        help="gamma-only: §5.2.5 fusion only, κ=1, no dim expansion — should be bit-identical to plaintext.")
    parser.add_argument("--expansion-size", type=int, default=128,
                        help="h: half of the dim expansion (default 128 → d → d+256)")
    parser.add_argument("--seed", type=int, default=42, help="Algorithm 1 seed (keymat mode)")
    parser.add_argument("--lam", type=float, default=0.3, help="Algorithm 1 λ coefficient (B = U + λV)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    info = rewrite_gguf(
        args.in_path, args.out_path, mode=args.mode, expansion=args.expansion_size, seed=args.seed, lam=args.lam,
    )
    log.info("done: %s", info)
    return 0


if __name__ == "__main__":
    sys.exit(main())

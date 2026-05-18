#!/usr/bin/env python3
"""
AloePri offline obfuscation rewriter for Qwen3 dense GGUF artifacts.

Qwen3's residual-stream norm topology is simpler than Gemma 4's:

  per-block:  attn_norm (pre)  →  attn_q/k/v  →  attn_output
              ffn_norm  (pre)  →  ffn_gate/up →  ffn_down
  global:     output_norm (pre)  →  output

  off-axis (head_dim, not d):  attn_q_norm, attn_k_norm  — unchanged
  no post-norms, no PLE, no Gemma-4-specific complications

  output.weight is already separate from token_embd.weight on disk.

§5.2.5 fuse-and-scale is mathematically exact at every (pre-)norm site,
so this rewriter should produce coherent output in keymat mode.

Modes:

  identity-pad   P̂_R = [I_d | 0]: zero padding. Math identical to
                 plaintext through the padded structure. Validates dim
                 plumbing only.

  gamma-only     §5.2.5 fusion only, κ = 1, no dim expansion. Must
                 produce **bit-identical** output to plaintext (modulo
                 fp16 quantisation). This is the fusion-correctness
                 regression test.

  keymat         Full Algorithm 1 obfuscation with §5.2.5 fusion.
                 Real per-batch-static AloePri without Π / Algorithm 2
                 attention transforms / noise yet — those come in
                 follow-on revisions.

Usage:
  python obfuscate_qwen3_gguf.py \\
    --in /path/to/plaintext-qwen3.gguf \\
    --out /path/to/obfuscated.gguf \\
    --mode {identity-pad|gamma-only|keymat} \\
    --expansion-size 128 --seed 42
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import gguf
import gguf.quants as gquants

log = logging.getLogger("obfuscate_qwen3_gguf")


# ────────────────────────────────────────────────────────────────────
# tensor classification (Qwen3 dense)
# ────────────────────────────────────────────────────────────────────

# ne[0] = d, READS from residual. Transform: arr_new = arr_old @ Q̂_R^T.
RESIDUAL_NE0_READ_TENSORS = {
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "output.weight",         # LM head: reads residual, outputs logits
}

# ne[0] = d, WRITES to residual (embedding lookup). Transform: arr_old @ P̂_R.
RESIDUAL_NE0_WRITE_TENSORS = {
    "token_embd.weight",
}

# ne[1] = d, WRITES to residual. Transform: arr_new = P̂_R^T @ arr_old.
RESIDUAL_NE1_TENSORS = {
    "attn_output.weight",
    "ffn_down.weight",
}

# 1-D γ vectors on the residual stream — all pre-norms in Qwen3.
RESIDUAL_NORM_TENSORS = {
    "attn_norm.weight",
    "ffn_norm.weight",
    "output_norm.weight",
}

# §5.2.5 norm-to-linear fusion. Pre-norms only — Qwen3 has nothing else.
PER_BLOCK_FUSION_MAP: dict[str, tuple[str, list[str]]] = {
    "attn_norm.weight": ("pre", ["attn_q.weight", "attn_k.weight", "attn_v.weight"]),
    "ffn_norm.weight":  ("pre", ["ffn_gate.weight", "ffn_up.weight"]),
}

GLOBAL_FUSION_MAP: dict[str, tuple[str, list[str]]] = {
    "output_norm.weight": ("pre", ["output.weight"]),
}

# Unchanged (off-residual): attn_q_norm, attn_k_norm operate on head_dim.


def stripped_tensor_name(full_name: str) -> str:
    if full_name.startswith("blk."):
        parts = full_name.split(".", 2)
        if len(parts) == 3:
            return parts[2]
    return full_name


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
# dequant helpers
# ────────────────────────────────────────────────────────────────────


def to_float_array(t: gguf.ReaderTensor) -> np.ndarray:
    qtype = t.tensor_type
    natural_shape = tuple(int(s) for s in reversed(t.shape))
    if qtype == gguf.GGMLQuantizationType.F32:
        return np.frombuffer(t.data, dtype=np.float32).reshape(natural_shape)
    if qtype == gguf.GGMLQuantizationType.F32:
        return np.frombuffer(t.data, dtype=np.float32).astype(np.float32).reshape(natural_shape)
    if qtype == gguf.GGMLQuantizationType.BF16:
        u16 = np.frombuffer(t.data, dtype=np.uint16)
        f32 = np.zeros(u16.shape, dtype=np.uint32)
        f32 |= u16.astype(np.uint32) << 16
        return f32.view(np.float32).reshape(natural_shape)
    raw = np.array(t.data)
    arr = gquants.dequantize(raw, qtype)
    return arr.reshape(natural_shape)


# ────────────────────────────────────────────────────────────────────
# transforms
# ────────────────────────────────────────────────────────────────────


def expand_axis_with_zeros(arr: np.ndarray, axis: int, d: int, expansion: int) -> np.ndarray:
    if arr.shape[axis] != d:
        raise ValueError(f"axis {axis} of shape {arr.shape} is {arr.shape[axis]}, expected {d}")
    pad_shape = list(arr.shape)
    pad_shape[axis] = expansion
    return np.concatenate([arr, np.zeros(tuple(pad_shape), dtype=arr.dtype)], axis=axis)


def identity_pad(arr: np.ndarray, cls: str, d: int, pad: int) -> np.ndarray:
    if cls in ("ne0_read", "ne0_write", "norm"):
        return expand_axis_with_zeros(arr, axis=-1, d=d, expansion=pad)
    if cls == "ne1":
        return expand_axis_with_zeros(arr, axis=0, d=d, expansion=pad)
    raise AssertionError(cls)


def fuse_gamma_pre(linear_arr: np.ndarray, gamma: np.ndarray, kappa: float) -> np.ndarray:
    """Pre-norm fusion: γ row-scales the *following* linear's d-axis.

    Replaces [RMSNorm(γ_per_dim) → Linear(W)] with [RMSNorm(κ·1) → Linear(W')]
    where W' = Diag(γ_per_dim) · W (i.e. γ baked into W as-is, NO κ scaling on γ).

    Verification of cancellation under obfuscation (x_obf = x · P̂_R):
        norm(x_obf; κ·1) · W̃' = κ · (x_obf · Q̂_R) · γ · W / RMS(x_obf)
                              = κ · (x · γ) · W / RMS(x_obf)
        plaintext target:    = (x · γ) · W / RMS(x)
        ratio                = κ · RMS(x)/RMS(x_obf) = κ / κ_correct = 1 when κ = κ_correct.

    In numpy with W natural shape (M, d): W_new = W * γ (broadcast on axis 0).
    The κ correction lives in the *norm site's* scalar γ_obf = κ_correct, NOT
    in the fusion step. kappa is unused here, kept in the signature for clarity."""
    del kappa  # the κ correction lives on the norm side, not in the fusion
    return linear_arr * gamma.astype(linear_arr.dtype)


def estimate_kappa(p_r: np.ndarray, d: int, num_samples: int = 2000, seed: int = 0) -> float:
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
    if arch != "qwen3":
        raise SystemExit(f"unsupported architecture: {arch} (expected qwen3)")
    d = int(r.fields["qwen3.embedding_length"].contents())
    n_layer = int(r.fields["qwen3.block_count"].contents())

    if mode == "gamma-only":
        pad = 0
    else:
        pad = 2 * expansion
    new_d = d + pad
    log.info("arch=%s d=%d 2h=%d new_d=%d n_layer=%d", arch, d, pad, new_d, n_layer)

    # ---- key material (only for keymat mode) ----
    p_r = None
    q_r = None
    q_r_t = None
    p_r_t = None
    kappa = 1.0
    kappa_e = None
    if mode == "keymat":
        sys.path.insert(0, "/home/timo/repos/private-rag-path-2/vendor/aloepri-py")
        sys.path.insert(0, "/home/timo/repos/private-rag-path-2/vendor/aloepri-py/src")
        from keymat import build_keymat_transform  # type: ignore

        log.info("sampling Algorithm 1 at d=%d h=%d λ=%.2f seed=%d", d, expansion, lam, seed)
        transform = build_keymat_transform(d=d, h=expansion, lam=lam, init_seed=seed)
        p_r = transform.key.numpy().astype(np.float64)
        q_r = transform.inverse.numpy().astype(np.float64)
        identity_err = float(np.max(np.abs(p_r @ q_r - np.eye(d))))
        log.info("‖P̂·Q̂ - I_d‖_max = %.3e", identity_err)
        kappa_e = estimate_kappa(p_r, d=d, num_samples=2000, seed=seed + 100)
        kappa = kappa_e * float(np.sqrt(d / float(d + 2 * expansion)))
        log.info("κ_E = %.5f  →  κ_correct = κ_E · √(d/(d+2h)) = %.5f", kappa_e, kappa)
        q_r_t = q_r.T  # (d, d+2h)
        p_r_t = p_r.T  # (d+2h, d)

    # ---- pass 1: load every tensor ----
    log.info("loading + dequantising tensors...")
    arrays: dict[str, np.ndarray] = {t.name: to_float_array(t) for t in r.tensors}
    log.info("loaded %d tensors", len(arrays))

    # ---- pass 2 (keymat / gamma-only): §5.2.5 fusion ----
    if mode in ("keymat", "gamma-only"):
        log.info("§5.2.5 fusion: γ → adjacent linears")
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
                    assert direction == "pre"
                    arrays[tgt_key] = fuse_gamma_pre(arrays[tgt_key], gamma, kappa)
        for norm_name, (direction, targets) in GLOBAL_FUSION_MAP.items():
            if norm_name not in arrays:
                continue
            gamma = arrays[norm_name]
            for tgt_name in targets:
                if tgt_name not in arrays:
                    continue
                assert direction == "pre"
                arrays[tgt_name] = fuse_gamma_pre(arrays[tgt_name], gamma, kappa)
        log.info("γ fusion complete")

    # ---- writer setup + metadata ----
    log.info("applying %s transform to residual-stream tensors", mode)
    writer = gguf.GGUFWriter(str(out_path), arch=arch)
    skipped_keys = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
    for key in sorted(r.fields.keys()):
        if key in skipped_keys:
            continue
        field = r.fields[key]
        value = field.contents()
        if key == "qwen3.embedding_length" and new_d != d:
            value = new_d
            log.info("metadata: %s = %d (was %d)", key, value, d)
        _write_field(writer, key, value, field)
    writer.add_string("aloepri.mode", mode)
    writer.add_uint32("aloepri.expansion_size", expansion if mode != "gamma-only" else 0)
    writer.add_uint32("aloepri.seed", seed)
    if mode == "keymat":
        writer.add_float32("aloepri.kappa_e", float(kappa_e))
        writer.add_float32("aloepri.kappa", float(kappa))
        writer.add_float32("aloepri.lambda", float(lam))

    # ---- pass 3: apply transforms + write ----
    n_changed = 0
    n_unchanged = 0
    for t in r.tensors:
        name = t.name
        arr = arrays[name]
        cls = classify(name)

        if cls == "unchanged":
            out_arr = arr.astype(np.float32) if arr.ndim <= 1 else arr.astype(np.float32)
            out_qtype = (gguf.GGMLQuantizationType.F32 if arr.ndim <= 1
                         else gguf.GGMLQuantizationType.F32)
            writer.add_tensor(name, out_arr, raw_dtype=out_qtype)
            n_unchanged += 1
            continue

        if cls == "norm":
            if mode == "identity-pad":
                out_arr = expand_axis_with_zeros(arr, axis=-1, d=d, expansion=pad).astype(np.float32)
            elif mode == "keymat":
                out_arr = np.full((new_d,), float(kappa), dtype=np.float32)
            else:  # gamma-only
                out_arr = np.full((d,), float(kappa), dtype=np.float32)
            writer.add_tensor(name, out_arr, raw_dtype=gguf.GGMLQuantizationType.F32)
            n_changed += 1
            continue

        # ne0_read / ne0_write / ne1
        if mode == "identity-pad":
            out_arr = identity_pad(arr, cls, d=d, pad=pad).astype(np.float32)
        elif mode == "keymat":
            arr_f64 = arr.astype(np.float64)
            if cls == "ne0_read":
                out_arr = (arr_f64 @ q_r_t).astype(np.float32)
            elif cls == "ne0_write":
                out_arr = (arr_f64 @ p_r).astype(np.float32)
            elif cls == "ne1":
                out_arr = (p_r_t @ arr_f64).astype(np.float32)
            else:
                raise AssertionError(cls)
        else:  # gamma-only — fuse already applied, no dim change, no Algorithm 1
            out_arr = arr.astype(np.float32)
        writer.add_tensor(name, out_arr, raw_dtype=gguf.GGMLQuantizationType.F32)
        n_changed += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    log.info("wrote %s (changed=%d unchanged=%d)", out_path, n_changed, n_unchanged)
    return {"mode": mode, "d": d, "new_d": new_d, "n_changed": n_changed, "kappa": kappa}


def _write_field(writer: gguf.GGUFWriter, key: str, value, field: gguf.ReaderField) -> None:
    types = field.types
    primary = types[0]
    if primary == gguf.GGUFValueType.STRING:
        writer.add_string(key, value)
    elif primary == gguf.GGUFValueType.BOOL:
        writer.add_bool(key, bool(value))
    elif primary == gguf.GGUFValueType.UINT8:    writer.add_uint8(key, int(value))
    elif primary == gguf.GGUFValueType.INT8:     writer.add_int8(key, int(value))
    elif primary == gguf.GGUFValueType.UINT16:   writer.add_uint16(key, int(value))
    elif primary == gguf.GGUFValueType.INT16:    writer.add_int16(key, int(value))
    elif primary == gguf.GGUFValueType.UINT32:   writer.add_uint32(key, int(value))
    elif primary == gguf.GGUFValueType.INT32:    writer.add_int32(key, int(value))
    elif primary == gguf.GGUFValueType.UINT64:   writer.add_uint64(key, int(value))
    elif primary == gguf.GGUFValueType.INT64:    writer.add_int64(key, int(value))
    elif primary == gguf.GGUFValueType.FLOAT32:  writer.add_float32(key, float(value))
    elif primary == gguf.GGUFValueType.FLOAT64:  writer.add_float64(key, float(value))
    elif primary == gguf.GGUFValueType.ARRAY:    writer.add_array(key, list(value))
    else:
        raise ValueError(f"unsupported gguf value type {primary} for key {key}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, required=True)
    parser.add_argument("--out", dest="out_path", type=Path, required=True)
    parser.add_argument("--mode", choices=["identity-pad", "keymat", "gamma-only"], default="keymat")
    parser.add_argument("--expansion-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lam", type=float, default=0.3)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    info = rewrite_gguf(args.in_path, args.out_path, mode=args.mode,
                       expansion=args.expansion_size, seed=args.seed, lam=args.lam)
    log.info("done: %s", info)
    return 0


if __name__ == "__main__":
    sys.exit(main())

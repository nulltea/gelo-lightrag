#!/usr/bin/env python3
"""Untie a tied-embedding Qwen3 GGUF.

Some Qwen3 sizes (0.6B, 4B) ship with `tie_word_embeddings: true` — the
LM head reuses `token_embd.weight` at inference. This script materialises
a separate `output.weight` tensor (byte-identical to `token_embd.weight`)
so downstream tools that operate on the LM head independently (such as
`obfuscate_qwen3_gguf.py`, whose ne0_write / ne0_read transforms assume
distinct input / output tensors) work correctly.

Inference behaviour of the untied GGUF is identical to the tied original
— llama.cpp uses `output.weight` for the LM head when present, and the
tensor is bit-for-bit the same as `token_embd.weight`.

Use:
  python untie_qwen3_gguf.py --in Qwen3-4B-Q8_0.gguf --out Qwen3-4B-Q8_0-untied.gguf
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import gguf
import numpy as np

log = logging.getLogger("untie_qwen3_gguf")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_path", type=Path, required=True)
    p.add_argument("--out", dest="out_path", type=Path, required=True)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    log.info("reading %s", args.in_path)
    r = gguf.GGUFReader(str(args.in_path))

    arch_field = r.fields["general.architecture"]
    arch = arch_field.contents()
    log.info("arch=%s", arch)

    by_name = {t.name: t for t in r.tensors}
    if "token_embd.weight" not in by_name:
        raise SystemExit("input GGUF lacks token_embd.weight")
    if "output.weight" in by_name:
        log.info("input GGUF already has output.weight — nothing to do")
        return 1

    embed_t = by_name["token_embd.weight"]
    log.info("token_embd.weight: shape=%s type=%s",
             list(reversed(list(embed_t.shape))), embed_t.tensor_type)

    # Build the output writer
    writer = gguf.GGUFWriter(str(args.out_path), arch=arch)
    skip_keys = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
    for key in sorted(r.fields.keys()):
        if key in skip_keys:
            continue
        field = r.fields[key]
        # Re-emit each KV faithfully via the writer
        primary = field.types[0]
        v = field.contents()
        if primary == gguf.GGUFValueType.STRING:    writer.add_string(key, v)
        elif primary == gguf.GGUFValueType.BOOL:    writer.add_bool(key, bool(v))
        elif primary == gguf.GGUFValueType.UINT8:   writer.add_uint8(key, int(v))
        elif primary == gguf.GGUFValueType.INT8:    writer.add_int8(key, int(v))
        elif primary == gguf.GGUFValueType.UINT16:  writer.add_uint16(key, int(v))
        elif primary == gguf.GGUFValueType.INT16:   writer.add_int16(key, int(v))
        elif primary == gguf.GGUFValueType.UINT32:  writer.add_uint32(key, int(v))
        elif primary == gguf.GGUFValueType.INT32:   writer.add_int32(key, int(v))
        elif primary == gguf.GGUFValueType.UINT64:  writer.add_uint64(key, int(v))
        elif primary == gguf.GGUFValueType.INT64:   writer.add_int64(key, int(v))
        elif primary == gguf.GGUFValueType.FLOAT32: writer.add_float32(key, float(v))
        elif primary == gguf.GGUFValueType.FLOAT64: writer.add_float64(key, float(v))
        elif primary == gguf.GGUFValueType.ARRAY:   writer.add_array(key, list(v))
        else:
            raise ValueError(f"unsupported kv type {primary} for {key}")

    # Re-emit all tensors verbatim; duplicate token_embd into output.weight
    natural_shape = tuple(int(s) for s in reversed(list(embed_t.shape)))
    # GGUF stores tensor.data as a 1-D numpy view over the raw bytes
    # (a block-quant tensor's view shape may be the block layout, not the
    # natural matrix shape). We pass `raw_dtype` to bypass re-quantisation.
    for t in r.tensors:
        # Copy the raw buffer directly — no recompute, no requant.
        # writer.add_tensor(name, data, raw_dtype=type) preserves bytes.
        writer.add_tensor(t.name, np.asarray(t.data),
                          raw_dtype=t.tensor_type)
        if t.name == "token_embd.weight":
            log.info("duplicating token_embd.weight -> output.weight "
                     "(byte-identical, type=%s)", t.tensor_type)
            writer.add_tensor("output.weight", np.asarray(t.data),
                              raw_dtype=t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    log.info("wrote %s", args.out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

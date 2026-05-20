#!/usr/bin/env python3
"""
Smoke-test builder for the matrix-Γ kernel patch.

Emits two F32 GGUFs from the same plaintext Qwen3 source:

  --out-scalar  → identical-to-source F32 GGUF with 1D γ_q at
                  blk.*.attn_q_norm.weight (legacy kernel path).
  --out-matrix  → F32 GGUF with 2D Diag(γ_q) at the same tensor name
                  and aloepri.qk_norm_matrix = True (patched path).

The matrix-Γ form with Γ = Diag(γ) is algebraically equivalent to the
scalar γ multiplication (M_q = I): every off-diagonal entry is exactly
zero, fp32 multiplication by 0 is exact, so the matmul produces
bit-identical results to the elementwise multiply.

Token output under greedy decoding must match between the two GGUFs.
Any difference means the kernel branch is broken.

Run from python/path-2/ with:
  .venv/bin/python scripts/build_matrix_gamma_identity_gguf.py \
    --in  /path/to/Qwen3-1.7B.gguf \
    --out-scalar /tmp/qwen3-fp32-scalar.gguf \
    --out-matrix /tmp/qwen3-fp32-matrix-identity.gguf
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import gguf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from obfuscate_qwen3_gguf import to_float_array, _write_field  # noqa: E402

log = logging.getLogger("build_matrix_gamma_identity_gguf")


def _write_one(out_path: Path, arch: str, fields, tensors_loaded, head_dim: int,
               *, as_matrix: bool, extra_kv: dict[str, str]) -> None:
    writer = gguf.GGUFWriter(str(out_path), arch=arch)
    skipped_keys = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
    for key in sorted(fields.keys()):
        if key in skipped_keys:
            continue
        field = fields[key]
        _write_field(writer, key, field.contents(), field)
    for k, v in extra_kv.items():
        if isinstance(v, bool):
            writer.add_bool(k, v)
        else:
            writer.add_string(k, v)
    n_expanded = 0
    n_other = 0
    for name, arr in tensors_loaded.items():
        if (
            as_matrix
            and name.startswith("blk.")
            and (name.endswith(".attn_q_norm.weight") or name.endswith(".attn_k_norm.weight"))
        ):
            assert arr.shape == (head_dim,), f"unexpected {name} shape {arr.shape}"
            out_arr = np.diag(arr).astype(np.float32)
            n_expanded += 1
        else:
            out_arr = arr.astype(np.float32)
            n_other += 1
        writer.add_tensor(name, out_arr, raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    log.info("wrote %s (matrix=%s expanded=%d other=%d)",
             out_path, as_matrix, n_expanded, n_other)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_path", type=Path, required=True)
    p.add_argument("--out-scalar", dest="out_scalar", type=Path, required=True)
    p.add_argument("--out-matrix", dest="out_matrix", type=Path, required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    log.info("opening %s", args.in_path)
    r = gguf.GGUFReader(str(args.in_path))
    arch = r.fields["general.architecture"].contents()
    if arch != "qwen3":
        raise SystemExit(f"unsupported architecture: {arch}")
    head_dim = int(r.fields["qwen3.attention.key_length"].contents())
    n_layer = int(r.fields["qwen3.block_count"].contents())
    log.info("arch=qwen3 n_layer=%d head_dim=%d", n_layer, head_dim)

    log.info("dequantising all tensors to fp32 …")
    tensors: dict[str, np.ndarray] = {t.name: to_float_array(t) for t in r.tensors}
    log.info("loaded %d tensors", len(tensors))

    _write_one(args.out_scalar, arch, r.fields, tensors, head_dim,
               as_matrix=False,
               extra_kv={"aloepri.mode": "matrix-gamma-smoke-baseline-fp32"})
    _write_one(args.out_matrix, arch, r.fields, tensors, head_dim,
               as_matrix=True,
               extra_kv={
                   "aloepri.mode": "matrix-gamma-identity-smoke",
                   "aloepri.qk_norm_matrix": True,
               })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

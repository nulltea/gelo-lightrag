"""M2.7 capture (Option B — file-dump protocol).

Drives an `aloepri-llama-server:m2_7` instance that was launched with
``--tensor-filter REGEX --tensor-dump-path FILE`` set. For each
prompt:

1. Truncate the dump file (delimits one forward pass from the next).
2. POST `/completion` with the τ-mapped obfuscated prompt ids and
   ``n_predict = 1`` so the forward pass actually runs.
3. Read the dump file. Parse the binary records (header layout
   defined in `common/debug.cpp` under the M2.7 patch).
4. Group records by tensor name, derive (layer, kind) from the name
   suffix (e.g. ``attn_norm-23`` → layer 23 kind ``attn_norm``).
5. Pack into the safetensors schema already consumed by
   ``snapshots_loader.SnapshotSet`` so the existing attack drivers
   (``run_nn``, ``run_ima``, ``run_isa``, ``run_ima_paper_like``)
   work unchanged.

Server spawn (HiddenState pass — flash-attn on is fine). Use the
GPU-safe launcher; it passes render/video groups plus `/dev/dri` and
`/dev/kfd` when present so the server does not silently fall back to CPU:

  OBF_GGUF=/home/timo/.cache/huggingface/path-2-aloepri/qwen3-1.7b/keymat-h128-pi-noise-alg2-fp32.gguf \
  PORT=8061 CONTAINER=aloepri-m2_7-server \
  TENSOR_FILTER='^attn_norm-(0|11|23)$' \
  TENSOR_DUMP_PATH=/dump/m2_7_dump.bin DUMP_DIR=/tmp \
  evals/aloepri-attacks/m2_7/spawn_obfuscated_server.sh

Server spawn (AttnScore pass — flash-attn MUST be off so `kq-*` materialises):

  FLASH_ATTN=off TENSOR_FILTER='^kq-23$' \
  TENSOR_DUMP_PATH=/dump/m2_7_dump.bin DUMP_DIR=/tmp \
  evals/aloepri-attacks/m2_7/spawn_obfuscated_server.sh

The Python wrapper accesses ``/dump/m2_7_dump.bin`` through the same
shared mount as the server (``--dump-path /tmp/m2_7_dump.bin``).
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import requests

# Import key + tokenizer machinery from the existing AloePri client.
PATH2 = Path("/home/timo/repos/private-rag-path-2/python/aloepri-llm")
sys.path.insert(0, str(PATH2))
from aloepri_client import AloePriClient  # type: ignore  # noqa: E402

# Safetensors writer.
from safetensors.numpy import save_file


# GGML dtype enum subset we accept.
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


def parse_dump_file(path: Path) -> list[dict]:
    """Parse the binary dump format the M2.7 patched
    `common/debug.cpp` writes. Returns a list of records keyed by
    tensor name with f32 numpy arrays.

    Record layout (little-endian):
      u32  name_len
      char name[name_len]
      i64  ne[4]
      u8   dtype       (0=F32, 1=F16)
      u64  n_bytes
      char data[n_bytes]
    """
    records: list[dict] = []
    with path.open("rb") as fh:
        while True:
            head = fh.read(4)
            if not head:
                break
            if len(head) < 4:
                raise RuntimeError(f"truncated record header at offset {fh.tell()}")
            (name_len,) = struct.unpack("<I", head)
            name = fh.read(name_len).decode("utf-8")
            ne_bytes = fh.read(8 * 4)
            ne = struct.unpack("<4q", ne_bytes)
            (dtype,) = struct.unpack("<B", fh.read(1))
            (n_bytes,) = struct.unpack("<Q", fh.read(8))
            raw = fh.read(n_bytes)
            if len(raw) < n_bytes:
                raise RuntimeError(
                    f"truncated tensor body for {name!r} at offset {fh.tell()}"
                )
            if dtype == GGML_TYPE_F32:
                arr = np.frombuffer(raw, dtype=np.float32).copy()
            elif dtype == GGML_TYPE_F16:
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32, copy=False)
            else:
                # Quantised or unsupported; skip with a note.
                records.append({"name": name, "ne": ne, "dtype": dtype, "skipped": True})
                continue
            arr = arr.reshape([d for d in ne if d > 0][::-1])
            records.append({"name": name, "ne": ne, "dtype": dtype, "data": arr})
    return records


def split_name(name: str) -> tuple[str, int] | None:
    """`attn_norm-23` → ('attn_norm', 23). Returns None if no layer suffix."""
    if "-" not in name:
        return None
    base, suffix = name.rsplit("-", 1)
    try:
        return base, int(suffix)
    except ValueError:
        return None


def truncate(path: Path) -> None:
    """Open in write mode and immediately close — 0-bytes the file."""
    with path.open("wb"):
        pass


sys.path.insert(0, str(Path(__file__).resolve().parent))
from m2_7_common import (  # noqa: E402
    add_min_mem_args,
    bounded_buffer_check,
    check_phase_memory,
)


def main() -> int:
    p = argparse.ArgumentParser(description="M2.7 capture frontend (Option B file-dump)")
    p.add_argument("--endpoint", default="http://127.0.0.1:8061",
                   help="Patched llama-server base URL")
    p.add_argument("--key-path", type=Path, required=True,
                   help="Path to .key.npz for the obfuscated GGUF "
                        "(used by AloePriClient to τ-map tokens)")
    p.add_argument("--dump-path", type=Path, required=True,
                   help="Same path the server is writing dumps to "
                        "(operator-side mount of --tensor-dump-path)")
    p.add_argument("--prompts-file", type=Path, required=True)
    p.add_argument("--max-prompts", type=int, default=8)
    p.add_argument("--max-prompt-tokens", type=int, default=32)
    p.add_argument("--mode", choices=("hidden", "attn"), required=True,
                   help="hidden → expects --tensor-filter '^attn_norm-(0|...)$' "
                        "on the server; attn → expects '^kq-(...)$' + --flash-attn off")
    p.add_argument("--output", type=Path, required=True,
                   help="Output safetensors path")
    p.add_argument("--meta-output", type=Path, default=None,
                   help="Sidecar meta.json (default: <output>.meta.json)")
    p.add_argument("--smoke", action="store_true",
                   help="One prompt + sketch of recovered tensor names")
    p.add_argument("--no-tau-map", action="store_true",
                   help="Skip the τ obfuscation map and send plain token ids "
                        "directly. Use this when targeting a plaintext GGUF "
                        "(the Plain control column in §08).")
    p.add_argument("--max-tensors-mb", type=float, default=8192.0,
                   help="Refuse to keep accumulating if the in-memory "
                        "tensors dict grows past this many MB. Guards "
                        "against a too-permissive --tensor-filter "
                        "regex on the server side.")
    add_min_mem_args(p, phase="hidden_capture")
    args = p.parse_args()

    check_phase_memory("hidden_capture", args.min_mem_gb, args.skip_mem_check)
    print(f"[M2.7 capture] mode={args.mode} endpoint={args.endpoint}")
    print(f"[M2.7 capture] dump_path={args.dump_path}")

    client = AloePriClient(endpoint=args.endpoint, key_path=args.key_path)

    prompts = [
        line.strip()
        for line in args.prompts_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    n = min(args.max_prompts, len(prompts)) if not args.smoke else 1
    prompts = prompts[:n]

    tensors: dict[str, np.ndarray] = {}
    snapshots: list[dict] = []
    prompt_token_ids: list[list[int]] = []
    seq_idx = 0
    captured_kinds: set[str] = set()
    completion_endpoint = args.endpoint.rstrip("/") + "/completion"

    t0 = time.perf_counter()
    for prompt_idx, prompt_text in enumerate(prompts):
        plain_ids = client._encode(prompt_text)[: args.max_prompt_tokens]  # type: ignore[attr-defined]
        if args.no_tau_map:
            wire_ids = plain_ids                                            # Plain control: send plain ids
        else:
            wire_ids = client._to_obf(plain_ids)                            # Default: τ-map for the obfuscated server  # type: ignore[attr-defined]
        prompt_token_ids.append(plain_ids)

        # Truncate before the call so we only capture this forward's tensors.
        truncate(args.dump_path)

        # n_predict=1 so the forward pass runs through all layers and our
        # tensor callback fires. The single generated token is discarded.
        # chat_parser=epsilon overrides llama-server's default PEG content
        # grammar so update_chat_msg never throws on strong-Π gibberish
        # output — see python/aloepri-llm/aloepri_client.py for the rationale.
        body = {
            "prompt": wire_ids,
            "n_predict": 1,
            "temperature": 0.0,
            "seed": 0,
            "cache_prompt": False,
            "stream": False,
            "chat_parser": '{"parsers":[{"type":"epsilon"}],"rules":{},"root":0}',
        }
        resp = requests.post(completion_endpoint, json=body, timeout=180)
        resp.raise_for_status()

        # Read the dump file (server has flushed it before responding —
        # debug.cpp opens/closes per record, so by response time the
        # final write is on disk).
        records = parse_dump_file(args.dump_path)

        # Pack matching records into the snapshot safetensors.
        for rec in records:
            if rec.get("skipped"):
                continue
            split = split_name(rec["name"])
            if split is None:
                continue
            kind, layer = split
            arr = rec["data"]
            # Squeeze leading singleton dims for cleaner shape contracts.
            arr = np.squeeze(arr)
            if arr.ndim == 1:
                arr = arr[None, :]
            shape = list(arr.shape)
            key = f"snap{seq_idx:05d}.{layer:03d}.{kind}.operand"
            tensors[key] = arr.astype(np.float32, copy=False)
            snapshots.append({
                "seq_idx": seq_idx,
                "prompt_idx": prompt_idx,
                "layer": layer,
                "kind": kind,
                "operand_shape": shape if len(shape) == 2 else
                                 [shape[0], int(np.prod(shape[1:]))],
                "output_shape": None,
                "n_data": shape[0] if len(shape) >= 1 else 0,
                "shield_k": 0,
            })
            captured_kinds.add(kind)
            seq_idx += 1

        print(
            f"  prompt[{prompt_idx:03}] (n_tok={len(plain_ids):>3}) "
            f"→ {len(records)} records → {sum(1 for r in records if not r.get('skipped'))} kept",
            flush=True,
        )

        # Bounded-buffer check — refuse to keep growing if a too-loose
        # filter is matching half the graph.
        current_bytes = sum(a.nbytes for a in tensors.values())
        bounded_buffer_check(
            current_bytes=current_bytes,
            limit_bytes=int(args.max_tensors_mb * 1024 * 1024),
            phase="hidden_capture",
            what="tensors dict",
        )

        if args.smoke:
            print(f"[M2.7 capture] smoke OK. Sample tensor names: "
                  f"{[r['name'] for r in records[:8]]}")
            break

    if not tensors:
        print(f"[M2.7 capture] WARNING — no tensors captured. "
              f"Check the server's --tensor-filter regex matches what the "
              f"forward pass emits (e.g. ^attn_norm-0$).")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output))
    meta_path = args.meta_output or args.output.with_suffix(".meta.json")
    meta = {
        "schema_version": "1",
        "model_id": "Qwen/Qwen3-1.7B (obfuscated, keymat-h128-pi-noise-alg2-fp32)",
        "condition": f"m2_7_{args.mode}",
        "config": {
            "shield_k": 0,
            "shield_energy_scale": 0.0,
            "per_forward_mask": False,
            "verify_probes": 0,
            "prompt_token_ids": prompt_token_ids,
            "captured_layers": sorted({s["layer"] for s in snapshots}),
            "captured_kinds": sorted(captured_kinds),
        },
        "snapshots": snapshots,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    elapsed = time.perf_counter() - t0
    print(f"[M2.7 capture] {len(prompts)} prompt(s) in {elapsed:.1f} s → "
          f"{args.output} ({len(tensors)} tensors)")
    print(f"[M2.7 capture] meta → {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

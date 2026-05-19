"""Drive the §05 obfuscated `llama-server` with the prompt corpus and
record the resulting obfuscated-token-id streams.

TFMA + SDA both consume the wire-side `τ(token_id)` stream. The
existing `AloePriClient` (`python/path-2/aloepri_client.py`) already
implements the tokenise → τ-map → /completion → response protocol;
we wrap it here to:

1. Tokenise the plaintext prompt and τ-map to obfuscated ids
   (these are the prompt-side ciphertext the attacker observes).
2. POST `/completion` to llama-server, get the obfuscated response
   token-id sequence (the response-side ciphertext).
3. Save (plain_prompt_ids, obf_prompt_ids, obf_response_ids) per
   prompt to JSONL for the attack drivers.

This script does NOT spawn the server — use `spawn_obfuscated_server.sh`
first, or point `--endpoint` at an already-running instance.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Import AloePriClient from the path-2 codebase.
PATH2 = Path("/home/timo/repos/private-rag-path-2/python/path-2")
sys.path.insert(0, str(PATH2))
from aloepri_client import AloePriClient, KeyMaterial  # type: ignore


def main() -> int:
    p = argparse.ArgumentParser(description="Capture obfuscated token streams from §05 llama-server")
    p.add_argument("--endpoint", default="http://127.0.0.1:8061",
                   help="llama-server base URL (no trailing /completion)")
    p.add_argument("--key-path", type=Path, required=True,
                   help="Path to .key.npz for the obfuscated GGUF")
    p.add_argument("--prompts-file", type=Path, required=True,
                   help="One prompt per line, UTF-8")
    p.add_argument("--max-prompts", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True,
                   help="JSONL output: one record per prompt")
    p.add_argument("--smoke", action="store_true",
                   help="Capture only the first prompt and print the record")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from m2_7_common import add_min_mem_args, check_phase_memory  # type: ignore
    add_min_mem_args(p, phase="token_capture")
    args = p.parse_args()

    check_phase_memory("token_capture", args.min_mem_gb, args.skip_mem_check)
    print(f"[M2.7 token-capture] endpoint={args.endpoint}")
    print(f"[M2.7 token-capture] key={args.key_path}")

    client = AloePriClient(endpoint=args.endpoint, key_path=args.key_path)

    prompts = [
        line.strip()
        for line in args.prompts_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    n = min(args.max_prompts, len(prompts)) if not args.smoke else 1
    prompts = prompts[:n]
    print(f"[M2.7 token-capture] {len(prompts)} prompt(s); max_new_tokens={args.max_new_tokens}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with args.output.open("w") as out_fh:
        for prompt_idx, prompt_text in enumerate(prompts):
            t_prompt = time.perf_counter()
            # Tokenise + τ-map locally.
            plain_prompt_ids = client._encode(prompt_text)  # type: ignore[attr-defined]
            obf_prompt_ids = client._to_obf(plain_prompt_ids)  # type: ignore[attr-defined]

            # POST to llama-server with the τ-mapped prompt.
            import requests

            payload = {
                "prompt": obf_prompt_ids,
                "n_predict": args.max_new_tokens,
                "temperature": args.temperature,
                "seed": args.seed,
                "return_tokens": True,
                "stream": False,
                "cache_prompt": False,
            }
            try:
                resp = requests.post(
                    client.endpoint, json=payload, timeout=180
                )
                resp.raise_for_status()
                body = resp.json()
            except Exception as exc:
                print(f"  prompt[{prompt_idx:03}] FAILED: {exc!r}", flush=True)
                record = {
                    "prompt_idx": prompt_idx,
                    "prompt_text": prompt_text,
                    "plain_prompt_ids": plain_prompt_ids,
                    "obf_prompt_ids": obf_prompt_ids,
                    "obf_response_ids": None,
                    "error": str(exc),
                }
                out_fh.write(json.dumps(record) + "\n")
                continue

            obf_response_ids = body.get("tokens") or []
            record = {
                "prompt_idx": prompt_idx,
                "prompt_text": prompt_text,
                "plain_prompt_ids": plain_prompt_ids,
                "obf_prompt_ids": obf_prompt_ids,
                "obf_response_ids": obf_response_ids,
                "content_decoded": body.get("content", ""),
            }
            out_fh.write(json.dumps(record) + "\n")
            out_fh.flush()
            print(
                f"  prompt[{prompt_idx:03}] ({len(plain_prompt_ids):>3} → {len(obf_response_ids):>3} tok)"
                f" — {time.perf_counter() - t_prompt:.2f} s — {prompt_text[:48]!r}…",
                flush=True,
            )
            if args.smoke:
                print(f"[M2.7 token-capture] smoke OK. Record: {json.dumps(record, indent=2)[:800]}…")
                break

    elapsed = time.perf_counter() - t0
    print(f"[M2.7 token-capture] {len(prompts)} prompt(s) in {elapsed:.1f} s → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

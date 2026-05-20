"""AloePri Python client wrapper for Qwen3 obfuscated GGUF artifacts.

Per-request protocol (item 6 of the deferred-work list, see
docs/plans/path-2-aloepri-next-steps.md):

  client (trusted; holds τ):
    1. tokenize(plaintext) → plain_ids
    2. obf_ids = τ[plain_ids]
    3. POST /completion {"prompt": [obf_ids], "return_tokens": true, ...}
    4. response: obf_response_ids = response["tokens"]
    5. plain_response_ids = τ⁻¹[obf_response_ids]
    6. detokenize(plain_response_ids)

We send and receive token IDs over the wire (the llama.cpp native
`/completion` endpoint accepts an int array as `prompt` and can
return the sampled token ids). This avoids a tokenize↔detokenize
roundtrip on the text on the wire, which would otherwise be a known
edge-case minefield with BPE merges and special tokens.

The wire payload (obf_ids array) carries no plaintext semantic
information: an attacker without τ sees only a sequence of integers
indexing into an obfuscated vocabulary.

Streaming is **not** supported in v1 (paper §5.3 partial-token
recovery edge cases punt to v2 in the plan).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from tokenizers import Tokenizer


@dataclass
class KeyMaterial:
    tau: np.ndarray         # shape (vocab_size,), int64; plain_id → obf_id
    inv_tau: np.ndarray     # shape (vocab_size,), int64; obf_id → plain_id
    vocab_size: int
    active_size: int        # range [0, active_size) is permuted; rest is identity
    pi_seed: int
    arch: str

    @classmethod
    def load(cls, path: Path) -> "KeyMaterial":
        z = np.load(path, allow_pickle=False)
        tau = z["tau"].astype(np.int64)
        return cls(
            tau=tau,
            inv_tau=np.argsort(tau).astype(np.int64),
            vocab_size=int(z["vocab_size"]),
            active_size=int(z["active_size"]),
            pi_seed=int(z["pi_seed"]),
            arch=str(z["arch"]),
        )


class AloePriClient:
    def __init__(
        self,
        endpoint: str,
        key_path: Path,
        tokenizer_id: str = "Qwen/Qwen3-1.7B",
    ):
        if not endpoint.endswith("/completion"):
            endpoint = endpoint.rstrip("/") + "/completion"
        self.endpoint = endpoint
        self.key = KeyMaterial.load(key_path)
        self.tokenizer = Tokenizer.from_pretrained(tokenizer_id)
        active_t = self.tokenizer.get_vocab_size(with_added_tokens=True)
        if active_t != self.key.active_size:
            raise ValueError(
                f"tokenizer active vocab {active_t} != key active range "
                f"{self.key.active_size}; refusing to map IDs through τ "
                f"(would corrupt outputs)."
            )

    def _encode(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if any(i >= self.key.active_size for i in ids):
            raise RuntimeError(
                "tokenizer emitted an id outside the active range — "
                "should not happen with Qwen3 BPE on normal input."
            )
        return ids

    def _to_obf(self, plain_ids: list[int]) -> list[int]:
        return self.key.tau[np.asarray(plain_ids, dtype=np.int64)].tolist()

    def _from_obf(self, obf_ids: list[int]) -> list[int]:
        return self.key.inv_tau[np.asarray(obf_ids, dtype=np.int64)].tolist()

    def _decode(self, plain_ids: list[int]) -> str:
        return self.tokenizer.decode(plain_ids, skip_special_tokens=False)

    def complete(
        self,
        prompt: str,
        max_tokens: int = 32,
        *,
        temperature: float = 0.0,
        seed: int = 0,
        timeout: float = 300.0,
    ) -> dict:
        plain_ids = self._encode(prompt)
        obf_ids = self._to_obf(plain_ids)
        # Stream mode is REQUIRED here. In non-stream mode, llama-server's
        # response pipeline runs a PEG chat-template parser on the
        # aggregated `content` string at the end of generation (see
        # common/chat.cpp:2536). With Π applied, the server's tokenizer
        # detokenises our obfuscated wire-token stream into multilingual
        # gibberish (since position `i` in the tokenizer's string table
        # is unchanged but the embedding/head rows at `i` were permuted),
        # the PEG parser fails on that gibberish, and llama-server returns
        # HTTP 500 — even though the integer `tokens` array we actually
        # consume is fine.
        # In stream mode the final response carries empty content (the
        # data is in per-chunk deltas), so the aggregate parse never
        # runs. We accumulate the obfuscated token IDs from the SSE
        # chunks and rely on τ⁻¹ + the client-side tokenizer for the
        # plaintext side. Diagnostic + fix: see
        # docs/handoffs/2026-05-20-aloepri-pi-special-token-fix.md and
        # the follow-on stream-mode handoff for the chat-template trap.
        body = {
            "prompt": obf_ids,
            "n_predict": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "stream": True,
            "return_tokens": True,
            # Why ignore_eos=True: α_e noise on the embedding/head subtly
            # biases logits, so the model occasionally emits an EOS token
            # earlier than it would in the unobfuscated path. For benchmark
            # evaluation we want to give the model a fixed budget to
            # produce a complete output and trim post-hoc on plaintext-
            # side stop sequences (matches the harness's existing
            # `_trim_at_stops`). Production deployments that need natural
            # EOS termination should flip this back to false at the call
            # site after the obfuscation is fully calibrated.
            "ignore_eos": True,
        }
        obf_resp_ids: list[int] = []
        final_meta: dict = {}
        # Read raw bytes and decode per-line with errors='replace'. Reason:
        # `iter_lines(decode_unicode=True)` silently aborts the stream when
        # the SSE chunk's `content` field carries invalid UTF-8 (which
        # happens routinely with Π'd outputs — the obfuscated wire tokens
        # detokenise to multilingual gibberish via the server tokenizer).
        # We don't care about the `content` field — we want the `tokens`
        # array — so an unrecoverable replacement char in content is fine.
        with requests.post(self.endpoint, json=body, timeout=timeout, stream=True) as r:
            r.raise_for_status()
            for raw_bytes in r.iter_lines(decode_unicode=False):
                if not raw_bytes:
                    continue
                raw = raw_bytes.decode("utf-8", errors="replace")
                if not raw.startswith("data:"):
                    continue
                payload = raw[len("data:"):].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    # A per-chunk JSON parse failure is unrecoverable; bail
                    # with what we have so we can still return partial tokens.
                    break
                # Per-chunk shape: {"content": "...", "tokens": [N], "stop": false, ...}
                # Final chunk: {"content": "", "tokens": [], "stop": true, "timings": {...}, ...}.
                tok = chunk.get("tokens")
                if isinstance(tok, list):
                    obf_resp_ids.extend(int(t) for t in tok)
                if chunk.get("stop"):
                    final_meta = {k: v for k, v in chunk.items()
                                  if k not in ("tokens", "content")}
                    break
        plain_resp_ids = self._from_obf(obf_resp_ids)
        # Sanity: every returned id should be in active range after τ⁻¹,
        # because τ leaves [active_size, vocab_size) as identity and we
        # constructed the artifact so the model can only address active ids.
        out_of_range = [i for i in plain_resp_ids if i >= self.key.active_size]
        text = self._decode(plain_resp_ids)
        return {
            "text": text,
            "plain_ids": plain_resp_ids,
            "obf_ids": obf_resp_ids,
            "out_of_range_ids": out_of_range,
            "server_raw": final_meta,
        }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True, help="http://host:port")
    ap.add_argument("--key", required=True, type=Path)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    args = ap.parse_args(argv)

    client = AloePriClient(args.endpoint, args.key, tokenizer_id=args.tokenizer)
    out = client.complete(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
    print(json.dumps(
        {
            "text": out["text"],
            "n_tokens": len(out["plain_ids"]),
            "out_of_range_ids": out["out_of_range_ids"],
            "stop_reason": out["server_raw"].get("stop_type")
                or out["server_raw"].get("stopped_eos")
                or out["server_raw"].get("stop", None),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())

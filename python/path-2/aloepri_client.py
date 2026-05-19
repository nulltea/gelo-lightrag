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
        body = {
            "prompt": obf_ids,
            "n_predict": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "stream": False,
            "return_tokens": True,
        }
        r = requests.post(self.endpoint, json=body, timeout=timeout)
        r.raise_for_status()
        resp = r.json()
        obf_resp_ids = resp.get("tokens")
        if obf_resp_ids is None:
            raise RuntimeError(
                f"server did not return tokens array; response keys: {list(resp)}"
            )
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
            "server_raw": {k: resp[k] for k in resp if k != "tokens"},
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

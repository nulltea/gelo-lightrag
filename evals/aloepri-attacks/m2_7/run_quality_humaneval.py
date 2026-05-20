"""Per-condition AloePri quality + HumanEval pass@1 gate.

For each (α_e, λ, h, β, γ) sweep cell we want a row of three numbers:

  (attack metric)  (quality OK?)  (HumanEval pass@1)

Attack lives in run_static_attacks.py + run_ima_embedrow_attacks.py.
This driver covers the other two columns. It assumes a llama-server is
running against the obfuscated GGUF at --endpoint; it routes everything
through AloePriClient so τ is applied to the prompt and τ⁻¹ to the
response — i.e., we test the actual paper protocol, not the raw
server's gibberish output.

Quality probe (always runs, ~5-10 s):
  Five fixed prompts spanning factual / reasoning / instruction-follow.
  Dump the de-obfuscated response + a coarse "human-readable" heuristic
  (≥ 60% ASCII printable, no >5x repetition of any token). Operator
  inspects the JSON for actual readability — the heuristic only flags
  obvious collapse.

HumanEval pass@1 (default n=50, paper subset; --fast-n 20 for sweep
crank, --skip-humaneval to omit entirely):
  Loads the HumanEval examples via the existing
  python/path-2/evals/tasks/humaneval.py module, but routes every
  completion through AloePriClient instead of the OpenAI-compat
  endpoint. Stop sequences are post-processed on the de-obfuscated
  text (the server can't match plain-text stops because the GGUF's
  vocabulary is τ-permuted).

Output JSON shape:
  {
    "endpoint": "...",
    "key_path": "...",
    "quality": {
      "prompts": [{"prompt": "...", "response": "...", "readable": true, ...}],
      "all_readable": bool,
    },
    "humaneval": {
      "n": 50, "passed": 31, "accuracy": 0.62, "details": [...]
    } | {"skipped": true}
  }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# AloePriClient lives under python/path-2 (same path used by capture_token_streams.py).
PATH2 = Path("/home/timo/repos/private-rag-path-2/python/path-2")
sys.path.insert(0, str(PATH2))
from aloepri_client import AloePriClient  # type: ignore

# HumanEval helpers — reuse the existing harness for load_examples / run_check.
# Import as the package `evals.tasks.humaneval` so the module's relative
# `from ..runner import …` resolves correctly.
from evals.tasks.humaneval import (  # type: ignore
    MAX_TOKENS as HE_MAX_TOKENS,
    STOP_SEQUENCES as HE_STOP_SEQUENCES,
    load_examples as he_load_examples,
    run_check as he_run_check,
)

QUALITY_PROMPTS = [
    "The capital of France is",
    "Q: What is 17 times 4? A:",
    "Write a haiku about autumn leaves:",
    "def is_prime(n):",
    "Translate to French: 'Good morning, how are you?' →",
]


def _is_human_readable(text: str) -> tuple[bool, str]:
    """Coarse heuristic: flag obvious collapse modes (non-printable,
    runaway repetition). Doesn't certify quality — operator reads the
    response field for that.
    """
    if not text:
        return False, "empty"
    printable = sum(1 for c in text if c.isprintable() or c in "\n\t")
    if printable / len(text) < 0.60:
        return False, f"non_printable_ratio={1 - printable/len(text):.2f}"
    # Crude repetition check: any 4-char substring appearing > 8x in 200 chars?
    sample = text[:400]
    for i in range(len(sample) - 4):
        ngram = sample[i : i + 4]
        if ngram.strip() and sample.count(ngram) > 8:
            return False, f"repetition_ngram={ngram!r}_count={sample.count(ngram)}"
    return True, "ok"


def _trim_at_stops(text: str, stops: list[str]) -> str:
    cut = len(text)
    for s in stops:
        idx = text.find(s)
        if idx >= 0 and idx < cut:
            cut = idx
    return text[:cut]


class _PlainClient:
    """Drop-in for AloePriClient that talks to a plain llama-server
    (no τ mapping). Same `.complete()` shape so the rest of the driver
    can stay symmetric — used to capture the plain-reference HumanEval
    pass@1 we diff against the obfuscated cell.
    """

    def __init__(self, endpoint: str, tokenizer_id: str = "Qwen/Qwen3-1.7B"):
        if not endpoint.endswith("/completion"):
            endpoint = endpoint.rstrip("/") + "/completion"
        self.endpoint = endpoint
        from tokenizers import Tokenizer  # type: ignore

        self.tokenizer = Tokenizer.from_pretrained(tokenizer_id)

    def complete(
        self,
        prompt: str,
        max_tokens: int = 32,
        *,
        temperature: float = 0.0,
        seed: int = 0,
        timeout: float = 300.0,
    ) -> dict:
        import requests

        body = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "stream": False,
            "return_tokens": True,
        }
        r = requests.post(self.endpoint, json=body, timeout=timeout)
        r.raise_for_status()
        resp = r.json()
        text = resp.get("content", "")
        return {
            "text": text,
            "plain_ids": resp.get("tokens") or [],
            "obf_ids": resp.get("tokens") or [],
            "out_of_range_ids": [],
            "server_raw": {k: resp[k] for k in resp if k != "tokens"},
        }


def run_quality_probe(client: AloePriClient, *, max_tokens: int = 48) -> dict[str, Any]:
    records = []
    all_ok = True
    t0 = time.perf_counter()
    for i, prompt in enumerate(QUALITY_PROMPTS):
        t_p = time.perf_counter()
        try:
            out = client.complete(prompt, max_tokens=max_tokens, temperature=0.0, seed=0)
            response = out["text"]
            ok, reason = _is_human_readable(response)
            records.append({
                "idx": i,
                "prompt": prompt,
                "response": response,
                "readable": ok,
                "readable_reason": reason,
                "n_response_tokens": len(out["plain_ids"]),
                "wall_s": time.perf_counter() - t_p,
            })
            if not ok:
                all_ok = False
            print(
                f"  q[{i}] readable={ok} ({reason}) "
                f"[{len(out['plain_ids']):>3} tok, "
                f"{time.perf_counter() - t_p:.1f} s] {prompt[:40]!r} → {response[:60]!r}",
                flush=True,
            )
        except Exception as exc:
            all_ok = False
            records.append({
                "idx": i,
                "prompt": prompt,
                "error": str(exc),
                "readable": False,
                "readable_reason": "error",
            })
            print(f"  q[{i}] ERROR: {exc!r}", flush=True)
    return {
        "prompts": records,
        "all_readable": all_ok,
        "wall_s": time.perf_counter() - t0,
    }


def run_humaneval_through_client(
    client: AloePriClient, *, n_examples: int | None = None
) -> dict[str, Any]:
    examples = he_load_examples()
    if n_examples is not None:
        examples = examples[:n_examples]
    passed = 0
    details: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for i, ex in enumerate(examples):
        t_p = time.perf_counter()
        try:
            out = client.complete(
                ex.prompt,
                max_tokens=HE_MAX_TOKENS,
                temperature=0.0,
                seed=0,
            )
            completion = _trim_at_stops(out["text"], HE_STOP_SEQUENCES)
            ok, err = he_run_check(ex.prompt, completion, ex.test, ex.entry_point)
            if ok:
                passed += 1
            details.append({
                "task_id": ex.task_id,
                "passed": ok,
                "completion": completion[:600],
                "error": err if not ok else "",
                "n_response_tokens": len(out["plain_ids"]),
                "wall_s": time.perf_counter() - t_p,
            })
            print(
                f"  he[{i:>2}/{len(examples)}] {ex.task_id:<16} "
                f"{'PASS' if ok else 'fail'} "
                f"({len(out['plain_ids']):>3} tok, {time.perf_counter()-t_p:.1f} s) "
                f"{'' if ok else err[:80]}",
                flush=True,
            )
        except Exception as exc:
            details.append({
                "task_id": ex.task_id,
                "passed": False,
                "completion": "",
                "error": f"client_error: {exc!r}",
            })
            print(f"  he[{i:>2}/{len(examples)}] {ex.task_id} ERROR: {exc!r}", flush=True)
    return {
        "n": len(examples),
        "passed": passed,
        "accuracy": passed / len(examples) if examples else 0.0,
        "wall_s": time.perf_counter() - t0,
        "details": details,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="AloePri quality + HumanEval gate")
    ap.add_argument("--endpoint", required=True,
                    help="llama-server base URL (e.g. http://127.0.0.1:8061)")
    ap.add_argument("--key", type=Path,
                    help="Path to .key.npz for the obfuscated GGUF "
                         "(required unless --plain-mode)")
    ap.add_argument("--plain-mode", action="store_true",
                    help="Skip τ — point at a plain (non-obfuscated) llama-server "
                         "to capture the reference HumanEval pass@1 we diff against.")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n-humaneval", type=int, default=50,
                    help="Number of HumanEval problems (paper default 50)")
    ap.add_argument("--skip-humaneval", action="store_true",
                    help="Quality probe only; skip HumanEval pass@1")
    ap.add_argument("--skip-quality", action="store_true",
                    help="HumanEval only; skip quality probe")
    args = ap.parse_args()

    print(f"[M2.7 quality+humaneval] endpoint={args.endpoint}")
    if args.plain_mode:
        print(f"[M2.7 quality+humaneval] plain-mode (no τ)")
        client = _PlainClient(endpoint=args.endpoint, tokenizer_id=args.tokenizer)
    else:
        if args.key is None:
            print("[M2.7 quality+humaneval] --key required unless --plain-mode")
            return 2
        print(f"[M2.7 quality+humaneval] key={args.key}")
        client = AloePriClient(
            endpoint=args.endpoint, key_path=args.key, tokenizer_id=args.tokenizer
        )

    out: dict[str, Any] = {
        "endpoint": args.endpoint,
        "key_path": str(args.key) if args.key else None,
        "plain_mode": args.plain_mode,
        "tokenizer": args.tokenizer,
    }

    if not args.skip_quality:
        print("[M2.7 quality+humaneval] quality probe (5 prompts)…")
        out["quality"] = run_quality_probe(client)
        print(f"  → all_readable={out['quality']['all_readable']} "
              f"({out['quality']['wall_s']:.1f} s)")
    else:
        out["quality"] = {"skipped": True}

    if not args.skip_humaneval:
        n = args.n_humaneval
        print(f"[M2.7 quality+humaneval] HumanEval pass@1 (n={n})…")
        out["humaneval"] = run_humaneval_through_client(client, n_examples=n)
        print(f"  → pass@1 = {out['humaneval']['passed']}/{out['humaneval']['n']} "
              f"= {out['humaneval']['accuracy']:.3f} "
              f"({out['humaneval']['wall_s']:.0f} s)")
    else:
        out["humaneval"] = {"skipped": True}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"[M2.7 quality+humaneval] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Gate B — temperature-0 determinism + plaintext-vs-keymat diff.

Per docs/plans/path-2-aloepri-next-steps.md (Gate B, revised after
agent flagged that byte-identical isn't realistic on Vulkan with
flash-attention). Measures longest-common-prefix across replicates
instead of asserting byte-equality; classifies determinism level.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import requests

PLAINTEXT_URL = "http://127.0.0.1:11441/v1/completions"
KEYMAT_URL = "http://127.0.0.1:11446/v1/completions"
MAX_TOKENS = 32
REPLICATES = 3
TIMEOUT_S = 60

PROMPTS = [
    "What is the capital of France?",
    "Write a haiku about autumn.",
    "def fibonacci(n):",
    "Translate to French: Hello, how are you?",
    "Once upon a time in a faraway land,",
]


def call(url: str, prompt: str) -> str:
    r = requests.post(
        url,
        json={
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": MAX_TOKENS,
            "seed": 0,
        },
        timeout=TIMEOUT_S,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["text"]


def lcp_chars(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def min_pairwise_lcp(runs: list[str]) -> int:
    return min(lcp_chars(a, b) for a, b in combinations(runs, 2))


def classify(lcp: int, mean_len: float) -> str:
    if mean_len == 0:
        return "empty"
    frac = lcp / mean_len
    if frac >= 0.99:
        return "fully-deterministic"
    if frac >= 0.50:
        return "stable-prefix"
    return "noisy"


def run_prompt(prompt: str) -> dict:
    plain = [call(PLAINTEXT_URL, prompt) for _ in range(REPLICATES)]
    keymat = [call(KEYMAT_URL, prompt) for _ in range(REPLICATES)]
    plain_lcp = min_pairwise_lcp(plain)
    keymat_lcp = min_pairwise_lcp(keymat)
    cross_lcp = lcp_chars(plain[0], keymat[0])
    plain_mean = sum(len(r) for r in plain) / REPLICATES
    keymat_mean = sum(len(r) for r in keymat) / REPLICATES
    return {
        "prompt": prompt,
        "plain_runs": plain,
        "keymat_runs": keymat,
        "plain_min_lcp_chars": plain_lcp,
        "keymat_min_lcp_chars": keymat_lcp,
        "plain_mean_len": plain_mean,
        "keymat_mean_len": keymat_mean,
        "plain_class": classify(plain_lcp, plain_mean),
        "keymat_class": classify(keymat_lcp, keymat_mean),
        "cross_lcp_chars": cross_lcp,
    }


def main() -> int:
    results = [run_prompt(p) for p in PROMPTS]
    for r in results:
        print(
            f"[{r['prompt'][:38]!r:42}] "
            f"plain LCP={r['plain_min_lcp_chars']}/{int(r['plain_mean_len']):d} ({r['plain_class']}) "
            f"keymat LCP={r['keymat_min_lcp_chars']}/{int(r['keymat_mean_len']):d} ({r['keymat_class']}) "
            f"cross LCP={r['cross_lcp_chars']}"
        )

    out = Path(__file__).resolve().parents[2] / "results" / "path-2-gate-b.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")

    plain_classes = {r["plain_class"] for r in results}
    keymat_classes = {r["keymat_class"] for r in results}
    print(f"\nPlaintext classes: {plain_classes}")
    print(f"Keymat    classes: {keymat_classes}")

    worst = lambda cs: (
        "noisy" if "noisy" in cs
        else "stable-prefix" if "stable-prefix" in cs
        else "fully-deterministic"
    )
    print(f"Plaintext determinism: {worst(plain_classes)}")
    print(f"Keymat    determinism: {worst(keymat_classes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""HTTP completion client for the Gate C mini accuracy harness.

The keymat and plaintext llama-servers both run with `-np 1` so
sequential requests are correct (no parallel-slot speedup available).
Keep this thin — task-specific scoring lives in `tasks/*.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class Endpoint:
    name: str
    url: str  # e.g. "http://127.0.0.1:11441/v1/completions"


def complete(
    endpoint: Endpoint,
    prompt: str,
    max_tokens: int,
    *,
    temperature: float = 0.0,
    seed: int = 0,
    stop: list[str] | None = None,
    timeout: float = 60.0,
) -> str:
    body = {
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    if stop:
        body["stop"] = stop
    r = requests.post(endpoint.url, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["text"]

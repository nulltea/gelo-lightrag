"""PIQA 0-shot (multi-choice A/B format).

Standard canonical scoring is log-likelihood comparison of the two
solutions; this harness uses the simpler A/B-letter-extraction
approach (same shape as MMLU) since the mini benchmark goal is
in-budget signal, not publishable numbers. If parse-rate dips below
~90% on either endpoint, switch to logprobs-based scoring.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from datasets import load_dataset

from ..runner import Endpoint, complete

NUM_PROMPTS = 200
SAMPLE_SEED = 42
MAX_TOKENS = 4
LETTERS = ("A", "B")
_LETTER_RE = re.compile(r"[AB]")


@dataclass
class PiqaExample:
    goal: str
    sol1: str
    sol2: str
    label: int  # 0 → sol1 (A); 1 → sol2 (B)


def load_examples() -> list[PiqaExample]:
    ds = load_dataset("lighteval/piqa", split="validation")
    rng = random.Random(SAMPLE_SEED)
    idxs = rng.sample(range(len(ds)), NUM_PROMPTS)
    out = []
    for i in idxs:
        row = ds[i]
        out.append(
            PiqaExample(
                goal=row["goal"],
                sol1=row["sol1"],
                sol2=row["sol2"],
                label=int(row["label"]),
            )
        )
    return out


def format_prompt(ex: PiqaExample) -> str:
    return (
        f"Question: {ex.goal}\n"
        f"A. {ex.sol1}\n"
        f"B. {ex.sol2}\n"
        f"Answer:"
    )


def extract_answer(text: str) -> str | None:
    m = _LETTER_RE.search(text)
    return m.group(0) if m else None


def score(endpoint: Endpoint, examples: list[PiqaExample]) -> dict:
    correct = 0
    parsed = 0
    details = []
    for ex in examples:
        prompt = format_prompt(ex)
        completion = complete(endpoint, prompt, max_tokens=MAX_TOKENS, stop=["\n"])
        pred = extract_answer(completion)
        gold = LETTERS[ex.label]
        is_correct = pred == gold
        if pred is not None:
            parsed += 1
        if is_correct:
            correct += 1
        details.append(
            {"gold": gold, "pred": pred, "raw": completion, "correct": is_correct}
        )
    return {
        "n": len(examples),
        "correct": correct,
        "parsed": parsed,
        "accuracy": correct / len(examples),
        "parse_rate": parsed / len(examples),
        "details": details,
    }

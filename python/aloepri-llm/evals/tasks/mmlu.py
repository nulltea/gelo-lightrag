"""MMLU 0-shot exact-match (first letter) scoring.

Standard lm-eval-harness format. 200 prompts sampled with a fixed
seed so plaintext and keymat see the same set.
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
LETTERS = ("A", "B", "C", "D")
_LETTER_RE = re.compile(r"[A-D]")


@dataclass
class MmluExample:
    question: str
    subject: str
    choices: list[str]
    answer_idx: int  # 0..3


def load_examples() -> list[MmluExample]:
    ds = load_dataset("cais/mmlu", "all", split="test")
    rng = random.Random(SAMPLE_SEED)
    idxs = rng.sample(range(len(ds)), NUM_PROMPTS)
    out = []
    for i in idxs:
        row = ds[i]
        out.append(
            MmluExample(
                question=row["question"],
                subject=row["subject"].replace("_", " "),
                choices=list(row["choices"]),
                answer_idx=int(row["answer"]),
            )
        )
    return out


def format_prompt(ex: MmluExample) -> str:
    return (
        f"The following are multiple choice questions (with answers) about {ex.subject}.\n\n"
        f"{ex.question}\n"
        f"A. {ex.choices[0]}\n"
        f"B. {ex.choices[1]}\n"
        f"C. {ex.choices[2]}\n"
        f"D. {ex.choices[3]}\n"
        f"Answer:"
    )


def extract_answer(text: str) -> str | None:
    m = _LETTER_RE.search(text)
    return m.group(0) if m else None


def score(endpoint: Endpoint, examples: list[MmluExample]) -> dict:
    correct = 0
    parsed = 0
    by_subject: dict[str, dict[str, int]] = {}
    details = []
    for ex in examples:
        prompt = format_prompt(ex)
        completion = complete(endpoint, prompt, max_tokens=MAX_TOKENS, stop=["\n"])
        pred = extract_answer(completion)
        gold = LETTERS[ex.answer_idx]
        is_correct = pred == gold
        if pred is not None:
            parsed += 1
        if is_correct:
            correct += 1
        bs = by_subject.setdefault(ex.subject, {"n": 0, "correct": 0})
        bs["n"] += 1
        if is_correct:
            bs["correct"] += 1
        details.append(
            {
                "subject": ex.subject,
                "gold": gold,
                "pred": pred,
                "raw": completion,
                "correct": is_correct,
            }
        )
    return {
        "n": len(examples),
        "correct": correct,
        "parsed": parsed,
        "accuracy": correct / len(examples),
        "parse_rate": parsed / len(examples),
        "by_subject": {
            s: {**v, "accuracy": v["correct"] / v["n"]} for s, v in by_subject.items()
        },
        "details": details,
    }

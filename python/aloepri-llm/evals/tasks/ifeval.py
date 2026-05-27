"""IFEval — strict-prompt accuracy on a verifier-supported subset.

Filters the IFEval set down to prompts whose every instruction has a
hand-rolled Python verifier here (no LLM judge). Covers ~8 of the
25 IFEval instruction types — enough for a directional accuracy
delta in the mini benchmark. Not the canonical lm-eval-harness
score; "in-budget signal, not publishable numbers" per Gate C scope.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from datasets import load_dataset

from ..runner import Endpoint, complete

NUM_PROMPTS = 50
SAMPLE_SEED = 42
MAX_TOKENS = 512


SUPPORTED = {
    "punctuation:no_comma",
    "length_constraints:number_words",
    "length_constraints:number_sentences",
    "length_constraints:number_paragraphs",
    "keywords:existence",
    "keywords:forbidden_words",
    "keywords:frequency",
    "change_case:english_lowercase",
    "change_case:english_capital",
    "startend:end_checker",
}


def _relation_pass(value: int, target: int, relation: str) -> bool:
    if relation in ("at least", "at_least"):
        return value >= target
    if relation in ("at most", "at_most"):
        return value <= target
    if relation in ("less than", "less_than"):
        return value < target
    return value == target


def _count_sentences(text: str) -> int:
    return len([s for s in re.split(r"[.!?]+", text) if s.strip()])


def _count_paragraphs(text: str) -> int:
    return len([p for p in re.split(r"\n\s*\n", text) if p.strip()])


def verify(instruction_id: str, kwargs: dict, response: str) -> bool:
    if instruction_id == "punctuation:no_comma":
        return "," not in response
    if instruction_id == "length_constraints:number_words":
        n = len(response.split())
        return _relation_pass(n, kwargs["num_words"], kwargs.get("relation") or "at_least")
    if instruction_id == "length_constraints:number_sentences":
        n = _count_sentences(response)
        return _relation_pass(n, kwargs["num_sentences"], kwargs.get("relation") or "at_least")
    if instruction_id == "length_constraints:number_paragraphs":
        n = _count_paragraphs(response)
        return _relation_pass(n, kwargs["num_paragraphs"], kwargs.get("relation") or "at_least")
    if instruction_id == "keywords:existence":
        return all(kw.lower() in response.lower() for kw in (kwargs.get("keywords") or []))
    if instruction_id == "keywords:forbidden_words":
        return not any(
            kw.lower() in response.lower() for kw in (kwargs.get("forbidden_words") or [])
        )
    if instruction_id == "keywords:frequency":
        kw = kwargs["keyword"].lower()
        n = response.lower().count(kw)
        return _relation_pass(n, kwargs["frequency"], kwargs.get("relation") or "at_least")
    if instruction_id == "change_case:english_lowercase":
        letters = [c for c in response if c.isalpha()]
        return all(c.islower() for c in letters)
    if instruction_id == "change_case:english_capital":
        letters = [c for c in response if c.isalpha()]
        return all(c.isupper() for c in letters)
    if instruction_id == "startend:end_checker":
        end_phrase = (kwargs.get("end_phrase") or "").strip()
        return response.rstrip().rstrip(".?!").endswith(end_phrase.rstrip(".?!"))
    return False


@dataclass
class IFExample:
    prompt: str
    instruction_id_list: list[str]
    kwargs: list[dict]


def load_examples() -> list[IFExample]:
    ds = load_dataset("google/IFEval", split="train")
    eligible = [i for i, ex in enumerate(ds) if all(iid in SUPPORTED for iid in ex["instruction_id_list"])]
    rng = random.Random(SAMPLE_SEED)
    rng.shuffle(eligible)
    chosen = eligible[:NUM_PROMPTS]
    return [
        IFExample(
            prompt=ds[i]["prompt"],
            instruction_id_list=list(ds[i]["instruction_id_list"]),
            kwargs=[dict(k) for k in ds[i]["kwargs"]],
        )
        for i in chosen
    ]


def score(endpoint: Endpoint, examples: list[IFExample]) -> dict:
    strict_correct = 0
    inst_correct = 0
    inst_total = 0
    details = []
    for ex in examples:
        response = complete(endpoint, ex.prompt, max_tokens=MAX_TOKENS, timeout=300)
        per_inst = []
        for iid, kw in zip(ex.instruction_id_list, ex.kwargs):
            ok = verify(iid, kw, response)
            per_inst.append(ok)
            inst_total += 1
            if ok:
                inst_correct += 1
        if all(per_inst):
            strict_correct += 1
        details.append(
            {
                "prompt": ex.prompt[:200],
                "instruction_id_list": ex.instruction_id_list,
                "per_instruction": per_inst,
                "strict": all(per_inst),
                "response": response[:600],
            }
        )
    return {
        "n": len(examples),
        "strict_correct": strict_correct,
        "accuracy": strict_correct / len(examples),  # strict-prompt accuracy
        "instruction_accuracy": inst_correct / max(inst_total, 1),
        "parse_rate": 1.0,
        "details": details,
    }

"""HumanEval pass@1.

50-problem subset. Generation stops at function/class boundaries or
3-newline boundary. Scoring runs each candidate in a subprocess with
a 10s wall-clock timeout against the canonical `check` harness.
"""

from __future__ import annotations

import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

from ..runner import Endpoint, complete

NUM_PROMPTS = 50
SAMPLE_SEED = 42
MAX_TOKENS = 384
EXEC_TIMEOUT_S = 10
STOP_SEQUENCES = ["\nclass ", "\ndef ", "\n\n\n", "\nif __name__"]


@dataclass
class HumanEvalExample:
    task_id: str
    prompt: str
    test: str
    entry_point: str


def load_examples() -> list[HumanEvalExample]:
    ds = load_dataset("openai/openai_humaneval", split="test")
    rng = random.Random(SAMPLE_SEED)
    idxs = rng.sample(range(len(ds)), NUM_PROMPTS)
    return [
        HumanEvalExample(
            task_id=ds[i]["task_id"],
            prompt=ds[i]["prompt"],
            test=ds[i]["test"],
            entry_point=ds[i]["entry_point"],
        )
        for i in idxs
    ]


def run_check(prompt: str, completion: str, test: str, entry_point: str) -> tuple[bool, str]:
    program = (
        prompt + completion + "\n\n" + test + f"\n\ncheck({entry_point})\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = Path(f.name)
    try:
        r = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            timeout=EXEC_TIMEOUT_S,
            text=True,
        )
        if r.returncode == 0:
            return True, ""
        return False, (r.stderr or r.stdout or "")[:200]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, f"exec error: {e}"
    finally:
        path.unlink(missing_ok=True)


def score(endpoint: Endpoint, examples: list[HumanEvalExample]) -> dict:
    passed = 0
    details = []
    for ex in examples:
        completion = complete(
            endpoint,
            ex.prompt,
            max_tokens=MAX_TOKENS,
            stop=STOP_SEQUENCES,
            timeout=120,
        )
        ok, err = run_check(ex.prompt, completion, ex.test, ex.entry_point)
        if ok:
            passed += 1
        details.append(
            {
                "task_id": ex.task_id,
                "passed": ok,
                "completion": completion[:600],
                "error": err if not ok else "",
            }
        )
    return {
        "n": len(examples),
        "passed": passed,
        "accuracy": passed / len(examples),
        "parse_rate": 1.0,
        "details": details,
    }

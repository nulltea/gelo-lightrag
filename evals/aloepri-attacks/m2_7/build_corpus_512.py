"""Build the 512-prompt corpus for the IMA paper-like sweep.

Mixes the hand-curated 64-prompt release-gate corpus with 448
filtered PIQA goal+solution sentences (cached locally). PIQA is
short physical-reasoning prose at the right length distribution
(6-25 tokens) and is BSD-licensed.

Output: `evals/aloepri-attacks/corpora/release-gate-512.txt`.
"""

from __future__ import annotations

import re
from pathlib import Path
from datasets import load_dataset

ROOT = Path("/home/timo/repos/private-rag-path-2/evals/aloepri-attacks/corpora")
SRC = ROOT / "release-gate-64.txt"
DST = ROOT / "release-gate-512.txt"

EXISTING = [l.strip() for l in SRC.read_text().splitlines() if l.strip()]
print(f"existing prompts: {len(EXISTING)}")

TARGET = 512
NEED = TARGET - len(EXISTING)

ds = load_dataset("lighteval/piqa", split="train", trust_remote_code=False)

def normalise(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if not s.endswith((".", "?", "!")):
        s = s + "."
    return s[0].upper() + s[1:]

picked: list[str] = []
seen: set[str] = set(EXISTING)
for ex in ds:
    if len(picked) >= NEED:
        break
    goal = ex["goal"].strip()
    sol = (ex["sol1"] if ex["label"] == 0 else ex["sol2"]).strip()
    sentence = normalise(goal + " " + sol)
    n_words = len(sentence.split())
    if n_words < 6 or n_words > 25:
        continue
    if sentence in seen:
        continue
    seen.add(sentence)
    picked.append(sentence)

assert len(picked) == NEED, f"only collected {len(picked)} of {NEED}"

ALL = EXISTING + picked
DST.write_text("\n".join(ALL) + "\n")
print(f"wrote {len(ALL)} prompts → {DST}")
lengths = [len(p.split()) for p in ALL]
print(f"length: min={min(lengths)} max={max(lengths)} mean={sum(lengths)/len(lengths):.1f}")

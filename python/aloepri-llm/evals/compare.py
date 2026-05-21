"""Gate C mini accuracy benchmark — plaintext vs keymat.

Per docs/plans/path-2-aloepri-next-steps.md (Gate C). Scope: one task
per category, ~30-45 min runtime, "in-budget per-task signal" not
publishable numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .runner import Endpoint
from .tasks import humaneval, ifeval, mmlu, piqa

PLAINTEXT = Endpoint("plaintext", "http://127.0.0.1:11441/v1/completions")
KEYMAT = Endpoint("keymat-h128-fp32", "http://127.0.0.1:11446/v1/completions")

TASKS = {
    "mmlu": mmlu,
    "piqa": piqa,
    "humaneval": humaneval,
    "ifeval": ifeval,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    task = TASKS[args.task]
    print(f"Loading task {args.task}…", file=sys.stderr)
    examples = task.load_examples()
    print(f"  {len(examples)} examples", file=sys.stderr)

    results = {}
    for endpoint in (PLAINTEXT, KEYMAT):
        print(f"Scoring {endpoint.name}…", file=sys.stderr)
        t0 = time.time()
        r = task.score(endpoint, examples)
        r["wall_s"] = time.time() - t0
        print(
            f"  acc={r['accuracy']:.4f}  parse={r['parse_rate']:.4f}  "
            f"wall={r['wall_s']:.1f}s",
            file=sys.stderr,
        )
        results[endpoint.name] = r

    delta = results["keymat-h128-fp32"]["accuracy"] - results["plaintext"]["accuracy"]
    summary = {
        "task": args.task,
        "plaintext_acc": results["plaintext"]["accuracy"],
        "keymat_acc": results["keymat-h128-fp32"]["accuracy"],
        "delta": delta,
        "delta_pct_pp": delta * 100,
        "n": results["plaintext"]["n"],
        "results": results,
    }

    out = args.out or (
        Path(__file__).resolve().parents[3] / "results" / f"path-2-gate-c-{args.task}.json"
    )
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(
        f"\n{args.task}: plain={summary['plaintext_acc']:.4f}  "
        f"keymat={summary['keymat_acc']:.4f}  "
        f"Δ={summary['delta_pct_pp']:+.2f}pp",
        file=sys.stderr,
    )
    print(f"Wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Shared OOM-robustness helpers for the M2.7 harness scripts.

Pattern lifted from the path-1 post-OOM safeguards (see
`docs/prototype/aloepri-attack-harness-findings.md`): pre-flight a
`MemAvailable` check at the start of each phase, document the
expected memory peak per phase so the threshold isn't arbitrary,
and offer a `--skip-mem-check` escape hatch for operators who have
measured headroom.

Per-phase expected peaks on Qwen3 1.7B (plaintext Q8_0 + obfuscated
fp32 GGUF):

| Phase                            | Peak RAM | Notes |
|---|---:|---|
| static attacks (VMA + IA)        | ~22 GB   | both GGUFs dequantised, source-pair matmuls subset by eval/pool |
| token-stream capture             | ~3 GB    | server-side; client just holds the captured JSONL |
| token-stream attacks (TFMA/SDA)  | ~2 GB    | bigram matrices + small inverters |
| hidden-state capture             | ~3 GB    | client holds parsed dump records; tensors flushed to safetensors per prompt batch |
| hidden-state attacks             | ~4 GB    | embedding table f32 (~600 MB) + captured tensors + inverter |

Phase peaks are sized for the 8-prompt fast variant. Doubling to
the 64-prompt release-gate scale adds linearly on the capture side
(more rows in the safetensors output, no per-prompt accumulation
on the attack side).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PHASE_MIN_GB: dict[str, float] = {
    "static_attacks": 25.0,    # ≥ 22 GB peak + safety margin
    "ima_embedrow_attacks": 25.0,  # same two-GGUF working set as static_attacks
    "token_capture": 4.0,
    "token_attacks": 3.0,
    "hidden_capture": 4.0,
    "hidden_attacks": 6.0,
}


def read_mem_available_gb() -> float:
    """Return current MemAvailable in GB, or +inf on parse failure
    (don't block on a probe error — just don't gate).
    """
    try:
        text = Path("/proc/meminfo").read_text()
    except Exception:
        return float("inf")
    m = re.search(r"^MemAvailable:\s+(\d+)\s+kB", text, re.MULTILINE)
    if not m:
        return float("inf")
    return int(m.group(1)) / 1024 / 1024


def check_phase_memory(
    phase: str, override_min_gb: float | None = None, skip: bool = False
) -> None:
    """Pre-flight memory check for a named M2.7 phase.

    Raises RuntimeError if MemAvailable is below the phase threshold
    and `skip` is False. The threshold comes from `PHASE_MIN_GB[phase]`
    unless `override_min_gb` is supplied.
    """
    if skip:
        print(f"[{phase}] mem pre-flight SKIPPED (--skip-mem-check)")
        return
    min_gb = override_min_gb if override_min_gb is not None else PHASE_MIN_GB.get(phase, 8.0)
    avail = read_mem_available_gb()
    msg = f"[{phase}] MemAvailable={avail:.1f} GB (min {min_gb} GB)"
    if avail < min_gb:
        raise RuntimeError(
            f"{msg} — refusing to start. Free memory (close llama-swap "
            f"children, dockers, browser tabs) or pass --skip-mem-check if "
            f"you've measured the headroom."
        )
    print(msg)


def add_min_mem_args(parser, phase: str) -> None:
    """Add `--min-mem-gb` + `--skip-mem-check` flags to an argparse
    parser, defaulting to the phase's recommended threshold.
    """
    parser.add_argument(
        "--min-mem-gb",
        type=float,
        default=PHASE_MIN_GB.get(phase, 8.0),
        help=f"Pre-flight MemAvailable floor in GB (phase '{phase}' default: "
             f"{PHASE_MIN_GB.get(phase, 8.0)})",
    )
    parser.add_argument(
        "--skip-mem-check",
        action="store_true",
        help="Bypass the pre-flight memory check",
    )


def bounded_buffer_check(
    *,
    current_bytes: int,
    limit_bytes: int,
    phase: str,
    what: str,
) -> None:
    """Raise if the in-memory buffer for `what` has grown past
    `limit_bytes`. Used by the capture frontends to bail out before
    a runaway accumulation OOMs the host.
    """
    if current_bytes > limit_bytes:
        gb_current = current_bytes / (1024 ** 3)
        gb_limit = limit_bytes / (1024 ** 3)
        raise RuntimeError(
            f"[{phase}] {what} grew to {gb_current:.2f} GB > limit "
            f"{gb_limit:.2f} GB — flushing to disk and aborting rather "
            f"than risking OOM. Reduce --max-prompts or raise the limit."
        )

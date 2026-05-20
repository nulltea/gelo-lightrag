"""M2.7 orchestrator — chain static-weight + token-stream + (deferred)
hidden-state attacks against the §05 obfuscated artifact.

This script does NOT spawn the obfuscated llama-server (use
`spawn_obfuscated_server.sh`) and does NOT auto-run anything when
imported. The intent is:

    python3 evals/aloepri-attacks/m2_7/run_all_m2_7.py --check

→ verify pre-flight (GGUFs exist, server reachable, memory OK),
  print a summary of what would run, exit 0/1.

    python3 evals/aloepri-attacks/m2_7/run_all_m2_7.py \\
        --plain <PLAIN.gguf> --obfuscated <OBF.gguf> \\
        --key <OBF.key.npz> --endpoint http://127.0.0.1:8061 \\
        --prompts <corpus.txt> --output-dir results/m2_7

→ run static (always) + token-stream (if endpoint healthy);
  skip hidden-state with a clear "deferred" stamp in the output JSON.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _check_path_exists(label: str, p: Path) -> bool:
    ok = p.exists()
    print(f"  {label:24s} {'✓' if ok else '✗'} {p}")
    return ok


def _check_endpoint(url: str, timeout: float = 3.0) -> bool:
    try:
        import urllib.request

        with urllib.request.urlopen(url + "/health", timeout=timeout) as resp:
            ok = resp.status == 200
    except Exception:
        ok = False
    print(f"  {'endpoint health':24s} {'✓' if ok else '✗'} {url}/health")
    return ok


def _check_available_mem_gb() -> float:
    import re

    text = Path("/proc/meminfo").read_text()
    m = re.search(r"^MemAvailable:\s+(\d+)\s+kB", text, re.MULTILINE)
    if not m:
        return 0.0
    return int(m.group(1)) / 1024 / 1024


def main() -> int:
    p = argparse.ArgumentParser(description="M2.7 orchestrator")
    p.add_argument("--check", action="store_true", help="Pre-flight only; do not run")
    p.add_argument("--plain", type=Path)
    p.add_argument("--obfuscated", type=Path)
    p.add_argument("--key", type=Path)
    p.add_argument("--endpoint", default="http://127.0.0.1:8061")
    p.add_argument("--prompts", type=Path,
                   default=Path("evals/aloepri-attacks/corpora/release-gate-64.txt"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("evals/aloepri-attacks/results"))
    p.add_argument("--max-prompts", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--vma-eval-size", type=int, default=256)
    p.add_argument("--vma-pool-size", type=int, default=4096)
    p.add_argument("--ia-eval-size", type=int, default=4096)
    p.add_argument("--ia-pool-size", type=int, default=8192)
    p.add_argument("--min-mem-gb", type=float, default=20.0)
    p.add_argument("--allow-token-stream", action="store_true",
                   help="Opt in to the token-stream step (requires llama-server up)")
    p.add_argument("--skip-ima-embedrow", action="store_true",
                   help="Skip the IMA-EmbedRow-ridge / IMA-EmbedRow-transformer step "
                        "(both attacks load both GGUFs again; ~5 min extra).")
    p.add_argument("--skip-ima-embedrow-transformer", action="store_true",
                   help="Run IMA-EmbedRow-ridge only; skip the slow "
                        "trained-inverter variant.")
    p.add_argument("--allow-quality-humaneval", action="store_true",
                   help="Opt in to the quality + HumanEval gate (requires llama-server up "
                        "at --endpoint with the obfuscated GGUF). Per-sweep-cell defence-vs-"
                        "accuracy gate; ~10-15 min at n-humaneval=50.")
    p.add_argument("--n-humaneval", type=int, default=50,
                   help="Number of HumanEval problems for the quality gate (default 50; "
                        "drop to 20 for fast sweep crank).")
    p.add_argument("--skip-quality", action="store_true",
                   help="Skip the 5-prompt quality probe portion of the gate.")
    args = p.parse_args()

    print("[M2.7 orchestrator] pre-flight checks")
    ok = True
    if args.plain:
        ok &= _check_path_exists("plaintext GGUF", args.plain)
    if args.obfuscated:
        ok &= _check_path_exists("obfuscated GGUF", args.obfuscated)
    if args.key:
        ok &= _check_path_exists("key.npz", args.key)
    ok &= _check_path_exists("prompts file", args.prompts)
    mem_gb = _check_available_mem_gb()
    mem_ok = mem_gb >= args.min_mem_gb
    print(f"  {'available memory':24s} {'✓' if mem_ok else '✗'} {mem_gb:.1f} GB "
          f"(min {args.min_mem_gb} GB)")
    ok &= mem_ok
    if args.allow_token_stream:
        ok &= _check_endpoint(args.endpoint)
    else:
        print("  endpoint health          ⏭  (token-stream not requested; "
              "pass --allow-token-stream to enable)")

    print("[M2.7 orchestrator] hidden-state attacks (NN / IMA / ISA) are "
          "DEFERRED — see HIDDEN_STATE_GAP.md")

    if args.check:
        print(f"[M2.7 orchestrator] check-only; pre-flight {'PASSED' if ok else 'FAILED'}")
        return 0 if ok else 1

    if not ok:
        print("[M2.7 orchestrator] pre-flight failed — refusing to run; "
              "fix the ✗ rows above or pass --check to inspect only")
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: static-weight attacks ──────────────────────────────
    static_out = args.output_dir / "m2_7-static.json"
    print(f"\n[M2.7 orchestrator] step 1/2: static-weight attacks → {static_out}")
    rc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "run_static_attacks.py"),
            "--plain", str(args.plain),
            "--obfuscated", str(args.obfuscated),
            "--output", str(static_out),
            "--vma-eval-size", str(args.vma_eval_size),
            "--vma-pool-size", str(args.vma_pool_size),
            "--ia-eval-size", str(args.ia_eval_size),
            "--ia-pool-size", str(args.ia_pool_size),
        ],
        check=False,
    ).returncode
    if rc != 0:
        print(f"[M2.7 orchestrator] static-weight step failed (rc={rc})")
        return rc

    # ── Step 1b: IMA-EmbedRow static-weight attacks ────────────────
    # Two prompt-inversion attacks on the obfuscated embedding-row
    # surface: ridge + trained-inverter on (W_e_plain, W_e_obf, τ).
    # See docs/handoffs/2026-05-19-aloepri-attack-surface-followups.md
    # thread 1 for why they're in-scope (recovering τ decodes every
    # wire-side prompt).
    if not args.skip_ima_embedrow:
        if args.key is None:
            print("[M2.7 orchestrator] IMA-EmbedRow skipped — no --key supplied "
                  "(τ must come from the obfuscator's .key.npz)")
        else:
            embedrow_out = args.output_dir / "m2_7-ima-embedrow.json"
            print(f"\n[M2.7 orchestrator] step 1b: IMA-EmbedRow attacks → {embedrow_out}")
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "run_ima_embedrow_attacks.py"),
                "--plain", str(args.plain),
                "--obfuscated", str(args.obfuscated),
                "--key", str(args.key),
                "--output", str(embedrow_out),
            ]
            if args.skip_ima_embedrow_transformer:
                cmd.append("--skip-transformer")
            rc = subprocess.run(cmd, check=False).returncode
            if rc != 0:
                print(f"[M2.7 orchestrator] IMA-EmbedRow step failed (rc={rc})")
                return rc

    # ── Step 2: token-stream attacks (opt-in) ──────────────────────
    if args.allow_token_stream:
        captured = args.output_dir / "m2_7-token-streams.jsonl"
        print(f"\n[M2.7 orchestrator] step 2/2: capture token streams → {captured}")
        rc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "capture_token_streams.py"),
                "--endpoint", args.endpoint,
                "--key-path", str(args.key),
                "--prompts-file", str(args.prompts),
                "--max-prompts", str(args.max_prompts),
                "--max-new-tokens", str(args.max_new_tokens),
                "--output", str(captured),
            ],
            check=False,
        ).returncode
        if rc != 0:
            print(f"[M2.7 orchestrator] capture step failed (rc={rc})")
            return rc

        token_out = args.output_dir / "m2_7-token.json"
        print(f"\n[M2.7 orchestrator] step 2/2: run TFMA + SDA → {token_out}")
        rc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "run_token_attacks.py"),
                "--captured", str(captured),
                "--output", str(token_out),
            ],
            check=False,
        ).returncode
        if rc != 0:
            print(f"[M2.7 orchestrator] token-attack step failed (rc={rc})")
            return rc
    else:
        print("\n[M2.7 orchestrator] step 2/2: token-stream SKIPPED (no --allow-token-stream)")

    # ── Step 3: quality probe + HumanEval pass@1 ───────────────────
    # Per-condition defence-vs-accuracy gate. Routes generation through
    # AloePriClient so τ is applied to prompts and τ⁻¹ to responses
    # (testing the actual paper protocol, not the server's gibberish).
    if args.allow_quality_humaneval:
        if args.key is None:
            print("\n[M2.7 orchestrator] step 3: SKIPPED — no --key supplied")
        else:
            qh_out = args.output_dir / "m2_7-quality-humaneval.json"
            print(f"\n[M2.7 orchestrator] step 3: quality + HumanEval → {qh_out}")
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "run_quality_humaneval.py"),
                "--endpoint", args.endpoint,
                "--key", str(args.key),
                "--output", str(qh_out),
                "--n-humaneval", str(args.n_humaneval),
            ]
            if args.skip_quality:
                cmd.append("--skip-quality")
            rc = subprocess.run(cmd, check=False).returncode
            if rc != 0:
                print(f"[M2.7 orchestrator] quality+HumanEval step failed (rc={rc})")
                return rc
    else:
        print("\n[M2.7 orchestrator] step 3: quality+HumanEval SKIPPED "
              "(no --allow-quality-humaneval)")

    print("\n[M2.7 orchestrator] done. NOT killing any container (manual: "
          "`docker stop aloepri-m2_7-server` if you started it).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

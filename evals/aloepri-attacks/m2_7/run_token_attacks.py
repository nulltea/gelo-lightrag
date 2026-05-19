"""Run TFMA + SDA against captured obfuscated token streams.

Consumes the JSONL produced by `capture_token_streams.py`:

  {
    "prompt_idx": 0,
    "plain_prompt_ids": [...],
    "obf_prompt_ids": [...],
    "obf_response_ids": [...],
    ...
  }

The attacker observes only `obf_response_ids` (paper §F.1 phrasing
"token-frequency attacks on the response stream"). TFMA matches
frequencies against a public-corpus prior to recover Π; SDA trains
a small substitution-cipher decipherer on (plain, cipher) pairs.

Calls AloePri's primitives unchanged from
`vendor/aloepri-py/src/security_qwen/tfma.py` and `sda.py` — we
just feed our captured stream where the reference code expected
its own simulated `_obfuscate_sequences` output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


# Re-use the harness's deferred-import helper to bypass the package
# __init__.py (which pulls transformers).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attack_drivers.common import (  # type: ignore  # noqa: E402
    AttackResult,
    classify_risk_level,
    load_aloepri_module,
)


_tfma = load_aloepri_module("src/security_qwen/tfma.py")
_sda = load_aloepri_module("src/security_qwen/sda.py")

_flatten_counts = _tfma._flatten_counts
_evaluate_rank_lists = _tfma._evaluate_rank_lists
_candidate_rank_lists = _tfma._candidate_rank_lists

_build_bigram_matrix = _sda._build_bigram_matrix
_sorted_bigram_signature = _sda._sorted_bigram_signature
_build_candidate_rank_lists = _sda._build_candidate_rank_lists
_token_bleu4 = _sda._token_bleu4


def load_captured(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def _split_obs_train_test(
    records: list[dict[str, Any]], train_frac: float = 0.75, seed: int = 0
) -> tuple[list[list[int]], list[list[int]], list[list[int]], list[list[int]]]:
    """Prompt-disjoint split: 75 % for the attacker's training prior,
    25 % held out for evaluation. Returns:
      (train_plain_seqs, train_obf_seqs, test_plain_seqs, test_obf_seqs).
    """
    rng = np.random.default_rng(seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    n_train = max(int(len(order) * train_frac), 1)
    train_idx = set(order[:n_train])
    train_plain = []
    train_obf = []
    test_plain = []
    test_obf = []
    for i, r in enumerate(records):
        plain = r.get("plain_prompt_ids") or []
        obf = r.get("obf_response_ids") or r.get("obf_prompt_ids") or []
        if not obf:
            continue
        if i in train_idx:
            train_plain.append(plain)
            train_obf.append(obf)
        else:
            test_plain.append(plain)
            test_obf.append(obf)
    return train_plain, train_obf, test_plain, test_obf


def run_tfma(
    records: list[dict[str, Any]],
    *,
    key_path: Path,
    candidate_pool_size: int = 256,
    seed: int = 0,
) -> AttackResult:
    """Frequency matching, paper §F.1.

    Calls AloePri's `_candidate_rank_lists` + `_evaluate_rank_lists`
    unchanged. Setup:

      * `reference_sequences` = the plain prompt-id streams (attacker's
        public corpus prior).
      * `private_obfuscated_sequences` = the captured obf_response_ids
        (the τ-permuted wire-side stream).
      * Ground truth Π is loaded from the .key.npz so the evaluator
        can score each obs_token's predicted plain candidates.
    """
    import torch
    from collections import Counter

    # Reference = plain prompt streams; observed = obfuscated responses.
    reference_sequences = [r["plain_prompt_ids"] for r in records if r.get("plain_prompt_ids")]
    observed_sequences = [
        r["obf_response_ids"] for r in records
        if r.get("obf_response_ids")
    ]
    if not observed_sequences:
        return AttackResult(
            attack="tfma", condition="obfuscated_§05",
            model_id="keymat-h128-pi-noise-alg2-fp32.gguf",
            n_prompts=len(records), n_train=0, n_test=0,
            ttrsr_top1=None, ttrsr_top10=None, risk_level="unknown",
            extra={"note": "no observed sequences in capture"},
        )

    reference_counter = _flatten_counts(reference_sequences)
    observed_counter = _flatten_counts(observed_sequences)

    # Candidate pools: top-N by frequency on each side.
    candidate_plain_ids = [tok for tok, _ in reference_counter.most_common(candidate_pool_size)]
    observed_token_ids = [tok for tok, _ in observed_counter.most_common(candidate_pool_size)]

    # Ground-truth Π for scoring — load from the obfuscated GGUF's key file.
    key = np.load(key_path, allow_pickle=False)
    tau = torch.as_tensor(key["tau"], dtype=torch.long)
    inv_tau = torch.argsort(tau)
    sensitive_plain_ids: set[int] = set()  # no sensitive-id filter for now

    rank_lists = _candidate_rank_lists(
        reference_counter=reference_counter,
        observed_counter=observed_counter,
        candidate_plain_ids=candidate_plain_ids,
        observed_token_ids=observed_token_ids,
    )
    metrics = _evaluate_rank_lists(
        candidate_lists=rank_lists,
        inv_perm_vocab=inv_tau,
        sensitive_plain_ids=sensitive_plain_ids,
        topk_values=(1, 10, 100),
    )

    top1 = float(metrics.get("token_top1_recovery_rate", 0.0))
    top10 = float(metrics.get("token_top10_recovery_rate", 0.0))
    return AttackResult(
        attack="tfma", condition="obfuscated_§05",
        model_id="keymat-h128-pi-noise-alg2-fp32.gguf",
        n_prompts=len(records), n_train=len(reference_sequences),
        n_test=len(observed_sequences),
        ttrsr_top1=top1, ttrsr_top10=top10,
        risk_level=classify_risk_level(top1),
        extra={
            "candidate_pool_size": candidate_pool_size,
            "n_observed_unique_tokens": len(observed_token_ids),
            "matching_strategy": "paper_frequency_rank",
            **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        },
    )


def run_sda(
    records: list[dict[str, Any]],
    *,
    candidate_pool_size: int = 256,
    seed: int = 0,
) -> AttackResult:
    """Bigram-based substitution decipherment + BLEU-4 (paper §F.1).

    Calls AloePri's `_build_candidate_rank_lists` (which internally
    builds bigram matrices and signatures) unchanged.
    """
    reference_sequences = [r["plain_prompt_ids"] for r in records if r.get("plain_prompt_ids")]
    observed_sequences = [
        r["obf_response_ids"] for r in records
        if r.get("obf_response_ids")
    ]
    if not observed_sequences:
        return AttackResult(
            attack="sda", condition="obfuscated_§05",
            model_id="keymat-h128-pi-noise-alg2-fp32.gguf",
            n_prompts=len(records), n_train=0, n_test=0,
            ttrsr_top1=None, ttrsr_top10=None, risk_level="unknown",
            extra={"note": "no observed sequences in capture"},
        )

    candidate_plain_ids = list({t for seq in reference_sequences for t in seq})[:candidate_pool_size]
    observed_token_ids = list({t for seq in observed_sequences for t in seq})[:candidate_pool_size]

    rank_lists = _build_candidate_rank_lists(
        reference_sequences=reference_sequences,
        private_obfuscated_sequences=observed_sequences,
        candidate_plain_ids=candidate_plain_ids,
        observed_token_ids=observed_token_ids,
    )

    # Recovered text by top-1 substitution.
    obf_to_plain = {oid: rank_lists[oid][0] for oid in observed_token_ids if oid in rank_lists}
    recovered = [
        [obf_to_plain.get(t, t) for t in seq] for seq in observed_sequences
    ]
    bleu4 = _token_bleu4(reference_sequences, recovered)

    return AttackResult(
        attack="sda", condition="obfuscated_§05",
        model_id="keymat-h128-pi-noise-alg2-fp32.gguf",
        n_prompts=len(records),
        n_train=len(reference_sequences), n_test=len(observed_sequences),
        ttrsr_top1=None, ttrsr_top10=None,
        risk_level="low" if bleu4 < 5.0 else "medium",
        extra={
            "bleu4": float(bleu4),
            "candidate_pool_size": candidate_pool_size,
            "matching_strategy": "paper_bigram_signature_substitution",
        },
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Run TFMA + SDA against captured §05 token streams")
    p.add_argument("--captured", type=Path, required=True,
                   help="JSONL from capture_token_streams.py")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--candidate-pool-size", type=int, default=256)
    p.add_argument("--key-path", type=Path, required=True,
                   help="Path to .key.npz (gives the inv-τ for TFMA scoring)")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from m2_7_common import add_min_mem_args, check_phase_memory  # type: ignore
    add_min_mem_args(p, phase="token_attacks")
    args = p.parse_args()

    check_phase_memory("token_attacks", args.min_mem_gb, args.skip_mem_check)
    records = load_captured(args.captured)
    print(f"[M2.7 token-attacks] {len(records)} prompts captured")

    print("[M2.7 token-attacks] running TFMA…")
    tfma = run_tfma(records, key_path=args.key_path, candidate_pool_size=args.candidate_pool_size)
    print(f"  tfma top1={tfma.ttrsr_top1} top10={tfma.ttrsr_top10} risk={tfma.risk_level}")

    print("[M2.7 token-attacks] running SDA…")
    sda = run_sda(records, candidate_pool_size=args.candidate_pool_size)
    print(f"  sda bleu4={sda.extra.get('bleu4')} risk={sda.risk_level}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "format": "aloepri_m2_7_token_v1",
        "captured": str(args.captured),
        "attacks": {"tfma": tfma.to_dict(), "sda": sda.to_dict()},
    }
    args.output.write_text(json.dumps(results, indent=2))
    print(f"[M2.7 token-attacks] wrote → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

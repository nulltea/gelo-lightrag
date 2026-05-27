"""Paper-faithful multi-key ISA driver for the AttnScore surface — STUB.

Companion to `run_isa_multikey.py` (HiddenState) and
`run_ima_embedrow_attacks_multikey.py` (EmbedRow). Both of those
drivers implement the §3.2 Kerckhoffs multi-key training trick:
the attacker doesn't know the deployment's K_d, so they synthesise
K=64 of their own K_a^k via Algorithm 1 and use the **covariance
claim** to manufacture per-key training tensors:

    HiddenState  : X_a^k = X_plain @ K_a^k       (residual axis)
    EmbedRow     : x_a^k = (W_e[id] + ε) @ K_a^k (residual axis)

The ridge inverter is then forced to be key-invariant across the K
synthetic copies, and at test time it's applied to the deployment's
real obfuscated captures.

**Why this driver is a stub.** The covariance claim is specifically a
statement about the *residual stream* — for any layer L, the
obfuscated residual is K_e · State_plain[L] up to higher-order
Algorithm-2 perturbations on the post-attention residual (see
`run_isa_multikey.py` module docstring, paragraph "The covariance
approximation …"). Attention scores are NOT in residual space:

      score = (X̃ · W̃_q) · (X̃ · W̃_k)^T / √d_h    per head

Under Algorithm 1 (keymat-only), W̃_q has its input axis carrying
Q̂ = K_a^{-1}-on-residual specifically so X̃ · W̃_q = X · W_q (paper
covariance). That means in the no-Algorithm-2 path the attention
scores are *invariant* under K_a — there is no covariant relation
`score_a^k = f(K_a^k, score_plain)` that the attacker can exploit
for multi-key training. (See
`python/aloepri-llm/obfuscate_qwen3_gguf.py:692-727` for the
per-tensor transforms — none of the Q/K/V transforms involve the
residual keymat K_a once the X̃ · W̃ product is collapsed.)

Under Algorithm 2, the deployment additionally applies:

  * Intra-head transforms R̂_qk on the head_dim axis (per-layer keys,
    sampled independently of K_a; see `lib/alg2.py:generate_r_qk`).
  * Head-axis permutations τ_kv, τ_group on the n_heads axis
    (per-layer keys; `lib/alg2.py:generate_head_perm`).

Both of these perturbations are functions of the per-layer
Algorithm-2 keyset, not of K_a. The attacker can sample their own
{τ_kv^k, τ_group^k, R̂_qk^k} pool by Algorithm 2, but that is a
*different* multi-key construction than the covariant-synthesis
pattern of `run_isa_multikey.py` / `run_ima_embedrow_multikey.py`.
It would need its own driver (provisional name:
`run_isa_attn_score_alg2_multikey.py`) and is out of scope for the
path-2 attack ledger as currently scoped — Algorithm 2's head-perm
+ intra-head transforms are independent draws per layer, and the
attacker has no synthesis target that mirrors deployment-side
ground truth without also possessing the layer's Algorithm-2 keys.

Per the task brief: "If neither interpretation makes sense, emit a
clear stub driver that returns `not_applicable` … This is a
legitimate outcome — paper-faithful multi-key has a specific
covariant-synthesis structure that may not transfer to attention
scores." That is the outcome here.

**What this driver does provide.** Two useful artefacts:

  1. CLI parity with `run_isa_multikey.py`, so the path-2 attack
     runner can invoke it identically and the per-condition result
     JSON has a stable schema with `attack="isa_attn_score_multikey"`.
  2. An optional `--identity-tau` calibration probe that runs the
     single-key labelled-ridge ISA on plain attn-score captures (no
     multi-key, no synthesis) — the analog of the `--identity-tau`
     branch in `run_isa_multikey.py`. This is the score-surface
     ceiling on plain captures and is useful diagnostically when
     the rest of the ledger fills in.

For all other invocations, the driver writes a `not_applicable` row
with the structural rationale in `extra`, and exits 0.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attack_drivers import run_ima, run_isa  # type: ignore  # noqa: E402
from attack_drivers.common import AttackResult, classify_risk_level  # type: ignore  # noqa: E402
from snapshots_loader import SnapshotSet  # type: ignore  # noqa: E402


# ───── Structural rationale (also used as the AttackResult.extra note) ─────

_STRUCTURAL_RATIONALE = (
    "Paper-faithful multi-key requires a covariance relation "
    "`surface_a^k = f(K_a^k, surface_plain)` so the attacker can "
    "synthesise per-key training inputs from public plain captures. "
    "For the AttnScore surface, no such relation exists. Under "
    "Algorithm 1 (keymat-only) the residual-axis K_a transforms "
    "W_q/W_k input axes such that X̃·W̃_q = X·W_q (paper covariance "
    "claim), so attention scores Q·K^T/√d_h are invariant under K_a "
    "— K_a does not act on the score axis. Under Algorithm 2, the "
    "score tensor is perturbed by intra-head R̂_qk (head_dim) and "
    "head permutations τ_kv/τ_group, but those keys are independent "
    "of K_a and would require a separate Algorithm-2 multi-key "
    "driver (out of scope for the K_a covariant-synthesis ledger). "
    "See module docstring of run_isa_attn_score_multikey.py for the "
    "full derivation."
)


# ───── Optional plain-on-plain calibration probe (single-key, identity-τ) ──


def _flatten_attn_score_view(attn_set: SnapshotSet, layer: int, kind: str):
    """Mirror m2_7.run_hidden_state_attacks._isa_attn_score: build a
    shim SnapshotSet whose per_prompt_layer_kind_tensors yields
    (n_q, n_heads * n_kv) flattened views per prompt.
    """
    flat_pairs: list[tuple[int, torch.Tensor]] = []
    for s in attn_set.select(kind=kind, layer=layer):
        op = attn_set.get_operand(s, strip_shield=False)
        if op.ndim == 3:
            n_heads, n_q, n_kv = op.shape
            op = op.permute(1, 0, 2).reshape(n_q, n_heads * n_kv)
        elif op.ndim != 2:
            continue
        flat_pairs.append((s.prompt_idx, op))
    shim = copy.copy(attn_set)
    shim.per_prompt_layer_kind_tensors = lambda **kw: flat_pairs  # type: ignore[method-assign]
    return shim, flat_pairs


def _run_identity_tau_calibration(
    *,
    plain_captures: Path,
    plain_model_id: str,
    layer: int,
    kind: str,
    seed: int,
    split_mode: str,
) -> AttackResult:
    """Single-key labelled-ridge ISA on plain attn-score captures.

    Analog of the identity_tau branch in run_isa_multikey.py: train
    + test on plain captures, no K_a synthesis. Establishes the
    score-surface inversion ceiling for the public model.
    """
    print(f"[ISA-attn-score-multikey] identity-τ calibration on plain captures")
    plain_snap = SnapshotSet.open("attn", root=plain_captures)
    print(f"  {plain_snap.n_prompts()} plain prompt(s), layers={plain_snap.captured_layers}, "
          f"kinds={plain_snap.captured_kinds}")
    embed_table = run_ima.load_qwen3_embedding_table(plain_model_id).to(torch.float32)
    print(f"  W_e shape = {tuple(embed_table.shape)}")

    shim, flat_pairs = _flatten_attn_score_view(plain_snap, layer, kind)
    if not flat_pairs:
        return AttackResult(
            attack="isa_attn_score_multikey",
            condition="plain_identity_tau",
            model_id=plain_model_id,
            n_prompts=plain_snap.n_prompts(),
            n_train=0,
            n_test=0,
            ttrsr_top1=None,
            ttrsr_top10=None,
            risk_level="unknown",
            extra={
                "phase": "identity_tau_no_snapshots",
                "note": f"no plain attn-score snapshots at layer={layer} kind={kind}",
                "expected_kinds": ["kq", "attn_score"],
                "available_kinds": sorted(set(s.kind for s in plain_snap.snapshots)),
            },
        )

    inner = run_isa.run(
        shim,
        embed_table=embed_table,
        layer=layer,
        kind=kind,
        seed=seed,
        split_mode=split_mode,
        strip_shield=False,
    )
    # Rebrand the inner driver's AttackResult so the result JSON
    # tags this row as the multikey driver's identity-τ branch.
    extra = dict(inner.extra or {})
    extra.update({
        "threat_model_regime": "single_key_identity_tau_calibration",
        "phase": "identity_tau",
        "delegated_inner_attack": inner.attack,
        "delegated_inner_extra": inner.extra,
        "structural_rationale": _STRUCTURAL_RATIONALE,
        "layer": int(layer),
        "kind": str(kind),
        "split_mode": str(split_mode),
    })
    return AttackResult(
        attack="isa_attn_score_multikey",
        condition="plain_identity_tau",
        model_id=plain_model_id,
        n_prompts=inner.n_prompts,
        n_train=inner.n_train,
        n_test=inner.n_test,
        ttrsr_top1=inner.ttrsr_top1,
        ttrsr_top10=inner.ttrsr_top10,
        risk_level=classify_risk_level(inner.ttrsr_top1),
        extra=extra,
    )


# ───── not_applicable stub for the multi-key invocation ────────────────────


def _emit_not_applicable(
    *,
    plain_captures: Path,
    obf_captures: Path | None,
    plain_model_id: str,
    layer: int,
    kind: str,
    attacker_expansion: int,
    attacker_lam: float,
    attacker_num_keys: int,
    attacker_seed: int,
    split_seed: int | None,
    split_mode: str,
    keymat_impl: str,
    device: str,
    runtime_seconds: float,
) -> AttackResult:
    """Construct the `not_applicable` AttackResult with full provenance."""
    try:
        plain_snap = SnapshotSet.open("attn", root=plain_captures)
        n_prompts = plain_snap.n_prompts()
        available_kinds = sorted(set(s.kind for s in plain_snap.snapshots))
    except Exception as exc:  # pragma: no cover — best-effort metadata
        n_prompts = 0
        available_kinds = []
        print(f"  warn: could not open plain attn captures ({exc!r}); "
              "emitting not_applicable with empty provenance")

    return AttackResult(
        attack="isa_attn_score_multikey",
        condition="obfuscated",
        model_id=plain_model_id,
        n_prompts=n_prompts,
        n_train=0,
        n_test=0,
        ttrsr_top1=None,
        ttrsr_top10=None,
        risk_level="not_applicable",
        extra={
            "phase": "structural_not_applicable",
            "threat_model_regime": "multikey_covariant_synthesis_paperfaithful",
            "structural_rationale": _STRUCTURAL_RATIONALE,
            "covariant_synthesis_target_surfaces": [
                "hidden_state (run_isa_multikey.py)",
                "embed_row (run_ima_embedrow_attacks_multikey.py)",
            ],
            "out_of_scope_alternative": (
                "Algorithm-2 multi-key on {τ_kv, τ_group, R̂_qk} would be "
                "a separate driver — not part of the K_a covariant-synthesis "
                "ledger this file companions."
            ),
            "layer": int(layer),
            "kind": str(kind),
            "split_mode": str(split_mode),
            "attacker_expansion": int(attacker_expansion),
            "attacker_lam": float(attacker_lam),
            "attacker_num_keys": int(attacker_num_keys),
            "attacker_seed": int(attacker_seed),
            "split_seed": (int(split_seed) if split_seed is not None
                           else int(attacker_seed) + 17),
            "keymat_impl": str(keymat_impl),
            "device": str(device),
            "plain_captures": str(plain_captures),
            "obf_captures": str(obf_captures) if obf_captures else None,
            "available_kinds": available_kinds,
            "expected_kinds": ["kq", "attn_score"],
            "runtime_seconds": round(runtime_seconds, 2),
        },
    )


# ───── CLI (mirrors run_isa_multikey.py exactly) ───────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Paper-faithful multi-key ISA against the AttnScore surface — "
            "structural STUB (see module docstring). Returns not_applicable "
            "by default; --identity-tau runs a single-key plain-on-plain "
            "calibration probe."
        )
    )
    p.add_argument("--plain-captures", type=Path, required=True,
                   help="Directory containing attn.{safetensors,meta.json} from plain-model capture.")
    p.add_argument("--obf-captures", type=Path, default=None,
                   help="Directory containing attn.{safetensors,meta.json} from obfuscated-model capture. "
                        "Accepted for CLI parity with run_isa_multikey.py; unused by the stub.")
    p.add_argument("--plain-model-id", type=str, default="Qwen/Qwen3-4B",
                   help="HF model id whose W_e is loaded as the candidate pool and inversion target.")
    p.add_argument("--layer", type=int, default=17,
                   help="Attn-score capture layer. Matches run_isa_multikey.py default (≈48 percent depth on Q3-4B).")
    p.add_argument("--kind", type=str, default="kq",
                   help="Snapshot kind for attn-score captures. Default 'kq' matches the patched "
                        "llama-server --tensor-filter '^kq-<L>$' emit kind.")
    p.add_argument("--identity-tau", action="store_true",
                   help="Calibration probe: single-key labelled-ridge ISA on plain attn-score captures, "
                        "no synthesis, no K_a. Score-surface ceiling on the public model.")
    p.add_argument("--attacker-expansion", type=int, default=128)
    p.add_argument("--attacker-lambda", type=float, default=0.3)
    p.add_argument("--attacker-num-keys", type=int, default=64)
    p.add_argument("--keymat-impl", type=str, default="vendor_cpu",
                   choices=("vendor_cpu", "gpu_native"),
                   help="Accepted for CLI parity with run_isa_multikey.py; unused by the stub (no K_a synthesis).")
    p.add_argument("--split-mode", type=str, default="row", choices=("row", "vocab"),
                   help="Train/test split mode passed through to the identity-τ calibration probe. "
                        "Unused by the not_applicable stub.")
    p.add_argument("--device", type=str, default="auto", choices=("auto", "gpu", "cpu", "cuda"),
                   help="Accepted for CLI parity with run_isa_multikey.py.")
    p.add_argument("--paper-checkpoint-dir", type=Path, default=None,
                   help="(accepted for compat with shared wrapper; ISA has no checkpoint)")
    p.add_argument("--attacker-seed", type=int, default=20260521)
    p.add_argument("--split-seed", type=int, default=None,
                   help="Seed for the train/val/test split RNG. Defaults to attacker_seed + 17 "
                        "(matches run_isa_multikey.py legacy behaviour).")
    p.add_argument("--ridge-alpha", type=float, action="append", default=None,
                   help="Accepted for CLI parity; the identity-τ calibration delegates to attack_drivers.run_isa "
                        "which uses its own multi-α grid.")
    p.add_argument("--topk", type=int, default=10,
                   help="Accepted for CLI parity with run_isa_multikey.py.")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    t0 = time.perf_counter()

    if args.identity_tau:
        # Calibration probe: single-key labelled-ridge ISA on plain
        # attn-score captures. Multi-key doesn't enter — this branch
        # exists for CLI parity with run_isa_multikey.py --identity-tau,
        # which is the score-surface analog of the HiddenState ceiling
        # check (top1 on plain, no defence).
        effective_seed = int(args.split_seed) if args.split_seed is not None else int(args.attacker_seed) + 17
        result = _run_identity_tau_calibration(
            plain_captures=args.plain_captures,
            plain_model_id=args.plain_model_id,
            layer=int(args.layer),
            kind=str(args.kind),
            seed=effective_seed,
            split_mode=str(args.split_mode),
        )
    else:
        # Multi-key invocation: emit the structural not_applicable row.
        print("[ISA-attn-score-multikey] structural not_applicable — "
              "no covariant K_a→score synthesis exists for the AttnScore surface")
        print("  (see module docstring for the derivation; pass --identity-tau "
              "for the single-key plain-on-plain calibration probe)")
        result = _emit_not_applicable(
            plain_captures=args.plain_captures,
            obf_captures=args.obf_captures,
            plain_model_id=args.plain_model_id,
            layer=int(args.layer),
            kind=str(args.kind),
            attacker_expansion=int(args.attacker_expansion),
            attacker_lam=float(args.attacker_lambda),
            attacker_num_keys=int(args.attacker_num_keys),
            attacker_seed=int(args.attacker_seed),
            split_seed=args.split_seed,
            split_mode=str(args.split_mode),
            keymat_impl=str(args.keymat_impl),
            device=str(args.device),
            runtime_seconds=time.perf_counter() - t0,
        )

    print(f"[ISA-attn-score-multikey] top1={result.ttrsr_top1} top10={result.ttrsr_top10} "
          f"risk={result.risk_level}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "format": "aloepri_m2_7_isa_attn_score_multikey_v1",
        "plain_captures": str(args.plain_captures),
        "obf_captures": str(args.obf_captures) if args.obf_captures else None,
        "plain_model_id": args.plain_model_id,
        "attack": result.to_dict(),
    }, indent=2))
    print(f"[ISA-attn-score-multikey] wrote → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""M2.7 Phase A.3 — run the four hidden-state attacks against
captured §05 obfuscated observables.

Consumes the safetensors output of `capture_hidden_states.py`:
  * `<dir>/hidden.safetensors`  + `hidden.meta.json`  (`attn_norm-*` tensors)
  * `<dir>/attn.safetensors`    + `attn.meta.json`    (`kq-*` tensors)

Routes:

* `NN`             → snapshots_loader.SnapshotSet on hidden.safetensors, attack_drivers.run_nn (layer 0)
* `IMA basic`      → same, attack_drivers.run_ima (layer 0)
* `IMA paper-like` → same, attack_drivers.run_ima_paper_like (layer 0; needs ≥256 prompts)
* `ISA HiddenState`→ same, attack_drivers.run_isa (layer 23)
* `ISA AttnScore`  → NEW adapter — loads attn.safetensors, flattens
                     `(n_heads, n_q, n_kv)` → `(n_q, n_heads*n_kv)` and
                     feeds attack_drivers.run_isa under the same
                     ridge primitive.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Make the existing attack drivers reachable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attack_drivers import run_ima, run_ima_paper_like, run_isa, run_nn  # type: ignore  # noqa: E402
from attack_drivers.common import AttackResult, classify_risk_level  # type: ignore  # noqa: E402
from snapshots_loader import SnapshotSet  # type: ignore  # noqa: E402


def _load_embed_table(model_id: str) -> torch.Tensor:
    """Re-use run_ima's loader (reads from HF cache safetensors)."""
    return run_ima.load_qwen3_embedding_table(model_id)


def _isa_attn_score(
    attn_set: SnapshotSet,
    embed_table: torch.Tensor,
    layer: int,
    seed: int,
    split_mode: str = "row",
) -> AttackResult:
    """ISA against the AttnScore observable.

    The attn safetensors stores `kq-{layer}` tensors per prompt. Each
    tensor has shape `(n_heads, n_q, n_kv)` — we flatten across the
    head + kv axes so each row corresponds to one query position and
    the feature dim is `n_heads * n_kv`. Then we apply the same
    multi-α ridge inversion as `attack_drivers.run_isa.run`.

    This is intentionally a thin adapter so the privacy claim stays
    aligned with the published AloePri ISA semantics — only the
    feature dim differs.
    """
    # We pretend the AttnScore tensor is the observation. The
    # existing run_isa.run() expects a SnapshotSet with the
    # observation in a single (n, d) layout — we build a fresh
    # SnapshotSet view that flattens (n_heads, n_q, n_kv) → (n_q, *)
    # and uses kind == "kq".
    #
    # The token-id alignment is straightforward: row r of the
    # flattened tensor corresponds to position r in the prompt,
    # same as the HiddenState case.
    #
    # The simplest implementation: monkey-patch the per-prompt tensor
    # accessor on a shallow copy of `attn_set` to return flattened
    # views, then call run_isa.run().

    # Build a per-prompt-flattened view.
    flat_pairs: list[tuple[int, torch.Tensor]] = []
    for s in attn_set.select(kind="kq", layer=layer):
        op = attn_set.get_operand(s, strip_shield=False)
        if op.ndim == 3:
            n_heads, n_q, n_kv = op.shape
            op = op.permute(1, 0, 2).reshape(n_q, n_heads * n_kv)
        elif op.ndim == 2:
            pass  # already flat
        else:
            continue
        flat_pairs.append((s.prompt_idx, op))

    # Replace per_prompt_layer_kind_tensors on a shallow copy.
    import copy
    shim = copy.copy(attn_set)
    shim.per_prompt_layer_kind_tensors = lambda **kw: flat_pairs  # type: ignore[method-assign]

    return run_isa.run(
        shim,
        embed_table=embed_table,
        layer=layer,
        kind="kq",
        seed=seed,
        split_mode=split_mode,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="M2.7 hidden-state attacks")
    p.add_argument("--captures-dir", type=Path, required=True,
                   help="Dir containing hidden.safetensors+.meta.json "
                        "and optionally attn.safetensors+.meta.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--ima-layer", type=int, default=0)
    p.add_argument("--ima-kind", default="attn_norm")
    p.add_argument("--isa-layer", type=int, default=23)
    p.add_argument("--isa-kind", default="attn_norm")
    p.add_argument("--include-attn-score", action="store_true",
                   help="Also run ISA against the AttnScore (kq-) "
                        "observable. Requires attn.safetensors present.")
    p.add_argument("--include-paper-like-ima", action="store_true",
                   help="Run the trained-transformer IMA paper-like "
                        "(slow; needs ≥256 prompts to fit).")
    p.add_argument("--split-mode", choices=("row", "vocab"), default="row",
                   help="Ridge train/val/test split for IMA basic + ISA: "
                        "'row' = legacy row-shuffle (memorising attacker); "
                        "'vocab' = paper-faithful vocab-disjoint "
                        "(generalising attacker — the reference impl uses this).")
    from m2_7_common import add_min_mem_args, check_phase_memory  # type: ignore
    add_min_mem_args(p, phase="hidden_attacks")
    args = p.parse_args()

    check_phase_memory("hidden_attacks", args.min_mem_gb, args.skip_mem_check)
    print(f"[M2.7 attacks] loading hidden.safetensors from {args.captures_dir}")
    hidden_set = SnapshotSet.open("hidden", root=args.captures_dir)
    embed_table = _load_embed_table(args.model_id)

    results: dict[str, dict] = {}

    print("[M2.7 attacks] running NN…")
    nn_res = run_nn.run(
        hidden_set, embed_table=embed_table,
        layer=args.ima_layer, kind=args.ima_kind, strip_shield=False,
    )
    results["nn"] = nn_res.to_dict()
    print(f"  nn top1={nn_res.ttrsr_top1} risk={nn_res.risk_level}")

    print(f"[M2.7 attacks] running IMA basic (multi-α ridge, split={args.split_mode})…")
    ima_res = run_ima.run(
        hidden_set, embed_table=embed_table,
        layer=args.ima_layer, kind=args.ima_kind, strip_shield=False,
        split_mode=args.split_mode,
    )
    results["ima"] = ima_res.to_dict()
    print(f"  ima top1={ima_res.ttrsr_top1} risk={ima_res.risk_level}")

    if args.include_paper_like_ima:
        print("[M2.7 attacks] running IMA paper-like (trained 2-layer inverter)…")
        pl_res = run_ima_paper_like.run(
            hidden_set, embed_table=embed_table,
            layer=args.ima_layer, kind=args.ima_kind, strip_shield=False,
        )
        results["ima_paper_like"] = pl_res.to_dict()
        print(f"  ima_paper_like top1={pl_res.ttrsr_top1} risk={pl_res.risk_level}")
    else:
        results["ima_paper_like"] = {
            "attack": "ima_paper_like",
            "condition": hidden_set.condition,
            "ttrsr_top1": None,
            "risk_level": "skipped",
            "extra": {"note": "pass --include-paper-like-ima; needs ≥256 prompts"},
        }

    print(f"[M2.7 attacks] running ISA at HiddenState (split={args.split_mode})…")
    isa_hs = run_isa.run(
        hidden_set, embed_table=embed_table,
        layer=args.isa_layer, kind=args.isa_kind, strip_shield=False,
        split_mode=args.split_mode,
    )
    results["isa_hidden_state"] = isa_hs.to_dict()
    print(f"  isa_hidden_state top1={isa_hs.ttrsr_top1} risk={isa_hs.risk_level}")

    if args.include_attn_score:
        attn_path = args.captures_dir / "attn.safetensors"
        if not attn_path.exists():
            print(f"  ⚠ attn.safetensors not found at {attn_path}; "
                  f"skipping AttnScore. Run capture_hidden_states.py --mode attn first.")
            results["isa_attn_score"] = {"ttrsr_top1": None,
                                         "risk_level": "skipped",
                                         "extra": {"note": "no attn capture"}}
        else:
            print(f"[M2.7 attacks] running ISA at AttnScore (split={args.split_mode})…")
            attn_set = SnapshotSet.open("attn", root=args.captures_dir)
            isa_as = _isa_attn_score(attn_set, embed_table, args.isa_layer,
                                     seed=20260518, split_mode=args.split_mode)
            results["isa_attn_score"] = isa_as.to_dict()
            print(f"  isa_attn_score top1={isa_as.ttrsr_top1} risk={isa_as.risk_level}")
    else:
        results["isa_attn_score"] = {
            "attack": "isa_attn_score",
            "condition": hidden_set.condition,
            "ttrsr_top1": None,
            "risk_level": "skipped",
            "extra": {"note": "pass --include-attn-score (requires --flash-attn off server pass)"},
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "format": "aloepri_m2_7_hidden_state_v1",
        "captures_dir": str(args.captures_dir),
        "model_id": args.model_id,
        "attacks": results,
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"[M2.7 attacks] wrote → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

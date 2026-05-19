"""IA — Invariant Attack driver (Gate-IA + Attn-IA).

Paper §F.1 specifies two variants — both **static-weight**
attacks that recover the secret token permutation Π by exploiting
mathematical invariants that survive AloePri's offline weight
obfuscation:

* **Gate-IA**: invariant `Avg(e_i · W_gate) = Avg(ẽ_i · W̃_gate)`.
  Survives because permutation + scaling on `W_gate` preserves
  the row-mean of the gate-projected embedding. Defence:
  add sufficient noise (α_e ≥ 1.0) so the row-mean drifts.

* **Attn-IA**: invariant `e_i (QᵀQ)⁻¹ e_iᵀ = ẽ_i (Q̃ᵀQ̃)⁻¹ ẽ_iᵀ`
  computed block-wise on `W_query / W_key`. Survives because the
  quadratic form is invariant under invertible right-multiplication.
  Defence: head and block permutations (paper §5.2.3 Algorithm 2).

**Both attacks target the static-weight obfuscation surface that
GELO does not have.** GELO is openweight by design (per
`docs/prototype/gelo.md` §2) — there is no `W̃_e` for the
attacker to compare against `W_e`. Running IA against
`(W_e, W_e)` returns an identity permutation with TTRSR = 100%
mechanically: not a privacy failure of GELO, but a tautology of
the threat model.

The driver reports both:
* `ttrsr_top1` — the mechanical static-weight number (100% for
  GELO). Matches AloePri Table 1's IA column framing.
* `extra.activation_axis_top1` — the meaningful GELO question:
  can the attacker recover Π from the **masked activation
  observation** instead of from weights? Should be near 0%
  under per-batch Haar mask.

If the reference impl ever ships a real `ia.py` (currently a
30-line phase-0 template), we can swap in their RowSort variant
and re-derive the static-weight number. For now this is the
honest port of `vendor/aloepri-py/src/security_qwen/ia.py:6` ←
`status: planned`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .common import (
    AttackResult,
    classify_risk_level,
    stack_prompt_observations,
)
from .run_ima import load_qwen3_embedding_table


# ─── Invariant computations ──────────────────────────────────────


def gate_ia_invariants(
    embed_table: torch.Tensor, w_gate: torch.Tensor
) -> torch.Tensor:
    """Paper §F.1: `Avg(e_i · W_gate)` per token row.

    Returns shape `(vocab_size,)`.
    """
    projected = embed_table.to(torch.float32) @ w_gate.to(torch.float32)
    return projected.mean(dim=1)


def attn_ia_invariants(
    embed_table: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    *,
    block_size: int = 16,
) -> torch.Tensor:
    """Paper §F.1: `e_i (QᵀQ)⁻¹ e_iᵀ` evaluated over blocks of the
    RoPE-block-aligned Q/K projection. We split the head dimension
    into `block_size`-wide chunks (default 16 matches the paper's
    `β = 8` × 2-dim RoPE pair); for each block we compute the
    quadratic form `e_i Mᵀ M e_iᵀ` where M is the block of `Wq` and
    `Wk` stacked.

    Returns shape `(vocab_size, n_blocks)`. The attacker matches
    rows of this matrix to recover Π.

    `w_q`, `w_k` are `(hidden_size, head_dim)` — already merged for
    a single head. For GQA we use the first head's worth of
    columns.
    """
    we = embed_table.to(torch.float32)
    wq = w_q.to(torch.float32)
    wk = w_k.to(torch.float32)
    head_dim = wq.shape[1]
    n_blocks = max(head_dim // block_size, 1)
    invariants = torch.zeros((we.shape[0], n_blocks), dtype=torch.float32)
    for b in range(n_blocks):
        s = b * block_size
        e = min(s + block_size, head_dim)
        # Stack Wq + Wk block columns → (hidden, 2*block_width)
        m = torch.cat([wq[:, s:e], wk[:, s:e]], dim=1)
        # Row-wise: invariant_i = || e_i · m ||^2
        proj = we @ m
        invariants[:, b] = (proj * proj).sum(dim=1)
    return invariants


# ─── Recovery: match plain invariants to "obfuscated" invariants ───


def _match_by_nearest(
    plain_inv: torch.Tensor, observed_inv: torch.Tensor, topk: int = 10
) -> torch.Tensor:
    """For each plain token id `i`, find the observed-id `j` whose
    invariant is closest. Returns `(vocab, topk)` predicted ids.

    Both tensors are `(vocab,)` or `(vocab, d)`. We compute pairwise
    L2 distance and take the smallest.
    """
    if plain_inv.dim() == 1:
        plain_inv = plain_inv.unsqueeze(1)
        observed_inv = observed_inv.unsqueeze(1)
    # Pairwise L2 distance: ||p_i - o_j||^2 = ||p_i||^2 - 2 p_i·o_j + ||o_j||^2
    p = plain_inv
    o = observed_inv
    pp = (p * p).sum(dim=1, keepdim=True)
    oo = (o * o).sum(dim=1, keepdim=True)
    dist2 = pp - 2.0 * (p @ o.T) + oo.T
    # Smallest distance = best match. topk neighbors per row.
    _, top_idx = torch.topk(-dist2, k=min(topk, dist2.shape[1]), dim=1)
    return top_idx


# ─── Driver ──────────────────────────────────────────────────────


def run(
    snapshots,
    *,
    embed_table: torch.Tensor,
    decoder_weights_dir: str | None = None,
    layer: int = 0,
    kind: str = "q_proj",
    topk: int = 10,
    eval_vocab_size: int = 4096,
    seed: int = 20260518,
) -> AttackResult:
    """Run both Gate-IA and Attn-IA against the snapshot set's
    metadata.

    `decoder_weights_dir` is the HF snapshot dir containing the
    Qwen3 safetensors (auto-discovered if None — same logic as
    `run_ima.load_qwen3_embedding_table`). We need `model.layers.{li}.mlp.gate_proj.weight`
    and `model.layers.{li}.self_attn.{q,k}_proj.weight`.

    `eval_vocab_size` bounds the per-attack cost. The paper
    evaluates on the full vocab (~150k); we sample a subset for
    runtime. The TTRSR is computed over the sampled subset.
    """
    # Load gate + q + k for the target layer. We use the snapshot's
    # condition only to label the result row; the attack itself runs
    # against the model's PUBLIC weights regardless of condition.
    w_gate = _load_decoder_weight(
        snapshots.model_id, decoder_weights_dir, f"model.layers.{layer}.mlp.gate_proj.weight"
    )
    w_q = _load_decoder_weight(
        snapshots.model_id, decoder_weights_dir, f"model.layers.{layer}.self_attn.q_proj.weight"
    )
    w_k = _load_decoder_weight(
        snapshots.model_id, decoder_weights_dir, f"model.layers.{layer}.self_attn.k_proj.weight"
    )

    # HuggingFace stores W as (out_features, in_features); we need
    # (in_features, out_features) for the right-multiply convention.
    w_gate = w_gate.T.contiguous()
    w_q = w_q.T.contiguous()
    w_k = w_k.T.contiguous()

    rng = np.random.default_rng(seed)
    vocab_size = embed_table.shape[0]
    eval_ids = torch.from_numpy(
        rng.choice(vocab_size, size=min(eval_vocab_size, vocab_size), replace=False)
    ).to(torch.long)
    eval_embed = embed_table[eval_ids]

    # ─── Static-weight branch (paper-faithful) ──────────────────
    # Under GELO's threat model the "obfuscated" weights == plain
    # weights, so the invariants match trivially.
    plain_gate_inv = gate_ia_invariants(eval_embed, w_gate)
    obs_gate_inv = plain_gate_inv.clone()  # GELO: identity
    gate_top_ids = _match_by_nearest(plain_gate_inv, obs_gate_inv, topk=topk)
    gate_static_top1 = float(
        (gate_top_ids[:, 0] == torch.arange(eval_ids.shape[0])).to(torch.float32).mean().item()
    )

    plain_attn_inv = attn_ia_invariants(eval_embed, w_q, w_k)
    obs_attn_inv = plain_attn_inv.clone()  # GELO: identity
    attn_top_ids = _match_by_nearest(plain_attn_inv, obs_attn_inv, topk=topk)
    attn_static_top1 = float(
        (attn_top_ids[:, 0] == torch.arange(eval_ids.shape[0])).to(torch.float32).mean().item()
    )

    # Combined static-weight TTRSR — voting between Gate-IA + Attn-IA.
    static_top1 = max(gate_static_top1, attn_static_top1)

    # ─── Activation-axis branch (GELO-meaningful) ──────────────
    # Use the layer-0 q_proj operand as the "obfuscated embedding"
    # observation. For each masked observation row, compute the
    # same invariants and try to match to the plain table. Under
    # per-batch Haar mask + shield this should fail (near-zero).
    act_gate_top1 = None
    act_attn_top1 = None
    try:
        from .common import stack_prompt_observations

        Xobs, yobs, _ = stack_prompt_observations(
            snapshots, layer=layer, kind=kind, strip_shield=True
        )
        if Xobs.shape[0] > 0 and Xobs.shape[1] == embed_table.shape[1]:
            obs_t = torch.from_numpy(Xobs).to(torch.float32)
            obs_y = torch.from_numpy(yobs).to(torch.long)
            # For each observed row, what plain id would Gate-IA pick?
            obs_gate_inv_act = gate_ia_invariants(obs_t, w_gate)
            # Match each observed row against the eval-vocab invariant table.
            eval_gate_inv = gate_ia_invariants(eval_embed, w_gate)
            gate_match = _match_by_nearest(obs_gate_inv_act, eval_gate_inv, topk=1)
            predicted_plain = eval_ids[gate_match.squeeze(1)]
            act_gate_top1 = float(
                (predicted_plain == obs_y).to(torch.float32).mean().item()
            )

            obs_attn_inv_act = attn_ia_invariants(obs_t, w_q, w_k)
            eval_attn_inv = attn_ia_invariants(eval_embed, w_q, w_k)
            attn_match = _match_by_nearest(obs_attn_inv_act, eval_attn_inv, topk=1)
            predicted_plain = eval_ids[attn_match.squeeze(1)]
            act_attn_top1 = float(
                (predicted_plain == obs_y).to(torch.float32).mean().item()
            )
    except Exception as exc:  # surface but don't kill the row
        return _ia_result(
            snapshots, static_top1, gate_static_top1, attn_static_top1,
            act_gate_top1=None, act_attn_top1=None,
            note=f"activation-axis IA failed: {exc!r}",
            layer=layer, vocab_eval=int(eval_ids.shape[0]),
        )

    return _ia_result(
        snapshots,
        static_top1,
        gate_static_top1,
        attn_static_top1,
        act_gate_top1=act_gate_top1,
        act_attn_top1=act_attn_top1,
        note=None,
        layer=layer,
        vocab_eval=int(eval_ids.shape[0]),
    )


def _ia_result(
    snapshots,
    static_top1: float,
    gate_static_top1: float,
    attn_static_top1: float,
    *,
    act_gate_top1: float | None,
    act_attn_top1: float | None,
    note: str | None,
    layer: int,
    vocab_eval: int,
) -> AttackResult:
    extra = {
        "layer": layer,
        "vocab_eval_size": vocab_eval,
        "gate_ia_static_top1": gate_static_top1,
        "attn_ia_static_top1": attn_static_top1,
        "gate_ia_activation_top1": act_gate_top1,
        "attn_ia_activation_top1": act_attn_top1,
        "note": (
            note
            if note is not None
            else (
                "IA targets static-weight obfuscation. GELO does not obfuscate "
                "weights, so the static-weight invariant match is trivially "
                "the identity (TTRSR=100% mechanically). The meaningful "
                "GELO question is `extra.{gate,attn}_ia_activation_top1` — "
                "can the attacker recover Π from masked activations? Under "
                "per-batch Haar mask this should be ≪ 1%."
            )
        ),
        "phase": "implemented",
    }
    return AttackResult(
        attack="ia",
        condition=snapshots.condition,
        model_id=snapshots.model_id,
        n_prompts=snapshots.n_prompts(),
        n_train=0,
        n_test=vocab_eval,
        ttrsr_top1=static_top1,
        ttrsr_top10=None,
        risk_level=classify_risk_level(static_top1),
        extra=extra,
    )


# ─── Weight loading (shared with IMA paper-like) ─────────────────


def _load_decoder_weight(
    model_id: str, decoder_weights_dir: str | None, key: str
) -> torch.Tensor:
    """Pull a specific tensor out of the HF-cached safetensors shards
    for the given model. Bypasses `transformers` — same pattern as
    `run_ima.load_qwen3_embedding_table`.
    """
    import glob
    import os

    if decoder_weights_dir is not None:
        snapshot_dir = decoder_weights_dir
    else:
        cache_root = os.environ.get(
            "HF_HOME", os.path.expanduser("~/.cache/huggingface")
        )
        repo_dir_name = "models--" + model_id.replace("/", "--")
        candidates = sorted(
            glob.glob(os.path.join(cache_root, "hub", repo_dir_name, "snapshots", "*"))
        )
        if not candidates:
            raise FileNotFoundError(
                f"No HF cache for {model_id!r}; pass --decoder-weights-dir or "
                "install transformers and download the model."
            )
        snapshot_dir = candidates[-1]

    index_path = os.path.join(snapshot_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        import json as _json

        with open(index_path) as fh:
            weight_map = _json.load(fh)["weight_map"]
        if key not in weight_map:
            raise KeyError(
                f"weight {key!r} not found in {index_path}. "
                f"Available example keys: {list(weight_map.keys())[:3]}…"
            )
        shard_path = os.path.join(snapshot_dir, weight_map[key])
    else:
        shard_path = os.path.join(snapshot_dir, "model.safetensors")
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"no safetensors under {snapshot_dir}")

    from safetensors import safe_open

    with safe_open(shard_path, framework="pt", device="cpu") as fh:
        return fh.get_tensor(key).to(torch.float32).clone()


def _cli() -> None:
    p = argparse.ArgumentParser(description="Run IA (Gate-IA + Attn-IA) against a snapshot set")
    p.add_argument("--snapshot-basename", required=True)
    p.add_argument("--snapshot-root", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--eval-vocab-size", type=int, default=4096)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from snapshots_loader import SnapshotSet  # type: ignore

    snaps = SnapshotSet.open(args.snapshot_basename, root=args.snapshot_root)
    embed_table = load_qwen3_embedding_table(args.model_id)
    result = run(
        snaps,
        embed_table=embed_table,
        layer=args.layer,
        eval_vocab_size=args.eval_vocab_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2))
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    _cli()

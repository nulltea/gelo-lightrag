"""IMA paper-like — trained 2-layer transformer inverter (paper §F.1).

The version of IMA AloePri actually publishes results for in Table 1.
Per §F.1 paragraph "Inversion Model Attack (IMA)":
*"we train a Qwen2 model with 2 decoder layers and 8 attention heads
to invert obfuscated embeddings to plaintext token embeddings."*

This driver mirrors the architecture and protocol:

* 2-layer pre-LN transformer with `num_heads=8`, optional input
  projection when the observation dim differs from the inverter's
  hidden size, and a final linear projection back to the target
  embedding dim.
* Training: MSE loss between predicted and target plain-embedding
  rows, AdamW, ~2 epochs by default.
* Eval: predicted embeddings → cosine top-K against the embedding
  table → TTRSR.

We split **by prompt** (not by row): 75% of prompts for training,
25% for eval. Within-prompt row leakage would inflate TTRSR
artifactually because nearby positions share context.

`transformers` is not used — we roll a minimal pre-LN block in
`torch.nn` so the harness keeps its zero-`transformers`
dependency surface. The trade-off is that we don't reuse Qwen3's
RoPE / QK-norm / SwiGLU — fine, because the inverter is a
generic learned function over hidden states, not a Qwen3 forward
pass. AloePri's `_PaperLikeIMAInverter` similarly uses
`AutoModel.from_config(... hidden_size=observed_dim,
num_layers=2)` which discards the original model's specifics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .common import (
    AttackResult,
    classify_risk_level,
    load_aloepri_module,
)
from .run_ima import load_qwen3_embedding_table


_ima = load_aloepri_module("src/security_qwen/ima.py")
_evaluate_inversion_predictions = _ima._evaluate_inversion_predictions


# ─── 2-layer pre-LN transformer inverter ───────────────────────────


class _InverterBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int, ffn_mult: int = 4) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(
            hidden, n_heads, batch_first=True, bias=False
        )
        self.ln2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * ffn_mult, bias=False),
            nn.GELU(),
            nn.Linear(hidden * ffn_mult, hidden, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class IMAInverter(nn.Module):
    """Generic `(B, T, observed_dim) → (B, T, output_dim)` inverter.

    Matches the role of AloePri's `_PaperLikeIMAInverter` but doesn't
    require `transformers`. The only architectural shortcut vs a
    real 2-layer Qwen3 is that we use vanilla LayerNorm / GELU FFN /
    standard MultiheadAttention — none of Qwen3's RoPE or QK-norm.
    The inverter is allowed any architecture; the privacy claim is
    that *no architecture* should be able to recover Π from the
    masked observations.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        inverter_hidden: int,
        n_layers: int = 2,
        n_heads: int = 8,
        output_dim: int,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        self.input_proj: nn.Module
        if observed_dim != inverter_hidden:
            self.input_proj = nn.Linear(observed_dim, inverter_hidden, bias=False)
        else:
            self.input_proj = nn.Identity()
        self.blocks = nn.ModuleList(
            [_InverterBlock(inverter_hidden, n_heads, ffn_mult) for _ in range(n_layers)]
        )
        self.output_proj = nn.Linear(inverter_hidden, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.output_proj(h)


# ─── Per-prompt data assembly ──────────────────────────────────────


def _gather_prompt_tensors(
    snapshots,
    *,
    layer: int,
    kind: str,
    strip_shield: bool,
) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    """Return `(prompt_idx, obs_tensor (T, d_obs), token_ids (T,))` per prompt.

    Truncates each to `min(snapshot_n_rows, len(prompt_token_ids))`.
    """
    out: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    pairs = snapshots.per_prompt_layer_kind_tensors(
        layer=layer, kind=kind, strip_shield=strip_shield
    )
    for prompt_idx, op in pairs:
        ids = torch.tensor(snapshots.prompt_token_ids[prompt_idx], dtype=torch.long)
        n = min(op.shape[0], ids.shape[0])
        if n == 0:
            continue
        out.append((prompt_idx, op[:n].to(torch.float32), ids[:n]))
    return out


def _split_by_prompt(
    per_prompt: list[tuple[int, torch.Tensor, torch.Tensor]],
    *,
    train_frac: float = 0.75,
    seed: int = 0,
) -> tuple[list[tuple[int, torch.Tensor, torch.Tensor]], list[tuple[int, torch.Tensor, torch.Tensor]]]:
    rng = np.random.default_rng(seed)
    order = list(range(len(per_prompt)))
    rng.shuffle(order)
    n_train = max(int(len(order) * train_frac), 1)
    train_idx = set(order[:n_train])
    train = [per_prompt[i] for i in range(len(per_prompt)) if i in train_idx]
    test = [per_prompt[i] for i in range(len(per_prompt)) if i not in train_idx]
    if not test and len(per_prompt) >= 2:
        # always reserve at least one for test
        test = [train.pop()]
    return train, test


# ─── Training + eval ───────────────────────────────────────────────


def _train(
    inverter: IMAInverter,
    train_set: list[tuple[int, torch.Tensor, torch.Tensor]],
    embed_table: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> list[float]:
    inverter.train()
    opt = torch.optim.AdamW(inverter.parameters(), lr=lr, weight_decay=weight_decay)
    losses: list[float] = []
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        order = list(range(len(train_set)))
        rng.shuffle(order)
        for batch_start in range(0, len(order), batch_size):
            batch_idx = order[batch_start : batch_start + batch_size]
            obs_list = [train_set[i][1] for i in batch_idx]
            tok_list = [train_set[i][2] for i in batch_idx]
            # Right-pad to batch's max sequence length.
            max_T = max(x.shape[0] for x in obs_list)
            d_obs = obs_list[0].shape[1]
            B = len(batch_idx)
            obs_batch = torch.zeros((B, max_T, d_obs), dtype=torch.float32)
            target_batch = torch.zeros((B, max_T, embed_table.shape[1]), dtype=torch.float32)
            mask = torch.zeros((B, max_T), dtype=torch.bool)
            for b, (obs, toks) in enumerate(zip(obs_list, tok_list)):
                T = obs.shape[0]
                obs_batch[b, :T] = obs
                target_batch[b, :T] = embed_table[toks]
                mask[b, :T] = True
            obs_batch = obs_batch.to(device)
            target_batch = target_batch.to(device)
            mask = mask.to(device)

            pred = inverter(obs_batch)
            mse = ((pred - target_batch) ** 2).sum(dim=-1)  # (B, T)
            loss = (mse * mask).sum() / mask.sum().clamp_min(1)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(inverter.parameters(), max_norm=1.0)
            opt.step()
            losses.append(float(loss.item()))
    return losses


def _evaluate(
    inverter: IMAInverter,
    test_set: list[tuple[int, torch.Tensor, torch.Tensor]],
    embed_table: torch.Tensor,
    *,
    candidate_pool: torch.Tensor,
    topk: int,
    device: torch.device,
) -> dict:
    inverter.eval()
    preds: list[torch.Tensor] = []
    truth: list[torch.Tensor] = []
    with torch.no_grad():
        for _, obs, toks in test_set:
            obs_t = obs.unsqueeze(0).to(device)
            pred = inverter(obs_t).squeeze(0).cpu()
            preds.append(pred)
            truth.append(toks)
    pred_flat = torch.cat(preds, dim=0)
    truth_flat = torch.cat(truth, dim=0)
    return _evaluate_inversion_predictions(
        predicted_embeddings=pred_flat,
        true_plain_ids=truth_flat,
        candidate_plain_ids=candidate_pool,
        baseline_embed=embed_table,
        topk=topk,
    )


# ─── Driver ─────────────────────────────────────────────────────────


def run(
    snapshots,
    *,
    embed_table: torch.Tensor,
    layer: int = 0,
    kind: str = "q_proj",
    strip_shield: bool = True,
    inverter_hidden: int = 256,
    n_layers: int = 2,
    n_heads: int = 8,
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    candidate_pool_size: int = 2048,
    topk: int = 10,
    seed: int = 20260518,
) -> AttackResult:
    per_prompt = _gather_prompt_tensors(
        snapshots, layer=layer, kind=kind, strip_shield=strip_shield
    )
    if len(per_prompt) < 2:
        return AttackResult(
            attack="ima_paper_like",
            condition=snapshots.condition,
            model_id=snapshots.model_id,
            n_prompts=snapshots.n_prompts(),
            n_train=0,
            n_test=0,
            ttrsr_top1=None,
            ttrsr_top10=None,
            risk_level="unknown",
            extra={"note": "not enough compatible prompts (need ≥ 2)"},
        )
    # obs_dim and embed_dim may differ — the inverter takes
    # `observed_dim` and `output_dim` separately, so it handles the
    # bridge naturally.
    obs_dim = per_prompt[0][1].shape[1]
    if obs_dim != embed_table.shape[1]:
        print(
            f"  [ima_paper_like] obs_dim={obs_dim} != embed_dim={embed_table.shape[1]} — "
            f"2-layer inverter learns the dim bridge"
        )
    train_set, test_set = _split_by_prompt(per_prompt, train_frac=0.75, seed=seed)
    if not train_set or not test_set:
        return AttackResult(
            attack="ima_paper_like",
            condition=snapshots.condition,
            model_id=snapshots.model_id,
            n_prompts=snapshots.n_prompts(),
            n_train=len(train_set),
            n_test=len(test_set),
            ttrsr_top1=None,
            ttrsr_top10=None,
            risk_level="unknown",
            extra={"note": "split produced an empty branch"},
        )

    device = torch.device("cpu")
    torch.manual_seed(seed)
    inverter = IMAInverter(
        observed_dim=per_prompt[0][1].shape[1],
        inverter_hidden=inverter_hidden,
        n_layers=n_layers,
        n_heads=n_heads,
        output_dim=embed_table.shape[1],
    ).to(device)
    losses = _train(
        inverter,
        train_set,
        embed_table,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )

    rng = np.random.default_rng(seed + 1)
    vocab_size = embed_table.shape[0]
    candidate_pool = torch.from_numpy(
        rng.choice(vocab_size, size=min(candidate_pool_size, vocab_size), replace=False)
    ).to(torch.long)
    test_ids = torch.cat([toks for _, _, toks in test_set])
    candidate_pool = torch.unique(torch.cat([candidate_pool, test_ids]))

    metrics = _evaluate(
        inverter,
        test_set,
        embed_table,
        candidate_pool=candidate_pool,
        topk=topk,
        device=device,
    )

    n_train_rows = sum(o.shape[0] for _, o, _ in train_set)
    n_test_rows = sum(o.shape[0] for _, o, _ in test_set)
    top1 = float(metrics["token_top1_recovery_rate"])
    top10 = float(metrics["token_top10_recovery_rate"])
    return AttackResult(
        attack="ima_paper_like",
        condition=snapshots.condition,
        model_id=snapshots.model_id,
        n_prompts=snapshots.n_prompts(),
        n_train=n_train_rows,
        n_test=n_test_rows,
        ttrsr_top1=top1,
        ttrsr_top10=top10,
        risk_level=classify_risk_level(top1),
        extra={
            "layer": layer,
            "kind": kind,
            "strip_shield": strip_shield,
            "inverter_hidden": inverter_hidden,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "final_train_loss": losses[-1] if losses else None,
            "n_train_prompts": len(train_set),
            "n_test_prompts": len(test_set),
            "candidate_pool_size": int(candidate_pool.shape[0]),
            "embedding_cosine_similarity": float(metrics["embedding_cosine_similarity"]),
        },
    )


def _cli() -> None:
    p = argparse.ArgumentParser(description="Run IMA paper-like (trained inverter) against a snapshot set")
    p.add_argument("--snapshot-basename", required=True)
    p.add_argument("--snapshot-root", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--kind", default="q_proj")
    p.add_argument("--inverter-hidden", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
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
        kind=args.kind,
        inverter_hidden=args.inverter_hidden,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        epochs=args.epochs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2))
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    _cli()

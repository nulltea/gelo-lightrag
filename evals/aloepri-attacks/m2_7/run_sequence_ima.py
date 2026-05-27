"""Sequence-IMA attack — sequence-level inversion of obfuscated activations.

## Why this attack exists (gap analysis vs paper IMA)

Paper §F.1 + Appendix D.1 IMA trains a Qwen2 2-layer 8-head inverter
that maps an obfuscated embedding ROW to a plaintext token id. The
training data uses 32-token sequence windows (paper Table 9), but the
inverter is structurally PER-ROW — each row is inverted independently.

This makes paper-IMA vulnerable to the **noise-averaging defence**:
AloePri's α_e = 1.0 embedding noise is large (≈ σ_W). Per-row, the
SNR is bounded by `1 / α_e ≈ 1`. The paper's Vec2Text reproducibility
study (arXiv 2507.07700) confirms Gaussian noise at λ=0.01 already
defeats Vec2Text-class per-row inverters; α_e=1.0 is 100× higher.

**Sequence-IMA hypothesis:** when the inverter conditions on the FULL
SEQUENCE of obfuscated activations (not just one row), it can average
the noise out across n_q positions, recovering signal that per-row
methods cannot. Effective SNR scales as `√n_q / α_e`. For typical
prompts (n_q ≈ 16-32 tokens), that's a 4-6× SNR gain over per-row.

If the hypothesis holds, Sequence-IMA recovers tokens at significantly
higher TTRSR than paper's IMA on the same cells — closing the
"per-row noise averages out at the sequence level" gap the harness
currently doesn't measure.

## Architecture

A small encoder-decoder transformer:

  - Encoder: n_q obfuscated activation rows (each d_obs-dim) →
    n_q context vectors via self-attention.
    2 layers, 8 heads, hidden=d_obs. Matches paper IMA architecture
    budget so the comparison is apples-to-apples — the only structural
    difference is the per-row vs sequence-level conditioning.

  - Decoder: cross-attends from encoder context to predict plaintext
    token id at each position. Output: (n_q, V) logits.

  - Loss: per-position cross-entropy on plaintext token ids.

## Surfaces this attacks

Driven by `--surface`:
  - `embed`: obfuscated embedding rows `W̃[τ[token_ids]]` (paper IMA's
    surface). Sequence-IMA on this surface tests whether the per-row
    paper result is bounded by noise averaging.
  - `hidden_l0`: hidden states at decoder layer 0 input (after the
    `Π·W·Q̂ + α_e·ε` chain). Closer to a runtime attack.
  - `kqv_out_lN`: per-head attention output at layer N before W_o
    (the §5.4-bounded surface; see 2B.1 measurement at
    `evals/aloepri-attacks/results/sweep/2B1-attn-output-vs-kq-comparison.md`).
  - `kq_lN`: pre-softmax attention scores at layer N.

For `embed`, training data can be synthesised purely from the
obfuscated model's static weights — no inference needed. For all other
surfaces, training data requires running prompts through the obfuscated
model to capture the corresponding hidden states (see capture pipeline
in `evals/aloepri-attacks/m2_7/capture_hidden_states.py`).

## Threat model

Same as paper IMA-EmbedRow-transformer (paper §F.1):
  - Adversary has obfuscated weights (W̃) and the public plaintext
    reference (W). They know the obfuscation algorithm (Kerckhoffs's
    principle) but not the secret keys (Π, P̂, Q̂, noise seed, Algorithm
    2 keys).
  - Adversary picks their OWN (Π_a, K_a, noise_a) and runs the
    obfuscator on a public corpus to generate synthetic
    (plain_seq, obf_seq) training pairs.
  - The trained inverter must learn a τ-INVARIANT inverse — same as
    paper IMA-EmbedRow-transformer.

## Status: SCAFFOLD ONLY (2026-05-26)

This file implements the ARCHITECTURE, TRAINING LOOP, and EVAL HARNESS
but is NOT run in this session. The user-facing CLI works (--help
parses); the training loop has been syntax-validated but not exercised
against real data. Next session: (a) generate synthetic training data
from the public corpus, (b) run a short training run on the `embed`
surface to compare against paper IMA-EmbedRow-transformer's existing
per-row result, (c) extend to `hidden_l0` and `kqv_out_lN` surfaces.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_gguf_weights import ModelWeights, load_model  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attack_drivers.common import AttackResult, classify_risk_level  # type: ignore


# ───── Architecture ────────────────────────────────────────────────


@dataclass
class SeqIMAConfig:
    """Configuration for the sequence-level inverter.

    Defaults match paper §F.1 / §F.2 inverter budget: 2 decoder layers,
    8 heads, hidden=d_obs (so the inverter has same parameter count
    as the paper inverter — comparison is on conditioning structure,
    not on capacity).
    """
    d_obs: int                          # input hidden dim per row
    vocab_size: int                     # output vocab |V|
    n_layers: int = 2
    n_heads: int = 8
    d_model: int | None = None          # hidden dim; defaults to d_obs
    d_ff: int | None = None             # FFN dim; defaults to 4*d_model
    seq_len: int = 32                   # max prompt length (paper Table 9)
    dropout: float = 0.0

    def __post_init__(self):
        if self.d_model is None:
            self.d_model = self.d_obs
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model


class _Block(nn.Module):
    """Pre-LN transformer block (matches paper IMA reference impl)."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        a = self.ln1(x)
        a, _ = self.attn(a, a, a, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class SequenceIMAInverter(nn.Module):
    """Encoder-only transformer that maps (B, seq_len, d_obs) → (B, seq_len, V).

    No decoder needed — this is a tagging task (predict plain token id
    at each position from the sequence of obfuscated activations).
    """
    def __init__(self, cfg: SeqIMAConfig):
        super().__init__()
        self.cfg = cfg
        # Project d_obs → d_model if they differ.
        self.input_proj = (
            nn.Identity() if cfg.d_obs == cfg.d_model
            else nn.Linear(cfg.d_obs, cfg.d_model)
        )
        # Learned positional embedding.
        self.pos_embed = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([
            _Block(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # Output projection to vocab.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x of shape (B, n_q, d_obs). Returns logits (B, n_q, V)."""
        B, n_q, _ = x.shape
        h = self.input_proj(x)
        pos = torch.arange(n_q, device=x.device).unsqueeze(0).expand(B, -1)
        h = h + self.pos_embed(pos)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        return self.lm_head(h)


# ───── Training data synthesis ─────────────────────────────────────


@dataclass
class SyntheticPair:
    """A single (plain_ids, obfuscated_activations) training pair."""
    plain_ids: np.ndarray            # (n_q,)
    obf_activations: np.ndarray      # (n_q, d_obs)


def synthesize_embed_pairs(
    plain: ModelWeights,
    *,
    attacker_tau: np.ndarray,
    attacker_keymat_Q: np.ndarray,
    attacker_noise_alpha_e: float,
    attacker_noise_seed: int,
    corpus_tokens: np.ndarray,
    n_sequences: int,
    seq_len: int,
    rng_seed: int = 20260526,
) -> list[SyntheticPair]:
    """Synthesise (plain_ids, obf_embed_rows) pairs for surface='embed'.

    The attacker picks their own (τ_a, Q̂_a, noise_a) and runs the
    obfuscator on a public corpus. The resulting pairs are the
    inverter's training set — see paper §F.1 Kerckhoffs assumption.

    Args:
        plain: plaintext model (provides W_embed and vocab_size).
        attacker_tau: attacker's chosen τ_a (V,).
        attacker_keymat_Q: attacker's Q̂_a (d, d_obs).
        attacker_noise_alpha_e: α_e_a chosen by attacker.
        corpus_tokens: 1-D array of public-corpus tokens.
        n_sequences: number of sequence windows to sample.
        seq_len: tokens per window.

    Returns: list of SyntheticPair.
    """
    rng = np.random.default_rng(rng_seed)
    W = plain.token_embd                                  # (V, d)
    V_, d = W.shape
    d_obs = attacker_keymat_Q.shape[1]
    # Build the attacker's W̃_a = (Π_a · W) · Q̂_a + α_e_a · ε
    perm_W = W[attacker_tau]                              # (V, d) row-permuted
    rng_noise = np.random.default_rng(attacker_noise_seed)
    noise = rng_noise.standard_normal(size=(V_, d_obs)).astype(np.float32)
    sigma = float(np.std(W))                              # match paper σ_e
    W_tilde_a = perm_W @ attacker_keymat_Q + attacker_noise_alpha_e * sigma * noise

    # Inverse permutation for lookup: tau_inv_a[obf_id] = plain_id
    tau_inv_a = np.argsort(attacker_tau).astype(np.int64)
    pairs: list[SyntheticPair] = []
    n_corpus = corpus_tokens.shape[0]
    for _ in range(n_sequences):
        start = int(rng.integers(0, max(1, n_corpus - seq_len)))
        plain_seq = corpus_tokens[start : start + seq_len].astype(np.int64)
        # Map plain → obfuscated via attacker's τ_a (attacker can
        # encode their own data; this gives them the obfuscated row).
        # NOTE: at inference time the attacker sees obfuscated rows
        # from the DEPLOYMENT τ, NOT τ_a. The inverter must learn a
        # τ-invariant mapping. Training on (random τ_a's, obf rows)
        # forces it.
        obf_ids = attacker_tau[plain_seq]
        obf_rows = W_tilde_a[obf_ids]                     # (seq_len, d_obs)
        pairs.append(SyntheticPair(plain_ids=plain_seq, obf_activations=obf_rows))
    return pairs


def synthesize_hidden_pairs_placeholder(*args, **kwargs):
    """Placeholder for non-`embed` surfaces.

    For `hidden_l0`, `kqv_out_lN`, `kq_lN` surfaces: synthesising
    training data requires running prompts through an obfuscated model
    (attacker's local synthetic obfuscation) to capture the
    corresponding tensor surface. See capture_hidden_states.py for the
    capture protocol. Implementing this requires:

      1. Build an attacker's local synthetic obfuscated cell (run
         obfuscate_qwen3_gguf.py with attacker's (τ_a, K_a) keys).
      2. Spawn the patched llama-server against the synthetic cell.
      3. Run capture_hidden_states.py on a public corpus to get
         (plain_ids, captured_hidden_states) pairs.
      4. Load the captures + plain_ids into the inverter training set.

    NOT IMPLEMENTED in this scaffold session. The architecture and
    training loop already accept (B, n_q, d_obs) activations agnostically
    — just need the pipeline that produces them for non-`embed` surfaces.
    """
    raise NotImplementedError(
        "non-`embed` surfaces require captures from a synthetic-key "
        "obfuscated cell — see docstring for the pipeline. Use --surface embed "
        "for the scaffold smoke test."
    )


# ───── Training loop ───────────────────────────────────────────────


def _batch_pairs(
    pairs: list[SyntheticPair], batch_size: int, device: str
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (X, y) batches: X (B, n_q, d_obs), y (B, n_q)."""
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        X = torch.from_numpy(np.stack([p.obf_activations for p in batch])).to(device)
        y = torch.from_numpy(np.stack([p.plain_ids for p in batch])).to(device)
        yield X, y


def train_inverter(
    model: SequenceIMAInverter,
    train_pairs: list[SyntheticPair],
    val_pairs: list[SyntheticPair],
    *,
    batch_size: int = 8,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    epochs: int = 2,
    device: str = "auto",
    log_every: int = 50,
) -> dict[str, Any]:
    """Train the inverter on synthetic pairs. Returns metrics dict.

    No checkpointing in this scaffold — caller is responsible for
    saving the model if needed. Hyperparameters match paper §F.2.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    metrics = {"train_loss_per_step": [], "val_top1_per_epoch": []}
    step = 0
    for epoch in range(epochs):
        model.train()
        rng = np.random.default_rng(20260526 + epoch)
        order = rng.permutation(len(train_pairs))
        shuffled = [train_pairs[i] for i in order]
        for X, y in _batch_pairs(shuffled, batch_size, device):
            logits = model(X)                                  # (B, n_q, V)
            loss = F.cross_entropy(
                logits.reshape(-1, model.cfg.vocab_size),
                y.reshape(-1),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % log_every == 0:
                metrics["train_loss_per_step"].append({
                    "step": step, "loss": float(loss.detach())
                })
            step += 1
        # Val pass.
        model.eval()
        with torch.no_grad():
            top1_hits, total = 0, 0
            for X, y in _batch_pairs(val_pairs, batch_size, device):
                pred = model(X).argmax(dim=-1)                # (B, n_q)
                top1_hits += int((pred == y).sum())
                total += int(y.numel())
        val_top1 = top1_hits / max(total, 1)
        metrics["val_top1_per_epoch"].append({"epoch": epoch, "val_top1": val_top1})
    return metrics


# ───── Eval ────────────────────────────────────────────────────────


def evaluate_inverter(
    model: SequenceIMAInverter,
    test_pairs: list[SyntheticPair],
    *,
    batch_size: int = 8,
    device: str = "auto",
    topk: int = 10,
) -> dict[str, Any]:
    """Compute TTRSR top-1 / top-K on the test set. Returns metrics."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    top1_hits, topk_hits, total = 0, 0, 0
    per_position_hits = np.zeros(model.cfg.seq_len, dtype=np.int64)
    per_position_total = np.zeros(model.cfg.seq_len, dtype=np.int64)
    with torch.no_grad():
        for X, y in _batch_pairs(test_pairs, batch_size, device):
            logits = model(X)                                  # (B, n_q, V)
            topk_pred = logits.topk(topk, dim=-1).indices      # (B, n_q, topk)
            top1_pred = topk_pred[..., 0]                       # (B, n_q)
            mask_top1 = (top1_pred == y)
            mask_topk = (topk_pred == y.unsqueeze(-1)).any(dim=-1)
            top1_hits += int(mask_top1.sum())
            topk_hits += int(mask_topk.sum())
            total += int(y.numel())
            for pos in range(min(model.cfg.seq_len, y.shape[1])):
                per_position_hits[pos] += int(mask_top1[:, pos].sum())
                per_position_total[pos] += int(y.shape[0])
    return {
        "ttrsr_top1": top1_hits / max(total, 1),
        "ttrsr_topk": topk_hits / max(total, 1),
        "topk": topk,
        "n_test_positions": int(total),
        "per_position_top1": (
            per_position_hits / np.maximum(per_position_total, 1)
        ).tolist(),
    }


# ───── End-to-end driver: synth → train → eval ────────────────────


def run_sequence_ima_embed_surface(
    plain: ModelWeights,
    *,
    attacker_keymat_Q: np.ndarray,
    attacker_alpha_e: float,
    attacker_noise_seed: int,
    attacker_tau_seed: int,
    corpus_tokens: np.ndarray,
    seq_len: int = 32,
    n_train: int = 128,
    n_val: int = 16,
    n_test: int = 16,
    n_layers: int = 2,
    n_heads: int = 8,
    batch_size: int = 8,
    lr: float = 3e-4,
    epochs: int = 2,
    device: str = "auto",
    rng_seed: int = 20260526,
) -> AttackResult:
    """End-to-end Sequence-IMA on the `embed` surface.

    Pipeline: synthesise pairs → instantiate inverter → train → eval.
    """
    rng = np.random.default_rng(attacker_tau_seed)
    V_ = plain.vocab_size
    attacker_tau = rng.permutation(V_).astype(np.int64)

    print(f"[seq-IMA] synthesising {n_train + n_val + n_test} pairs "
          f"(seq_len={seq_len}, d_obs={attacker_keymat_Q.shape[1]})")
    all_pairs = synthesize_embed_pairs(
        plain,
        attacker_tau=attacker_tau,
        attacker_keymat_Q=attacker_keymat_Q,
        attacker_noise_alpha_e=attacker_alpha_e,
        attacker_noise_seed=attacker_noise_seed,
        corpus_tokens=corpus_tokens,
        n_sequences=n_train + n_val + n_test,
        seq_len=seq_len,
        rng_seed=rng_seed,
    )
    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train : n_train + n_val]
    test_pairs = all_pairs[n_train + n_val : n_train + n_val + n_test]

    cfg = SeqIMAConfig(
        d_obs=attacker_keymat_Q.shape[1],
        vocab_size=V_,
        n_layers=n_layers,
        n_heads=n_heads,
        seq_len=seq_len,
    )
    model = SequenceIMAInverter(cfg)

    t0 = time.perf_counter()
    train_metrics = train_inverter(
        model, train_pairs, val_pairs,
        batch_size=batch_size, lr=lr, epochs=epochs, device=device,
    )
    train_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    eval_metrics = evaluate_inverter(model, test_pairs, batch_size=batch_size, device=device)
    eval_s = time.perf_counter() - t1

    return AttackResult(
        attack="sequence_ima_embed",
        condition="obfuscated",
        model_id=str(plain.path.name),
        n_prompts=n_train + n_val + n_test,
        n_train=n_train,
        n_test=n_test * seq_len,
        ttrsr_top1=float(eval_metrics["ttrsr_top1"]),
        ttrsr_top10=float(eval_metrics["ttrsr_topk"]),
        risk_level=classify_risk_level(float(eval_metrics["ttrsr_top1"])),
        extra={
            "config": asdict(cfg),
            "train_runtime_s": round(train_s, 2),
            "eval_runtime_s": round(eval_s, 2),
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "attacker_alpha_e": attacker_alpha_e,
            "attacker_tau_seed": attacker_tau_seed,
            "attacker_noise_seed": attacker_noise_seed,
        },
    )


# ───── CLI ─────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description="Sequence-IMA — sequence-level inverter for AloePri "
                    "obfuscated activations. SCAFFOLD only as of 2026-05-26 "
                    "— architecture + training + eval implemented; no runs "
                    "executed in this session."
    )
    p.add_argument("--plain", type=Path, required=True,
                   help="Plaintext GGUF. Used to access W_embed for synthesis.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--surface", type=str, default="embed",
                   choices=("embed",),
                   help="Activation surface to attack. 'embed' is implemented; "
                        "hidden_l0 / kqv_out / kq require captures (see docstring).")
    p.add_argument("--corpus-file", type=Path, required=True,
                   help="Public corpus text file for synthetic-pair generation. "
                        "Will be tokenised by --tokenizer-model.")
    p.add_argument("--tokenizer-model", type=str, default="Qwen/Qwen3-4B")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--n-train", type=int, default=128)
    p.add_argument("--n-val", type=int, default=16)
    p.add_argument("--n-test", type=int, default=16)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--device", type=str, default="auto",
                   choices=("auto", "cuda", "cpu"))
    # Attacker-side Algorithm 1 keys (Kerckhoffs: algorithm public).
    p.add_argument("--attacker-expansion", type=int, default=128,
                   help="Algorithm 1 h the attacker uses.")
    p.add_argument("--attacker-alpha-e", type=float, default=1.0,
                   help="Algorithm 1 α_e the attacker uses for training-pair "
                        "synthesis. Should match the deployment's α_e for "
                        "best transfer.")
    p.add_argument("--attacker-tau-seed", type=int, default=99999,
                   help="Seed for the attacker's own τ_a. Different from the "
                        "deployment τ; the inverter must be τ-invariant.")
    p.add_argument("--attacker-noise-seed", type=int, default=88888)
    p.add_argument("--keymat-seed", type=int, default=77777,
                   help="Seed for the attacker's keymat Q̂_a.")
    p.add_argument("--no-run", action="store_true",
                   help="Build the model and synthesise data but skip "
                        "train+eval (smoke-test mode).")
    args = p.parse_args()

    print(f"[seq-IMA] loading plaintext GGUF: {args.plain}")
    plain = load_model(args.plain, "plaintext", embed_only=True)
    print(f"  vocab={plain.vocab_size} d_eff={plain.d_eff}")

    print(f"[seq-IMA] tokenising corpus {args.corpus_file} via {args.tokenizer_model}")
    from transformers import AutoTokenizer  # type: ignore
    tok = AutoTokenizer.from_pretrained(args.tokenizer_model)
    text = args.corpus_file.read_text(encoding="utf-8")
    corpus_tokens = np.asarray(
        tok(text, add_special_tokens=False)["input_ids"], dtype=np.int64
    )
    print(f"  corpus tokens: {corpus_tokens.size}")

    # Sample the attacker's keymat Q̂_a. Real production would call into
    # `vendor/aloepri-py/src/keymat.build_keymat_transform` to use the
    # paper Algorithm 1 construction; here we use a simple Gaussian
    # synthetic for the scaffold smoke test.
    d = plain.d_eff
    d_obs = d + 2 * args.attacker_expansion
    rng = np.random.default_rng(args.keymat_seed)
    Q_attacker = rng.standard_normal((d, d_obs), dtype=np.float64).astype(np.float32)
    Q_attacker /= np.sqrt(d)
    print(f"[seq-IMA] attacker keymat Q̂_a shape={Q_attacker.shape}")

    if args.no_run:
        print("[seq-IMA] --no-run set; building model + a single synth pair to "
              "smoke-test the pipeline, then exiting without training.")
        cfg = SeqIMAConfig(
            d_obs=d_obs, vocab_size=plain.vocab_size,
            n_layers=args.n_layers, n_heads=args.n_heads, seq_len=args.seq_len,
        )
        model = SequenceIMAInverter(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  model params: {n_params:,}")
        # One synth pair.
        rng2 = np.random.default_rng(args.attacker_tau_seed)
        attacker_tau = rng2.permutation(plain.vocab_size).astype(np.int64)
        pair = synthesize_embed_pairs(
            plain,
            attacker_tau=attacker_tau,
            attacker_keymat_Q=Q_attacker,
            attacker_noise_alpha_e=args.attacker_alpha_e,
            attacker_noise_seed=args.attacker_noise_seed,
            corpus_tokens=corpus_tokens,
            n_sequences=1,
            seq_len=args.seq_len,
        )[0]
        print(f"  synth pair: plain_ids shape={pair.plain_ids.shape} "
              f"obf_activations shape={pair.obf_activations.shape}")
        # Forward pass on a single batch of 1.
        X = torch.from_numpy(pair.obf_activations).unsqueeze(0)
        logits = model(X)
        print(f"  forward OK: logits shape={tuple(logits.shape)}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "status": "scaffold_smoke_test_ok",
            "model_params": n_params,
            "config": asdict(cfg),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        }, indent=2))
        print(f"[seq-IMA] wrote → {args.output}")
        return 0

    result = run_sequence_ima_embed_surface(
        plain,
        attacker_keymat_Q=Q_attacker,
        attacker_alpha_e=args.attacker_alpha_e,
        attacker_noise_seed=args.attacker_noise_seed,
        attacker_tau_seed=args.attacker_tau_seed,
        corpus_tokens=corpus_tokens,
        seq_len=args.seq_len,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
    )
    print(f"[seq-IMA] top1={result.ttrsr_top1:.4f} top10={result.ttrsr_top10:.4f} "
          f"risk={result.risk_level}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "sequence_ima_embed": asdict(result),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }, indent=2))
    print(f"[seq-IMA] wrote → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

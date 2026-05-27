"""ArrowMatch attack (Wang et al., USENIX Sec '25) ported to AloePri.

Paper: "Game of Arrows: On the (In-)Security of Weight Obfuscation for
On-Device TEE-Shielded LLM Partition Algorithms"
PDF: https://ziqi-zhang.github.io/files/SEC25-GameOfArrows.pdf
Code: https://github.com/qsxltss/Game-of-Arrows

## What ArrowMatch does

Direction-similarity attack against weight-obfuscation schemes. The
attack exploits two observations:

  Obs1. Fine-tuned model weights M_vic have LOW direction discrepancy
        from their pre-training initialisation M_pre, because
        fine-tuning's weight updates lie in a low-intrinsic-dimensional
        subspace (Aghajanyan et al. 2020) — directions barely change.

  Obs2. Lightweight obfuscation schemes (Π permutation + D diagonal
        scaling, *per-vector* operations) DO NOT change vector
        DIRECTIONS — only matrix-multiplication operations do, and
        these are too expensive for TEEs.

Therefore the attacker recovers the permutation Π by cosine-matching
each obfuscated weight row to its nearest plaintext-direction
counterpart in M_pre. Two stages:

  S1 — Distance-based direction recovery:
      σ(i) = argmin_j cos(w_obf^i, w_pre^j)            (paper Eq. 1)

  S2 — Learning-based length adjustment:
      ŝ_i = ||w_pre^{σ(i)}|| / ||w_obf^i||
      w_init^{σ(i)} = ŝ_i · w_obf^i

## Why ArrowMatch's applicability to AloePri is bounded

ArrowMatch's foundational claim is that obfuscation cannot change
vector direction. AloePri's Algorithm 1 keymat construction is
specifically MATRIX-MULTIPLICATION (the keymat P̂ / Q̂ are non-diagonal
invertible matrices) — exactly the operation ArrowMatch identifies as
"too expensive for TEEs" and therefore absent from the schemes it
attacks. AloePri pays the matrix-multiplication cost OFFLINE (the
obfuscated model is rewritten once), making it immune to ArrowMatch's
core insight in theory.

**Empirically open:** at λ=0.3 the keymat P̂ = [B|C|E]·Z with
B = U+λV (U orthogonal, V random Gaussian) is close-to-orthogonal in
its top-d×d block. The remaining 2h columns are non-trivially mixing.
Whether ArrowMatch's cosine-matching can still partially recover Π
through the residual direction signal is what this driver measures.

Expected outcome on AloePri (prediction, to be measured):
  - τ recovery: significantly lower than current VMA (RowSort) which
    is structure-aware (operates on sorted-quantile invariants robust
    to per-row direction changes).
  - Best-case: ~5-15 % top-1 τ recovery on Q3-4B (well above random
    1/V but well below VMA's 25 %).
  - Length-adjustment Stage 2 is an artefact reconstruction step (paper
    builds W_init for fine-tune-based functional recovery) — for
    AloePri's prompt-privacy threat model it's measurement-only.

## How this script differs from `run_static_attacks.py::run_vma`

VMA uses RowSort + sorted-quantile features (paper §F.1, dimension-
agnostic; robust to small perturbations and dimension expansion).
ArrowMatch uses direct cosine matching (assumes direction preserved).
Different operating points on the same underlying problem.

## Threat-model framing for AloePri

ArrowMatch was originally a *model-piracy* attack (device user wants
to steal the proprietary M_vic weights). In AloePri's setting,
weights aren't secret (W̃ is published). What recovering σ buys the
attacker here is the **τ permutation** — the same target as VMA.
Recovering τ enables wire-prompt decoding for every request. So
in our threat model ArrowMatch is a τ-recovery attack, not a
weight-piracy attack.

## Output

JSON compatible with `run_static_attacks.py` schema. Reports:
  - per-tensor (token_embd primarily; also lm_head, attn_q, etc.)
    top-1 / top-K τ recovery rate.
  - length-adjustment residual (Stage 2 — how close are reconstructed
    lengths to ground-truth).
  - runtime + chunk size.

## Implementation notes

  - V × V pairwise cosine matrix is ~23 GB at V=151 K — must chunk.
  - GPU acceleration via torch is supported; fall back to CPU.
  - Both plain and obfuscated weights must have the same dim_in on the
    matched axis. token_embd: plain (V, d), obf (V, d+2h). Direct
    cosine matching needs same d. We project obfuscated to plain dim
    by truncation (first d cols) or pad plain with zeros to (d+2h);
    truncation matches AloePri's Algorithm 1 structure (the extra 2h
    cols are mixed but the first d carry most of the signal).

  - For non-embedding tensors (W_q, W_k, etc.) the relevant axis is
    head_dim or hidden_size; per-row cosine still works.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_gguf_weights import ModelWeights, load_model  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attack_drivers.common import AttackResult, classify_risk_level  # type: ignore


# ───── τ loader (shared convention with run_ima_embedrow_attacks) ──


def load_tau(key_path: Path) -> tuple[np.ndarray, int]:
    """Load τ + active_size from the obfuscator's .key.npz."""
    z = np.load(key_path, allow_pickle=False)
    tau = z["tau"].astype(np.int64)
    active_size = int(z["active_size"])
    return tau, active_size


# ───── ArrowMatch Stage 1: distance-based direction recovery ───────


def _cosine_match_chunked(
    W_obf: torch.Tensor,
    W_pre: torch.Tensor,
    *,
    chunk_obf: int = 1024,
    chunk_pre: int = 32768,
    topk: int = 10,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each row of `W_obf`, find the top-K most-cosine-similar
    rows of `W_pre`. Returns `(indices, scores)` of shape
    `(W_obf.shape[0], topk)`.

    Chunked over both axes so the V × V pairwise matrix never
    materialises. With V=151 K, d=2560, chunk_obf=1024,
    chunk_pre=32768: peak memory ≈ 32 K × 1024 × 4 bytes = 128 MB
    for one chunk's cosine slice. Fits easily on iGPU.
    """
    n_obf = W_obf.shape[0]
    n_pre = W_pre.shape[0]
    # L2-normalise once.
    W_obf_n = torch.nn.functional.normalize(W_obf.to(device), dim=1)
    W_pre_n = torch.nn.functional.normalize(W_pre.to(device), dim=1)

    best_idx = torch.full((n_obf, topk), -1, dtype=torch.long, device=device)
    best_val = torch.full((n_obf, topk), -float("inf"), device=device)

    for i_start in range(0, n_obf, chunk_obf):
        i_end = min(i_start + chunk_obf, n_obf)
        Q = W_obf_n[i_start:i_end]  # (chunk_obf, d)
        # Maintain running top-K across pre-chunks.
        running_idx = torch.full((Q.shape[0], topk), -1, dtype=torch.long, device=device)
        running_val = torch.full((Q.shape[0], topk), -float("inf"), device=device)
        for j_start in range(0, n_pre, chunk_pre):
            j_end = min(j_start + chunk_pre, n_pre)
            K = W_pre_n[j_start:j_end]  # (chunk_pre, d)
            sim = Q @ K.T  # (chunk_obf, chunk_pre)
            # Merge with running top-K.
            merged_val = torch.cat([running_val, sim], dim=1)
            merged_idx = torch.cat([
                running_idx,
                torch.arange(j_start, j_end, device=device).expand(Q.shape[0], -1),
            ], dim=1)
            top_val, top_pos = merged_val.topk(topk, dim=1)
            running_val = top_val
            running_idx = merged_idx.gather(1, top_pos)
        best_idx[i_start:i_end] = running_idx
        best_val[i_start:i_end] = running_val
    return best_idx, best_val


def arrowmatch_stage1(
    W_obf: np.ndarray,
    W_pre: np.ndarray,
    tau: np.ndarray,
    *,
    active_size: int,
    topk: int = 10,
    chunk_obf: int = 1024,
    chunk_pre: int = 32768,
    device: str = "auto",
    align_dims: str = "truncate",
) -> dict[str, Any]:
    """Run ArrowMatch S1 direction-distance recovery + score against τ.

    Args:
        W_obf: obfuscated weight matrix shape (V, d_obs). The row index
            i is the obfuscated token id (= τ[plain_id]).
        W_pre: plaintext / public weight matrix shape (V, d_plain).
        tau: ground-truth permutation. `tau[plain_id] = obf_id`.
        active_size: only positions [0, active_size) of τ are permuted.
        align_dims: how to align d_plain vs d_obs.
            'truncate': use first min(d_plain, d_obs) cols of both.
            'pad': zero-pad the shorter to the longer.

    Returns:
        dict with keys: top1_recovery, topk_recovery, n_active,
        per_chunk_runtime, predicted_tau (np.ndarray of obf_id -> recovered_plain_id),
        cosine_at_correct (np.ndarray of cosine sim at the ground-truth pair).
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Align dimensions.
    d_pre = W_pre.shape[1]
    d_obs = W_obf.shape[1]
    if align_dims == "truncate":
        d = min(d_pre, d_obs)
        W_obf_aligned = W_obf[:, :d]
        W_pre_aligned = W_pre[:, :d]
    elif align_dims == "pad":
        d = max(d_pre, d_obs)
        W_obf_aligned = np.zeros((W_obf.shape[0], d), dtype=np.float32)
        W_obf_aligned[:, :d_obs] = W_obf
        W_pre_aligned = np.zeros((W_pre.shape[0], d), dtype=np.float32)
        W_pre_aligned[:, :d_pre] = W_pre
    else:
        raise ValueError(f"unknown align_dims: {align_dims!r}")

    t0 = time.perf_counter()
    W_obf_t = torch.from_numpy(W_obf_aligned.astype(np.float32))
    W_pre_t = torch.from_numpy(W_pre_aligned.astype(np.float32))
    best_idx, best_val = _cosine_match_chunked(
        W_obf_t, W_pre_t,
        chunk_obf=chunk_obf, chunk_pre=chunk_pre, topk=topk, device=device,
    )
    runtime_s = time.perf_counter() - t0
    best_idx_np = best_idx.cpu().numpy()  # (V, topk)
    best_val_np = best_val.cpu().numpy()

    # σ̂(obf_id) = best_idx[obf_id, 0] is the recovered plain id.
    # Ground truth: τ[plain_id] = obf_id, so τ⁻¹[obf_id] = plain_id.
    tau_inv = np.argsort(tau).astype(np.int64)  # τ⁻¹

    # Compare ONLY positions in the active range (Π only permutes
    # [0, active_size); special tokens stay identity).
    n_active = min(active_size, W_obf.shape[0])
    obf_in_active = np.arange(n_active, dtype=np.int64)
    # The obfuscated id at position obf_in_active[i] corresponds to
    # plain id tau_inv[obf_in_active[i]]. But the SLOT-INDEX in
    # W_obf is the obfuscated id directly (because W̃[τ[i]] is at row
    # τ[i]). So for each obf_id k ∈ [0, active_size), the true plain
    # id is tau_inv[k].
    true_plain = tau_inv[obf_in_active]  # (n_active,)
    pred_plain_top1 = best_idx_np[obf_in_active, 0]  # (n_active,)
    pred_plain_topk = best_idx_np[obf_in_active]  # (n_active, topk)

    top1 = float((pred_plain_top1 == true_plain).mean())
    topk_hits = (pred_plain_topk == true_plain[:, None]).any(axis=1)
    topk_rate = float(topk_hits.mean())

    # Cosine at the ground-truth (plain, obf) pair — diagnostic.
    # Re-compute to avoid storing the full pairwise matrix.
    W_obf_n = torch.nn.functional.normalize(W_obf_t.to(device), dim=1)
    W_pre_n = torch.nn.functional.normalize(W_pre_t.to(device), dim=1)
    paired_cosine = (W_obf_n[obf_in_active] * W_pre_n[true_plain]).sum(dim=1)
    cosine_at_correct = paired_cosine.cpu().numpy()

    return {
        "stage1_top1_recovery": top1,
        "stage1_topk_recovery": topk_rate,
        "stage1_topk": topk,
        "n_active": n_active,
        "runtime_s": round(runtime_s, 2),
        "predicted_tau_inverse_top1": pred_plain_top1.astype(np.int64),
        "cosine_at_correct_mean": float(cosine_at_correct.mean()),
        "cosine_at_correct_std": float(cosine_at_correct.std()),
        "cosine_at_correct_p10": float(np.percentile(cosine_at_correct, 10)),
        "cosine_at_correct_p90": float(np.percentile(cosine_at_correct, 90)),
        "device": device,
        "align_dims": align_dims,
        "d_aligned": int(W_obf_aligned.shape[1]),
    }


# ───── ArrowMatch Stage 2: learning-based length adjustment ────────


def arrowmatch_stage2(
    W_obf: np.ndarray,
    W_pre: np.ndarray,
    predicted_sigma_inv: np.ndarray,
    *,
    align_dims: str = "truncate",
) -> dict[str, Any]:
    """Run ArrowMatch S2 length adjustment.

    Args:
        W_obf: obfuscated weights (V, d_obs).
        W_pre: plaintext weights (V, d_plain).
        predicted_sigma_inv: σ̂(obf_id) -> recovered plain_id (the output
            of stage1's top-1). Same length as W_obf rows.

    Returns: dict with the reconstructed W_init and a residual metric
        ||W_init - W_pre[σ̂(·)]||_F as the reconstruction error.
    """
    d_pre = W_pre.shape[1]
    d_obs = W_obf.shape[1]
    if align_dims == "truncate":
        d = min(d_pre, d_obs)
        W_obf_a = W_obf[:, :d]
        W_pre_a = W_pre[:, :d]
    else:
        d = max(d_pre, d_obs)
        W_obf_a = np.zeros((W_obf.shape[0], d), dtype=np.float32)
        W_obf_a[:, :d_obs] = W_obf
        W_pre_a = np.zeros((W_pre.shape[0], d), dtype=np.float32)
        W_pre_a[:, :d_pre] = W_pre

    pre_norm = np.linalg.norm(W_pre_a[predicted_sigma_inv], axis=1)  # (V,)
    obf_norm = np.linalg.norm(W_obf_a, axis=1)                       # (V,)
    eps = 1e-8
    s_hat = pre_norm / (obf_norm + eps)
    W_init = W_obf_a * s_hat[:, None]
    target = W_pre_a[predicted_sigma_inv]
    residual_fro = float(np.linalg.norm(W_init - target))
    target_fro = float(np.linalg.norm(target))
    rel_residual = residual_fro / max(target_fro, eps)
    return {
        "stage2_s_hat_mean": float(s_hat.mean()),
        "stage2_s_hat_std": float(s_hat.std()),
        "stage2_reconstruction_residual_fro": residual_fro,
        "stage2_reconstruction_residual_relative": rel_residual,
    }


# ───── Top-level attack against a chosen tensor ────────────────────


def run_arrowmatch_on_tensor(
    plain: ModelWeights,
    obfuscated: ModelWeights,
    tau: np.ndarray,
    *,
    tensor_name: str = "token_embd",
    active_size: int,
    topk: int = 10,
    chunk_obf: int = 1024,
    chunk_pre: int = 32768,
    device: str = "auto",
    align_dims: str = "truncate",
    skip_stage2: bool = False,
) -> AttackResult:
    """Run ArrowMatch S1 (+ optional S2) on a chosen tensor name.

    Currently supports `token_embd` (the embedding table — what AloePri's
    Π permutes). Future: extend to per-layer attn_q / attn_k / ffn_*
    rows for cross-layer voting.
    """
    if tensor_name != "token_embd":
        raise NotImplementedError(
            f"ArrowMatch currently only supports tensor_name='token_embd' "
            f"(got {tensor_name!r}). Per-layer attn_q/attn_k/ffn_* extension "
            f"is on the to-do list."
        )
    W_pre = plain.token_embd
    W_obf = obfuscated.token_embd
    s1 = arrowmatch_stage1(
        W_obf=W_obf, W_pre=W_pre, tau=tau,
        active_size=active_size, topk=topk,
        chunk_obf=chunk_obf, chunk_pre=chunk_pre,
        device=device, align_dims=align_dims,
    )
    extra: dict[str, Any] = {k: v for k, v in s1.items() if k != "predicted_tau_inverse_top1"}
    if not skip_stage2:
        s2 = arrowmatch_stage2(
            W_obf=W_obf, W_pre=W_pre,
            predicted_sigma_inv=s1["predicted_tau_inverse_top1"],
            align_dims=align_dims,
        )
        extra.update(s2)
    return AttackResult(
        attack="arrowmatch",
        condition="obfuscated",
        model_id=str(obfuscated.path.name),
        n_prompts=0,
        n_train=0,
        n_test=int(s1["n_active"]),
        ttrsr_top1=float(s1["stage1_top1_recovery"]),
        ttrsr_top10=float(s1["stage1_topk_recovery"]),
        risk_level=classify_risk_level(float(s1["stage1_top1_recovery"])),
        extra=extra,
    )


# ───── CLI ─────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description="ArrowMatch attack (Wang et al., USENIX Sec '25) ported to AloePri"
    )
    p.add_argument("--plain", type=Path, required=True)
    p.add_argument("--obfuscated", type=Path, required=True)
    p.add_argument(
        "--key", type=Path,
        help=".key.npz with τ from obfuscate_qwen3_gguf.py",
    )
    p.add_argument(
        "--identity-tau", action="store_true",
        help="Use τ = identity (plain-side control). Attack should "
             "succeed at ~100%% since the bijection is trivial.",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tensor", type=str, default="token_embd",
                   choices=("token_embd",))
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--chunk-obf", type=int, default=1024)
    p.add_argument("--chunk-pre", type=int, default=32768)
    p.add_argument("--device", type=str, default="auto",
                   choices=("auto", "cuda", "cpu"))
    p.add_argument("--align-dims", type=str, default="truncate",
                   choices=("truncate", "pad"))
    p.add_argument("--skip-stage2", action="store_true",
                   help="Skip Stage 2 length adjustment (S1 only).")
    from m2_7_common import add_min_mem_args, check_phase_memory  # type: ignore
    add_min_mem_args(p, phase="arrowmatch")
    args = p.parse_args()
    check_phase_memory("arrowmatch", args.min_mem_gb, args.skip_mem_check)

    print(f"[ArrowMatch] loading plaintext {args.plain}")
    plain = load_model(args.plain, "plaintext", embed_only=True)
    print(f"  vocab={plain.vocab_size} d_eff={plain.d_eff}")

    print(f"[ArrowMatch] loading obfuscated {args.obfuscated}")
    obfuscated = load_model(args.obfuscated, "obfuscated", embed_only=True)
    print(f"  vocab={obfuscated.vocab_size} d_eff={obfuscated.d_eff}")

    if args.identity_tau:
        active_size = plain.vocab_size
        tau = np.arange(plain.vocab_size, dtype=np.int64)
        print(f"[ArrowMatch] τ = identity (control); active_size={active_size}")
    else:
        if args.key is None:
            raise SystemExit("--key is required unless --identity-tau")
        tau, active_size = load_tau(args.key)
        print(f"[ArrowMatch] τ active_size={active_size}")

    result = run_arrowmatch_on_tensor(
        plain, obfuscated, tau,
        tensor_name=args.tensor,
        active_size=active_size,
        topk=args.topk,
        chunk_obf=args.chunk_obf,
        chunk_pre=args.chunk_pre,
        device=args.device,
        align_dims=args.align_dims,
        skip_stage2=args.skip_stage2,
    )
    print(
        f"[ArrowMatch] tensor={args.tensor} S1 top1={result.ttrsr_top1:.4f} "
        f"top{args.topk}={result.ttrsr_top10:.4f} risk={result.risk_level} "
        f"runtime={result.extra['runtime_s']}s "
        f"cosine_correct_mean={result.extra['cosine_at_correct_mean']:.4f}"
    )
    if not args.skip_stage2:
        print(
            f"  S2 ŝ mean={result.extra['stage2_s_hat_mean']:.4f} "
            f"reconstruction_rel_residual={result.extra['stage2_reconstruction_residual_relative']:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "arrowmatch": asdict(result),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }, indent=2))
    print(f"[ArrowMatch] wrote → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

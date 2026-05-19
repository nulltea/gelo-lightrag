"""Run AloePri's static-weight attacks against the §05 obfuscated GGUF.

Attacks covered here (no inference needed):

* **VMA** — Vocabulary-Matching Attack via RowSort + sorted-quantile
  features (paper §F.1 Table 8 / AloePri reference
  `vma.py::_sorted_quantile_features` at line 305).
* **IA Gate-IA** — exploits `Avg(eW_gate) = Avg(ẽW̃_gate)` invariant.
* **IA Attn-IA** — exploits the block quadratic form
  `e(QᵀQ)⁻¹eᵀ = ẽ(Q̃ᵀQ̃)⁻¹ẽᵀ`.

The attacker assumption (paper §F.1 / Table 1 caption): adversary
has both the plaintext weights θ and the obfuscated weights θ̃. We
load both GGUFs via `extract_gguf_weights` and run the attacks.

Writes per-attack JSON to `--output`, schema-compatible with
`evals/aloepri-attacks/results/path-1-attacks.json` so the doc can
ingest it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

# Local import — extractor lives next to this script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_gguf_weights import ModelWeights, load_model

# Reuse the AttackResult dataclass + risk classifier from the harness.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attack_drivers.common import AttackResult, classify_risk_level  # type: ignore


# ───── VMA: sorted-quantile RowSort matching ────────────────────────


def _sorted_quantile_features(matrix: np.ndarray, bins: int = 64) -> np.ndarray:
    """Port of AloePri vma.py:305-313. Returns (rows, bins) of sorted
    row quantiles — dimension-agnostic, so plaintext (cols=2048) and
    obfuscated (cols=2304) yield comparable signatures.
    """
    sorted_rows = np.sort(matrix.astype(np.float32), axis=1)
    positions = np.linspace(0, sorted_rows.shape[1] - 1, num=bins).astype(np.int64)
    return sorted_rows[:, positions]


def _rowwise_standardize(score_matrix: np.ndarray) -> np.ndarray:
    """Z-score each row so the dot-product step in voting is scale-free."""
    mean = score_matrix.mean(axis=1, keepdims=True)
    std = score_matrix.std(axis=1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (score_matrix - mean) / std


def _topk_match(plain_features: np.ndarray, obs_features: np.ndarray, topk: int) -> np.ndarray:
    """For each plain row i, return the indices of the topk closest
    rows in `obs_features` by cosine similarity.
    """
    p = plain_features / np.maximum(
        np.linalg.norm(plain_features, axis=1, keepdims=True), 1e-8
    )
    o = obs_features / np.maximum(
        np.linalg.norm(obs_features, axis=1, keepdims=True), 1e-8
    )
    scores = p @ o.T
    return np.argpartition(-scores, kth=topk, axis=1)[:, :topk]


def run_vma(
    plain: ModelWeights,
    obfuscated: ModelWeights,
    *,
    eval_size: int = 256,
    candidate_pool_size: int = 4096,
    topk: int = 10,
    bins: int = 64,
    projection_kinds: tuple[str, ...] = ("attn_q", "attn_k", "attn_v", "ffn_gate", "ffn_up"),
    layer_indices: tuple[int, ...] = (0, 11, 23),
    seed: int = 20260518,
) -> AttackResult:
    """RowSort + sorted-quantile + voting. Two source families:

    * **direct**: `W_embed @ W_head^T` — the simplest paper case (Table 8 row 1).
    * **projection-derived**: for each (layer, kind) in `layer_indices ×
      projection_kinds`, compute `W_embed @ W_proj` on both sides,
      extract sorted-quantile features, and vote.

    Voting: per-source we get a (eval_size, candidate_pool_size)
    similarity matrix; we standardise rows, sum across sources, take
    top-k.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    vocab = plain.vocab_size
    eval_size = min(eval_size, vocab)
    candidate_pool_size = min(candidate_pool_size, vocab)
    eval_ids = rng.choice(vocab, size=eval_size, replace=False)
    # Ensure ground truth always reachable in candidate pool.
    pool_ids = rng.choice(vocab, size=candidate_pool_size, replace=False)
    pool_ids = np.unique(np.concatenate([pool_ids, eval_ids]))

    plain_W_e = plain.token_embd  # (vocab, d_plain)
    obs_W_e = obfuscated.token_embd  # (vocab, d_obs)

    # Subset W_embed rows ONCE so per-source matmuls stay cheap.
    # Materialising (vocab, vocab) or (vocab, intermediate) would
    # cost 50–100 GB at f32 — we just need (eval_size, out) and
    # (pool_size, out).
    plain_W_e_eval = plain_W_e[eval_ids]   # (eval_size, d_plain)
    obs_W_e_pool = obs_W_e[pool_ids]       # (pool_size, d_obs)

    # Collect source-row pairs (plain_eval_rows, obs_pool_rows, name).
    source_pairs: list[tuple[np.ndarray, np.ndarray, str]] = []

    # Direct source 1: W_embed rows themselves.
    source_pairs.append((plain_W_e_eval, obs_W_e_pool, "embed"))

    # Direct source 2: W_output rows (also vocab-indexed).
    plain_out_eval = plain.output[eval_ids]
    obs_out_pool = obfuscated.output[pool_ids]
    source_pairs.append((plain_out_eval, obs_out_pool, "output"))

    # Projection-derived sources (paper Table 8): W_embed @ W_proj^T.
    # We compute this on the eval/pool row subset only.
    for li in layer_indices:
        if li >= plain.n_layers or li >= obfuscated.n_layers:
            continue
        for kind in projection_kinds:
            plain_W = plain.per_layer[li].get(kind)
            obs_W = obfuscated.per_layer[li].get(kind)
            if plain_W is None or obs_W is None:
                continue
            # nn.Linear weights are (out, in); we want
            # rows of W_embed projected onto the kind output:
            # (eval_size, d) @ (d, out) = (eval_size, out).
            plain_src = plain_W_e_eval @ plain_W.T  # (eval_size, out)
            obs_src = obs_W_e_pool @ obs_W.T        # (pool_size, out)
            source_pairs.append((plain_src, obs_src, f"l{li}_{kind}"))

    # Per-source quantile features + cosine similarity + voting.
    n_sources = len(source_pairs)
    accum = np.zeros((eval_size, pool_ids.shape[0]), dtype=np.float32)
    for plain_src, obs_src, _ in source_pairs:
        plain_eval = _sorted_quantile_features(plain_src, bins=bins)
        obs_pool = _sorted_quantile_features(obs_src, bins=bins)
        # Cosine similarity matrix (eval_size × pool_size).
        pn = plain_eval / np.maximum(np.linalg.norm(plain_eval, axis=1, keepdims=True), 1e-8)
        on = obs_pool / np.maximum(np.linalg.norm(obs_pool, axis=1, keepdims=True), 1e-8)
        sim = pn @ on.T
        accum += _rowwise_standardize(sim)

    # Top-k voting result.
    k = min(topk, accum.shape[1])
    top_idx = np.argpartition(-accum, kth=k - 1, axis=1)[:, :k]
    # Resolve to pool_ids and check ground-truth hits.
    predicted = pool_ids[top_idx]
    truth = eval_ids[:, None]
    top1 = float((predicted[:, 0] == eval_ids).mean())
    top10 = float((predicted[:, :min(10, k)] == truth).any(axis=1).mean())

    return AttackResult(
        attack="vma",
        condition="obfuscated",
        model_id=str(obfuscated.path.name),
        n_prompts=0,
        n_train=0,
        n_test=eval_size,
        ttrsr_top1=top1,
        ttrsr_top10=top10,
        risk_level=classify_risk_level(top1),
        extra={
            "matching_strategy": "row_sort_quantile_voting",
            "n_sources": n_sources,
            "feature_bins": bins,
            "projection_kinds": list(projection_kinds),
            "layer_indices": list(layer_indices),
            "candidate_pool_size": int(pool_ids.shape[0]),
            "runtime_seconds": round(time.perf_counter() - t0, 2),
        },
    )


# ───── IA: Gate-IA + Attn-IA invariants ─────────────────────────────


def _gate_ia_invariants(W_e: np.ndarray, W_gate: np.ndarray) -> np.ndarray:
    """Paper §F.1: `Avg(e_i · W_gate)` per token row.

    `W_e` is `(vocab, d)`; `W_gate` is stored nn.Linear-style as
    `(intermediate, d)` so the matmul we want is `W_e @ W_gate.T`
    yielding `(vocab, intermediate)` — then mean across intermediate.
    """
    projected = W_e.astype(np.float32) @ W_gate.astype(np.float32).T
    return projected.mean(axis=1)


def _attn_ia_invariants(
    W_e: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, block_size: int = 16
) -> np.ndarray:
    """Paper §F.1: `e_i (QᵀQ)⁻¹ e_iᵀ` per block of the head dim.

    `W_q` / `W_k` are `(q_dim, d)` / `(kv_dim, d)`. Q = W_e @ W_q.T
    has shape `(vocab, q_dim)`. We split q_dim into blocks of
    `block_size` and for each block compute `||e_i · M||²` where M
    stacks the block of Wq and the analogous Wk columns.
    """
    we = W_e.astype(np.float32)
    wq = W_q.astype(np.float32)
    wk = W_k.astype(np.float32)
    q_dim = wq.shape[0]
    kv_dim = wk.shape[0]
    head_dim = min(q_dim, kv_dim)
    n_blocks = max(head_dim // block_size, 1)
    invariants = np.zeros((we.shape[0], n_blocks), dtype=np.float32)
    for b in range(n_blocks):
        s = b * block_size
        e = min(s + block_size, head_dim)
        m_q = wq[s:e, :].T  # (d, block_width)
        m_k = wk[s:e, :].T
        m = np.concatenate([m_q, m_k], axis=1)  # (d, 2*block_width)
        proj = we @ m  # (vocab, 2*block_width)
        invariants[:, b] = (proj * proj).sum(axis=1)
    return invariants


def _nearest_match_1d(plain_vec: np.ndarray, obs_vec: np.ndarray, topk: int) -> np.ndarray:
    """For each plain[i], rank obs[j] by abs(plain[i] - obs[j])."""
    diff = np.abs(plain_vec[:, None] - obs_vec[None, :])  # (eval, pool)
    return np.argpartition(diff, kth=topk - 1, axis=1)[:, :topk]


def _nearest_match_nd(
    plain_mat: np.ndarray, obs_mat: np.ndarray, topk: int
) -> np.ndarray:
    """For each plain[i] (vec), rank obs[j] (vec) by L2 distance."""
    pp = (plain_mat * plain_mat).sum(axis=1, keepdims=True)
    oo = (obs_mat * obs_mat).sum(axis=1, keepdims=True)
    dist2 = pp - 2.0 * plain_mat @ obs_mat.T + oo.T
    return np.argpartition(dist2, kth=topk - 1, axis=1)[:, :topk]


def run_ia(
    plain: ModelWeights,
    obfuscated: ModelWeights,
    *,
    layer: int = 0,
    eval_size: int = 4096,
    candidate_pool_size: int = 8192,
    topk: int = 10,
    seed: int = 20260518,
) -> AttackResult:
    """Compute Gate-IA + Attn-IA invariants on both models, match by
    nearest neighbour, report TTRSR.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    vocab = plain.vocab_size
    eval_size = min(eval_size, vocab)
    candidate_pool_size = min(candidate_pool_size, vocab)
    eval_ids = rng.choice(vocab, size=eval_size, replace=False)
    pool_ids = rng.choice(vocab, size=candidate_pool_size, replace=False)
    pool_ids = np.unique(np.concatenate([pool_ids, eval_ids]))

    plain_W_e = plain.token_embd
    obs_W_e = obfuscated.token_embd
    plain_gate = plain.per_layer[layer]["ffn_gate"]
    obs_gate = obfuscated.per_layer[layer]["ffn_gate"]
    plain_q = plain.per_layer[layer]["attn_q"]
    plain_k = plain.per_layer[layer]["attn_k"]
    obs_q = obfuscated.per_layer[layer]["attn_q"]
    obs_k = obfuscated.per_layer[layer]["attn_k"]

    # Gate-IA invariants per row.
    plain_gate_inv_all = _gate_ia_invariants(plain_W_e, plain_gate)
    obs_gate_inv_all = _gate_ia_invariants(obs_W_e, obs_gate)
    plain_gate_inv = plain_gate_inv_all[eval_ids]
    obs_gate_inv = obs_gate_inv_all[pool_ids]
    gate_top = _nearest_match_1d(plain_gate_inv, obs_gate_inv, topk)
    gate_predicted = pool_ids[gate_top]
    gate_top1 = float((gate_predicted[:, 0] == eval_ids).mean())
    gate_top10 = float(
        (gate_predicted[:, :min(10, gate_top.shape[1])] == eval_ids[:, None]).any(axis=1).mean()
    )

    # Attn-IA invariants.
    plain_attn_inv_all = _attn_ia_invariants(plain_W_e, plain_q, plain_k)
    obs_attn_inv_all = _attn_ia_invariants(obs_W_e, obs_q, obs_k)
    plain_attn_inv = plain_attn_inv_all[eval_ids]
    obs_attn_inv = obs_attn_inv_all[pool_ids]
    attn_top = _nearest_match_nd(plain_attn_inv, obs_attn_inv, topk)
    attn_predicted = pool_ids[attn_top]
    attn_top1 = float((attn_predicted[:, 0] == eval_ids).mean())
    attn_top10 = float(
        (attn_predicted[:, :min(10, attn_top.shape[1])] == eval_ids[:, None]).any(axis=1).mean()
    )

    # Headline TTRSR: max of the two variants (an attacker uses
    # whichever wins). Paper Table 1's IA column is also the max.
    top1 = max(gate_top1, attn_top1)
    top10 = max(gate_top10, attn_top10)

    return AttackResult(
        attack="ia",
        condition="obfuscated",
        model_id=str(obfuscated.path.name),
        n_prompts=0,
        n_train=0,
        n_test=eval_size,
        ttrsr_top1=top1,
        ttrsr_top10=top10,
        risk_level=classify_risk_level(top1),
        extra={
            "gate_ia_top1": gate_top1,
            "gate_ia_top10": gate_top10,
            "attn_ia_top1": attn_top1,
            "attn_ia_top10": attn_top10,
            "layer": layer,
            "candidate_pool_size": int(pool_ids.shape[0]),
            "runtime_seconds": round(time.perf_counter() - t0, 2),
        },
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Run static-weight attacks against the §05 obfuscated GGUF")
    p.add_argument("--plain", type=Path, required=True, help="Path to plaintext Qwen3 GGUF")
    p.add_argument("--obfuscated", type=Path, required=True, help="Path to §05 obfuscated GGUF")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--vma-eval-size", type=int, default=256)
    p.add_argument("--vma-pool-size", type=int, default=4096)
    p.add_argument("--ia-eval-size", type=int, default=4096)
    p.add_argument("--ia-pool-size", type=int, default=8192)
    from m2_7_common import add_min_mem_args, check_phase_memory  # type: ignore
    add_min_mem_args(p, phase="static_attacks")
    args = p.parse_args()

    # Pre-flight: both GGUFs together need ~22 GB of working RAM. Refuse
    # to start if we don't have headroom — the post-OOM lesson from path-1.
    check_phase_memory("static_attacks", args.min_mem_gb, args.skip_mem_check)

    print("[M2.7] loading plaintext GGUF…")
    t0 = time.perf_counter()
    plain = load_model(args.plain, "plaintext")
    print(f"  loaded in {time.perf_counter() - t0:.1f} s — "
          f"vocab={plain.vocab_size} d_eff={plain.d_eff} n_layers={plain.n_layers}")

    print("[M2.7] loading obfuscated GGUF…")
    t0 = time.perf_counter()
    obfuscated = load_model(args.obfuscated, "obfuscated")
    print(f"  loaded in {time.perf_counter() - t0:.1f} s — "
          f"vocab={obfuscated.vocab_size} d_eff={obfuscated.d_eff} n_layers={obfuscated.n_layers}")

    print("[M2.7] running VMA…")
    vma = run_vma(
        plain,
        obfuscated,
        eval_size=args.vma_eval_size,
        candidate_pool_size=args.vma_pool_size,
    )
    print(f"  vma top1={vma.ttrsr_top1:.4f} top10={vma.ttrsr_top10:.4f} risk={vma.risk_level}")

    print("[M2.7] running IA (Gate-IA + Attn-IA)…")
    ia = run_ia(
        plain,
        obfuscated,
        eval_size=args.ia_eval_size,
        candidate_pool_size=args.ia_pool_size,
    )
    print(f"  ia top1={ia.ttrsr_top1:.4f} (gate={ia.extra['gate_ia_top1']:.4f} "
          f"attn={ia.extra['attn_ia_top1']:.4f}) risk={ia.risk_level}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "format": "aloepri_m2_7_static_v1",
        "plain_path": str(args.plain),
        "obfuscated_path": str(args.obfuscated),
        "attacks": {
            "vma": vma.to_dict(),
            "ia": ia.to_dict(),
        },
    }
    args.output.write_text(json.dumps(results, indent=2))
    print(f"[M2.7] wrote → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

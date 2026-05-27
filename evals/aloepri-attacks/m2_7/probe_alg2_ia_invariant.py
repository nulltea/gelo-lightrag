"""Synthetic IA Attn-IA invariant probe — per-Alg2-component defense isolation.

For each Algorithm 2 component (R̂_qk, Ĥ_qk Walsh-Hadamard ±1, Ĥ_qk non-unit
scale, Ẑ_block at varying β, Π_head, Û_vo), apply it in isolation to random
W_q / W_k / W_v / W_o tensors of deployment shape, recompute the IA Attn-IA
invariant `‖e_i · [W_q[block] | W_k[block]]‖²` over a random vocab, and measure
how nearest-neighbour matching (plain vs obfuscated invariants, τ=identity)
degrades.

Components that leave the invariant per-row unchanged contribute zero defense
against IA Attn-IA. Components that shift it row-dependently break matching.

Run from repo root:
    PYTHONPATH=python/aloepri-llm:evals/aloepri-attacks/m2_7 \
        python3 evals/aloepri-attacks/m2_7/probe_alg2_ia_invariant.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python" / "aloepri-llm"))
sys.path.insert(0, str(REPO / "evals" / "aloepri-attacks" / "m2_7"))

from lib import alg2  # noqa: E402
from run_static_attacks import _attn_ia_invariants, _nearest_match_nd  # noqa: E402


# ── deployment shapes (Q3-4B) ─────────────────────────────────────────
D = 2560
N_Q_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
Q_DIM = N_Q_HEADS * HEAD_DIM   # 4096
KV_DIM = N_KV_HEADS * HEAD_DIM  # 1024
NUM_GROUPS = N_Q_HEADS // N_KV_HEADS  # 4

BLOCK_SIZE = 16  # matches `_attn_ia_invariants` default
VOCAB_POOL = 4096
VOCAB_EVAL = 512
SEED = 20260525


def apply_intra_head_per_head(
    weight_natural: np.ndarray, M_per_head: np.ndarray, n_heads: int, head_dim: int
) -> np.ndarray:
    """Apply per-head head_dim-axis transform M to W of natural shape (n_heads*head_dim, d).

    Equivalent in paper convention to `W̃ = W · M` (M on right). In numpy
    natural shape `apply_qkv_output_transform` does `M.T @ W`. We repeat that
    block-diagonally across heads.
    """
    W_h = weight_natural.reshape(n_heads, head_dim, -1)  # (n_heads, head_dim, d_in)
    W_h_out = np.einsum("ij,hjk->hik", M_per_head.T.astype(np.float32), W_h)
    return W_h_out.reshape(n_heads * head_dim, -1)


def apply_head_perm(
    weight_natural: np.ndarray, tau: np.ndarray, n_heads: int, head_dim: int
) -> np.ndarray:
    """Permute the head axis (head-major reshape) by tau."""
    W_h = weight_natural.reshape(n_heads, head_dim, -1)
    return W_h[tau].reshape(n_heads * head_dim, -1)


def match_topk(plain_eval: np.ndarray, obs_pool: np.ndarray, eval_ids: np.ndarray, topk: int = 10):
    """Returns (top1, topk) hit rates: did plain eval row i find obs pool id i?"""
    idx = _nearest_match_nd(plain_eval, obs_pool, topk=topk)
    top1 = (idx[:, 0] == eval_ids).mean()
    topk_hit = np.any(idx == eval_ids[:, None], axis=1).mean()
    return float(top1), float(topk_hit)


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"## Synthetic IA Attn-IA invariant probe — d={D}, q={Q_DIM}, kv={KV_DIM}, head_dim={HEAD_DIM}, block_size={BLOCK_SIZE}")
    print(f"## vocab_eval={VOCAB_EVAL}, vocab_pool={VOCAB_POOL}, τ=identity, no Alg1 keymat, no noise.")
    print()

    W_e = rng.standard_normal((VOCAB_POOL, D)).astype(np.float32)
    W_q_plain = rng.standard_normal((Q_DIM, D)).astype(np.float32)
    W_k_plain = rng.standard_normal((KV_DIM, D)).astype(np.float32)
    W_v_plain = rng.standard_normal((KV_DIM, D)).astype(np.float32)

    eval_ids = np.arange(VOCAB_EVAL)
    plain_inv = _attn_ia_invariants(W_e, W_q_plain, W_k_plain, block_size=BLOCK_SIZE)
    plain_eval = plain_inv[eval_ids]
    n_blocks = plain_inv.shape[1]
    print(f"   invariant shape per row: (n_blocks={n_blocks},)")
    print()

    # Reusable Alg2 keys (same seed → reproducible)
    layer_seed = SEED + 1
    full_keys = alg2.build_layer_keys(
        head_dim=HEAD_DIM, num_kv_heads=N_KV_HEADS, num_groups=NUM_GROUPS,
        seed=layer_seed,
        qk_scale_range=(1.0, 1.0),
        beta=8, gamma=1e3, rope_base=1e6,
        h_hadamard_signs=True, enable_u_vo=True,
    )

    # Pre-build per-component M_q, M_k matrices to apply in isolation.
    R_qk = alg2.generate_r_qk(HEAD_DIM, seed=layer_seed + 1)
    H_hadamard = alg2.generate_h_qk(HEAD_DIM, (1.0, 1.0), seed=layer_seed + 2, hadamard_signs=True)
    H_nonunit = alg2.generate_h_qk(HEAD_DIM, (0.95, 1.05), seed=layer_seed + 2, hadamard_signs=False)
    H_strong = alg2.generate_h_qk(HEAD_DIM, (0.5, 1.5), seed=layer_seed + 2, hadamard_signs=False)
    I = np.eye(HEAD_DIM, dtype=np.float32)

    z_b8 = alg2.generate_block_perm(num_blocks=HEAD_DIM // 2, beta=8, gamma=1e3, rope_base=1e6, seed=layer_seed + 3)
    z_b16 = alg2.generate_block_perm(num_blocks=HEAD_DIM // 2, beta=16, gamma=1e3, rope_base=1e6, seed=layer_seed + 3)
    z_b64 = alg2.generate_block_perm(num_blocks=HEAD_DIM // 2, beta=64, gamma=1e3, rope_base=1e6, seed=layer_seed + 3)

    tau_q = full_keys.tau_kv if full_keys.tau_kv is not None else np.arange(N_KV_HEADS)
    # Build full Q-head permutation from τ_kv + τ_group (mirrors _query_head_order)
    q_head_order = alg2._query_head_order(N_Q_HEADS, N_KV_HEADS, NUM_GROUPS, full_keys.tau_kv, full_keys.tau_group)

    # Per-scenario: produce (W̃_q, W̃_k) and measure invariant matching vs plain.
    scenarios: list[tuple[str, np.ndarray, np.ndarray]] = []

    # 0. Plain control
    scenarios.append(("plain control (no Alg2)", W_q_plain, W_k_plain))

    # 1. R̂_qk only (M_q = M_k = R̂_qk)
    Wq = apply_intra_head_per_head(W_q_plain, R_qk, N_Q_HEADS, HEAD_DIM)
    Wk = apply_intra_head_per_head(W_k_plain, R_qk, N_KV_HEADS, HEAD_DIM)
    scenarios.append(("R̂_qk per-pair rotation only", Wq, Wk))

    # 2. Ĥ_qk Walsh-Hadamard ±1 only
    Wq = apply_intra_head_per_head(W_q_plain, H_hadamard, N_Q_HEADS, HEAD_DIM)
    Wk = apply_intra_head_per_head(W_k_plain, H_hadamard, N_KV_HEADS, HEAD_DIM)  # H ±1 = H⁻¹
    scenarios.append(("Ĥ_qk Walsh-Hadamard ±1 only", Wq, Wk))

    # 3. Ĥ_qk non-unit scale (deployed default reset to ±1, but worth measuring)
    H_nonunit_inv = np.linalg.inv(H_nonunit).astype(np.float32)
    Wq = apply_intra_head_per_head(W_q_plain, H_nonunit, N_Q_HEADS, HEAD_DIM)
    Wk = apply_intra_head_per_head(W_k_plain, H_nonunit_inv, N_KV_HEADS, HEAD_DIM)
    scenarios.append(("Ĥ_qk non-unit (0.95-1.05) only", Wq, Wk))

    # 4. Ĥ_qk stronger non-unit scale (0.5-1.5)
    H_strong_inv = np.linalg.inv(H_strong).astype(np.float32)
    Wq = apply_intra_head_per_head(W_q_plain, H_strong, N_Q_HEADS, HEAD_DIM)
    Wk = apply_intra_head_per_head(W_k_plain, H_strong_inv, N_KV_HEADS, HEAD_DIM)
    scenarios.append(("Ĥ_qk stronger (0.5-1.5) only", Wq, Wk))

    # 5. Ẑ_block β=8 only (paper default, deployed)
    Wq = apply_intra_head_per_head(W_q_plain, z_b8, N_Q_HEADS, HEAD_DIM)
    Wk = apply_intra_head_per_head(W_k_plain, z_b8, N_KV_HEADS, HEAD_DIM)
    scenarios.append(("Ẑ_block β=8 only (DEPLOYED)", Wq, Wk))

    # 6. Ẑ_block β=16
    Wq = apply_intra_head_per_head(W_q_plain, z_b16, N_Q_HEADS, HEAD_DIM)
    Wk = apply_intra_head_per_head(W_k_plain, z_b16, N_KV_HEADS, HEAD_DIM)
    scenarios.append(("Ẑ_block β=16 only", Wq, Wk))

    # 7. Ẑ_block β=64 (full half-d shuffle)
    Wq = apply_intra_head_per_head(W_q_plain, z_b64, N_Q_HEADS, HEAD_DIM)
    Wk = apply_intra_head_per_head(W_k_plain, z_b64, N_KV_HEADS, HEAD_DIM)
    scenarios.append(("Ẑ_block β=64 only", Wq, Wk))

    # 8. Π_head only (τ_kv + τ_group)
    Wq = apply_head_perm(W_q_plain, q_head_order, N_Q_HEADS, HEAD_DIM)
    Wk = apply_head_perm(W_k_plain, full_keys.tau_kv, N_KV_HEADS, HEAD_DIM)
    scenarios.append(("Π_head (τ_kv + τ_group) only", Wq, Wk))

    # 9. Û_vo only — control: Û_vo touches V/O, NOT W_q/W_k. So invariant unchanged by construction.
    scenarios.append(("Û_vo only (control — does not touch W_q/W_k)", W_q_plain, W_k_plain))

    # 10. Full deployed Alg2: R̂·H(±1)·Ẑ_β8 on Q, R̂·H(±1)·Ẑ_β8 on K (path-2 convention)
    M_q_full = full_keys.q_matrix
    M_k_full = full_keys.k_matrix  # equals M_q_full with hadamard ±1
    Wq = apply_intra_head_per_head(W_q_plain, M_q_full, N_Q_HEADS, HEAD_DIM)
    Wq = apply_head_perm(Wq, q_head_order, N_Q_HEADS, HEAD_DIM)
    Wk = apply_intra_head_per_head(W_k_plain, M_k_full, N_KV_HEADS, HEAD_DIM)
    Wk = apply_head_perm(Wk, full_keys.tau_kv, N_KV_HEADS, HEAD_DIM)
    scenarios.append(("FULL Alg2 deployed (R̂·H±1·Ẑ_β8 · Π_head)", Wq, Wk))

    # 11. Drop-the-deadweight: ONLY Ẑ_block + Π_head (remove R̂_qk, Hadamard, Û_vo)
    Wq = apply_intra_head_per_head(W_q_plain, z_b8, N_Q_HEADS, HEAD_DIM)
    Wq = apply_head_perm(Wq, q_head_order, N_Q_HEADS, HEAD_DIM)
    Wk = apply_intra_head_per_head(W_k_plain, z_b8, N_KV_HEADS, HEAD_DIM)
    Wk = apply_head_perm(Wk, full_keys.tau_kv, N_KV_HEADS, HEAD_DIM)
    scenarios.append(("Ẑ_block β=8 + Π_head only", Wq, Wk))

    print(f"{'scenario':<55s} {'top1':>8s} {'top10':>8s} {'invariant Δ':>14s}")
    print("-" * 90)
    plain_top1, plain_top10 = match_topk(plain_eval, plain_inv, eval_ids)
    for name, Wq_obf, Wk_obf in scenarios:
        obs_inv = _attn_ia_invariants(W_e, Wq_obf, Wk_obf, block_size=BLOCK_SIZE)
        top1, topk = match_topk(plain_eval, obs_inv, eval_ids)
        # Quantify how much the invariant per-row shifted (relative L2 distance)
        if np.array_equal(Wq_obf, W_q_plain) and np.array_equal(Wk_obf, W_k_plain):
            rel = 0.0
        else:
            rel = float(np.linalg.norm(obs_inv - plain_inv) / np.linalg.norm(plain_inv))
        print(f"{name:<55s} {top1:>8.4f} {topk:>8.4f} {rel:>14.4e}")

    print()
    print(f"Plain control sanity: top1={plain_top1:.4f}, top10={plain_top10:.4f} (expected 1.0)")
    print()
    print("Reading: top1=1.0 means attack succeeds perfectly → component provides ZERO defense.")
    print("        top1<<1.0 means component disrupts the invariant per-row → real defense.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Concern-1 verification (handoff §6.4): does the alg2.py construction
  M_q = R̂_qk · Ĥ_qk · Ẑ_block
  M_k = R̂_qk · Ĥ_qk⁻¹ · Ẑ_blockᵀ
yield attention-score invariance under RoPE?

Tests, smallest-to-broadest:
  1. M_q · M_kᵀ ≈ I ?  (the naive cancellation requirement)
  2. After NEOX RoPE applied to Q_obf and K_obf, does
       (Q_obf_rope) · (K_obf_rope)ᵀ  ≈  (Q_rope) · (K_rope)ᵀ ?
  3. Same as (2) but with R̂_qk alone (h=I, z=I)
     to isolate the R̂_qk · RoPE commutation property.
  4. With h=I, varying z to identity vs non-trivial.

Run from python/aloepri-llm/ with: .venv/bin/python scripts/check_alg2_invariance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2].parent
sys.path.insert(0, str(REPO_ROOT / "python" / "path-2"))

from lib import alg2  # noqa: E402


HEAD_DIM = 128      # Qwen3-realistic dimension
N_TOKENS = 8
ROPE_BASE = 1e6
SEED = 12345
BETA = 8            # Ẑ_block window size (matches obfuscator default)


def rope_neox(x: np.ndarray, positions: np.ndarray, base: float) -> np.ndarray:
    """NEOX-style RoPE on the last axis (head_dim).

    Splits last axis at d_h/2; for index i in [0, d_h/2) and position pos,
    rotates the (x[i], x[i+d_h/2]) pair by angle pos * base^(-2i/d_h).
    """
    d_h = x.shape[-1]
    half = d_h // 2
    inv_freq = base ** (-np.arange(half, dtype=np.float64) * 2.0 / d_h)
    # angles[t, i] = positions[t] * inv_freq[i]
    angles = positions[:, None] * inv_freq[None, :]
    cos_a = np.cos(angles).astype(x.dtype)
    sin_a = np.sin(angles).astype(x.dtype)
    x_lo = x[..., :half]
    x_hi = x[..., half:]
    # standard NEOX rotation, applied to each token position
    # (x_lo, x_hi) -> (x_lo * cos - x_hi * sin, x_lo * sin + x_hi * cos)
    if x.ndim == 2:
        # broadcast over (n_tokens, d_h)
        y_lo = x_lo * cos_a - x_hi * sin_a
        y_hi = x_lo * sin_a + x_hi * cos_a
    else:
        raise NotImplementedError("only 2D (n_tokens, d_h) supported here")
    return np.concatenate([y_lo, y_hi], axis=-1)


def report(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    diff = actual - expected
    rel = np.linalg.norm(diff) / (np.linalg.norm(expected) + 1e-12)
    print(f"  {name:<48s}  ‖Δ‖={np.linalg.norm(diff):.3e}  rel={rel:.3e}")


def test_mq_mkt_identity(keys: alg2.LayerAlg2Keys, label: str) -> None:
    M_q = keys.q_matrix
    M_k = keys.k_matrix
    prod = M_q @ M_k.T
    I = np.eye(M_q.shape[0])
    report(f"[{label}] M_q · M_kᵀ vs I", prod, I)
    prod_inv = M_q @ np.linalg.inv(M_k)
    report(f"[{label}] M_q · M_k⁻¹ vs I (alt cancellation)", prod_inv, I)
    Mq_Mk = M_q.T @ M_k
    report(f"[{label}] M_qᵀ · M_k vs I", Mq_Mk, I)


def test_rope_attention(keys: alg2.LayerAlg2Keys, label: str) -> None:
    rng = np.random.default_rng(SEED)
    Q = rng.standard_normal((N_TOKENS, HEAD_DIM)).astype(np.float64)
    K = rng.standard_normal((N_TOKENS, HEAD_DIM)).astype(np.float64)
    positions = np.arange(N_TOKENS, dtype=np.int64)

    # plaintext
    Q_rope = rope_neox(Q, positions, ROPE_BASE)
    K_rope = rope_neox(K, positions, ROPE_BASE)
    scores_plain = Q_rope @ K_rope.T

    # obfuscated: bake M_q into output axis (Q_obf = Q @ M_q), then RoPE
    Q_obf = Q @ keys.q_matrix
    K_obf = K @ keys.k_matrix
    Q_obf_rope = rope_neox(Q_obf, positions, ROPE_BASE)
    K_obf_rope = rope_neox(K_obf, positions, ROPE_BASE)
    scores_obf = Q_obf_rope @ K_obf_rope.T

    report(f"[{label}] attention scores match (RoPE both)", scores_obf, scores_plain)


def test_matrix_gamma_construction(keys: alg2.LayerAlg2Keys, label: str) -> None:
    """End-to-end: does the matrix-Γ kernel reproduce Q_plain_normed @ M_q?

    Plain QK-norm: Q_normed = γ ⊙ (Q / RMS(Q)).
    Matrix-Γ kernel: Q_normed = (Q_obf / RMS(Q_obf)) @ Γ, where Γ = MᵀDM
                                and Q_obf = Q_plain @ M.
    Expected: matrix-Γ output == Q_plain_normed @ M, up to fp32 noise.
    """
    rng = np.random.default_rng(SEED + 1)
    M = keys.q_matrix.astype(np.float64)
    gamma = rng.uniform(0.5, 2.0, size=HEAD_DIM).astype(np.float64)
    # NB: also try a γ vector with the Qwen3-shape outliers (γ values up to 68)
    # to exercise numerical robustness.
    Q_plain = rng.standard_normal((N_TOKENS, HEAD_DIM)).astype(np.float64)

    # plaintext path
    rms_plain = np.sqrt(np.mean(Q_plain ** 2, axis=-1, keepdims=True))
    Q_plain_normed = gamma * (Q_plain / rms_plain)
    expected = Q_plain_normed @ M

    # matrix-Γ kernel path
    Gamma = M.T @ np.diag(gamma) @ M
    Q_obf = Q_plain @ M
    rms_obf = np.sqrt(np.mean(Q_obf ** 2, axis=-1, keepdims=True))
    actual = (Q_obf / rms_obf) @ Gamma

    report(f"[{label}] matrix-Γ kernel ≡ Q_plain_normed @ M", actual, expected)
    # And: Γ symmetric? (the kernel implementation relies on this)
    sym_err = np.linalg.norm(Gamma - Gamma.T)
    print(f"  [{label}] Γ symmetric ‖Γ - Γᵀ‖={sym_err:.3e}")


def main() -> int:
    print("=" * 80)
    print(f"alg2 invariance check: head_dim={HEAD_DIM}, n_tokens={N_TOKENS}, "
          f"rope_base={ROPE_BASE}")
    print("=" * 80)

    # Full alg2.py defaults
    keys_full = alg2.build_layer_keys(
        head_dim=HEAD_DIM,
        num_kv_heads=1,
        num_groups=1,
        seed=SEED,
        qk_scale_range=(0.95, 1.05),
        beta=BETA,
        gamma=1e3,
        rope_base=ROPE_BASE,
    )
    print(f"\n--- FULL alg2: r·h·z (h non-unit, z general) ---")
    test_mq_mkt_identity(keys_full, "full")
    test_rope_attention(keys_full, "full")

    # h = I (orthogonal scale)
    keys_h1 = alg2.build_layer_keys(
        head_dim=HEAD_DIM,
        num_kv_heads=1,
        num_groups=1,
        seed=SEED,
        qk_scale_range=(1.0, 1.0),
        beta=BETA,
        gamma=1e3,
        rope_base=ROPE_BASE,
    )
    print(f"\n--- h=I (orthogonal) but z general ---")
    test_mq_mkt_identity(keys_h1, "h=I")
    test_rope_attention(keys_h1, "h=I")

    # h = I, z = I (only R̂_qk)
    keys_rz1 = alg2.build_layer_keys(
        head_dim=HEAD_DIM,
        num_kv_heads=1,
        num_groups=1,
        seed=SEED,
        qk_scale_range=(1.0, 1.0),
        beta=1,           # forces z_block window size 1 → identity
        gamma=1e3,
        rope_base=ROPE_BASE,
    )
    print(f"\n--- h=I, z=I (only R̂_qk rotation) ---")
    test_mq_mkt_identity(keys_rz1, "rotation-only")
    test_rope_attention(keys_rz1, "rotation-only")

    print(f"\n--- matrix-Γ kernel construction (M=full alg2 keys) ---")
    test_matrix_gamma_construction(keys_full, "full(0.95-1.05)")
    test_matrix_gamma_construction(keys_h1, "h=I")
    test_matrix_gamma_construction(keys_rz1, "rotation-only")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

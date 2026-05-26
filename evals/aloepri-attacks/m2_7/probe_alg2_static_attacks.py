"""Synthetic VMA + IA Gate-IA + IMA-EmbedRow per-Alg2-component impact probe.

Generates random ModelWeights at moderate Q3-shape, applies each Algorithm 2
component in isolation to attn_q / attn_k / attn_v / attn_output, then runs
the real `run_vma` and `run_ia` drivers against (plain vs obs) pairs with
τ=identity, no Alg1, no §5.2.2.

**LIMITATION (added 2026-05-26).** This probe measures Alg2 component
contributions **without §5.2.2 substrate** (no W_e noise, no Π token-perm).
That under-predicts the real-cell contribution by ~10× because §5.2.2 and Alg2
interact **superadditively** on VMA: R̂_qk + Ĥ_qk±1 marginal is 1.96 pp on
Alg1-only substrate but 17.84 pp on Alg1+§5.2.2 substrate (a 9× amplification).

The probe's numbers are correct in their setting (no-§5.2.2 substrate) and
match real measurements at Alg1→Alg1+minAlg2. They do NOT predict deployed-cell
contributions where §5.2.2 is present. To compute deployed-cell contributions,
measure on real cells with the appropriate §5.2.2 substrate.

See `docs/handoffs/2026-05-25-alg2-attack-crossmap.md` § "Within-Alg2 bisection
(2026-05-26)" + § "Synthetic probe scope correction" for the full picture.

Predicted outcomes (in the no-§5.2.2 substrate that this probe runs):
  - Π_head, Ẑ_block (β≤block_size): zero VMA + zero IA, because sort/sum
    are permutation-invariant.
  - R̂_qk, Ĥ_qk non-unit / strong: 0.4-1 pp marginal (much higher under §5.2.2).
  - Û_vo: ~1 pp marginal (much higher under §5.2.2).
  - IA Gate-IA: zero across ALL Alg2 components (Alg2 doesn't touch W_gate).
  - IMA-EmbedRow (W̃_e surface): zero across ALL Alg2 components (no W_e touch).

Run:
  PYTHONPATH=python/aloepri-llm:evals/aloepri-attacks/m2_7 \\
      python3 evals/aloepri-attacks/m2_7/probe_alg2_static_attacks.py
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python" / "aloepri-llm"))
sys.path.insert(0, str(REPO / "evals" / "aloepri-attacks" / "m2_7"))
sys.path.insert(0, str(REPO / "evals" / "aloepri-attacks"))

from lib import alg2  # noqa: E402
from extract_gguf_weights import ModelWeights  # noqa: E402
from run_static_attacks import run_vma, run_ia  # noqa: E402


# ── shapes (moderate, ~Q3-4B-shaped) ────────────────────────────────
D = 1024
N_Q_HEADS = 8
N_KV_HEADS = 4
HEAD_DIM = 128
Q_DIM = N_Q_HEADS * HEAD_DIM  # 1024
KV_DIM = N_KV_HEADS * HEAD_DIM  # 512
NUM_GROUPS = N_Q_HEADS // N_KV_HEADS  # 2
INTERMEDIATE = 2048
VOCAB = 4096
N_LAYERS = 2
SEED = 20260525

VMA_EVAL = 256
VMA_POOL = 2048
IA_EVAL = 1024
IA_POOL = 2048


def random_plain_weights(rng: np.random.Generator) -> ModelWeights:
    """Random plaintext ModelWeights at the moderate Q3-shape."""
    token_embd = rng.standard_normal((VOCAB, D)).astype(np.float32)
    output = rng.standard_normal((VOCAB, D)).astype(np.float32)
    per_layer: list[dict[str, np.ndarray]] = []
    for _ in range(N_LAYERS):
        per_layer.append({
            "attn_q":      rng.standard_normal((Q_DIM,  D)).astype(np.float32),
            "attn_k":      rng.standard_normal((KV_DIM, D)).astype(np.float32),
            "attn_v":      rng.standard_normal((KV_DIM, D)).astype(np.float32),
            "attn_output": rng.standard_normal((D, Q_DIM)).astype(np.float32),
            "ffn_gate":    rng.standard_normal((INTERMEDIATE, D)).astype(np.float32),
            "ffn_up":      rng.standard_normal((INTERMEDIATE, D)).astype(np.float32),
            "ffn_down":    rng.standard_normal((D, INTERMEDIATE)).astype(np.float32),
        })
    return ModelWeights(
        label="plain", path=Path("/tmp/synthetic-plain.gguf"),
        d_eff=D, intermediate_eff=INTERMEDIATE, n_layers=N_LAYERS,
        vocab_size=VOCAB, token_embd=token_embd, output=output, per_layer=per_layer,
    )


def apply_per_head(W_natural: np.ndarray, M_per_head: np.ndarray,
                   n_heads: int, head_dim: int, axis: int = 0) -> np.ndarray:
    """Apply per-head head_dim-axis transform M (head_dim, head_dim) to W.

    axis=0: W shape (n_heads*head_dim, d) → W̃ = M.T @ W per head (used for
            attn_q, attn_k, attn_v).
    axis=1: W shape (d, n_heads*head_dim) → W̃ = W @ M.T per head (used for
            attn_output, where head_dim is on the input axis).
    """
    M_T = M_per_head.T.astype(np.float32)
    if axis == 0:
        W_h = W_natural.reshape(n_heads, head_dim, -1)
        W_h_out = np.einsum("ij,hjk->hik", M_T, W_h)
        return W_h_out.reshape(n_heads * head_dim, -1)
    elif axis == 1:
        W_h = W_natural.reshape(-1, n_heads, head_dim)
        W_h_out = np.einsum("dhj,ij->dhi", W_h, M_T)
        return W_h_out.reshape(-1, n_heads * head_dim)
    else:
        raise ValueError(f"axis must be 0 or 1, got {axis}")


def apply_head_perm(W_natural: np.ndarray, tau: np.ndarray,
                    n_heads: int, head_dim: int, axis: int = 0) -> np.ndarray:
    """Permute heads along the requested axis."""
    if axis == 0:
        return W_natural.reshape(n_heads, head_dim, -1)[tau].reshape(n_heads * head_dim, -1)
    elif axis == 1:
        out = W_natural.reshape(-1, n_heads, head_dim)[:, tau, :]
        return out.reshape(-1, n_heads * head_dim)
    else:
        raise ValueError(axis)


def build_obs(plain: ModelWeights, scenario: str,
              M_q: np.ndarray, M_k: np.ndarray, M_v: np.ndarray, M_o_inv: np.ndarray,
              q_head_perm: np.ndarray, kv_head_perm: np.ndarray,
              use_intra_qk: bool, use_intra_vo: bool,
              use_head_perm: bool) -> ModelWeights:
    """Build an obfuscated ModelWeights with the requested Alg2 components on."""
    obs = ModelWeights(
        label=f"obs[{scenario}]", path=Path(f"/tmp/synthetic-{scenario}.gguf"),
        d_eff=D, intermediate_eff=INTERMEDIATE, n_layers=N_LAYERS,
        vocab_size=VOCAB,
        token_embd=plain.token_embd.copy(),  # Alg2 doesn't touch W_e
        output=plain.output.copy(),          # Alg2 doesn't touch W_h
        per_layer=[],
    )
    for layer in plain.per_layer:
        new = {kind: arr.copy() for kind, arr in layer.items()}
        if use_intra_qk:
            new["attn_q"] = apply_per_head(new["attn_q"], M_q, N_Q_HEADS, HEAD_DIM, axis=0)
            new["attn_k"] = apply_per_head(new["attn_k"], M_k, N_KV_HEADS, HEAD_DIM, axis=0)
        if use_intra_vo:
            new["attn_v"] = apply_per_head(new["attn_v"], M_v, N_KV_HEADS, HEAD_DIM, axis=0)
            new["attn_output"] = apply_per_head(new["attn_output"], M_o_inv, N_Q_HEADS, HEAD_DIM, axis=1)
        if use_head_perm:
            new["attn_q"] = apply_head_perm(new["attn_q"], q_head_perm, N_Q_HEADS, HEAD_DIM, axis=0)
            new["attn_k"] = apply_head_perm(new["attn_k"], kv_head_perm, N_KV_HEADS, HEAD_DIM, axis=0)
            new["attn_v"] = apply_head_perm(new["attn_v"], kv_head_perm, N_KV_HEADS, HEAD_DIM, axis=0)
            new["attn_output"] = apply_head_perm(new["attn_output"], q_head_perm, N_Q_HEADS, HEAD_DIM, axis=1)
        obs.per_layer.append(new)
    return obs


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"## Synthetic VMA + IA Gate-IA per-Alg2-component probe")
    print(f"## d={D}, n_q={N_Q_HEADS}, n_kv={N_KV_HEADS}, head_dim={HEAD_DIM}, "
          f"vocab={VOCAB}, n_layers={N_LAYERS}, τ=identity, no Alg1, no §5.2.2")
    print()

    plain = random_plain_weights(rng)

    # ── Build Alg2 keys ────────────────────────────────────────────
    layer_seed = SEED + 1
    full_keys = alg2.build_layer_keys(
        head_dim=HEAD_DIM, num_kv_heads=N_KV_HEADS, num_groups=NUM_GROUPS,
        seed=layer_seed,
        qk_scale_range=(1.0, 1.0),
        beta=8, gamma=1e3, rope_base=1e6,
        h_hadamard_signs=True, enable_u_vo=True,
    )
    R_qk = alg2.generate_r_qk(HEAD_DIM, seed=layer_seed + 1)
    H_hadamard = alg2.generate_h_qk(HEAD_DIM, (1.0, 1.0), seed=layer_seed + 2, hadamard_signs=True)
    H_nonunit = alg2.generate_h_qk(HEAD_DIM, (0.95, 1.05), seed=layer_seed + 2, hadamard_signs=False)
    H_strong = alg2.generate_h_qk(HEAD_DIM, (0.5, 1.5), seed=layer_seed + 2, hadamard_signs=False)
    Z_b8 = alg2.generate_block_perm(HEAD_DIM // 2, beta=8, gamma=1e3, rope_base=1e6, seed=layer_seed + 3)
    Z_b64 = alg2.generate_block_perm(HEAD_DIM // 2, beta=64, gamma=1e3, rope_base=1e6, seed=layer_seed + 3)
    U_vo = full_keys.u_vo if full_keys.u_vo is not None else np.eye(HEAD_DIM, dtype=np.float32)
    U_vo_inv = full_keys.u_vo_inv if full_keys.u_vo_inv is not None else np.eye(HEAD_DIM, dtype=np.float32)
    I = np.eye(HEAD_DIM, dtype=np.float32)

    q_head_order = alg2._query_head_order(N_Q_HEADS, N_KV_HEADS, NUM_GROUPS,
                                          full_keys.tau_kv, full_keys.tau_group)
    kv_head_order = alg2._kv_head_order(N_KV_HEADS, full_keys.tau_kv)

    # ── Scenarios ──────────────────────────────────────────────────
    scenarios: list[tuple[str, dict]] = [
        ("plain control (no Alg2)",
            dict(M_q=I, M_k=I, M_v=I, M_o_inv=I,
                 q_head_perm=np.arange(N_Q_HEADS), kv_head_perm=np.arange(N_KV_HEADS),
                 use_intra_qk=False, use_intra_vo=False, use_head_perm=False)),
        ("R̂_qk per-pair rotation only",
            dict(M_q=R_qk, M_k=R_qk, M_v=I, M_o_inv=I,
                 q_head_perm=np.arange(N_Q_HEADS), kv_head_perm=np.arange(N_KV_HEADS),
                 use_intra_qk=True, use_intra_vo=False, use_head_perm=False)),
        ("Ĥ_qk Walsh-Hadamard ±1 only",
            dict(M_q=H_hadamard, M_k=H_hadamard, M_v=I, M_o_inv=I,
                 q_head_perm=np.arange(N_Q_HEADS), kv_head_perm=np.arange(N_KV_HEADS),
                 use_intra_qk=True, use_intra_vo=False, use_head_perm=False)),
        ("Ĥ_qk non-unit (0.95-1.05) only",
            dict(M_q=H_nonunit, M_k=np.linalg.inv(H_nonunit).astype(np.float32), M_v=I, M_o_inv=I,
                 q_head_perm=np.arange(N_Q_HEADS), kv_head_perm=np.arange(N_KV_HEADS),
                 use_intra_qk=True, use_intra_vo=False, use_head_perm=False)),
        ("Ĥ_qk strong (0.5-1.5) only",
            dict(M_q=H_strong, M_k=np.linalg.inv(H_strong).astype(np.float32), M_v=I, M_o_inv=I,
                 q_head_perm=np.arange(N_Q_HEADS), kv_head_perm=np.arange(N_KV_HEADS),
                 use_intra_qk=True, use_intra_vo=False, use_head_perm=False)),
        ("Ẑ_block β=8 only (DEPLOYED)",
            dict(M_q=Z_b8, M_k=Z_b8, M_v=I, M_o_inv=I,
                 q_head_perm=np.arange(N_Q_HEADS), kv_head_perm=np.arange(N_KV_HEADS),
                 use_intra_qk=True, use_intra_vo=False, use_head_perm=False)),
        ("Ẑ_block β=64 only",
            dict(M_q=Z_b64, M_k=Z_b64, M_v=I, M_o_inv=I,
                 q_head_perm=np.arange(N_Q_HEADS), kv_head_perm=np.arange(N_KV_HEADS),
                 use_intra_qk=True, use_intra_vo=False, use_head_perm=False)),
        ("Π_head only",
            dict(M_q=I, M_k=I, M_v=I, M_o_inv=I,
                 q_head_perm=q_head_order, kv_head_perm=kv_head_order,
                 use_intra_qk=False, use_intra_vo=False, use_head_perm=True)),
        ("Û_vo only",
            dict(M_q=I, M_k=I, M_v=U_vo, M_o_inv=U_vo_inv,
                 q_head_perm=np.arange(N_Q_HEADS), kv_head_perm=np.arange(N_KV_HEADS),
                 use_intra_qk=False, use_intra_vo=True, use_head_perm=False)),
        ("FULL deployed Alg2 (R̂·H±1·Ẑ_β8 + Π_head + Û_vo)",
            dict(M_q=full_keys.q_matrix, M_k=full_keys.k_matrix, M_v=U_vo, M_o_inv=U_vo_inv,
                 q_head_perm=q_head_order, kv_head_perm=kv_head_order,
                 use_intra_qk=True, use_intra_vo=True, use_head_perm=True)),
    ]

    print(f"{'scenario':<50s}  {'VMA t1':>7s} {'VMA t10':>8s}  {'Gate-IA t1':>10s} {'Gate-IA t10':>11s}  {'Attn-IA t1':>10s} {'Attn-IA t10':>11s}")
    print("-" * 130)

    for name, kwargs in scenarios:
        obs = build_obs(plain, name, **kwargs)
        vma = run_vma(plain, obs, tau=None,
                      eval_size=VMA_EVAL, candidate_pool_size=VMA_POOL, bins=64)
        ia = run_ia(plain, obs, tau=None,
                    eval_size=IA_EVAL, candidate_pool_size=IA_POOL)
        gate_t1 = ia.extra["gate_ia_top1"]
        gate_t10 = ia.extra["gate_ia_top10"]
        attn_t1 = ia.extra["attn_ia_top1"]
        attn_t10 = ia.extra["attn_ia_top10"]
        print(f"{name:<50s}  {vma.ttrsr_top1:>7.4f} {vma.ttrsr_top10:>8.4f}  "
              f"{gate_t1:>10.4f} {gate_t10:>11.4f}  {attn_t1:>10.4f} {attn_t10:>11.4f}")

    print()
    print("Reading: top1=1.0 → attack succeeds perfectly → ZERO defence from that component.")
    print("         top1<<1.0 → component disrupts the attack → real defence.")
    print()
    print("Note: IMA-EmbedRow-ridge (W̃_e static row attack) operates on token_embd; Alg2 doesn't")
    print("      touch it, so all scenarios are equivalent to plain by construction. Skipped.")


if __name__ == "__main__":
    main()

"""Algorithm 2 (intra-head + inter-head) attention obfuscation in numpy.

Paper §5.2.3 + reference `vendor/aloepri-py/src/{attention_keys,
stage_h_attention_static}.py`. Ported from torch to numpy with the
same seed conventions (so a given `seed` produces matching keys).

Profile: `rqk_hqk_block_taukv_taugroup` — matches reference default.

**Û_vo handling.** The V↔O random projection from paper §5.2.3 (step 4
+ step 6 alt + step 7) is opt-in via `enable_u_vo=True` on
`build_layer_keys`. The reference impl always passes `dense_transform
=None` for V — equivalent to `enable_u_vo=False` here. Paper Table 4's
0.0 % HiddenState TTRSR is measured with full Algorithm 2 including
Û_vo; without it, paper Table 4 reports 0.82 % (Noise+KeyMat row).
The 2026-05-21 audit (`docs/handoffs/2026-05-21-ima-transformer-paper-disparity.md`)
attributed the path-2 ISA HiddenState attenuation gap (4B ≈ 50 %, 8B
≈ 4 %) partly to this missing Û_vo component, so we expose the flag
to allow re-obfuscating with the full paper Algorithm 2 recipe.

Acts on the **head_dim axis** of W_q, W_k, W_v, W_o. Commutes with
the existing keymat transform which acts on the **residual (d) axis**.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ────────────────────────────────────────────────────────────────────
# key generation
# ────────────────────────────────────────────────────────────────────


def generate_r_qk(head_dim: int, seed: int) -> np.ndarray:
    """Block-diag of 2D rotations R(ρ_i), one per RoPE pair.

    Convention matches `attention_keys.generate_r_qk`: each block lives
    at positions (i, i+half) on the diagonal, not at (2i, 2i+1).
    Compatible with the half-rotated RoPE layout used by Qwen's RoPE.
    """
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even")
    rng = np.random.default_rng(seed)
    num_blocks = head_dim // 2
    angles = rng.uniform(0.0, 2.0 * np.pi, size=num_blocks).astype(np.float32)
    matrix = np.zeros((head_dim, head_dim), dtype=np.float32)
    half = num_blocks
    for i, angle in enumerate(angles):
        c = np.cos(angle)
        s = np.sin(angle)
        j = i + half
        matrix[i, i] = c
        matrix[i, j] = -s
        matrix[j, i] = s
        matrix[j, j] = c
    return matrix


def generate_h_qk(
    head_dim: int,
    scale_range: tuple[float, float],
    seed: int,
    *,
    hadamard_signs: bool = False,
) -> np.ndarray:
    """Per-RoPE-pair scale diagonal.

    When `hadamard_signs=True`, sample each block scale uniformly from
    {-1, +1} (Walsh-Hadamard ±1 form). Keeps `M_q = R · H · Z`
    orthogonal so the matrix-Γ kernel algebra stays exact, while
    adding per-pair sign flips to the obfuscation.

    Otherwise `scale_range` controls uniform sampling in (low, high)
    — the reference convention. With non-unit scale, M_q becomes
    non-orthogonal and matrix-Γ algebra drifts; safe only when not
    deploying through the matrix-Γ kernel.
    """
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even")
    rng = np.random.default_rng(seed)
    num_blocks = head_dim // 2
    if hadamard_signs:
        block_scales = rng.choice(np.array([-1.0, 1.0], dtype=np.float32),
                                  size=num_blocks).astype(np.float32)
    else:
        low, high = scale_range
        if low <= 0 or high <= 0 or low > high:
            raise ValueError(f"invalid scale range {scale_range}")
        block_scales = rng.uniform(low, high, size=num_blocks).astype(np.float32)
    # Reference uses cat([block_scales, block_scales]) — same value on both
    # halves of the diagonal so it pairs with the half-rotated R_qk layout.
    diag = np.concatenate([block_scales, block_scales])
    return np.diag(diag).astype(np.float32)


def generate_block_perm(
    num_blocks: int,
    beta: int,
    gamma: float,
    rope_base: float,
    seed: int,
    mode: str = "fixed_window",
) -> np.ndarray:
    """Block-wise locality-preserving permutation of RoPE pairs.

    Paper §5.2.3 Ẑ_block permutes RoPE pairs but only within bands of
    similar angular frequency, otherwise attention scores drift after
    RoPE (R̂_qk's commutation with RoPE assumes the data at position
    i continues to see RoPE frequency θ_i).

    `mode="fixed_window"` is the hardened path-2 default: each
    consecutive group of β RoPE pairs is shuffled internally, ignoring
    γ and rope_base. `mode="dynamic_window"` retains the paper-like
    softmax window sampler; with γ=1e3 it often collapses to identity,
    which is useful for paper-fidelity probes but weak as a defence.
    """
    if mode not in {"fixed_window", "dynamic_window"}:
        raise ValueError(f"unsupported block permutation mode {mode!r}")
    rng = np.random.default_rng(seed)
    beta = max(1, min(int(beta), num_blocks))
    perm_blocks: list[int] = []
    start = 0
    if mode == "fixed_window":
        while start < num_blocks:
            c = min(beta, num_blocks - start)
            window = np.arange(start, start + c, dtype=np.int64)
            rng.shuffle(window)
            perm_blocks.extend(window.tolist())
            start += c
    else:
        zeta_log = (-2.0 * np.arange(num_blocks, dtype=np.float64) / max(1, num_blocks)) * np.log(float(rope_base))
        while start < num_blocks:
            c = min(beta, num_blocks - start)
            if c == 1:
                perm_blocks.append(start)
                start += 1
                continue
            local = gamma * (zeta_log[start:start + c] - zeta_log[start])
            local = local - np.max(local)
            probs = np.exp(local)
            probs = probs / np.sum(probs)
            window_size = int(rng.choice(np.arange(1, c + 1), p=probs))
            window = np.arange(start, start + window_size, dtype=np.int64)
            rng.shuffle(window)
            perm_blocks.extend(window.tolist())
            start += window_size
    perm = np.array(perm_blocks, dtype=np.int64)

    # Build the (head_dim, head_dim) permutation matrix in half-rotated layout
    head_dim = num_blocks * 2
    half = num_blocks
    block_matrix = np.zeros((head_dim, head_dim), dtype=np.float32)
    for original_block_idx in range(num_blocks):
        target_block_idx = int(perm[original_block_idx])
        block_matrix[target_block_idx, original_block_idx] = 1.0
        block_matrix[target_block_idx + half, original_block_idx + half] = 1.0
    return block_matrix


def generate_head_perm(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random non-identity permutation of [0, n), plus its inverse."""
    rng = np.random.default_rng(seed)
    tau = rng.permutation(n).astype(np.int64)
    if n > 1:
        attempts = 0
        while np.array_equal(tau, np.arange(n)) and attempts < 8:
            tau = rng.permutation(n).astype(np.int64)
            attempts += 1
        if np.array_equal(tau, np.arange(n)):
            tau = np.roll(np.arange(n, dtype=np.int64), 1)
    inv = np.argsort(tau).astype(np.int64)
    return tau, inv


# ────────────────────────────────────────────────────────────────────
# composed per-layer config
# ────────────────────────────────────────────────────────────────────


def generate_u_vo(
    head_dim: int,
    seed: int,
    *,
    paper_literal: bool = False,
    mode: str | None = None,
    pow2_exp: int = 1,
) -> np.ndarray:
    """Random projection Û_vo for the V→O cancellation (paper §5.2.3 step 4).

    Sampled from N(0, (1/head_dim) · I) per paper Algorithm 2 line 4.

    `mode` controls the sampled family:

    - `raw-gaussian`: return the raw Gaussian sample directly
      (paper-faithful). The matrix is later inverted (step 7:
      W̃_o = Û_vo⁻¹ · W_o); high condition number on the raw sample may
      introduce numerical loss when bf16-casting Û_vo⁻¹.

    - `qr-perturb` (default, our deployed cell): the raw Gaussian is
      QR-orthogonalised then perturbed by a small Gaussian. This produces
      a near-orthogonal matrix with well-conditioned inverse at bf16, at
      the cost of preserving more per-head structure than paper's pure
      Gaussian.

    - `orthogonal`: QR-orthogonalised only. This keeps condition number
      exactly 1, reducing bf16 V/O cancellation error versus `qr-perturb`,
      but gives up the singular-value perturbation.

    - `signed-permutation`: a random signed permutation matrix. This is
      bf16-exact and adds essentially no accuracy loss beyond the base
      bf16 weight cast, but is a much weaker V/O mixing defence.

    - `pow2-monomial`: a signed permutation with per-channel power-of-two
      scales sampled from `[-pow2_exp, +pow2_exp]`. Multiplication by
      powers of two commutes with bf16 rounding for normal-range values,
      so this preserves the near-zero accuracy cost of signed permutations
      while adding magnitude perturbation on the V/O channels.

    Returns: (head_dim, head_dim) float32.
    """
    if mode is None:
        mode = "raw-gaussian" if paper_literal else "qr-perturb"
    if paper_literal and mode == "qr-perturb":
        mode = "raw-gaussian"
    valid_modes = {"raw-gaussian", "qr-perturb", "orthogonal", "signed-permutation", "pow2-monomial"}
    if mode not in valid_modes:
        raise ValueError(f"unsupported U_vo mode {mode!r}; expected one of {sorted(valid_modes)}")

    rng = np.random.default_rng(seed)
    if mode in {"signed-permutation", "pow2-monomial"}:
        perm = rng.permutation(head_dim)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=head_dim)
        scales = np.ones(head_dim, dtype=np.float32)
        if mode == "pow2-monomial":
            e = max(0, int(pow2_exp))
            exponents = rng.integers(-e, e + 1, size=head_dim, dtype=np.int32)
            scales = np.exp2(exponents).astype(np.float32)
        out = np.zeros((head_dim, head_dim), dtype=np.float32)
        out[np.arange(head_dim), perm] = signs * scales
        return out

    # Standard Gaussian with paper's variance 1/head_dim.
    raw = rng.standard_normal(size=(head_dim, head_dim)).astype(np.float64)
    raw *= (1.0 / np.sqrt(head_dim))
    if mode == "raw-gaussian":
        return raw.astype(np.float32)
    # QR-stabilise: gives Q (orthogonal) · R (upper triangular with positive
    # diagonal). We return Q · (I + δ·R_norm) where R_norm is the
    # diagonal-of-R scaled small — keeps the matrix invertible with a
    # well-conditioned inverse while preserving the Gaussian-projection
    # spirit. Without this, the raw Gaussian can have condition number
    # > 1e3 at head_dim=128 and the bf16 cast of Û_vo⁻¹ loses precision
    # on W̃_o.
    q, r = np.linalg.qr(raw)
    diag = np.diag(r)
    diag_sign = np.sign(diag)
    diag_sign[diag_sign == 0] = 1.0
    q = q * diag_sign  # fix Q sign convention (Householder)
    if mode == "orthogonal":
        return q.astype(np.float32)
    # Small Gaussian perturbation on the orthogonal Q — keeps it close to
    # an orthogonal matrix but breaks the head_dim symmetry the attacker
    # would otherwise rely on.
    perturb = rng.standard_normal(size=(head_dim, head_dim)).astype(np.float64) * 0.05
    out = q + perturb
    return out.astype(np.float32)


@dataclass(frozen=True)
class LayerAlg2Keys:
    q_matrix: np.ndarray       # (head_dim, head_dim) — R_qk · H_qk · Z_block
    k_matrix: np.ndarray       # (head_dim, head_dim) — R_qk · H_qk⁻¹ · Z_block
    u_vo: np.ndarray | None    # (head_dim, head_dim) — Û_vo, applied to W_v
    u_vo_inv: np.ndarray | None  # (head_dim, head_dim) — Û_vo⁻¹, applied to W_o input axis
    tau_kv: np.ndarray | None  # (num_kv_heads,)
    inv_tau_kv: np.ndarray | None
    tau_group: np.ndarray | None  # (num_groups,)
    inv_tau_group: np.ndarray | None


def build_layer_keys(
    *,
    head_dim: int,
    num_kv_heads: int,
    num_groups: int,
    seed: int,
    qk_scale_range: tuple[float, float] = (0.95, 1.05),
    beta: int = 8,
    gamma: float = 1e3,
    rope_base: float = 1e6,
    h_hadamard_signs: bool = False,
    enable_u_vo: bool = False,
    u_vo_mode: str = "qr-perturb",
    u_vo_pow2_exp: int = 1,
    paper_literal: bool = False,
    paper_literal_k: bool = False,
    paper_literal_u_vo: bool = False,
    paper_literal_k_no_r: bool = False,
    block_perm_mode: str = "fixed_window",
) -> LayerAlg2Keys:
    paper_literal_k = bool(paper_literal or paper_literal_k)
    paper_literal_u_vo = bool(paper_literal or paper_literal_u_vo)
    paper_literal_k_no_r = bool(paper_literal_k_no_r)
    r_qk = generate_r_qk(head_dim, seed + 1)
    h_qk = generate_h_qk(head_dim, qk_scale_range, seed + 2,
                         hadamard_signs=h_hadamard_signs)
    h_qk_inv = np.linalg.inv(h_qk).astype(np.float32)
    z_block = generate_block_perm(
        num_blocks=head_dim // 2,
        beta=beta,
        gamma=gamma,
        rope_base=rope_base,
        seed=seed + 3,
        mode=block_perm_mode,
    )
    z_block_inv = z_block.T

    q_matrix = (r_qk @ h_qk @ z_block).astype(np.float32)
    # Score-invariance requires M_q · M_kᵀ = I. With M_q = R·H·Z, the
    # algebraic solution is M_k = R · H⁻¹ · Z (same Z, no transpose):
    #   M_k.T = Zᵀ · H⁻¹ · Rᵀ
    #   M_q · M_k.T = R · H · (Z · Zᵀ) · H⁻¹ · Rᵀ = R · I · I · Rᵀ = I
    # The original construction used Z⁻¹ = Zᵀ on M_k which gives an
    # extra factor of Z² in the cancellation — only collapses to I when
    # Z² = I (which the identity-Z degeneracy above silently provided).
    # See docs/handoffs/2026-05-19-alg2-z-block-degeneracy.md.
    #
    # CAVEAT (added 2026-05-25). The `M_q · M_kᵀ = I` identity is necessary
    # but NOT sufficient for end-to-end score invariance under RoPE: RoPE
    # is a per-RoPE-pair rotation at position-dependent frequency, and Ẑ
    # permutes the RoPE-pair index → M_q does NOT commute with RoPE per
    # pair when β > 1. Measured score Δ vs plain (head_dim=128, β=8
    # DEPLOYED): 35 % relative. β-sweep:
    #   β= 1 → 3.6e-8 ; β= 2 → 9.4 % ; β= 4 → 33 % ; β= 8 → 35 % ;
    #   β=16 → 54 %  ; β=64 → 75 %.
    # So the "matrix-Γ algebra is still exact under that M_q" claim in the
    # 2026-05-19 handoff only holds at β=1; the deployed β=8 introduces a
    # ~35 % score perturbation. Treated as a *deliberate non-covariant*
    # defence component (it accounts for most of Alg2's measured ISA
    # HiddenState delta — see docs/handoffs/2026-05-25-alg2-attack-crossmap.md).
    if paper_literal_k_no_r:
        # Experimental beyond-paper variant from the 2026-05-26 PM diagnosis:
        # omit R̂ on the K side. This is *not* Algorithm 2 line 6 in the PDF;
        # it creates a much larger score perturbation and is kept only as an
        # explicit hardening knob for follow-up tests.
        k_matrix = (h_qk_inv @ z_block.T).astype(np.float32)
    elif paper_literal_k:
        # Paper Algorithm 2 line 6 literal, verified against 2603.01499v2.pdf:
        # W̃_k = Q̂_k W_k R̂_qk Ĥ_qk⁻¹ Ẑ_blockᵀ. The transpose on Z leaves a
        # Z² residual in M_q · M_k.T when Z is non-involutive, but β=1 still
        # cancels exactly.
        k_matrix = (r_qk @ h_qk_inv @ z_block.T).astype(np.float32)
    else:
        k_matrix = (r_qk @ h_qk_inv @ z_block).astype(np.float32)

    if enable_u_vo:
        u_vo = generate_u_vo(
            head_dim,
            seed + 7,
            paper_literal=paper_literal_u_vo,
            mode=u_vo_mode,
            pow2_exp=u_vo_pow2_exp,
        )
        u_vo_inv = np.linalg.inv(u_vo.astype(np.float64)).astype(np.float32)
    else:
        u_vo = None
        u_vo_inv = None

    if num_kv_heads > 1:
        tau_kv, inv_tau_kv = generate_head_perm(num_kv_heads, seed + 4)
    else:
        tau_kv = inv_tau_kv = None
    if num_groups > 1:
        tau_group, inv_tau_group = generate_head_perm(num_groups, seed + 5)
    else:
        tau_group = inv_tau_group = None

    return LayerAlg2Keys(
        q_matrix=q_matrix,
        k_matrix=k_matrix,
        u_vo=u_vo,
        u_vo_inv=u_vo_inv,
        tau_kv=tau_kv,
        inv_tau_kv=inv_tau_kv,
        tau_group=tau_group,
        inv_tau_group=inv_tau_group,
    )


# ────────────────────────────────────────────────────────────────────
# head-feature ordering for the GQA inter-head shuffle
# ────────────────────────────────────────────────────────────────────


def _query_head_order(
    num_q_heads: int,
    num_kv_heads: int,
    num_groups: int,
    tau_kv: np.ndarray | None,
    tau_group: np.ndarray | None,
) -> np.ndarray:
    """Construct the GQA-aware Q-head permutation.

    Matches `_query_head_order` + `GQALayout.permute_query_groups` from
    the reference. Reshape (num_q_heads,) → (num_kv_heads, num_groups),
    permute axes by tau_kv then tau_group, then flatten.
    """
    grouped = np.arange(num_q_heads, dtype=np.int64).reshape(num_kv_heads, num_groups)
    if tau_kv is not None:
        grouped = grouped[tau_kv, :]
    if tau_group is not None:
        grouped = grouped[:, tau_group]
    return grouped.reshape(-1)


def _kv_head_order(num_kv_heads: int, tau_kv: np.ndarray | None) -> np.ndarray:
    if tau_kv is None:
        return np.arange(num_kv_heads, dtype=np.int64)
    return tau_kv.astype(np.int64)


def _expand_feature_order(head_order: np.ndarray, head_dim: int) -> np.ndarray:
    """Lift a head-index permutation to a feature-axis permutation."""
    return np.concatenate(
        [np.arange(int(h) * head_dim, (int(h) + 1) * head_dim, dtype=np.int64) for h in head_order]
    )


# ────────────────────────────────────────────────────────────────────
# weight transform
# ────────────────────────────────────────────────────────────────────


def _block_diag_repeat(matrix: np.ndarray, repeats: int) -> np.ndarray:
    """Build a (n*m, n*m) block-diagonal repeat of a (m, m) matrix.

    Equivalent to torch.block_diag(*[matrix]*repeats).
    """
    m = matrix.shape[0]
    out = np.zeros((m * repeats, m * repeats), dtype=matrix.dtype)
    for i in range(repeats):
        out[i * m : (i + 1) * m, i * m : (i + 1) * m] = matrix
    return out


def apply_qkv_output_transform(
    weight: np.ndarray,
    dense_transform: np.ndarray | None,
    feature_order: np.ndarray,
) -> np.ndarray:
    """Apply intra-head dense + head-shuffle to a QKV weight tensor.

    Args:
        weight: GGUF natural-shape (n_heads · head_dim, d_in).
        dense_transform: (n_heads · head_dim, n_heads · head_dim)
            block-diag intra-head matrix, or None for V (no intra-head).
        feature_order: (n_heads · head_dim,) row permutation.

    Returns:
        Transformed weight of same shape.
    """
    out = weight
    if dense_transform is not None:
        # Paper: W̃ = W_paper · M, where M acts on the head_dim (output) axis.
        # numpy natural shape transposes paper convention, so this becomes
        # M.T @ W_numpy on axis 0.
        out = (dense_transform.T.astype(out.dtype)) @ out
    out = out[feature_order]
    return out


def apply_o_output_transform(
    weight: np.ndarray,
    feature_order: np.ndarray,
    dense_input_transform: np.ndarray | None = None,
) -> np.ndarray:
    """Apply head-shuffle + optional Û_vo⁻¹ input-axis transform to W_o.

    W_o has natural shape `(d_model, n_q_heads · head_dim)` — the
    head-feature axis is axis 1.

    `dense_input_transform`, when given, is the
    `_block_diag_repeat(Û_vo⁻¹, n_q_heads)` matrix from
    `build_layer_keys`. Paper §5.2.3 step 7: `W̃_o = Û_vo⁻¹ · W_o`.
    Translating to numpy natural shape: V̄ = (X · W_v^T · Û_vo) flows
    through attention; the residual contribution then is V̄ · W_o^T =
    X · W_v^T · Û_vo · W_o^T. For this to equal the plain `X · W_v^T ·
    W_o^T` we need `Û_vo · W_o^T = W_o^T` after substitution, which
    gives `W̃_o^T = Û_vo⁻¹ · W_o^T` i.e. `W̃_o = W_o · Û_vo⁻¹.T`. In
    block-diagonal-per-head form, that's a right-multiply of
    W_o_natural by `(Û_vo_inv_block_diag).T`.
    """
    out = weight
    if dense_input_transform is not None:
        out = out @ dense_input_transform.T.astype(out.dtype)
    out = out[:, feature_order]
    return out

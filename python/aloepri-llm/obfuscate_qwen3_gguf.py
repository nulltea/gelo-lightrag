#!/usr/bin/env python3
"""
AloePri offline obfuscation rewriter for Qwen3 dense GGUF artifacts.

Qwen3's residual-stream norm topology is simpler than Gemma 4's:

  per-block:  attn_norm (pre)  →  attn_q/k/v  →  attn_output
              ffn_norm  (pre)  →  ffn_gate/up →  ffn_down
  global:     output_norm (pre)  →  output

  off-axis (head_dim, not d):  attn_q_norm, attn_k_norm  — unchanged
  no post-norms, no PLE, no Gemma-4-specific complications

  output.weight is already separate from token_embd.weight on disk.

§5.2.5 fuse-and-scale is mathematically exact at every (pre-)norm site,
so this rewriter should produce coherent output in keymat mode.

Modes:

  identity-pad   P̂_R = [I_d | 0]: zero padding. Math identical to
                 plaintext through the padded structure. Validates dim
                 plumbing only.

  gamma-only     §5.2.5 fusion only, κ = 1, no dim expansion. Must
                 produce **bit-identical** output to plaintext (modulo
                 fp16 quantisation). This is the fusion-correctness
                 regression test.

  keymat         Full Algorithm 1 obfuscation with §5.2.5 fusion.
                 Real per-batch-static AloePri without Π / Algorithm 2
                 attention transforms / noise yet — those come in
                 follow-on revisions.

Usage:
  python obfuscate_qwen3_gguf.py \\
    --in /path/to/plaintext-qwen3.gguf \\
    --out /path/to/obfuscated.gguf \\
    --mode {identity-pad|gamma-only|keymat} \\
    --expansion-size 128 --seed 42
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import gguf
import gguf.quants as gquants

from lib import alg2

log = logging.getLogger("obfuscate_qwen3_gguf")


# ────────────────────────────────────────────────────────────────────
# tensor classification (Qwen3 dense)
# ────────────────────────────────────────────────────────────────────

# ne[0] = d, READS from residual. Transform: arr_new = arr_old @ Q̂_R^T.
RESIDUAL_NE0_READ_TENSORS = {
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "output.weight",         # LM head: reads residual, outputs logits
}

# ne[0] = d, WRITES to residual (embedding lookup). Transform: arr_old @ P̂_R.
RESIDUAL_NE0_WRITE_TENSORS = {
    "token_embd.weight",
}

# ne[1] = d, WRITES to residual. Transform: arr_new = P̂_R^T @ arr_old.
RESIDUAL_NE1_TENSORS = {
    "attn_output.weight",
    "ffn_down.weight",
}

# 1-D γ vectors on the residual stream — all pre-norms in Qwen3.
RESIDUAL_NORM_TENSORS = {
    "attn_norm.weight",
    "ffn_norm.weight",
    "output_norm.weight",
}

# §5.2.5 norm-to-linear fusion. Pre-norms only — Qwen3 has nothing else.
PER_BLOCK_FUSION_MAP: dict[str, tuple[str, list[str]]] = {
    "attn_norm.weight": ("pre", ["attn_q.weight", "attn_k.weight", "attn_v.weight"]),
    "ffn_norm.weight":  ("pre", ["ffn_gate.weight", "ffn_up.weight"]),
}

GLOBAL_FUSION_MAP: dict[str, tuple[str, list[str]]] = {
    "output_norm.weight": ("pre", ["output.weight"]),
}

# Unchanged (off-residual): attn_q_norm, attn_k_norm operate on head_dim.


def stripped_tensor_name(full_name: str) -> str:
    if full_name.startswith("blk."):
        parts = full_name.split(".", 2)
        if len(parts) == 3:
            return parts[2]
    return full_name


def classify(full_name: str) -> str:
    stripped = stripped_tensor_name(full_name)
    if stripped in RESIDUAL_NE0_READ_TENSORS:
        return "ne0_read"
    if stripped in RESIDUAL_NE0_WRITE_TENSORS:
        return "ne0_write"
    if stripped in RESIDUAL_NE1_TENSORS:
        return "ne1"
    if stripped in RESIDUAL_NORM_TENSORS:
        return "norm"
    return "unchanged"


# ────────────────────────────────────────────────────────────────────
# dequant helpers
# ────────────────────────────────────────────────────────────────────


def to_float_array(t: gguf.ReaderTensor) -> np.ndarray:
    qtype = t.tensor_type
    natural_shape = tuple(int(s) for s in reversed(t.shape))
    if qtype == gguf.GGMLQuantizationType.F32:
        return np.frombuffer(t.data, dtype=np.float32).reshape(natural_shape)
    if qtype == gguf.GGMLQuantizationType.F32:
        return np.frombuffer(t.data, dtype=np.float32).astype(np.float32).reshape(natural_shape)
    if qtype == gguf.GGMLQuantizationType.BF16:
        u16 = np.frombuffer(t.data, dtype=np.uint16)
        f32 = np.zeros(u16.shape, dtype=np.uint32)
        f32 |= u16.astype(np.uint32) << 16
        return f32.view(np.float32).reshape(natural_shape)
    raw = np.array(t.data)
    arr = gquants.dequantize(raw, qtype)
    return arr.reshape(natural_shape)


# ────────────────────────────────────────────────────────────────────
# transforms
# ────────────────────────────────────────────────────────────────────


def expand_axis_with_zeros(arr: np.ndarray, axis: int, d: int, expansion: int) -> np.ndarray:
    if arr.shape[axis] != d:
        raise ValueError(f"axis {axis} of shape {arr.shape} is {arr.shape[axis]}, expected {d}")
    pad_shape = list(arr.shape)
    pad_shape[axis] = expansion
    return np.concatenate([arr, np.zeros(tuple(pad_shape), dtype=arr.dtype)], axis=axis)


def identity_pad(arr: np.ndarray, cls: str, d: int, pad: int) -> np.ndarray:
    if cls in ("ne0_read", "ne0_write", "norm"):
        return expand_axis_with_zeros(arr, axis=-1, d=d, expansion=pad)
    if cls == "ne1":
        return expand_axis_with_zeros(arr, axis=0, d=d, expansion=pad)
    raise AssertionError(cls)


def fp32_to_bf16(arr: np.ndarray) -> np.ndarray:
    """fp32 → bf16 with round-to-nearest-even (matches IEEE-754 RTNE).

    Returns a uint16 ndarray (same shape as input) holding the bf16 bit
    pattern. Pass to `GGUFWriter.add_tensor` together with
    `raw_dtype=GGMLQuantizationType.BF16`.

    Why bf16 for AloePri-obfuscated GGUFs: the keymat × λ-perturbation
    construction introduces a long lower tail (~1e-9 magnitudes) in the
    transformed weight tensors. fp16's smallest normal is ~6e-5, so it
    flushes ~1 %% of attn_q entries to zero and breaks the P̂·Q̂ = I_d
    cancellation. bf16 keeps fp32's 8-bit exponent (range ~1e-38) so the
    cancellation survives; only mantissa precision is reduced. Empirically
    matches fp32 HumanEval pass@1 at half the file size and 2× the decode
    throughput.
    """
    assert arr.dtype == np.float32, f"expected fp32 input, got {arr.dtype}"
    u32 = arr.view(np.uint32)
    # RTNE: bias = 0x7fff + lsb_of_truncated. Handles ties correctly.
    rounded = (u32 + 0x7fff + ((u32 >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


def write_tensor_typed(writer: gguf.GGUFWriter, name: str, arr: np.ndarray,
                       output_dtype: str, force_fp32: bool = False) -> None:
    """Write `arr` to `writer` at the requested precision.

    Norm-class tensors (1D vectors, matrix-Γ Γ_q/Γ_k) must stay F32 for
    precision; pass `force_fp32=True` for those. Everything else honours
    `output_dtype` ∈ {'fp32', 'bf16'}.
    """
    arr = arr.astype(np.float32, copy=False)
    if force_fp32 or output_dtype == "fp32":
        writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
    elif output_dtype == "bf16":
        writer.add_tensor(name, fp32_to_bf16(arr),
                          raw_dtype=gguf.GGMLQuantizationType.BF16)
    else:
        raise ValueError(f"unsupported output_dtype: {output_dtype!r}")


def fuse_gamma_pre(linear_arr: np.ndarray, gamma: np.ndarray, kappa: float) -> np.ndarray:
    """Pre-norm fusion: γ row-scales the *following* linear's d-axis.

    Replaces [RMSNorm(γ_per_dim) → Linear(W)] with [RMSNorm(κ·1) → Linear(W')]
    where W' = Diag(γ_per_dim) · W (i.e. γ baked into W as-is, NO κ scaling on γ).

    Verification of cancellation under obfuscation (x_obf = x · P̂_R):
        norm(x_obf; κ·1) · W̃' = κ · (x_obf · Q̂_R) · γ · W / RMS(x_obf)
                              = κ · (x · γ) · W / RMS(x_obf)
        plaintext target:    = (x · γ) · W / RMS(x)
        ratio                = κ · RMS(x)/RMS(x_obf) = κ / κ_correct = 1 when κ = κ_correct.

    In numpy with W natural shape (M, d): W_new = W * γ (broadcast on axis 0).
    The κ correction lives in the *norm site's* scalar γ_obf = κ_correct, NOT
    in the fusion step. kappa is unused here, kept in the signature for clarity."""
    del kappa  # the κ correction lives on the norm side, not in the fusion
    return linear_arr * gamma.astype(linear_arr.dtype)


def estimate_kappa(p_r: np.ndarray, d: int, num_samples: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_samples, d)).astype(np.float64)
    y = x @ p_r.astype(np.float64)
    ratio = np.linalg.norm(y, axis=-1) / np.linalg.norm(x, axis=-1)
    return float(ratio.mean())


# ────────────────────────────────────────────────────────────────────
# main pipeline
# ────────────────────────────────────────────────────────────────────


def rewrite_gguf(
    in_path: Path,
    out_path: Path,
    mode: str,
    expansion: int,
    seed: int,
    lam: float = 0.3,
    apply_pi: bool = False,
    pi_seed: int = 0,
    pi_include_specials: bool = False,
    key_out: Path | None = None,
    noise_alpha_e: float = 0.0,
    noise_alpha_h: float = 0.0,
    noise_seed: int = 0,
    apply_alg2: bool = False,
    alg2_seed: int = 0,
    alg2_beta: int = 8,
    alg2_gamma: float = 1e3,
    alg2_qk_scale_range: tuple[float, float] = (0.95, 1.05),
    # When True: bake real R̂_qk intra-head transform into W_q/W_k output
    # axis AND replace 1D γ_q/γ_k with 2D Γ = M_qᵀ · Diag(γ) · M_q.
    # Requires the patched llama.cpp kernel (aloepri_qk_norm_matrix
    # branch) — see docs/research/aloepri-qk-norm-matrix-gamma-threat-model.md.
    # Forces qk_scale_range to (1.0, 1.0) so M_q stays orthogonal.
    alg2_qk_norm_matrix: bool = False,
    # When True (in combination with alg2_qk_norm_matrix): use ±1
    # Walsh-Hadamard for Ĥ_qk instead of identity. Keeps M_q
    # orthogonal (since ±1 diag is involutive). Adds per-pair sign
    # flips to the obfuscation; M_q stays equal to M_k (because
    # H = H⁻¹ for ±1 diag), so Q/K cancel cleanly in attention.
    alg2_h_hadamard_signs: bool = False,
    # When True: enable paper §5.2.3 Û_vo random projection on the V↔O
    # composition. V output is right-multiplied by Û_vo per head; W_o
    # input axis is right-multiplied by Û_vo⁻¹ block-diag per head.
    # The two cancel through attention (V̄ · W̃_o^T = V · W_o^T) so
    # residual covariance is preserved, but per-head V outputs and
    # W_o cols carry deployment-specific randomness that ISA
    # HiddenState ridge cannot extract. Closes paper Table 4's
    # 0.82 %→0.0 % gap (Noise+KeyMat vs Noise+KeyMat+Head&BlockPerm).
    # Default off for backward-compat with prior deployments.
    alg2_enable_u_vo: bool = False,
    # Output precision for the large matmul tensors (token_embd,
    # output, attn_*, ffn_*). Norm tensors and the matrix-Γ Γ_q/Γ_k
    # tensors always stay F32 regardless of this setting. Default
    # bf16: half the on-disk size and 2× decode throughput vs fp32,
    # accuracy-equivalent on the obfuscated keymat chain. fp16 is
    # NOT safe — see fp32_to_bf16 docstring.
    output_dtype: str = "bf16",
) -> dict:
    log.info("opening %s (mode=%s pi=%s)", in_path, mode, apply_pi)
    r = gguf.GGUFReader(str(in_path))

    arch = r.fields["general.architecture"].contents()
    if arch != "qwen3":
        raise SystemExit(f"unsupported architecture: {arch} (expected qwen3)")
    d = int(r.fields["qwen3.embedding_length"].contents())
    n_layer = int(r.fields["qwen3.block_count"].contents())

    if mode == "gamma-only":
        pad = 0
    else:
        pad = 2 * expansion
    new_d = d + pad
    log.info("arch=%s d=%d 2h=%d new_d=%d n_layer=%d", arch, d, pad, new_d, n_layer)

    # ---- key material (only for keymat mode) ----
    p_r = None
    q_r = None
    q_r_t = None
    p_r_t = None
    kappa = 1.0
    kappa_e = None
    if mode == "keymat":
        sys.path.insert(0, "/home/timo/repos/private-rag-path-2/vendor/aloepri-py")
        sys.path.insert(0, "/home/timo/repos/private-rag-path-2/vendor/aloepri-py/src")
        from keymat import build_keymat_transform  # type: ignore

        log.info("sampling Algorithm 1 at d=%d h=%d λ=%.2f seed=%d", d, expansion, lam, seed)
        transform = build_keymat_transform(d=d, h=expansion, lam=lam, init_seed=seed)
        # Sample at fp64 for identity-residual check + κ estimate (cheap ops),
        # then keep fp32 copies for the per-tensor matmuls in pass 3. Holding
        # only fp32 versions saves ~70 MB × 4 = 280 MB at the largest model
        # scale but more importantly avoids per-tensor `.astype(np.float32)`
        # scratch which compounds across the 36 layers × ~10 tensors.
        p_r_64 = transform.key.numpy().astype(np.float64)
        q_r_64 = transform.inverse.numpy().astype(np.float64)
        identity_err = float(np.max(np.abs(p_r_64 @ q_r_64 - np.eye(d))))
        log.info("‖P̂·Q̂ - I_d‖_max = %.3e", identity_err)
        kappa_e = estimate_kappa(p_r_64, d=d, num_samples=2000, seed=seed + 100)
        kappa = kappa_e * float(np.sqrt(d / float(d + 2 * expansion)))
        log.info("κ_E = %.5f  →  κ_correct = κ_E · √(d/(d+2h)) = %.5f", kappa_e, kappa)
        p_r = p_r_64.astype(np.float32)
        q_r = q_r_64.astype(np.float32)
        q_r_t = q_r.T  # (d, d+2h)  fp32
        p_r_t = p_r.T  # (d+2h, d)  fp32
        del p_r_64, q_r_64

    # ---- pass 1: load every tensor ----
    log.info("loading + dequantising tensors...")
    arrays: dict[str, np.ndarray] = {t.name: to_float_array(t) for t in r.tensors}
    log.info("loaded %d tensors", len(arrays))

    # ---- §5.2.2: additive Gaussian noise on embed + head ----
    # Applied BEFORE Π and keymat. Π is a row permutation so it commutes
    # with iid noise; applying earlier keeps the noise i.i.d. across
    # plaintext-space rows which is what paper §5.2.2 specifies.
    noise_info: dict = {}
    if noise_alpha_e > 0.0 or noise_alpha_h > 0.0:
        noise_rng = np.random.default_rng(noise_seed)
        if noise_alpha_e > 0.0 and "token_embd.weight" in arrays:
            we = arrays["token_embd.weight"]
            sigma_e = float(np.std(we))
            eps_e = noise_rng.standard_normal(we.shape).astype(we.dtype)
            arrays["token_embd.weight"] = we + np.float32(noise_alpha_e * sigma_e) * eps_e
            noise_info["embed"] = {"alpha": noise_alpha_e, "sigma": sigma_e}
            log.info("embed noise α_e=%.3f σ_e=%.5f", noise_alpha_e, sigma_e)
        if noise_alpha_h > 0.0 and "output.weight" in arrays:
            wh = arrays["output.weight"]
            sigma_h = float(np.std(wh))
            eps_h = noise_rng.standard_normal(wh.shape).astype(wh.dtype)
            arrays["output.weight"] = wh + np.float32(noise_alpha_h * sigma_h) * eps_h
            noise_info["head"] = {"alpha": noise_alpha_h, "sigma": sigma_h}
            log.info("head noise α_h=%.3f σ_h=%.5f", noise_alpha_h, sigma_h)

    # ---- Π: row-permute token_embd and output by τ⁻¹ on the vocab axis ----
    tau: np.ndarray | None = None
    pi_active_size: int | None = None
    pi_special_ids: list[int] = []
    if apply_pi:
        if "token_embd.weight" not in arrays:
            raise SystemExit("token_embd.weight missing — cannot apply Π")
        n_vocab = int(arrays["token_embd.weight"].shape[0])
        # Restrict τ to the tokenizer's *active* range. Slots above that
        # are GGUF padding (empty strings), kept identity so the model
        # cannot sample an obf_id whose τ⁻¹ maps out of decodable range.
        # For Qwen3 1.7B the active range is 151669 — hardcoded for now.
        pi_active_size = 151669
        if pi_active_size > n_vocab:
            raise SystemExit(f"active range {pi_active_size} > n_vocab {n_vocab}")

        # Exclude special tokens (EOS / BOS / im_start / im_end / fim_*
        # / tool markers etc.) from Π. They stay at identity so the
        # inference server's standard stop-on-EOS / chat-template
        # plumbing keeps working — without this, the model emits
        # `inv_τ[151645]` to mean "stop" but the server is looking for
        # `151645` in the wire stream, generation runs to max_tokens,
        # the model drifts off-manifold, and HumanEval pass@1 collapses
        # (see 2026-05-20 sweep diagnosis under `evals/aloepri-attacks/
        # results/sweep/`). Public knowledge — token-type metadata is
        # already exposed in the GGUF and reveals nothing useful to an
        # attacker; the privacy guarantee comes from permuting the
        # *content-bearing* tokens.
        token_type_field = r.fields.get("tokenizer.ggml.token_type")
        if token_type_field is None:
            raise SystemExit(
                "tokenizer.ggml.token_type missing — needed to exclude "
                "special tokens from Π. Re-source the GGUF with full tokenizer "
                "metadata or set --pi-active-size explicitly."
            )
        token_types = np.asarray(token_type_field.contents(), dtype=np.int32)
        # llama.cpp token type codes: 1=NORMAL, 2=UNKNOWN, 3=CONTROL,
        # 4=USER_DEFINED, 5=UNUSED, 6=BYTE.
        #
        # Default mode (pi_include_specials=False): permute only NORMAL +
        # BYTE so the server's stop-on-EOS / chat-template plumbing keeps
        # working without client-side `ignore_eos`. Leaks ~293 identity-
        # fixed pairs to a passive attacker who reads this source file.
        #
        # Strong mode (pi_include_specials=True): permute everything in
        # `[0, pi_active_size)`. Closes the structural leak. Requires
        # the client to set `ignore_eos: True` and bytes-stream the
        # response — otherwise the server's EOS detection fires on the
        # wrong (permuted) id and generation runs to max_tokens with
        # multi-language drift output that may crash the chat-parser
        # PEG.
        if pi_include_specials:
            permutable_ids = np.arange(pi_active_size, dtype=np.int32)
            pi_special_ids = []
            log.info(
                "Π strong mode: permuting all %d ids in active range; "
                "no identity-fixed specials", pi_active_size,
            )
        else:
            permutable_mask = np.isin(token_types[:pi_active_size], [1, 6])
            permutable_ids = np.where(permutable_mask)[0].astype(np.int32)
            pi_special_ids = sorted(set(range(pi_active_size)) - set(permutable_ids.tolist()))
            log.info(
                "Π special-token exclusion: %d permutable, %d kept identity "
                "(non-NORMAL/BYTE token-type) within active range %d",
                len(permutable_ids), len(pi_special_ids), pi_active_size,
            )

        pi_rng = np.random.default_rng(pi_seed)
        # Permute only `permutable_ids` among themselves. tau starts at
        # identity, then we shuffle the permutable subset by reassigning
        # tau[permutable_ids] = shuffled(permutable_ids).
        tau = np.arange(n_vocab, dtype=np.int32)
        shuffled = pi_rng.permutation(permutable_ids).astype(np.int32)
        tau[permutable_ids] = shuffled
        inv_tau = np.argsort(tau).astype(np.int32)
        # τ : plain_id → obf_id ;  inv_tau : obf_id → plain_id.
        # W̃[i, :] = W[inv_tau[i], :] so the obfuscated table at obf_id i
        # serves the original embedding of plain_id inv_tau[i].
        for vocab_tensor in ("token_embd.weight", "output.weight"):
            if vocab_tensor not in arrays:
                continue
            assert arrays[vocab_tensor].shape[0] == n_vocab
            arrays[vocab_tensor] = arrays[vocab_tensor][inv_tau]
        log.info("applied Π (τ pi_seed=%d, active=%d/%d, permuted=%d, "
                 "specials-identity=%d) to %d vocab tensors",
                 pi_seed, pi_active_size, n_vocab,
                 len(permutable_ids), len(pi_special_ids),
                 sum(1 for t in ("token_embd.weight", "output.weight") if t in arrays))

    # ---- Algorithm 2 prep: per-layer keys (item 7) ----
    #
    # Qwen3-specific restriction: paper §5.2.3's intra-head transforms
    # (R̂_qk, Ĥ_qk, Ẑ_block) require fusing γ_qk into W_q/W_k via §5.2.5.
    # That fusion is mathematically exact only under i.i.d. Gaussian-input
    # assumptions on Q/K. Empirically, Qwen3's per-head-dim γ_q/γ_k have
    # high variance and the per-input bias of κ ≈ √(mean(γ²)) breaks the
    # model (smoke tests degenerate to high-frequency-token loops).
    #
    # So on Qwen3 we apply ONLY the inter-head shuffle (τ_kv, τ_group).
    # Head-shuffle is a row permutation across whole heads — γ_qk (per
    # head_dim) is broadcast across heads and is preserved by the
    # shuffle. No γ_qk modification needed. This loses the R̂_qk/Ĥ_qk/
    # Ẑ_block components of the paper's Algorithm 2 (a real loss of
    # ISA defense) but keeps the model working. Documented in
    # docs/plans/path-2-status.md as a Qwen3-specific divergence from
    # the paper's full Algorithm 2.
    alg2_per_layer: dict[int, alg2.LayerAlg2Keys] = {}
    alg2_q_feature_orders: dict[int, np.ndarray] = {}
    alg2_kv_feature_orders: dict[int, np.ndarray] = {}
    n_q_heads = n_kv_heads = head_dim_a = num_groups_a = 0
    if apply_alg2:
        n_q_heads = int(r.fields["qwen3.attention.head_count"].contents())
        n_kv_heads = int(r.fields["qwen3.attention.head_count_kv"].contents())
        head_dim_a = int(r.fields["qwen3.attention.key_length"].contents())
        num_groups_a = n_q_heads // n_kv_heads
        rope_base = float(r.fields["qwen3.rope.freq_base"].contents())
        if alg2_qk_norm_matrix:
            # M_q must be orthogonal for the matrix-Γ kernel algebra to be
            # exact. Either force Ĥ_qk = I (scale_range=(1.0,1.0)) or use
            # ±1 Walsh-Hadamard (involutive, still orthogonal).
            # See docs/research/aloepri-qk-norm-matrix-gamma-threat-model.md.
            if not alg2_h_hadamard_signs:
                alg2_qk_scale_range = (1.0, 1.0)
            log.info(
                "alg2: full intra-head + matrix-Γ QK-norm. "
                "n_q=%d n_kv=%d head_dim=%d groups=%d  Ĥ=%s",
                n_q_heads, n_kv_heads, head_dim_a, num_groups_a,
                "Walsh-Hadamard ±1" if alg2_h_hadamard_signs else "I",
            )
        else:
            log.info(
                "alg2: head-shuffle only (Qwen3 QK-norm blocks intra-head). "
                "n_q=%d n_kv=%d head_dim=%d groups=%d",
                n_q_heads, n_kv_heads, head_dim_a, num_groups_a,
            )
        for il in range(n_layer):
            full_keys = alg2.build_layer_keys(
                head_dim=head_dim_a,
                num_kv_heads=n_kv_heads,
                num_groups=num_groups_a,
                seed=alg2_seed + il * 1000,
                qk_scale_range=alg2_qk_scale_range,
                beta=alg2_beta,
                gamma=alg2_gamma,
                rope_base=rope_base,
                h_hadamard_signs=alg2_h_hadamard_signs,
                enable_u_vo=alg2_enable_u_vo,
            )
            if alg2_qk_norm_matrix:
                keys = full_keys
            else:
                # Legacy: head-shuffle only; intra-head q/k_matrix → I.
                keys = alg2.LayerAlg2Keys(
                    q_matrix=np.eye(head_dim_a, dtype=np.float32),
                    k_matrix=np.eye(head_dim_a, dtype=np.float32),
                    tau_kv=full_keys.tau_kv,
                    inv_tau_kv=full_keys.inv_tau_kv,
                    tau_group=full_keys.tau_group,
                    inv_tau_group=full_keys.inv_tau_group,
                )
            alg2_per_layer[il] = keys
            q_head_order = alg2._query_head_order(
                n_q_heads, n_kv_heads, num_groups_a, keys.tau_kv, keys.tau_group
            )
            kv_head_order = alg2._kv_head_order(n_kv_heads, keys.tau_kv)
            alg2_q_feature_orders[il] = alg2._expand_feature_order(q_head_order, head_dim_a)
            alg2_kv_feature_orders[il] = alg2._expand_feature_order(kv_head_order, head_dim_a)
        log.info("alg2: head-shuffle keys generated for %d layers", n_layer)

    # ---- pass 2 (keymat / gamma-only): §5.2.5 fusion ----
    if mode in ("keymat", "gamma-only"):
        log.info("§5.2.5 fusion: γ → adjacent linears")
        for il in range(n_layer):
            for norm_name, (direction, targets) in PER_BLOCK_FUSION_MAP.items():
                gamma_key = f"blk.{il}.{norm_name}"
                if gamma_key not in arrays:
                    continue
                gamma = arrays[gamma_key]
                for tgt_name in targets:
                    tgt_key = f"blk.{il}.{tgt_name}"
                    if tgt_key not in arrays:
                        continue
                    assert direction == "pre"
                    arrays[tgt_key] = fuse_gamma_pre(arrays[tgt_key], gamma, kappa)
        for norm_name, (direction, targets) in GLOBAL_FUSION_MAP.items():
            if norm_name not in arrays:
                continue
            gamma = arrays[norm_name]
            for tgt_name in targets:
                if tgt_name not in arrays:
                    continue
                assert direction == "pre"
                arrays[tgt_name] = fuse_gamma_pre(arrays[tgt_name], gamma, kappa)
        log.info("γ fusion complete")

    # ---- writer setup + metadata ----
    log.info("applying %s transform to residual-stream tensors", mode)
    writer = gguf.GGUFWriter(str(out_path), arch=arch)
    skipped_keys = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
    for key in sorted(r.fields.keys()):
        if key in skipped_keys:
            continue
        field = r.fields[key]
        value = field.contents()
        if key == "qwen3.embedding_length" and new_d != d:
            value = new_d
            log.info("metadata: %s = %d (was %d)", key, value, d)
        _write_field(writer, key, value, field)
    writer.add_string("aloepri.mode", mode)
    writer.add_uint32("aloepri.expansion_size", expansion if mode != "gamma-only" else 0)
    writer.add_uint32("aloepri.seed", seed)
    writer.add_bool("aloepri.pi_applied", bool(apply_pi))
    writer.add_float32("aloepri.noise_alpha_e", float(noise_alpha_e))
    writer.add_float32("aloepri.noise_alpha_h", float(noise_alpha_h))
    writer.add_bool("aloepri.alg2_applied", bool(apply_alg2))
    if apply_alg2:
        writer.add_uint32("aloepri.alg2_beta", int(alg2_beta))
        writer.add_float32("aloepri.alg2_gamma", float(alg2_gamma))
        writer.add_bool("aloepri.qk_norm_matrix", bool(alg2_qk_norm_matrix))
        writer.add_bool("aloepri.alg2_u_vo_applied", bool(alg2_enable_u_vo))
    if mode == "keymat":
        writer.add_float32("aloepri.kappa_e", float(kappa_e))
        writer.add_float32("aloepri.kappa", float(kappa))
        writer.add_float32("aloepri.lambda", float(lam))

    # ---- pass 3: apply transforms + write ----
    n_changed = 0
    n_unchanged = 0
    for t in r.tensors:
        name = t.name
        arr = arrays[name]
        cls = classify(name)

        # Matrix-Γ form: intercept attn_{q,k}_norm.weight BEFORE the cls
        # dispatch so we replace the 1D γ tensor with the 2D Γ = MᵀDM.
        # Otherwise `cls == "unchanged"` writes the 1D tensor and continues.
        if (
            alg2_qk_norm_matrix
            and apply_alg2
            and name.startswith("blk.")
        ):
            stripped = stripped_tensor_name(name)
            if stripped in ("attn_q_norm.weight", "attn_k_norm.weight"):
                layer_idx = int(name.split(".", 2)[1])
                keys = alg2_per_layer.get(layer_idx)
                assert keys is not None, f"missing alg2 keys for layer {layer_idx}"
                gamma = arr.astype(np.float64)
                assert gamma.ndim == 1 and gamma.shape[0] == head_dim_a, \
                    f"unexpected {stripped} shape {gamma.shape}"
                M = (keys.q_matrix if stripped == "attn_q_norm.weight"
                     else keys.k_matrix).astype(np.float64)
                gamma_matrix = (M.T @ np.diag(gamma) @ M).astype(np.float32)
                # Γ_q / Γ_k is the matrix-Γ tensor — orthogonality
                # precision matters; keep it F32 regardless of output_dtype.
                write_tensor_typed(writer, name, gamma_matrix,
                                   output_dtype, force_fp32=True)
                n_changed += 1
                continue

        if cls == "unchanged":
            # Most "unchanged" tensors in Qwen3 are 1D norm / position
            # tensors. Keep at F32 — they are small, precision-sensitive,
            # and not on the bandwidth-bound critical path.
            write_tensor_typed(writer, name, arr, output_dtype, force_fp32=True)
            n_unchanged += 1
            continue

        if cls == "norm":
            if mode == "identity-pad":
                out_arr = expand_axis_with_zeros(arr, axis=-1, d=d, expansion=pad).astype(np.float32)
            elif mode == "keymat":
                out_arr = np.full((new_d,), float(kappa), dtype=np.float32)
            else:  # gamma-only
                out_arr = np.full((d,), float(kappa), dtype=np.float32)
            # Norm γ is a per-channel scale; 7 bits of mantissa is too
            # coarse, keep F32.
            write_tensor_typed(writer, name, out_arr, output_dtype, force_fp32=True)
            n_changed += 1
            continue

        # ne0_read / ne0_write / ne1
        if mode == "identity-pad":
            out_arr = identity_pad(arr, cls, d=d, pad=pad).astype(np.float32)
        elif mode == "keymat":
            # Keymat matmul in fp32. Previously upcast to fp64 + downcast,
            # which created a 3× peak per tensor (4-byte → 8-byte source +
            # 8-byte output + 4-byte downcast). Load-bearing for 8B+ where
            # the fp32 dict alone is ~32 GB and the fp64 scratch on the
            # token_embd row was an extra ~7 GB peak per such tensor.
            # Precision error from fp64→fp32 was already discarded at the
            # final cast; the |P̂·Q̂ - I_d| residual stays at ~1e-7.
            arr_f32 = arr.astype(np.float32, copy=False)
            if cls == "ne0_read":
                out_arr = arr_f32 @ q_r_t
            elif cls == "ne0_write":
                out_arr = arr_f32 @ p_r
            elif cls == "ne1":
                out_arr = p_r_t @ arr_f32
            else:
                raise AssertionError(cls)
            # Drop the source tensor — it's been transformed; not referenced
            # again. Saves the duplicate of the largest tensors at peak.
            arrays[name] = None  # type: ignore[assignment]
            del arr_f32
        else:  # gamma-only — fuse already applied, no dim change, no Algorithm 1
            out_arr = arr.astype(np.float32)

        # ---- Algorithm 2: per-attention-tensor head_dim-axis transform ----
        if apply_alg2:
            stripped = stripped_tensor_name(name)
            if name.startswith("blk."):
                layer_idx = int(name.split(".", 2)[1])
                keys = alg2_per_layer.get(layer_idx)
                if keys is not None and stripped in (
                    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"
                ):
                    q_feat = alg2_q_feature_orders[layer_idx]
                    kv_feat = alg2_kv_feature_orders[layer_idx]
                    # Intra-head dense transform: block-diag M repeated per head.
                    # When alg2_qk_norm_matrix=False, keys.q_matrix = I — the
                    # dense_transform reduces to identity, so passing it is a
                    # no-op equivalent to passing None.
                    q_dense = alg2._block_diag_repeat(keys.q_matrix, n_q_heads) \
                        if alg2_qk_norm_matrix else None
                    k_dense = alg2._block_diag_repeat(keys.k_matrix, n_kv_heads) \
                        if alg2_qk_norm_matrix else None
                    # V dense transform: Û_vo block-diag per KV head when
                    # enable_u_vo. Without it V is unobfuscated on head_dim.
                    v_dense = alg2._block_diag_repeat(keys.u_vo, n_kv_heads) \
                        if (alg2_enable_u_vo and keys.u_vo is not None) else None
                    # O input-axis transform: Û_vo⁻¹ block-diag per Q head.
                    o_input_dense = alg2._block_diag_repeat(keys.u_vo_inv, n_q_heads) \
                        if (alg2_enable_u_vo and keys.u_vo_inv is not None) else None
                    if stripped == "attn_q.weight":
                        out_arr = alg2.apply_qkv_output_transform(out_arr, q_dense, q_feat)
                    elif stripped == "attn_k.weight":
                        out_arr = alg2.apply_qkv_output_transform(out_arr, k_dense, kv_feat)
                    elif stripped == "attn_v.weight":
                        out_arr = alg2.apply_qkv_output_transform(out_arr, v_dense, kv_feat)
                    elif stripped == "attn_output.weight":
                        out_arr = alg2.apply_o_output_transform(
                            out_arr, q_feat, dense_input_transform=o_input_dense,
                        )

        # Large matmul tensors: honour output_dtype. bf16 is the default
        # because the AloePri keymat construction creates ~1e-9 tail
        # values that survive bf16's 1e-38 exponent floor but flush to
        # zero under fp16's 6e-5 floor (breaks P̂·Q̂ = I_d cancellation).
        write_tensor_typed(writer, name, out_arr, output_dtype)
        n_changed += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    log.info("wrote %s (changed=%d unchanged=%d)", out_path, n_changed, n_unchanged)

    if apply_pi:
        assert tau is not None
        key_path = key_out if key_out is not None else out_path.with_suffix(out_path.suffix + ".key.npz")
        save_kwargs: dict = dict(
            tau=tau,
            pi_seed=np.int64(pi_seed),
            vocab_size=np.int64(tau.shape[0]),
            active_size=np.int64(pi_active_size if pi_active_size else tau.shape[0]),
            arch=np.array(arch),
            version=np.int32(2 if apply_alg2 else 1),
        )
        if apply_alg2:
            # Per-layer alg2 keys — not used by the client (which only needs τ
            # for token-level I/O) but saved for reproducibility / attack-bench.
            save_kwargs["alg2_applied"] = np.array(True)
            save_kwargs["alg2_seed"] = np.int64(alg2_seed)
            save_kwargs["alg2_n_q_heads"] = np.int64(n_q_heads)
            save_kwargs["alg2_n_kv_heads"] = np.int64(n_kv_heads)
            save_kwargs["alg2_head_dim"] = np.int64(head_dim_a)
            for il, keys in alg2_per_layer.items():
                save_kwargs[f"alg2_l{il}_q_matrix"] = keys.q_matrix
                save_kwargs[f"alg2_l{il}_k_matrix"] = keys.k_matrix
                if keys.tau_kv is not None:
                    save_kwargs[f"alg2_l{il}_tau_kv"] = keys.tau_kv
                if keys.tau_group is not None:
                    save_kwargs[f"alg2_l{il}_tau_group"] = keys.tau_group
        np.savez_compressed(key_path, **save_kwargs)
        try:
            key_path.chmod(0o600)
        except OSError:
            pass
        log.info("wrote key %s (size=%d)", key_path, key_path.stat().st_size)

    return {"mode": mode, "d": d, "new_d": new_d, "n_changed": n_changed, "kappa": kappa,
            "pi_applied": apply_pi, "pi_seed": pi_seed if apply_pi else None,
            "noise": noise_info}


def _write_field(writer: gguf.GGUFWriter, key: str, value, field: gguf.ReaderField) -> None:
    types = field.types
    primary = types[0]
    if primary == gguf.GGUFValueType.STRING:
        writer.add_string(key, value)
    elif primary == gguf.GGUFValueType.BOOL:
        writer.add_bool(key, bool(value))
    elif primary == gguf.GGUFValueType.UINT8:    writer.add_uint8(key, int(value))
    elif primary == gguf.GGUFValueType.INT8:     writer.add_int8(key, int(value))
    elif primary == gguf.GGUFValueType.UINT16:   writer.add_uint16(key, int(value))
    elif primary == gguf.GGUFValueType.INT16:    writer.add_int16(key, int(value))
    elif primary == gguf.GGUFValueType.UINT32:   writer.add_uint32(key, int(value))
    elif primary == gguf.GGUFValueType.INT32:    writer.add_int32(key, int(value))
    elif primary == gguf.GGUFValueType.UINT64:   writer.add_uint64(key, int(value))
    elif primary == gguf.GGUFValueType.INT64:    writer.add_int64(key, int(value))
    elif primary == gguf.GGUFValueType.FLOAT32:  writer.add_float32(key, float(value))
    elif primary == gguf.GGUFValueType.FLOAT64:  writer.add_float64(key, float(value))
    elif primary == gguf.GGUFValueType.ARRAY:    writer.add_array(key, list(value))
    else:
        raise ValueError(f"unsupported gguf value type {primary} for key {key}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, required=True)
    parser.add_argument("--out", dest="out_path", type=Path, required=True)
    parser.add_argument("--mode", choices=["identity-pad", "keymat", "gamma-only"], default="keymat")
    parser.add_argument("--expansion-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lam", type=float, default=0.3)
    parser.add_argument("--pi", action="store_true",
                        help="apply Π token-permutation to token_embd + output (item 6).")
    parser.add_argument("--pi-seed", type=int, default=42424242,
                        help="seed for τ generation (kept out of GGUF metadata).")
    parser.add_argument("--pi-include-specials", action="store_true",
                        help="Include CONTROL/USER_DEFINED/UNKNOWN/UNUSED token types "
                             "in the Π permutation (default: keep them at identity for "
                             "server stop-on-EOS compatibility). Closes the ~293-pair "
                             "structural leak that lets an attacker fit a partial-τ "
                             "ridge inverter from publicly-known identity-fixed ids. "
                             "Requires the client to pass `ignore_eos: True` and "
                             "stream-decode the response — without that, the server "
                             "won't stop on EOS and may emit chat-template gibberish "
                             "that crashes the PEG chat-parser.")
    parser.add_argument("--key-out", type=Path, default=None,
                        help="path for τ key file (defaults to <out>.key.npz).")
    parser.add_argument("--noise-alpha-e", type=float, default=0.0,
                        help="paper §5.2.2 W_e Gaussian noise scale (0.0 disables; paper default 1.0).")
    parser.add_argument("--noise-alpha-h", type=float, default=0.0,
                        help="paper §5.2.2 W_h Gaussian noise scale (0.0 disables; paper default 0.2).")
    parser.add_argument("--noise-seed", type=int, default=13371337,
                        help="seed for ε_embed, ε_head sampling (separate RNG from τ).")
    parser.add_argument("--alg2", action="store_true",
                        help="apply Algorithm 2 intra-head + inter-head attention obfuscation (item 7).")
    parser.add_argument("--alg2-seed", type=int, default=987654321,
                        help="base seed for Algorithm 2 per-layer keys.")
    parser.add_argument("--alg2-beta", type=int, default=8,
                        help="max RoPE-block window size for dynamic_window Ẑ_block (paper default 8).")
    parser.add_argument("--alg2-gamma", type=float, default=1e3,
                        help="dynamic_window similarity-score scale (paper default 1e3).")
    parser.add_argument("--alg2-qk-scale-min", type=float, default=0.95,
                        help="Ĥ_qk per-block scale lower bound (reference default 0.95).")
    parser.add_argument("--alg2-qk-scale-max", type=float, default=1.05,
                        help="Ĥ_qk per-block scale upper bound (reference default 1.05).")
    parser.add_argument("--alg2-qk-norm-matrix", action="store_true",
                        help="Bake R̂_qk into W_q/W_k output axis AND replace 1D γ_q/γ_k "
                             "with 2D Γ = MᵀDM (paper Algorithm 2 intra-head on Qwen3). "
                             "Requires the patched llama.cpp kernel branch that detects "
                             "aloepri.qk_norm_matrix metadata. Forces scale_range to "
                             "(1.0, 1.0). See docs/research/aloepri-qk-norm-matrix-gamma-threat-model.md.")
    parser.add_argument("--alg2-h-hadamard-signs", action="store_true",
                        help="Use ±1 Walsh-Hadamard Ĥ_qk instead of identity. Combine "
                             "with --alg2-qk-norm-matrix: keeps M_q orthogonal (H is "
                             "involutive), adds per-pair sign flips to the obfuscation.")
    parser.add_argument("--alg2-u-vo", action="store_true",
                        help="Enable U_vo random projection on the V-O composition "
                             "(paper sec 5.2.3 step 4 + steps 6 alt / 7). V output is "
                             "right-multiplied by U_vo (sampled per layer, paper "
                             "default variance 1/d_head with QR-stabilisation); W_o "
                             "input axis gets U_vo inverse block-diag per head. The "
                             "two cancel through attention to preserve residual "
                             "covariance, while adding per-deployment randomness on "
                             "the V-O head_dim axis. Closes paper Table 4's "
                             "0.82 to 0.0 percent HiddenState gap. Default off for "
                             "backward compatibility with prior deployments; tag "
                             "obfuscated GGUFs under a distinct name when enabling.")
    parser.add_argument("--output-dtype", choices=("fp32", "bf16"), default="bf16",
                        help="Precision for the large matmul tensors. Default bf16: "
                             "half the file size and 2× decode throughput vs fp32, "
                             "accuracy-equivalent on the obfuscated keymat chain. "
                             "fp16 is unsafe (collapses the obfuscation chain — see "
                             "fp32_to_bf16 docstring). Norm tensors and matrix-Γ Γ_q/Γ_k "
                             "always stay F32 regardless of this setting.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    info = rewrite_gguf(args.in_path, args.out_path, mode=args.mode,
                       expansion=args.expansion_size, seed=args.seed, lam=args.lam,
                       apply_pi=args.pi, pi_seed=args.pi_seed,
                       pi_include_specials=args.pi_include_specials,
                       key_out=args.key_out,
                       noise_alpha_e=args.noise_alpha_e, noise_alpha_h=args.noise_alpha_h,
                       noise_seed=args.noise_seed,
                       apply_alg2=args.alg2, alg2_seed=args.alg2_seed,
                       alg2_beta=args.alg2_beta, alg2_gamma=args.alg2_gamma,
                       alg2_qk_scale_range=(args.alg2_qk_scale_min, args.alg2_qk_scale_max),
                       alg2_qk_norm_matrix=args.alg2_qk_norm_matrix,
                       alg2_h_hadamard_signs=args.alg2_h_hadamard_signs,
                       alg2_enable_u_vo=args.alg2_u_vo,
                       output_dtype=args.output_dtype)
    log.info("done: %s", info)
    return 0


if __name__ == "__main__":
    sys.exit(main())

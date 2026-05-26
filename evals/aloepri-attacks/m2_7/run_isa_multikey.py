"""Paper-faithful labelled-ridge ISA on hidden states, with multi-key attacker synthesis.

The standard ridge ISA driver
(`evals/aloepri-attacks/attack_drivers/run_isa.py`) trains a ridge
inverter `hidden_state[L] → W_e[plain_id]` on labelled pairs where
the labels come from running known prompts through the *deployment*
with the deployment's secret τ applied client-side. Under paper §3.2
the attacker does not have τ, so they cannot label deployment-side
hidden-state captures with plain_ids. That makes the standard driver
non-paper-faithful (it has the same τ-leak we just fixed in IMA).

Reconstructing the attack within paper §3.2's threat model, the attacker:

1. Has Kerckhoffs access to the obfuscation algorithm (h, λ, α_e, α_h,
   Algorithm 1, Algorithm 2) and to the public plaintext model.
2. Runs the *plain* model on prompts whose plain_ids they know — no
   τ knowledge needed (the plain model has identity τ). This produces
   `State_plain[L][position]` paired with `plain_id[position]`.
3. Generates their own attacker-side keymats K_a^k via Algorithm 1.
   They cannot run the obfuscated model end-to-end (would need to
   apply Algorithm 2 transforms to every weight and rebuild the
   network), but under AloePri's covariance claim the obfuscated
   residual at any layer L equals K_e · State_plain[L]. So the
   attacker can *synthesise* what their own obfuscated model's
   hidden state would have been:
       `State_a^k[L][i] = State_plain[L][i] @ K_a^k`
4. Trains a ridge inverter on the concatenation across K keys:
       `(State_a^k[L][i], W_e[plain_id_i])` for k=1..K, i=1..n_pairs
   forcing key-invariant inversion (same multi-key trick as
   `run_ima_embedrow_attacks_multikey.py`).
5. At inference time, captures deployment-side hidden states
   `State_d[L]` (server-side observable, no τ needed — these are
   actual obfuscated runtime states). Applies the trained inverter,
   does cosine-NN against the public W_e to recover the top-1
   plain_id.

`--identity-tau` runs the calibration probe: train + test directly
on `State_plain` with no synthesis and no K. Tests the ridge
attacker's ceiling on a no-defense task.

The covariance approximation skips per-layer additive noise (path-2
only adds noise at the embedding and the head, not at every
intermediate layer — see `python/aloepri-llm/obfuscate_qwen3_gguf.py:332-352`).
Algorithm 2's intra-head transforms also preserve the residual basis
by design, so synthesis matches the deployment up to higher-order
Algorithm-2 perturbations on the post-attention residual.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Existing reusable building blocks.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snapshots_loader import SnapshotSet  # type: ignore  # noqa: E402
from attack_drivers import run_ima  # type: ignore  # noqa: E402
from attack_drivers.common import (  # type: ignore  # noqa: E402
    AttackResult,
    classify_risk_level,
    stack_prompt_observations,
)


# ───── Multi-key attacker construction (mirrors IMA multi-key driver) ──────


def _build_attacker_keymat_pool_vendor(
    *,
    d: int,
    expansion: int,
    lam: float,
    num_keys: int,
    attacker_seed: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Reference keymat pool builder using vendor `keymat.build_keymat_transform`
    (CPU-only torch via `torch.Generator(device='cpu')`). Returns
    (K, d, d + 2h) float32, transferred to `device` at the end.

    Trusted reference: same code path that built the deployment's K_d.
    Slow on CPU at large d (~few minutes per pool at d=4096, K=64), but
    correct.
    """
    sys.path.insert(0, "/home/timo/repos/private-rag-path-2/vendor/aloepri-py")
    sys.path.insert(0, "/home/timo/repos/private-rag-path-2/vendor/aloepri-py/src")
    from keymat import build_keymat_transform  # type: ignore  # noqa: E402

    d_obs = d + 2 * expansion
    pool = torch.empty((num_keys, d, d_obs), dtype=torch.float32)
    for k in range(num_keys):
        transform = build_keymat_transform(
            d=d, h=expansion, lam=lam, init_seed=attacker_seed + 1 + 10_000 * k,
        )
        pool[k] = transform.key.to(torch.float32)
    return pool.to(device)


def _build_attacker_keymat_pool_gpu_native(
    *,
    d: int,
    expansion: int,
    lam: float,
    num_keys: int,
    attacker_seed: int,
    device: str = "cpu",
) -> torch.Tensor:
    """GPU-native keymat pool builder. Same Algorithm 1 math as vendor
    but uses one `torch.Generator(device=gen_device)` per k advancing
    through 8 draws — pattern that vendor avoids by using 8 separate
    generators (one per draw with offset seeds +1..+7 / +11).

    A 2026-05-21 5-seed sweep at Q3-4B Û_vo, L=17, K=64, row-split
    found vendor_cpu and gpu_native sample TTRSR top-1 from
    indistinguishable distributions (Welch t = 0.40, p = 0.70):

        | impl       | mean    | std    | range          |
        | ---------- | ------- | ------ | -------------- |
        | vendor_cpu | 6.22 %  | 5.02   | 1.91 – 12.79 % |
        | gpu_native | 5.04 %  | 4.19   | 1.24 – 11.66 % |

    Earlier-same-day analysis flagged a single-seed reading of 11.92 %
    here as "structurally impossible" above the 10.18 % plain-τ
    ceiling — that diagnosis was retracted after the seed sweep
    showed vendor reaches 12.79 % at attacker_seed=2 alone. The K=64
    pool's TTRSR has std ≈ 5 pp at d=2560; single-seed comparisons
    within that range are noise.

    Either impl is correct. Prefer vendor_cpu for consistency with
    the deployment-side keymat builder (no measurable difference,
    but matches the reference algorithm verbatim). Full investigation
    writeup: `docs/research/aloepri-keymat-variance.md`.
    """
    if expansion <= 0 or expansion % 2 != 0:
        raise ValueError(f"expansion h must be positive and even, got {expansion}")
    if lam < 0:
        raise ValueError(f"lam must be non-negative, got {lam}")
    d_obs = d + 2 * expansion
    half_h = expansion // 2
    dtype = torch.float64
    gen_device = "cuda" if device.startswith("cuda") else "cpu"
    pool = torch.empty((num_keys, d, d_obs), dtype=torch.float32, device=device)

    def _orthogonal(dim: int, gen: torch.Generator) -> torch.Tensor:
        g = torch.randn(dim, dim, generator=gen, dtype=dtype, device=gen_device)
        q, r = torch.linalg.qr(g, mode="reduced")
        sign = torch.sign(torch.diagonal(r))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        return q * sign.unsqueeze(0)

    def _nullspace_basis(matrix: torch.Tensor) -> torch.Tensor:
        _, sv, vh = torch.linalg.svd(matrix, full_matrices=True)
        cutoff = max(1e-10, 1e-10 * float(sv.max().item()))
        rank = int((sv > cutoff).sum().item())
        basis = vh[rank:].T.contiguous()
        if basis.numel() == 0:
            raise ValueError("Null space empty; cannot construct Algorithm 1 key.")
        return basis

    for k in range(num_keys):
        gen = torch.Generator(device=gen_device)
        gen.manual_seed(int(attacker_seed + 1 + 10_000 * k))
        u = _orthogonal(d, gen)
        v = torch.randn(d, d, generator=gen, dtype=dtype, device=gen_device) * (d ** -0.5)
        b = u + lam * v
        e1 = torch.randn(d, half_h, generator=gen, dtype=dtype, device=gen_device) * (d ** -0.5)
        e2 = torch.randn(half_h, expansion, generator=gen, dtype=dtype, device=gen_device) * (d ** -0.5)
        e_mat = e1 @ e2
        f1 = torch.randn(expansion, half_h, generator=gen, dtype=dtype, device=gen_device) * (d ** -0.5)
        f2 = torch.randn(half_h, d, generator=gen, dtype=dtype, device=gen_device) * (d ** -0.5)
        f_mat = f1 @ f2
        z_mat = _orthogonal(d_obs, gen)
        basis_ft = _nullspace_basis(f_mat.T)
        coeffs_c = torch.randn(d, basis_ft.shape[1], generator=gen, dtype=dtype, device=gen_device)
        c_mat = coeffs_c @ basis_ft.T
        left = torch.cat([b, c_mat, e_mat], dim=1)
        key = left @ z_mat
        pool[k] = key.to(dtype=torch.float32)
        del u, v, b, e1, e2, e_mat, f1, f2, f_mat, z_mat
        del basis_ft, coeffs_c, c_mat, left, key
    return pool


def _build_attacker_keymat_pool(
    *,
    impl: str = "vendor_cpu",
    **kwargs,
) -> torch.Tensor:
    """Dispatch on impl name: 'vendor_cpu' (trusted, slow at large d)
    or 'gpu_native' (fast on GPU but currently produces divergent
    attack results — diagnostic only)."""
    if impl == "vendor_cpu":
        return _build_attacker_keymat_pool_vendor(**kwargs)
    if impl == "gpu_native":
        return _build_attacker_keymat_pool_gpu_native(**kwargs)
    raise ValueError(f"unknown keymat impl: {impl!r}; expected vendor_cpu or gpu_native")


# ───── Ridge solver (multi-α, val-selected) ────────────────────────────────


def _fit_ridge(
    X: torch.Tensor, Y: torch.Tensor, *, ridge_alpha: float,
) -> dict[str, torch.Tensor]:
    """Standard closed-form ridge — same primitive as
    `attack_drivers/run_isa.py` and `vendor/aloepri-py` reference.

    Runs on the device X is on. Caller should move X, Y to GPU before
    invoking when the synthetic training set fits on the GPU (~tens of
    GB at K=64 on Q3-8B). torch.linalg.solve auto-routes to cuSOLVER
    / rocSOLVER for the (n × n) solve.
    """
    device = X.device
    x_mean = X.mean(dim=0, keepdim=True)
    x_std = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_mean = Y.mean(dim=0, keepdim=True)
    y_std = Y.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xn = (X - x_mean) / x_std
    Yn = (Y - y_mean) / y_std
    ones = torch.ones((Xn.shape[0], 1), dtype=Xn.dtype, device=device)
    Xa = torch.cat([Xn, ones], dim=1)
    n = Xa.shape[1]
    I = torch.eye(n, dtype=Xn.dtype, device=device)
    I[-1, -1] = 0.0
    lhs = Xa.T @ Xa + ridge_alpha * I
    rhs = Xa.T @ Yn
    # `torch.linalg.solve` on ROCm Strix Halo hits HIPBLAS_STATUS_ALLOC_FAILED
    # at the hipblasStrsm triangular-solve step at d=4353 even though
    # peak memory is only ~4 GB on a 68 GB iGPU — rocBLAS-internal
    # allocator failure, not a real OOM. Try on GPU; on failure, move
    # the (small, ~150 MB) lhs/rhs to CPU and run LAPACK solve there.
    try:
        W = torch.linalg.solve(lhs, rhs)
    except RuntimeError as e:
        if "HIPBLAS_STATUS_ALLOC_FAILED" not in str(e) and "CUDA error" not in str(e):
            raise
        print(f"  ridge solve: GPU failed ({type(e).__name__}); falling back to CPU LAPACK")
        W_cpu = torch.linalg.solve(lhs.cpu(), rhs.cpu())
        W = W_cpu.to(device)
        del W_cpu
    return {"weight": W, "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


def _predict_ridge(model: dict[str, torch.Tensor], X: torch.Tensor) -> torch.Tensor:
    Xn = (X - model["x_mean"]) / model["x_std"]
    ones = torch.ones((Xn.shape[0], 1), dtype=Xn.dtype, device=X.device)
    Xa = torch.cat([Xn, ones], dim=1)
    Yn = Xa @ model["weight"]
    return Yn * model["y_std"] + model["y_mean"]


# ───── Cosine-NN top-1/top-10 against full vocab ───────────────────────────


def _cosine_topk(
    pred: torch.Tensor, embed_table: torch.Tensor, true_ids: torch.Tensor, topk: int = 10,
    chunk: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (top1_hits, topk_hits) as bool tensors over the test rows."""
    pn = pred / pred.norm(dim=1, keepdim=True).clamp_min(1e-8)
    best_scores = torch.full((pn.shape[0], 0), float("-inf"), dtype=pn.dtype, device=pn.device)
    best_ids = torch.empty((pn.shape[0], 0), dtype=torch.long, device=pn.device)
    vocab = embed_table.shape[0]
    k_eff = min(topk, vocab)
    for s in range(0, vocab, chunk):
        e = min(s + chunk, vocab)
        cn = embed_table[s:e]
        cn = cn / cn.norm(dim=1, keepdim=True).clamp_min(1e-8)
        sc = pn @ cn.T  # (N_test, e-s)
        c_scores, c_local = torch.topk(sc, k=min(k_eff, sc.shape[1]), dim=1)
        c_ids = c_local + s
        merged_scores = torch.cat([best_scores, c_scores], dim=1)
        merged_ids = torch.cat([best_ids, c_ids], dim=1)
        new_scores, new_idx = torch.topk(merged_scores, k=k_eff, dim=1)
        best_scores = new_scores
        best_ids = merged_ids.gather(1, new_idx)
    hits = best_ids.eq(true_ids.unsqueeze(1))
    top1 = hits[:, 0]
    topk = hits[:, :k_eff].any(dim=1)
    return top1, topk


# ───── Main entrypoint ────────────────────────────────────────────────────


def run_isa_multikey(
    *,
    plain_snapshots: SnapshotSet,
    obf_snapshots: SnapshotSet | None,
    embed_table: torch.Tensor,
    layer: int,
    kind: str = "attn_norm",
    attacker_expansion: int = 128,
    attacker_lam: float = 0.3,
    attacker_num_keys: int = 64,
    attacker_seed: int = 20260521,
    keymat_impl: str = "vendor_cpu",
    ridge_alphas: tuple[float, ...] = (1e-4, 1e-2, 1.0),
    train_frac: float = 0.5,
    val_frac: float = 0.25,
    identity_tau: bool = False,
    split_mode: str = "row",
    topk: int = 10,
    device: str = "auto",
) -> AttackResult:
    """Run paper-faithful multi-key labelled-ridge ISA at one (layer, kind).

    plain_snapshots provides the State_plain inputs for training-data
    synthesis; obf_snapshots provides the deployment-side State_d for
    test eval. In identity_tau mode obf_snapshots is unused — train +
    test both come from plain captures (calibration probe).
    """
    t0 = time.perf_counter()

    # Resolve device. GPU eliminates the CPU memory bottleneck on the
    # ridge solve (multi-key training matrix can reach ~14 GB at K=158
    # on Q3-8B). torch.linalg.solve auto-routes to cuSOLVER /
    # rocSOLVER on the corresponding device.
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "gpu":
        resolved_device = "cuda"
    else:
        resolved_device = device
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"  warn: requested device={device!r} but CUDA/ROCm not available; falling back to CPU")
        resolved_device = "cpu"
    print(f"  device = {resolved_device}")
    embed_table = embed_table.to(resolved_device)

    # 1) Load plain hidden states (and their plain_id labels)
    X_plain, y_ids, _ = stack_prompt_observations(
        plain_snapshots, layer=layer, kind=kind, strip_shield=True,
    )
    if X_plain.shape[0] == 0:
        raise RuntimeError(f"no plain snapshots at layer={layer} kind={kind}")
    X_plain_t = torch.from_numpy(X_plain).to(torch.float32).to(resolved_device)
    y_ids_t = torch.from_numpy(y_ids).to(torch.long).to(resolved_device)
    d_plain = int(X_plain_t.shape[1])
    n_total = int(X_plain_t.shape[0])

    print(f"  plain captures: {n_total} (state, plain_id) rows at "
          f"layer={layer} kind={kind} d_plain={d_plain}")
    if d_plain != int(embed_table.shape[1]):
        raise RuntimeError(
            f"plain state dim {d_plain} != embed_table dim {embed_table.shape[1]} — "
            f"unexpected; the plain model's residual stream should match W_e dim"
        )

    # 2) Train/val/test split.
    #
    # split_mode="row" (default, threat-model realistic): partition
    # token positions randomly, both splits share the unique plain_id
    # vocab. Matches the realistic attacker who pre-trains on a public
    # corpus covering > 99 % of the Qwen3 vocab, so test queries'
    # plain_ids are essentially always in training vocab.
    #
    # split_mode="vocab" (paper reference-impl methodology): partition
    # unique plain_ids into disjoint sets, all rows for each id go to
    # one split. Stress test for vocab generalisation. Ridge cannot
    # extrapolate to unseen W_e[plain_id] across the split, so this
    # gives 0 % top-1 by construction on small data — measures the
    # methodology more than the defence. Available as a secondary
    # reading.
    unique_ids = torch.unique(y_ids_t).tolist()
    rng = np.random.default_rng(attacker_seed + 17)
    if split_mode == "vocab":
        shuffled = rng.permutation(unique_ids).tolist()
        n_train_ids = int(len(shuffled) * train_frac)
        n_val_ids = int(len(shuffled) * val_frac)
        train_ids = set(shuffled[:n_train_ids])
        val_ids = set(shuffled[n_train_ids : n_train_ids + n_val_ids])
        test_ids = set(shuffled[n_train_ids + n_val_ids :])
        def _mask(ids_set):
            return torch.tensor(
                [int(i) in ids_set for i in y_ids_t.tolist()],
                dtype=torch.bool, device=resolved_device,
            )
        tr_mask = _mask(train_ids)
        va_mask = _mask(val_ids)
        te_mask = _mask(test_ids)
        print(f"  vocab-disjoint split: train={int(tr_mask.sum())} val={int(va_mask.sum())} "
              f"test={int(te_mask.sum())} rows (over {len(unique_ids)} unique plain_ids)")
    elif split_mode == "row":
        n_rows = int(X_plain_t.shape[0])
        perm = rng.permutation(n_rows)
        n_train_rows = int(n_rows * train_frac)
        n_val_rows = int(n_rows * val_frac)
        tr_idx = perm[:n_train_rows]
        va_idx = perm[n_train_rows : n_train_rows + n_val_rows]
        te_idx = perm[n_train_rows + n_val_rows :]
        tr_mask = torch.zeros(n_rows, dtype=torch.bool, device=resolved_device)
        tr_mask[tr_idx] = True
        va_mask = torch.zeros(n_rows, dtype=torch.bool, device=resolved_device)
        va_mask[va_idx] = True
        te_mask = torch.zeros(n_rows, dtype=torch.bool, device=resolved_device)
        te_mask[te_idx] = True
        train_ids = set(y_ids_t[tr_mask].tolist())
        val_ids = set(y_ids_t[va_mask].tolist())
        test_ids = set(y_ids_t[te_mask].tolist())
        print(f"  row split: train={int(tr_mask.sum())} val={int(va_mask.sum())} "
              f"test={int(te_mask.sum())} rows; vocab overlap "
              f"train∩test={len(train_ids & test_ids)} of {len(train_ids)} train ids")
    else:
        raise ValueError(f"split_mode must be 'row' or 'vocab', got {split_mode!r}")

    # 3) Build training inputs
    if identity_tau:
        # Calibration probe: train + test on plain states directly.
        print("  identity-τ calibration: ridge on (State_plain, W_e[plain_id]) — no K_a")
        X_train = X_plain_t[tr_mask]
        y_train = embed_table[y_ids_t[tr_mask]]
        X_val = X_plain_t[va_mask]
        y_val_ids = y_ids_t[va_mask]
        X_test = X_plain_t[te_mask]
        y_test_ids = y_ids_t[te_mask]
    else:
        # Paper-faithful: synthesise K attacker-keymat-transformed inputs
        # per training row. Test inputs come from obf captures.
        if obf_snapshots is None:
            raise RuntimeError("obf_snapshots is required in non-identity_tau mode")
        print(f"  pre-generating multi-key pool: K={attacker_num_keys} keymats "
              f"(impl={keymat_impl}, h={attacker_expansion}, λ={attacker_lam}, "
              f"seed={attacker_seed})")
        keymat_pool = _build_attacker_keymat_pool(
            impl=keymat_impl,
            d=d_plain, expansion=int(attacker_expansion), lam=float(attacker_lam),
            num_keys=int(attacker_num_keys), attacker_seed=int(attacker_seed),
            device=resolved_device,
        )

        # Synthesise: X_a^k[i] = X_plain[i] @ K_a^k. Stack across k.
        X_plain_train = X_plain_t[tr_mask]
        X_plain_val = X_plain_t[va_mask]
        synth_train_chunks: list[torch.Tensor] = []
        synth_train_y_chunks: list[torch.Tensor] = []
        synth_val_chunks: list[torch.Tensor] = []
        for k in range(int(attacker_num_keys)):
            K_k = keymat_pool[k]  # (d_plain, d_obs)
            X_a_train = X_plain_train @ K_k  # (n_train, d_obs)
            X_a_val = X_plain_val @ K_k
            synth_train_chunks.append(X_a_train)
            synth_train_y_chunks.append(embed_table[y_ids_t[tr_mask]])
            synth_val_chunks.append(X_a_val)
        X_train = torch.cat(synth_train_chunks, dim=0)
        y_train = torch.cat(synth_train_y_chunks, dim=0)
        X_val = torch.cat(synth_val_chunks, dim=0)
        # Val labels repeat across the K synth copies of the val set
        y_val_ids_single = y_ids_t[va_mask]
        y_val_ids = y_val_ids_single.repeat(int(attacker_num_keys))
        print(f"  synthesised training tensor: X_train {tuple(X_train.shape)} "
              f"y_train {tuple(y_train.shape)} (K × n_train)")
        # Free GPU memory held by the per-key chunks (~K · 2 · n_train · d_obs)
        # and the full keymat pool (~K · d · d_obs). Both are now redundant —
        # the concatenated X_train / y_train / X_val carry everything the
        # ridge solve needs. On Q3-8B at K=64 this frees ~5 GB GPU, which
        # the rocSOLVER triangular-solve needs as workspace; without this
        # we hit HIPBLAS_STATUS_ALLOC_FAILED inside torch.linalg.solve.
        del synth_train_chunks, synth_train_y_chunks, synth_val_chunks
        del keymat_pool
        if resolved_device.startswith("cuda"):
            torch.cuda.empty_cache()

        # Test inputs: deployment's obf captures at the same layer, but
        # filtered to test plain_ids only (vocab-disjoint from train).
        X_obf_full, y_obf_ids, _ = stack_prompt_observations(
            obf_snapshots, layer=layer, kind=kind, strip_shield=True,
        )
        X_obf_t = torch.from_numpy(X_obf_full).to(torch.float32).to(resolved_device)
        y_obf_t = torch.from_numpy(y_obf_ids).to(torch.long).to(resolved_device)
        te_obf_mask = torch.tensor(
            [int(i) in test_ids for i in y_obf_t.tolist()],
            dtype=torch.bool, device=resolved_device,
        )
        X_test = X_obf_t[te_obf_mask]
        y_test_ids = y_obf_t[te_obf_mask]
        print(f"  obf test rows: {int(te_obf_mask.sum())} (filtered to test plain_ids)")

    # 4) Multi-α ridge selection on val
    best_alpha: float | None = None
    best_val_top1: float = -1.0
    best_model: dict[str, torch.Tensor] | None = None
    alpha_scores: list[dict[str, Any]] = []
    for alpha in ridge_alphas:
        model = _fit_ridge(X_train, y_train, ridge_alpha=float(alpha))
        val_pred = _predict_ridge(model, X_val)
        v_top1, v_topk = _cosine_topk(val_pred, embed_table, y_val_ids, topk=topk)
        v_top1_f = float(v_top1.float().mean().item())
        v_topk_f = float(v_topk.float().mean().item())
        alpha_scores.append({"ridge_alpha": float(alpha), "val_top1": v_top1_f, "val_topk": v_topk_f})
        print(f"    α={alpha:.0e}: val_top1={v_top1_f:.4f} val_top{topk}={v_topk_f:.4f}")
        if v_top1_f > best_val_top1:
            best_val_top1 = v_top1_f
            best_alpha = float(alpha)
            best_model = model
    assert best_model is not None

    # 5) Final test eval
    test_pred = _predict_ridge(best_model, X_test)
    te_top1, te_topk = _cosine_topk(test_pred, embed_table, y_test_ids, topk=topk)
    top1 = float(te_top1.float().mean().item())
    top10 = float(te_topk.float().mean().item())

    return AttackResult(
        attack="isa_multikey",
        condition="obfuscated" if not identity_tau else "plain_identity_tau",
        model_id=plain_snapshots.model_id,
        n_prompts=plain_snapshots.n_prompts() if hasattr(plain_snapshots, "n_prompts") else 0,
        n_train=int(X_train.shape[0]),
        n_test=int(X_test.shape[0]),
        ttrsr_top1=top1,
        ttrsr_top10=top10,
        risk_level=classify_risk_level(top1),
        extra={
            "layer": int(layer),
            "kind": str(kind),
            "threat_model_regime": "multikey_covariant_synthesis_paperfaithful",
            "split_mode": str(split_mode),
            "attacker_identity_probe": bool(identity_tau),
            "attacker_expansion": int(attacker_expansion),
            "attacker_lam": float(attacker_lam),
            "attacker_num_keys": int(attacker_num_keys) if not identity_tau else 0,
            "attacker_seed": int(attacker_seed),
            "keymat_impl": str(keymat_impl),
            "best_ridge_alpha": best_alpha,
            "alpha_scan": alpha_scores,
            "n_unique_plain_ids": len(unique_ids),
            "n_train_ids": len(train_ids),
            "n_val_ids": len(val_ids),
            "n_test_ids": len(test_ids),
            "runtime_seconds": round(time.perf_counter() - t0, 2),
        },
    )


# ───── CLI ─────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description="Paper-faithful labelled-ridge ISA with multi-key attacker synthesis"
    )
    p.add_argument("--plain-captures", type=Path, required=True,
                   help="Directory containing hidden.{safetensors,meta.json} from plain-model capture.")
    p.add_argument("--obf-captures", type=Path, default=None,
                   help="Directory containing hidden.{safetensors,meta.json} from obfuscated-model capture. "
                        "Required unless --identity-tau is set.")
    p.add_argument("--plain-model-id", type=str, default="Qwen/Qwen3-4B",
                   help="HF model id whose W_e is loaded as the candidate pool and inversion target.")
    p.add_argument("--layer", type=int, default=17,
                   help="Hidden-state capture layer to attack. Default 17 ≈ 48 percent depth on 36-layer Q3.")
    p.add_argument("--kind", type=str, default="attn_norm")
    p.add_argument("--identity-tau", action="store_true",
                   help="Calibration probe: ridge on plain captures only, no synthesis, no K_a.")
    p.add_argument("--attacker-expansion", type=int, default=128)
    p.add_argument("--attacker-lambda", type=float, default=0.3)
    p.add_argument("--attacker-num-keys", type=int, default=64)
    p.add_argument("--keymat-impl", type=str, default="vendor_cpu",
                   choices=("vendor_cpu", "gpu_native"),
                   help="vendor_cpu (default): trusted vendor build_keymat_transform "
                        "on CPU; correct but slow at large d (~few min per pool at "
                        "d=4096 K=64). gpu_native: experimental GPU port of Algorithm "
                        "1; fast (~30s) but currently produces divergent attack results "
                        "(4B Û_vo TTRSR 11.9 % vs vendor's 3.4 %) — keep for diagnostic "
                        "side-by-side; do not use for production measurements.")
    p.add_argument("--split-mode", type=str, default="row", choices=("row", "vocab"),
                   help="row (default): random position-level split, train+test share vocab — "
                        "realistic threat-model reading. vocab: vocab-disjoint, stress test "
                        "for vocab generalisation (ridge gives 0 percent by construction on small data).")
    p.add_argument("--device", type=str, default="auto", choices=("auto", "gpu", "cpu", "cuda"),
                   help="Device for the ridge solve + cosine-NN eval + multi-key synthesis. "
                        "auto picks GPU if available else CPU. K=64 on Q3-4B fits CPU, "
                        "but K=158 on Q3-8B benefits from GPU (~14 GB working set, "
                        "torch.linalg.solve routes to rocSOLVER / cuSOLVER).")
    # The Docker wrapper run_in_gpu_container.sh auto-injects this for
    # the IMA driver's checkpointing. ISA does closed-form ridge, no
    # iterative training, no checkpoint needed — accept and ignore.
    p.add_argument("--paper-checkpoint-dir", type=Path, default=None,
                   help="(accepted for compat with shared wrapper; ISA has no checkpoint)")
    p.add_argument("--attacker-seed", type=int, default=20260521)
    p.add_argument("--ridge-alpha", type=float, action="append", default=None,
                   help="Override the default multi-α grid. Can be passed multiple times.")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    if not args.identity_tau and args.obf_captures is None:
        raise SystemExit("--obf-captures is required unless --identity-tau is set")

    print(f"[ISA-multikey] plain captures: {args.plain_captures}")
    plain_snap = SnapshotSet.open(args.plain_captures / "hidden")
    print(f"  {plain_snap.n_prompts()} prompt(s), layers={plain_snap.captured_layers}, "
          f"kinds={plain_snap.captured_kinds}")
    if args.identity_tau:
        obf_snap = None
    else:
        print(f"[ISA-multikey] obf captures: {args.obf_captures}")
        obf_snap = SnapshotSet.open(args.obf_captures / "hidden")
        print(f"  {obf_snap.n_prompts()} prompt(s), layers={obf_snap.captured_layers}, "
              f"kinds={obf_snap.captured_kinds}")

    print(f"[ISA-multikey] loading embed table for {args.plain_model_id}")
    embed_table = run_ima.load_qwen3_embedding_table(args.plain_model_id).to(torch.float32)
    print(f"  W_e shape = {tuple(embed_table.shape)}")

    ridge_alphas = tuple(args.ridge_alpha) if args.ridge_alpha else (1e-4, 1e-2, 1.0)

    result = run_isa_multikey(
        plain_snapshots=plain_snap,
        obf_snapshots=obf_snap,
        embed_table=embed_table,
        layer=int(args.layer),
        kind=str(args.kind),
        attacker_expansion=int(args.attacker_expansion),
        attacker_lam=float(args.attacker_lambda),
        attacker_num_keys=int(args.attacker_num_keys),
        attacker_seed=int(args.attacker_seed),
        keymat_impl=str(args.keymat_impl),
        ridge_alphas=ridge_alphas,
        identity_tau=bool(args.identity_tau),
        split_mode=str(args.split_mode),
        device=str(args.device),
    )

    print(f"[ISA-multikey] top1={result.ttrsr_top1:.4f} top10={result.ttrsr_top10:.4f} "
          f"risk={result.risk_level} α*={result.extra.get('best_ridge_alpha')}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "format": "aloepri_m2_7_isa_multikey_v1",
        "plain_captures": str(args.plain_captures),
        "obf_captures": str(args.obf_captures) if args.obf_captures else None,
        "plain_model_id": args.plain_model_id,
        "attack": result.to_dict(),
    }, indent=2))
    print(f"[ISA-multikey] wrote → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

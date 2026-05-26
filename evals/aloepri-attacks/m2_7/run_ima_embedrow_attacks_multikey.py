"""Multi-key paper-faithful IMA-EmbedRow-transformer attack.

Fork of `run_ima_embedrow_attacks.py`. Replaces the single-key attacker
(one synthetic (τ_a, K_a) sampled once per run) with **multi-key
training**: K independent K_aᵏ are pre-generated, and per training
batch a random k is sampled and the input is computed on-the-fly as
`(W_e[plain_id] + fresh_noise) @ K_aᵏ`.

Why: with a single attacker key the inverter overfits to one specific
K_a and cannot transfer to the deployment's K_d at test time. With
many K_a samples drawn from the *same algorithm distribution* as K_d,
the inverter is forced to learn a key-invariant inversion procedure —
i.e. to invert the obfuscation **algorithm** rather than memorize
one keymat. Multi-key is paper-faithful (the threat model permits the
attacker to run Algorithm 1 any number of times with their own keys)
and matches the strongest-attacker reading.

Construction note: when training pairs are accessed as
`W̃_aᵏ[τ_aᵏ[plain_id]]`, the τ_aᵏ permutation cancels — we both route
the row to position τ_aᵏ[plain_id] AND look it up via the same index.
So τ_aᵏ is irrelevant for training-input generation; only K_aᵏ and
noise_aᵏ matter. We skip per-key permutation materialisation entirely
and compute `x_train[i] = (W_e[plain_id_i] + fresh_noise) @ K_aᵏ`
on-the-fly per batch.

Recommendation A from the IMA-attacker research agent (2026-05-21).

──────────────────────────────────────────────────────────────────────

Run the IMA-EmbedRow prompt-inversion attacks against the §05 obfuscated GGUF.

These are the two **prompt-inversion via static-weight** flavours of
the Inversion Model Attack described in paper §F.1, ported as path-2
attack drivers with self-descriptive names parallel to the existing
`IMA-L0-activation` / `IMA-L0-transformer` surface attacks in §08:

* **IMA-EmbedRow-ridge** — ridge regression on `(W̃_embed[τ[i]], W_embed[i])`
  pairs. Port of `vendor/aloepri-py/src/security_qwen/ima.py::run_ima_baseline`.
  Surface = static rows of the obfuscated embedding table. Inverter =
  multi-α ridge with val selection.
* **IMA-EmbedRow-transformer** — trained 2-layer transformer inverter on
  the same pairs. Port of `run_ima_paper_like`. The paper trains a
  Qwen2 backbone with 2 decoder layers + 8 heads; we use the same
  pre-LN block as `attack_drivers/run_ima_paper_like.py` (vanilla MHA,
  GELU FFN) — the privacy claim is that *no* learnable inverter can
  recover τ from masked observations.

Threat model (paper §F.1 / Table 1 caption): adversary has both the
plaintext weights θ and the obfuscated weights θ̃. They also have
*some* known `(plain_id, obf_id)` training pairs — in the paper's
reference these come straight from `perm_vocab` (= τ) on a subset of
the tokenizer. This simulates the realistic attack where high-frequency
tokens leak their τ-mapping through corpus statistics or other side
channels, and the attacker uses those known pairs to fit an inverter
that generalises to unknown tokens.

The attacker's *goal* is to recover the plaintext token id from each
obfuscated row — i.e. to invert the permutation τ. This is a
**prompt-inversion attack** (recovering τ lets you decode every
wire-side prompt to this deployment), NOT a "weight inversion" attack.
Path-2's docs previously mislabelled it as out-of-scope on the latter
ground. See `docs/handoffs/2026-05-19-aloepri-attack-surface-followups.md`
thread 1.

Naming note: parallel to `IMA-L0-{activation,transformer}` which name
the *surface* (layer-0 hidden state) and *inverter type*. Here the
surface is `EmbedRow` (a row of the obfuscated embedding table) and
the inverter type is explicit (`ridge` or `transformer`).

Output JSON schema matches `run_static_attacks.py` so the §08 doc
ingestion treats both static-weight families uniformly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Local imports — share the GGUF loader with run_static_attacks.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_gguf_weights import ModelWeights, load_model

# Re-use AttackResult + risk classifier.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attack_drivers.common import (  # type: ignore  # noqa: E402
    AttackResult,
    classify_risk_level,
    load_aloepri_module,
)

_ima = load_aloepri_module("src/security_qwen/ima.py")
_fit_ridge_regressor = _ima._fit_ridge_regressor
_predict_ridge = _ima._predict_ridge
_evaluate_inversion_predictions = _ima._evaluate_inversion_predictions

# Paper-faithful trained-inverter primitives (paper §F.1, reference
# impl run_ima_paper_like). The reference's Qwen2-backbone inverter +
# sequence-window pipeline is the canonical IMA — see
# docs/research/aloepri-attacks.md for the threat-model writeup.
_build_paper_like_inverter_config = _ima.build_paper_like_inverter_config
_PaperLikeIMAInverter = _ima._PaperLikeIMAInverter
_collect_public_token_windows = _ima._collect_public_token_windows
_evaluate_sequence_inversion_predictions = _ima._evaluate_sequence_inversion_predictions
_collect_sensitive_plain_ids = _ima._collect_sensitive_plain_ids
_resolve_ima_device = _ima._resolve_ima_device


# ───── τ loader ────────────────────────────────────────────────────


def load_tau(key_path: Path) -> tuple[np.ndarray, int]:
    """Load τ + active_size from the obfuscator's .key.npz."""
    z = np.load(key_path, allow_pickle=False)
    tau = z["tau"].astype(np.int64)
    active_size = int(z["active_size"])
    return tau, active_size


# ───── Splits ──────────────────────────────────────────────────────


def _sample_splits(
    *,
    active_size: int,
    train_size: int,
    val_size: int,
    test_size: int,
    candidate_pool_size: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Vocab-disjoint splits over `[0, active_size)`. Mirrors the
    AloePri reference but without tokenizer-driven 'sensitive token'
    weighting — path-2 doesn't carry the same DEFAULT_PROMPTS list.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(active_size)
    required = train_size + val_size + test_size
    if required > active_size:
        raise ValueError(
            f"requested splits {required} > active_size {active_size}"
        )
    test_ids = perm[:test_size]
    val_ids = perm[test_size : test_size + val_size]
    train_ids = perm[test_size + val_size : test_size + val_size + train_size]
    candidate = perm[: min(candidate_pool_size, active_size)]
    # Ensure test ⊂ candidate.
    candidate = np.unique(np.concatenate([candidate, test_ids]))
    return {
        "train_plain_ids": train_ids.astype(np.int64),
        "val_plain_ids": val_ids.astype(np.int64),
        "test_plain_ids": test_ids.astype(np.int64),
        "candidate_plain_ids": candidate.astype(np.int64),
    }


# ───── IMA-EmbedRow-ridge ──────────────────────────────────────────


def run_ima_embedrow_ridge(
    plain: ModelWeights,
    obfuscated: ModelWeights,
    tau: np.ndarray,
    *,
    active_size: int,
    train_size: int = 1024,
    val_size: int = 128,
    test_size: int = 128,
    candidate_pool_size: int = 2048,
    topk: int = 10,
    ridge_alphas: tuple[float, ...] = (1e-4, 1e-2, 1.0),
    seed: int = 20260518,
) -> AttackResult:
    """Ridge inverter on `(W̃_embed[τ[i]], W_embed[i])` pairs.

    Returns TTRSR top-1/top-10 plus full diagnostic block (alpha scan,
    cosine, runtime).
    """
    t0 = time.perf_counter()

    splits = _sample_splits(
        active_size=active_size,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        candidate_pool_size=candidate_pool_size,
        seed=seed,
    )

    plain_W_e = torch.from_numpy(plain.token_embd.astype(np.float32))
    obs_W_e = torch.from_numpy(obfuscated.token_embd.astype(np.float32))
    tau_t = torch.from_numpy(tau)

    train_ids = torch.from_numpy(splits["train_plain_ids"])
    val_ids = torch.from_numpy(splits["val_plain_ids"])
    test_ids = torch.from_numpy(splits["test_plain_ids"])
    candidate_ids = torch.from_numpy(splits["candidate_plain_ids"])

    x_train = obs_W_e[tau_t[train_ids]]
    y_train = plain_W_e[train_ids]
    x_val = obs_W_e[tau_t[val_ids]]
    x_test = obs_W_e[tau_t[test_ids]]

    val_candidate_ids = torch.unique(torch.cat([val_ids, candidate_ids]))

    best_alpha: float | None = None
    best_val_top1 = -1.0
    best_model: dict[str, torch.Tensor] | None = None
    alpha_scores: list[dict[str, Any]] = []

    for alpha in ridge_alphas:
        model = _fit_ridge_regressor(x_train, y_train, ridge_alpha=float(alpha))
        val_pred = _predict_ridge(model, x_val)
        val_metrics = _evaluate_inversion_predictions(
            predicted_embeddings=val_pred,
            true_plain_ids=val_ids,
            candidate_plain_ids=val_candidate_ids,
            baseline_embed=plain_W_e,
            topk=topk,
        )
        alpha_scores.append(
            {"ridge_alpha": float(alpha), "val_top1": val_metrics["token_top1_recovery_rate"]}
        )
        if val_metrics["token_top1_recovery_rate"] > best_val_top1:
            best_val_top1 = val_metrics["token_top1_recovery_rate"]
            best_alpha = float(alpha)
            best_model = model

    assert best_model is not None

    test_pred = _predict_ridge(best_model, x_test)
    metrics = _evaluate_inversion_predictions(
        predicted_embeddings=test_pred,
        true_plain_ids=test_ids,
        candidate_plain_ids=candidate_ids,
        baseline_embed=plain_W_e,
        topk=topk,
    )

    top1 = float(metrics["token_top1_recovery_rate"])
    top10 = float(metrics["token_top10_recovery_rate"])

    return AttackResult(
        attack="ima_embedrow_ridge",
        condition="obfuscated",
        model_id=str(obfuscated.path.name),
        n_prompts=0,
        n_train=int(train_ids.shape[0]),
        n_test=int(test_ids.shape[0]),
        ttrsr_top1=top1,
        ttrsr_top10=top10,
        risk_level=classify_risk_level(top1),
        extra={
            "best_ridge_alpha": best_alpha,
            "alpha_scan": alpha_scores,
            "embedding_cosine_similarity": float(metrics["embedding_cosine_similarity"]),
            "candidate_pool_size": int(candidate_ids.shape[0]),
            "runtime_seconds": round(time.perf_counter() - t0, 2),
        },
    )


# ───── IMA-EmbedRow-transformer (paper §F.1, trained 2-layer inverter) ───────
#
# Direct port of `vendor/aloepri-py/src/security_qwen/ima.py::run_ima_paper_like`:
#
# - Inverter = 2-layer Qwen-backbone transformer (`AutoConfig.from_pretrained(
#   baseline_model_dir)` with `hidden_size = d_obs`, `num_hidden_layers=2`,
#   `num_attention_heads=8`, `num_key_value_heads=8`, `intermediate_size =
#   max(d_obs * 4, base)`) + `nn.Linear(d_obs, d_plain, bias=False)` head.
# - Inputs = sequence windows of length 32 from a public corpus, each plain
#   token id looked up via `observed_embed[tau[plain_id]]` to get the
#   deployment-obfuscated row at that index.
# - Targets = `plain_embed[plain_id]` per position.
# - Training = AdamW lr=3e-4 wd=0, batch_size=8, epochs=2, MSE loss.
# - Eval = per-position cosine-NN against candidate vocab.
#
# This matches paper §F.1 + paper §F.2 hyperparams. Previous path-2 driver
# was a stripped-down residual-MLP per-row model (no attention, no sequence
# context) that failed plain-identity control across every variant tried —
# see docs/handoffs/2026-05-20-ima-embedrow-transformer-investigation.md.


DEFAULT_PUBLIC_CORPUS_PATHS: tuple[str, ...] = (
    "vendor/aloepri-py/docs/Towards Privacy-Preserving LLM Inference via Collaborative Obfuscation (Technical Report).txt",
    "vendor/aloepri-py/docs/AloePri 论文中的部署适配机制整理.md",
    "vendor/aloepri-py/docs/AloePri_技术报告梳理与复现方案.md",
    "vendor/aloepri-py/README.md",
)


def _ima_xformer_ckpt_path(
    *,
    checkpoint_dir: Path,
    plain_path: str | None,
    obfuscated_path: str | None,
    tau_fingerprint: str,
    baseline_model_dir: str,
    sequence_length: int,
    train_sequence_count: int,
    val_sequence_count: int,
    test_sequence_count: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    candidate_pool_size: int | None,
    seed: int,
    attacker_fingerprint: str = "v2-paperfaithful",
) -> Path:
    """Content-addressed checkpoint path. Keyed on everything except
    `epochs` so a 100-epoch run can resume from a 50-epoch checkpoint.

    The fingerprint deliberately includes the GGUF *paths* (not file
    hashes) because the obfuscated artifact's identity is uniquely
    determined by its path within the path-2 layout — same path → same
    bytes. If a path is reused for a different artifact, delete the
    matching checkpoint manually.

    `attacker_fingerprint` captures the threat-model regime + attacker
    hyperparameters. Old checkpoints (under the τ-leaking attack) used
    the default tag and must not be re-used after the 2026-05-21
    paper-faithful patch — bump the tag to invalidate.
    """
    import hashlib
    fp_parts = [
        f"plain={plain_path or ''}",
        f"obfuscated={obfuscated_path or ''}",
        f"tau={tau_fingerprint}",
        f"baseline={baseline_model_dir}",
        f"seq_len={sequence_length}",
        f"seq_train={train_sequence_count}",
        f"seq_val={val_sequence_count}",
        f"seq_test={test_sequence_count}",
        f"batch={batch_size}",
        f"lr={learning_rate}",
        f"wd={weight_decay}",
        f"pool={candidate_pool_size}",
        f"seed={seed}",
        f"attacker={attacker_fingerprint}",
    ]
    fp_str = "|".join(fp_parts)
    fp_short = hashlib.sha256(fp_str.encode("utf-8")).hexdigest()[:16]
    return checkpoint_dir / f"ima_xformer_{fp_short}.pt"


def _build_multikey_pool(
    *,
    d: int,
    expansion: int,
    lam: float,
    num_keys: int,
    attacker_seed: int,
) -> torch.Tensor:
    """Pre-generate K independent attacker-side keymats via Algorithm 1.

    Returns a tensor of shape (K, d, d + 2h) on CPU. Per-batch noise
    is fresh — only the keymats are pooled here. Threat model: the
    attacker has Kerckhoffs access to (h, λ, α_e) and can run
    Algorithm 1 an unbounded number of times with their own keys.

    Memory: K × d × (d + 2h) × 4 bytes. For Q3-4B (d=2560, h=128, K=64):
    ≈ 1.7 GB. For Q3-8B (d=4096, h=128) at K=64: ≈ 4.4 GB. Allocate on
    CPU and move per-batch slices to GPU.
    """
    import sys
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
    return pool


def _make_multikey_batch_input(
    *,
    plain_W_e_rows: torch.Tensor,
    keymat_pool: torch.Tensor,
    key_idx: int,
    alpha_e: float,
    sigma_e: float,
    rng_torch: torch.Generator,
) -> torch.Tensor:
    """Build a training/eval batch input as
    `(W_e[plain_ids] + fresh_noise) @ K_aᵏ` on-the-fly.

    See module docstring for why τ_aᵏ cancels — only K_aᵏ + noise
    matter for training-input construction.

    Args:
        plain_W_e_rows: (B*T, d) — plaintext embedding rows for the
            batch's plain_ids, flattened across batch and time.
        keymat_pool: (K, d, d+2h) — pre-generated attacker keymats.
        key_idx: which k to use for this batch.
        alpha_e, sigma_e: noise scaling. `sigma_e = std(plain_W_e)`.
        rng_torch: torch Generator on the same device as plain_W_e_rows
            for fresh per-batch noise.

    Returns:
        (B*T, d+2h) input tensor.
    """
    noise = torch.randn(
        plain_W_e_rows.shape,
        generator=rng_torch,
        dtype=plain_W_e_rows.dtype,
        device=plain_W_e_rows.device,
    )
    K = keymat_pool[key_idx].to(plain_W_e_rows.device, dtype=plain_W_e_rows.dtype)
    noisy = plain_W_e_rows + (alpha_e * sigma_e) * noise
    return noisy @ K  # (B*T, d+2h)


def run_ima_embedrow_transformer(
    plain: ModelWeights,
    obfuscated: ModelWeights,
    tau: np.ndarray,
    *,
    baseline_model_dir: str,
    public_corpus_paths: tuple[str, ...] = DEFAULT_PUBLIC_CORPUS_PATHS,
    sequence_length: int = 32,
    train_sequence_count: int = 128,
    val_sequence_count: int = 16,
    test_sequence_count: int = 16,
    batch_size: int = 8,
    epochs: int = 2,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.0,
    candidate_pool_size: int | None = None,
    topk: int = 10,
    device: str = "auto",
    seed: int = 20260518,
    checkpoint_dir: Path | None = None,
    identity_tau: bool = False,
    attacker_expansion: int = 128,
    attacker_lam: float = 0.3,
    attacker_alpha_e: float = 1.0,
    attacker_seed: int | None = None,
    attacker_num_keys: int = 64,
    use_bf16: bool = True,
) -> AttackResult:
    """Paper-faithful trained-inverter attack on the static embedding-row
    surface. Port of paper §F.1 / reference impl ``run_ima_paper_like``.

    Threat model — paper §3.2 + §D.1. The attacker has:

    - the obfuscation algorithm (Kerckhoffs) including h, λ, α_e
    - the deployment's published obfuscated table W̃ (server-side, public)
    - a public corpus for generating plain_id sequences

    The attacker does NOT have:

    - the deployment's secret τ_deployment
    - the deployment's keymat K_e_deployment

    So they cannot label deployment W̃ rows with plain_ids. Instead they
    re-run Algorithm 1 with their *own* (τ_attacker, K_attacker,
    noise_attacker) to generate synthetic training pairs
    (W̃_attacker[τ_attacker[plain_id]], W_e[plain_id]), train an
    inverter on those, and apply it to the deployment's W̃ at test time.

    The TTRSR top-1 measured here is the paper-faithful inversion rate:
    fraction of test-corpus positions where cosine-NN of the inverter's
    prediction against the public W_e table lands on the true plain_id.

    Identity-τ control. When ``identity_tau=True`` (calibration probe),
    the input == target task is used directly — no attacker obfuscation,
    the inverter just has to learn the identity mapping. This is the
    attacker-capacity ceiling; a competent attacker should approach 100%
    on this task. The paper-faithful obfuscated TTRSR is meaningful as a
    defense bound only when this calibration probe saturates.

    Checkpointing. If `checkpoint_dir` is set, the function looks for a
    content-addressed `.pt` file keyed on all training hyperparameters
    except `epochs`. If found and its stored `epoch_idx >= epochs`, the
    run skips training entirely and reports the cached metrics. If the
    stored `epoch_idx < epochs`, training resumes from the saved
    best-state for `(epochs - epoch_idx)` more epochs. Otherwise (no
    checkpoint), training runs from random init.
    """
    t0 = time.perf_counter()
    torch.manual_seed(seed)

    from transformers import AutoTokenizer

    print(f"  loading tokenizer from {baseline_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(baseline_model_dir, trust_remote_code=True)

    plain_W_e = torch.from_numpy(plain.token_embd.astype(np.float32))
    obs_W_e = torch.from_numpy(obfuscated.token_embd.astype(np.float32))
    tau_t = torch.from_numpy(tau)
    d_obs = int(obs_W_e.shape[1])
    d_plain = int(plain_W_e.shape[1])
    vocab_size = int(plain.vocab_size)

    print(f"  collecting public-corpus token windows "
          f"(seq_len={sequence_length}, train={train_sequence_count}, "
          f"val={val_sequence_count}, test={test_sequence_count})")
    resolved_corpus_paths = tuple(str(p) for p in public_corpus_paths)
    corpus = _collect_public_token_windows(
        tokenizer=tokenizer,
        corpus_paths=resolved_corpus_paths,
        sequence_length=sequence_length,
        train_sequence_count=train_sequence_count,
        val_sequence_count=val_sequence_count,
        test_sequence_count=test_sequence_count,
        seed=seed,
    )

    train_plain_ids = corpus["train_plain_ids"]   # (B_train, T)
    val_plain_ids = corpus["val_plain_ids"]
    test_plain_ids = corpus["test_plain_ids"]

    # Clamp any out-of-vocab ids from the tokenizer's special-token
    # cliff (rare in the path-2 GGUF artifacts but defensive).
    train_plain_ids = train_plain_ids.clamp_(0, vocab_size - 1)
    val_plain_ids = val_plain_ids.clamp_(0, vocab_size - 1)
    test_plain_ids = test_plain_ids.clamp_(0, vocab_size - 1)

    # Multi-key paper-faithful training data.
    #
    # In `identity_tau` mode the inverter is given the trivial identity
    # task (x == y) — calibration probe, no attacker obfuscation.
    #
    # In the obfuscated case, we pre-generate K independent K_aᵏ via
    # Algorithm 1 and at each training batch sample one k and synthesize
    # input on-the-fly: `x = (W_e[plain_id] + fresh_noise) @ K_aᵏ`. The
    # attacker never sees the deployment's W̃ during training; the
    # inverter is forced to learn a *key-invariant* inversion procedure.
    sigma_e = float(np.std(plain.token_embd.astype(np.float32)))
    if identity_tau:
        keymat_pool = None
        x_train_static = plain_W_e[train_plain_ids]  # input == target
    else:
        _att_seed = int(attacker_seed if attacker_seed is not None else seed + 99999)
        print(f"  pre-generating multi-key pool: K={attacker_num_keys} attacker keymats "
              f"via Algorithm 1 (h={attacker_expansion}, λ={attacker_lam}, "
              f"α_e={attacker_alpha_e}, base_seed={_att_seed})")
        keymat_pool = _build_multikey_pool(
            d=d_plain,
            expansion=int(attacker_expansion),
            lam=float(attacker_lam),
            num_keys=int(attacker_num_keys),
            attacker_seed=_att_seed,
        )
        if keymat_pool.shape[1:] != (d_plain, d_obs):
            raise RuntimeError(
                f"keymat pool shape {tuple(keymat_pool.shape)} mismatches "
                f"(d_plain={d_plain}, d_obs={d_obs}); check --attacker-expansion"
            )
        # Static rows we'll consume per batch — keeps DataLoader simple.
        # Per batch we'll sample fresh noise + pick a key index.
        x_train_static = plain_W_e[train_plain_ids]  # (B, T, d) — clean

    # We also need val/test inputs. Test uses deployment's actual W̃ +
    # τ (what the attacker reads at inference time from the wire).
    # Val mirrors test so best-state selection generalises to test
    # distribution.
    y_train = plain_W_e[train_plain_ids]
    x_val = obs_W_e[tau_t[val_plain_ids]]
    x_test = obs_W_e[tau_t[test_plain_ids]]

    # Candidate pool for cosine-NN ranking — paper-faithful "full
    # movable vocab" by default; user can shrink for fast smoke tests.
    if candidate_pool_size is None or candidate_pool_size >= vocab_size:
        candidate_plain_ids = torch.arange(vocab_size, dtype=torch.long)
    else:
        rng = np.random.default_rng(seed + 7)
        pool = rng.choice(vocab_size, size=int(candidate_pool_size), replace=False)
        # Ensure every test id is in the pool so accuracy isn't capped.
        pool = np.unique(np.concatenate([pool, test_plain_ids.flatten().numpy()]))
        candidate_plain_ids = torch.from_numpy(pool.astype(np.int64))

    sensitive_plain_ids = _collect_sensitive_plain_ids(tokenizer)

    print(f"  building paper-like inverter (d_obs={d_obs} d_plain={d_plain} "
          f"vocab={vocab_size})")
    attack_config = _build_paper_like_inverter_config(
        observed_hidden_size=d_obs,
        vocab_size=vocab_size,
        baseline_model_dir=baseline_model_dir,
    )
    model = _PaperLikeIMAInverter(
        backbone_config=attack_config,
        target_embedding_dim=d_plain,
    )

    resolved_device = _resolve_ima_device(device)
    model = model.to(resolved_device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    # x_train_static holds clean plain embeddings; per-batch input is
    # synthesised on-device via _make_multikey_batch_input. DataLoader
    # iterates indices so we can sample noise + key on the same device.
    train_index_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.arange(x_train_static.shape[0], dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    # Per-batch noise and key-index RNG. Lives on the inverter's device
    # for fast noise sampling without host↔device transfer.
    _noise_rng_seed = int(attacker_seed if attacker_seed is not None else seed + 99999) + 7
    noise_rng = torch.Generator(device=resolved_device)
    noise_rng.manual_seed(_noise_rng_seed)
    key_rng = np.random.default_rng(_noise_rng_seed + 1)
    keymat_pool_device = keymat_pool.to(resolved_device) if keymat_pool is not None else None

    x_val_device = x_val.to(resolved_device)
    x_test_device = x_test.to(resolved_device)
    baseline_embed_device = plain_W_e.to(resolved_device)
    candidate_plain_ids_device = candidate_plain_ids.to(resolved_device)
    sensitive_plain_ids_device = sensitive_plain_ids.to(resolved_device)
    val_plain_ids_device = val_plain_ids.to(resolved_device)
    test_plain_ids_device = test_plain_ids.to(resolved_device)

    # Mixed precision (see single-key driver for rationale).
    _amp_enabled = bool(use_bf16) and resolved_device.startswith("cuda")
    _amp_device_type = "cuda" if resolved_device.startswith("cuda") else "cpu"
    if _amp_enabled:
        print(f"  bf16 autocast enabled (device_type={_amp_device_type})")

    def _amp_ctx():
        return torch.amp.autocast(
            device_type=_amp_device_type, dtype=torch.bfloat16, enabled=_amp_enabled,
        )

    def _eval_split(x_device: torch.Tensor, true_ids: torch.Tensor) -> dict[str, Any]:
        model.eval()
        with torch.no_grad(), _amp_ctx():
            pred = model(x_device).float()
            return _evaluate_sequence_inversion_predictions(
                predicted_embeddings=pred,
                true_plain_ids=true_ids,
                candidate_plain_ids=candidate_plain_ids_device,
                baseline_embed=baseline_embed_device,
                sensitive_plain_ids=sensitive_plain_ids_device,
                topk=topk,
            )

    # Checkpoint discovery + load. Content-addressed on every
    # hyperparameter except `epochs` — so longer epoch budgets can
    # resume from shorter prior runs at the same config. The checkpoint
    # stores `state_dict + optimizer + best_state + best_val_top1 +
    # epoch_idx_done + epoch_summaries`.
    ckpt_path: Path | None = None
    epochs_already_done = 0
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tau_fp = "identity" if identity_tau else (
            f"len={tau.shape[0]},sum={int(tau.sum())}"
        )
        # Multi-key driver uses a distinct fingerprint tag so single-
        # key v2 checkpoints don't shadow these runs and vice-versa.
        _bf16_tag = "bf16" if use_bf16 else "fp32"
        if identity_tau:
            attacker_fp = f"v3-multikey-identityprobe-{_bf16_tag}"
        else:
            _att_seed_for_fp = int(attacker_seed if attacker_seed is not None else seed + 99999)
            attacker_fp = (
                f"v3-multikey|h={attacker_expansion}|lam={attacker_lam}|"
                f"alpha_e={attacker_alpha_e}|K={attacker_num_keys}|seed={_att_seed_for_fp}|{_bf16_tag}"
            )
        ckpt_path = _ima_xformer_ckpt_path(
            checkpoint_dir=checkpoint_dir,
            plain_path=str(plain.path) if hasattr(plain, "path") else None,
            obfuscated_path=str(obfuscated.path) if hasattr(obfuscated, "path") else None,
            tau_fingerprint=tau_fp,
            baseline_model_dir=baseline_model_dir,
            sequence_length=sequence_length,
            train_sequence_count=train_sequence_count,
            val_sequence_count=val_sequence_count,
            test_sequence_count=test_sequence_count,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            candidate_pool_size=candidate_pool_size,
            seed=seed,
            attacker_fingerprint=attacker_fp,
        )
        # Sibling best-state file; see save block below.
        best_state_path = ckpt_path.parent / f"{ckpt_path.stem}.best{ckpt_path.suffix}"
        if ckpt_path.exists():
            print(f"  found checkpoint {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=resolved_device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            except (ValueError, KeyError) as e:
                # Optimizer state can fail to load across torch versions; safe to skip.
                print(f"  warn: could not restore optimizer state ({e}); re-initialising AdamW")
            epochs_already_done = int(ckpt.get("epochs_done", 0))
            print(f"  resumed at epoch {epochs_already_done} "
                  f"(prev best val top1={ckpt.get('best_val_top1', '?')})")

    # Evaluate at step 0 (BEFORE any training) so a paper-default
    # 2-epoch run that degrades the random-init state can still pick
    # up the better checkpoint. Cheap insurance — the reference impl
    # doesn't do this but it's strictly an improvement.
    init_metrics = _eval_split(x_val_device, val_plain_ids_device)

    if epochs_already_done > 0 and ckpt_path is not None and ckpt_path.exists():
        # Resume bookkeeping from the checkpoint, not init.
        best_epoch = int(ckpt.get("best_epoch", epochs_already_done))
        best_val_top1 = float(ckpt.get("best_val_top1", init_metrics["token_top1_recovery_rate"]))
        # best_state lookup priority: sibling .best.pt → legacy inline →
        # current model_state fall-through. See save block.
        if best_state_path.exists():
            print(f"  loading best-state from sibling file {best_state_path.name}")
            _best_ckpt = torch.load(best_state_path, map_location="cpu", weights_only=False)
            best_state = {k: v.clone() for k, v in _best_ckpt["best_state"].items()}
            del _best_ckpt
        elif "best_state" in ckpt:
            print("  loading best-state inline (legacy checkpoint format)")
            best_state = {k: v.clone() for k, v in ckpt["best_state"].items()}
        else:
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        epoch_summaries = list(ckpt.get("epoch_summaries", []))
    else:
        best_epoch = 0
        best_val_top1 = float(init_metrics["token_top1_recovery_rate"])
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        epoch_summaries = [{
            "epoch": 0,
            "train_loss": None,
            "val_token_top1_recovery_rate": best_val_top1,
            "val_token_top10_recovery_rate": float(init_metrics["token_top10_recovery_rate"]),
            "val_embedding_cosine_similarity": float(init_metrics["embedding_cosine_similarity"]),
        }]

    if epochs_already_done >= int(epochs):
        print(f"  checkpoint covers requested epochs={epochs} "
              f"(done={epochs_already_done}); skipping additional training")
        epochs_to_run = 0
    else:
        epochs_to_run = int(epochs) - epochs_already_done

    # Eval cadence — full-vocab cosine-NN against 151k candidates is
    # ~5 s/eval on GPU; per-epoch eval at 100 epochs spent 80 % of total
    # wall time on monitoring. We only need val for (a) best-state
    # selection and (b) the convergence-curve readout — both are fine
    # at coarser resolution. The schedule below: dense early (when the
    # curve is moving fast), log-spaced later, plus the final epoch.
    def _should_eval(global_epoch: int, total_epochs: int) -> bool:
        # Always eval at the last epoch.
        if global_epoch == total_epochs:
            return True
        # Dense early (every epoch ≤ 5).
        if global_epoch <= 5:
            return True
        # Mid-range every 5.
        if global_epoch <= 50 and global_epoch % 5 == 0:
            return True
        # Long-tail every 25.
        if global_epoch <= 500 and global_epoch % 25 == 0:
            return True
        # Very long every 100.
        if global_epoch % 100 == 0:
            return True
        return False

    # Move static clean plain embeddings to device once (cheap;
    # train_size_sequences × seq_len × d float32).
    x_train_static_device = x_train_static.to(resolved_device)
    y_train_static_device = y_train.to(resolved_device)

    for epoch_idx in range(epochs_to_run):
        epoch_idx = epochs_already_done + epoch_idx  # global epoch counter
        model.train()
        total_loss = 0.0
        total_batches = 0
        for (idx_batch,) in train_index_loader:
            idx_batch = idx_batch.to(resolved_device)
            # Gather this batch's clean rows + targets
            clean_rows = x_train_static_device.index_select(0, idx_batch)  # (B, T, d)
            batch_targets = y_train_static_device.index_select(0, idx_batch)  # (B, T, d)

            optimizer.zero_grad(set_to_none=True)
            with _amp_ctx():
                # Identity-τ path: input = target (no obfuscation).
                # Obfuscated path: synth input per batch with sampled k.
                if identity_tau or keymat_pool_device is None:
                    batch_inputs = clean_rows
                else:
                    # Sample a key index for this batch — uniform over the K pool.
                    k = int(key_rng.integers(0, int(attacker_num_keys)))
                    B, T, d = clean_rows.shape
                    flat_clean = clean_rows.reshape(B * T, d)
                    flat_input = _make_multikey_batch_input(
                        plain_W_e_rows=flat_clean,
                        keymat_pool=keymat_pool_device,
                        key_idx=k,
                        alpha_e=float(attacker_alpha_e),
                        sigma_e=sigma_e,
                        rng_torch=noise_rng,
                    )
                    d_obs_local = flat_input.shape[-1]
                    batch_inputs = flat_input.reshape(B, T, d_obs_local)

                pred = model(batch_inputs)
                loss = torch.nn.functional.mse_loss(pred, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_batches += 1

        global_epoch = epoch_idx + 1
        if not _should_eval(global_epoch, int(epochs)):
            continue

        val_metrics = _eval_split(x_val_device, val_plain_ids_device)
        epoch_summaries.append({
            "epoch": global_epoch,
            "train_loss": total_loss / max(total_batches, 1),
            "val_token_top1_recovery_rate": float(val_metrics["token_top1_recovery_rate"]),
            "val_token_top10_recovery_rate": float(val_metrics["token_top10_recovery_rate"]),
            "val_embedding_cosine_similarity": float(val_metrics["embedding_cosine_similarity"]),
        })
        if val_metrics["token_top1_recovery_rate"] > best_val_top1:
            best_val_top1 = float(val_metrics["token_top1_recovery_rate"])
            best_epoch = global_epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    # Persist checkpoint so longer epoch budgets at the same config
    # can resume. Two files: the main `.pt` holds model + optimizer
    # state needed for continued training; the sibling `.best.pt`
    # holds the best-val state for the final eval restore. Splitting
    # them saves ~1 GB per checkpoint vs the legacy inline format.
    # Continue-training quality is unchanged — only the inert
    # best_state copy was moved.
    if ckpt_path is not None:
        # Defensive write: surface any error loudly rather than silently
        # losing the checkpoint. The first time we shipped this, the
        # container had HOME unset under --user 1000:1000, Python's
        # Path.home() resolved to "/" inside the container, and the
        # 11-minute training run wrote nothing to disk with no message.
        try:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer_state": optimizer.state_dict(),
                "epochs_done": int(epochs),
                "best_epoch": int(best_epoch),
                "best_val_top1": float(best_val_top1),
                "epoch_summaries": epoch_summaries,
            }, ckpt_path)
            print(f"  saved checkpoint → {ckpt_path}")
            torch.save({"best_state": best_state}, best_state_path)
            print(f"  saved best-state → {best_state_path.name}")
        except Exception as e:
            print(f"  ERROR saving checkpoint → {ckpt_path}: {e!r}")
            print(f"  (training output is still valid; only the resumable "
                  f"state was lost. Investigate: is the path writable by "
                  f"the current user? cwd={Path.cwd()}, HOME={os.environ.get('HOME', '<unset>')})")

    # Restore best-val checkpoint, run final test eval.
    model.load_state_dict(best_state)
    model.to(resolved_device)
    test_metrics = _eval_split(x_test_device, test_plain_ids_device)

    top1 = float(test_metrics["token_top1_recovery_rate"])
    top10 = float(test_metrics["token_top10_recovery_rate"])

    return AttackResult(
        attack="ima_embedrow_transformer",
        condition="obfuscated",
        model_id=str(obfuscated.path.name),
        n_prompts=int(train_plain_ids.shape[0]),
        n_train=int(train_plain_ids.numel()),
        n_test=int(test_plain_ids.numel()),
        ttrsr_top1=top1,
        ttrsr_top10=top10,
        risk_level=classify_risk_level(top1),
        extra={
            "baseline_model_dir": baseline_model_dir,
            "sequence_length": int(sequence_length),
            "train_sequence_count": int(train_plain_ids.shape[0]),
            "val_sequence_count": int(val_plain_ids.shape[0]),
            "test_sequence_count": int(test_plain_ids.shape[0]),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "best_epoch": int(best_epoch),
            "epoch_summaries": epoch_summaries,
            "embedding_cosine_similarity": float(test_metrics["embedding_cosine_similarity"]),
            "candidate_pool_size": int(candidate_plain_ids.numel()),
            "device": str(resolved_device),
            "runtime_seconds": round(time.perf_counter() - t0, 2),
            "threat_model_regime": "v3_multikey_paperfaithful",
            "attacker_identity_probe": bool(identity_tau),
            "attacker_expansion": int(attacker_expansion),
            "attacker_lam": float(attacker_lam),
            "attacker_alpha_e": float(attacker_alpha_e),
            "attacker_seed": int(attacker_seed) if attacker_seed is not None else int(seed + 99999),
            "attacker_num_keys": int(attacker_num_keys),
            "use_bf16": bool(use_bf16),
        },
    )


# ───── CLI ─────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description="Run IMA-EmbedRow prompt-inversion attacks")
    p.add_argument("--plain", type=Path, required=True)
    p.add_argument("--obfuscated", type=Path, required=True)
    p.add_argument(
        "--key",
        type=Path,
        help=".key.npz produced by obfuscate_qwen3_gguf.py (contains τ). "
             "Omit with --identity-tau for plain-side control runs.",
    )
    p.add_argument(
        "--identity-tau",
        action="store_true",
        help="Use τ = identity instead of loading from --key. Use with "
             "--plain == --obfuscated to measure the plain-side control "
             "(attack should succeed at ~100 % since the bijection is "
             "trivial). Verifies the attack itself works.",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ridge-train-size", type=int, default=1024)
    p.add_argument("--ridge-val-size", type=int, default=128)
    p.add_argument("--ridge-test-size", type=int, default=128)
    p.add_argument("--ridge-candidate-pool-size", type=int, default=2048)
    # Paper-faithful transformer-inverter parameters (paper §F.1 / §F.2,
    # reference impl `run_ima_paper_like`).
    p.add_argument(
        "--baseline-model-dir", type=str, default="Qwen/Qwen3-4B",
        help="HF model id whose AutoConfig + tokenizer drive the inverter's "
             "backbone architecture. The trained inverter uses 2 hidden "
             "layers from this config with hidden_size overridden to d_obs. "
             "Default Qwen3-4B; pass Qwen/Qwen3-8B for the 8B cell.",
    )
    p.add_argument(
        "--public-corpus-path", type=Path, action="append", default=None,
        help="Public corpus text/markdown files for sequence-window training "
             "(repeatable). Default: vendor/aloepri-py/docs/*.{md,txt} + "
             "vendor/aloepri-py/README.md (matches reference impl exactly).",
    )
    p.add_argument("--paper-sequence-length", type=int, default=32,
                   help="Token-window length (paper Table 9).")
    p.add_argument("--paper-train-sequence-count", type=int, default=128,
                   help="Number of training windows (paper Table 9).")
    p.add_argument("--paper-val-sequence-count", type=int, default=16)
    p.add_argument("--paper-test-sequence-count", type=int, default=16)
    p.add_argument("--paper-batch-size", type=int, default=8,
                   help="paper §F.2.")
    p.add_argument("--paper-epochs", type=int, default=2,
                   help="paper §F.2.")
    p.add_argument("--paper-lr", type=float, default=3e-4,
                   help="paper §F.2.")
    p.add_argument("--paper-weight-decay", type=float, default=0.0,
                   help="paper §F.2.")
    p.add_argument(
        "--paper-candidate-pool-size", type=int, default=0,
        help="Candidate vocab pool for cosine-NN ranking. 0 (default) = "
             "full vocab (paper-faithful). Smaller → faster smoke runs.",
    )
    p.add_argument(
        "--paper-device", type=str, default="auto",
        choices=("auto", "gpu", "cpu", "cuda"),
        help="Device for the trained inverter. 'auto' uses GPU if "
             "available, else CPU. 'gpu' = ROCm/HIP on AMD or CUDA on "
             "NVIDIA (PyTorch reuses the 'cuda' device-string for both). "
             "'cuda' is an alias for 'gpu' for backwards compatibility "
             "with reference impl naming.",
    )
    p.add_argument(
        "--paper-checkpoint-dir", type=Path,
        default=Path.home() / ".cache" / "aloepri-ima-checkpoints",
        help="Where to cache trained-inverter checkpoints. Path is "
             "content-addressed on all hyperparameters except --paper-epochs, "
             "so a 1000-epoch run will resume from a prior 100-epoch run at "
             "the same config. Set to empty string to disable checkpointing.",
    )
    p.add_argument(
        "--skip-transformer",
        action="store_true",
        help="Skip the slow trained-inverter attack (ridge only).",
    )
    # Attacker-side Algorithm 1 parameters. Under Kerckhoffs the
    # attacker knows these (algorithm public, keys secret). Default to
    # the path-2 paper-default knobs; override if the deployment used
    # a different (h, λ, α_e). The attacker_seed is independent from
    # both the deployment and the training/eval RNG so different attack
    # runs against the same deployment exercise different attacker keys.
    p.add_argument(
        "--attacker-expansion", type=int, default=128,
        help="Algorithm 1 expansion size h the attacker uses. Public.",
    )
    p.add_argument(
        "--attacker-lambda", type=float, default=0.3,
        help="Algorithm 1 λ the attacker uses. Public.",
    )
    p.add_argument(
        "--attacker-alpha-e", type=float, default=1.0,
        help="Algorithm 1 α_e the attacker uses. Public.",
    )
    p.add_argument(
        "--attacker-seed", type=int, default=None,
        help="Base seed for the attacker's keymat pool + noise RNG. If "
             "omitted, derived as `seed + 99999`. Setting this lets the "
             "user vary the attacker keys while holding everything else "
             "fixed — useful for confirming the attack is τ-key-invariant.",
    )
    p.add_argument(
        "--attacker-num-keys", type=int, default=64,
        help="K = number of independent attacker keymats pre-generated "
             "for multi-key training. Each batch samples one k uniformly. "
             "Larger K → more key-invariance but more memory (≈ K · d · "
             "(d+2h) · 4 bytes). Default 64 → ~1.7 GB on Q3-4B, ~4.4 GB "
             "on Q3-8B.",
    )
    p.add_argument(
        "--no-bf16", action="store_true",
        help="Disable bf16 autocast (use fp32 forward+loss). Default is "
             "bf16 autocast; ~2× speed on ROCm Strix Halo with no "
             "measurable accuracy loss.",
    )
    from m2_7_common import add_min_mem_args, check_phase_memory  # type: ignore
    add_min_mem_args(p, phase="ima_embedrow_attacks")
    args = p.parse_args()

    check_phase_memory("ima_embedrow_attacks", args.min_mem_gb, args.skip_mem_check)

    print(f"[IMA-EmbedRow] loading plaintext GGUF: {args.plain}")
    plain = load_model(args.plain, "plaintext", embed_only=True)
    print(
        f"  loaded vocab={plain.vocab_size} d_eff={plain.d_eff} "
        f"n_layers={plain.n_layers}"
    )

    print(f"[IMA-EmbedRow] loading obfuscated GGUF: {args.obfuscated}")
    obfuscated = load_model(args.obfuscated, "obfuscated", embed_only=True)
    print(
        f"  loaded vocab={obfuscated.vocab_size} d_eff={obfuscated.d_eff} "
        f"n_layers={obfuscated.n_layers}"
    )

    if plain.vocab_size != obfuscated.vocab_size:
        raise SystemExit(
            f"vocab size mismatch: plain={plain.vocab_size} "
            f"obs={obfuscated.vocab_size} — refusing to run IMA-EmbedRow"
        )

    if args.identity_tau:
        # Plain control: τ = identity → x_train == y_train (up to noise).
        # The active vocab range only matters for splitting train/val/test —
        # use the full loaded vocab so the splits cover the whole table.
        # (Pre-2026-05-20 this was hard-coded 151669 = Qwen3-1.7B's
        # permutable count, which under-sampled the 4B/8B test pool.)
        active_size = plain.vocab_size
        tau = np.arange(plain.vocab_size, dtype=np.int64)
        print(f"[IMA-EmbedRow] τ = identity (plain control); "
              f"active_size={active_size} (= vocab_size)")
    else:
        if args.key is None:
            raise SystemExit(
                "--key is required unless --identity-tau is set"
            )
        print(f"[IMA-EmbedRow] loading τ from {args.key}")
        tau, active_size = load_tau(args.key)
        if tau.shape[0] != plain.vocab_size:
            raise SystemExit(
                f"τ length {tau.shape[0]} != vocab_size {plain.vocab_size}"
            )
        print(f"  τ active_size={active_size} (rest identity)")

    results: dict[str, dict[str, Any]] = {}

    print("[IMA-EmbedRow] running IMA-EmbedRow-ridge (multi-α ridge on embed rows)…")
    ridge = run_ima_embedrow_ridge(
        plain,
        obfuscated,
        tau,
        active_size=active_size,
        train_size=args.ridge_train_size,
        val_size=args.ridge_val_size,
        test_size=args.ridge_test_size,
        candidate_pool_size=args.ridge_candidate_pool_size,
    )
    print(
        f"  ima_embedrow_ridge top1={ridge.ttrsr_top1:.4f} top10={ridge.ttrsr_top10:.4f} "
        f"risk={ridge.risk_level} α*={ridge.extra['best_ridge_alpha']}"
    )
    results["ima_embedrow_ridge"] = ridge.to_dict()

    if not args.skip_transformer:
        print("[IMA-EmbedRow] running IMA-EmbedRow-transformer "
              "(paper §F.1 trained Qwen-backbone inverter, public-corpus pipeline)…")
        corpus_paths = (
            tuple(str(p) for p in args.public_corpus_path)
            if args.public_corpus_path
            else DEFAULT_PUBLIC_CORPUS_PATHS
        )
        xformer = run_ima_embedrow_transformer(
            plain,
            obfuscated,
            tau,
            baseline_model_dir=args.baseline_model_dir,
            public_corpus_paths=corpus_paths,
            sequence_length=args.paper_sequence_length,
            train_sequence_count=args.paper_train_sequence_count,
            val_sequence_count=args.paper_val_sequence_count,
            test_sequence_count=args.paper_test_sequence_count,
            batch_size=args.paper_batch_size,
            epochs=args.paper_epochs,
            learning_rate=args.paper_lr,
            weight_decay=args.paper_weight_decay,
            candidate_pool_size=(
                args.paper_candidate_pool_size if args.paper_candidate_pool_size > 0 else None
            ),
            # Map our friendlier 'gpu' alias to PyTorch's 'cuda' device
            # string (used for both NVIDIA CUDA and AMD ROCm/HIP).
            device=("cuda" if args.paper_device == "gpu" else args.paper_device),
            checkpoint_dir=(
                args.paper_checkpoint_dir
                if args.paper_checkpoint_dir and str(args.paper_checkpoint_dir)
                else None
            ),
            identity_tau=args.identity_tau,
            attacker_expansion=args.attacker_expansion,
            attacker_lam=args.attacker_lambda,
            attacker_alpha_e=args.attacker_alpha_e,
            attacker_seed=args.attacker_seed,
            attacker_num_keys=args.attacker_num_keys,
            use_bf16=not args.no_bf16,
        )
        print(
            f"  ima_embedrow_transformer top1={xformer.ttrsr_top1:.4f} "
            f"top10={xformer.ttrsr_top10:.4f} risk={xformer.risk_level} "
            f"best_epoch={xformer.extra['best_epoch']}"
        )
        results["ima_embedrow_transformer"] = xformer.to_dict()
    else:
        results["ima_embedrow_transformer"] = {
            "attack": "ima_embedrow_transformer",
            "risk_level": "skipped",
            "extra": {"note": "--skip-transformer was set"},
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "format": "aloepri_m2_7_ima_embedrow_v1",
        "plain_path": str(args.plain),
        "obfuscated_path": str(args.obfuscated),
        "key_path": str(args.key),
        "attacks": results,
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"[IMA-EmbedRow] wrote → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

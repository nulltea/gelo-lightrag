"""Run the IMA-EmbedRow prompt-inversion attacks against the §05 obfuscated GGUF.

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


# ───── IMA-EmbedRow-transformer (trained 2-layer inverter) ───────


class _InverterBlock(nn.Module):
    """Pre-LN GELU FFN residual block with identity-init.

    No MHA — the attack target is a per-row mapping (each obfuscated
    embedding row → its plain partner), so sequence attention adds
    capacity without useful inductive bias and the optimiser settles
    on memorising solutions that don't transfer to vocab-disjoint
    test rows. We keep the trunk pure residual MLP so the block has
    a single learnable path (the FFN delta) and zero-init the FFN's
    second Linear so the block ≡ identity at t = 0.
    """

    def __init__(self, hidden: int, ffn_mult: int = 4) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * ffn_mult, bias=False),
            nn.GELU(),
            nn.Linear(hidden * ffn_mult, hidden, bias=False),
        )
        nn.init.zeros_(self.ffn[2].weight)  # block ≡ x at t=0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.ln(x))


class _RowInverter(nn.Module):
    """`(B, observed_dim) → (B, output_dim)` row inverter.

    No input bottleneck: `inverter_hidden` defaults to `observed_dim`,
    so the input projection is an Identity (or a small mixing matrix
    only when obs ≠ hidden by user override). The first version of
    this driver hard-coded `inverter_hidden=256` against a 2 304-dim
    input — that discarded 89 % of the row signal at the input layer
    and the trunk could only produce rank-256 projections of the
    target, collapsing every prediction to the corpus centroid.

    Trunk is a stack of identity-init residual MLP blocks (no
    cross-row attention). The output projection then has to learn an
    approximate inverse of the obfuscation — initialised randomly,
    refined by AdamW under MSE + weight decay.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        inverter_hidden: int,
        n_layers: int = 2,
        output_dim: int,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        self.input_proj: nn.Module
        if observed_dim != inverter_hidden:
            self.input_proj = nn.Linear(observed_dim, inverter_hidden, bias=False)
        else:
            self.input_proj = nn.Identity()
        self.blocks = nn.ModuleList(
            [_InverterBlock(inverter_hidden, ffn_mult) for _ in range(n_layers)]
        )
        self.output_proj = nn.Linear(inverter_hidden, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, observed_dim).
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.output_proj(h)


def run_ima_embedrow_transformer(
    plain: ModelWeights,
    obfuscated: ModelWeights,
    tau: np.ndarray,
    *,
    active_size: int,
    train_size: int = 4096,
    val_size: int = 256,
    test_size: int = 256,
    candidate_pool_size: int = 2048,
    inverter_hidden: int | None = None,
    n_layers: int = 2,
    epochs: int = 16,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    topk: int = 10,
    seed: int = 20260518,
) -> AttackResult:
    """Trained-inverter version on flat row pairs (no sequence packing).

    Architecture: input_proj → n_layers × identity-init residual MLP
    blocks → output_proj. With <code>inverter_hidden = observed_dim</code>
    by default, the input projection is identity; the trunk starts as
    exact identity (FFN deltas zero-init); the output projection is
    random and the optimiser learns the obfuscation inverse via MSE +
    weight-decay AdamW. Sequence-input + MHA were removed because they
    let the optimiser memorise sequence-level patterns that don't
    transfer to vocab-disjoint test rows.
    """
    t0 = time.perf_counter()
    torch.manual_seed(seed)

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

    if inverter_hidden is None:
        inverter_hidden = obs_W_e.shape[1]  # no bottleneck

    device = torch.device("cpu")
    model = _RowInverter(
        observed_dim=obs_W_e.shape[1],
        inverter_hidden=inverter_hidden,
        n_layers=n_layers,
        output_dim=plain_W_e.shape[1],
    ).to(device)

    # Warm-start output_proj at the ridge closed-form solution. With
    # identity-init residual trunk (FFN second-linear zero-init), the
    # network at t=0 IS the ridge inverter. From random init AdamW
    # only manages ~2 % of the W movement needed in paper-default
    # 1024 GD steps (||W − I||_F drops 52.2 → 51.2 on the minimal-pipeline
    # diagnostic); the network never reaches ridge's closed-form
    # solution. Warm-starting gets us there at step 0 — any further
    # AdamW movement is non-linear refinement on top.
    with torch.no_grad():
        # Match `inverter_hidden` to the input projection: if input_proj
        # is Identity (inverter_hidden==observed_dim), the ridge regressor
        # acts directly on observed rows; otherwise `model.input_proj`
        # would re-project, and warm-start would be wrong without
        # accounting for it. The default path (inverter_hidden=observed_dim)
        # is the common case.
        if isinstance(model.input_proj, nn.Identity):
            ridge_alpha_for_init = 1e-2
            xt_x = x_train.T @ x_train
            xt_y = x_train.T @ y_train
            n_in = x_train.shape[1]
            reg = ridge_alpha_for_init * torch.eye(n_in, dtype=x_train.dtype)
            # Ridge yields `W_ridge: (in, out)` such that `x @ W_ridge ≈ y`.
            w_ridge = torch.linalg.solve(xt_x + reg, xt_y)
            # output_proj.weight is (out, in) — transpose.
            model.output_proj.weight.copy_(w_ridge.T.contiguous())

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = x_train.shape[0]

    best_val_top1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    epoch_summaries: list[dict[str, Any]] = []

    for epoch in range(int(epochs)):
        model.train()
        order = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            batch_idx = order[start : start + batch_size]
            x_batch = x_train[batch_idx].to(device)
            y_batch = y_train[batch_idx].to(device)
            pred = model(x_batch)
            loss = nn.functional.mse_loss(pred, y_batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val.to(device)).cpu()
        val_metrics = _evaluate_inversion_predictions(
            predicted_embeddings=val_pred,
            true_plain_ids=val_ids,
            candidate_plain_ids=val_candidate_ids,
            baseline_embed=plain_W_e,
            topk=topk,
        )
        epoch_summaries.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / max(n_batches, 1),
                "val_top1": val_metrics["token_top1_recovery_rate"],
                "val_top10": val_metrics["token_top10_recovery_rate"],
            }
        )
        if val_metrics["token_top1_recovery_rate"] > best_val_top1:
            best_val_top1 = float(val_metrics["token_top1_recovery_rate"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(x_test.to(device)).cpu()

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
        attack="ima_embedrow_transformer",
        condition="obfuscated",
        model_id=str(obfuscated.path.name),
        n_prompts=0,
        n_train=int(train_ids.shape[0]),
        n_test=int(test_ids.shape[0]),
        ttrsr_top1=top1,
        ttrsr_top10=top10,
        risk_level=classify_risk_level(top1),
        extra={
            "inverter_hidden": inverter_hidden,
            "n_layers": n_layers,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "epoch_summaries": epoch_summaries,
            "embedding_cosine_similarity": float(metrics["embedding_cosine_similarity"]),
            "candidate_pool_size": int(candidate_ids.shape[0]),
            "runtime_seconds": round(time.perf_counter() - t0, 2),
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
    p.add_argument("--transformer-train-size", type=int, default=4096)
    p.add_argument("--transformer-val-size", type=int, default=256)
    p.add_argument("--transformer-test-size", type=int, default=256)
    p.add_argument("--transformer-candidate-pool-size", type=int, default=2048)
    p.add_argument("--transformer-epochs", type=int, default=16)
    p.add_argument("--transformer-hidden", type=int, default=0,
                   help="Inverter hidden dim. 0 (default) = observed_dim (no "
                        "bottleneck). Setting this below observed_dim drops "
                        "input dims and the model collapses to centroid — "
                        "see docstring on _RowInverter for the v1 bug that "
                        "motivated the default.")
    p.add_argument(
        "--skip-transformer",
        action="store_true",
        help="Skip the slow trained-inverter attack (ridge only).",
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
        print("[IMA-EmbedRow] running IMA-EmbedRow-transformer (trained 2-layer inverter)…")
        xformer = run_ima_embedrow_transformer(
            plain,
            obfuscated,
            tau,
            active_size=active_size,
            train_size=args.transformer_train_size,
            val_size=args.transformer_val_size,
            test_size=args.transformer_test_size,
            candidate_pool_size=args.transformer_candidate_pool_size,
            epochs=args.transformer_epochs,
            inverter_hidden=(args.transformer_hidden if args.transformer_hidden > 0 else None),
        )
        print(
            f"  ima_embedrow_transformer top1={xformer.ttrsr_top1:.4f} "
            f"top10={xformer.ttrsr_top10:.4f} risk={xformer.risk_level}"
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

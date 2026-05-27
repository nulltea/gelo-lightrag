"""V/O per-channel magnitude attack — targets Û_vo specifically.

Threat model: adversary has plain + obfuscated GGUF. Algorithm-2 transforms
V and O *jointly* — W_v right-mult by Û_vo per head, W_o left-mult by
Û_vo⁻¹ per head. The product `W_v · W_o` is INVARIANT to Û_vo, so any
attack working on the product sees Û_vo as a no-op. We want the OPPOSITE
attack: look at W_v and W_o **separately** with signatures that DO depend
on the per-head dense transform.

Per-channel L2 magnitude is such a signature: for head block B
∈ R^{head_dim × d_eff}, row L2 norms `||B[i, :]||₂` only survive an exactly
orthogonal right-multiplication. AloePri samples Û_vo from a random
Gaussian and QR-stabilises it, so the result is *near-orthogonal* — row
norms drift modestly on V (Û_vo right-mult, near-orthogonal preserves them)
and more visibly on O (Û_vo⁻¹ left-mult on the input-axis block changes
per-input-channel column magnitudes proportional to Û_vo⁻¹'s row norms,
which QR-with-scaling does NOT fully cancel).

Π_head meanwhile shuffles heads → magnitude *vectors* survive as wholes,
attached to different indices. NN matching therefore recovers head identity
through Π_head while degrading under Û_vo — measuring Û_vo's contribution.

Per-tensor head slicing (GGUF natural shape):
  * W_v: (kv_dim, d_eff)  — heads on axis 0, head h rows [h·hd : (h+1)·hd]
  * W_o: (d_eff, q_dim)   — heads on axis 1, head h cols [h·hd : (h+1)·hd]

Surprise during design: Û_vo's QR-stabilisation makes V-side row magnitudes
barely move, so we added a top-K spectral signature alongside L2 magnitudes
(`--singular-values 8`, opt-out via `--no-spectra`). Headline metric is
top-1 (V, O)-pair match rate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_gguf_weights import ModelWeights, load_model  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attack_drivers.common import AttackResult, classify_risk_level  # type: ignore  # noqa: E402


# ───── per-channel magnitude signatures ─────────────────────────────


def _v_head_magnitudes(W_v: np.ndarray, n_kv_heads: int, head_dim: int) -> np.ndarray:
    """Per-row L2 norms of each V head block.

    Returns shape (n_kv_heads, head_dim) — one magnitude vector per head.
    """
    arr = W_v.astype(np.float32, copy=False)
    out = np.zeros((n_kv_heads, head_dim), dtype=np.float32)
    for h in range(n_kv_heads):
        block = arr[h * head_dim : (h + 1) * head_dim, :]
        out[h] = np.linalg.norm(block, axis=1)
    return out


def _o_head_magnitudes(W_o: np.ndarray, n_q_heads: int, head_dim: int) -> np.ndarray:
    """Per-column-within-head L2 norms of each O head block.

    W_o natural shape is (d_eff, q_dim). Head h occupies columns
    [h*head_dim : (h+1)*head_dim]. We take the L2 norm of each column
    in the block over the d_eff axis → (n_q_heads, head_dim).
    """
    arr = W_o.astype(np.float32, copy=False)
    out = np.zeros((n_q_heads, head_dim), dtype=np.float32)
    for h in range(n_q_heads):
        block = arr[:, h * head_dim : (h + 1) * head_dim]
        out[h] = np.linalg.norm(block, axis=0)
    return out


def _head_spectra(
    W: np.ndarray, n_heads: int, head_dim: int, *, axis: int, k: int
) -> np.ndarray:
    """Top-k singular values per head block (descending, zero-padded).

    Auxiliary spectral fingerprint that picks up the residual scaling
    Û_vo's QR-stabilisation leaves behind after row L2 norms fail to
    distinguish near-orthogonal Q from identity.
    """
    out = np.zeros((n_heads, k), dtype=np.float32)
    for h in range(n_heads):
        s = slice(h * head_dim, (h + 1) * head_dim)
        block = W[s, :] if axis == 0 else W[:, s]
        sv = np.linalg.svd(block.astype(np.float32, copy=False), compute_uv=False)
        out[h, : min(sv.shape[0], k)] = sv[:k]
    return out


def _match_nn(plain: np.ndarray, obs: np.ndarray, topk: int) -> np.ndarray:
    """For each obs row, return the indices of the topk-closest plain
    rows by L2 distance of the feature vector. Shape: (n_obs, topk).
    """
    pp = (plain * plain).sum(axis=1, keepdims=True)
    oo = (obs * obs).sum(axis=1, keepdims=True)
    dist2 = oo - 2.0 * obs @ plain.T + pp.T
    k = min(topk, plain.shape[0])
    return np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]


# ───── ground truth from key.npz ────────────────────────────────────


def _build_head_perm_from_key(
    key: dict[str, np.ndarray],
    layer: int,
    n_q_heads: int,
    n_kv_heads: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct (q_perm, kv_perm) per `lib/alg2.py`'s GQA layout.

    Each is obs-indexed: `q_perm[j] = plain Q-head id of obs Q-head j`,
    likewise for kv. Identity vectors if alg2 keys absent.
    """
    tau_kv = key.get(f"alg2_l{layer}_tau_kv")
    tau_group = key.get(f"alg2_l{layer}_tau_group")
    num_groups = n_q_heads // n_kv_heads
    if num_groups * n_kv_heads != n_q_heads:
        raise ValueError(
            f"GQA mismatch: n_q={n_q_heads} not multiple of n_kv={n_kv_heads}"
        )
    grouped = np.arange(n_q_heads, dtype=np.int64).reshape(n_kv_heads, num_groups)
    if tau_kv is not None:
        grouped = grouped[tau_kv, :]
    if tau_group is not None:
        grouped = grouped[:, tau_group]
    q_perm = grouped.reshape(-1)
    kv_perm = (
        tau_kv.astype(np.int64) if tau_kv is not None
        else np.arange(n_kv_heads, dtype=np.int64)
    )
    return q_perm, kv_perm


# ───── attack runner ────────────────────────────────────────────────


def run_vo_channel_pair(
    plain: ModelWeights,
    obfuscated: ModelWeights,
    *,
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    head_perm_by_layer: dict[int, tuple[np.ndarray, np.ndarray]] | None,
    singular_values: int = 8,
    topk: int = 5,
    use_spectra_aux: bool = True,
    layer_indices: tuple[int, ...] | None = None,
) -> dict[str, AttackResult]:
    """Run the V/O per-channel attack across all layers; return three
    AttackResults: v_match, o_match, vo_pair_match.

    Matching strategy:
      * V-match: NN on (magnitudes ⊕ optional spectra) for V heads.
      * O-match: NN on (magnitudes ⊕ optional spectra) for O heads.
      * (V, O)-pair: O head j's predicted Q-head id is matches_o[j, 0];
        its KV-group parent is `predicted_q // num_groups`; pair-hit if
        that equals the ground-truth KV head for obs V at index
        `q // num_groups`. Concretely we ask: does the O-side prediction
        agree with the V-side prediction modulo GQA grouping?
    """
    t0 = time.perf_counter()
    n_layers = min(plain.n_layers, obfuscated.n_layers)
    if layer_indices is None:
        layer_indices = tuple(range(n_layers))
    num_groups = n_q_heads // n_kv_heads

    totals_v = totals_o = totals_pair = 0
    hits_v_top1 = hits_v_topk = 0
    hits_o_top1 = hits_o_topk = 0
    hits_pair_top1 = 0
    per_layer: list[dict[str, Any]] = []

    for li in layer_indices:
        plain_v = plain.per_layer[li].get("attn_v")
        plain_o = plain.per_layer[li].get("attn_output")
        obs_v = obfuscated.per_layer[li].get("attn_v")
        obs_o = obfuscated.per_layer[li].get("attn_output")
        if any(W is None for W in (plain_v, plain_o, obs_v, obs_o)):
            continue

        # Sort head_dim coordinates within each head's magnitude vector
        # so the signature is invariant to *intra-head* axis reordering
        # (none applies in Û_vo, but it removes a degree of freedom that
        # would inflate noise). Pair this with sorted-spectrum (which is
        # already sorted by SVD).
        v_mag_plain = np.sort(
            _v_head_magnitudes(plain_v, n_kv_heads, head_dim), axis=1
        )
        v_mag_obs = np.sort(
            _v_head_magnitudes(obs_v, n_kv_heads, head_dim), axis=1
        )
        o_mag_plain = np.sort(
            _o_head_magnitudes(plain_o, n_q_heads, head_dim), axis=1
        )
        o_mag_obs = np.sort(
            _o_head_magnitudes(obs_o, n_q_heads, head_dim), axis=1
        )

        if use_spectra_aux:
            kk = singular_values
            v_plain = np.concatenate(
                [v_mag_plain, _head_spectra(plain_v, n_kv_heads, head_dim, axis=0, k=kk)], axis=1)
            v_obs = np.concatenate(
                [v_mag_obs, _head_spectra(obs_v, n_kv_heads, head_dim, axis=0, k=kk)], axis=1)
            o_plain = np.concatenate(
                [o_mag_plain, _head_spectra(plain_o, n_q_heads, head_dim, axis=1, k=kk)], axis=1)
            o_obs = np.concatenate(
                [o_mag_obs, _head_spectra(obs_o, n_q_heads, head_dim, axis=1, k=kk)], axis=1)
        else:
            v_plain, v_obs = v_mag_plain, v_mag_obs
            o_plain, o_obs = o_mag_plain, o_mag_obs

        # NN matches: matches[j, :] = topk plain head ids ranked for obs j.
        v_matches = _match_nn(v_plain, v_obs, topk)
        o_matches = _match_nn(o_plain, o_obs, topk)

        if head_perm_by_layer is not None and li in head_perm_by_layer:
            q_truth, kv_truth = head_perm_by_layer[li]
        else:
            q_truth = np.arange(n_q_heads, dtype=np.int64)
            kv_truth = np.arange(n_kv_heads, dtype=np.int64)

        v_top1 = int((v_matches[:, 0] == kv_truth).sum())
        v_topk_h = int((v_matches == kv_truth[:, None]).any(axis=1).sum())
        o_top1 = int((o_matches[:, 0] == q_truth).sum())
        o_topk_h = int((o_matches == q_truth[:, None]).any(axis=1).sum())

        # (V, O)-pair: for each obs Q head j, ask whether the O-side
        # predicted plain Q-head agrees with the V-side prediction modulo
        # GQA grouping. I.e. predicted_q_head // num_groups (the predicted
        # KV-group of the O side) should equal the V-side predicted KV-head
        # for that group.
        # Map obs Q-head j → obs KV-head j // num_groups.
        obs_q_to_kv = np.arange(n_q_heads, dtype=np.int64) // num_groups
        v_pred_kv = v_matches[:, 0][obs_q_to_kv]   # predicted plain KV per obs Q
        o_pred_q = o_matches[:, 0]                 # predicted plain Q per obs Q
        o_pred_kv_group = o_pred_q // num_groups   # predicted plain KV per obs Q
        pair_top1 = int((v_pred_kv == o_pred_kv_group).sum())
        # Additionally tighten the pair criterion to require that BOTH
        # predictions agree with ground truth (not just with each other).
        gt_kv_per_obs_q = kv_truth[obs_q_to_kv]
        pair_correct = (
            (v_pred_kv == gt_kv_per_obs_q)
            & (o_pred_kv_group == gt_kv_per_obs_q)
        )
        pair_top1_correct = int(pair_correct.sum())

        hits_v_top1 += v_top1
        hits_v_topk += v_topk_h
        hits_o_top1 += o_top1
        hits_o_topk += o_topk_h
        hits_pair_top1 += pair_top1_correct
        totals_v += n_kv_heads
        totals_o += n_q_heads
        totals_pair += n_q_heads

        per_layer.append({
            "layer": int(li),
            "v_top1": v_top1, "v_topk": v_topk_h, "v_n": n_kv_heads,
            "o_top1": o_top1, "o_topk": o_topk_h, "o_n": n_q_heads,
            "pair_agree_top1": pair_top1,
            "pair_correct_top1": pair_top1_correct,
        })

    def _result(name: str, top1_hits: int, topk_hits: int, total: int,
                extra_local: dict[str, Any]) -> AttackResult:
        top1 = float(top1_hits / total) if total > 0 else 0.0
        topk_rate = float(topk_hits / total) if total > 0 else 0.0
        return AttackResult(
            attack=name,
            condition="obfuscated" if head_perm_by_layer is not None else "identity_perm",
            model_id=str(obfuscated.path.name),
            n_prompts=0,
            n_train=0,
            n_test=int(total),
            ttrsr_top1=top1,
            ttrsr_top10=topk_rate,
            risk_level=classify_risk_level(top1),
            extra=extra_local,
        )

    runtime = round(time.perf_counter() - t0, 2)
    common_extra = {
        "n_q_heads": int(n_q_heads),
        "n_kv_heads": int(n_kv_heads),
        "head_dim": int(head_dim),
        "n_layers": int(len(layer_indices)),
        "topk": int(topk),
        "n_singular_values": int(singular_values) if use_spectra_aux else 0,
        "feature": "sorted_l2_magnitudes" + ("+spectra" if use_spectra_aux else ""),
        "runtime_seconds_total": runtime,
        "per_layer": per_layer,
    }
    return {
        "vo_v_match": _result("vo_v_match", hits_v_top1, hits_v_topk, totals_v, common_extra),
        "vo_o_match": _result("vo_o_match", hits_o_top1, hits_o_topk, totals_o, common_extra),
        "vo_pair_match": _result(
            "vo_pair_match", hits_pair_top1, hits_pair_top1, totals_pair, common_extra,
        ),
    }


# ───── CLI ──────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description="V/O per-channel magnitude attack — isolates Û_vo."
    )
    p.add_argument("--plain", type=Path, required=True)
    p.add_argument("--obfuscated", type=Path, required=True)
    p.add_argument("--key", type=Path,
                   help=".key.npz containing alg2 head permutations. Used for "
                        "ground-truth head ids in the obfuscated cell.")
    p.add_argument("--identity-perm", action="store_true",
                   help="Use identity head permutation (plain control cell).")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=5,
                   help="Top-k matching for top-k recovery metric.")
    p.add_argument("--singular-values", type=int, default=8,
                   help="K — auxiliary spectral fingerprint dim (alongside "
                        "per-channel magnitudes). Set 0 to disable.")
    p.add_argument("--no-spectra", action="store_true",
                   help="Disable the auxiliary spectral signature; use "
                        "sorted L2 magnitudes only.")
    p.add_argument("--head-perm-from-key", action="store_true", default=True,
                   help="Load alg2 head permutations from --key (default).")
    p.add_argument("--no-head-perm-from-key", dest="head_perm_from_key",
                   action="store_false")
    p.add_argument("--n-q-heads", type=int, default=None)
    p.add_argument("--n-kv-heads", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=None)
    from m2_7_common import add_min_mem_args, check_phase_memory  # type: ignore  # noqa: E402
    add_min_mem_args(p, phase="static_attacks")
    args = p.parse_args()

    check_phase_memory("static_attacks", args.min_mem_gb, args.skip_mem_check)

    print("[vo_channel_pair] loading plaintext GGUF…")
    t0 = time.perf_counter()
    plain = load_model(args.plain, "plaintext")
    print(f"  loaded in {time.perf_counter() - t0:.1f}s — d_eff={plain.d_eff} "
          f"n_layers={plain.n_layers}")
    print("[vo_channel_pair] loading obfuscated GGUF…")
    t0 = time.perf_counter()
    obfuscated = load_model(args.obfuscated, "obfuscated")
    print(f"  loaded in {time.perf_counter() - t0:.1f}s — d_eff={obfuscated.d_eff} "
          f"n_layers={obfuscated.n_layers}")

    key_dict: dict[str, np.ndarray] | None = None
    if args.key is not None:
        z = np.load(args.key, allow_pickle=False)
        key_dict = {name: z[name] for name in z.files}
        print(f"[vo_channel_pair] loaded key {args.key} ({len(key_dict)} entries)")

    def _resolve(cli_val: int | None, key_name: str, default: int) -> int:
        if cli_val is not None:
            return int(cli_val)
        if key_dict is not None and key_name in key_dict:
            return int(key_dict[key_name])
        return default

    n_q_heads = _resolve(args.n_q_heads, "alg2_n_q_heads", 16)
    n_kv_heads = _resolve(args.n_kv_heads, "alg2_n_kv_heads", 8)
    head_dim = _resolve(args.head_dim, "alg2_head_dim", 128)
    print(f"[vo_channel_pair] head geometry: n_q={n_q_heads} n_kv={n_kv_heads} "
          f"head_dim={head_dim}")

    head_perm_by_layer: dict[int, tuple[np.ndarray, np.ndarray]] | None
    if args.identity_perm or key_dict is None or not args.head_perm_from_key:
        head_perm_by_layer = None
        print("[vo_channel_pair] ground truth = identity permutation")
    else:
        head_perm_by_layer = {}
        for li in range(min(plain.n_layers, obfuscated.n_layers)):
            head_perm_by_layer[li] = _build_head_perm_from_key(
                key_dict, li, n_q_heads, n_kv_heads,
            )
        print(f"[vo_channel_pair] loaded ground-truth head perms for "
              f"{len(head_perm_by_layer)} layers")

    use_spectra = (not args.no_spectra) and args.singular_values > 0

    print("[vo_channel_pair] running attack…")
    results = run_vo_channel_pair(
        plain,
        obfuscated,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        head_perm_by_layer=head_perm_by_layer,
        singular_values=args.singular_values,
        topk=args.top_k,
        use_spectra_aux=use_spectra,
    )
    for name, r in results.items():
        print(f"  {name:18s} top1={r.ttrsr_top1:.4f} top{args.top_k}={r.ttrsr_top10:.4f} "
              f"risk={r.risk_level}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "aloepri_m2_7_static_v1",
        "plain_path": str(args.plain),
        "obfuscated_path": str(args.obfuscated),
        "key_path": str(args.key) if args.key else "identity_perm",
        "attacks": {name: r.to_dict() for name, r in results.items()},
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"[vo_channel_pair] wrote → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

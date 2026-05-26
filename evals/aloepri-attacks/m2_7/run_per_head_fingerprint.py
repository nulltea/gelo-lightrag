"""Per-head spectral-fingerprint attack against §05 Algorithm-2 obfuscation.

Threat model (paper §F.1): adversary holds plaintext + obfuscated GGUF and
must recover the per-layer head permutation Π_head = τ_kv ⊗ τ_group.

Attack: for each plaintext Q/K/V/O head we compute the top-K singular values
of the head's weight slice. SVD spectra are invariant under orthogonal
right-multiplication, so this signature is INVARIANT to R̂_qk (orthogonal
Q/K rotations), Ĥ_qk (Hadamard ±1 flips), and Û_vo's QR-orthogonal factor.
The only Algorithm-2 component the spectrum does NOT survive is the
*block-row* (W_q, W_k, W_v) / *block-column* (W_o) permutation Π_head: row
sets are preserved per head, but heads are attached to different indices.
NN matching on per-head spectra therefore recovers Π_head while being blind
to the other components — isolating its contribution to the joint defence.

Per-tensor head slicing (GGUF natural shape, confirmed via IA driver's
`wq.shape[0] = q_dim`):
  * W_q: (q_dim, d_eff)     — heads on axis 0, q_dim  = n_q ·head_dim
  * W_k: (kv_dim, d_eff)    — heads on axis 0, kv_dim = n_kv·head_dim
  * W_v: (kv_dim, d_eff)    — heads on axis 0
  * W_o: (d_eff, q_dim)     — heads on axis 1
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


# ───── spectral signatures ──────────────────────────────────────────


def _topk_singular_values(mat: np.ndarray, k: int) -> np.ndarray:
    """Return the top-k singular values of `mat` (descending), padded with
    zeros if rank < k. Uses `np.linalg.svd(..., compute_uv=False)` for
    accuracy; per-head dims here are small (head_dim × d_eff ≈ 128 × 2304)
    so the cost is sub-second per layer."""
    arr = mat.astype(np.float32, copy=False)
    s = np.linalg.svd(arr, compute_uv=False)
    if s.shape[0] >= k:
        return s[:k]
    out = np.zeros(k, dtype=np.float32)
    out[: s.shape[0]] = s
    return out


def _per_head_signatures(
    W: np.ndarray, n_heads: int, head_dim: int, *, axis: int, k: int
) -> np.ndarray:
    """Slice W into `n_heads` head blocks along `axis` and return a
    (n_heads, k) array of top-k singular values per head.

    For axis=0 the head block is W[h*head_dim : (h+1)*head_dim, :].
    For axis=1 the head block is W[:, h*head_dim : (h+1)*head_dim].
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    sigs = np.zeros((n_heads, k), dtype=np.float32)
    for h in range(n_heads):
        s = slice(h * head_dim, (h + 1) * head_dim)
        block = W[s, :] if axis == 0 else W[:, s]
        sigs[h] = _topk_singular_values(block, k)
    return sigs


def _match_nn(plain_sigs: np.ndarray, obs_sigs: np.ndarray, topk: int) -> np.ndarray:
    """For each obs head, return the indices of the topk-closest plain
    heads by L2 distance of the singular-value signature. Shape:
    `(n_obs_heads, topk)`."""
    pp = (plain_sigs * plain_sigs).sum(axis=1, keepdims=True)
    oo = (obs_sigs * obs_sigs).sum(axis=1, keepdims=True)
    # dist²(obs_j, plain_i) = ||obs||² - 2·obs·plain + ||plain||²
    dist2 = oo - 2.0 * obs_sigs @ plain_sigs.T + pp.T
    k = min(topk, plain_sigs.shape[0])
    return np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]


# ───── ground-truth construction from key.npz ───────────────────────


def _build_head_perm_from_key(
    key: dict[str, np.ndarray],
    layer: int,
    n_q_heads: int,
    n_kv_heads: int,
) -> dict[str, np.ndarray]:
    """Reconstruct obs→plain head permutation per `lib/alg2.py`.

    KV heads (W_k, W_v): `plain_head_for_obs_kv[j] = tau_kv[j]`.
    Q heads (W_q, W_o): GQA layout reshapes (n_q,) → (n_kv, n_groups) then
    permutes rows by tau_kv and columns by tau_group; flattened head_order
    is the obs→plain Q mapping. Returns identity if tau_* missing.
    """
    tau_kv = key.get(f"alg2_l{layer}_tau_kv")
    tau_group = key.get(f"alg2_l{layer}_tau_group")
    num_groups = n_q_heads // n_kv_heads
    if num_groups * n_kv_heads != n_q_heads:
        raise ValueError(
            f"GQA layout mismatch: n_q_heads={n_q_heads} not multiple of "
            f"n_kv_heads={n_kv_heads}"
        )
    grouped = np.arange(n_q_heads, dtype=np.int64).reshape(n_kv_heads, num_groups)
    if tau_kv is not None:
        grouped = grouped[tau_kv, :]
    if tau_group is not None:
        grouped = grouped[:, tau_group]
    q_perm = grouped.reshape(-1)
    kv_perm = (
        tau_kv.astype(np.int64)
        if tau_kv is not None
        else np.arange(n_kv_heads, dtype=np.int64)
    )
    return {"q": q_perm, "kv": kv_perm}


# ───── attack runner ────────────────────────────────────────────────


def run_per_head_fingerprint(
    plain: ModelWeights,
    obfuscated: ModelWeights,
    *,
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    head_perm_by_layer: dict[int, dict[str, np.ndarray]] | None,
    singular_values: int = 16,
    topk: int = 5,
    layer_indices: tuple[int, ...] | None = None,
) -> dict[str, AttackResult]:
    """Run the per-head spectral-fingerprint attack across all layers and
    Q/K/V/O variants. Returns a dict keyed by variant name."""
    t0 = time.perf_counter()
    n_layers = min(plain.n_layers, obfuscated.n_layers)
    if layer_indices is None:
        layer_indices = tuple(range(n_layers))

    variants = {
        "attn_q": ("q", 0, n_q_heads),
        "attn_k": ("kv", 0, n_kv_heads),
        "attn_v": ("kv", 0, n_kv_heads),
        "attn_output": ("q", 1, n_q_heads),  # W_o has heads on axis 1
    }

    # Accumulators per variant.
    totals = {v: 0 for v in variants}
    hits_top1 = {v: 0 for v in variants}
    hits_topk = {v: 0 for v in variants}
    per_layer_summary: dict[str, list[dict[str, Any]]] = {v: [] for v in variants}

    for li in layer_indices:
        for kind, (perm_key, axis, n_heads) in variants.items():
            plain_W = plain.per_layer[li].get(kind)
            obs_W = obfuscated.per_layer[li].get(kind)
            if plain_W is None or obs_W is None:
                continue
            # Both models must share the head_dim — only d_eff differs.
            plain_sigs = _per_head_signatures(
                plain_W.astype(np.float32, copy=False),
                n_heads, head_dim, axis=axis, k=singular_values,
            )
            obs_sigs = _per_head_signatures(
                obs_W.astype(np.float32, copy=False),
                n_heads, head_dim, axis=axis, k=singular_values,
            )
            matches = _match_nn(plain_sigs, obs_sigs, topk)
            # matches[j, 0] = plain-head index predicted for obs head j.
            if head_perm_by_layer is not None and li in head_perm_by_layer:
                truth = head_perm_by_layer[li][perm_key]
            else:
                truth = np.arange(n_heads, dtype=np.int64)
            top1 = (matches[:, 0] == truth).sum()
            topk_hit = (matches == truth[:, None]).any(axis=1).sum()
            hits_top1[kind] += int(top1)
            hits_topk[kind] += int(topk_hit)
            totals[kind] += int(n_heads)
            per_layer_summary[kind].append({
                "layer": int(li),
                "n_heads": int(n_heads),
                "top1_hits": int(top1),
                "topk_hits": int(topk_hit),
            })

    results: dict[str, AttackResult] = {}
    for kind in variants:
        total = totals[kind]
        top1 = float(hits_top1[kind] / total) if total > 0 else 0.0
        topk_rate = float(hits_topk[kind] / total) if total > 0 else 0.0
        results[kind] = AttackResult(
            attack=f"per_head_fingerprint_{kind}",
            condition="obfuscated" if head_perm_by_layer is not None else "identity_perm",
            model_id=str(obfuscated.path.name),
            n_prompts=0,
            n_train=0,
            n_test=total,
            ttrsr_top1=top1,
            ttrsr_top10=topk_rate,
            risk_level=classify_risk_level(top1),
            extra={
                "variant": kind,
                "n_singular_values": int(singular_values),
                "topk": int(topk),
                "n_layers": int(len(layer_indices)),
                "n_q_heads": int(n_q_heads),
                "n_kv_heads": int(n_kv_heads),
                "head_dim": int(head_dim),
                "per_layer": per_layer_summary[kind],
                "runtime_seconds_total": round(time.perf_counter() - t0, 2),
            },
        )
    return results


# ───── CLI ──────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description="Per-head spectral-fingerprint attack — isolates Π_head."
    )
    p.add_argument("--plain", type=Path, required=True, help="Plaintext Qwen3 GGUF.")
    p.add_argument("--obfuscated", type=Path, required=True, help="Algorithm-2 obfuscated GGUF.")
    p.add_argument("--key", type=Path,
                   help=".key.npz containing alg2_l<l>_tau_kv / alg2_l<l>_tau_group. "
                        "Required for the obfuscated cell. Omit with --identity-perm "
                        "for the plain-side control.")
    p.add_argument("--identity-perm", action="store_true",
                   help="Use identity head permutation (plain control cell). "
                        "Pair with --plain == --obfuscated to verify the attack "
                        "recovers identity.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--singular-values", type=int, default=16,
                   help="K — number of leading singular values per head.")
    p.add_argument("--top-k", type=int, default=5,
                   help="Top-k matching for top-k recovery metric.")
    p.add_argument("--head-perm-from-key", action="store_true", default=True,
                   help="Load alg2 head permutations from --key (default True). "
                        "Disable with --no-head-perm-from-key to force identity "
                        "ground-truth even with --key supplied.")
    p.add_argument("--no-head-perm-from-key", dest="head_perm_from_key",
                   action="store_false")
    p.add_argument("--n-q-heads", type=int, default=None,
                   help="Override n_q_heads (default: read from key.npz "
                        "alg2_n_q_heads, else 16 for Qwen3-1.7B).")
    p.add_argument("--n-kv-heads", type=int, default=None,
                   help="Override n_kv_heads (default: read from key.npz, "
                        "else 8 for Qwen3-1.7B).")
    p.add_argument("--head-dim", type=int, default=None,
                   help="Override head_dim (default: read from key.npz, "
                        "else 128 for Qwen3-1.7B).")
    from m2_7_common import add_min_mem_args, check_phase_memory  # type: ignore  # noqa: E402
    add_min_mem_args(p, phase="static_attacks")
    args = p.parse_args()

    check_phase_memory("static_attacks", args.min_mem_gb, args.skip_mem_check)

    if args.identity_perm and args.key is not None:
        # Allow --key with --identity-perm: the key is used only for head
        # geometry (n_*_heads, head_dim), not for ground truth.
        print("[per_head_fingerprint] --identity-perm with --key: using key "
              "only for head geometry, ignoring tau_kv/tau_group.")

    print("[per_head_fingerprint] loading plaintext GGUF…")
    t0 = time.perf_counter()
    plain = load_model(args.plain, "plaintext")
    print(f"  loaded in {time.perf_counter() - t0:.1f}s — "
          f"vocab={plain.vocab_size} d_eff={plain.d_eff} n_layers={plain.n_layers}")

    print("[per_head_fingerprint] loading obfuscated GGUF…")
    t0 = time.perf_counter()
    obfuscated = load_model(args.obfuscated, "obfuscated")
    print(f"  loaded in {time.perf_counter() - t0:.1f}s — "
          f"vocab={obfuscated.vocab_size} d_eff={obfuscated.d_eff} "
          f"n_layers={obfuscated.n_layers}")

    key_dict: dict[str, np.ndarray] | None = None
    if args.key is not None:
        z = np.load(args.key, allow_pickle=False)
        key_dict = {name: z[name] for name in z.files}
        print(f"[per_head_fingerprint] loaded key {args.key} "
              f"(keys: {len(key_dict)}, alg2={'alg2_applied' in key_dict})")

    # Resolve head geometry: CLI override > key.npz > Qwen3-1.7B default.
    def _resolve(cli_val: int | None, key_name: str, default: int) -> int:
        if cli_val is not None:
            return int(cli_val)
        if key_dict is not None and key_name in key_dict:
            return int(key_dict[key_name])
        return default

    n_q_heads = _resolve(args.n_q_heads, "alg2_n_q_heads", 16)
    n_kv_heads = _resolve(args.n_kv_heads, "alg2_n_kv_heads", 8)
    head_dim = _resolve(args.head_dim, "alg2_head_dim", 128)
    print(f"[per_head_fingerprint] head geometry: n_q={n_q_heads} "
          f"n_kv={n_kv_heads} head_dim={head_dim}")

    head_perm_by_layer: dict[int, dict[str, np.ndarray]] | None = None
    if args.identity_perm or key_dict is None or not args.head_perm_from_key:
        head_perm_by_layer = None
        print("[per_head_fingerprint] ground truth = identity permutation")
    else:
        head_perm_by_layer = {}
        for li in range(min(plain.n_layers, obfuscated.n_layers)):
            head_perm_by_layer[li] = _build_head_perm_from_key(
                key_dict, li, n_q_heads, n_kv_heads,
            )
        print(f"[per_head_fingerprint] loaded ground-truth head perms for "
              f"{len(head_perm_by_layer)} layers")

    print("[per_head_fingerprint] running attack…")
    results = run_per_head_fingerprint(
        plain,
        obfuscated,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        head_perm_by_layer=head_perm_by_layer,
        singular_values=args.singular_values,
        topk=args.top_k,
    )
    for kind, r in results.items():
        print(f"  {kind:12s} top1={r.ttrsr_top1:.4f} top{args.top_k}={r.ttrsr_top10:.4f} "
              f"risk={r.risk_level}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "aloepri_m2_7_static_v1",
        "plain_path": str(args.plain),
        "obfuscated_path": str(args.obfuscated),
        "key_path": str(args.key) if args.key else "identity_perm",
        "attacks": {
            f"per_head_fingerprint_{kind}": r.to_dict()
            for kind, r in results.items()
        },
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"[per_head_fingerprint] wrote → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

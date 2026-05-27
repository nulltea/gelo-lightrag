#!/usr/bin/env python3
"""GPU-native layer × cell × per-head ISA-AttnScore sweep.

Self-contained — does NOT call into the CPU-only run_isa.py driver.
Pushes all tensors to cuda, ridge solve via torch.linalg.solve →
rocSOLVER on Strix Halo gfx1151. Reports per-config TTRSR top-1/top-10.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path("/home/timo/repos/private-rag-path-2")
sys.path.insert(0, str(REPO / "evals" / "aloepri-attacks"))
from snapshots_loader import SnapshotSet  # type: ignore
from attack_drivers.run_ima import load_qwen3_embedding_table  # type: ignore


def fit_ridge_gpu(X: torch.Tensor, Y: torch.Tensor, alpha: float) -> dict:
    """torch.linalg.solve closed-form ridge on the input device."""
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
    lhs = Xa.T @ Xa + alpha * I
    rhs = Xa.T @ Yn
    try:
        W = torch.linalg.solve(lhs, rhs)
    except RuntimeError as e:
        if "HIPBLAS_STATUS_ALLOC_FAILED" not in str(e) and "CUDA" not in str(e):
            raise
        try:
            W = torch.linalg.solve(lhs.cpu(), rhs.cpu()).to(device)
        except Exception:
            return None
    except Exception:
        return None
    return {"W": W, "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


def predict_ridge(m, X):
    Xn = (X - m["x_mean"]) / m["x_std"]
    ones = torch.ones((Xn.shape[0], 1), dtype=Xn.dtype, device=X.device)
    Xa = torch.cat([Xn, ones], dim=1)
    return Xa @ m["W"] * m["y_std"] + m["y_mean"]


def cosine_topk(pred, embed_table, true_ids, topk=10, chunk=4096):
    pn = pred / pred.norm(dim=1, keepdim=True).clamp_min(1e-8)
    best_s = torch.full((pn.shape[0], 0), float("-inf"), dtype=pn.dtype, device=pn.device)
    best_i = torch.empty((pn.shape[0], 0), dtype=torch.long, device=pn.device)
    V = embed_table.shape[0]
    k_eff = min(topk, V)
    for s in range(0, V, chunk):
        e = min(s + chunk, V)
        c = embed_table[s:e]
        c = c / c.norm(dim=1, keepdim=True).clamp_min(1e-8)
        sc = pn @ c.T
        cs, ci_local = torch.topk(sc, k=min(k_eff, sc.shape[1]), dim=1)
        ci = ci_local + s
        ms = torch.cat([best_s, cs], dim=1)
        mi = torch.cat([best_i, ci], dim=1)
        ns, nx = torch.topk(ms, k=k_eff, dim=1)
        best_s = ns
        best_i = mi.gather(1, nx)
    hits = best_i.eq(true_ids.unsqueeze(1))
    return hits[:, 0], hits[:, :k_eff].any(dim=1)


def load_flat(captures_dir: Path, layer: int, kind: str, head_idx: int | None = None,
              capture_basename: str = "attn"):
    """Return (X, y) torch CPU tensors.

    For 3-D `kq`-style operands `(n_heads, n_q, n_kv)`:
      head_idx=None  -> flatten heads onto the feature axis: (n_q, n_heads*n_kv)
      head_idx=int   -> per-head slice: (n_q, n_kv)

    For 2-D `kqv_out`-style operands `(n_q, n_heads*head_dim)` (the per-head
    attention output `softmax(Q·Kᵀ/√d)·V` flattened across heads BEFORE W_o):
      head_idx=None  -> use the 2-D tensor directly
      head_idx=int   -> slice the head's `head_dim` columns
    """
    attn = SnapshotSet.open(capture_basename, root=captures_dir)
    Xs, ys = [], []
    for s in attn.select(kind=kind, layer=layer):
        op = attn.get_operand(s, strip_shield=False)
        ids = attn.prompt_token_ids[s.prompt_idx]
        if op.ndim == 3:
            n_valid = min(op.shape[1], len(ids))
            if head_idx is None:
                feat = op[:, :n_valid, :].permute(1, 0, 2).reshape(n_valid, -1)
            else:
                feat = op[head_idx, :n_valid, :].contiguous()
        elif op.ndim == 2:
            n_valid = min(op.shape[0], len(ids))
            if head_idx is None:
                feat = op[:n_valid, :].contiguous()
            else:
                # kqv_out: columns are (n_heads * head_dim). Need head_dim
                # from the operand: total / n_heads.
                total = op.shape[1]
                # head_dim known to caller via per-head probe; we infer
                # n_heads from a meta hint by leaving it to the caller.
                # For now, allow head_idx to address head_dim-sized slices
                # only if the caller passes a tuple (head_idx, head_dim).
                if isinstance(head_idx, tuple):
                    h, hd = head_idx
                    feat = op[:n_valid, h * hd:(h + 1) * hd].contiguous()
                else:
                    raise ValueError(
                        "kqv_out per-head load requires head_idx=(h, head_dim); "
                        f"got {head_idx!r}")
        else:
            continue
        Xs.append(feat)
        ys.append(torch.tensor(ids[:n_valid], dtype=torch.long))
    if not Xs:
        return None, None
    return torch.cat(Xs, dim=0).to(torch.float32), torch.cat(ys, dim=0)


def run_one(X_cpu, y_cpu, embed_table_gpu, *, seed: int, split: str = "row",
            alpha_grid=(1e-4, 1e-2, 1.0), train_frac=0.5, val_frac=0.25):
    """One config: load, split, multi-α GPU ridge, return (top1, top10, alpha*)."""
    dev = embed_table_gpu.device
    X = X_cpu.to(dev)
    y = y_cpu.to(dev)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    if split == "row":
        perm = rng.permutation(n)
        n_tr = int(n * train_frac); n_va = int(n * val_frac)
        tr = torch.tensor(perm[:n_tr], dtype=torch.long, device=dev)
        va = torch.tensor(perm[n_tr:n_tr + n_va], dtype=torch.long, device=dev)
        te = torch.tensor(perm[n_tr + n_va:], dtype=torch.long, device=dev)
    else:
        uniq = torch.unique(y).cpu().tolist()
        sh = rng.permutation(uniq).tolist()
        n_tr_ids = int(len(sh) * train_frac); n_va_ids = int(len(sh) * val_frac)
        tr_s = set(sh[:n_tr_ids]); va_s = set(sh[n_tr_ids:n_tr_ids + n_va_ids])
        te_s = set(sh[n_tr_ids + n_va_ids:])
        ycpu = y.cpu().tolist()
        tr = torch.tensor([i for i, v in enumerate(ycpu) if v in tr_s], dtype=torch.long, device=dev)
        va = torch.tensor([i for i, v in enumerate(ycpu) if v in va_s], dtype=torch.long, device=dev)
        te = torch.tensor([i for i, v in enumerate(ycpu) if v in te_s], dtype=torch.long, device=dev)

    if len(tr) == 0 or len(va) == 0 or len(te) == 0:
        return None, None, None

    X_tr, X_va, X_te = X[tr], X[va], X[te]
    y_tr = embed_table_gpu[y[tr]]
    y_va_ids, y_te_ids = y[va], y[te]

    best_a, best_v1, best_m = None, -1.0, None
    for a in alpha_grid:
        m = fit_ridge_gpu(X_tr, y_tr, float(a))
        if m is None:
            continue
        v_pred = predict_ridge(m, X_va)
        v_top1, _ = cosine_topk(v_pred, embed_table_gpu, y_va_ids)
        v1 = float(v_top1.float().mean())
        if v1 > best_v1:
            best_v1, best_a, best_m = v1, float(a), m
    if best_m is None:
        return None, None, None
    t_pred = predict_ridge(best_m, X_te)
    t_top1, t_top10 = cosine_topk(t_pred, embed_table_gpu, y_te_ids)
    return float(t_top1.float().mean()), float(t_top10.float().mean()), best_a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plain", type=Path, required=True)
    p.add_argument("--obf", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3-4B")
    p.add_argument("--layers", type=int, nargs="+", default=[0, 5, 11, 17, 23])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--split", default="row")
    p.add_argument("--per-head-layer", type=int, default=17,
                   help="Layer to run per-head ridge probe on (after layer sweep)")
    p.add_argument("--kind", default="kq",
                   help="Tensor kind to load (e.g. 'kq' for pre-softmax Q·Kᵀ "
                        "or 'kqv_out' for the per-head attention output "
                        "softmax(Q·Kᵀ/√d)·V before W_o).")
    p.add_argument("--head-dim", type=int, default=128,
                   help="Per-head dim used for kqv_out per-head slicing.")
    p.add_argument("--skip-per-head", action="store_true",
                   help="Skip the per-head probe section.")
    args = p.parse_args()

    assert torch.cuda.is_available(), "GPU required"
    dev = "cuda"
    print(f"[gpu_sweep] device={dev} name={torch.cuda.get_device_name(0)}")
    embed = load_qwen3_embedding_table(args.model_id).to(torch.float32).to(dev)
    print(f"[gpu_sweep] embed shape={tuple(embed.shape)} mem={torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ===== Layer sweep =====
    print(f"\n===== LAYER SWEEP plain vs obf (kind={args.kind}, split={args.split}, {args.seeds} seeds) =====")
    print(f"{'layer':>5} | {'cell':>5} | {'top1 mean ± std':>20} | {'top10':>10} | {'wall':>6}")
    layer_results = {}
    for L in args.layers:
        for label, cap in [("PLAIN", args.plain), ("OBF", args.obf)]:
            X, y = load_flat(cap, L, args.kind, head_idx=None)
            if X is None:
                print(f"  L={L} {label}: no captures"); continue
            t0 = time.perf_counter()
            t1s, t10s, alphas = [], [], []
            for k in range(args.seeds):
                r = run_one(X, y, embed, seed=20260518 + k, split=args.split)
                if r[0] is None: continue
                t1s.append(r[0]); t10s.append(r[1]); alphas.append(r[2])
            a1 = np.array(t1s) * 100; a10 = np.array(t10s) * 100
            print(f"  L={L:>3} | {label:>5} | {a1.mean():>6.2f}% ± {a1.std():>5.2f} (n={len(t1s)}, X={tuple(X.shape)}) | "
                  f"{a10.mean():>5.2f}% | {time.perf_counter()-t0:>5.1f}s  alphas={alphas[:3]}...")
            layer_results[(L, label)] = (a1.mean(), a1.std(), a10.mean())

    if args.skip_per_head:
        return

    # ===== Per-head probe @ specified layer =====
    PH = args.per_head_layer
    print(f"\n===== PER-HEAD RIDGE @ L={PH} kind={args.kind} (3 seeds, alpha=1e-2) =====")
    for label, cap in [("PLAIN", args.plain), ("OBF", args.obf)]:
        # First infer n_heads from sample shape (3-D kq) or from kqv_out width.
        attn = SnapshotSet.open("attn", root=cap)
        s0 = next(iter(attn.select(kind=args.kind, layer=PH)))
        sample = attn.get_operand(s0, strip_shield=False)
        if sample.ndim == 3:
            n_heads = sample.shape[0]
            print(f"  L={PH} {label}: n_heads={n_heads} per-head shape=(?, {sample.shape[2]})")
        elif sample.ndim == 2:
            n_heads = sample.shape[1] // args.head_dim
            print(f"  L={PH} {label}: n_heads={n_heads} per-head shape=(?, {args.head_dim})")
        else:
            print(f"  L={PH} {label}: unsupported ndim={sample.ndim}"); continue
        head_t1 = np.zeros((n_heads, 3))
        t0 = time.perf_counter()
        for h in range(n_heads):
            head_arg = h if sample.ndim == 3 else (h, args.head_dim)
            X, y = load_flat(cap, PH, args.kind, head_idx=head_arg)
            for k in range(3):
                r = run_one(X, y, embed, seed=20260518 + k, split=args.split, alpha_grid=(1e-2,))
                head_t1[h, k] = r[0] or 0.0
        means = head_t1.mean(axis=1) * 100
        print(f"    {label} per-head top-1: min={means.min():.2f}  median={np.median(means):.2f}  "
              f"mean={means.mean():.2f}  p90={np.percentile(means, 90):.2f}  max={means.max():.2f}  "
              f"best head idx={int(np.argmax(means))}  (wall {time.perf_counter()-t0:.1f}s)")


if __name__ == "__main__":
    main()

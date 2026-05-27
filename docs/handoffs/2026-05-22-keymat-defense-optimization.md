# Handoff — AloePri keymat defense optimization plan

**Date:** 2026-05-22
**Branch:** `path-2-aloepri-gemma`
**Companion docs:**
- `docs/research/aloepri-keymat-variance.md` — full session findings (variance decomposition, K_a × K_d universality, component decomposition, defender-lever table)
- Previous handoff: `2026-05-21-uvo-isa-multikey-and-gpu-keymat-bug.md` — closed (Û_vo Algorithm 2 patch + ISA multi-key driver + retracted GPU-keymat-bug diagnosis)

## Headline

The 2×2 PRNG×LinAlg "GPU-keymat-bug" diagnosis was retracted as K=64 single-seed noise. Subsequent variance-decomposition + K_a × K_d universality probes shifted the picture:

- **σ_pool dominates σ_split by 13×** at K=64, d=2560. Single-seed §08 cells comparing within 5 pp are noise.
- **Pure Algorithm 1 has a K_a × K_d interaction bounded at ~5 pp.** Pool ranks rotate with K_d.
- **Algorithm 2 amplifies the interaction ~3–5×**, producing the real-deployment 13 pp spread (pool-2 = 13.8 % vs pool-1 = 0.7 % at K_d=42).
- **Synthetic-mode K_d ranking does NOT predict real-mode K_d ranking** — pool-2 was *worst* in synthetic K_d=42 (3.1 %) and *best* in real K_d=42 (13.8 %). Algorithm 2 rotates the ranking, not just scales it.

## Forward plan — optimize Algorithm 1 defense before moving to Algorithm 2

Three phases in dependency order. **c1 first**, then a2 (reconsider after c1), then b1 (shaped by c1 + a2), then Algorithm 2 ablation.

### Phase c1 — Luckiness-signature probe (to be launched)

Script: `evals/aloepri-attacks/m2_7/probe_luckiness_signature.py`

**Goal:** find the scalar feature `f(K_a^k)` or `f(K_a^k, K_d)` that predicts synthetic-mode TTRSR. Result classifies the lucky-pool mechanism into one of three categories and routes the rest of the plan.

| Winning feature category | What it means | Next phase target |
|---|---|---|
| Intrinsic (Category 1) | "Lucky" K_a^k samples are universal — function of K_a^k alone | Phase b modifies Algorithm 1's noise distribution |
| Alignment (Category 2) | K_d × K_a interaction — function of pair | a1 (adversarial K_d selection) becomes principled with this as scoring function (reduces from ~6 h to ~1 h since scoring becomes cheap) |
| Component (Category 3) | Specific Algorithm-1 block (U / V / E / F / Z / C) carries the variance | Phase b targets that component (likely b1 = anti-correlated C-coeffs if C wins) |

#### Run decision (2026-05-22)

- **N = 10 pool seeds** (1..10) at K=64, fixed K_d_seed=42.
- **Option 1 — parallel synthetic-TTRSR sweep**: run `run_isa_multikey.py` at `--kd-test-seed 42 --attacker-seed N` for N ∈ {6..10} to fill in TTRSR for the 5 new pools (1..5 already covered by the universality probe). Existing data: seeds 1..3 from the universality probe (4.0, 3.1, 0.9 %); seeds 4..5 from the disentangle sweep need to be re-run in synthetic mode since they were on real K_d.

Rationale for downscale to N=10: |r| ≥ 0.6 → p < 0.07 — borderline-significant correlation. Acceptable for first iteration. Expand to N=20 only if N=10 gives ambiguous results (|r| in 0.4–0.6 range across multiple competing features).

Total wall time: c1 features ~30 min + parallel synthetic-TTRSR sweep ~15 min, both fit in ~30 min wall.

#### Parameter rationale

| Param | Value | Rationale | Downscale option |
|---|---|---|---|
| `D` | 2560 | Q3-4B hidden size, fixed by deployment | Not scalable |
| `H` | 128 | Algorithm 1 expansion, paper default + matches deployment | Could parameterise to test h ∈ {64, 256} but doubles cost per added value |
| `LAM` | 0.3 | Algorithm 1 V-noise weight, paper default | Could parameterise; orthogonal to luckiness test |
| `K` | 64 | Matches deployment + existing data | K=32 halves per-pool work but loses K=64 bimodal structure. Floor: K=32 |
| `KD_SEED` | 42 | Matches deployment; isolates "what predicts pool quality at *this* K_d" | Multi-K_d would mix two effects; orthogonal question |
| `POOL_SEEDS` | 1..10 | N=10 → \|r\|≥0.6 → p<0.07, OK for first iteration; 5 seeds reusable from prior probes | Expand to 1..20 if borderline |

#### Features per category

**Category 1 — Intrinsic** (per K_a^k, no K_d reference). Detects universal-lucky-sample mode.

| Feature | Computation | Significance if it correlates with TTRSR |
|---|---|---|
| `frobenius_norm` | `‖K_a^k‖_F` | Matrix magnitude alone predicts luckiness → attacker pre-filters offline |
| `sigma_1` | top singular value | Sharp dominant direction → universal lucky direction; Algorithm 1 leaks structure |
| `sigma_min` | smallest non-zero SV | Ill-conditioned K_a^k samples are lucky → Algorithm 1 should reject these draws |
| `condition_number` | `σ_1 / σ_min` | Same as above; ratio form |
| `spectral_concentration_top1` | `σ_1 / Σσ` | How much spectral mass in top-1 direction |
| `spectral_kurtosis` | `Σσ⁴ / (Σσ²)²` | Heavy-tailed spectrum (peaked = close to 1, flat = close to 1/d). Suggests modifying Algorithm 1 noise to enforce flat spectrum |

**Category 2 — Alignment** (function of `(K_a^k, K_d)`). Tests interaction-driven luck.

| Feature | Computation | Significance |
|---|---|---|
| `frobenius_alignment` | `‖K_a^k · pinv(K_d)‖_F` | How much of K_d's inverse direction K_a^k covers. Win → defender picks K_d minimising expected alignment |
| `top_sv_overlap_r128` | Σ squared SVs of `Vh_a[:h].T · Vh_d[:h]` cross-matrix | Top-h "important" subspace overlap. Targets h-dimensional dominant subspace; direct h-tuning lever |
| `principal_angle_mean` | mean principal angle between row spaces of K_a^k and K_d | Coarse subspace overlap |

If Category 2 wins: K_d × K_a interaction is the lever → adversarial K_d selection works with this metric as scoring function (the original Phase a1, now justified).

**Category 3 — Per-component** (decomposes K_a^k back into U, V, E, F, Z, C via seed regeneration).

| Feature | Computation | What it tests |
|---|---|---|
| `U_diag_overlap_with_kd` | `tr(U_a · U_d^T) / d` | U rotation overlap (sanity — Haar should NOT correlate) |
| `V_norm` | `‖V‖_F` | V noise magnitude (sanity — iid Gaussian, should be near-constant) |
| `E_norm` | `‖E‖_F` | E block magnitude — tests E carries variance |
| `F_top_sv` | `σ_1(F)` | F's leading singular value; F should be rank-h/2 |
| `Z_diag_overlap_with_kd` | `tr(Z_a · Z_d^T) / (d+2h)` | Z rotation overlap (sanity — Haar should NOT correlate) |
| **`C_nullspace_angle_with_kd`** | **principal angle between nullspace(F_a^T) and nullspace(F_d^T)** | **Direct test of C-block hypothesis** — small angle = K_a^k's C lives in same subspace as K_d's C → multi-key ridge generalises. If wins: targets Phase b1 (anti-correlated C-coeffs) and Phase a2 (larger h enlarges nullspace, reducing angle-by-chance) |

The `C_nullspace_angle_with_kd` feature is **load-bearing** for the doc's C-block hypothesis — that's the central claim of the "Component-level decomposition of Algorithm 1" section in `aloepri-keymat-variance.md` and the empirical test of it.

#### Optimizations in place

| Optimization | Effect |
|---|---|
| **Batched SVD on (K, d, d+2h) stack** | 64 SVDs in parallel on GPU. ~10× vs serial (which the killed probe did) |
| **`full_matrices=False` in batched SVD** | Computes top-`d` singular vectors only, not the full (d+2h)² basis. ~2× speedup + 5× memory savings |
| **Sequential per-pool with `torch.cuda.empty_cache()`** | Stack peak 3.7 GB on GPU per pool instead of 37 GB for all 10 pools simultaneously. Fits the iGPU comfortably |
| **Incremental JSON write per pool** | Crash recovery + intermediate inspection. Read partial results from `/tmp/aloepri-gpu-validation/c1_luckiness_features.json` mid-run |
| **`flush=True` on every print + `python -u` recommended** | Avoids the "silent for 50 min" bug from the killed probe |
| **Component analysis on CPU via numpy** | Bases are small ((d,d) max); CPU is faster for these since GPU transfer dominates |
| **Reuse `Vh_d_full` across pools** | K_d's SVD computed once; reused for all 10 alignment correlations |

Not optimized (skipped for cost/benefit): multi-threaded pool building (vendor's MT19937 single-thread is paper-faithful), truncated/randomized SVD (accuracy concern), numba/Cython for per-k loops (not hot enough).

#### Output

- `/tmp/aloepri-gpu-validation/c1_luckiness_features.json` — per-pool aggregates (mean, max, std, top5_mean) + per-K_a^k arrays for each of ~14 features.
- Synthetic-TTRSR data for pool seeds 1..10 from the parallel sweep.
- Follow-up analysis step (post-run): Pearson r per (feature, pool-aggregate stat) vs the 10-pool TTRSR vector. Highest |r| in each category determines the winning mechanism.

### Phase a2 — h sweep (run after c1, reconsider based on findings)

If c1 confirms C-block (Category 2 alignment or Category 3 component-C):
- a2 becomes confirmation. Narrow to h ∈ {128, 256} at the optimized K_d_seed (= one from c1's pool ranking). ~1 h.

If c1 refutes C-block:
- a2 is likely a negative result; skip or run h ∈ {128, 256} as a sanity check then pivot to whichever component c1 fingered.

a2's role updated: **not a discovery experiment, a hypothesis confirmation**. Specific values of h to try depend on c1's mechanism finding.

### Phase b1 — Anti-correlated C-coeffs (shaped by c1 + a2)

Code mod in `vendor/aloepri-py/src/keymat.py:sample_null_columns`: replace iid Gaussian `coeffs` with QR-orthonormalised draws.

Trigger condition: only worth implementing if c1 confirms C-block as the lucky-pool site **AND** a2 shows h sweep yields meaningful gains (i.e., the C-block hypothesis is real and the lever has magnitude).

Implementation steps once triggered:
1. Add `coeff_sampling="orthonormal"` flag to `sample_null_columns`.
2. Verify covariance preservation analytically (orthonormal Gaussian draws are still rotation-equivariant — should hold).
3. Re-obfuscate Q3-4B with the modified Algorithm 1; recapture hidden states.
4. Re-run universality probe to compare synthetic spread vs original.
5. Run multi-pool best-of-5 attacker on the new deployment; compare to original baseline.

Expected gain: ~1-3 pp on top of a2's gain in synthetic mode; real-deployment gain bounded by Algorithm 2 contribution.

### Phase 4b (now Phase d) — Algorithm 2 ablation

> **Naming correction (2026-05-25).** The three ablations below lump
> §5.2.2 (Π + W_e/W_h noise) and §5.2.3 (Algorithm 2 attention
> obfuscation) under one "Algorithm 2 ablation" header. Paper-strict:
> Π and noise are §5.2.2 (paper item 6); only Û_vo (and R̂_qk, Ĥ_qk,
> Ẑ_block, Π_head) is §5.2.3 (paper item 7). The Ẑ_block component —
> theoretically the dominant Algorithm 2 contributor at deployed β=8 —
> is missing from this list and should be added (`--alg2-beta 1` to
> disable). Full per-component picture:
> `docs/handoffs/2026-05-25-alg2-attack-crossmap.md`.

After Algorithm 1 is exhausted, ablate Algorithm 2 components to identify the amplifier:

| Ablation | What it tests | §5.2 paper item | Cost |
|---|---|---|---|
| Disable Û_vo (`--alg2` without `--alg2-u-vo`) | Whether V↔O random projection is the amplifier | §5.2.3 (Alg2) | ~30 min obf + capture + sweep |
| Disable Ẑ_block (`--alg2-beta 1`) | Whether the RoPE-pair shuffle is the amplifier (theoretical prediction: yes, dominant) | §5.2.3 (Alg2) | ~30 min |
| Disable Π_head (run on a model with `n_kv_heads = 1` or patch `tau_kv = identity`) | Whether the head-permutation amplifies | §5.2.3 (Alg2) | ~30 min |
| Disable π token-permutation (`--no-pi`) | Whether τ-permutation interaction with K_d amplifies | **§5.2.2** (NOT Alg2) | ~30 min |
| Disable additive Gaussian noise (`--noise-alpha-e 0 --noise-alpha-h 0`) | Whether the noise on W_e/W_h amplifies | **§5.2.2** (NOT Alg2) | ~30 min |

Each ablation re-runs the universality probe (3 K_d_test × 3 pool seeds, ~30 min attack). Compare spread vs full Algorithm 2 (13 pp) and Algorithm 1 only (3-7 pp). Smallest-spread ablation identifies the amplifier.

Trigger condition: after Phases c1, a2, b1 are exhausted.

## Open items not in the current plan

- **C — Kerckhoffs-adaptive minimax threat model** — deferred per earlier session decision (B = best-of-N attacker chosen). C requires an inner-loop minimax solver; significant infrastructure. Revisit only if defender-lever gains from b1/d motivate the harder threat model.
- **Q3-8B replication** — every finding here is at Q3-4B Û_vo, L=17, d=2560. 8B (d=4096) may show different pool-variance magnitude. Replicate the universality probe at 8B before claiming the C-block hypothesis generalises.
- **Plain identity-τ ceiling at 5+ seeds** — current 10.18 % ceiling is single-seed. Re-measure with the same multi-pool-seed methodology to give the ceiling a real error bar. Critical before any "above ceiling = structurally impossible" claim.
- **Multi-K_d rotation** — server-side defense that rotates K_d per session. Out of single-deployment optimization scope but worth considering as a final layer once Algorithm 1 + 2 are tuned.

## Code state at handoff

Working tree changes since last commit (`1019018`):

```
M evals/aloepri-attacks/m2_7/run_isa_multikey.py
    + --split-seed flag (decouples pool seed from train/val/test split seed)
    + --kd-test-seed flag (synthetic universality probe — synthesises test
      inputs from K_d_test @ plain captures instead of real obf captures)
A evals/aloepri-attacks/m2_7/probe_luckiness_signature.py
    c1 script; designed but NOT yet executed at handoff time
A evals/aloepri-attacks/m2_7/probe_pool_alignment.py
    Original Phase 1.1 probe; killed mid-run when universality results showed
    K_d-only alignment was the wrong question. Kept as scaffold; functionally
    superseded by probe_luckiness_signature.py
M docs/research/aloepri-keymat-variance.md
    + Key findings section + Implications for the defender plan section
    + Two-seeds-two-draws conceptual definitions
    + Confirmation sweep table + ANOVA decomposition
    + Bimodal pool distribution finding
    + K_d universality probe section
    + Component-level decomposition of Algorithm 1
    + Defender-lever ranking
A docs/handoffs/2026-05-22-keymat-defense-optimization.md  (this file)
```

Memory entries:
- `aloepri_attack_keymat_cuda_philox_bias` rewritten as a retraction of the original Philox/rocSOLVER bug diagnosis. Records the K=64 std ≈ 5 pp finding.

## Next-session-first action — c1 launch

Two concurrent jobs. **Both must use the `aloepri-ima-trainer:latest` Docker image with ROCm passthrough.** Total wall time ~30 min.

### Data inventory before launch

Existing synthetic-mode TTRSR at `kd_test_seed=42`, K=64, split=101 (from universality probe):

| pool seed | TTRSR top-1 | source JSON |
|---|---|---|
| 1 | 4.0 % | `/tmp/aloepri-gpu-validation/universality-kd42-p1.json` |
| 2 | 3.1 % | `/tmp/aloepri-gpu-validation/universality-kd42-p2.json` |
| 3 | 0.9 % | `/tmp/aloepri-gpu-validation/universality-kd42-p3.json` |

**Need to fill in**: pool seeds 4..10 (7 new runs at ~90 s each = ~11 min wall). Cells will land in `/tmp/aloepri-gpu-validation/c1-synthttrsr-p{N}.json`.

### Job 1 — c1 feature probe (foreground, ~25 min)

```bash
REPO=/home/timo/repos/private-rag-path-2
HF=$HOME/.cache/huggingface
RENDER_GID=$(getent group render | cut -d: -f3)
VIDEO_GID=$(getent group video | cut -d: -f3)

docker run --rm \
    --name aloepri-c1-probe \
    --device /dev/dri --device /dev/kfd \
    --group-add "$VIDEO_GID" --group-add "$RENDER_GID" \
    --user "$(id -u):$(id -g)" \
    --shm-size 16G \
    -v "$REPO:$REPO" \
    -v "$HF:$HF" \
    -v "/tmp:/tmp" \
    -e HOME="$HOME" \
    -w "$REPO" \
    aloepri-ima-trainer:latest \
    python3 -u evals/aloepri-attacks/m2_7/probe_luckiness_signature.py \
    > /tmp/aloepri-gpu-validation/c1_luckiness_features.log 2>&1 &
```

Watch progress: `tail -F /tmp/aloepri-gpu-validation/c1_luckiness_features.log`. Per-pool aggregates print every ~2 min; partial JSON in `/tmp/aloepri-gpu-validation/c1_luckiness_features.json` after each pool.

### Job 2 — parallel synthetic-TTRSR sweep (background, ~11 min)

Run after Job 1 is launched but in a separate shell (different Docker container; both fit on the iGPU). Fills in synthetic TTRSR for pool seeds 4..10:

```bash
for POOL in 4 5 6 7 8 9 10; do
    OUT=/tmp/aloepri-gpu-validation/c1-synthttrsr-p${POOL}.json
    [[ -f "$OUT" ]] && continue
    IMA_DRIVER=run_isa_multikey.py bash evals/aloepri-attacks/m2_7/run_in_gpu_container.sh \
        --plain-captures evals/aloepri-attacks/results/sweep/cell-qwen3-4b-plain-rerun-20260520/captures \
        --plain-model-id Qwen/Qwen3-4B \
        --layer 17 --attacker-num-keys 64 --split-mode row \
        --keymat-impl gpu_native --device gpu \
        --attacker-seed "$POOL" --split-seed 101 \
        --kd-test-seed 42 \
        --output "$OUT" 2>&1 | tee /tmp/aloepri-gpu-validation/c1-synthttrsr-p${POOL}.log
    echo "pool=$POOL top1=$(jq -r '.attack.ttrsr_top1' $OUT)"
done
```

### Post-run analysis (~1 min)

Once both jobs are done, compute Pearson correlations:

```bash
python3 << 'EOF'
import json, numpy as np
from pathlib import Path
D = Path("/tmp/aloepri-gpu-validation")
# Load TTRSR per pool
ttrsr = {1: 0.040, 2: 0.031, 3: 0.009}  # from universality
for p in range(4, 11):
    j = json.loads((D / f"c1-synthttrsr-p{p}.json").read_text())
    ttrsr[p] = j["attack"]["ttrsr_top1"]
# Load features
feats = json.loads((D / "c1_luckiness_features.json").read_text())
pool_seeds = list(range(1, 11))
ttrsr_vec = np.array([ttrsr[p] for p in pool_seeds])
print("Pearson r vs synthetic TTRSR top-1:")
print(f"{'feature.stat':<40} {'r':>8} {'|r|':>8} {'category':>10}")
ranked = []
for p in pool_seeds:
    pf = feats["pool_features"][str(p)]["aggregates"]
    feat_names = list(pf.keys())
    break
for name in feat_names:
    for stat in ("mean", "max", "top5_mean", "median"):
        vals = np.array([feats["pool_features"][str(p)]["aggregates"][name][stat] for p in pool_seeds])
        if vals.std() < 1e-12: continue
        r = float(np.corrcoef(vals, ttrsr_vec)[0, 1])
        cat = ("Cat1" if name in {"frobenius_norm","sigma_1","sigma_min","condition_number","spectral_concentration_top1","spectral_kurtosis"}
               else "Cat2" if name in {"frobenius_alignment","top_sv_overlap_r128","principal_angle_mean"}
               else "Cat3")
        ranked.append((abs(r), r, f"{name}.{stat}", cat))
ranked.sort(reverse=True)
for absr, r, label, cat in ranked[:20]:
    print(f"{label:<40} {r:+8.3f} {absr:8.3f} {cat:>10}")
EOF
```

### Pass/fail interpretation

- **Any feature with \|r\| ≥ 0.7**: clear winner. Route the plan according to the feature's category (see § Phase c1 table).
- **\|r\| in 0.4–0.6 across multiple competing features**: ambiguous. Expand to N=20 by extending the script's `POOL_SEEDS` and running the sweep with seeds 11..20.
- **All \|r\| < 0.3**: no clean predictor. The mechanism is more complex than scalar features can capture; consider non-linear (XGBoost on aggregates, or per-K_a^k-feature regression).

### Stop signal

c1 + post-run analysis answers "which feature category predicts luckiness." That single answer routes the next phase (a2 / b1 / b2 / b3). Don't queue the next phase until c1 lands; the routing depends on it.

## Suggested skills for next session

- `/diagnose` if probe outputs are unexpected (e.g., all features show |r| < 0.2 — no clean predictor).
- `/grill-with-docs` if updating `aloepri-keymat-variance.md` with the mechanism finding — keep terminology aligned with paper §5.2 and §F.1.

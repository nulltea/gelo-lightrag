---
type: handoff
status: current
created: 2026-05-21
updated: 2026-05-21
tags: [aloepri, alg2, gpu]
---

# Handoff — Û_vo Algorithm 2 patch + ISA multi-key + GPU keymat seed-convention bug

**Date:** 2026-05-21 PM
**Branch:** `path-2-aloepri-gemma`
**Companion handoffs:**
- `2026-05-21-ima-transformer-paper-disparity.md` — earlier-today handoff covering the τ-leak diagnosis + IMA multi-key fix.
- `2026-05-21-strong-pi-server-patch.md` — strong-Π workstream (parallel).

This handoff focuses on (1) the ISA HiddenState investigation that landed `run_isa_multikey.py`, (2) the Û_vo Algorithm 2 audit + patch, and (3) a GPU-keymat port that *appears* to produce systematically weaker keymats than the CPU vendor builder — open bug to investigate.

## Headline

- Paper-faithful labelled-ridge ISA driver (`run_isa_multikey.py`) ships with multi-key attacker synthesis (K=64 keymats), row-split realistic methodology, GPU support, and CPU-fallback for the ridge solve.
- Audit determined aloepri's Algorithm 2 implementation **omits Û_vo** (the V↔O random projection from paper §5.2.3). Both `vendor/aloepri-py` and our `python/aloepri-llm/lib/alg2.py` had the same gap. Paper Table 4 attributes the last ~0.82 % → 0.0 % HiddenState reduction to Û_vo specifically.
- Patched: `lib/alg2.py` and `obfuscate_qwen3_gguf.py` now support `--alg2-u-vo`. Math is end-to-end verified (`(X · W̃_v.T) · W̃_o.T = X · W_v.T · W_o.T` to 1e-6 relative on in-memory test).
- Re-obfuscated Q3-4B + Q3-8B with Û_vo; captured plain Q3-8B hidden states; ran ISA multi-key on the Û_vo deployments.
- **4B Û_vo result with vendor CPU keymat**: 3.41 % top-1 (down from 5.11 % pre-Û_vo) — Û_vo working as expected.
- **4B Û_vo result with GPU-native keymat port**: 11.92 % top-1 on the SAME inputs — exceeds the plain calibration ceiling (10.18 %), which is structurally impossible. Confirmed reproducible. **Real bug in the GPU port** despite distributionally-equivalent statistics. Diagnosis in §3.
- **8B Û_vo paper-faithful TTRSR**: 9.00 % top-1, 22.14 % top-10. Landed after CPU-fallback ridge solve (GPU rocBLAS still failing at `hipblasStrsm` despite both `ROCBLAS_USE_HIPBLASLT=1` and `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` env vars). Û_vo barely attenuates at 8B — only 0.73 pp drop (7.5 % relative) vs the 1.70 pp drop (33 % relative) on 4B. **The 8B defense gap is structural, not closeable by Û_vo at our scale.**

## §1 What landed this session

### 1a — `run_isa_multikey.py` driver

Paper-faithful labelled-ridge ISA on hidden states. Attacker:

1. Runs the **public plain model** locally on own plaintext prompts → captures `State_plain[L]`.
2. Generates K independent attacker keymats `K_a^k` via Algorithm 1.
3. Synthesizes `State_a^k = State_plain[L] @ K_a^k` (covariance approximation).
4. Trains ridge on the K × n_train concatenation — multi-key forces key-invariant inversion.
5. Tests on deployment's actual obfuscated hidden states (legitimately captured server-side).

Threat model: paper §3.2 Kerckhoffs — never uses deployment τ or K. Split mode default = `row` (vocab-overlapping, realistic); `vocab` mode available as secondary stress-test reading.

Driver supports CPU + GPU (`--device gpu`), CPU-fallback ridge solve, two keymat implementations (`--keymat-impl vendor_cpu` | `gpu_native`).

Files:
- `evals/aloepri-attacks/m2_7/run_isa_multikey.py`
- Wrapper: same `run_in_gpu_container.sh` (parametric via `IMA_DRIVER` env var) — already exposes `ROCBLAS_USE_HIPBLASLT=1` + `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`.

### 1b — Û_vo Algorithm 2 patch

Audit (background agent task #24) found Algorithm 2's Û_vo random projection is **not implemented** in `lib/alg2.py` (V's `dense_transform=None`, O's input axis unaffected). Same omission in `vendor/aloepri-py/src/stage_h_attention_static.py:166`. Paper Table 4: KeyMat alone leaks 0.82 % HiddenState; +Head&BlockPerm closes to 0.0 % — so Û_vo is the load-bearing missing component.

Patched (commit not yet made, working tree):
- `python/aloepri-llm/lib/alg2.py`: added `generate_u_vo()` (QR-stabilised N(0, 1/d_head·I) sample), `u_vo`+`u_vo_inv` fields on `LayerAlg2Keys`, `enable_u_vo` flag in `build_layer_keys`, `dense_input_transform` parameter in `apply_o_output_transform`.
- `python/aloepri-llm/obfuscate_qwen3_gguf.py`: `alg2_enable_u_vo` parameter, CLI `--alg2-u-vo`, wires V's dense_transform to `block_diag(u_vo, n_kv_heads)`, wires O's input axis to `block_diag(u_vo_inv, n_q_heads)`, `aloepri.alg2_u_vo_applied` GGUF metadata key.

Math verification (in-memory smoke test): `Û_vo · Û_vo⁻¹` error ≤ 4.77e-7, condition number 3.57-5.87, E2E V→O cancellation `(X @ W̃_v.T) @ W̃_o.T - (X @ W_v.T @ W_o.T)` relative error 1.18e-6 → covariance preserved correctly.

### 1c — Û_vo re-obfuscation + captures

GGUFs at:
- `/home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-bf16-native.gguf`
- `/home/timo/.cache/huggingface/path-2-aloepri/qwen3-8b/keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-bf16.gguf`

Captures at:
- `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-uvo-20260521/captures/` (L=[0, 17, 21, 30])
- `evals/aloepri-attacks/results/sweep/cell-qwen3-8b-uvo-20260521/captures/` (L=[0, 17, 21, 30])
- `evals/aloepri-attacks/results/sweep/cell-qwen3-8b-plain-rerun-20260521/captures/` (plain Q3-8B, L=[0, 17, 21, 30])

Plain Q3-4B captures from prior workstream still valid; reused.

### 1d — ISA multi-key results (final at L=17, K=64, row-split, vendor CPU keymat)

Q3-4B at L=17 (paper-matched ~48 % depth):

| Setup | Top-1 | Top-10 | Plain ceiling |
|---|---|---|---|
| No Û_vo, vendor CPU keymat | 5.11 % | 21.17 % | 10.18 % |
| **Û_vo, vendor CPU keymat** | **3.41 %** | 12.90 % | 10.18 % |
| Û_vo, gpu_native keymat (diagnostic only) | 11.92 % | 25.55 % | 10.18 % ← above ceiling, invalid |

Q3-8B at L=17:

| Setup | Top-1 | Top-10 | Plain ceiling |
|---|---|---|---|
| No Û_vo, vendor CPU keymat | 9.73 % | 22.14 % | 10.18 % |
| **Û_vo, vendor CPU keymat** | **9.00 %** | 22.14 % | 10.18 % |

**Û_vo Δ summary:**

| Model | Δ top-1 (non-Û_vo → Û_vo) | Relative attenuation |
|---|---|---|
| 4B | −1.70 pp | **33 %** |
| 8B | −0.73 pp | **7.5 %** |

Headline reading: **Û_vo provides meaningful attenuation on 4B (33 % relative) but barely moves the needle on 8B (7.5 %).** Both still pass paper's `< 15 %` gate; 8B is at the edge — 9.00 % obfuscated vs 10.18 % plain ceiling = 1.18 pp margin only.

The dimensional-headroom hypothesis (bigger `d` defends better) is falsified by these measurements: at d=4096 the multi-key paper-faithful attacker recovers ~9 % of tokens regardless of whether Û_vo is enabled. The 8B defense gap is structural — full paper Algorithm 2 (KeyMat + Head&BlockPerm + Û_vo) is not enough at this scale.

#### 8B GPU ridge-solve OOM trail

Every Q3-8B attempt failed inside `torch.linalg.solve` with `HIPBLAS_STATUS_ALLOC_FAILED` on the iGPU. Mitigation history:
- Initial run (no OOM fix): failed at first ridge solve
- `del keymat_pool; torch.cuda.empty_cache()` before solve: failed at first ridge solve
- `ROCBLAS_USE_HIPBLASLT=1` env var: failed at first ridge solve (same hipblasStrsm call)
- `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` + try/except CPU-fallback around `torch.linalg.solve`: **landed**. CPU fallback fired three times (one per α in the grid); 432 s total runtime.

The 8B run JSON is `/tmp/aloepri-gpu-validation/isa-multikey-obf-8B-uvo-L17-row-cpufallback.json`. The CPU-fallback path in `_fit_ridge` is now permanent in the driver — future runs at 8B+ on the iGPU will automatically fall back if rocBLAS allocator fails.

§08 HTML still has 8B Û_vo as "pending" — needs update to `9.00 % top-1, 22.14 % top-10`.

## §2 What's left

### Immediate

1. **Update §08 HTML cells** for ISA HiddenState (paper-faithful) with the final figures:
   - Obfuscated (4B) Û_vo: **3.41 % top-1, 12.90 % top-10** (replace prior 5.11 % which was non-Û_vo)
   - Obfuscated (8B) Û_vo: **9.00 % top-1, 22.14 % top-10** (replace "pending")
   - Notes should call out: Û_vo lands but defense scales poorly with `d` — 4B gets 33 % relative attenuation, 8B gets 7.5 %.
2. **Document the 8B GPU ridge OOM workaround** in §08 or `aloepri-attacks.md`: CPU-fallback ridge solve is the path that worked; rocBLAS allocator failure on Strix Halo iGPU at d=4353 is real even with `ROCBLAS_USE_HIPBLASLT=1` + `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`.

### Open: GPU-keymat seed-convention bug

The user identified the divergence as a likely bug in the gpu_native keymat port, NOT mere sample variance. Evidence supporting "real bug":

- 4B Û_vo side-by-side (same inputs, same `attacker_seed`): vendor 3.41 % vs port 11.92 %. The port reading exceeds the plain identity-τ calibration ceiling (10.18 %), which is structurally impossible for a paper-faithful attacker (attacker output is bounded above by the inverter's ceiling on the no-defense task). So the port is sampling keymats that are *systematically more effective* at attacking the Û_vo deployment than vendor's keymats — implies the port's keymats are closer to "useful for inverting K_d" than the underlying Algorithm 1 distribution should produce.
- Distributional statistics agree closely (mean ~0, std ~0.137, Frobenius ~36 in both). Per-pair correlation across K=64 keymats also matches (~3e-3 mean). The bug is subtler than the obvious mismatch a coarse-distribution check catches.
- Synth-covariance eigenvalue spectrum differs: vendor's top eval is 86.7 vs port's 64.2 — port's K=64 samples are more isotropic. That alone doesn't explain *direction* of TTRSR shift, but it is one detectable signal that the two impls produce different empirical concentrations.

#### What the seed convention difference is

`vendor/aloepri-py/src/keymat.py` builds each random component (U, V, E1, E2, F1, F2, Z, C) by **constructing a fresh `torch.Generator(device='cpu')` and manually seeding it** with `init_seed + N` for fixed offsets N ∈ {1, 2, 3, 4, 5, 6, 7, 1011}. Eight independent MT19937 streams per keymat.

`run_isa_multikey._build_attacker_keymat_pool_gpu_native` constructs **one generator per k**, seeded with `attacker_seed + 1 + 10_000 * k`, and **advances** it through all 8 draws. One MT19937 stream per keymat.

These produce statistically-valid Algorithm 1 keymats but draw different specific samples. We expected the discrepancy to be K=64 sample noise; the empirical evidence (TTRSR above plain ceiling, reproducibly) suggests it's *not* sample noise — it's a systematic bias from how the seed convention interacts with one of the heavy linalg ops (QR, SVD, nullspace projection).

#### Hypotheses (from concrete to speculative)

| # | Hypothesis | Test |
|---|---|---|
| 1 | **rocSOLVER QR/SVD sign-convention or numerical behaviour differs from CPU LAPACK** — the port runs heavy linalg through rocSOLVER on the iGPU; vendor's CPU LAPACK uses different numerics. Sign / rotation / nullspace-orientation differences could push the port toward a particular "effective rotation" of the (B \| C \| E) basis that aligns with K_d. | Run port at `device='cpu'` (CPU MT19937 + CPU LAPACK, just port logic). If port-on-CPU gives ~3.4 %, the bug is GPU-specific (rocSOLVER). If port-on-CPU also gives ~12 %, the bug is in the port's seed-stream layout itself. |
| 2 | **Sample variance at K=64 across seed conventions** — both impls draw valid samples but with different bit patterns; one happens to give a more "lucky" set. | Sweep `attacker_seed ∈ {1, 2, 3, ...}` for both impls; compare distributions. If they bracket each other across seeds, sample variance dominates. |
| 3 | **MT19937 stream-position correlation** — within a single advancing generator, the 8 consecutive draws of different shapes may have subtle correlations the fresh-generator-per-draw pattern doesn't. Hard to reason about without empirical test. | Modify port to use 8 fresh generators with seed offsets (matching vendor's pattern) but still on GPU device. Compare. |

### The seed sweep (queued, not yet launched)

Script ready at `/tmp/aloepri-gpu-validation/keymat-seed-sweep.sh`. Runs 6 attacks (2 impls × 3 attacker_seeds ∈ {1, 2, 3}) on Q3-4B Û_vo at L=17, K=64. Reports top-1/top-10 in a summary table.

Expected ~15 min total: 3 vendor runs × ~3 min (CPU keymat build + GPU synthesis + GPU ridge) + 3 port runs × ~1 min.

Outcomes:
- If both impls fluctuate by ~5-10pp across seeds with overlapping ranges → hypothesis 2 (sample variance) confirmed, port is fine.
- If vendor stays ~3-5 % and port stays ~10-15 % across all seeds → hypothesis 1 or 3 — real port bug. Then move to port-on-CPU diagnostic to localise rocSOLVER vs seed-stream.

## §3 Concrete next-session plan

In priority order:

### Step 0 — Land the 8B Û_vo paper-faithful number (~10 min)

Wait for `b62131dnf` to land. If it succeeds, update §08 HTML cell `Obfuscated (8B)` for `ISA HiddenState (paper-faithful)` with the new value. If the CPU-fallback path executed, note that in the run-notes so future readers know the GPU ridge solve was bypassed.

### Step 1 — Run the seed sweep (~15 min)

```bash
bash /tmp/aloepri-gpu-validation/keymat-seed-sweep.sh
```

Outputs sweep results in `/tmp/aloepri-gpu-validation/sweep-4B-uvo-<impl>-seed<N>.json`. Tabulate top-1 across all 6 runs.

### Step 2 — Diagnose based on sweep outcome

- **If sample variance dominates** (overlapping distributions): the port is fine, both can be used. Document the K=64 noise floor in the §08 notes and move on.
- **If port has a structural bias** (consistent gap): localise via port-on-CPU.

To run port-on-CPU diagnostically: temporarily edit `run_isa_multikey.py` to pass `device="cpu"` to `_build_attacker_keymat_pool_gpu_native` regardless of caller's device argument. This isolates "port logic" from "rocSOLVER on iGPU". Run the same 4B Û_vo input. Compare to vendor's 3.41 %.

### Step 3 — If port-on-CPU still differs from vendor

The bug is in the port's seed-stream layout, not the device. Most likely culprit: the **C-matrix nullspace projection**. `_nullspace_basis` returns vh[rank:].T where vh is from `torch.linalg.svd(matrix, full_matrices=True)`. SVD's right-singular vectors are determined up to sign — different runs may pick different sign conventions for vectors in the nullspace. This propagates to `coeffs_c @ basis.T` and into the final K matrix's structure.

To test: capture vendor's and port's nullspace bases for the same F matrix, check whether they span the same subspace but with different basis orientations. If yes, the C-matrix orientation differs and that may be the bias source.

### Step 4 — Once port is verified equivalent to vendor

The original goal of the port was GPU acceleration. The vendor's CPU keymat build at d=4096, K=64 takes ~3-5 min — tolerable for the current measurement budget. Only worth investing in port-fix work if we plan to run K=158+ or repeat-experiment sweeps where CPU build dominates wall-clock.

If we don't need GPU keymat for the immediate measurement program, mark the port as `# experimental — known divergence vs vendor, use vendor_cpu for production` in the docstring and move on.

## §4 Operator references

### Files

- Driver: `evals/aloepri-attacks/m2_7/run_isa_multikey.py`
- Obfuscator: `python/aloepri-llm/obfuscate_qwen3_gguf.py` (+ `lib/alg2.py`)
- Vendor keymat: `vendor/aloepri-py/src/keymat.py` (CPU-only `torch.Generator(device='cpu')`)
- Docker wrapper: `evals/aloepri-attacks/m2_7/run_in_gpu_container.sh` (now sets `ROCBLAS_USE_HIPBLASLT=1` and `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`)

### Result JSONs

- 4B Û_vo vendor: `/tmp/aloepri-gpu-validation/isa-multikey-obf-4B-uvo-vendor_cpu.json`
- 4B Û_vo port: `/tmp/aloepri-gpu-validation/isa-multikey-obf-4B-uvo-gpu_native.json`
- 4B Û_vo first run (vendor): `/tmp/aloepri-gpu-validation/isa-multikey-obf-4B-uvo-L17-row.json`
- 8B Û_vo pending: `/tmp/aloepri-gpu-validation/isa-multikey-obf-8B-uvo-L17-row-cpufallback.json` (b62131dnf)
- Prior (no Û_vo) baselines: `isa-multikey-obf-4B-L17-row.json`, `isa-multikey-obf-8B-L17-row.json`

### GGUFs

- Q3-4B plain: `/home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/Qwen3-4B-Q8_0-untied.gguf`
- Q3-4B obf (non-Û_vo): `untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-bf16-native.gguf`
- Q3-4B obf (Û_vo): `untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-bf16-native.gguf`
- Q3-8B plain: bartowski cache `Qwen_Qwen3-8B-bf16.gguf`
- Q3-8B obf (non-Û_vo): `keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-bf16.gguf`
- Q3-8B obf (Û_vo): `keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-bf16.gguf`

### Open work tree state

Uncommitted (as of handoff):

```
M  docs/prototype/aloepri-llm.html        (ISA HiddenState paper-faithful row)
M  evals/aloepri-attacks/m2_7/run_in_gpu_container.sh   (env vars for rocBLAS)
M  evals/aloepri-attacks/m2_7/run_isa_multikey.py       (GPU support, dual keymat impl, CPU-fallback ridge)
M  python/aloepri-llm/lib/alg2.py                       (Û_vo)
M  python/aloepri-llm/obfuscate_qwen3_gguf.py           (Û_vo wiring)
A  docs/research/aloepri-attacks.md                     (ISA HiddenState section)
```

Commit recommendation: bundle into "**aloepri: Û_vo Algorithm 2 + ISA multi-key paper-faithful driver + GPU support**" once the 8B number lands. The GPU-keymat port can stay in the commit with the experimental docstring.

## §5 Suggested skills for next session

- `/diagnose` for the GPU-keymat seed-convention bug — disciplined hypothesis-test loop is the right shape after the sweep lands.
- `/grill-with-docs` if updating `aloepri-attacks.md` § ISA HiddenState with the final Û_vo numbers — keep terminology aligned with paper §F.1 + §5.2.3.
- No `/handoff` again unless an unexpected branch opens.

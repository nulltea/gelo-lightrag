---
type: dev-log
status: current
created: 2026-05-18
updated: 2026-05-26
tags: [aloepri, attacks, ttrsr, alg2, m2.7, chronicle]
companion:
  [
    aloepri-attack-bench-log,
    aloepri-attack-harness-findings,
    alg2-threat-model-log,
    aloepri-deployment-log,
    aloepri-status,
  ]
---

# AloePri attack chronicle — comprehensive

> One-stop reference for everything AloePri-attack-related: taxonomy,
> harness design, threat-model, dated configs + measurements + conclusions,
> deferred attacks, methodology notes. Distilled from ~18 handoffs,
> the M2.7 evals harness, JSON gate results, and the
> research/prototype docs at `/home/timo/repos/worktrees/private-rag/docs-refactor/private-rag/`.
>
> The dated chronicle in §4 is the spine; everything else is reference
> material for interpreting those entries.

## Contents

1. [Background](#1-background)
2. [Harness design (Phase 1/2/3) + deferred attacks (D1–D6)](#2-harness-design-phase-123--deferred-attacks-d1d6)
3. [Threat-model: matrix-Γ kernel on Qwen3](#3-threat-model-matrix-γ-kernel-on-qwen3)
4. [Dated chronicle](#4-dated-chronicle)
5. [Applicability under openweight threat model](#5-applicability-under-openweight-threat-model)
6. [Methodology + measurement discipline](#6-methodology--measurement-discipline)
7. [Open follow-ups](#7-open-follow-ups)
8. [Cross-references](#8-cross-references)

---

## 1. Background

### Attack taxonomy

| Code                                 | Class                                               | Observation surface              | Recovery target               | Scope under GELO threat model                                                   |
| ------------------------------------ | --------------------------------------------------- | -------------------------------- | ----------------------------- | ------------------------------------------------------------------------------- |
| **VMA**                              | weight inversion (sorted-quantile RowSort)          | static W tensors (W̃ vs W)        | token permutation τ           | out of scope (weight privacy not load-bearing)                                  |
| **IA**                               | weight invariants                                   | weight tensor relationships      | head perms / scaling factors  | out of scope                                                                    |
| **IMA-EmbedRow-ridge**               | activation inversion (ridge)                        | embed-row pairs (W̃ vs W)         | τ via linear map V·x = y      | bypassed by deployment-τ scarcity (TFMA+SDA ≤ 20 pairs < ridge bootstrap floor) |
| **IMA-EmbedRow-transformer**         | activation inversion (trained transformer inverter) | embed pairs (τ-invariant)        | τ via inverter                | **load-bearing for paper claim** (paper §F.1); pending validation               |
| **IMA-L0-activation**                | activation inversion at layer 0                     | attn_norm-0 (pre-W_q residual)   | token ids via ridge           | structural — Alg2 acts post-W_q, unreachable                                    |
| **IMA @ Qcur_normed-0**              | post-W_q + Q-norm activation                        | post-W_q + Q-norm at L=0         | token ids                     | partial; Alg2 orthogonal rotation absorbed by paired-data ridge                 |
| **ISA-HiddenState**                  | deep-layer residual stream                          | attn_norm-L (typically L=17, 23) | token ids via multi-key ridge | defended by Ẑ_block (post-repair)                                               |
| **ISA-AttnScore (`kq`)**             | pre-softmax Q·Kᵀ                                    | kq-L                             | tokens via attention pattern  | defended by head-shuffle Π_head + Alg2 (Q·Kᵀ designed to cancel)                |
| **Score-surface `kqv_out`**          | attention output (pre-W_o)                          | kqv_out-L                        | tokens via output covariance  | paper §5.4-bounded; paper-literal Alg2 lands 40 pp defense at L=5               |
| **Per-head fingerprint**             | SVD spectra on heads                                | W_q/W_k/W_v/W_o heads            | Π_head                        | incidentally defended by Algorithm 1 rectangular projection                     |
| **V/O channel-pair**                 | per-channel L2 magnitudes                           | W_v/W_o columns                  | Û_vo + Π_head                 | incidentally defended (same reason)                                             |
| **§5.4 Attention-output covariance** | output-level bound                                  | post-W_o                         | output error e_C^attn         | paper-literal Alg2 closes most of paper Table 4 gap                             |
| **NN**                               | cosine retrieval at any layer                       | hidden state                     | token id via embed-table NN   | defended (keymat expansion blocks naive NN)                                     |
| **TFMA**                             | wire-side token-frequency                           | obfuscated token stream          | τ from bigram stats           | weakly defended (paper §7.6: ≤ 20 pairs recovered under strong-Π)               |
| **SDA**                              | wire-side n-gram substitution                       | obfuscated token bigrams         | τ via decipherment            | weakly defended (BLEU-4 ≈ 2 — insufficient for meaningful text)                 |
| **Eigendecomposition leak**          | similarity-transform structure                      | `Γ_q = Mᵀ·Diag(γ)·M`             | M_q via `np.linalg.eig`       | **open**; analytically unambiguous, empirical measurement pending               |

### Methodology splits

- **Row-split** — attacker trains on random 50 % of eval vocab, tests on remaining 50 %. Realistic for server-side attacker with multiple queries.
- **Vocab-disjoint** — attacker trains on N = 64–512 prompts, tests on disjoint out-of-vocab prompts. Paper-faithful; measures generalisation. Floors ridge effectiveness on fixed-feature surfaces (where Q/K depend on token id, not content) — ~0 % is a measurement floor, not a defense victory.

### Gate convention (M2.7 Phase 2)

- **IMA C2 TTRSR < 10 %** (load-bearing for activation-privacy claim)
- **ISA C2 TTRSR < 10 %** (load-bearing for activation-privacy claim)
- **C0 control floor ≥ 95 %** (measurement sanity check)
- **TFMA top-1 ≤ 15 %, top-10 ≤ 15 %**
- **SDA BLEU-4 ≤ 5**

Both IMA + ISA gates must pass for non-zero CI exit code. C1 (mask-only) is intermediate; C2 (mask + shield k=8 σ=4.0) is the production gate.

### Three-condition control matrix

| Condition        | Executor                     | Mask               | Shield                      | Expected C2 TTRSR |
| ---------------- | ---------------------------- | ------------------ | --------------------------- | ----------------- |
| **C0 plain**     | `CapturingPlaintextExecutor` | none               | n/a                         | ~100 % (sanity)   |
| **C1 mask-only** | `InProcessTrustedExecutor`   | per-batch Haar/HD₃ | `ShieldConfig::NONE`        | < C0, > C2        |
| **C2 default**   | `InProcessTrustedExecutor`   | per-batch Haar/HD₃ | `ShieldConfig::new(8, 4.0)` | < 10 % target     |

---

## 2. Harness design (Phase 1/2/3) + deferred attacks (D1–D6)

### Phase 1 — Snapshot capture (landed 2026-05-18)

Rust-side infrastructure capturing PCIe-crossing activations in test-only mode:

- **`PcieSnapshot` struct** records `(seq_idx, layer, kind, masked_operand, masked_output)`
- **`SnapshotCapture` aggregator** with configurable buffer (`capture_outputs`, `max_snapshots`), `drain`, `reset`, `snapshots`, `dropped`
- **`InProcessTrustedExecutor::with_snapshot_capture(cfg)`** builder method
- **`InProcessTrustedExecutor::{enable,disable}_snapshot_capture`** in-place toggles
- **Hook sites:** `offload_linear`, `offload_qkv` (3 snapshots per call), `offload_linear_many`
- **Tests:** 11 covering default-off invariant, opt-in capture, seq-idx ordering, drain semantics, multi-output batching

Default: **capture disabled**, zero overhead and zero allocations in production. Capture strictly opt-in inside attack-harness binaries.

### Phase 2 — Python attack harness (in progress)

Directory layout at `evals/aloepri-attacks/`:

```
├── README.md                  — operator runbook
├── pyproject.toml             — pinned to AloePri commit + transformers + torch
├── conftest.py                — pytest fixtures (snapshot loader, 3-condition matrix)
├── snapshots_loader.py        — read safetensors snapshots → AloePri-shaped tensors
├── run_vma.py / run_ima.py / run_isa.py / run_tfma.py / run_sda.py / run_ia.py
├── run_all.py                 — single-shot runner producing one row per attack
├── results/                   — JSON outputs keyed by (model, config, attack)
└── tests/test_smoke_*.py      — pytest wrappers for CI
```

**Snapshot serialisation contract.** Rust-side test util at `crates/gelo-embedder/src/attack_export.rs` writes `.safetensors` + sidecar `.meta.json`:

- Keys: `snap{seq_idx:05}.{layer:03}.{kind}.{operand|output}` (Array2<f32>)
- Sidecar: schema_version, model_id, config (shield_k, shield_energy_scale, per_forward_mask, verify_probes, prompt_token_ids), snapshots array

The `n_data` field lets the Python harness strip shield rows (last `shield_k` rows of each operand) before attacks.

**AloePri commit pin.** Use pyproject `[tool.uv.sources]` git-pin against the commit recorded in `../../plans/aloepri-gemma.md` M2.1 — NOT a fragile sibling-worktree path.

**Prompt corpus.** AloePri's `DEFAULT_PROMPTS` (natural-language), capped at 256 prompts per condition to keep harness under 30 min.

### Phase 3 — CI release-gate

After Phase 2 lands:

1. Fast variant of `run_all.py` (~64 prompts, <5 min) wired into `.github/workflows/aloepri-gate.yml`.
2. **Threshold:** fail if IMA or ISA C2 ≥ 10 %.
3. Tag gate `aloepri-attacks`; require on PRs modifying mask/shield/sim/snapshot code.
4. Archive each gated PR's results for longitudinal tracking.

### Deferred attacks (D1–D6)

| ID     | Attack/Surface                              | Driver             | Snapshot capture | Why deferred                                                                                                                                                                                                                                                                             |
| ------ | ------------------------------------------- | ------------------ | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **D1** | Cross-batch Gram leak                       | not built          | already captured | Harness aggregates per-prompt; meaningful comparison needs C1-with-shield-on as a 4th condition (currently C1 clears shield).                                                                                                                                                            |
| **D2** | AttnScore observable                        | not built          | **not captured** | Phase 1 hooks `offload_linear/qkv/many`; attention-score path (`offload_attention_qkt`, `offload_attention_permuted_cached`) is unhooked. Requires new `SnapshotCapture` hook. Per AloePri Table 3: AttnScore is the harder surface (87 % under "Noise only", 0 % under Head&BlockPerm). |
| **D3** | KV-cache observable                         | not built          | **not captured** | KV cache stays inside `KvCache`, never crosses executor offload seam. If GPU has VRAM read access (our threat model), KV cache IS observable. Requires capture-after-write hook on `KvCache::append`.                                                                                    |
| **D4** | IMA paper-like at scale (256-prompt corpus) | built              | already captured | At 64 prompts (~680 training rows) inverter undertrains. Paper defaults: 128 seqs × 32 tok = 4096 rows. Gate `c0_ima_paper_like_at_least_50pct` fails on short runs; use `c0_ima_ridge_at_least_95pct` as sanity instead.                                                                |
| **D5** | PIIRSR (sensitive-token slice)              | not built          | already captured | AloePri reports PII Recovery Success Ratio as separate metric for medical/financial deployments. Filter `test_ids` to PII subset; recompute TTRSR. Small implementation cost.                                                                                                            |
| **D6** | RowSort weight-pair VMA                     | n/a (inapplicable) | n/a              | GELO doesn't obfuscate weights → no W_obfuscated → no Π to recover. Driver is `not_applicable` stub. No follow-up unless GELO adds weight-obfuscation defense (it won't — openweight is the design point).                                                                               |

---

## 3. Threat-model: matrix-Γ kernel on Qwen3

### Why the QK-norm site is a problem

Qwen3 inserts per-element RMSNorm (`attn_q_norm.γ_q`, `attn_k_norm.γ_k`, shape `(head_dim,) = (128,)`) **after** Q/K projections and **before** RoPE + attention. Paper's Algorithm 2 (§5.2.3) assumes Q flows directly from W_q into RoPE + dot product; this site doesn't exist in Qwen2.5 / Llama3 / DeepSeek-R1-Distill-Qwen.

```
W_q [d→128] → attn_q_norm(γ_q) → RoPE → ┐
                                         ├→ softmax(Q·Kᵀ/√d_h) → attn_v → W_o
W_k [d→128] → attn_k_norm(γ_k) → RoPE → ┘
```

For `attn_q_norm(W_q · x · M_q, γ_q)` to equal `attn_q_norm(W_q · x, γ_q) · M_q`, M_q must commute with `Diag(γ_q)`. In general it doesn't (γ_q non-uniform, co-adapted with trained Q activations). Per-input correlation `E[q² · γ²] > 0` pushes attention toward uniformity ("a again ... ..." loops).

### Matrix-Γ kernel extension

Replace elementwise γ multiplication with a per-head matmul:

```
Γ_q[h] = M_q[h]ᵀ · Diag(γ_q) · M_q[h]      shape (d_h, d_h)
```

Kernel:

```
Q_obf_normed = (Q_obf / RMS) · Γ_q[h]
             = (Q_plain / RMS) · Diag(γ_q) · M_q
             = Q_plain_normed · M_q    ✓
```

Per-input exact, no approximation, provided M_q is orthogonal (composition of rotation + Hadamard + permutation).

**Cost:** γ tensor grows from 128 floats to 128² per (layer, head, q|k). For Qwen3-1.7B (d_h=128, n_q=16, n_kv=8, 28 layers): ~42 MB extra against ~3.4 GB base = <1.3 %. Decode-time per token: ~11 M MACs/token = <0.5 % of forward FLOPs.

### What the tensor leaks

`Γ_q[h] = Mᵀ · Diag(γ) · M` is a **similarity transform of a diagonal**. `numpy.linalg.eig` recovers:

- **Eigenvalues:** γ_q multiset
- **Eigenvectors:** columns of M_q[h] up to permutation and sign

O(d_h³) per head — for d_h=128, sub-millisecond. Once M_q[h] is known, attacker computes `M_q[h]⁻¹ · Q_obs = Q_plain` at every QK-norm observation.

### Decision: Option 4.1 (scope-narrow)

| Option                                               | Cost               | Verdict                                                                                                                       |
| ---------------------------------------------------- | ------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| **4.1 Scope the deployment to paper's threat model** | zero               | **CHOSEN** — exclude weight-analysis attacks (including QK-norm eigendecomposition). Matches paper's own exclusion of VMA/IA. |
| 4.2 Additive noise on Γ_q                            | 1–2 weeks research | fragile — γ_q eigenvalue spread enormous (−1 to +68); fixed σ swaps noise-to-signal ratio orders of magnitude                 |
| 4.3 Per-prompt fresh M_q via thin-client handshake   | ~2 weeks           | "this is GELO" — if we wanted this, use path-1, not AloePri                                                                   |
| 4.4 Hide γ_q eigenvalue structure (fold γ into W̃_q)  | negative           | requires paper extension not derived in literature; §5.2.5 doesn't cover output-axis γ                                        |

**For future weight-privacy requirements:** the answer is not "tweak AloePri" — it is **use the path-1 GELO stack**.

### M2.7 Phase 2 acceptance gates

- IMA < 10 %, ISA < 10 % (load-bearing for activation-privacy)
- **Defended-surface list:** IMA, ISA, TFMA, SDA, NN
- **Out of scope:** VMA, IA, eigendecomposition of Γ tensors (weight-side attacks)

---

## 4. Dated chronicle

### 2026-05-18 — Phase 1 snapshot capture lands

**Source:** `docs/archive/handoffs/2026-05-18-aloepri-attack-resistance.md`

**Config tested:** Qwen3-1.7B plain (control); no obfuscated GGUF yet exercised by the harness.

**Phase 1 deliverables:** snapshot infrastructure in `crates/gelo-protocol/src/snapshot.rs`; 11 tests; capture disabled by default.

**Key decisions:**

- **Do not port** AloePri's static-weight obfuscation protocol; port only the attack suite.
- Use empirical TTRSR as a release gate.
- Three-condition control matrix (C0/C1/C2); target C2 < 10 % on IMA + ISA.
- 256-prompt corpus from `vendor/aloepri-py/src/defaults.py::DEFAULT_PROMPTS`.

**Handoff target:** Phase 2 worker should land safetensors serialisation contract, build three-condition matrix, gate on four numbered criteria. Phase 2 acceptance on Qwen3-1.7B (not Gemma 4 — real weights blocked on Phase 1.5).

---

### 2026-05-18 — Gemma 4 path-1 deferred

**Source:** `docs/archive/handoffs/2026-05-18-gemma4-architecture-support.md`

Gemma 4 has **5 residual-norm sites per block** (3 post-norms + 2 pre-norms). Paper §5.2.5 RMSNorm fusion is exact for pre-norms but **not** for post-norms; error compounds over 35 layers × 3 post-norms catastrophically. Active pivot away from Gemma 4 toward Qwen3.

**Carry-forward findings (validated):**

- llama.cpp upstream `LLM_ARCH_GEMMA4` is shipped; no fork needed for architecture itself.
- `hidden_size = d + 2h` propagates cleanly through llama.cpp metadata.
- K=V tying is runtime cache sharing only; GGUF stores separate attn_k/attn_v at every layer.
- PLE (Per-Layer Embeddings) is one fused `[8960, 262144]` tensor; τ permutation = `arr[:, tau]`.
- Algorithm 1 KeyMat math verified to ≤ 3×10⁻⁷ max-error at fp64.
- κ_correct ≈ 7.42 for E2B (d=1536, h=128, λ=0.3).

**Three independent work items when resuming (deferred):**

1. Covariant post-norm — llama.cpp `ggml_rms_norm_then_scale` op (~2–3 weeks) OR algebraic reformulation (research).
2. PLE token-axis permutation under τ.
3. p-RoPE-aware R̂_qk for global layers.

---

### 2026-05-19 (early) — Qwen3 QK-norm gap analysis

**Source:** `docs/archive/handoffs/2026-05-19-alg2-qwen3-shape-analysis.md`

**Three options evaluated:**

- **Option A (empirical κ_qk calibration):** medium risk — variance of ratio on K-side where γ_max = 68.
- **Option B (γ-commuting rotation):** **dead** per pre-flight measurement.

| ε    | mean pair-symm % | mean Ẑ-coverage % |
| ---- | ---------------- | ----------------- |
| 0.01 | 3.6              | 0.0               |
| 0.05 | 16.0             | 0.0               |
| 0.10 | 25.9             | 1.4               |
| 0.25 | 43.8             | 22.0              |

Layer 0 (M2.7 attack surface) fails threshold (20 % Q coverage, 25 % K coverage at ε=0.25). K-side γ values have extreme outliers (γ_max = 68 at layer 0); clusters collapse.

- **Option C (runtime κ correction in llama.cpp):** algebraically certain (1 week). Violates "no infra change" thesis but we already ship a forked llama.cpp. **Recommended.**

---

### 2026-05-19 (early) — Ẑ_block degeneracy bug

**Source:** `docs/archive/handoffs/2026-05-19-alg2-z-block-degeneracy.md`

**Bug:** `generate_block_perm` in `lib/alg2.py` (and `vendor/aloepri-py/src/attention_keys.py`) produces **identity permutation** under default parameters (`beta=8, gamma=1e3, rope_base=1e6`) for all head_dim ∈ {8, 16, 32, 64, 128} and seeds {7, 42, 99, 12345}.

**Mechanism:** softmax over ζ-log differences collapses to `[1.0, ~0, ~0, …]` at fp32. `rng.choice` always returns index 0 → window_size = 1 → identity.

**Why it didn't surface:** Qwen3 deployment forced `q_matrix = k_matrix = I` anyway (Option C deferred), so Ẑ_block was never exercised in inference.

**Fix sketches (deferred):** (a) drop γ to ~1; (b) use rope_base ~ 1e3 in score function only; (c) uniform random permutation. Each requires re-deriving the paper's RoPE-locality argument.

---

### 2026-05-19 — §05 baseline measurement (partial Alg2)

**Source:** `docs/archive/handoffs/2026-05-19-m2-7-attack-findings.md`

**GGUF:** `keymat-h128-pi-noise-alg2-fp32.gguf` (§05 partial Algorithm 2: inter-head shuffle Π_head + keymat h=128 + α-noise, no intra-head transforms).

**Model:** Qwen3-1.7B fp32. Methodology: vocab-disjoint. 64-prompt corpus.

| Attack                             |   Plain |           Obf | Gate     | Verdict                                         |
| ---------------------------------- | ------: | ------------: | -------- | ----------------------------------------------- |
| NN                                 | 100.0 % |         0.0 % | ≤ 15 %   | ✓                                               |
| **IMA basic** (attn_norm-0)        |  98.1 % |    **88.9 %** | ≤ 15 %   | **✗ fails 6×**                                  |
| **IMA paper-like** (N=256)         |  49.3 % |    **38.0 %** | ≤ 15 %   | **✗ fails 2.5×**                                |
| **ISA HiddenState** (attn_norm-23) |   8.7 % |    **16.3 %** | ≤ 15 %   | **✗ marginal — obf paradoxically raises TTRSR** |
| ISA AttnScore (kq-23)              |   2.9 % |         0.0 % | ≤ 15 %   | ✓                                               |
| TFMA                               |       — | 1.6 % / 5.1 % | ≤ 15 %   | ✓                                               |
| SDA                                |       — | 1.6×10⁻⁵ BLEU | ≤ 5 BLEU | ✓                                               |

**Data-scale sweep (IMA paper-like, ep=16):**

| N   | Plain top-1 | Obf top-1 | Plain top-10 |             Obf top-10 |
| --- | ----------: | --------: | -----------: | ---------------------: |
| 64  |       7.6 % |     6.2 % |       26.2 % |  17.8 % (undertrained) |
| 128 |      24.4 % |    11.5 % |       45.5 % |                 37.9 % |
| 192 |      39.1 % |    34.1 % |       54.9 % |                 47.1 % |
| 256 |      49.3 % |    38.0 % |       66.3 % | 47.5 % (paper-matched) |
| 384 |      62.4 % |    47.6 % |       76.7 % |                 55.3 % |
| 512 |      72.8 % |    46.8 % |       87.9 % |    55.1 % (saturating) |

**Diagnosis:** §05 ships only inter-head shuffle (Π_head), not intra-head transforms. Decoy dims from keymat expansion (2048 → 2304) carry recoverable structure for ridge.

**Conclusion:** partial Algorithm 2 insufficient; need Option C ramp.

---

### 2026-05-19 — Option C ramp + matrix-Γ kernel deploys

**Source:** `docs/archive/handoffs/2026-05-19-option-c-m2-7-rerun-findings.md`

**GGUF:** `keymat-h128-pi-noise-alg2-FULL-fp32.gguf` — matrix-Γ kernel in patched llama.cpp + R̂_qk on output axis; Ẑ_block silently identity; Ĥ_qk = I.

**Result:** all M2.7 numbers **bit-equal to §05**. M_q is post-W_q; layer-0 captures are pre-W_q (structurally unreachable). Algorithm 2 designed to make M_q · M_kᵀ = I, so Q·Kᵀ also preserved.

**Key finding:** Option C is correctly deployed algebraically, but its defense is **invisible to the current M2.7 attack suite**. The surface where M_q actually lives is **post-q_norm Q values** — not currently dumped. Tensor filter extended to:

```
--tensor-filter '^(attn_norm-(0|11|23)|Qcur_normed-(0|11|23)|Kcur_normed-(0|11|23)|kq-23)$'
```

---

### 2026-05-19 — Steps 0/1/2a (full Alg2 build sweep)

**Source:** `docs/archive/handoffs/2026-05-19-option-c-steps-0-1-2a-findings.md`

**GGUFs:**

- `…FULL-fp32.gguf` (R̂_qk only)
- `…FULL-zfix-fp32.gguf` (+ Ẑ_block fix)
- `…FULL-zfix-hadamard-fp32.gguf` (+ ±1 Walsh-Hadamard Ĥ_qk)

**Full ledger (vocab-disjoint):**

| Attack        | Surface        |         §05 |      + R̂_qk |  + Ẑ_block | + Ĥ_qk ±1 | Gate                              |
| ------------- | -------------- | ----------: | ----------: | ---------: | --------: | --------------------------------- |
| NN            | —              |       0.0 % |       0.0 % |      0.0 % |     0.0 % | ≤ 15 % ✓                          |
| **IMA basic** | attn_norm-0    |  **88.9 %** |      88.9 % |     88.9 % |    88.9 % | ≤ 15 % ✗                          |
| IMA basic     | Qcur_normed-0  |      88.9 % |      88.0 % |     88.0 % |    88.0 % | ≤ 15 % ✗                          |
| IMA basic     | Kcur_normed-0  |      76.4 % |      76.9 % |     76.9 % |    76.9 % | ≤ 15 % ✗                          |
| **ISA HS**    | attn_norm-23   |      16.3 % |      16.3 % | **11.5 %** |    11.5 % | ≤ 15 % **✓ passes after Ẑ_block** |
| ISA HS        | Qcur_normed-23 |      12.5 % |       9.6 % |     13.5 % |    13.5 % | ≤ 15 % ✓                          |
| ISA HS        | Kcur_normed-23 |       8.7 % |       7.7 % |      7.7 % |     7.7 % | ≤ 15 % ✓                          |
| ISA AS        | kq-23          |       0.0 % |       0.0 % |          — |         — | ≤ 15 % ✓                          |
| TFMA          | wire           | 1.6 / 5.1 % | 0.8 / 2.7 % |          — |         — | ≤ 15 % ✓                          |
| SDA           | wire           |      1.6e-5 |      1.4e-5 |          — |         — | ≤ 5 BLEU ✓                        |

**Key findings:**

- **Ẑ_block fix is the only step that moved a metric** (ISA HS 16.3 → 11.5 %, now passes).
- **±1 Hadamard Ĥ_qk is algebraically dead-end** against ridge (orthogonal sign flips trivially absorbed). With ±1 H, `M_q = M_k` so no Q/K diversity. Non-unit H requires runtime κ correction in kernel.
- **IMA basic structurally unreachable** at 88.9 % — layer-0 surface is pre-W_q. To move IMA below 15 %, levers outside Algorithm 2: larger keymat (h=256+), heavier α-noise (degrades generation), or non-linear keymat fold (paper-divergent).

**Bonus algebra fix:** Original M_k construction `R · H⁻¹ · Z⁻¹` only collapses M_q · M_kᵀ to I when Z² = I (identity-Z silently provided). Changed to **M_k = R · H⁻¹ · Z** (same Z, no inverse); `M_q · M_kᵀ = I` for any orthogonal Z.

---

### 2026-05-19 — Five-thread plan for next session

**Source:** `docs/archive/handoffs/2026-05-19-aloepri-attack-surface-followups.md`

Five threads queued (some superseded by later work):

1. **Thread 1 — re-measure "weight-inversion" attacks as prompt-inversion via weights.** VMA / IA / paper-IMA-basic / paper-IMA-paper-like all run offline (attacker has both W_e_obf and W_e_plain). Paper Table 1 reports them all passing 15 % gate on Qwen2.5-14B; we have not measured against Qwen3-1.7B FULL-zfix-hadamard.
2. **Thread 2 — sweep Algorithm 1 parameters** (α_e and h) to close IMA-L0-activation. Pre-flight: at what (α_e, h) does TTRSR drop ≤ 15 %? Cost vs perplexity tradeoff.
3. **Thread 3 — measure eigendecomposition leak** empirically. Run `numpy.linalg.eig` on Γ_q matrices; verify recovered M defeats intra-head obfuscation. ~5 min work.
4. **Thread 4 — port GELO-like dynamic defenses** + measure Gram leakage (2–3 weeks spike).
5. **Thread 5 — resolve paper-vs-deployment surface mismatch** in public docs (§08).

> **Note:** Threads 1–3 mostly absorbed by subsequent work (2026-05-20 paired-data analysis, 2026-05-26 paper-literal). The 2026-05-19 doc is marked `status: stale` with `superseded_by: 2026-05-20-aloepri-attacks-status-and-paired-data-defences`.

---

### 2026-05-20 (morning–afternoon) — Special-token Π fix + paired-data ridge discovery

**Sources:** `docs/archive/handoffs/2026-05-20-aloepri-pi-special-token-fix.md`, `docs/archive/handoffs/2026-05-20-aloepri-attacks-status-and-paired-data-defences.md`

**Special-token bug:**

- **Symptom:** HumanEval pass@1 → 0/20 with multi-language gibberish; llama-server returned 500 on 11/20 prompts.
- **Root cause:** `obfuscate_qwen3_gguf.py` permuted _all_ tokens in `[0, 151669)` including EOS (151645), BOS (151643), 26 special tokens. Server's "stop on EOS 151645" check never fired.
- **Fix:** read `tokenizer.ggml.token_type`; permute only `type ∈ {NORMAL=1, BYTE=6}`; keep `CONTROL/USER_DEFINED/UNUSED` at identity. For Qwen3-1.7B: 151,643 permuted, 26 identity.
- **Privacy note:** zero confidentiality loss (special token IDs are public in GGUF metadata).
- **Accuracy after fix:** keymat-h128-pi-noise-ae0.3-ah0.1-fp32 (fixed Π) → 7/20 = 35 % (Δ vs plain = −15 pp). Residual 5/20 server-500s (down from 11/20).

**Paired-data ridge discovery (Qwen3-1.7B, α_e=0.1, α_h=0.033):**

| Attack                  | Surface           |      Plain |  Obf (no-Alg2) |    Obf (+Alg2) | Status                                |
| ----------------------- | ----------------- | ---------: | -------------: | -------------: | ------------------------------------- |
| VMA top-1               | static W          |     99.6 % |          0.0 % |          0.0 % | ✓                                     |
| IA top-1                | invariants        |     98.6 % |          0.0 % |       0.0244 % | ✓                                     |
| **IMA-EmbedRow-ridge**  | embed pairs       | **99.2 %** |    **99.22 %** |    **99.22 %** | **✗ structural — paired-data attack** |
| NN                      | attn_norm-0       |          — |          0.0 % |          0.0 % | ✓                                     |
| **IMA-L0-activation**   | attn_norm-0       |          — |    **22.58 %** |    **22.58 %** | **✗ pre-W_q unreachable**             |
| ISA HS                  | attn_norm-23      |          — |         9.68 % |         9.68 % | ✓                                     |
| **IMA @ Qcur_normed-0** | post-W_q + Q-norm |          — |        35.48 % |    **38.71 %** | **✗ Alg2 +3 pp degradation**          |
| TFMA                    | wire              |          — | 0.0 % / 5.08 % | 0.0 % / 4.69 % | ✓                                     |
| SDA                     | wire              |          — |         1.5e-5 |         1.4e-5 | ✓                                     |
| HumanEval pass@1 (n=20) | gen               |       50 % |           40 % |           35 % | −5 pp Alg2 cost                       |

**Key finding:** **paired-data ridge attacker is the structural limit of static linear obfuscation.** Attacker with both W (plaintext from HuggingFace) and W̃ (server-hosted) fits a bijection V via closed-form ridge; recovers ~99 % at d=2048 on Qwen3-1.7B.

**Three mitigation strategies:**

- **2A — Hide the obfuscated model.** Attacker loses W̃ access; paired-data assumption breaks. Zero algorithm change; breaks "public LMaaS infrastructure" thesis.
- **2B — Obfuscate inside a TEE (insufficient alone).** Keeps keymat / τ inside attestation; doesn't prevent paired-data attack.
- **2C — Dynamic masking (fresh M_q per request, GELO-style).** Cross-batch pair collection fails. Breaks "no infrastructure change" but matches project's research direction. ~4–8 weeks engineering.

---

### 2026-05-20 PM — Direction 1 sweep + Path-B untie + bf16 default

**Source:** `docs/archive/handoffs/2026-05-20-aloepri-attacks-status-and-paired-data-defences.md` (PM update section)

**Qwen3-4B (d=2560, α_e=1.0, h=128 untied):**

| Attack                          |   Plain |       Obf (paper-literal Alg2) |
| ------------------------------- | ------: | -----------------------------: |
| IMA-EmbedRow-ridge              | 98.02 % | **57.03 %** (−41 pp from 1.7B) |
| IMA-L0-activation @ attn_norm-0 | 28.87 % |                        25.13 % |
| ISA HS @ attn_norm-23           | 11.67 % |                      10.64 % ✓ |
| HumanEval pass@1 (n=20)         |       — |                           30 % |

**Qwen3-8B (d=4096, α_e=1.0, h=128):**

| Attack                  |                                              Obf |
| ----------------------- | -----------------------------------------------: |
| **IMA-EmbedRow-ridge**  | **96.88 % — regression to near-plain at d=4096** |
| ISA HS @ attn_norm-23   |                                          ~10 % ✓ |
| HumanEval pass@1 (n=20) |                                 40 % (best ever) |

**Key revelation:** 4B → 8B IMA-EmbedRow-ridge **non-monotonicity** falsifies "dimensional headroom fixes everything" thesis. At d=4096, ridge still recovers clean closed-form (best ridge α rose 0.01 → 1.0), suggesting different qualitative solution regime at higher d, not continuation of d=2048 → 2560 trend.

**Bugs fixed in same session:**

- **Qwen3-4B tied embeddings:** Qwen3-4B ships `tie_word_embeddings: true`. Separate P̂_R / Q̂_R⁻ᵀ transforms invalid on shared `token_embd`. New utility `python/aloepri-llm/untie_qwen3_gguf.py` duplicates `token_embd.weight` to a new `output.weight` slot.
- **bf16 became default:** fp16 fails (denormal flush at ~6e-5 → ~1.15 % of attn_q entries to zero, breaks P̂·Q̂ = I_d cancellation → 0/20 HumanEval). bf16 (8-bit exponent, range ~1e-38) preserves the cancellation; 4B drift: fp32 57.81 % → bf16 57.03 % ridge (negligible).
- **Static-attack loader OOM:** 8B was loading at 66 GB RAM. Fixed via `extract_gguf_weights.py` bf16-native lazy loader (`ml_dtypes.bfloat16`); peak drops to 33 GB.
- **Docker Vulkan GPU device binding:** `--group-add video` exposes display (`renderD128`) but NOT compute (gid 992). Inference 6.3 → 22.1 tps (4B) on GPU after fix.

**Suggested next-session ordering:** re-run canonical 4B attacks on final code; run 8B IMA-L0-transformer (~8–10 min); diagnose 8B ridge regression; bring Q/K activation attacks to N=256; re-measure 1.7B at bf16 for parity.

---

### 2026-05-20 — IMA-EmbedRow-transformer plain-control failure

**Source:** `docs/archive/handoffs/2026-05-20-ima-embedrow-transformer-investigation.md`

**Finding:** trained-transformer IMA inverter fails identity plain control. Ridge gets 99.2 % top-1 on plain; transformer gets 0.0–0.4 % across 4 architecture variants.

**Root cause:** ridge is the closed-form least-squares solution. Transformer tries to approach W via AdamW GD with 1024 update steps; bounded ~5×10⁻³ per parameter per step; identity diagonal needs ~30 step-units to climb. AloePri reference uses paper-default budget (epochs=2, batch=8, 256 rows); over 256 updates GD doesn't converge.

**Interpretation:** paper's "IMA = 0 %" may be the constrained-attacker (ep=2) reading, not "no attacker can recover."

**Verdict:** drop IMA-EmbedRow-transformer from measurement table. Ridge (99.2 % plain → 97.66 % obf, 6.5× defense gap) is the load-bearing static-embedding-row measurement.

---

### 2026-05-21 — Û_vo lands + ISA multi-key + GPU keymat bug

**Source:** `docs/handoffs/2026-05-21-uvo-isa-multikey-and-gpu-keymat-bug.md`

**Û_vo Algorithm 2 omission fixed:** patched `lib/alg2.py` + `obfuscate_qwen3_gguf.py` with `generate_u_vo()` (QR-stabilised N(0, 1/d_head·I)). CLI flag `--alg2-u-vo`. Math verified: condition number 3.57–5.87; E2E V→O cancellation error 1.18×10⁻⁶.

**ISA HiddenState multi-key (paper-faithful, K=64, L=17):**

| Model  | Config                        |      Top-1 |      Top-10 | Plain ceiling | Δ (non-Û_vo → Û_vo) | Relative attenuation      |
| ------ | ----------------------------- | ---------: | ----------: | ------------- | ------------------- | ------------------------- |
| 4B     | No Û_vo, vendor CPU keymat    |     5.11 % |     21.17 % | 10.18 %       | —                   | —                         |
| **4B** | **Û_vo, vendor CPU keymat**   | **3.41 %** | **12.90 %** | 10.18 %       | −1.70 pp            | **33 %**                  |
| 4B     | Û_vo, gpu_native keymat (BUG) |    11.92 % |     25.55 % | 10.18 %       | —                   | **exceeds plain ceiling** |
| 8B     | No Û_vo, vendor CPU keymat    |     9.73 % |     22.14 % | 10.18 %       | —                   | —                         |
| **8B** | **Û_vo, vendor CPU keymat**   | **9.00 %** | **22.14 %** | 10.18 %       | −0.73 pp            | **7.5 %**                 |

**Verdict:**

- **Û_vo provides meaningful attenuation at 4B** (33 % relative) but **barely moves needle at 8B** (7.5 %). Defense gap at 8B is **structural** — not closeable by Û_vo at d=4096.
- **Dimensional-headroom hypothesis falsified.**

**GPU-keymat seed-convention divergence (open bug):**

- Vendor CPU keymat: fresh `torch.Generator` per component (8 streams per keymat). Port GPU-native: one advancing generator per keymat (1 stream).
- Distributional stats agree (mean ~0, std ~0.137); synth-covariance eigenvalues differ (vendor top eval 86.7 vs port 64.2).
- Root cause hypothesis: rocSOLVER QR/SVD sign convention or seed-stream layout.

**8B GPU ridge OOM** mitigated via CPU-fallback ridge solve (432 s total runtime; made permanent in driver).

---

### 2026-05-21 — IMA-EmbedRow-transformer paper disparity

**Source:** `docs/archive/handoffs/2026-05-21-ima-transformer-paper-disparity.md`

**Inverter:** Qwen2.5-0.5B (AutoModel with paper defaults: 2 decoder layers, 8 heads, 8 KV heads). 32-token sequence windows. AdamW(lr=3e-4), MSE loss, batch=8, ep=100. AMD Strix Halo ROCm 7.2.3.

**Measurements:**

| Cell                                     |       Top-1 |  Top-10 | Best epoch | Notes                                     |
| ---------------------------------------- | ----------: | ------: | ---------- | ----------------------------------------- |
| Plain identity-τ control (Qwen3-4B Q8_0) | **15.62 %** | 26.56 % | 99         | Checkpoint lost to crash                  |
| Obfuscated 4B real-τ (bf16 obf)          | **12.70 %** | 23.24 % | 98         | Defence gap 2.92 pp top-1, 3.32 pp top-10 |

Both curves monotonically climbing at ep=100; not saturated.

**Three competing hypotheses (paper claims 0 % on Qwen2.5-14B):**

1. **Our attack ≠ paper's attack (most likely):** inverter arch / corpus / hyperparam mismatch. Paper may have measured ep=2 only (we also got ~2 % at ep=2).
2. **Our obfuscation weaker:** structural leak (specials + UNUSED tail identity-τ pre-strong-Π) — but transformer attack uses public corpus (no τ-leak training data).
3. **Configuration / harness bug:** plain reaching 15.62 % at ep=100 (not ~99 %) suggests inverter itself is limited. **At ep=1000:** if plain rises to 50–90 % and obf catches up, defense broken; if obf lags, defense real.

**Next steps:** re-create plain checkpoint, extend both to ep=500, run inverter against reference Stage-K artifact.

**Infrastructure landed:** ROCm 7.2.3 + PyTorch 2.10 on AMD Strix Halo; `aloepri-ima-trainer:latest` image; `run_in_gpu_container.sh`; resumable training with content-addressed checkpoints; log-spaced eval cadence (3.5× speedup at ep=100).

---

### 2026-05-21 — Algorithm 2 quantisation gates

**Source:** `docs/archive/handoffs/2026-05-21-aloepri-quantisation-and-alg2-gaps.md`

**Quantisation × Algorithm 2 verdict:** fp32 required.

| Format | Size   | Output                       | Verdict      |
| ------ | ------ | ---------------------------- | ------------ |
| fp32   | 8.6 GB | coherent                     | ✅ reference |
| Q8_0   | 2.3 GB | degenerate "(((,chein,zech…" | ❌ breaks    |
| Q6_K   | 1.8 GB | 500 error                    | ❌ breaks    |
| Q5_K_M | 1.6 GB | word salad                   | ❌ breaks    |

**Mechanism:** AloePri keymat weights are heavy-tailed per row (max ≈ 55, std ≈ 4.7). Q8_0 stores 32-element blocks with one fp16 scale; within a heavy-tailed block small values round to zero, breaking per-row covariance.

**QK-norm decision deferred:** Option C (llama.cpp runtime κ correction) chosen over Option A (empirical κ, high risk) and Option B (γ-commuting rotation, dead). ~1 week effort.

---

### 2026-05-21 — Path-2 next-steps (gates queue)

**Source:** `docs/archive/handoffs/2026-05-21-aloepri-next-steps.md`

Three sequential gates queued for next session:

1. **Quantisation verdict** (fp32 vs Q8_0). Done in following session; verdict fp32.
2. **Algorithm 2 research** (paper-literal vs deployed default). Done 2026-05-26.
3. **Gemma 4 port** (post-Qwen3 validation). Deferred indefinitely.

---

### 2026-05-21 — AloePri on Gemma deferred

**Source:** `docs/archive/handoffs/2026-05-21-aloepri-gemma-deferred.md`

Active work pivoted from Gemma 4 to Qwen3-class on 2026-05-18. Blocker: post-norms break paper §5.2.5 fusion. Carry-forward findings preserved (see [`gemma-path1-log.md`](gemma-path1-log.md)).

---

### 2026-05-26 — Paper-literal Algorithm 2 closes most of Table 4 gap

**Source:** `docs/handoffs/2026-05-26-alg2-paper-literal-defense-gap.md`

**GGUF:** `untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-PAPERLIT-bf16-native.gguf` (9.7 GB).

**Two deviations vs our deployed Alg2:**

- **A1:** `k_matrix = Ĥ⁻¹ · Ẑᵀ` (paper) vs `R̂_qk · Ĥ⁻¹ · Ẑ` (ours)
- **A2:** `Û_vo = raw N(0, 1/d_head)` (paper, condition number ~500) vs `QR-stabilised + 0.05 σ` (ours, condition number ~6)

**Methodology:** Qwen3-4B, 512-prompt corpus, ridge multi-α {1e-4, 1e-2, 1.0}, 10-seed row-split mean, cosine-NN top-1 against Qwen3-4B embedding table (V=151,936). Vocab-disjoint also measured for comparison.

**Results — `kq` surface (pre-softmax Q·Kᵀ):**

| Layer | Plain row | Default obf | Paper-lit obf | Plain vocab | Default obf vocab |
| ----: | --------: | ----------: | ------------: | ----------: | ----------------: |
|     0 |   48.63 % |     47.22 % |   **43.22 %** |      0.43 % |            0.07 % |
|     5 |   38.69 % |     38.49 % |    **7.79 %** |      0.08 % |            0.04 % |
|    11 |   27.73 % |     26.95 % |    **7.52 %** |      0.02 % |            0.00 % |
|    17 |   22.41 % |     21.17 % |    **6.35 %** |      0.00 % |            0.00 % |
|    23 |   30.13 % |     29.67 % |    **6.49 %** |      0.00 % |            0.01 % |

**Results — `kqv_out` surface (attention output per head, pre-W_o; §5.4 "bounded"):**

| Layer | Plain row | Default obf | Paper-lit obf | Defense delta (default) | Defense delta (paper-literal) |
| ----: | --------: | ----------: | ------------: | ----------------------- | ----------------------------- |
|     0 |   97.46 % |     83.77 % |   **47.32 %** | +13.7 pp                | **+50.1 pp**                  |
|     5 |   50.77 % |     50.22 % |   **10.49 %** | −0.5 pp                 | **+40.3 pp**                  |
|    11 |   18.50 % |     19.83 % |   **10.57 %** | +1.3 pp                 | similar                       |
|    17 |   16.67 % |     17.75 % |   **10.04 %** | +1.1 pp                 | similar                       |
|    23 |   25.27 % |     25.99 % |   **10.04 %** | −0.7 pp                 | similar                       |

**Combined finding:** Paper Table 4's "87.14 % Noise+KeyMat → 0.0 % + Alg2" is explained as the compound of (a) paper-literal A1+A2 lowering obf TTRSR to single digits at L≥5, plus (b) vocab-disjoint split flooring ridge effectiveness near 0 % beyond L=0. The 87 % baseline itself remains unmeasured (likely requires paper's gradient-opt ISA attack or a different cell).

---

### 2026-05-26 — pow2-monomial UVO restores utility but leaves QK row-split open

**Source:** `docs/prototype/aloepri-pow2-uvo-findings-2026-05-26.md`, `docs/prototype/aloepri-llm.html`, `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-uvo-pow2e1-*`.

**Current best 4B obfuscation cell:**

```text
untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-pow2e1-bf16-native.gguf
```

Config: Qwen3-4B Q8_0 untied base, h=128, β=8, α_e=1.0, α_h=0.2, Alg2 matrix-Γ/Hadamard, bf16-native GGUF, `--alg2-u-vo-mode pow2-monomial --alg2-u-vo-pow2-exp 1`.

**Accuracy gate:**

| Cell                  | Quality probe |  HumanEval n=20 | Reading                                          |
| --------------------- | ------------: | --------------: | ------------------------------------------------ |
| Dense/default UVO     |          pass |     3/20 = 15 % | bf16 dense UVO hurts utility                     |
| Non-UVO canonical     |          pass |     6/20 = 30 % | reference utility for current obfuscation family |
| **pow2-monomial UVO** |      **pass** | **6/20 = 30 %** | recovers dense-UVO utility loss                  |
| h256 pow2 UVO         |          fail |         skipped | larger h is not a current accuracy fix           |

**Attack gate highlights, pow2-UVO h128 β8:**

| Surface / attack            |                         Result | Gate reading                                 |
| --------------------------- | -----------------------------: | -------------------------------------------- |
| Static VMA                  |  8.98 % top-1 / 23.44 % top-10 | pass on top-1; top-10 nontrivial             |
| Static IA                   |   0.05 % top-1 / 0.20 % top-10 | pass                                         |
| IMA EmbedRow transformer    |   0.00 % top-1 / 0.00 % top-10 | pass                                         |
| IMA EmbedRow ridge          | 55.47 % top-1 / 60.16 % top-10 | fail / out-of-paper side-probe               |
| Token TFMA                  |   0.78 % top-1 / 4.69 % top-10 | pass                                         |
| Token SDA                   |                 BLEU-4 1.28e-5 | pass                                         |
| ISA row-split `kq` L0       |    plain 48.63 % → obf 47.20 % | fail / invariant QK surface                  |
| ISA row-split `kqv_out` L0  |    plain 97.46 % → obf 82.45 % | partial; L0 output-surface defense preserved |
| ISA row-split `kqv_out` L17 |    plain 16.68 % → obf 16.68 % | no later-layer gain                          |

**Engineering result:** `evals/aloepri-attacks/m2_7/diagnose_isa/gpu_sweep.py` now has transparent long-run logging via `--progress-jsonl PATH` and flushed progress lines. The completed kqv_out run wrote:

- `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-uvo-pow2e1-attn-and-output-512-20260526/logs/gpu_sweep_kqv_out.progress.jsonl`
- `evals/aloepri-attacks/results/sweep/cell-qwen3-4b-uvo-pow2e1-attn-and-output-512-20260526/logs/gpu_sweep_kqv_out.summary.json`

**Roadmap implication:** pow2-monomial UVO is the current utility-preserving UVO form for bf16 deployment. It does **not** solve the strongest row-split Q/K attack; future defenses need Q/K-side changes or TEE/path-1 coverage for raw `kq`.

---

### 2026-05-26 — Path-2 recommendations (extracted to archive handoff)

**Source:** `docs/archive/handoffs/2026-05-26-aloepri-recommendations.md` (extracted from `aloepri-attacks.md` §"Implications for path-2")

**Recommendations:**

1. **Recommended deployment construction: paper-literal Alg2**, contingent on accuracy preservation under bf16. Deployed cell was understating actual defense by 7–40 pp on both surfaces.
2. **AloePri §5.4 protects the attention output surface** more than previously measured. Subject to accuracy validation, output-surface defense delta at L=0 is **50 pp** under paper-literal (vs 14 pp under default). At L≥5 the delta is **40 pp** (vs 0.5 pp).
3. **AloePri's score-surface defense under paper-literal is non-trivial at L≥5.** Paper-literal `kq` delta at L≥5 is 16–31 pp, dropping obf to single digits. L=0 surplus (~5 pp) is still small but no longer "no defense."
4. **TEE-protected attention (path-1) remains the gold standard** for adversaries who can capture either surface at L=0. Even paper-literal Alg2 leaks 43 % on `kq` and 47 % on `kqv_out` at L=0 — only an in-TEE first decoder layer eliminates the embedding-noise shadow.

### 2026-05-26 — Gate B + Gate C accuracy measurements

**Source:** `results/path-2-gate-{b,c-humaneval,c-mmlu,c-piqa}.json`

**Gate B (behavioural consistency, 5 prompts × 3 runs each):**

| Prompt                           | Plain class           | Keymat class          | Cross LCP (chars) | Verdict                                                                           |
| -------------------------------- | --------------------- | --------------------- | ----------------- | --------------------------------------------------------------------------------- |
| "What is the capital of France?" | deterministic         | deterministic         | 19                | **PASS**                                                                          |
| "Write a haiku about autumn."    | deterministic (123 c) | deterministic (48 c)  | 1                 | **FAIL** — obf output garbled (pipe chars; Û_vo numerical instability under bf16) |
| "def fibonacci(n):"              | deterministic         | deterministic         | 5                 | **PASS**                                                                          |
| "Translate to French: Hello..."  | deterministic         | deterministic         | 5                 | **PASS**                                                                          |
| "Once upon a time..."            | deterministic (161 c) | deterministic (143 c) | 48                | **PASS**                                                                          |

Largely PASS with minor coherence degradation on structured-format tasks (haiku).

**Gate C — HumanEval (50 tasks):**

| Model            | passed |   accuracy | parse_rate | wall                  |
| ---------------- | -----: | ---------: | ---------: | --------------------- |
| Plaintext        |     20 | **40.0 %** |        1.0 | 73.8 s                |
| keymat-h128-fp32 |     17 | **34.0 %** |        1.0 | 335.8 s (4.5× slower) |

**−6.0 pp** accuracy loss; within acceptable range.

**Gate C — MMLU (200 questions, 47 subjects):**

| Model            | correct | parsed |   accuracy | parse_rate |
| ---------------- | ------: | -----: | ---------: | ---------: |
| Plaintext        |     109 |    197 | **54.5 %** |      0.985 |
| keymat-h128-fp32 |     110 |    199 | **55.0 %** |      0.995 |

**+0.5 pp** (within noise). Negligible.

**Gate C — PIQA (200 prompts):**

| Model            | correct | parsed |   accuracy | parse_rate |
| ---------------- | ------: | -----: | ---------: | ---------: |
| Plaintext        |     137 |    200 | **68.5 %** |        1.0 |
| keymat-h128-fp32 |     129 |    199 | **64.5 %** |      0.995 |

**−4.0 pp** accuracy loss; same scale as HumanEval.

**Combined accuracy verdict:** paper-literal Alg2 exhibits 0–6 pp accuracy loss on the bench suite (MMLU +0.5, HumanEval −6, PIQA −4). Consistent with paper's "0–3 % loss" claim (assuming loss-magnitude not direction).

---

### 2026-05-27 — Qwen3-8B paperK/no-H/pow2 sweep + β-bifurcation derivation

**Source:** `docs/prototype/aloepri-qk-pow2-hybrid-findings-2026-05-27.md`,
`docs/research/aloepri-h-beta-interaction-2026-05-27.md`,
`evals/aloepri-attacks/results/sweep/cell-qwen3-8b-paperK-noH-uvo-pow2e1-b{2,4}-h{128,256}-20260527/`,
`evals/aloepri-attacks/results/sweep/cell-qwen3-8b-plain-reference-20260527/`.

**Hypothesis going in:** at Qwen3-8B (d=4096) the extra residual headroom
should let the 4B β=2 / no-H / pow2-monomial recipe absorb stronger Q/K
perturbations (β=4 or h=256 or both). Empirics falsify this on every
lever.

#### β / h cliff (8B, paperK / no-H / pow2-monomial / αₑ=1.0 / αₕ=0.2 / matrix-Γ / bf16)

|     h |     β | κ(K_d) | Quality probe                             |   HumanEval n=20 | Δ vs plain |
| ----: | ----: | -----: | ----------------------------------------- | ---------------: | ---------: |
|   128 | **2** |   7.79 | pass                                      |  **8/20 = 40 %** |     −10 pp |
|   128 |     4 |   7.79 | **fail** (single-token loops `is is is…`) |          skipped |          — |
|   256 |     2 |  10.67 | readable; no task-coherence               |         **0/20** |     −50 pp |
|   256 |     4 |  10.67 | **fail**                                  |          skipped |          — |
| plain |     — |   1.00 | —                                         | **10/20 = 50 %** |          — |

**Plain baseline thinking-mode doublecheck:** plaintext Qwen3-8B through
the _same_ `/completion` pipeline (AloePriClient routes to `/completion`,
never `/v1/chat/completions`; no chat template runs) gives 10/20 with
**0/20 plain completions containing `<think>` tokens**. 3/20 plain
failures show plain-English reasoning drift mid-code; the drift doesn't
crash plain accuracy. So Qwen3-8B's instruct/thinking-mode training
does not silently leak into raw base-completion regime, and the 50 %
plain ceiling is on-distribution for raw `/completion` (chat + thinking
reaches 70–80 % on this model).

#### β-bifurcation derivation

The empirical β cliff is structural, not gradual. With `H=I` (no
Hadamard signs):

```
M_q · M_kᵀ = R̂ · Ẑ² · R̂ᵀ            (paper-literal-K)
            = I                          (default-K)
```

So paper-literal-K differs from default-K only by `Ẑ²` between the two
rotations. `Ẑ_block` is the direct sum of independent in-window
permutations from `S_β`; `Ẑ² = I` iff every window's σ is involutive.
Counting involutions (OEIS A000085) over 32 windows at β=2 and 16 windows
at β=4:

|   β | Prob(σ²=I) per window | Global Prob(Ẑ²=I) |
| --: | --------------------: | ----------------: |
|   2 |                 1.000 |             **1** |
|   4 |                 0.417 |        4.4 × 10⁻⁷ |
|   8 |                 0.019 |       1.7 × 10⁻¹⁴ |

So **β=2 reduces paper-literal-K to default-K at the score surface**
(zero K-side defense), and **β=4 is the first β where the surface is
generically distorted**. Between them lies no smooth midpoint — β=3
isn't supported by the fixed-window sampler (RoPE-pair geometry). The
empirical "β=2 passes utility / β=4 collapses generation" cliff is
predicted exactly.

#### h-cliff: κ(K_d) depth compounding

κ(K_d) scales with h (independent of β):

|   h | κ(K_d) | per-layer worst-case bf16 drift κ·ε_bf16 | depth-compound at L=36 |
| --: | -----: | ---------------------------------------: | ---------------------: |
| 128 |   7.79 |                               6.1 × 10⁻² |         e^2.19 − 1 ≈ 8 |
| 256 |  10.67 |                               8.3 × 10⁻² |        e^3.00 − 1 ≈ 19 |

The 37 % κ jump (h=128 → h=256) blows up ~2.4× under the depth-
compounded bound, which is the operator-norm-level argument. The
actual decoder argmax sensitivity is sharper because 8B's vocabulary-
local geometry is denser — small operator drift flips top-1 over
32k vocab once it correlates across tokens, producing the
"loop attractor" failure mode (`if not isinstance(strings, list):
return None` repeated 15×) we observe at h=256 / β=2 even though
the readability heuristic passes on short prompts (France → Paris).

Full derivation in
`docs/research/aloepri-h-beta-interaction-2026-05-27.md` §§ 1–3, with
§4 sketching four candidate levers that don't collapse to the β=2 / β=4
dichotomy (mixed-window γ-sampler, low-rank additive R̂_qk delta,
anti-correlated cross-layer keys, layer-shared Ẑ_block).

#### Attack-harness readings on the working 8B cell (paperK / no-H / pow2-monomial / h=128 / β=2)

| Surface / attack                                                                   |        TTRSR top-1 | TTRSR top-10 | Risk       | Notes                                                                 |
| ---------------------------------------------------------------------------------- | -----------------: | -----------: | ---------- | --------------------------------------------------------------------- |
| Static VMA (256 prompts, 36 layers, 7 proj kinds)                                  |         **1.17 %** |       3.52 % | low        | strong defense                                                        |
| Static IA-gate (4096 test tokens)                                                  |         **0.07 %** |       0.27 % | low        | very strong                                                           |
| Static IA-attn (4096 test tokens)                                                  |         **0.02 %** |       0.05 % | low        | very strong                                                           |
| IMA-EmbedRow-transformer (paper §F.1 trained Qwen-backbone inverter, n_train=4096) |         **0.00 %** |       0.00 % | low        | actual paper attack defeated                                          |
| Per-head fingerprint, attn_q                                                       |             3.73 % |      18.14 % | low        | Q-head identity hidden                                                |
| Per-head fingerprint, attn_k                                                       |        **13.19 %** |      63.54 % | **medium** | K head identity partially recoverable                                 |
| Per-head fingerprint, attn_v                                                       |        **13.89 %** |      63.54 % | **medium** | V head identity partially recoverable                                 |
| Per-head fingerprint, attn_output                                                  |             3.21 % |      15.97 % | low        | O head identity hidden                                                |
| V/O channel-pair, vo_v_match                                                       |        **12.50 %** |      63.54 % | **medium** | V channel identity partially recoverable                              |
| V/O channel-pair, vo_o_match                                                       |             3.21 % |      15.89 % | low        | O channel identity hidden                                             |
| V/O channel-pair, vo_pair_match                                                    |             3.82 % |       3.82 % | low        | paired V+O identity hidden                                            |
| ISA AttnScore `kq` L=0 (row split, 3 seeds, 512 prompts; plain 49.02 % ± 0.61)     | **48.56 % ± 0.67** |      58.86 % | high       | **+0.46 pp defense** — predicted exact by β-bifurcation               |
| ISA AttnScore `kq` L=5 (plain 39.52 % ± 0.51)                                      | **39.59 % ± 0.65** |      53.71 % | high       | **−0.07 pp defense**                                                  |
| ISA AttnScore `kqv_out` L=0 (plain 93.02 % ± 0.93)                                 | **82.26 % ± 1.38** |      96.04 % | high       | **+10.76 pp defense** (Ûvo pow2 channel-scale lever)                  |
| ISA AttnScore `kqv_out` L=5 (plain 50.23 % ± 0.93)                                 | **49.44 % ± 1.22** |      62.10 % | high       | **+0.79 pp defense**                                                  |
| ISA HiddenState multikey K=64 at attn_norm-17 (paper-faithful, seed 20260521)      |        **10.22 %** |      20.68 % | **medium** | **higher than 4B β=2 ref (3.89 %, low)** — confirms 8B HS d=4096 leak |

> **IMA-EmbedRow-ridge attack DELETED 2026-05-27** from
> `m2_7/run_ima_embedrow_attacks.py` + `..._multikey.py` + orchestrator.
> It over-triggered on the ~293 identity-fixed special-token pairs
> regardless of obfuscation strength (96.88 % top-1 here, 97.66 % on
> 4B prior — both well above the 4B no-Alg2 row's 99.22 % control,
> indicating the attack reads identity-fixed tokens, not τ-permuted
> ones). It was not the paper's actual IMA attack surface. See
> `feedback_aloepri_attack_harness_disparities.md` memory note; the
> paper-faithful trained-transformer variant (IMA-EmbedRow-transformer)
> is retained and lands 0.00 % on this cell.

**Per-head and V/O reading:** K/V-side TTRSR (~13 %) is the
**incidental** recoverability of _which head a row belongs to_,
not of the row contents themselves — Q/O-side land at ~3 % (random-
guess floor on 32 heads is 3.13 %). This mirrors the 4B sweep
behavior. The bf16-commuting pow2-monomial UVO leaves the
V/O magnitude structure visible (signed permutation × power-of-two
scaling preserves L2 ranks per channel), so per-head V/O fingerprints
remain a small statistical leak; full-head row contents do not
recover. No surface in this measurement set is in the "high" risk
band.

**Wall + artifacts:**

- 8B obf GGUF builds: ≈3 min each (h=128 17.4 GB κ=7.79, h=256 18.4 GB
  κ=10.67) on host CPU via `obfuscate_qwen3_gguf.py`.
- Per cell, HumanEval n=20 = ≈11 min, quality probe = 22 s,
  static + IMA-EmbedRow attacks combined ≈ 4 min.
- Per-head fingerprint = 87 s, V/O channel-pair = 84 s.
- Attn capture (512 prompts, kq+kqv_out at L=0,5, flash-attn off) ≈
  35 min on the m2_7-attn-output image (Vulkan node visible but
  patched matrix-Γ kernels appear to fall back to CPU on flash-attn
  off path).

**ISA AttnScore reading:** `kq` defense is essentially zero at both
L=0 (+0.46 pp) and L=5 (−0.07 pp) — predicted exact by the
β-bifurcation: paper-literal-K at β=2 reduces to default-K at the
score surface (`Ẑ² = I` deterministically per § 1.3 of the research
note). The Ûvo pow2-monomial lever drives **+10.76 pp** at `kqv_out`
L=0 (plain 93.02 % → obf 82.26 %), but vanishes by L=5
(+0.79 pp). This mirrors the 4B β=2 cell pattern exactly. Both
absolute obf TTRSR numbers stay in the 48–82 % band — far above any
"defended" threshold — so this surface is **not effectively
protected** by static obfuscation alone.

**ISA HiddenState reading — the 8B HS gap re-confirmed.** At
attn_norm-17, K=64 multikey ridge, paper-faithful (row split,
attacker seed 20260521, vendor CPU keymat pool, ridge solve fell
back from GPU to CPU LAPACK on OOM at d=4096×4352):

| Cell                               | top-1 TTRSR | top-10 TTRSR | risk       |
| ---------------------------------- | ----------: | -----------: | ---------- |
| 4B β=2 / no-H / pow2 / h=128 (ref) |      3.89 % |      20.92 % | low        |
| **8B β=2 / no-H / pow2 / h=128**   | **10.22 %** |      20.68 % | **medium** |

This re-confirms the 2026-05-21 finding (above) that **Ûvo
attenuation drops with d**: at 8B's d=4096 there are more observed
coordinates for the ridge attacker to fit, and the bf16-commuting
pow2-monomial UVO no longer provides meaningful HiddenState
attenuation. The 8B β=2 cell lands at **medium** risk on
HiddenState where 4B β=2 was low — falsifying any "8B is more
private because larger d" hypothesis on this surface.

**Decisions / next steps:**

1. **Working operating point on 8B = paperK / no-H / pow2 / h=128 /
   β=2.** Only viable utility cell measured. Defense profile:
   strong on all static / weight-inversion surfaces (≤ 1.2 % VMA,
   ≤ 0.07 % IA, 0.00 % paper-faithful IMA-EmbedRow-transformer);
   weak on `kq` (no defense, structural by β-bifurcation); weak on
   HiddenState (10.22 % medium, worse than 4B); only `kqv_out` L=0
   moves +10.76 pp on the Ûvo lever.
2. **The β axis is exhausted on both 4B and 8B.** No smooth
   midpoint between "β=2 = no defense" and "β=4 = quality death" under
   `paper-literal-K + matrix-Γ` on Qwen3 dense.
3. **`kq` row-split is structurally undefended at β=2 on every model
   size** — the bifurcation makes paper-literal-K equivalent to
   default-K on the score surface. To defend `kq` you must change
   the K-side construction kind (mixed-window γ-sampler, low-rank
   additive R̂_qk delta, layer-shared Ẑ_block, or anti-correlated
   cross-layer keys — see `aloepri-h-beta-interaction-2026-05-27.md`
   §4) — or use path-1 (TEE-protected attention) for an in-TEE first
   decoder layer.
4. **HiddenState defense on 8B requires more than h=128 / β=2 /
   pow2-Ûvo.** The 4B → 8B HS gap is structural in d, not in our
   construction choices. Same followup candidates as #3 apply.

**GPU note:** `diagnose_isa/gpu_sweep.py` (AttnScore) successfully
used ROCm/rocSOLVER throughout. `run_isa_multikey.py` (HiddenState
multikey K=64) attempted GPU ridge solve but hit
`RuntimeError`/OOM at the 28928×4352 normal-equations form and
silently fell back to CPU LAPACK — same behaviour the
2026-05-21 8B run reported. The CPU fallback is correctness-
preserving but slow (~8 min total at K=64). Tracking as
infra-TODO: tighter rocSOLVER memory plan at d=4096 K≥64 would let
the multikey run stay on GPU end-to-end.

---

## 5. Applicability under openweight threat model

Nine of AloePri's ten primitives are **inapplicable** to GELO's threat model (they protect the wrong axis):

| AloePri primitive                                 | GELO applicability                                                                     |
| ------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Token-level vocab permutation τ                   | ❌ Token IDs are TEE-internal; embedding lookup never crosses PCIe                     |
| Embedding/lm_head Gaussian noise α_e·ε            | ❌ GPU has W_e directly; noise costs accuracy without changing what GPU already sees   |
| Key matrices P̂, Q̂ with expansion h                | ❌ Shield rows fill the "extra dims" role with per-batch freshness (strictly stronger) |
| Block-RoPE permutation Ẑ_block                    | ❌ Static defense against W_q ↔ W_k matching; we mask activations, not weights         |
| RoPE-aware Q/K rotation R̂_qk                      | ❌ Cancels in Q·Kᵀ; doesn't hide the dot product the GPU computes                      |
| Q/K scaling Ĥ_qk                                  | ❌ Same cancellation                                                                   |
| V/O paired invertible Û_vo                        | ❌ Cancels in V·O composition                                                          |
| Inter-head permutation τ_kv, τ_group              | ❌ Permutes static weight storage; adversary matches by content                        |
| RMSNorm κI approximation                          | ❌ Compensates for P̂ expansion we don't introduce                                      |
| **Empirical attack suite** (`src/security_qwen/`) | ✅ **Highly portable; the one we ported.**                                             |

**The structural difference:** AloePri is fully static — one τ, one {P̂, Q̂}, one ε_e, all baked into weights. Same prompt twice produces identical obfuscated traffic; observations accumulate across requests. GELO+TwinShield resamples fresh Haar A_b per forward pass; each batch is information-theoretically independent. GELO is in a categorically stronger security class.

---

## 6. Methodology + measurement discipline

### K=64 keymat-pool sample variance (≈5 pp at d=2560)

Single-seed multi-key TTRSR readings at K=64 attacker-keymat-pool carry ~5 pp standard deviation at d=2560 (Qwen3-4B). A 2×2 factorial over PRNG (MT19937 vs Philox) and LinAlg (LAPACK vs rocSOLVER) produced apparently contradictory readings (3.41 % vs 11.92 %); 6-seed sweeps revealed all corners sample TTRSR from indistinguishable distributions (Welch t-test p > 0.4 pairwise). The 8.5 pp gap was a noise sample (z-score ±1.0 and ±0.5), not a structural signal. **The original "GPU port bug" diagnosis is retracted.**

**Implication:** any single-seed multi-key TTRSR reading carries ~5 pp uncertainty. Comparisons within 5 pp are noise. Re-runs should use ≥5 seeds with reported mean ± std.

### Vocab-disjoint methodology floor

Under vocab-disjoint split, ridge attacker on fixed-feature surfaces (where Q/K depend on token id, not content) can only learn the identity-like baseline. The "0 %" in paper Table 4 is a measurement floor, not a defense victory. Paper Table 4's reproducibility story:

- Paper-literal A1+A2 construction lowers obf TTRSR to single digits at L≥5.
- Vocab-disjoint methodology floors both plain and obf to ~0 % beyond L=0.
- Compound: paper's "0.0 %" reproduces.

### Three-condition matrix (C0/C1/C2)

| Cond         | Purpose                                 | Expected C2 IMA/ISA |
| ------------ | --------------------------------------- | ------------------- |
| C0 plain     | Sanity: attacks work (no defense)       | ≥ 95 %              |
| C1 mask-only | Baseline: what does the mask alone buy? | < C0, > C2          |
| C2 default   | Production gate (mask + shield)         | < 10 %              |

### Gate-acceptance summary

Phase 2 acceptance requires:

1. `run_all.py --condition c2` produces results JSON with all 6 attacks reporting TTRSR.
2. C0 reports TTRSR ≥ 95 % on IMA + ISA + VMA.
3. C2 reports TTRSR < 10 % on IMA + ISA.
4. C1 shows gap between C0 and C2 (shield adds measurable defense).
5. Results JSON committed to `results/path-1-attacks.json`.

---

## 7. Open follow-ups

### Critical (gates load-bearing claims)

1. **Diagnose Qwen3-8B ridge regression** (IMA-EmbedRow 96.88 % vs 4B's 57 %). SVD analysis of `W_e_obf @ W_e_plain.pinv()` across model sizes + candidate-pool sensitivity sweep. Estimated: SVD ~2 hr + sweep ~4 hr.
2. **Measure eigendecomposition leak empirically.** Run `numpy.linalg.eig` on Γ_q matrices in current GGUF; verify recovered M defeats intra-head obfuscation by re-running IMA on Q-normed surface with M⁻¹ applied. ~5 min compute.
3. **Per-component A1 vs A2 attribution** for paper-literal Alg2. Build cells with only one deviation; settle whether to recommend paper-literal or hybrid deployment. ~2 hr build + 30 min measurement.

### Important (close paper-reproducibility story)

4. **Compound paper-literal × vocab-disjoint ridge** for both kq and kqv_out surfaces. ~20 min measurement.
5. **Paper-literal accuracy validation.** HumanEval + coherence-check + bf16 condition-number risk verification before deployment migration.
6. **IMA-EmbedRow-transformer ep=500 extension.** Run both plain + obf checkpoints to ep=500 (resumable). Inspect divergence; settles hypothesis A/B/C.

### Deferred (require new infrastructure or threat-model shift)

7. **D1 cross-batch Gram-leak** — needs C1-with-shield-on as 4th harness condition.
8. **D2 AttnScore observable** — needs new SnapshotCapture hook in offload_attention paths.
9. **D3 KV-cache observable** — needs capture-after-write hook on `KvCache::append`.
10. **D5 PIIRSR sensitive-token slice** — small effort; not yet in CI gate.
11. **Direction 2C dynamic-mask design sketch** — if static obfuscation gates fail on production-scale models, sketch per-prompt fresh-mask handshake. Prerequisite for Gram-leakage measurement.

### Already done (resolved during chronicle period)

- Special-token Π fix (2026-05-20).
- Û_vo Alg2 omission (2026-05-21).
- Strong-Π server patch via chat_parser=epsilon (2026-05-21).
- Static-attack loader OOM via bf16 lazy loader (2026-05-20 PM).
- Qwen3-4B tied-embeddings untie utility (2026-05-20 PM).
- bf16 → default output dtype (2026-05-20 PM).
- Paper-literal Alg2 measurement on Qwen3-4B (2026-05-26).
- K=64 variance investigation closed (vendor vs port = same distribution; "GPU port bug" retracted).

---

## 8. Cross-references

### Companion docs (same workstream, narrower scope)

- [`aloepri-attack-bench-log.md`](../../archive/dev/logs/aloepri-attack-bench-log.md) (archived) — earlier distillation focused on TTRSR tables (subset of §4 here).
- [`aloepri-attack-harness-findings.md`](../../archive/dev/logs/aloepri-attack-harness-findings.md) (archived) — Phase 1 OOM incident (subsumed into §4 entry for 2026-05-19; original preserved as standalone reference).
- [`alg2-threat-model-log.md`](../../archive/dev/logs/alg2-threat-model-log.md) (archived) — broader threat-model + bug log (Qwen3 architectural gaps, Ẑ_block / QK-norm / Û_vo deep dives).
- [`aloepri-deployment-log.md`](../../archive/dev/logs/aloepri-deployment-log.md) (archived) — deployment fixes (special-token, server patches, quantisation).
- [`aloepri-status.md`](aloepri-status.md) — running plan log for path-2 work.

### Static reference

- [`../prototype/aloepri-attack-harness.md`](../prototype/aloepri-attack-harness.md) — Phase 2/3 harness spec.
- [`../prototype/aloepri-attack-harness-followups.md`](../prototype/aloepri-attack-harness-followups.md) — D1–D6 deferred attacks.
- [`../prototype/aloepri-qk-norm-matrix-gamma-threat-model.md`](../prototype/aloepri-qk-norm-matrix-gamma-threat-model.md) — matrix-Γ threat model.
- [`../../research/aloepri-attacks.md`](../../research/aloepri-attacks.md) — research-flavored conceptual descriptions of each attack.
- [`../../research/aloepri-vs-gelo.md`](../../research/aloepri-vs-gelo.md) — technique-by-technique applicability matrix.
- [`../../research/aloepri-keymat-variance.md`](../../research/aloepri-keymat-variance.md) — K=64 variance investigation closeout.

### Code + data

- `crates/gelo-protocol/src/snapshot.rs` — Phase 1 snapshot capture.
- `crates/gelo-embedder/src/attack_export.rs` — serialisation utility (planned).
- `evals/aloepri-attacks/{README, ATTACK_FAMILIES, m2_7/README, m2_7/HIDDEN_STATE_GAP}.md` — operator runbook + attack matrix.
- `evals/aloepri-attacks/m2_7/{run_isa_multikey, run_ima_embedrow_attacks}.py` — attack drivers.
- `python/aloepri-llm/{lib/alg2.py, obfuscate_qwen3_gguf.py, untie_qwen3_gguf.py}` — obfuscator + untie utility.
- `vendor/aloepri-py/src/security_qwen/` — AloePri reference attack suite.
- `results/path-2-gate-{b,c-humaneval,c-mmlu,c-piqa}.json` — gate B/C measurement results.

### Source handoffs distilled into §4 (chronicle)

| Date       | Handoff                                                         |
| ---------- | --------------------------------------------------------------- |
| 2026-05-18 | `2026-05-18-aloepri-attack-resistance.md`                       |
| 2026-05-18 | `2026-05-18-gemma4-architecture-support.md`                     |
| 2026-05-19 | `2026-05-19-alg2-qwen3-shape-analysis.md`                       |
| 2026-05-19 | `2026-05-19-alg2-z-block-degeneracy.md`                         |
| 2026-05-19 | `2026-05-19-m2-7-attack-findings.md`                            |
| 2026-05-19 | `2026-05-19-option-c-m2-7-rerun-findings.md`                    |
| 2026-05-19 | `2026-05-19-option-c-steps-0-1-2a-findings.md`                  |
| 2026-05-19 | `2026-05-19-aloepri-attack-surface-followups.md`                |
| 2026-05-20 | `2026-05-20-aloepri-pi-special-token-fix.md`                    |
| 2026-05-20 | `2026-05-20-aloepri-attacks-status-and-paired-data-defences.md` |
| 2026-05-20 | `2026-05-20-ima-embedrow-transformer-investigation.md`          |
| 2026-05-21 | `2026-05-21-aloepri-quantisation-and-alg2-gaps.md`              |
| 2026-05-21 | `2026-05-21-aloepri-next-steps.md`                              |
| 2026-05-21 | `2026-05-21-aloepri-gemma-deferred.md`                          |
| 2026-05-21 | `2026-05-21-ima-transformer-paper-disparity.md`                 |
| 2026-05-21 | `2026-05-21-uvo-isa-multikey-and-gpu-keymat-bug.md`             |
| 2026-05-26 | `2026-05-26-alg2-paper-literal-defense-gap.md`                  |
| 2026-05-26 | `2026-05-26-aloepri-recommendations.md`                         |

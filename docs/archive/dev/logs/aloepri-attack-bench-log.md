---
type: dev-log
status: stale
created: 2026-05-18
updated: 2026-05-26
tags: [aloepri, attacks, ttrsr, alg2, m2.7]
companion: [2026-05-18-aloepri-attack-resistance, 2026-05-19-m2-7-attack-findings, 2026-05-19-option-c-m2-7-rerun-findings, 2026-05-19-option-c-steps-0-1-2a-findings, 2026-05-19-aloepri-attack-surface-followups, 2026-05-20-aloepri-attacks-status-and-paired-data-defences, 2026-05-26-alg2-paper-literal-defense-gap]
superseded_by: aloepri-attack-chronicle
archive_reason: "Earlier (2026-05-26) distillation of TTRSR tables across 7 handoffs. Fully absorbed into the comprehensive chronicle, which adds taxonomy, threat-model, harness design, and 11 additional handoffs."
---

# AloePri attack bench — distilled measurement log

> Knowledge layer aggregating TTRSR measurements and defense decisions
> across the AloePri attack workstream (2026-05-18 → 2026-05-26). Source
> handoffs retain full diagnostic reasoning and exploration threads; this
> log is the canonical reference for operational leak metrics:
> "did we defend attack X at layer L under config C?"

## Attack taxonomy

Six model-agnostic attacks ported from AloePri (ByteDance, arXiv 2603.01499), measured against Qwen3-1.7B / 4B / 8B on PCIe-side observables:

| Code | Class | Observation surface | Recovery target | Scope |
|---|---|---|---|---|
| **VMA** | weight inversion | weight tensors | permutation τ | out of scope (weight privacy not load-bearing) |
| **IA** | weight invariants | weight tensor relationships | permutation τ | out of scope (same) |
| **IMA** | activation inversion (ridge, transformer) | hidden-state rows + attention Q/K post-norm | token embedding / token id stream | in-scope; layer-0 activation unreachable by Algorithm 2 |
| **ISA** | score surface (hidden state deep layer; attention scores pre-softmax) | residual stream attn_norm-L + attention Q·Kᵀ | embeddings / token logits | in-scope; defended by Algorithm 2 + head permutation |
| **TFMA** | token-frequency matching | wire-side token stream | plaintext sequence | in-scope; defended by τ token permutation |
| **SDA** | bigram-substitution | wire-side token bigrams | plaintext bigrams | in-scope; defended by τ permutation |

Methodology splits: **row-split** (attacker trains on random 50 % of eval vocab, tests on remaining 50 % of same vocab) vs **vocab-disjoint** (attacker trains on N=64–512 prompts, tests on disjoint out-of-vocab prompts — paper-faithful, measures generalisation).

---

## Cumulative TTRSR trajectory

### §05 baseline (2026-05-19, partial Alg2)

Config: `keymat-h128-pi-noise-alg2-fp32.gguf` — partial Algorithm 2 (inter-head shuffle Π_head only, missing intra-head R̂_qk / Ĥ_qk / Ẑ_block). Qwen3-1.7B fp32.

| Attack | Layer/surface | Plain (row) | Obf (row) | Gate | Verdict |
|---|---|---|---|---|---|
| NN | attn_norm-0 | 100.0 % | 0.0 % | ≤ 15 % | ✓ |
| IMA basic (ridge) | attn_norm-0 | 98.1 % | **88.9 %** | ≤ 15 % | **✗ fails 6×** — pre-W_q surface unreachable |
| IMA paper-like (transformer, N=256) | attn_norm-0 | 49.3 % | **38.0 %** | ≤ 15 % | **✗ fails 2.5×** |
| ISA HiddenState | attn_norm-23 | 8.7 % | **16.3 %** | ≤ 15 % | **✗ marginal fail** (obfuscation paradoxically raises) |
| ISA AttnScore | kq-23 | 2.9 % | 0.0 % | ≤ 15 % | ✓ |
| TFMA | token stream | — | 1.6 % top-1 / 5.1 % top-10 | ≤ 15 % | ✓ |
| SDA | token stream | — | BLEU-4 = 1.6×10⁻⁵ | ≤ 5 BLEU | ✓ |

Diagnosis: partial Alg2; IMA-basic + ISA-HS shared root cause is decoy dims from keymat expansion (2048→2304) recoverable via ridge without intra-head transforms.

### Post-repair: full Alg2 + matrix-Γ + steps 0/1/2a (2026-05-19)

Config: `keymat-h128-pi-noise-alg2-FULL-zfix-hadamard-fp32.gguf` — complete Algorithm 2 deployed (R̂_qk rotation, Ẑ_block fixed-window perm, ±1 Hadamard Ĥ_qk). Matrix-Γ kernel extends per-head M_q into Q/K post-norm.

| Attack | Surface | §05 frozen | Opt-C (R̂_qk) | +Ẑ_block | +Ĥ_qk ±1 | Gate | Change |
|---|---|---|---|---:|---:|---|---|
| NN | attn_norm-0 | 0.0 % | 0.0 % | 0.0 % | 0.0 % | ≤ 15 % | ✓ |
| IMA basic | attn_norm-0 | 88.9 % | 88.9 % | 88.9 % | 88.9 % | ≤ 15 % | **✗ unchanged** |
| IMA basic | Qcur_normed-0 | n/a | 88.0 % | 88.0 % | 88.0 % | ≤ 15 % | **✗ unchanged** |
| ISA HS | attn_norm-23 | 16.3 % | 16.3 % | **11.5 %** | 11.5 % | ≤ 15 % | **✓ passes after Ẑ_block** |
| ISA HS | Qcur_normed-23 | n/a | 9.6 % | 13.5 % | 13.5 % | ≤ 15 % | ✓ |
| ISA AttnScore | kq-23 | 0.0 % | 0.0 % | — | — | ≤ 15 % | ✓ |
| TFMA | token stream | 1.6 / 5.1 % | 0.8 / 2.7 % | — | — | ≤ 15 % | ✓ |
| SDA | token stream | 1.6×10⁻⁵ | 1.4×10⁻⁵ | — | — | ≤ 5 BLEU | ✓ |

Key findings:
- **Ẑ_block was the only step that moved a metric** (ISA HS attn_norm-23 16.3 → 11.5 %, now passes).
- **±1 Hadamard Ĥ_qk algebraically dead-end** against ridge (orthogonal sign flips trivially absorbed by linear fit).
- **IMA basic remains structural** — layer-0 pre-W_q, unreachable by M_q regardless of intra-head transforms.
- Matrix-Γ kernel correctly deployed (M_q · M_kᵀ = I invariant holds to 1×10⁻⁷ rel; coherent generation preserved).

### Model scaling + α-noise sweep (2026-05-20)

Qwen3-1.7B / 4B / 8B with paper-default α_e=1.0 (vs earlier α_e=0.1). Discovery: **static linear obfuscation cannot defeat paired-data ridge attacker** (attacker has both plaintext W and obfuscated W̃ from HuggingFace + server, fits bijection V via ridge).

**Qwen3-1.7B (d=2048, α_e=0.1):**

| Attack | Surface | Plain (row) | Obf (row) | Verdict |
|---|---|---|---|---|
| IMA-EmbedRow-ridge | W_e pairs | 99.22 % | 99.22 % | **✗ 99 % — paired-data structural** |
| IMA-L0-activation | attn_norm-0 | 22.58 % | 22.58 % | **✗ 23 % — pre-W_q unreachable** |
| IMA @ Qcur_normed-0 | post-W_q + Q-norm | 35.48 % | 38.71 % | **✗ 39 % — Alg2 +3 pp degradation** |
| ISA HS | attn_norm-23 | 9.68 % | 9.68 % | ✓ |
| HumanEval pass@1 | n=20 | — | 7/20 (35 %) | ~−5 pp Alg2 cost |

**Qwen3-4B (d=2560, α_e=1.0, h=128 untied):**

| Attack | Surface | Plain | Obf | Verdict |
|---|---|---|---|---|
| IMA-EmbedRow-ridge | W_e pairs | 98.02 % | **57.03 %** | ✗ but **−41 pp** vs 1.7B |
| IMA-L0-activation | attn_norm-0 | 28.87 % | 25.13 % | ✗ |
| ISA HS | attn_norm-23 | 11.67 % | 10.64 % | ✓ |
| HumanEval pass@1 | n=20 | — | 6/20 (30 %) | Acceptable |

**Qwen3-8B (d=4096, α_e=1.0, h=128):**

| Attack | Surface | Plain | Obf | Verdict |
|---|---|---|---|---|
| IMA-EmbedRow-ridge | W_e pairs | — | **96.88 %** | **✗ regression to near-plain at d=4096** |
| ISA HS | attn_norm-23 | ~10 % | ~10 % | ✓ |
| HumanEval pass@1 | n=20 | — | 8/20 (40 %) | Best accuracy |

Interpretation: scaling to 4B materially improves paired-data resilience (99 → 57 %). But 8B's regression to 97 % indicates that at d=4096, ridge attacker finds a different qualitative solution. **Dimensional headroom argument incomplete.**

Lesson: static linear obfuscation operational limit reached at Qwen3-1.7B/4B under paired-data threat model. Further progress requires (a) higher-d model (8B still open after diagnosis), (b) dynamic masking (fresh M_q per request — breaks pair accumulation), or (c) hide W̃ (access control).

### Paper-literal Algorithm 2 (2026-05-26)

Config: `keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-uvo-PAPERLIT-bf16-native.gguf` — two deviations vs our deployed Alg2 corrected:

- **A1:** `k_matrix = Ĥ⁻¹ · Ẑᵀ` (paper) vs `R̂_qk · Ĥ⁻¹ · Ẑ` (ours).
- **A2:** `Û_vo = raw N(0, 1/d_head)` (paper) vs `QR-stabilised + 0.05 σ perturbation` (ours).

Qwen3-4B. Measured ridge on two surfaces paper claims defended.

**`kq` surface (paper: "0.0 % ObfuscatedScore", Table 4):**

| Layer | Plain (row) | Plain (vocab) | Obf (row) | Obf (vocab) | Defense delta |
|---|---:|---:|---:|---:|---|
| 0 | 48.63 % | 0.43 % | 47.22 % | 0.07 % | ~0.4 pp |
| 5 | 38.69 % | 0.08 % | 38.49 % | 0.04 % | ~0.04 pp |
| 11 | 27.73 % | 0.02 % | 26.95 % | 0.00 % | ~0.02 pp |
| 17 | 22.41 % | 0.00 % | 21.17 % | 0.00 % | ~0 pp |
| 23 | 30.13 % | 0.00 % | 29.67 % | 0.01 % | ~0 pp |

Paper-literal lowers obf kq at L=5 from 38.49 % (default Alg2) → **7.79 %**. **But vocab-disjoint methodology floors both plain and obf to ~0 % beyond layer 0**, so the gain is invisible in paper's reporting methodology.

**`kqv_out` surface (attention output, pre-W_o; §5.4 "bounded"):**

| Layer | Plain (row) | Plain (vocab) | Obf (row, default) | Defense delta (default) | Obf (paper-literal) | Defense delta (paper-literal) |
|---|---:|---:|---:|---|---:|---|
| 0 | 97.46 % | 71.68 % | 83.77 % | +13.1 pp | 47.32 % | **+50 pp** |
| 5 | 50.77 % | 0.76 % | 50.22 % | −0.5 pp | 10.49 % | **+40 pp** |
| 11 | 18.50 % | 0.01 % | 19.83 % | +1.3 pp | 10.57 % | similar |
| 17 | 16.67 % | 0.00 % | 17.75 % | +1.1 pp | 10.04 % | similar |
| 23 | 25.27 % | 0.09 % | 25.99 % | −0.7 pp | 10.04 % | similar |

**Paper-literal delivers 40 pp defense gain at layer 5** on the bounded output surface vs our deployed Alg2. Combined finding: Paper Table 4's "87.14 % Noise+KeyMat → 0.0 % + Alg2" is explained as the compound of (a) paper-literal construction A1+A2 lowering obf TTRSR to single digits at L≥5, PLUS (b) vocab-disjoint split flooring ridge effectiveness beyond L=0. The 87 % baseline remains unmeasured (likely requires paper's gradient-opt ISA attack, Step 3, or a different cell).

---

## Per-attack-class current state

### IMA-EmbedRow-ridge (paired-data attack)

- **Best measured TTRSR:** 57.03 % (Qwen3-4B, paper-literal Alg2). Plain 98.02 %. Defense delta **−41 pp**, still 42 pp short of gate.
- **Defense applies:** paired-data threat model only. If W̃ private or weights not publicly available, attack collapses.
- **Open gap:** Qwen3-8B reversion to 96.88 % indicates ridge attacker finds undamped solution at d=4096. Root cause unknown.
- **Status:** **In-scope but structurally limited by paired-data assumption.**

### IMA-L0-activation (layer-0 hidden state)

- **Best measured TTRSR:** ~25 % (Qwen3-4B, paper-literal). Defense delta ~0 (pre-W_q unreachable).
- **Defense applies:** None from Alg2 (structural unreachability). Mitigations: larger keymat, heavier α-noise (breaks generation at d=2048), non-linear keymat fold (paper-divergent).
- **Status:** **Structurally undefended by Alg2.** Gate failure; choices are (1) accept + document as known-weakness, (2) dynamic masking, (3) scale (incomplete).

### IMA @ Qcur_normed-0 (post-W_q, post-Q-norm L=0)

- **Best measured TTRSR:** 38.71 % (default Alg2, Qwen3-1.7B, α_e=0.1). Alg2 paradoxically **+3 pp worse**.
- **Mechanism:** matrix-Γ design preserves Q_plain_normed · M_q (orthogonal rotation), which a paired-data ridge inverter learns in closed form.
- **Status:** **Partially defended.** 2.6× gate; needs next-session paper-literal re-measurement to see if construction A1 changes this.

### ISA-HiddenState (layer-23 residual stream)

- **Best measured TTRSR:** 11.5 % (Qwen3-1.7B, full Alg2). **✓ Passes 15 % gate.**
- **At Qwen3-4B paper-literal:** ~10 %.
- **Defense:** Ẑ_block (fixed-window permutation) + head-shuffle Π_head.
- **Status:** **Defended.** Robust across 1.7B → 4B scaling.

### ISA-AttnScore (`kq` surface)

- **Paper claim (Table 4):** 0.0 % under full Alg2.
- **Our row-split:** 0.0 % at layer-23 under partial Alg2 (head-shuffle alone sufficient).
- **Our vocab-disjoint:** plain floors at ~0 % at L≥5; defense delta unmeasurable (methodology floor).
- **Theorem (pending formal write-up):** under vocab-disjoint split, ridge attacker on fixed-feature surfaces can only learn the identity-like baseline. The "0 %" is a measurement floor, not a defense victory.
- **Status:** **Defended (but methodology floor governs claimed defense).**

### Score-surface `kqv_out` (§5.4 "bounded")

- **Paper claim:** 0.82 % → 0.0 % under full Alg2.
- **Our paper-literal measurement (Qwen3-4B, row-split):** 50.22 % (default) → 10.49 % (paper-literal) at L=5. **40 pp gain.**
- **Paper-literal at vocab-disjoint:** ~0.54 % at L=5, ~0 % at L≥11 (methodology floor again).
- **Status:** **Defended under paper-literal Alg2.** Gate passes comfortably. Per-component (A1 vs A2) attribution pending.

### TFMA / SDA (wire-side)

- **TFMA:** 0.78 % top-1 / 2.73 % top-10 — defended by τ token permutation.
- **SDA:** BLEU-4 = 1.4×10⁻⁵ — defended by τ permutation.
- **Status:** **Defended.** Robust across all measured model scales.

### Eigendecomposition leak (Γ_q similarity transform, unresolved)

- **Surface:** Matrix-Γ kernel writes `Γ_q = Mᵀ · Diag(γ_q) · M` to GGUF (similarity transform of diagonal).
- **Attack:** `numpy.linalg.eig` recovers both eigenvalues γ_q and eigenvectors M (up to permutation) in milliseconds.
- **If confirmed empirically:** downstream IMA on Q-normed would drop ~38 % → ~98 %, defeating intra-head obfuscation entirely.
- **Status:** **Analytically unambiguous leak; empirical measurement pending** (~5 min of numpy + downstream IMA re-run). Flagged in threat-model doc but not yet quantified.

---

## Paper-literal vs deployed-default decision frame

| Construction | Our default | Paper-literal | Defense gain (L=5 kqv_out) | Accuracy impact |
|---|---|---|---|---|
| k_matrix | R̂_qk · Ĥ⁻¹ · Ẑ | Ĥ⁻¹ · Ẑᵀ | +40 pp (50.22 → 10.49 %) | unknown (untested) |
| Û_vo | QR-stabilised + 0.05 σ | raw N(0, 1/d_head) | (bundled above) | unknown; risks bf16 denormals |

**Next-session work:** separate A1 and A2 effects; validate accuracy on paper-literal cell; decide whether to re-deploy paper-literal as the canonical variant.

---

## Paired-data defenses

Three mitigation strategies, increasing engineering cost:

### 2A. Hide the obfuscated model

Attacker loses W̃ access; paired-data assumption breaks. Activation-side attacks require server memory access; if server in TEE, collapses to wire-side only. Zero algorithm change; breaks "public LMaaS infrastructure" thesis.

### 2B. Obfuscate inside a TEE (insufficient alone)

Keeps keymat / τ inside attestation boundary; doesn't prevent paired-data attack. Only helps combined with 2A.

### 2C. Dynamic masking (fresh M_q per request, GELO-style)

**The structural fix.** Attacker sees different W̃ on every forward; cross-batch pair collection fails. Breaks "no infrastructure change" but matches project's research direction (GELO + TwinShield).

Architectural questions:
- **Mask source:** client-resident (stateless server) vs TEE-resident (requires signing stream).
- **What gets masked:** W_q/W_k alone vs W_e + W_q/W_k.
- **Inference cost:** GELO HD₃ reports −28 % TTFT; AloePri-side will differ (mask applies to weights not activations).
- **Stock llama.cpp compatibility:** almost certainly no; fused-mask-multiply kernel required.

Gram-leakage toolkit ready: four attack drivers (`run_anchor_ica.py`, `run_jade.py`, `run_jd.py`, `run_gram_error.py`) wired per path-1 round 2/3 research.

---

## Defended surfaces summary

### Unambiguously defended (gate cleared)

| Attack | Surface | Config | TTRSR obf | Mechanism |
|---|---|---|---|---|
| NN | attn_norm-0 | any Alg2 | 0.0 % | rotation (Π_head) |
| ISA-AttnScore | kq-L | Alg2 + Π_head | ≤ 0.0 % (row) / ~0 % (vocab) | head-shuffle + vocab-disjoint floor |
| ISA-HS | attn_norm-L≥5 | Alg2 + Ẑ_block | ~9–11 % | Ẑ_block |
| TFMA | token stream | τ permutation | 0.78 % | τ masks vocab correspondence |
| SDA | token stream | τ permutation | 1.4×10⁻⁵ BLEU | τ masks token frequency |

### Partially defended (gate failure with identified mitigation)

| Attack | Config | TTRSR obf | Gap | Mitigation |
|---|---|---|---|---|
| IMA-EmbedRow-ridge | paper-literal Alg2 | 57.03 % (4B) / 96.88 % (8B) | 42 / 82 pp | dynamic masking / W̃ access control |
| IMA-L0-activation | paper-literal Alg2 | ~25 % | 10 pp | larger keymat / α / non-linear fold |
| IMA @ Qcur_normed-0 | default Alg2 | 38.71 % | 24 pp | dynamic masking |

### Pending measurement

- Eigendecomposition (Γ_q similarity eig) — analytically unambiguous leak; empirical number pending.
- Paper-literal × vocab-disjoint compound — pending (~20 min).
- Per-component A1 vs A2 attribution — deferred.

---

## Operational leak budget (static deployment)

Assuming public GGUF (W̃ visible), honest-but-curious server (activation access), all 15 % gates:

**Qwen3-4B, paper-literal Alg2:**
- L=0: IMA-EmbedRow-ridge 57 %, IMA-L0-activation 25 %, IMA@Qcur 47 % — all fail individually
- L≥5: ISA-HS 10 %, ISA-AttnScore ~0 %, TFMA 0.8 %, SDA <1e-4 BLEU — all pass

**Leak budget sits at L=0 activation surface** (25–57 %). Deep layers (L≥5) cleared. An attacker with server memory access can recover layer-0 token embeddings or token IDs from attn_norm-0 captures with 25–57 % TTRSR. Mitigation: (a) dynamic masking, (b) TEE isolation of memory, (c) explicit threat-model boundary ("activation privacy only at L≥5").

---

## Open follow-ups

1. **Diagnose Qwen3-8B ridge regression** (IMA-EmbedRow 96.88 % vs 4B's 57 %). SVD analysis of `W_e_obf @ W_e_plain.pinv()` across model sizes + candidate-pool sensitivity sweep.
2. **Measure eigendecomposition leak empirically.** ~5 min work; determines whether static-obfuscation envelope collapses this surface.
3. **Paper-literal accuracy validation.** HumanEval + coherence-check on paper-literal Alg2 cell.
4. **Per-component attribution (A1 vs A2).** Build cells with only one deviation; settle whether to recommend paper-literal or hybrid.
5. **Compound paper-literal × vocab-disjoint ridge.** Closes the Table 4 reproducibility story.
6. **Direction 2C dynamic-mask design sketch.** Prerequisite for Gram-leakage measurement.

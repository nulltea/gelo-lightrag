# Alg2 parameters × attacks — cross-interaction reference

**Date opened:** 2026-05-25
**Status:** Living document. Update inline whenever a new measurement or
theoretical clarification lands. Each row of the cross-map has a status
tag (**M** measured, **T** theory only, **C** code ready but not run).
**Branch at open:** `path-2-aloepri-gemma`

## TL;DR (2026-05-25, post real-data attack runs)

- **Algorithm 2** in path-2 has 5 runtime-touching components: `R̂_qk`,
  `Ĥ_qk`, `Ẑ_block`, `Π_head` (= `τ_kv` + `τ_group`), `Û_vo`.
  Definitions follow paper §5.2.3 with one deliberate path-2 deviation
  on the K-side formula (`lib/alg2.py:241-262`).
- The path-2 attack ledger has 12 attacks across four families.
  **Empirically validated Alg2 defences (2026-05-25 real-data runs on
  the deployed Q3-4B Û_vo cell):**
  - **Per-head fingerprint:** Q/K/V/O all at ≈ random chance (Π_head
    defends near-completely).
  - **V/O channel-pair:** V & O & pair all at random chance (Π_head +
    Û_vo combined defend near-completely).
  - **ISA-attn-score:** at noise floor on *both* plain and obf at L ∈
    {0, 5, 11, 17, 23} → attack class itself is too weak to discriminate
    Alg2 defence on this surface; not Alg2's fault, attack's fault.
- **Empirically attributed but not isolated:** the aggregate ISA
  HiddenState delta on the 4B Û_vo cell (1.70 pp) and 8B (0.73 pp) is
  the only per-component-untested Alg2 measurement on the runtime side.
  Theoretical attribution: ~70 % Ẑ_block, ~30 % bf16-Û_vo, ~0 % the rest.
- Among the 5 Alg2 components, only **Ẑ_block (β > block_size)** and
  **Π_head** carry *significant* defence on the current ledger.
  R̂_qk, Ĥ_qk ±1, and Û_vo do real work specifically against the
  per-head + V/O attacks (validated 2026-05-25 — they shadow under
  Π_head though, so individual contribution can't be isolated from a
  single deployed cell run).
- The "Algorithm 2 amplifies K_a × K_d 3-5×" framing in
  `aloepri-keymat-variance.md` and `2026-05-22-keymat-defense-optimization.md`
  conflates §5.2.2 (noise + Π) with §5.2.3 (Algorithm 2) and with
  bf16/Q8_0 quantization. **Framing corrections landed 2026-05-25** in
  both docs.
- **Code bugs fixed 2026-05-25:** Bug #2 (legacy `LayerAlg2Keys`
  TypeError), Bug #3 (stale `python/path-2` import), Bug #1 (misleading
  matrix-Γ comment), Bug #6 (doc conflation). See [Bugs](#bugs).

## Companion artifacts (don't duplicate)

- `python/aloepri-llm/lib/alg2.py` — implementation; comment block at lines
  220-256 documents the M_q/M_k cancellation choice that deviates from paper.
- `python/aloepri-llm/obfuscate_qwen3_gguf.py:468-546` — Alg2 wiring + the
  legacy-path bug in the `LayerAlg2Keys` reconstruction (see [Bugs](#bugs)).
- `python/aloepri-llm/scripts/check_alg2_invariance.py` — M_q M_k^T and
  RoPE-attention-score invariance test. Has a stale `python/path-2` import
  on line 26 (rename leftover; works via `PYTHONPATH=$(pwd)`).
- `evals/aloepri-attacks/m2_7/probe_alg2_ia_invariant.py` — synthetic IA
  Attn-IA per-component probe.
- `evals/aloepri-attacks/m2_7/probe_alg2_static_attacks.py` — synthetic
  VMA + IA Gate-IA + IA Attn-IA per-component probe.
- `evals/aloepri-attacks/m2_7/run_per_head_fingerprint.py` — new static
  attack driver (code only, never run).
- `evals/aloepri-attacks/m2_7/run_vo_channel_pair.py` — new static attack
  driver (code only, never run).
- `evals/aloepri-attacks/ATTACK_FAMILIES.md` — ledger overview.
- `docs/research/aloepri-keymat-variance.md` — Alg1 variance work
  (referenced for the "Alg2 amplifies" framing that this doc corrects).
- `docs/handoffs/2026-05-21-uvo-isa-multikey-and-gpu-keymat-bug.md` and
  `docs/handoffs/2026-05-22-keymat-defense-optimization.md` — prior
  context on the Û_vo patch and the K_a×K_d sweep.
- `2603.01499v2.pdf` §5.2.3 + Algorithm 2 + Table 4 — paper reference.

## Algorithm 2 parameter / property inventory

| Property | CLI flag | Default | Code site | Active in deployed cell `cell-qwen3-4b-uvo-20260521`? |
|---|---|---|---|---|
| **Module enable** | `--alg2` | off | `obfuscate_qwen3_gguf.py:830` | **on** |
| `alg2_seed` (base, per-layer +1000·il) | `--alg2-seed` | 987654321 | `obfuscate_qwen3_gguf.py:519` | **public seed** (in script source) — flagged in [Gaps](#gaps) |
| **R̂_qk** per-RoPE-pair 2D rotation | gated by `--alg2-qk-norm-matrix` | off → I | `alg2.py:generate_r_qk` (36-58) | **on** (matrix-Γ on) |
| **Ĥ_qk Walsh-Hadamard ±1** | `--alg2-h-hadamard-signs` | off | `alg2.py:generate_h_qk` (61-95) | **on** |
| **Ĥ_qk uniform scale range** | `--alg2-qk-scale-min/max` | (0.95, 1.05) | same | forced to (1.0, 1.0) by matrix-Γ + no-hadamard branch |
| **Ẑ_block β-windowed permutation** | `--alg2-beta` | 8 | `alg2.py:generate_block_perm` (98-140) | **β = 8** |
| `alg2-gamma` (legacy softmax param) | `--alg2-gamma` | 1e3 | (accepted, **ignored** since 2026-05-19 fix) | dead CLI cruft |
| `rope_base` (legacy softmax param) | (none; read from GGUF) | 1e6 | (ignored) | dead |
| **Π_head = (τ_kv, τ_group)** | implicit when `--alg2` on | n_kv > 1, num_groups > 1 | `alg2.py:generate_head_perm` (143-155) | **on** |
| **Û_vo (V↔O random projection)** | `--alg2-u-vo` | off | `alg2.py:generate_u_vo` (163-199) | **on** |
| **Matrix-Γ kernel mode** | `--alg2-qk-norm-matrix` | off | replaces `attn_q_norm.weight` 1D γ with 2D Γ = MᵀDM in `obfuscate_qwen3_gguf.py:615-636` | **on** |
| `--alg2-h-hadamard-signs` (forces M_q == M_k) | as above | off | when on, k_matrix = q_matrix bit-identical | **on** |

## Path-2-implemented attack inventory

| Attack | Family | Surface | Driver | Last measured on a current cell? |
|---|---|---|---|---|
| **VMA** (Vocabulary-Matching) | static-weight | W̃_e + projections through W̃_q/k/v/o + W̃_gate/up/down + W̃_output | `run_static_attacks.py:run_vma` | not since pre-2026-05-20 Π fix |
| **IA Gate-IA** | static-weight | W̃_e + W̃_gate (invariant `Avg(e_i · W_gate^T)`) | `run_static_attacks.py:_gate_ia_invariants` | not currently |
| **IA Attn-IA** | static-weight | W̃_e + W̃_q + W̃_k (block-quadratic invariant `‖e · [W_q‖block ‖ W_k‖block]‖²`) | `run_static_attacks.py:_attn_ia_invariants` | synthetic probe only (2026-05-25) |
| **Per-head fingerprint** | static-weight | per-head SVD spectra on W̃_q/k/v/o | `run_per_head_fingerprint.py` (NEW 2026-05-25) | never |
| **V/O channel-pair** | static-weight | per-head L2 magnitudes + top-K spectra on W̃_v, W̃_o | `run_vo_channel_pair.py` (NEW 2026-05-25) | never |
| **IMA-EmbedRow-ridge** | static-weight | W̃_e rows | `run_ima_embedrow_attacks.py` | not on a current Alg2 cell |
| **IMA-EmbedRow-transformer** | static-weight | W̃_e rows (2-layer xformer inverter) | same | not on a current Alg2 cell |
| **TFMA** (token-frequency) | token-stream | obfuscated token id stream | `run_token_attacks.py:run_tfma` | not on a current Alg2 cell |
| **SDA** (sequence distribution) | token-stream | same | `run_token_attacks.py:run_sda` | not on a current Alg2 cell |
| **NN / IMA / IMA-paper-like / single-key ISA** | hidden-state | h̃_L residual | `run_hidden_state_attacks.py` | not isolated per Alg2 component |
| **ISA HiddenState multi-key (paper-faithful)** | hidden-state | h̃_L residual at L=17 with K=64 keymat-pool ridge | `run_isa_multikey.py` | aggregate Alg2 Δ measured (1.70 pp on 4B, 0.73 pp on 8B); not isolated per component |
| **ISA-attn-score** | runtime score-stream | per-head attention scores `(n_heads, n_q, n_kv)` | `run_hidden_state_attacks.py::_isa_attn_score` | **never run on any cell** |

## Cross-interaction map

Status tags:
- **M** = measured empirically (test name in []).
- **S** = measured synthetically on random tensors of deployment shape.
- **T** = theoretical only, derived from the math (no experiment).
- **C** = driver code exists but has never been run on a real cell.
- **n/a** = attack surface mathematically untouched by this component.

### Static-weight attacks

| Component | VMA | IA Gate-IA | IA Attn-IA | Per-head fpt | V/O channel-pair | IMA-EmbedRow |
|---|---|---|---|---|---|---|
| **R̂_qk** | 0.4 pp ↓ **S** [`probe_alg2_static_attacks.py`] | **0 M** [same] | 30-64 pp ↓ **S** [`probe_alg2_ia_invariant.py`] | predicted ≈0 **C** (SVD-spectrum invariant) | predicted ≈0 **C** | **0 T** (W_e untouched) |
| **Ĥ_qk ±1 Walsh-Hadamard** (deployed) | **0 S** | **0 M** | **0 S** (L2-invariant) | predicted 0 **C** | predicted 0 **C** | **0 T** |
| **Ĥ_qk non-unit (0.95-1.05)** | **0 S** | **0 M** | **0 S** (too narrow) | 0 **C** | 0 **C** | **0 T** |
| **Ĥ_qk strong (0.5-1.5)** | **0 S** | **0 M** | 8-43 pp ↓ **S** | small **C** | small **C** | **0 T** |
| **Ẑ_block β=8** (deployed) | **0 S** | **0 M** | **0 S** (block-aligned with `block_size=16`) | predicted 0 **C** | n/a | **0 T** |
| **Ẑ_block β=16** | not isolated | predicted 0 | **0 S** | predicted 0 **C** | n/a | **0 T** |
| **Ẑ_block β=64** | **0 S** | **0 M** | 98-99 pp ↓ **S** | small **C** | n/a | **0 T** |
| **Π_head** (τ_kv + τ_group) | **0 S** (sort-invariant) | **0 M** | 94-99 pp ↓ **S** | **near-complete M** (drops to ≈ random 1/n_heads on real cell, see below) | **near-complete M** (V & O both at random chance on real cell) | **0 T** |
| **Û_vo** | 1.2 pp ↓ **S** | **0 M** | **0 S** (control — doesn't touch Q/K) | predicted near-0 **C** (QR-stable + 0.05 σ) | weak V / moderate O **C** | **0 T** |
| **FULL deployed Alg2 stack** | 21 pp ↓ **S** | **0 M** | 99+ pp ↓ **S** | **near-complete M** (top-1 1.0× random across Q/K/V/O) | **near-complete M** (V t1 12.5 % = random 1/8; O t1 3.1 % = random 1/32; pair t1 3.1 %) | **0 T** |

### Runtime attacks

| Component | TFMA / SDA | NN / IMA / IMA-paper-like / single-key ISA | ISA HiddenState multi-key (paper-faithful) | ISA-attn-score |
|---|---|---|---|---|
| **R̂_qk** | n/a | **0 T** (self-cancels in QK^T at exact precision) | **0 T** | **0 T** (cancels in QK^T) |
| **Ĥ_qk ±1** | n/a | **0 T** | **0 T** | **0 T** |
| **Ẑ_block β=8** | n/a | dominant **T** (35 % rel score Δ propagates to residual via attention) | **dominant T** — 70 % of the 1.70 pp Alg2 delta on 4B per theoretical attribution; aggregate Δ measured **M** at 4B Û_vo cell, component split is theory + score-Δ **M** in `check_alg2_invariance.py` | dominant **T** |
| **Ẑ_block β=64** | n/a | larger **T** (75 % score Δ) | larger **T** | larger **T** |
| **Π_head** | n/a | **0 T** (per-head sums cancel in residual) | **0 T** | dominant **T** (head index shift; relevant only if attack uses head identity rather than head-permutation-invariant statistics) |
| **Û_vo** | n/a | ≈ 0 at fp32 **T**; ~2-3 pp at bf16 **T** (cond(Û_vo) ≈ 3.6-5.9 → numerical leak per layer) | small (~2-3 pp **T**, bf16 numerical leak only; rest of 1.70 pp aggregate attributed to Ẑ_block via theory) | small **T** |
| **FULL deployed Alg2 stack** | n/a | (aggregate observed in 4B Û_vo cell) | **1.70 pp ↓ M** on Q3-4B L=17 K=64 (vendor CPU keymat, single seed; ±5 pp pool noise); **0.73 pp ↓ M** on Q3-8B L=17 | **noise-floor M** (≤ 2 % top-1 on both plain and obf at L ∈ {0,5,11,17,23}; attack is fundamentally too weak to demonstrate Alg2 defence on this surface) |

### §5.2.2 components (not Alg2, but stated for completeness)

These are NOT part of Algorithm 2. They live in `obfuscate_qwen3_gguf.py:355-465`
under the "§5.2.2 additive Gaussian noise" and "Π token permutation" sections.

| Component | VMA | IA Gate-IA | IA Attn-IA | TFMA / SDA | ISA HiddenState | IMA-EmbedRow |
|---|---|---|---|---|---|---|
| **Π token-perm** (`--pi`) | dominant **T** | dominant **T** | partial **T** | dominant **T** | partial **T** (per-id mapping unknown to attacker) | dominant **T** |
| **α_e embed noise** (`--noise-alpha-e`) | partial **T** | partial **T** | partial **T** | n/a | partial **T** (residual perturbation via embed) | partial **T** |
| **α_h head noise** (`--noise-alpha-h`) | partial **T** | n/a | n/a | n/a | partial **T** | partial **T** |
| **Π special-token exclusion** (`--pi-include-specials=False`) | leak ~293 fixed pairs **T** | same | same | same | same | same |

`docs/handoffs/2026-05-20-aloepri-pi-special-token-fix.md` has the §5.2.2 cell
metrics post-fix (HumanEval pass@1, etc).

## Test inventory & methodology

| Test | File | What it generates | What it measures | Limitations |
|---|---|---|---|---|
| `check_alg2_invariance.py` | `python/aloepri-llm/scripts/` | M_q M_k^T = I and post-RoPE attention score | Score-Δ vs β | 1 head, 8 tokens; doesn't propagate through full attention |
| β-sweep one-liner | inline in this session | Score Δ across β ∈ {1, 2, 4, 8, 16, 64} | β-vs-score-Δ curve: 3e-8 / 9 % / 33 % / **35 %** / 54 % / 75 % | random Q, K (uniform Gaussian); real Qwen3 weight statistics may differ |
| `probe_alg2_ia_invariant.py` | `evals/aloepri-attacks/m2_7/` | random W_e/q/k at Q3-4B deployment shape | IA Attn-IA top-1/top-10 per Alg2 component | synthetic, no Alg1, no §5.2.2 |
| `probe_alg2_static_attacks.py` | `evals/aloepri-attacks/m2_7/` | random ModelWeights at moderate Q3-shape | VMA + IA Gate-IA + IA Attn-IA top-1/top-10 per Alg2 component | synthetic; only attn projections affected; FFN/output stay plain |
| 4B Û_vo cell aggregate | `cell-qwen3-4b-uvo-20260521/` capture + `run_isa_multikey.py` | real obfuscated forward-pass captures at L=17 | aggregate ISA HiddenState TTRSR: 3.41 % top-1 (Û_vo) vs 5.11 % (no Û_vo) = 1.70 pp Alg2 delta | single attacker_seed; ±5 pp K=64 pool noise per `aloepri-keymat-variance.md`. Per-component split is theoretical, not empirical |
| 8B Û_vo cell aggregate | `cell-qwen3-8b-uvo-20260521/` | as above at L=17 | 9.00 % top-1 (Û_vo) vs 9.73 % (no Û_vo) = 0.73 pp Alg2 delta | same caveats |
| **Real-data per-head fingerprint** (2026-05-25) | `cell-qwen3-4b-uvo-20260525` (just-built deployed config); run `run_per_head_fingerprint.py` | top-1 hit rate per Q/K/V/O on real GGUF pair | attn_q t1=4.25 % (random 3.13 % at n=32); attn_k t1=13.54 % (random 12.5 % at n=8); attn_v t1=12.50 % (= random); attn_o t1=3.21 % (≈ random). **Π_head defends to random chance.** | single seed; SVD spectrum can't probe R̂_qk/Hadamard/Û_vo separately (right-mult invariant) |
| **Real-data V/O channel-pair** (2026-05-25) | same cell; run `run_vo_channel_pair.py` | per-head V row L2 + O column L2 magnitude matching | vo_v t1=12.50 % (= random 1/8); vo_o t1=3.12 % (= random 1/32); vo_pair t1=3.12 %. **Π_head + Û_vo combined defend V/O channel-pair to random chance.** | combined effect; can't isolate Û_vo's contribution beyond Π_head from this measurement alone |
| **Real-data ISA-attn-score sweep** (2026-05-25, 64-prompt) | both `cell-qwen3-4b-plain-attn-multilayer-20260525` and `cell-qwen3-4b-uvo-attn-multilayer-20260525`; layers {0, 5, 11, 17, 23} | per-layer ISA-attn-score on captured kq tensors | All layers ≤ 2 % top-1 on **both** plain and obf | **RETRACTED 2026-05-25** — outlier-low seed sample. Row-split + 10-seed mean: plain = 11.87 % ± 3.4, obf = 11.87 % ± 3.4 → ZERO Alg2 defense. **Original "noise floor" framing further rebased 2026-05-26** — 11.87 % itself was a 64-prompt sample artefact (34 % test/train vocab overlap capped memorisation). See [512-prompt re-capture](#isa-attnscore-512-prompt-re-capture-2026-05-26) row. |
| **Real-data ISA-attn-score sweep** (2026-05-26, 512-prompt, GPU) | `cell-qwen3-4b-{plain,uvo}-attn-512-20260526/captures`; layers {0, 5, 11, 17, 23}; 75 % test/train vocab overlap; GPU ridge via `aloepri-ima-trainer:latest` | per-layer ISA-attn-score, 10-seed row-split mean | **L=0:** plain 48.63 ± 0.80 / obf 47.22 ± 0.84 (Δ +1.41 pp, ~1.2σ). **L=17:** plain 22.41 / obf 21.17. K_a-invariance empirically confirmed at every layer (max Δ 1.41 pp). | Per-head ridge @ L=17 caps at 11.66 % plain / 11.20 % obf — cross-head flattening load-bearing. **Defense delta: ~0 pp at every layer** despite full deployed obfuscation. See `evals/aloepri-attacks/m2_7/diagnose_isa/gpu_sweep.py`. |

## Alg1-only Q3-4B attack matrix (2026-05-25)

**Cell:** `untied-keymat-h128-alg1-only-bf16.gguf` — Algorithm 1 keymat only.
No `--pi`, no `--noise-*`, no `--alg2`. Built in 47 s; 9.1 GB.
Captures: `cell-qwen3-4b-alg1only-attn-20260525/captures/{hidden,attn}.safetensors`
(64 prompts, n_tokens ≈ 10-20, layers {0, 17}, kinds {attn_norm, kq}).

Purpose: isolate Algorithm 1's contribution to every §08 attack so we can
measure what §5.2.2 (noise + Π) and Algorithm 2 add on top.

### Static-weight attacks

| Attack | Setting | Plain ceiling | Alg1-only obf | Δ |
|---|---|---:|---:|---|
| VMA | identity-τ | 98.4 % | **98.4 %** | 0 pp |
| IA Gate-IA | identity-τ | 98.5 % | **0.00 %** | −98.5 pp |
| IA Attn-IA | identity-τ | (same) | **0.02 %** | −98.5 pp |
| Per-head fingerprint Q | identity perm | 100 % (trivial) | **7.6 %** (≈ random 6.25 %, 1/n_q_heads_actual) | −92 pp |
| Per-head fingerprint K | identity perm | 100 % | **13.5 %** (= random 12.5 %, 1/n_kv) | −86 pp |
| Per-head fingerprint V | identity perm | 100 % | **13.2 %** (= random) | −87 pp |
| Per-head fingerprint O | identity perm | 100 % | **6.8 %** (≈ random) | −93 pp |
| V/O channel-pair V | identity perm | (similar) | **13.2 %** | −87 pp |
| V/O channel-pair O | identity perm | | **6.8 %** | −93 pp |
| V/O channel-pair (V,O) joint | identity perm | | **1.9 %** | random² match |
| IMA-EmbedRow-ridge multi-key K=64 | identity-τ | 98.4 % | **99.2 %** | +0.8 pp |

### Runtime attacks (n=64 prompts, row-split)

| Attack | Layer | Plain ceiling | Alg1-only obf | Δ |
|---|---|---:|---:|---|
| NN | attn_norm-0 | 100.0 % | **0.0 %** | −100 pp |
| IMA basic ridge single-key | attn_norm-0 | 92.7 % | **90.5 %** | −2.2 pp |
| ISA HS single-key ridge | attn_norm-17 (depth 47.2 %) | 12.5 % | 26.6 % (row-split memorization artifact) | inconclusive |
| ISA HS multi-key K=64 (paper-faithful) | attn_norm-17 | 15.0 % (identity-τ) | **11.6 %** | −3.4 pp (within ±5 pp K=64 pool noise) |
| ISA-AttnScore (row-split, 10-seed mean) | kq-17 (64-prompt, rebased — see 512-prompt row) | 11.9 % ± 3.4 | **13.6 % ± 5.0** | +1.7 pp (within noise) |
| ISA-AttnScore (row-split, 10-seed mean, **512-prompt 2026-05-26**) | kq-0 | 48.6 % ± 0.8 | (Alg1-only not re-measured at 512) | n/a |
| ISA-AttnScore (row-split, 10-seed mean, **512-prompt 2026-05-26**) | kq-17 | 22.4 % ± 0.8 | (Alg1-only not re-measured at 512) | n/a |

### Reading

| Outcome | Attacks |
|---|---|
| **Alg1 alone defeats** (rectangular d→d_obs keymat is enough) | IA Gate-IA, IA Attn-IA, NN, Per-head fingerprint (Q/K/V/O), V/O channel-pair (V, O, pair) |
| **Alg1 alone fails** (defense comes from §5.2.2 + Alg2 later) | VMA (98 %), IMA basic L=0 (90.5 %), IMA-EmbedRow-ridge multi-key (99 %), ISA-AttnScore (no defense), partial: ISA HS multi-key (−3.4 pp, within noise) |
| **No attack can defeat** (structural) | ISA-AttnScore — K_a-invariant surface by construction |

### Implications for the §08 row attributions

- **IA Gate-IA + Attn-IA "Pass"** — already passes at Alg1 alone. §5.2.2 + Alg2 add NO measurable defense on these two.
- **NN "Pass"** — already passes at Alg1 alone.
- **Per-head fingerprint** + **V/O channel-pair** — Alg1 already defeats them. Π_head (Alg2) and Û_vo's contribution we measured yesterday was over-credited; Alg1's keymat alone collapsed them to random chance.
- **VMA + IMA-EmbedRow-***. — Alg1 alone fails completely. Defense source is α_e=1.0 + Π (paper §5.2.2), NOT Algorithm 2.
- **ISA HS multi-key** — Alg1 alone gives ~3.4 pp attenuation (noise-bounded). The deployed cell's ~5 pp delta means Alg2 + §5.2.2 combined contribute the remaining ~2 pp on top. Per crossmap-doc theoretical attribution, that 2 pp is mostly Ẑ_block + bf16-Û_vo numerical leak.
- **ISA-AttnScore** — confirmed zero defense at any layer; surface is K_a-invariant.

## Alg1 + minimal-optimal-Alg2 Q3-4B attack matrix (2026-05-25)

**Cell:** `untied-keymat-h128-alg2min-zblock-bf16.gguf` — Algorithm 1 keymat **plus** the minimal Algorithm 2 configuration our crossmap analysis identified as carrying measurable defense:
- `--alg2 --alg2-qk-norm-matrix --alg2-h-hadamard-signs --alg2-beta 8`
- enables: R̂_qk (head-dim rotation), Ĥ_qk ±1 Walsh-Hadamard, Ẑ_block β=8, Π_head (τ_kv + τ_group)
- **excludes:** Û_vo (zero measurable contribution at fp32), §5.2.2 noise + Π token-perm
- Built in 58 s; 9.1 GB. Captures: `cell-qwen3-4b-alg2min-attn-20260525/captures/`.

### Side-by-side delta (vs Alg1-only baseline)

| Attack | Plain | Alg1-only | Alg1 + minAlg2 | Δ from Alg1 |
|---|---:|---:|---:|---:|
| **VMA** (sorted-quantile RowSort) | 98.4 % (id-τ) | 98.4 % | **96.5 %** | −1.9 pp |
| **IA Gate-IA** | (id-τ) | 0.00 % | 0.00 % | 0 |
| **IA Attn-IA** | | 0.02 % | 0.00 % | within noise |
| **Per-head fpt Q** (random 1/n_q) | id 100 % | 7.6 % | **4.25 %** | −3.4 pp (both ≈ random) |
| **Per-head fpt K** (random 1/n_kv) | id 100 % | 13.5 % | 13.5 % | 0 |
| **Per-head fpt V** | id 100 % | 13.2 % | 13.2 % | 0 |
| **Per-head fpt O** | id 100 % | 6.8 % | **3.21 %** | −3.6 pp (both ≈ random) |
| **V/O ch-pair V** | id 100 % | 13.2 % | 13.2 % | 0 |
| **V/O ch-pair O** | id 100 % | 6.8 % | **3.12 %** | −3.6 pp (both ≈ random) |
| **V/O ch-pair (V,O) joint** | id 100 % | 1.9 % | 3.12 % | +1.2 pp (within random² noise) |
| **IMA-EmbedRow-ridge multi-key K=64** | 98.4 % (id-τ) | 99.2 % | **99.2 %** | 0 |
| **NN** @ attn_norm-0 | 100.0 % | 0.0 % | 0.0 % | 0 |
| **IMA basic ridge** @ attn_norm-0 | 92.7 % | 90.5 % | 90.5 % | 0 |
| **ISA HS single-key** @ attn_norm-17 | 12.5 % | 26.6 % (row-split artefact) | 31.3 % | (artefact; ignore) |
| **ISA HS multi-key K=64** @ attn_norm-17 | 15.0 % (id-τ) | 11.6 % | **12.8 %** | +1.2 pp (within ±5 pp pool noise) |
| **ISA-AttnScore** (row-split, 10-seed mean) @ kq-17 (64-prompt — rebased) | 11.9 % ± 3.4 | 13.6 % ± 5.0 | **13.0 % ± 3.3** | within noise |
| **ISA-AttnScore** (row-split, 10-seed mean, **512-prompt 2026-05-26**) @ kq-0 | 48.6 % ± 0.8 | (not re-measured) | **47.2 % ± 0.8** | within noise; K_a-invariance |
| **ISA-AttnScore** (row-split, 10-seed mean, **512-prompt 2026-05-26**) @ kq-17 | 22.4 % ± 0.8 | (not re-measured) | **21.2 % ± 0.9** | within noise; K_a-invariance |

### Headline (2026-05-25, post-Alg1+minAlg2 measurement)

**The minimal-optimal Alg2 configuration (R̂_qk + Hadamard + Ẑ_block β=8 + Π_head) adds essentially ZERO measurable defense on top of Alg1 alone across the entire path-2 attack ledger.**

Every cell deltas to within ±5 pp noise of Alg1-only. The largest measurable shift is VMA dropping from 98.4 → 96.5 % (still trivially recoverable). The crossmap-doc theoretical attribution that Ẑ_block carries ~70 % of the Alg2 ISA HiddenState delta (~1.7 pp on the deployed cell) **fails to reproduce** when measured in isolation — Alg1's keymat is what carries the heavy lifting, and Ẑ_block's marginal contribution is within pool-noise of zero.

**Reconciliation with the earlier 1.70 pp aggregate Alg2 delta** (deployed cell vs no-Û_vo on 4B Û_vo cell): that delta was a single-seed measurement. Given ±5 pp K=64 pool noise per `aloepri-keymat-variance.md`, the 1.70 pp number was almost certainly noise-dominated.

### What carries the defense on each attack (revised conclusions)

| Attack | Defense source |
|---|---|
| IA Gate-IA, IA Attn-IA, NN, Per-head fpt (Q/K/V/O), V/O channel-pair | **Alg1 alone** (rectangular keymat d→d_obs is enough) |
| VMA, IMA basic ridge L=0, IMA-EmbedRow-ridge multi-key | **§5.2.2 alone** (α_e noise + Π token-perm); Alg1 doesn't defend, Alg2 doesn't defend |
| ISA HS multi-key K=64 | **Alg1 + §5.2.2** combined; Alg2 adds nothing measurable |
| ISA-AttnScore | **No defense from any component** — K_a-invariant surface by construction |
| QK-norm Γ eigendecomposition | **Not defended by design** (Alg2-specific surface; structural break) |
| TFMA / SDA | **Π alone** (§5.2.2 token-perm); orthogonal to Alg1/Alg2 |

**Initial conclusion (later revised, see below):** Algorithm 2 in path-2 carries no measurable defensive contribution on the current attack ledger.

## VMA attribution diagnose — §5.2.2 × Alg2 interaction (2026-05-25)

Triggered by the user noting: "previous measured VMA figures were 18.4 % on Q3-4B; current obfuscation shows no defense (98 %). What changed?"

### Stacked-component VMA on Q3-4B (3-seed mean)

| Stack | VMA top-1 mean | std | Δ from previous |
|---|---:|---:|---:|
| Plain (identity-τ control) | 98.4 % | — | — |
| Alg1 only (keymat) | 98.4 % | — | **0 pp** — Alg1 alone does nothing for VMA |
| Alg1 + minAlg2 (R̂_qk + Hadamard + Ẑ_block β=8 + Π_head, no Û_vo) | 96.5 % | — | −1.9 pp — Alg2 alone barely moves it |
| **Alg1 + §5.2.2** (α_e=1.0 + α_h=0.2 + Π token-perm, no Alg2) | **35.4 %** | 2.9 | **−63 pp** — §5.2.2 is load-bearing |
| **Alg1 + §5.2.2 + full Alg2** (deployed Û_vo config) | **9.5 %** | 2.5 | **−26 pp interaction** — Alg2 amplifies §5.2.2 only when stacked |

### Reconciliation with §08's 18.4 % cell

§08 quotes 18.4 % single-seed. My 3-seed mean is 9.5 % ± 2.5. 18.4 % is ~3 σ above the mean — outlier-high but plausible within seed noise. Re-measuring the §08 cell at 5+ seeds would close the gap.

### Revised conclusion — Algorithm 2's value is in the §5.2.2 × Alg2 interaction

The "Alg2 contributes ~0 measurable defense" conclusion (from the Alg1 → Alg1+minAlg2 comparison earlier) was **incomplete**. Alg2's marginal contribution to **VMA alone is −1.9 pp** (~zero), but its **interaction with §5.2.2** is −26 pp:

- §5.2.2 alone: −63 pp (Alg1 → Alg1+§5.2.2)
- Alg2 alone: −2 pp (Alg1 → Alg1+minAlg2)
- §5.2.2 + Alg2 stacked: −89 pp (Alg1 → deployed)
- Interaction term: −89 − (−63 + −2) = **−24 pp** (matches the 26 pp gap)

So Algorithm 2 is **synergistic** with §5.2.2 on VMA: roughly tripling §5.2.2's defense (98 → 35 vs 98 → 9.5). Without §5.2.2 as substrate, Alg2 carries no value on VMA. With §5.2.2 present, Alg2 contributes a real 26 pp.

### Likely mechanism (theoretical, not yet isolated)

- **§5.2.2 alone:** α_e noise scrambles W_e ROW values per-id; Π re-shuffles ROW indices. VMA's sorted-quantile features per row become poorly-matched to plain features → recovery drops 98 → 35.
- **Alg2 alone:** R̂_qk + Hadamard + Ẑ_block + Π_head reshape attention-weight COLUMNS per-layer. VMA's per-row sorted features are sort-invariant under column permutation → recovery ~98 % unchanged.
- **§5.2.2 + Alg2 stacked:** §5.2.2's per-row scrambling is amplified across VMA's 17 layer × projection-kind sources because Alg2's per-layer column re-indexing prevents the multi-source voting from reinforcing a stable per-row signature → recovery drops 35 → 9.5.

### Within-Alg2 bisection (2026-05-26) — which Alg2 component carries the 25.9 pp?

Built 4 isolation cells on Alg1+§5.2.2 substrate. Ran VMA 3-seed each.

| Cell | Configuration | VMA mean | std | Δ from substrate (35.4 %) |
|---|---|---:|---:|---:|
| A | Alg1 + §5.2.2 (no Alg2) | 35.42 % | 2.88 | — |
| B | + `--alg2` (Π_head only) | **35.42 %** | 2.88 | **0 pp** |
| C | + `--alg2 --alg2-qk-norm-matrix --alg2-h-hadamard-signs` (matrix-Γ: R̂+H±1+Ẑ_β8+Π_head, no Û_vo) | 17.58 % | 0.64 | **−17.8 pp** |
| F | C with `--alg2-beta 1` (Ẑ_block = I; R̂+H±1+Π_head only) | **17.58 %** | 0.64 | **−17.8 pp** (identical to C) |
| D | + `--alg2 --alg2-u-vo` (Π_head + Û_vo, no matrix-Γ) | 22.53 % | 1.76 | **−12.9 pp** |
| E | Deployed (matrix-Γ + Û_vo) | 9.51 % | 2.42 | **−25.9 pp** |

### Attribution (corrects earlier "Ẑ_block dominates" theoretical attribution)

| Component | Marginal VMA Δ | Status |
|---|---:|---|
| **Π_head** (τ_kv + τ_group) | **0 pp** | Cell B = Cell A exactly. Sort kills column permutation. |
| **Ẑ_block β=8** | **0 pp** | Cell F = Cell C exactly. Sort also kills per-pair index swap. |
| **R̂_qk + Ĥ_qk ±1 (matrix-Γ)** | **−17.8 pp** | Confirmed by F: with Ẑ block disabled, R̂+H still drops 35 → 17.6. Per-pair 2D rotation + sign flips on real TRAINED Q/K weights break sorted-quantile per-row matching. |
| **Û_vo** | **−12.9 pp** | Cell D. QR + 0.05 Gaussian perturb on per-head V/O channels. |
| **Interaction (R̂+H × Û_vo)** | +4.8 pp (subadditive) | Combined −25.9 vs sum-of-marginals −30.7. ~5 pp overlap. |

### Theoretical correction — trained-weight statistics matter

`probe_alg2_static_attacks.py` (synthetic uniform-Gaussian weights) measured R̂_qk = 0.4 pp + Hadamard ±1 = 0 pp on VMA. **Real Q3-4B trained weights give the same components −17.8 pp.** The structured per-row distribution of trained Q/K is sensitive to per-pair 2D rotation + sign flips in a way that random Gaussian weights aren't. The earlier synthetic probe systematically underestimated R̂_qk + Hadamard's VMA contribution.

### Path-2 deployment defense recipe (post-bisection)

For **VMA** specifically:
- §5.2.2 (Π + α_e + α_h): primary, −63 pp
- R̂_qk + Ĥ_qk ±1 (matrix-Γ kernel): secondary, −18 pp
- Û_vo: tertiary, −13 pp
- Ẑ_block β=8: zero contribution (can drop without VMA cost)
- Π_head: zero contribution (can drop without VMA cost)

For **ISA HiddenState** and other runtime attacks, the attribution may differ — Ẑ_block's score-perturbation (β=8: 35 % rel score Δ) was theorized to contribute there, though our earlier Alg1 vs Alg1+minAlg2 comparison found ~0 pp delta. Need cell-B / cell-F deltas on the runtime attack ledger to close the per-component story for runtime.

## VMA × Alg2 — final ranked table (2026-05-26)

Ordered by marginal contribution on §5.2.2 substrate (Alg1+§5.2.2 = 35.42 % baseline):

| Param | Δ from substrate | Mechanism on VMA | Other attacks where it defends |
|---|---:|---|---|
| **R̂_qk + Ĥ_qk±1** (matrix-Γ kernel) | **−17.8 pp** | Per-RoPE-pair 2D rotation + ±1 sign flip on Q/K head_dim. On trained Q3-4B weights, shifts each row's value distribution enough that sorted-quantile features no longer match across 36 × 2 = 72 Q+K sources. (Synthetic Gaussian under-predicts; trained structure is what makes it bite.) | IA Attn-IA (but Alg1 already at 0); no measurable contribution to ISA HS, ISA-AttnScore, NN, IMA, TFMA, SDA |
| **Û_vo** (V↔O random projection) | **−12.9 pp** | QR-orthogonal + 0.05 Gaussian per-head perturb shifts V/O column statistics across 36 × 2 = 72 V+O sources | V/O channel-pair (Alg1 alone already at random); small ISA HS via bf16 numerical leak (~2-3 pp); zero at fp32 elsewhere |
| **Ẑ_block β=8** | **0 pp** | Per-RoPE-pair index permutation is a column permutation — sorted-quantile features are sort-invariant. Confirmed Cell F (β=1) = Cell C (β=8) exactly | **Theoretically** ISA HS via 35 % rel attention-score Δ; **empirically** ≤ noise (+1.2 pp on Alg1+minAlg2 within ±5 pp pool noise). Also: causes 35 % rel RoPE attention-score perturbation → accuracy cost. **No measured attack benefits meaningfully.** |
| **Π_head** (τ_kv + τ_group) | **0 pp** | Head-index shuffle relabels which head's slice lives where; same set of rows, just reordered. Sort kills it. Confirmed Cell B = Cell A exactly | Per-head fingerprint + V/O channel-pair targets — but Alg1's rectangular keymat already collapses those to random chance. **No measured attack benefits meaningfully.** |

## Cross-param interactions

### Within Alg2, on §5.2.2 substrate

| Combination | Δ observed | Σ marginals | Interaction |
|---|---:|---:|---:|
| R̂_qk+H alone (Cell C/F) | −17.8 pp | — | — |
| Û_vo alone (Cell D) | −12.9 pp | — | — |
| R̂_qk+H × Û_vo (deployed Cell E) | **−25.9 pp** | −30.7 pp | **+4.8 pp subadditive** — ~5 pp shared coverage of VMA's per-row signature space |

### §5.2.2 × Alg2 (vs Alg1 baseline 98.4 %)

| Combination | Δ observed | Σ marginals | Interaction |
|---|---:|---:|---:|
| §5.2.2 alone | −63 pp | — | — |
| Alg2-min alone (no §5.2.2) | −1.9 pp | — | — |
| §5.2.2 + R̂_qk+H (Cell F on §5.2.2) | −80.8 pp | ≈ −64 pp | **−16.8 pp superadditive** — §5.2.2 amplifies R̂+H's VMA hit ~9× |
| §5.2.2 + Û_vo (Cell D) | −75.9 pp | ≈ −63 pp | **−12.9 pp superadditive** |
| §5.2.2 + full Alg2 (deployed) | −88.9 pp | ≈ −65 pp | **−23.9 pp superadditive** — full §5.2.2 × Alg2 amplification |

**Structure:** §5.2.2 (row-scrambling) and Alg2 (column-scrambling) attack VMA's signature on two independent axes — sum-of-marginals systematically under-predicts the combined effect by 12-24 pp. Within Alg2, R̂+H and Û_vo overlap defensively by ~5 pp (subadditive).

## Optimal configs

### VMA-optimal

```
--mode keymat --expansion-size 128 \
--pi --noise-alpha-e 1.0 --noise-alpha-h 0.2 \
--alg2 --alg2-qk-norm-matrix --alg2-h-hadamard-signs \
--alg2-beta 1  \    # Ẑ_block = identity; 0 contribution
--alg2-u-vo \
--output-dtype bf16
```

Expected VMA top-1: **9.5 %** (identical to deployed, since Ẑ_block contributes 0).

### Broadly-optimal (all measured attacks)

```
--mode keymat --expansion-size 128 \
--pi --noise-alpha-e 1.0 --noise-alpha-h 0.2 \
--alg2 --alg2-qk-norm-matrix --alg2-h-hadamard-signs \
--alg2-beta 1  \    # drop Ẑ_block — 0 measured contribution + avoids 35 % rel RoPE score perturb
--alg2-u-vo \
--output-dtype bf16
```

**Identical to VMA-optimal.** Reasoning:

| Component | Best for VMA | Best for IA / NN / Per-head / V/O | Best for ISA HS | Best for ISA-AttnScore | Best for IMA-EmbedRow | Best for TFMA / SDA | Accuracy cost |
|---|---|---|---|---|---|---|---|
| §5.2.2 | keep | n/a (Alg1 alone defeats) | partial | n/a | keep (primary) | keep (primary) | small (−15 pp HumanEval per 2026-05-20 handoff) |
| R̂_qk + Ĥ_qk±1 | keep (−17.8 pp) | n/a | 0 (cancels in QK^T) | 0 | n/a | n/a | minor |
| Û_vo | keep (−12.9 pp) | n/a | ~2-3 pp bf16 leak | 0 | n/a | n/a | minor |
| **Ẑ_block β=8** | **drop** (0 pp) | n/a | 0 measured (≤ noise) | 0 | n/a | n/a | **negative — 35 % rel RoPE attention-score perturb, accuracy hit** |
| Π_head | keep (auto with `--alg2`) | 0 (Alg1 defeats) | 0 | minor | n/a | n/a | trivial |

### Why they coincide

No measured attack benefits from Ẑ_block β=8 enough to justify its 35 % RoPE attention-score perturbation. Dropping it (β=1) is a strict improvement: same security on the current ledger, better model accuracy.

### When the two configs would differ

If we added an attack the current ledger doesn't include:
- **Per-RoPE-pair frequency-band attack on W_q** → would benefit from Ẑ_block. Not implemented.
- **Stronger ISA-AttnScore inverter** (paper-like trained transformer) → could re-expose Ẑ_block's score perturbation. Not implemented.
- **Per-head function fingerprinting with non-spectral matcher** (current SVD-spectrum is right-mult-invariant) → would benefit from Π_head. Not implemented.

Until those land in the ledger and are measured, the broadly-optimal drops Ẑ_block for free accuracy.

## Synthetic probe scope correction (2026-05-26)

`probe_alg2_static_attacks.py` measured R̂_qk = 0.4 pp + Ĥ±1 = 0 pp on uniform Gaussian random weights, n_layers=2, no §5.2.2 substrate. Real Q3-4B + §5.2.2 substrate gives R̂_qk + Ĥ±1 = −17.8 pp combined. **9× discrepancy.**

**Root cause: the probe was incomplete, not wrong.** It measured the no-§5.2.2-substrate case correctly. In that setting the marginal is small — matched by the real Alg1 → Alg1+minAlg2 measurement (−1.9 pp). The discrepancy comes from §5.2.2 × Alg2 superadditivity, which isn't captured because the probe didn't have a §5.2.2 substrate.

Per-row sorted-quantile feature shift (cosine similarity between plain and post-R̂+H features) is similar on real Q3-4B vs synthetic Gaussian (0.998 vs 0.999 median). Trained-weight structure plays a small role; the BIG factor is the substrate.

Probe header now documents this limitation (see `probe_alg2_static_attacks.py` docstring). Future synthetic probes for sort-based attacks should:
- include a §5.2.2-equivalent substrate (noise on W_e + row-perm) as a second test
- explicitly state what substrate they measure on
- be treated as "components in isolation" estimates, not deployed-cell predictions

### Implications for the rest of the cross-map

The "Alg2 ≈ 0 defense" conclusion drawn from the Alg1 vs Alg1+minAlg2 deltas needs to be re-checked for every attack class. The minAlg2-on-Alg1 baseline misses §5.2.2-Alg2 interaction effects. To properly attribute Alg2 across the ledger, we'd need to measure:

1. Alg1 + §5.2.2 (no Alg2) — done above for VMA + IA only.
2. Alg1 + §5.2.2 + Alg2 (deployed) — done.

For every attack class, compute `interaction = deployed - (Alg1+minAlg2) - (Alg1+§5.2.2) + Alg1`. Positive interaction means Alg2 amplifies §5.2.2's defense; near-zero means Alg2 truly contributes nothing.

The IA, NN, Per-head-fpt, V/O channel-pair attacks all hit floor (~0 or random) at Alg1 alone, so the interaction is bounded ≤ random-chance — Alg2 has no room to add measurable defense. **VMA, IMA basic, IMA-EmbedRow, ISA HS multi-key are the attacks where §5.2.2 × Alg2 interaction could matter — only VMA confirmed so far.**

### Open empirical gaps

- IMA basic, IMA-EmbedRow, ISA HS multi-key on the Alg1+§5.2.2 cell — to compute §5.2.2 × Alg2 interaction for each. ~15 min each.
- 5-seed re-measurement of the deployed cell's VMA (currently single-seed in §08).

## Single-key vs multi-key audit (2026-05-25)

Multi-key paper-faithfulness (K=64 K_a synthesis per paper §3.2 Kerckhoffs) status across the §08 ledger:

| §08 attack | Multi-key status | Driver |
|---|---|---|
| **NN** | n/a — no learnable inverter | `attack_drivers/run_nn.py` |
| **ISA HiddenState (legacy L=23, 3 surfaces)** | covered operationally by running `run_isa_multikey.py --layer 23 --kind {attn_norm,Qcur_normed,Kcur_normed}` | `m2_7/run_isa_multikey.py` (multi-key) |
| **ISA HiddenState (paper-faithful L=17, K=64)** | **multi-key ✓** | `m2_7/run_isa_multikey.py` |
| **ISA AttnScore** | **STRUCTURALLY n/a** — see below | stub at `m2_7/run_isa_attn_score_multikey.py` (returns `not_applicable` with rationale) |
| **VMA** | n/a — sorted-quantile feature match | `attack_drivers/run_vma.py` |
| **IA (Gate + Attn)** | n/a — invariant feature match | `attack_drivers/run_ia.py` |
| **IMA-EmbedRow-ridge** | **multi-key ✓** | `m2_7/run_ima_embedrow_attacks_multikey.py` |
| **IMA-EmbedRow-transformer** | **multi-key ✓** | same file, separate variant |
| **QK-norm Γ eigendecomposition** | n/a — direct `linalg.eig`, no inverter | inline / docs only |
| **TFMA / SDA** | n/a — bigram match on wire-side stream | `attack_drivers/run_{tfma,sda}.py` |

### Why multi-key is structurally inapplicable to ISA AttnScore

Paper-faithful multi-key rests on a covariant-synthesis pattern
`surface_a^k = f(K_a^k, surface_plain)` where K_a (Algorithm 1 keymat) acts
on the residual axis. The attacker synthesises K=64 training versions of
the surface, forcing key-invariant inversion.

For the attention-score surface `Q·K^T / √d_h`, no such relation holds:

- Under **Algorithm 1 alone** (keymat-only), the residual-axis K_a transforms
  W_q / W_k input axes so that `X̃·W̃_q = X·W_q` — the **scores are invariant
  under K_a by construction** (the paper's covariance claim). There is no
  K_a → score function to multi-key against.
- Under **Algorithm 2**, scores are perturbed by intra-head `R̂_qk`
  (head-dim axis) and head permutations `τ_kv` / `τ_group` — but those keys
  are sampled independently of K_a (`lib/alg2.py:generate_r_qk` /
  `:generate_head_perm`). Multi-keying over Algorithm-2 keys would be a
  separate threat model (and a separate driver, e.g.
  `run_isa_attn_score_alg2_multikey.py`); it's not what "paper-faithful K_a
  multi-key" means.

This is consistent with the 2026-05-25 empirical finding above (plain + obf
both at ≤ 2 % top-1 across L ∈ {0, 5, 11, 17, 23}). The score surface
admits no K_a-covariant lever and the attention computation softmax-mixes
positional signal beyond what a linear ridge can recover at any depth.

## Bugs (open or noted-only)

| # | Where | Severity | Status |
|---|---|---|---|
| 1 | `lib/alg2.py:241-262` comment claims "matrix-Γ algebra is still exact under that M_q" but it's only exact at β=1; deployed β=8 has 35 % rel score Δ | medium | **fixed 2026-05-25** — caveat block + β-sweep table added to `lib/alg2.py` and retroactive note added to `2026-05-19-alg2-z-block-degeneracy.md` |
| 2 | `obfuscate_qwen3_gguf.py:531-542` legacy reconstruction omits `u_vo`/`u_vo_inv` → `TypeError` if anyone runs `--alg2 --alg2-u-vo` without `--alg2-qk-norm-matrix` | low (deployed config doesn't hit it) | **fixed 2026-05-25** — pass `full_keys.u_vo` and `full_keys.u_vo_inv` through |
| 3 | `python/aloepri-llm/scripts/check_alg2_invariance.py:26` imports from `python/path-2` (stale after rename) | low | **fixed 2026-05-25** — renamed to `python/aloepri-llm` |
| 4 | `--alg2-gamma`, `rope_base` are dead CLI args after 2026-05-19 Ẑ_block fix | low | left as documented cruft — signature ripple to remove is uglier than the benefit |
| 5 | Path-2 deviates from paper Alg2 line 6 (`M_k = R̂_qk · Ĥ_qk⁻¹ · Ẑ_block` vs paper's `Ĥ_qk⁻¹ · Ẑ_block^T`); algebra makes `M_q · M_k^T = I` exactly (paper's requires Ẑ² = I) | design choice | documented; security-proof implications untested |
| 6 | Doc-level "Alg2 amplifies K_a × K_d" conflates §5.2.2 + §5.2.3 + bf16 + Q8_0; Phase d ablation plan in `2026-05-22-keymat-defense-optimization.md` inherits the confusion | medium | **fixed 2026-05-25** — framing-correction blocks added to both `aloepri-keymat-variance.md` and the Phase d table now has a §5.2 column + Ẑ_block + Π_head rows |

## Gaps

### Open empirical gaps

- **ISA-attn-score: never run on any cell.** Driver exists, returns
  `not_applicable` for GELO but should run for path-2 cells. Theoretical
  prediction: Π_head and Ẑ_block dominate; R̂_qk and Ĥ_qk ±1 cancel.
  Cost: ~45 min once a deployed Û_vo GGUF is regenerated (none on disk).
- **Per-head fingerprint and V/O channel-pair drivers: never run.** Both
  written 2026-05-25. Same regen-GGUF cost.
- **ISA HiddenState per-component empirical isolation:** the 1.70 pp 4B
  aggregate Alg2 delta is split theoretically (70 % Ẑ_block, 30 % bf16
  Û_vo). No empirical ablation has actually disabled each component
  individually. Would need ~3 h of obfuscation + capture + multi-pool
  attack runs.
- **Real-cell VMA + IA Gate-IA on a current Alg2 cell:** synthetic probes
  cover the within-Alg2 attribution, but the aggregate VMA/Gate-IA top-1
  on the deployed cell hasn't been measured since pre-2026-05-20 Π fix.

### Theoretical gaps

- **Paper deviation (Bug #5) security-proof implications.** Path-2's M_k
  formula gives `M_q · M_k^T = I` instead of paper's `R̂_qk` residual.
  Eliminates R̂_qk from the obfuscated runtime score. Whether paper §C's
  RmDP bound still applies is open.
- **Quantization-as-defence accounting.** Theory says bf16 of Û_vo
  contributes most of Û_vo's ~2-3 pp defence at the deployed precision.
  Moving to fp32 weights would *remove* defence (the "quantization
  piggyback" effect). Not currently called out in any deployment doc.
- **`alg2_seed` is in the public obfuscate script.** Per-layer offset
  `alg2_seed + 1000·il` is also public. An attacker who reads the script
  can reconstruct Z_block, R̂_qk, Hadamard signs, head-perm exactly. The
  intended secret is presumably the `--alg2-seed` value, but it's also
  stored in GGUF metadata as `aloepri.alg2_beta` etc., and the seed
  itself isn't currently treated as secret. **Treat as secret** (rotate
  per deployment, store in `.key.npz`, omit from GGUF metadata).

### Static-weight attacks not implemented

The path-2 ledger is missing several attacks that R̂_qk / Ĥ_qk
non-unit / Û_vo would defend against if they were in the ledger:

- **Per-head fingerprint** — code now exists (see [Inventory](#path-2-implemented-attack-inventory)).
- **V/O channel-pair** — code now exists.
- **RoPE-pair angular-clustering** on W̃_q (target of R̂_qk
  specifically; not implemented).
- **Per-pair magnitude attack** on W̃_q (target of Ĥ_qk non-unit;
  not implemented).

Until those are run, the components they target appear as "dead weight"
in this map. They may not be dead weight against the full threat
model — just against the *measured* threat model.

## Headline finding & recommended deployment changes

**Theory + synthetic probes converge on the same statement:** of
Algorithm 2's 5 components, only **Ẑ_block** (against ISA HiddenState
at β=8 — empirically and theoretically) and **Π_head** (against IA
Attn-IA, empirically) carry measurable defence on the current path-2
attack ledger. R̂_qk, Ĥ_qk ±1, and Û_vo provide near-zero against every
ledger attack we have measurements or synthetic probes for.

Three options consistent with this evidence:

1. **Strip dead-weight from the deployment.** Drop `--alg2-qk-norm-matrix`,
   `--alg2-h-hadamard-signs`, `--alg2-u-vo`. Keep `--alg2` (head-perm) +
   `--alg2-beta 8` (for ISA HiddenState defence). Net measurable defence
   unchanged on the current ledger; removes Bug #1 and Bug #2 from
   reachability; simplifies the obfuscation code path.
2. **Add the missing static-weight attacks to the ledger before
   stripping.** Run `run_per_head_fingerprint.py` +
   `run_vo_channel_pair.py` on the current deployment to confirm whether
   R̂_qk / Hadamard / Û_vo do real work against attacks not currently
   measured. ~45 min once GGUFs are regenerated.
3. **Tune Ẑ_block β for *real* IA Attn-IA defence.** At deployed β=8 the
   window aligns with IA's block_size=16 → invariant unchanged.
   β ≥ 16 crosses block boundaries; β=64 gives near-complete defence at
   the cost of larger accuracy perturbation. Trade-off untested.

Option 1 is the lowest-risk, highest-information action given current
evidence. Option 2 is the principled "verify before stripping" path.

## Update protocol

When a new measurement / theory clarification / attack lands:

1. Update the relevant cell in the [Cross-interaction map](#cross-interaction-map).
2. Bump the status tag (T → S → M as we gather evidence).
3. Add the test/probe to [Test inventory](#test-inventory--methodology).
4. If the finding changes the headline (e.g. R̂_qk suddenly shows real
   defence on per-head fingerprint), update [TL;DR](#tldr-2026-05-25).
5. **Keep TL;DR dated.** Edit the date in the header `## TL;DR (YYYY-MM-DD)`
   each time the headline section changes.
6. Don't duplicate content into other docs — link out via paths.

## Suggested skills for next session

- `/diagnose` — if attack ablation results conflict with theoretical
  predictions in this map (e.g. R̂_qk shows unexpected non-zero defence
  on a real cell).
- `/grill-with-docs` — if updating `aloepri-keymat-variance.md` and the
  Phase d plan to remove the §5.2.2-vs-§5.2.3 conflation flagged in
  Bug #6.
- `/code-review` — before any deployment strip (Option 1 above).

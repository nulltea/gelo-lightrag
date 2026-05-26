---
type: dev-log
status: stale
created: 2026-05-19
updated: 2026-05-26
tags: [aloepri, alg2, qwen3, threat-model, keymat]
companion: [2026-05-19-alg2-qwen3-shape-analysis, 2026-05-19-alg2-z-block-degeneracy, 2026-05-20-ima-embedrow-transformer-investigation, 2026-05-21-aloepri-quantisation-and-alg2-gaps, 2026-05-21-ima-transformer-paper-disparity, 2026-05-21-uvo-isa-multikey-and-gpu-keymat-bug]
superseded_by: aloepri-attack-chronicle
archive_reason: "Earlier (2026-05-26) distillation focused on Qwen3 architectural delta, matrix-Γ kernel, Ẑ_block bug, Û_vo deployment, quantisation × Alg2. Absorbed into chronicle §3 (threat model) + §4 dated entries."
---

# Algorithm 2 + threat-model on Qwen3 — distilled analysis log

> Knowledge layer for the Algorithm-2-on-Qwen3 workstream. Consolidates the
> architectural gaps between the paper's Qwen2-baseline design and Qwen3's
> QK-norm topology, bugs encountered during deployment, quantization
> constraints, and paper-disparity findings. Companion to 6 handoffs spanning
> 2026-05-19 → 2026-05-21.

## Qwen3 vs paper's Qwen2 — architectural delta

### QK-norm site (the core gap)

Qwen3 inserts per-element RMSNorm (`attn_q_norm.γ_q`, `attn_k_norm.γ_k`, shape `(head_dim,) = (128,)`) **after** the Q/K projections and **before** RoPE and the attention dot product. Paper's Algorithm 2 (§5.2.3) assumes Q flows directly from W_q into RoPE + dot; this site does not exist in Qwen2.5 / Llama3 / DeepSeek-R1-Distill-Qwen.

Topology (Qwen3-1.7B):
```
W_q [d → 128] ──► attn_q_norm(γ_q) ──► RoPE ──┐
                                               ├─► softmax(Q·Kᵀ/√d_h) ──► attn_v ──► W_o
W_k [d → 128] ──► attn_k_norm(γ_k) ──► RoPE ──┘
```

**Why it breaks Algorithm 2.** Intra-head transforms `M_q = R̂_qk · Ĥ_qk · Ẑ_block` are designed to fold into W_q's output axis (`W̃_q = W_q · M_q`). For `attn_q_norm(W_q · x · M_q, γ_q)` to equal `attn_q_norm(W_q · x, γ_q) · M_q`, we need M_q to commute with `Diag(γ_q)`. In general it doesn't, because γ_q is non-uniform across head_dim and co-adapted with trained Q activations. Per-input correlation `E[q² · γ²] > 0` pushes attention toward uniformity (repetitive "a again ... ..." loops).

### head_dim = 128 quirks

Qwen3 uses `head_dim = 128` (vs Qwen2's 64). Combined with per-element γ at full dimension:
- Larger γ variance across head_dim (some training heads put γ_max = 68 on the K side — extreme outliers).
- Flatter γ-value distributions that don't cluster. Pre-flight measurement on 56 vectors: only 22 % coverage at ε=0.25, 1.4 % at ε=0.10.

### GQA / KV head count

Qwen3-1.7B has n_q=16, n_kv=8 (2:1 GQA). Attention mechanics unchanged; transforms must respect group structure in head permutation (τ_kv, τ_group).

---

## Algorithm 2 primitives — what deployed vs what didn't

### M_q / M_k construction (matrix-Γ kernel extension)

- **Paper design (§5.2.3):** `W̃_q = Q̂_q · W_q · R̂_qk · Ĥ_qk · Ẑ_block`, `W̃_k = Q̂_k · W_k · R̂_qk · Ĥ_qk⁻¹ · Ẑ_block_η`.
- **Status on Qwen3 (pre-Opt-C):**
  - ✅ `Q̂_q`, `Q̂_k` — input-axis folds of `attn_norm.γ_input` into W_q/W_k — deployed, algebraically exact.
  - ❌ `R̂_qk · Ĥ_qk · Ẑ_block` on output axis — identity-stubbed until QK-norm is handled.
- **Root cause:** Paper §5.2.5 fold construction is only proven for input-axis norm sites. The QK-norm site is output-axis and the paper doesn't derive the fix.

### Ẑ_block permutation (degeneracy finding)

- **Bug:** `generate_block_perm` produces identity permutation under default parameters (`beta=8, gamma=1e3, rope_base=1e6`) for all `head_dim ∈ {8, 16, 32, 64, 128}` and seeds {7, 42, 99, 12345}.
- **Mechanism:** softmax over ζ-log differences collapses to `[1.0, ~0, ~0, …]` at fp32 precision. `rng.choice` always returns index 0 → window_size always 1 → no permutation.
- **Why it didn't surface:** Qwen3 deployment forced `q_matrix = k_matrix = I` anyway, so Ẑ_block was never exercised. Paper Table 3 "Head&BlockPerm" → 0 % HS reduction conflates head-shuffle (Π_head) and block-perm (Ẑ_block); marginal contribution of Ẑ_block is unmeasured.
- **Fix sketches (deferred):** drop γ to ~1; use rope_base ~1e3 in score only; uniform random permutation. Each requires re-deriving the paper's RoPE-locality argument.

### Û_vo (paper-literal vs deployed)

- **Paper design:** V↔O random projection (§5.2.3). Paper Table 4 attributes ~0.82 % → 0 % HiddenState reduction to Û_vo.
- **Status:** omitted in both `vendor/aloepri-py` and our `lib/alg2.py` (V's `dense_transform=None`, O's input axis unaffected).
- **Fixed 2026-05-21:** patched `lib/alg2.py` + `obfuscate_qwen3_gguf.py` with `generate_u_vo()` (QR-stabilised N(0, 1/d_head·I)); CLI flag `--alg2-u-vo`. Math verified: condition number 3.57–5.87, E2E V→O cancellation error 1.18×10⁻⁶ relative.
- **Impact on Qwen3-4B ISA-HS:** 5.11 % → 3.41 % (−1.70 pp, **33 % relative**).
- **Impact on Qwen3-8B ISA-HS:** 9.73 % → 9.00 % (−0.73 pp, **7.5 % relative**).
- **Verdict:** Defense scales poorly with dimension — 8B defense gap is structural; Û_vo isn't sufficient at d=4096.

### Keymat (Algorithm 1) on Qwen3

- Per-row heavy-tailed distributions (max ≈ 55, std ≈ 4.7 on Qwen3-1.7B).
- Multi-key ISA driver (`run_isa_multikey.py`) generates K=64 independent keymats.
- **GPU-native port discovered seed-convention divergence:** vendor's CPU keymat uses fresh `torch.Generator` per component (8 streams per keymat); port's GPU-native uses one advancing generator per keymat (1 stream). At K=64 on Qwen3-4B Û_vo: vendor 3.41 % top-1, port **11.92 %** (exceeds plain ceiling 10.18 %, structurally impossible). Diagnosis pending seed-sweep and port-on-CPU isolation test.

---

## Bugs found and (sometimes) fixed

### Ẑ_block silent identity
- **Symptom:** block permutation never happens; intra-head obfuscation weaker than expected.
- **Root cause:** softmax over window sizes collapses due to `gamma · ΔζLog` magnitude dominance at default gamma=1e3, rope_base=1e6.
- **Fix:** **not yet applied** (deferred pending paper-locality argument re-derivation).
- **Refs:** `2026-05-19-alg2-z-block-degeneracy.md`, `lib/alg2.py:generate_block_perm`, `vendor/aloepri-py/src/attention_keys.py`.

### QK-norm fold breaks attention
- **Symptom:** smoke test output degenerates to "... ... ... a again ..." (high-frequency loops) when Algorithm 2 intra-head transforms are wired with §5.2.5 fold applied to QK-norm γ_q, γ_k.
- **Root cause:** paper §5.2.5 fold assumes i.i.d. Gaussian input; trained Qwen3 Q/K vectors aren't. Per-input correlation breaks the fold's κ approximation; even small per-input error amplifies through softmax, pushing attention toward uniformity.
- **Three options evaluated:**
  - **Option A (empirical κ_qk calibration):** medium risk — variance of ratio might still cause softmax error on outliers.
  - **Option B (γ-commuting rotation):** **dead** — clusters tiny at operational ε (1.4 % coverage at ε=0.10, 22 % at ε=0.25 already too loose).
  - **Option C (runtime κ correction in llama.cpp):** algebraically certain; violates "no infra change" thesis but we already ship patched llama.cpp. **Recommended.** ~1 week effort.
- **Current deployment:** head-shuffle only (τ_kv, τ_group), no QK-norm fold, no intra-head dense.
- **Refs:** `2026-05-19-alg2-qwen3-shape-analysis.md` §3, `2026-05-21-aloepri-quantisation-and-alg2-gaps.md` §(b).

### GPU keymat divergence (diagnosis pending)
- **Symptom:** `run_isa_multikey.py` GPU-native keymat port on Qwen3-4B Û_vo produces TTRSR 11.92 % top-1, exceeding plain ceiling 10.18 % (structurally impossible if keymats are drawn from the correct distribution).
- **Root cause hypothesis:** seed-convention difference + possible rocSOLVER iGPU sign convention in QR/SVD biasing nullspace projection toward directions aligned with K_d.
- **Tests queued:**
  1. Seed sweep across attacker_seed ∈ {1, 2, 3} for both impls.
  2. Port-on-CPU diagnostic to isolate rocSOLVER vs seed-stream layout.
  3. Nullspace basis sign-convention analysis.
- **Refs:** `2026-05-21-uvo-isa-multikey-and-gpu-keymat-bug.md` §2.

---

## Quantization × Algorithm 2

**Verdict: fp32 required; all production quantization formats break obfuscation.**

| Format | Size | Output | Verdict |
|---|---:|---|---|
| fp32 | 8.6 GB | coherent | ✅ reference |
| Q8_0 | 2.3 GB | degenerate "(((,chein,zech…" | ❌ breaks |
| Q6_K | 1.8 GB | 500 error | ❌ breaks |
| Q5_K_M | 1.6 GB | word salad | ❌ breaks |

**Mechanism:** AloePri keymat weights are heavy-tailed per row (max ≈ 55, std ≈ 4.7). Q8_0 stores 32-element blocks with one fp16 scale; within a heavy-tailed block small values round to zero, breaking per-row covariance. K-quants have more flexible scaling but still insufficient precision.

**Future paths explored (none implemented):**
1. QR-project keymat B matrix toward orthonormal manifold — tightens per-row variance; smaller obfuscation group.
2. Smaller h (64 or 32) instead of 128 — less internal expansion; smaller AloePri group.
3. K-quant with importance matrix — llama.cpp calibration on plaintext runs.
4. Quantization-aware obfuscation design — unexplored.

**Expected effort:** ~1 week to land Q8_0-friendly construction that also passes accuracy gates.

---

## Paper-disparities

### IMA-EmbedRow-transformer plain-control failure

- **Finding (2026-05-20):** trained-transformer IMA inverter fails identity plain control. Ridge inverter gets 99.2 % top-1 on plain; transformer gets 0.0–0.4 % across 4 architecture variants.
- **Root cause:** ridge is the closed-form least-squares solution. Transformer tries to approach W via AdamW GD with 1024 update steps; each parameter's movement bounded ~5×10⁻³ per step; identity diagonal needs ~30 step-units to climb.
- **Interpretation:** AloePri reference uses paper-default budget (epochs=2, batch=8, 256 rows). Over 256 updates GD doesn't converge to ridge's solution; paper's "IMA = 0 %" may be the constrained-attacker reading (paper §3.2), not "no attacker can recover."
- **Verdict:** drop IMA-EmbedRow-transformer from measurement table. Ridge (99.2 % plain → 97.66 % obf, 6.5× defense gap) is the load-bearing static-embedding-row measurement.
- **Refs:** `2026-05-20-ima-embedrow-transformer-investigation.md`.

### IMA GPU baseline disparity

- **Finding (2026-05-21):** paper-faithful `_PaperLikeIMAInverter` on Qwen2.5-0.5B backbone, AMD Strix Halo iGPU. Plain identity-τ baseline: 15.62 % top-1 at ep=100. Obfuscated Qwen3-4B: 12.70 % top-1 at ep=100.
- **Paper comparison:** paper claims IMA = 0 % on Qwen2.5-14B (Table 2, §F.1). We get 12.70 % on 4B and climbing. Both curves still monotonic at ep=100.
- **Three competing hypotheses:**
  1. Our attack ≠ paper's attack (most likely): inverter arch / corpus / hyperparam mismatch. Paper might use domain-mismatched corpus (medical); we use AloePri docstrings.
  2. Our obfuscation weaker: `obfuscate_qwen3_gguf.py` could have a bug. Structural-leak risk identified (specials/UNUSED tail still identity-τ until strong-Π patch): ~293 free pairs.
  3. Configuration/harness bug: undertrained. At ep=500–1000 divergence between plain and obf will reveal whether defense is real.
- **Tests queued:** ep=500 extension + Stage-K artifact comparison.
- **Refs:** `2026-05-21-ima-transformer-paper-disparity.md`.

### Û_vo defense scaling vs dimension

- **Finding (2026-05-21):** full Algorithm 2 with Û_vo:
  - **4B:** 3.41 % ISA top-1 (−33 % relative vs non-Û_vo).
  - **8B:** 9.00 % ISA top-1 (−7.5 % relative vs non-Û_vo).
- **Interpretation:** dimensional-headroom hypothesis (bigger `d` defends better) **falsified**. At d=4096 multi-key paper-faithful attacker recovers ~9 % regardless of Û_vo. Defense gap is structural — not closeable by Û_vo at this scale.
- **Refs:** `2026-05-21-uvo-isa-multikey-and-gpu-keymat-bug.md` §1d.

---

## Decisions taken

1. **QK-norm fix:** defer to Option C (llama.cpp runtime correction) unless Gate B/C pressure forces earlier closure. Option B is dead; Option A is high risk.
2. **Ẑ_block:** keep identity stub. Marginal defense contribution unmeasured on Qwen3. Revisit if M2.7 ISA TTRSR exceeds 15 % after full Algorithm 2 deploys.
3. **Û_vo:** deployed. Math verified to 1×10⁻⁶ relative. Meaningful attenuation on 4B (33 %), marginal on 8B (7.5 %).
4. **Quantization:** fp32 mandatory. No production format survives keymat's heavy-tailed structure.
5. **IMA-EmbedRow-transformer:** removed from measurement table (broken attack).
6. **GPU-keymat port:** experimental; use CPU vendor build for production until divergence is diagnosed.

---

## Open questions

| Question | Status | Blocker? |
|---|---|---|
| What is the paper's QK-norm solution on Qwen3? | Unanswered; not in public release | No — workarounds exist (Option A/C) |
| Does Option C (llama.cpp correction) reduce ISA TTRSR below 15 % at 4B/8B? | Pending implementation | Yes, for full Alg2 on Qwen3 |
| What causes GPU-keymat seed-convention divergence? | Pending seed sweep + port-on-CPU diagnostic | No — use vendor CPU build |
| Does Û_vo defense scale to d=8192 (Qwen3-14B+)? | Preliminary: 7.5 % attenuation at 8B suggests not | No — structural gap, not Û_vo's fault |
| Is paper's IMA = 0 % the constrained-attacker (ep=2) reading? | Pending Stage-K artifact comparison | No |
| Why are per-row keymat weights heavy-tailed (max=55)? | Not investigated — Algorithm 1 design choice | No — fp32 cost accepted |

---

## Cross-references

- **Architecture deep-dive:** `2026-05-19-alg2-qwen3-shape-analysis.md` (topology, §5.2.5 fold analysis, Option A/B/C).
- **Ẑ_block bug:** `2026-05-19-alg2-z-block-degeneracy.md`.
- **QK-norm bisect:** `2026-05-21-aloepri-quantisation-and-alg2-gaps.md` §(b).
- **Quantization gaps:** `2026-05-21-aloepri-quantisation-and-alg2-gaps.md` §(a).
- **IMA transformer:** `2026-05-20-ima-embedrow-transformer-investigation.md`, `2026-05-21-ima-transformer-paper-disparity.md`.
- **Û_vo + ISA multi-key + GPU bug:** `2026-05-21-uvo-isa-multikey-and-gpu-keymat-bug.md`.
- **Implementation:** `python/aloepri-llm/{lib/alg2.py, obfuscate_qwen3_gguf.py}`.
- **Evaluation:** `evals/aloepri-attacks/m2_7/{run_isa_multikey.py, run_ima_embedrow_attacks.py}`.
- **Protocol doc:** `docs/prototype/aloepri-llm.html` §07–§08 (perf, threat-model gates, acceptance criteria).

---
type: handoff
status: current
created: 2026-05-18
updated: 2026-05-21
tags: [path-2, aloepri]
---

# Path 2 (AloePri) — next steps handoff

> **Written:** 2026-05-18, after first working AloePri obfuscation on
> Qwen3 1.7B (commit [`124703e`](../../../../commit/124703e), fp32 keymat
> h=128 producing coherent on-topic continuations through stock
> llama.cpp + Vulkan).
>
> **Audience:** the agent that picks up `path-2-aloepri-gemma` next.
> Sequenced: three gates **must** complete in order before any of the
> deferred privacy work (items 6–10) starts. Each gate either unblocks
> the next, or surfaces a finding that changes scope.
>
> **Don't duplicate** the following — reference them by path:
>
> - Plan: [`../../plans/path-2-aloepri-gemma.md`](../../plans/path-2-aloepri-gemma.md)
> - Running status (chronological): [`../../dev/logs/path-2-status.md`](../../dev/logs/path-2-status.md)
> - Protocol doc: [`../prototype/aloepri-llm.html`](../prototype/aloepri-llm.html)
> - Gemma 4 deferred work: [`2026-05-21-aloepri-gemma-deferred.md`](2026-05-21-aloepri-gemma-deferred.md)
> - Paper: AloePri, arXiv 2603.01499

---

## Where we are now

| Item | Status |
|---|---|
| Architecture | Qwen3 1.7B (pre-norm-only, 28 layers × 2 norms + 1 output_norm = 57 residual norm sites) |
| Source GGUF | `/home/timo/.cache/huggingface/.../Qwen_Qwen3-1.7B-Q8_0.gguf` (2.1 GB Q8_0) |
| Obfuscation level | Algorithm 1 key matrices + §5.2.5 fusion only. **No Π / Algorithm 2 / noise yet.** |
| Obfuscated artifact | `keymat-h128-fp32.gguf` (9.1 GB, **fp32-stored** because fp16 collapses chain) |
| Rewriter | `python/aloepri-llm/obfuscate_qwen3_gguf.py` — three modes: `identity-pad`, `gamma-only`, `keymat` |
| Verification | `gamma-only` mode produces **bit-identical** output to plaintext (fusion verified); `keymat h=128 fp32` produces coherent on-topic continuations (different tokens from plaintext) |

### Running containers (current, on Strix Halo Vulkan iGPU)

| Port | Container | Artifact | Status |
|---:|---|---|---|
| `:11437` | `llama-gemma4-e2b-aloepri-baseline` | Gemma 4 E2B Q8_0 plaintext (deferred) | ✓ |
| `:11438` | `llama-gemma4-e2b-aloepri-identity-pad` | Gemma 4 identity-pad fp16 | ✓ regression ref |
| `:11441` | `llama-qwen3-1p7b-aloepri-baseline` | **Qwen3 1.7B Q8_0 plaintext** | ✓ |
| `:11442` | `llama-qwen3-1p7b-aloepri-gamma` | Qwen3 gamma-only (§5.2.5 fusion check) | ✓ bit-identical |
| `:11446` | `llama-qwen3-1p7b-aloepri-h128-fp32` | **Qwen3 keymat h=128 fp32** | ✓ coherent |
| `:11443`–`:11445` | various keymat fp16/fp2 attempts | (all degenerate; safe to `docker rm -f`) | — |

Reclaim disk by stopping the failed-experiment containers:

```bash
docker rm -f llama-qwen3-1p7b-aloepri-h2 \
              llama-qwen3-1p7b-aloepri-h128 \
              llama-qwen3-1p7b-aloepri-h2-fp32 \
              llama-gemma4-e2b-aloepri-keymat \
              llama-gemma4-e2b-aloepri-gamma
```

---

## Gated next-steps (do in order)

### Gate A — Q8_0 requantisation test

**Question:** Does the obfuscated artifact survive Q8_0 quantisation, or
does fp32 stay required for deployment?

**Why this matters:** fp32 → 9.1 GB; Q8_0 → ~2.5 GB. ~3.6× smaller +
~3× faster decode if it works. If it doesn't, deployment cost picture
changes (or we try Q5_K_M / Q6_K).

**Caveat from earlier analysis (recorded so it's not re-discovered):**
Q8_0 stores 32-element blocks with one fp16 scale factor. Within a
block, per-element absolute error is ±max/254 ≈ 0.4 % of the block
max. This is *good* for blocks with uniform magnitudes but *bad* for
blocks mixing large and small values — the small values can round to
zero. AloePri-obfuscated weights at `blk.27` show max=55 with std=4.7,
so within-row variance is significant. **Outcome is empirical.**

**Steps:**

1. `vendor/llama.cpp/build/bin/llama-quantize \
   /home/timo/.cache/huggingface/path-2-aloepri/qwen3-1.7b/keymat-h128-fp32.gguf \
   /home/timo/.cache/huggingface/path-2-aloepri/qwen3-1.7b/keymat-h128-Q8_0.gguf \
   Q8_0`
2. Launch on port 11447 via the same `llama.cpp:server-vulkan` container
   pattern as the existing :11446.
3. Same prompt comparison: `"What is the capital of France?"` at
   `temperature: 0.0`, `max_tokens: 24`.

**Acceptance:**

- If output is **coherent on-topic** (similar quality to current fp32
  keymat output): **Q8_0 holds the chain.** Use Q8_0 as the production
  format going forward. Re-run gates B and C on the Q8_0 artifact.
- If output **degenerates** (newlines, `?`, single-token loops):
  Q8_0's per-block scaling isn't sufficient. Try Q6_K next, then
  Q5_K_M. If all of those fail, stay on fp32.

**Effort:** ~30 minutes (quantize + launch + sanity completion).

---

### Gate B — Temperature-0 determinism + plaintext-vs-keymat diff

**Question:** Is the plaintext-vs-keymat output divergence
(`"largest city in the United States"` vs `"capital of Italy"`)
accuracy loss from the κ_E approximation, or just non-deterministic
sampling drift at temperature 0?

**Why this matters:** Gate C's accuracy benchmark needs a clean
plaintext baseline. If even plaintext is non-deterministic at
temperature 0, we need to switch to greedy / a fixed seed before
measuring anything.

**Steps:**

1. Call **plaintext** baseline (`:11441`, or post-Gate-A Q8_0 if
   adopted) three times with identical prompt, `temperature: 0.0`,
   `max_tokens: 32`, no other sampling params. Capture outputs A1, A2, A3.
2. **Expect:** A1 == A2 == A3 byte-for-byte. If not, document the
   non-determinism source (KV cache slot reuse, threading, kernel
   variance) and adjust.
3. Call **keymat fp32** (or Q8_0 from Gate A) similarly three times.
   Capture B1, B2, B3. Expect B1 == B2 == B3.
4. **Compare** A1 vs B1: count token positions where they diverge.
   Earliest-divergence position is the e_C-induced drift onset.

**Acceptance:**

- A1==A2==A3 and B1==B2==B3: plaintext + keymat each fully
  deterministic. The plaintext-vs-keymat difference is **real
  accuracy drift**, not sampling noise. Proceed to Gate C.
- A1==A2==A3 but B1≠B2: keymat is non-deterministic. Investigate —
  likely an fp32 round-off non-associativity issue across runs;
  unlikely to be a fundamental bug but worth tracking.
- A1≠A2: plaintext is non-deterministic at temperature 0. Reproduce
  with `--seed` flag pinned; if still non-deterministic, llama.cpp
  threading / Vulkan workgroup ordering is the cause. Switch to a
  greedy sampler and re-test.

**Effort:** ~1 hour.

---

### Gate C — Mini accuracy benchmark

**Question:** What is `e_C^AloePri` empirically on Qwen3 1.7B? Does it
fall within the paper's 0–3.5 % range, or has Gemma-style error
compounding leaked into this architecture too?

**Why this matters:** All deferred work (items 6–10 below) assumes
the obfuscated artifact is *competently inferring*. If accuracy is
already 10 %+ worse than plaintext before we add Π / Algorithm 2 /
noise, the cumulative loss after those layers will be too high to be
useful — and we should tune hyperparameters (h, λ, κ-source) before
adding more obfuscation steps.

**Scope:**

Mini benchmark (not the full M0.2 harness — keep it cheap and
isolated). Target one task per category:

| Task | Subset | Metric | Plaintext expected (paper / model card) |
|---|---|---|---|
| MMLU 0-shot | 200 prompts random sample | exact-match accuracy | ~55 % for Qwen3 1.7B |
| IFEval | 50 prompts | instruction follow rate | — |
| PIQA | 200 prompts | binary accuracy | ~75 % |
| HumanEval | 50 prompts | pass@1 | low for 1.7B; report whatever |

This is the **mini** version of the framework's M0.2 (full corpus is
1000+ prompts per task). Goal here is "in-budget per-task signal," not
publishable numbers. 30-45 min runtime per gate-C run.

**Implementation:**

- Re-use existing eval harnesses (`lm-evaluation-harness` if installed
  locally; otherwise hand-roll minimal versions in
  `python/aloepri-llm/evals/`). Don't yet wire this into the full M0.2
  shared harness — that's framework-level and blocked behind Path 1.
- Run twice: once on plaintext (`:11441` or Q8_0 if Gate A succeeded),
  once on keymat (`:11446` or Q8_0). Same sampling config (greedy,
  temperature 0, fixed seed).
- Output a single comparison table: `(task, plaintext_score,
  keymat_score, delta, delta_pct)`.

**Acceptance / decision tree:**

| Δ accuracy (keymat vs plain) | Action |
|---|---|
| **≤ 3.5 %** on every task | Paper-bound territory. Proceed to deferred items (6 → 7 → 8 → 10 ordering below). |
| **3.5 % – 10 %** | Slightly above paper claim — Gemma-style compounding may be partial. Investigate κ source first (try sampling κ_E against actual layer-output activation distribution rather than `N(0,1)`); if that doesn't recover, proceed but flag in M0.4 report. |
| **> 10 %** | Too lossy to layer Π / Algorithm 2 / noise on top. **Stop and tune** before adding any more obfuscation. Knobs in priority order: (a) per-norm-site κ instead of one global κ, (b) λ → 0 to make P̂_R more orthogonal, (c) increase h (180? 256?) to tighten κ_E concentration, (d) restrict P̂_R via QR-projection to orthonormal-row manifold. |
| **Degenerate** (model fails to do task at all) | Re-run Gate B to confirm output isn't garbage; if confirmed, the obfuscation chain is more broken than the single-prompt smoke test revealed. Reassess. |

**Effort:** ~3–5 days including harness scaffolding and the
investigation if Δ is in the moderate range.

---

## Outstanding (deferred until Gate C passes)

The single-prompt smoke test that produced coherent output means the
chain isn't broken. But the artifact **only obfuscates internal
activations** (defends against ISA / IMA / NN attacks per paper §7).
It does not obfuscate the token-level I/O. The wire payload of the
current setup is plaintext — only the weights are obfuscated. To get
the full AloePri privacy claim:

### 6. Π token-level permutation + client wrapper

- Sample τ ∼ S_n once at offline rewrite (n = 151 936 for Qwen3).
- Apply Π row-permutation to embedding (`token_embd`) and column-
  permutation (Π^T) to LM head (`output`). Same global pair of
  matrices; already untied in Qwen3 GGUF so no extra work there.
- Client wrapper: `python/aloepri-llm/aloepri_client.py` — tokenise →
  Z = {V[i] ↦ V[τ(i)]} mapping → detokenise → POST → decode → Z⁻¹.
  Per [`../prototype/aloepri-llm.html`](../prototype/aloepri-llm.html)
  §06 FIG. 03a.
- Tokeniser roundtrip property: assert
  `tokenizer.encode(tokenizer.decode(ids)) == ids` per request.
  SentencePiece (Qwen3 uses gpt2-style BPE actually — check the
  tokeniser model from GGUF metadata) edge cases on special tokens.

Defends against: VMA + IA at the I/O boundary. Without this, the
attacker who reads the wire payload reads plaintext.

### 7. Algorithm 2 attention transforms

Paper §5.2.3 Algorithm 2:

- Per-attention-head intra-head: `R̂_qk` (2D rotary obfuscation on
  the RoPE-block axis), `Ĥ_qk` (2-block scaling), `Ẑ_block` (RoPE
  block permutation via BlockPerm), `Û_vo` (V↔O paired transform).
- Inter-head: `τ_kv` (shuffles K/V heads), `τ_group` (shuffles
  Q/O heads within each GQA group).

For Qwen3 GQA(16,8): straightforward — no K=V tying complication. RoPE
freq_base = 1 000 000 (single value, no p-RoPE / hybrid-attention
complications). `BlockPerm` parameter `rope_base` directly takes
this value. No layer-class-specific config needed (vs Gemma 4 which
had two RoPE bases for local/global layers).

Defends against: ISA on attention scores specifically (not just
hidden states).

### 8. Embedding / head Gaussian noise (α_e, α_h)

Paper §5.2.2:

```
W̃_embed = Π · (W_e + α_e · ε_embed) · P̂_embed
W̃_head  = Q̂_head · (W_h + α_h · ε_head) · Π^T
```

ε_embed, ε_head sampled `N(0, σ² I)` where `σ_e = std(W_e)`,
`σ_h = std(W_h)`. Paper defaults `α_e = 1.0`, `α_h = 0.2`. Paper §7.3
shows VMA TTRSR drops from >30 % at α_e=0.5 to <15 % at α_e=1.0
— this is **load-bearing for VMA defence**.

In our rewriter: add noise immediately after the dequantisation step,
before the Algorithm 1 transforms. Use a seeded RNG for
reproducibility.

### 10. Attack benchmark

Port `sheng1feng/Aloepri/src/security_qwen/{vma,ia,isa,ima,nn,tfma,sda}.py`
into `evals/aloepri-attacks/` (already on the to-do list as Path 1's
M0.3 — but the source code lives in our Path 2 vendor tree, so we
own it). Run the seven attacks against the obfuscated Qwen3 1.7B
artifact, report TTRSR per attack. Acceptance: ≤ 15 % per paper
Table 2 (measured on Qwen2.5-14B; we're on Qwen3 1.7B which is
smaller — TTRSR may differ).

---

## Suggested skills for the next session

- **diagnose** if Gate A or B output is unexpected and root cause
  isn't obvious.
- **grill-me** before committing to a Gate-C accuracy budget — useful
  to stress-test whether the chosen task list is the right one before
  spending the days to build the harness.
- **handoff** at the end of Gate C — capture what e_C^AloePri
  actually was, and the decision (proceed / tune / stop) for the
  next handoff.

---

## Pointers

- Rewriter: [`../../python/aloepri-llm/obfuscate_qwen3_gguf.py`](../../python/aloepri-llm/obfuscate_qwen3_gguf.py)
- Vendored AloePri: `vendor/aloepri-py/` (gitignored,
  `sheng1feng/Aloepri @ 60e8ea3`)
- Vendored llama.cpp: `vendor/llama.cpp/` (gitignored, mainline);
  Qwen3 model class at `vendor/llama.cpp/src/models/qwen3.cpp`
- Commits on `path-2-aloepri-gemma` branch (most recent first):
  - `124703e` — first working AloePri obfuscation on Qwen3 1.7B
  - `532f3a7` — Gemma 4 deferred + handoff doc
  - `51363b7` — revert M2.3 verdict downgrade (§5.2.5 lives offline)
  - `b50f807` — M2.3 gate cleared (identity-pad on Gemma 4)
  - `7788c02` — M2.0 Vulkan baseline running for Gemma 4 E2B Q8_0
  - `f8c1482` — protocol doc + plan retarget to llama.cpp

---

*Updated 2026-05-18 as a handoff. Next agent: start with Gate A.*

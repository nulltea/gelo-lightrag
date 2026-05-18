# Path 2 — AloePri Offline-Rewrite Inference for Gemma E2B/E4B

> **Worktree:** `../private-rag-path-2/` (separate git worktree on
> branch `path-2-aloepri-gemma`).
>
> **Sibling plan:** [`path-1-gelo-gemma.md`](path-1-gelo-gemma.md)
> develops in the original worktree.
>
> **Shared framework:** [`private-inference-comparison-framework.md`](private-inference-comparison-framework.md).
>
> **Goal:** Implement AloePri's covariant-obfuscation private LLM
> inference as a **second private-inference option** for Gemma
> E2B/E4B, deliberately accepting a weakened threat model in
> exchange for scalability and TEE-independence. Produce
> performance, accuracy, and attack-resistance numbers comparable
> to Path 1 on the same models.

---

## 0. Status

| Date | Note |
|---|---|
| 2026-05-18 | Plan written. Pending kickoff. |

---

## 1. Handoff context — read this first if you are not the person who wrote this plan

### 1.1 What this project is

This repository implements **private RAG** — retrieval-augmented
generation where prompts and retrieved context are kept private
from the inference provider. The current production protocol is
**GELO+TwinShield**: a per-batch fresh Haar-uniform activation
mask applied inside a SEV-SNP CVM, with attention matmuls offloaded
to a co-located consumer GPU under the mask. See
[`../prototype/gelo.md`](../prototype/gelo.md) for the canonical
reference; [`../research/private-llm-inference-round-2.md`](../research/private-llm-inference-round-2.md)
for the surrounding research landscape.

### 1.2 Why we're adding AloePri

GELO's per-batch fresh randomness is the load-bearing security
property under our **openweight** threat model (the model weights
are public — anyone can download Gemma 4 from HuggingFace). But
GELO is:
- Tied to a TEE — requires a SEV-SNP CVM. Cannot run on standard
  cloud serving infrastructure.
- Validated only at 0.6B-parameter scale today; per-batch cost
  scales `O(d²)` (Householder QR), making it less attractive at
  frontier scales.
- Not compatible with vLLM / SGLang / production serving stacks —
  the TEE is in the loop per offload.

**AloePri** (arXiv 2603.01499, ByteDance, March 2026) takes a
different tradeoff:
- No TEE. Pure obfuscation.
- Offline-only protocol: client rewrites the model once, ships
  obfuscated weights to a normal vLLM/SGLang server, and during
  online inference only does a cheap O(seq_len) token-ID
  permutation.
- Validated to 671B parameters (DeepSeek-V3.1-Terminus).
- **In exchange:** static-key scheme (no per-request entropy);
  TTRSR 5–15% under published attacks given openweight knowledge
  (vs ~0% for GELO's per-batch fresh mask, theoretically).

We accept this weakened threat model deliberately as a second,
complementary deployment option for scenarios where TEE
unavailability or scaling cost rules out GELO. See
[`../research/aloepri-vs-gelo.md`](../research/aloepri-vs-gelo.md)
for the full technique-by-technique comparison.

### 1.3 What you should read first

In order:

1. **This plan**, top to bottom (you're here).
2. [`private-inference-comparison-framework.md`](private-inference-comparison-framework.md)
   — the shared evaluation framework. M0.1 (corpus), M0.2 (eval
   harness), M0.3 (attack harness) are produced by Path 1 in the
   sibling worktree; you consume them.
3. AloePri paper (`arXiv 2603.01499`) — read §3 (threat model),
   §4 (covariant obfuscation theory), §5 (concrete construction),
   §6 (Rényi-mDP analysis), §7 (experiments). Most relevant
   sections: §5.2 (offline obfuscation) and §5.3 (online phase).
4. [`../research/aloepri-vs-gelo.md`](../research/aloepri-vs-gelo.md)
   — our analysis of AloePri vs GELO, particularly §2 (threat
   model) and §3 (technique-by-technique).
5. AloePri reference code: `github.com/sheng1feng/Aloepri` (commit
   `60e8ea3` is the version analyzed in our doc). This is a
   **community reproduction**, not the official ByteDance fork.
   Source files in `src/` map to paper sections:
   - `keymat.py` → Algorithm 1
   - `obfuscate_attention_complex.py` → Algorithm 2
   - `obfuscate_ffn.py` → FFN obfuscation
   - `obfuscate_rmsnorm.py` → RMSNorm covariant obfuscation
   - `obfuscate_embed_head.py` → §5.2.2
   - `security_qwen/{vma,ima,isa,...}.py` → §7 attack suite
   - `stage_*` files are the author's incremental build artifacts;
     mostly skip these.

### 1.4 What you should NOT do

- **Do not attempt to make AloePri "GELO-equivalent" in security
  properties.** They're different threat models. AloePri's static
  obfuscation is structurally weaker; that's the accepted tradeoff.
  Document the limitation in M2.7 attack-resistance results.
- **Do not modify GELO-side code.** Path 1's directories are
  off-limits (see §5 disjoint-directory contract below).
- **Do not commit to master without a sync with Path 1.** Shared
  M0.* infrastructure changes go through PR to master.
- **Do not extend AloePri to model-weight privacy.** The paper §8
  mentions CPU-TEE + FHE extensions for that scenario; we are
  explicitly not pursuing this. The user has stated: "we do not
  care about model weights privacy at this stage of the project."

### 1.5 Current architecture summary (so you don't have to dig)

```
Path 2 deployment shape:
  Client (any device, no TEE):
    1. Offline (one-time): obfuscate Gemma 4 model:
       - Sample secret token-permutation τ ∈ S_262144
       - Run Algorithm 1 to make key matrices {P̂_i, Q̂_i}
       - Run Algorithm 2 to obfuscate attention weights
       - Apply token-perm + key-matrix transforms to all weights
       - Permute PLE table by τ
       - Add Gaussian noise α_e·ε to W_e, α_h·ε to W_h
       - Ship θ̃ to server
    2. Online (per prompt):
       - Tokenize prompt locally
       - Map token IDs through τ
       - Send permuted IDs to vLLM server (running θ̃)
       - Receive obfuscated response IDs
       - Map through τ⁻¹
       - Detokenize

  Server (untrusted, no TEE):
    - Standard vLLM/SGLang stack
    - Runs unmodified Gemma 4 inference on θ̃
    - Never sees plaintext tokens or original weights
```

The key insight: **after offline rewrite, the obfuscated model is
structurally a normal LLM that vLLM serves with zero modification.**
This is AloePri's superpower — it composes with existing serving
infrastructure.

### 1.6 Compatibility caveats (Gemma 4 specific)

Round 2 research established that AloePri works with Gemma 4
**with three adjustments**:

1. **K=V global-layer trick must be un-tied.** Gemma 4 stores K
   and V as the same tensor in global layers (memory optimization).
   AloePri's Algorithm 2 applies different transforms to K vs V
   (R̂_qk applies to K only; Û_vo applies to V only). Resolution:
   un-tie K and V *after* obfuscation, store them separately on
   the server (~10% extra KV cache memory in global layers, which
   is ~1.5–2% of total memory). Do NOT set Q̂_k = Q̂_v as a
   workaround — that introduces an undocumented security
   degradation.

2. **PLE table must be token-permuted.** Gemma 4 has a per-layer
   embedding table `[262144 × 256 × N_layers]` indexed by token ID.
   The offline rewrite must permute the first axis by τ:
   `W̃_PLE[i, ℓ, :] = W_PLE[τ(i), ℓ, :]`. The PLE projection
   matrices (per-layer, project 256 → d_hidden) get the standard
   AloePri key-matrix transform on the d_hidden axis.

3. **p-RoPE in global layers**. Gemma 4 applies RoPE only to the
   first p·d_head dims (p=0.25). Algorithm 2's R̂_qk (a 2D rotation
   in RoPE space) must be applied only to the rotated dim subset.
   On the non-rotated subset, R̂_qk should be identity.

### 1.7 Open risk: vLLM Gemma 4 support timing

vLLM (and SGLang) need a Gemma 4 model implementation in their
mainline to serve our obfuscated model. Gemma 4 released Q1 2026;
today is 2026-05-18. **Verify support before M2.3 kickoff:**

```bash
pip install vllm
python -c "from vllm import LLM; LLM(model='google/gemma-4-E4B', dtype='float16')"
```

If this errors with "unsupported model", **fall back to Gemma 3**
(1B → 4B same hybrid family, no PLE). Document the fallback in
M2.1 acceptance criteria and inform Path 1 + comparison framework
of the change. The Gemma 3 path is strictly easier (no PLE complications).

---

## 2. Baseline state

What already exists:

- Nothing AloePri-related in this repo today.
- AloePri reference code at `github.com/sheng1feng/Aloepri` (Python,
  ~120 KB of obfuscation + attack code).
- Path 2 will vendor this as a Python dep with pinned commit.

What needs to be added:

1. Import AloePri reference code as vendored dep
2. Adapt obfuscation pipeline for Gemma 4 (hybrid attention, K=V,
   PLE, p-RoPE)
3. Build offline weight-rewriter CLI
4. Verify vLLM Gemma 4 support → run obfuscated model
5. Build client wrapper (tokenize → τ → server → τ⁻¹ → detokenize)
6. Run benchmarks and accuracy harness (shared M0.2)
7. Run attack-resistance suite (shared M0.3)

---

## 3. Milestones

### M2.0 — Worktree + environment setup

**Scope:** Verify the worktree is correctly created from master.
Set up Python environment for AloePri vendored code.

**Files to add:**
- `python/path-2/pyproject.toml` — venv definition
- `python/path-2/requirements.txt` — pinned deps
- `python/path-2/.python-version`

**Acceptance:**
- `python -V` reports 3.11+
- `pip install -e .` succeeds
- `pytest python/path-2/tests/smoke.py` (a one-line "import works")
  passes

**Effort:** 0.5 weeks.

**Dependencies:** None.

---

### M2.1 — AloePri code import + Gemma 4 adapter

**Scope:** Vendor `sheng1feng/Aloepri` at commit `60e8ea3` into
`vendor/aloepri-py/`. Build a Gemma 4 adapter on top of
`src/model_loader.py` that handles:
- Layer count 35 (E2B) or 42 (E4B)
- Per-layer attention class (local-512 vs global-8K)
- K=V global layers — flagged for un-tying in M2.2
- PLE table layout
- p-RoPE config

**Files to add:**
- `vendor/aloepri-py/` (git submodule or copy with LICENSE
  preservation)
- `python/path-2/gemma4_adapter.py` — Gemma 4 → AloePri model
  interface

**vLLM verification step (gates downstream work):**
```bash
python -c "from vllm import LLM; LLM(model='google/gemma-4-E4B', dtype='float16')"
```
If unsupported, **stop and pivot to Gemma 3** before proceeding.
Update `docs/plans/path-2-status.md` with the fallback decision.

**Acceptance:**
- Adapter loads Gemma 4 E2B and E4B weights from HuggingFace cache
  (or local safetensors) into AloePri's internal representation.
- vLLM imports + initializes the unobfuscated model successfully.
- (If fallback): Gemma 3 4B initializes; document the substitution.

**Effort:** 2 weeks.

**Dependencies:** M2.0.

**Risk:** Moderate. vLLM support is the gating uncertainty.

---

### M2.2 — Offline model obfuscation pipeline

**Scope:** Implement the full offline rewrite for Gemma 4, with
the three adjustments from §1.6.

**Sub-steps:**

#### M2.2a — Algorithm 1 + Algorithm 2 for Gemma 4 dimensions

Adapt `keymat.py` + `obfuscate_attention_complex.py` to Gemma 4's
hidden sizes (1536 / 2560), head dims (256), GQA layout (8-to-1),
and per-layer attention classes.

#### M2.2b — K=V global-layer un-tying

For each global layer, store the obfuscated K and V tensors
**separately** in the output safetensors, even though the input
had them merged. Document the memory delta in
`results/path-2-memory.json`.

#### M2.2c — PLE table permutation

`W̃_PLE[i, :, :] = W_PLE[τ(i), :, :]`. Also apply the standard
key-matrix right-multiplication on the 256-dim axis for each PLE
projection matrix.

#### M2.2d — p-RoPE-aware R̂_qk

In `obfuscate_attention_complex.py`, modify the R̂_qk construction
to apply 2D rotation only to the first `p · d_head` dims (per
global layer). Non-rotated dims get identity.

#### M2.2e — FFN, RMSNorm, embed/head obfuscation

Mostly reuse AloePri reference code. Verify each runs cleanly on
Gemma 4 weight shapes.

**Files to add:**
- `python/path-2/obfuscate_gemma4.py` — top-level offline rewriter
  CLI
- `python/path-2/lib/{algorithm_1,algorithm_2,k_eq_v,ple,p_rope}.py`
  — Gemma-specific deltas, each thin wrapper around vendored code

**Acceptance:**
- Offline rewrite of E2B and E4B completes; outputs valid
  safetensors readable by vLLM.
- **Forward-parity test:** for a plain prompt tokenized to
  `[t_1, t_2, ...]`, the obfuscated model fed `[τ(t_1), τ(t_2), ...]`
  produces logits that, after un-permutation, match plain logits
  to within `e_C^AloePri` (paper-bounded; expect cosine similarity
  ≥ 0.95 on hidden states at each layer, top-1 token match ≥ 0.97).
- Key-permutation never leaks: τ is sampled inside the CLI, written
  to a `.key` file with restrictive perms (0600), never logged.

**Effort:** 3 weeks.

**Dependencies:** M2.1.

**Risk:** Moderate. The K=V un-tying and PLE permutation are
non-trivial new code; rest is adaptation.

---

### M2.3 — vLLM integration

**Scope:** Serve the obfuscated model via vLLM. If vLLM mainline
supports Gemma 4 cleanly, this is mostly configuration. If
mainline doesn't yet support Gemma 4, options:
1. Wait for vLLM support (track upstream)
2. Backport from PR
3. Use SGLang as alternative
4. Pivot to Gemma 3 (most reliable fallback)

**Files to add:**
- `scripts/path-2/serve.sh` — launches vLLM with obfuscated
  weights + config
- `python/path-2/vllm_config.py` — model card + serving params

**Acceptance:**
- vLLM serves the obfuscated E2B and E4B at `http://localhost:8000`.
- A single permuted-token-ID prompt round-trips through the server
  and returns a plausible response.
- TPOT and TTFT measured on a 10-prompt smoke test.

**Effort:** 2 weeks (mainline support clean); up to 4 weeks if
backporting needed.

**Dependencies:** M2.2.

**Risk:** **High**. vLLM Gemma 4 support timing is the biggest
schedule risk.

---

### M2.4 — Client wrapper

**Scope:** Build the client-side tokenize → τ → request → τ⁻¹ →
detokenize pipeline. Both Python and Rust client (Rust for
integration with the existing `gelo-rag` flow).

**Files to add:**
- `python/path-2/aloepri_client.py` — async HTTP client
- `crates/aloepri-client/` (optional Rust crate for parity with
  `gelo-rag` ingestion paths)

**Acceptance:**
- Python client tokenizes, permutes, sends to vLLM server, receives
  obfuscated response, un-permutes, returns plaintext.
- Round-trip latency ≤ TPOT + RTT (no measurable client overhead).
- (Optional Rust crate) integration test against the running vLLM
  server.

**Effort:** 1 week.

**Dependencies:** M2.3.

**Risk:** Low.

---

### M2.5 — E2B end-to-end benchmark

**Scope:** Run shared M0.1 corpus on E2B-AloePri using shared M0.2
harness. Same metrics as Path 1 M1.6.

**Acceptance:**
- E2B + AloePri runs the Tier 1 smoke corpus end-to-end.
- TPOT overhead vs plain Gemma E2B is within paper's <10% claim.
- Results to `results/path-2-e2b.json`.

**Effort:** 0.5 weeks.

**Dependencies:** M2.4 + M0.2.

**Risk:** Low.

---

### M2.6 — E4B scaling benchmark

**Scope:** Same on E4B. Measure scaling delta.

**Acceptance:**
- E4B + AloePri runs.
- Scaling delta E2B → E4B documented; expected near-linear (AloePri
  paper shows this through 671B).

**Effort:** 0.5 weeks.

**Dependencies:** M2.5.

**Risk:** Low.

---

### M2.7 — Accuracy validation

**Scope:** Run M0.2 eval harness against AloePri E2B and E4B.

**Acceptance:**
- MMLU / IFEval / PIQA / HumanEval accuracy measured.
- Accuracy delta vs plain documented. Paper claims 0–3.5% loss
  on Qwen2.5/Qwen3/Llama3/DeepSeek; Gemma 4 is novel. Expect
  similar or possibly higher (Gemma 4 not in paper's tested set).
- If accuracy loss exceeds 5pp on any benchmark, investigate
  hyperparameter tuning: λ, h, α_e, α_h, β, γ (defaults in paper
  §7.1).

**Effort:** 1 week.

**Dependencies:** M2.5.

**Risk:** Moderate. Gemma 4's PLE machinery is novel to AloePri;
the permuted-PLE adaptation may introduce more drift than the
paper's tested models.

---

### M2.8 — Attack-resistance benchmark (M0.3 wiring)

**Scope:** Wire M0.3 attack harness against AloePri-served Gemma.
Capture obfuscated-token streams (and, with care, intermediate
hidden states from a debugging build of vLLM) for VMA / IA / ISA /
IMA / NN / TFMA / SDA.

**Acceptance:**
- TTRSR measured per attack on E2B and E4B.
- Expected to land in 5–15% range per AloePri paper.
- If significantly worse (>30%), investigate hyperparameter
  configuration.

**Effort:** 2 weeks (after M0.3 lands).

**Dependencies:** M0.3, M2.5, M2.6.

**Risk:** Moderate. The paper's TTRSR numbers are on
Qwen/Llama/DeepSeek; Gemma 4 with PLE may behave differently
under TFMA/SDA in particular (PLE is an additional leak axis).

---

### M2.9 — (Stretch) Gemma 4 31B dense

**Scope:** Run M2.5–M2.8 on Gemma 4 31B.

**Acceptance:**
- 31B obfuscated weights serve via vLLM.
- Performance numbers match scaling expectations.
- Accuracy delta similar to E2B/E4B (paper claims AloePri scales
  smoothly).

**Effort:** 1 week.

**Dependencies:** M2.7.

**Risk:** Low (AloePri's scaling has been validated to 671B).

---

## 4. Aggregate effort

| Milestone | Effort (weeks) | Cumulative |
|---|---:|---:|
| M2.0 | 0.5 | 0.5 |
| M2.1 | 2.0 | 2.5 |
| M2.2 | 3.0 | 5.5 |
| M2.3 | 2.0 (clean) – 4.0 (backport) | 7.5–9.5 |
| M2.4 | 1.0 | 8.5–10.5 |
| M2.5 | 0.5 | 9.0–11.0 |
| M2.6 | 0.5 | 9.5–11.5 |
| M2.7 | 1.0 | 10.5–12.5 |
| M2.8 | 2.0 (after M0.3) | 12.5–14.5 |
| M2.9 stretch | +1.0 | 13.5–15.5 |

**Total: ~10.5–14.5 weeks v1 (E2B + E4B); 13.5–15.5 weeks with 31B
stretch.** Plus shared work (M0.* coordination, M0.4 comparison
report).

The wide range reflects the vLLM Gemma 4 support uncertainty.

---

## 5. Disjoint-directory contract with Path 1

To minimize merge pain, Path 2 only writes to:

- `vendor/aloepri-py/**` (vendored AloePri Python code)
- `python/path-2/**` (new Python code)
- `crates/aloepri-client/**` (optional Rust client)
- `scripts/path-2/**`
- `docs/plans/path-2-*.md`
- `results/path-2-*.json`

Path 1 only writes to:

- `crates/gelo-embedder/**`, `crates/gelo-protocol/**`,
  `crates/gelo-gpu-wgpu/**`, `crates/gelo-snp-runner/**`
- `evals/private-inference-corpus/**` (M0.1)
- `evals/run-eval.py` + `evals/lib/**` (M0.2)
- `evals/attack-harness/**` (M0.3)
- `docs/plans/path-1-*.md`
- `results/path-1-*.json`

If Path 2 needs changes to Path-1-owned files (e.g., attack-harness
API), file a PR to master. Path 1 reviews + merges before Path 2
consumes.

---

## 6. Open questions / decisions deferred

- **Vendoring approach for AloePri**: git submodule vs copy with
  LICENSE preservation. Submodule is cleaner but adds setup
  friction. Decide at M2.0 kickoff.
- **AloePri hyperparameter tuning on Gemma 4**: paper's defaults
  (λ=0.3, h=128, α_e=1.0, α_h=0.2, β=8, γ=1000) were tuned for
  Qwen2.5 / Llama3 / DeepSeek. Gemma 4 may benefit from re-tuning.
  Schedule as a follow-on after M2.7 if accuracy is unexpectedly
  poor.
- **Multimodal**: Gemma E2B/E4B have native audio/vision encoders
  (~150–300M params each). Out of scope for v1; document as
  future work.
- **MoE (Gemma 4 26B A4B)**: out of scope; requires CryptoMoE
  balanced-dispatch defense (round 2 §C). Future work.

---

## 7. Getting started — concrete first commands

In the worktree (`../private-rag-path-2/`):

```bash
# 1. Verify worktree state
git status
git log --oneline -5

# 2. Read the handoff and shared framework
less docs/plans/path-2-aloepri-gemma.md            # this file
less docs/plans/private-inference-comparison-framework.md
less docs/research/aloepri-vs-gelo.md
less docs/research/private-llm-inference-round-2.md

# 3. Pull reference code
mkdir -p vendor
git clone --branch main https://github.com/sheng1feng/Aloepri \
    vendor/aloepri-py
( cd vendor/aloepri-py && git checkout 60e8ea3 )

# 4. Verify the paper PDF is accessible (it's indexed in EdgeQuake;
#    or download from arXiv 2603.01499)

# 5. M2.0: set up Python env
mkdir -p python/path-2
cd python/path-2
python -m venv .venv
source .venv/bin/activate
pip install vllm transformers torch safetensors

# 6. M2.1 gating: verify vLLM Gemma 4 support
python -c "from vllm import LLM; LLM(model='google/gemma-4-E4B', dtype='float16')"
# If this errors: stop, update path-2-status.md with fallback decision
```

---

## 8. References

- [`private-inference-comparison-framework.md`](private-inference-comparison-framework.md)
  (shared framework)
- [`path-1-gelo-gemma.md`](path-1-gelo-gemma.md) (sibling)
- [`../prototype/gelo.md`](../prototype/gelo.md) — GELO protocol
  reference (background)
- [`../research/private-llm-inference-round-2.md`](../research/private-llm-inference-round-2.md)
  §D (Gemma 4 architecture)
- [`../research/aloepri-vs-gelo.md`](../research/aloepri-vs-gelo.md)
  — full AloePri-vs-GELO comparison
- AloePri paper: arXiv 2603.01499
- AloePri reference code:
  `github.com/sheng1feng/Aloepri` commit `60e8ea3`

---
type: plan
status: current
created: 2026-05-18
updated: 2026-05-21
tags: [path-2, aloepri, gemma]
companion: [path-2-status]
---

# AloePri Private LLM Inference — Gemma 4 E2B / E4B on llama.cpp

> **Worktree:** `../private-rag-path-2/` (separate git worktree on
> branch `path-2-aloepri-gemma`).
>
> **Sibling plan:** [`path-1-gelo-gemma.md`](path-1-gelo-gemma.md) (GELO
> + TEE) develops in the original worktree. Shared evaluation
> framework: [`private-inference-comparison-framework.md`](private-inference-comparison-framework.md).
>
> **Protocol reference:** [`../prototype/aloepri-llm.html`](../prototype/aloepri-llm.html)
> — the canonical protocol doc, kept in sync with this plan.
>
> **Goal:** Implement AloePri covariant-obfuscation private LLM
> inference for Gemma 4 E2B (primary) then E4B (port), served by an
> unmodified llama.cpp / `llama-server` binary. Deliberately accepting
> a weakened threat model in exchange for TEE-independence, CPU-first
> deployment, and serving-stack simplicity.

---

## 0. Status

| Date | Note |
|---|---|
| 2026-05-18 | Plan rewritten for llama.cpp serving target (was vLLM); E2B-first sequencing; BF16 baseline + Q8_0 production quantisation; single-user / per-tenant deployment. |
| 2026-05-18 | E2B GGUF weights download in progress (user) — local path TBD. |

---

## 1. Handoff context — read this first if you're new

### 1.1 What this project is

This repository implements **private RAG** — retrieval-augmented
generation where prompts and retrieved context are kept private from
the inference provider. Embedding, storage, and reranking run inside
an SEV-SNP CVM with GELO + TwinShield masked offload to a commodity
GPU (see [`../dev/prototype/gelo.md`](../dev/prototype/gelo.md),
[`../prototype/reranking.html`](../prototype/reranking.html)).

**This plan adds the generative LLM step** — the post-retrieval LLM
that consumes the reranked chunks and produces an answer. It does so
with a categorically different trust model from the rest of the
stack: no TEE, no GPU masking, no per-batch fresh randomness. The
single trust anchor is the client process that holds the secret
permutation τ.

### 1.2 Why AloePri + llama.cpp

GELO's per-batch fresh Haar mask is the load-bearing security
property for embedder + reranker — it must keep working at small
scales on consumer hardware. But generation has a different cost
shape:

- Models are 5–50× larger than the embedder.
- Decode is autoregressive: per-token mask resampling and per-offload
  PCIe round-trips dominate wall-clock.
- The CVM + commodity-GPU envelope that works for ~0.6B models gets
  expensive fast at ≥4B.

**AloePri** (Lin et al., arXiv 2603.01499, ByteDance, March 2026)
moves the entire forward pass to a normal server in exchange for an
empirical security guarantee instead of an information-theoretic one:

- One-shot offline weight rewrite produces an obfuscated artifact.
- Per request, the client does a token-ID permutation; the server
  runs unmodified inference.
- Validated to 671B parameters in the paper.
- Empirically: TTRSR ≤ 15% under VMA / IA / ISA / IMA / NN / TFMA /
  SDA at recommended hyperparameters (paper Table 2).

**llama.cpp** is the right serving target for our deployment envelope:

- Pure C++ binary; no Python serving stack.
- CPU-first; runs on commodity EPYC with no GPU passthrough required.
- Gemma 3 + Gemma 3n upstream support (PLE machinery already in tree).
- GGUF metadata reads `hidden_size` per artifact, so AloePri's
  expanded internal dim (`d → d+2h`) is transparent if the kernel
  doesn't assume `hidden_size == n_heads · head_dim`.
- Native Q8_0 quantisation, which is our production target.
- `llama-server` exposes an OpenAI-compatible HTTP API — the client
  wrapper code is identical to what we'd write against vLLM.

See [`../research/aloepri-vs-gelo.md`](../research/aloepri-vs-gelo.md)
for the full technique-by-technique comparison; [`../prototype/aloepri-llm.html`](../prototype/aloepri-llm.html)
for the canonical protocol and trust-boundary documentation.

### 1.3 What you should read first

In order:

1. **This plan**, top to bottom.
2. [`../prototype/aloepri-llm.html`](../prototype/aloepri-llm.html) —
   protocol reference (threat model, components, compute flow).
3. [`private-inference-comparison-framework.md`](private-inference-comparison-framework.md)
   — shared evaluation framework with the GELO path. M0.* (corpus,
   eval harness, attack harness) are produced by Path 1; we consume.
4. AloePri paper (arXiv 2603.01499) — §3 (threat model), §4 (covariant
   obfuscation theory), §5 (concrete construction), §6 (RmDP), §7
   (experiments). Most relevant: §5.2 (offline obfuscation), §5.3
   (online inference).
5. [`../research/aloepri-vs-gelo.md`](../research/aloepri-vs-gelo.md)
   — the analysis we did before committing to this path.
6. AloePri reference code: `github.com/sheng1feng/Aloepri` at commit
   `60e8ea3`. **Qwen2-architected.** It is not a drop-in for Gemma
   4 — see §1.6 for the deltas.
7. llama.cpp Gemma support: read `src/llama-model.cpp` for the Gemma
   model class registration and the per-architecture forward graph;
   check `examples/server/` for the HTTP server entry point.

### 1.4 What you should NOT do

- **Do not attempt to make AloePri "GELO-equivalent" in security
  properties.** Different threat models, different guarantees.
  AloePri's static obfuscation is structurally weaker; that's the
  accepted trade. Document the boundary; don't try to erase it.
- **Do not modify the GELO-side code.** Path 1's directories are
  off-limits (§5 disjoint-directory contract).
- **Do not extend AloePri to model-weight privacy.** Paper §8
  discusses CPU-TEE + FHE extensions; we are explicitly not pursuing
  this. User stated: "we do not care about model weights privacy at
  this stage of the project."
- **Do not fork llama.cpp until you have measured whether you need
  to.** First confirm that an obfuscated artifact with
  `hidden_size = d + 2h` loads and runs through the stock Gemma
  forward pass. Only fork if that proves impossible.

### 1.5 Current architecture summary

```
Per-request flow:

  Client process (trusted; no TEE; holds τ):
    1. tokenize(plaintext_prompt) → ids
    2. ids' = [Z[i] for i in ids]                          # τ-map
    3. obf_text = tokenizer.decode(ids')                   # detok back to STRING
    4. POST http://llama-server:8080/v1/completions
         body: {"prompt": obf_text, ...}
    5. resp_text ← HTTP response
    6. resp_ids = tokenizer.encode(resp_text)
    7. plain_ids = [Z⁻¹[i] for i in resp_ids]
    8. return tokenizer.decode(plain_ids)

  llama-server (untrusted; no TEE; pure C++ binary):
    - loads obfuscated Gemma 4 E2B/E4B GGUF
    - tokenises obf_text → same ids' the client produced
    - runs unmodified Gemma forward pass over the obfuscated weights
    - samples next obfuscated token ID
    - detokenises → response string
    - returns string to client
```

The server-side weights have already absorbed every AloePri
transform (Π, P̂, Q̂, R̂_qk, Ĥ_qk, Ẑ_block, Û_vo, α_e·ε, α_h·ε, the
PLE vocab-axis permutation, and the K and V untie). The forward
pass is structurally a normal Gemma forward. **No protocol-aware
code on the server side.**

### 1.6 Gemma-4-specific deviations from the AloePri paper

The paper evaluates Qwen2.5 / Qwen3 / Llama3 / DeepSeek — none of
which has PLE, K=V tying, p-RoPE, or hybrid attention. Three
extensions to the offline construction:

1. **K=V untie in global layers.** Gemma 4 stores K and V as one
   tensor in global layers. Algorithm 2 applies different transforms
   (`R̂_qk · Ĥ_qk⁻¹ · Ẑ_block^η` to K; `Û_vo` to V). Resolution:
   un-tie K and V into separate GGUF tensors during offline rewrite.
   Inter-head `τ_kv` still permutes them in lockstep. Memory: ~16%
   extra on global-layer KV cache, ≈ 3% of total. **Do NOT** use
   `Q̂_k = Q̂_v` as a workaround — undocumented security degradation.

2. **PLE table token-axis permutation.** Gemma 4 E2B/E4B have a PLE
   table of shape `[262144 × 256 × N_layers]`. Permute the vocab
   axis: `W̃_PLE[i, :, ℓ] = W_PLE[τ(i), :, ℓ]`. Per-layer projection
   (256 → d_hidden) gets right-multiplied by **the same P̂_ℓ** the
   layer-ℓ attention block outputs into — otherwise the PLE
   contribution doesn't land in the layer's obfuscated residual
   space and the composition theorem breaks.

3. **p-RoPE-aware R̂_qk.** Gemma 4 global layers apply RoPE only to
   the first `p · d_head` dims (p=0.25). R̂_qk must be a partial
   rotation (identity on non-rotated dims); Ĥ_qk and Ẑ_block
   likewise restricted. BlockPerm's `rope_base`-dependent decay
   needs two configs per layer-class (10K local, 1M global).

### 1.7 Hidden risk: the +10% internal expansion (h=128 default)

AloePri's Algorithm 1 key matrices grow the per-block residual /
projection input dim from `d` to `d + 2h`. With paper-default
`h=128`:

- E2B: d=1536 → d+2h=1792 (+16.7%)
- E4B: d=2560 → d+2h=2816 (+10.0%)

This means the obfuscated GGUF must advertise `hidden_size = d + 2h`,
not the plaintext value. **First milestone-zero validation** (M2.1):
load a plaintext Gemma 4 E2B GGUF, change the `hidden_size`
metadata to `d + 2h` without changing any tensor data, and confirm
llama.cpp either (a) errors out cleanly, or (b) runs without
asserting `hidden_size == n_heads · head_dim` somewhere internally.
This determines whether we need to fork llama.cpp or not.

### 1.8 Hidden risk: Q8_0 + AloePri noise stacking

User-decided production quantisation is Q8_0. AloePri intentionally
adds `α_e · ε` Gaussian noise to embed + `α_h · ε` to head. Q8_0
adds ~0.3% relative quantisation error on top. The paper's TTRSR
and accuracy bounds were measured at BF16. Two specific risks:

- **VMA may strengthen.** At α_e=0.5, paper §7.3 already shows VMA
  jumping to >30% TTRSR. Q8 quantisation effectively reduces the
  noise floor that VMA must overcome.
- **Accuracy may degrade past the paper's 3.5% ceiling.** Need to
  re-measure on Gemma 4 at our hyperparameters before committing
  Q8 to production.

Mitigation: BF16 baseline lands first (M2.6); Q8_0 lands behind a
gate that checks both accuracy delta and VMA TTRSR vs the BF16
baseline (M2.7).

---

## 2. Baseline state

What already exists:

- Nothing AloePri-related in this repo today.
- AloePri reference code at `github.com/sheng1feng/Aloepri` @
  `60e8ea3` — **Qwen2-architected**. The seven attacks in
  `src/security_qwen/` are reusable. Most of the model-side code is
  not (different model family, different framework).
- llama.cpp upstream (verify per M2.0): Gemma 3 supported; Gemma 3n
  partial support (PLE); Gemma 4 — unknown at time of plan-writing,
  verify at kickoff.
- E2B GGUF weights download in progress (user) — local path to be
  filled in at M2.0 kickoff.

What needs to be added:

1. llama.cpp upstream support verification (M2.0)
2. Vendored AloePri reference at pinned commit (M2.1)
3. Offline GGUF rewriter for Gemma 4 (M2.2)
4. Server-side validation (M2.3) — does the stock llama.cpp Gemma 4
   kernel accept `hidden_size = d + 2h`?
5. Client wrapper (M2.4)
6. E2B end-to-end (M2.5)
7. Accuracy + quantisation gate at BF16 → Q8_0 (M2.6 → M2.7)
8. Port to E4B (M2.8)
9. Attack-resistance benchmark (M2.9)

---

## 3. Milestones

### M2.0 — Worktree, environment, upstream verification

**Scope:** Stand up the Python environment for offline obfuscation;
verify llama.cpp Gemma 3n / Gemma 4 support; confirm where the E2B
GGUF weights live.

**Files to add:**
- `python/aloepri-llm/pyproject.toml` · `requirements.txt` · `.python-version`
- `docs/dev/logs/path-2-status.md` — running notes log

**Concrete checks:**

```bash
# (a) Python env
mkdir -p python/aloepri-llm && cd python/aloepri-llm
python3.11 -m venv .venv && source .venv/bin/activate
pip install gguf safetensors torch numpy

# (b) llama.cpp build
git clone https://github.com/ggml-org/llama.cpp vendor/llama.cpp
cd vendor/llama.cpp && cmake -B build -DGGML_NATIVE=ON && cmake --build build -j

# (c) Verify Gemma E2B loads under stock llama.cpp
./build/bin/llama-cli -m <PATH_TO_E2B_GGUF> -p "hello" -n 8

# (d) Read the Gemma model class in src/llama-model.cpp; identify
#     where hidden_size is read from GGUF metadata vs hardcoded.
#     Flag any assertion that hidden_size == n_heads * head_dim.

# (e) Verify gguf-py package can read + write E2B tensors
python -c "import gguf; r = gguf.GGUFReader('<PATH>'); print([t.name for t in r.tensors][:10])"
```

**Acceptance:**
- Plaintext E2B runs an 8-token completion under stock llama.cpp.
- `gguf-py` reads E2B tensors and metadata fields.
- Documented finding in `path-2-status.md`: does Gemma 4 architecture
  exist in llama.cpp mainline yet? If not, document the fallback
  (Gemma 3 4B has no PLE — strictly easier protocol-wise but loses
  the E-series PLE work).

**Effort:** 0.5 weeks.

**Dependencies:** E2B GGUF weights at a local path.

**Risk:** Moderate. Gemma 4 upstream support is the gating
uncertainty.

---

### M2.1 — AloePri reference vendoring + attack-suite isolation

**Scope:** Vendor `sheng1feng/Aloepri @ 60e8ea3` into
`vendor/aloepri-py/` with LICENSE preservation. Isolate the
`src/security_qwen/` attack suite into `evals/aloepri-attacks/` so
both paths can run it without dragging in the Qwen-specific model
code.

**Files to add:**
- `vendor/aloepri-py/` (git submodule or copy with LICENSE)
- `evals/aloepri-attacks/{vma,ia,isa,ima,nn,tfma,sda}.py` — thin
  wrappers around the vendored attack code
- `python/aloepri-llm/lib/keymat.py` — Python port of Algorithm 1
  (KeyMatGen / InvKeyMatGen) since the reference is Qwen-architected
  but the math is model-agnostic; the reference's `src/keymat.py`
  may be imported as-is

**Acceptance:**
- `vendor/aloepri-py/` contains the pinned commit with LICENSE.
- Sanity test: generate `P̂`, `Q̂` for `d=1536, h=128` and assert
  `‖P̂ · Q̂ - I_d‖_F < 1e-5`.
- Attack-suite Python imports cleanly outside the vendored package
  (no Qwen-model dependencies leak in).

**Effort:** 1 week.

**Dependencies:** M2.0.

**Risk:** Low.

---

### M2.2 — Offline GGUF rewriter for Gemma 4

**Scope:** Read plaintext E2B GGUF, apply the full AloePri offline
obfuscation, write obfuscated GGUF. Operates on GGUF tensors
directly via the `gguf` Python package (no safetensors detour). All
hyperparameters configurable; paper defaults baked in.

**Sub-steps:**

#### M2.2a — Algorithm 1 + Algorithm 2 for Gemma 4 dimensions

Port `keymat.py`, `attention_keys.py`, `obfuscate_attention_complex`
to Gemma 4 shapes: `d=1536` (E2B) / `2560` (E4B), `head_dim=256`,
GQA layout, per-layer attention class (local W=512 vs global).

#### M2.2b — K=V un-tying in global layers

For each global layer, read the merged K=V tensor, decompose into
separate K and V, apply the divergent Algorithm-2 transforms, write
two GGUF tensors. Update model metadata to flag K and V as separate.

#### M2.2c — PLE table vocab-axis permutation

```
W̃_PLE[i, :, ℓ] = W_PLE[τ(i), :, ℓ]   for i in 0..262143, ℓ in 0..N_layers-1
```
Per-layer projection matrix gets right-multiplied by `P̂_ℓ`
(the layer-ℓ residual key matrix, not a fresh one).

#### M2.2d — p-RoPE-aware R̂_qk / Ĥ_qk / Ẑ_block

For global layers (p=0.25), R̂_qk restricted to the first
`p · d_head = 64` dims; identity on the remaining 192 dims. Ĥ_qk
and Ẑ_block likewise. BlockPerm with `rope_base=1M`.

For local layers, R̂_qk over the full 256 head-dim; `rope_base=10K`.

#### M2.2e — FFN, RMSNorm, embed, head obfuscation

Per paper §5.2.2 + §5.2.4. RMSNorm's `γ` weight is folded into the
adjacent key matrix per the covariant construction (paper §5.2.4).
Embedding + head get noise (`α_e=1.0`, `α_h=0.2` by default) and Π.

#### M2.2f — GGUF metadata update

Set `gemma.hidden_size = d + 2h` (so 1792 for E2B). Keep
`gemma.head_dim = 256` and `gemma.attention.head_count` unchanged.
Bump artifact version; embed AloePri version + seed-hash in
metadata so the artifact is reproducible.

**Files to add:**
- `python/aloepri-llm/obfuscate_gemma4_gguf.py` — top-level CLI
- `python/aloepri-llm/lib/{attention,ffn,rmsnorm,embed,ple,p_rope}.py`
- `python/aloepri-llm/lib/key_material.py` — generate + persist
  `aloepri.key` (POSIX 0600)

**Acceptance:**
- Offline rewrite of E2B completes; outputs valid BF16 GGUF readable
  by `gguf-py`.
- Tensor shapes match expectation: residual-stream tensors at
  `d + 2h`; QKV / FFN tensors at the AloePri-derived shapes.
- `aloepri.key` is 0600, contains seed + τ + version tag.
- Bit-for-bit reproducibility: running the rewriter twice with the
  same seed produces identical output tensors.

**Effort:** 3 weeks.

**Dependencies:** M2.1.

**Risk:** Moderate. The K=V untie and PLE permutation are
non-trivial new code; everything else is mechanical.

---

### M2.3 — Server-side validation: does stock llama.cpp accept it?

**Scope:** The pivotal question. Load the obfuscated E2B GGUF into
stock llama.cpp and observe whether the forward pass runs without
asserting on shape mismatches.

**Three possible outcomes:**

1. **Best case — stock llama.cpp accepts it.** `hidden_size` is read
   from GGUF metadata; the Gemma kernel does not hardcode
   `hidden_size == n_heads · head_dim` anywhere; forward pass
   completes. We do not fork llama.cpp. Proceed to M2.4.

2. **Middle case — small patch required.** A specific assertion
   blocks the path but the underlying op handles non-square dims
   fine. Land a minimal patch upstream-style (a separate
   `vendor/llama.cpp-aloepri/` worktree with the patch on a branch)
   and proceed.

3. **Worst case — architecture fork required.** The forward graph
   needs structural changes to handle the expanded residual stream
   alongside head-projection dims that don't match. Add a new
   architecture class `gemma_aloepri` to the fork. Stop and reassess
   schedule before proceeding — this is a 4–6 week side-quest.

**Acceptance:**
- `llama-cli -m obfuscated-e2b.gguf -p "<obfuscated_test_prompt>"
  -n 8` produces *some* output (correctness is M2.5's job; this
  milestone is just "does it run").
- Documented outcome in `path-2-status.md`; if (2) or (3), patch /
  fork plan written.

**Effort:** 1 week (outcome 1) to 6 weeks (outcome 3). Plan as 2
weeks expected with a 4-week reserve.

**Dependencies:** M2.2.

**Risk:** **High** — this is the biggest schedule unknown.

---

### M2.4 — Client wrapper (Python)

**Scope:** The trusted-side library. Tokenise plaintext, map IDs
through Z, detokenise to obfuscated string, POST to `llama-server`'s
`/v1/completions` endpoint, decode response.

**Files to add:**
- `python/aloepri-llm/aloepri_client.py` — async + sync HTTP wrapper
- `python/aloepri-llm/lib/tokenizer_roundtrip.py` — fuzz harness
  asserting `tokenizer.encode(tokenizer.decode(ids)) == ids` for
  representative ID lists including edge cases (special tokens,
  byte-fallback, leading whitespace)
- `crates/aloepri-client/` (optional Rust crate for parity with
  `gelo-rag`'s ingestion path)

**Acceptance:**
- Roundtrip property holds for 1k random ID sequences drawn from
  the Gemma 4 vocab + 32 hand-crafted edge cases.
- Python client end-to-end: takes a plaintext prompt, hits a stock
  `llama-server` instance loaded with the obfuscated E2B GGUF,
  returns a plaintext response.
- Round-trip latency: client overhead ≤ 5 ms per request beyond
  network RTT + server TPOT.

**Effort:** 1 week.

**Dependencies:** M2.3.

**Risk:** Low. Main concern is tokenizer edge cases; the fuzz
harness catches them up front.

---

### M2.5 — E2B end-to-end correctness

**Scope:** Drive M2.4 + obfuscated-E2B-on-llama-server against the
shared Tier-1 smoke corpus (M0.1). Validate top-1 token agreement
and final-hidden-state cosine similarity vs plaintext-Gemma-E2B
running on the same llama.cpp build.

**Acceptance:**
- BF16 obfuscated E2B + AloePri client produces sensible completions
  on the Tier-1 smoke corpus.
- Top-1 token agreement vs plaintext baseline: ≥ 0.95 across the
  32 smoke prompts.
- No silent corruption: any prompt that fails the tokenizer
  roundtrip property is rejected at the client wrapper, not on the
  server.

**Effort:** 0.5 weeks.

**Dependencies:** M2.4 + M0.1.

**Risk:** Low.

---

### M2.6 — E2B BF16 accuracy gate

**Scope:** Run shared M0.2 eval harness against the BF16 obfuscated
E2B. Measure MMLU / IFEval / PIQA / HumanEval accuracy vs plaintext
baseline. Establishes the AloePri-only accuracy budget before
quantisation enters the picture.

**Acceptance:**
- Per-benchmark accuracy delta vs plaintext documented in
  `results/path-2-e2b-bf16.json`.
- Paper claims 0–3.5% loss on Qwen / Llama / DeepSeek — expect
  similar on Gemma 4 (architecture is novel; PLE is new). If
  accuracy loss > 5pp on any benchmark, investigate hyperparameter
  tuning (λ, h, α_e, α_h, β, γ).
- Document the discovered ceiling for use as the Q8_0 budget gate.

**Effort:** 1 week.

**Dependencies:** M2.5 + M0.2.

**Risk:** Moderate. PLE adaptation is novel; accuracy may drift more
than paper baselines on Qwen.

---

### M2.7 — Q8_0 production quantisation + VMA re-check

**Scope:** Quantise the obfuscated BF16 GGUF to Q8_0 via standard
`llama-quantize`. Re-run M2.6 eval harness; re-run VMA from the M0.3
attack harness against the Q8_0 artifact. Decide whether Q8_0 is
safe to ship as production default.

**Acceptance:**
- Q8_0 obfuscated E2B passes the same evals.
- Accuracy delta Q8_0 vs BF16 obfuscated ≤ 1pp on each benchmark
  (i.e. quantisation noise stacks linearly with AloePri noise at most).
- VMA TTRSR on Q8_0 ≤ paper's TTRSR ceiling × 1.5 (allow some
  degradation; reject if it exceeds 22.5%).
- If either gate fails: stay on BF16 for production, document why.

**Effort:** 1 week.

**Dependencies:** M2.6.

**Risk:** Moderate. The Q8_0 + AloePri noise interaction is novel —
the paper never tested it.

---

### M2.8 — Port to Gemma 4 E4B

**Scope:** Re-run M2.2 through M2.7 on Gemma 4 E4B. The protocol is
identical; the new content is:

- Tensor shapes: `d=2560`, `d+2h=2816`, 42 layers instead of 35,
  hybrid ratio 5:1 instead of 4:1.
- PLE table scales: ~1.3 GB int8 vs ~1.1 GB for E2B.
- Memory budget: E4B BF16 obfuscated ≈ 9 GB; Q8_0 ≈ 4.7 GB.

**Acceptance:**
- E4B BF16 + Q8_0 artifacts produced via the M2.2 pipeline (no new
  Gemma-specific code; just larger config).
- M2.6 + M2.7 gates pass on E4B with the same thresholds.
- Scaling delta E2B → E4B documented in `results/path-2-scaling.json`.

**Effort:** 1.5 weeks.

**Dependencies:** M2.7.

**Risk:** Low. Architecture is identical; only scale changes.

---

### M2.9 — Attack-resistance benchmark

**Scope:** Wire M0.3 attack harness against AloePri-served Gemma 4
E2B and E4B. Capture obfuscated-token streams; for ISA / IMA / NN,
capture intermediate hidden states from a debug build of llama.cpp
(or instrument via `llama-cpp-python` bindings).

Specific attacks per AloePri paper Table 2:
- **VMA, IA** — exploit plaintext-vs-obfuscated weight pairs
- **ISA, IMA, NN** — train inverters on internal states
- **TFMA, SDA** — token-frequency exploits
- **PLE-TFMA (novel)** — observe per-layer PLE gather addresses
  across N_layers · seq_len observations per prompt; check whether
  the additional observation volume increases TFMA effectiveness
  vs the paper's Qwen-baseline numbers. **This is novel work** —
  paper does not evaluate it.

**Acceptance:**
- TTRSR ≤ 15% per attack on E2B and E4B at paper-recommended
  hyperparameters.
- PLE-TFMA result documented honestly: if it exceeds the paper's
  20% Top-100 ceiling, this is a Gemma-specific weakness of
  AloePri and goes in [`../research/aloepri-vs-gelo.md`](../research/aloepri-vs-gelo.md).

**Effort:** 2 weeks.

**Dependencies:** M0.3, M2.8.

**Risk:** Moderate. PLE-TFMA is novel; outcome shapes how aggressive
the §1.6 PLE-permutation defense is.

---

## 4. Aggregate effort

| Milestone | Effort (weeks) | Cumulative (low) | Cumulative (high) |
|---|---:|---:|---:|
| M2.0 | 0.5 | 0.5 | 0.5 |
| M2.1 | 1.0 | 1.5 | 1.5 |
| M2.2 | 3.0 | 4.5 | 4.5 |
| M2.3 | 1.0–6.0 | 5.5 | 10.5 |
| M2.4 | 1.0 | 6.5 | 11.5 |
| M2.5 | 0.5 | 7.0 | 12.0 |
| M2.6 | 1.0 | 8.0 | 13.0 |
| M2.7 | 1.0 | 9.0 | 14.0 |
| M2.8 | 1.5 | 10.5 | 15.5 |
| M2.9 | 2.0 | 12.5 | 17.5 |

**Total: 12.5–17.5 weeks.** The wide range is dominated by M2.3
(does stock llama.cpp accept the expanded `hidden_size`?). Best case
~12 weeks; if a llama.cpp architecture fork is required, ~17 weeks.

Plus shared work coordinating M0.* with Path 1 and writing M0.4
(comparison report).

---

## 5. Disjoint-directory contract with Path 1

To minimise merge pain, Path 2 only writes to:

- `vendor/aloepri-py/**` (vendored AloePri reference)
- `vendor/llama.cpp/**` (vendored llama.cpp; only modified if M2.3
  outcome 2 or 3)
- `python/aloepri-llm/**` (offline rewriter, client wrapper, lib)
- `crates/aloepri-client/**` (optional Rust client)
- `scripts/path-2/**`
- `docs/plans/path-2-*.md`
- `docs/prototype/aloepri-llm.html` (the protocol doc)
- `evals/aloepri-attacks/**` (attack suite isolation per M2.1)
- `results/path-2-*.json`

Path 1 only writes to:

- `crates/gelo-embedder/**`, `crates/gelo-protocol/**`,
  `crates/gelo-gpu-wgpu/**`, `crates/gelo-snp-runner/**`,
  `crates/gelo-reranker/**`
- `evals/private-inference-corpus/**` (M0.1)
- `evals/run-eval.py` + `evals/lib/**` (M0.2)
- `evals/attack-harness/**` (M0.3 — though ownership may flip to
  Path 2 since the attack code lives in our vendored tree; see §6)
- `docs/plans/path-1-*.md`
- `results/path-1-*.json`

If Path 2 needs changes to Path-1-owned files, file a PR to master;
Path 1 reviews and merges before Path 2 consumes.

---

## 6. Open questions / decisions deferred

- **M0.3 attack-harness ownership.** Framework doc gives it to Path 1.
  But the source attacks live in `vendor/aloepri-py/src/security_qwen/`,
  which is Path-2-owned. Cleanest resolution: move M0.3 ownership to
  Path 2; Path 1 consumes the harness. Pending sync with Path 1 owner.
- **Tokenizer dual-implementation.** llama.cpp has its own
  tokeniser; the client wrapper uses HuggingFace's. Need to verify
  byte-level equivalence beyond the basic roundtrip property (M2.4)
  — a token boundary that disagrees between the two would silently
  corrupt outputs. M2.5 will catch this empirically; flag it
  explicitly if observed.
- **Streaming tokens.** `llama-server` supports streaming completions
  (SSE). With AloePri, partial obfuscated tokens may not detokenise
  cleanly mid-stream (SentencePiece byte-fallback edge cases).
  Defer streaming support to v2; v1 uses non-streaming completions.
- **AloePri hyperparameter retuning on Gemma 4.** Paper defaults
  were tuned for Qwen2.5 / Llama3 / DeepSeek. If M2.6 accuracy comes
  in worse than paper baselines, schedule a hyperparameter sweep
  (λ, h, α_e, α_h, β, γ) as a follow-on between M2.7 and M2.8.
- **Multimodal encoders** (audio / vision, ~150 M params each on
  E-series). Out of scope for v1. Open research questions in
  [`../research/private-llm-inference-round-2.md`](../research/private-llm-inference-round-2.md) §D.9.
- **Gemma 4 26B A4B (MoE)**. Out of scope; requires composing AloePri
  with CryptoMoE balanced-dispatch (round-2 §C). Separate research
  stream.

---

## 7. Getting started — concrete first commands

In the worktree (`../private-rag-path-2/`):

```bash
# 1. Verify worktree state
git status
git log --oneline -5

# 2. Read the handoff + the protocol doc
less docs/plans/path-2-aloepri-gemma.md             # this file
less docs/prototype/aloepri-llm.html                # protocol reference
less docs/plans/private-inference-comparison-framework.md
less docs/research/aloepri-vs-gelo.md
less docs/research/private-llm-inference-round-2.md

# 3. Vendor AloePri reference (M2.1 prep)
mkdir -p vendor
git clone https://github.com/sheng1feng/Aloepri vendor/aloepri-py
( cd vendor/aloepri-py && git checkout 60e8ea3 )

# 4. Set up Python env (M2.0)
mkdir -p python/aloepri-llm && cd python/aloepri-llm
python3.11 -m venv .venv && source .venv/bin/activate
pip install gguf safetensors torch numpy requests pytest

# 5. Build llama.cpp (M2.0)
cd ../../vendor
git clone https://github.com/ggml-org/llama.cpp llama.cpp
cd llama.cpp
cmake -B build -DGGML_NATIVE=ON -DLLAMA_BUILD_SERVER=ON
cmake --build build -j

# 6. Sanity-check plaintext E2B (M2.0 acceptance)
./build/bin/llama-cli -m <PATH_TO_E2B_GGUF> -p "What is the capital of France?" -n 16

# 7. Read llama.cpp's Gemma model code to find where hidden_size is consumed
grep -n "hidden_size\|n_embd\|hparams.n_embd" src/llama-model.cpp | head -40
```

---

## 8. References

- [`../prototype/aloepri-llm.html`](../prototype/aloepri-llm.html) —
  canonical protocol documentation (threat model, components, compute
  flow, trust boundaries)
- [`private-inference-comparison-framework.md`](private-inference-comparison-framework.md) —
  shared evaluation framework with Path 1
- [`path-1-gelo-gemma.md`](path-1-gelo-gemma.md) — sibling plan
- [`../dev/prototype/gelo.md`](../dev/prototype/gelo.md) — GELO protocol
  reference (background context)
- [`../research/private-llm-inference-round-2.md`](../research/private-llm-inference-round-2.md)
  §D — Gemma 4 architecture analysis (PLE, hybrid attention, K=V,
  p-RoPE)
- [`../research/aloepri-vs-gelo.md`](../research/aloepri-vs-gelo.md)
  — full technique-by-technique comparison
- AloePri paper: arXiv 2603.01499 (Lin et al., ByteDance + Nanjing
  Univ., March 2026)
- AloePri reference code: `github.com/sheng1feng/Aloepri` commit
  `60e8ea3`
- llama.cpp: `github.com/ggml-org/llama.cpp` — read `src/llama-model.cpp`
  for the Gemma forward graph, `examples/server/` for the HTTP server
- `gguf-py` package: PyPI `gguf` — read/write GGUF tensors from Python

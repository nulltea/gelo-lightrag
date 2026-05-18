# Path 2 — running status log

Update at the end of each milestone or whenever findings invalidate
a plan assumption. Most recent entry on top.

---

## 2026-05-18 · Item 7 done **partial** — head-shuffle works, intra-head blocked by Qwen3 QK-norm

Ported paper §5.2.3 Algorithm 2 to numpy in `python/path-2/lib/alg2.py`
(key generation: R̂_qk, Ĥ_qk, Ẑ_block, τ_kv, τ_group; static weight
rewrite via `_apply_output_transform`-equivalent + GQA-aware feature
ordering). Added `--alg2` flag to the rewriter.

**Smoke-test outcome:** the full paper Algorithm 2 (intra-head dense
+ head-shuffle + QK-norm §5.2.5 fold) **breaks the model** —
degenerate output like `" ..................... a again  ..."` on
`"What is the capital of France?"`. Bisecting:

| Transform configuration | Smoke output |
|---|---|
| Full alg2 (intra + Ẑ + head + QK-norm fold) | degenerate (".... a again ...") |
| Ẑ_block off (β=1), rest on | degenerate |
| Intra-head off, head+QK-norm-fold on | degenerate (".... of ......") |
| QK-norm fold only, head off, intra off | degenerate |
| **Head-shuffle only** (intra off, QK fold off) | **coherent — "The capital of France is Paris."** |

**Root cause: QK-norm §5.2.5 fold mathematically broken for Qwen3.**
Paper §5.2.5's fusion replaces per-element γ with scalar κ and folds
γ backward into the adjacent linear. The construction is **exact in
expectation** under i.i.d. Gaussian-input assumption — κ ≈
√(mean(γ²)) makes the RMS of (q·γ) approximately equal κ·RMS(q).
Empirically, Qwen3's `attn_q_norm.weight` / `attn_k_norm.weight`
have non-uniform per-dim γ, AND the model's Q/K vectors are
trained to put information in the high-γ dims (so q² and γ² are
positively correlated). The Gaussian assumption fails, the κ
approximation is off per-input, and attention scores degrade enough
to make the model produce high-prior-token loops.

This was foreseeable from the earlier grill but I underweighted
the *empirical* impact. The norm-doubling concern in path-2-status
(57 → 113 κ-sites) wasn't just an accuracy budget worry — for
QK-norm specifically, even one site breaks attention because the
per-input bias compounds inside the softmax.

Without the QK-norm fold, **all intra-head transforms (R̂_qk, Ĥ_qk,
Ẑ_block) become inapplicable.** They each require the Q/K vectors
to flow through a scalar-only norm so they can commute with R̂_qk
(rotation). Per-element γ_qk doesn't commute. Same issue for Ẑ_block
(permutes pair-positions with non-uniform γ).

**Head-shuffle (τ_kv, τ_group) is the one Algorithm 2 component
that survives.** It permutes whole heads — γ_qk is broadcast across
heads and is invariant under head reordering. Mathematically clean,
empirically verified.

**Item 7 final implementation (Qwen3-restricted):**
- Per-layer τ_kv ∈ S_8, τ_group ∈ S_2 (random non-identity)
- W_q, W_k, W_v rows permuted by the GQA-aware feature order
- W_o columns permuted by the Q-head feature order
- No intra-head dense, no QK-norm fold, no R̂_qk / Ĥ_qk / Ẑ_block
- `attn_q_norm.weight` and `attn_k_norm.weight` left untouched

**What we lose vs paper Algorithm 2:** R̂_qk / Ĥ_qk / Ẑ_block were
the principal ISA-defense transforms (they scramble the per-head
QK^T values directly). Head-shuffle alone scrambles only **which
head** produced which score, which is a weaker defense — an ISA
attacker can train an inverter on the full set of (head, position)
attention patterns, treating "permuted head ordering" as
recoverable noise.

The gap is the same one identified earlier between paper claims and
public reference code: the paper §7.1 says Qwen3 was evaluated, but
the public `sheng1feng/Aloepri @ 60e8ea3` has no Qwen3-specific
QK-norm handling. ByteDance's internal industrial version may have
a fix (per-norm-site κ tuning? a different fusion scheme?); the
public release doesn't, and re-deriving it from first principles
hits the i.i.d. Gaussian assumption wall.

**Future work options if intra-head defense becomes critical:**
- (a) Per-norm-site κ calibration — measure actual `RMS(q·γ)/RMS(q)`
  on a real input distribution rather than assuming Gaussian.
  Requires running the plaintext model on a corpus.
- (b) Restrict R̂_qk / Ĥ_qk / Ẑ_block to commute with γ structurally
  — e.g., only permute pairs `(i, i+half)` where γ_i ≈ γ_{i+half}.
  Likely yields a very small effective group.
- (c) Modify llama.cpp to add a "scaled-RMSNorm" op that compensates
  for fold bias at runtime. Forks the serving stack — opposite of
  AloePri's "no infra change" design goal.

All deferred to a post-item-10 (attack benchmark) decision point.
If head-shuffle alone defeats ISA enough on Qwen3 1.7B, the gap is
academic; if ISA still gets > 15% TTRSR, we revisit.

**Artifact:** `keymat-h128-pi-noise-alg2-fp32.gguf` (8.6 GB,
identical size to items 6+8 since head-shuffle is just a row
permutation). Key file 459 KB (up from 421 KB) — adds per-layer
`tau_kv`, `tau_group`, plus identity q_matrix / k_matrix stored for
metadata completeness.

**Smoke test:** `"What is the capital of France?"` → `"The capital of
France is Paris.\n\nNo, that's not right..."` — byte-identical to
items 6+8 artifact at the same prompt, which confirms head-shuffle
is correctness-preserving (model produces the same trajectory; only
the *internal head indexing* changes).

---

## 2026-05-18 · Item 8 done — α_e/α_h Gaussian noise on embed + head

Added `--noise-alpha-e`, `--noise-alpha-h`, `--noise-seed` flags to
the rewriter. Noise is sampled from `N(0, σ² I)` with `σ = std(W)`
of the relevant matrix, scaled by α, and added BEFORE Π / §5.2.5 /
keymat transforms.

**Paper defaults applied:** α_e = 1.0 (load-bearing for VMA defense
per §7.3), α_h = 0.2.

**Empirically observed σ:** σ_e = σ_h = 0.0345 (identical because
the bartowski source GGUF stores `token_embd` and `output` as
separate tensors with byte-identical content — Qwen3 ships tied
embeddings; the GGUF just duplicates them. The applied noise is
independently sampled so the post-noise W_e and W_h differ.)

**Smoke test** (`keymat-h128-pi-noise-fp32.gguf` on port 11451):

- Prompt: `"What is the capital of France?"`, max_tokens=24, seed=0.
- Output: `" The capital of France is Paris.\n\nNo, that's not right.
  The capital of France is Paris, but the capital"`
- Genuinely **better** than keymat-only's output for this prompt
  (which was `"Also, what is the largest city in the United States?"`).
  The shift is real (different noise → different model behavior)
  but well within "coherent" — α_e=1.0 doesn't break generation.
- `out_of_range_ids: []` again — Π/active-range guarantees hold.

**GGUF metadata additions:** `aloepri.noise_alpha_e` (float32),
`aloepri.noise_alpha_h` (float32). Seed is **not** stored (kept
client-side; reproducibility requires the rewriter args, not the
artifact metadata).

**Artifact:** `keymat-h128-pi-noise-fp32.gguf` (8.6 GB) +
`.gguf.key.npz` (the τ-only key — noise has no key, just adds bits
to the artifact). The container `llama-qwen3-keymat-pi-noise-fp32`
was spawned for smoke and torn down after.

**Acceptance:** model still produces coherent output under paper-
default noise. Numeric accuracy delta vs the no-noise keymat is
deferred to next Gate-C run.

---

## 2026-05-18 · Item 6 done — Π token-permutation + AloePri client landed

Added `--pi` mode to the rewriter and built `python/path-2/aloepri_client.py`.
Wire-payload privacy now closed (was: plaintext over the network).

**Offline rewrite changes** (`obfuscate_qwen3_gguf.py`):

- New flags: `--pi`, `--pi-seed`, `--key-out`.
- τ sampled as a permutation of `[0, 151669)` only (the tokenizer's
  active vocab); IDs `[151669, 151936)` are GGUF padding slots and
  stay identity. This eliminates the ~5%-per-32-token corruption
  risk from sampling into the un-decodable padding range.
- Row-permute `token_embd` and `output` by τ⁻¹ before the §5.2.5
  fusion + Algorithm 1 keymat transforms. Π commutes with keymat
  (different axes), so ordering doesn't change correctness — but
  cleaner to apply Π first.
- Key file `<out>.gguf.key.npz` written 0600: `{tau, pi_seed,
  vocab_size, active_size, arch, version}`. `pi_seed` is **not**
  written to GGUF metadata so the server cannot reconstruct τ.
- New GGUF metadata `aloepri.pi_applied: bool` (server-visible — the
  server's tokenizer config still names BOS=151643 etc., but those
  IDs now point at random embeddings, so this flag is purely
  informational).

**Client** (`aloepri_client.py`):

- Uses llama.cpp's **native** `/completion` endpoint with `prompt`
  as an int array and `return_tokens=true`. Bypasses the OpenAI-compat
  text-roundtrip protocol entirely; no tokenize↔detokenize on the
  wire to break with BPE edge cases. Server still treats the int
  array as if it tokenized text to those IDs.
- `KeyMaterial.load()` reads the key file; `tau`, `inv_tau`,
  `active_size` exposed.
- Refuses to run if `tokenizer.vocab_size != key.active_size` —
  cross-checks that the tokenizer used by the client matches the
  one the artifact was built against.
- Streaming + EOS handling explicitly **deferred to v2** (per
  next-steps handoff). v1 uses bounded `n_predict`.

**Smoke test** (port 11450, single-container spawn):

- Prompt: `"What is the capital of France?"`, max_tokens=24, seed=0.
- Plaintext IDs: `[3838, 374, 279, 6722, 315, 9625, 30]`
- Obfuscated IDs sent on wire: `[137397, 44230, 90908, 60247, 33846,
  135351, 102727]`
- What the server's stock tokenizer *would* decode those obf IDs to
  (i.e. what a passive observer sees as the "prompt"):
  `'いるとupgrade Printf\tmodeocyCIÓN芬'`
- Model output via client: `" Also, what is the largest city in the
  United States? What is the capital of Japan? What is the capital of"`
- **Byte-identical** to the keymat-only (no-Π) baseline on `:11446`
  for the same prompt/seed — confirms Π is correctness-preserving
  (commutes with keymat on the residual axis) and was generated
  + applied symmetrically.
- `out_of_range_ids: []` — the active-size restriction worked; no
  un-decodable IDs in the response.

**Artifact:** `keymat-h128-pi-fp32.gguf` (8.6 GB; same size as
keymat-only since Π is just a row permutation) + `.gguf.key.npz`
(422 KB, mode 0600).

**Container hygiene:** spawned a single fresh container
(`llama-qwen3-keymat-pi-fp32` on :11450) for smoke; tearing it down
before moving to item 8.

---

## 2026-05-18 · Gate C — partial: MMLU/PIQA/HumanEval done, IFEval deferred

Ran 3 of 4 Gate C tasks on Qwen3 1.7B Q8_0 plaintext vs keymat h128
fp32. Results below. IFEval deferred (initial run crashed on read
timeout while many idle containers were competing for Vulkan iGPU
memory; per user instruction we tore down all model containers
rather than re-run).

| Task | n | Plaintext | Keymat (fp32) | Δ (pp) | Notes |
|---|---:|---:|---:|---:|---|
| MMLU 0-shot | 200 | 54.5% | 55.0% | **+0.5** | within sampling noise (SE ≈ 3.5pp); keymat numerically *better* |
| PIQA 0-shot | 200 | 68.5% | 64.5% | **−4.0** | just past paper 3.5pp; SE ≈ 3.3pp |
| HumanEval pass@1 | 50 | 40.0% | 34.0% | **−6.0** | SE ≈ 6.9pp at n=50; marginal |
| IFEval (subset) | 50 | — | — | — | deferred |

**Pattern:** multi-choice / knowledge tasks (MMLU) survive cleanly;
generative tasks (PIQA solutions selection on a base model, HumanEval
code completion) drift more. Plausible mechanism: e_C^AloePri
compounds multiplicatively across hundreds of generated tokens vs.
a single-token decision in multi-choice.

**Decision-tree position** per `path-2-aloepri-next-steps.md` Gate C:
- MMLU clearly **≤ 3.5%** (paper-bound)
- PIQA / HumanEval in the **3.5%–10% band** ("paper-bound territory…
  proceed but flag in M0.4 report"), with sampling noise making the
  direction-of-effect statistically ambiguous on 50–200 prompts.
- No task is in the > 10% "stop and tune" band.

**Verdict:** Proceed to deferred items (Π token permutation → Algorithm
2 attention → α_e/α_h noise → attack benchmark) per user direction.
The 4th-task IFEval and a larger-n re-run (1000+ prompts each) are
deferred to a later session when (a) a sweep-of-hyperparameters is
also under consideration, or (b) M0.2 full framework lands.

**Harness scaffold landed** in `python/path-2/evals/` — reusable for
later sweeps. Results in `results/path-2-gate-c-{mmlu,piqa,humaneval}.json`.

**Container hygiene fix** (after user feedback): future benchmarks
spawn only the containers under test and tear them down after; saved
as `feedback_llama_containers_ephemeral` memory.

---

## 2026-05-18 · Gate B — both endpoints fully deterministic on Vulkan + flash-attn

Ran `python/path-2/gate_b_determinism.py`: 5 prompts × 3 replicates ×
2 endpoints (`:11441` plaintext, `:11446` keymat h128 fp32),
`temperature=0.0`, `seed=0`, `max_tokens=32`.

All 30 replicates **byte-identical within their (prompt, endpoint)
group.** Determinism class on every prompt: `fully-deterministic`.
The earlier concern about Vulkan workgroup ordering + flash-attn
reduction-order non-determinism does not manifest here — likely
because `-np 1` (single slot) removes batching variability, and the
Strix Halo Vulkan FA path appears to use a deterministic tile
schedule under fixed launch geometry.

**Plaintext-vs-keymat divergence** (cross-LCP, chars / mean response
length):

| Prompt | divergence char | response length |
|---|---:|---:|
| "What is the capital of France?" | 19 | ~145 |
| "Write a haiku about autumn." | 1 | ~86 |
| "def fibonacci(n):" | 5 | ~93 |
| "Translate to French: Hello…" | 5 | ~142 |
| "Once upon a time in a faraway land," | 48 | ~152 |

Conclusion: e_C^AloePri produces **real, deterministic** divergence
from plaintext early in every completion. Gate C runs on Vulkan
(no CPU fallback needed); the accuracy delta we measure will be
signal, not sampling noise.

Results: `results/path-2-gate-b.json`.

---

## 2026-05-18 · Gate A — Q8_0/Q6_K/Q5_K_M all fail; fp32 required for keymat artifact

Quantised `keymat-h128-fp32.gguf` (8.6 GB) to Q8_0, Q6_K, Q5_K_M and
ran identical smoke prompt (`"What is the capital of France?"`,
temperature 0.0, max_tokens 24, seed 0) on each. All three quantised
artifacts produce **degenerate** output:

| Format | Size | Output (first 24 tokens) |
|---|---:|---|
| fp32 (ref) | 8.6 GB | `" Also, what is the largest city in the United States? What is the capital of Japan? What is the capital of"` |
| Q8_0 | 2.3 GB | `" ( ( ( ( ,chein,zech,  \tswitch , , alysis , ,     く"` |
| Q6_K | 1.8 GB | `glm_termumbaและการacre庾iones Chattanooga…` (server returned 500 — output failed re-tokenisation) |
| Q5_K_M | 1.6 GB | `"phenhqymologyaverholes澈%nCANasse有名imusted的女儿战争べきuctor célib.slice…"` |

**Root cause** (predicted in the next-steps handoff §A caveat,
confirmed empirically): AloePri-obfuscated weights have heavy-tailed
per-row distributions (paper-default `λ` makes `P̂_R` non-orthonormal;
keymat at `h=128` further amplifies the spread). Q8_0's 32-element
fixed-scale blocks lose the small values to zero when a block also
contains an outlier. K-quants (Q6_K, Q5_K_M) have more flexible
scaling but still can't preserve the precision needed to keep the
covariant chain intact.

**Verdict:** fp32 is the **production format** for AloePri-obfuscated
GGUF artifacts. Gate B and Gate C run on `:11446` (keymat h128 fp32).

Implications:
- Disk cost per model variant ≈ 9 GB. Manageable for v1; revisit at
  E4B (~20 GB fp32 estimated) if scale becomes a problem.
- Decode speed per the existing `:11446` baseline is acceptable
  (Vulkan iGPU handles 8.6 GB; see M2.0 notes for per-token timing).
- **Potential remediation knobs** for a future "obfuscated artifact
  that survives Q8_0" research stream: smaller `λ` (P̂_R closer to
  orthonormal → smaller within-row variance → friendlier to
  per-block scaling); QR-project P̂_R to the orthogonal manifold;
  reduce `h` (trade-off: less internal expansion, weaker
  obfuscation). Deferred — fp32 unblocks Gate B/C immediately.

**Failed-experiment artifacts kept on disk:**
- `keymat-h128-Q8_0.gguf` (2.3 GB), `keymat-h128-Q6_K.gguf` (1.8 GB),
  `keymat-h128-Q5_K_M.gguf` (1.6 GB) — total 5.7 GB to reclaim
- Containers: `llama-qwen3-1p7b-aloepri-h128-Q8_0` (:11447),
  `…-Q6_K` (:11448), `…-Q5_K_M` (:11449) — keep until Gate B/C
  decide we don't need them as a regression reference; tear down
  during Gate C cleanup if not needed.

---

## 2026-05-18 · M2.3 verdict re-confirmed at outcome (1) after §5.2.5 reread

The earlier "verdict downgrade" entry in this log was wrong. I had
not noticed the paper's §5.2.5 *Layer Normalization Transformation*
construction, which provides a fully-offline weight rewrite for
RMSNorm. Reverting to **outcome (1) — just runs on stock
llama.cpp** without a patch. The runtime `KeyMatRMSNormBridge` in
the reference codebase is a research-time convenience and **not the
construction the paper claims** for production.

### What §5.2.5 says

For an RMSNorm with per-dim γ vector `w_norm` and a following linear
of weights `W`:

1. Replace `RMSNorm(x; w_norm) → Linear(W)` with three ops:
   `RMSNorm(x; κ·1) → Linear(Diag(w_norm)) → Linear(W)`. The new
   RMSNorm has γ = a single scalar κ (constant across all dims).
2. Fuse `Diag(w_norm) · W` together into a single linear, applied
   *before* AloePri obfuscation: `W' = Diag(w_norm) · W`.
3. Apply Algorithm 1 obfuscation to the fused `W'` as normal.
4. κ is a scalar correction `κ = E[‖xP̂‖/‖x‖]` over the assumed-
   Gaussian input distribution; for orthonormal-row P̂ it's ≈ 1.

Why this works offline: a *scalar* γ at the RMSNorm site **does**
commute with rotation (it's just an isotropic rescaling), while
a per-dim γ does not. The per-dim part has been baked into the
adjacent linear weight before obfuscation — no extra ops at
runtime.

For post-norms (sitting between a sub-block's output and the
residual add), the fusion goes *backward* into the previous linear:
`W'_prev = W_prev · Diag(w_norm)` (column scaling).

### Updated implementation plan for M2.2 step 2

The offline rewriter (`obfuscate_gemma4_gguf.py`) gets a `--mode keymat`
flag that, for each tensor:

1. Maps every residual-stream RMSNorm `w_norm` (γ vector) to its
   adjacent linear weight(s). Direction (pre or post) depends on the
   norm's role in the gemma4 forward graph.
2. Pre-multiplies (or post-multiplies, depending) the adjacent weight
   by `Diag(w_norm)`.
3. Replaces the γ tensor with a constant vector `(κ, κ, …, κ)` of
   length `d + 2h`.
4. Applies Algorithm 1 obfuscation (`Q̂_R @ W'` for input weights,
   `W' @ P̂_R` for output weights) to the fused weights.

Stock llama.cpp runs the resulting artifact unchanged.

### Live security risk: Gemma 4's higher norm count

Gemma 4 has ~5 residual-stream RMSNorm sites per block (attn_norm,
post_attention_norm, ffn_norm, post_ffw_norm, post_norm — plus the
per_layer_post_norm after PLE addition) vs. Llama/Qwen's 2. Across
E2B (35 blocks): ~175 norm sites vs. Llama 3 8B's 64.

Each §5.2.5 fusion introduces a bounded approximation error
`e_C^norm` (since κ is an expectation, not exact for individual
inputs). The paper's total error bound

```
e_C^AloePri ≤ M_0 · e_C^embed + Σ_i M_i · e_C_i^decoder + e_C^head
```

accumulates `e_C^norm` at every site, multiplied by per-layer
Lipschitz factors. Paper-measured accuracy loss (0–3.5 %) is on
2-norm-per-block architectures. Gemma 4's ~2.7× higher norm count
could plausibly compound to 7–12 % loss, possibly more.

**This is treated as an M2.6 question, not a blocker.** Implementation
proceeds with paper-default hyperparameters; if M2.6 shows
unacceptable loss, the remediation knobs are:

- Smaller λ (B = U + λV; smaller λ → P̂_R closer to orthonormal → κ closer to 1 → smaller per-site error).
- Per-norm-site κ tuning instead of one global κ (measure per-site activation statistics).
- Restrict P̂_R to orthonormal (project via QR onto the orthogonal manifold) — eliminates dim-correction bias at the cost of a smaller obfuscation group.
- Last resort: drop down to Gemma 3 4B (2 norms per block, in paper-tested regime).

---

## 2026-05-18 · M2.0 complete

**Headline:** Gemma 4 architecture (`gemma4`) is already in llama.cpp
upstream. Q8_0 E2B GGUF runs on Vulkan via the stock
`ghcr.io/ggml-org/llama.cpp:server-vulkan` container with no
modifications. Reading `src/models/gemma4.cpp` indicates the M2.3
"does stock llama.cpp accept `hidden_size = d + 2h`?" question is
**very likely outcome (1) — just runs.**

### Baseline running

| Item | Value |
|---|---|
| Container | `llama-gemma4-e2b-aloepri-baseline` |
| Image | `ghcr.io/ggml-org/llama.cpp:server-vulkan` |
| Endpoint | `http://127.0.0.1:11437` (OpenAI-compat) |
| Vulkan device | `Radeon 8060S Graphics (RADV STRIX_HALO)` on AMD Strix Halo iGPU |
| Model | Gemma 4 E2B-it Q8_0 (`unsloth/gemma-4-E2B-it-GGUF` @ snapshot `90f9618…`) |
| Per-token decode | ~15 ms (≈65 tok/s) |
| Model alias | `gemma 4 (E2B) plaintext baseline` |

Container config mirrors the user's existing `llama-gemma4-vision`
reference: `--network host`, `/dev/dri`, `video` group, Vulkan ICD
passthrough, `-ngl 999 -np 4 --flash-attn on -c 131072
--ubatch-size 2048 --sleep-idle-seconds 120`. Restart policy
`unless-stopped`.

### Architecture findings from `src/models/gemma4.cpp`

- `LLM_ARCH_GEMMA4` is a full first-class architecture; model class
  `llama_model_gemma4`. Supports E2B (35 layers), E4B (42),
  26B A4B (30, with MoE), 31B (60). Type-detected from
  `gemma4.block_count`.
- `n_embd` is read directly from `gemma4.embedding_length` metadata.
  **No assertion that `n_embd == n_heads · head_dim` anywhere in
  the forward graph.** Q projection shape `{n_embd,
  n_embd_head * n_head}` is constructed independently — changing
  `n_embd` to `d + 2h = 1792` should propagate cleanly.
- One real assertion (lines 37–42 of `gemma4.cpp`): requires
  `n_embd_head_k == n_embd_head_v` and same for SWA. AloePri does
  not change head dim — fine.
- Final logit softcapping (`f_final_logit_softcapping = 30.0`) is
  applied to the output logits. Monotonic and applied after the
  obfuscated head, so the τ-permuted token-ID argmax is unchanged.

### Gemma 4 E2B Q8_0 tensor inventory

601 tensors, all per-layer tensors uniform across all 35 blocks. Per-block tensors:

| Tensor | Shape | dtype | AloePri treatment |
|---|---|---|---|
| `attn_norm` | `[1536]` | F32 | covariant RMSNorm (γ folded into Q̂ chain) |
| `attn_q` | `[1536, 2048]` | Q8_0 | Algorithm 2: `Q̂_q · W_q · R̂_qk · Ĥ_qk · Ẑ_block` |
| `attn_k` | `[1536, 256]` (local) / `[1536, 512]` (global) | Q8_0 | Algorithm 2: `Q̂_k · W_k · R̂_qk · Ĥ_qk⁻¹ · Ẑ_block^η` |
| `attn_v` | `[1536, 256]` (local) / `[1536, 512]` (global) | Q8_0 | Algorithm 2: `Q̂_v · W_v · Û_vo` |
| `attn_q_norm`, `attn_k_norm` | `[256]` | F32 | covariant RMSNorm |
| `attn_output` | `[2048, 1536]` | Q8_0 | Algorithm 2: `Û_vo⁻¹ · W_o · P̂_o` |
| `post_attention_norm`, `post_norm`, `post_ffw_norm`, `ffn_norm` | `[1536]` | F32 | covariant RMSNorm |
| `ffn_gate`, `ffn_up` | `[1536, 6144]` | Q8_0 | Algorithm 1 key-matrix obfuscation |
| `ffn_down` | `[6144, 1536]` | Q8_0 | Algorithm 1 key-matrix obfuscation |
| `inp_gate` (= `per_layer_inp_gate`) | `[1536, 256]` | F32 | covariant — gates per-layer contribution per-token |
| `proj` (= `per_layer_proj`) | `[256, 1536]` | F32 | covariant projection into residual stream |
| `post_norm` (`per_layer_post_norm`) | `[1536]` | F32 | covariant RMSNorm |
| `layer_output_scale` | `[1]` | F32 | scalar — Gemma 4 specific layer-output gain |

Plus six global tensors:

| Tensor | Shape | dtype | AloePri treatment |
|---|---|---|---|
| `token_embd` | `[1536, 262144]` | Q8_0 | `Π · (W_e + α_e·ε) · P̂_embed` |
| `output` (tied to `token_embd` per architecture, but stored separately in this GGUF) | — | — | (tied head case: same as token_embd) |
| `output_norm` | `[1536]` | F32 | covariant RMSNorm |
| `per_layer_token_embd` (PLE table) | `[8960, 262144]` | Q8_0 | **vocab-axis permute by τ** — same τ as `token_embd` |
| `per_layer_model_proj` | `[1536, 8960]` | BF16 | covariant |
| `per_layer_proj_norm` | `[256]` | F32 | covariant |
| `rope_freqs` | `[256]` | F32 | unchanged (RoPE frequencies, public) |

### Surprises vs the original plan

| Plan assumption | Reality | Plan impact |
|---|---|---|
| "K=V untie in global layers" needs a non-trivial offline rewrite step (`M2.2b`) | GGUF stores `attn_k` and `attn_v` as **separate tensors at every layer**, regardless of `shared_kv_layers`. The "sharing" is at the **KV cache** runtime layer, not the weight layer. | `M2.2b` collapses to a no-op. Offline rewriter just obfuscates every present `attn_k` / `attn_v` tensor independently. |
| PLE is `[262144 × 256 × N_layers]` indexed per-layer-per-token | PLE is **one fused `[8960 × 262144]` table** where 8960 = 256 × 35 layers. Vocabulary axis is dim 1 (262144). | `M2.2c` is simpler: permute axis 1 of `per_layer_token_embd` by τ (same τ as `token_embd`). One operation, not per-layer. |
| p-RoPE is the rotation knob in Gemma 4 global layers | Confirmed: `rope_freqs.weight` is loaded only for `!is_swa(i)` layers (global), shape `{n_embd_head / 2}`. The metadata key `rope.dimension_count = 512` vs `rope.dimension_count_swa = 256` confirms partial RoPE on global. | `M2.2d` retains its scope — R̂_qk / Ĥ_qk / Ẑ_block must be restricted to the rotated subset. |
| `n_layer_kv_from_start` is a per-block weight property | It's a **runtime cache-sharing optimisation**: layers `n_layer_kv_from_start..n_layer-1` reuse the KV cache of earlier layers but have their own Q. For E2B with `shared_kv_layers = 20`, layers 15..34 reuse cached K/V from layers 0..14. | **New M2.2 constraint:** AloePri's `Q̂_k`, `Q̂_v` matrices for layer `i ≥ n_layer_kv_from_start` must equal the matrices for the layer it shares with. Otherwise Q·K^T dot products land in mismatched obfuscation spaces. To be specified concretely once we identify the exact sharing map (likely `i → i mod n_layer_kv_from_start` or similar — TODO: confirm from source). |

### Mystery tensors decoded

The inspector showed three per-layer tensors not enumerated in the
plan; resolved from `src/models/gemma4.cpp:125–130`:

- `inp_gate.weight` = `per_layer_inp_gate` — gates how the
  per-layer embedding adds into the residual stream
- `proj.weight` = `per_layer_proj` — projects 256-dim per-layer
  vector → 1536-dim hidden
- `layer_output_scale.weight` = `out_scale` — scalar gain on the
  whole block's output, Gemma 4 specific

All three sit inside the covariant-obfuscation framework via paper
§4.2's sequential composition theorem — no new construction needed.

### Q8_0-only baseline note

User provided Q8_0 weights, not BF16. Plan's M2.6 (BF16 baseline)
becomes "plaintext Q8_0 vs obfuscated Q8_0" instead of "plaintext
BF16 vs obfuscated BF16". This is a tighter single-track gate — we
measure AloePri delta on top of already-Q8 quantisation, with no
separate "quantisation stacking" risk to evaluate at M2.7.

If we later want a BF16 reference for the paper-parity correctness
argument, options are:

- Download `unsloth/gemma-4-E2B-it-GGUF` BF16 variant (likely
  exists), or
- Convert from the HF safetensors original with
  `convert_hf_to_gguf.py` from llama.cpp tools.

Out of scope for v1 unless M2.6 surfaces a question that requires it.

### Build artefact locations

- llama.cpp source: `vendor/llama.cpp/` @ `053e01d`
- CPU build: `vendor/llama.cpp/build/bin/{llama-cli,llama-server,llama-quantize}`
  (built with `-DGGML_NATIVE=ON -DLLAMA_BUILD_SERVER=ON
  -DLLAMA_CURL=OFF`; CPU-only — kept for the offline rewriter
  pipeline and `llama-quantize`-based re-quant work, not for
  serving)
- Python env: `python/path-2/.venv` (Python 3.12.3) — `gguf`,
  `safetensors`, `numpy`, `torch`, `requests`, `pytest` installed
- Vulkan baseline server: container
  `llama-gemma4-e2b-aloepri-baseline` on `:11437`

### What unblocks M2.1 / M2.2

- M2.3 risk has dropped from **high → low-medium**. We will likely
  not need to fork llama.cpp.
- The offline rewriter (`M2.2`) can target the Q8_0 GGUF directly
  via `gguf-py`: read tensors, dequantise per-tensor to F32, apply
  AloePri, re-quantise to Q8_0, write.
- The KV-cache-sharing constraint is a new design point for
  `M2.2a` — specify the per-layer matching of `Q̂_k`, `Q̂_v` before
  any obfuscation code is written.

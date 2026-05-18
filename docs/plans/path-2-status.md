# Path 2 — running status log

Update at the end of each milestone or whenever findings invalidate
a plan assumption. Most recent entry on top.

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

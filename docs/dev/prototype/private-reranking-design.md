---
type: prototype-note
status: current
created: 2026-05-15
updated: 2026-05-26
tags: [reranking, gelo, caprise, design]
companion: [reranking, gelo, gelo-llm]
---

# Private Reranking — Design (Qwen3-Reranker primary)

> Design + plan content originated in a "round-2" research doc
> (2026-05-15); split off here 2026-05-26 so the research doc stays a
> literature artifact and the implementation choices live with the rest
> of our prototype design notes. The literature half of the original
> round-2 was merged into the rev-5 update section of the research doc
> (linked below).
>
> **Research baseline:** [`../../research/private-reranking-research.md`](../../research/private-reranking-research.md) — taxonomy of 10 approaches + the rev-5 (2026-05-15) literature update / GELO+TwinShield compatibility table.
>
> **Sibling prototype docs:** [`reranking.md`](reranking.md) (broader reranking design including cross-encoder vs causal-LM choice), [`gelo.md`](gelo.md) §6 (protocol), [`gelo-llm.md`](gelo-llm.md) §3 (fused permuted attention prerequisite for jina-v3).

## Definitions (design-specific)

- **LBNLI** — *Listwise BERT-NLI*. Jina-reranker-v3's "last but not late" scoring scheme: pack `[query | doc_1 | … | doc_N]` into one causal forward; gather per-doc last-token hidden states; score by `cos(MLP(q_last), MLP(d_i_last))`.
- **Discriminator reranker** — Causal LM prompted with `"{q}\n{d}"` and trained to output `yes`/`no` as the next token; `softmax([no_logit, yes_logit])[1]` is the relevance score. Qwen3-Reranker uses this pattern.
- **Score-export leakage** — Class of leakage specific to rerankers: even with TEE-protected inference, the *scalar relevance score* must exit the TEE to be useful, and that scalar can be inverted under enough queries.

## Architecture comparison: bge / Qwen3 / jina

| Dimension | bge-reranker-v2-m3 | Qwen3-Reranker-0.6B | jina-reranker-v3 |
|---|---|---|---|
| Backbone | XLM-RoBERTa-large (24L, 1024, GELU, post-LN) | Qwen3-0.6B (28L, 1024, SwiGLU, RMSNorm, GQA 16/8, RoPE) | Qwen3-0.6B (same) |
| Output head | `Linear(1024 → 1)` on `[CLS]` | LM-head + softmax over `{yes, no}` | 2-layer MLP `1024→512→256` + cosine |
| Scoring shape | Per-pair forward | Per-pair forward | **Listwise**: one forward over `[q;d_1;…;d_N]` |
| Typical n per forward | 200–512 | 200–600 | up to 16k (64 docs × ~256 tokens) |
| BEIR NDCG@10 | 56.51 | ~58 (inferred from MTEB-R 65.80) | **61.94** |
| License | Apache-2.0 | Apache-2.0 | **CC-BY-NC-4.0** |
| GELO compat | green | **green** | yellow (needs fused permuted attention) |
| Existing crate reuse | ~85% (`bert/` path) | **~98%** (`decoder/` path, byte-identical) | ~80% (after `gelo-llm.md` §3 ships) |
| Estimated latency / (q, d) | ~130–160 ms | ~155 ms | ~12 ms amortized (needs fused attn); else impractical |
| Special prep work | classifier-head loader | LM-head + chat-template + yes/no tok-id pin | listwise packing + fused-attention dep |

### Why Qwen3-Reranker-0.6B is the primary

The Qwen3-Reranker-0.6B backbone is **byte-identical** to Qwen3-Embedding-0.6B already shipping in `crates/gelo-embedder/src/decoder/`. Every GELO protocol primitive — mask, shield rows, U-Verify, OutAttnMult, permuted attention, length auto-switch, sensitive-layer exclusion — applies without modification.

Implementation cost is dominated by:

1. LM-head linear loader (or reuse `token_embedding` since `tie_word_embeddings=true`).
2. Chat-template plumbing for the Qwen-style `<|im_start|>system…<|im_end|>` prompt format.
3. Yes/no token-id lookup at load (must be SHA-pinned alongside the tokenizer JSON).
4. A `score(q, d) -> f32` entry point that gathers the last-token logits, slices on the two pinned token IDs, computes `softmax([no, yes])[1]`.

Estimated effort: **2–3 days** for a working private reranker.
Estimated wall-clock: **~155 ms/(q, d) on AOCL-BLIS post-Tier-2** (within run-to-run variance of the documented Qwen3-Embedding baseline at 1.27× plain).

Quality is mid-pack but credible (MTEB-R 65.80, inferred BEIR ~58). Apache-2.0 license is commercially clean.

### Why bge-reranker-v2-m3 is the fallback

XLM-RoBERTa-large is the boring, well-trodden option. Post-LN BERT GEMMs are exactly what GELO was first validated on. The only protocol addition is a `Linear(1024 → 1)` classifier head — a single offload-able or in-TEE linear, depending on policy.

Effort: **~3–4 days**. Latency: comparable to Qwen3 (24 layers × 1024 ≈ Qwen3's 28 × 1024 in total work; FFN is wider but no SwiGLU gate). Quality is the lowest of the three at BEIR 56.51 but the implementation risk is the lowest.

This is the right call if Qwen3-Reranker's yes/no probability calibration turns out to be unstable on the production corpus, or if we want a parity check that the protocol's correctness extends beyond the Qwen family.

### Why jina-reranker-v3 is deferred

Three reasons:

1. **CC-BY-NC-4.0 license** kills commercial deployment without separate negotiation. Apache/MIT competitors are preferable.
2. **The listwise-packed forward pushes n to 16k+**, which requires the `gelo-llm.md` §3 fused permuted-attention work (FlashAttention inside our protocol) to land first. That's a 5–7 week prerequisite.
3. Per-query full recompute (no doc-encoding cache) makes it expensive when the listwise advantage is small (single-doc reranking).

It is the strongest candidate to revisit **after** the LLM-serving stack is up, since it shares fully with that work. The 5+ point BEIR NDCG@10 lead over the other two is real — but contingent on fused attention being available.

## Open problems specific to reranking

Most of these are *new* concerns that didn't appear in the embedding-only protocol design.

### Score-export leakage

The reranker's *output* is a plaintext scalar per (q, d) pair. Even with the forward pass fully protected by GELO+TwinShield, the exported score reveals query-document alignment under enough queries. None of our current encryption layers protect this.

Mitigation candidates (no published paper picks one yet):

- **Rank-only export**: return ordered chunk IDs, not scores. Loses fine-grained pipeline composability.
- **Quantized score export**: round to N bits before exit. Bounds per-query information.
- **Bounded score noise**: add small DP noise to the exported score. Requires an accounting model for cumulative leakage across queries.
- **Score-tensor masking inside the TEE**: mask the scores under the same mask as the embeddings (CAPRISE-style); decrypt only at the client. Heaviest option; cleanest privacy.

This is the **single largest gap** in the reranker design. Recommend spec'ing it before implementation — `gelo.md` §6 doesn't address it because embeddings exit CAPRISE-encrypted, but rerank scores don't.

### Cross-encoder + AES-decrypted chunks inside the CVM

Cross-encoder rerankers must AES-decrypt the k' candidate chunks' *text* inside the TEE. `gelo.md` doesn't currently address the AES key-transfer path for chunk text. Verify:

- The attestation flow (`SnpTrustedExecutor::evidence`) already binds `model_identity` + `scheme_identity`; an additional `aes_key_holder` identity must enter REPORT_DATA for the relying party to know the chunk-AES key is provisioned correctly.
- The Qwen3-Reranker prompt template embeds chunk text directly; the AES decryption happens inside the CVM before tokenization. No protocol-layer change to GELO, but a runner-layer change to wire the key.

This is an engineering concern more than a research one; flagged for the implementation pass.

### Empirical attack resistance for the reranker shape

All published attack papers (2602.11088, 2505.18332, `qsxltss/Game-of-Arrows`) target embedding/decoder inversion. None evaluate against the (q, d) → scalar reranker forward. The memory note in `gelo_research_round_2.md` argues GELO is safe by construction, but that argument hasn't been empirically validated against the reranker forward.

Recommend: run `Game-of-Arrows` against the reranker forward in mock mode as a release-gate test, similar to the existing `tests/bss_recovery.rs` regression guard. ~1 week of work; not on the critical path but should land before any production deployment claim.

### Listwise (jina-v3) under the protocol

Deferred entirely. The listwise-packed n=16k forward needs:

- Fused permuted attention (`gelo-llm.md` §3) — open prerequisite.
- Encrypted-KV-in-SWIOTLB (per-batch state, not weights) — open design.
- Per-doc score gather logic with permutation-invariant correctness — straightforward but new.

File for revisit after the LLM-serving stack lands.

## Recommendations

### Architecture target

- **Primary: Qwen3-Reranker-0.6B.** Lowest friction, highest code reuse, validated backbone, Apache-2.0 license.
- **Fallback: bge-reranker-v2-m3.** Parity check + safer bet if Qwen3-Reranker's calibration is unstable.
- **Defer: jina-reranker-v3.** Revisit after `gelo-llm.md` §3 lands.

### Protocol composition for the primary path

The existing GELO+TwinShield+CAPRISE+SEV-SNP stack covers Qwen3-Reranker-0.6B directly. Required *new* design work:

1. **Score-export discipline** (above) — pick a mitigation, document the threat model, gate behind a runner config.
2. **AES key path for chunk text** (above) — extend `scheme_identity`'s REPORT_DATA contribution, add a runner-layer key-provision step.

Optional add-ons (defense-in-depth, not on critical path):

3. ARROWCLOAK per-row scaling for OutAttnMult — only relevant if the permuted-attention path is forced on for long-doc reranking.
4. SCX-style stateless HKDF mask derivation — required for multi-tenant deployment but skippable for single-tenant.

### What still needs research

1. **Cross-encoder transformer under MPC.** Still unsolved. EncFormer is closest but seconds-class. Not on critical path; relevant only if TEE assumption later breaks.
2. **Score-leakage formal model.** No published paper defines the right notion of "rerank score privacy" beyond DP. Worth a 1–2 week research spike before locking in a mitigation in §Protocol composition above.
3. **Game-of-Arrows on the reranker shape.** Empirical attack-resistance bench (above).
4. **Listwise rerankers under the protocol** (above). Filed for after LLM serving.

## Decisions deferred to implementation

These are notes for the implementer, not commitments:

1. **Per-pair vs batched-listwise API.** Recommend per-pair for the Qwen3-Reranker path; add batched later only if needed.
2. **Yes/no token resolution.** Look up at load time, pin alongside tokenizer SHA-256 in `model_identity`.
3. **Chat-template handling.** Inline (deterministic + attested) rather than reading from `tokenizer_config.json` at runtime.
4. **Skip-last-layer default.** Consider defaulting `skip_last_layer = true` for the reranker path — the LM head's final projection is adjacent to the cleartext score and is the highest-leakage layer. Cost: ~5 ms/text.
5. **Truncation policy.** Qwen3-Reranker accepts 32k context; suggest 2048 per-pair limit (covers BEIR queries + most docs). Surface via env.

## References

Reranker model docs (referenced in this design):

- Chen et al., "BGE-M3 / bge-reranker-v2-m3." arXiv 2309.07597.
- "Qwen3-Reranker." arXiv 2506.05176 (Tongyi Lab, 2025).
- Wang et al., "Jina-Reranker-v3: Last but Not Late Interaction." arXiv 2509.25085.
- Maringan & Fitrianah 2026, "HR+QDA," *Discover Computing* — lightweight on-prem reranker baseline.

Protocol primitives (background):

- `gelo.md` — current protocol.
- `gelo-llm.md` — LLM extension plan (fused permuted attention, SCX KV-cache).
- `memory/gelo_research_round_2.md` — 2026-05-14 attack-resistance analysis (Claude memory).

Attack papers driving the threat model:

- Wang et al., "Vulnerabilities in Partial TEE-Shielded LLM Inference with Precomputed Noise." arXiv 2602.11088.
- Wang et al., "Hidden No More: Attacking and Defending Private Third-Party LLM Inference." ICML 2025, arXiv 2505.18332.
- Wang et al., "Game of Arrows / ARROWMATCH / ARROWCLOAK." USENIX Security 2025.

# Private Reranking — Round 2 (GELO+CAPRISE compatibility pass)

> Research date: 2026-05-15. Sources: WebSearch (post-Apr-2026 papers),
> WebFetch (arXiv abstracts + HF model cards), OpenAlex citation
> expansion, Edgequake corpus check, local `docs/prototype/`.
> Complements `private-reranking-research.md` (rev-4, 2026-04-21).
>
> **Goal of this round.** Given the current GELO+TwinShield+CAPRISE
> direction documented in `docs/prototype/gelo.md` and the LLM extension
> notes in `docs/prototype/gelo-llm.md`, narrow the reranker design space
> to a concrete architecture and identify the primitives that need to
> land before implementation can start. Three architectures on the
> short-list: **bge-reranker-v2-m3**, **Qwen3-Reranker-0.6B**,
> **jina-reranker-v3**.

---

## 0. Definitions

Acronyms specific to this document on top of the ones in
`gelo.md` §0 / `inference-optimization.md` §0.

- **LBNLI** — *Listwise BERT-NLI*. Jina-reranker-v3's "last but not late"
  scoring scheme: pack `[query | doc_1 | ... | doc_N]` into one causal
  forward; gather per-doc last-token hidden states; score by
  `cos(MLP(q_last), MLP(d_i_last))`.
- **Cross-encoder** — Single transformer that joint-encodes
  `[CLS] q [SEP] d [SEP]` and emits one relevance scalar per pair.
  Contrasted with *bi-encoder* (independent embeddings + cosine).
- **Discriminator reranker** — Causal LM prompted with
  `"{q}\n{d}"` and trained to output `yes`/`no` as the next token;
  `softmax([no_logit, yes_logit])[1]` is the relevance score.
  Qwen3-Reranker uses this pattern.
- **Score-export leakage** — Class of leakage specific to rerankers:
  even with TEE-protected inference, the *scalar relevance score* must
  exit the TEE to be useful, and that scalar can be inverted under
  enough queries.
- **ArrowMatch** — Wang et al. (USENIX Sec '25, `qsxltss/Game-of-Arrows`)
  attack on weight-side obfuscation. Sister attack to ARROWCLOAK
  (defense).

---

## 1. What changed since rev-4

Three new TEE+GPU split-inference papers appeared between April and
December 2025 (indexed in 2026) plus one important MPC+GPU result. None
of them targets reranking specifically; all are LLM-inference protocols
whose primitives transfer.

### 1.1 New TEE+GPU split papers

**SecureInfer** (arXiv 2510.19979, Oct 2025) — heterogeneous TEE-GPU,
XOR-based one-time-pad masks per transfer. Targets **private weights**
(opposite of our openweight assumption). Construction operates on both
weight and activation axes; the moment weights become public it falls
into the ArrowMatch-broken family. Reported 4.7× speedup over TEE-only,
2.06× latency over unprotected GPU, 8.44 tok/s LLaMA-2. **Not a fit**
for our threat model; useful only as an end-to-end latency benchmark.

**Privacy-Aware Split Inference with Speculative Decoding** (arXiv
2602.16760, Feb 2026, Cunningham). Naked fp16 activations cross PCIe
between trusted local GPU and untrusted remote GPU over WAN. 3-layer
MLP inversion decoder recovers 59% top-1 at 2-layer split, 35% at
8-layer split. **Useful as a negative result**: confirms that the
activation mask in GELO is doing real work — without it, an attacker
with 880 sample pairs cracks the protocol.

**Opal: Private Memory for Personal AI** (arXiv 2604.02522, Kaviani et
al., April 2026). Intel TDX + NVIDIA B200 in confidential-compute mode,
ORAM-backed encrypted disk for embeddings + chunks, knowledge-graph +
semantic retrieval inside the enclave. **Rev-4's claim that "Opal
explicitly bakes cross-encoder reranking inside TEE" is not supported
by the published abstract** — the data-dependent reasoning step does
*not* mention cross-encoder rerank; it uses semantic-search-plus-graph.
[UNCLEAR — needs full-paper fetch.] Threat model assumes confidential
GPU (B200 CC), which is exactly the assumption GELO deliberately avoids.
29× lower infra cost vs secure baseline.

**SoK: Analysis of Accelerator TEE Designs** (NDSS 2026). PDF text
extraction failed; contents [UNCLEAR — manual fetch needed]. Likely
relevant for the consumer-GPU-passthrough vs CC-GPU framing.

### 1.2 New MPC+GPU result

**EncFormer** (arXiv 2604.09975, April 2026). Two-party FHE+MPC with
CKKS kernels on A100 GPU. **1.3-9.8× lower latency than prior FHE-MPC
on BERT-base.** Targets both BERT and GPT. Strongest 2026 MPC+GPU
result on BERT-class — the architecture relevant to bge-reranker-v2-m3.

Even at this improvement, BERT-base inference under FHE-MPC is
**seconds**-class, not the ~150 ms/text we hit under GELO at
Qwen3-0.6B scale. For 50-candidate reranking that's >5s/query under
EncFormer vs <1s under GELO+CAPRISE today. **MPC+GPU remains
non-competitive** for the <2× wall-clock target. The conclusion from
rev-4 §6 stands: cross-encoder transformer under MPC/FHE is still
seconds-per-query class.

### 1.3 New non-applicable

**Collaborative Obfuscation** (arXiv 2603.01499, Lin et al., Mar 2026).
"Covariant obfuscation" with dynamic (not static) masks; same family
as GELO but covariance-structured rather than Haar-orthogonal. Claims
resistance to precomputed-basis attack but construction details need
full-paper read. [UNCLEAR — flagged for ingestion.] If the security
argument holds, this is an alternative to GELO's Haar sampling; if
not, GELO remains the only published construction that survives both
ArrowMatch (2602.11088) and Hidden No More (2505.18332) attack classes.

---

## 2. Re-evaluation of rev-4 papers under the reranker lens

Through the lens of (a) composes with GELO fresh-per-batch orthogonal
mask, (b) survives precomputed-basis + Hidden No More attacks, (c)
applicability to cross-encoder vs causal-LM-discriminator reranker
shapes:

| Technique | Mask compose | 2602.11088 safe | 2505.18332 safe | Cross-enc fit | Causal-LM-rerank fit | Notes |
|---|---|---|---|---|---|---|
| GELO + TwinShield (ours) | n/a (is the mask) | Yes (full-rank per-batch) | Yes (not a permutation) | Plausible | Plausible | No published reranker eval. |
| ObfuscaTune | No (mixes mask with W) | No (static W-mix) | No | Possible | Possible | Memory note: ArrowMatch-broken. **Reject.** |
| STIP / SOTER / TSQP / TLG | No | Broken (6-min recovery) | n/a | — | — | **Reject.** |
| PermLLM / Centaur | No (fixed permutations) | n/a | Broken (99% recovery) | — | — | **Reject.** |
| Delta / AsymML / 3LegRace | Yes | Yes (low-rank, not static basis) | Yes (additive Gaussian + DP) | Possible | Possible | DP-formal but requires per-matrix factorization. Heavy lift for cross-encoder. |
| Shredder (`[CLS]` cut) | Yes (additive noise) | Yes | Yes | **Best near-term cross-encoder option** | n/a | Cheapest to deploy; train noise distribution only. |
| SCX | Yes (different axis: per-user keys) | Yes (per-session) | Yes (one-time keys per generation) | n/a (no KV cache needed) | **Yes** (designed for decoder) | Already in Edgequake. Useful for LLM serving, not single-pass rerank. |
| Amulet (softmax-equiv π) | Composes with GELO | Yes (per-batch π) | Yes if combined with σ-noise + shield rows | Yes | Yes | Already implemented in our stack. Off by default at short n. |
| ARROWCLOAK (per-row scaling) | Defensive add-on | Yes | n/a | Possible | Possible | Defense-in-depth for OutAttnMult; no known attack against scalar version. |
| p²RAG / PRAG / SANNS / Panther / Tiptoe | n/a (MPC/HE) | n/a | n/a | **No cross-encoder under any** | n/a | All cap at bi-encoder cosine. |
| HR+QDA (1.77M params) | n/a (small model) | n/a | n/a | Lightweight CE | n/a | Run client-side or in-TEE without offload. |
| Opal (TDX + B200 CC) | n/a (different threat model) | Yes | Yes | Possibly no CE used | Yes | Confidential GPU assumption we avoid. |
| Slalom (CNN-era Freivalds) | Yes | Yes | Yes | Yes | Plausible | Integrity primitive only; our U-Verify is its descendant. |
| EncFormer (MPC+GPU) | n/a (pure crypto) | Yes | Yes | Yes (BERT-base) | Yes | Seconds-class; not competitive. |
| Fission (IACR 2025/653) | n/a | Yes | Yes | Yes (ModernBERT <5s) | Yes (LLaMA-3-1B <20s) | Order-of-magnitude slower than TEE+GPU. |

**Bottom line:** the GELO+TwinShield+Amulet stack remains the **only
published construction** that plausibly covers a cross-encoder or
causal-LM-discriminator reranker under the openweight + per-batch-mask
+ TEE-trusted + GPU-untrusted threat model. Every other extant
TEE-based proposal either (a) targets private weights, (b) assumes a
confidential GPU, or (c) falls to one of the two attack papers above.

The cross-encoder-under-MPC/FHE gap from rev-4 §6 remains open.

---

## 3. Architecture comparison: bge / Qwen3 / jina

Quick reference (full details in agent reports):

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

### 3.1 Why Qwen3-Reranker-0.6B is the primary

The Qwen3-Reranker-0.6B backbone is **byte-identical** to
Qwen3-Embedding-0.6B already shipping in
`crates/gelo-embedder/src/decoder/`. Every GELO protocol primitive —
mask, shield rows, U-Verify, OutAttnMult, permuted attention, length
auto-switch, sensitive-layer exclusion — applies without modification.

Implementation cost is dominated by:
1. LM-head linear loader (or reuse `token_embedding` since
   `tie_word_embeddings=true`).
2. Chat-template plumbing for the Qwen-style
   `<|im_start|>system…<|im_end|>` prompt format.
3. Yes/no token-id lookup at load (must be SHA-pinned alongside the
   tokenizer JSON).
4. A `score(q, d) -> f32` entry point that gathers the last-token
   logits, slices on the two pinned token IDs, computes
   `softmax([no, yes])[1]`.

Estimated effort: **2–3 days** for a working private reranker.
Estimated wall-clock: **~155 ms/(q, d) on AOCL-BLIS post-Tier-2**
(within run-to-run variance of the documented Qwen3-Embedding
baseline at 1.27× plain).

Quality is mid-pack but credible (MTEB-R 65.80, inferred BEIR ~58).
Apache-2.0 license is commercially clean.

### 3.2 Why bge-reranker-v2-m3 is the fallback

XLM-RoBERTa-large is the boring, well-trodden option. Post-LN BERT
GEMMs are exactly what GELO was first validated on. The only protocol
addition is a `Linear(1024 → 1)` classifier head — a single offload-able
or in-TEE linear, depending on policy.

Effort: **~3–4 days**. Latency: comparable to Qwen3 (24 layers ×
1024 ≈ Qwen3's 28 × 1024 in total work; FFN is wider but no SwiGLU
gate). Quality is the lowest of the three at BEIR 56.51 but the
implementation risk is the lowest.

This is the right call if Qwen3-Reranker's yes/no probability
calibration turns out to be unstable on the production corpus, or if
we want a parity check that the protocol's correctness extends beyond
the Qwen family.

### 3.3 Why jina-reranker-v3 is deferred

Three reasons:

1. **CC-BY-NC-4.0 license** kills commercial deployment without
   separate negotiation. Apache/MIT competitors are preferable.
2. **The listwise-packed forward pushes n to 16k+**, which requires
   the `gelo-llm.md` §3 fused permuted-attention work (FlashAttention
   inside our protocol) to land first. That's a 5–7 week prerequisite.
3. Per-query full recompute (no doc-encoding cache) makes it
   expensive when the listwise advantage is small (single-doc
   reranking).

It is the strongest candidate to revisit **after** the LLM-serving
stack is up, since it shares fully with that work. The 5+ point
BEIR NDCG@10 lead over the other two is real — but contingent on
fused attention being available.

---

## 4. Open problems specific to reranking

Most of these are *new* concerns that didn't appear in the
embedding-only protocol design.

### 4.1 Score-export leakage

The reranker's *output* is a plaintext scalar per (q, d) pair. Even
with the forward pass fully protected by GELO+TwinShield, the exported
score reveals query-document alignment under enough queries. None of
our current encryption layers protect this.

Mitigation candidates (no published paper picks one yet):

- **Rank-only export**: return ordered chunk IDs, not scores. Loses
  fine-grained pipeline composability.
- **Quantized score export**: round to N bits before exit. Bounds
  per-query information.
- **Bounded score noise**: add small DP noise to the exported score.
  Requires an accounting model for cumulative leakage across queries.
- **Score-tensor masking inside the TEE**: mask the scores under the
  same mask as the embeddings (CAPRISE-style); decrypt only at the
  client. Heaviest option; cleanest privacy.

This is the **single largest gap** in the round-2 design. Recommend
spec'ing it before implementation — `gelo.md` §6 doesn't address it
because embeddings exit CAPRISE-encrypted, but rerank scores don't.

### 4.2 Cross-encoder + AES-decrypted chunks inside the CVM

Rev-4 noted that cross-encoder rerankers must AES-decrypt the k'
candidate chunks' *text* inside the TEE. `gelo.md` doesn't currently
address the AES key-transfer path for chunk text. Verify:

- The attestation flow (`SnpTrustedExecutor::evidence`) already binds
  `model_identity` + `scheme_identity`; an additional `aes_key_holder`
  identity must enter REPORT_DATA for the relying party to know the
  chunk-AES key is provisioned correctly.
- The Qwen3-Reranker prompt template embeds chunk text directly; the
  AES decryption happens inside the CVM before tokenization. No
  protocol-layer change to GELO, but a runner-layer change to wire
  the key.

This is an engineering concern more than a research one; flagged for
the implementation pass.

### 4.3 Empirical attack resistance for the reranker shape

All published attack papers (2602.11088, 2505.18332,
`qsxltss/Game-of-Arrows`) target embedding/decoder inversion. None
evaluate against the (q, d) → scalar reranker forward. The memory note
in `gelo_research_round_2.md` argues GELO is safe by construction, but
that argument hasn't been empirically validated against the reranker
forward.

Recommend: run `Game-of-Arrows` against the reranker forward in mock
mode as a release-gate test, similar to the existing
`tests/bss_recovery.rs` regression guard. ~1 week of work; not on the
critical path but should land before any production deployment claim.

### 4.4 Listwise (jina-v3) under the protocol

Deferred entirely. The listwise-packed n=16k forward needs:

- Fused permuted attention (`gelo-llm.md` §3) — open prerequisite.
- Encrypted-KV-in-SWIOTLB (per-batch state, not weights) — open design.
- Per-doc score gather logic with permutation-invariant correctness
  — straightforward but new.

File for revisit after the LLM-serving stack lands.

---

## 5. Recommendations

### 5.1 Architecture target

- **Primary: Qwen3-Reranker-0.6B.** Lowest friction, highest code
  reuse, validated backbone, Apache-2.0 license.
- **Fallback: bge-reranker-v2-m3.** Parity check + safer bet if
  Qwen3-Reranker's calibration is unstable.
- **Defer: jina-reranker-v3.** Revisit after `gelo-llm.md` §3 lands.

### 5.2 Protocol composition for the primary path

The existing GELO+TwinShield+CAPRISE+SEV-SNP stack covers
Qwen3-Reranker-0.6B directly. Required *new* design work:

1. **Score-export discipline** (§4.1) — pick a mitigation, document
   the threat model, gate behind a runner config.
2. **AES key path for chunk text** (§4.2) — extend
   `scheme_identity`'s REPORT_DATA contribution, add a runner-layer
   key-provision step.

Optional add-ons (defense-in-depth, not on critical path):

3. ARROWCLOAK per-row scaling for OutAttnMult — only relevant if the
   permuted-attention path is forced on for long-doc reranking.
4. SCX-style stateless HKDF mask derivation — required for
   multi-tenant deployment but skippable for single-tenant.

### 5.3 What still needs research

1. **Cross-encoder transformer under MPC.** Still unsolved. EncFormer
   is closest but seconds-class. Not on critical path; relevant only
   if TEE assumption later breaks.
2. **Score-leakage formal model.** No published paper defines the
   right notion of "rerank score privacy" beyond DP. Worth a 1–2 week
   research spike before locking in a mitigation in §5.2.1.
3. **Game-of-Arrows on the reranker shape.** Empirical attack-resistance
   bench (§4.3).
4. **Listwise rerankers under the protocol** (§4.4). Filed for after
   LLM serving.

### 5.4 Edgequake ingestion list

Highest priority (new, post-Apr-2026, not in corpus):

1. **Opal: Private Memory for Personal AI** (arXiv 2604.02522,
   April 2026). The CC-GPU+TDX+ORAM comparison point.
2. **SecureInfer** (arXiv 2510.19979). Closest architectural peer,
   different threat model.
3. **Privacy-Aware Split Inference with Speculative Decoding**
   (arXiv 2602.16760). Naked-fp16 negative result.
4. **EncFormer** (arXiv 2604.09975). Strongest 2026 MPC+GPU on BERT.
5. **Collaborative Obfuscation** (arXiv 2603.01499). Possible GELO
   alternative; needs full-paper read to verify attack resistance.
6. **Privacy-Preserving LLM Inference in Practice — Comparative
   Survey** (IACR ePrint 2026/105). Field-state survey to replace
   rev-4's bibliography.

Reranker-specific (referenced in design discussion, not in corpus):

7. **bge-reranker-v2-m3** / FlagEmbedding technical report
   (Chen et al., 2024).
8. **Qwen3-Reranker technical card / paper** (Yang et al., 2025).
9. **Jina-Reranker-v3** (arXiv 2509.25085, Wang et al.).
10. **HR+QDA** (Maringan & Fitrianah 2026, *Discover Computing*).

Cited in rev-4 but absent from Edgequake:

11. **p²RAG** (arXiv 2603.14778, Ming et al., 2026).
12. **FlashAttention-2/-3** (Dao 2023/2024) + FLASH-D (arXiv
    2505.14201). Required for `gelo-llm.md` §3.
13. **Delta / AsymML / 3LegRace** (Niu et al. 2022/2023).
14. **Shredder** (Mireshghallah et al.).
15. **Hidden No More** (arXiv 2505.18332). Already referenced
    extensively in `gelo_research_round_2.md`; ingest as standalone PDF.
16. **Tiptoe** (Henzinger et al., SOSP 2023).

The corpus is **adequate** for the protocol-side research but **thin**
on the reranker model side. Items 7–10 above should be ingested
before the implementation pass begins.

---

## 6. Decisions deferred to implementation

These are notes for the implementer, not commitments:

1. **Per-pair vs batched-listwise API.** Recommend per-pair for the
   Qwen3-Reranker path; add batched later only if needed.
2. **Yes/no token resolution.** Look up at load time, pin alongside
   tokenizer SHA-256 in `model_identity`.
3. **Chat-template handling.** Inline (deterministic + attested)
   rather than reading from `tokenizer_config.json` at runtime.
4. **Skip-last-layer default.** Consider defaulting `skip_last_layer
   = true` for the reranker path — the LM head's final projection is
   adjacent to the cleartext score and is the highest-leakage layer.
   Cost: ~5 ms/text.
5. **Truncation policy.** Qwen3-Reranker accepts 32k context;
   suggest 2048 per-pair limit (covers BEIR queries + most docs).
   Surface via env.

---

## References

- `docs/research/private-reranking-research.md` — rev-4 baseline this
  round complements.
- `docs/prototype/gelo.md` — current protocol.
- `docs/prototype/gelo-llm.md` — LLM extension plan (fused permuted
  attention, SCX KV-cache).
- `memory/gelo_research_round_2.md` — 2026-05-14 attack-resistance
  analysis.
- Kaviani et al., "Opal: Private Memory for Personal AI." arXiv
  2604.02522 (April 2026).
- "SecureInfer: Heterogeneous TEE-GPU Architecture for Privacy-Critical
  Tensors." arXiv 2510.19979 (Oct 2025).
- Cunningham, "Privacy-Aware Split Inference with Speculative Decoding
  for LLMs over WANs." arXiv 2602.16760 (Feb 2026).
- "EncFormer: Secure and Efficient Transformer Inference over
  Encrypted Data." arXiv 2604.09975 (April 2026).
- Lin et al., "Towards Privacy-Preserving LLM Inference via
  Collaborative Obfuscation." arXiv 2603.01499 (March 2026).
- "Privacy-Preserving LLM Inference in Practice: A Comparative Survey."
  IACR ePrint 2026/105.
- Chen et al., "BGE-M3 / bge-reranker-v2-m3." arXiv 2309.07597.
- "Qwen3-Reranker." arXiv 2506.05176 (Tongyi Lab, 2025).
- Wang et al., "Jina-Reranker-v3: Last but Not Late Interaction."
  arXiv 2509.25085.
- Wang et al., "Vulnerabilities in Partial TEE-Shielded LLM Inference
  with Precomputed Noise." arXiv 2602.11088.
- Wang et al., "Hidden No More: Attacking and Defending Private
  Third-Party LLM Inference." ICML 2025, arXiv 2505.18332.
- Wang et al., "Game of Arrows / ARROWMATCH / ARROWCLOAK." USENIX
  Security 2025.
- Dao et al., "FlashAttention-2/-3."
- "FLASH-D: FlashAttention with Hidden Softmax Division." arXiv
  2505.14201.
- Zeng et al., "SCX: Stateless KV-Cache Encoding for Cloud-Scale
  Confidential Transformer Serving." SIGCOMM 2025.
- Mao et al., "Amulet: Fast TEE-Shielded Inference for On-Device
  Model Protection." arXiv 2512.07495.

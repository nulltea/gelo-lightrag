---
type: reference
status: current
created: 2026-05-18
updated: 2026-05-18
tags: [comparison, path-1, path-2]
---

# Private LLM Inference Comparison: GELO vs AloePri on Gemma E2B/E4B

> **Plan date:** 2026-05-18. Shared framework for two parallel
> development paths:
> - **Path 1** (`path-1-gelo-gemma.md`): GELO TEE-GPU split inference
>   for Gemma E2B/E4B. Continues in the original worktree.
> - **Path 2** (`path-2-aloepri-gemma.md`): AloePri offline-rewrite
>   inference for Gemma E2B/E4B. Develops in `../private-rag-path-2`
>   worktree.
>
> **Goal:** Direct head-to-head comparison of two private-inference
> approaches with categorically different threat models, on the same
> model family at the same scaling delta, with shared evaluation
> infrastructure to make the comparison meaningful.
>
> **Decision context:** Round-2 research determined GELO and AloePri
> address different adversaries with structurally different security
> arguments (per-batch information-theoretic vs static empirical).
> The user accepted AloePri's weakened threat model in exchange for
> its scalability claim (validated to 671B) and TEE-independence.
> See [`../research/aloepri-vs-gelo.md`](../research/aloepri-vs-gelo.md)
> for the technical comparison.

---

## Definitions

| Term | Meaning |
|---|---|
| **Path 1** | GELO TEE-GPU split inference (this project's existing protocol family). |
| **Path 2** | AloePri offline-rewrite inference (new addition; pre-existing reference code at `sheng1feng/Aloepri`). |
| **E2B / E4B** | Gemma 4 effective-2B and effective-4B variants. |
| **PLE** | Per-Layer Embeddings — Gemma 3n / Gemma 4 cache table indexed by `token_id`. |
| **K=V trick** | Gemma 4 global-layer optimization: K and V tensors share storage. |
| **p-RoPE** | Proportional RoPE: rotation applied only to first p·d_head dims (p=0.25 for Gemma 4 global). |
| **TPOT** | Time Per Output Token (steady-state decoding throughput). |
| **TTFT** | Time To First Token (prefill latency). |
| **TTRSR** | Text Token Recovery Success Ratio under inversion/recovery attacks. |
| **MatFormer** | Nested-model architecture; Gemma 4 uses it. **Note:** E2B and E4B have different attention ratios (4:1 vs 5:1), so they are NOT pure MatFormer slices of each other. |

---

## 1. Validation: Both paths are compatible with Gemma E2B/E4B

Confirmed compatibility per architectural feature (see chat
2026-05-18 for derivation):

| Feature | Path 1 (GELO) | Path 2 (AloePri) |
|---|---|---|
| Hybrid attention 5:1 (E4B) / 4:1 (E2B), W=512 | ✓ Clean (in-TEE local, offload global) | ✓ Clean (Algorithm 2 is attention-pattern-agnostic) |
| PLE table 262144×256×N | ⚠ Must live in TEE DRAM (~1.3 GB E4B int8) | ✓ Token-permute the table at offline-rewrite |
| K=V trick in global layers | ✓ Cheaper than separate K/V (one mask) | ⚠ Breaks Algorithm 2; **must un-tie K/V on server side** (~10% extra KV cache) |
| p-RoPE (p=0.25) | ✓ One-line change in RoPE module | ✓ R̂_qk becomes p-fraction block-diagonal |
| 8-to-1 GQA | ✓ Already handled (Qwen3 uses GQA) | ✓ Paper explicit: works on MHA/MQA/GQA |
| MatFormer (E2B ⊂ E4B) | ✓ Compile-time slice | ⚠ Different attention ratios → two separate obfuscated artifacts |
| Multimodal encoders | Out of scope (text-only v1) | Out of scope (text-only v1) |
| vLLM Gemma 4 mainline support | N/A | **RISK** — must verify before M2.3; fallback: Gemma 3 |

**No hard blockers. Two soft caveats** captured per path.

---

## 2. Model choice + scaling axis

**Primary target:** Gemma 4 E2B → E4B (2.3B → 4.5B effective).

- Both share the PLE machinery — validates the PLE-in-TEE fix end-to-end.
- Both share the hybrid-attention pattern (with the 4:1 vs 5:1 ratio
  difference between E2B and E4B noted).
- 2× scaling delta is enough to expose scaling pathologies if any.
- Edge-class deployment target aligns with the project's overall
  hardware envelope (Hetzner EPYC + consumer GPU passthrough; no
  H100-CC dependency).

**Stretch goal:** Gemma 4 31B dense (5:1 hybrid, W=1024, no PLE).
Closes ~50× of GELO's currently-validated scale gap; matches
AloePri's published frontier-dense regime. Memory budget: 31 GB
int8 fits tight in 32 GB CVM, comfortable in 64 GB.

**Explicit future work:** Gemma 4 26B A4B (MoE). Requires the
CryptoMoE balanced-dispatch defense (round 2 §C) — separate
research stream. Out of scope for v1 comparison.

---

## 3. Shared milestones (M0.*)

These milestones must complete **before** either path can produce
publishable comparison numbers. They live in the **original
worktree** (Path 1) and are referenced by Path 2.

### M0.1 — Common test corpus

**Goal:** A pinned, version-controlled set of prompts both paths
evaluate on, sized for fast iteration + release-grade comparison.

**Tier 1 (per-iteration smoke, < 1 min/run):**
- 32 prompts drawn from MMLU + IFEval, balanced across task types
- Used by every CI run

**Tier 2 (release gate, ~30 min/run):**
- 500 MMLU 0-shot
- 500 IFEval instruction-following
- 200 PIQA physical commonsense
- 200 HumanEval code generation
- Matches AloePri paper's eval set for direct paper-comparison

**Artifact:** `evals/private-inference-corpus/` — JSON files,
pinned commit hash, scripts to refresh from upstream.

**Owner:** Path 1 worker (this session). Path 2 reads.

**Effort:** ~3 days.

### M0.2 — Common eval harness

**Goal:** One CLI that takes a model+protocol identifier and runs
the tiered corpus, emitting standardized metrics.

```
evals/run-eval.py \
    --model gemma-e4b \
    --protocol gelo|aloepri|plain \
    --tier 1|2 \
    --out results/<run-id>.json
```

**Metrics emitted per run:**
- Accuracy: MMLU accuracy, IFEval pass rate, PIQA accuracy,
  HumanEval pass@1
- Performance: TTFT, TPOT, peak-memory (TEE-RAM for Path 1, total
  for Path 2), throughput (tok/s)
- Quality: top-1 token match against plain reference, cosine
  similarity of final hidden states

**Artifact:** `evals/run-eval.py` + `evals/lib/`.

**Owner:** Path 1 worker.

**Effort:** ~5 days.

### M0.3 — Common attack benchmark

**Goal:** Port AloePri's `src/security_qwen/` attack suite as a
standalone harness, runnable against either protocol's output.

Detailed plan in `../research/aloepri-vs-gelo.md` §4.1. Three
phases:
1. Snapshot capture in TrustedExecutor (Path 1) and in the
   AloePri client wrapper (Path 2)
2. Attack-harness wiring against pinned AloePri commit
3. Integration into release-gate CI

**Owner:** Path 1 worker writes the harness; Path 2 wires its own
snapshot capture into the same harness.

**Effort:** ~2-3 weeks (already on Path 1's roadmap as P1 spike).

### M0.4 — Comparison report

**Goal:** Single document collecting results from M1.* and M2.*
into the comparison criteria of §5 below.

**Artifact:** `docs/research/gelo-vs-aloepri-on-gemma.md`.

**Owner:** Whoever finishes second writes; whoever finishes first
contributes their numbers.

**Effort:** ~1 week after both paths complete.

---

## 4. Cadence and synchronization

| Phase | Path 1 (this worktree) | Path 2 (worktree 2) | Sync point |
|---|---|---|---|
| Setup | Write Path 1 plan (this doc batch) | Read handoff doc, set up env | — |
| Week 1-2 | M0.1 corpus + M0.2 harness | Verify vLLM Gemma 4 support; fall back to Gemma 3 if needed | M0.1 + M0.2 land on master |
| Week 3-4 | M1.1 Gemma loader | M2.1 AloePri import + Gemma adapter | — |
| Week 5-8 | M1.2 PLE in TEE, M1.3 hybrid attn, M1.4 K=V, M1.5 p-RoPE | M2.2 offline obfuscation pipeline | — |
| Week 9-10 | M1.6 E2B bench | M2.3 vLLM integration, M2.4 client | — |
| Week 11-12 | M1.7 E4B scaling, M1.8 accuracy | M2.5-M2.7 benches + accuracy | **Comparison sync** |
| Week 13-14 | M0.3 attack harness on Path 1 | M2.8 attack harness on Path 2 | Both run same suite |
| Week 15 | M0.4 comparison report | M0.4 comparison report | Joint write-up |

**Total: ~15 weeks** in calendar time if both paths progress in
parallel. Path 1 critical path is M1.3 (hybrid attention).
Path 2 critical path is M2.3 (vLLM integration, dependent on
upstream Gemma 4 support).

**Communication:**
- Shared `master` branch holds M0.* artifacts.
- Path 1 develops on `path-1-gelo-gemma` branch.
- Path 2 develops on `path-2-aloepri-gemma` branch (separate
  worktree at `../private-rag-path-2`).
- Weekly handoff: each side updates its plan doc's "status" section.

---

## 5. Comparison criteria

These are the questions M0.4 (the final comparison report) must
answer:

### 5.1 Performance

| Metric | How measured |
|---|---|
| TTFT @ 512-token prompt | One run per (model, protocol); average of 10 |
| TPOT (steady-state) @ 256-token continuation | Same |
| Peak memory (TEE RAM for Path 1; total VRAM for Path 2) | OS-level read |
| Throughput (tok/s) at batch=1 and batch=32 | Wall-clock |
| Wall-clock overhead vs plain baseline (% slowdown) | TPOT relative to plain Gemma |

**Expected**: AloePri wins on overhead (<10% TPOT per paper); GELO
6-27% per `gelo.md` §7. Path 1's win is the per-batch fresh
randomness, not the wall-clock.

### 5.2 Scaling

| Metric | How measured |
|---|---|
| Overhead delta from E2B → E4B | Does either approach degrade faster than the other as model grows? |
| Memory delta E2B → E4B | Same |
| TTFT delta E2B → E4B | Same |
| If stretch 31B lands: same set | Same |

**Expected**: AloePri's offline cost is one-time and scales with
weight volume. GELO's online cost scales with hidden_size²
(Householder sample) + per-layer offload. At 31B, GELO's per-batch
mask sample is ~9-15 ms (vs 9 ms at 0.6B today) — still small
absolute.

### 5.3 Accuracy

| Metric | How measured |
|---|---|
| MMLU 0-shot accuracy delta vs plain | M0.2 harness |
| IFEval pass-rate delta | Same |
| PIQA accuracy delta | Same |
| HumanEval pass@1 delta | Same |
| Top-1 token match rate vs plain reference | Same |
| Final hidden-state cosine similarity vs plain | Same |

**Expected**: Path 1 is bit-exact in fp32 (`top1 = 1.00` per
`gelo.md` Appendix). Path 2 loses 0-3.5% per AloePri paper on
Qwen/Llama/DeepSeek — Gemma 4 untested by paper. Direct
measurement is the point.

### 5.4 Attack-resistance

| Attack | What it measures |
|---|---|
| VMA (Vocabulary-Matching) | Recovery of obfuscation given plaintext+obfuscated weights |
| IA (Invariant) | Recovery via weight-relation invariants |
| ISA (Internal State) | Prompt recovery from hidden states |
| IMA (Inversion Model) | Adaptive learned inverter from observation→token |
| NN (Nearest Neighbor) | Token recovery via embedding NN lookup |
| TFMA (Token Frequency Matching) | Recovery via corpus frequency stats |
| SDA (Substitution Deciphering) | n-gram based recovery |

Per-attack TTRSR (lower is better). The AloePri paper reports
< 15% under each on Qwen2.5-14B; we'll re-run on Gemma E2B/E4B
for both protocols.

**Expected**: GELO should approach 0% (per-batch fresh Haar mixing
is the strong primitive); AloePri's TTRSR should track its
published numbers (5-15%). If GELO numbers are much higher than
expected, that's a regression-grade bug. If AloePri numbers are
much higher than the paper claims, that's a generalization gap
worth documenting.

### 5.5 Engineering complexity

| Metric | How measured |
|---|---|
| Lines of code added | `git diff` against current `master` |
| New external dependencies | Cargo.toml / requirements.txt deltas |
| Ongoing maintenance surface | Subjective — each path summarizes |
| Compatibility with existing inference stacks | Path 2: vLLM/SGLang; Path 1: ours alone |

**Expected**: Path 2 wins on integration (vLLM unmodified). Path 1
wins on dependency footprint (no Python serving stack).

### 5.6 Threat-model framing

Recapped for the report (no measurement; descriptive only).

| Property | Path 1 | Path 2 |
|---|---|---|
| Adversary | Honest-but-curious GPU + host operator | Honest-but-curious server |
| Per-request entropy | Fresh Haar `A_b` per forward | None — static `τ`, static `θ̃` |
| Security argument class | Information-theoretic per batch | Empirical (passes named attacks) |
| Trust assumption | SEV-SNP silicon | None (no TEE) |
| Open-weight safety | Per-batch mixing immune to public-W attacks | Degrades to TTRSR 5-15% under public W (per paper) |

---

## 6. Risks and contingencies

### R1: vLLM mainline support for Gemma 4 not ready

**Symptom**: M2.3 blocks because vLLM doesn't have Gemma 4 model
implementation in mainline.

**Mitigation**: Path 2 falls back to Gemma 3 (1B → 4B same hybrid
family, no PLE). Document the substitution; rerun M0.* corpus on
Gemma 3. Path 1 sticks with Gemma 4 unless instructed otherwise.
Comparison becomes "Path 1 on Gemma 4 vs Path 2 on Gemma 3" with
an explicit caveat that the model architectures differ slightly
(no PLE in Gemma 3 → no PLE-leak fix needed → not directly
comparable on that axis).

### R2: GELO mask cost grows unsustainably at E4B

**Symptom**: M1.6 E2B bench shows acceptable overhead; M1.7 E4B
bench shows 4-5× regression. Per-batch Householder sample cost
grows as O(d²) — at d=2560 (E4B), that's ~3× the d=1536 (E2B)
cost. Should still be <50 ms/forward, but worth measuring.

**Mitigation**: Hidden-size scaling is known; if it bites, the
existing "stateless mask derivation via HKDF" lever (`gelo.md` §8
lever #6) becomes more attractive — HKDF expansion is ~µs vs ms
for Haar QR.

### R3: K=V un-tying on AloePri global hurts memory budget more than expected

**Symptom**: E4B AloePri server-side memory exceeds available
VRAM by more than the predicted ~10%.

**Mitigation**: The 10% estimate is on KV cache only (global
layers are 1/6 of total in E4B), not total weight memory. If real
delta exceeds budget, consider whether to allow Q̂_k = Q̂_v (sec
loss to be documented) or to drop AloePri's per-V `Û_vo`
transform. Both are stopgaps; document in M0.4.

### R4: Attack suite porting takes longer than 3 weeks

**Symptom**: M0.3 slips past Week 14.

**Mitigation**: Both paths still produce comparison-quality
numbers without M0.3; document M0.4 with an "attack-resistance:
deferred" section. Land M0.3 post-comparison.

### R5: Diverging code between worktrees causes merge pain

**Mitigation**: Both paths only write to **disjoint directories**.
- Path 1 modifies `crates/gelo-embedder`, `crates/gelo-protocol`,
  `crates/gelo-gpu-wgpu`, `crates/gelo-snp-runner`.
- Path 2 writes to `vendor/aloepri-py/` (Python), `evals/`,
  `scripts/path-2-*`. No Rust touches.
- Shared infrastructure (`evals/private-inference-corpus/`,
  `evals/run-eval.py`, `evals/attack-harness/`) lives on master
  and is updated by Path 1, consumed by both. Path 2 doesn't
  modify these; if Path 2 needs changes, send a PR back to master.

---

## 7. What "done" looks like

The comparison is complete when M0.4 (`gelo-vs-aloepri-on-gemma.md`)
includes:

- [ ] Table 5.1 fully populated for both paths on E2B and E4B (and
      31B if stretch lands).
- [ ] Table 5.2 fully populated, with a paragraph commentary on
      scaling behavior.
- [ ] Table 5.3 fully populated; any accuracy losses documented
      with hypothesized root cause.
- [ ] Table 5.4 fully populated; gaps between GELO and AloePri
      TTRSR documented; gaps between paper-AloePri claims and
      empirical AloePri-on-Gemma documented.
- [ ] Table 5.5 fully populated; LOC counts and dependency
      manifests included as appendices.
- [ ] Recommendation paragraph: under which deployment scenarios
      should the project use Path 1 vs Path 2.

This output feeds back into [`../research/private-llm-inference-round-2.md`](../research/private-llm-inference-round-2.md)
as round-3.

---

## References

- [`../research/aloepri-vs-gelo.md`](../research/aloepri-vs-gelo.md)
  — technique-level comparison rationale.
- [`../research/private-llm-inference-round-2.md`](../research/private-llm-inference-round-2.md)
  — Gemma 4 architectural analysis (round 2 §D).
- [`../dev/prototype/gelo.md`](../dev/prototype/gelo.md) — GELO protocol
  reference.
- [`../dev/prototype/gelo-llm.md`](../dev/prototype/gelo-llm.md) — LLM-
  generation forward plan (Path 1 baseline).
- AloePri paper: arXiv 2603.01499.
- AloePri reference code: `github.com/sheng1feng/Aloepri`.

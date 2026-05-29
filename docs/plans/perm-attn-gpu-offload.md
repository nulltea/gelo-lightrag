---
type: plan
status: current
created: 2026-05-29
updated: 2026-05-29
tags: [gelo, dgpu, attention, gpu, persistent-kv, permutation, security, flash-attention]
companion: [2026-05-22-dgpu-attention-revival]
---

# Permutation-shielded GPU attention offload — frozen-prefix / active-tail persistent K/V

The design for moving the decode in-TEE attention bottleneck onto the
dGPU (RTX 5090) without violating the F1+ threat model. Supersedes the
"Item 1 persistent K/V" sketch in
[`2026-05-22-dgpu-attention-revival.md`](../handoffs/2026-05-22-dgpu-attention-revival.md)
with a concrete cache structure, decode-step mechanism, and threat
model.

## Why this exists (the binding measurement)

The Step-0 triage (`amulet_attention_r1_4`, RTX 5090, 2026-05-29 —
`bench-results/amulet-attn-triage-5090-2026-05-29.log`) measured the
naive GPU attention path at the decode shape:

| n_kv (decode, B=8) | in-TEE rayon | GPU full-upload (no mask) | GPU ÷ in-TEE |
|---:|---:|---:|---:|
| 256  | 1.08 ms | 71.6 ms | **66× slower** |
| 1024 | 4.47 ms | 281 ms  | **63× slower** |
| 2048 | 11.35 ms| 510 ms  | **45× slower** |

The GPU time scales linearly with n_kv — an *upload* signature, not a
compute one. `fused_attention_batched` re-uploads **and re-converts**
the entire K/V cache (f32→f16) on every call, then blocks on readback;
that fixed pipeline cost (0.13 GB/s effective, ~200× below PCIe
bandwidth) is the entire 45–66× gap. The 5090's HBM and tensor cores
never get to matter. **Naive GPU attention is non-viable; viability is
gated entirely on persistent K/V** — keeping the cache device-resident
so only the per-step delta moves.

## Decisions taken (grill, 2026-05-29)

- **Scope:** phased — VRAM-resident K/V first (solves the production
  n_kv≤~16k shape), with an NVMe spill tier designed-in for
  beyond-VRAM context (n_kv ≥ ~16–32k, where the cache exceeds the
  32 GB card). The session-handle API must be spill-ready from day one.
- **Cover scheme:** block-fresh-π (hold the existing Amulet +
  Hidden-No-More permuted-attention cover's permutation fixed across N
  decode steps). **Fallback** if the perf gate fails: TwinShield-Xue
  additive softmax-blinding (arXiv 2507.03278).
- **Perf gate:** a fail-fast persistent-buffer microbench (~1 day) that
  holds `k_t`/`v_t` device-resident across iterations and measures
  steady-state per-step cost **and** the boundary re-permute cost,
  against the 11.35 ms in-TEE baseline — *before* committing to the
  2–3 week session-resident substrate refactor. The σ-vs-N security
  spike runs in parallel.
- **Backend:** **CUDA is the production default** (the deployment target
  is Nvidia dGPU under SEV-SNP — CUDA is the deployment reality, not a
  fork); **Vulkan is the development backend** (portable, cheap SPIR-V
  autotune for fast iteration on the iGPU dev box + the iGPU track). The
  engine already swaps `WgpuRuntime` ↔ `CudaRuntime` behind `Rt`/`Dev`
  aliases, so this is a build-flag choice. CUDA's ~10–16% kernel win
  (chronicle §8/§9) is a bonus on top of the native path; its heavy nvrtc
  cold-autotune (~27 s one-time at B=8) is amortized by a long-running
  prod server but is why dev stays on Vulkan. **Kernel consequence:** the
  production partial-stats kernel must be authored in **cubecl** (compiles
  to both CUDA and SPIR-V) — a hand-rolled WGSL kernel is Vulkan-only and
  cannot serve the CUDA prod path, so it drops to at most a dev-only
  optimization. End-to-end acceptance perf (gate tiers 2–4) is measured
  on the **CUDA** backend (warm).
- **Scope (prefill):** decode-only for v1 — the hybrid targets the
  decode attention bucket (34% B=8 / 52% B=1 of decode wall). Prefill
  attention offload (~12–21% of prefill wall; the ~4 GB scores-tensor
  materialization on dGPU) is a **fast-follow**: it shares the deferred
  FlashAttention-D kernel (prefill tiling) but needs no session, so it
  slots in once the kernel lands.
- **Cache structure:** frozen-prefix / active-tail hybrid (below).
- **Kernel:** phased. The gate-1 microbench uses a minimal
  resident-buffer variant of `fused_attention_batched` (whole cache
  resident, full GPU softmax, no tail-split, no partial stats) — enough
  to measure resident-read per-step cost vs the 11.35 ms baseline. The
  production partial-stats / GQA-aware / tail-merge kernel is **deferred**
  until gates 1–3 clear — chosen with measured numbers in hand. Because
  CUDA is the prod backend, the kernel must be **cubecl-authored**
  (compiles to CudaRuntime + SPIR-V); the choice narrows to a
  **cubecl-custom FlashAttention-D vs an upstream-burn `flash_attention`
  extension**. Raw WGSL is Vulkan-only → off the prod table (at most a
  dev-only optimization). The microbench stub may use the existing
  `fused_attention_batched` (already cubecl) on either backend.
- **Session-resident K/V API:** engine-owned session handle
  (`create_session` / `append` / `attend` / `refresh_block` / `drop`) on
  `GpuOffloadEngine`, with a pluggable `SpillProvider` for the Phase-2
  NVMe cold tier (Phase 1 = VRAM-only null provider). See the
  substrate-refactor section; lands post-gates.
- **V hardening:** on the resident cache, V carries the token-axis
  `perm_kv` *plus* a feature-axis orthogonal rotation `O_v` (with a
  matched `O_qk` on Q,K that leaves Q·Kᵀ invariant). Exactly correctable
  (TEE applies `O_vᵀ` after the online merge); hides V's absolute
  coordinates. **v1 default: apply `O_v` once at prefill (session-fixed)**
  — this keeps the boundary re-permute cheap, since the per-block `O_v`
  rotation is otherwise the dominant re-permute cost (gate 1). But `O_v`
  carries its own recovery clock (covariance / cumulant alignment — see
  threat model), independent of `perm_kv`'s HNM clock; the
  covariance-alignment spike (gate 3) is the **gate on this default** —
  if it shows `O_v` recoverable within a production context, the default
  is demoted to refresh every `M` blocks (with the structured-orthogonal
  O(L·d) signed-permutation trick to keep that affordable). Note `perm_kv`
  and the σ-noise on K still refresh per block regardless — only `O_v`'s
  cadence is relaxed. Hiding V's *geometry* (Gram / pairwise distances,
  rotation-invariant) is **deferred** — no correctable transform on the
  permutation path achieves it; if it becomes a hard requirement it
  re-ranks TwinShield-Xue to primary (fallback section).

## The structure

The resident K/V cache is split at a moving boundary `p`:

```
            absolute positions 0 ───────────────────► n_kv
VRAM-resident:  [ ████████ FROZEN PREFIX [0,p) ████████ │ active tail [p, p+j) ]
                  permuted under perm_kv^(b), K σ-noised   stays IN-TEE, plaintext
                  GPU-resident, read at HBM ~1.8 TB/s      ≤ N rows, never uploaded
```

- **Frozen prefix** — the context committed at the *start* of the current block. Uploaded once, under a single fixed `perm_kv^(b)`, with σ-noise already baked into the K rows. The GPU holds these bytes unchanged for all N steps of the block. This is the bulk (long context) and it's what gets read every step at HBM bandwidth — the 20× win over DDR5.
- **Active tail** — the ≤ N tokens generated *during* this block. Small. Lives in TEE enclave memory, plaintext, never uploaded.

## A decode step (the load-bearing mechanism: online-softmax split)

Attention over `prefix ∪ tail` is computed by splitting the key set and merging with FlashAttention's exact running-stats trick — softmax is associative through `(max, sumexp, acc)`:

1. TEE forms the new query `q_t`, permutes it on the q-axis (trivial, n_q=1) and adds σ-noise. Uploads **only `q_t`** — `heads × d_head` ≈ a few KB, microseconds.
2. **GPU** computes the prefix's partial attention state against the resident K/V: `(m_A, l_A, acc_A)` per head — running max, sum-of-exp, and the *unnormalized* value accumulator. It does **not** do the final divide. Reads back `acc_A` (d_head/head) + two scalars — tiny.
3. **TEE** computes the tail's partial `(m_B, l_B, acc_B)` over ≤ N plaintext keys — trivial work.
4. TEE merges the two states exactly (`m=max; l=l_A e^{m_A−m}+l_B e^{m_B−m}; acc=…`), applies `O_vᵀ` to undo the V feature-axis rotation (`acc = acc_true·O_v`), divides, un-permutes via π_q⁻¹. Output.

The expensive O(context) part runs on the GPU over resident data; the per-step PCIe traffic is one query up + one partial-state down. **No tail upload, no per-step re-permute.** That's what gets per-step cost toward sub-millisecond.

Note this *requires* a kernel that returns partial `(m, l, acc)` rather than normalized `probs·V` — which is precisely the FlashAttention-D kernel (revival Item 2+3). So the hybrid and the custom-kernel decision are the same decision; the existing `fused_attention_batched` (full normalized softmax) doesn't expose the stats.

## The block boundary (every N steps)

When the tail fills (N tokens) or a refresh is due:
- TEE folds the tail into the prefix → new prefix `[0, p+N)`.
- Samples a **fresh** `perm_kv^(b+1)` over the new length, re-noises K, uploads. Cost ∝ context length, paid once per N steps. **This is the only heavy transfer, and it's the refresh tax the microbench must measure.**

## Threat model

The trust boundary is the SEV-SNP enclave. **Trusted:** the TEE holds plaintext q/k/v, the secret permutations, the noise RNG, does the merge and un-permute. **Untrusted:** the VFIO-passed GPU — it executes matmul/softmax over whatever it's handed and can read its own VRAM and observe dispatch shapes/timing. The adversary controls the GPU.

Over one block of N steps the GPU sees:
- The frozen prefix: K rows **permuted (`perm_kv^(b)`) + σ-noised**, V rows **permuted, _not_ noised** (the code adds Gaussian noise to Q and K only). Fixed bytes for N steps.
- N query vectors, each q-permuted (trivial) + σ-noised.
- N partial-softmax dispatches against the fixed prefix.

What protects the context:
- **`perm_kv` hides row order** — the adversary sees a *shuffled* set of K/V vectors, not the sequence. Amulet's softmax-permutation equivariance makes this exact: `softmax(PX)·PV` recovers correctly after un-permuting queries, so the GPU never needs to know the order.
- **σ-noise on Q,K** defeats statistical recovery of `perm_kv` across repeated observations (Hidden No More, arXiv 2505.18332).

Where the clock ticks — **and this is the part to grill**:
- HNM recovers a *fixed* permutation at 99%+ given enough observations; σ=0.01 is calibrated for **fresh-per-call** π. Holding `perm_kv^(b)` fixed for N steps gives the adversary N correlated looks; the signal grows ~√N, so σ must scale ~√N to hold resistance. **The block size N _is_ the HNM observation count.** That's the σ-vs-N spike, and it caps N (≈32–64 before the noise degrades model accuracy).
- **The hybrid does _not_ improve this** versus monolithic — same N, same clock. What it buys is purely (a) per-step cost (tail in-TEE, prefix at HBM, no per-step re-permute) and (b) the prefix/tail boundary as the NVMe-spill seam. I want to be explicit so we don't credit it with security it doesn't have.
- **V exposure (and the feature-axis-rotation mitigation):** V carries `perm_kv` *and* a feature-axis orthogonal rotation `O_v` (matched `O_qk` on Q,K leaves the scores invariant). `O_v` is exactly correctable — the GPU returns `acc = acc_true·O_v` and the TEE applies `O_vᵀ` in the merge — so the adversary no longer reads the shuffled value vectors directly; absolute coordinates are hidden. The **accepted residual** is geometry: orthogonal transforms preserve the Gram matrix and pairwise distances `‖v_i − v_j‖`, so the value cloud's configuration leaks regardless of `O_v` / `perm_kv`. Destroying geometry needs *additive* noise on V, which is uncorrectable on this path (the `probs·V` contraction is over the token axis with softmax weights the TEE never sees — see the token-axis argument in the fallback section). For v1 this residual is gated by the c5 AloePri / σ-vs-N spike; making geometry-hiding mandatory re-ranks TwinShield to primary.
- **Two independent clocks (the cadence tension).** `perm_kv` and `O_v` defend different things and are recovered by different attacks, so they tick independently:
  - `perm_kv` hides *position* (sequence order); recovered by the HNM-class attack on attention/score statistics, **fed by query observations** (N per block), defended by σ-noise on Q,K + per-block refresh. → gate 2.
  - `O_v` hides *content* (the value coordinates, hence token identity via vocabulary matching); recovered by **covariance / cumulant alignment** (Procrustes / ICA — JADE, anchor_ica) against the model's known activation statistics, **fed by the number of distinct token-values observed** (grows with context length). → gate 3.
  - The two don't help each other: `perm_kv` shuffles rows but covariance / Gram are computed over the row *set* (permutation-invariant), so `perm_kv` does **not** slow `O_v` recovery; and σ-noise is on Q,K only, so `V·O_v` is observed *noiselessly*, making `O_v` alignment *easier* than `perm_kv` recovery (and we can't noise `V·O_v` — that's the uncorrectable case). Hence `O_v` needs its own refresh cadence `M`, traded against the per-block rotation cost (gate 1).
- One small *benefit*: the freshest N tokens (often the most sensitive, most-attended) stay in-TEE for the whole block and only ever reach the GPU permuted+noised, after a boundary fold.

F1+ is preserved throughout: the GPU only ever sees permuted+noised operands, and the softmax it runs is over permuted scores (equivariant) — it never learns π. The TEE-side merge is a small plaintext correction, not a softmax-over-real-positions.

---

## Session-resident K/V API (substrate refactor — gated on the spikes)

The persistence the design needs is an **engine-owned session**, not the
per-call K/V views `offload_attention_permuted_cached` takes today. The
`GpuOffloadEngine` gains a session handle:

- `create_session(prefix_k, prefix_v, perm_kv, O_v) -> SessionId` —
  uploads the rotated + permuted + noised prefix once.
- `append(id, k_row, v_row)` — adds one decode token; the tail is small
  and may stay TEE-side (the online-merge path).
- `attend(id, q) -> partial(m, l, acc)` — the prefix's partial softmax
  state for the TEE-side merge.
- `refresh_block(id, perm_kv', noise)` — re-gather the canonical cache
  under fresh `perm_kv` + re-noise K (every block). `O_v` is **not**
  re-applied (session-fixed default; gate 3).
- `drop_session(id)`.

Residency is engine-owned; the cold tier is a **pluggable
`SpillProvider`** (Phase 1 = null / VRAM-only; Phase 2 = NVMe), so the
spill seam exists from day one and the NVMe tier slots in with no API
change. The TEE retains the canonical rotated cache (`V·O_v` in canonical
order) for the `refresh_block` re-gather — so there are two copies (TEE
canonical + GPU permuted), ~2× cache memory (≤ ~1 GB at 16k B=8 un-
replicated bf16, negligible on the 32 GB card).

This is the load-bearing trait change every engine impl must follow; it
lands only after gates 1–3 clear (it is the 2–3 week substrate refactor,
not part of the microbench).

## Fallback: TwinShield-Xue additive blinding

If the block-fresh-π perf gate fails (the ∝L re-permute tax doesn't
amortize within the security-permitted N), the fallback is the
additive softmax-blinding scheme of TwinShield-Xue (arXiv 2507.03278).
Instead of permutation-equivariance, it blinds the scores additively in
the exponent: `e^{X+R} = e^X · e^R`. The blinding `R` is regenerated
**fresh per call** and the TEE divides it back out, so — unlike a fixed
permutation — it carries no across-step accumulation clock. That lets
K/V persist on the GPU under a *fixed* operand cover while the
fresh-per-call `R` supplies the freshness that defeats HNM-class
recovery, **decoupling persistence from the security clock** that
block-fresh-π pays the re-permute tax to manage.

It is also the one mechanism that could perturb V's *geometry*
correctably. Under our permutation path, additive noise on V is
uncorrectable: `out = probs·V` contracts over the token axis with
softmax weights the TEE never sees, so `Σ_j e^{s_j−m} ε_j` cannot be
subtracted (the same reason the in-tree cover noises Q and K only, and
why feature-axis rotations — which pass through the contraction — are
the only correctable V-hardening on the permutation path, and those
preserve geometry). TwinShield builds the correction into the protocol,
so the additive blinding the permutation path cannot do becomes
available — at the cost of the `R`-rank correction.

### Structural difference for this design

| Axis | Block-fresh-π (primary) | TwinShield additive (fallback) |
|---|---|---|
| What enables persistence | fixed `perm_kv` across N steps | fixed operand cover **+ fresh-per-call `R`** |
| Freshness clock | **N is the HNM observation count** → σ-vs-N gate caps N ≈32–64 | `R` regenerated every step → **no fixed-cover accumulation clock** |
| Refresh / re-permute tax | full re-permute ∝ context, every N steps (the hybrid's one remaining cost) | **none** — `R` is small and per-step |
| Correction cost | ~free (perm un-apply + online merge) | **f(rank R)** — full-rank `R` ⇒ correction ≈ cost of the attention itself ⇒ no win |
| V-geometry hiding | no (orthogonal / permutation only — geometry-preserving) | **yes, if `R` reaches the value path** — the additive thing the permutation path cannot do |
| Maturity | Amulet + HNM, in-tree, fresh-per-call already validated | published 2025; **threat model predates HNM**; needs independent validation |
| Engineering | extends `permuted_attention_cached` | new port + a TEE-side correction pipeline |

The two schemes trade one open risk for a different one. Block-fresh-π
fails on **perf** ("N too small to amortize the ∝L re-permute") — cheaply
falsifiable with the microbench. TwinShield fails on **correction cost**
("R too high-rank to un-blind cheaply") — *not* cheaply falsifiable; it
needs the paper's threat model audited against ours and the `R`-rank
characterized (the 1B security spike). That asymmetry is why the
sequencing runs the cheap in-tree path first and reaches for the
structurally-cleaner-but-higher-unknown path only on a measured perf
failure.

## Open questions (the load-bearing gates)

The kernel / backend / session-handle-API choices below these are
comparatively mechanical; these three gates decide whether block-fresh-π
ships or we fall to TwinShield-Xue.

1. **Prefix re-permute cost (gate 1, perf).** **Per-step half: PASSED
   (2026-05-29).** The `gpu_resident_b8` microbench measures resident
   per-step at **0.30 / 0.373 / 0.465 ms** (n_kv 256 / 1024 / 2048, B=8)
   vs the in-TEE baseline 1.08 / 5.28 / 11.15 ms — **24× faster at n=2048,
   widening with context.** Decomposition confirmed: ~99.9% of the
   full-upload cost is upload+convert+sync (the term persistence deletes).
   See chronicle §10 / `bench-results/amulet-attn-resident-5090-2026-05-29.log`.
   **Re-permute half — conditional.** The microbench also bounds the
   re-permute *upload* directly: `no_mask − resident` = (convert+upload
   K/V) = **~488 ms @ n=2048** on the *current* pipeline (GQA-expanded,
   f32→f16 convert). Amortized over N=16 that is ~30 ms/step — it would
   **lose** to the 11 ms in-TEE baseline. So the re-permute half **only
   wins once the upload is optimized**: un-replicated storage (4× less
   data) + bf16-native K/V (no f32→f16 convert) → modeled ~5 ms (gather
   1.5 + upload 2 + re-noise ~1), ÷16 ≈ 0.3 ms/step. That optimization is
   part of the substrate refactor — **gate-1 is PASS on the per-step read,
   PASS-conditional on the re-permute optimization landing.** The boundary
   re-permute
   decomposes into: gather (`perm_kv`, memory-bound — ~1.5 ms @2048 /
   ~12 ms @16k), the `O_v`/`O_qk` rotation (compute-bound — ~86 ms @2048
   / ~0.7 s @16k *if refreshed*, at an assumed ~100 GFLOP/s CPU), and
   re-noise + bf16 upload (bandwidth-bound — a few ms / ~30 ms,
   un-replicated B=8). The **rotation is the swing term**, so the
   amortized re-permute is governed by the `O_v` cadence `M` (gate 3),
   *not* by `N`: at `O_v` session-fixed it collapses to gather+upload
   (~0.2 ms/step @2048, ~2 ms/step @16k after ÷N — negligible); at
   per-block `O_v` it eats ~half the in-TEE baseline. The fail-fast
   microbench profiles the actual split — its **first deliverable** is
   decomposing the 510 ms triage cost into convert / stage / DMA / sync,
   since the "fixed-overhead-bound" attribution is currently *inferred*,
   not measured. Async double-buffering is **not** a v1 lever — it hides
   transfer, not the CPU rotation that dominates when `O_v` refreshes.

2. **σ-vs-N thresholds (gate 2 — the `perm_kv` clock).** **Partial
   measurement done (2026-05-29, Rust)** — `gate2_perm_recovery_vs_sigma_and_n`
   in `crates/gelo-protocol/tests/permutation_attention.rs`, log
   `bench-results/gate2-perm-recovery-sigma-n-2026-05-29.log`. Two
   measured findings (ARROWMATCH cosine recovery, cleartext reference =
   worst case, random Q at d=128):
   - **σ-noise is not a usable lever.** Single-observation recovery is
     100% until σ≈1.2, but attention quality is destroyed by σ≈0.3
     (drift 0.08) — the quality ceiling sits **~20× below** where the
     attack even begins to fail. No quality-compatible σ defeats a
     reference-equipped cosine attack at production d=128.
   - **Persistence is strictly worse, via √N denoising (confirmed).** At
     σ=5, single-obs recovery 0.064 → **N=64 fixed-π observations recover
     fully (1.00)**. Fixed-π-across-N lets the attacker average out the
     noise. ⇒ **prefill-only is the *worst* case for `perm_kv`** (maximal
     accumulation); this validates the design decision that `perm_kv`
     **refreshes per block** (bounded N), and `O_v` alone is session-fixed.
   - **Implication:** the cover's security rests on the **no-clean-reference
     GELO mask invariant**, not σ-noise. The reference-*free* HNM attack
     (the real adversary, who lacks clean K) is **not yet measured** — it
     needs the Python HNM driver + real activations (see gate-status note).
   The remaining quantitative output (max N before reference-free recovery)
   comes from that driver.

3. **Covariance-alignment thresholds (gate 3 — the `O_v` clock).** Runs
   in parallel to gate 2 against the *same* attack suite, but targets the
   rotation rather than the permutation. Feed JADE / anchor_ica / JD the
   observed `V·O_v` cloud (noiseless) plus the model's
   activation-covariance prior; measure how many distinct token-values
   must be observed before `O_v` is recovered up to sign/axis flips at
   our shapes (d_head=128, production context lengths). Output: the max
   `O_v`-fixed observation budget → the refresh cadence `M`. Determines
   whether v1 ships `O_v` session-fixed (cheap — gate 1) or must refresh
   every `M` blocks, and whether the structured-orthogonal O(L·d)
   signed-permutation trick is needed to make a finite `M` affordable.

### Gate-measurement status + environment split

The gates run in two environments, and only part runs on the dGPU box:

- **Rust, on the dGPU box (done 2026-05-29):** the **quality ceiling**
  (`permutation_attention.rs` drift-vs-σ: σ=0.01 drift < 5e-2, tolerable;
  σ≥0.3 destroys output) and the **reference-equipped** perm-recovery
  attack + the √N-accumulation effect (`gate2_perm_recovery_vs_sigma_and_n`).
  These establish: σ is not the security lever, and persistence amplifies
  recovery via √N — so `perm_kv` must refresh per block.
- **Python AloePri harness, on the eval env / CI (NOT runnable on the
  dGPU box — no pip/ensurepip/apt/sudo; numpy/scipy/sklearn absent):**
  the **reference-free** attacks that produce the actual cadence numbers —
  HNM statistical permutation recovery (gate 2 proper → max N) and
  JADE / anchor_ica covariance-alignment for `O_v` (gate 3 → max T / M).
  Drivers exist (`evals/aloepri-attacks/attack_drivers/run_{jade,anchor_ica}.py`)
  but target the linear-mask channel; the attention-cover scenario
  (fixed `perm_kv`+noise on K across N; fixed `O_v` on the V cloud) is a
  new condition to add. **Gate 3 additionally requires real Qwen3-4B
  activation dumps** — random/isotropic activations make `O_v` recovery
  trivially fail (false pass), so step 0 is capturing real K/V at the
  production shape.

### First real-activation gate run (2026-05-29)

Pipeline now wired end-to-end: `attn_cover_capture.rs` dumps the real
Qwen3-4B adversary view (prefill-only cover: `perm_kv` + σ=0.01 on K +
`O_qk`/`O_v`, n_kv=545, layers 0/17/35) → `persistent_attn_gate.py`
(in the `gelo-attack` container) attacks it. Artefacts:
`evals/aloepri-attacks/{persistent_attn_gate.py, captures/gate_results.json}`.

| measurement | result | reading |
|---|---|---|
| baseline (direct cos-match `v_sent`→`v_clean`) | cos ≈ 0.07, perm ≈ chance | the cover defeats naive coordinate matching |
| **gate 3 (content / `O_v`): FastICA corr** | **0.249 vs 0.347 no-attack baseline** | **`O_v` HOLDS** — ICA recovers nothing above the coincidental high-d column-overlap floor; V coordinates stay hidden |
| **gate 2 (position): perm recovery from row-norm geometry** | **0.70 (L0) / 1.00 (L17,L35)** vs chance 0.0018 | the **geometry residual is real and severe** — row norms are `O_v`-invariant, so `perm_kv` leaks almost completely; `O_v` gives zero position protection |

**Honest caveats.** (1) The gate-2 perm-recovery is **reference-equipped**
(scores against the clean row-norms) — same worst-case framing as the
σ-sweep probe; reference-*free*, the leak is the value-norm *multiset* +
Gram (the documented accepted residual), not the position mapping itself.
But it definitively shows `O_v`+`perm` do not hide geometry. (2) Gate 3
used PCA-to-99%-var + capped FastICA (tol 1e-2); a 4th-order **JADE**
joint-diagonalisation is the stronger escalation to confirm "`O_v` holds"
— FastICA failing is strong but not final. (3) Single prompt, 3 layers,
σ=0.01 — needs breadth for a release verdict.

**Design implication.** Content-hiding (`O_v`) looks viable; the open
problem is the **geometry/position leak**, which no rotation fixes
(only additive noise destroys geometry — uncorrectable on this path →
the TwinShield escalation if geometry-hiding becomes mandatory). This
sharpens the v1 question: *is leaking value-norm geometry (the cloud's
shape, and — reference-equipped — the position order) acceptable?* If
yes, ship perm+`O_v`; if no, TwinShield.

## Acceptance gate (v1)

Layered — failing any tier reopens the TwinShield-Xue fallback:

1. **Fail-fast (microbench).** Resident per-step cost < the in-TEE
   baseline (11.35 ms @ n_kv=2048, B=8). Go/no-go *before* the substrate
   refactor.
2. **Perf (end-to-end).** ≥ 30% decode-wall reduction on top of R3 at
   n=2048, **and** a demonstrably larger reduction at a long-context cell
   (n ≥ 8k — the phased target where HBM's bandwidth edge dominates).
3. **Quality.** Greedy-token parity preserved at the σ chosen by gate 2
   (no extraction-quality regression from the √N-scaled noise). This
   couples acceptance to the gate-2 σ.
4. **Round-trips.** No growth in TEE↔GPU round-trips beyond the design's
   ≤ 1 per decode step, and no growth in mask-offload count (revival
   Step-5 invariants).

## Sequencing (committed forward plan)

**Strategy: build the prefill-only permute cover (the simplest, fastest,
*weakest* variant), attack it with a real reference-free HNM bench as
early as possible, and harden only if it fails — with TwinShield-Xue as
the parallel-de-risked fallback.** The ordering change vs a naive
"build-all-then-test" is to **gate the expensive kernel + decode wire-up
on the security result**, because the adversary view is just the cover
applied to real activations (Phase-1 output) — it does not need the
kernel or the integration.

### Done

0. **Triage + gate-1 microbench** — ✅ **2026-05-29.** `gpu_resident_b8`:
   resident read 0.465 ms @ n=2048 (24× under the 11.15 ms in-TEE baseline);
   ~99.9% of the full-upload cost is upload+convert+sync. `gpu_resident_append_b8`
   (per-step O(1) append, prefill-only): **0.662 ms @ n=2048 → 16.4×**,
   scaling 2.7× → 16.4× with context. **Optimistic case is worth it.**
   Conditional: the re-permute upload (`no_mask − resident` ≈ 488 ms
   current-pipeline) needs the un-replicated + bf16-native optimization
   (~5 ms modeled) to amortize — lands in Phase 2.
   Gate-2 σ-sweep probe (`gate2_perm_recovery_vs_sigma_and_n`): σ is not a
   lever, √N accumulation real ⇒ `perm_kv` refresh per block; but this is a
   *cleartext-reference* probe, **not** the realistic gate.

### The reordered critical path

```
Phase 1 (cover incl. O_v/O_qk, σ=0 parity)  + real-activation capture
   → HNM/ICA bench on the adversary view          ← cheap, highest-risk, runs FIRST
        ‖ parallel: Phase 2 substrate (cover-AGNOSTIC: session API incl.
        ‖           refresh_block, SpillProvider, un-replicated+bf16 upload)
        ‖ parallel: TwinShield 1B spike (R-rank / correction-cost viability)
   → SECURITY GATE result → THEN commit Phase 3 kernel + Phase 4 wire-up
   → Acceptance (4 tiers) → Flip default behind c5 AloePri condition
```

1. **Phase 1 — cover math + parity** (`gelo-protocol/attention.rs`) —
   ✅ **DONE 2026-05-29.** `O_qk` (Q,K feature rotation — cancels in
   scores) + `O_v` (V feature rotation — corrected by `O_vᵀ`) added to
   `permuted_attention_cached`, behind `PermAttnConfig::feature_rotation`
   (default false → production unchanged) with `sample_orthogonal`
   (Gram-Schmidt). Verified: `feature_rotation_sigma_zero_matches_plain`
   (bit-exact at σ=0, d=32 & 128) and `feature_rotation_preserves_scores`
   (ON ≡ OFF). The lever gate-2 showed σ-noise can't provide. *Cover
   sampled fresh per call here; the session fixes it at prefill.*
2. **Real-activation capture** (Rust, this box): dump real Qwen3-4B
   attention Q/K/V at the production decode shape + apply the cover →
   the adversary view the HNM/ICA bench consumes.
3. **Security bench** (Python eval env — *not* this box): reference-free
   HNM (gate 2 → max N) + JADE/anchor_ica covariance-alignment
   (gate 3 → max T / `O_v` cadence M) against the captured adversary
   view. **Prefill-only first; if recovery is bad, flip on per-block /
   per-N-block `perm_kv` refresh and re-attack.** This is the gate on
   Phases 4+.
4. **Phase 2 — session substrate (cover-agnostic, parallel to 1–3)**:
   `GpuOffloadEngine` session API (`create/append/attend→partial(m,l,acc)/
   refresh_block/drop`), `SpillProvider` (null), **un-replicated GQA +
   bf16-native upload** (the gate-1 re-permute optimization).
   **Build `refresh_block` into the API now** even though prefill-only
   doesn't call it — makes the "add per-block refresh" step nearly free.
5. **Phase 3 — production kernel** (gated on the security result):
   cubecl partial-stats / GQA-aware FlashAttention-D (cubecl-custom vs
   upstream-burn; both CUDA-capable).
6. **Phase 4 — decode wire-up** (gated): thread the session through
   `decoder_block_cached_batched`; prefill builds session, decode
   appends+attends+merges; in-TEE fallback behind a config flag.
7. **Acceptance + flip** — the 4-tier gate, then default-on behind the
   c5 AloePri condition (mirrors R3).
8. **Fast-follows** — prefill FlashAttention-D; NVMe `SpillProvider`.

### TwinShield reuse (what a pivot costs)

If the cover fails HNM at every cadence, fall back to TwinShield-Xue.
Most of Phases 2–4 is cover-agnostic and survives the pivot:

| Component | Permutation path | TwinShield | Reused on pivot |
|---|---|---|---|
| Session substrate (API, resident K/V, SpillProvider, un-replicated+bf16) | ✓ | ✓ (also persists K/V) | **100%** |
| Decode wire-up (prefill builds / decode appends+attends) | ✓ | ✓ | **100%** |
| Kernel skeleton (resident reads, GQA broadcast, matmuls, partial-stats/merge) | ✓ | ✓ | **mostly** |
| Cover math (perm + σ + `O_v`, πᵀ/`O_vᵀ` recovery) | ✓ | ✗ — `e^(X+R)` blinding + correction | **no** |
| Kernel softmax stage | standard over permuted scores | blinded-exponent + correction | **differs** |
| Block-refresh machinery | needed (√N) | not needed (fresh per-call R) | TwinShield *removes* |

⇒ **~60–70% of Phases 2–4 survives a pivot.** Keep the session API
**cover-agnostic** (cover is a swappable strategy) so the cost of a pivot
is the cover layer + the softmax stage, not the substrate. De-risk
TwinShield's R-rank in parallel (the 1B spike) so the fallback is
*known-viable* before it's needed.

## References

- [`2026-05-22-dgpu-attention-revival.md`](../handoffs/2026-05-22-dgpu-attention-revival.md) — the Item 1/2/3 design this concretizes; σ-vs-N table, 1A vs 1B trade
- [`2026-05-29-dgpu-attention-offload.md`](../handoffs/2026-05-29-dgpu-attention-offload.md) — dGPU bring-up handoff; the §2 headline that set up this task
- [`gelo-llm-perf-roadmap.md`](gelo-llm-perf-roadmap.md) §4.C.2 — the EV/engineering table for these levers
- `docs/dev/logs/gelo-llm-perf-chronicle_dgpu.md` §8 — the per-call-readback correction (the backend-invariant bottleneck)
- `bench-results/amulet-attn-triage-5090-2026-05-29.log` — the triage that gates viability on persistent K/V
- `crates/gelo-protocol/src/attention.rs::permuted_attention_cached` — the existing fresh-per-call cover this extends
- `crates/gelo-protocol/src/substrate.rs::offload_attention_permuted_cached` — the trait seam
- `crates/gelo-gpu-wgpu/src/lib.rs::fused_attention_batched` — the current re-upload-every-call engine path
- Amulet softmax-permutation equivariance (arXiv 2512.07495); Hidden No More (arXiv 2505.18332); TwinShield-Xue additive blinding (arXiv 2507.03278)

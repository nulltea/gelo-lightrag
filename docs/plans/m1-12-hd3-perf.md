# M1.12 — HD₃ FWHT perf (the lever bucket 3a missed)

> **Parent context:**
> - Handoff: [`2026-05-22-perf-bucket-roadmap-r3-default.md`](../handoffs/2026-05-22-perf-bucket-roadmap-r3-default.md) §3 — perf-bucket-3 spec. Originally framed around Haar mask GEMM; this plan corrects scope to HD₃ since production switched to `MaskKind::Auto` (→ HD₃ at pow2-friendly shapes) on 2026-05-21.
> - Plan: [`m1-12-bf16-activation-pipeline.md`](m1-12-bf16-activation-pipeline.md) — bucket 3a (bf16 Haar GEMM) + 3b (bf16-native activations). Ships independently of this plan but is **structurally inert for current production** because Auto never dispatches to Haar. This plan replaces 3a's "20.6 % prefill wall reduction" projection with measurements against the actual HD₃ code path that production runs.
> - Plan: [`m1-12-blis-thread-dispatch.md`](m1-12-blis-thread-dispatch.md) — parallel follow-up for the BLIS GEMM path. Orthogonal to HD₃ (different transform); both can ship in parallel.
> - Code: `crates/gelo-protocol/src/hd3.rs` — current radix-8 SIMD FWHT with rayon-over-row-chunks parallelism. ~1k LOC.
>
> **Status:** plan, post-discovery-correction. No engineering committed.
> **Author date:** 2026-05-22.

---

## 0. Why this plan exists (the bucket 3a oversight)

Bucket 3a (`m1-12-bf16-activation-pipeline.md` §3) added a bf16 GEMM
path to the Haar mask family on the assumption that Haar was
production's mask family. The 2026-05-21 default switch to
`MaskKind::Auto` (per `hd3_mask_landed`, `qwen3_4b_perf_2026_05_20`,
the `MaskKind::Auto (HD₃ + DCT-IV)` commit comment at `sim.rs:386`)
moved production to HD₃ at pow2-friendly shapes and DCT-IV
elsewhere. Neither uses BLIS GEMM. Bucket 3a's win is **structurally
inert** for current production.

What IS the production critical path:

| Stage | Today's code | Bottleneck cost |
|---|---|---|
| `gelo:mask_apply:hd3` / `:dct4` | `Hd3Mask::apply_in_place` / `Dct4Mask::apply_in_place` | 3 × FWHT + 3 × ±1 sign-flip + 1 × scale per call |
| `gelo:mask_unapply:hd3` / `:dct4` | mirror — 3 × FWHT + 3 × sign-flip + 1 × scale | symmetric to apply |

The per-forward share of these buckets at Qwen3-4B B=8 prefill is
the ~39 % (`mask_apply` 14.9 % + `mask_unapply` 24.5 %) the M1.12
roadmap measured — but **the actual underlying work is HD₃ FWHT**,
not BLIS GEMM. The bf16 GEMM path bucket 3a optimised never fires
on this critical path.

This plan targets the actual HD₃ FWHT cost.

---

## 1. Acceptance gates (provisional — pending §3 measurement)

The 39 % aggregate mask bucket from the M1.12 roadmap is the
opportunity ceiling. Target:

- **`gelo:mask_apply:hd3` + `gelo:mask_unapply:hd3` wall** at Qwen3-4B B=8 n=2048 drops **≥ 30 %** (best case ~12 % prefill wall reduction; conservative ~7 % at the lower bound below)
- **No regression** on `mask_apply:dct4` / `mask_unapply:dct4` (the non-pow2 fallback path that runs on the same `Hd3Mask` machinery's DCT-IV cousin)
- **No regression** on decode m=1 FWHT cost (shape-adaptive overlay shield at k=15 makes stacked_n=16 → HD₃ pow2; these are small kernels where parallelism overhead can dominate)
- **Greedy generation token parity** preserved on `qwen3_generation_e2e` + v7 extraction bench, to bf16-floor if 4b/4c land (≤ 1e-3 abs per element at the bucket boundary)

Refined gates land after §3 measurement settles which sub-bucket
(4a / 4b / 4c) carries the headline.

---

## 2. Threat model — what changes

**Sub-buckets 4a (threading) and 4b (bf16 boundary):** no change.
Sign vectors and inv_norm scale are unchanged; FWHT mathematics
unchanged; mask material (`d1, d2, d3, inv_norm`) stays TEE-only.
Adversary observation (the masked operand sent to GPU) is
bit-identical to today within numerical floor.

**Sub-bucket 4c (bf16-native FWHT):** the buffer mid-FWHT
materialises in bf16 (between butterfly stages). Same security
posture as bucket 3a §2: bf16 quantisation noise on adversary
observations is **strictly noisier than f32**, so existing
AloePri c1-c5 attack bounds carry over without empirical
re-validation. The mask material itself (sign vectors `d1, d2,
d3`) stays in TEE; only the working buffer is bf16.

No new AloePri gate required for any sub-bucket (math-only
argument per the bucket-3 plan §5 precedent).

---

## 3. Phase 0 spike — measure where HD₃'s time actually goes (½ day)

Before designing the fix we need a per-stage breakdown of the
current HD₃ implementation. Two structural questions to answer:

### 3.1 Is HD₃'s rayon parallelism saturating?

`fwht_rows_inplace` (`hd3.rs:290`) uses `slice.par_chunks_mut(chunk_size)`
with `chunk_size = 8 * h * d` at the radix-8 path and `2 * h * d`
at the radix-2 tail. The chunk count is `n / (chunk_factor · h)`:

| Stage `h` | radix-8 chunks at n=2048 | rayon parallelism at 16 cores |
|---:|---:|---|
| 1   | 256 | saturates 16 cores easily |
| 8   | 32  | saturates |
| 64  | 4   | 4-way parallel (4 cores idle on a 16-core box) |
| 512 | 1   | **serial — no rayon parallelism** |
| 1024 (radix-2 tail) | 1 | **serial — no rayon parallelism** |

**Hypothesis:** late stages (h ≥ 256) serialise the FWHT despite
the multi-core machine. If late stages dominate FWHT wall, threading
gains stop at ~½ the available parallelism.

**Spike:** add `profile::time` brackets around each FWHT stage at
`hd3.rs:310-355` and measure per-stage time at Qwen3-4B prefill
shape (n=2048 or 2056 padded → 2048 pow2, d=2560). Look for the
shape of the per-stage curve:

- Flat curve → memory-bandwidth bound, threading already saturated
- Rising late-stage curve → parallelism collapses at large h
- Falling late-stage curve → radix-8 fusion working well

### 3.2 Is HD₃ compute-bound or memory-bandwidth-bound?

Per-stage estimate (n=2048, d=2560):

| Resource | Value |
|---:|---:|
| Per-FWHT data read | n · d × 4 bytes = 20 MB |
| Per-FWHT memory passes | log₂ n / log₂ radix = 11 / 3 ≈ 4 passes (radix-8) |
| Per-FWHT memory traffic | 4 × 40 MB ≈ 160 MB |
| Memory bandwidth (DDR5 effective) | ~40 GB/s realised |
| Theoretical FWHT time | 160 MB / 40 GB/s = **4 ms** |
| Compute (n · d · log n add-sub ops) | 5M × 11 = 55M ops |
| AVX-512 peak (16 lanes × 4 GHz × 2 add/cycle) | 128 GFLOPS |
| Theoretical compute-only time | 55M / 128 GFLOPS = **0.4 ms** |

**Memory-bound by ~10×.** Implication: bf16 storage (halves memory
traffic) could give up to **2×** speedup at late stages; threading
past memory bandwidth saturates near 4-8 cores.

**Spike:** measure actual FWHT wall vs theoretical at 1, 4, 8, 16
rayon thread counts. If actual wall hits the memory-bandwidth floor
at threads=4-8, threads=16 buys nothing and 4a is bounded; if
actual wall scales linearly past threads=8, parallelism still
buys.

### 3.3 Production share decomposition

Run M1.12 microbench with per-family profile breakdown:

```bash
GELO_BENCH_VARIANT=4b GELO_BENCH_B=8 GELO_BENCH_N=2048 \
GELO_BENCH_MAX_TOKENS=64 RUST_LOG=gelo_protocol=debug \
  cargo test -p gelo-gpu-wgpu --release \
  --test qwen3_m1_12_r1_q1_microbench -- --ignored --nocapture \
  m1_12_per_op_breakdown_prefill_decode
```

Expect to see `gelo:mask_apply:hd3` vs `gelo:mask_apply:dct4`
split. If HD₃ is > 80 % of the mask bucket (likely at pow2-
aligned prefill), this plan's prefill scope is just HD₃; if
DCT-IV is > 20 %, §6 covers it explicitly.

### 3.4 Decision tree out of Phase 0

| Spike result | Engineering direction |
|---|---|
| Late-stage parallelism collapses + memory-bandwidth-bound | **4a (threading tune)** is small; **4b (bf16 boundary)** is the main lever — halves memory traffic |
| Late-stage parallelism collapses + compute-bound | **4a** primary — restructure parallelism axis |
| Threading already saturates DRAM at threads=4 | **4b** is the only available lever |
| Compute-bound at all stages | **4c (bf16 butterflies)** — only if compute headroom remains; needs careful AVX-512_BF16 design |
| Production split shows DCT-IV ≥ 20 % | **§6 explicit DCT-IV variant of 4b** ships alongside HD₃ |

---

## 4. Sub-buckets

### 4a — HD₃ FWHT threading strategy tune (~1 week)

Reshape the rayon parallelism axis at late stages so the FWHT
doesn't serialise when `h` grows. Two complementary moves:

**Move 1 — column-axis parallelism at late stages.** Today the
chunking is on the row axis: when `h` is large, you have few
butterfly groups (` n / (8h)`) and they're large. Switch to
parallelising **across the d (column) axis** when row-chunk count
drops below ~16:

```rust
if rayon_row_chunks(n, h, radix) < num_cpus {
    // Late stage — chunk d axis instead.
    par_chunks_d_axis(slice, chunk_size_d, h, d);
} else {
    par_chunks_mut(slice, chunk_size_n);
}
```

Each butterfly row is `d` elements; slicing `d` into chunks of
~`d / num_cpus` lets rayon parallelise the butterfly's per-row
work. SIMD vectorization within each chunk stays intact.

**Move 2 — recursive `rayon::join` inside large chunks.** At the
single-chunk late stage, recursively split the chunk via
`rayon::join((|| left_half, || right_half))` until reaching a
size that fits a thread's L2 cache (~256 KB ≈ 64 K floats).
Standard divide-and-conquer.

**Engineering surface:** ~50 LOC in `hd3.rs`. Touches only the
`fwht_rows_inplace` dispatch logic. Existing butterfly kernels
unchanged. SIMD paths unchanged.

**Risk:** column-axis parallelism has worse cache behaviour
(each butterfly touches all rows, splitting across columns means
each thread reloads the row pair). Spike 3.2's memory-bandwidth
finding determines whether the column-axis approach is feasible
or just shifts the bottleneck.

### 4b — bf16 storage at the buffer boundary (~3-4 days)

The work mid-FWHT is a sequence of `(row[i] ± row[i+h])` butterflies.
The intermediate values stay in f32 today. The **buffer the
caller hands in** is `Array2<f32>` today; with 3b's bf16-native
activations, it'll be `Array2<bf16>`. Two patterns:

**Pattern 1 — bf16 buffer, f32 FWHT internals.** The buffer
arrives as bf16, gets one-time-converted to a f32 working buffer
at the start of `apply_in_place`, FWHT runs at f32, one-time
narrows back to bf16 at end. Saves the **upload memory traffic**
of the buffer (half the bytes from caller) without changing the
FWHT internals.

```rust
pub fn apply_in_place_bf16(&self, buf: &mut Array2<bf16>) {
    // One-time bf16 → f32 expand. SIMD-vectorisable.
    let mut f32_buf = bf16_buf_to_f32(buf);
    self.apply_in_place(&mut f32_buf);  // existing f32 FWHT path
    f32_to_bf16_buf(&f32_buf, buf);
}
```

**Cost of the one-time conversions:** at n=2048, d=2560: 5M
conversions per direction × 2 (apply + unapply) = 10M conversions
per offload. At AVX-512 throughput (~16 conversions per cycle ≈
16 GS/s peak, ~5 GS/s realistic), that's ~2 ms per offload of
conversion overhead. Across 360 offloads = ~700 ms per forward.

Savings: the bf16 buffer at the caller boundary halves the
memcpy bandwidth on the buffer (12 MB → 6 MB per offload boundary
times some number of crossings). Net win depends on what dominates
between conversion overhead and saved memcpy bandwidth.

**Composes with 3b's activation pipeline.** Once activations are
bf16-native, the bf16 buffer is the natural input to the mask
boundary. Pattern 1 makes HD₃ a friendly consumer of 3b without
changing FWHT internals.

### 4c — bf16-native HD₃ FWHT internals (~2-3 weeks; research)

Replace the f32 add-sub butterfly with a bf16 add-sub butterfly.
AVX-512_BF16 doesn't have a native bf16 add/sub instruction
(only `vdpbf16ps` which is fma into f32). So bf16 add-sub means:

- Load bf16 pair → zero-extend + shift left 16 bits → f32 pair
  in register
- f32 add / f32 sub in register
- Truncate f32 → bf16 (via `vcvtneps2bf16` or shift right 16
  bits with rounding)
- Store bf16

Throughput per butterfly element:
- f32 baseline: 1 load + 1 add + 1 store = 3 µops, 16 lanes
- bf16 variant: 1 load + 1 expand + 1 add + 1 narrow + 1 store
  = 5 µops, 32 lanes (zmm processes 32 bf16)
- Per-lane throughput: bf16 variant uses ~5/3 = 1.67× more µops
  for 2× the data per zmm → **theoretical ~1.2× speedup per
  ZMM-cycle** at compute-bound; **at memory-bound the bf16 variant
  wins ~2× from the halved memory traffic**.

**Net win depends on whether HD₃ is memory- or compute-bound** —
which is what spike §3.2 establishes.

**Engineering surface:** large. Every butterfly kernel
(`butterfly_pair`, `butterfly_oct`, AVX-512 and AVX-2 variants)
gets a bf16 cousin. Sign vectors (`d1, d2, d3`) stay f32
(they're tiny — 8 KB at n=2048). The scale step needs a bf16
variant too.

**Sign-flip step under bf16:** `apply_diag_inplace` multiplies
each row by ±1.0. At bf16: signs are ±1.0 stored as bf16 bits.
The "multiplication" reduces to xor of the sign bit — same as
f32. Trivial port.

**Mask sample-side:** keep the f32 source-of-truth `d1, d2, d3`
as f32 (Mezzadri-correctness analogue — actually HD₃ doesn't
have Mezzadri but the sign vectors are sampled from an unbiased
±1 distribution that doesn't need post-correction). Downcast
lazy or eager — same call pattern as bucket 3a's bf16 cache.

**Risk:** custom bf16 SIMD kernel maintenance. The existing
f32 FWHT kernels in `hd3.rs` are ~600 LOC of intricately-tuned
intrinsics; the bf16 cousins would add another ~600 LOC.
Probably 2-3 weeks for "works, well-tested"; longer for
"matches f32 performance per-op" if compute-bound.

**Conditional:** **only pursue if spike §3.2 shows compute
headroom at late stages**. If memory-bound throughout, 4b's
buffer-boundary conversion captures the same ~2× memory
bandwidth saving without the kernel maintenance.

---

## 5. DCT-IV is the same story (mostly)

`Dct4Mask` (`dct4.rs`, 526 LOC) follows the same `D₃·C·D₂·C·D₁·C`
cascade. The "C" is rustdct's DCT-IV plan (probably calling
into a tuned dct4f kernel) — not directly comparable to HD₃'s
hand-rolled FWHT. Plan:

- §4a (threading): rustdct may already parallelise internally;
  measure first
- §4b (bf16 buffer boundary): identical pattern; same engineering
- §4c (bf16-native DCT-IV): rustdct is a third-party crate;
  patching it for bf16 would require upstreaming or forking;
  much higher engineering cost than HD₃-internal. **Defer
  unless DCT-IV shows ≥ 15 % production wall share.**

If spike §3.3 shows DCT-IV is a small share (likely at our
shapes since Auto picks HD₃ at pow2 alignment), DCT-IV stays
out of scope.

---

## 6. Phase plan

| Phase | Effort | Deliverable |
|---|---|---|
| **Phase 0** | ½ day | spikes 3.1, 3.2, 3.3 ran; decision tree §3.4 picks 4a / 4b / 4c |
| **Phase A** (4a) | ~1 week | rayon axis tune at late stages; benchmark vs threads=1..16 |
| **Phase B** (4b) | ~3-4 days | bf16 buffer boundary; composes with 3b; benchmark vs f32 buffer |
| **Phase C** (4c — conditional) | ~2-3 weeks | bf16-native FWHT kernel; AloePri math-only argument; benchmark vs 4b |
| **Integration** | ~1-2 days | re-run M1.12 microbench; assert §1 acceptance gates clear |

Total: **2-4 weeks** depending on whether 4c is pursued.

Phase A and Phase B can ship in parallel; they touch different
parts of `hd3.rs`. Phase C is conditional on §3.2's memory-bound
finding.

---

## 7. Strategic context — where this sits vs other levers

Comparing the prefill-wall-reduction levers we know about today:

| Lever | Prefill wall reduction | Engineering | Status |
|---|---:|---:|---|
| Bucket 3a (Haar bf16 GEMM) | **0 %** at current production | shipped (4 commits) | inert because production uses Auto, not Haar |
| Per-shape BLIS thread dispatch (3a follow-up) | ~33 % **if production hits Haar** | ~3 days post-spikes | also inert at current production — Auto bypasses BLIS |
| **HD₃ FWHT perf (this plan)** | **conservative ~7-12 %**, best case ~30 % of `mask_apply:hd3` + `mask_unapply:hd3` bucket | ~2-4 weeks | the lever that actually targets production code |
| Bucket 3b (bf16-native activations) | ~5 % marginal on top | ~2-3 weeks | composes with this plan's 4b |
| Bucket 4 / R4 (async pipelining) | ~15 % at iGPU best case | ~5-8 days, blocked on Q#2 RADV spike | independent |

**This plan is the highest-EV next move for production prefill
performance.** The bucket-3a "20.6 % wall reduction" projection
that motivated the bf16 work was correct for the Haar code path
but doesn't apply to the Auto code path production runs. The
~7-12 % conservative target here is the realistic next gain.

---

## 8. Out of scope

- **DCT-IV custom bf16 kernel** — defer per §5 unless DCT-IV
  share is ≥ 15 %.
- **Switching production back to Haar to use bucket 3a** —
  would regress the HD₃ wins. Per `hd3_mask_landed` HD₃ at pow2
  alignment is structurally faster than Haar BLIS-mt-1; rolling
  back the Auto default loses more than 3a gains.
- **HD₃ multi-thread mask sampling** — Hd3Mask::fresh is O(n)
  (just `3·n` ±1 random bits). Not the bottleneck.
- **HD₃ structured-block FWHT** — research item (block-diagonal
  HD₃ trades security for less memory traffic). Filed in
  future-rnd; not v1.

---

## 9. Open questions

1. **DCT-IV thread-parallelism status under rustdct** — rustdct's
   internal threading is unmeasured. Phase 0 spike includes a
   per-stage breakdown of DCT-IV alongside HD₃.
2. **Per-call rayon overhead at late stages** — column-axis
   parallelism adds rayon-spawn cost per stage. If the per-spawn
   cost is comparable to the per-stage compute, 4a is a wash.
   Measure during Phase A.
3. **AVX-512_BF16 throughput on Zen 5** — published numbers
   suggest 2 µops/cycle for `vdpbf16ps` on a single SKU. Our
   actual Strix Halo MFMA-or-similar bf16 throughput needs
   confirmation; AGNER tables for Zen 5 are still partial as of
   2026-05-22.
4. **Compose with bucket 3b's f32-internal kernel widening** —
   if 3b's RMSNorm / RoPE / softmax internally widen to f32, the
   activation tensor on the FWHT boundary is already f32-shaped
   even though "storage" is bf16. Verify the buffer dtype seen
   by `apply_in_place` after 3b lands.

---

## 10. References

- `crates/gelo-protocol/src/hd3.rs:104` — `Hd3Mask` struct
- `crates/gelo-protocol/src/hd3.rs:290` — `fwht_rows_inplace` (dispatch)
- `crates/gelo-protocol/src/hd3.rs:361, 384, 451` — butterfly kernels
- `crates/gelo-protocol/src/dct4.rs` — DCT-IV cousin
- `crates/gelo-protocol/src/mask.rs:21-101` — `MaskFamily` enum + profile categories
- `docs/plans/m1-12-bf16-activation-pipeline.md` §10 — measured bf16 vs f32 BLIS at the (inert-for-production) Haar shape
- `docs/plans/m1-12-blis-thread-dispatch.md` — parallel follow-up for the BLIS path
- `docs/handoffs/2026-05-22-perf-bucket-roadmap-r3-default.md` §3 — original bucket-3 spec
- `~/.claude/projects/.../memory/hd3_mask_landed.md` — 2026-05-19 HD₃ landing with measured pow2-vs-non-pow2 perf characteristics
- `~/.claude/projects/.../memory/qwen3_4b_perf_2026_05_20.md` — Qwen3-4B HD₃ perf numbers at production scale

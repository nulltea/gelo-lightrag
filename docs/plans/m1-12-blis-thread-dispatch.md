# M1.12 — Per-shape BLIS thread dispatch (3a follow-up)

> **Parent context:**
> - Plan: [`m1-12-bf16-activation-pipeline.md`](m1-12-bf16-activation-pipeline.md) §10 — measured threads-scaling at the prefill mask GEMM shape that motivates this follow-up. The threads-scaling lever alone is ~1.6× the size of the bf16 lever at long-n prefill, but unlocking it requires moving away from the current process-global thread setting.
> - Memory: `blis_default_on_and_layer_skip_regression.md` (2026-05-19) — original "threads=1 protects small shapes" rationale.
> - Memory: `tee_direct_m1_gemv_slowness.md` (2026-05-19) — analogous "small-shape GEMV needs different dispatch" finding for the in-TEE direct path.
>
> **Status:** plan / spike sketch. No engineering committed.
> **Author date:** 2026-05-22.

---

## 0. The opportunity

Measured at Qwen3-4B prefill mask shape (`s = 2056, d = 2560`):

| `GELO_BLIS_THREADS` | f32 BLIS GEMM | Speedup vs threads=1 |
|---:|---:|---:|
| 1  | 81.93 ms | 1.00× |
| 16 |  8.62 ms | **9.50×** |

Per `m1-12-bf16-activation-pipeline.md` §10.3 the mask buckets are
**39 % of B=8 prefill wall** at threads=1 (~72 s of 192 s). Lifting
the mask GEMM to threads=16 alone (no bf16 protocol change) saves
~64 s of mask wall → **−33.5 % prefill wall reduction**. Combined
with bucket 3a's bf16 lever: ~67 s saved → **−35.1 %**.

The current process-global `GELO_BLIS_THREADS=1` default leaves
this ~33 % wall reduction on the table because the same global
setting **regresses** three other workloads that share the binary.

## 1. Why this isn't already done

`mask::blis_init_single_thread` pins BLIS to a single thread on the
first GEMM call via a per-thread `OnceCell`, with `GELO_BLIS_THREADS`
as the override. This is correct for the three workloads that
regress at threads=N:

| Regressing workload | Why threads=N hurts |
|---|---|
| Embedder + rerank | Per-call GEMM is small AND outer rayon already parallelises across candidates. Multi-thread BLIS over-subscribes; each thread's barrier setup exceeds the actual GEMM. |
| Decode m=1 (`s ≈ 16`) | Per-call work is ~6 µs at f32 BLIS-mt-1 (measured). Multi-thread barrier setup is comparable. Negative scaling. |
| Batched prefill (M1.11 PerSequence, rayon-over-B) | Same over-subscription as embedder. Each rayon worker holds one per-sequence mask; multi-thread BLIS inside each worker would multiply contention by `B × N_threads`. |

The fourth workload — **single-stream long-n prefill** — is the
*only* place threads=N wins. Today's setting protects three at the
cost of one. We want all four optimised.

## 2. Detection signal — what tells us "use threads=N"?

Two properties identify the long-n-single-stream path:

1. **Large `m` in the GEMM**. `m ≥ 1024` is the empirical threshold
   above which BLIS-mt amortises its per-call barrier (see
   `TEE_BLIS_THRESHOLD_ROWS = 64` in `mask::tee_matmul` for the
   smaller-scale analogue — that threshold gates `matmul_blis` vs
   `ndarray::dot`, not the BLIS thread count).
2. **No outer rayon parallelism**. If we're inside a
   `rayon::iter::ParallelIterator::for_each` closure, the outer
   loop already owns the parallelism budget.

`rayon::current_thread_index()` returns `Some(_)` inside any rayon
worker, `None` at the top level. The combination
`s ≥ THRESHOLD && rayon::current_thread_index().is_none()` is the
signal.

## 3. Design options

### Option A — per-call `bli_thread_set_num_threads`

Change `sgemm_blis` / `matmul_blis` to set the thread count
**inside each call** based on `rayon::current_thread_index()` and
operand size, then reset to 1 after the GEMM. AOCL-BLIS already
supports thread-count changes at runtime; the OnceCell pin is just
our wiring choice.

```rust
fn sgemm_blis_smart(a, b, transpose_a) -> Array2<f32> {
    let target_threads = select_threads_for_shape(a.nrows(), b.ncols());
    // BLIS thread count is per-thread (thread-local state); changing
    // it here only affects this rayon worker (or main thread).
    set_blis_num_threads(target_threads);
    let out = sgemm_blis(a, b, transpose_a);
    // Reset to 1 so the next call (possibly small / inside rayon)
    // doesn't inherit the multi-thread setting.
    set_blis_num_threads(1);
    out
}

fn select_threads_for_shape(m: usize, n: usize) -> i64 {
    // Inside rayon: always 1.
    if rayon::current_thread_index().is_some() {
        return 1;
    }
    // Top-level: scale with operand size. Conservative threshold —
    // tune from a 2-3 point measurement sweep at intermediate s.
    if m >= 1024 { 16 } else if m >= 256 { 4 } else { 1 }
}
```

**Cost of `bli_thread_set_num_threads`**: needs measurement, but
the literature suggests µs-scale. At 360 mask GEMM calls per Qwen3
forward, 2 µs/call setup = ~720 µs total = negligible vs the
multi-second mask buckets.

**Pros:**
- Self-contained in `mask.rs`. No API change.
- Works for all 4 workloads without caller intervention.
- Easy to A/B test (revert to OnceCell-pin via env var).

**Cons:**
- Adds a per-call `bli_thread_set_num_threads` overhead (TBD).
- Implicit behaviour: which thread count fired depends on call
  site context. Slightly harder to reason about than today's
  "one setting per process".

### Option B — explicit per-call hint via executor mode

`InProcessTrustedExecutor` gains a state flag (`ParallelismMode::Outer` /
`Inner`) that the caller sets at top-of-forward. The forward path
sets `Outer` once; rayon parallel loops switch to `Inner` for the
duration of the closure.

```rust
enum ParallelismMode {
    Outer,  // single-stream, GEMM may parallelise → use threads=N
    Inner,  // inside outer rayon → GEMM stays threads=1
}

impl InProcessTrustedExecutor {
    pub fn with_parallelism(mut self, mode: ParallelismMode) -> Self { ... }
}
```

Mask code reads the flag at each call.

**Pros:** explicit, easy to reason about, no rayon-API dependency.
**Cons:** every parallel-loop call site needs `.with_parallelism(Inner)`
plumbing. Adds API surface. Caller-side discipline becomes load-
bearing.

### Option C — separate threads-aware GEMM entry point

Add `mask::matmul_blis_mt(a, b, threads)` and `matmul_blis_st(a, b)`
as explicit variants. The caller picks per shape and context.

**Pros:** explicit, zero hidden state.
**Cons:** every call site has to know which variant to call. More
shotgun edits. Easy to miss a site and regress.

## 4. Recommended path

**Option A** — per-call `bli_thread_set_num_threads` driven by
`rayon::current_thread_index()` + operand size. Smallest diff,
self-contained, handles all 4 workloads automatically.

Two spikes needed before committing engineering:

### Spike 1 — `bli_thread_set_num_threads` per-call cost (~½ day)

Microbench: 10 000 calls to `set_blis_num_threads(N)` and back. If
each call is < 5 µs, the overhead is irrelevant. If it's ms-scale,
need a different design (e.g., cache the current thread count and
no-op if unchanged).

### Spike 2 — measure all 4 workloads at the proposed dispatch (~1 day)

- Long-n prefill (single-stream Qwen3-4B): expect ~9.5× speedup vs
  threads=1 baseline (matches §0).
- Embedder + rerank: expect parity with current threads=1 path
  (rayon-detected → BLIS stays threads=1).
- Decode m=1: expect parity (small `m` → BLIS stays threads=1).
- Batched prefill (M1.11 PerSequence, B=8 rayon): expect parity
  (rayon-detected → BLIS stays threads=1).

If all 4 are parity-or-better, ship. If any regresses, retune the
threshold or fall back to a sub-shape-specific variant.

### Spike 3 — interaction with LPGEMM thread settings (½ day)

AOCL LPGEMM may have its own thread mechanism (possibly using the
same `bli_thread_set_num_threads` pool, possibly its own). Verify
that the bf16 path scales the same way as the f32 path at the
proposed dispatch. The §10.1 table shows scaling does occur, so
LPGEMM is at least responding to the BLIS setting — but the per-
call set/reset pattern needs to be confirmed compatible.

## 5. Engineering

Assuming spikes clear:

- `mask::blis_init_single_thread` → renamed `mask::blis_init_per_call_dispatch`
  or similar; OnceCell removed.
- `mask::sgemm_blis` + `mask::matmul_blis` get the per-call
  set-then-reset pattern wrapped via a `with_blis_threads(n, ||
  call)` helper.
- LPGEMM call sites in `aocl_lpgemm` get the same wrapper.
- New unit test that exercises the four workload patterns and
  asserts no regression vs the baselines.
- `GELO_BLIS_THREADS` env var **kept** as an override for
  debugging / measurement; defaults to "use dispatch" when unset.

Estimated effort post-spikes: **2-3 days** for the wire-up + tests.

## 6. Acceptance gates

Same shape as the bucket-3a §1 gates, applied per workload:

| Workload | Target |
|---|---|
| Single-stream long-n prefill (Qwen3-4B B=1 n=2048) | ≥ 30 % prefill wall reduction vs threads=1 baseline |
| Embedder | ≥ −5 % wall (no regression) at the BEIR fixture |
| Reranker | ≥ −5 % wall (no regression) at the comparative bench |
| Decode m=1 | ≥ −5 % wall (no regression) at the M1.12 microbench |
| Batched prefill (B=8) | ≥ 30 % prefill wall vs threads=1 baseline (the same gate as bucket 3a) |

## 7. Compose-with-3a interaction

Per `m1-12-bf16-activation-pipeline.md` §10.3, compound headline at
threads=16 + bf16 is 35.1 % prefill wall reduction vs the 20.6 %
from bf16-alone or 33.5 % from threads-alone. **The per-shape
thread dispatch lever is therefore a near-total replacement for
bucket 3a's prefill wall win** with about half the engineering
cost (~3 days post-spikes vs ~3-4 weeks for 3a+3b).

**This raises a strategic question** the next perf session should
decide: do we ship 3a as planned, OR pivot directly to per-shape
thread dispatch since it captures most of the same wall reduction
without the substrate-rework cost? Two considerations:

1. **3a's bf16 lever stays useful as a memory-bandwidth lever
   even at threads=16** — saves an extra 1.6 percentage points
   even after threads is maxed out.
2. **3b's broader bf16-native activation work is the load-bearing
   prerequisite for any future dGPU bucket-2 revival** (the
   bucket-2 abort post-mortem showed the f32→f16 conversion was
   the binding upload-pipeline cost). 3b doesn't depend on the
   threads decision.

Recommendation: **ship 3a (already mostly done; 4 commits this
session); spike + ship per-shape thread dispatch as a parallel
track; defer 3b decision until both 3a and threads-dispatch numbers
are integrated.**

## 8. Out of scope

- **Adaptive thread count per machine** — would react to runtime
  core-count detection rather than a hard-coded `16`. Useful but
  premature; cap at the threshold via `num_cpus::get()` or
  `std::thread::available_parallelism()` for portability instead.
- **NUMA-aware thread placement** — Strix Halo is single-socket;
  not relevant. Filed for future multi-socket deployment.
- **LPGEMM batched API** (`aocl_batch_gemm_bf16bf16f32of32`) — a
  separate optimisation that would let us submit multiple mask
  GEMMs together. Composes with per-shape thread dispatch but is
  independently filed.

## 9. References

- `m1-12-bf16-activation-pipeline.md` §10 — measured threads-
  scaling at the prefill mask shape
- `blis_default_on_and_layer_skip_regression.md` — 2026-05-19
  original threads=1 default decision
- `tee_direct_m1_gemv_slowness.md` — analogous shape-dispatch
  argument for the in-TEE direct-path GEMM
- `feedback_memory_efficiency_priority.md` — production-default
  bias toward conservative resource settings
- `crates/gelo-protocol/src/mask.rs:283-294` —
  `blis_init_single_thread` / `set_blis_num_threads` current impl
- `crates/gelo-protocol/benches/mask_bf16_lpgemm.rs` — bench used
  to measure §0 / §1 numbers (re-runnable)

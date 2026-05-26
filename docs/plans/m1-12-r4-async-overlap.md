# §4.D R4 — async compute pipelining

Status: **green-lit 2026-05-26**; implementation in progress.
Supersedes the 4-step sketch in `gelo-llm-perf-roadmap.md` §4.D.

## Why this exists

The Q#2 RADV-async spike (`ea29602`) measured 58 % CPU/GPU overlap
at production prefill shape (Qwen3-4B B=8 n=2048, d_out=2560 O
projection). The spike runs `engine.matmul` (sync API: upload +
matmul + download, 19 ms) concurrent with `Dct4Mask` apply+unapply
on a separate buffer (10 ms) and measures 23 ms total — 1.25× speedup
vs the 29 ms serial baseline. Strix Halo UMA shares DDR5 between
CPU and GPU at ~58 % efficiency under contention; well above the
"no meaningful overlap" floor, in the "partial overlap, R4 viable"
band.

The spike validates a *capability*, not a real-pipeline win. Real
forward.rs has strict data dependencies — `apply M_{i+1}` needs the
unapplied output of matmul M_i, transitively. With `per_forward_mask=true`
(paper-parity default per `paper_parity_default.md`), the mask is
shared across the whole forward and `mask.generate()` isn't there
to hide either. The honest projection is **1-3 % wall** unless the
plan extends beyond a naive async swap into a shield-hoist refactor
that overlaps the data-light shield row sampling with the cascade.

## Design decisions (locked in grilling session 2026-05-26)

### A. Pipeline shape — 1-deep + thread-split + engine bus-pipeline

At most one matmul in flight at the substrate level. Multi-deep
buys nothing because matmuls within and across layers are serially
dependent. Thread-split runs shield generation on a worker thread
while the main thread waits for matmul completion. Engine internally
pipelines submit-N+1 / kernel-N / download-N-1 across its own
command queue (transparent to substrate).

### B. Trait surface — opaque `MatmulToken` + default sync fallback

```rust
pub trait GpuOffloadEngine: Send {
    // existing methods...

    fn matmul_async(
        &self,
        handle: WeightHandle,
        input: ArrayView2<f32>,
    ) -> Result<MatmulToken> {
        // default: run sync, stash the result behind a token
        let out = self.matmul(handle, input)?;
        Ok(MatmulToken::sync_fallback(out))
    }

    fn read_result(&self, token: MatmulToken) -> Result<Array2<f32>> {
        // default: unwrap the stashed result
        token.into_sync_result()
    }
}
```

`MatmulToken` is an opaque struct whose internals are
`enum { SyncFallback(Array2<f32>), WgpuPending(slab::Key) }` (or
similar). `WgpuVulkanEngine` overrides with real async via
`burn_tensor::into_data_async`. `matmul_many_async` mirrors the
same shape (returning `Vec<MatmulToken>` or a single token
representing N matmuls — TBD at step 1).

Object-safe (no associated types). No tokio dependency.

### C. Op scope — all 4 mask-cascade sites; attention deferred

| Site | Substrate call | Sites per layer |
|---|---|---|
| QKV | `offload_qkv_async` | 1 |
| O | `offload_linear_async` | 1 |
| gate/up | `offload_linear_many_async` | 1 |
| down | `offload_linear_async` | 1 |

`offload_attention_permuted` stays synchronous in this R4 pass —
different shielding scheme (permutation, not mask cascade), no
overlap window to exploit.

### D. Path scope — prefill only (batched + single)

Wired into `decoder_block_batched` (forward.rs L958) and
`decoder_block` (L1177). Decode paths (`decoder_block_cached*`)
stay sync — decode mask bucket is ~4 % of decode wall per the
handoff, so R4 buys ~1-2 % there. Not worth the code surface.
If post-R4 perf work shrinks the other decode buckets to where
mask matters, re-evaluate.

### E. Handle lifecycle — block-and-discard via RAII

```rust
#[must_use]
pub struct OffloadHandle {
    mask: Arc<MaskFamily>,
    masked: Array2<f32>,
    n_data: usize,
    token: MatmulToken,
    // engine + scratch-return back-channel for Drop
}

impl Drop for OffloadHandle {
    fn drop(&mut self) {
        // block on engine.read_result, discard result, return scratch
    }
}
```

If a `?` propagates up before `wait_offload` is called, RAII
guarantees the scratch buffer is returned. Cost: panic mid-forward
will wait ~19 ms (iGPU) or ~2 ms (dGPU) for the pending matmul
before unwinding — acceptable since panic paths aren't perf-critical.

### F. Mask-cascade ordering — shield-hoist + engine bus-pipeline

Refactor `build_shielded_and_apply` (sim.rs:844-1031) into three
phases:

```rust
fn compute_sigma_and_mean_norm(&self, hidden: ArrayView2<f32>) -> f32;
fn sample_shield_async(&mut self, sigma: f32, shape: (usize, usize)) -> ShieldHandle;
fn apply_mask_after_shield(&mut self, hidden, shield: ShieldHandle, mask) -> Array2<f32>;
```

Once sigma is known for site N+1 (post-RMSNorm of the previous
layer's down output), the substrate kicks off shield row sampling
on a worker thread. The main thread is free to issue matmul-N
async and wait on its result; by the time the worker finishes
shield-N+1, the main thread is ready to do cascade-apply-N+1 +
matmul-N+1 issue.

Engine bus-pipeline is opaque to the substrate but documented in
`WgpuVulkanEngine`: internal command queue holds at most 3 stages
(submit pending, kernel running, download in progress).

### G. Profile attribution — cross-thread aggregator + split buckets

`profile.rs` is thread-local with `RefCell`. Step 0 of the
implementation extends it with a global registry: worker threads
call `profile::register_thread()` on entry, the main thread calls
`profile::aggregate_threads()` at snapshot time, and worker buckets
get merged into the main profile.

Bucket changes:
- `engine:matmul` → split into `engine:matmul_submit` (cheap) and
  `engine:matmul_wait` (the visible GPU wait).
- New bucket `gelo:shield_async` for shield work that ran on a
  worker thread.
- Existing buckets (`gelo:mask_apply:*`, `gelo:mask_unapply:*`,
  `engine:matmul_many`, etc.) keep their names so before/after
  charts stay comparable.

### H. verify_probes — dropped from async production path

`offload_linear_async` and siblings panic if
`self.verify_probes > 0`. Verify lives on the sync path during the
validation window; at cutover (step 6), verify migrates to a
standalone test-only scaffold under `crates/gelo-protocol/tests/`
(direct `engine.matmul` calls + `verify_offload` helper, no
substrate involvement).

This matches the "no dual paths" principle (memory
`feedback_no_dual_paths.md`): the production substrate has one
implementation; test scaffolding lives in tests.

### I. Default + cutover plan — env-var opt-in → validate → cutover

```
GELO_ASYNC_OFFLOAD=1  # enables async in forward.rs prefill paths
```

Default OFF during validation window. After step 4 (parity test +
microbench) passes AND step 5 (AloePri attack-suite re-run) clears,
step 6 commits the cutover: delete sync `offload_*`, remove the
env var, move verify to a standalone test scaffold.

The AloePri gate exists because the async refactor changes per-call
ordering, which may surface timing side-channel concerns (handoff
flagged `grill-with-docs` before flipping default-on behaviour).

## Implementation steps

| Step | What | Effort |
|---|---|---|
| 0 | **Profile prep**: extend `profile.rs` with cross-thread aggregator; split `engine:matmul` → submit + wait; add `gelo:shield_async`. | 1 day |
| 1 | **Engine async API**: `matmul_async` + `read_result` + opaque `MatmulToken` on `GpuOffloadEngine` (default sync fallback). `WgpuVulkanEngine` override with `into_data_async` + internal 3-stage bus pipeline. `matmul_many_async` parallel. | 2 days |
| 2 | **Substrate async API**: `offload_linear_async` / `offload_qkv_async` / `offload_linear_many_async`. `OffloadHandle` + RAII Drop. Shield-hoist refactor of `build_shielded_and_apply`. Panic-on-verify_probes contract. | 3 days |
| 3 | **Forward.rs wiring**: `decoder_block_batched` + `decoder_block` use async substrate API. Layer-boundary stitching for QKV → attention → O → gate/up → down with shield N+1 on worker thread. | 2 days |
| 4 | **Parity + bench (STOP BEFORE THIS)**: `qwen3_generation_e2e` parity + production-shape microbench. Per-shape sweep (QKV, O, gate-up, down). Variance sweep §3.1 #2. | 1 day |
| 5 | **AloePri validation (out-of-band)**: attack suite re-run with async on. User coordinates GPU contention per `feedback_ask_before_running_attacks`. | — |
| 6 | **Cutover**: delete sync `offload_*`; relocate verify to test scaffold; remove `GELO_ASYNC_OFFLOAD` env var. | 1 day |

**Total: ~10 days engineering** + out-of-band validation.

## Open risks

1. **Spike-vs-real-pipeline gap.** The 12 % headline rests on the
   spike, but strict data dependencies in forward.rs cap real
   savings. Realistic projection is **1-3 % wall** unless shield-hoist
   delivers as designed (then potentially 5-8 %). Step 4 must
   compare against the lower-bound expectation; below 2 % savings
   means shield-hoist isn't doing real work and we reassess scope
   before cutover.

2. **Per-shape overlap variance** (handoff open risk #1). QKV, O,
   gate-up, down have different `d_out` and matmul times. Bench must
   sweep per-shape, not just aggregate. Some shapes may *regress*
   under the bus-pipeline if launch overhead exceeds the saving.

3. **AloePri timing side-channel** (handoff open risk #2). Per-call
   ordering changes when shield runs on a worker thread. The full
   AloePri attack suite (especially ISA-AttnScore and any timing-aware
   variants) must re-clear before cutover. If a side-channel surfaces,
   either revert to sync or design a constant-time shield-issue
   protocol.

4. **Engine bus-pipeline failure modes.** Multiple matmuls in flight
   inside the engine mean an error in matmul N may surface while
   the substrate is waiting on matmul N+1. Engine must serialize
   error reporting per token; substrate's `read_result` for the
   wrong token must not race against an error from another.

5. **Drop semantics during panic unwind.** RAII Drop blocks on
   pending matmul (~19 ms iGPU). If a panic occurs inside the worker
   thread's shield work, the main thread's pending matmul must
   still complete safely. Test: induce panic during shield generation,
   verify no leaks / dangling tokens.

## Cross-references

- Spike: `crates/gelo-gpu-wgpu/tests/q2_radv_async_spike.rs`,
  `bench-results/q2-radv-async-spike-2026-05-26_14-14-23.log`
- Roadmap: `docs/plans/gelo-llm-perf-roadmap.md` §4.D
- Handoff: `docs/handoffs/2026-05-26-r4-greenlight-bf16-aborted.md`
- Memory: `feedback_no_dual_paths.md`,
  `feedback_never_mask_on_untrusted_gpu.md`,
  `paper_parity_default.md`,
  `shield_simd_gaussian_landed.md`,
  `feedback_ask_before_running_attacks.md`
- Code touchpoints:
  - `crates/gelo-protocol/src/substrate.rs` — trait
  - `crates/gelo-protocol/src/sim.rs:844-1056` — `build_shielded_and_apply` (shield-hoist target)
  - `crates/gelo-protocol/src/sim.rs:1801-1849` — `offload_linear` (async sibling pattern)
  - `crates/gelo-protocol/src/profile.rs` — thread-local profiler (step 0 target)
  - `crates/gelo-gpu-wgpu/src/lib.rs:337` — current sync `matmul` impl
  - `crates/gelo-embedder/src/decoder/forward.rs:958,1177` — prefill paths to wire

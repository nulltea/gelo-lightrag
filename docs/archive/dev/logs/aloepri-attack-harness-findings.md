---
type: dev-log
status: stale
created: 2026-05-19
updated: 2026-05-26
tags: [aloepri, attacks]
companion: [aloepri-attack-harness]
superseded_by: aloepri-attack-chronicle
archive_reason: "2026-05-19 Phase 1 OOM incident. Absorbed into chronicle §4 entry for 2026-05-19 (and §2 harness design)."
---

# AloePri attack harness — OOM incident + safeguards

**Date:** 2026-05-19.
**Status:** Safeguards landed. **DO NOT re-run the capture binary until
the user explicitly authorises.**

## What happened

Tried to run the §2.6 fast variant (`--max-prompts 8 --condition all
--engine gpu`) on the AMD Strix Halo iGPU. C0 walked through all 8
prompts in 132.6 s producing **zero snapshots per prompt**, then the
first C1 prompt OOM-killed the process (and as collateral, the kernel
killed `pipewire` and `dbus`):

```
Out of memory: Killed process 1091265 (capture_snapsho)
  total-vm:12681116kB anon-rss:7788244kB ...
session-c540.scope: A process of this unit has been killed by the OOM killer.
```

`MemAvailable` was already tight pre-run (36 GB used, 7.5 / 8 GB
swap). The capture process pushed it over a cliff.

## Two independent root causes

### Bug 1 — `generate(max_tokens=0)` short-circuits before prefill

[`crates/gelo-embedder/src/decoder/generation.rs:144`](../../crates/gelo-embedder/src/decoder/generation.rs)

```rust
if gen_cfg.max_tokens == 0 {
    return Ok(GenerationOutput {
        tokens: Vec::new(),
        stopped_on_eos: false,
    });
}
```

`max_tokens=0` returns **before** `forward::run_prefill` runs, so no
snapshots get captured. The 132 s wall-clock was all the
weight-provisioning we did before each (immediately-returning) call.
This is the right behaviour for the embedder API in general — it's
"how many new tokens to generate" — but the snapshot binary
specifically wanted "prefill only" semantics.

**Fix:** the binary now calls `forward::run_prefill` directly when
`--max-tokens 0`, with a fresh `KvCache::new(...)` sized for the
prompt. `generate()` is only used when the user asks for ≥ 1 decode
token.

### Bug 2 — engine + weight provisioning duplicated per prompt

The original `run_condition` did:

```rust
for prompt in prompts {
    let engine = make_engine()?;            // NEW WgpuVulkanEngine
    let mut exec = ...::new(engine)...;
    provision_decoder_weights(...)?;        // ~3.4 GB upload + clone
    generate(...)?;
}
```

On the iGPU (shared system RAM) this stages ~3.4 GB of f32 weights to
the GPU on every iteration, and wgpu's allocator doesn't eagerly free
between iterations. Plus `InProcessTrustedExecutor` keeps its **own**
TEE-side weight cache via `provision_weight` (another ~3.4 GB clone
of every weight tensor, even with `verify_probes=0`). 8 prompts × a
few GB of fresh allocations is enough to land in OOM territory on a
machine that's already at 36 GB used.

**Fix:** the executor is built **once per condition**:

```rust
let mut executor = ExecVariant::... ;       // one engine, one TEE cache
for (handle, arc) in weight_arcs {          // provision once
    e.provision_weight_shared(*handle, Arc::clone(arc))?;
}
for prompt in prompts {
    let _ = executor.drain();               // reset capture
    run_one_prompt(..., &mut executor, ...)?;
}
```

`weight_arcs` is built once at startup as
`Vec<(WeightHandle, Arc<Array2<f32>>)>` — one `f32` clone per
provisioned tensor — and reused across all three conditions. The
InProc executor uses `provision_weight_shared` to register the
TEE-side cache as an `Arc::clone` rather than a second byte copy,
saving ~3.4 GB on Qwen3-1.7B vs the cloning path.

## Other safeguards landed

3. **Bounded snapshot cap** (`--max-snapshots-per-prompt`, default
   4096). Replaces the previous `max_snapshots: None`
   (unbounded) configuration. Qwen3-1.7B prefill produces 28 × 7 =
   196 snapshots per prompt; the 4096 cap leaves ~20× headroom and
   short-circuits any runaway forward that would otherwise eat
   all of RAM.

4. **Pre-flight `MemAvailable` check** (`--min-mem-gb`, default 8.0
   GB). Reads `/proc/meminfo` and aborts before allocating if the
   host has less than the floor available. `--skip-mem-check`
   bypasses for cases where the operator has measured the actual
   headroom.

5. **Hard failure on 0-snapshot prompts**. If `run_one_prompt`
   returns an empty `Vec`, `run_condition` errors out with a clear
   message — protects against future silent-prefill regressions.

## Where the binary stands now

* `cargo check -p aloepri-attack-snapshot-runner` — clean.
* `cargo test -p aloepri-attack-snapshot-runner --lib` — 2/2 pass
  (export-roundtrip + capturing-plaintext-records-unmasked-operand).
* `cargo test -p gelo-protocol --test snapshot_capture` — 6/6 pass.
* **Not re-run** against real Qwen3-1.7B weights. The user explicitly
  asked to plant safeguards but defer the next run until they say.

## Pre-run checklist for the next session

When the user authorises the re-run:

1. `free -h` shows ≥ 12 GB available (10 GB Qwen3-1.7B working set +
   ~2 GB safetensors export buffer).
2. Containers that aren't needed for the test are stopped. Skip
   `llama-swap` — the user has flagged it as out-of-scope for kills.
3. Start with `--max-prompts 4 --condition c0 --engine gpu` to
   confirm prefill-direct works on the small case before sweeping
   all three conditions.
4. If `c0` succeeds, run `--condition c2 --max-prompts 8` next (the
   release-gate target) before stitching `--condition all` together.
5. Disk: per-condition safetensors at 8 prompts ≈ 200 MB; at 64
   prompts ≈ 1.5 GB; verify `--output` has room.

## Open follow-up

The Phase 2 handoff §4.3 question on "decode-shape vs prefill-shape
attacks" remains untouched — when the binary is happy in
prefill-only mode, the next study is `--max-tokens 8` on a small
prompt subset and re-running IMA/ISA against the new mixed
snapshot set.

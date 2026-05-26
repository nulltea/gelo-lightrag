# Handoff — 2026-05-26 — R4 green-lit, bf16 activations aborted

Focus for next session: **implement §4.D R4 async overlap on iGPU
substrate** (~5-8 days). Q#2 RADV-async spike (`ea29602`) cleared at
~58 % CPU/GPU overlap on Strix Halo UMA, projecting ~12 % production
prefill wall reduction. bf16 activation pipeline was disconfirmed
mid-session and aborted; the infrastructure shipped (phase 1 / 2 / 3a)
remains useful for dGPU revival.

## What landed (12 commits, master branch, pushed through `ea29602`)

Source of truth: `git log 50f6dbd..master --oneline`. Key shifts:

| Commit | What | Outcome |
|---|---|---|
| `55d7c19` | Mask round: per-family profile + Auto 8/5=1.6 threshold + roadmap reorg | Bucket attribution unblocked |
| `5b157b9` | §3.1 measurement-gaps sweep folded into roadmap §1/§4.A + B=1 attn bucket name fix | §3.1 #1/#3/#4 resolved |
| `cd1a008` | **Tile-fused DCT-IV cascade** | **−22.7 % prefill wall at production shape** (174.9 s → 135.1 s; 121 tok/s agg) |
| `792738b` / `7472d2f` / `58d2c28` | `gelo-llm.html` §07 rewrite + stat-strip refactor | Current state visible |
| `310b7ad` | UMA allocator unblock spike | **Non-issue**: B=16 runs clean, no aggregate gain at prod shape |
| `98271f0` | Phase 1 — `decoder::bf16_kernels` (rms_norm/qk_norm/residual bf16) | Infra; not wired |
| `7abb9f1` | Phase 2 — Path β engine `matmul_bf16_input` overrides | Infra; not wired |
| `a05eb8a` | Phase 3a — `Hd3Mask` + `Dct4Mask` bf16 cascade variants | Infra; not wired |
| `909b0a3` | **bf16 cascade microbench** — hypothesis disconfirmed | bf16 chain **ABORTED** for iGPU |
| `ea29602` | **Q#2 RADV-async spike** | **R4 GREEN-LIT** at 58 % overlap; ~12 % wall projected |

## Roadmap state (`docs/plans/gelo-llm-perf-roadmap.md`)

**Shipped this session**:
- §4.A.1 DCT-IV column-locality cascade ✅
- §3.2 #1 R1 weight Arc drop ✅ (already in `4686b8f`, confirmed alive)
- §3.2 #2 UMA allocator unblock ✅ resolved as non-issue
- §3.1 #1 long-n HD₃ sweep cell ✅
- §3.1 #3 B=1 attention bucket capture ✅
- §3.1 #4 pad-ratio probe ✅
- Q#2 RADV-async spike ✅ — partial overlap, R4 viable

**Aborted / deprioritised**:
- §4.E bf16 activation pipeline (phases 1+2+3a infra shipped; cascade
  microbench `909b0a3` showed DCT-IV bf16 gains +8 % standalone =
  ~1.6 % wall projected, below 7 % variance floor; HD₃ bf16
  regresses 2× standalone). Roadmap §4.E carries the disconfirmation.
- §4.E.1 bf16 inner kernels — deferred earlier session

**Still pending**:
- §3.1 #2 variance sweep (80 min; gates every single-cell EV claim)
- **§4.D R4 async overlap** — green-lit, awaiting implementation
- Q4 weight quantization (`docs/plans/q4-gpu-weights.md`)
- dGPU substrate (M5.9, hardware-gated)

## R4 implementation plan (the next session's work)

Per `gelo-llm-perf-roadmap.md` §4.D, the 5-8 day refactor:

| Step | What | Effort |
|---|---|---|
| 1 | Add `engine.matmul_async(handle, input) -> Tensor` + `engine.read_result(t) -> Array2<f32>` to `GpuOffloadEngine` trait. `WgpuVulkanEngine` override using `tensor.into_data_async()` (the burn-tensor primitive is already async-capable — current sync calls are just `try_read_sync(into_data_async())` wrappers; see `~/.cargo/registry/.../burn-tensor-0.20.1/src/tensor/api/base.rs:1832`). | 1 day |
| 2 | Add `substrate.offload_linear_async(handle, hidden) -> OffloadHandle` on `TrustedExecutor` returning a struct holding (mask, n_data, pending_tensor). Companion `wait_offload(h) -> Array2<f32>` that calls `read_result`, runs unmask + strip, returns. Implement in `InProcessTrustedExecutor`. | 2 days |
| 3 | Pipeline `forward.rs` `decoder_block_cached` + `_batched`: issue layer N's QKV+O+gate/up/down matmuls via `_async`, then compute layer N+1's pre-attn RMSNorm + (next layer's) mask cascade host-side while layer N matmuls run on GPU. Wait + unmask for layer N before consuming its output. | 2 days |
| 4 | Parity test (`qwen3_generation_e2e`) + production-shape microbench. Verify the ~12 % wall reduction projection survives end-to-end (vs the single-call spike measurement). | 1 day |

**Spike-validated assumptions**:
- wgpu submit is non-blocking; burn-tensor exposes async via `into_data_async`
- CPU/GPU overlap at ~58 % efficiency on Strix Halo UMA (`bench-results/q2-radv-async-spike-2026-05-26_14-14-23.log`)
- At production shape: T_gpu 19 ms, T_cpu 10 ms per matmul × cascade pair

**Open risks for R4**:
1. Per-shape overlap varies: spike used d_out=2560 (O projection); QKV is smaller (faster GPU, mask is larger relative), gate/up is larger (slower GPU, mask easier to hide). Need to measure cross-shape and pipeline accordingly.
2. Forward-pass pipelining may need a 2-stage queue (layer N matmul + layer N+1 mask + layer N-1 unmask all in flight). Care needed with `OffloadHandle` ownership across the loop iteration.
3. R4 doesn't help decode much — at decode, the mask bucket is only 4 % of decode wall, so overlap savings are tiny per step. R4's value is prefill-dominated.

## Pages deploy status

`master` was pushed (`909b0a3..ea29602`) but GitHub Pages deploy
failed twice on `actions/configure-pages@v5` download. Root cause:
**GitHub-side active critical incident** ("Authentication issues
leading to failure in starting Actions runs and downloading
actions", started 10:57 UTC, still investigating as of 12:17 UTC).
Not actionable from our side.

**To redeploy when GitHub recovers**:

```bash
gh run rerun 26447521101
# or watch the next push trigger it automatically
gh run list --workflow=pages.yml --limit 1
```

Check upstream status before rerun: `curl -s https://www.githubstatus.com/api/v2/incidents/unresolved.json | jq -r '.incidents[].name'`.

## Cross-cutting findings (worth carrying to memory)

1. **UMA allocator cap is a non-issue post-cubecl `tasks_max=32`**.
   The 2026-05-22 handoff's "8 GiB per-submission cap" was a
   three-executors-alive squeeze; current sequential-executor pattern
   doesn't hit it. B=16 n=2048 K=8 runs clean.

2. **B-scaling does NOT amortise prefill at production shape**.
   B=8 → B=16 measured prefill aggregate 121.3 → 114.8 tok/s
   (−5 %, slightly worse). GPU compute already saturated at B=8 on
   long-n. Decode amortises +22 % (4.62 → 5.66 tok/s aggregate; the
   surviving launch-overhead-amortised gain at `n_q=1`).

3. **bf16 hypothesis disconfirmed**. Cascade refactor already
   captured the L2-resident-tile bandwidth win. bf16 storage at the
   tile boundary halves only the residual ~2 buffer-passes. DCT-IV
   wins +8 % standalone → ~1.6 % wall (below variance). HD₃ bf16
   regresses 2× (bulk widen-narrow + per-call allocation).

4. **Zen 5 has no bf16 add/sub SIMD**. AVX-512_BF16 only has
   `VDPBF16PS` (dot-product to f32). bf16 arithmetic in FWHT/DCT-IV
   means widen → f32 compute → narrow. Byte savings, not compute
   savings. dGPU (CUDA Tensor Cores) inverts this — bf16 compute is
   native, so the precision-vs-bandwidth math rewrites itself.

5. **Real-weight microbench framework standardised**. Variance at
   long-n B=8 production shape is ~7 % single-cell; §3.1 #2 sweep
   should fire before any further single-cell EV claim is treated
   as ground truth. ~80 min run.

## Bench artefacts saved

All under `bench-results/`:
- `dct4-cascade-microbench-2026-05-26_11-35-01.{log,tsv}` — cascade-refactor headline
- `measurement-gaps-2026-05-26_10-34-30.{log,tsv}` — §3.1 sweep (3 cells)
- `measurement-gaps-cell4-rerun2-2026-05-26_11-00-58.{log,tsv}` — B=1 attn bucket
- `uma-spike-2026-05-26_12-34-26.log[.summary]` — B=16 non-issue
- `bf16-cascade-microbench-2026-05-26_13-47-56.log` — bf16 disconfirmation
- `q2-radv-async-spike-2026-05-26_14-14-23.log` — R4 green-light

## Pinned reading for next session

- `docs/plans/gelo-llm-perf-roadmap.md` §4.D (R4) — the implementation gate is now lifted
- `docs/plans/m1-12-bf16-activation-pipeline.md` — the bf16 plan; preserved but §4.E in the roadmap supersedes its priority verdict
- `crates/gelo-gpu-wgpu/tests/q2_radv_async_spike.rs` — spike measurement code; pattern to follow when adding `matmul_async` parity tests
- `~/.cargo/registry/.../burn-tensor-0.20.1/src/tensor/api/base.rs:1817-1835` — `into_data` vs `into_data_async` (the async API surface R4 will consume)
- `crates/gelo-gpu-wgpu/src/lib.rs:337` — current `matmul` impl; the async sibling lives here
- `crates/gelo-protocol/src/sim.rs:1801` — `InProcessTrustedExecutor::offload_linear`; pattern for the async sibling

## Suggested skills for the next session

- **`diagnose`** if R4's async pipeline shows unexpected wall numbers —
  the spike says 58 % overlap but real-pass-through may differ; bisect
  per layer-stage to identify which matmul shape contributes most.
- **`grill-with-docs`** before flipping any default-on R4 behaviour —
  the async refactor changes per-call ordering which may surface
  AloePri timing-side-channel concerns (paper §3.2-ish argument
  worth re-reading before commit).
- **`verify`** after R4 lands — confirm token parity on
  `qwen3_generation_e2e` + microbench shows the projected ~12 % wall
  reduction at the production shape.

## Memory entries worth saving for future sessions

(Not yet written — candidates for the next session to capture if the
findings hold under further measurement)

1. **iGPU UMA submission cap is no longer the binding gate** —
   resolved by cubecl `tasks_max=32` default chunking. Don't pursue
   "raise the wgpu cap" as a perf lever (per `2026-05-22-q3-4b-b8-mask-sweep.md` item #3).
2. **bf16 cascade microbench disconfirmation** — DCT-IV +8 %
   standalone projects below variance; HD₃ regresses 2× under bulk
   widen-narrow. Don't re-spike bf16 cascade on iGPU until §4.E.1
   inner-kernel work materially changes the math.
3. **Q#2 RADV-async spike outcome** — partial overlap 58 % at
   production shape; CPU/GPU contend on DDR5 but not crushingly.
   R4 wall projection ~12 %.

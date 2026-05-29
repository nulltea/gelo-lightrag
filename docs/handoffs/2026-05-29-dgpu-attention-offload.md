---
type: handoff
status: current
created: 2026-05-29
updated: 2026-05-29
tags: [gelo, dgpu, nvidia, rtx5090, cuda, vulkan, attention, handoff]
focus: GPU attention offload (next step)
---

# Handoff — dGPU bring-up done; next step is GPU attention offload

/ Box: shared remote, **AMD Ryzen 9 7900X + NVIDIA RTX 5090** (Blackwell,
sm_120, 32 GB), Ubuntu 24.04, CUDA toolkit 13.0.3. This is the discrete
GPU the GELO perf roadmap kept deferring to ("dGPU substrate, M5.9,
hardware-gated") — it is now in hand. /

## 1. What this session accomplished (don't re-derive — read the artifacts)

- **Build portability**: the gelo crates now build + run on a non-AMD box
  with **no ROCm and no cmake**. Committed (branch `dgpu-nvidia-bringup`,
  commit `be6e809`). Key moves: rcgen `aws_lc_rs`→`ring` (drops the only
  cmake dep), removed cubecl-hip + its vendored patch + the
  `q4_hip_kernel_spike` test.
- **Vulkan→CUDA port (opt-in)**: `cuda` feature on `gelo-gpu-wgpu`
  compile-time swaps the cubecl runtime (`WgpuRuntime`→`CudaRuntime`)
  behind `Rt`/`Dev` aliases; Vulkan stays the default. Runs end-to-end on
  the 5090. Same commit `be6e809`.
- **Profile trace split** (commit `5b50da2`): GPU offload now emits
  `engine:matmul` (single-weight: O, FfnDown, R3 LM-head) vs
  `engine:matmul_many` (fused QKV, gate∥up), replacing the merged
  `engine:registered_linear` bucket.
- **Main bench renamed** `gelo_llm_prefill_decode_breakdown` (R3-only),
  added `GELO_BENCH_WARMUP`; documented as the canonical Gelo-LLM bench in
  `CLAUDE.md`.
- **All measurements** (iGPU comparison, CUDA-vs-Vulkan A/B at B=1 and
  B=8, per-op tables) are in
  **`docs/dev/logs/gelo-llm-perf-chronicle_dgpu.md`** (§4 iGPU compare,
  §8 CUDA A/B narrative, §9 full per-op tables). Read that, don't requote.

## 2. The headline finding that sets up the next step

CUDA is a **modest ~10–16 % wall lever** (GPU matmul ~1.2–1.9× faster);
**not** a step change. Why: the binding bottlenecks are **backend-
invariant** and the GPU backend can't touch them:

1. **In-TEE attention** (`tee:attn_cached*`) — CPU, **~52 % of decode wall
   at B=1, ~34 % at B=8, ~21 % of prefill**. Runs in the TEE on the CPU
   (softmax must stay in-TEE under F1+). Identical across Vulkan/CUDA.
2. **Per-call masked round-trip** — every offloaded matmul reads its
   result back to the TEE to unmask, serialising dispatches (effective
   GPU throughput ~1–2 TFLOP/s on a ~300-TFLOP/s card). Backend-agnostic.

**⇒ The next lever is moving attention onto the GPU.** That is exactly
the focus of this handoff.

## 3. NEXT STEP — GPU attention offload (the actual task)

The design already exists; do not reinvent it. Read:
- **`docs/handoffs/2026-05-22-dgpu-attention-revival.md`** — the full
  design: Item 1 persistent K/V on GPU (1A block-fresh-π vs 1B
  TwinShield-Xue, with σ-vs-N security gate), Item 2 GQA-aware WGSL
  kernel (4× K/V motion cut), Item 3 single-pass FlashAttention (FLASH-D).
  Includes a **Step-0 bench-triage** recipe to run first on dGPU.
- **`docs/plans/gelo-llm-perf-roadmap.md` §4.C.2** — EV/engineering at a
  glance for the same levers.
- **`docs/dev/logs/gelo-llm-perf-chronicle.md` §4 (2026-05-22 entry)** —
  why iGPU bucket-2 was aborted (16.4× slower on UMA: upload pipeline) and
  why dGPU changes the math (separate HBM bus). The chronicle's "Methodology
  → anti-patterns" section is load-bearing (don't push the mask to GPU,
  don't break OEM-agnostic, fresh-per-batch mask, etc.).

Critical constraints to preserve:
- **F1+ threat model**: the causal softmax must not leak π to the GPU; any
  attention offload must keep the softmax-blinding / in-TEE-softmax
  property. This is *the* reason attention is still on CPU.
- Persistent K/V needs a **security spike** (σ-vs-N curve or
  TwinShield-Xue validation) before it can land — gate, not just code.
- The async substrate (R4) is a precondition for overlap on PCIe and was
  retained on a feature branch on the iGPU track; check whether it's
  relevant here.

Suggested first action on the dGPU: run the Step-0 triage from the revival
handoff (re-measure `gpu_batched_b8` vs `in_tee_rayon_b8` at n_kv=2048 on
the 5090) to see where the HBM-vs-upload ratio now lands before investing
in the WGSL kernel.

## 4. How to build & run on this box (environment is non-obvious)

```bash
source "$HOME/.cargo/env"          # rustup-installed toolchain (1.96), not on default PATH
# BLIS already built at vendor/aocl-install (libblis-mt.so); .cargo/config.toml wires the rpath.
# Canonical bench (Vulkan default):
GELO_BENCH_VARIANT=4b GELO_BENCH_B=1 GELO_BENCH_N=2048 GELO_BENCH_MAX_TOKENS=32 \
  cargo test --release -p gelo-gpu-wgpu --test qwen3_m1_12_r1_q1_microbench \
  gelo_llm_prefill_decode_breakdown -- --ignored --nocapture
# CUDA backend: add --features cuda
# Fair A/B: prepend GELO_BENCH_WARMUP=1 (discards one forward to cache autotune)
```
Gotchas:
- **Always `--release`.** Debug pays minute-scale cubecl shader compile.
- **Vulkan warm ≈ cold** (cheap SPIR-V autotune), but **CUDA cold is
  2–4× slower than warm** (nvrtc per-shape autotune; ~27 s one-time at B=8
  vs ~1 s at B=1). Use the warmup knob for any CUDA comparison.
- Qwen3-4B weights are cached at `~/.cache/huggingface` (7.6 GB, bf16) —
  no re-download.
- wgpu auto-selects the 5090 (`NVIDIA GeForce RTX 5090 (DiscreteGpu)`);
  confirm via the bench banner / `nvidia-smi` memory movement.
- Engine `cuda` cfg-alias structure lives in
  `crates/gelo-gpu-wgpu/src/lib.rs` (imports + `Rt`/`Dev` + `gpu_ctx`).

## 5. Repo / VCS state

- Branch **`dgpu-nvidia-bringup`**, 2 commits ahead of `master`
  (`be6e809`, `5b50da2`). **Push may still be pending** — it was the
  user's action (secure throwaway-PAT push, then revoke).
- Local commit identity set repo-only as `Timo <timofey.luin@gmail.com>`
  (matches history; harness email differs — `timofey@chainsafe.io`).
- `Cargo.lock` is **gitignored** (don't expect it in diffs).
- `skills/` is a pre-existing untracked Claude-plugin dir — **leave it**.
- **SECURITY**: a fine-grained PAT was pasted in the prior session's
  transcript. It must be **revoked** in GitHub settings (single-repo,
  Contents-only, short-expiry — but treat as compromised). Do not reuse.
  No token is persisted in the repo/`.git/config` (audited clean).

## 6. Ephemeral artifacts (capture to `bench-results/` if you want them kept)

`/tmp/warm_vulkan_b1.log`, `/tmp/warm_cuda_b1.log`, `/tmp/gelo_b8_n2048.log`
(Vulkan B=8 cold), `/tmp/warm_cuda_b8.log` (CUDA B=8 warm). These are the
sources behind chronicle §9; they live in /tmp and will not survive.

## 7. Suggested skills for the next session

- **`grill-me`** (or `grill-with-docs`) — before committing to the
  attention-offload design (persistent-K/V cover scheme 1A vs 1B, kernel
  approach). The σ-vs-N security trade and the F1+ leak constraint are
  load-bearing and deserve grilling, as the revival handoff itself flags.
- **`verify`** — at the bench-acceptance step (the revival handoff's
  Step-5 gate: ≥30 % decode-wall reduction on top of R3, no extra
  TEE↔GPU round-trips).
- **`code-review`** — before any `GpuOffloadEngine` trait change for
  session-resident K/V (every engine impl must follow).
- **`handoff`** — to roll forward again when this leg completes.

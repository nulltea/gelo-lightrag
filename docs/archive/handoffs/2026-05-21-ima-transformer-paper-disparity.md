---
type: handoff
status: current
created: 2026-05-21
updated: 2026-05-21
tags: [ima, attacks]
companion: [2026-05-20-ima-embedrow-transformer-investigation]
---

# Handoff — IMA-EmbedRow-transformer GPU baseline + paper-disparity investigation

**Date:** 2026-05-21
**Branch:** `path-2-aloepri-gemma` (uncommitted: Docker GPU pipeline, harness overhaul, checkpoint cache, eval-cadence patch)
**Companion handoffs:**
- `docs/archive/handoffs/2026-05-21-strong-pi-server-patch.md` — strong-Π + `chat_parser=epsilon` workaround (separate workstream)
- `docs/archive/handoffs/2026-05-20-ima-embedrow-transformer-investigation.md` — prior diagnosis ("broken-attack" reading from the residual-MLP era)

## Headline

We rebuilt `IMA-EmbedRow-transformer` to be paper-faithful (reference impl's exact `_PaperLikeIMAInverter`: 2-layer Qwen2 backbone + sequence-window pipeline + paper hyperparams) and ran it on the Strix Halo iGPU.

**At ep=100, plain identity-τ = 15.62 % top-1, obfuscated 4B = 12.70 % top-1.** Both curves are **still monotonically climbing — not saturated**. Paper claims 0 % on IMA for Qwen2.5-14B; we're at 12.70 % and growing. Either our attack isn't the same attack as paper's, or there's a config / obfuscator divergence, or paper's reading itself is the undertrained-attacker number (we saw a similar disparity story in earlier sessions). The next session's job is to investigate.

## What landed this session

### 1. Full overhaul of `IMA-EmbedRow-transformer` in `evals/aloepri-attacks/m2_7/run_ima_embedrow_attacks.py`

Replaced the prior residual-MLP per-row inverter with a byte-faithful port of `vendor/aloepri-py/src/security_qwen/ima.py::run_ima_paper_like`:

- Inverter = `AutoConfig.from_pretrained(baseline_model_dir)` + `AutoModel.from_config(config)` with `hidden_size = d_obs`, `num_hidden_layers=2`, `num_attention_heads=8`, `num_key_value_heads=8`, `intermediate_size = max(d_obs * 4, base.intermediate_size)`. Output head = `nn.Linear(d_obs, d_plain, bias=False)`.
- Inputs = sequence windows of 32 consecutive plain ids from a public corpus → look up `obs_W_e[tau[plain_id]]` per position → `(B, T=32, d_obs)`.
- Targets = `plain_W_e[plain_id]` per position.
- Training = `AdamW(lr=3e-4, wd=0)`, `batch_size=8`, MSE loss, paper-default `epochs=2` (overridable).
- Public corpus default: `vendor/aloepri-py/docs/*.{md,txt}` + `vendor/aloepri-py/README.md` (same as reference impl).
- CLI: `--baseline-model-dir Qwen/Qwen2.5-0.5B-Instruct`, `--public-corpus-path`, `--paper-{sequence-length,train-sequence-count,val-sequence-count,test-sequence-count,batch-size,epochs,lr,weight-decay,candidate-pool-size,device,checkpoint-dir}`. Default device = `auto` (= `cuda` on ROCm or NVIDIA).

Reused the reference's actual functions via `load_aloepri_module` (`build_paper_like_inverter_config`, `_PaperLikeIMAInverter`, `_collect_public_token_windows`, `_evaluate_sequence_inversion_predictions`). One side change in `attack_drivers/common.py:_ensure_module_stub`: skip stubbing `transformers` / `tokenizers` when the real packages are importable — otherwise the new code crashes with "AutoTokenizer has no attribute from_pretrained" because the loader had been silently stubbing them.

### 2. GPU pipeline — ROCm 7.2.3 on AMD Strix Halo gfx1151

Pipeline (verified end-to-end, matmul 2048×2560 @ 2560×2560 = 9.6 ms/iter, full AdamW step = 87 ms/iter):

- **Base image**: `rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.10.0` — official AMD, ROCm 7.2.3 + PyTorch 2.10 + Python 3.12, **gfx1151 support compiled in** (we confirmed this is the only common image that works on Strix Halo — see "dead ends" below). 40 GB disk / 10 GB content.
- **Trainer image**: `aloepri-ima-trainer:latest` — layered on the base, adds `transformers + safetensors + numpy + gguf + accelerate + ml_dtypes`. Dockerfile at `evals/aloepri-attacks/m2_7/Dockerfile.ima-trainer`.
- **Run wrapper**: `evals/aloepri-attacks/m2_7/run_in_gpu_container.sh` — forwards all CLI args, bind-mounts repo + HF cache + `/tmp` + checkpoint dir at the **same paths inside the container as on host** so host-side absolute paths in args resolve unchanged. Injects `HOME` + `--paper-checkpoint-dir` automatically.

Operator command (already working today):

```bash
bash evals/aloepri-attacks/m2_7/run_in_gpu_container.sh \
    --plain /home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/Qwen3-4B-Q8_0-untied.gguf \
    --obfuscated /home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-bf16-native.gguf \
    --key      /home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/untied-keymat-h128-pi-noise-ae1.0-ah0.2-alg2-matrix-gamma-hadamard-bf16-native.gguf.key.npz \
    --output   /tmp/aloepri-gpu-validation/ima-ep500-obf-4B.json \
    --baseline-model-dir Qwen/Qwen2.5-0.5B-Instruct \
    --paper-epochs 500 \
    --paper-device gpu
```

For plain identity-τ control: pass `--identity-tau` and `--obfuscated $PLAIN` (same path twice, no `--key`).

### 3. Checkpointing — resumable training across epoch budgets

- Default checkpoint dir: `~/.cache/aloepri-ima-checkpoints/` (host) mounted at the same path inside the container.
- Checkpoint path is **content-addressed** on every hyperparameter except `epochs`. Key derivation in `run_ima_embedrow_attacks.py::_ima_xformer_ckpt_path`: SHA256 of `plain_path | obfuscated_path | tau_fingerprint | baseline_model_dir | seq_len | seq_train | seq_val | seq_test | batch | lr | wd | pool | seed`. Filename = `ima_xformer_<16-char-hash>.pt`.
- Stored: `model_state + optimizer_state + best_state + best_val_top1 + best_epoch + epochs_done + epoch_summaries`. ~7 GB per checkpoint (large because Qwen2.5-0.5B backbone × 3 for model/optimizer/best — see "trim opportunity" below).
- Behaviour:
  - On run start, look up the matching `.pt`. If `epochs_done >= --paper-epochs`, **skip training**, restore best_state, run final test eval, return cached metrics.
  - If `epochs_done < --paper-epochs`, **resume from `epochs_done`** and train `(epochs - epochs_done)` more epochs.
  - On save, write the updated state + epoch counter atomically.
- Save is wrapped in try/except with explicit error logging — earlier silent failure (lost an 11-minute training run) is fixed.
- Disable checkpointing with `--paper-checkpoint-dir ''`.

**Currently cached checkpoints:**
- `~/.cache/aloepri-ima-checkpoints/ima_xformer_2cb1dff683633e8a.pt` (7.1 GB) — obfuscated 4B real-τ, ep=100. Hash derived from current path-2-aloepri/qwen3-4b paths.

### 4. Eval-cadence patch — full-vocab cosine NN was 80 % of runtime

Per-epoch eval against full vocab (151,936 candidates) was 5.5 s × N_epochs. Patched `run_ima_embedrow_transformer` to use a log-spaced schedule (`_should_eval`):

```
epoch ≤ 5             every epoch
5 < epoch ≤ 50        every 5
50 < epoch ≤ 500      every 25
epoch > 500           every 100
always evaluate at the final epoch
```

Expected speedup at common budgets: 3.5× at ep=100, 4.3× at ep=1000, 5.5× at ep=5000. Eval is not in the gradient path (it's just best-state selection + curve logging) so this is loss-free for final TTRSR.

**Note:** the just-completed ep=100 runs were *before* this patch. Subsequent runs benefit.

### 5. CLI device-name cleanup

`--paper-device cuda` was misleading on this AMD box. Updated to accept `auto` / `gpu` / `cpu` / `cuda`. Internally still maps to PyTorch's `'cuda'` device string (which PyTorch reuses for both CUDA and ROCm/HIP). `'cuda'` kept as a backward-compat alias.

### 6. Dead ends explored — don't repeat

- Direct `pip install torch --index-url …/rocm7.0` on host venv: 12 GB install, **SIGSEGV on every tensor op** (gfx1151 not in wheel's kernel target list).
- `kyuz0/amd-strix-halo-toolboxes:rocm-7.2.3` (already on box, Strix-Halo-specific): ships ROCm 7.2.3 runtime but **no PyTorch**. Layering pip torch 2.9.1+rocm6.4 on top → `hipErrorInvalidDeviceFunction` (wheel's bundled rocblas missing gfx1151 kernels, even though `/opt/rocm` runtime has them). Cleaned up.
- Build PyTorch from source against `/opt/rocm`: too slow for the current cycle (3–4 h compile); revisit if image size becomes a problem.
- Vulkan PyTorch backend: experimental mobile-inference only, no training support.

### 7. Auto-memory rule added

`feedback-no-cpu-for-gpu-workloads` ([memory/feedback_no_cpu_for_gpu_workloads.md](../../../.claude/projects/-home-timo-repos-private-rag/memory/feedback_no_cpu_for_gpu_workloads.md)). Heavy ML workloads on this box MUST use GPU; CPU is never the right answer when a GPU backend exists.

## Headline measurements

| Cell | top-1 | top-10 | best epoch | runtime | notes |
|---|---|---|---|---|---|
| Plain identity-τ 4B @ ep=100 | **15.62 %** | 26.56 % | 99 | 11.5 min | Checkpoint LOST (pre-fix run) |
| Obfuscated 4B real-τ @ ep=100 | **12.70 %** | 23.24 % | 98 | 24 min (degraded by parallel-run collision; would be ~12 min alone) | Checkpoint saved at `ima_xformer_2cb1dff683633e8a.pt` |

Defence gap at ep=100: **top-1 = 2.92 pp, top-10 = 3.32 pp**. Both curves still climbing (best epochs 98-99 of 100).

Result JSONs in `/tmp/aloepri-gpu-validation/`:
- `ima-ep100-plain-4B.json` — plain identity-τ
- `ima-ep100-obf-4B.json` — obfuscated 4B real-τ
- `ima-paper-faithful-identity-tau-4B-ep2-gpu.json` — earlier sanity run at paper-default ep=2 (top-1 = 1.95 %)

Configuration both cells share: `Qwen/Qwen2.5-0.5B-Instruct` inverter arch (paper §F.1 says "Qwen2 with 2 decoder layers and 8 attention heads" — Qwen2.5 = Qwen2 architecture), `sequence_length=32`, `train_sequence_count=128`, `batch_size=8`, `lr=3e-4`, `wd=0`, full-vocab candidate pool (151,936), `seed=20260518`.

## The disparity — what the next session needs to investigate

**Paper §F.1 claims `IMA = 0 %`** on AloePri-defended Qwen2.5-14B (paper Table 2). **Our reproduction reports 12.70 % on 4B at ep=100 and growing.** Three competing explanations to test in order:

### Hypothesis A — Our attack isn't paper's attack (most likely)

Even though we ported `run_ima_paper_like` byte-for-byte from the reference impl, there are still possible divergences:

- **Inverter arch**: paper says "Qwen2 model with 2 decoder layers and 8 attention heads." We use `Qwen/Qwen2.5-0.5B-Instruct` config → Qwen2Model class via `AutoModel.from_config`. Need to verify Qwen2.5 hf-transformers wrapper is identical to whatever the paper-authors used in 2025.
- **Public corpus**: we use `vendor/aloepri-py/docs/*` (paper, internal docs, README) — total ~110 KB. Paper §7.1 mentions "CCI3, Huatuo26M-Lite, MedDialog" datasets but doesn't specify which is used to train the IMA inverter. If paper uses a domain-mismatched corpus (medical text against general English target deployment), training data might be much weaker than ours.
- **`observed_embed` source**: reference impl loads `observed_embed` from the deployment's actual obfuscated weights (`server_dir/model.safetensors`). We load from our obfuscated GGUF. Should be byte-equivalent if our obfuscator + dequantiser is correct. But: any divergence between reference's obfuscator and ours (path-2's `obfuscate_qwen3_gguf.py`) would produce a different `W̃` and a different attack baseline.
- **Hyperparam mismatch**: paper §F.2 fixes `epochs=2`. We're at ep=100. **Paper might genuinely consider "constrained-attacker = 2 epochs" the load-bearing claim** — i.e., paper's 0 % is the ep=2 reading, which we also got (~2 %). If so, the disparity is on the *interpretation* axis: paper's claim is "constrained attacker can't recover," not "no attacker can recover."

**Test:** run our ep=100+ inverter against the **reference impl's Stage-K obfuscated artifact** (vendored at `vendor/aloepri-py/artifacts/stage_k_release/`). If we get the same 12.70 %-and-climbing, the disparity is paper's claim ≠ what their own measurement supports — our attack is the truer reading. If we get 0 %, the disparity is in our deployment (path-2 GGUF differs from reference's stage_k obfuscation in a way that lets ridge-class attacks succeed).

### Hypothesis B — Our AloePri obfuscation is weaker than paper's

Path-2's `python/aloepri-llm/obfuscate_qwen3_gguf.py` could have a bug or weaker default that paper's reference doesn't:

- α_e = 1.0 (paper default — match)
- α_h = 0.2 (paper default — match)
- λ = 0.3 (paper default — match)
- h = 128 (paper default — match)
- β = 8 (paper default — match)
- γ = 1e3 (paper default — match)
- bf16 quantisation (we changed default 2026-05-20; paper uses fp16/bf16)
- Algorithm 2 matrix-Γ enabled (we enable it; paper does too)

The structural-leak risk: see the `2026-05-21-strong-pi-server-patch.md` handoff — until that lands, **specials + UNUSED tail are identity-τ in our obfuscator** (~293 free pairs). Reference impl might Π'd these. This could be the disparity source for *ridge* attacks but **the transformer attack doesn't use τ-leak training data** — it uses public-corpus + the deployment's W̃ table. So the specials leak isn't the explanation here.

**Test:** diff our generated `W̃` against the reference impl's Stage-K artifact, head-by-head, to find any structural difference.

### Hypothesis C — Configuration / harness bug

- The plain identity-τ run reaching 15.62 % at ep=100 is harder to explain by "broken attack" — even on plain (no obfuscation) the test top-1 is only 15 %, not the ~99 % ridge gets. This says the **trained inverter pipeline itself is limited**, *not* that the attack on obfuscated is exposing a defence gap. The 12.70 % vs 15.62 % comparison may just be measuring the inverter's training noise floor, not the defence at all.
- That said, both numbers are climbing in lockstep. **At ep=1000 we should see plain rise to 50–90 % and obfuscated either catch up (→ broken defence) or fall behind (→ real defence)**. Until we measure that, we can't distinguish "obfuscation working" from "attack undertrained, gap is just noise."

**Test:** continue training to ep=500 and ep=1000 on both cells. Use the checkpoints. The shape of plain-vs-obfuscated divergence as epochs grow is the load-bearing signal.

## Next session's concrete plan

### Step 0 — Re-create the lost plain checkpoint (~10 min)

```bash
bash evals/aloepri-attacks/m2_7/run_in_gpu_container.sh \
    --plain /home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/Qwen3-4B-Q8_0-untied.gguf \
    --obfuscated /home/timo/.cache/huggingface/path-2-aloepri/qwen3-4b/Qwen3-4B-Q8_0-untied.gguf \
    --identity-tau \
    --output /tmp/aloepri-gpu-validation/ima-ep100-plain-4B.json \
    --baseline-model-dir Qwen/Qwen2.5-0.5B-Instruct \
    --paper-epochs 100 \
    --paper-device gpu
```

With the eval-cadence patch this should be ~3.5 min instead of 11.5. Writes the plain checkpoint we lost on the first run.

### Step 1 — Extend both cells to ep=500 (~15 min total on GPU, checkpoint-resumed)

Same commands as Step 0 with `--paper-epochs 500`. The harness picks up the cached state and trains the additional 400 epochs.

Inspect plain-vs-obfuscated divergence. If plain rises significantly above obfuscated, defence is real (just paper undersold via 2-epoch budget). If both stay near each other, defence is structurally weaker than paper claims.

### Step 2 — Run our inverter against the reference impl's Stage-K artifact

This isolates "is the attack different?" from "is the deployment different?":

- Reference Stage-K artifact: `vendor/aloepri-py/artifacts/stage_k_release/` (need to verify presence + load path).
- Reuse the same Qwen2.5 inverter + same hyperparams.
- Compare top-1 trajectory vs our path-2 obfuscated 4B.

If reference Stage-K behaves like our path-2 → paper's 0 % is the 2-epoch reading. If reference Stage-K behaves differently → path-2's obfuscator diverges from reference.

### Step 3 — Slim the checkpoint format (optional, after disparity is resolved)

7.1 GB per checkpoint is large. Options to trim:
- Skip `optimizer_state` (saves ~4 GB; loses Adam momentum on resume — minimal impact after ≥ 100 warmup epochs)
- Save only `best_state`, not the current `model_state` too (saves ~2 GB)

Together: ~7.1 GB → ~1 GB. Easy win once we're past the disparity investigation and into long sweeps.

## Operator references

- New driver: `evals/aloepri-attacks/m2_7/run_ima_embedrow_attacks.py` (paper-faithful `run_ima_embedrow_transformer`)
- Wrapper: `evals/aloepri-attacks/m2_7/run_in_gpu_container.sh`
- Dockerfile: `evals/aloepri-attacks/m2_7/Dockerfile.ima-trainer`
- Trainer image: `aloepri-ima-trainer:latest` (depends on `rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.10.0`)
- Checkpoint cache: `~/.cache/aloepri-ima-checkpoints/`
- Threat-model writeup: `docs/research/aloepri-attacks.md`
- Reference impl: `vendor/aloepri-py/src/security_qwen/ima.py::run_ima_paper_like`
- Public corpus (default): `vendor/aloepri-py/docs/*.{md,txt}` + `vendor/aloepri-py/README.md`

## Open uncommitted work

```
M  docs/research/aloepri-attacks.md           (threat-model overhaul — already coherent)
M  docs/prototype/aloepri-llm.html            (§08 cleaned of "row" / "path-2" mentions)
M  evals/aloepri-attacks/m2_7/run_ima_embedrow_attacks.py   (paper-faithful inverter + checkpoint + eval cadence)
M  evals/aloepri-attacks/m2_7/run_in_gpu_container.sh        (new file — Docker wrapper)
M  evals/aloepri-attacks/m2_7/Dockerfile.ima-trainer         (new file)
M  evals/aloepri-attacks/attack_drivers/common.py            (load_aloepri_module: skip stub when real package importable)
… (plus the strong-Π workstream from the companion handoff)
```

Commit recommendation: bundle the IMA-EmbedRow-transformer overhaul + GPU pipeline + checkpoint cache + eval-cadence in **one logical commit**; the docs (`aloepri-attacks.md`, §08) reference the new attack semantics so they belong in the same commit.

## Suggested skills for the next session

- **`/diagnose`** when running Step 2 (reference vs path-2 obfuscation diff) — disciplined hypothesis-test loop is the right shape for "where does this 12.70 % come from?"
- **`/grill-with-docs`** if drafting the "paper's IMA claim is actually the constrained-attacker reading" framing for `aloepri-attacks.md` — the threat-model writeup needs careful language alignment with paper §5.1's "constrained attackers" remark.
- Skip `/handoff` at the end of the next session unless an unexpected branch opens — the disparity investigation is well-scoped.

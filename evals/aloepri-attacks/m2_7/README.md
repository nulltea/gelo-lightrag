# M2.7 — capture frontend for the §05 obfuscated Qwen3 1.7B artifact

Run the AloePri attack suite against the actual obfuscated GGUF
(`keymat-h128-pi-noise-alg2-fp32.gguf`, §05 of
`docs/prototype/aloepri-llm.html`), producing the per-attack
TTRSR numbers that the existing path-1 measurements can't supply.

## Coverage at land

| Attack | Frontend | Status |
|---|---|---|
| **VMA** | static-weight: load both GGUFs, RowSort + sorted-quantile feature matching | code landed (`run_static_attacks.py`) — **not run** |
| **IA** (Gate-IA + Attn-IA) | static-weight: invariant matching on `(W_e, W_gate / Q / K)` pairs | code landed (`run_static_attacks.py`) — **not run** |
| **TFMA** | token-stream: drive `llama-server` `/completion`, capture obfuscated token-id sequences | code landed (`capture_token_streams.py` + `run_token_attacks.py`) — **not run** |
| **SDA** | token-stream: bigram signature + substitution decipherment + BLEU-4 | code landed (`run_token_attacks.py`) — **not run** |
| **NN** | hidden-state: tensor dump via patched llama.cpp + `capture_hidden_states.py --mode hidden` | code landed (Option B file-dump) — **not run** |
| **IMA basic** | hidden-state at layer 0, same path | code landed — **not run** |
| **IMA paper-like** | hidden-state at layer 0 + ≥ 256 prompts, same path | code landed (needs corpus growth) — **not run** |
| **ISA** | hidden-state at layer 23 (HiddenState) + `kq-23` tensor (AttnScore, requires `--flash-attn off`) | code landed — **not run** |

## What's on disk

* `extract_gguf_weights.py` — loads a Qwen3 GGUF (plaintext Q8_0 or
  obfuscated fp32) via the `gguf` library, dequantises on the fly,
  returns the load-bearing weight tensors (token_embd, output,
  per-layer `attn_q/k/v/output`, `ffn_gate/up/down`).
* `run_static_attacks.py` — wraps `extract_gguf_weights` for both
  GGUFs, ports AloePri's RowSort + sorted-quantile VMA and the
  Gate-IA / Attn-IA invariant attacks. Writes a results JSON.
* `spawn_obfuscated_server.sh` — Docker spawn for a dedicated
  `llama.cpp:server-vulkan` instance bound to localhost:8061,
  serving the §05 GGUF. Container name `aloepri-m2_7-server` —
  separate from the persistent `llama-swap` container.
* `capture_token_streams.py` — drives the spawned server with the
  64-prompt corpus via the existing
  `python/path-2/aloepri_client.py::AloePriClient`, captures
  `(plain_prompt_ids, obf_prompt_ids, obf_response_ids)` per
  prompt to JSONL.
* `run_token_attacks.py` — feeds the captured JSONL to AloePri's
  TFMA (frequency-rank matching) and SDA (bigram-signature
  substitution + BLEU-4) primitives.
* `HIDDEN_STATE_GAP.md` — documents the Hard-effort branch (NN /
  IMA / ISA at intermediate layers) and the three options for
  filling it.

## Operator runbook (does NOT auto-run anything)

All of the below are commands you would invoke yourself once you
decide M2.7 should be exercised. Pre-flight memory: both GGUFs
together need ~20 GB of working RAM (plaintext dequant + obfuscated
fp32 + working buffers); check `free -h` shows ≥ 25 GB available
before running the static attacks.

### 1. Static-weight attacks (VMA, IA) — ~5 min, no server

```
PLAIN=$(readlink -f /home/timo/.cache/huggingface/hub/models--bartowski--Qwen_Qwen3-1.7B-GGUF/snapshots/*/Qwen_Qwen3-1.7B-Q8_0.gguf | head -1)
OBF=/home/timo/.cache/huggingface/path-2-aloepri/qwen3-1.7b/keymat-h128-pi-noise-alg2-fp32.gguf

python3 evals/aloepri-attacks/m2_7/run_static_attacks.py \
    --plain "$PLAIN" --obfuscated "$OBF" \
    --output evals/aloepri-attacks/results/m2_7-static.json \
    --vma-eval-size 256 --vma-pool-size 4096 \
    --ia-eval-size 4096 --ia-pool-size 8192
```

For a smoke check first, drop the sizes (e.g.
`--vma-eval-size 32 --vma-pool-size 256`). The smoke produces
non-informative all-zero TTRSR but validates the code paths and
memory budget.

### 2. Token-stream attacks (TFMA, SDA) — server + driver

```
# Spawn dedicated obfuscated llama-server (Docker, port 8061).
evals/aloepri-attacks/m2_7/spawn_obfuscated_server.sh

# Wait for the server to be ready (~10 s warmup):
curl --retry 30 --retry-delay 1 -s http://127.0.0.1:8061/health

# Drive it with the 64-prompt corpus, capture obfuscated streams.
python3 evals/aloepri-attacks/m2_7/capture_token_streams.py \
    --endpoint http://127.0.0.1:8061 \
    --key-path /home/timo/.cache/huggingface/path-2-aloepri/qwen3-1.7b/keymat-h128-pi-noise-alg2-fp32.gguf.key.npz \
    --prompts-file evals/aloepri-attacks/corpora/release-gate-64.txt \
    --max-prompts 64 --max-new-tokens 24 \
    --output evals/aloepri-attacks/snapshots/m2_7-token-streams.jsonl

# Run TFMA + SDA on the captured streams.
python3 evals/aloepri-attacks/m2_7/run_token_attacks.py \
    --captured evals/aloepri-attacks/snapshots/m2_7-token-streams.jsonl \
    --output evals/aloepri-attacks/results/m2_7-token.json

# Tear down — only kill containers we spawned.
docker stop aloepri-m2_7-server
```

For smoke, pass `--smoke` to `capture_token_streams.py` (captures
one prompt and prints the record).

### 3. Hidden-state attacks (NN, IMA, ISA) — patched `llama-server` + file-dump

**Build the patched server image first** (one-time, ~15 min, ~5 GB
Docker storage). The patch lives at
`evals/aloepri-attacks/m2_7/patches/0001-M2.7-tensor-dump-hook-for-llama-server.patch`
and is applied to the `vendor/llama.cpp` submodule working tree
before `docker build`:

```
# One-time: initialise the submodule
git submodule update --init --recursive vendor/llama.cpp

# Apply the M2.7 patch onto the submodule's working tree
bash evals/aloepri-attacks/m2_7/apply-patches.sh

# Build the image (Dockerfile copies the patched source)
docker build \
    -f evals/aloepri-attacks/m2_7/vulkan-m2_7.Dockerfile \
    -t aloepri-llama-server:m2_7 \
    vendor/llama.cpp

# Revert when done — useful before pulling a newer submodule pin
bash evals/aloepri-attacks/m2_7/apply-patches.sh --revert
```

`apply-patches.sh` supports `apply` (default), `--check`
(dry-run), and `--revert` modes. The submodule pointer in the
parent repo stays at the clean upstream commit; the patch only
modifies the working tree.

The patch adds `--tensor-filter REGEX` + `--tensor-dump-path FILE`
flags that write captured tensors to a binary file per forward
pass.

**Operational notes (verified during smoke 2026-05-19):**

* Spawn the container with `--user 1000:1000` (your host uid:gid) — the
  server writes the dump file from inside the container; the Python
  harness needs to truncate it from the host between prompts, which
  only works when the file is owned by the host user.
* The Python harness pulls `tokenizers` via the path-2 `AloePriClient`.
  System Python on Ubuntu 24+ refuses pip installs (PEP 668), so run
  the harness through the path-2 venv:
  `/home/timo/repos/private-rag-path-2/python/path-2/.venv/bin/python`.

**HiddenState pass** (NN / IMA basic / IMA paper-like / ISA
HiddenState — flash-attn on is fine):

```
mkdir -p /tmp/m2_7-dump
docker run --rm -d --name aloepri-m2_7-server \
    --user 1000:1000 \
    -p 127.0.0.1:8061:8080 \
    -v /home/timo/.cache/huggingface/path-2-aloepri/qwen3-1.7b:/models:ro \
    -v /tmp/m2_7-dump:/dump \
    --device /dev/dri \
    aloepri-llama-server:m2_7 \
    -m /models/keymat-h128-pi-noise-alg2-fp32.gguf \
    -ngl 999 -np 1 --flash-attn on -c 4096 \
    --tensor-filter '^attn_norm-(0|11|23)$' \
    --tensor-dump-path /dump/m2_7_dump.bin

curl --retry 30 --retry-delay 1 -s http://127.0.0.1:8061/health

VENV=/home/timo/repos/private-rag-path-2/python/path-2/.venv/bin/python
$VENV evals/aloepri-attacks/m2_7/capture_hidden_states.py \
    --endpoint http://127.0.0.1:8061 \
    --key-path /home/timo/.cache/huggingface/path-2-aloepri/qwen3-1.7b/keymat-h128-pi-noise-alg2-fp32.gguf.key.npz \
    --dump-path /tmp/m2_7-dump/m2_7_dump.bin \
    --prompts-file evals/aloepri-attacks/corpora/release-gate-64.txt \
    --max-prompts 8 --mode hidden \
    --output evals/aloepri-attacks/snapshots/m2_7-hidden/hidden.safetensors

docker stop aloepri-m2_7-server
```

**AttnScore pass** (ISA AttnScore — *flash-attn must be off* so
`kq-23` materialises in the compute graph):

```
docker run --rm -d --name aloepri-m2_7-server \
    --user 1000:1000 \
    -p 127.0.0.1:8061:8080 \
    -v /home/timo/.cache/huggingface/path-2-aloepri/qwen3-1.7b:/models:ro \
    -v /tmp/m2_7-dump:/dump \
    --device /dev/dri \
    aloepri-llama-server:m2_7 \
    -m /models/keymat-h128-pi-noise-alg2-fp32.gguf \
    -ngl 999 -np 1 --flash-attn off -c 4096 \
    --tensor-filter '^kq-23$' \
    --tensor-dump-path /dump/m2_7_dump.bin

curl --retry 30 --retry-delay 1 -s http://127.0.0.1:8061/health

$VENV evals/aloepri-attacks/m2_7/capture_hidden_states.py \
    --endpoint http://127.0.0.1:8061 \
    --key-path /home/timo/.cache/huggingface/path-2-aloepri/qwen3-1.7b/keymat-h128-pi-noise-alg2-fp32.gguf.key.npz \
    --dump-path /tmp/m2_7-dump/m2_7_dump.bin \
    --prompts-file evals/aloepri-attacks/corpora/release-gate-64.txt \
    --max-prompts 8 --mode attn \
    --output evals/aloepri-attacks/snapshots/m2_7-hidden/attn.safetensors

docker stop aloepri-m2_7-server
```

**Run the attacks on the captured snapshots:**

```
python3 evals/aloepri-attacks/m2_7/run_hidden_state_attacks.py \
    --captures-dir evals/aloepri-attacks/snapshots/m2_7-hidden \
    --output evals/aloepri-attacks/results/m2_7-hidden.json \
    --include-attn-score
```

### Patch on top of `vendor/llama.cpp` submodule (Option B file-dump)

`vendor/llama.cpp` is a git submodule pinned at upstream master.
The M2.7 hook lives as a single patch at
`patches/0001-M2.7-tensor-dump-hook-for-llama-server.patch`
(`git format-patch` output, replays cleanly via `git apply` or
`apply-patches.sh`). Bumping the submodule pin: `--revert` first,
update the pin, then `apply-patches.sh` again — refresh the patch
file if there's a conflict.


| File | Diff |
|---|---|
| `common/common.h` | +1 line (`tensor_dump_path` field) |
| `common/debug.h` | +13 lines (`set_dump_path` method) |
| `common/debug.cpp` | +29 lines (file-write block) |
| `common/arg.cpp` | +12 lines (`--tensor-dump-path` flag + extend `--tensor-filter` to LLAMA_EXAMPLE_SERVER) |
| `common/common.cpp` | +28 lines (auto-wire callback when `--tensor-filter` is set) |
| **Total** | **~80 lines, 5 files** |

Same `cb_eval` infrastructure the existing `llama-debug` tool uses
— just extended with binary-file output and registered to
`llama-server` via the existing CLI-flag machinery. No
new task types, route handlers, or compute-graph touches.

## Pre-flight gates

Every M2.7 script that allocates non-trivial memory now does a
`MemAvailable` pre-flight check (lifted from the path-1 post-OOM
safeguards in `docs/prototype/aloepri-attack-harness-findings.md`).
Thresholds are encoded in `m2_7_common.PHASE_MIN_GB`:

| Phase | Pre-flight floor | Expected peak | Notes |
|---|---:|---:|---|
| `static_attacks` (VMA + IA) | 25 GB | 22 GB | both GGUFs dequantised simultaneously |
| `token_capture` | 4 GB | 3 GB | server-side memory; client just holds JSONL |
| `token_attacks` (TFMA + SDA) | 3 GB | 2 GB | bigram matrices + small inverters |
| `hidden_capture` | 4 GB | 3 GB | client parses dump file per prompt; flushes to safetensors |
| `hidden_attacks` (NN/IMA/ISA) | 6 GB | 4 GB | embedding table f32 + captured tensors + inverter |

Each script accepts `--min-mem-gb FLOAT` to override the default and
`--skip-mem-check` to bypass entirely (only after measuring actual
headroom). `capture_hidden_states.py` adds a runtime
`--max-tensors-mb` bounded-buffer check that aborts if a
too-permissive `--tensor-filter` regex on the server side causes
in-memory accumulation to exceed the limit.

Other pre-flight gates before any run:

* `swapon --show` doesn't show heavy swap pressure.
* `docker ps` confirms the `llama-swap` container is **untouched**
  (we don't kill containers we didn't spawn).
* The §05 GGUF + `.key.npz` exist at the paths above (rewriter has
  run).
* Port 8061 is free (`ss -tlnp | grep 8061` empty) before spawning
  the M2.7 server.
* For hidden-state attacks: docker volume `/tmp/m2_7-dump` exists
  and is writable; the server writes there, the Python harness reads.

## After running

Results JSONs land in `evals/aloepri-attacks/results/m2_7-*.json`.
Update `docs/prototype/aloepri-llm.html` §08 "Obfuscated" column
with the measured numbers and toggle the relevant `*_obfuscated_*`
acceptance gate rows from "pending" to ✓/✗.

## Why this is a separate harness from path-1

Path-1 (the `capture_snapshots` Rust binary) hooks into the Rust
`InProcessTrustedExecutor` and records per-offload activations
under a per-batch Haar-mask + shield-rows defence. The §05
artifact uses a different defence (keymat + Π + α-noise +
partial Algorithm 2) baked into the GGUF, served by an
unmodified `llama-server`. The two share the *attack code*
(Python drivers under `attack_drivers/`) but need different
capture frontends.

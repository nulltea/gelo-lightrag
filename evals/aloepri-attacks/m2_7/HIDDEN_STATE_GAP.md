# M2.7 — hidden-state capture: deferred (Hard effort)

The four attacks that need intermediate-layer hidden states from the
obfuscated forward pass are NOT yet implemented for M2.7:

| Attack | Observable needed | Required hook |
|---|---|---|
| **NN** | hidden state at any early layer | last-layer pooled via `/embeddings` *or* intermediate layer |
| **IMA basic** | hidden state at layer 0 (q_proj input) | intermediate layer |
| **IMA paper-like** | hidden state at layer 0 + ≥ 256 prompts | intermediate layer + training corpus |
| **ISA** | hidden state at layer 23 (paper default) | intermediate layer |

`llama.cpp:server-vulkan` (the §05 deployment target) exposes:

* `/completion` — final tokens (used by TFMA / SDA)
* `/embeddings` — **last-layer** pooled embedding (single vector per
  prompt; suitable for NN at the output projection only, not at
  layer 0 or any intermediate layer)

It does NOT expose per-layer hidden states. To fill the table for
NN / IMA / ISA we need one of three frontends:

## Option A — patch `llama.cpp` to expose intermediate hidden states

Add an HTTP endpoint that runs the obfuscated forward pass and
returns the hidden state at a requested layer index. Several
existing forks (`pixeli99/llama.cpp`, `mostlygeek/llama.cpp`) carry
similar hooks for embedding-extraction at a non-final layer.

**Effort:** medium-hard. Touches llama.cpp C++ source and rebuilds
the Docker image. Operates against the §05 design goal of "no
infra change" but is acceptable for the eval-only path.

## Option B — `llama-cpp-python` with `embedding=True` + layer hook

The Python bindings have an `Llama` class that loads a GGUF
directly and supports `Llama.create_embedding()` plus access to
internal state. Per-layer hidden states require setting
`embedding=True` at construction and using the `state` extraction
API; some versions expose layer indices, some don't.

**Effort:** medium. Requires installing `llama-cpp-python` with
Vulkan support (compiled with `CMAKE_ARGS="-DLLAMA_VULKAN=on"`).
Tooling on the dev box: `pip install llama-cpp-python` would work
but won't pick up the iGPU automatically — needs the right CMake
flags.

## Option C — GGUF → PyTorch and run forward in `transformers` / native code

Convert the obfuscated GGUF back to a PyTorch state dict (vendored
`gguf` library has the tensor readers we already use in
`extract_gguf_weights.py`), reconstruct the obfuscated forward
pass — including the partial-Algorithm-2 attention path the §05
artifact uses — in `transformers` or our own Rust code, attach
PyTorch forward-hooks at the layers we want, capture activations.

**Effort:** hard. Requires:

1. A model class that knows how to consume the d_eff = 2304 (post-
   expansion) hidden size, the keymat-rewritten attention block,
   and the head-shuffled τ_kv / τ_group permutations.
2. Recreating §5.2.5's κ-fold inference path (the existing
   plaintext `decoder` Rust code doesn't deploy that).
3. Either a fp32 PyTorch port or a Rust-side runtime able to
   serve the obfuscated artifact.

This is the cleanest path to per-layer access but is essentially
"build a second inference frontend for the obfuscated artifact". The
existing `obfuscate_qwen3_gguf.py` rewriter handles the offline
side; an obfuscated-side `infer_qwen3_obfuscated.py` would be its
runtime sibling.

## Recommendation for the immediate next step

Once the static-weight + token-stream M2.7 numbers land:

1. **Decide whether the ISA AttnScore watchpoint really needs the
   M2.7 hidden-state capture**, or whether AloePri's published Table 3
   ablation numbers (Qwen2.5-14B) + our partial-Algorithm-2 reasoning
   are sufficient to make the §09 "switch backbone / κ-tune / patch"
   decision without a Qwen3 1.7B measurement.
2. **If hidden-state capture IS needed**, pursue Option A (smallest
   blast radius — single endpoint addition to llama.cpp, no
   inference-frontend duplication) and re-pin the §05 Docker image
   to the patched build.

The capture script `capture_hidden_states.py` and per-attack
drivers (`run_nn.py`, `run_ima.py`, `run_isa.py`) already exist
in `attack_drivers/`; they just need a snapshot frontend that
talks to whichever Option lands.

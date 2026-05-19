# private-rag

Rust workspace for the GeloRAG prototype in `docs/private-rag-system-design.md`.

Current focus:
- vector DB storage encryption
- SAP port as the first concrete scheme
- CAPRISE crate boundaries and algorithm scaffolding
- simple encrypted retrieval over local embeddings

Workspace layout:
- `crates/core`: shared domain types, crypto, embedding adapter, and in-memory encrypted storage
- `crates/gelo-rag`: orchestration layer standing in for the future embedding TEE boundary

Commands:

```bash
cargo test
cargo test -- --ignored
```

The ignored integration test downloads a small embedding model through `fastembed`.

## Path-2 — AloePri attack harness (M2.7)

`evals/aloepri-attacks/` houses the AloePri attack-resistance harness
(eight attacks ported from `vendor/aloepri-py/src/security_qwen/`).
M2.7 captures observables from an obfuscated Qwen3 1.7B GGUF served
by a patched `llama-server` and runs the attacks against them. See
`docs/prototype/aloepri-llm.html` §08 for measured results and
`evals/aloepri-attacks/m2_7/README.md` for the operator runbook.

The patched `llama-server` is built from the `vendor/llama.cpp`
submodule, which is pinned at
[github.com/nulltea/llama.cpp](https://github.com/nulltea/llama.cpp)
branch `m2_7-tensor-dump` (= upstream `ggml-org/llama.cpp` master +
one commit adding `--tensor-filter REGEX` and `--tensor-dump-path
FILE`). Build:

```bash
git submodule update --init --recursive vendor/llama.cpp
docker build \
    -f evals/aloepri-attacks/m2_7/vulkan-m2_7.Dockerfile \
    -t aloepri-llama-server:m2_7 \
    vendor/llama.cpp
```

Fresh clones get the patched source directly; there is no manual
patch-apply step. Rebase recipe for bumping the fork onto newer
upstream is in `evals/aloepri-attacks/m2_7/README.md`.


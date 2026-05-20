# private-rag

Rust workspace for a working **retrieval-only private RAG** stack: a
regulated client outsources a confidential corpus to an untrusted
cloud, queries it interactively, and gets ranked results back —
without the cloud seeing plaintext text, plaintext embeddings, or
which document matched.

The stack composes an attested SEV-SNP CVM, a per-forward
orthogonal-mask split-inference protocol (GELO) against a commodity
Vulkan GPU, and distance-preserving ciphertext (CAPRISE) at rest.
Extends end-to-end to private LightRAG retrieval over Ring-ORAM +
XorMM volume-hiding multimaps.

Design and measurement docs: **<https://nulltea.github.io/rag-privacy/>**

## Workspace layout

| crate | role |
|---|---|
| `crates/core` | shared types, CAPRISE / AES-GCM, two-party HKDF, in-memory encrypted index |
| `crates/gelo-protocol` | `TrustedExecutor` / `GpuOffloadEngine` traits + per-forward mask · shield · U-Verify · HD₃ |
| `crates/gelo-gpu-wgpu` | Vulkan offload engine (`WgpuVulkanEngine`) |
| `crates/gelo-embedder` | masked BERT / Qwen3 embedder + decoder substrate |
| `crates/gelo-reranker` | cross-encoder & causal-LM-discriminator reranker under the same mask |
| `crates/gelo-rag` | orchestration · `GeloRagTwoPartyService` · `LightRagTwoPartyService` |
| `crates/gelo-tee-sev-snp` | SEV-SNP attestation issuer + verifier · RATLS plumbing |
| `crates/gelo-snp-runner` | axum service binding all of the above |
| `crates/ring-oram` | Ring-ORAM semi-honest baseline + treetop cache |
| `crates/compass-index` | Compass over Ring-ORAM + Directional Filter (HNSW-ORAM) |
| `crates/xormm-emm` | XorMM volume-hiding encrypted multi-map |
| `crates/compass-rest-backend` | axum + sled untrusted storage server for Compass |
| `crates/light-kg-store` | LightRAG-shaped storage facade · 3× CompassIndex + 2× XorMM + AES chunks |
| `crates/lightrag-private` | Rust port of LightRAG `kg_query` (Local + Hybrid modes) |
| `crates/graphrag-bench` | end-to-end stage-timed bench harness |
| `evals/aloepri-attacks` | static-weight obfuscation attack-resistance harness (see below) |

## Commands

```bash
cargo test
cargo test -- --ignored        # downloads embedding model via fastembed
```

## AloePri attack harness

`evals/aloepri-attacks/` ports the AloePri attack suite from
`vendor/aloepri-py/src/security_qwen/` and runs it against an
obfuscated Qwen3-1.7B GGUF served by a patched `llama-server`.
Captures observables, runs prompt-inversion attacks against them.
Measured results in
[`docs/prototype/aloepri-llm.html`](https://nulltea.github.io/rag-privacy/aloepri-llm.html);
operator runbook in `evals/aloepri-attacks/README.md`.

The patched `llama-server` builds from the `vendor/llama.cpp`
submodule, pinned at
[github.com/nulltea/llama.cpp](https://github.com/nulltea/llama.cpp)
branch `m2_7-tensor-dump` (upstream `ggml-org/llama.cpp` master + one
commit adding `--tensor-filter REGEX` and `--tensor-dump-path FILE`):

```bash
git submodule update --init --recursive vendor/llama.cpp
docker build \
    -f evals/aloepri-attacks/m2_7/vulkan-m2_7.Dockerfile \
    -t aloepri-llama-server:latest \
    vendor/llama.cpp
```

Fresh clones get the patched source directly — no manual patch-apply
step. Rebase recipe for bumping the fork onto newer upstream is in the
eval's README.

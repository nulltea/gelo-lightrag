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


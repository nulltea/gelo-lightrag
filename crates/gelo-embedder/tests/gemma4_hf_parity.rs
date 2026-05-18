//! M1.8 scaffolding — HuggingFace `transformers` parity gate.
//!
//! Implements the M1.8 accuracy validation acceptance criterion in
//! its smoke-test shape: greedy `generate()` at `temperature=0` on a
//! pinned prompt must match the HF reference output **token-for-token**
//! (top-1 token match ≥ 0.99 by the milestone's wording; for greedy
//! single-shot this collapses to "equal").
//!
//! Pattern: a sidecar JSON file at
//! `crates/gelo-embedder/tests/fixtures/gemma4_e2b_hf_reference.json`
//! holds the HF reference output for a specific model_id + prompt +
//! sampler config (greedy, temperature=0). The test loads the fixture,
//! runs our stack, compares tokens.
//!
//! **Generating the fixture (M1.8 worker handoff):**
//!
//! ```python
//! from transformers import AutoTokenizer, AutoModelForCausalLM
//! import torch, json, hashlib
//!
//! MODEL = "google/gemma-4-e2b"  # pin actual ID once published
//! tok = AutoTokenizer.from_pretrained(MODEL)
//! model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
//! prompt = "The transformer architecture revolutionised"
//! ids = tok(prompt, return_tensors="pt").input_ids
//! with torch.no_grad():
//!     out = model.generate(
//!         ids, max_new_tokens=32, do_sample=False, temperature=0.0,
//!     )
//! generated = out[0, ids.shape[1]:].tolist()
//! sha = hashlib.sha256(json.dumps({"prompt": prompt, "tokens": generated}).encode()).hexdigest()
//! fixture = {
//!     "model_id": MODEL,
//!     "transformers_version": __import__("transformers").__version__,
//!     "prompt": prompt,
//!     "prompt_ids": ids[0].tolist(),
//!     "reference_tokens": generated,
//!     "sampler": {"kind": "greedy", "temperature": 0.0},
//!     "fixture_sha256": sha,
//! }
//! with open("fixtures/gemma4_e2b_hf_reference.json", "w") as f:
//!     json.dump(fixture, f, indent=2)
//! ```
//!
//! **Blockers (un-ignore prerequisites):**
//!  - All M1.6 prerequisites (loader, model_id, GPU access).
//!  - The fixture JSON must exist at the path above. Until the M1.8
//!    worker generates it, this test ignores itself.
//!
//! Per `docs/plans/path-1-gelo-gemma.md` M1.8: HF `transformers` is the
//! only accept-gate baseline; vLLM and llama.cpp are informational
//! only.

use anyhow::Result;

/// Returns the fixture path. Centralised so the doc comment above and
/// the test body stay in sync.
fn fixture_path() -> std::path::PathBuf {
    let manifest = std::env!("CARGO_MANIFEST_DIR");
    std::path::Path::new(manifest)
        .join("tests")
        .join("fixtures")
        .join("gemma4_e2b_hf_reference.json")
}

#[test]
#[ignore = "M1.8 scaffolding — gated on M1.6 prerequisites AND fixture JSON existing"]
fn gemma4_e2b_greedy_matches_hf_transformers() -> Result<()> {
    let path = fixture_path();
    if !path.exists() {
        return Err(anyhow::anyhow!(
            "M1.8 fixture missing at {} — see file-level docstring for the \
             one-shot Python snippet that produces it",
            path.display(),
        ));
    }

    // 1. Load fixture (prompt_ids, reference_tokens, sampler config).
    // 2. Build E2B config + weights (same loader as M1.6 e2e test).
    // 3. Run greedy generate with the fixture's prompt_ids and
    //    `max_new_tokens = reference_tokens.len()`.
    // 4. assert_eq!(our_tokens, reference_tokens). The milestone gate
    //    is top-1 match ≥ 0.99; for greedy single-shot this is
    //    bit-equality across all positions.
    // 5. Record `model_identity` (SHA-256 from `DecoderWeights`) and
    //    `transformers_version` in `results/path-1-accuracy.json` per
    //    the M1.8 acceptance criterion. Same format the bench harness
    //    uses; the M1.8 worker wires it via `evals/run-eval.py`.

    Err(anyhow::anyhow!(
        "M1.8 fixture exists but Gemma 4 loader not yet wired — see M1.6"
    ))
}

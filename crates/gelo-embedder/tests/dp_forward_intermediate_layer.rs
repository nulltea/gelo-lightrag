//! M7.1 — functional smoke test for DP-Forward applied at an intermediate
//! transformer layer (the `add_and_norm_2` position from the
//! `xiangyue9607/DP-Forward` paper, `noise_layer = 10` on BERT-base).
//!
//! Verifies:
//! - `with_dp_forward(layer_index = Some(10))` does not panic during embed.
//! - Two consecutive calls produce *different* embeddings (DP RNG is
//!   `OsRng`-seeded, so noise must be non-deterministic across calls).
//! - The pooled embedding is still finite (no NaN/Inf from upstream
//!   layers reacting badly to mid-stream noise).
//!
//! The "does retrieval utility recover" question is M7.3's job — the
//! BEIR/NFCorpus bench measures that against a real IR baseline. This
//! test just guarantees the forward-pass plumbing is correct.

#![cfg(feature = "dp-forward")]

use dp_forward::DpForwardConfig;
use gelo_embedder::GeloBertEmbedder;
use gelo_protocol::rng::MaskSeed;
use gelo_protocol::{InProcessTrustedExecutor, RayonCpuEngine};
use rag_core::Embedder;

const MODEL: &str = "BAAI/bge-small-en-v1.5";

fn make_exec() -> InProcessTrustedExecutor<RayonCpuEngine> {
    InProcessTrustedExecutor::with_seed(RayonCpuEngine::new(), MaskSeed::from_bytes([7u8; 32]))
}

#[test]
#[ignore = "downloads bge-small (~130 MB) on first run"]
fn intermediate_layer_dp_does_not_panic_and_produces_non_deterministic_output() {
    let cfg = DpForwardConfig::calibrate(4.0, 1e-5, 1.0).with_layer_index(Some(10));
    let mut embedder = GeloBertEmbedder::from_pretrained(MODEL, make_exec())
        .expect("download BGE-small")
        .with_dp_forward(cfg);

    let texts = vec!["Rust enforces memory safety through ownership.".to_string()];

    let v1 = embedder.embed(&texts).expect("embed call 1");
    let v2 = embedder.embed(&texts).expect("embed call 2");

    // Shape is preserved.
    assert_eq!(v1.len(), 1);
    assert_eq!(v2.len(), 1);
    assert_eq!(v1[0].len(), v2[0].len());

    // Finite values everywhere — noise didn't blow up the forward pass.
    for x in v1[0].iter().chain(v2[0].iter()) {
        assert!(
            x.is_finite(),
            "DP-on intermediate-layer embedding contains NaN/Inf: {x}"
        );
    }

    // Two calls produced different embeddings — DP RNG is
    // non-deterministic across calls. (Without DP this would be exactly
    // equal to f32 round-off; with DP it differs by an amount commensurate
    // with the noise budget.)
    let max_abs: f32 = v1[0]
        .iter()
        .zip(v2[0].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs > 1e-4,
        "two DP-on embeds of same input should differ; max_abs = {max_abs}"
    );
}

#[test]
#[ignore = "downloads bge-small (~130 MB) on first run"]
fn intermediate_layer_vs_pooled_output_distinct_paths() {
    // Build two embedders: one with intermediate-layer DP, one with
    // pooled-output DP. Both should produce valid (finite, well-shaped)
    // embeddings — confirms the two code paths are independently wired
    // up. We do NOT assert relative retrieval quality here; that's M7.3.
    let cfg_intermediate =
        DpForwardConfig::calibrate(4.0, 1e-5, 1.0).with_layer_index(Some(10));
    let cfg_pooled = DpForwardConfig::calibrate(4.0, 1e-5, 1.0); // layer_index = None

    let mut emb_intermediate = GeloBertEmbedder::from_pretrained(MODEL, make_exec())
        .expect("download BGE-small (intermediate)")
        .with_dp_forward(cfg_intermediate);
    let mut emb_pooled = GeloBertEmbedder::from_pretrained(MODEL, make_exec())
        .expect("download BGE-small (pooled)")
        .with_dp_forward(cfg_pooled);

    let texts = vec!["Test embedding through both DP paths.".to_string()];
    let v_intermediate = emb_intermediate.embed(&texts).expect("intermediate embed");
    let v_pooled = emb_pooled.embed(&texts).expect("pooled embed");

    assert_eq!(v_intermediate[0].len(), v_pooled[0].len());
    for x in v_intermediate[0].iter() {
        assert!(x.is_finite(), "intermediate-layer DP produced NaN/Inf");
    }
    for x in v_pooled[0].iter() {
        assert!(x.is_finite(), "pooled-output DP produced NaN/Inf");
    }
}

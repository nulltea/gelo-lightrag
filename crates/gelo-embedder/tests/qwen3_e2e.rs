//! End-to-end test of `GeloQwenEmbedder` against `Qwen/Qwen3-Embedding-0.6B`.
//!
//! Asserts masked vs plaintext executor parity on the pooled embedding.
//! Forces OutAttnMult on at any sequence length on both sides
//! (`out_attn_mult_min_seq_len = Some(0)`) so the protected attention
//! path keeps regression coverage even though the auto-switch would
//! normally route short bench prompts to in-TEE attention.
//!
//! Downloads ~1.2 GB on first run; gated behind `#[ignore]`.

use gelo_embedder::GeloQwenEmbedder;
use gelo_protocol::rng::MaskSeed;
use gelo_protocol::{InProcessTrustedExecutor, PlaintextExecutor, ReferenceCpuEngine};
use rag_core::Embedder;

const MODEL: &str = "Qwen/Qwen3-Embedding-0.6B";

#[test]
#[ignore = "downloads ~1.2 GB on first run"]
fn qwen3_decoder_parity() {
    let mut cpu_plain = GeloQwenEmbedder::from_pretrained(
        MODEL,
        PlaintextExecutor::new(ReferenceCpuEngine::new()),
    )
    .expect("load Qwen3-Embedding-0.6B (plaintext)")
    .with_out_attn_mult(true)
    .with_out_attn_mult_min_seq_len(Some(0));

    let mut cpu_masked = GeloQwenEmbedder::from_pretrained(
        MODEL,
        InProcessTrustedExecutor::with_seed(ReferenceCpuEngine::new(), MaskSeed::from_bytes([29u8; 32])),
    )
    .expect("load Qwen3-Embedding-0.6B (masked)")
    .with_out_attn_mult(true)
    .with_out_attn_mult_min_seq_len(Some(0));

    let texts = vec![
        "Confidential computing protects user data inside attested enclaves.".to_string(),
        "Rotary position embeddings rotate query and key vectors per token.".to_string(),
    ];

    let plain = cpu_plain.embed(&texts).unwrap();
    let masked = cpu_masked.embed(&texts).unwrap();

    assert_eq!(plain.len(), masked.len());
    for (p, m) in plain.iter().zip(masked.iter()) {
        assert_eq!(p.len(), m.len());
        let max_abs = p
            .iter()
            .zip(m.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        // Decoder is deeper (28 layers vs BERT's 12) so f32 roundoff compounds;
        // a few-mil tolerance is generous but still <<1% of the unit-norm
        // embedding scale.
        assert!(max_abs < 1e-2, "Qwen3 masked vs plaintext: max abs {max_abs}");
    }
}

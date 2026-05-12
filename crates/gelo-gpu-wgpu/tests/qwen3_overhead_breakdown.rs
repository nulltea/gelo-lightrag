//! Per-step breakdown of where the `gpu + GELO + OutAttnMult` overhead
//! goes on `Qwen/Qwen3-Embedding-0.6B`, with the **mock SEV-SNP TEE
//! boundary** wired in around the GELO executor.
//!
//! Runs the unprotected (gpu_plain) and protected
//! (`SnpTrustedExecutor<… , MockReportIssuer>` wrapping the GELO
//! `InProcessTrustedExecutor`) configurations on the same shared weights,
//! snapshotting the thread-local `gelo_protocol::profile` accumulator
//! between runs. Categories:
//!
//! - `tee:*` — pure TEE-side work (RMSNorm, RoPE, softmax+AV, SwiGLU
//!   activation, residual adds, attention)
//! - `engine:*` — wgpu/Vulkan dispatches (matmul / matmul_dynamic)
//! - `gelo:*` — GELO mask machinery (sample A, apply A·H, unapply Aᵀ·V,
//!   shield stack, shield strip)
//! - `outattn:*` — TwinShield OutAttnMult bookkeeping (setup + stack
//!   2n-wide masked operands, recover Q·Kᵀ from the four partitions)
//!
//! Before measuring, the bench issues one **session attestation**
//! through the mock SEV-SNP path (`SnpTrustedExecutor::evidence()`),
//! verifies it round-trips through `SnpAttestationVerifier`, and prints
//! its wall-clock cost. This mirrors what a real deployment does once
//! per session — the steady-state embed loop happens *after* attestation
//! and shouldn't pick up its cost.
//!
//! Downloads ~1.2 GB on first run; gated behind `#[ignore]`.

use std::sync::Arc;

use gelo_embedder::decoder::config::DecoderConfig;
use gelo_embedder::decoder::rope::RopeTables;
use gelo_embedder::decoder::weights::DecoderWeights;
use gelo_embedder::GeloQwenEmbedder;
use gelo_gpu_wgpu::WgpuVulkanEngine;
use gelo_protocol::profile;
use gelo_protocol::rng::MaskSeed;
use gelo_protocol::{InProcessTrustedExecutor, PlaintextExecutor};
use gelo_tee_sev_snp::{
    AttestedBinding, SnpAttestationVerifier, SnpRootTrust, SnpTrustedExecutor,
    mock::MockReportIssuer,
};
use rag_core::Embedder;

const MODEL: &str = "Qwen/Qwen3-Embedding-0.6B";

fn corpus() -> Vec<String> {
    vec![
        "Confidential computing protects user data inside attested enclaves.".into(),
        "Rotary position embeddings rotate query and key vectors per token.".into(),
        "TwinShield's OutAttnMult outsources the attention QK^T matmul to the GPU.".into(),
    ]
}

fn warmup_then_capture(
    embedder: &mut dyn Embedder,
    texts: &[String],
) -> (profile::Profile, std::time::Duration) {
    // One throw-away pass to settle GPU autotune + first-touch caches.
    let _ = embedder.embed(texts).unwrap();
    profile::reset();
    let t0 = std::time::Instant::now();
    let _ = embedder.embed(texts).unwrap();
    let wall = t0.elapsed();
    (profile::snapshot(), wall)
}

#[test]
#[ignore = "downloads ~1.2 GB Qwen3-Embedding-0.6B; requires Vulkan GPU"]
fn qwen3_overhead_step_breakdown() {
    let texts = corpus();

    eprintln!("[load] downloading + materialising Qwen3 weights...");
    let cpu_seed = GeloQwenEmbedder::from_pretrained(
        MODEL,
        PlaintextExecutor::new(gelo_protocol::RayonCpuEngine::new()),
    )
    .expect("Qwen3 from_pretrained")
    .with_out_attn_mult(false);
    let weights_arc: Arc<DecoderWeights> = cpu_seed.weights_arc();
    let rope_arc: Arc<RopeTables> = cpu_seed.rope_arc();
    let tokenizer = cpu_seed.tokenizer().clone();
    let cfg: DecoderConfig = cpu_seed.config().clone();

    let gpu_root = WgpuVulkanEngine::new().expect("Vulkan adapter");
    let adapter_line = format!(
        "{} ({:?}, driver={}, info={})",
        gpu_root.adapter_info().name,
        gpu_root.adapter_info().device_type,
        gpu_root.adapter_info().driver,
        gpu_root.adapter_info().driver_info,
    );
    assert!(gpu_root.is_real_gpu());

    // Build the two embedders we want to compare against each other.
    let mut gpu_plain = GeloQwenEmbedder::new(
        cfg.clone(),
        tokenizer.clone(),
        Arc::clone(&weights_arc),
        Arc::clone(&rope_arc),
        PlaintextExecutor::new(gpu_root.clone_shared()),
    )
    .expect("gpu_plain")
    .with_out_attn_mult(false);

    // Protected path: GELO mask + shield + OutAttnMult, wrapped in the
    // mock SEV-SNP boundary. The wrapper forwards every TrustedExecutor
    // method to its inner `InProcessTrustedExecutor`, so the per-text
    // overhead is unchanged — what we get is a real attestation surface
    // we can exercise once at session start.
    const MODEL_IDENT: &[u8] = b"Qwen/Qwen3-Embedding-0.6B@bench";
    const SCHEME_IDENT: &[u8] = b"gelo+twinshield@v1";
    let mut gpu_gelo_outattn = {
        let mut c = cfg.clone();
        c.use_out_attn_mult = true;
        let inner = InProcessTrustedExecutor::with_seed(
            gpu_root.clone_shared(),
            MaskSeed::from_bytes([1u8; 32]),
        );
        let issuer = MockReportIssuer::from_bundled().expect("load bundled mock VCEK key");
        let snp = SnpTrustedExecutor::new(inner, issuer, MODEL_IDENT, SCHEME_IDENT);
        GeloQwenEmbedder::new(
            c,
            tokenizer.clone(),
            Arc::clone(&weights_arc),
            Arc::clone(&rope_arc),
            snp,
        )
        .expect("gpu_gelo_outattn (snp-wrapped)")
    };

    eprintln!("Vulkan adapter: {adapter_line}");
    eprintln!("Corpus: {} texts.", texts.len());

    // ── Session attestation through the mock SEV-SNP path ──
    // Done once *before* the warmup-and-measure cycle so the steady-
    // state embed numbers below don't include attestation cost.
    {
        // The embedder consumed the executor by value, but the snp
        // wrapper's identity strings + the same MockReportIssuer can be
        // re-instantiated for the round-trip — issuer state is just the
        // bundled signing key, no per-instance state to leak.
        let issuer = MockReportIssuer::from_bundled().unwrap();
        let snp_for_attest = SnpTrustedExecutor::new(
            InProcessTrustedExecutor::with_seed(
                gelo_protocol::RayonCpuEngine::new(),
                MaskSeed::from_bytes([1u8; 32]),
            ),
            issuer,
            MODEL_IDENT,
            SCHEME_IDENT,
        );
        let t0 = std::time::Instant::now();
        let evidence = snp_for_attest.evidence(Some(b"qwen3-bench-session")).unwrap();
        let issue_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let verifier = SnpAttestationVerifier::new(SnpRootTrust::with_mock_root());
        let t1 = std::time::Instant::now();
        verifier
            .verify(
                &evidence.report_bytes,
                &evidence.vcek_cert_pem,
                AttestedBinding {
                    model_identity: MODEL_IDENT,
                    scheme_identity: SCHEME_IDENT,
                    nonce: Some(b"qwen3-bench-session"),
                },
            )
            .expect("session attestation must round-trip through SnpAttestationVerifier");
        let verify_ms = t1.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[attest] mock SEV-SNP issue={issue_ms:.2} ms, verify={verify_ms:.2} ms, \
             report={} B, vcek_pem={} B",
            evidence.report_bytes.len(),
            evidence.vcek_cert_pem.len(),
        );
    }

    let (plain_profile, plain_wall) = warmup_then_capture(&mut gpu_plain, &texts);
    plain_profile.dump("gpu_plain (no privacy)");
    eprintln!(
        "→ wall-clock: {:.2} ms total, {:.2} ms/text",
        plain_wall.as_secs_f64() * 1000.0,
        plain_wall.as_secs_f64() * 1000.0 / texts.len() as f64,
    );

    let (gelo_profile, gelo_wall) = warmup_then_capture(&mut gpu_gelo_outattn, &texts);
    gelo_profile.dump("gpu + GELO + OutAttnMult + mock SEV-SNP boundary");
    eprintln!(
        "→ wall-clock: {:.2} ms total, {:.2} ms/text",
        gelo_wall.as_secs_f64() * 1000.0,
        gelo_wall.as_secs_f64() * 1000.0 / texts.len() as f64,
    );

    // Per-category delta table (protected − plain), sorted by absolute
    // overhead contribution. Negative values land naturally for any
    // category whose work happens in the unprotected path but is replaced
    // by a different code path in the protected one.
    let mut delta: std::collections::BTreeMap<&'static str, f64> = Default::default();
    for (name, (d, _)) in &gelo_profile.buckets {
        *delta.entry(name).or_default() += d.as_secs_f64() * 1000.0;
    }
    for (name, (d, _)) in &plain_profile.buckets {
        *delta.entry(name).or_default() -= d.as_secs_f64() * 1000.0;
    }

    let overhead_ms =
        gelo_wall.as_secs_f64() * 1000.0 - plain_wall.as_secs_f64() * 1000.0;
    eprintln!();
    eprintln!("=== overhead delta (protected − plain) ===");
    eprintln!(
        "{:<32} {:>10} {:>10}",
        "category", "Δ time (ms)", "share of Δ"
    );
    eprintln!("{}", "-".repeat(56));
    let mut rows: Vec<_> = delta.iter().collect();
    rows.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    let abs_total: f64 = delta.values().map(|d| d.abs()).sum();
    for (name, d) in &rows {
        let share = if abs_total > 0.0 { 100.0 * d.abs() / abs_total } else { 0.0 };
        eprintln!("{:<32} {:>+10.2} {:>9.1}%", name, d, share);
    }
    eprintln!("{}", "-".repeat(56));
    eprintln!(
        "{:<32} {:>+10.2}    ({:+.1}% over baseline)",
        "WALL-CLOCK Δ",
        overhead_ms,
        100.0 * overhead_ms / plain_wall.as_secs_f64() / 1000.0,
    );
}

//! End-to-end: run `bge-small-en-v1.5` with the wgpu Vulkan engine.
//!
//! Confirms that a real BERT forward pass driven through the GELO protocol
//! and offloaded to a Vulkan GPU produces the same embedding as the CPU
//! engine. Gated behind `#[ignore]` because it downloads ~130 MB on first run.

use gelo_embedder::GeloBertEmbedder;
use gelo_gpu_wgpu::WgpuVulkanEngine;
use gelo_protocol::rng::MaskSeed;
use gelo_protocol::{InProcessTrustedExecutor, RayonCpuEngine};
use rag_core::Embedder;

#[test]
#[ignore = "downloads bge-small (~130 MB) from Hugging Face on first run; requires Vulkan"]
fn bge_small_wgpu_matches_cpu() {
    let gpu = match WgpuVulkanEngine::new() {
        Ok(g) => {
            eprintln!("wgpu backend: {}", g.backend());
            g
        }
        Err(err) => {
            eprintln!("skipping wgpu e2e: no Vulkan adapter ({err})");
            return;
        }
    };

    let cpu_exec =
        InProcessTrustedExecutor::with_seed(RayonCpuEngine::new(), MaskSeed::from_bytes([5u8; 32]));
    let gpu_exec = InProcessTrustedExecutor::with_seed(gpu, MaskSeed::from_bytes([5u8; 32]));

    let mut cpu_embedder = GeloBertEmbedder::from_pretrained("BAAI/bge-small-en-v1.5", cpu_exec)
        .expect("bge-small via cpu engine");
    let mut gpu_embedder = GeloBertEmbedder::from_pretrained("BAAI/bge-small-en-v1.5", gpu_exec)
        .expect("bge-small via wgpu engine");

    let texts = vec![
        "Confidential computing protects user data.".to_string(),
        "Vulkan exposes the GPU compute pipeline.".to_string(),
    ];
    let cpu = cpu_embedder.embed(&texts).unwrap();
    let gpu = gpu_embedder.embed(&texts).unwrap();

    assert_eq!(cpu.len(), gpu.len());
    for (c, g) in cpu.iter().zip(gpu.iter()) {
        assert_eq!(c.len(), g.len());
        let max_abs = c
            .iter()
            .zip(g.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_abs < 5e-3, "bge-small diverges on wgpu: max abs {max_abs}");
    }
}

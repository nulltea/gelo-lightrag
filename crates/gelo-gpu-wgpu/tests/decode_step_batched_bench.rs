//! **M1.11 D1.6** — decode-step microbench: serial × B vs batched at B.
//!
//! Strips the comparison down to the decode loop itself — no
//! tokenisation, no prefill, no LM-head sampling. Synthetic Qwen3-4B-
//! shaped weights (4 layers; per-layer dims pinned to the real Q4B
//! config) primed with a `n_kv = 64` random KV cache prefix per
//! sequence. We then time `K = 8` decode steps each way:
//!
//! - **Serial**: B independent executors, B independent
//!   single-sequence KV caches; for each `step in 0..K`, loop
//!   `for b in 0..B { run_decode_step(...) }`. This is the
//!   "no batching at all" baseline per
//!   `feedback_perf_baseline_unbatched_not_rayon`.
//! - **Batched**: one executor, one batched KV cache at B=B; loop
//!   `for step in 0..K { run_decode_step_batched(&[token; B], ...) }`.
//!
//! Both paths are warm-up'd to amortise first-launch JIT compile.
//! Per-op profile breakdowns dump on stderr so we can see where the
//! decode wall actually goes — mask sample, mask apply/unapply,
//! engine matmul/matmul_many, in-TEE attention.
//!
//! Default `BATCHED_DECODE_SHARED_A` is OFF — bench measures the
//! per-sequence A_b decode topology. Set to `1` to compare the
//! shared-dense-A path.
//!
//! Invoke:
//!
//! ```text
//! cargo test -p gelo-gpu-wgpu --release --test decode_step_batched_bench \
//!     -- --ignored --nocapture
//! ```

use std::sync::Arc;
use std::time::Instant;

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

use gelo_embedder::decoder::config::DecoderConfig;
use gelo_embedder::decoder::forward;
use gelo_embedder::decoder::kv_cache::KvCache;
use gelo_embedder::decoder::rope::RopeTables;
use gelo_embedder::decoder::weights::{DecoderLayerWeights, DecoderWeights};
use gelo_gpu_wgpu::WgpuVulkanEngine;
use gelo_protocol::rng::MaskSeed;
use gelo_protocol::{
    GpuOffloadEngine, InProcessTrustedExecutor, WeightHandle, WeightKind,
};

/// Qwen3-4B per-layer dims with `num_layers` (4 for the microbench).
fn qwen3_4b_shape_cfg(num_layers: usize) -> DecoderConfig {
    DecoderConfig {
        vocab_size: 64, // synth small vocab — we don't sample real tokens
        hidden_size: 2560,
        intermediate_size: 9728,
        num_hidden_layers: num_layers,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        head_dim: Some(128),
        max_position_embeddings: 512,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        hidden_act: "silu".into(),
        tie_word_embeddings: true,
        max_seq_len: 512,
        skip_first_layers: 0,
        skip_last_layer: false,
        use_out_attn_mult: false,
        out_attn_mult_min_seq_len: None,
        use_perm_attention: false,
        perm_attention_min_seq_len: None,
        attention_classes: None,
        partial_rope: None,
        kv_shared_in_global: false,
        final_logit_softcapping: None,
    }
}

fn rand2_bf16(rows: usize, cols: usize, rng: &mut impl rand::RngCore) -> Array2<half::bf16> {
    let normal = StandardNormal;
    Array2::from_shape_fn((rows, cols), |_| {
        half::bf16::from_f32(
            <StandardNormal as Distribution<f32>>::sample(&normal, rng) * 0.02,
        )
    })
}

fn synth_weights(cfg: &DecoderConfig, rng: &mut impl rand::RngCore) -> DecoderWeights {
    let d = cfg.hidden_size;
    let q = cfg.q_dim();
    let kv = cfg.kv_dim();
    let f = cfg.intermediate_size;
    let layers = (0..cfg.num_hidden_layers)
        .map(|_| DecoderLayerWeights {
            norm_attn: Array1::from_elem(d, 1.0),
            wq: Some(Arc::new(rand2_bf16(d, q, rng))),
            wk: Some(Arc::new(rand2_bf16(d, kv, rng))),
            wv: Some(Arc::new(rand2_bf16(d, kv, rng))),
            wo: Some(Arc::new(rand2_bf16(q, d, rng))),
            norm_ffn: Array1::from_elem(d, 1.0),
            w_gate: Some(Arc::new(rand2_bf16(d, f, rng))),
            w_up: Some(Arc::new(rand2_bf16(d, f, rng))),
            w_down: Some(Arc::new(rand2_bf16(f, d, rng))),
            q_norm: None,
            k_norm: None,
        })
        .collect();
    DecoderWeights {
        token_embedding: rand2_bf16(cfg.vocab_size, d, rng),
        final_norm: Array1::from_elem(d, 1.0),
        layers,
        model_identity: [0u8; 32],
    }
}

fn provision_decoder<E: GpuOffloadEngine>(
    weights: &DecoderWeights,
    cfg: &DecoderConfig,
    engine: &mut E,
) {
    for (li, layer) in weights.layers.iter().enumerate() {
        if !cfg.offload_layer(li) {
            continue;
        }
        let li16 = li as u16;
        for (kind, w) in [
            (WeightKind::Q, layer.wq.as_ref().unwrap()),
            (WeightKind::K, layer.wk.as_ref().unwrap()),
            (WeightKind::V, layer.wv.as_ref().unwrap()),
            (WeightKind::O, layer.wo.as_ref().unwrap()),
            (WeightKind::FfnGate, layer.w_gate.as_ref().unwrap()),
            (WeightKind::FfnUp, layer.w_up.as_ref().unwrap()),
            (WeightKind::FfnDown, layer.w_down.as_ref().unwrap()),
        ] {
            engine
                .register_weight_bf16(WeightHandle::new(li16, kind), w.view())
                .unwrap();
        }
    }
}

/// Prime `kv_cache` for sequence `b` (or all sequences if `b = None`)
/// with `n_kv` rows of random K and V at every layer. Bypasses RoPE
/// + RMSNorm + projection — we don't care about KV cache content
/// realism, only that attention has `n_kv` rows of work to do.
fn prime_kv_cache_random(
    kv_cache: &mut KvCache,
    n_kv: usize,
    rng: &mut impl rand::RngCore,
) {
    let kv_dim = kv_cache.kv_dim();
    let normal = StandardNormal;
    for li in 0..kv_cache.num_layers() {
        for b in 0..kv_cache.batch_size() {
            let k_rand = Array2::from_shape_fn((n_kv, kv_dim), |_| {
                <StandardNormal as Distribution<f32>>::sample(&normal, rng) * 0.02
            });
            let v_rand = Array2::from_shape_fn((n_kv, kv_dim), |_| {
                <StandardNormal as Distribution<f32>>::sample(&normal, rng) * 0.02
            });
            kv_cache
                .append_prefill(li, b, k_rand.view(), v_rand.view())
                .expect("prime kv cache");
        }
    }
}

const NUM_LAYERS: usize = 4;
const N_KV_PRIME: usize = 64;
const K_STEPS: usize = 8;
const BATCH_SIZE: usize = 8;

#[test]
#[ignore = "runs synth Qwen3-4B-shape decode bench on Vulkan/wgpu (~30s)"]
fn decode_step_serial_vs_batched() {
    // Force per-sequence A_b default (clear any stale env from
    // previous tests).
    unsafe {
        std::env::remove_var("BATCHED_DECODE_SHARED_A");
    }

    let cfg = qwen3_4b_shape_cfg(NUM_LAYERS);
    let mut weight_rng = ChaCha20Rng::from_seed([7u8; 32]);
    let weights = Arc::new(synth_weights(&cfg, &mut weight_rng));
    let rope = RopeTables::new(
        cfg.head_dim_value(),
        cfg.max_position_embeddings,
        cfg.rope_theta,
    );

    let gpu = WgpuVulkanEngine::new().expect("Vulkan adapter must be available");
    eprintln!(
        "[d1.6] Vulkan adapter: {} ({:?})",
        gpu.adapter_info().name,
        gpu.adapter_info().device_type,
    );
    eprintln!(
        "[d1.6] cfg: layers={NUM_LAYERS}, hidden={}, heads={}/{}, head_dim={}, intermediate={}",
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim_value(),
        cfg.intermediate_size,
    );
    eprintln!("[d1.6] B = {BATCH_SIZE}, n_kv prime = {N_KV_PRIME}, K steps = {K_STEPS}");

    let max_cache_len = N_KV_PRIME + K_STEPS + 1;

    // ── Variant A: serial × B ────────────────────────────────────
    // Build B independent (executor, kv_cache) pairs. Each step
    // runs `for b in 0..B { run_decode_step(b) }`.
    let mut serial_execs: Vec<InProcessTrustedExecutor<WgpuVulkanEngine>> =
        Vec::with_capacity(BATCH_SIZE);
    let mut serial_caches: Vec<KvCache> = Vec::with_capacity(BATCH_SIZE);
    eprintln!("[d1.6] building serial executors + KV caches ...");
    let t_setup = Instant::now();
    for b in 0..BATCH_SIZE {
        let exec = InProcessTrustedExecutor::with_seed(
            gpu.clone_shared(),
            MaskSeed::from_bytes([0xa0 ^ b as u8; 32]),
        );
        serial_execs.push(exec);
        let mut kv = KvCache::new(NUM_LAYERS, max_cache_len, cfg.kv_dim());
        let mut prime_rng = ChaCha20Rng::from_seed([0xb0 ^ b as u8; 32]);
        prime_kv_cache_random(&mut kv, N_KV_PRIME, &mut prime_rng);
        serial_caches.push(kv);
    }
    // Provision weights once into the GPU (all serial executors
    // share the same WgpuVulkanEngine via clone_shared).
    {
        let mut provision_gpu = gpu.clone_shared();
        provision_decoder(&weights, &cfg, &mut provision_gpu);
    }
    eprintln!("[d1.6]   serial setup done in {:.2?}", t_setup.elapsed());

    // Warm up: one decode step per sequence to amortise JIT compile.
    let warm_token: u32 = 1;
    for b in 0..BATCH_SIZE {
        let _ = forward::run_decode_step(
            &cfg,
            &weights,
            &rope,
            &mut serial_execs[b],
            warm_token,
            &mut serial_caches[b],
        )
        .expect("serial warm");
    }

    // Measured: K decode steps for each sequence.
    gelo_protocol::profile::reset();
    let t0 = Instant::now();
    for _step in 0..K_STEPS {
        for b in 0..BATCH_SIZE {
            let _ = forward::run_decode_step(
                &cfg,
                &weights,
                &rope,
                &mut serial_execs[b],
                warm_token,
                &mut serial_caches[b],
            )
            .expect("serial decode");
        }
    }
    let elapsed_serial = t0.elapsed();
    let profile_serial = gelo_protocol::profile::snapshot();
    let per_step_per_seq_serial = elapsed_serial / (K_STEPS * BATCH_SIZE) as u32;

    // Drop serial executors / caches before building the batched
    // setup to release Vulkan handles.
    drop(serial_execs);
    drop(serial_caches);

    // ── Variant B: batched ───────────────────────────────────────
    eprintln!("[d1.6] building batched executor + KV cache ...");
    let t_setup = Instant::now();
    let mut batched_exec = InProcessTrustedExecutor::with_seed(
        gpu.clone_shared(),
        MaskSeed::from_bytes([0xc0; 32]),
    );
    let mut batched_kv =
        KvCache::new_batched(BATCH_SIZE, NUM_LAYERS, max_cache_len, cfg.kv_dim());
    let mut prime_rng = ChaCha20Rng::from_seed([0xd0; 32]);
    prime_kv_cache_random(&mut batched_kv, N_KV_PRIME, &mut prime_rng);
    eprintln!("[d1.6]   batched setup done in {:.2?}", t_setup.elapsed());

    let batched_tokens: Vec<u32> = vec![warm_token; BATCH_SIZE];

    // Warm up: one batched decode step.
    let _ = forward::run_decode_step_batched(
        &cfg,
        &weights,
        &rope,
        &mut batched_exec,
        &batched_tokens,
        &mut batched_kv,
    )
    .expect("batched warm");

    // Measured: K batched decode steps.
    gelo_protocol::profile::reset();
    let t0 = Instant::now();
    for _step in 0..K_STEPS {
        let _ = forward::run_decode_step_batched(
            &cfg,
            &weights,
            &rope,
            &mut batched_exec,
            &batched_tokens,
            &mut batched_kv,
        )
        .expect("batched decode");
    }
    let elapsed_batched = t0.elapsed();
    let profile_batched = gelo_protocol::profile::snapshot();
    let per_step_per_seq_batched = elapsed_batched / (K_STEPS * BATCH_SIZE) as u32;

    // ── Results ───────────────────────────────────────────────────
    eprintln!("\n[d1.6] ─── results ───");
    eprintln!(
        "[d1.6] serial:  total={elapsed_serial:.2?} per-step-per-seq={per_step_per_seq_serial:.2?}"
    );
    eprintln!(
        "[d1.6] batched: total={elapsed_batched:.2?} per-step-per-seq={per_step_per_seq_batched:.2?}"
    );
    let speedup = elapsed_serial.as_secs_f64() / elapsed_batched.as_secs_f64();
    eprintln!("[d1.6] speedup (serial/batched): {speedup:.2}×");

    profile_serial.dump(&format!(
        "d1.6 serial breakdown (B={BATCH_SIZE}, K={K_STEPS}, total wall={elapsed_serial:.2?})"
    ));
    profile_batched.dump(&format!(
        "d1.6 batched breakdown (B={BATCH_SIZE}, K={K_STEPS}, total wall={elapsed_batched:.2?})"
    ));
}

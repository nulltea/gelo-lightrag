//! R4 async-offload parity tests.
//!
//! Validates that the forward.rs prefill paths produce the same outputs
//! when run under `GELO_ASYNC_OFFLOAD=1` (R4 async dispatch) as when
//! run under the legacy sync path. This is a wiring-correctness test
//! at the integration level — semantic parity of the substrate's async
//! methods is already covered by `gelo-protocol` unit tests; this
//! file confirms `forward.rs::offload_*_dispatch` routes correctly.
//!
//! Runs synthetic tiny weights so the test is offline + fast.
//!
//! ## Test isolation
//!
//! `GELO_ASYNC_OFFLOAD` is a process-global env var; concurrent tests
//! flipping it would race. A process-wide mutex serializes async tests
//! and snapshots/restores the env var around each scope.

use std::sync::{Mutex, OnceLock};

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

use gelo_embedder::decoder::config::DecoderConfig;
use gelo_embedder::decoder::forward;
use gelo_embedder::decoder::rope::RopeTables;
use gelo_embedder::decoder::weights::{DecoderLayerWeights, DecoderWeights};
use gelo_protocol::rng::MaskSeed;
use gelo_protocol::{
    GpuOffloadEngine, InProcessTrustedExecutor, RayonCpuEngine, TrustedExecutor, WeightHandle,
    WeightKind,
};

const ENV_VAR: &str = "GELO_ASYNC_OFFLOAD";

/// Serialize env-var manipulation across tests in this file.
fn env_mutex() -> &'static Mutex<()> {
    static M: OnceLock<Mutex<()>> = OnceLock::new();
    M.get_or_init(|| Mutex::new(()))
}

/// Run `f` with the env var set; restore the prior value on scope exit.
fn with_async_offload<F: FnOnce() -> R, R>(on: bool, f: F) -> R {
    let _g = env_mutex().lock().unwrap_or_else(|p| p.into_inner());
    let prior = std::env::var(ENV_VAR).ok();
    // SAFETY: serialized via env_mutex(); no other thread in this
    // process touches the var while the guard is held. Other processes
    // (cargo test workers) get their own env state at spawn so this
    // doesn't affect them.
    unsafe {
        if on {
            std::env::set_var(ENV_VAR, "1");
        } else {
            std::env::remove_var(ENV_VAR);
        }
    }
    let out = f();
    unsafe {
        match prior {
            Some(v) => std::env::set_var(ENV_VAR, v),
            None => std::env::remove_var(ENV_VAR),
        }
    }
    out
}

// ─── synthetic decoder helpers (copy of decoder_parity.rs's helpers, kept local
//     so this file is self-contained) ────────────────────────────────

fn tiny_decoder_config(
    num_layers: usize,
    hidden: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    intermediate: usize,
) -> DecoderConfig {
    DecoderConfig {
        vocab_size: 64,
        hidden_size: hidden,
        intermediate_size: intermediate,
        num_hidden_layers: num_layers,
        num_attention_heads: n_q_heads,
        num_key_value_heads: n_kv_heads,
        head_dim: Some(head_dim),
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        hidden_act: "silu".into(),
        tie_word_embeddings: false,
        max_seq_len: 64,
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

fn rand2(rows: usize, cols: usize, rng: &mut impl rand::RngCore, scale: f32) -> Array2<f32> {
    let normal = StandardNormal;
    Array2::from_shape_fn((rows, cols), |_| {
        <StandardNormal as Distribution<f32>>::sample(&normal, rng) * scale
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
            wq: Some(std::sync::Arc::new(rand2(d, q, rng, 0.05).mapv(half::bf16::from_f32))),
            wk: Some(std::sync::Arc::new(rand2(d, kv, rng, 0.05).mapv(half::bf16::from_f32))),
            wv: Some(std::sync::Arc::new(rand2(d, kv, rng, 0.05).mapv(half::bf16::from_f32))),
            wo: Some(std::sync::Arc::new(rand2(q, d, rng, 0.05).mapv(half::bf16::from_f32))),
            norm_ffn: Array1::from_elem(d, 1.0),
            w_gate: Some(std::sync::Arc::new(rand2(d, f, rng, 0.05).mapv(half::bf16::from_f32))),
            w_up: Some(std::sync::Arc::new(rand2(d, f, rng, 0.05).mapv(half::bf16::from_f32))),
            w_down: Some(std::sync::Arc::new(rand2(f, d, rng, 0.05).mapv(half::bf16::from_f32))),
            q_norm: None,
            k_norm: None,
        })
        .collect();
    DecoderWeights {
        token_embedding: rand2(cfg.vocab_size, d, rng, 0.05).mapv(half::bf16::from_f32),
        final_norm: Array1::from_elem(d, 1.0),
        layers,
        model_identity: [0u8; 32],
    }
}

fn provision_offload<E: GpuOffloadEngine>(weights: &DecoderWeights, cfg: &DecoderConfig, engine: &mut E) {
    for (li, layer) in weights.layers.iter().enumerate() {
        if !cfg.offload_layer(li) {
            continue;
        }
        let li16 = li as u16;
        engine
            .register_weight_bf16(WeightHandle::new(li16, WeightKind::Q), layer.wq.as_ref().unwrap().view())
            .unwrap();
        engine
            .register_weight_bf16(WeightHandle::new(li16, WeightKind::K), layer.wk.as_ref().unwrap().view())
            .unwrap();
        engine
            .register_weight_bf16(WeightHandle::new(li16, WeightKind::V), layer.wv.as_ref().unwrap().view())
            .unwrap();
        engine
            .register_weight_bf16(WeightHandle::new(li16, WeightKind::O), layer.wo.as_ref().unwrap().view())
            .unwrap();
        engine
            .register_weight_bf16(WeightHandle::new(li16, WeightKind::FfnGate), layer.w_gate.as_ref().unwrap().view())
            .unwrap();
        engine
            .register_weight_bf16(WeightHandle::new(li16, WeightKind::FfnUp), layer.w_up.as_ref().unwrap().view())
            .unwrap();
        engine
            .register_weight_bf16(WeightHandle::new(li16, WeightKind::FfnDown), layer.w_down.as_ref().unwrap().view())
            .unwrap();
    }
}

fn run_prefill_single_stream(seed_bytes: [u8; 32]) -> Array2<f32> {
    let cfg = tiny_decoder_config(2, 32, 4, 2, 8, 64);
    let mut rng = ChaCha20Rng::from_seed([21u8; 32]);
    let weights = synth_weights(&cfg, &mut rng);
    let rope = RopeTables::new(cfg.head_dim_value(), cfg.max_position_embeddings, cfg.rope_theta);

    let mut engine = RayonCpuEngine::new();
    provision_offload(&weights, &cfg, &mut engine);
    let mut exec =
        InProcessTrustedExecutor::with_seed(engine, MaskSeed::from_bytes(seed_bytes));

    let input_ids: Vec<u32> = vec![1, 5, 9, 13, 17, 21, 25, 29];
    forward::run(&cfg, &weights, &rope, &mut exec, &input_ids).unwrap()
}

#[test]
fn r4_async_prefill_single_matches_sync() {
    // Validates: with GELO_ASYNC_OFFLOAD=1, decoder_block (single-stream
    // prefill, called from `forward::run`) routes through the async
    // substrate API and produces the same output as the sync path
    // within mask-roundtrip tolerance.
    //
    // The dispatch helpers (offload_qkv_dispatch / offload_linear_dispatch /
    // offload_linear_many_dispatch) are shared with decoder_block_batched,
    // so this test also covers the batched prefill wiring at the
    // dispatch-routing level. End-to-end batched parity on real model
    // weights is a Step 4 bench-time concern.
    let seed = [42u8; 32];
    let sync_out = with_async_offload(false, || run_prefill_single_stream(seed));
    let async_out = with_async_offload(true, || run_prefill_single_stream(seed));

    assert_eq!(sync_out.shape(), async_out.shape());
    let mut max_abs = 0.0_f32;
    for ((i, j), v) in sync_out.indexed_iter() {
        let diff = (v - async_out[[i, j]]).abs();
        if diff > max_abs {
            max_abs = diff;
        }
    }
    assert!(
        max_abs < 5e-3,
        "R4 async single-stream prefill diverges from sync: max abs {max_abs}",
    );
}

#[test]
fn r4_async_default_off_is_sync() {
    // With env var unset, the dispatch helpers should call the sync
    // path. The sync path was already verified by decoder_parity.rs
    // (PlaintextExecutor vs InProcessTrustedExecutor). Here we just
    // confirm that the env-var-unset run produces a non-degenerate
    // result (i.e., the wiring didn't accidentally short-circuit
    // everything to zero).
    let _g = env_mutex().lock().unwrap_or_else(|p| p.into_inner());
    // SAFETY: serialized.
    unsafe {
        std::env::remove_var(ENV_VAR);
    }
    let out = run_prefill_single_stream([72u8; 32]);
    assert!(out.iter().any(|&v| v.abs() > 1e-6), "all zeros — wiring bug");
}

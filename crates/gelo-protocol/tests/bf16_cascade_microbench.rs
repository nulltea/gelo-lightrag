//! Standalone microbench — bf16 cascade vs f32 cascade at the
//! production cascade shape.
//!
//! Validates the §4.E.3 phase-3 hypothesis: bf16 storage at the
//! cascade tile boundary delivers measurable wall reduction. If yes,
//! green-light the multi-week phase 3b/3c substrate + forward.rs
//! wire-up. If no, the multi-week chain can't deliver — abort and
//! pivot.
//!
//! Run:
//!   cargo test -p gelo-protocol --release --test bf16_cascade_microbench \
//!       -- --ignored --nocapture
//!
//! Output: per-path mean wall ± stddev, speedup ratio, go/no-go signal.
//!
//! The benches run apply+unapply round-trip (the actual production
//! flow) at the two cascade shapes that matter:
//!   - DCT-IV at n=2056 d=2560 (post-shield production prefill, pad 1.99)
//!   - HD₃ at n=4096 d=2560 (post-shield production prefill, pad < 1.6)

use std::time::Instant;

use gelo_protocol::dct4::Dct4Mask;
use gelo_protocol::hd3::Hd3Mask;
use half::bf16;
use ndarray::Array2;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

const N_WARMUP: usize = 3;
const N_ITER: usize = 25;

fn sample_normal(rng: &mut ChaCha20Rng, n: usize, d: usize) -> Array2<f32> {
    let normal = StandardNormal;
    Array2::from_shape_fn((n, d), |_| normal.sample(rng))
}

fn stats(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// Format a duration in ms with a few digits of precision.
fn fmt_ms(seconds: f64) -> String {
    format!("{:8.3} ms", seconds * 1000.0)
}

#[test]
#[ignore = "microbench: minutes; run on a quiet box for reliable numbers"]
fn dct4_bf16_vs_f32_cascade_at_production_shape() {
    let n = 2056; // n_prompt 2048 + k_shield 8 = production post-shield row count
    let d = 2560; // Qwen3-4B hidden_size
    let mut rng = ChaCha20Rng::from_seed([42u8; 32]);
    let mask = Dct4Mask::fresh(n, &mut rng);
    let h_init = sample_normal(&mut rng, n, d);

    eprintln!("=== DCT-IV cascade microbench — n={n} d={d} ===");
    eprintln!("warmup: {N_WARMUP} iter; measure: {N_ITER} iter");
    eprintln!();

    // ── f32 path ──
    let mut f32_apply: Vec<f64> = Vec::with_capacity(N_ITER);
    let mut f32_unapply: Vec<f64> = Vec::with_capacity(N_ITER);
    let mut f32_round_trip: Vec<f64> = Vec::with_capacity(N_ITER);
    for i in 0..(N_WARMUP + N_ITER) {
        let mut buf = h_init.clone();
        let buf_slice = buf.as_slice_mut().unwrap();
        let t0 = Instant::now();
        mask.apply_in_place_slice(buf_slice, d);
        let apply_dt = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        mask.unapply_in_place_slice(buf_slice, d);
        let unapply_dt = t1.elapsed().as_secs_f64();
        let total_dt = apply_dt + unapply_dt;
        if i >= N_WARMUP {
            f32_apply.push(apply_dt);
            f32_unapply.push(unapply_dt);
            f32_round_trip.push(total_dt);
        }
    }

    // ── bf16 path ──
    let h_init_bf16: Vec<bf16> = h_init.iter().map(|&v| bf16::from_f32(v)).collect();
    let mut bf16_apply: Vec<f64> = Vec::with_capacity(N_ITER);
    let mut bf16_unapply: Vec<f64> = Vec::with_capacity(N_ITER);
    let mut bf16_round_trip: Vec<f64> = Vec::with_capacity(N_ITER);
    for i in 0..(N_WARMUP + N_ITER) {
        let mut buf = h_init_bf16.clone();
        let t0 = Instant::now();
        mask.apply_in_place_slice_bf16(&mut buf, d);
        let apply_dt = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        mask.unapply_in_place_slice_bf16(&mut buf, d);
        let unapply_dt = t1.elapsed().as_secs_f64();
        let total_dt = apply_dt + unapply_dt;
        if i >= N_WARMUP {
            bf16_apply.push(apply_dt);
            bf16_unapply.push(unapply_dt);
            bf16_round_trip.push(total_dt);
        }
    }

    let (f32_apply_mean, f32_apply_std) = stats(&f32_apply);
    let (f32_unapply_mean, f32_unapply_std) = stats(&f32_unapply);
    let (f32_rt_mean, f32_rt_std) = stats(&f32_round_trip);
    let (bf16_apply_mean, bf16_apply_std) = stats(&bf16_apply);
    let (bf16_unapply_mean, bf16_unapply_std) = stats(&bf16_unapply);
    let (bf16_rt_mean, bf16_rt_std) = stats(&bf16_round_trip);

    eprintln!("                       f32 mean ± stddev          bf16 mean ± stddev         ratio (bf16/f32)");
    eprintln!(
        "  apply            {} ± {}     {} ± {}     {:.3}",
        fmt_ms(f32_apply_mean), fmt_ms(f32_apply_std),
        fmt_ms(bf16_apply_mean), fmt_ms(bf16_apply_std),
        bf16_apply_mean / f32_apply_mean,
    );
    eprintln!(
        "  unapply          {} ± {}     {} ± {}     {:.3}",
        fmt_ms(f32_unapply_mean), fmt_ms(f32_unapply_std),
        fmt_ms(bf16_unapply_mean), fmt_ms(bf16_unapply_std),
        bf16_unapply_mean / f32_unapply_mean,
    );
    eprintln!(
        "  apply+unapply    {} ± {}     {} ± {}     {:.3}",
        fmt_ms(f32_rt_mean), fmt_ms(f32_rt_std),
        fmt_ms(bf16_rt_mean), fmt_ms(bf16_rt_std),
        bf16_rt_mean / f32_rt_mean,
    );

    eprintln!();
    let speedup = f32_rt_mean / bf16_rt_mean;
    let saving_pct = 100.0 * (1.0 - bf16_rt_mean / f32_rt_mean);
    eprintln!("  → bf16 speedup vs f32: {:.3}× ({:+.1}% wall change)", speedup, -saving_pct);
    eprintln!(
        "DCT4_RESULT shape=({n},{d}) f32_ms={:.3} bf16_ms={:.3} speedup={:.3} saving_pct={:.2}",
        f32_rt_mean * 1000.0,
        bf16_rt_mean * 1000.0,
        speedup,
        saving_pct,
    );
}

#[test]
#[ignore = "microbench: minutes; run on a quiet box for reliable numbers"]
fn hd3_bf16_vs_f32_cascade_at_production_shape() {
    let n = 4096; // production HD₃ shape (pow2-padded shield + prompt)
    let d = 2560;
    let mut rng = ChaCha20Rng::from_seed([42u8; 32]);
    let mask = Hd3Mask::fresh(n, &mut rng);
    let h_init = sample_normal(&mut rng, n, d);

    eprintln!("\n=== HD₃ cascade microbench — n={n} d={d} ===");
    eprintln!("warmup: {N_WARMUP} iter; measure: {N_ITER} iter");
    eprintln!();

    let mut f32_apply: Vec<f64> = Vec::with_capacity(N_ITER);
    let mut f32_unapply: Vec<f64> = Vec::with_capacity(N_ITER);
    let mut f32_round_trip: Vec<f64> = Vec::with_capacity(N_ITER);
    for i in 0..(N_WARMUP + N_ITER) {
        let mut buf = h_init.clone();
        let buf_slice = buf.as_slice_mut().unwrap();
        let t0 = Instant::now();
        mask.apply_in_place_slice(buf_slice, d);
        let apply_dt = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        mask.unapply_in_place_slice(buf_slice, d);
        let unapply_dt = t1.elapsed().as_secs_f64();
        let total_dt = apply_dt + unapply_dt;
        if i >= N_WARMUP {
            f32_apply.push(apply_dt);
            f32_unapply.push(unapply_dt);
            f32_round_trip.push(total_dt);
        }
    }

    let h_init_bf16: Vec<bf16> = h_init.iter().map(|&v| bf16::from_f32(v)).collect();
    let mut bf16_apply: Vec<f64> = Vec::with_capacity(N_ITER);
    let mut bf16_unapply: Vec<f64> = Vec::with_capacity(N_ITER);
    let mut bf16_round_trip: Vec<f64> = Vec::with_capacity(N_ITER);
    for i in 0..(N_WARMUP + N_ITER) {
        let mut buf = h_init_bf16.clone();
        let t0 = Instant::now();
        mask.apply_in_place_slice_bf16(&mut buf, d);
        let apply_dt = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        mask.unapply_in_place_slice_bf16(&mut buf, d);
        let unapply_dt = t1.elapsed().as_secs_f64();
        let total_dt = apply_dt + unapply_dt;
        if i >= N_WARMUP {
            bf16_apply.push(apply_dt);
            bf16_unapply.push(unapply_dt);
            bf16_round_trip.push(total_dt);
        }
    }

    let (f32_apply_mean, f32_apply_std) = stats(&f32_apply);
    let (f32_unapply_mean, f32_unapply_std) = stats(&f32_unapply);
    let (f32_rt_mean, f32_rt_std) = stats(&f32_round_trip);
    let (bf16_apply_mean, bf16_apply_std) = stats(&bf16_apply);
    let (bf16_unapply_mean, bf16_unapply_std) = stats(&bf16_unapply);
    let (bf16_rt_mean, bf16_rt_std) = stats(&bf16_round_trip);

    eprintln!("                       f32 mean ± stddev          bf16 mean ± stddev         ratio (bf16/f32)");
    eprintln!(
        "  apply            {} ± {}     {} ± {}     {:.3}",
        fmt_ms(f32_apply_mean), fmt_ms(f32_apply_std),
        fmt_ms(bf16_apply_mean), fmt_ms(bf16_apply_std),
        bf16_apply_mean / f32_apply_mean,
    );
    eprintln!(
        "  unapply          {} ± {}     {} ± {}     {:.3}",
        fmt_ms(f32_unapply_mean), fmt_ms(f32_unapply_std),
        fmt_ms(bf16_unapply_mean), fmt_ms(bf16_unapply_std),
        bf16_unapply_mean / f32_unapply_mean,
    );
    eprintln!(
        "  apply+unapply    {} ± {}     {} ± {}     {:.3}",
        fmt_ms(f32_rt_mean), fmt_ms(f32_rt_std),
        fmt_ms(bf16_rt_mean), fmt_ms(bf16_rt_std),
        bf16_rt_mean / f32_rt_mean,
    );

    eprintln!();
    let speedup = f32_rt_mean / bf16_rt_mean;
    let saving_pct = 100.0 * (1.0 - bf16_rt_mean / f32_rt_mean);
    eprintln!("  → bf16 speedup vs f32: {:.3}× ({:+.1}% wall change)", speedup, -saving_pct);
    eprintln!(
        "HD3_RESULT shape=({n},{d}) f32_ms={:.3} bf16_ms={:.3} speedup={:.3} saving_pct={:.2}",
        f32_rt_mean * 1000.0,
        bf16_rt_mean * 1000.0,
        speedup,
        saving_pct,
    );
}

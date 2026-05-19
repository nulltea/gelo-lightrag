//! bf16 mask GEMM parity simulation.
//!
//! Step 2 of the round-3 perf plan
//! (`docs/research/private-llm-inference-round-3.md` §7 item 2) asks:
//! would a bf16 mask GEMM with f32 accumulate preserve the `Aᵀ·A=I`
//! round-trip identity closely enough that GELO's downstream model
//! parity (paper Table 1: ≥98.8 % top-1 token equality at bf16) is
//! unchanged?
//!
//! The vendored AOCL-BLIS in this repo
//! (`vendor/aocl-install/lib/libblis-mt.so.5.2.2`) does not export
//! `sbgemm_` / `bli_gemm_bf16bf16f32` symbols — confirmed via
//! `nm | grep -i bf16` returning empty. So a real bf16 mask GEMM is
//! gated on either upgrading AOCL or hand-rolling AVX-512 BF16
//! intrinsics. This test answers the **arithmetic** question without
//! either path: we simulate the bf16 truncation of operands, then run
//! the f32 GEMM (which mirrors the `bf16 inputs + f32 accumulate`
//! behaviour of AOCL's mixed-precision sgemm — the f32 accumulator
//! has the same dynamic range as our existing path, only the operand
//! rounding is bf16).
//!
//! Reports max-abs and mean-abs round-trip error vs an f32 baseline
//! and vs a "bf16-everywhere" target (`H_bf · W_bf` directly), which
//! is the operationally correct comparison if downstream Qwen3
//! inference is also bf16. The released paper-parity bench is f32, so
//! we report both.

use gelo_protocol::GeloMask;
use half::bf16;
use ndarray::{Array2, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

/// Round every entry of `m` through bf16 and back to f32. This models
/// the input-rounding error you would see with a bf16-inputs +
/// f32-accumulate GEMM (the actual GEMM math itself is unchanged
/// because we still run the multiply in f32 — we only rounded the
/// operands).
fn round_through_bf16(m: ArrayView2<'_, f32>) -> Array2<f32> {
    Array2::from_shape_fn(m.raw_dim(), |idx| {
        let v = m[idx];
        bf16::from_f32(v).to_f32()
    })
}

fn sample_normal(rng: &mut ChaCha20Rng, n: usize, d: usize) -> Array2<f32> {
    let normal = StandardNormal;
    Array2::from_shape_fn((n, d), |_| normal.sample(rng))
}

fn max_abs(a: ArrayView2<'_, f32>, b: ArrayView2<'_, f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn mean_abs(a: ArrayView2<'_, f32>, b: ArrayView2<'_, f32>) -> f32 {
    let mut acc = 0.0_f32;
    let mut n = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        acc += (x - y).abs();
        n += 1;
    }
    if n == 0 { 0.0 } else { acc / (n as f32) }
}

fn rms(a: ArrayView2<'_, f32>) -> f32 {
    let n = a.len() as f32;
    let s: f32 = a.iter().map(|v| v * v).sum();
    (s / n).sqrt()
}

/// Run one parity comparison at a given (n, d, p) shape.
///
/// n = stacked token-axis size, d = input-hidden width, p = output width.
fn run_one(label: &str, n: usize, d: usize, p: usize, seed: [u8; 32]) {
    let mut rng = ChaCha20Rng::from_seed(seed);

    let h = sample_normal(&mut rng, n, d);
    let w = sample_normal(&mut rng, d, p);
    let mask = GeloMask::fresh(n, &mut rng);

    // ----- f32 reference path -----
    let target = h.dot(&w); // H · W
    let u_f32 = mask.apply(h.view()); // A · H
    let v_f32 = u_f32.dot(&w); // (A · H) · W
    let recovered_f32 = mask.unapply(v_f32.view()); // Aᵀ · ((A · H) · W) ≈ H · W

    let err_f32_max = max_abs(recovered_f32.view(), target.view());
    let err_f32_mean = mean_abs(recovered_f32.view(), target.view());

    // ----- bf16 simulated path -----
    // Round operands to bf16 once; GEMMs still run f32 (the input-
    // rounding error is the only one a real bf16-input + f32-accumulate
    // GEMM would introduce).
    let h_bf = round_through_bf16(h.view());
    let w_bf = round_through_bf16(w.view());
    // The mask matrix is rounded through bf16 too — that's what a real
    // bf16 mask GEMM would feed the hardware.
    let mask_a_bf = round_through_bf16(mask.matrix());
    let mask_at_bf = round_through_bf16(mask.matrix().t());

    let u_bf = mask_a_bf.dot(&h_bf);
    let v_bf = u_bf.dot(&w_bf);
    let recovered_bf16 = mask_at_bf.dot(&v_bf);

    // Compare bf16-trip against the f32 target (what the model expects
    // if upstream/downstream is still f32).
    let err_bf_vs_f32_max = max_abs(recovered_bf16.view(), target.view());
    let err_bf_vs_f32_mean = mean_abs(recovered_bf16.view(), target.view());

    // Compare bf16-trip against the bf16-everywhere target (what the
    // model expects if upstream/downstream is bf16 too — the paper's
    // Table 1 regime where ≥98.8% top-1 holds at bf16).
    let target_bf = h_bf.dot(&w_bf);
    let err_bf_vs_bf_max = max_abs(recovered_bf16.view(), target_bf.view());
    let err_bf_vs_bf_mean = mean_abs(recovered_bf16.view(), target_bf.view());

    // Magnitudes for context.
    let target_rms = rms(target.view());
    let h_rms = rms(h.view());

    eprintln!();
    eprintln!("--- {label} (n={n}, d={d}, p={p}) ---");
    eprintln!("  H rms                           : {:.3e}", h_rms);
    eprintln!("  target H·W rms                  : {:.3e}", target_rms);
    eprintln!();
    eprintln!("  f32 round-trip vs target:");
    eprintln!("    max  abs error                : {:.3e}", err_f32_max);
    eprintln!("    mean abs error                : {:.3e}", err_f32_mean);
    eprintln!("    mean/rms ratio                : {:.3e}", err_f32_mean / target_rms);
    eprintln!();
    eprintln!("  bf16 round-trip vs f32 target:");
    eprintln!("    max  abs error                : {:.3e}", err_bf_vs_f32_max);
    eprintln!("    mean abs error                : {:.3e}", err_bf_vs_f32_mean);
    eprintln!("    mean/rms ratio                : {:.3e}", err_bf_vs_f32_mean / target_rms);
    eprintln!();
    eprintln!("  bf16 round-trip vs bf16-everywhere target:");
    eprintln!("    max  abs error                : {:.3e}", err_bf_vs_bf_max);
    eprintln!("    mean abs error                : {:.3e}", err_bf_vs_bf_mean);
    eprintln!("    mean/rms ratio                : {:.3e}", err_bf_vs_bf_mean / target_rms);
}

#[test]
fn bf16_parity_small_shape() {
    // Quick smoke test at a tiny shape — runs in <1 s.
    run_one("small shape (sanity)", 64, 64, 64, [11u8; 32]);
}

#[test]
#[ignore = "realistic Qwen3-1.7B mask shape — takes ~30 s of CPU GEMMs"]
fn bf16_parity_qwen3_shapes() {
    // Three shapes mirror the per-layer call sites:
    //   QKV/O/gate_up apply  : (n+k=2056, d_in=2048)
    //   FfnDown apply        : (2056, 6144)
    //   gate, up unapply     : output width f=6144 (we model with p=6144)
    // Mask side length s = 2056 for all (per-forward-pass shared A).
    let s = 2056;
    run_one("apply: d_in=2048", s, 2048, 2048, [21u8; 32]);
    run_one("apply: d_in=6144 (FfnDown)", s, 6144, 2048, [23u8; 32]);
    run_one("unapply: d_out=6144 (gate/up)", s, 2048, 6144, [25u8; 32]);
}

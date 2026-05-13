//! Micro-bench for `sample_haar_orthogonal` at sizes representative of
//! BGE-base + NFCorpus workloads. The mask is sampled at seq_len, so
//! n ranges from ~30 (short docs) to 512 (max BGE). For n in [16, 256]
//! we expect the BLAS-3 inner update to dominate; below n=16 the
//! allocation overhead of `Array1::to_owned()` may even regress vs the
//! scalar baseline.
//!
//! Run: cargo test -p gelo-protocol --release --test qr_bench -- --ignored --nocapture

use std::time::Instant;

use gelo_protocol::{GeloMask, MaskSeed};
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;

#[test]
#[ignore]
fn qr_speed_by_n() {
    let sizes = [16, 32, 64, 128, 256, 384, 512];
    let n_iter = 20;
    eprintln!();
    eprintln!("{:>5} {:>12} {:>12} {:>12}", "n", "median_us", "min_us", "max_us");
    eprintln!("{}", "-".repeat(48));
    for &n in &sizes {
        let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
        // Warm-up
        let _ = GeloMask::fresh(n, &mut rng);
        let mut samples = Vec::with_capacity(n_iter);
        for _ in 0..n_iter {
            let t0 = Instant::now();
            let _m = GeloMask::fresh(n, &mut rng);
            samples.push(t0.elapsed().as_micros());
        }
        samples.sort_unstable();
        let med = samples[samples.len() / 2];
        let min = samples[0];
        let max = samples[samples.len() - 1];
        eprintln!("{:>5} {:>12} {:>12} {:>12}", n, med, min, max);
    }
}

/// Sanity check at one representative point — confirms the mask
/// is orthogonal after the BLAS-accelerated path. Catches algorithm
/// regressions distinct from `mask::tests::orthogonality`.
#[test]
fn blas_haar_is_orthogonal_at_real_seq_len() {
    let n = 128;
    let mask = GeloMask::from_seed(n, MaskSeed::from_bytes([42u8; 32]));
    let a = mask.matrix();
    let ata = a.t().dot(&a);
    let mut max_err = 0.0_f32;
    for i in 0..n {
        for j in 0..n {
            let want = if i == j { 1.0 } else { 0.0 };
            let err = (ata[[i, j]] - want).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    assert!(
        max_err < 5e-4,
        "‖AᵀA − I‖_max = {max_err} at n={n}; new BLAS-3 path is non-orthogonal"
    );
}

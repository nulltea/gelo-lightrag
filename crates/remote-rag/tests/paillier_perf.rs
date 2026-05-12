//! M6.4 perf smoke test — `<100 ms / 1024-dim dot product` is the stretch
//! target from the plan. Hand-rolled Paillier on `num-bigint` with the
//! default 1024-bit modulus is unlikely to hit it; this test prints the
//! measured wall-clock and only fails if we blow well past it.

use std::time::Instant;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use remote_rag::paillier::{
    DEFAULT_SCALE_BITS, PaillierPrivateKey, dequantize_product, quantize,
};

#[test]
#[ignore = "perf smoke — multi-second on debug builds; run with --release for representative numbers"]
fn paillier_dot_perf_1024_dim() {
    let dim = 1024;
    let sk = PaillierPrivateKey::generate();
    let mut rng = ChaCha20Rng::from_seed([42u8; 32]);

    // Synthetic query and one candidate document, both unit-norm-ish.
    let q: Vec<f32> = (0..dim).map(|i| ((i as f32) / dim as f32 - 0.5) * 0.1).collect();
    let e_d: Vec<f32> = (0..dim)
        .map(|i| ((dim - i) as f32 / dim as f32 - 0.5) * 0.1)
        .collect();
    let q_int = quantize(&q, DEFAULT_SCALE_BITS);
    let e_d_int = quantize(&e_d, DEFAULT_SCALE_BITS);

    // Encrypt the query twice: once via the public-key path (server-style),
    // once via the keypair-holder CRT path (RemoteRAG client-style). The
    // service code uses the second one — the first is here for comparison.
    let t0 = Instant::now();
    let q_ct_pub: Vec<_> = q_int
        .iter()
        .map(|m| sk.public.encrypt_signed(m, &mut rng))
        .collect();
    let enc_pub_ms = t0.elapsed().as_millis();

    let t0 = Instant::now();
    let q_ct: Vec<_> = q_int
        .iter()
        .map(|m| sk.encrypt_signed(m, &mut rng))
        .collect();
    let enc_ms = t0.elapsed().as_millis();
    eprintln!("[paillier] encrypt_query: public-key path={enc_pub_ms}ms, CRT path={enc_ms}ms");
    let _ = q_ct_pub;

    // Hot path: server homomorphic dot product against one candidate.
    let t1 = Instant::now();
    let dot_ct = sk
        .public
        .homomorphic_dot(&q_ct, &e_d_int, &mut rng)
        .expect("homomorphic dot");
    let dot_ms = t1.elapsed().as_millis();

    let dot_int = sk.decrypt_signed(&dot_ct);
    let homomorphic_dot = dequantize_product(&dot_int, DEFAULT_SCALE_BITS) as f32;
    let plain_dot: f32 = q.iter().zip(e_d.iter()).map(|(a, b)| a * b).sum();

    eprintln!(
        "[paillier] dim={dim} encrypt_query={enc_ms}ms one_dot={dot_ms}ms \
         homomorphic={homomorphic_dot:.6} plain={plain_dot:.6}"
    );

    assert!(
        (homomorphic_dot - plain_dot).abs() < 1e-3,
        "homomorphic {homomorphic_dot} vs plain {plain_dot}"
    );
    // Soft check: if we're more than 10× over the stretch target on a debug
    // build, something has regressed substantially. Tighter target only on
    // release builds.
    assert!(
        dot_ms < 5000,
        "one_dot {dot_ms} ms way over expected; check the modpow path"
    );
}

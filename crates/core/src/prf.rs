use hmac::{Hmac, Mac};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

pub fn derive_rng(key: &[u8], nonce: &[u8], label: &[u8]) -> anyhow::Result<ChaCha20Rng> {
    let mut mac = HmacSha256::new_from_slice(key)?;
    mac.update(label);
    mac.update(nonce);
    let digest = mac.finalize().into_bytes();

    let mut seed = [0_u8; 32];
    seed.copy_from_slice(&digest[..32]);
    Ok(ChaCha20Rng::from_seed(seed))
}

pub fn sample_unit_vector(rng: &mut ChaCha20Rng, dims: usize) -> Vec<f32> {
    let mut raw = Vec::with_capacity(dims);
    while raw.len() < dims {
        let u1 = rng.random::<f32>().clamp(f32::EPSILON, 1.0);
        let u2 = rng.random::<f32>();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        raw.push(radius * theta.cos());
        if raw.len() < dims {
            raw.push(radius * theta.sin());
        }
    }

    let norm = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm == 0.0 {
        return vec![0.0; dims];
    }

    raw.into_iter().map(|v| v / norm).collect()
}

pub fn sample_uniform_open01(rng: &mut ChaCha20Rng) -> f32 {
    rng.random::<f32>().clamp(f32::EPSILON, 1.0 - f32::EPSILON)
}

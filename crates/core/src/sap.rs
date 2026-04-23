use anyhow::{Result, bail};
use rand::Rng;

use crate::prf::{derive_rng, sample_uniform_open01, sample_unit_vector};
use crate::{EmbeddingEncryptionScheme, EncryptedEmbedding};

#[derive(Debug, Clone)]
pub struct SapKey {
    pub scale_factor: f32,
    pub beta: f32,
    pub seed_key: [u8; 32],
}

impl SapKey {
    pub fn generate(scale_factor: f32, beta: f32) -> Self {
        let mut seed_key = [0_u8; 32];
        rand::rng().fill(&mut seed_key);
        Self {
            scale_factor,
            beta,
            seed_key,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SapScheme {
    key: SapKey,
}

impl SapScheme {
    pub fn new(key: SapKey) -> Self {
        Self { key }
    }

    fn encrypt_with_nonce(&self, embedding: &[f32], nonce: [u8; 16]) -> Result<EncryptedEmbedding> {
        if embedding.is_empty() {
            bail!("embedding must not be empty");
        }

        let dims = embedding.len();
        let mut direction_rng = derive_rng(&self.key.seed_key, &nonce, b"sap:direction")?;
        let mut radius_rng = derive_rng(&self.key.seed_key, &nonce, b"sap:radius")?;

        let unit = sample_unit_vector(&mut direction_rng, dims);
        let u = sample_uniform_open01(&mut radius_rng);
        let radius = u.powf(1.0 / dims as f32) * self.key.scale_factor * self.key.beta / 4.0;

        let vector = embedding
            .iter()
            .zip(unit.iter())
            .map(|(value, noise)| value * self.key.scale_factor + noise * radius)
            .collect();

        Ok(EncryptedEmbedding {
            scheme: "sap",
            vector,
            nonce: nonce.to_vec(),
            original_dimension: dims,
        })
    }

    fn decrypt_with_nonce(&self, ciphertext: &EncryptedEmbedding) -> Result<Vec<f32>> {
        if ciphertext.scheme != "sap" {
            bail!("ciphertext scheme mismatch: expected sap");
        }
        if ciphertext.vector.is_empty() {
            bail!("ciphertext vector must not be empty");
        }

        let nonce: [u8; 16] = ciphertext
            .nonce
            .clone()
            .try_into()
            .map_err(|_| anyhow::anyhow!("sap nonce must be 16 bytes"))?;

        let dims = ciphertext.vector.len();
        let mut direction_rng = derive_rng(&self.key.seed_key, &nonce, b"sap:direction")?;
        let mut radius_rng = derive_rng(&self.key.seed_key, &nonce, b"sap:radius")?;
        let unit = sample_unit_vector(&mut direction_rng, dims);
        let u = sample_uniform_open01(&mut radius_rng);
        let radius = u.powf(1.0 / dims as f32) * self.key.scale_factor * self.key.beta / 4.0;

        Ok(ciphertext
            .vector
            .iter()
            .zip(unit.iter())
            .map(|(value, noise)| (value - noise * radius) / self.key.scale_factor)
            .collect())
    }
}

impl EmbeddingEncryptionScheme for SapScheme {
    fn scheme_name(&self) -> &'static str {
        "sap"
    }

    fn encrypt_document(&mut self, embedding: &[f32]) -> Result<EncryptedEmbedding> {
        let mut nonce = [0_u8; 16];
        rand::rng().fill(&mut nonce);
        self.encrypt_with_nonce(embedding, nonce)
    }

    fn encrypt_query(&mut self, embedding: &[f32]) -> Result<EncryptedEmbedding> {
        let mut nonce = [0_u8; 16];
        rand::rng().fill(&mut nonce);
        self.encrypt_with_nonce(embedding, nonce)
    }

    fn decrypt_document(&mut self, ciphertext: &EncryptedEmbedding) -> Result<Vec<f32>> {
        self.decrypt_with_nonce(ciphertext)
    }
}

#[cfg(test)]
mod tests {
    use super::{SapKey, SapScheme};
    use crate::EmbeddingEncryptionScheme;

    #[test]
    fn sap_round_trip_recovers_vector() {
        let key = SapKey::generate(10.0, 0.2);
        let mut scheme = SapScheme::new(key);
        let plain = vec![0.25, -0.5, 0.75, 0.1];

        let cipher = scheme.encrypt_document(&plain).unwrap();
        let recovered = scheme.decrypt_document(&cipher).unwrap();

        for (lhs, rhs) in plain.iter().zip(recovered.iter()) {
            assert!((lhs - rhs).abs() < 1e-5);
        }
    }
}

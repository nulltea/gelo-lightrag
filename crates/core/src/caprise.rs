use anyhow::{Result, bail};
use rand::Rng;

use crate::prf::{derive_rng, sample_uniform_open01, sample_unit_vector};
use crate::{EmbeddingEncryptionScheme, EncryptedEmbedding};

#[derive(Debug, Clone)]
pub struct CapriseKey {
    pub scale_factor: f32,
    pub beta: f32,
    pub seed_key: [u8; 32],
}

impl CapriseKey {
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
pub struct Caprise {
    key: CapriseKey,
}

impl Caprise {
    pub fn new(key: CapriseKey) -> Self {
        Self { key }
    }

    fn encrypt_internal(
        &self,
        embedding: &[f32],
        nonce: [u8; 16],
        label: &[u8],
        coefficient: f32,
        scheme: &'static str,
    ) -> Result<EncryptedEmbedding> {
        if embedding.is_empty() {
            bail!("embedding must not be empty");
        }

        let dims = embedding.len();
        let mut direction_rng = derive_rng(&self.key.seed_key, &nonce, label)?;
        let mut radius_rng = derive_rng(&self.key.seed_key, &nonce, b"caprise:radius")?;
        let direction = sample_unit_vector(&mut direction_rng, dims);
        let u = sample_uniform_open01(&mut radius_rng);
        let radius =
            coefficient * self.key.scale_factor * self.key.beta * u.powf(1.0 / dims as f32);

        let vector = embedding
            .iter()
            .zip(direction.iter())
            .map(|(value, noise)| value * self.key.scale_factor + noise * radius)
            .collect();

        Ok(EncryptedEmbedding {
            scheme,
            vector,
            nonce: nonce.to_vec(),
            original_dimension: dims,
        })
    }
}

impl EmbeddingEncryptionScheme for Caprise {
    fn scheme_name(&self) -> &'static str {
        "caprise"
    }

    fn encrypt_document(&mut self, embedding: &[f32]) -> Result<EncryptedEmbedding> {
        let mut nonce = [0_u8; 16];
        rand::rng().fill(&mut nonce);
        self.encrypt_internal(embedding, nonce, b"caprise:db", 3.0 / 8.0, "caprise-db")
    }

    fn encrypt_query(&mut self, embedding: &[f32]) -> Result<EncryptedEmbedding> {
        let mut nonce = [0_u8; 16];
        rand::rng().fill(&mut nonce);
        self.encrypt_internal(
            embedding,
            nonce,
            b"caprise:query",
            1.0 / 8.0,
            "caprise-query",
        )
    }

    fn decrypt_document(&mut self, ciphertext: &EncryptedEmbedding) -> Result<Vec<f32>> {
        if ciphertext.scheme != "caprise-db" {
            bail!("ciphertext scheme mismatch: expected caprise-db");
        }

        let nonce: [u8; 16] = ciphertext
            .nonce
            .clone()
            .try_into()
            .map_err(|_| anyhow::anyhow!("caprise nonce must be 16 bytes"))?;
        let dims = ciphertext.vector.len();
        let mut direction_rng = derive_rng(&self.key.seed_key, &nonce, b"caprise:db")?;
        let mut radius_rng = derive_rng(&self.key.seed_key, &nonce, b"caprise:radius")?;
        let direction = sample_unit_vector(&mut direction_rng, dims);
        let u = sample_uniform_open01(&mut radius_rng);
        let radius =
            (3.0 / 8.0) * self.key.scale_factor * self.key.beta * u.powf(1.0 / dims as f32);

        Ok(ciphertext
            .vector
            .iter()
            .zip(direction.iter())
            .map(|(value, noise)| (value - noise * radius) / self.key.scale_factor)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::{Caprise, CapriseKey};
    use crate::EmbeddingEncryptionScheme;

    #[test]
    fn caprise_uses_distinct_query_and_document_schemes() {
        let mut scheme = Caprise::new(CapriseKey::generate(10.0, 0.2));
        let vector = vec![0.3, 0.4, 0.5];

        let doc = scheme.encrypt_document(&vector).unwrap();
        let query = scheme.encrypt_query(&vector).unwrap();

        assert_eq!(doc.scheme, "caprise-db");
        assert_eq!(query.scheme, "caprise-query");
    }
}

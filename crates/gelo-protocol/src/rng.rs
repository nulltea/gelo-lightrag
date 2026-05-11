use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// 32-byte seed used to derive a per-batch mask RNG.
#[derive(Debug, Clone, Copy)]
pub struct MaskSeed(pub [u8; 32]);

impl MaskSeed {
    /// Draw a fresh seed from the OS CSPRNG.
    pub fn from_os_rng() -> Self {
        let mut seed = [0u8; 32];
        rand::rng().fill_bytes(&mut seed);
        Self(seed)
    }

    /// Deterministic seed for tests / parity checks.
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Build a [`ChaCha20Rng`] from this seed.
    pub fn rng(&self) -> ChaCha20Rng {
        ChaCha20Rng::from_seed(self.0)
    }
}

impl Default for MaskSeed {
    fn default() -> Self {
        Self::from_os_rng()
    }
}

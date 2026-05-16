//! Synthetic KG fixture builder. Deterministic given a seed; sizes
//! scale linearly off `n_entities`.

use light_kg_store::{Chunk, Entity, ExtractedKg, Relation};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// One sizing knob → full KG. Entity count drives chunks (×1.5) and
/// relations (×0.33), so a 100-entity KG carries 150 chunks + 33
/// relations — a coarse match to LightRAG-paper corpora.
pub struct SynthConfig {
    pub n_entities: usize,
    pub dim: usize,
    pub seed: [u8; 32],
}

impl SynthConfig {
    pub fn new(n_entities: usize, dim: usize) -> Self {
        let mut seed = [0u8; 32];
        seed[..4].copy_from_slice(&(n_entities as u32).to_le_bytes());
        seed[4..8].copy_from_slice(&(dim as u32).to_le_bytes());
        Self {
            n_entities,
            dim,
            seed,
        }
    }

    pub fn n_chunks(&self) -> usize {
        (self.n_entities as f32 * 1.5).ceil() as usize
    }

    pub fn n_relations(&self) -> usize {
        (self.n_entities as f32 * 0.33).ceil() as usize
    }
}

pub fn build_kg(cfg: &SynthConfig) -> ExtractedKg {
    let mut rng = ChaCha20Rng::from_seed(cfg.seed);
    let n_chunks = cfg.n_chunks();
    let n_relations = cfg.n_relations();

    let chunks: Vec<Chunk> = (0..n_chunks)
        .map(|i| Chunk {
            id: format!("chunk-{i:06}"),
            text: format!(
                "Body of chunk {i}. lorem ipsum dolor sit amet consectetur \
                 adipiscing elit. {i} {i} {i}"
            ),
            embedding: random_unit_vec(&mut rng, cfg.dim),
        })
        .collect();

    let entities: Vec<Entity> = (0..cfg.n_entities)
        .map(|i| Entity {
            name: format!("entity-{i:06}"),
            description: format!("Entity {i}: synthesised for bench."),
            embedding: random_unit_vec(&mut rng, cfg.dim),
            source_chunks: vec![
                chunks[i % chunks.len()].id.clone(),
                chunks[(i * 7 + 3) % chunks.len()].id.clone(),
            ],
        })
        .collect();

    let mut relations = Vec::with_capacity(n_relations);
    for i in 0..n_relations {
        let a = format!("entity-{:06}", i % cfg.n_entities);
        let b = format!(
            "entity-{:06}",
            (i * 13 + 5) % cfg.n_entities
        );
        if a == b {
            continue;
        }
        relations.push(Relation {
            src: a,
            tgt: b,
            description: format!("relation {i}"),
            embedding: random_unit_vec(&mut rng, cfg.dim),
            source_chunks: vec![chunks[i % chunks.len()].id.clone()],
        });
    }

    ExtractedKg {
        chunks,
        entities,
        relations,
    }
}

pub fn random_unit_vec(rng: &mut ChaCha20Rng, dim: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

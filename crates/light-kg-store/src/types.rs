//! Plaintext KG types shipped by the client to the CVM over RATLS.
//! These are the contents of `ExtractedKg` produced by the client-
//! side entity-extraction LLM (OQ#5, plan §9). Inside the CVM they
//! exist only for the duration of `LightKgStore::build_from_kg` and
//! are then re-encoded into the encrypted components.
//!
//! Naming convention follows upstream LightRAG closely so the Python
//! reference trace at M7 can be diffed line-by-line.

use zeroize::ZeroizeOnDrop;

/// One chunk of source text. `id` matches upstream LightRAG's
/// document-fragment id (e.g. `chunk-0007`). `embedding` is the
/// embedder's output on `text`; the client computes it before
/// shipping (CAPRISE-encrypted on the wire).
#[derive(Debug, Clone)]
pub struct Chunk {
    pub id: String,
    pub text: String,
    pub embedding: Vec<f32>,
}

/// One entity extracted from the corpus. `name` is the canonical
/// entity identifier — same string upstream LightRAG would write to
/// disk. `description` is the free-text summary; we embed it. The
/// `source_chunks` list maps the entity back to the chunks it was
/// extracted from (used by the chunk fan-out step in `kg_query`).
#[derive(Debug, Clone)]
pub struct Entity {
    pub name: String,
    pub description: String,
    pub embedding: Vec<f32>,
    pub source_chunks: Vec<String>,
}

/// One relation. The upstream LightRAG representation is a (src, tgt,
/// description) triple; we extend with an embedding (the description
/// embedding, computed by the client) and the source chunks.
#[derive(Debug, Clone)]
pub struct Relation {
    pub src: String,
    pub tgt: String,
    pub description: String,
    pub embedding: Vec<f32>,
    pub source_chunks: Vec<String>,
}

/// Full input to `LightKgStore::build_from_kg`. The client-side
/// extraction LLM produces this and ships it RATLS-encrypted to the
/// CVM. The CVM zeroizes after the build completes — see
/// `LightKgStore::build_from_kg` for the explicit drop.
#[derive(Debug, Clone, ZeroizeOnDrop)]
pub struct ExtractedKg {
    #[zeroize(skip)]
    pub chunks: Vec<Chunk>,
    #[zeroize(skip)]
    pub entities: Vec<Entity>,
    #[zeroize(skip)]
    pub relations: Vec<Relation>,
}

impl Relation {
    /// Stable canonical key for the relation. Used as both the
    /// CompassIndex block ID seed and the EMM lookup key.
    /// `"{src}\x00{tgt}"` — sorted lexicographically to make
    /// (A, B) and (B, A) equivalent (relations in upstream LightRAG
    /// are undirected; we follow).
    pub fn canonical_key(&self) -> String {
        let (a, b) = if self.src <= self.tgt {
            (&self.src, &self.tgt)
        } else {
            (&self.tgt, &self.src)
        };
        format!("{a}\x00{b}")
    }
}

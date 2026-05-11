//! Architecture-agnostic helpers shared between BERT and decoder embedders.

pub mod pool;
pub mod tokenizer;

pub use tokenizer::HfTokenizer;

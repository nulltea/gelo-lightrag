use anyhow::Result;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use crate::Embedder;

pub struct FastEmbedEmbedder {
    inner: TextEmbedding,
}

impl FastEmbedEmbedder {
    pub fn new_smallest() -> Result<Self> {
        let inner = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2Q).with_show_download_progress(false),
        )?;
        Ok(Self { inner })
    }

    pub fn new(model: EmbeddingModel) -> Result<Self> {
        let inner =
            TextEmbedding::try_new(InitOptions::new(model).with_show_download_progress(false))?;
        Ok(Self { inner })
    }
}

impl Embedder for FastEmbedEmbedder {
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.inner.embed(texts, None)
    }
}

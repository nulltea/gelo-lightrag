use std::path::Path;

use anyhow::{Context, Result, anyhow};
use tokenizers::Tokenizer;

pub struct BertTokenizer {
    inner: Tokenizer,
}

impl BertTokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| anyhow!("loading tokenizer from {}: {e}", path.display()))?;
        Ok(Self { inner })
    }

    /// Encode a single string into token ids. `add_special_tokens=true`
    /// inserts `[CLS]` / `[SEP]` as configured by the tokenizer.
    pub fn encode(&self, text: &str, max_len: usize) -> Result<Vec<u32>> {
        let encoded = self
            .inner
            .encode(text, true)
            .map_err(|e| anyhow!("tokenizer encode failure: {e}"))
            .context("encoding text")?;
        let mut ids: Vec<u32> = encoded.get_ids().to_vec();
        if ids.len() > max_len {
            ids.truncate(max_len);
            // Re-cap with [SEP] (id varies; tokenizer's special token map
            // would resolve it, but for bge-small [SEP] is id 102 for the
            // bert-base-uncased vocabulary used by BGE).
            if let Some(last) = ids.last_mut() {
                *last = 102;
            }
        }
        Ok(ids)
    }
}

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use tokenizers::Tokenizer;

/// Thin wrapper over the HuggingFace `tokenizers` crate. Works for both BERT-
/// style and decoder-LLM-style tokenizers (any `tokenizer.json`).
///
/// Cloneable so multiple embedders constructed from a single load can each
/// own a separate, cheap copy.
#[derive(Clone)]
pub struct HfTokenizer {
    inner: Tokenizer,
    /// Token id used to backfill the truncated position when `add_special_tokens`
    /// is requested. For BERT-family this is `[SEP]` (id 102 in the
    /// bert-base-uncased vocab BGE inherits). For decoder-LLM-family this is
    /// typically the EOS token. Caller sets it via [`Self::with_truncation_token`].
    truncation_close_token: Option<u32>,
}

impl HfTokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| anyhow!("loading tokenizer from {}: {e}", path.display()))?;
        Ok(Self {
            inner,
            truncation_close_token: None,
        })
    }

    pub fn with_truncation_token(mut self, id: u32) -> Self {
        self.truncation_close_token = Some(id);
        self
    }

    /// Encode a single string into token ids with special tokens added.
    pub fn encode(&self, text: &str, max_len: usize) -> Result<Vec<u32>> {
        let encoded = self
            .inner
            .encode(text, true)
            .map_err(|e| anyhow!("tokenizer encode failure: {e}"))
            .context("encoding text")?;
        let mut ids: Vec<u32> = encoded.get_ids().to_vec();
        if ids.len() > max_len {
            ids.truncate(max_len);
            if let (Some(last), Some(close)) =
                (ids.last_mut(), self.truncation_close_token)
            {
                *last = close;
            }
        }
        Ok(ids)
    }

    /// Resolve a special token id by name, e.g. `[SEP]`, `[CLS]`, `<|endoftext|>`.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}

/// Back-compat alias preserved while we finish moving callers off the BERT-only
/// name.
pub type BertTokenizer = HfTokenizer;

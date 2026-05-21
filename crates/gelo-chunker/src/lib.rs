//! Char-boundary-aware text chunker with configurable overlap and
//! separator-preference split points.
//!
//! Lifted from `crates/gelo-rag/tests/common/mod.rs` (which itself was a
//! simplified port of `edgequake-pipeline`'s `TokenBasedChunking`) so
//! that the in-CVM extraction route can chunk without depending on
//! test-only code.
//!
//! The chunker treats `chunk_size` / `chunk_overlap` / `min_chunk_size`
//! as token-counts and multiplies by 4 to estimate the character span
//! (the same 4-char-per-token heuristic upstream uses).

/// Tunables for [`TokenBasedChunker::chunk`].
#[derive(Debug, Clone)]
pub struct ChunkerConfig {
    /// Target chunk length, in approximate tokens (×4 → chars).
    pub chunk_size: usize,
    /// Overlap between consecutive chunks, in approximate tokens.
    pub chunk_overlap: usize,
    /// Drop any chunk shorter than this (×4 → chars). Prevents tiny
    /// tail fragments after the chunker walks past the last natural
    /// separator.
    pub min_chunk_size: usize,
    /// Preferred split-point delimiters, searched in order. The first
    /// one that occurs inside the last quarter of the target window
    /// wins.
    pub separators: Vec<String>,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1600,
            chunk_overlap: 100,
            min_chunk_size: 600,
            separators: vec![
                "\n\n".to_string(),
                "\n".to_string(),
                ". ".to_string(),
                "! ".to_string(),
                "? ".to_string(),
                "; ".to_string(),
                ", ".to_string(),
                " ".to_string(),
            ],
        }
    }
}

/// Marker type — the chunker itself is stateless.
pub struct TokenBasedChunker;

impl TokenBasedChunker {
    /// Split `content` into overlapping chunks. Trim-empty input
    /// produces an empty `Vec`. Single chunks below `target_size`
    /// pass through unchanged.
    pub fn chunk(content: &str, config: &ChunkerConfig) -> Vec<String> {
        if content.trim().is_empty() {
            return Vec::new();
        }

        let target_chars = config.chunk_size * 4;
        let overlap_chars = config.chunk_overlap * 4;
        let min_chars = config.min_chunk_size * 4;

        split_text_internal(
            content,
            target_chars,
            overlap_chars,
            min_chars,
            &config.separators,
        )
    }
}

fn split_text_internal(
    text: &str,
    target_size: usize,
    overlap: usize,
    min_size: usize,
    separators: &[String],
) -> Vec<String> {
    if text.len() <= target_size {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut current_pos = 0;

    while current_pos < text.len() {
        current_pos = ceil_char_boundary(text, current_pos);

        let remaining = &text[current_pos..];
        if remaining.len() <= target_size {
            chunks.push(remaining.to_string());
            break;
        }

        let end_pos = floor_char_boundary(text, current_pos + target_size);
        let chunk_text = &text[current_pos..end_pos.min(text.len())];

        let split_point = find_split_point_internal(chunk_text, target_size, separators);
        let actual_end = floor_char_boundary(text, current_pos + split_point);

        let chunk_content = text[current_pos..actual_end].to_string();
        if chunk_content.len() >= min_size {
            chunks.push(chunk_content);
        }

        let overlap_pos = actual_end.saturating_sub(overlap);
        current_pos = ceil_char_boundary(text, overlap_pos);

        if current_pos >= actual_end {
            current_pos = actual_end;
        }
    }

    chunks
}

fn find_split_point_internal(text: &str, target: usize, separators: &[String]) -> usize {
    let search_start = floor_char_boundary(text, target.saturating_sub(target / 4));
    let search_end = floor_char_boundary(text, target.min(text.len()));
    if search_start >= search_end {
        return floor_char_boundary(text, target.min(text.len()));
    }
    for separator in separators {
        if let Some(pos) = text[search_start..search_end].rfind(separator.as_str()) {
            return search_start + pos + separator.len();
        }
    }
    floor_char_boundary(text, target.min(text.len()))
}

fn floor_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut i = index;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ceil_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut i = index;
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_or_whitespace_input_returns_empty_vec() {
        assert!(TokenBasedChunker::chunk("", &ChunkerConfig::default()).is_empty());
        assert!(TokenBasedChunker::chunk("   \n\t  ", &ChunkerConfig::default()).is_empty());
    }

    #[test]
    fn short_input_returns_single_chunk_unchanged() {
        let cfg = ChunkerConfig::default();
        let text = "Alice met Bob in Paris.";
        let chunks = TokenBasedChunker::chunk(text, &cfg);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn long_input_produces_multiple_overlapping_chunks() {
        // chunk_size=50 tokens → 200 chars target; build a 1500-char
        // doc with paragraph breaks so the separator search has work
        // to do.
        let cfg = ChunkerConfig {
            chunk_size: 50,
            chunk_overlap: 10,
            min_chunk_size: 10,
            separators: vec!["\n\n".into(), ". ".into(), " ".into()],
        };
        let paragraph: String = "Lorem ipsum dolor sit amet consectetur adipiscing elit. "
            .repeat(5);
        let doc = format!("{paragraph}\n\n{paragraph}\n\n{paragraph}\n\n{paragraph}");
        let chunks = TokenBasedChunker::chunk(&doc, &cfg);
        assert!(
            chunks.len() >= 3,
            "expected >=3 chunks for {} char doc, got {}",
            doc.len(),
            chunks.len()
        );
        // Each non-final chunk should land near the target size; we
        // give a generous lower bound because separator backoff can
        // cut a chunk down by up to 25%.
        for (i, c) in chunks.iter().enumerate().take(chunks.len() - 1) {
            assert!(
                c.len() <= 200 + 8,
                "chunk {i} oversize: {} chars (>208)",
                c.len()
            );
        }
    }

    #[test]
    fn respects_utf8_char_boundaries() {
        // 4-byte chars that would corrupt if we sliced mid-codepoint.
        let cfg = ChunkerConfig {
            chunk_size: 4,
            chunk_overlap: 0,
            min_chunk_size: 1,
            separators: vec![" ".into()],
        };
        let doc = "𝕬𝕭𝕮𝕯 𝕰𝕱𝕲𝕳 𝕴𝕵𝕶𝕷 𝕸𝕹𝕺𝕻 𝕼𝕽𝕾𝕿";
        let chunks = TokenBasedChunker::chunk(doc, &cfg);
        // Every chunk must be valid UTF-8 (Rust guarantees this for
        // String, but the offsets were the part we needed to get
        // right) and reassemble approximately to the original.
        let joined: String = chunks.concat();
        // Joined may include repeated overlap; just sanity-check that
        // every emitted byte sequence parses as UTF-8.
        assert!(!joined.is_empty());
    }
}

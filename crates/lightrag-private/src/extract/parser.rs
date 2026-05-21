//! Tuple-stream parser for the upstream LightRAG entity-extraction
//! format.
//!
//! Input shape (one chunk's worth of LLM output):
//! ```text
//! ("entity"<|>Alice<|>person<|>A person ...)##
//! ("entity"<|>Bob<|>person<|>Another person ...)##
//! ("relationship"<|>Alice<|>Bob<|>met in Paris<|>meeting<|>7)
//! <|COMPLETE|>
//! ```
//!
//! Parser contract:
//! - Tolerant: every malformed record is dropped + counted, never an
//!   error. Real-world small-model output drifts; we account, not
//!   abort.
//! - Records are split on `record_delimiter` after stripping the
//!   completion marker. Partial trailing tuples (no closing `)`) are
//!   discarded.
//! - `EntityDraft::name` and `RelationDraft::src` / `tgt` are stored
//!   verbatim from the model output after whitespace trim + outer-
//!   quote strip. Upstream LightRAG canonicalises by upper-casing;
//!   we defer that to the merge step.

use super::prompt::{
    DEFAULT_COMPLETION_DELIMITER, DEFAULT_RECORD_DELIMITER, DEFAULT_TUPLE_DELIMITER,
};

/// Delimiters for the tuple-stream wire format.
#[derive(Debug, Clone)]
pub struct TupleConfig {
    pub tuple_delimiter: &'static str,
    pub record_delimiter: &'static str,
    pub completion_delimiter: &'static str,
}

impl Default for TupleConfig {
    fn default() -> Self {
        Self {
            tuple_delimiter: DEFAULT_TUPLE_DELIMITER,
            record_delimiter: DEFAULT_RECORD_DELIMITER,
            completion_delimiter: DEFAULT_COMPLETION_DELIMITER,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntityDraft {
    pub name: String,
    pub entity_type: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RelationDraft {
    pub src: String,
    pub tgt: String,
    pub description: String,
    pub keywords: String,
    pub strength: f32,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParseTrailer {
    pub saw_completion: bool,
    pub malformed_records: usize,
}

/// Parse the raw decoder output for one chunk. Returns the extracted
/// drafts plus a count of records that couldn't be parsed.
pub fn parse_tuple_stream(
    raw: &str,
    cfg: &TupleConfig,
) -> (Vec<EntityDraft>, Vec<RelationDraft>, ParseTrailer) {
    let mut trailer = ParseTrailer::default();

    // Strip the completion marker (and anything after it — small
    // models sometimes append extra text after `<|COMPLETE|>`).
    let stream = if let Some(idx) = raw.find(cfg.completion_delimiter) {
        trailer.saw_completion = true;
        &raw[..idx]
    } else {
        raw
    };

    let mut entities = Vec::new();
    let mut relations = Vec::new();

    for record in stream.split(cfg.record_delimiter) {
        let trimmed = record.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Must look like `(...)` — discard partial trailing records
        // that have no closing paren.
        let inner = match strip_outer_parens(trimmed) {
            Some(s) => s,
            None => {
                trailer.malformed_records += 1;
                continue;
            }
        };
        let fields: Vec<&str> = inner.split(cfg.tuple_delimiter).collect();
        if fields.is_empty() {
            trailer.malformed_records += 1;
            continue;
        }
        let kind = unquote(fields[0].trim());
        match kind.as_str() {
            "entity" => match parse_entity(&fields[1..]) {
                Some(e) => entities.push(e),
                None => trailer.malformed_records += 1,
            },
            "relationship" => match parse_relationship(&fields[1..]) {
                Some(r) => relations.push(r),
                None => trailer.malformed_records += 1,
            },
            _ => trailer.malformed_records += 1,
        }
    }

    (entities, relations, trailer)
}

/// Strip a leading `(` and trailing `)`. Returns `None` if either is
/// missing or if `s` is shorter than 2 chars.
fn strip_outer_parens(s: &str) -> Option<&str> {
    let s = s.trim();
    if !s.starts_with('(') || !s.ends_with(')') {
        return None;
    }
    // Indices: `(` is 1 byte, `)` is 1 byte (both ASCII).
    if s.len() < 2 {
        return None;
    }
    Some(&s[1..s.len() - 1])
}

/// Strip optional surrounding straight or smart double-quotes. Small
/// models sometimes emit `"entity"`, sometimes `entity`, occasionally
/// `“entity”`.
fn unquote(s: &str) -> String {
    let s = s.trim();
    let chars: Vec<char> = s.chars().collect();
    if chars.len() >= 2 {
        let first = chars[0];
        let last = chars[chars.len() - 1];
        if (first == '"' && last == '"')
            || (first == '\'' && last == '\'')
            || (first == '“' && last == '”')
        {
            let inner: String = chars[1..chars.len() - 1].iter().collect();
            return inner.trim().to_string();
        }
    }
    s.to_string()
}

fn parse_entity(fields: &[&str]) -> Option<EntityDraft> {
    if fields.len() < 3 {
        return None;
    }
    let name = unquote(fields[0]);
    let entity_type = unquote(fields[1]);
    // Allow the description to contain `<|>` itself by re-joining the
    // tail. (Small models sometimes embed the delimiter in prose.)
    let description = if fields.len() == 3 {
        unquote(fields[2])
    } else {
        fields[2..].join(DEFAULT_TUPLE_DELIMITER).trim().to_string()
    };
    if name.is_empty() {
        return None;
    }
    Some(EntityDraft {
        name,
        entity_type,
        description,
    })
}

fn parse_relationship(fields: &[&str]) -> Option<RelationDraft> {
    if fields.len() < 5 {
        return None;
    }
    let src = unquote(fields[0]);
    let tgt = unquote(fields[1]);
    let description = unquote(fields[2]);
    let keywords = unquote(fields[3]);
    let strength_raw = unquote(fields[4]);
    let strength: f32 = strength_raw.parse().ok()?;
    if src.is_empty() || tgt.is_empty() {
        return None;
    }
    Some(RelationDraft {
        src,
        tgt,
        description,
        keywords,
        strength,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> TupleConfig {
        TupleConfig::default()
    }

    #[test]
    fn well_formed_stream_parses_all_records() {
        let raw = r#"("entity"<|>Alice<|>person<|>A person.)##("entity"<|>Bob<|>person<|>Another person.)##("relationship"<|>Alice<|>Bob<|>They met.<|>meeting<|>7)<|COMPLETE|>"#;
        let (entities, relations, t) = parse_tuple_stream(raw, &cfg());
        assert!(t.saw_completion);
        assert_eq!(t.malformed_records, 0);
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].name, "Alice");
        assert_eq!(entities[0].entity_type, "person");
        assert_eq!(entities[1].name, "Bob");
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].src, "Alice");
        assert_eq!(relations[0].tgt, "Bob");
        assert_eq!(relations[0].strength, 7.0);
    }

    #[test]
    fn missing_completion_marker_still_parses() {
        let raw = r#"("entity"<|>Alice<|>person<|>A person.)##("entity"<|>Bob<|>person<|>Another.)"#;
        let (entities, _relations, t) = parse_tuple_stream(raw, &cfg());
        assert!(!t.saw_completion);
        assert_eq!(t.malformed_records, 0);
        assert_eq!(entities.len(), 2);
    }

    #[test]
    fn partial_trailing_tuple_drops() {
        // Mid-tuple truncation — the trailing record has no `)`.
        let raw = r#"("entity"<|>Alice<|>person<|>A person.)##("entity"<|>Bob<|>person<|>unfinished"#;
        let (entities, _, t) = parse_tuple_stream(raw, &cfg());
        assert_eq!(entities.len(), 1);
        assert_eq!(t.malformed_records, 1);
    }

    #[test]
    fn malformed_strength_drops_relationship() {
        let raw = r#"("relationship"<|>A<|>B<|>desc<|>kw<|>not-a-number)<|COMPLETE|>"#;
        let (entities, relations, t) = parse_tuple_stream(raw, &cfg());
        assert!(entities.is_empty());
        assert!(relations.is_empty());
        assert_eq!(t.malformed_records, 1);
    }

    #[test]
    fn unknown_record_type_drops_and_counts() {
        let raw = r#"("ghost"<|>X<|>person<|>nope)<|COMPLETE|>"#;
        let (entities, relations, t) = parse_tuple_stream(raw, &cfg());
        assert!(entities.is_empty());
        assert!(relations.is_empty());
        assert_eq!(t.malformed_records, 1);
    }

    #[test]
    fn handles_unquoted_kind_field() {
        // Some small-model outputs drop the quotes around the kind tag.
        let raw = r#"(entity<|>Alice<|>person<|>A person.)<|COMPLETE|>"#;
        let (entities, _, t) = parse_tuple_stream(raw, &cfg());
        assert_eq!(entities.len(), 1);
        assert_eq!(t.malformed_records, 0);
    }

    #[test]
    fn entity_with_too_few_fields_drops() {
        let raw = r#"("entity"<|>OnlyName)<|COMPLETE|>"#;
        let (entities, _, t) = parse_tuple_stream(raw, &cfg());
        assert!(entities.is_empty());
        assert_eq!(t.malformed_records, 1);
    }

    #[test]
    fn empty_records_between_delimiters_are_skipped() {
        let raw = "##(\"entity\"<|>Alice<|>person<|>A person.)####<|COMPLETE|>";
        let (entities, _, t) = parse_tuple_stream(raw, &cfg());
        assert_eq!(entities.len(), 1);
        assert_eq!(t.malformed_records, 0);
    }

    #[test]
    fn text_after_completion_marker_is_ignored() {
        let raw = r#"("entity"<|>Alice<|>person<|>A person.)<|COMPLETE|>some trailing chatter from the model"#;
        let (entities, _, t) = parse_tuple_stream(raw, &cfg());
        assert!(t.saw_completion);
        assert_eq!(entities.len(), 1);
    }
}

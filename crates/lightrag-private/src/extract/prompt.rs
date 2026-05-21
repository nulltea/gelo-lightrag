//! Upstream-parity LightRAG entity-extraction prompt builder.
//!
//! Mirrors `lightrag/prompt.py::PROMPTS["entity_extraction"]` from the
//! Python reference: same delimiters, same tuple shape, same role-play
//! preamble. Kept terse — small models (Qwen3-4B) handle short prompts
//! better, and longer few-shot stretches grow TTFT on the GELO masked
//! path linearly with prompt tokens.

pub const DEFAULT_TUPLE_DELIMITER: &str = "<|>";
pub const DEFAULT_RECORD_DELIMITER: &str = "##";
pub const DEFAULT_COMPLETION_DELIMITER: &str = "<|COMPLETE|>";

/// Builds the prompt string handed to the extraction LLM for one
/// chunk. Stateless once constructed; `build` returns a fresh `String`
/// per call so the caller can tokenize it directly.
pub struct LightRagExtractionPrompt {
    entity_types: Vec<String>,
    tuple_delimiter: &'static str,
    record_delimiter: &'static str,
    completion_delimiter: &'static str,
    language: String,
}

impl LightRagExtractionPrompt {
    /// Upstream LightRAG defaults — entity types and delimiters as
    /// committed in `lightrag/prompt.py:DEFAULT_ENTITY_TYPES`.
    pub fn paper_defaults() -> Self {
        Self {
            entity_types: vec![
                "organization".to_string(),
                "person".to_string(),
                "geo".to_string(),
                "event".to_string(),
                "category".to_string(),
            ],
            tuple_delimiter: DEFAULT_TUPLE_DELIMITER,
            record_delimiter: DEFAULT_RECORD_DELIMITER,
            completion_delimiter: DEFAULT_COMPLETION_DELIMITER,
            language: "English".to_string(),
        }
    }

    pub fn with_entity_types(mut self, types: Vec<String>) -> Self {
        self.entity_types = types;
        self
    }

    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    pub fn tuple_delimiter(&self) -> &'static str {
        self.tuple_delimiter
    }
    pub fn record_delimiter(&self) -> &'static str {
        self.record_delimiter
    }
    pub fn completion_delimiter(&self) -> &'static str {
        self.completion_delimiter
    }

    /// Render the prompt for one chunk. Output is the full string to
    /// tokenize and feed to `generate` — no further wrapping.
    pub fn build(&self, chunk_text: &str) -> String {
        let entity_types_joined = self.entity_types.join(", ");
        let tup = self.tuple_delimiter;
        let rec = self.record_delimiter;
        let comp = self.completion_delimiter;
        let lang = &self.language;

        let example = format!(
            "Example:\n\
             Entity_types: organization, person, geo\n\
             Text: Alice met Bob in Paris to discuss the OpenSouce project at Acme Corp.\n\
             ######################\n\
             Output:\n\
             (\"entity\"{tup}Alice{tup}person{tup}A person who met Bob in Paris.){rec}\
             (\"entity\"{tup}Bob{tup}person{tup}A person Alice met in Paris.){rec}\
             (\"entity\"{tup}Paris{tup}geo{tup}A city where Alice and Bob met.){rec}\
             (\"entity\"{tup}Acme Corp{tup}organization{tup}A company involved in the OpenSouce project.){rec}\
             (\"relationship\"{tup}Alice{tup}Bob{tup}Alice met Bob to discuss a project.{tup}meeting, collaboration{tup}7){rec}\
             (\"relationship\"{tup}Alice{tup}Paris{tup}Alice visited Paris for a meeting.{tup}location, visit{tup}5){rec}\
             (\"relationship\"{tup}OpenSouce{tup}Acme Corp{tup}The OpenSouce project is associated with Acme Corp.{tup}project, organization{tup}6){comp}\n"
        );

        format!(
            "-Goal-\n\
             Given a text document that is potentially relevant to this activity \
             and a list of entity types, identify all entities of those types from \
             the text and all relationships among the identified entities.\n\
             \n\
             -Steps-\n\
             1. Identify all entities. For each identified entity, extract:\n\
             - entity_name: Name of the entity, capitalized as it appears in the text.\n\
             - entity_type: One of: [{entity_types_joined}]\n\
             - entity_description: Comprehensive description of the entity's attributes and activities.\n\
             Format each entity as (\"entity\"{tup}<entity_name>{tup}<entity_type>{tup}<entity_description>)\n\
             \n\
             2. From the entities identified in step 1, identify all pairs of \
             (source_entity, target_entity) that are *clearly related* in the text.\n\
             For each related pair, extract:\n\
             - source_entity: name of the source entity, as identified in step 1.\n\
             - target_entity: name of the target entity, as identified in step 1.\n\
             - relationship_description: explanation as to why source and target are related.\n\
             - relationship_keywords: comma-separated high-level keywords summarizing the relationship.\n\
             - relationship_strength: integer 1-10, strength of the relationship.\n\
             Format each relationship as (\"relationship\"{tup}<source_entity>{tup}<target_entity>{tup}<relationship_description>{tup}<relationship_keywords>{tup}<relationship_strength>)\n\
             \n\
             3. Return output in {lang} as a single list of all entities and \
             relationships identified in steps 1 and 2. Use **{rec}** as the list delimiter.\n\
             \n\
             4. When finished, output {comp}\n\
             \n\
             -Example-\n\
             {example}\n\
             -Real Data-\n\
             Entity_types: {entity_types_joined}\n\
             Text: {chunk_text}\n\
             ######################\n\
             Output:\n"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_contains_chunk_text_and_completion_marker() {
        let p = LightRagExtractionPrompt::paper_defaults();
        let s = p.build("Hello world.");
        assert!(s.contains("Hello world."));
        assert!(s.contains("<|COMPLETE|>"));
        assert!(s.contains("<|>"));
        assert!(s.contains("##"));
        // Must list the default entity types.
        for t in &["organization", "person", "geo", "event", "category"] {
            assert!(s.contains(t), "missing entity type {t} in prompt");
        }
    }

    #[test]
    fn with_entity_types_overrides_defaults() {
        let p = LightRagExtractionPrompt::paper_defaults()
            .with_entity_types(vec!["protein".to_string(), "gene".to_string()]);
        let s = p.build("acetyl-CoA carboxylase");
        // The custom types must appear in the "Entity_types:" lines
        // (both in the Steps block and the Real Data block — there
        // are two occurrences).
        assert!(s.matches("protein, gene").count() >= 2);
    }
}

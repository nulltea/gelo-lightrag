//! In-CVM entity/relation extraction from raw chunk text via a
//! GELO-masked LLM.
//!
//! Closes OQ#5 (variant-A plan §9): instead of the client extracting
//! entities/relations on plaintext and shipping an `ExtractedKg` over
//! RATLS, the CVM itself runs the extraction LLM (Qwen3-4B) under the
//! masked `InProcessTrustedExecutor` and assembles the same
//! `ExtractedKg` shape.
//!
//! Output format mirrors upstream LightRAG's tuple-delimited stream
//! (`("entity"<|>name<|>type<|>desc)##("relationship"<|>src<|>tgt<|>desc<|>kw<|>strength)##…<|COMPLETE|>`)
//! so the M7 trace stays diffable against the Python reference.

pub mod orchestrator;
pub mod parser;
pub mod prompt;

pub use orchestrator::{
    ChunkInput, ChunkTiming, DecoderOutput, DecoderTiming, DescriptionEmbedder, ExtractionConfig,
    ExtractionDecoder, ExtractionReport, MergePolicy, extract_kg_from_chunks,
};
pub use parser::{EntityDraft, ParseTrailer, RelationDraft, TupleConfig, parse_tuple_stream};
pub use prompt::{DEFAULT_COMPLETION_DELIMITER, DEFAULT_RECORD_DELIMITER, DEFAULT_TUPLE_DELIMITER, LightRagExtractionPrompt};

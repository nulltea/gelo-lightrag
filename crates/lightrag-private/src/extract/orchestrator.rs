//! Per-chunk extraction loop + cross-chunk merge → `ExtractedKg`.
//!
//! Stays free of any concrete LLM/embedder dependency by going through
//! the [`ExtractionDecoder`] and [`DescriptionEmbedder`] traits. The
//! runtime adapters (over `gelo-embedder`'s `generate` and
//! `GeloQwenEmbedder::embed`) live in `gelo-snp-runner` so this crate
//! doesn't need to compile the embedder graph.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use light_kg_store::{Chunk, Entity, ExtractedKg, Relation};

use super::parser::{EntityDraft, ParseTrailer, RelationDraft, TupleConfig, parse_tuple_stream};
use super::prompt::LightRagExtractionPrompt;

/// Decoder-side abstraction: prompt → raw string output. The concrete
/// runtime impl is responsible for tokenize → `generate` → decode and
/// for tracking `stopped_on_eos` (it should mark `Truncated` when the
/// loop hit `max_tokens` before the completion marker).
pub trait ExtractionDecoder {
    /// Run one extraction prompt. Returns the decoded output text +
    /// whether generation hit the EOS / completion-marker stop
    /// condition (`true`) or was truncated at `max_tokens` (`false`).
    fn generate_extraction(
        &mut self,
        prompt: &str,
        max_tokens: usize,
    ) -> anyhow::Result<DecoderOutput>;
}

#[derive(Debug, Clone, Default)]
pub struct DecoderOutput {
    pub text: String,
    pub stopped_on_eos: bool,
    /// Per-stage timings for the decoder run. Defaults to all-zero
    /// when the impl doesn't instrument (mock decoders in tests).
    pub timing: DecoderTiming,
}

/// Fine-grained timings for one decoder invocation. Populated by the
/// runtime adapter (`DecoderRuntime` in `gelo-snp-runner`); left as
/// zero by other impls.
#[derive(Debug, Clone, Copy, Default)]
pub struct DecoderTiming {
    /// `tokenizer.encode(prompt, ...)`.
    pub tokenize: Duration,
    /// `decoder::generation::generate(...)` — prefill + decode loop.
    pub generate: Duration,
    /// `tokenizer.decode(&output_tokens, ...)`.
    pub decode: Duration,
    pub prompt_tokens: usize,
    pub output_tokens: usize,
}

/// Embedder-side abstraction: batch text → row-major embeddings. The
/// runtime impl wraps `GeloQwenEmbedder::embed`.
pub trait DescriptionEmbedder {
    fn embed_batch(&mut self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>>;
}

#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    /// Per-chunk hard cap for output tokens. Default 1024 — generous
    /// enough for chunks at the default `chunk_size=1600` (8 tokens
    /// per entity/relation tuple × ~30 records).
    pub max_tokens_per_chunk: usize,
    pub merge: MergePolicy,
    pub tuple_config: TupleConfig,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            max_tokens_per_chunk: 1024,
            merge: MergePolicy::FirstWinsAppendDescription,
            tuple_config: TupleConfig::default(),
        }
    }
}

/// Cross-chunk merge strategy for the same canonicalised entity name
/// (or sorted relation key).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergePolicy {
    /// First chunk's `entity_type` and `description` win. Subsequent
    /// chunks append their description (newline-separated) and union
    /// `source_chunks`.
    FirstWinsAppendDescription,
}

#[derive(Debug, Clone)]
pub struct ChunkInput {
    pub id: String,
    pub text: String,
}

#[derive(Debug, Clone, Default)]
pub struct ExtractionReport {
    pub chunks_processed: usize,
    pub chunks_skipped_empty: usize,
    pub generations_truncated: usize,
    pub malformed_records_total: usize,
    pub dropped_dangling_relations_total: usize,
    pub per_chunk_trailers: Vec<ParseTrailer>,
    /// Per-chunk wall-clock breakdown — populated for every chunk
    /// passed through `extract_kg_from_chunks` in the same order as
    /// `per_chunk_trailers`.
    pub chunk_timings: Vec<ChunkTiming>,
    /// Time spent in the cross-chunk entity + relation merge loop
    /// (after every per-chunk parse, before description embedding).
    pub merge: Duration,
    /// Time spent dropping dangling relations.
    pub drop_dangling: Duration,
    /// One `embed_batch` call over all chunk bodies.
    pub embed_chunks: Duration,
    /// One `embed_batch` call over all entity descriptions.
    pub embed_entities: Duration,
    /// One `embed_batch` call over all relation descriptions.
    pub embed_relations: Duration,
    /// Final `Vec<Chunk>` / `Vec<Entity>` / `Vec<Relation>` assembly
    /// (zipping embeddings into the store types).
    pub assemble: Duration,
    /// Total wall-clock for `extract_kg_from_chunks` (everything
    /// above plus orchestration overhead).
    pub total: Duration,
}

/// Per-chunk wall-clock breakdown. Captured during the per-chunk loop
/// in `extract_kg_from_chunks`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChunkTiming {
    /// `LightRagExtractionPrompt::build` — pure string-formatting.
    pub prompt_build: Duration,
    /// Decoder call (tokenize + generate + decode). For the
    /// runtime adapter, sub-stages are in `decoder_sub`.
    pub decoder_total: Duration,
    /// Sub-stages of the decoder call. Zero when the impl doesn't
    /// instrument.
    pub decoder_sub: DecoderTiming,
    /// `parse_tuple_stream` over the decoded text.
    pub parse: Duration,
    pub entities_extracted: usize,
    pub relations_extracted: usize,
    pub stopped_on_eos: bool,
}

/// Drive the extraction LLM across every chunk, merge into a single
/// `ExtractedKg`, embed every chunk/entity/relation description.
///
/// Memory note: all chunk embeddings + description embeddings stay in
/// memory until the call returns. Caller-side parallelism is *not*
/// implemented — `decoder` is `&mut`, so chunks are serialized.
pub fn extract_kg_from_chunks<D, E>(
    chunks: Vec<ChunkInput>,
    decoder: &mut D,
    embedder: &mut E,
    cfg: &ExtractionConfig,
) -> anyhow::Result<(ExtractedKg, ExtractionReport)>
where
    D: ExtractionDecoder,
    E: DescriptionEmbedder,
{
    let prompt_builder = LightRagExtractionPrompt::paper_defaults();
    let mut report = ExtractionReport::default();
    let t_total = Instant::now();

    // Cross-chunk accumulators. BTreeMap for stable iteration order
    // (so embedding batches + ExtractedKg row order are deterministic
    // given the same per-chunk draft order).
    let mut entities: BTreeMap<String, MergedEntity> = BTreeMap::new();
    let mut relations: BTreeMap<(String, String), MergedRelation> = BTreeMap::new();

    // Filter empty chunks up-front so the chunk-embedding batch stays
    // aligned with the surviving chunk list.
    let mut live_chunks: Vec<ChunkInput> = Vec::with_capacity(chunks.len());
    for c in chunks {
        if c.text.trim().is_empty() {
            report.chunks_skipped_empty += 1;
            continue;
        }
        live_chunks.push(c);
    }

    // Run extraction per chunk, accumulating drafts.
    let mut merge_acc = Duration::ZERO;
    let total_chunks = live_chunks.len();
    for (idx, chunk) in live_chunks.iter().enumerate() {
        // Per-chunk GELO+forward profile breakdown — reset
        // gelo_protocol::profile thread-local before the chunk so
        // the dump below reflects exactly this chunk's mask /
        // attention / matmul stages. Decode runs on the same thread
        // as the orchestrator loop (`extract_kg_from_chunks` is
        // called directly, no `spawn_blocking`), so the thread-local
        // captures the decoder's `profile::time` calls.
        gelo_protocol::profile::reset();
        tracing::info!(
            target: "lightrag_private::extract",
            chunk_idx = idx + 1,
            chunk_total = total_chunks,
            chunk_id = %chunk.id,
            chunk_chars = chunk.text.len(),
            "extract: chunk start"
        );
        let mut timing = ChunkTiming::default();
        let t = Instant::now();
        let prompt = prompt_builder.build(&chunk.text);
        timing.prompt_build = t.elapsed();

        let t = Instant::now();
        let out = decoder.generate_extraction(&prompt, cfg.max_tokens_per_chunk)?;
        timing.decoder_total = t.elapsed();
        timing.decoder_sub = out.timing;
        timing.stopped_on_eos = out.stopped_on_eos;
        if !out.stopped_on_eos {
            report.generations_truncated += 1;
        }

        let t = Instant::now();
        let (draft_entities, draft_relations, trailer) =
            parse_tuple_stream(&out.text, &cfg.tuple_config);
        timing.parse = t.elapsed();
        timing.entities_extracted = draft_entities.len();
        timing.relations_extracted = draft_relations.len();

        // Emit the decoded text at DEBUG when nothing parsed —
        // makes silent-failure cases ("model output garbled" vs
        // "parser bug" vs "model returned wrong format") debuggable
        // without a separate dump pass. At INFO level we still see
        // the chunk_done summary with the counts.
        if draft_entities.is_empty() && draft_relations.is_empty() {
            tracing::info!(
                target: "lightrag_private::extract",
                chunk_idx = idx + 1,
                chunk_id = %chunk.id,
                output_chars = out.text.chars().count(),
                "extract: 0 entities + 0 relations — model output follows"
            );
            tracing::info!(
                target: "lightrag_private::extract",
                chunk_idx = idx + 1,
                chunk_id = %chunk.id,
                output_preview = %out.text.chars().take(800).collect::<String>(),
                "extract: decoded output (first 800 chars)"
            );
        }

        report.malformed_records_total += trailer.malformed_records;
        report.per_chunk_trailers.push(trailer);

        let t = Instant::now();
        merge_entities(&mut entities, draft_entities, &chunk.id, cfg.merge);
        merge_relations(&mut relations, draft_relations, &chunk.id, cfg.merge);
        merge_acc += t.elapsed();

        // Mask shape diagnostic — `s = n + k_shield` with k_shield = 8
        // (paper-parity default in `InProcessTrustedExecutor`).
        // s_pad is the pow2 mask side HD₃ would use. Auto picks HD₃
        // when `s_pad * 3 <= s * 4` (pad ratio ≤ 4/3), DCT-IV
        // otherwise. See `mask.rs::resolve_mask_kind_for_shape`.
        let s = timing.decoder_sub.prompt_tokens + 8;
        let s_pad = s.next_power_of_two().max(2);
        tracing::info!(
            target: "lightrag_private::extract",
            chunk_idx = idx + 1,
            chunk_total = total_chunks,
            chunk_id = %chunk.id,
            decoder_total_ms = timing.decoder_total.as_millis() as u64,
            generate_ms = timing.decoder_sub.generate.as_millis() as u64,
            prompt_tokens = timing.decoder_sub.prompt_tokens,
            output_tokens = timing.decoder_sub.output_tokens,
            mask_s_prefill = s,
            mask_s_pad_prefill = s_pad,
            mask_pad_ratio_x1000 = (s_pad * 1000 / s.max(1)) as u64,
            entities = timing.entities_extracted,
            relations = timing.relations_extracted,
            stopped_on_eos = timing.stopped_on_eos,
            "extract: chunk done"
        );
        // Dump the per-chunk profile breakdown to stderr — GELO
        // mask cost, TEE-side ops (RMSNorm, RoPE, attention,
        // qkv_direct, embed_lookup), GPU offload duration. The
        // GPU util observed at the OS level (nvtop) is the
        // (GPU bucket) / (total wall) ratio; this dump pinpoints
        // which non-GPU buckets are eating wall time.
        gelo_protocol::profile::snapshot()
            .dump(&format!("chunk-{:06} gelo+forward profile", idx));

        report.chunk_timings.push(timing);
        report.chunks_processed += 1;
    }
    report.merge = merge_acc;

    // Drop dangling relations (src/tgt not in the entity set after
    // the full-document pass).
    let t = Instant::now();
    let entity_names: std::collections::HashSet<&str> =
        entities.keys().map(|s| s.as_str()).collect();
    relations.retain(|_, r| {
        let keep =
            entity_names.contains(r.src.as_str()) && entity_names.contains(r.tgt.as_str());
        if !keep {
            report.dropped_dangling_relations_total += 1;
        }
        keep
    });
    report.drop_dangling = t.elapsed();

    // Batch-embed chunk bodies.
    let chunk_texts: Vec<String> = live_chunks.iter().map(|c| c.text.clone()).collect();
    let t = Instant::now();
    let chunk_embeddings = embedder.embed_batch(&chunk_texts)?;
    report.embed_chunks = t.elapsed();
    if chunk_embeddings.len() != chunk_texts.len() {
        anyhow::bail!(
            "embedder returned {} rows for {} chunk texts",
            chunk_embeddings.len(),
            chunk_texts.len()
        );
    }

    // Batch-embed entity descriptions.
    let entity_keys: Vec<String> = entities.keys().cloned().collect();
    let entity_texts: Vec<String> = entity_keys
        .iter()
        .map(|k| {
            let e = entities.get(k).expect("key just iterated");
            format!("{}: {}", e.name, e.description)
        })
        .collect();
    let entity_embeddings = if entity_texts.is_empty() {
        Vec::new()
    } else {
        let t = Instant::now();
        let v = embedder.embed_batch(&entity_texts)?;
        report.embed_entities = t.elapsed();
        if v.len() != entity_texts.len() {
            anyhow::bail!(
                "embedder returned {} rows for {} entity descriptions",
                v.len(),
                entity_texts.len()
            );
        }
        v
    };

    // Batch-embed relation descriptions.
    let relation_keys: Vec<(String, String)> = relations.keys().cloned().collect();
    let relation_texts: Vec<String> = relation_keys
        .iter()
        .map(|k| {
            let r = relations.get(k).expect("key just iterated");
            format!("{} -> {}: {}", r.src, r.tgt, r.description)
        })
        .collect();
    let relation_embeddings = if relation_texts.is_empty() {
        Vec::new()
    } else {
        let t = Instant::now();
        let v = embedder.embed_batch(&relation_texts)?;
        report.embed_relations = t.elapsed();
        if v.len() != relation_texts.len() {
            anyhow::bail!(
                "embedder returned {} rows for {} relation descriptions",
                v.len(),
                relation_texts.len()
            );
        }
        v
    };

    // Assemble ExtractedKg.
    let t = Instant::now();
    let kg_chunks: Vec<Chunk> = live_chunks
        .into_iter()
        .zip(chunk_embeddings.into_iter())
        .map(|(c, emb)| Chunk {
            id: c.id,
            text: c.text,
            embedding: emb,
        })
        .collect();

    let kg_entities: Vec<Entity> = entity_keys
        .into_iter()
        .zip(entity_embeddings.into_iter())
        .map(|(k, emb)| {
            let m = entities.remove(&k).expect("key from same map");
            Entity {
                name: m.name,
                description: m.description,
                embedding: emb,
                source_chunks: m.source_chunks,
            }
        })
        .collect();

    let kg_relations: Vec<Relation> = relation_keys
        .into_iter()
        .zip(relation_embeddings.into_iter())
        .map(|(k, emb)| {
            let m = relations.remove(&k).expect("key from same map");
            Relation {
                src: m.src,
                tgt: m.tgt,
                description: m.description,
                embedding: emb,
                source_chunks: m.source_chunks,
            }
        })
        .collect();
    report.assemble = t.elapsed();
    report.total = t_total.elapsed();

    Ok((
        ExtractedKg {
            chunks: kg_chunks,
            entities: kg_entities,
            relations: kg_relations,
        },
        report,
    ))
}

/// Working copy of an entity during cross-chunk accumulation.
struct MergedEntity {
    name: String,
    description: String,
    source_chunks: Vec<String>,
}

struct MergedRelation {
    src: String,
    tgt: String,
    description: String,
    source_chunks: Vec<String>,
}

fn merge_entities(
    acc: &mut BTreeMap<String, MergedEntity>,
    drafts: Vec<EntityDraft>,
    chunk_id: &str,
    policy: MergePolicy,
) {
    for d in drafts {
        let key = d.name.clone();
        match acc.get_mut(&key) {
            None => {
                acc.insert(
                    key,
                    MergedEntity {
                        name: d.name,
                        description: d.description,
                        source_chunks: vec![chunk_id.to_string()],
                    },
                );
            }
            Some(existing) => {
                match policy {
                    MergePolicy::FirstWinsAppendDescription => {
                        if !d.description.is_empty()
                            && !existing.description.contains(&d.description)
                        {
                            if existing.description.is_empty() {
                                existing.description = d.description;
                            } else {
                                existing.description.push('\n');
                                existing.description.push_str(&d.description);
                            }
                        }
                    }
                }
                if !existing.source_chunks.iter().any(|c| c == chunk_id) {
                    existing.source_chunks.push(chunk_id.to_string());
                }
            }
        }
    }
}

fn merge_relations(
    acc: &mut BTreeMap<(String, String), MergedRelation>,
    drafts: Vec<RelationDraft>,
    chunk_id: &str,
    policy: MergePolicy,
) {
    for d in drafts {
        // Canonical key — same shape as `Relation::canonical_key`
        // (sorted endpoints; undirected relations).
        let (a, b) = if d.src <= d.tgt {
            (d.src.clone(), d.tgt.clone())
        } else {
            (d.tgt.clone(), d.src.clone())
        };
        let key = (a.clone(), b.clone());
        match acc.get_mut(&key) {
            None => {
                acc.insert(
                    key,
                    MergedRelation {
                        src: a,
                        tgt: b,
                        description: d.description,
                        source_chunks: vec![chunk_id.to_string()],
                    },
                );
            }
            Some(existing) => {
                match policy {
                    MergePolicy::FirstWinsAppendDescription => {
                        if !d.description.is_empty()
                            && !existing.description.contains(&d.description)
                        {
                            if existing.description.is_empty() {
                                existing.description = d.description;
                            } else {
                                existing.description.push('\n');
                                existing.description.push_str(&d.description);
                            }
                        }
                    }
                }
                if !existing.source_chunks.iter().any(|c| c == chunk_id) {
                    existing.source_chunks.push(chunk_id.to_string());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CannedDecoder {
        scripts: Vec<String>,
        idx: usize,
    }

    impl ExtractionDecoder for CannedDecoder {
        fn generate_extraction(
            &mut self,
            _prompt: &str,
            _max_tokens: usize,
        ) -> anyhow::Result<DecoderOutput> {
            let i = self.idx;
            self.idx += 1;
            Ok(DecoderOutput {
                text: self.scripts.get(i).cloned().unwrap_or_default(),
                stopped_on_eos: true,
                ..Default::default()
            })
        }
    }

    /// One-hot embedder: index-into-vocab hash. Deterministic;
    /// length = 8.
    struct OneHotEmbedder;

    impl DescriptionEmbedder for OneHotEmbedder {
        fn embed_batch(&mut self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| {
                    let mut v = vec![0.0f32; 8];
                    let h: u32 = t.bytes().fold(0u32, |a, b| a.wrapping_add(b as u32));
                    v[(h % 8) as usize] = 1.0;
                    v
                })
                .collect())
        }
    }

    #[test]
    fn extracts_entities_and_relations_from_single_chunk() {
        let raw = r#"("entity"<|>Alice<|>person<|>A person.)##("entity"<|>Bob<|>person<|>Another.)##("relationship"<|>Alice<|>Bob<|>met<|>meeting<|>7)<|COMPLETE|>"#;
        let mut decoder = CannedDecoder {
            scripts: vec![raw.to_string()],
            idx: 0,
        };
        let mut emb = OneHotEmbedder;
        let chunks = vec![ChunkInput {
            id: "chunk-000000".into(),
            text: "Alice met Bob in Paris.".into(),
        }];
        let (kg, report) = extract_kg_from_chunks(
            chunks,
            &mut decoder,
            &mut emb,
            &ExtractionConfig::default(),
        )
        .unwrap();
        assert_eq!(report.chunks_processed, 1);
        assert_eq!(kg.chunks.len(), 1);
        assert_eq!(kg.entities.len(), 2);
        assert_eq!(kg.relations.len(), 1);
        let names: Vec<&str> = kg.entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"Alice"));
        assert!(names.contains(&"Bob"));
        // Source-chunk tracking.
        for e in &kg.entities {
            assert_eq!(e.source_chunks, vec!["chunk-000000".to_string()]);
        }
        for r in &kg.relations {
            assert_eq!(r.source_chunks, vec!["chunk-000000".to_string()]);
        }
        // Relation key canonicalised — sorted endpoints.
        assert_eq!(kg.relations[0].src, "Alice");
        assert_eq!(kg.relations[0].tgt, "Bob");
        // Embedding lengths.
        assert_eq!(kg.chunks[0].embedding.len(), 8);
        assert_eq!(kg.entities[0].embedding.len(), 8);
    }

    #[test]
    fn merges_duplicate_entities_across_chunks_first_wins() {
        let raw1 = r#"("entity"<|>Alice<|>person<|>First desc.)<|COMPLETE|>"#;
        let raw2 = r#"("entity"<|>Alice<|>person<|>Second desc.)<|COMPLETE|>"#;
        let mut decoder = CannedDecoder {
            scripts: vec![raw1.into(), raw2.into()],
            idx: 0,
        };
        let mut emb = OneHotEmbedder;
        let chunks = vec![
            ChunkInput { id: "c0".into(), text: "Alice text 1.".into() },
            ChunkInput { id: "c1".into(), text: "Alice text 2.".into() },
        ];
        let (kg, _) = extract_kg_from_chunks(
            chunks,
            &mut decoder,
            &mut emb,
            &ExtractionConfig::default(),
        )
        .unwrap();
        assert_eq!(kg.entities.len(), 1);
        let alice = &kg.entities[0];
        assert!(alice.description.contains("First desc."));
        assert!(alice.description.contains("Second desc."));
        assert_eq!(alice.source_chunks, vec!["c0".to_string(), "c1".to_string()]);
    }

    #[test]
    fn dangling_relations_are_dropped() {
        // Relation references an entity ("Ghost") that never appears
        // in any entity record.
        let raw = r#"("entity"<|>Alice<|>person<|>A person.)##("relationship"<|>Alice<|>Ghost<|>nope<|>kw<|>3)<|COMPLETE|>"#;
        let mut decoder = CannedDecoder {
            scripts: vec![raw.into()],
            idx: 0,
        };
        let mut emb = OneHotEmbedder;
        let chunks = vec![ChunkInput { id: "c0".into(), text: "Some text.".into() }];
        let (kg, report) = extract_kg_from_chunks(
            chunks,
            &mut decoder,
            &mut emb,
            &ExtractionConfig::default(),
        )
        .unwrap();
        assert_eq!(kg.entities.len(), 1);
        assert!(kg.relations.is_empty());
        assert_eq!(report.dropped_dangling_relations_total, 1);
    }

    #[test]
    fn empty_chunks_are_skipped() {
        let mut decoder = CannedDecoder {
            scripts: vec![r#"("entity"<|>X<|>category<|>x)<|COMPLETE|>"#.into()],
            idx: 0,
        };
        let mut emb = OneHotEmbedder;
        let chunks = vec![
            ChunkInput { id: "empty".into(), text: "   ".into() },
            ChunkInput { id: "real".into(), text: "real text".into() },
        ];
        let (kg, report) = extract_kg_from_chunks(
            chunks,
            &mut decoder,
            &mut emb,
            &ExtractionConfig::default(),
        )
        .unwrap();
        assert_eq!(report.chunks_skipped_empty, 1);
        assert_eq!(report.chunks_processed, 1);
        assert_eq!(kg.chunks.len(), 1);
        assert_eq!(kg.chunks[0].id, "real");
    }

    #[test]
    fn truncated_generation_is_counted_but_not_an_error() {
        struct TruncatedDecoder;
        impl ExtractionDecoder for TruncatedDecoder {
            fn generate_extraction(
                &mut self,
                _prompt: &str,
                _max_tokens: usize,
            ) -> anyhow::Result<DecoderOutput> {
                Ok(DecoderOutput {
                    text: r#"("entity"<|>Alice<|>person<|>A person.)"#.into(),
                    stopped_on_eos: false,
                    ..Default::default()
                })
            }
        }
        let mut emb = OneHotEmbedder;
        let chunks = vec![ChunkInput { id: "c0".into(), text: "Alice.".into() }];
        let (kg, report) = extract_kg_from_chunks(
            chunks,
            &mut TruncatedDecoder,
            &mut emb,
            &ExtractionConfig::default(),
        )
        .unwrap();
        assert_eq!(report.generations_truncated, 1);
        assert_eq!(kg.entities.len(), 1);
    }
}

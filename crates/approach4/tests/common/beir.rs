//! BEIR JSONL + qrels loader for the M7.3 accuracy bench.
//!
//! BEIR datasets are canonically distributed as zip archives via the
//! UKP-TUDA mirror (referenced by the BEIR README):
//! `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip`
//!
//! Each zip contains:
//! - `{name}/corpus.jsonl`  — one JSON object per doc: `{_id, title, text}`
//! - `{name}/queries.jsonl` — one JSON object per query: `{_id, text}`
//! - `{name}/qrels/test.tsv` — query-id<TAB>doc-id<TAB>relevance lines
//!
//! The HuggingFace `BeIR/*` datasets are in parquet format which would
//! require a heavier dep tree to parse; the UKP zip is JSONL and ~3 MB
//! for NFCorpus. We fetch + extract once, cache under
//! `target/beir-cache/{name}/`.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use rag_core::{ChunkId, DocumentChunk};
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct BeirDataset {
    pub name: &'static str,
    pub docs: Vec<DocumentChunk>,
    pub queries: Vec<(String, String)>, // (query_id, query_text)
    /// `qrels[query_id]` → map of `doc_id` → graded relevance (0 = not
    /// relevant, ≥1 = relevant; some datasets are binary, others graded).
    pub qrels: HashMap<String, HashMap<String, u8>>,
}

/// Load BEIR's NFCorpus (3,633 docs, 323 test queries, graded qrels).
/// Cached under `target/beir-cache/nfcorpus/` after the first run.
pub fn load_nfcorpus() -> Result<BeirDataset> {
    load("nfcorpus")
}

/// Generic loader for any BeIR/* dataset that follows the standard
/// layout. Most BEIR datasets do; some (TREC-COVID round-2, BioASQ)
/// have extra files we ignore.
pub fn load(name: &'static str) -> Result<BeirDataset> {
    let cache_dir = cache_root().join(name);
    fs::create_dir_all(&cache_dir).with_context(|| {
        format!("creating cache dir {}", cache_dir.display())
    })?;

    // Single zip download → extract once → all three files cached locally.
    ensure_dataset_extracted(&cache_dir, name)?;
    let corpus_path = cache_dir.join("corpus.jsonl");
    let queries_path = cache_dir.join("queries.jsonl");
    let qrels_path = cache_dir.join("qrels").join("test.tsv");
    for p in [&corpus_path, &queries_path, &qrels_path] {
        anyhow::ensure!(
            p.exists(),
            "expected BEIR file missing after extraction: {}",
            p.display()
        );
    }

    let docs = parse_corpus(&corpus_path)
        .with_context(|| format!("parsing {}", corpus_path.display()))?;
    let queries = parse_queries(&queries_path)
        .with_context(|| format!("parsing {}", queries_path.display()))?;
    let qrels = parse_qrels(&qrels_path)
        .with_context(|| format!("parsing {}", qrels_path.display()))?;

    Ok(BeirDataset {
        name,
        docs,
        queries,
        qrels,
    })
}

fn cache_root() -> PathBuf {
    // The integration-test binary runs from inside `target/`; walk up
    // to find the workspace root cache dir.
    let mut p = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("."));
    while let Some(parent) = p.parent() {
        if parent.file_name().and_then(|s| s.to_str()) == Some("target") {
            return parent.join("beir-cache");
        }
        p = parent.to_path_buf();
    }
    PathBuf::from("target/beir-cache")
}

/// Fetch the BEIR zip for `name` from the UKP-TUDA mirror and extract
/// into `cache_dir`. No-op if already extracted (checked via marker
/// file `_extracted.ok`).
fn ensure_dataset_extracted(cache_dir: &Path, name: &str) -> Result<()> {
    let marker = cache_dir.join("_extracted.ok");
    if marker.exists() {
        return Ok(());
    }
    let url = format!(
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
    );
    let zip_path = cache_dir.join(format!("{name}.zip"));
    if !zip_path.exists() {
        eprintln!("[beir] downloading {url}");
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()?;
        let resp = client
            .get(&url)
            .send()
            .with_context(|| format!("fetching {url}"))?;
        anyhow::ensure!(
            resp.status().is_success(),
            "BEIR mirror returned {} for {url}",
            resp.status()
        );
        let bytes = resp.bytes()?;
        eprintln!("[beir]   ✓ downloaded {} bytes", bytes.len());
        let mut f = File::create(&zip_path)?;
        f.write_all(&bytes)?;
    }

    eprintln!("[beir] extracting {} ...", zip_path.display());
    let zip_file = File::open(&zip_path)?;
    let mut archive = zip::ZipArchive::new(zip_file)?;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let entry_path = match entry.enclosed_name() {
            Some(p) => p.to_path_buf(),
            None => continue,
        };
        // The zip's top-level dir is `{name}/`; strip it so files land
        // directly under cache_dir (corpus.jsonl, queries.jsonl, qrels/...).
        let stripped = entry_path
            .strip_prefix(name)
            .unwrap_or(&entry_path)
            .to_path_buf();
        if stripped.as_os_str().is_empty() {
            continue;
        }
        let out_path = cache_dir.join(&stripped);
        if entry.is_dir() {
            fs::create_dir_all(&out_path)?;
        } else {
            if let Some(parent) = out_path.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut out_file = File::create(&out_path)?;
            std::io::copy(&mut entry, &mut out_file)?;
        }
    }
    File::create(&marker)?;
    eprintln!("[beir]   ✓ extracted to {}", cache_dir.display());
    Ok(())
}

#[derive(Deserialize)]
struct CorpusRow {
    #[serde(rename = "_id")]
    id: String,
    #[serde(default)]
    title: Option<String>,
    text: String,
}

fn parse_corpus(path: &Path) -> Result<Vec<DocumentChunk>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: CorpusRow = serde_json::from_str(&line)
            .with_context(|| format!("parsing corpus line {}", lineno + 1))?;
        let text = match row.title.as_deref() {
            Some(t) if !t.is_empty() => format!("{} {}", t, row.text),
            _ => row.text,
        };
        out.push(DocumentChunk {
            id: ChunkId(row.id),
            text,
        });
    }
    Ok(out)
}

#[derive(Deserialize)]
struct QueryRow {
    #[serde(rename = "_id")]
    id: String,
    text: String,
}

fn parse_queries(path: &Path) -> Result<Vec<(String, String)>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: QueryRow = serde_json::from_str(&line)
            .with_context(|| format!("parsing query line {}", lineno + 1))?;
        out.push((row.id, row.text));
    }
    Ok(out)
}

fn parse_qrels(path: &Path) -> Result<HashMap<String, HashMap<String, u8>>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut out: HashMap<String, HashMap<String, u8>> = HashMap::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        // Skip a possible TREC-style header row.
        if lineno == 0 && line.starts_with("query-id") {
            continue;
        }
        // BEIR qrels format: query-id<TAB>doc-id<TAB>relevance.
        // Some files use spaces or have an extra "iteration" column.
        let parts: Vec<&str> = line.split('\t').collect();
        let parts: Vec<&str> = if parts.len() == 1 {
            line.split_whitespace().collect()
        } else {
            parts
        };
        let (qid, did, rel) = match parts.len() {
            3 => (parts[0], parts[1], parts[2]),
            4 => (parts[0], parts[2], parts[3]), // TREC format: qid 0 docid rel
            _ => {
                return Err(anyhow!(
                    "qrels line {} has {} columns: {:?}",
                    lineno + 1,
                    parts.len(),
                    parts
                ));
            }
        };
        let rel: u8 = rel
            .parse()
            .with_context(|| format!("parsing relevance at line {}", lineno + 1))?;
        out.entry(qid.to_string())
            .or_default()
            .insert(did.to_string(), rel);
    }
    Ok(out)
}

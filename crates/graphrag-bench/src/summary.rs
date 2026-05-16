//! Summary stats over a `Vec<StageTimings>`: p50/p95/p99/mean/stddev
//! per stage, plus the headline `layer0_reads` count.

use std::time::Duration;

use crate::stages::StageTimings;

#[derive(Debug, Clone, Copy)]
pub struct Pcts {
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub mean: f64,
    pub stddev: f64,
}

impl Pcts {
    fn from_micros(mut samples: Vec<f64>) -> Self {
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = samples.len();
        let p50 = pct(&samples, 0.50);
        let p95 = pct(&samples, 0.95);
        let p99 = pct(&samples, 0.99);
        let mean = samples.iter().sum::<f64>() / n as f64;
        let variance: f64 =
            samples.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
        let stddev = variance.sqrt();
        Pcts { p50, p95, p99, mean, stddev }
    }
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn micros(d: Duration) -> f64 {
    d.as_secs_f64() * 1_000_000.0
}

pub struct PerStagePcts {
    pub total: Pcts,
    pub perturb: Pcts,
    pub entities_search: Pcts,
    pub relations_search: Pcts,
    pub adjacency: Pcts,
    pub src_chunks: Pcts,
    pub chunk_decrypt: Pcts,
    pub layer0_reads_mean: f64,
}

impl PerStagePcts {
    pub fn from(samples: &[StageTimings]) -> Self {
        Self {
            total: Pcts::from_micros(samples.iter().map(|s| micros(s.total)).collect()),
            perturb: Pcts::from_micros(samples.iter().map(|s| micros(s.perturb)).collect()),
            entities_search: Pcts::from_micros(
                samples.iter().map(|s| micros(s.entities_search)).collect(),
            ),
            relations_search: Pcts::from_micros(
                samples.iter().map(|s| micros(s.relations_search)).collect(),
            ),
            adjacency: Pcts::from_micros(samples.iter().map(|s| micros(s.adjacency)).collect()),
            src_chunks: Pcts::from_micros(samples.iter().map(|s| micros(s.src_chunks)).collect()),
            chunk_decrypt: Pcts::from_micros(
                samples.iter().map(|s| micros(s.chunk_decrypt)).collect(),
            ),
            layer0_reads_mean: samples
                .iter()
                .map(|s| s.layer0_reads_delta as f64)
                .sum::<f64>()
                / samples.len().max(1) as f64,
        }
    }

    pub fn print_human(&self, scenario: &str, n_entities: usize, mode: &str, n_queries: usize) {
        eprintln!();
        eprintln!(
            "── {scenario}/{n_entities}-entities/{mode} — {n_queries} queries ──"
        );
        eprintln!(
            "{:<18} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "stage", "p50_us", "p95_us", "p99_us", "mean_us", "stddev_us"
        );
        let row = |label: &str, p: &Pcts| {
            eprintln!(
                "{:<18} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1}",
                label, p.p50, p.p95, p.p99, p.mean, p.stddev
            );
        };
        row("total", &self.total);
        row("  perturb", &self.perturb);
        row("  entities_search", &self.entities_search);
        row("  relations_search", &self.relations_search);
        row("  adjacency", &self.adjacency);
        row("  src_chunks", &self.src_chunks);
        row("  chunk_decrypt", &self.chunk_decrypt);
        eprintln!("layer0_reads/q     {:>10.1}", self.layer0_reads_mean);
    }
}

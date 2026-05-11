//! Thread-local timing aggregator. Cheap when sections are coarse (per-matmul
//! or per-layer rather than per-element). Used by the overhead-breakdown
//! benchmark; safe to leave enabled in tests and benches because reading the
//! `Instant` clock is ~50 ns on x86 and we only sample at coarse boundaries.
//!
//! No effect on correctness if instrumentation is skipped: `record` and
//! `time` are pure observation.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

#[derive(Default, Clone, Debug)]
pub struct Profile {
    /// `category → (cumulative time, call count)`.
    pub buckets: BTreeMap<&'static str, (Duration, u64)>,
}

impl Profile {
    pub fn total(&self) -> Duration {
        self.buckets.values().map(|(d, _)| *d).sum()
    }

    /// Print a sorted, human-readable breakdown to stderr.
    pub fn dump(&self, header: &str) {
        let total = self.total().as_secs_f64() * 1000.0;
        eprintln!();
        eprintln!("=== {header} ===");
        eprintln!(
            "{:<32} {:>10} {:>10} {:>10}",
            "category", "time (ms)", "share", "calls"
        );
        eprintln!("{}", "-".repeat(64));
        let mut rows: Vec<_> = self.buckets.iter().collect();
        rows.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
        for (name, (d, n)) in rows {
            let ms = d.as_secs_f64() * 1000.0;
            let share = if total > 0.0 { 100.0 * ms / total } else { 0.0 };
            eprintln!(
                "{:<32} {:>10.2} {:>9.1}% {:>10}",
                name, ms, share, n,
            );
        }
        eprintln!("{}", "-".repeat(64));
        eprintln!("{:<32} {:>10.2}", "TOTAL", total);
    }
}

thread_local! {
    static PROFILE: RefCell<Profile> = RefCell::new(Profile::default());
}

/// Add a timing sample to the current thread's profile.
pub fn record(name: &'static str, d: Duration) {
    PROFILE.with(|p| {
        let mut p = p.borrow_mut();
        let entry = p.buckets.entry(name).or_default();
        entry.0 += d;
        entry.1 += 1;
    });
}

/// Time the closure `f` and record its duration under `name`.
pub fn time<R>(name: &'static str, f: impl FnOnce() -> R) -> R {
    let t0 = Instant::now();
    let r = f();
    record(name, t0.elapsed());
    r
}

/// Clear the current thread's profile.
pub fn reset() {
    PROFILE.with(|p| p.borrow_mut().buckets.clear());
}

/// Snapshot the current thread's profile (independent copy).
pub fn snapshot() -> Profile {
    PROFILE.with(|p| p.borrow().clone())
}

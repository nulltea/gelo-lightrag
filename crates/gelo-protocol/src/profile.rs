//! Thread-local timing aggregator. Cheap when sections are coarse (per-matmul
//! or per-layer rather than per-element). Used by the overhead-breakdown
//! benchmark; safe to leave enabled in tests and benches because reading the
//! `Instant` clock is ~50 ns on x86 and we only sample at coarse boundaries.
//!
//! No effect on correctness if instrumentation is skipped: `record` and
//! `time` are pure observation.
//!
//! ## Cross-thread aggregation
//!
//! `record` writes to a per-thread `RefCell` (no mutex on the hot path).
//! Worker threads spawned by the R4 async path (`offload_*_async` shield
//! generation, future bus-pipeline helpers) must flush their thread-local
//! profile to the global aggregator before joining — easiest pattern is
//! `let _g = profile::WorkerProfileGuard::new();` at the top of the
//! worker closure. The guard's `Drop` impl flushes on scope exit (including
//! panic unwind). The main thread calls `profile::aggregate_threads()`
//! before snapshot/dump to merge the flushed worker profiles into its
//! local accumulator.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
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

    /// Merge another profile's buckets into this one (additive).
    pub fn merge(&mut self, other: &Profile) {
        for (name, (d, n)) in &other.buckets {
            let entry = self.buckets.entry(*name).or_default();
            entry.0 += *d;
            entry.1 += *n;
        }
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

/// Process-global staging area for profiles flushed from worker threads.
/// The main thread drains this via [`aggregate_threads`] before snapshot.
static GLOBAL_AGGREGATOR: OnceLock<Mutex<Vec<Profile>>> = OnceLock::new();

fn global() -> &'static Mutex<Vec<Profile>> {
    GLOBAL_AGGREGATOR.get_or_init(|| Mutex::new(Vec::new()))
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

/// Reset the current thread's profile *and* drain the global worker
/// aggregator. Call at the start of a fresh measurement run (e.g., at
/// `begin_forward_pass` for benches that snapshot per-forward).
pub fn reset_all() {
    PROFILE.with(|p| p.borrow_mut().buckets.clear());
    let _ = global().lock().map(|mut g| g.clear());
}

/// Snapshot the current thread's profile (independent copy).
///
/// To capture worker-thread samples too, call [`aggregate_threads`] first.
pub fn snapshot() -> Profile {
    PROFILE.with(|p| p.borrow().clone())
}

/// Push this thread's accumulated profile into the global aggregator and
/// clear the thread-local. Worker threads should call this (typically via
/// [`WorkerProfileGuard`]) before joining so their samples survive into
/// the main snapshot.
pub fn flush_to_global() {
    let snap = PROFILE.with(|p| {
        let local = p.borrow().clone();
        p.borrow_mut().buckets.clear();
        local
    });
    if !snap.buckets.is_empty() {
        if let Ok(mut g) = global().lock() {
            g.push(snap);
        }
    }
}

/// Drain the global worker aggregator into the current thread's profile.
/// Typically called on the main thread after joining workers, before
/// `snapshot`/`dump`. Idempotent.
pub fn aggregate_threads() {
    let drained: Vec<Profile> = match global().lock() {
        Ok(mut g) => g.drain(..).collect(),
        Err(_) => return,
    };
    PROFILE.with(|p| {
        let mut main = p.borrow_mut();
        for sub in drained {
            main.merge(&sub);
        }
    });
}

/// RAII guard that flushes the current thread's profile to the global
/// aggregator on scope exit (including panic unwind). Place at the top
/// of any worker-thread closure that calls `profile::record` /
/// `profile::time`.
///
/// ```ignore
/// std::thread::spawn(|| {
///     let _g = gelo_protocol::profile::WorkerProfileGuard::new();
///     gelo_protocol::profile::time("gelo:shield_async", || sample_shield());
///     // _g drops here; samples land in the global aggregator
/// });
/// ```
pub struct WorkerProfileGuard;

impl WorkerProfileGuard {
    pub fn new() -> Self {
        Self
    }
}

impl Default for WorkerProfileGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for WorkerProfileGuard {
    fn drop(&mut self) {
        flush_to_global();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard, OnceLock};
    use std::thread;

    /// Serialize profile tests: they all touch the process-global
    /// `GLOBAL_AGGREGATOR`, so running in parallel under `cargo test`
    /// would race. Production use has at most one forward pass at a
    /// time so the global state is safe there.
    fn serial_lock() -> MutexGuard<'static, ()> {
        static M: OnceLock<Mutex<()>> = OnceLock::new();
        M.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|p| p.into_inner())
    }

    fn run_isolated<F: FnOnce() + Send + 'static>(f: F) -> Profile {
        let _guard = serial_lock();
        // Run the test body in a fresh thread so the main test runner's
        // thread-local profile isn't polluted by parallel tests, then
        // aggregate worker flushes on that thread.
        thread::spawn(move || {
            reset_all();
            f();
            aggregate_threads();
            snapshot()
        })
        .join()
        .expect("test thread")
    }

    #[test]
    fn record_accumulates_on_one_thread() {
        let snap = run_isolated(|| {
            time("test:a", || {});
            time("test:a", || {});
            time("test:b", || {});
        });
        assert_eq!(snap.buckets.get("test:a").map(|(_, n)| *n), Some(2));
        assert_eq!(snap.buckets.get("test:b").map(|(_, n)| *n), Some(1));
    }

    #[test]
    fn worker_guard_flushes_to_main_on_aggregate() {
        let snap = run_isolated(|| {
            time("test:main_a", || {});
            let h = thread::spawn(|| {
                let _g = WorkerProfileGuard::new();
                time("test:worker_a", || {});
                time("test:worker_a", || {});
                time("test:worker_b", || {});
            });
            h.join().expect("worker");
            time("test:main_b", || {});
            // Note: aggregate_threads is called inside run_isolated after
            // f returns. We assert on the post-aggregate snapshot below.
        });
        assert_eq!(snap.buckets.get("test:main_a").map(|(_, n)| *n), Some(1));
        assert_eq!(snap.buckets.get("test:main_b").map(|(_, n)| *n), Some(1));
        assert_eq!(snap.buckets.get("test:worker_a").map(|(_, n)| *n), Some(2));
        assert_eq!(snap.buckets.get("test:worker_b").map(|(_, n)| *n), Some(1));
    }

    #[test]
    fn worker_guard_flushes_on_panic() {
        let snap = run_isolated(|| {
            let h = thread::spawn(|| {
                let _g = WorkerProfileGuard::new();
                time("test:panic_a", || {});
                panic!("intentional");
            });
            let _ = h.join();
        });
        assert_eq!(snap.buckets.get("test:panic_a").map(|(_, n)| *n), Some(1));
    }

    #[test]
    fn aggregate_threads_is_idempotent() {
        let snap = run_isolated(|| {
            let h = thread::spawn(|| {
                let _g = WorkerProfileGuard::new();
                time("test:idem", || {});
            });
            h.join().expect("worker");
            aggregate_threads();
            aggregate_threads(); // second call should be a no-op
        });
        assert_eq!(snap.buckets.get("test:idem").map(|(_, n)| *n), Some(1));
    }

    #[test]
    fn merge_is_additive() {
        let mut a = Profile::default();
        a.buckets
            .insert("x", (Duration::from_millis(10), 1));
        let mut b = Profile::default();
        b.buckets
            .insert("x", (Duration::from_millis(20), 2));
        b.buckets.insert("y", (Duration::from_millis(5), 1));
        a.merge(&b);
        assert_eq!(a.buckets["x"], (Duration::from_millis(30), 3));
        assert_eq!(a.buckets["y"], (Duration::from_millis(5), 1));
    }
}

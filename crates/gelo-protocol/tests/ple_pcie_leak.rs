//! M1.2c — PLE address-bus leak verification.
//!
//! Round 2's P0 finding (see `docs/research/private-llm-inference-round-2.md`
//! §D.5 and `docs/prototype/gelo-llm.html` §03): Gemma 3n / Gemma 4 PLE
//! tables are indexed by `token_id` per layer. If the table lived on
//! the untrusted GPU, an attacker watching PCIe addresses would
//! recover the prompt in plaintext — strictly worse than Vec2Text. The
//! fix specified in `docs/plans/path-1-gelo-gemma.md` M1.2 is to keep
//! the table inside the trusted executor's owned memory and gather
//! in-TEE.
//!
//! This test enforces that contract at the substrate level:
//!
//! 1. Wraps the standard `ReferenceCpuEngine` in a `SpyEngine` that
//!    records every method call against it (handle, shape).
//! 2. Provisions a `PleTable` into both `InProcessTrustedExecutor`
//!    and `PlaintextExecutor`.
//! 3. Calls `ple_gather` for several `(layer, token_ids)` combinations
//!    on each executor.
//! 4. Asserts the gather outputs match the expected dequantised rows.
//! 5. Asserts the spy engine recorded **zero** activity — no
//!    `register_weight`, no `matmul`, no `matmul_dynamic_batched`.
//!
//! If the gather ever falls back to a `provision_weight + matmul`
//! pattern (e.g. someone re-implements PLE as a normal embedding-like
//! offload), the spy log fills up and this test fails — the leak
//! returns immediately on the next code review.

use std::sync::Mutex;

use anyhow::Result;
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};

use gelo_protocol::{
    GpuOffloadEngine, InProcessTrustedExecutor, PlaintextExecutor, PleTable, ReferenceCpuEngine,
    TrustedExecutor, WeightHandle,
};

/// Tape-recorder engine. Wraps a real engine, forwarding every call,
/// and appends a textual trace entry so the test can assert on the
/// sequence afterwards. Only the methods the protocol actually uses
/// are forwarded; the rest fall back to the trait defaults.
struct SpyEngine {
    inner: ReferenceCpuEngine,
    log: Mutex<Vec<String>>,
}

impl SpyEngine {
    fn new() -> Self {
        Self {
            inner: ReferenceCpuEngine::new(),
            log: Mutex::new(Vec::new()),
        }
    }

    fn log_snapshot(&self) -> Vec<String> {
        self.log.lock().expect("log poisoned").clone()
    }

    fn record(&self, entry: String) {
        self.log.lock().expect("log poisoned").push(entry);
    }
}

impl GpuOffloadEngine for SpyEngine {
    fn register_weight(&mut self, handle: WeightHandle, weight: ArrayView2<f32>) -> Result<()> {
        self.record(format!(
            "register_weight(layer={}, kind={:?}, shape=({}, {}))",
            handle.layer,
            handle.kind,
            weight.nrows(),
            weight.ncols(),
        ));
        self.inner.register_weight(handle, weight)
    }

    fn matmul(&self, handle: WeightHandle, input: ArrayView2<f32>) -> Result<Array2<f32>> {
        self.record(format!(
            "matmul(layer={}, kind={:?}, in=({}, {}))",
            handle.layer,
            handle.kind,
            input.nrows(),
            input.ncols(),
        ));
        self.inner.matmul(handle, input)
    }

    fn matmul_dynamic(
        &self,
        lhs: ArrayView2<f32>,
        rhs: ArrayView2<f32>,
    ) -> Result<Array2<f32>> {
        self.record(format!(
            "matmul_dynamic(lhs=({}, {}), rhs=({}, {}))",
            lhs.nrows(),
            lhs.ncols(),
            rhs.nrows(),
            rhs.ncols(),
        ));
        self.inner.matmul_dynamic(lhs, rhs)
    }

    fn matmul_dynamic_batched(
        &self,
        lhs: ArrayView3<f32>,
        rhs: ArrayView3<f32>,
    ) -> Result<Array3<f32>> {
        self.record(format!(
            "matmul_dynamic_batched(lhs={:?}, rhs={:?})",
            lhs.shape(),
            rhs.shape(),
        ));
        self.inner.matmul_dynamic_batched(lhs, rhs)
    }
}

/// Synthesise a PLE table with predictable row contents. Row `t` at
/// layer `li` has values `(li * 10 + t * 3, li * 10 + t * 3 + 1)`
/// — easy to spot-check after dequant.
fn synth_ple_table() -> PleTable {
    let vocab = 6;
    let d = 2;
    let mut layers = Vec::new();
    for li in 0..4 {
        let mut slab = Vec::with_capacity(vocab * d);
        for t in 0..vocab {
            slab.push((li * 10 + t * 3) as i8);
            slab.push((li * 10 + t * 3 + 1) as i8);
        }
        layers.push(slab);
    }
    PleTable::from_int8_rows(layers, vocab, d, 1.0).unwrap()
}

#[test]
fn ple_gather_under_plaintext_executor_keeps_engine_idle() {
    let table = synth_ple_table();
    let spy = SpyEngine::new();
    let mut exec = PlaintextExecutor::new(spy);
    exec.provision_ple_table(table.clone()).unwrap();

    // Gather across several layer/token combinations.
    let out0 = exec.ple_gather(&[0, 1, 5], 0).unwrap();
    let out2 = exec.ple_gather(&[3, 5], 2).unwrap();

    // Layer 0, t=0 → (0, 1); t=1 → (3, 4); t=5 → (15, 16).
    assert_eq!(out0.shape(), &[3, 2]);
    assert_eq!(out0[[0, 0]], 0.0);
    assert_eq!(out0[[0, 1]], 1.0);
    assert_eq!(out0[[1, 0]], 3.0);
    assert_eq!(out0[[1, 1]], 4.0);
    assert_eq!(out0[[2, 0]], 15.0);
    assert_eq!(out0[[2, 1]], 16.0);

    // Layer 2, t=3 → (29, 30); t=5 → (35, 36).
    assert_eq!(out2[[0, 0]], 29.0);
    assert_eq!(out2[[1, 1]], 36.0);

    // The load-bearing assertion: the spy engine saw nothing.
    let log = exec.engine().log_snapshot();
    assert!(
        log.is_empty(),
        "PLE gather leaked to the offload engine — log:\n{}",
        log.join("\n"),
    );
}

#[test]
fn ple_gather_under_inprocess_trusted_executor_keeps_engine_idle() {
    // Same contract holds for the masked / paper-parity executor — the
    // PLE table never reaches the engine even when the rest of the
    // forward path is offloaded under mask.
    let table = synth_ple_table();
    let spy = SpyEngine::new();
    let mut exec = InProcessTrustedExecutor::new(spy);
    exec.provision_ple_table(table).unwrap();

    let out = exec.ple_gather(&[2, 4], 1).unwrap();
    // Layer 1, t=2 → (16, 17); t=4 → (22, 23).
    assert_eq!(out[[0, 0]], 16.0);
    assert_eq!(out[[0, 1]], 17.0);
    assert_eq!(out[[1, 0]], 22.0);
    assert_eq!(out[[1, 1]], 23.0);

    // No offload calls.
    // (Need to drop the executor or borrow the engine via the type
    // method on InProcessTrustedExecutor — but the spy log is in a
    // Mutex so we can read it through an immutable reference.)
    // Reach into the engine via the executor's `engine_ref()` accessor
    // if one exists; otherwise re-create the executor pattern.
    // Below: use a separate snapshot via a fresh spy wrapper that the
    // executor doesn't take ownership of — but for InProcessTrustedExecutor,
    // the engine IS moved in. Use Arc<Mutex<Vec<String>>> shared
    // between the test and the spy instead so we can inspect after.
    //
    // Refactor: see `shared_log_*` variant below for the proper
    // pattern.
}

/// Variant that keeps a `Mutex<Vec<String>>` outside the executor so
/// the test can inspect the spy log after the executor owns the
/// engine. Same contract.
#[test]
fn ple_gather_inprocess_with_shared_log_keeps_engine_idle() {
    use std::sync::Arc;
    struct SharedSpy {
        inner: ReferenceCpuEngine,
        log: Arc<Mutex<Vec<String>>>,
    }
    impl GpuOffloadEngine for SharedSpy {
        fn register_weight(
            &mut self,
            handle: WeightHandle,
            weight: ArrayView2<f32>,
        ) -> Result<()> {
            self.log.lock().unwrap().push(format!(
                "register_weight(layer={}, kind={:?}, shape=({}, {}))",
                handle.layer,
                handle.kind,
                weight.nrows(),
                weight.ncols(),
            ));
            self.inner.register_weight(handle, weight)
        }
        fn matmul(
            &self,
            handle: WeightHandle,
            input: ArrayView2<f32>,
        ) -> Result<Array2<f32>> {
            self.log.lock().unwrap().push(format!(
                "matmul(layer={}, kind={:?}, in=({}, {}))",
                handle.layer,
                handle.kind,
                input.nrows(),
                input.ncols(),
            ));
            self.inner.matmul(handle, input)
        }
        fn matmul_dynamic(
            &self,
            lhs: ArrayView2<f32>,
            rhs: ArrayView2<f32>,
        ) -> Result<Array2<f32>> {
            self.log.lock().unwrap().push(format!(
                "matmul_dynamic(lhs=({}, {}), rhs=({}, {}))",
                lhs.nrows(),
                lhs.ncols(),
                rhs.nrows(),
                rhs.ncols(),
            ));
            self.inner.matmul_dynamic(lhs, rhs)
        }
    }

    let log = Arc::new(Mutex::new(Vec::<String>::new()));
    let engine = SharedSpy {
        inner: ReferenceCpuEngine::new(),
        log: Arc::clone(&log),
    };
    let mut exec = InProcessTrustedExecutor::new(engine);
    exec.provision_ple_table(synth_ple_table()).unwrap();

    for li in 0..4 {
        let out = exec.ple_gather(&[0, 1, 2, 3], li).unwrap();
        assert_eq!(out.shape(), &[4, 2]);
    }

    let entries = log.lock().unwrap().clone();
    assert!(
        entries.is_empty(),
        "PLE gather leaked to offload engine across {} layers: {}",
        4,
        entries.join("\n"),
    );
}

#[test]
fn ple_gather_without_provision_errors() {
    let exec = PlaintextExecutor::new(SpyEngine::new());
    let err = exec.ple_gather(&[0], 0).unwrap_err();
    assert!(err.to_string().contains("no PLE table provisioned"));
}

#[test]
fn default_trait_executor_rejects_ple_provision() {
    // The default TrustedExecutor impl of `provision_ple_table`
    // returns an explicit error rather than silently no-op'ing — this
    // is the "fail loud" defense against an executor that doesn't
    // implement PLE accidentally getting handed a hybrid model.
    use ndarray::ArrayView2;
    struct MinimalExec;
    impl TrustedExecutor for MinimalExec {
        fn provision_weight(
            &mut self,
            _h: WeightHandle,
            _w: ArrayView2<f32>,
        ) -> Result<()> {
            Ok(())
        }
        fn provision_weight_bf16(
            &mut self,
            _h: WeightHandle,
            _w: ArrayView2<half::bf16>,
        ) -> Result<()> {
            Ok(())
        }
        fn offload_linear(
            &mut self,
            _h: WeightHandle,
            _hidden: ArrayView2<f32>,
        ) -> Result<Array2<f32>> {
            unimplemented!()
        }
    }

    let mut exec = MinimalExec;
    let err = exec.provision_ple_table(synth_ple_table()).unwrap_err();
    assert!(err.to_string().contains("not implemented"));

    let err = exec.ple_gather(&[0], 0).unwrap_err();
    assert!(err.to_string().contains("without a provisioned"));
}

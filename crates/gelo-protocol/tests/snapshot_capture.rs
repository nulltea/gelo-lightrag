//! Smoke tests for PCIe-side snapshot capture (Phase 1 of the AloePri
//! attack-resistance integration — see `docs/research/aloepri-vs-gelo.md`
//! §4.1).
//!
//! Covers:
//! 1. Default-off invariant: a normal embedder forward pass produces zero
//!    snapshots when capture is not opted into.
//! 2. Opt-in capture: enabling capture before a `begin_forward_pass /
//!    offload_linear / end_forward_pass` cycle records exactly one
//!    snapshot per offload, in seq-idx order, with correct
//!    (layer, kind) tagging.
//! 3. `offload_qkv` records three snapshots (Q, K, V) sharing the same
//!    masked operand by row-equality.
//! 4. `offload_linear_many` records one snapshot per handle in input
//!    order.
//! 5. Drain returns the captured snapshots and leaves the buffer empty
//!    for the next forward pass.
//!
//! These are unit-level tests against `InProcessTrustedExecutor` +
//! `RayonCpuEngine`; the Qwen3-1.7B end-to-end smoke against real
//! weights lives in `crates/gelo-embedder/tests/aloepri_snapshot_capture.rs`
//! (gated `#[ignore]` like the other real-model tests).

use ndarray::Array2;

use gelo_protocol::rng::MaskSeed;
use gelo_protocol::{
    InProcessTrustedExecutor, RayonCpuEngine, SnapshotConfig, TrustedExecutor, WeightHandle,
    WeightKind,
};

fn synth_executor() -> InProcessTrustedExecutor<RayonCpuEngine> {
    InProcessTrustedExecutor::with_seed(RayonCpuEngine::new(), MaskSeed::from_bytes([7u8; 32]))
}

fn provision_three_layers(
    exec: &mut InProcessTrustedExecutor<RayonCpuEngine>,
    d_in: usize,
    d_out: usize,
) {
    // Distinct weight per (layer, kind) so the per-handle masked output
    // is verifiably distinct in tests that assert output divergence.
    for li in 0..3u16 {
        for (k_idx, kind) in [
            WeightKind::Q,
            WeightKind::K,
            WeightKind::V,
            WeightKind::O,
            WeightKind::FfnGate,
            WeightKind::FfnUp,
        ]
        .iter()
        .enumerate()
        {
            let salt = (li as usize) * 100 + k_idx;
            let w = Array2::<f32>::from_shape_fn((d_in, d_out), |(r, c)| {
                ((salt + r) as f32 * 0.01) + (c as f32 * 0.005)
            });
            exec.provision_weight(WeightHandle::new(li, *kind), w.view())
                .unwrap();
        }
    }
}

#[test]
fn capture_disabled_by_default_returns_none() {
    let exec = synth_executor();
    assert!(
        exec.pcie_snapshots().is_none(),
        "capture should be off out of the box — production embedder/reranker \
         paths must never allocate snapshot buffers",
    );
}

#[test]
fn opt_in_capture_records_one_snapshot_per_offload_linear() {
    let mut exec = synth_executor().with_snapshot_capture(SnapshotConfig::default());
    provision_three_layers(&mut exec, 16, 32);
    let hidden = Array2::<f32>::from_shape_fn((4, 16), |(r, c)| (r * 16 + c) as f32 * 0.01);

    exec.begin_forward_pass(hidden.nrows()).unwrap();
    for li in 0..3u16 {
        for kind in [WeightKind::Q, WeightKind::O] {
            exec.offload_linear(WeightHandle::new(li, kind), hidden.view())
                .unwrap();
        }
    }
    exec.end_forward_pass().unwrap();

    let snaps = exec.pcie_snapshots().expect("capture enabled");
    assert_eq!(snaps.len(), 6, "expected 3 layers × 2 kinds = 6 snapshots");
    for (i, snap) in snaps.iter().enumerate() {
        assert_eq!(snap.seq_idx, i, "seq_idx must increase monotonically");
        assert_eq!(snap.layer, (i / 2) as u16);
        let want_kind = if i % 2 == 0 { WeightKind::Q } else { WeightKind::O };
        assert_eq!(snap.kind, want_kind);
        // Shape: (n_data + shield.k, d_in). Default shield is k=8.
        assert_eq!(snap.masked_operand.shape(), &[4 + 8, 16]);
        assert!(snap.masked_output.is_some(), "default config captures outputs");
        assert_eq!(snap.masked_output.as_ref().unwrap().shape(), &[4 + 8, 32]);
    }
    assert_eq!(
        exec.pcie_snapshot_capture().unwrap().dropped(),
        0,
        "default cap of 4096 should never drop on a 6-snapshot run",
    );
}

#[test]
fn offload_qkv_records_three_snapshots_sharing_operand() {
    let mut exec = synth_executor().with_snapshot_capture(SnapshotConfig::default());
    provision_three_layers(&mut exec, 16, 32);
    let hidden = Array2::<f32>::from_shape_fn((4, 16), |(r, c)| (r * 16 + c) as f32 * 0.01);

    exec.begin_forward_pass(hidden.nrows()).unwrap();
    exec.offload_qkv(0, hidden.view()).unwrap();
    exec.end_forward_pass().unwrap();

    let snaps = exec.pcie_snapshots().unwrap();
    assert_eq!(snaps.len(), 3, "Q, K, V — one snapshot each");
    assert_eq!(snaps[0].kind, WeightKind::Q);
    assert_eq!(snaps[1].kind, WeightKind::K);
    assert_eq!(snaps[2].kind, WeightKind::V);
    // The three snapshots must share the same masked operand bit-for-bit —
    // that's the actual property the PCIe-side attacker exploits in
    // batched-matmul layouts.
    assert_eq!(snaps[0].masked_operand, snaps[1].masked_operand);
    assert_eq!(snaps[1].masked_operand, snaps[2].masked_operand);
    // Outputs differ because Q/K/V are distinct weights.
    let oq = snaps[0].masked_output.as_ref().unwrap();
    let ok = snaps[1].masked_output.as_ref().unwrap();
    assert_ne!(oq, ok, "Q and K masked outputs must differ");
}

#[test]
fn offload_linear_many_records_one_snapshot_per_handle_in_order() {
    let mut exec = synth_executor().with_snapshot_capture(SnapshotConfig::default());
    provision_three_layers(&mut exec, 16, 32);
    let hidden = Array2::<f32>::from_shape_fn((4, 16), |(r, c)| (r * 16 + c) as f32 * 0.01);

    let handles = [
        WeightHandle::new(0, WeightKind::FfnGate),
        WeightHandle::new(0, WeightKind::FfnUp),
    ];
    exec.begin_forward_pass(hidden.nrows()).unwrap();
    exec.offload_linear_many(&handles, hidden.view()).unwrap();
    exec.end_forward_pass().unwrap();

    let snaps = exec.pcie_snapshots().unwrap();
    assert_eq!(snaps.len(), 2);
    assert_eq!(snaps[0].kind, WeightKind::FfnGate);
    assert_eq!(snaps[1].kind, WeightKind::FfnUp);
    // Same operand drives both projections.
    assert_eq!(snaps[0].masked_operand, snaps[1].masked_operand);
}

#[test]
fn drain_returns_snapshots_and_clears_buffer() {
    let mut exec = synth_executor().with_snapshot_capture(SnapshotConfig::default());
    provision_three_layers(&mut exec, 16, 32);
    let hidden = Array2::<f32>::from_shape_fn((4, 16), |(r, c)| (r * 16 + c) as f32 * 0.01);

    exec.begin_forward_pass(hidden.nrows()).unwrap();
    exec.offload_linear(WeightHandle::new(0, WeightKind::Q), hidden.view())
        .unwrap();
    exec.offload_linear(WeightHandle::new(0, WeightKind::K), hidden.view())
        .unwrap();
    exec.end_forward_pass().unwrap();

    assert_eq!(exec.pcie_snapshots().unwrap().len(), 2);
    let drained = exec.drain_pcie_snapshots();
    assert_eq!(drained.len(), 2);
    assert_eq!(
        exec.pcie_snapshots().unwrap().len(),
        0,
        "drain must leave the buffer empty",
    );

    // Next pass continues at seq_idx 2 (monotonic across drains).
    exec.begin_forward_pass(hidden.nrows()).unwrap();
    exec.offload_linear(WeightHandle::new(1, WeightKind::V), hidden.view())
        .unwrap();
    exec.end_forward_pass().unwrap();
    let next = exec.pcie_snapshots().unwrap();
    assert_eq!(next.len(), 1);
    assert_eq!(next[0].seq_idx, 2);
}

#[test]
fn disable_capture_drops_buffer_and_makes_next_calls_no_op() {
    let mut exec = synth_executor().with_snapshot_capture(SnapshotConfig::default());
    provision_three_layers(&mut exec, 16, 32);
    let hidden = Array2::<f32>::from_shape_fn((4, 16), |(r, c)| (r * 16 + c) as f32 * 0.01);

    exec.begin_forward_pass(hidden.nrows()).unwrap();
    exec.offload_linear(WeightHandle::new(0, WeightKind::Q), hidden.view())
        .unwrap();
    exec.end_forward_pass().unwrap();
    assert_eq!(exec.pcie_snapshots().unwrap().len(), 1);

    exec.disable_snapshot_capture();
    assert!(exec.pcie_snapshots().is_none());

    // Run another pass — no panic, just no capture.
    exec.begin_forward_pass(hidden.nrows()).unwrap();
    exec.offload_linear(WeightHandle::new(0, WeightKind::Q), hidden.view())
        .unwrap();
    exec.end_forward_pass().unwrap();
    assert!(exec.pcie_snapshots().is_none());
}

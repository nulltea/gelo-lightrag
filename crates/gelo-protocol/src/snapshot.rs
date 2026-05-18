//! PCIe-side snapshot capture for empirical attack-resistance evaluation.
//!
//! AloePri ships an attack suite (`security_qwen/{vma,ima,isa,tfma,sda,ia}.py`)
//! that takes intermediate tensors and attempts to recover the input token
//! sequence. To run that suite against GELO we need to be able to dump what
//! the **PCIe-side attacker** would actually observe — i.e. the post-mask
//! (and optionally post-shield-stacking) operands and matmul outputs that
//! cross from the trusted side to the engine.
//!
//! Phase 1 (this module) is in-memory capture only. Phase 2 wires a Python
//! harness at `evals/aloepri-attacks/` that reads serialised snapshots and
//! drives the AloePri attack suite. Phase 3 promotes the attack
//! ASR (attack success rate) into a CI release-gate threshold. See
//! `docs/research/aloepri-vs-gelo.md` §4.1 for the full plan.
//!
//! ## Threat-model alignment
//!
//! The capture point sits **between mask-apply and engine-matmul**:
//!
//! ```text
//!   hidden  ──► build_shielded_and_apply ──► [SNAPSHOT POINT] ──► engine.matmul
//!                                              ^ pre-engine: shield-stacked + masked
//!                                              ^ post-engine: masked output
//! ```
//!
//! This is exactly what an adversary co-located with the GPU (or sitting on
//! the PCIe bus, or possessing a TEE-side breach below the protocol layer)
//! sees. It is **not** what crosses an attested TEE boundary in production —
//! the unmasked outputs stay inside the executor's address space and are
//! never serialised.

use ndarray::Array2;

use crate::substrate::{WeightHandle, WeightKind};

/// One per-offload PCIe-crossing tensor pair. The masked operand is what the
/// GPU receives from the trusted side; the masked output is what the GPU
/// returns. Together they describe the entire visible trace of the matmul
/// to a PCIe-side attacker.
#[derive(Debug, Clone)]
pub struct PcieSnapshot {
    /// Monotone counter within a single capture session, starting at 0 and
    /// incrementing per call to `record(...)`. Lets the Python harness
    /// re-establish the temporal ordering AloePri's TFMA/SDA attacks need.
    pub seq_idx: usize,
    /// `WeightHandle::layer` of the matmul this snapshot belongs to.
    pub layer: u16,
    /// `WeightHandle::kind` of the matmul — Q/K/V/O/FfnGate/FfnUp/FfnDown.
    /// AloePri's ISA partitions snapshots by op_kind when fitting its
    /// inversion regressors.
    pub kind: WeightKind,
    /// The masked, shield-stacked operand that crossed PCIe.
    /// Shape: `(stacked_n, in_features)` where
    /// `stacked_n = n_data + shield.k` (data rows first, shield rows after).
    pub masked_operand: Array2<f32>,
    /// The masked output the engine returned. Shape: `(stacked_n, out_features)`.
    /// Some when capture was configured with `with_outputs(true)`; None when
    /// only the upstream-direction snapshot was requested (cheaper —
    /// out_features is often equal to or larger than in_features).
    pub masked_output: Option<Array2<f32>>,
}

/// Configuration controlling what gets captured.
#[derive(Debug, Clone, Copy)]
pub struct SnapshotConfig {
    /// Capture the masked output in addition to the masked operand. Default
    /// `true` — AloePri's IMA/ISA need both. Set `false` to halve memory
    /// when only operand-side attacks (VMA/IA) are being run.
    pub capture_outputs: bool,
    /// Hard cap on snapshots retained per session. Once the buffer reaches
    /// this limit additional snapshots are silently dropped (no error —
    /// running out of memory mid-forward is worse than truncating). `None`
    /// is unbounded; default 4096 fits a Qwen3-1.7B prefill of 1024 tokens
    /// at every (layer × op_kind) pair (28 × 7 = 196 snapshots per forward
    /// pass — even 20 forward passes stays well under the cap).
    pub max_snapshots: Option<usize>,
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self {
            capture_outputs: true,
            max_snapshots: Some(4096),
        }
    }
}

/// In-memory snapshot aggregator. Owned by the executor (and only allocated
/// when capture is opted in via `with_snapshot_capture(_)`). The buffer is
/// drained / consumed by the test harness via
/// `InProcessTrustedExecutor::drain_pcie_snapshots()`.
#[derive(Debug, Default)]
pub struct SnapshotCapture {
    cfg: SnapshotConfig,
    buffer: Vec<PcieSnapshot>,
    next_seq: usize,
    dropped: usize,
}

impl SnapshotCapture {
    /// Build with the given config. Use `SnapshotCapture::default()` for the
    /// AloePri-recommended defaults (capture outputs, cap at 4096 entries).
    pub fn new(cfg: SnapshotConfig) -> Self {
        Self {
            cfg,
            buffer: Vec::new(),
            next_seq: 0,
            dropped: 0,
        }
    }

    /// Record one snapshot. `operand` is the post-mask tensor that crossed
    /// PCIe; `output` is the engine's return (None when only operand
    /// capture is wanted). The capture site is the only authority on
    /// snapshot shape — this method does not validate against
    /// (n_data, shield.k), since callers may capture either the
    /// shield-stacked or the post-strip form.
    pub fn record(
        &mut self,
        handle: WeightHandle,
        operand: &Array2<f32>,
        output: Option<&Array2<f32>>,
    ) {
        if let Some(cap) = self.cfg.max_snapshots {
            if self.buffer.len() >= cap {
                self.dropped += 1;
                return;
            }
        }
        let masked_output = if self.cfg.capture_outputs {
            output.map(|o| o.clone())
        } else {
            None
        };
        self.buffer.push(PcieSnapshot {
            seq_idx: self.next_seq,
            layer: handle.layer,
            kind: handle.kind,
            masked_operand: operand.clone(),
            masked_output,
        });
        self.next_seq += 1;
    }

    /// Borrow the current buffer for read-only inspection (assertion-style
    /// tests prefer this over draining).
    pub fn snapshots(&self) -> &[PcieSnapshot] {
        &self.buffer
    }

    /// Take ownership of the captured snapshots, clearing the internal
    /// buffer. The `next_seq` counter is **not** reset — the next snapshot
    /// continues monotonically, matching how Python harnesses concatenate
    /// drains across multiple forward passes.
    pub fn drain(&mut self) -> Vec<PcieSnapshot> {
        std::mem::take(&mut self.buffer)
    }

    /// Reset both the buffer and the sequence counter. Use between
    /// independent attack runs against the same executor instance.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.next_seq = 0;
        self.dropped = 0;
    }

    /// Number of snapshots silently dropped because `max_snapshots` was hit.
    /// Tests should assert this is zero on a typical Qwen3-1.7B forward pass
    /// (28 layers × 7 op_kinds = 196 — the default cap of 4096 has 20×
    /// headroom).
    pub fn dropped(&self) -> usize {
        self.dropped
    }

    pub fn config(&self) -> SnapshotConfig {
        self.cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn dummy_op(rows: usize, cols: usize, seed: f32) -> Array2<f32> {
        Array2::from_shape_fn((rows, cols), |(r, c)| seed + (r * cols + c) as f32)
    }

    #[test]
    fn record_buffers_in_seq_order_and_assigns_monotone_idx() {
        let mut cap = SnapshotCapture::default();
        for (i, kind) in [WeightKind::Q, WeightKind::K, WeightKind::V].iter().enumerate() {
            let h = WeightHandle::new(i as u16, *kind);
            let operand = dummy_op(8, 16, i as f32);
            let output = dummy_op(8, 32, i as f32 + 100.0);
            cap.record(h, &operand, Some(&output));
        }
        let snaps = cap.snapshots();
        assert_eq!(snaps.len(), 3);
        assert_eq!(snaps[0].seq_idx, 0);
        assert_eq!(snaps[1].seq_idx, 1);
        assert_eq!(snaps[2].seq_idx, 2);
        assert_eq!(snaps[0].kind, WeightKind::Q);
        assert_eq!(snaps[2].kind, WeightKind::V);
        // Output captured with default config.
        assert!(snaps[0].masked_output.is_some());
    }

    #[test]
    fn capture_outputs_false_skips_output_clone() {
        let mut cap = SnapshotCapture::new(SnapshotConfig {
            capture_outputs: false,
            max_snapshots: None,
        });
        let h = WeightHandle::new(0, WeightKind::Q);
        cap.record(h, &dummy_op(4, 8, 0.0), Some(&dummy_op(4, 16, 0.0)));
        assert_eq!(cap.snapshots().len(), 1);
        assert!(cap.snapshots()[0].masked_output.is_none());
    }

    #[test]
    fn max_snapshots_cap_drops_silently_after_limit() {
        let mut cap = SnapshotCapture::new(SnapshotConfig {
            capture_outputs: false,
            max_snapshots: Some(2),
        });
        let h = WeightHandle::new(0, WeightKind::Q);
        for _ in 0..5 {
            cap.record(h, &dummy_op(4, 8, 0.0), None);
        }
        assert_eq!(cap.snapshots().len(), 2);
        assert_eq!(cap.dropped(), 3);
        // seq_idx 0,1 only — dropped records don't increment.
        assert_eq!(cap.snapshots()[0].seq_idx, 0);
        assert_eq!(cap.snapshots()[1].seq_idx, 1);
    }

    #[test]
    fn drain_clears_buffer_but_preserves_seq_counter() {
        let mut cap = SnapshotCapture::default();
        let h = WeightHandle::new(0, WeightKind::Q);
        cap.record(h, &dummy_op(4, 8, 0.0), None);
        cap.record(h, &dummy_op(4, 8, 1.0), None);
        let drained = cap.drain();
        assert_eq!(drained.len(), 2);
        assert_eq!(cap.snapshots().len(), 0);
        // Next record continues at seq 2 — Python harness concatenation
        // across forward passes stays monotonic.
        cap.record(h, &dummy_op(4, 8, 2.0), None);
        assert_eq!(cap.snapshots()[0].seq_idx, 2);
    }

    #[test]
    fn reset_clears_buffer_and_seq_counter() {
        let mut cap = SnapshotCapture::default();
        let h = WeightHandle::new(0, WeightKind::Q);
        cap.record(h, &dummy_op(4, 8, 0.0), None);
        cap.reset();
        cap.record(h, &dummy_op(4, 8, 1.0), None);
        assert_eq!(cap.snapshots().len(), 1);
        assert_eq!(cap.snapshots()[0].seq_idx, 0);
    }
}

//! AloePri attack-resistance harness — Rust side.
//!
//! This crate is the "Phase 2" Rust deliverable from
//! `docs/prototype/aloepri-attack-harness.md`. It does **not** modify
//! any GELO crate; it consumes the public `PcieSnapshot` /
//! `TrustedExecutor` API frozen at the Phase 1 boundary.
//!
//! Two pieces live here:
//!
//! 1. [`export_snapshots`] — serialise a `Vec<PcieSnapshot>` plus a
//!    per-run config blob into a `.safetensors` file (operand /
//!    optional output tensors keyed
//!    `snap{seq_idx:05}.{layer:03}.{kind}.{operand|output}`) plus a
//!    sidecar `<basename>.meta.json` describing each snapshot's
//!    shape, kind, and data-row count. Format pinned by
//!    `docs/prototype/aloepri-attack-harness.md` §2.2.
//!
//! 2. [`CapturingPlaintextExecutor`] — a thin `TrustedExecutor` wrapper
//!    around `PlaintextExecutor` that records the unmasked operand
//!    and output of every offload to its own `SnapshotCapture`. Lets
//!    the C0 plain control share the same downstream loader code as
//!    the C1/C2 masked conditions — the Python harness reads all
//!    three through one snapshot adapter.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Axis};
use safetensors::Dtype;
use safetensors::tensor::TensorView;
use serde::Serialize;

use gelo_protocol::{
    GpuOffloadEngine, PcieSnapshot, PlaintextExecutor, SnapshotCapture, SnapshotConfig,
    TrustedExecutor, WeightHandle, WeightKind,
};

/// The control conditions defined in
/// `docs/prototype/aloepri-attack-harness.md` §2.4, plus the HD₃
/// extension added for the round-3 attack-defence gate (B.3): a
/// fourth condition that holds shield constant but swaps the mask
/// family from Haar to the QuIP#/QuaRot HD₃ cascade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Condition {
    /// C0 — `PlaintextExecutor` (no mask, no shield). Control that
    /// proves the attacks themselves are wired correctly: TTRSR
    /// should approach 100% on at least IMA/ISA/VMA.
    C0Plain,
    /// C1 — `InProcessTrustedExecutor` with `with_per_offload_mask()`
    /// (mask only, `ShieldConfig::NONE`). Isolates the mask's
    /// contribution.
    C1MaskOnly,
    /// C2 — `InProcessTrustedExecutor::with_seed` defaults
    /// (per-forward-pass Haar mask, `ShieldConfig::new(8, 4.0)`).
    /// The release-gate target: IMA & ISA TTRSR must be < 10%.
    C2Default,
    /// C3 — same as C2 except the Haar mask is swapped for the HD₃
    /// Hadamard cascade via `.with_hd3_mask()`. Tests whether HD₃'s
    /// discrete `2^{3·s}`-element orbit defends as well as Haar's
    /// continuous measure under the AloePri / GELO §4.3 attack
    /// suites. Holding shield constant between C2 and C3 isolates
    /// the mask family as the only variable.
    C3Hd3,
}

impl Condition {
    pub fn slug(self) -> &'static str {
        match self {
            Condition::C0Plain => "c0_plain",
            Condition::C1MaskOnly => "c1_mask_only",
            Condition::C2Default => "c2_default",
            Condition::C3Hd3 => "c3_hd3",
        }
    }

    pub fn from_slug(slug: &str) -> Result<Self> {
        match slug {
            "c0" | "c0_plain" | "plain" => Ok(Condition::C0Plain),
            "c1" | "c1_mask_only" | "mask_only" => Ok(Condition::C1MaskOnly),
            "c2" | "c2_default" | "default" => Ok(Condition::C2Default),
            "c3" | "c3_hd3" | "hd3" => Ok(Condition::C3Hd3),
            other => Err(anyhow!(
                "unknown condition slug '{other}' (expected c0 / c1 / c2 / c3)"
            )),
        }
    }
}

/// AloePri-pipeline kind names. The Python attack drivers index by
/// these strings, matching HuggingFace transformers conventions.
pub fn kind_slug(kind: WeightKind) -> &'static str {
    match kind {
        WeightKind::Q => "q_proj",
        WeightKind::K => "k_proj",
        WeightKind::V => "v_proj",
        WeightKind::O => "o_proj",
        WeightKind::FfnGate => "gate_proj",
        WeightKind::FfnUp => "up_proj",
        WeightKind::FfnDown => "down_proj",
    }
}

/// Top-level metadata blob written to `<basename>.meta.json`. Encodes
/// the snapshot-set's provenance so the Python loader can route to
/// the right AloePri attack primitives without re-introspecting the
/// safetensors keys.
#[derive(Serialize)]
pub struct ExportMeta {
    pub schema_version: String,
    pub model_id: String,
    pub condition: String,
    pub config: ExportRunConfig,
    pub snapshots: Vec<SnapshotMeta>,
}

#[derive(Serialize)]
pub struct ExportRunConfig {
    /// Shield-row count (k). 0 in C0/C1, 8 in C2 by default.
    pub shield_k: usize,
    /// Shield energy scale. 0.0 in C0/C1, 4.0 in C2 by default.
    pub shield_energy_scale: f32,
    /// True in C2 (per-forward mask), false in C1 (per-offload),
    /// N/A in C0 (no mask) — recorded as `false` for that case.
    pub per_forward_mask: bool,
    /// U-Verify Freivalds probe count. 0 by default for the harness;
    /// non-zero is recorded if the runner forces probes on for a
    /// follow-up sweep.
    pub verify_probes: usize,
    /// 1 row per prompt: the tokenizer ids the prompt expanded to.
    /// Lets the Python harness reconstruct ground truth without re-
    /// tokenising and gives ISA/IMA their training target.
    pub prompt_token_ids: Vec<Vec<u32>>,
    /// The shape of the masked operand at every layer×kind capture.
    /// `(n_data + shield_k, in_features)`. Recorded as-snapshot, not
    /// inferred from `prompt_token_ids` — KV-cache prefill resets
    /// can produce `n_data ≠ len(prompt)` and the Python loader
    /// trusts what's on disk.
    pub captured_layers: Vec<u16>,
    pub captured_kinds: Vec<String>,
}

#[derive(Serialize)]
pub struct SnapshotMeta {
    pub seq_idx: usize,
    pub layer: u16,
    pub kind: String,
    pub operand_shape: [usize; 2],
    pub output_shape: Option<[usize; 2]>,
    /// `(n_data + shield_k) − shield_k` — the number of data rows
    /// the Python harness should keep when running attacks in
    /// "data-only" mode. Equal to operand_shape[0] in C0/C1
    /// (shield_k = 0) and `operand_shape[0] − 8` in C2.
    pub n_data: usize,
    pub shield_k: usize,
    /// Index into `config.prompt_token_ids` of the prompt this
    /// snapshot belongs to. The Python loader uses this to slice
    /// snapshots into per-prompt traces without having to recompute
    /// ops-per-forward-pass from layer counts.
    pub prompt_idx: usize,
}

/// Write a snapshot batch to `<out_dir>/<basename>.safetensors` plus
/// the sidecar `<out_dir>/<basename>.meta.json`. The Python harness
/// reads both as a unit — see `evals/aloepri-attacks/snapshots_loader.py`.
///
/// Layout:
///
/// * Tensor key per snapshot: `snap{seq_idx:05}.{layer:03}.{kind}.operand`
///   and (when present) `…operand.output`.
/// * Safetensors top-level metadata: pointer to the meta.json basename
///   and a one-line schema version, for forensic forensics if a
///   matched pair gets separated.
pub fn export_snapshots(
    snapshots: &[PcieSnapshot],
    prompt_idx_per_snapshot: &[usize],
    model_id: &str,
    condition: Condition,
    prompt_token_ids: Vec<Vec<u32>>,
    shield_k: usize,
    shield_energy_scale: f32,
    per_forward_mask: bool,
    verify_probes: usize,
    out_dir: &Path,
    basename: &str,
) -> Result<ExportArtifacts> {
    if prompt_idx_per_snapshot.len() != snapshots.len() {
        return Err(anyhow!(
            "prompt_idx_per_snapshot has length {} but snapshots has length {}",
            prompt_idx_per_snapshot.len(),
            snapshots.len()
        ));
    }
    std::fs::create_dir_all(out_dir)
        .with_context(|| format!("creating output dir {}", out_dir.display()))?;

    // Build snapshot metadata + tensor key strings up front so we own
    // the lifetimes for the safetensors writer.
    let mut snapshot_metas = Vec::with_capacity(snapshots.len());
    let mut key_payloads: Vec<(String, Vec<usize>, Vec<u8>)> =
        Vec::with_capacity(snapshots.len() * 2);
    let mut captured_layers = Vec::new();
    let mut captured_kinds = Vec::new();

    for (snap, prompt_idx) in snapshots.iter().zip(prompt_idx_per_snapshot.iter().copied()) {
        let kind_str = kind_slug(snap.kind).to_string();
        let operand_key =
            format!("snap{:05}.{:03}.{}.operand", snap.seq_idx, snap.layer, kind_str);
        let operand_shape = [snap.masked_operand.nrows(), snap.masked_operand.ncols()];
        key_payloads.push((
            operand_key,
            vec![operand_shape[0], operand_shape[1]],
            f32_array_to_bytes(&snap.masked_operand),
        ));

        let output_shape = snap.masked_output.as_ref().map(|out| {
            let shape = [out.nrows(), out.ncols()];
            let output_key =
                format!("snap{:05}.{:03}.{}.output", snap.seq_idx, snap.layer, kind_str);
            key_payloads.push((
                output_key,
                vec![shape[0], shape[1]],
                f32_array_to_bytes(out),
            ));
            shape
        });

        // For Haar (C0/C1/C2), operand_shape[0] = n_data + shield_k, so
        // subtracting shield_k recovers n_data correctly. For HD₃ (C3),
        // the executor pads stacked_n to next_power_of_two(n_data +
        // shield_k); operand_shape[0] is then s_pad, not n_data + shield_k.
        // Trust the prompt's tokenised length instead — it always equals
        // the original data row count regardless of mask family.
        let n_data = prompt_token_ids
            .get(prompt_idx)
            .map(|ids| ids.len())
            .unwrap_or_else(|| operand_shape[0].saturating_sub(shield_k));
        snapshot_metas.push(SnapshotMeta {
            seq_idx: snap.seq_idx,
            layer: snap.layer,
            kind: kind_str.clone(),
            operand_shape,
            output_shape,
            n_data,
            shield_k,
            prompt_idx,
        });

        if !captured_layers.contains(&snap.layer) {
            captured_layers.push(snap.layer);
        }
        if !captured_kinds.contains(&kind_str) {
            captured_kinds.push(kind_str);
        }
    }
    captured_layers.sort_unstable();
    captured_kinds.sort();

    // Now build the safetensors view set. The view borrows the bytes
    // out of `key_payloads`, so the writer must consume both in the
    // same scope.
    let mut views: Vec<(&str, TensorView<'_>)> = Vec::with_capacity(key_payloads.len());
    for (key, shape, bytes) in &key_payloads {
        let view = TensorView::new(Dtype::F32, shape.clone(), bytes)
            .with_context(|| format!("building TensorView for key {key}"))?;
        views.push((key.as_str(), view));
    }

    let safetensors_path = out_dir.join(format!("{basename}.safetensors"));
    let metadata_pointer = format!("{basename}.meta.json");
    let mut top_meta = std::collections::HashMap::new();
    top_meta.insert("schema_version".to_string(), "1".to_string());
    top_meta.insert("meta_json".to_string(), metadata_pointer.clone());
    top_meta.insert("condition".to_string(), condition.slug().to_string());
    safetensors::serialize_to_file(views, &Some(top_meta), &safetensors_path)
        .with_context(|| format!("writing safetensors to {}", safetensors_path.display()))?;

    let meta_path = out_dir.join(&metadata_pointer);
    let export_meta = ExportMeta {
        schema_version: "1".to_string(),
        model_id: model_id.to_string(),
        condition: condition.slug().to_string(),
        config: ExportRunConfig {
            shield_k,
            shield_energy_scale,
            per_forward_mask,
            verify_probes,
            prompt_token_ids,
            captured_layers,
            captured_kinds,
        },
        snapshots: snapshot_metas,
    };
    let mut meta_file = File::create(&meta_path)
        .with_context(|| format!("opening meta.json at {}", meta_path.display()))?;
    let pretty = serde_json::to_string_pretty(&export_meta)?;
    meta_file
        .write_all(pretty.as_bytes())
        .with_context(|| format!("writing meta.json at {}", meta_path.display()))?;

    Ok(ExportArtifacts {
        safetensors_path,
        meta_path,
        snapshot_count: snapshots.len(),
    })
}

#[derive(Debug)]
pub struct ExportArtifacts {
    pub safetensors_path: std::path::PathBuf,
    pub meta_path: std::path::PathBuf,
    pub snapshot_count: usize,
}

fn f32_array_to_bytes(arr: &Array2<f32>) -> Vec<u8> {
    // Force a row-major contiguous copy so the safetensors byte layout
    // matches the (rows, cols) shape header. The decoder's `Array2`
    // can come back in either layout depending on the engine's slice
    // pattern, so going through `.to_owned()` on a default-layout
    // target is the safest path.
    let contiguous = if arr.is_standard_layout() {
        arr.view().to_owned()
    } else {
        Array2::from_shape_fn(arr.dim(), |(i, j)| arr[[i, j]])
    };
    let (head, body, tail) = unsafe { contiguous.as_slice().unwrap().align_to::<u8>() };
    debug_assert!(head.is_empty() && tail.is_empty());
    body.to_vec()
}

// ─── C0 control: CapturingPlaintextExecutor ────────────────────────────

/// Thin wrapper around `PlaintextExecutor` that captures every offload
/// to its own [`SnapshotCapture`] before forwarding.
///
/// We need this because `SnapshotCapture` is currently only wired into
/// `InProcessTrustedExecutor` (Phase 1 scope) — and the handoff rule
/// is "don't touch gelo-protocol". For C0 the "PCIe-side adversary"
/// observation is the *raw* hidden state (no mask, no shield), which
/// the wrapper records directly.
pub struct CapturingPlaintextExecutor<E: GpuOffloadEngine> {
    inner: PlaintextExecutor<E>,
    capture: Option<SnapshotCapture>,
}

impl<E: GpuOffloadEngine> CapturingPlaintextExecutor<E> {
    pub fn new(engine: E) -> Self {
        Self {
            inner: PlaintextExecutor::new(engine),
            capture: None,
        }
    }

    pub fn with_snapshot_capture(mut self, cfg: SnapshotConfig) -> Self {
        self.capture = Some(SnapshotCapture::new(cfg));
        self
    }

    pub fn pcie_snapshots(&self) -> &[PcieSnapshot] {
        self.capture
            .as_ref()
            .map(SnapshotCapture::snapshots)
            .unwrap_or(&[])
    }

    pub fn drain_pcie_snapshots(&mut self) -> Vec<PcieSnapshot> {
        self.capture
            .as_mut()
            .map(SnapshotCapture::drain)
            .unwrap_or_default()
    }

    /// Equivalent of `SnapshotCapture::dropped()` — should be zero on
    /// the default 4096-cap for Qwen3-class models.
    pub fn dropped_snapshots(&self) -> usize {
        self.capture.as_ref().map(SnapshotCapture::dropped).unwrap_or(0)
    }

    fn record_linear(&mut self, handle: WeightHandle, input: &Array2<f32>, output: &Array2<f32>) {
        if let Some(cap) = self.capture.as_mut() {
            cap.record(handle, input, Some(output));
        }
    }
}

impl<E: GpuOffloadEngine> TrustedExecutor for CapturingPlaintextExecutor<E> {
    fn provision_weight(&mut self, handle: WeightHandle, weight: ArrayView2<f32>) -> Result<()> {
        self.inner.provision_weight(handle, weight)
    }

    fn provision_weight_shared(
        &mut self,
        handle: WeightHandle,
        weight: std::sync::Arc<Array2<f32>>,
    ) -> Result<()> {
        self.inner.provision_weight_shared(handle, weight)
    }

    fn provision_weight_bf16(
        &mut self,
        handle: WeightHandle,
        weight: ArrayView2<half::bf16>,
    ) -> Result<()> {
        self.inner.provision_weight_bf16(handle, weight)
    }

    fn provision_weight_bf16_shared(
        &mut self,
        handle: WeightHandle,
        weight: std::sync::Arc<Array2<half::bf16>>,
    ) -> Result<()> {
        self.inner.provision_weight_bf16_shared(handle, weight)
    }

    fn begin_forward_pass(&mut self, n: usize) -> Result<()> {
        self.inner.begin_forward_pass(n)
    }

    fn end_forward_pass(&mut self) -> Result<()> {
        self.inner.end_forward_pass()
    }

    fn offload_linear(
        &mut self,
        handle: WeightHandle,
        hidden: ArrayView2<f32>,
    ) -> Result<Array2<f32>> {
        let input_owned = hidden.to_owned();
        let out = self.inner.offload_linear(handle, hidden)?;
        self.record_linear(handle, &input_owned, &out);
        Ok(out)
    }

    fn offload_qkv(
        &mut self,
        layer: u16,
        hidden: ArrayView2<f32>,
    ) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>)> {
        let input_owned = hidden.to_owned();
        let (q, k, v) = self.inner.offload_qkv(layer, hidden)?;
        // Match the InProcess executor's recording shape: three snapshots
        // sharing one operand. seq_idx assignments come from the
        // SnapshotCapture sequencer in record() order.
        self.record_linear(WeightHandle::new(layer, WeightKind::Q), &input_owned, &q);
        self.record_linear(WeightHandle::new(layer, WeightKind::K), &input_owned, &k);
        self.record_linear(WeightHandle::new(layer, WeightKind::V), &input_owned, &v);
        Ok((q, k, v))
    }

    fn offload_linear_many(
        &mut self,
        handles: &[WeightHandle],
        hidden: ArrayView2<f32>,
    ) -> Result<Vec<Array2<f32>>> {
        let input_owned = hidden.to_owned();
        let outs = self.inner.offload_linear_many(handles, hidden)?;
        for (h, o) in handles.iter().zip(outs.iter()) {
            self.record_linear(*h, &input_owned, o);
        }
        Ok(outs)
    }

    fn offload_attention_qkt(
        &mut self,
        q: ArrayView2<f32>,
        kt: ArrayView2<f32>,
    ) -> Result<Array2<f32>> {
        // PlaintextExecutor::offload_attention_qkt is unimplemented!() by
        // default — the embedder never reaches it in the encoder path,
        // and the decoder uses fused_attention. We mirror the upstream
        // behaviour so a silent fallback can't mask a missing capture.
        self.inner.offload_attention_qkt(q, kt)
    }

    fn offload_attention_permuted(
        &mut self,
        q: ArrayView3<f32>,
        k: ArrayView3<f32>,
        v: ArrayView3<f32>,
        scale: f32,
        mask: gelo_protocol::attention::AttentionMask,
    ) -> Result<Array3<f32>> {
        // The Phase 1 snapshot module only records linear-projection
        // pairs; attention-permuted is the fused softmax(QK^T)V path
        // and has no analog in the AloePri attack inputs. We forward
        // without recording.
        self.inner.offload_attention_permuted(q, k, v, scale, mask)
    }

    fn offload_attention_permuted_cached(
        &mut self,
        q: ArrayView3<f32>,
        k: ArrayView3<f32>,
        v: ArrayView3<f32>,
        scale: f32,
        q_pos_offset: usize,
        mask: gelo_protocol::attention::AttentionMask,
    ) -> Result<Array3<f32>> {
        self.inner
            .offload_attention_permuted_cached(q, k, v, scale, q_pos_offset, mask)
    }

    fn offload_attention_qkt_batched(
        &mut self,
        q: ArrayView3<f32>,
        kt: ArrayView3<f32>,
    ) -> Result<Array3<f32>> {
        let h = q.shape()[0];
        let n = q.shape()[1];
        let mut out = Array3::<f32>::zeros((h, n, n));
        for i in 0..h {
            let r = self.offload_attention_qkt(
                q.index_axis(Axis(0), i),
                kt.index_axis(Axis(0), i),
            )?;
            out.index_axis_mut(Axis(0), i).assign(&r);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gelo_protocol::{RayonCpuEngine, SnapshotConfig};
    use ndarray::Array2;

    /// Roundtrip: build a small `Vec<PcieSnapshot>` by hand, write it
    /// through `export_snapshots`, then re-read the safetensors +
    /// meta.json and assert on shape/name parity.
    #[test]
    fn export_roundtrip_shapes_and_keys_match() {
        use std::collections::HashMap;

        let snaps = vec![
            PcieSnapshot {
                seq_idx: 0,
                layer: 0,
                kind: WeightKind::Q,
                masked_operand: Array2::from_shape_fn((4, 8), |(r, c)| (r * 8 + c) as f32),
                masked_output: Some(Array2::from_shape_fn((4, 16), |(r, c)| (r + c) as f32)),
            },
            PcieSnapshot {
                seq_idx: 1,
                layer: 0,
                kind: WeightKind::FfnGate,
                masked_operand: Array2::from_shape_fn((4, 8), |(r, c)| (r + c) as f32),
                masked_output: None,
            },
        ];

        let dir = tempdir();
        let prompt_idx = vec![0usize, 0usize];
        let out = export_snapshots(
            &snaps,
            &prompt_idx,
            "test/qwen3-1.7B",
            Condition::C0Plain,
            vec![vec![1, 2, 3, 4]],
            0,
            0.0,
            false,
            0,
            &dir,
            "fixture",
        )
        .expect("export_snapshots succeeds");
        assert_eq!(out.snapshot_count, 2);

        let st_bytes = std::fs::read(&out.safetensors_path).unwrap();
        let st = safetensors::SafeTensors::deserialize(&st_bytes).unwrap();

        // operand keys both present, output only for snapshot 0.
        let names: HashMap<&str, _> = st.names().iter().map(|n| (n.as_str(), ())).collect();
        assert!(names.contains_key("snap00000.000.q_proj.operand"));
        assert!(names.contains_key("snap00000.000.q_proj.output"));
        assert!(names.contains_key("snap00001.000.gate_proj.operand"));
        assert!(!names.contains_key("snap00001.000.gate_proj.output"));

        let op = st.tensor("snap00000.000.q_proj.operand").unwrap();
        assert_eq!(op.shape(), &[4, 8]);
        assert_eq!(op.dtype(), Dtype::F32);
        let out0 = st.tensor("snap00000.000.q_proj.output").unwrap();
        assert_eq!(out0.shape(), &[4, 16]);

        let meta_text = std::fs::read_to_string(&out.meta_path).unwrap();
        let meta: serde_json::Value = serde_json::from_str(&meta_text).unwrap();
        assert_eq!(meta["schema_version"], "1");
        assert_eq!(meta["condition"], "c0_plain");
        assert_eq!(meta["snapshots"].as_array().unwrap().len(), 2);
        assert_eq!(meta["snapshots"][0]["kind"], "q_proj");
        assert_eq!(meta["snapshots"][1]["output_shape"], serde_json::Value::Null);
    }

    /// C0 capturing executor sees unmasked operands. We verify by
    /// provisioning a weight, calling `offload_linear`, and checking
    /// the captured operand is byte-identical to the input we passed.
    #[test]
    fn capturing_plaintext_records_unmasked_operand() {
        let engine = RayonCpuEngine::new();
        let mut exec = CapturingPlaintextExecutor::new(engine)
            .with_snapshot_capture(SnapshotConfig::default());

        let weight = Array2::from_shape_fn((8, 16), |(r, c)| (r + c) as f32);
        let handle = WeightHandle::new(0, WeightKind::Q);
        exec.provision_weight(handle, weight.view()).unwrap();

        let input = Array2::from_shape_fn((4, 8), |(r, c)| (r * 8 + c) as f32);
        exec.begin_forward_pass(4).unwrap();
        let _out = exec.offload_linear(handle, input.view()).unwrap();
        exec.end_forward_pass().unwrap();

        let snaps = exec.drain_pcie_snapshots();
        assert_eq!(snaps.len(), 1);
        assert_eq!(snaps[0].masked_operand, input);
        let expected_out = input.dot(&weight);
        assert_eq!(snaps[0].masked_output.as_ref().unwrap(), &expected_out);
    }

    fn tempdir() -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "aloepri-attack-roundtrip-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}

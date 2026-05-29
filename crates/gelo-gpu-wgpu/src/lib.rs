//! [`GpuOffloadEngine`] backed by **burn-cubecl** on the **wgpu/Vulkan** runtime.
//!
//! Replaces the prior cubecl-matmul direct path with `burn_tensor::Tensor::matmul`
//! over the `CubeBackend<Rt, …>` backend. burn-cubecl wires:
//! - Real autotune with disk-persistent cache (via `cubecl-runtime::TuneCache`,
//!   configured by workspace-root `cubecl.toml`).
//! - Lazy / deferred dispatch — sync only happens at `.into_data()`.
//! - Built-in buffer pooling and kernel fusion (`burn-cubecl-fusion`).
//!
//! The trait surface (`GpuOffloadEngine` from `gelo-protocol`) is unchanged.
//! The GELO mask round-trip math stays on the trusted/TEE side (CPU);
//! only the masked product `A·H` becomes a `Tensor` on the engine side.
//!
//! On Linux this dispatches via Vulkan; on macOS via Metal; on Windows via DX12.
//!
//! ## Precision modes
//!
//! - **`new()`** — default, f32 throughout. Highest fidelity, full
//!   U-Verify compatibility.
//! - **`new_fp16()`** — engine internal element type is f16 (`half::f16`).
//!   Inputs/outputs are converted at the f32 ↔ f16 boundary inside the
//!   engine; the trait surface remains f32 so trusted-side code is
//!   unchanged. Expected ~1.5–2× faster GEMM kernels on Vulkan
//!   `shader-f16`-capable adapters at the cost of ~3–4 decimal digits
//!   of precision. **U-Verify probes must be widened or disabled** under
//!   fp16 — the engine's matmul output is not bit-equal to the trusted
//!   side's f32 reference. Confirms via the `WgpuVulkanEngine::is_fp16()`
//!   accessor.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Result, anyhow};
use burn_backend::Backend;
use burn_cubecl::CubeBackend;
use burn_tensor::{Tensor, TensorData, Transaction, activation};
// Backend runtime + device, selected at compile time: wgpu/Vulkan by
// default (OEM-agnostic), cubecl-cuda under the `cuda` feature. The rest
// of the file refers to the runtime/device only through `Rt` / `Dev`.
#[cfg(not(feature = "cuda"))]
use cubecl_common::future;
#[cfg(not(feature = "cuda"))]
use cubecl_wgpu::{AutoGraphicsApi, RuntimeOptions, WgpuDevice, WgpuRuntime, init_setup_async};
#[cfg(feature = "cuda")]
use cubecl_cuda::{CudaDevice, CudaRuntime};

#[cfg(not(feature = "cuda"))]
type Rt = WgpuRuntime;
#[cfg(not(feature = "cuda"))]
type Dev = WgpuDevice;
#[cfg(feature = "cuda")]
type Rt = CudaRuntime;
#[cfg(feature = "cuda")]
type Dev = CudaDevice;
use half::{bf16, f16};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};

use gelo_protocol::{GpuOffloadEngine, MatmulToken, WeightHandle};

/// burn-cubecl backend specialised to f32 floats. The default engine
/// precision.
type CubeWgpu32 = CubeBackend<Rt, f32, i32, u8>;

/// burn-cubecl backend specialised to f16 floats. Used by the fp16
/// engine path. Requires the wgpu adapter to support the `shader-f16`
/// extension (true on AMD RDNA2/3, NVIDIA Maxwell+, most modern Intel
/// iGPUs).
type CubeWgpu16 = CubeBackend<Rt, f16, i32, u8>;

/// Backend device class, abstracted over the wgpu / CUDA split so the
/// engine reports device identity uniformly across both runtimes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDeviceType {
    Discrete,
    Integrated,
    Virtual,
    Cpu,
    Other,
}

/// Per-process GPU identity, captured once at first device init.
pub struct GpuContext {
    pub name: String,
    pub device_type: GpuDeviceType,
    pub backend: String,
}

static GPU_CTX: OnceLock<GpuContext> = OnceLock::new();

#[cfg(not(feature = "cuda"))]
fn gpu_ctx() -> &'static GpuContext {
    GPU_CTX.get_or_init(|| {
        let device = Dev::default();
        let setup = future::block_on(init_setup_async::<AutoGraphicsApi>(
            &device,
            RuntimeOptions::default(),
        ));
        let info = setup.adapter.get_info();
        let device_type = match info.device_type {
            wgpu::DeviceType::DiscreteGpu => GpuDeviceType::Discrete,
            wgpu::DeviceType::IntegratedGpu => GpuDeviceType::Integrated,
            wgpu::DeviceType::VirtualGpu => GpuDeviceType::Virtual,
            wgpu::DeviceType::Cpu => GpuDeviceType::Cpu,
            _ => GpuDeviceType::Other,
        };
        GpuContext {
            backend: format!("{:?}", info.backend),
            name: info.name,
            device_type,
        }
    })
}

#[cfg(feature = "cuda")]
fn gpu_ctx() -> &'static GpuContext {
    // cubecl-cuda has no wgpu-style adapter enumeration; the CUDA context
    // is created lazily on the first `Backend::sync`. Report a fixed
    // identity for device 0 — nvidia-smi confirms which physical card.
    GPU_CTX.get_or_init(|| GpuContext {
        name: "CUDA device 0 (cubecl-cuda)".to_string(),
        device_type: GpuDeviceType::Discrete,
        backend: "Cuda".to_string(),
    })
}

/// Dispatch enum holding the precision-specific weight map.
enum WeightStore {
    F32(HashMap<WeightHandle, Tensor<CubeWgpu32, 2>>),
    F16(HashMap<WeightHandle, Tensor<CubeWgpu16, 2>>),
}

/// burn-cubecl/wgpu-backed offload engine.
///
/// Registered weights live device-resident as `Tensor<…, 2>` in either
/// f32 or f16. `clone_shared()` produces a second handle pointing at the
/// same weight cache. Precision is fixed at construction
/// ([`Self::new`] vs [`Self::new_fp16`]).
pub struct WgpuVulkanEngine {
    device: Dev,
    weights: Arc<Mutex<WeightStore>>,
    fp16: bool,
}

impl WgpuVulkanEngine {
    /// Initialise a Vulkan-preferred wgpu device via burn-cubecl, with
    /// f32 internal precision.
    pub fn new() -> Result<Self> {
        let _ = gpu_ctx();
        let device = Dev::default();
        <CubeWgpu32 as Backend>::sync(&device)
            .map_err(|e| anyhow!("burn-cubecl device sync at init: {e:?}"))?;
        Ok(Self {
            device,
            weights: Arc::new(Mutex::new(WeightStore::F32(HashMap::new()))),
            fp16: false,
        })
    }

    /// Initialise the engine with **f16 internal precision** for the
    /// GEMM kernel. Inputs/outputs cross the trait boundary as f32; the
    /// conversion to/from f16 happens inside `register_weight` and each
    /// `matmul*` call. See module docs for the U-Verify caveat.
    ///
    /// Fails if the adapter doesn't support `shader-f16`. (cubecl checks
    /// this lazily; the first matmul will surface the error.)
    pub fn new_fp16() -> Result<Self> {
        let _ = gpu_ctx();
        let device = Dev::default();
        <CubeWgpu16 as Backend>::sync(&device)
            .map_err(|e| anyhow!("burn-cubecl device sync at init: {e:?}"))?;
        Ok(Self {
            device,
            weights: Arc::new(Mutex::new(WeightStore::F16(HashMap::new()))),
            fp16: true,
        })
    }

    /// Second handle sharing the registered-weight cache with `self`.
    pub fn clone_shared(&self) -> Self {
        Self {
            device: self.device.clone(),
            weights: Arc::clone(&self.weights),
            fp16: self.fp16,
        }
    }
}

/// `Clone` delegates to `clone_shared` — both handles point at the same
/// `Arc`-backed weight cache, so the `Embedder::embed` rayon fan-out can
/// hand each worker its own engine handle without duplicating the
/// device-resident weight tensors.
impl Clone for WgpuVulkanEngine {
    fn clone(&self) -> Self {
        self.clone_shared()
    }
}

/// Device-resident f16 K/V for the gate-1 persistent-attention
/// microbench (`docs/plans/perm-attn-gpu-offload.md`). Holds the K/V
/// tensors on the GPU across decode steps so the per-step path uploads
/// only Q — isolating the resident-read cost from the per-call K/V
/// upload+convert that dominates `fused_attention_batched`. Seed of the
/// eventual session-resident K/V API (not the production surface yet).
pub struct ResidentKvF16 {
    k_t: Tensor<CubeWgpu16, 3>,
    v_t: Tensor<CubeWgpu16, 3>,
}

impl WgpuVulkanEngine {
    /// Upload + convert K/V to device-resident f16 tensors **once**.
    /// `k`, `v`: `(B·H, n_kv, d_head)`. fp16 engine only.
    pub fn upload_resident_kv(
        &self,
        k: ArrayView3<'_, f32>,
        v: ArrayView3<'_, f32>,
    ) -> Result<ResidentKvF16> {
        if !self.fp16 {
            return Err(anyhow!("upload_resident_kv requires the fp16 engine"));
        }
        Ok(ResidentKvF16 {
            k_t: array3_to_tensor_f16(k, &self.device),
            v_t: array3_to_tensor_f16(v, &self.device),
        })
    }

    /// Attention against device-resident K/V: uploads only `q`
    /// `(B·H, n_q, d_head)`, computes `softmax(q·kᵀ·scale)·v`, reads
    /// back the context `(B·H, n_q, d_head)`. No mask (decode m=1 causal
    /// is a no-op). This is `fused_attention_batched` **minus** the
    /// per-call K/V upload+convert — the gate-1 resident-read
    /// measurement. fp16 engine only.
    pub fn attend_resident(
        &self,
        q: ArrayView3<'_, f32>,
        kv: &ResidentKvF16,
        scale: f32,
    ) -> Result<Array3<f32>> {
        if !self.fp16 {
            return Err(anyhow!("attend_resident requires the fp16 engine"));
        }
        let q_t = array3_to_tensor_f16(q, &self.device);
        let kt = kv.k_t.clone().permute([0, 2, 1]);
        let scores = q_t.matmul(kt).mul_scalar(scale);
        let probs = activation::softmax(scores, 2);
        let out = probs.matmul(kv.v_t.clone());
        tensor3_to_array_f16(out)
    }
}

/// Device-resident **growing** K/V session for the representative
/// decode microbench (gate 1, `docs/plans/perm-attn-gpu-offload.md`).
/// Models the **optimistic prefill-only-permute** case: the cover is
/// applied once at `create_kv_session` (prefill); decode only *appends*
/// new rows and attends over the active slice — **no per-block
/// re-permute** (the N=∞ best case the security gate will later test).
/// K/V are pre-allocated to `capacity` on the n_kv axis so `append_kv`
/// is an O(1) in-place row write (`slice_assign` on a sole-owner tensor),
/// not an O(n) recopy.
pub struct ResidentKvSession {
    k_t: Tensor<CubeWgpu16, 3>,
    v_t: Tensor<CubeWgpu16, 3>,
    len: usize,
    capacity: usize,
}

impl WgpuVulkanEngine {
    /// Prefill: pre-allocate `(B·H, capacity, d_head)` and write the
    /// (already TEE-permuted/noised) prefix into `[0..n0]`. One-time;
    /// the per-step decode path never re-touches it. fp16 only.
    pub fn create_kv_session(
        &self,
        k_prefix: ArrayView3<'_, f32>,
        v_prefix: ArrayView3<'_, f32>,
        capacity: usize,
    ) -> Result<ResidentKvSession> {
        if !self.fp16 {
            return Err(anyhow!("create_kv_session requires the fp16 engine"));
        }
        let (bh, n0, d) = k_prefix.dim();
        if n0 > capacity {
            return Err(anyhow!("create_kv_session: prefix {n0} > capacity {capacity}"));
        }
        let k_full = Tensor::<CubeWgpu16, 3>::zeros([bh, capacity, d], &self.device);
        let v_full = Tensor::<CubeWgpu16, 3>::zeros([bh, capacity, d], &self.device);
        let k_t = k_full.slice_assign([0..bh, 0..n0, 0..d], array3_to_tensor_f16(k_prefix, &self.device));
        let v_t = v_full.slice_assign([0..bh, 0..n0, 0..d], array3_to_tensor_f16(v_prefix, &self.device));
        Ok(ResidentKvSession { k_t, v_t, len: n0, capacity })
    }

    /// Decode append: write one token's `(B·H, 1, d_head)` K/V row at
    /// the current length (O(1) in-place). At capacity it overwrites the
    /// last slot (the bench holds context fixed); production grows `len`.
    pub fn append_kv(
        &self,
        session: &mut ResidentKvSession,
        k_row: ArrayView3<'_, f32>,
        v_row: ArrayView3<'_, f32>,
    ) -> Result<()> {
        let (bh, _one, d) = k_row.dim();
        let idx = session.len.min(session.capacity - 1);
        let kr = array3_to_tensor_f16(k_row, &self.device);
        let vr = array3_to_tensor_f16(v_row, &self.device);
        // mem::replace so each tensor is the sole owner during
        // slice_assign → cubecl mutates the resident buffer in place.
        let dummy = || Tensor::<CubeWgpu16, 3>::zeros([1, 1, 1], &self.device);
        let k_t = std::mem::replace(&mut session.k_t, dummy());
        let v_t = std::mem::replace(&mut session.v_t, dummy());
        session.k_t = k_t.slice_assign([0..bh, idx..idx + 1, 0..d], kr);
        session.v_t = v_t.slice_assign([0..bh, idx..idx + 1, 0..d], vr);
        if session.len < session.capacity {
            session.len += 1;
        }
        Ok(())
    }

    /// Attend over the active `[0..len]` slice: upload only `q`
    /// `(B·H, n_q, d_head)`, `softmax(q·kᵀ·scale)·v`, read back context.
    /// The per-step decode read against the growing resident cache.
    pub fn attend_session(
        &self,
        q: ArrayView3<'_, f32>,
        session: &ResidentKvSession,
        scale: f32,
    ) -> Result<Array3<f32>> {
        let (bh, _nq, d) = q.dim();
        let q_t = array3_to_tensor_f16(q, &self.device);
        let k_act = session.k_t.clone().slice([0..bh, 0..session.len, 0..d]);
        let v_act = session.v_t.clone().slice([0..bh, 0..session.len, 0..d]);
        let kt = k_act.permute([0, 2, 1]);
        let scores = q_t.matmul(kt).mul_scalar(scale);
        let probs = activation::softmax(scores, 2);
        let out = probs.matmul(v_act);
        tensor3_to_array_f16(out)
    }
}

impl WgpuVulkanEngine {
    /// `true` if this engine handle runs GEMM kernels in f16. Trusted-
    /// side code that needs bit-equal matmul output (e.g. U-Verify) must
    /// gate on this.
    pub fn is_fp16(&self) -> bool {
        self.fp16
    }

    /// Backend name (`"Vulkan"`/`"Metal"`/… on wgpu; `"Cuda"` under the
    /// `cuda` feature).
    pub fn backend(&self) -> String {
        gpu_ctx().backend.clone()
    }

    /// GPU identity captured at first init (`.name`, `.device_type`).
    pub fn adapter_info(&self) -> &'static GpuContext {
        gpu_ctx()
    }

    /// `true` if the selected device is a real GPU (discrete, integrated,
    /// or virtual) — not a software rasterizer like lavapipe.
    pub fn is_real_gpu(&self) -> bool {
        matches!(
            gpu_ctx().device_type,
            GpuDeviceType::Discrete | GpuDeviceType::Integrated | GpuDeviceType::Virtual
        )
    }
}

// ─── f32 conversion helpers ───────────────────────────────────────────

fn array2_to_tensor_f32(view: ArrayView2<'_, f32>, device: &Dev) -> Tensor<CubeWgpu32, 2> {
    let rows = view.nrows();
    let cols = view.ncols();
    let v: Vec<f32> = view.as_standard_layout().iter().copied().collect();
    Tensor::<CubeWgpu32, 2>::from_data(TensorData::new(v, [rows, cols]), device)
}

fn array3_to_tensor_f32(view: ArrayView3<'_, f32>, device: &Dev) -> Tensor<CubeWgpu32, 3> {
    let b = view.shape()[0];
    let m = view.shape()[1];
    let k = view.shape()[2];
    let v: Vec<f32> = view.as_standard_layout().iter().copied().collect();
    Tensor::<CubeWgpu32, 3>::from_data(TensorData::new(v, [b, m, k]), device)
}

fn tensor2_to_array_f32(t: Tensor<CubeWgpu32, 2>) -> Result<Array2<f32>> {
    let shape = t.dims();
    let v: Vec<f32> = t
        .into_data()
        .into_vec()
        .map_err(|e| anyhow!("burn f32 tensor → Vec<f32>: {e:?}"))?;
    Array2::from_shape_vec((shape[0], shape[1]), v)
        .map_err(|e| anyhow!("Array2 from tensor data: {e}"))
}

fn tensor3_to_array_f32(t: Tensor<CubeWgpu32, 3>) -> Result<Array3<f32>> {
    let shape = t.dims();
    let v: Vec<f32> = t
        .into_data()
        .into_vec()
        .map_err(|e| anyhow!("burn f32 tensor → Vec<f32>: {e:?}"))?;
    Array3::from_shape_vec((shape[0], shape[1], shape[2]), v)
        .map_err(|e| anyhow!("Array3 from tensor data: {e}"))
}

// ─── f16 conversion helpers ───────────────────────────────────────────

fn array2_to_tensor_f16(view: ArrayView2<'_, f32>, device: &Dev) -> Tensor<CubeWgpu16, 2> {
    let rows = view.nrows();
    let cols = view.ncols();
    // f32→f16 conversion. LLVM auto-vectorises this to vcvtps2ph on x86
    // with AVX2 enabled (default in release).
    let v: Vec<f16> = view
        .as_standard_layout()
        .iter()
        .map(|&x| f16::from_f32(x))
        .collect();
    Tensor::<CubeWgpu16, 2>::from_data(TensorData::new(v, [rows, cols]), device)
}

/// **bf16-native** weight upload. Skips the bf16 → f32 host
/// intermediate that `view_to_f32` would otherwise force on the
/// loader. Each bf16 element is converted directly to f16 via the
/// `f16::from_f32(bf16::to_f32(x))` round-trip — same numeric path
/// as the f32 entry point but without ever materialising an f32
/// host copy of the full weight matrix.
fn array2_bf16_to_tensor_f16(
    view: ArrayView2<'_, bf16>,
    device: &Dev,
) -> Tensor<CubeWgpu16, 2> {
    let rows = view.nrows();
    let cols = view.ncols();
    let v: Vec<f16> = view
        .as_standard_layout()
        .iter()
        .map(|&x| f16::from_f32(x.to_f32()))
        .collect();
    Tensor::<CubeWgpu16, 2>::from_data(TensorData::new(v, [rows, cols]), device)
}

/// **bf16 → f32 GPU upload**. Used when the engine is in F32 mode but
/// the caller supplied bf16. Still avoids a host f32 array — the
/// per-element widening happens once during the upload Vec build.
fn array2_bf16_to_tensor_f32(
    view: ArrayView2<'_, bf16>,
    device: &Dev,
) -> Tensor<CubeWgpu32, 2> {
    let rows = view.nrows();
    let cols = view.ncols();
    let v: Vec<f32> = view
        .as_standard_layout()
        .iter()
        .map(|&x| x.to_f32())
        .collect();
    Tensor::<CubeWgpu32, 2>::from_data(TensorData::new(v, [rows, cols]), device)
}

fn array3_to_tensor_f16(view: ArrayView3<'_, f32>, device: &Dev) -> Tensor<CubeWgpu16, 3> {
    let b = view.shape()[0];
    let m = view.shape()[1];
    let k = view.shape()[2];
    let v: Vec<f16> = view
        .as_standard_layout()
        .iter()
        .map(|&x| f16::from_f32(x))
        .collect();
    Tensor::<CubeWgpu16, 3>::from_data(TensorData::new(v, [b, m, k]), device)
}

fn tensor2_to_array_f16(t: Tensor<CubeWgpu16, 2>) -> Result<Array2<f32>> {
    let shape = t.dims();
    let v_f16: Vec<f16> = t
        .into_data()
        .into_vec()
        .map_err(|e| anyhow!("burn f16 tensor → Vec<f16>: {e:?}"))?;
    let v: Vec<f32> = v_f16.into_iter().map(|x| x.to_f32()).collect();
    Array2::from_shape_vec((shape[0], shape[1]), v)
        .map_err(|e| anyhow!("Array2 from tensor data: {e}"))
}

fn tensor3_to_array_f16(t: Tensor<CubeWgpu16, 3>) -> Result<Array3<f32>> {
    let shape = t.dims();
    let v_f16: Vec<f16> = t
        .into_data()
        .into_vec()
        .map_err(|e| anyhow!("burn f16 tensor → Vec<f16>: {e:?}"))?;
    let v: Vec<f32> = v_f16.into_iter().map(|x| x.to_f32()).collect();
    Array3::from_shape_vec((shape[0], shape[1], shape[2]), v)
        .map_err(|e| anyhow!("Array3 from tensor data: {e}"))
}

fn registered_weight_f32(
    map: &HashMap<WeightHandle, Tensor<CubeWgpu32, 2>>,
    handle: WeightHandle,
    input_cols: usize,
    op: &str,
) -> Result<Tensor<CubeWgpu32, 2>> {
    let weight = map
        .get(&handle)
        .ok_or_else(|| anyhow!("weight {handle:?} not registered"))?
        .clone();
    if input_cols != weight.dims()[0] {
        return Err(anyhow!(
            "{op} shape mismatch on {handle:?}: input cols {input_cols} != weight rows {}",
            weight.dims()[0]
        ));
    }
    Ok(weight)
}

fn registered_weight_f16(
    map: &HashMap<WeightHandle, Tensor<CubeWgpu16, 2>>,
    handle: WeightHandle,
    input_cols: usize,
    op: &str,
) -> Result<Tensor<CubeWgpu16, 2>> {
    let weight = map
        .get(&handle)
        .ok_or_else(|| anyhow!("weight {handle:?} not registered"))?
        .clone();
    if input_cols != weight.dims()[0] {
        return Err(anyhow!(
            "{op} shape mismatch on {handle:?}: input cols {input_cols} != weight rows {}",
            weight.dims()[0]
        ));
    }
    Ok(weight)
}

fn registered_weights_f32(
    map: &HashMap<WeightHandle, Tensor<CubeWgpu32, 2>>,
    handles: &[WeightHandle],
    input_cols: usize,
    op: &str,
) -> Result<Vec<Tensor<CubeWgpu32, 2>>> {
    handles
        .iter()
        .map(|h| registered_weight_f32(map, *h, input_cols, op))
        .collect()
}

fn registered_weights_f16(
    map: &HashMap<WeightHandle, Tensor<CubeWgpu16, 2>>,
    handles: &[WeightHandle],
    input_cols: usize,
    op: &str,
) -> Result<Vec<Tensor<CubeWgpu16, 2>>> {
    handles
        .iter()
        .map(|h| registered_weight_f16(map, *h, input_cols, op))
        .collect()
}

fn tensor_data_to_array_f32(data: TensorData, rows: usize, cols: usize) -> Result<Array2<f32>> {
    let v: Vec<f32> = data
        .into_vec()
        .map_err(|e| anyhow!("burn f32 TensorData -> Vec<f32>: {e:?}"))?;
    Array2::from_shape_vec((rows, cols), v).map_err(|e| anyhow!("Array2 from tensor data: {e}"))
}

fn tensor_data_to_array_f16(data: TensorData, rows: usize, cols: usize) -> Result<Array2<f32>> {
    let v_f16: Vec<f16> = data
        .into_vec()
        .map_err(|e| anyhow!("burn f16 TensorData -> Vec<f16>: {e:?}"))?;
    let v: Vec<f32> = v_f16.into_iter().map(|x| x.to_f32()).collect();
    Array2::from_shape_vec((rows, cols), v).map_err(|e| anyhow!("Array2 from tensor data: {e}"))
}

fn execute_registered_many_f32(
    lhs: Tensor<CubeWgpu32, 2>,
    weights: Vec<Tensor<CubeWgpu32, 2>>,
) -> Result<Vec<Array2<f32>>> {
    let mut out_dims: Vec<(usize, usize)> = Vec::with_capacity(weights.len());
    let mut tx = Transaction::<CubeWgpu32>::default();
    for w in weights {
        let out = lhs.clone().matmul(w);
        let d = out.dims();
        out_dims.push((d[0], d[1]));
        tx = tx.register(out);
    }
    tx.execute()
        .into_iter()
        .zip(out_dims)
        .map(|(data, (rows, cols))| tensor_data_to_array_f32(data, rows, cols))
        .collect()
}

fn execute_registered_many_f16(
    lhs: Tensor<CubeWgpu16, 2>,
    weights: Vec<Tensor<CubeWgpu16, 2>>,
) -> Result<Vec<Array2<f32>>> {
    let mut out_dims: Vec<(usize, usize)> = Vec::with_capacity(weights.len());
    let mut tx = Transaction::<CubeWgpu16>::default();
    for w in weights {
        let out = lhs.clone().matmul(w);
        let d = out.dims();
        out_dims.push((d[0], d[1]));
        tx = tx.register(out);
    }
    tx.execute()
        .into_iter()
        .zip(out_dims)
        .map(|(data, (rows, cols))| tensor_data_to_array_f16(data, rows, cols))
        .collect()
}

fn submit_registered_many_f32(
    lhs: Tensor<CubeWgpu32, 2>,
    weights: Vec<Tensor<CubeWgpu32, 2>>,
) -> Vec<MatmulToken> {
    weights
        .into_iter()
        .map(|w| {
            let pending = lhs.clone().matmul(w);
            MatmulToken::from_fn(move || tensor2_to_array_f32(pending))
        })
        .collect()
}

fn submit_registered_many_f16(
    lhs: Tensor<CubeWgpu16, 2>,
    weights: Vec<Tensor<CubeWgpu16, 2>>,
) -> Vec<MatmulToken> {
    weights
        .into_iter()
        .map(|w| {
            let pending = lhs.clone().matmul(w);
            MatmulToken::from_fn(move || tensor2_to_array_f16(pending))
        })
        .collect()
}

// ─── GpuOffloadEngine impl ─────────────────────────────────────────────

impl GpuOffloadEngine for WgpuVulkanEngine {
    fn register_weight(&mut self, handle: WeightHandle, weight: ArrayView2<'_, f32>) -> Result<()> {
        let mut guard = self.weights.lock().unwrap();
        match &mut *guard {
            WeightStore::F32(map) => {
                let t = array2_to_tensor_f32(weight, &self.device);
                map.insert(handle, t);
            }
            WeightStore::F16(map) => {
                let t = array2_to_tensor_f16(weight, &self.device);
                map.insert(handle, t);
            }
        }
        Ok(())
    }

    fn register_weight_bf16(
        &mut self,
        handle: WeightHandle,
        weight: ArrayView2<'_, bf16>,
    ) -> Result<()> {
        let mut guard = self.weights.lock().unwrap();
        match &mut *guard {
            WeightStore::F32(map) => {
                let t = array2_bf16_to_tensor_f32(weight, &self.device);
                map.insert(handle, t);
            }
            WeightStore::F16(map) => {
                let t = array2_bf16_to_tensor_f16(weight, &self.device);
                map.insert(handle, t);
            }
        }
        Ok(())
    }

    fn matmul(&self, handle: WeightHandle, input: ArrayView2<'_, f32>) -> Result<Array2<f32>> {
        let k = input.ncols();
        let guard = self.weights.lock().unwrap();
        match &*guard {
            WeightStore::F32(map) => {
                let weight = registered_weight_f32(map, handle, k, "matmul")?;
                drop(guard);
                let lhs = array2_to_tensor_f32(input, &self.device);
                tensor2_to_array_f32(lhs.matmul(weight))
            }
            WeightStore::F16(map) => {
                let weight = registered_weight_f16(map, handle, k, "matmul")?;
                drop(guard);
                let lhs = array2_to_tensor_f16(input, &self.device);
                tensor2_to_array_f16(lhs.matmul(weight))
            }
        }
    }

    fn matmul_many(
        &self,
        handles: &[WeightHandle],
        input: ArrayView2<'_, f32>,
    ) -> Result<Vec<Array2<f32>>> {
        if handles.is_empty() {
            return Ok(Vec::new());
        }
        let k = input.ncols();
        let guard = self.weights.lock().unwrap();
        match &*guard {
            WeightStore::F32(map) => {
                let weights = registered_weights_f32(map, handles, k, "matmul_many")?;
                drop(guard);
                let lhs = array2_to_tensor_f32(input, &self.device);
                execute_registered_many_f32(lhs, weights)
            }
            WeightStore::F16(map) => {
                let weights = registered_weights_f16(map, handles, k, "matmul_many")?;
                drop(guard);
                let lhs = array2_to_tensor_f16(input, &self.device);
                execute_registered_many_f16(lhs, weights)
            }
        }
    }

    /// **R4 async** override. Splits the existing sync `matmul` into
    /// (upload + kernel issue) and (download), returning a
    /// [`MatmulToken`] whose closure captures the pending burn-tensor.
    ///
    /// The kernel issue (`lhs.matmul(weight)`) is non-blocking on
    /// burn-cubecl — the actual GPU sync happens inside the closure
    /// when the substrate drains the token via `into_array`. This
    /// frees the substrate's calling thread to run shield/cascade
    /// work for the next offload site while the GPU is busy.
    ///
    /// Plan: `docs/plans/m1-12-r4-async-overlap.md` §B.
    fn matmul_async(
        &self,
        handle: WeightHandle,
        input: ArrayView2<'_, f32>,
    ) -> Result<MatmulToken> {
        let k = input.ncols();
        let guard = self.weights.lock().unwrap();
        match &*guard {
            WeightStore::F32(map) => {
                let weight = registered_weight_f32(map, handle, k, "matmul_async")?;
                drop(guard);
                let lhs = array2_to_tensor_f32(input, &self.device);
                let pending = lhs.matmul(weight);
                Ok(MatmulToken::from_fn(move || tensor2_to_array_f32(pending)))
            }
            WeightStore::F16(map) => {
                let weight = registered_weight_f16(map, handle, k, "matmul_async")?;
                drop(guard);
                let lhs = array2_to_tensor_f16(input, &self.device);
                let pending = lhs.matmul(weight);
                Ok(MatmulToken::from_fn(move || tensor2_to_array_f16(pending)))
            }
        }
    }

    /// **R4 async** override for `matmul_many`. Shares one upload of
    /// `input` across all N kernel launches (same algebra as the sync
    /// `matmul_many` but each output is captured into its own token
    /// rather than batched via [`Transaction`]). The first token
    /// drained triggers a device sync that completes *all* N kernels;
    /// subsequent token drains just read pre-completed buffers, so
    /// the bus savings of the sync path are preserved.
    fn matmul_many_async(
        &self,
        handles: &[WeightHandle],
        input: ArrayView2<'_, f32>,
    ) -> Result<Vec<MatmulToken>> {
        if handles.is_empty() {
            return Ok(Vec::new());
        }
        let k = input.ncols();
        let guard = self.weights.lock().unwrap();
        match &*guard {
            WeightStore::F32(map) => {
                let weights = registered_weights_f32(map, handles, k, "matmul_many_async")?;
                drop(guard);
                let lhs = array2_to_tensor_f32(input, &self.device);
                Ok(submit_registered_many_f32(lhs, weights))
            }
            WeightStore::F16(map) => {
                let weights = registered_weights_f16(map, handles, k, "matmul_many_async")?;
                drop(guard);
                let lhs = array2_to_tensor_f16(input, &self.device);
                Ok(submit_registered_many_f16(lhs, weights))
            }
        }
    }

    /// Path β bf16-input override. The bf16 → device-precision
    /// conversion runs once during the upload Vec build via the
    /// existing `array2_bf16_to_tensor_*` helpers — no transient
    /// host f32 buffer is materialised. Compared to the default
    /// trait impl (`bf16 → f32 → f16 upload`), this saves one
    /// full-tensor DRAM pass at the substrate boundary.
    fn matmul_bf16_input(
        &self,
        handle: WeightHandle,
        input: ArrayView2<'_, bf16>,
    ) -> Result<Array2<f32>> {
        let k = input.ncols();
        let guard = self.weights.lock().unwrap();
        match &*guard {
            WeightStore::F32(map) => {
                let weight = registered_weight_f32(map, handle, k, "matmul_bf16_input")?;
                drop(guard);
                let lhs = array2_bf16_to_tensor_f32(input, &self.device);
                tensor2_to_array_f32(lhs.matmul(weight))
            }
            WeightStore::F16(map) => {
                let weight = registered_weight_f16(map, handle, k, "matmul_bf16_input")?;
                drop(guard);
                let lhs = array2_bf16_to_tensor_f16(input, &self.device);
                tensor2_to_array_f16(lhs.matmul(weight))
            }
        }
    }

    /// Path β bf16-input variant of [`Self::matmul_many`]. Same
    /// fused-dispatch structure as the f32 path — one upload of the
    /// bf16 input (no f32 intermediate), shared across all N kernel
    /// launches via the Transaction-batched download.
    fn matmul_many_bf16_input(
        &self,
        handles: &[WeightHandle],
        input: ArrayView2<'_, bf16>,
    ) -> Result<Vec<Array2<f32>>> {
        if handles.is_empty() {
            return Ok(Vec::new());
        }
        let k = input.ncols();
        let guard = self.weights.lock().unwrap();
        match &*guard {
            WeightStore::F32(map) => {
                let weights = registered_weights_f32(map, handles, k, "matmul_many_bf16_input")?;
                drop(guard);
                let lhs = array2_bf16_to_tensor_f32(input, &self.device);
                execute_registered_many_f32(lhs, weights)
            }
            WeightStore::F16(map) => {
                let weights = registered_weights_f16(map, handles, k, "matmul_many_bf16_input")?;
                drop(guard);
                let lhs = array2_bf16_to_tensor_f16(input, &self.device);
                execute_registered_many_f16(lhs, weights)
            }
        }
    }

    fn matmul_dynamic(
        &self,
        lhs: ArrayView2<'_, f32>,
        rhs: ArrayView2<'_, f32>,
    ) -> Result<Array2<f32>> {
        if lhs.ncols() != rhs.nrows() {
            return Err(anyhow!(
                "matmul_dynamic shape mismatch: lhs cols {} != rhs rows {}",
                lhs.ncols(),
                rhs.nrows()
            ));
        }
        if self.fp16 {
            let lhs_t = array2_to_tensor_f16(lhs, &self.device);
            let rhs_t = array2_to_tensor_f16(rhs, &self.device);
            tensor2_to_array_f16(lhs_t.matmul(rhs_t))
        } else {
            let lhs_t = array2_to_tensor_f32(lhs, &self.device);
            let rhs_t = array2_to_tensor_f32(rhs, &self.device);
            tensor2_to_array_f32(lhs_t.matmul(rhs_t))
        }
    }

    fn matmul_dynamic_batched(
        &self,
        lhs: ArrayView3<'_, f32>,
        rhs: ArrayView3<'_, f32>,
    ) -> Result<Array3<f32>> {
        let b = lhs.shape()[0];
        let lhs_k = lhs.shape()[2];
        let rhs_k = rhs.shape()[1];
        if rhs.shape()[0] != b || rhs_k != lhs_k {
            return Err(anyhow!(
                "matmul_dynamic_batched shape mismatch: lhs {:?} vs rhs {:?}",
                lhs.shape(),
                rhs.shape()
            ));
        }
        if self.fp16 {
            let lhs_t = array3_to_tensor_f16(lhs, &self.device);
            let rhs_t = array3_to_tensor_f16(rhs, &self.device);
            tensor3_to_array_f16(lhs_t.matmul(rhs_t))
        } else {
            let lhs_t = array3_to_tensor_f32(lhs, &self.device);
            let rhs_t = array3_to_tensor_f32(rhs, &self.device);
            tensor3_to_array_f32(lhs_t.matmul(rhs_t))
        }
    }

    fn softmax_batched(&self, input: ArrayView3<'_, f32>) -> Result<Array3<f32>> {
        // Last-axis softmax via burn_tensor::activation::softmax. Runs on
        // the wgpu device — used by permutation-shielded attention so the
        // softmax doesn't bounce back to the TEE between Q·Kᵀ and ·V.
        if self.fp16 {
            let t = array3_to_tensor_f16(input, &self.device);
            tensor3_to_array_f16(activation::softmax(t, 2))
        } else {
            let t = array3_to_tensor_f32(input, &self.device);
            tensor3_to_array_f32(activation::softmax(t, 2))
        }
    }

    /// **A1 (Phase 1b enabler)** — single-dispatch-chain fused
    /// attention.  Uploads Q, K, V, mask **once**; runs
    /// `Q·Kᵀ → scale → +mask → softmax → ·V` entirely on-device via
    /// chained `burn::Tensor` ops; downloads the output **once**.
    ///
    /// The five sub-ops still execute as separate burn kernels (no
    /// FlashAttention-style single-pass fusion — that would need a
    /// hand-rolled CubeCL kernel against the `O(B·n_q·n_kv)` scores
    /// intermediate).  But all intermediates live in GPU device memory
    /// — no GPU↔CPU round-trips between sub-ops, no K^T staging on
    /// the host.  This is the load-bearing change against the trait's
    /// default impl, which materialises `K^T` and `scores` host-side
    /// between dispatches.
    ///
    /// At decode m=1 the dominant residual cost is per-kernel launch
    /// latency on Vulkan (~0.2-0.5 ms per dispatch on Strix Halo).
    /// Removing those further requires either (a) a hand-rolled
    /// FlashAttention-style kernel that runs the chain in one
    /// dispatch, or (b) burn-cubecl's operator-fusion pass picking up
    /// the chain — under investigation.
    fn fused_attention_batched(
        &self,
        q: ArrayView3<'_, f32>,
        k: ArrayView3<'_, f32>,
        v: ArrayView3<'_, f32>,
        scale: f32,
        mask: Option<ArrayView3<'_, f32>>,
    ) -> Result<Array3<f32>> {
        let (b, n_q, d_head) = (q.shape()[0], q.shape()[1], q.shape()[2]);
        let n_kv = k.shape()[1];
        debug_assert_eq!(q.shape()[0], b);
        debug_assert_eq!(k.shape(), &[b, n_kv, d_head]);
        debug_assert_eq!(v.shape(), &[b, n_kv, d_head]);
        if let Some(m) = mask {
            debug_assert_eq!(m.shape(), &[b, n_q, n_kv]);
        }

        if self.fp16 {
            let q_t = array3_to_tensor_f16(q, &self.device);
            let k_t = array3_to_tensor_f16(k, &self.device);
            let v_t = array3_to_tensor_f16(v, &self.device);
            // Device-side K^T via permute (no host transpose).
            let kt = k_t.permute([0, 2, 1]);
            let scores = q_t.matmul(kt);
            // A2: mask is None at decode (no-op) — skip the upload +
            // add-kernel-dispatch entirely.
            let scores = match mask {
                Some(m) => {
                    let mask_t = array3_to_tensor_f16(m, &self.device);
                    scores.mul_scalar(scale).add(mask_t)
                }
                None => scores.mul_scalar(scale),
            };
            let probs = activation::softmax(scores, 2);
            let out = probs.matmul(v_t);
            tensor3_to_array_f16(out)
        } else {
            let q_t = array3_to_tensor_f32(q, &self.device);
            let k_t = array3_to_tensor_f32(k, &self.device);
            let v_t = array3_to_tensor_f32(v, &self.device);
            let kt = k_t.permute([0, 2, 1]);
            let scores = q_t.matmul(kt);
            let scores = match mask {
                Some(m) => {
                    let mask_t = array3_to_tensor_f32(m, &self.device);
                    scores.mul_scalar(scale).add(mask_t)
                }
                None => scores.mul_scalar(scale),
            };
            let probs = activation::softmax(scores, 2);
            let out = probs.matmul(v_t);
            tensor3_to_array_f32(out)
        }
    }
}

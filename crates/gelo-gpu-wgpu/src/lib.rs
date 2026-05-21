//! [`GpuOffloadEngine`] backed by **burn-cubecl** on the **wgpu/Vulkan** runtime.
//!
//! Replaces the prior cubecl-matmul direct path with `burn_tensor::Tensor::matmul`
//! over the `CubeBackend<WgpuRuntime, …>` backend. burn-cubecl wires:
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
use cubecl_common::future;
use cubecl_wgpu::{AutoGraphicsApi, RuntimeOptions, WgpuDevice, WgpuRuntime, init_setup_async};
use half::{bf16, f16};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};

use gelo_protocol::{GpuOffloadEngine, WeightHandle};

/// burn-cubecl backend specialised to f32 floats. The default engine
/// precision.
type CubeWgpu32 = CubeBackend<WgpuRuntime, f32, i32, u8>;

/// burn-cubecl backend specialised to f16 floats. Used by the fp16
/// engine path. Requires the wgpu adapter to support the `shader-f16`
/// extension (true on AMD RDNA2/3, NVIDIA Maxwell+, most modern Intel
/// iGPUs).
type CubeWgpu16 = CubeBackend<WgpuRuntime, f16, i32, u8>;

/// Per-process GPU adapter info, captured once at first device init.
struct GpuContext {
    adapter_info: wgpu::AdapterInfo,
}

static GPU_CTX: OnceLock<GpuContext> = OnceLock::new();

fn gpu_ctx() -> &'static GpuContext {
    GPU_CTX.get_or_init(|| {
        let device = WgpuDevice::default();
        let setup = future::block_on(init_setup_async::<AutoGraphicsApi>(
            &device,
            RuntimeOptions::default(),
        ));
        GpuContext {
            adapter_info: setup.adapter.get_info(),
        }
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
    device: WgpuDevice,
    weights: Arc<Mutex<WeightStore>>,
    fp16: bool,
}

impl WgpuVulkanEngine {
    /// Initialise a Vulkan-preferred wgpu device via burn-cubecl, with
    /// f32 internal precision.
    pub fn new() -> Result<Self> {
        let _ = gpu_ctx();
        let device = WgpuDevice::default();
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
        let device = WgpuDevice::default();
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

impl WgpuVulkanEngine {

    /// `true` if this engine handle runs GEMM kernels in f16. Trusted-
    /// side code that needs bit-equal matmul output (e.g. U-Verify) must
    /// gate on this.
    pub fn is_fp16(&self) -> bool {
        self.fp16
    }

    /// Backend name reported by the selected wgpu adapter (e.g. `"Vulkan"`).
    pub fn backend(&self) -> String {
        format!("{:?}", gpu_ctx().adapter_info.backend)
    }

    /// Full adapter information.
    pub fn adapter_info(&self) -> &'static wgpu::AdapterInfo {
        &gpu_ctx().adapter_info
    }

    /// `true` if the selected adapter is a real (discrete, integrated, or
    /// virtual) GPU — not a software rasterizer like lavapipe.
    pub fn is_real_gpu(&self) -> bool {
        matches!(
            gpu_ctx().adapter_info.device_type,
            wgpu::DeviceType::DiscreteGpu
                | wgpu::DeviceType::IntegratedGpu
                | wgpu::DeviceType::VirtualGpu
        )
    }
}

// ─── f32 conversion helpers ───────────────────────────────────────────

fn array2_to_tensor_f32(view: ArrayView2<'_, f32>, device: &WgpuDevice) -> Tensor<CubeWgpu32, 2> {
    let rows = view.nrows();
    let cols = view.ncols();
    let v: Vec<f32> = view.as_standard_layout().iter().copied().collect();
    Tensor::<CubeWgpu32, 2>::from_data(TensorData::new(v, [rows, cols]), device)
}

fn array3_to_tensor_f32(view: ArrayView3<'_, f32>, device: &WgpuDevice) -> Tensor<CubeWgpu32, 3> {
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

fn array2_to_tensor_f16(view: ArrayView2<'_, f32>, device: &WgpuDevice) -> Tensor<CubeWgpu16, 2> {
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
    device: &WgpuDevice,
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
    device: &WgpuDevice,
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

fn array3_to_tensor_f16(view: ArrayView3<'_, f32>, device: &WgpuDevice) -> Tensor<CubeWgpu16, 3> {
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

    fn matmul(
        &self,
        handle: WeightHandle,
        input: ArrayView2<'_, f32>,
    ) -> Result<Array2<f32>> {
        let k = input.ncols();
        let guard = self.weights.lock().unwrap();
        match &*guard {
            WeightStore::F32(map) => {
                let weight = map
                    .get(&handle)
                    .ok_or_else(|| anyhow!("weight {handle:?} not registered"))?
                    .clone();
                if k != weight.dims()[0] {
                    return Err(anyhow!(
                        "matmul shape mismatch: input cols {k} != weight rows {}",
                        weight.dims()[0]
                    ));
                }
                drop(guard);
                let lhs = array2_to_tensor_f32(input, &self.device);
                tensor2_to_array_f32(lhs.matmul(weight))
            }
            WeightStore::F16(map) => {
                let weight = map
                    .get(&handle)
                    .ok_or_else(|| anyhow!("weight {handle:?} not registered"))?
                    .clone();
                if k != weight.dims()[0] {
                    return Err(anyhow!(
                        "matmul shape mismatch: input cols {k} != weight rows {}",
                        weight.dims()[0]
                    ));
                }
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
                let weights: Vec<Tensor<CubeWgpu32, 2>> = handles
                    .iter()
                    .map(|h| {
                        let w = map
                            .get(h)
                            .ok_or_else(|| anyhow!("weight {h:?} not registered"))?
                            .clone();
                        if w.dims()[0] != k {
                            return Err(anyhow!(
                                "matmul_many shape mismatch on {h:?}: input cols {k} != weight rows {}",
                                w.dims()[0]
                            ));
                        }
                        Ok(w)
                    })
                    .collect::<Result<_>>()?;
                drop(guard);
                let lhs = array2_to_tensor_f32(input, &self.device);
                let mut out_dims: Vec<(usize, usize)> = Vec::with_capacity(handles.len());
                let mut tx = Transaction::<CubeWgpu32>::default();
                for w in weights {
                    let out = lhs.clone().matmul(w);
                    let d = out.dims();
                    out_dims.push((d[0], d[1]));
                    tx = tx.register(out);
                }
                let datas: Vec<TensorData> = tx.execute();
                datas
                    .into_iter()
                    .zip(out_dims)
                    .map(|(data, (rows, cols))| {
                        let v: Vec<f32> = data
                            .into_vec()
                            .map_err(|e| anyhow!("burn f32 TensorData → Vec<f32>: {e:?}"))?;
                        Array2::from_shape_vec((rows, cols), v)
                            .map_err(|e| anyhow!("Array2 from tensor data: {e}"))
                    })
                    .collect()
            }
            WeightStore::F16(map) => {
                let weights: Vec<Tensor<CubeWgpu16, 2>> = handles
                    .iter()
                    .map(|h| {
                        let w = map
                            .get(h)
                            .ok_or_else(|| anyhow!("weight {h:?} not registered"))?
                            .clone();
                        if w.dims()[0] != k {
                            return Err(anyhow!(
                                "matmul_many shape mismatch on {h:?}: input cols {k} != weight rows {}",
                                w.dims()[0]
                            ));
                        }
                        Ok(w)
                    })
                    .collect::<Result<_>>()?;
                drop(guard);
                let lhs = array2_to_tensor_f16(input, &self.device);
                let mut out_dims: Vec<(usize, usize)> = Vec::with_capacity(handles.len());
                let mut tx = Transaction::<CubeWgpu16>::default();
                for w in weights {
                    let out = lhs.clone().matmul(w);
                    let d = out.dims();
                    out_dims.push((d[0], d[1]));
                    tx = tx.register(out);
                }
                let datas: Vec<TensorData> = tx.execute();
                datas
                    .into_iter()
                    .zip(out_dims)
                    .map(|(data, (rows, cols))| {
                        let v_f16: Vec<f16> = data
                            .into_vec()
                            .map_err(|e| anyhow!("burn f16 TensorData → Vec<f16>: {e:?}"))?;
                        let v: Vec<f32> = v_f16.into_iter().map(|x| x.to_f32()).collect();
                        Array2::from_shape_vec((rows, cols), v)
                            .map_err(|e| anyhow!("Array2 from tensor data: {e}"))
                    })
                    .collect()
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
}

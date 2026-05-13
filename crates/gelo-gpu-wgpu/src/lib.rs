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

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Result, anyhow};
use burn_backend::Backend;
use burn_cubecl::CubeBackend;
use burn_tensor::{Tensor, TensorData, Transaction};
use cubecl_common::future;
use cubecl_wgpu::{AutoGraphicsApi, RuntimeOptions, WgpuDevice, WgpuRuntime, init_setup_async};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};

use gelo_protocol::{GpuOffloadEngine, WeightHandle};

/// burn-cubecl backend specialized to f32 floats, i32 ints, u8 bools.
type CubeWgpu = CubeBackend<WgpuRuntime, f32, i32, u8>;

/// Per-process GPU adapter info, captured once at first device init.
/// burn / cubecl-runtime cache the actual client per `WgpuDevice`, so the
/// only thing we need to remember at the process level is the adapter
/// metadata for `is_real_gpu()` / `backend()` queries.
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

/// burn-cubecl/wgpu-backed offload engine.
///
/// Registered weights are held as `Tensor<CubeWgpu, 2>` so they stay
/// device-resident across matmul calls. `clone_shared` produces a second
/// handle pointing at the same weight cache.
pub struct WgpuVulkanEngine {
    device: WgpuDevice,
    weights: Arc<Mutex<HashMap<WeightHandle, Tensor<CubeWgpu, 2>>>>,
}

impl WgpuVulkanEngine {
    /// Initialise a Vulkan-preferred wgpu device via burn-cubecl. First
    /// call registers the device's compute server; subsequent calls reuse
    /// it through the shared `cubecl-runtime` client cache.
    pub fn new() -> Result<Self> {
        let _ = gpu_ctx();
        let device = WgpuDevice::default();
        // Sync once at init so the device is fully constructed before any
        // matmul fires. Cheap.
        <CubeWgpu as Backend>::sync(&device)
            .map_err(|e| anyhow!("burn-cubecl device sync at init: {e:?}"))?;
        Ok(Self {
            device,
            weights: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Second handle sharing the registered-weight cache with `self`.
    pub fn clone_shared(&self) -> Self {
        Self {
            device: self.device.clone(),
            weights: Arc::clone(&self.weights),
        }
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

fn array2_to_tensor(view: ArrayView2<'_, f32>, device: &WgpuDevice) -> Tensor<CubeWgpu, 2> {
    let rows = view.nrows();
    let cols = view.ncols();
    let v: Vec<f32> = view.as_standard_layout().iter().copied().collect();
    Tensor::<CubeWgpu, 2>::from_data(TensorData::new(v, [rows, cols]), device)
}

fn array3_to_tensor(view: ArrayView3<'_, f32>, device: &WgpuDevice) -> Tensor<CubeWgpu, 3> {
    let b = view.shape()[0];
    let m = view.shape()[1];
    let k = view.shape()[2];
    let v: Vec<f32> = view.as_standard_layout().iter().copied().collect();
    Tensor::<CubeWgpu, 3>::from_data(TensorData::new(v, [b, m, k]), device)
}

fn tensor2_to_array(t: Tensor<CubeWgpu, 2>) -> Result<Array2<f32>> {
    let shape = t.dims();
    let data = t.into_data();
    let v: Vec<f32> = data
        .into_vec()
        .map_err(|e| anyhow!("burn tensor → Vec<f32>: {e:?}"))?;
    Array2::from_shape_vec((shape[0], shape[1]), v)
        .map_err(|e| anyhow!("Array2 from tensor data: {e}"))
}

fn tensor3_to_array(t: Tensor<CubeWgpu, 3>) -> Result<Array3<f32>> {
    let shape = t.dims();
    let data = t.into_data();
    let v: Vec<f32> = data
        .into_vec()
        .map_err(|e| anyhow!("burn tensor → Vec<f32>: {e:?}"))?;
    Array3::from_shape_vec((shape[0], shape[1], shape[2]), v)
        .map_err(|e| anyhow!("Array3 from tensor data: {e}"))
}

impl GpuOffloadEngine for WgpuVulkanEngine {
    fn register_weight(&mut self, handle: WeightHandle, weight: ArrayView2<'_, f32>) -> Result<()> {
        let t = array2_to_tensor(weight, &self.device);
        self.weights.lock().unwrap().insert(handle, t);
        Ok(())
    }

    fn matmul(
        &self,
        handle: WeightHandle,
        input: ArrayView2<'_, f32>,
    ) -> Result<Array2<f32>> {
        let m = input.nrows();
        let k = input.ncols();
        let weight = {
            let weights = self.weights.lock().unwrap();
            weights
                .get(&handle)
                .ok_or_else(|| anyhow!("weight {handle:?} not registered"))?
                .clone()
        };
        let w_dims = weight.dims();
        if k != w_dims[0] {
            return Err(anyhow!(
                "matmul shape mismatch: input cols {k} != weight rows {}",
                w_dims[0]
            ));
        }
        let _ = m; // shape captured for the error path above; tensor carries its own shape
        let lhs = array2_to_tensor(input, &self.device);
        let out = lhs.matmul(weight);
        tensor2_to_array(out)
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
        // Resolve weights up front + shape-check.
        let weights = {
            let map = self.weights.lock().unwrap();
            handles
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
                .collect::<Result<Vec<_>>>()?
        };

        // ONE upload of the masked input.
        let lhs = array2_to_tensor(input, &self.device);

        // Issue all N matmuls lazily. Capture out-dims for the eventual
        // ndarray rebuild; we lose access to dims() after registering with
        // the transaction (which consumes the tensors).
        let mut out_dims: Vec<(usize, usize)> = Vec::with_capacity(handles.len());
        let mut transaction = Transaction::<CubeWgpu>::default();
        for w in weights {
            let out = lhs.clone().matmul(w);
            let dims = out.dims();
            out_dims.push((dims[0], dims[1]));
            transaction = transaction.register(out);
        }

        // Single batched read: burn flushes the stream and downloads ALL
        // registered tensors in one transaction. Documented at
        // `burn_tensor::Transaction` as the canonical pattern for "reading
        // multiple tensors at once" — better than N separate `into_data()`
        // calls, which each force a device sync.
        let datas: Vec<TensorData> = transaction.execute();
        anyhow::ensure!(
            datas.len() == out_dims.len(),
            "Transaction returned {} datas; expected {}",
            datas.len(),
            out_dims.len()
        );

        datas
            .into_iter()
            .zip(out_dims.into_iter())
            .map(|(data, (rows, cols))| {
                let v: Vec<f32> = data
                    .into_vec()
                    .map_err(|e| anyhow!("burn TensorData → Vec<f32>: {e:?}"))?;
                Array2::from_shape_vec((rows, cols), v)
                    .map_err(|e| anyhow!("Array2 from tensor data: {e}"))
            })
            .collect()
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
        let lhs_t = array2_to_tensor(lhs, &self.device);
        let rhs_t = array2_to_tensor(rhs, &self.device);
        let out = lhs_t.matmul(rhs_t);
        tensor2_to_array(out)
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
        let lhs_t = array3_to_tensor(lhs, &self.device);
        let rhs_t = array3_to_tensor(rhs, &self.device);
        let out = lhs_t.matmul(rhs_t);
        tensor3_to_array(out)
    }
}

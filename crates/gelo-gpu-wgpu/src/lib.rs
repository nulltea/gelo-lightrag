//! [`GpuOffloadEngine`] backed by **CubeCL** on the **wgpu/Vulkan** runtime.
//!
//! Replaces the M2 naive WGSL kernel with `cubecl-matmul`'s autotuned SGEMM
//! (workgroup-tiled + register-tiled + cooperative-matrix-where-available),
//! while keeping the same `GpuOffloadEngine` trait surface so the GELO
//! protocol layer is unchanged.
//!
//! On Linux this dispatches via Vulkan; on macOS via Metal; on Windows via
//! DX12. The adapter info is exposed so callers can assert real-GPU
//! hardware vs. a software ICD (e.g. lavapipe).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use anyhow::{Result, anyhow};
use cubecl_common::future;
use cubecl_core::Runtime;
use cubecl_core::client::ComputeClient;
use cubecl_core::prelude::CubePrimitive;
use cubecl_core::prelude::TensorHandleRef;
use cubecl_core::server::Handle;
use cubecl_matmul as matmul;
use cubecl_matmul::MatmulInputHandleRef;
use cubecl_matmul::Strategy;
use cubecl_matmul::components::MatmulElems;
use cubecl_wgpu::{AutoGraphicsApi, RuntimeOptions, WgpuDevice, WgpuRuntime, init_setup_async};
use ndarray::{Array2, ArrayView2};

use gelo_protocol::{GpuOffloadEngine, WeightHandle};

const ELEM_SIZE: usize = std::mem::size_of::<f32>();

struct WeightBuffer {
    handle: Handle,
    /// `(in_features, out_features)`
    rows: usize,
    cols: usize,
}

/// Per-process GPU context. `cubecl-runtime` keeps a singleton server per
/// device id, so we register it exactly once and share both the
/// [`ComputeClient`] and the captured adapter info across every
/// [`WgpuVulkanEngine`] instance.
struct GpuContext {
    client: ComputeClient<WgpuRuntime>,
    adapter_info: wgpu::AdapterInfo,
}

static GPU_CTX: OnceLock<GpuContext> = OnceLock::new();

fn gpu_ctx() -> &'static GpuContext {
    GPU_CTX.get_or_init(|| {
        let device = WgpuDevice::default();
        // `init_setup_async` both builds the setup *and* registers the
        // server for `device`. Subsequent `WgpuRuntime::client(&device)`
        // calls just resolve a fresh client handle pointing at it.
        let setup = future::block_on(init_setup_async::<AutoGraphicsApi>(
            &device,
            RuntimeOptions::default(),
        ));
        let adapter_info = setup.adapter.get_info();
        let client = WgpuRuntime::client(&device);
        GpuContext { client, adapter_info }
    })
}

/// CubeCL/wgpu-backed offload engine. Internally uses `cubecl-matmul`'s
/// autotuned SGEMM (no hand-written kernel in this crate).
pub struct WgpuVulkanEngine {
    weights: Mutex<HashMap<WeightHandle, WeightBuffer>>,
}

impl WgpuVulkanEngine {
    /// Initialise a Vulkan-preferred wgpu device, build the CubeCL client.
    ///
    /// The first call to this constructor performs the wgpu setup and
    /// registers the device's server. Subsequent calls (e.g. one per
    /// benchmark configuration) reuse the same shared client. Returns
    /// `Err` if no Vulkan-capable adapter is available.
    pub fn new() -> Result<Self> {
        let _ = gpu_ctx();
        Ok(Self {
            weights: Mutex::new(HashMap::new()),
        })
    }

    fn client(&self) -> &'static ComputeClient<WgpuRuntime> {
        &gpu_ctx().client
    }

    /// Backend name reported by the selected wgpu adapter (e.g. `"Vulkan"`).
    pub fn backend(&self) -> String {
        format!("{:?}", gpu_ctx().adapter_info.backend)
    }

    /// Full adapter information — vendor, device name, driver, device type.
    pub fn adapter_info(&self) -> &'static wgpu::AdapterInfo {
        &gpu_ctx().adapter_info
    }

    /// `true` if the selected adapter is a discrete, integrated, or virtual
    /// GPU (i.e. not a software rasterizer such as `llvmpipe`/`lavapipe`).
    pub fn is_real_gpu(&self) -> bool {
        matches!(
            gpu_ctx().adapter_info.device_type,
            wgpu::DeviceType::DiscreteGpu
                | wgpu::DeviceType::IntegratedGpu
                | wgpu::DeviceType::VirtualGpu
        )
    }
}

fn row_major_strides(_rows: usize, cols: usize) -> [usize; 2] {
    [cols, 1]
}

impl GpuOffloadEngine for WgpuVulkanEngine {
    fn register_weight(&mut self, handle: WeightHandle, weight: ArrayView2<'_, f32>) -> Result<()> {
        let rows = weight.nrows();
        let cols = weight.ncols();
        let bytes: Vec<f32> = weight
            .as_standard_layout()
            .iter()
            .copied()
            .collect();
        let raw: &[u8] = bytemuck::cast_slice(&bytes);
        let gpu_handle = self.client().create_from_slice(raw);
        self.weights
            .lock()
            .unwrap()
            .insert(handle, WeightBuffer { handle: gpu_handle, rows, cols });
        Ok(())
    }

    fn matmul(
        &self,
        handle: WeightHandle,
        input: ArrayView2<'_, f32>,
    ) -> Result<Array2<f32>> {
        let weights = self.weights.lock().unwrap();
        let w = weights
            .get(&handle)
            .ok_or_else(|| anyhow!("weight {handle:?} not registered"))?;

        let m = input.nrows();
        let k = input.ncols();
        let n = w.cols;
        if k != w.rows {
            return Err(anyhow!(
                "matmul shape mismatch: input cols {k} != weight rows {}",
                w.rows
            ));
        }

        // Upload masked input.
        let input_data: Vec<f32> = input.as_standard_layout().iter().copied().collect();
        let input_bytes: &[u8] = bytemuck::cast_slice(&input_data);
        let client = self.client();
        let lhs_handle = client.create_from_slice(input_bytes);
        let out_handle = client.empty(m * n * ELEM_SIZE);

        let lhs_shape = [m, k];
        let lhs_strides = row_major_strides(m, k);
        let rhs_shape = [w.rows, w.cols];
        let rhs_strides = row_major_strides(w.rows, w.cols);
        let out_shape = [m, n];
        let out_strides = row_major_strides(m, n);

        let dtype = f32::as_type_native_unchecked();

        // SAFETY: handles outlive their refs for the duration of launch_ref.
        let lhs_ref = unsafe {
            TensorHandleRef::<WgpuRuntime>::from_raw_parts(
                &lhs_handle,
                &lhs_strides,
                &lhs_shape,
                ELEM_SIZE,
            )
        };
        let rhs_ref = unsafe {
            TensorHandleRef::<WgpuRuntime>::from_raw_parts(
                &w.handle,
                &rhs_strides,
                &rhs_shape,
                ELEM_SIZE,
            )
        };
        let out_ref = unsafe {
            TensorHandleRef::<WgpuRuntime>::from_raw_parts(
                &out_handle,
                &out_strides,
                &out_shape,
                ELEM_SIZE,
            )
        };

        let lhs_input = MatmulInputHandleRef::Normal(lhs_ref, dtype);
        let rhs_input = MatmulInputHandleRef::Normal(rhs_ref, dtype);

        let mut dtypes = MatmulElems::new::<f32>();

        matmul::launch_ref::<WgpuRuntime>(
            &Strategy::Auto,
            client,
            &lhs_input,
            &rhs_input,
            &out_ref,
            &mut dtypes,
        )
        .map_err(|e| anyhow!("cubecl matmul launch failed: {e:?}"))?;

        let bytes = client.read_one(out_handle);
        let floats: &[f32] = bytemuck::cast_slice(&bytes);
        let out = Array2::from_shape_vec((m, n), floats.to_vec())
            .map_err(|e| anyhow!("shape build failed: {e}"))?;
        Ok(out)
    }
}


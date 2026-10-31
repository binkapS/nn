//! Minimal wgpu host-side scaffolding (Stage 1.1 of PLAN.md).
//!
//! Responsibilities at this stage:
//! * adapter discovery with a deterministic preference order (Vulkan/Metal/
//!   DX12 drivers via `wgpu`), reporting what it picks to stderr,
//! * a blocking `block_on` for the async wgpu future API — written on std
//!   primitives only (thread + one-shot channel + noop waker), because the
//!   crate has no async runtime dependency.
//! * a trivial device round-trip so the scaffold is proven before kernels
//!   land (Stage 2).
//!
//! The numeric `Backend` trait and WGSL kernels arrive in Stage 2; until
//! then `select_backend` never returns GPU for use by training (the caller
//! treats "GPU" as "probe only").
#![allow(dead_code)] // device/roundtrip land with Stage 2 kernels

use std::sync::mpsc;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

/// Which engine runs the math. CPU is and stays the reference; GPU selects
/// the wgpu path once kernels are ported.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Engine {
    Cpu,
    Gpu,
}

/// Selects the engine from the `--backend` flag.
///
/// * `cpu` — always CPU.
/// * `gpu` — GPU required: fails the run when no usable adapter exists
///   (no silent fallback when the user explicitly asked).
/// * `auto` — GPU when an adapter is available **and** the GPU path is
///   implemented; otherwise CPU with an informational note.
///
/// Returns `(engine, adapter_report)`.
pub fn select_backend(flag: &str, gpu_implemented: bool) -> Result<(Engine, String), String> {
    match flag {
        "cpu" => Ok((Engine::Cpu, "cpu engine (--backend cpu)".to_string())),
        "gpu" | "auto" => {
            let report = probe_adapters();
            match report {
                Ok(info) if gpu_implemented || flag == "gpu" => {
                    if gpu_implemented {
                        Ok((Engine::Gpu, info))
                    } else {
                        // Probe succeeded but kernels not ported yet.
                        Ok((
                            Engine::Cpu,
                            format!("{info} — GPU kernels not ported yet; using CPU (Stage 2)"),
                        ))
                    }
                }
                Ok(info) => Ok((
                    Engine::Cpu,
                    format!("{info} — using CPU engine (--backend auto)"),
                )),
                Err(e) => {
                    if flag == "gpu" {
                        Err(format!("--backend gpu requested but no usable adapter: {e}"))
                    } else {
                        Ok((Engine::Cpu, format!("no usable GPU ({e}) — using CPU engine")))
                    }
                }
            }
        }
        other => Err(format!(
            "unknown --backend `{other}` (expected cpu|gpu|auto)"
        )),
    }
}

/// Enumerates drivers via wgpu and returns a human-readable report with the
/// chosen adapter. Errors when no adapter fits.
pub fn probe_adapters() -> Result<String, String> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|e| format!("wgpu adapter request failed: {e}"))?;
    let info = adapter.get_info();
    let limits = adapter.limits();
    Ok(format!(
        "GPU adapter: {} ({}, Vulkan/Metal/DX12 driver: {}), max buffer {} MiB",
        info.name,
        info.backend,
        info.driver,
        limits.max_buffer_size / (1024 * 1024),
    ))
}

/// A device handle for the GPU engine (populated in Stage 2; the round-trip
/// test exercises it end-to-end).
pub struct Device {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

/// Requests the default adapter and returns a device+queue pair.
pub fn open_device() -> Result<Device, String> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|e| format!("wgpu adapter request failed: {e}"))?;
    let (device, queue) = block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("nn-gpu"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::default(),
        },
    ))
    .map_err(|e| format!("wgpu device request failed: {e}"))?;
    Ok(Device { device, queue })
}

/// Submits a copy/execute round trip: uploads `data`, runs a passthrough
/// shader, reads it back. Proves device, buffers, shader module, encoders
/// and staging readback all work before any real kernel is written.
pub fn device_roundtrip(dev: &Device, data: &[f32]) -> Result<Vec<f32>, String> {
    // 1:1 mapping identity shader: b_out[i] = b_in[i].
    let shader = dev.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("passthrough"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
            r#"
@group(0) @binding(0) var<storage, read> b_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> b_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&b_in) { return; }
    b_out[i] = b_in[i];
}
"#,
        )),
    });

    let n = data.len();
    let bytes: &[u8] = bytemuck_bytes(data);
    let in_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rt-in"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rt-out"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rt-read"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    dev.queue.write_buffer(&in_buf, 0, bytes);

    let pipeline = dev
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("passthrough-pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rt-bind"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = dev
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rt-enc") });
    let groups = n.div_ceil(64) as u32;
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rt-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, bytes.len() as u64);
    dev.queue.submit(Some(encoder.finish()));

    let slice = read_buf.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res.map_err(|e| e.to_string()));
    });
    let _ = dev.device.poll(wgpu::PollType::Wait);
    rx.recv()
        .map_err(|e| format!("gpu readback channel failed: {e}"))?
        .map_err(|e| format!("gpu map_async failed: {e}"))?;
    let out: Vec<f32> = slice
        .get_mapped_range()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    read_buf.unmap();
    Ok(out)
}

use std::borrow::Cow;

/// `bytemuck`-free little-endian byte view of f32 data (stdin of the GPU).
fn bytemuck_bytes(data: &[f32]) -> &[u8] {
    // SAFETY: &[f32] is plain old data; a flat byte reinterpretation of the
    // same slice is always valid (alignment of u8 < f32).
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) }
}

// ---------------------------------------------------------------------------
// Stage 2.1: tiled GEMM kernels — forward + the two backward GEMMs
// ---------------------------------------------------------------------------

/// Forward kernel: `y (M,N) = x (M,K) @ wt (K,N) + bias(N)`.
///
/// Each workgroup (16×16 threads) computes a 16×16 tile of `y` by marching
/// over `k` in chunks of 16: a chunk of `x` rows and `wt` rows is staged in
/// workgroup memory, all threads compute against it, barriers delimit every
/// chunk (no early return: the `k` bound is uniform so the loop exit is
/// uniform too). Out-of-range loads are clamped in-bounds and their
/// contribution is zeroed via `select` — never a raw OOB access.
const GEMM_FWD_WGSL: &str = r#"
struct Dims {
    rows: u32,    // m
    inner: u32,   // k
    cols: u32,    // n
    use_bias: u32,
}

@group(0) @binding(0) var<storage, read> x_buf: array<f32>;
@group(0) @binding(1) var<storage, read> w_buf: array<f32>;
@group(0) @binding(2) var<storage, read> bias_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> y_buf: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

var<workgroup> tile_x: array<f32, 256>; // 16 x 16 chunk of x
var<workgroup> tile_w: array<f32, 256>; // 16 x 16 chunk of wt

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let m = dims.rows;
    let k = dims.inner;
    let n = dims.cols;
    let row = gid.y;                 // m index of this thread's output cell
    let col = gid.x;                 // n index of this thread's output cell
    let base_row = row - lid.y;      // workgroup tile origin
    let base_col = col - lid.x;
    let ty = lid.y;
    let tx = lid.x;

    var acc = 0.0;
    var chunk: u32 = 0u;
    loop {
        let k0 = chunk * 16u;
        if (k0 >= k) {
            break; // uniform across the workgroup, so the barrier set is uniform
        }
        // Stage one 16x16 chunk of x (rows base_row..) and wt (rows k0..).
        // Clamped, then zeroed via select: the OOB lanes contribute 0.
        let xr = min(base_row + ty, m - 1u);
        let xc = min(k0 + tx, k - 1u);
        tile_x[ty * 16u + tx] = select(0.0, x_buf[xr * k + xc],
            (base_row + ty) < m && (k0 + tx) < k);
        let wr = min(k0 + ty, k - 1u);
        let wc = min(base_col + tx, n - 1u);
        tile_w[ty * 16u + tx] = select(0.0, w_buf[wr * n + wc],
            (k0 + ty) < k && (base_col + tx) < n);
        workgroupBarrier();

        for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
            acc = acc + tile_x[ty * 16u + kk] * tile_w[kk * 16u + tx];
        }
        workgroupBarrier();
        chunk = chunk + 1u;
    }

    if (row < m && col < n) {
        var out = acc;
        if (dims.use_bias != 0u) {
            out = out + bias_buf[col];
        }
        y_buf[row * n + col] = out;
    }
}
"#;

/// Weight-gradient kernel: `dwt (K,N) = Σ_m x[m,K] * gy[m,N]`.
///
/// Each workgroup computes a 16×16 tile of `dwt` (thread (k-row, n-col));
/// every thread accumulates its scalar over `m` in chunks of 16, staging one
/// 16-lane column of `x` and `gy` in workgroup memory per chunk. Each output
/// cell is written exactly once (accumulated locally first), so the output
/// buffer needs no pre-zeroing.
const GEMM_DWT_WGSL: &str = r#"
struct Dims {
    rows: u32,    // m
    inner: u32,   // k
    cols: u32,    // n
    pad: u32,
}

@group(0) @binding(0) var<storage, read> x_buf: array<f32>;
@group(0) @binding(1) var<storage, read> gy_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> dwt_buf: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let m = dims.rows;
    let k = dims.inner;
    let n = dims.cols;
    let krow = gid.y;                // k index of this thread's dwt cell
    let ncol = gid.x;                // n index of this thread's dwt cell

    var acc = 0.0;
    var chunk: u32 = 0u;
    loop {
        let m0 = chunk * 16u;
        if (m0 >= m) {
            break;
        }
        // No workgroup staging here: the tile's two varying coordinates are
        // already taken by (krow, ncol), so a third (the m lane) cannot be
        // assigned to a thread dimension without colliding writes. Each
        // thread accumulates its scalar straight from global memory; the
        // redundant loads (whole lane-rows load the same x values) hit L1.
        for (var i: u32 = 0u; i < 16u; i = i + 1u) {
            let mr = min(m0 + i, m - 1u);
            let kc = min(krow, k - 1u);
            let xv = select(0.0, x_buf[mr * k + kc], (m0 + i) < m && krow < k);
            let nc = min(ncol, n - 1u);
            let gv = select(0.0, gy_buf[mr * n + nc], (m0 + i) < m && ncol < n);
            acc = acc + xv * gv;
        }
        chunk = chunk + 1u;
    }

    if (krow < k && ncol < n) {
        dwt_buf[krow * n + ncol] = acc;
    }
}
"#;

/// Cached GEMM engine: pipelines (and their compiled WGSL) are built once at
/// construction; per-call work is buffer creation + dispatch. Residency
/// (weights/activations persisting between steps) is Stage 2.7.
pub struct GpuGemm {
    device: wgpu::Device,
    queue: wgpu::Queue,
    fwd_pipeline: wgpu::ComputePipeline,
    dwt_pipeline: wgpu::ComputePipeline,
}

impl GpuGemm {
    pub fn new() -> Result<Self, String> {
        let Device { device, queue } = open_device()?;
        let fwd_pipeline = pipeline(&device, "gemm-fwd", GEMM_FWD_WGSL);
        let dwt_pipeline = pipeline(&device, "gemm-dwt", GEMM_DWT_WGSL);
        Ok(Self {
            device,
            queue,
            fwd_pipeline,
            dwt_pipeline,
        })
    }

    /// `y (M,N) = x (M,K) @ wt (K,N) + bias(N)` — same contract as
    /// `ops::matmul_fwd`.
    pub fn matmul_fwd(
        &self,
        x: &[f32],
        wt: &[f32],
        bias: Option<&[f32]>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        assert_eq!(x.len(), m * k, "x shape");
        assert_eq!(wt.len(), k * n, "wt shape");
        let (bias_data, use_bias) = match bias {
            Some(b) => {
                assert_eq!(b.len(), n, "bias shape");
                (b.to_vec(), 1u32)
            }
            None => (vec![0.0f32; 1], 0u32),
        };
        let dims = [m as u32, k as u32, n as u32, use_bias];

        let x_buf = upload_f32(&self.device, &self.queue, x);
        let wt_buf = upload_f32(&self.device, &self.queue, wt);
        let bias_buf = upload_f32(&self.device, &self.queue, &bias_data);
        let y_buf = self.storage_out(m * n);
        let dims_buf = upload_uniform_u32(&self.device, &self.queue, &dims);

        Self::dispatch(
            &self.device,
            &self.queue,
            &self.fwd_pipeline,
            [ceil_div(n, 16) as u32, ceil_div(m, 16) as u32],
            &[
                bind(0, &x_buf),
                bind(1, &wt_buf),
                bind(2, &bias_buf),
                bind(3, &y_buf),
                bind(4, &dims_buf),
            ],
        );
        readback_f32(&self.device, &self.queue, &y_buf, m * n)
    }

    /// `(dx (M,K), dwt (K,N))` through `y = x @ wt`: same contract as
    /// `ops::matmul_bwd`. `dx = gy @ wtᵀ` reuses the forward kernel against a
    /// host-transposed weight (mirrors the CPU Stage-0 design); `dwt` runs
    /// the dedicated scatter kernel.
    pub fn matmul_bwd(
        &self,
        x: &[f32],
        wt: &[f32],
        gy: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(x.len(), m * k, "x shape");
        assert_eq!(wt.len(), k * n, "wt shape");
        assert_eq!(gy.len(), m * n, "gy shape");

        // dx = gy (M,N) @ wtᵀ (N,K): materialize the transposed weight once.
        let mut wtt = vec![0.0f32; n * k];
        for r in 0..k {
            let wr = &wt[r * n..(r + 1) * n];
            for (j, &wv) in wr.iter().enumerate() {
                wtt[j * k + r] = wv;
            }
        }
        let dx = self.matmul_fwd(gy, &wtt, None, m, n, k);

        // dwt = xᵀ @ gy via the dedicated kernel.
        let dims = [m as u32, k as u32, n as u32, 0u32];
        let x_buf = upload_f32(&self.device, &self.queue, x);
        let gy_buf = upload_f32(&self.device, &self.queue, gy);
        let dwt_buf = self.storage_out(k * n);
        let dims_buf = upload_uniform_u32(&self.device, &self.queue, &dims);

        Self::dispatch(
            &self.device,
            &self.queue,
            &self.dwt_pipeline,
            [ceil_div(n, 16) as u32, ceil_div(k, 16) as u32],
            &[
                bind(0, &x_buf),
                bind(1, &gy_buf),
                bind(2, &dwt_buf),
                bind(3, &dims_buf),
            ],
        );
        let dwt = readback_f32(&self.device, &self.queue, &dwt_buf, k * n);
        (dx, dwt)
    }

    fn storage_out(&self, len_f32: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gemm-out"),
            size: (len_f32 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// One compute pass: bind group + dispatch. Readback is a separate step.
    fn dispatch(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &wgpu::ComputePipeline,
        groups: [u32; 2],
        entries: &[wgpu::BindGroupEntry<'_>],
    ) {
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gemm-bind"),
            layout: &layout,
            entries,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gemm-enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemm-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups[0], groups[1], 1);
        }
        queue.submit(Some(encoder.finish()));
    }
}

fn pipeline(device: &wgpu::Device, label: &str, wgsl: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl)),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None, // derive from the module's bindings
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// GL-style bind entry for an entire buffer.
fn bind(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

/// Uploads f32 data into a read-only storage buffer.
fn upload_f32(device: &wgpu::Device, queue: &wgpu::Queue, data: &[f32]) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gemm-in"),
        size: (data.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, bytemuck_bytes(data));
    buf
}

/// Uploads u32 data into a uniform buffer (16-byte aligned blocks).
fn upload_uniform_u32(device: &wgpu::Device, queue: &wgpu::Queue, data: &[u32]) -> wgpu::Buffer {
    let mut bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    // Uniform buffer sizes must be a multiple of 16; pad with zeros.
    while !bytes.len().is_multiple_of(16) {
        bytes.push(0);
    }
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gemm-dims"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, &bytes);
    buf
}

/// Copies a buffer to a MAP_READ staging buffer, waits, and returns the f32
/// contents. Requires `src` to carry `COPY_SRC`.
fn readback_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    len_f32: usize,
) -> Vec<f32> {
    let size = (len_f32 * 4) as u64;
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gemm-read"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gemm-read-enc"),
    });
    encoder.copy_buffer_to_buffer(src, 0, &read, 0, size);
    queue.submit(Some(encoder.finish()));
    let slice = read.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res.map_err(|e| e.to_string()));
    });
    device
        .poll(wgpu::PollType::Wait)
        .map_err(|e| format!("gpu poll failed: {e}"))
        .unwrap();
    rx.recv()
        .map_err(|e| format!("gpu readback channel failed: {e}"))
        .unwrap()
        .map_err(|e| format!("gpu map_async failed: {e}"))
        .unwrap();
    let out: Vec<f32> = slice
        .get_mapped_range()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    read.unmap();
    out
}

/// Standard-library `block_on`: drives a future on the *calling* thread with
/// a noop waker, polling until ready. Used only at startup (adapter/device
/// request); the compute pipeline itself never blocks on host futures —
/// readback goes through explicit `device.poll(wgpu::PollType::Wait)`.
pub fn block_on<F: Future>(fut: F) -> F::Output {
    let mut fut = Box::pin(fut);
    let raw = RawWaker::new(std::ptr::null::<()>(), &NOOP_VTABLE);
    let waker = unsafe { Waker::from_raw(raw) };
    let mut cx = Context::from_waker(&waker);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => std::hint::spin_loop(),
        }
    }
}

const NOOP_VTABLE: RawWakerVTable = RawWakerVTable::new(
    |p| RawWaker::new(p, &NOOP_VTABLE), // clone
    |_| {},                             // wake
    |_| {},                             // wake_by_ref
    |_| {},                             // drop
);

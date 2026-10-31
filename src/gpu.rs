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
        let y_buf = storage_out(&self.device, m * n);
        let dims_buf = upload_uniform_u32(&self.device, &self.queue, &dims);

        dispatch(
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
        let dwt_buf = storage_out(&self.device, k * n);
        let dims_buf = upload_uniform_u32(&self.device, &self.queue, &dims);

        dispatch(
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
}

// ---------------------------------------------------------------------------
// Stage 2.2: LayerNorm + GeLU kernels (elementwise, written fusable)
// ---------------------------------------------------------------------------

/// Forward LayerNorm, one row per workgroup (256 threads):
/// `y = (x - mean) * rstd * g + b` with population variance, matching the
/// CPU reference exactly. Pass 1 stages each lane's partial Σx/Σx² in
/// workgroup memory and tree-reduces; pass 2 re-reads the row and
/// normalizes. The `var = max(Σx²/n − mean², 0)` clamp guards against a
/// negative variance from f32 cancellation (the CPU form Σ(x−m)² can't go
/// negative; mathematically identical).
const LN_FWD_WGSL: &str = r#"
struct LnDims {
    rows: u32,
    cols: u32,   // c
    eps: f32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> x_buf: array<f32>;
@group(0) @binding(1) var<storage, read> g_buf: array<f32>;
@group(0) @binding(2) var<storage, read> b_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> y_buf: array<f32>;
@group(0) @binding(4) var<uniform> dims: LnDims;

const WG: u32 = 256u;
var<workgroup> red0: array<f32, 256>;
var<workgroup> red1: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    let c = dims.cols;
    let lane = lid.x;
    let base = row * c;

    var s = 0.0;
    var sq = 0.0;
    for (var cc = lane; cc < c; cc = cc + WG) {
        let v = x_buf[base + cc];
        s = s + v;
        sq = sq + v * v;
    }
    red0[lane] = s;
    red1[lane] = sq;
    workgroupBarrier();

    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (lane < stride) {
            red0[lane] = red0[lane] + red0[lane + stride];
            red1[lane] = red1[lane] + red1[lane + stride];
        }
        workgroupBarrier();
    }
    let mean = red0[0] / f32(c);
    let var_raw = red1[0] / f32(c) - mean * mean;
    let variance = select(var_raw, 0.0, var_raw < 0.0);
    let rstd = 1.0 / sqrt(variance + dims.eps);
    workgroupBarrier();

    for (var cc = lane; cc < c; cc = cc + WG) {
        y_buf[base + cc] = (x_buf[base + cc] - mean) * rstd * g_buf[cc] + b_buf[cc];
    }
}
"#;

/// Backward LayerNorm, one row per workgroup. Recomputes the row stats
/// (forward math), folds `m1 = Σ gy·g` and `m2 = Σ gy·g·x̂` into per-row
/// scalars via the shared tree, writes the whole `dx` row, and leaves
/// per-row `dg`/`db` contributions in a (rows × c) partial buffer that a
/// follow-up column-reduce kernel sums (f32 atomics are not in WebGPU).
const LN_BWD_WGSL: &str = r#"
struct LnDims {
    rows: u32,
    cols: u32,   // c
    eps: f32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> x_buf: array<f32>;
@group(0) @binding(1) var<storage, read> gy_buf: array<f32>;
@group(0) @binding(2) var<storage, read> g_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx_buf: array<f32>;
@group(0) @binding(4) var<storage, read_write> dgp_buf: array<f32>;
@group(0) @binding(5) var<storage, read_write> dbp_buf: array<f32>;
@group(0) @binding(6) var<uniform> dims: LnDims;

const WG: u32 = 256u;
var<workgroup> red0: array<f32, 256>;
var<workgroup> red1: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    let c = dims.cols;
    let lane = lid.x;
    let base = row * c;

    var s = 0.0;
    var sq = 0.0;
    for (var cc = lane; cc < c; cc = cc + WG) {
        let v = x_buf[base + cc];
        s = s + v;
        sq = sq + v * v;
    }
    red0[lane] = s;
    red1[lane] = sq;
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (lane < stride) {
            red0[lane] = red0[lane] + red0[lane + stride];
            red1[lane] = red1[lane] + red1[lane + stride];
        }
        workgroupBarrier();
    }
    let mean = red0[0] / f32(c);
    let var_raw = red1[0] / f32(c) - mean * mean;
    let variance = select(var_raw, 0.0, var_raw < 0.0);
    let rstd = 1.0 / sqrt(variance + dims.eps);
    workgroupBarrier();

    // m1 = Σ gy·g, m2 = Σ gy·g·x̂ ; per-row dg/db partials per column.
    var m1 = 0.0;
    var m2 = 0.0;
    for (var cc = lane; cc < c; cc = cc + WG) {
        let gv = gy_buf[base + cc];
        let xhat = (x_buf[base + cc] - mean) * rstd;
        let gyg = gv * g_buf[cc];
        m1 = m1 + gyg;
        m2 = m2 + gyg * xhat;
        dgp_buf[base + cc] = gv * xhat;
        dbp_buf[base + cc] = gv;
    }
    red0[lane] = m1;
    red1[lane] = m2;
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (lane < stride) {
            red0[lane] = red0[lane] + red0[lane + stride];
            red1[lane] = red1[lane] + red1[lane + stride];
        }
        workgroupBarrier();
    }
    let m1t = red0[0];
    let m2t = red1[0];
    workgroupBarrier();

    // dx = rstd·gy·g − (rstd/c)·(m1 + x̂·m2)
    for (var cc = lane; cc < c; cc = cc + WG) {
        let xhat = (x_buf[base + cc] - mean) * rstd;
        dx_buf[base + cc] = rstd * gy_buf[base + cc] * g_buf[cc]
            - rstd / f32(c) * (m1t + xhat * m2t);
    }
}
"#;

/// Column reduction over a (w × c) partial buffer: `out[col] = Σ_r p[r·c + col]`.
/// One thread per column, no barriers needed; used to finish LN dg/db.
const REDUCE_COLS_WGSL: &str = r#"
struct RdDims {
    w: u32,    // rows of partials (= rows of LN input)
    c: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> p_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> o_buf: array<f32>;
@group(0) @binding(2) var<uniform> dims: RdDims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let c = dims.c;
    if (col >= c) {
        return;
    }
    var acc = 0.0;
    for (var r = 0u; r < dims.w; r = r + 1u) {
        acc = acc + p_buf[r * c + col];
    }
    o_buf[col] = acc;
}
"#;

/// GeLU forward (tanh approximation, same constants as `ops::gelu_fwd`):
/// `y = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`.
const GELU_FWD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> y_buf: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&x_buf)) {
        return;
    }
    let v = x_buf[i];
    let u = 0.7978846 * (v + 0.044715 * v * v * v);
    y_buf[i] = 0.5 * v * (1.0 + tanh(u));
}
"#;

/// GeLU backward given the saved forward input x: elementwise.
const GELU_BWD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x_buf: array<f32>;
@group(0) @binding(1) var<storage, read> dy_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx_buf: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&x_buf)) {
        return;
    }
    let v = x_buf[i];
    let u = 0.7978846 * (v + 0.044715 * v * v * v);
    let t = tanh(u);
    let sech2 = 1.0 - t * t;
    let dudx = 0.7978846 * (1.0 + 3.0 * 0.044715 * v * v);
    dx_buf[i] = dy_buf[i] * (0.5 * (1.0 + t) + 0.5 * v * sech2 * dudx);
}
"#;

/// AdamW elementwise update, identical arithmetic to `AdamState::step`:
/// `m = b1·m + (1−b1)·g; v = b2·v + (1−b2)·g²; θ −= lr·(m̂/(√v̂+ε) + wd·θ)`.
/// The bias corrections `bc1/bc2` are computed on the host from the step
/// counter (`powf` is not in WGSL core).
const ADAMW_WGSL: &str = r#"
struct AdamDims {
    n: u32,
    lr: f32,
    b1: f32,
    b2: f32,
    bc1: f32,
    bc2: f32,
    eps: f32,
    wd: f32,
}

@group(0) @binding(0) var<storage, read> g_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> p_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> m_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> v_buf: array<f32>;
@group(0) @binding(4) var<uniform> dims: AdamDims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.n) {
        return;
    }
    let g = g_buf[i];
    m_buf[i] = dims.b1 * m_buf[i] + (1.0 - dims.b1) * g;
    v_buf[i] = dims.b2 * v_buf[i] + (1.0 - dims.b2) * g * g;
    let m_hat = m_buf[i] / dims.bc1;
    let v_hat = v_buf[i] / dims.bc2;
    p_buf[i] = p_buf[i] - dims.lr * (m_hat / (sqrt(v_hat) + dims.eps) + dims.wd * p_buf[i]);
}
"#;

/// Sum-of-squares of a gradient tensor, block-reduced: each 256-thread
/// workgroup owns a contiguous chunk, lanes stride-reduce, one shared-tree
/// barrier finishes the block, and `o_buf[wid]` holds one f32 partial.
/// (The final f64 fold over the 32–64 partials happens host-side, mirroring
/// the CPU's f64 gradient-norm accumulation.)
const SQNORM_WGSL: &str = r#"
struct SqDims {
    n: u32,
    chunk: u32,   // elements per workgroup
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> g_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> o_buf: array<f32>;
@group(0) @binding(2) var<uniform> dims: SqDims;

const WG: u32 = 256u;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let start = wid.x * dims.chunk;
    let end = min(start + dims.chunk, dims.n);

    var acc = 0.0;
    for (var i = start + lid.x; i < end; i = i + WG) {
        let g = g_buf[i];
        acc = acc + g * g;
    }
    red[lid.x] = acc;
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (lid.x < stride) {
            red[lid.x] = red[lid.x] + red[lid.x + stride];
        }
        workgroupBarrier();
    }
    o_buf[wid.x] = red[0];
}
"#;

/// Elementwise gradient scale ("multiply all grads by s") used by grad
/// clipping; a no-op skip for `s == 1.0` is a host-side decision.
const GRAD_SCALE_WGSL: &str = r#"
struct ScaleDims {
    s: f32,
    n: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> g_buf: array<f32>;
@group(0) @binding(1) var<uniform> dims: ScaleDims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.n) {
        return;
    }
    g_buf[i] = g_buf[i] * dims.s;
}
"#;

/// Softmax cross-entropy forward, one row per workgroup: computes the
/// numerically stable row softmax, writes `probs`, and emits the per-row
/// loss `-ln(max p[target], EPS)` into a per-row buffer (the host folds
/// the mean in f64, mirroring the CPU accumulation).
const CE_FWD_WGSL: &str = r#"
struct CeDims {
    rows: u32,
    vocab: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> logits_buf: array<f32>;
@group(0) @binding(1) var<storage, read> tgt_buf: array<u32>;
@group(0) @binding(2) var<storage, read_write> probs_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> loss_buf: array<f32>;
@group(0) @binding(4) var<uniform> dims: CeDims;

const WG: u32 = 256u;
var<workgroup> red0: array<f32, 256>;
var<workgroup> red1: array<f32, 256>;

const LN_EPS_P: f32 = 1.1920929e-07; // f32::EPSILON, matches CPU clamp

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;
    let v = dims.vocab;
    let lane = lid.x;
    let base = row * v;
    let tgt = tgt_buf[row];

    // pass 1: row max
    var mx = -3.4e38; // f32 approx of -inf for reduction purposes
    for (var cc = lane; cc < v; cc = cc + WG) {
        mx = max(mx, logits_buf[base + cc]);
    }
    red0[lane] = mx;
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (lane < stride) {
            red0[lane] = max(red0[lane], red0[lane + stride]);
        }
        workgroupBarrier();
    }
    let rm = red0[0];
    workgroupBarrier();

    // pass 2: Σ exp(x − max)
    var z = 0.0;
    for (var cc = lane; cc < v; cc = cc + WG) {
        z = z + exp(logits_buf[base + cc] - rm);
    }
    red0[lane] = z;
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (lane < stride) {
            red0[lane] = red0[lane] + red0[lane + stride];
        }
        workgroupBarrier();
    }
    let rz = red0[0];
    workgroupBarrier();

    // pass 3: probs
    for (var cc = lane; cc < v; cc = cc + WG) {
        probs_buf[base + cc] = exp(logits_buf[base + cc] - rm) / rz;
    }
    workgroupBarrier();

    // pass 4: per-row loss from the target logit (one lane); clamp p at
    // f32::EPSILON exactly like the CPU (`p.max(f32::EPSILON).ln()`).
    if (lane == 0u) {
        let pv = probs_buf[base + tgt];
        let pt = select(1.0, pv, tgt < v);
        loss_buf[row] = -log(max(pt, LN_EPS_P));
    }
}
"#;

/// Softmax-CE backward, elementwise over (rows × vocab):
/// `dlogits = (p − onehot(target)) / rows`.
const CE_BWD_WGSL: &str = r#"
struct CeDims {
    rows: u32,
    vocab: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> p_buf: array<f32>;
@group(0) @binding(1) var<storage, read> tgt_buf: array<u32>;
@group(0) @binding(2) var<storage, read_write> d_buf: array<f32>;
@group(0) @binding(3) var<uniform> dims: CeDims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = dims.rows * dims.vocab;
    if (i >= n) {
        return;
    }
    let row = i / dims.vocab;
    let col = i % dims.vocab;
    let is_tgt = col == tgt_buf[row];
    let d = p_buf[i] - select(0.0, 1.0, is_tgt);
    d_buf[i] = d / f32(dims.rows);
}
"#;

// ---------------------------------------------------------------------------
// Stage 2.6: embedding gather/scatter kernels.
// ---------------------------------------------------------------------------

/// Embedding forward: gather `emb_fwd(ids) -> out` with
/// `out[r·dim + j] = table[ids[r]·dim + j]`, thread-per-element. Ids come
/// in as a u32 storage buffer.
const EMB_FWD_WGSL: &str = r#"
struct EmbDims {
    n: u32,    // n = rows*dim
    dim: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> table_buf: array<f32>;
@group(0) @binding(1) var<storage, read> ids_buf: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<f32>;
@group(0) @binding(3) var<uniform> dims: EmbDims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.n) {
        return;
    }
    let dim = dims.dim;
    let r = i / dim;
    let j = i - r * dim;
    out_buf[i] = table_buf[ids_buf[r] * dim + j];
}
"#;

/// Embedding backward, thread per (id, column): the whole masked sum
/// `dtable[id,j] = old + Σ_r (ids[r]==id)·dx[r,j]` — one thread serially
/// scans the rows in r-order, so accumulation order matches the CPU's
/// scatter-add exactly (no atomics involved). O(vocab·dim·rows) reads of
/// dx but pass 1 of the loop is a dollar-and-nothing op once the update
/// runs per token column; vocab at model scale is small (char-level).
const EMB_BWD_WGSL: &str = r#"
struct EmbDims {
    n: u32,     // n = vocab*dim (table length)
    rows: u32,
    dim: u32,
    pad0: u32,
}

@group(0) @binding(0) var<storage, read> dx_buf: array<f32>;
@group(0) @binding(1) var<storage, read> ids_buf: array<u32>;
@group(0) @binding(2) var<storage, read_write> dt_buf: array<f32>;
@group(0) @binding(3) var<uniform> dims: EmbDims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.n) {
        return;
    }
    let dim = dims.dim;
    let id = i / dim;
    let j = i - id * dim;
    // in-place `+=` per row in r-order — same association as the CPU
    // scatter-add (including the preexisting table value), bit-for-bit.
    for (var r = 0u; r < dims.rows; r = r + 1u) {
        if (ids_buf[r] == id) {
            dt_buf[i] = dt_buf[i] + dx_buf[r * dim + j];
        }
    }
}
"#;

/// Elementwise + reduction GPU ops (LayerNorm fwd/bwd, GeLU fwd/bwd).
/// Mirrors the CPU contracts in `ops.rs` element for element.
pub struct GpuOps {
    device: wgpu::Device,
    queue: wgpu::Queue,
    ln_fwd: wgpu::ComputePipeline,
    ln_bwd: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    gelu_fwd: wgpu::ComputePipeline,
    gelu_bwd: wgpu::ComputePipeline,
    adamw: wgpu::ComputePipeline,
    sqnorm: wgpu::ComputePipeline,
    grad_scale: wgpu::ComputePipeline,
    ce_fwd: wgpu::ComputePipeline,
    ce_bwd: wgpu::ComputePipeline,
    emb_fwd: wgpu::ComputePipeline,
    emb_bwd: wgpu::ComputePipeline,
}

impl GpuOps {
    pub fn new() -> Result<Self, String> {
        let Device { device, queue } = open_device()?;
        let ln_fwd = pipeline(&device, "ln-fwd", LN_FWD_WGSL);
        let ln_bwd = pipeline(&device, "ln-bwd", LN_BWD_WGSL);
        let reduce = pipeline(&device, "reduce-cols", REDUCE_COLS_WGSL);
        let gelu_fwd = pipeline(&device, "gelu-fwd", GELU_FWD_WGSL);
        let gelu_bwd = pipeline(&device, "gelu-bwd", GELU_BWD_WGSL);
        let adamw = pipeline(&device, "adamw", ADAMW_WGSL);
        let sqnorm = pipeline(&device, "sqnorm", SQNORM_WGSL);
        let grad_scale = pipeline(&device, "grad-scale", GRAD_SCALE_WGSL);
        let ce_fwd = pipeline(&device, "ce-fwd", CE_FWD_WGSL);
        let ce_bwd = pipeline(&device, "ce-bwd", CE_BWD_WGSL);
        let emb_fwd = pipeline(&device, "emb-fwd", EMB_FWD_WGSL);
        let emb_bwd = pipeline(&device, "emb-bwd", EMB_BWD_WGSL);
        Ok(Self {
            device,
            queue,
            ln_fwd,
            ln_bwd,
            reduce,
            gelu_fwd,
            gelu_bwd,
            adamw,
            sqnorm,
            grad_scale,
            ce_fwd,
            ce_bwd,
            emb_fwd,
            emb_bwd,
        })
    }

    /// `ops::layernorm_fwd` on the GPU.
    pub fn layernorm_fwd(
        &self,
        x: &[f32],
        g: &[f32],
        b: &[f32],
        rows: usize,
        c: usize,
        eps: f32,
    ) -> Vec<f32> {
        assert_eq!(x.len(), rows * c, "x shape");
        assert_eq!(g.len(), c, "g shape");
        assert_eq!(b.len(), c, "b shape");
        if rows == 0 {
            return Vec::new();
        }
        let dims = [rows as u32, c as u32, eps.to_bits(), 0u32];
        let x_buf = upload_f32(&self.device, &self.queue, x);
        let g_buf = upload_f32(&self.device, &self.queue, g);
        let b_buf = upload_f32(&self.device, &self.queue, b);
        let y_buf = storage_out(&self.device, rows * c);
        let dims_buf = upload_uniform_u32(&self.device, &self.queue, &dims);
        dispatch(
            &self.device,
            &self.queue,
            &self.ln_fwd,
            [rows as u32, 1],
            &[
                bind(0, &x_buf),
                bind(1, &g_buf),
                bind(2, &b_buf),
                bind(3, &y_buf),
                bind(4, &dims_buf),
            ],
        );
        readback_f32(&self.device, &self.queue, &y_buf, rows * c)
    }

    /// `ops::layernorm_bwd` on the GPU: `(dx (rows,c), dg (c), db (c))`.
    pub fn layernorm_bwd(
        &self,
        x: &[f32],
        gy: &[f32],
        g: &[f32],
        rows: usize,
        c: usize,
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        assert_eq!(x.len(), rows * c, "x shape");
        assert_eq!(gy.len(), rows * c, "gy shape");
        assert_eq!(g.len(), c, "g shape");
        if rows == 0 {
            return (Vec::new(), vec![0.0; c], vec![0.0; c]);
        }
        let dims = [rows as u32, c as u32, eps.to_bits(), 0u32];
        let x_buf = upload_f32(&self.device, &self.queue, x);
        let gy_buf = upload_f32(&self.device, &self.queue, gy);
        let g_buf = upload_f32(&self.device, &self.queue, g);
        let dx_buf = storage_out(&self.device, rows * c);
        let dgp_buf = storage_out(&self.device, rows * c);
        let dbp_buf = storage_out(&self.device, rows * c);
        let dims_buf = upload_uniform_u32(&self.device, &self.queue, &dims);
        dispatch(
            &self.device,
            &self.queue,
            &self.ln_bwd,
            [rows as u32, 1],
            &[
                bind(0, &x_buf),
                bind(1, &gy_buf),
                bind(2, &g_buf),
                bind(3, &dx_buf),
                bind(4, &dgp_buf),
                bind(5, &dbp_buf),
                bind(6, &dims_buf),
            ],
        );

        // dg/db: column-reduce both partials (rows × c → c).
        let rd_dims = [rows as u32, c as u32, 0u32, 0u32];
        let dg_buf = storage_out(&self.device, c);
        let db_buf = storage_out(&self.device, c);
        let rd_buf = upload_uniform_u32(&self.device, &self.queue, &rd_dims);
        for (p, o) in [(&dgp_buf, &dg_buf), (&dbp_buf, &db_buf)] {
            dispatch(
                &self.device,
                &self.queue,
                &self.reduce,
                [ceil_div(c, 256) as u32, 1],
                &[bind(0, p), bind(1, o), bind(2, &rd_buf)],
            );
        }
        let dx = readback_f32(&self.device, &self.queue, &dx_buf, rows * c);
        let dg = readback_f32(&self.device, &self.queue, &dg_buf, c);
        let db = readback_f32(&self.device, &self.queue, &db_buf, c);
        (dx, dg, db)
    }

    /// `ops::gelu_fwd` tanh approximation; returns `y` (the CPU version's
    /// returned saved input stays on the host side here).
    pub fn gelu_fwd(&self, x: &[f32]) -> Vec<f32> {
        let x_buf = upload_f32(&self.device, &self.queue, x);
        let y_buf = storage_out(&self.device, x.len());
        dispatch(
            &self.device,
            &self.queue,
            &self.gelu_fwd,
            [ceil_div(x.len(), 256) as u32, 1],
            &[bind(0, &x_buf), bind(1, &y_buf)],
        );
        readback_f32(&self.device, &self.queue, &y_buf, x.len())
    }

    /// `ops::gelu_bwd` given the saved forward input.
    pub fn gelu_bwd(&self, x: &[f32], dy: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), dy.len(), "dy shape");
        let x_buf = upload_f32(&self.device, &self.queue, x);
        let dy_buf = upload_f32(&self.device, &self.queue, dy);
        let dx_buf = storage_out(&self.device, x.len());
        dispatch(
            &self.device,
            &self.queue,
            &self.gelu_bwd,
            [ceil_div(x.len(), 256) as u32, 1],
            &[bind(0, &x_buf), bind(1, &dy_buf), bind(2, &dx_buf)],
        );
        readback_f32(&self.device, &self.queue, &dx_buf, x.len())
    }

    /// One AdamW step over a whole tensor: `AdamState::step` semantics.
    /// Bias corrections from `step` are computed host-side; `m`/`v` are the
    /// persistent moments (returned updated), `wd` is pre-gated by the
    /// caller for decay-excluded tensors (biases, LN scales).
    #[allow(clippy::too_many_arguments)]
    pub fn adamw(
        &self,
        params: &mut [f32],
        grads: &[f32],
        m: &mut [f32],
        v: &mut [f32],
        step: u32,
        lr: f32,
        b1: f32,
        b2: f32,
        eps: f32,
        wd: f32,
    ) {
        let n = params.len();
        assert_eq!(grads.len(), n);
        assert_eq!(m.len(), n);
        assert_eq!(v.len(), n);
        let t = step as f32;
        let bc1 = 1.0 - b1.powf(t);
        let bc2 = 1.0 - b2.powf(t);
        let dims: [u32; 8] = [
            n as u32,
            lr.to_bits(),
            b1.to_bits(),
            b2.to_bits(),
            bc1.to_bits(),
            bc2.to_bits(),
            eps.to_bits(),
            wd.to_bits(),
        ];
        let g_buf = upload_f32(&self.device, &self.queue, grads);
        let p_buf = buffer_f32_duplex(&self.device, n);
        let m_buf = buffer_f32_duplex(&self.device, n);
        let v_buf = buffer_f32_duplex(&self.device, n);
        self.queue.write_buffer(&p_buf, 0, bytemuck_bytes(params));
        self.queue.write_buffer(&m_buf, 0, bytemuck_bytes(m));
        self.queue.write_buffer(&v_buf, 0, bytemuck_bytes(v));
        let dims_buf = upload_uniform_u32(&self.device, &self.queue, &dims);
        dispatch(
            &self.device,
            &self.queue,
            &self.adamw,
            [ceil_div(n, 256) as u32, 1],
            &[
                bind(0, &g_buf),
                bind(1, &p_buf),
                bind(2, &m_buf),
                bind(3, &v_buf),
                bind(4, &dims_buf),
            ],
        );
        let p_out = readback_f32(&self.device, &self.queue, &p_buf, n);
        params.copy_from_slice(&p_out);
        let m_out = readback_f32(&self.device, &self.queue, &m_buf, n);
        m.copy_from_slice(&m_out);
        let v_out = readback_f32(&self.device, &self.queue, &v_buf, n);
        v.copy_from_slice(&v_out);
    }

    /// ‖g‖₂ over one gradient tensor, for grad clipping. GPU computes 64
    /// (or fewer) workgroup partials in f32; the host folds them in f64 —
    /// matching the CPU's f64 `total_grad_sq` accumulation exactly in
    /// structure, to rounding within f32 partials.
    pub fn grad_norm(&self, grads: &[f32]) -> f32 {
        let n = grads.len();
        if n == 0 {
            return 0.0;
        }
        let workgroups = 64usize.min(n.div_ceil(256));
        let chunk = n.div_ceil(workgroups); // fits u32 for model-scale tensors (< 2^32 / 256)
        let dims = [n as u32, chunk as u32, 0u32, 0u32];
        let g_buf = upload_f32(&self.device, &self.queue, grads);
        let o_buf = storage_out(&self.device, workgroups);
        let dims_buf = upload_uniform_u32(&self.device, &self.queue, &dims);
        dispatch(
            &self.device,
            &self.queue,
            &self.sqnorm,
            [workgroups as u32, 1],
            &[bind(0, &g_buf), bind(1, &o_buf), bind(2, &dims_buf)],
        );
        let partials = readback_f32(&self.device, &self.queue, &o_buf, workgroups);
        // Each partial is already a sum of squares; fold plainly (no-repeat
        // the square like the host would with raw values).
        let sq_total: f64 = partials.iter().map(|&p| p as f64).sum();
        sq_total.sqrt() as f32
    }

    /// In-place `grads *= s` (the clipping scale application).
    pub fn scale_grads(&self, grads: &mut [f32], s: f32) {
        if s == 1.0 {
            return;
        }
        let n = grads.len();
        let dims: [u32; 4] = [s.to_bits(), n as u32, 0, 0];
        let g_buf = buffer_f32_duplex(&self.device, n);
        self.queue.write_buffer(&g_buf, 0, bytemuck_bytes(grads));
        let dims_buf = upload_uniform_u32(&self.device, &self.queue, &dims);
        dispatch(
            &self.device,
            &self.queue,
            &self.grad_scale,
            [ceil_div(n, 256) as u32, 1],
            &[bind(0, &g_buf), bind(1, &dims_buf)],
        );
        let g_out = readback_f32(&self.device, &self.queue, &g_buf, n);
        grads.copy_from_slice(&g_out);
    }

    /// Softmax cross-entropy, `ops::cross_entropy` semantics: returns
    /// `(mean_loss, dlogits, probs)` where `dlogits = (p − onehot)/rows`
    /// and the mean is folded host-side in f64 exactly like the CPU.
    pub fn cross_entropy(
        &self,
        logits: &[f32],
        targets: &[usize],
        rows: usize,
        vocab: usize,
    ) -> (f32, Vec<f32>, Vec<f32>) {
        assert_eq!(logits.len(), rows * vocab, "logits shape");
        assert_eq!(targets.len(), rows, "targets shape");
        if rows == 0 {
            return (0.0, Vec::new(), Vec::new());
        }
        for &t in targets {
            assert!(t < vocab, "target out of vocab");
        }
        let tgt_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();
        let dims = [rows as u32, vocab as u32, 0u32, 0u32];
        let x_buf = upload_f32(&self.device, &self.queue, logits);
        let tgt_buf = upload_u32_storage(&self.device, &self.queue, &tgt_u32);
        let p_buf = storage_out(&self.device, rows * vocab);
        let loss_buf = storage_out(&self.device, rows);
        let dims_buf = upload_uniform_u32(&self.device, &self.queue, &dims);
        dispatch(
            &self.device,
            &self.queue,
            &self.ce_fwd,
            [rows as u32, 1],
            &[
                bind(0, &x_buf),
                bind(1, &tgt_buf),
                bind(2, &p_buf),
                bind(3, &loss_buf),
                bind(4, &dims_buf),
            ],
        );

        // dlogits = (p − onehot)/rows via the elementwise bwd kernel.
        let d_buf = storage_out(&self.device, rows * vocab);
        dispatch(
            &self.device,
            &self.queue,
            &self.ce_bwd,
            [ceil_div(rows * vocab, 256) as u32, 1],
            &[bind(0, &p_buf), bind(1, &tgt_buf), bind(2, &d_buf), bind(3, &dims_buf)],
        );

        let probs = readback_f32(&self.device, &self.queue, &p_buf, rows * vocab);
        let per_row = readback_f32(&self.device, &self.queue, &loss_buf, rows);
        let loss =
            (per_row.iter().map(|&l| l as f64).sum::<f64>() / rows as f64) as f32;
        let dlogits = readback_f32(&self.device, &self.queue, &d_buf, rows * vocab);
        (loss, dlogits, probs)
    }

    /// Embedding gather (`modules::Embedding::forward`): row-per-token.
    pub fn embedding_fwd(&self, table: &[f32], ids: &[usize], dim: usize) -> Vec<f32> {
        let rows = ids.len();
        assert!(table.len() % dim == 0, "table shape");
        for &t in ids {
            assert!(t * dim < table.len(), "id out of vocab");
        }
        if rows == 0 {
            return Vec::new();
        }
        let ids_u32: Vec<u32> = ids.iter().map(|&t| t as u32).collect();
        let dims = [(rows * dim) as u32, dim as u32, 0u32, 0u32];
        let table_buf = upload_f32(&self.device, &self.queue, table);
        let ids_buf = upload_u32_storage(&self.device, &self.queue, &ids_u32);
        let out_buf = storage_out(&self.device, rows * dim);
        let dims_buf = upload_uniform_u32(&self.device, &self.queue, &dims);
        dispatch(
            &self.device,
            &self.queue,
            &self.emb_fwd,
            [ceil_div(rows * dim, 256) as u32, 1],
            &[
                bind(0, &table_buf),
                bind(1, &ids_buf),
                bind(2, &out_buf),
                bind(3, &dims_buf),
            ],
        );
        readback_f32(&self.device, &self.queue, &out_buf, rows * dim)
    }

    /// Embedding scatter-add backward (`Embedding::backward`): accumulates
    /// the masked row-sums into the existing `grads` table (duplex buffer,
    /// `+=` semantics like the CPU).
    pub fn embedding_bwd(&self, ids: &[usize], dx: &[f32], grads: &mut [f32], dim: usize) {
        let rows = ids.len();
        assert_eq!(dx.len(), rows * dim, "dx shape");
        assert!(grads.len() % dim == 0, "grads shape");
        let vocab = grads.len() / dim;
        for &t in ids {
            assert!(t < vocab, "id out of vocab");
        }
        if rows == 0 || vocab == 0 || dim == 0 {
            return;
        }
        let ids_u32: Vec<u32> = ids.iter().map(|&t| t as u32).collect();
        let dims = [(vocab * dim) as u32, rows as u32, dim as u32, 0u32];
        let dx_buf = upload_f32(&self.device, &self.queue, dx);
        let ids_buf = upload_u32_storage(&self.device, &self.queue, &ids_u32);
        let dt_buf = buffer_f32_duplex(&self.device, vocab * dim);
        self.queue.write_buffer(&dt_buf, 0, bytemuck_bytes(grads));
        let dims_buf = upload_uniform_u32(&self.device, &self.queue, &dims);
        dispatch(
            &self.device,
            &self.queue,
            &self.emb_bwd,
            [ceil_div(vocab * dim, 256) as u32, 1],
            &[
                bind(0, &dx_buf),
                bind(1, &ids_buf),
                bind(2, &dt_buf),
                bind(3, &dims_buf),
            ],
        );
        let dt_out = readback_f32(&self.device, &self.queue, &dt_buf, vocab * dim);
        grads.copy_from_slice(&dt_out);
    }
}

// ---------------------------------------------------------------------------
// Stage 2.5: causal attention kernels. `GpuAttention` is fixed-shape at
// construction and keeps its attention and QKV Scratch on-device between the
// forward and backward calls — the first real residency slice.
// ---------------------------------------------------------------------------

/// Split the fused QKV projection (B·T, 3C) into per-head buffers
/// q/k/v, laid out flat over the (batch, head) pairs:
/// `buf[(bi·nh + h)·T·d + tt·d + j]`. Thread-per-element over b·nh·T·d.
const SPLIT_QKV_WGSL: &str = r#"
struct AttnDims {
    b: u32,     // batch
    t: u32,     // sequence length T
    c: u32,     // embed dim C (3C is the fused QKV width)
    d: u32,     // per-head dim, nh = c / d
    scale: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}


@group(0) @binding(0) var<storage, read> qkv_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> q_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> k_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> v_buf: array<f32>;
@group(0) @binding(4) var<uniform> dims: AttnDims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let t = dims.t;
    let c = dims.c;
    let d = dims.d;
    let td = t * d;
    if (i >= dims.b * (c / d) * td) {
        return;
    }
    let bh = i / td;
    let r = i - bh * td;
    let tt = r / d;
    let j = r - tt * d;
    let bi = bh / (c / d);
    let h = bh - bi * (c / d);
    let src = (bi * t + tt) * 3 * c;
    let hd = h * d;
    q_buf[i] = qkv_buf[src + hd + j];
    k_buf[i] = qkv_buf[src + c + hd + j];
    v_buf[i] = qkv_buf[src + 2 * c + hd + j];
}
"#;

/// Causal attention forward, one workgroup per (batch, head) with 128
/// threads each owning rows `i = lid, lid+WG, …` (no barriers — every
/// thread touches only its own rows). Per row i:
/// raw scores for s ≤ i go into p (used as same-thread scratch), then the
/// max-shifted softmax divides them in place, transposed copies land in
/// pT (upper triangle stays zero for the backward GEMMs), and the output
/// row `concat[i, h·d..]` accumulates `p[i,s]·v[s]` (first term written
/// `=` like the CPU, so no prezero of concat is needed).
const ATTN_FWD_WGSL: &str = r#"
struct AttnDims {
    b: u32,     // batch
    t: u32,     // sequence length T
    c: u32,     // embed dim C (3C is the fused QKV width)
    d: u32,     // per-head dim, nh = c / d
    scale: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}


@group(0) @binding(0) var<storage, read> q_buf: array<f32>;
@group(0) @binding(1) var<storage, read> k_buf: array<f32>;
@group(0) @binding(2) var<storage, read> v_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> p_buf: array<f32>;
@group(0) @binding(4) var<storage, read_write> p_t_buf: array<f32>;
@group(0) @binding(5) var<storage, read_write> concat_buf: array<f32>;
@group(0) @binding(6) var<uniform> dims: AttnDims;

const WG: u32 = 128u;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let bh = wid.x;
    let t = dims.t;
    let c = dims.c;
    let d = dims.d;
    let nh = c / d;
    let bi = bh / nh;
    let h = bh - bi * nh;
    let pbase = bh * t * t;

    for (var i = lid.x; i < t; i = i + WG) {
        let p_i = pbase + i * t;
        // raw scores into p (s ≤ i; the s > i tail stays zero forever)
        for (var s = 0u; s <= i; s = s + 1u) {
            var acc = 0.0;
            let qrow = bh * t * d + i * d;
            let krow = bh * t * d + s * d;
            for (var j = 0u; j < d; j = j + 1u) {
                acc = acc + q_buf[qrow + j] * k_buf[krow + j];
            }
            p_buf[p_i + s] = acc * dims.scale;
        }
        // max + Σ over the unmasked positions only
        var mx = -3.4e38;
        for (var s = 0u; s <= i; s = s + 1u) {
            mx = max(mx, p_buf[p_i + s]);
        }
        var z = 0.0;
        for (var s = 0u; s <= i; s = s + 1u) {
            z = z + exp(p_buf[p_i + s] - mx);
        }
        // in-place softmax + transposed copies
        for (var s = 0u; s <= i; s = s + 1u) {
            let e = exp(p_buf[p_i + s] - mx) / z;
            p_buf[p_i + s] = e;
            p_t_buf[pbase + s * t + i] = e;
        }
        // output row: concat[(bi·t+i), h·d..] = Σ_{s≤i} p[i,s]·v[s]
        let dst = (bi * t + i) * c + h * d;
        let vbase = bh * t * d;
        for (var s = 0u; s <= i; s = s + 1u) {
            let w = p_buf[p_i + s];
            let vrow = vbase + s * d;
            for (var j = 0u; j < d; j = j + 1u) {
                concat_buf[dst + j] = select(concat_buf[dst + j] + w * v_buf[vrow + j], w * v_buf[vrow + j], s == 0u);
            }
        }
    }
}
"#;

/// Split `dconcat` (B·T, C) into per-head rows matching q/k/v layout.
const SPLIT_DH_WGSL: &str = r#"
struct AttnDims {
    b: u32,     // batch
    t: u32,     // sequence length T
    c: u32,     // embed dim C (3C is the fused QKV width)
    d: u32,     // per-head dim, nh = c / d
    scale: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}


@group(0) @binding(0) var<storage, read> dc_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> dh_buf: array<f32>;
@group(0) @binding(2) var<uniform> dims: AttnDims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let t = dims.t;
    let c = dims.c;
    let d = dims.d;
    let td = t * d;
    if (i >= dims.b * (c / d) * td) {
        return;
    }
    let bh = i / td;
    let r = i - bh * td;
    let tt = r / d;
    let j = r - tt * d;
    let bi = bh / (c / d);
    let h = bh - bi * (c / d);
    dh_buf[i] = dc_buf[(bi * t + tt) * c + h * d + j];
}
"#;

/// Attention backward, per-thread row `i` of each (b,h): 
/// `dp[i,s] = ⟨dh[i], v[s]⟩` (staged in the ds buffer as same-thread scratch),
/// `dot = Σ_s dp·p`,  `ds[i,s] = p[i,s]·(dp − dot)·scale` written to both the
/// row-major ds buffer and its transpose (readers see the s > i tail as the
/// prezeroed zeros the math needs).
const ATTN_BWD_DS_WGSL: &str = r#"
struct AttnDims {
    b: u32,     // batch
    t: u32,     // sequence length T
    c: u32,     // embed dim C (3C is the fused QKV width)
    d: u32,     // per-head dim, nh = c / d
    scale: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}


@group(0) @binding(0) var<storage, read> dh_buf: array<f32>;
@group(0) @binding(1) var<storage, read> v_buf: array<f32>;
@group(0) @binding(2) var<storage, read> p_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> ds_buf: array<f32>;
@group(0) @binding(4) var<storage, read_write> ds_t_buf: array<f32>;
@group(0) @binding(5) var<uniform> dims: AttnDims;

const WG: u32 = 128u;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let bh = wid.x;
    let t = dims.t;
    let d = dims.d;
    let pbase = bh * t * t;
    let vbase = bh * t * d;
    let dhd = bh * t * d;

    for (var i = lid.x; i < t; i = i + WG) {
        let p_i = pbase + i * t;
        for (var s = 0u; s <= i; s = s + 1u) {
            var dp = 0.0;
            let vrow = vbase + s * d;
            for (var j = 0u; j < d; j = j + 1u) {
                dp = dp + dh_buf[dhd + i * d + j] * v_buf[vrow + j];
            }
            ds_buf[p_i + s] = dp;
        }
        var dot = 0.0;
        for (var s = 0u; s <= i; s = s + 1u) {
            dot = dot + ds_buf[p_i + s] * p_buf[p_i + s];
        }
        for (var s = 0u; s <= i; s = s + 1u) {
            let dsc = p_buf[p_i + s] * (ds_buf[p_i + s] - dot) * dims.scale;
            ds_buf[p_i + s] = dsc;
            ds_t_buf[pbase + s * t + i] = dsc;
        }
    }
}
"#;

/// Batched GEMM for the head gradients: per batch slice
/// `y (m,n) = x (m,k) @ w (k,n)` with fixed m=T k=T n=d and batch strides
/// m·k / k·n / m·n. Thread-per-element over nb·m·n (tiny tiles, T·T·d work
/// each — no tiling needed at this size).
const BATCHED_GEMM_WGSL: &str = r#"
struct BgDims {
    nb: u32,
    m: u32,
    k: u32,
    n: u32,
}

@group(0) @binding(0) var<storage, read> x_buf: array<f32>;
@group(0) @binding(1) var<storage, read> w_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> y_buf: array<f32>;
@group(0) @binding(3) var<uniform> dims: BgDims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.nb * dims.m * dims.n) {
        return;
    }
    let mn = dims.m * dims.n;
    let b = i / mn;
    let r = i - b * mn;
    let row = r / dims.n;
    let col = r - row * dims.n;
    var acc = 0.0;
    let xbase = b * dims.m * dims.k + row * dims.k;
    let wbase = b * dims.k * dims.n;
    for (var kk = 0u; kk < dims.k; kk = kk + 1u) {
        acc = acc + x_buf[xbase + kk] * w_buf[wbase + kk * dims.n + col];
    }
    y_buf[i] = acc;
}
"#;

/// Scatter dq/dk/dv (per-head layout) back into the fused QKV gradient
/// (B·T, 3C). Head column ranges are exclusive, so plain writes are safe.
const SCATTER_DQKV_WGSL: &str = r#"
struct AttnDims {
    b: u32,     // batch
    t: u32,     // sequence length T
    c: u32,     // embed dim C (3C is the fused QKV width)
    d: u32,     // per-head dim, nh = c / d
    scale: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}


@group(0) @binding(0) var<storage, read> dq_buf: array<f32>;
@group(0) @binding(1) var<storage, read> dk_buf: array<f32>;
@group(0) @binding(2) var<storage, read> dv_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> qkv_buf: array<f32>;
@group(0) @binding(4) var<uniform> dims: AttnDims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let t = dims.t;
    let c = dims.c;
    let d = dims.d;
    let td = t * d;
    if (i >= dims.b * (c / d) * td) {
        return;
    }
    let bh = i / td;
    let r = i - bh * td;
    let tt = r / d;
    let j = r - tt * d;
    let bi = bh / (c / d);
    let h = bh - bi * (c / d);
    let hd = h * d;
    let dst = (bi * t + tt) * 3 * c;
    qkv_buf[dst + hd + j] = dq_buf[i];
    qkv_buf[dst + c + hd + j] = dk_buf[i];
    qkv_buf[dst + 2 * c + hd + j] = dv_buf[i];
}
"#;

/// Fixed-shape GPU causal attention (`attention::Attention` core, without
/// the surrounding Linears — compose those with `GpuGemm`/CPU). The scratch
/// buffers (p, pT, ds, dsT, q/k/v, dh, dq/dk/dv) are created once and stay
/// on device across forward→backward: no per-call reallocation.
pub struct GpuAttention {
    device: wgpu::Device,
    queue: wgpu::Queue,
    b: usize,
    t: usize,
    c: usize,
    d: usize,
    bh: usize,
    split_qkv: wgpu::ComputePipeline,
    attn_fwd: wgpu::ComputePipeline,
    split_dh: wgpu::ComputePipeline,
    attn_bwd_ds: wgpu::ComputePipeline,
    batched_gemm: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    // persistent device buffers + uniforms
    qkv_in: wgpu::Buffer,     // (b·t, 3c) fused QKV input, host-uploaded
    dconcat_in: wgpu::Buffer, // (b·t, c) upstream grad, host-uploaded
    concat_out: wgpu::Buffer, // (b·t, c) attention output, host readback
    dqkv_out: wgpu::Buffer,   // (b·t, 3c) fused QKV grad, host readback
    q_buf: wgpu::Buffer,
    k_buf: wgpu::Buffer,
    v_buf: wgpu::Buffer,
    dh_buf: wgpu::Buffer,
    p_buf: wgpu::Buffer,
    p_t_buf: wgpu::Buffer,
    ds_buf: wgpu::Buffer,
    ds_t_buf: wgpu::Buffer,
    dq_buf: wgpu::Buffer,
    dk_buf: wgpu::Buffer,
    dv_buf: wgpu::Buffer,
    attn_dims: wgpu::Buffer,
    bg_dims: wgpu::Buffer,
}

impl GpuAttention {
    /// Shapes are fixed: everything downstream asserts against them.
    pub fn new(b: usize, t: usize, c: usize, n_head: usize) -> Result<Self, String> {
        assert_eq!(c % n_head, 0, "c must split evenly among heads");
        let Device { device, queue } = open_device()?;
        let d = c / n_head;
        let bh = b * n_head;
        let split_qkv = pipeline(&device, "attn-split-qkv", SPLIT_QKV_WGSL);
        let attn_fwd = pipeline(&device, "attn-fwd", ATTN_FWD_WGSL);
        let split_dh = pipeline(&device, "attn-split-dh", SPLIT_DH_WGSL);
        let attn_bwd_ds = pipeline(&device, "attn-bwd-ds", ATTN_BWD_DS_WGSL);
        let batched_gemm = pipeline(&device, "attn-bgemm", BATCHED_GEMM_WGSL);
        let scatter = pipeline(&device, "attn-scatter", SCATTER_DQKV_WGSL);

        let scale = 1.0 / (d as f32).sqrt();
        let attn_dims_buf = upload_uniform_u32(&device, &queue, &[
            b as u32,
            t as u32,
            c as u32,
            d as u32,
            scale.to_bits(),
            0u32,
            0u32,
            0u32,
        ]);
        let bg_dims_buf =
            upload_uniform_u32(&device, &queue, &[bh as u32, t as u32, t as u32, d as u32]);

        let storage = |n: usize, zero: bool| {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("attn-scratch"),
                size: (n * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: zero,
            });
            if zero {
                buf.unmap(); // zero-filled via the mapped creation path
            }
            buf
        };
        let duplex = |n: usize, label: &str, copy_src: bool| {
            let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
            if copy_src {
                usage |= wgpu::BufferUsages::COPY_SRC;
            }
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (n * 4) as u64,
                usage,
                mapped_at_creation: false,
            })
        };

        let qkv_in = duplex(b * t * 3 * c, "attn-qkv-in", false);
        let dconcat_in = duplex(b * t * c, "attn-dconcat-in", false);
        let concat_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("attn-concat-out"),
            size: (b * t * c * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dqkv_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("attn-dqkv-out"),
            size: (b * t * 3 * c * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let q_buf = storage(bh * t * d, false);
        let k_buf = storage(bh * t * d, false);
        let v_buf = storage(bh * t * d, false);
        let dh_buf = storage(bh * t * d, false);
        let dq_buf = storage(bh * t * d, false);
        let dk_buf = storage(bh * t * d, false);
        let dv_buf = storage(bh * t * d, false);
        // p / pT / ds / dsT: lower triangle written by kernels, upper tail
        // must read zero → zero-fill at creation once.
        let p_buf = storage(bh * t * t, true);
        let p_t_buf = storage(bh * t * t, true);
        let ds_buf = storage(bh * t * t, true);
        let ds_t_buf = storage(bh * t * t, true);

        Ok(Self {
            device,
            queue,
            b,
            t,
            c,
            d,
            bh,
            split_qkv,
            attn_fwd,
            split_dh,
            attn_bwd_ds,
            batched_gemm,
            scatter,
            qkv_in,
            dconcat_in,
            concat_out,
            dqkv_out,
            q_buf,
            k_buf,
            v_buf,
            dh_buf,
            p_buf,
            p_t_buf,
            ds_buf,
            ds_t_buf,
            dq_buf,
            dk_buf,
            dv_buf,
            attn_dims: attn_dims_buf,
            bg_dims: bg_dims_buf,
        })
    }

    /// Attention forward given the already-computed fused QKV projection
    /// `(b·t, 3c)`; returns the head-concatenated output `(b·t, c)` (the
    /// caller runs c_proj on it). Leaves q/k/v/p/pT resident on device for
    /// the matching `backward` call.
    pub fn forward(&self, qkv_flat: &[f32]) -> Vec<f32> {
        assert_eq!(qkv_flat.len(), self.b * self.t * 3 * self.c, "qkv shape");
        self.queue.write_buffer(&self.qkv_in, 0, bytemuck_bytes(qkv_flat));

        let n = self.bh * self.t * self.d;
        dispatch(
            &self.device,
            &self.queue,
            &self.split_qkv,
            [ceil_div(n, 256) as u32, 1],
            &[
                bind(0, &self.qkv_in),
                bind(1, &self.q_buf),
                bind(2, &self.k_buf),
                bind(3, &self.v_buf),
                bind(4, &self.attn_dims),
            ],
        );
        dispatch(
            &self.device,
            &self.queue,
            &self.attn_fwd,
            [self.bh as u32, 1],
            &[
                bind(0, &self.q_buf),
                bind(1, &self.k_buf),
                bind(2, &self.v_buf),
                bind(3, &self.p_buf),
                bind(4, &self.p_t_buf),
                bind(5, &self.concat_out),
                bind(6, &self.attn_dims),
            ],
        );
        readback_f32(&self.device, &self.queue, &self.concat_out, self.b * self.t * self.c)
    }

    /// Attention backward given `dconcat` (the c_proj'd upstream gradient,
    /// `(b·t, c)`); returns the fused QKV gradient `(b·t, 3c)` (the caller
    /// runs c_qkv.backward on it). Consumes the forward call's resident
    /// device state.
    pub fn backward(&self, dconcat: &[f32]) -> Vec<f32> {
        assert_eq!(dconcat.len(), self.b * self.t * self.c, "dconcat shape");
        self.queue.write_buffer(&self.dconcat_in, 0, bytemuck_bytes(dconcat));

        let n = self.bh * self.t * self.d;
        dispatch(
            &self.device,
            &self.queue,
            &self.split_dh,
            [ceil_div(n, 256) as u32, 1],
            &[
                bind(0, &self.dconcat_in),
                bind(1, &self.dh_buf),
                bind(2, &self.attn_dims),
            ],
        );
        dispatch(
            &self.device,
            &self.queue,
            &self.attn_bwd_ds,
            [self.bh as u32, 1],
            &[
                bind(0, &self.dh_buf),
                bind(1, &self.v_buf),
                bind(2, &self.p_buf),
                bind(3, &self.ds_buf),
                bind(4, &self.ds_t_buf),
                bind(5, &self.attn_dims),
            ],
        );
        // dq = ds@k, dk = dsT@q, dv = pT@dh — one batched kernel, 3 dispatches.
        let gem = |x: &wgpu::Buffer, w: &wgpu::Buffer, y: &wgpu::Buffer| {
            dispatch(
                &self.device,
                &self.queue,
                &self.batched_gemm,
                [ceil_div(n, 256) as u32, 1],
                &[bind(0, x), bind(1, w), bind(2, y), bind(3, &self.bg_dims)],
            );
        };
        gem(&self.ds_buf, &self.k_buf, &self.dq_buf);
        gem(&self.ds_t_buf, &self.q_buf, &self.dk_buf);
        gem(&self.p_t_buf, &self.dh_buf, &self.dv_buf);
        dispatch(
            &self.device,
            &self.queue,
            &self.scatter,
            [ceil_div(n, 256) as u32, 1],
            &[
                bind(0, &self.dq_buf),
                bind(1, &self.dk_buf),
                bind(2, &self.dv_buf),
                bind(3, &self.dqkv_out),
                bind(4, &self.attn_dims),
            ],
        );
        readback_f32(
            &self.device,
            &self.queue,
            &self.dqkv_out,
            self.b * self.t * 3 * self.c,
        )
    }
}

/// Read-only-storage-size output buffer (kernel writes, host copies out).
fn storage_out(device: &wgpu::Device, len_f32: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu-out"),
        size: (len_f32 * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// In-out storage buffer: host writes via COPY_DST, kernel mutates, host
/// reads back via COPY_SRC (needed for in-place ops like AdamW).
fn buffer_f32_duplex(device: &wgpu::Device, len_f32: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu-inout"),
        size: (len_f32 * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
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
        label: Some("gpu-bind"),
        layout: &layout,
        entries,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu-enc"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups[0], groups[1], 1);
    }
    queue.submit(Some(encoder.finish()));
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

/// Uploads u32 data into a read-only storage buffer (token targets, ids).
fn upload_u32_storage(device: &wgpu::Device, queue: &wgpu::Queue, data: &[u32]) -> wgpu::Buffer {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu-u32-in"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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

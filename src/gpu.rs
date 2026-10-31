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

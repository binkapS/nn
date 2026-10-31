use crate::gpu;
use crate::transformer::{Config, Gpt, GptParams};
use std::cell::RefCell;

const ADD_WGSL: &str = r#"
struct Dims { n: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> a_buf: array<f32>;
@group(0) @binding(1) var<storage, read> b_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> o_buf: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.n) { return; }
    o_buf[i] = a_buf[i] + b_buf[i];
}
"#;

const FILL_WGSL: &str = r#"
struct Dims { n: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read_write> o_buf: array<f32>;
@group(0) @binding(1) var<uniform> dims: Dims;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.n) { return; }
    o_buf[i] = 0.0;
}
"#;

const LN_EPS: f32 = 1e-5;

const TRANSPOSE_WGSL: &str = r#"
struct Dims { in_d: u32, out_d: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> in_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_buf: array<f32>;
@group(0) @binding(2) var<uniform> dims: Dims;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = dims.in_d * dims.out_d;
    if (i >= n) { return; }
    let row = i / dims.out_d;
    let col = i % dims.out_d;
    out_buf[col * dims.in_d + row] = in_buf[i];
}
"#;

struct ParamBuf {
    n: usize,
    decay: bool,
    shape_in: usize,
    shape_out: usize,
    data: wgpu::Buffer,
    data_t: Option<wgpu::Buffer>,
    grad: wgpu::Buffer,
    m: wgpu::Buffer,
    v: wgpu::Buffer,
}

struct AttnBufs {
    q: wgpu::Buffer, k: wgpu::Buffer, v: wgpu::Buffer,
    dh: wgpu::Buffer, dq: wgpu::Buffer, dk: wgpu::Buffer, dv: wgpu::Buffer,
    p: wgpu::Buffer, pt: wgpu::Buffer,
    ds: wgpu::Buffer, dst: wgpu::Buffer,
    attn_dims: wgpu::Buffer, bg_dims: wgpu::Buffer,
}

struct BlockParamsGpu {
    ln1_g: ParamBuf, ln1_b: ParamBuf,
    cq_w: ParamBuf, cq_b: ParamBuf,
    proj_w: ParamBuf, proj_b: ParamBuf,
    ln2_g: ParamBuf, ln2_b: ParamBuf,
    fc1_w: ParamBuf, fc1_b: ParamBuf,
    fc2_w: ParamBuf, fc2_b: ParamBuf,
}

#[allow(dead_code)]
pub struct GpuEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pool: RefCell<gpu::Pool>,

    gemm_fwd: wgpu::ComputePipeline, gemm_dwt: wgpu::ComputePipeline,
    ln_fwd: wgpu::ComputePipeline, ln_bwd: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline, gelu_fwd: wgpu::ComputePipeline, gelu_bwd: wgpu::ComputePipeline,
    adamw: wgpu::ComputePipeline, sqnorm: wgpu::ComputePipeline, grad_scale: wgpu::ComputePipeline,
    ce_fwd: wgpu::ComputePipeline, ce_bwd: wgpu::ComputePipeline,
    emb_fwd: wgpu::ComputePipeline, emb_bwd: wgpu::ComputePipeline,
    split_qkv: wgpu::ComputePipeline, attn_fwd: wgpu::ComputePipeline,
    split_dh: wgpu::ComputePipeline, attn_bwd_ds: wgpu::ComputePipeline,
    batched_gemm: wgpu::ComputePipeline, scatter: wgpu::ComputePipeline,
    add: wgpu::ComputePipeline, fill: wgpu::ComputePipeline,
    transpose_pipe: wgpu::ComputePipeline,

    cfg: Config, b: usize, t: usize, c: usize, v: usize, nh: usize, d: usize,
    rows: usize, n_layer: usize,

    tok_emb: ParamBuf, pos_emb: ParamBuf,
    blocks: Vec<BlockParamsGpu>,
    ln_f_g: ParamBuf, ln_f_b: ParamBuf,
    head_w: ParamBuf, head_b: ParamBuf,

    x_stream: Vec<wgpu::Buffer>,
    blk_x1: Vec<wgpu::Buffer>, blk_ln1out: Vec<wgpu::Buffer>,
    blk_ln2out: Vec<wgpu::Buffer>, blk_fc1out: Vec<wgpu::Buffer>,
    blk_qkv: Vec<wgpu::Buffer>,
    blk_attnout: Vec<wgpu::Buffer>, attns: Vec<AttnBufs>,
    ln_f_out: wgpu::Buffer, logits_buf: wgpu::Buffer,
    probs: wgpu::Buffer, loss_buf: wgpu::Buffer,
    tmp_rc: wgpu::Buffer, tmp_4rc: wgpu::Buffer,
    sc1_rc: wgpu::Buffer, sc2_rc: wgpu::Buffer, sc3_rc: wgpu::Buffer, sc4rc: wgpu::Buffer,

    token_ids_buf: wgpu::Buffer, target_ids_buf: wgpu::Buffer, pos_ids_buf: wgpu::Buffer,

    step: u32,
}

// helpers
fn duplex(device: &wgpu::Device, n: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("duplex"),
        size: (n*4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}
fn make_uniform_u32(device: &wgpu::Device, queue: &wgpu::Queue, data: &[u32]) -> wgpu::Buffer {
    let mut bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    while !bytes.len().is_multiple_of(16) { bytes.push(0); }
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniform"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, &bytes);
    buf
}
fn upload_f32(device: &wgpu::Device, queue: &wgpu::Queue, data: &[f32]) -> wgpu::Buffer {
    let b = duplex(device, data.len());
    queue.write_buffer(&b, 0, gpu::bytemuck_bytes(data));
    b
}
fn zstorage(device: &wgpu::Device, n: usize) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("zs"),
        size: (n*4) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: true,
    });
    buf.unmap();
    buf
}
fn fill_buf(fill: &wgpu::ComputePipeline, device: &wgpu::Device, queue: &wgpu::Queue, pool: &RefCell<gpu::Pool>, buf: &wgpu::Buffer, n: usize) {
    let dimbuf = gpu::pool_uniform_u32(pool, device, queue, &[n as u32, 0, 0, 0]);
    gpu::dispatch(device, queue, fill, [gpu::ceil_div(n, 256) as u32, 1], &[
        gpu::bind(0, buf),
        gpu::bind(1, &dimbuf),
    ]);
    gpu::pool_release(pool, vec![dimbuf]);
}
fn zero_duplex(fill: &wgpu::ComputePipeline, device: &wgpu::Device, queue: &wgpu::Queue, pool: &RefCell<gpu::Pool>, buf: &wgpu::Buffer, n: usize) {
    if n == 0 { return; }
    fill_buf(fill, device, queue, pool, buf, n);
}
#[allow(clippy::too_many_arguments)]
fn make_param(device:&wgpu::Device, queue:&wgpu::Queue, fill:&wgpu::ComputePipeline, pool:&RefCell<gpu::Pool>,
    n:usize, decay:bool, data:&[f32], data_t:Option<Vec<f32>>, shape_in:usize, shape_out:usize) -> ParamBuf {
    let buf = upload_f32(device, queue, data);
    let tbuf = data_t.map(|d| upload_f32(device, queue, &d));
    let grad = duplex(device, n); zero_duplex(fill, device, queue, pool, &grad, n);
    let m = duplex(device, n); zero_duplex(fill, device, queue, pool, &m, n);
    let v = duplex(device, n); zero_duplex(fill, device, queue, pool, &v, n);
    ParamBuf { n, decay, shape_in, shape_out, data: buf, data_t: tbuf, grad, m, v }
}
fn transpose(src:&[f32], in_d:usize, out_d:usize) -> Vec<f32> {
    let mut t = vec![0.0f32; in_d*out_d];
    for i in 0..in_d { for o in 0..out_d { t[o*in_d+i] = src[i*out_d+o]; } }
    t
}
fn copy_buf(device:&wgpu::Device, queue:&wgpu::Queue, src:&wgpu::Buffer, dst:&wgpu::Buffer, size:usize) {
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("copy") });
    enc.copy_buffer_to_buffer(src,0,dst,0,size as u64);
    queue.submit(Some(enc.finish()));
}
#[allow(clippy::too_many_arguments)]
fn gemm_fwd_call(device:&wgpu::Device, queue:&wgpu::Queue, pipeline:&wgpu::ComputePipeline, pool:&RefCell<gpu::Pool>,
    x:&wgpu::Buffer, wt:&wgpu::Buffer, bias:Option<&wgpu::Buffer>, y:&wgpu::Buffer, m:usize,k:usize,n:usize) {
    let (bias_buf, use_bias) = match bias { Some(b) => (b.clone(),1u32), None => (duplex(device,1),0u32) };
    let dims=[m as u32,k as u32,n as u32,use_bias];
    let dims_buf=gpu::pool_uniform_u32(pool,device,queue,&dims);
    gpu::dispatch(device,queue,pipeline,[gpu::ceil_div(n,16) as u32,gpu::ceil_div(m,16) as u32],&[
        gpu::bind(0,x), gpu::bind(1,wt), gpu::bind(2,&bias_buf), gpu::bind(3,y), gpu::bind(4,&dims_buf)]);
    gpu::pool_release(pool,vec![dims_buf]);
}
#[allow(clippy::too_many_arguments)]
fn gemm_dwt_call(device:&wgpu::Device, queue:&wgpu::Queue, pipeline:&wgpu::ComputePipeline, pool:&RefCell<gpu::Pool>,
    x:&wgpu::Buffer, gy:&wgpu::Buffer, dwt:&wgpu::Buffer, m:usize,k:usize,n:usize) {
    let dims=[m as u32,k as u32,n as u32,0u32];
    let dims_buf=gpu::pool_uniform_u32(pool,device,queue,&dims);
    gpu::dispatch(device,queue,pipeline,[gpu::ceil_div(n,16) as u32,gpu::ceil_div(k,16) as u32],&[
        gpu::bind(0,x), gpu::bind(1,gy), gpu::bind(2,dwt), gpu::bind(3,&dims_buf)]);
    gpu::pool_release(pool,vec![dims_buf]);
}
#[allow(clippy::too_many_arguments)]
fn reduce_cols_call(device:&wgpu::Device, queue:&wgpu::Queue, pipeline:&wgpu::ComputePipeline, pool:&RefCell<gpu::Pool>,
    src:&wgpu::Buffer, dst:&wgpu::Buffer, rows:usize, cols:usize) {
    let dims=[rows as u32, cols as u32,0,0];
    let dims_buf=gpu::pool_uniform_u32(pool,device,queue,&dims);
    gpu::dispatch(device,queue,pipeline,[gpu::ceil_div(cols,256) as u32,1],&[
        gpu::bind(0,src), gpu::bind(1,dst), gpu::bind(2,&dims_buf)]);
    gpu::pool_release(pool,vec![dims_buf]);
}
#[allow(clippy::too_many_arguments)]
fn sqsum_tensor(sq:&wgpu::ComputePipeline, device:&wgpu::Device, queue:&wgpu::Queue, pool:&RefCell<gpu::Pool>, grad:&wgpu::Buffer, n:usize) -> f64 {
    if n==0 { return 0.0; }
    let wg=64usize.min(n.div_ceil(256));
    let chunk=n.div_ceil(wg);
    let dims=[n as u32,chunk as u32,0,0];
    let dims_buf=gpu::pool_uniform_u32(pool,device,queue,&dims);
    let o=duplex(device,wg);
    gpu::dispatch(device,queue,sq,[wg as u32,1],&[gpu::bind(0,grad),gpu::bind(1,&o),gpu::bind(2,&dims_buf)]);
    let partials=gpu::readback_f32(device,queue,&o,wg);
    let s:f64=partials.iter().map(|&p| p as f64).sum();
    gpu::pool_release(pool,vec![dims_buf]);
    s
}

impl GpuEngine {
    pub fn from_gpt(gpt:&Gpt, batch:usize) -> Result<Self,String> {
        let p=gpt.export_params();
        let cfg=p.cfg;
        let n_layer=cfg.n_layer; let nh=cfg.n_head; let c=cfg.n_embd; let t=cfg.block_size; let v=cfg.vocab_size;
        let b=batch; let rows=b*t; let d=c/nh;
        let dev=gpu::open_device()?; let device=dev.device; let queue=dev.queue;
        // pipelines
        let gemm_fwd=gpu::pipeline(&device,"gemm-fwd",gpu::GEMM_FWD_WGSL);
        let gemm_dwt=gpu::pipeline(&device,"gemm-dwt",gpu::GEMM_DWT_WGSL);
        let ln_fwd=gpu::pipeline(&device,"ln-fwd",gpu::LN_FWD_WGSL);
        let ln_bwd=gpu::pipeline(&device,"ln-bwd",gpu::LN_BWD_WGSL);
        let reduce=gpu::pipeline(&device,"reduce-cols",gpu::REDUCE_COLS_WGSL);
        let gelu_fwd=gpu::pipeline(&device,"gelu-fwd",gpu::GELU_FWD_WGSL);
        let gelu_bwd=gpu::pipeline(&device,"gelu-bwd",gpu::GELU_BWD_WGSL);
        let adamw=gpu::pipeline(&device,"adamw",gpu::ADAMW_WGSL);
        let sqnorm=gpu::pipeline(&device,"sqnorm",gpu::SQNORM_WGSL);
        let grad_scale=gpu::pipeline(&device,"grad-scale",gpu::GRAD_SCALE_WGSL);
        let ce_fwd=gpu::pipeline(&device,"ce-fwd",gpu::CE_FWD_WGSL);
        let ce_bwd=gpu::pipeline(&device,"ce-bwd",gpu::CE_BWD_WGSL);
        let emb_fwd=gpu::pipeline(&device,"emb-fwd",gpu::EMB_FWD_WGSL);
        let emb_bwd=gpu::pipeline(&device,"emb-bwd",gpu::EMB_BWD_WGSL);
        let split_qkv=gpu::pipeline(&device,"attn-split-qkv",gpu::SPLIT_QKV_WGSL);
        let attn_fwd_p=gpu::pipeline(&device,"attn-fwd",gpu::ATTN_FWD_WGSL);
        let split_dh=gpu::pipeline(&device,"attn-split-dh",gpu::SPLIT_DH_WGSL);
        let attn_bwd_ds=gpu::pipeline(&device,"attn-bwd-ds",gpu::ATTN_BWD_DS_WGSL);
        let batched_gemm=gpu::pipeline(&device,"attn-bgemm",gpu::BATCHED_GEMM_WGSL);
        let scatter=gpu::pipeline(&device,"attn-scatter",gpu::SCATTER_DQKV_WGSL);
        let add=gpu::pipeline(&device,"add",ADD_WGSL);
        let fill=gpu::pipeline(&device,"fill",FILL_WGSL);
        let transpose_pipe=gpu::pipeline(&device,"transpose_pipe",TRANSPOSE_WGSL);
        let pool=RefCell::new(gpu::Pool::default());

        let mut blocks=Vec::with_capacity(n_layer);
        let mut blk_x1=Vec::with_capacity(n_layer);
        let mut blk_ln1out=Vec::with_capacity(n_layer);
        let mut blk_ln2out=Vec::with_capacity(n_layer);
        let mut blk_fc1out=Vec::with_capacity(n_layer);
        let mut blk_qkv=Vec::with_capacity(n_layer);
        let mut blk_attnout=Vec::with_capacity(n_layer);
        let mut attns=Vec::with_capacity(n_layer);

        for bp in &p.blocks {
            let ln1_g=make_param(&device,&queue,&fill,&pool,c,false,&bp.ln1_g,None,0,0);
            let ln1_b=make_param(&device,&queue,&fill,&pool,c,false,&bp.ln1_b,None,0,0);
            let cq_w=make_param(&device,&queue,&fill,&pool,c*3*c,true,&bp.cq_w,Some(transpose(&bp.cq_w,c,3*c)),c,3*c);
            let cq_b=make_param(&device,&queue,&fill,&pool,3*c,false,&bp.cq_b,None,0,0);
            let proj_w=make_param(&device,&queue,&fill,&pool,c*c,true,&bp.proj_w,Some(transpose(&bp.proj_w,c,c)),c,c);
            let proj_b=make_param(&device,&queue,&fill,&pool,c,false,&bp.proj_b,None,0,0);
            let ln2_g=make_param(&device,&queue,&fill,&pool,c,false,&bp.ln2_g,None,0,0);
            let ln2_b=make_param(&device,&queue,&fill,&pool,c,false,&bp.ln2_b,None,0,0);
            let fc1_w=make_param(&device,&queue,&fill,&pool,c*4*c,true,&bp.fc1_w,Some(transpose(&bp.fc1_w,c,4*c)),c,4*c);
            let fc1_b=make_param(&device,&queue,&fill,&pool,4*c,false,&bp.fc1_b,None,0,0);
            let fc2_w=make_param(&device,&queue,&fill,&pool,4*c*c,true,&bp.fc2_w,Some(transpose(&bp.fc2_w,4*c,c)),4*c,c);
            let fc2_b=make_param(&device,&queue,&fill,&pool,c,false,&bp.fc2_b,None,0,0);
            blocks.push(BlockParamsGpu{ln1_g,ln1_b,cq_w,cq_b,proj_w,proj_b,ln2_g,ln2_b,fc1_w,fc1_b,fc2_w,fc2_b});
            blk_x1.push(duplex(&device,rows*c));
            blk_ln1out.push(duplex(&device,rows*c));
            blk_ln2out.push(duplex(&device,rows*c));
            blk_fc1out.push(duplex(&device,rows*4*c));
            blk_qkv.push(duplex(&device,rows*3*c));
            blk_attnout.push(duplex(&device,rows*c));
            let bh=b*nh; let nscratch=bh*t*d; let nprob=bh*t*t;
            let scale=1.0/(d as f32).sqrt();
            let attn_dims=make_uniform_u32(&device,&queue,&[b as u32,t as u32,c as u32,d as u32,scale.to_bits(),0,0,0]);
            let bg_dims=make_uniform_u32(&device,&queue,&[bh as u32,t as u32,t as u32,d as u32]);
            let mk_z=|n:usize| zstorage(&device,n);
            attns.push(AttnBufs{
                q:mk_z(nscratch),k:mk_z(nscratch),v:mk_z(nscratch),
                dh:mk_z(nscratch),dq:mk_z(nscratch),dk:mk_z(nscratch),dv:mk_z(nscratch),
                p:mk_z(nprob),pt:mk_z(nprob),ds:mk_z(nprob),dst:mk_z(nprob),
                attn_dims,bg_dims,
            });
        }

        let tok_emb=make_param(&device,&queue,&fill,&pool,v*c,true,&p.tok_emb,None,0,0);
        let pos_emb=make_param(&device,&queue,&fill,&pool,t*c,true,&p.pos_emb,None,0,0);
        let ln_f_g=make_param(&device,&queue,&fill,&pool,c,false,&p.ln_f_g,None,0,0);
        let ln_f_b=make_param(&device,&queue,&fill,&pool,c,false,&p.ln_f_b,None,0,0);
        let head_w=make_param(&device,&queue,&fill,&pool,c*v,true,&p.head_w,Some(transpose(&p.head_w,c,v)),c,v);
        let head_b=make_param(&device,&queue,&fill,&pool,v,false,&p.head_b,None,0,0);

        let mut x_stream=Vec::with_capacity(n_layer+1);
        for _ in 0..=n_layer { x_stream.push(duplex(&device,rows*c)); }
        let ln_f_out=duplex(&device,rows*c);
        let logits_buf=duplex(&device,rows*v);
        let probs=duplex(&device,rows*v);
        let loss_buf=duplex(&device,rows);
        let tmp_rc=duplex(&device,rows*c);
        let tmp_4rc=duplex(&device,rows*4*c);
        let sc1_rc=duplex(&device,rows*c);
        let sc2_rc=duplex(&device,rows*c);
        let sc3_rc=duplex(&device,rows*c);
        let sc4rc=duplex(&device,rows*4*c);

        let pos_ids: Vec<u32> = (0..b).flat_map(|_| (0..t).map(|tt| tt as u32)).collect();
        let pos_ids_buf=gpu::upload_u32_storage(&device,&queue,&pos_ids);
        let token_ids_buf=gpu::upload_u32_storage(&device,&queue,&vec![0u32;rows]);
        let target_ids_buf=gpu::upload_u32_storage(&device,&queue,&vec![0u32;rows]);

        Ok(Self { device,queue,pool,
            gemm_fwd, gemm_dwt, ln_fwd, ln_bwd, reduce, gelu_fwd, gelu_bwd, adamw, sqnorm, grad_scale,
            ce_fwd, ce_bwd, emb_fwd, emb_bwd, split_qkv, attn_fwd:attn_fwd_p, split_dh, attn_bwd_ds,
            batched_gemm, scatter, add, fill, transpose_pipe,
            cfg,b,t,c,v,nh,d,rows,n_layer,
            tok_emb,pos_emb,blocks,ln_f_g,ln_f_b,head_w,head_b,
            x_stream,blk_x1,blk_ln1out,blk_ln2out,blk_fc1out,blk_qkv,blk_attnout,attns,ln_f_out,logits_buf,probs,loss_buf,tmp_rc,tmp_4rc,sc1_rc,sc2_rc,sc3_rc,sc4rc,
            token_ids_buf,target_ids_buf,pos_ids_buf,step:0 })
    }

    pub fn forward_pass(&mut self, ids: &[usize], targets: &[usize]) -> f32 {
        let pool = &self.pool;
        let n_layer=self.n_layer; let c=self.c; let rows=self.rows; let v=self.v; let _b=self.b; let _nh=self.nh; let t=self.t;
        let _=t;
        // upload ids
        let ids_u: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
        let tgt_u: Vec<u32> = targets.iter().map(|&x| x as u32).collect();
        let mut ib = Vec::with_capacity(ids_u.len()*4);
        for u in &ids_u { ib.extend_from_slice(&u.to_le_bytes()); }
        self.queue.write_buffer(&self.token_ids_buf, 0, &ib);
        let mut tb = Vec::with_capacity(tgt_u.len()*4);
        for u in &tgt_u { tb.extend_from_slice(&u.to_le_bytes()); }
        self.queue.write_buffer(&self.target_ids_buf, 0, &tb);

        // tok emb
        {
            let n=rows*c;
            let ub=gpu::pool_uniform_u32(pool,&self.device,&self.queue,&[n as u32,c as u32,0,0]);
            gpu::dispatch(&self.device,&self.queue,&self.emb_fwd,[gpu::ceil_div(n,256) as u32,1],&[
                gpu::bind(0,&self.tok_emb.data), gpu::bind(1,&self.token_ids_buf),
                gpu::bind(2,&self.x_stream[0]), gpu::bind(3,&ub)]);
            gpu::pool_release(pool,vec![ub]);
        }
        // pos emb -> tmp_rc
        {
            let n=rows*c;
            let ub=gpu::pool_uniform_u32(pool,&self.device,&self.queue,&[n as u32,c as u32,0,0]);
            gpu::dispatch(&self.device,&self.queue,&self.emb_fwd,[gpu::ceil_div(n,256) as u32,1],&[
                gpu::bind(0,&self.pos_emb.data), gpu::bind(1,&self.pos_ids_buf),
                gpu::bind(2,&self.tmp_rc), gpu::bind(3,&ub)]);
            gpu::pool_release(pool,vec![ub]);
        }
        // add
        {
            let n=rows*c;
            let ub=make_uniform_u32(&self.device,&self.queue,&[n as u32,0,0,0]);
            gpu::dispatch(&self.device,&self.queue,&self.add,[gpu::ceil_div(n,256) as u32,1],&[
                gpu::bind(0,&self.x_stream[0]), gpu::bind(1,&self.tmp_rc),
                gpu::bind(2,&self.tmp_4rc), gpu::bind(3,&ub)]);
            copy_buf(&self.device,&self.queue,&self.tmp_4rc,&self.x_stream[0],n*4);
        }

        for li in 0..n_layer {
            let bp=&self.blocks[li];
            // ln1
            {
                let ub=gpu::pool_uniform_u32(pool,&self.device,&self.queue,&[rows as u32,c as u32,LN_EPS.to_bits(),0]);
                gpu::dispatch(&self.device,&self.queue,&self.ln_fwd,[rows as u32,1],&[
                    gpu::bind(0,&self.x_stream[li]), gpu::bind(1,&bp.ln1_g.data),
                    gpu::bind(2,&bp.ln1_b.data), gpu::bind(3,&self.blk_ln1out[li]), gpu::bind(4,&ub)]);
                gpu::pool_release(pool,vec![ub]);
            }
            // cq gemm
            gemm_fwd_call(&self.device,&self.queue,&self.gemm_fwd,pool,
                &self.blk_ln1out[li],&bp.cq_w.data,Some(&bp.cq_b.data),&self.blk_qkv[li],rows,c,3*c);
            // attn core
            {
                let ab=&self.attns[li];
                let nsc=rows*c;
                gpu::dispatch(&self.device,&self.queue,&self.split_qkv,[gpu::ceil_div(nsc,256) as u32,1],&[
                    gpu::bind(0,&self.blk_qkv[li]),gpu::bind(1,&ab.q),gpu::bind(2,&ab.k),
                    gpu::bind(3,&ab.v),gpu::bind(4,&ab.attn_dims)]);
                let bh=(self.b*self.nh) as u32;
                gpu::dispatch(&self.device,&self.queue,&self.attn_fwd,[bh,1],&[
                    gpu::bind(0,&ab.q),gpu::bind(1,&ab.k),gpu::bind(2,&ab.v),
                    gpu::bind(3,&ab.p),gpu::bind(4,&ab.pt),
                    gpu::bind(5,&self.blk_attnout[li]),gpu::bind(6,&ab.attn_dims)]);
            }
            // c_proj
            gemm_fwd_call(&self.device,&self.queue,&self.gemm_fwd,pool,
                &self.blk_attnout[li],&bp.proj_w.data,Some(&bp.proj_b.data),&self.tmp_rc,rows,c,c);
            // residual -> blk_x1
            {
                let n=rows*c;
                let ub=make_uniform_u32(&self.device,&self.queue,&[n as u32,0,0,0]);
                gpu::dispatch(&self.device,&self.queue,&self.add,[gpu::ceil_div(n,256) as u32,1],&[
                    gpu::bind(0,&self.x_stream[li]),gpu::bind(1,&self.tmp_rc),
                    gpu::bind(2,&self.blk_x1[li]),gpu::bind(3,&ub)]);
            }
            copy_buf(&self.device,&self.queue,&self.blk_x1[li],&self.x_stream[li+1],rows*c*4);
            // ln2
            {
                let ub=gpu::pool_uniform_u32(pool,&self.device,&self.queue,&[rows as u32,c as u32,LN_EPS.to_bits(),0]);
                gpu::dispatch(&self.device,&self.queue,&self.ln_fwd,[rows as u32,1],&[
                    gpu::bind(0,&self.blk_x1[li]),gpu::bind(1,&bp.ln2_g.data),
                    gpu::bind(2,&bp.ln2_b.data),gpu::bind(3,&self.blk_ln2out[li]),gpu::bind(4,&ub)]);
                gpu::pool_release(pool,vec![ub]);
            }
            // fc1
            gemm_fwd_call(&self.device,&self.queue,&self.gemm_fwd,pool,
                &self.blk_ln2out[li],&bp.fc1_w.data,Some(&bp.fc1_b.data),&self.blk_fc1out[li],rows,c,4*c);
            // gelu
            {
                let n=rows*4*c;
                gpu::dispatch(&self.device,&self.queue,&self.gelu_fwd,[gpu::ceil_div(n,256) as u32,1],&[
                    gpu::bind(0,&self.blk_fc1out[li]),gpu::bind(1,&self.tmp_4rc)]);
            }
            // fc2
            gemm_fwd_call(&self.device,&self.queue,&self.gemm_fwd,pool,
                &self.tmp_4rc,&bp.fc2_w.data,Some(&bp.fc2_b.data),&self.tmp_rc,rows,4*c,c);
            // residual
            {
                let n=rows*c;
                let ub=make_uniform_u32(&self.device,&self.queue,&[n as u32,0,0,0]);
                gpu::dispatch(&self.device,&self.queue,&self.add,[gpu::ceil_div(n,256) as u32,1],&[
                    gpu::bind(0,&self.blk_x1[li]),gpu::bind(1,&self.tmp_rc),
                    gpu::bind(2,&self.x_stream[li+1]),gpu::bind(3,&ub)]);
            }
        }

        // ln_f
        {
            let ub=gpu::pool_uniform_u32(pool,&self.device,&self.queue,&[rows as u32,c as u32,LN_EPS.to_bits(),0]);
            gpu::dispatch(&self.device,&self.queue,&self.ln_fwd,[rows as u32,1],&[
                gpu::bind(0,&self.x_stream[n_layer]),gpu::bind(1,&self.ln_f_g.data),
                gpu::bind(2,&self.ln_f_b.data),gpu::bind(3,&self.ln_f_out),gpu::bind(4,&ub)]);
            gpu::pool_release(pool,vec![ub]);
        }
        // head
        gemm_fwd_call(&self.device,&self.queue,&self.gemm_fwd,pool,
            &self.ln_f_out,&self.head_w.data,Some(&self.head_b.data),&self.logits_buf,rows,c,v);
        // CE
        {
            let dims=[rows as u32,v as u32,0,0];
            let ub=gpu::pool_uniform_u32(pool,&self.device,&self.queue,&dims);
            gpu::dispatch(&self.device,&self.queue,&self.ce_fwd,[rows as u32,1],&[
                gpu::bind(0,&self.logits_buf),gpu::bind(1,&self.target_ids_buf),
                gpu::bind(2,&self.probs),gpu::bind(3,&self.loss_buf),gpu::bind(4,&ub)]);
            gpu::pool_release(pool,vec![ub]);
        }
        let loss_vals=gpu::readback_f32(&self.device,&self.queue,&self.loss_buf,rows);
        let loss:f32={ let s:f64=loss_vals.iter().map(|&x|x as f64).sum(); (s/rows as f64) as f32 };
        loss
    }

    pub fn train_step(&mut self, ids:&[usize], targets:&[usize], lr:f32, wd:f32, clip:f32) -> f32 {
        let n_layer=self.n_layer; let c=self.c; let _t=self.t; let _b=self.b; let rows=self.rows; let v=self.v;
        let ids_u: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
        let tgt_u: Vec<u32> = targets.iter().map(|&x| x as u32).collect();
        self.token_ids_buf=gpu::upload_u32_storage(&self.device,&self.queue,&ids_u);
        self.target_ids_buf=gpu::upload_u32_storage(&self.device,&self.queue,&tgt_u);

        // zero grads
        let z=|buf:&wgpu::Buffer,n:usize,dev:&Self| { zero_duplex(&dev.fill, &dev.device,&dev.queue,&self.pool,buf,n); };
        // explicit:
        zero_duplex(&self.fill,&self.device,&self.queue,&self.pool,&self.tok_emb.grad,self.tok_emb.n);
        zero_duplex(&self.fill,&self.device,&self.queue,&self.pool,&self.pos_emb.grad,self.pos_emb.n);
        for i in 0..n_layer { let bp=&self.blocks[i];
            zero_duplex(&self.fill,&self.device,&self.queue,&self.pool,&bp.ln1_g.grad,bp.ln1_g.n);
            zero_duplex(&self.fill,&self.device,&self.queue,&self.pool,&bp.ln1_b.grad,bp.ln1_b.n);
            zero_duplex(&self.fill,&self.device,&self.queue,&self.pool,&bp.cq_w.grad,bp.cq_w.n);
            zero_duplex(&self.fill,&self.device,&self.queue,&self.pool,&bp.cq_b.grad,bp.cq_b.n);
            zero_duplex(&self.fill,&self.device,&self.queue,&self.pool,&bp.proj_w.grad,bp.proj_w.n);
            zero_duplex(&self.fill,&self.device,&self.queue,&self.pool,&bp.proj_b.grad,bp.proj_b.n);
            zero_duplex(&self.fill,&self.device,&self.queue,&self.pool,&bp.ln2_g.grad,bp.ln2_g.n);
            zero_duplex(&self.fill,&self.device,&self.queue,&self.pool,&bp.ln2_b.grad,bp.ln2_b.n);
            zero_duplex(&self.fill,&self.device,&self.queue,&self.pool,&bp.fc1_w.grad,bp.fc1_w.n);
            zero_duplex(&self.fill,&self.device,&self.queue,&self.pool,&bp.fc1_b.grad,bp.fc1_b.n);
            zero_duplex(&self.fill,&self.device,&self.queue,&self.pool,&bp.fc2_w.grad,bp.fc2_w.n);
            zero_duplex(&self.fill,&self.device,&self.queue,&self.pool,&bp.fc2_b.grad,bp.fc2_b.n);
        }
        zero_duplex(&self.fill,&self.device,&self.queue,&self.pool,&self.ln_f_g.grad,self.ln_f_g.n);
        zero_duplex(&self.fill,&self.device,&self.queue,&self.pool,&self.ln_f_b.grad,self.ln_f_b.n);
        zero_duplex(&self.fill,&self.device,&self.queue,&self.pool,&self.head_w.grad,self.head_w.n);
        zero_duplex(&self.fill,&self.device,&self.queue,&self.pool,&self.head_b.grad,self.head_b.n);
        let _=z;

        let loss = self.forward_pass(ids, targets);
        let pool=&self.pool;
        // BACKWARD
        // CE bwd
        {
            let dims=[rows as u32,v as u32,0,0];
            let ub=gpu::pool_uniform_u32(pool,&self.device,&self.queue,&dims);
            gpu::dispatch(&self.device,&self.queue,&self.ce_bwd,[gpu::ceil_div(rows*v,256) as u32,1],&[
                gpu::bind(0,&self.probs),gpu::bind(1,&self.target_ids_buf),
                gpu::bind(2,&self.logits_buf),gpu::bind(3,&ub)]);
            gpu::pool_release(pool,vec![ub]);
        }
        // head
        gemm_dwt_call(&self.device,&self.queue,&self.gemm_dwt,pool,
            &self.ln_f_out,&self.logits_buf,&self.head_w.grad,rows,c,v);
        reduce_cols_call(&self.device,&self.queue,&self.reduce,pool,&self.logits_buf,&self.head_b.grad,rows,v);
        // dx head -> tmp_rc
        {
            let wt=self.head_w.data_t.as_ref().unwrap();
            gemm_fwd_call(&self.device,&self.queue,&self.gemm_fwd,pool,
                &self.logits_buf,wt,None,&self.tmp_rc,rows,v,c);
        }
        // ln_f bwd
        let dgp_buf=duplex(&self.device,rows*c);
        let dbp_buf=duplex(&self.device,rows*c);
        let dx_scratch=duplex(&self.device,rows*c);
        {
            let ub=gpu::pool_uniform_u32(pool,&self.device,&self.queue,&[rows as u32,c as u32,LN_EPS.to_bits(),0]);
            gpu::dispatch(&self.device,&self.queue,&self.ln_bwd,[rows as u32,1],&[
                gpu::bind(0,&self.x_stream[n_layer]), gpu::bind(1,&self.tmp_rc),
                gpu::bind(2,&self.ln_f_g.data), gpu::bind(3,&dx_scratch),
                gpu::bind(4,&dgp_buf), gpu::bind(5,&dbp_buf), gpu::bind(6,&ub)]);
            gpu::pool_release(pool,vec![ub]);
        }
        reduce_cols_call(&self.device,&self.queue,&self.reduce,pool,&dgp_buf,&self.ln_f_g.grad,rows,c);
        reduce_cols_call(&self.device,&self.queue,&self.reduce,pool,&dbp_buf,&self.ln_f_b.grad,rows,c);
        copy_buf(&self.device,&self.queue,&dx_scratch,&self.tmp_rc,rows*c*4);

        // per block backward
        for li in (0..n_layer).rev() {
            let i=li;
            let bp=&self.blocks[i];

            // recalc gelu output into tmp_4rc
            {
                let n=rows*4*c;
                gpu::dispatch(&self.device,&self.queue,&self.gelu_fwd,[gpu::ceil_div(n,256) as u32,1],&[
                    gpu::bind(0,&self.blk_fc1out[i]), gpu::bind(1,&self.tmp_4rc)]);
            }
            // fc2 backward
            gemm_dwt_call(&self.device,&self.queue,&self.gemm_dwt,pool,
                &self.tmp_4rc,&self.tmp_rc,&bp.fc2_w.grad,rows,4*c,c);
            reduce_cols_call(&self.device,&self.queue,&self.reduce,pool,&self.tmp_rc,&bp.fc2_b.grad,rows,c);
            // dx gelu
            {
                let wt=bp.fc2_w.data_t.as_ref().unwrap();
                gemm_fwd_call(&self.device,&self.queue,&self.gemm_fwd,pool,
                    &self.tmp_rc,wt,None,&self.tmp_4rc,rows,c,4*c);
            }
            // gelu_bwd
            let grad_fc1=duplex(&self.device,rows*4*c);
            {
                let n=rows*4*c;
                gpu::dispatch(&self.device,&self.queue,&self.gelu_bwd,[gpu::ceil_div(n,256) as u32,1],&[
                    gpu::bind(0,&self.blk_fc1out[i]), gpu::bind(1,&self.tmp_4rc),
                    gpu::bind(2,&grad_fc1)]);
            }
            // fc1 backward
            gemm_dwt_call(&self.device,&self.queue,&self.gemm_dwt,pool,
                &self.blk_ln2out[i],&grad_fc1,&bp.fc1_w.grad,rows,c,4*c);
            reduce_cols_call(&self.device,&self.queue,&self.reduce,pool,&grad_fc1,&bp.fc1_b.grad,rows,4*c);
            // dx pre (rows,c) = grad_fc1 @ fc1_wT -> blk_x1? use local
            let dx_pre=duplex(&self.device,rows*c);
            {
                let wt=bp.fc1_w.data_t.as_ref().unwrap();
                gemm_fwd_call(&self.device,&self.queue,&self.gemm_fwd,pool,
                    &grad_fc1,wt,None,&dx_pre,rows,4*c,c);
            }
            // ln2 backward, cached x=blk_x1[i], gy=dx_pre
            let dgp2=duplex(&self.device,rows*c);
            let dbp2=duplex(&self.device,rows*c);
            let dx_attnin=duplex(&self.device,rows*c);
            {
                let ub=gpu::pool_uniform_u32(pool,&self.device,&self.queue,&[rows as u32,c as u32,LN_EPS.to_bits(),0]);
                gpu::dispatch(&self.device,&self.queue,&self.ln_bwd,[rows as u32,1],&[
                    gpu::bind(0,&self.blk_x1[i]), gpu::bind(1,&dx_pre),
                    gpu::bind(2,&bp.ln2_g.data), gpu::bind(3,&dx_attnin),
                    gpu::bind(4,&dgp2), gpu::bind(5,&dbp2), gpu::bind(6,&ub)]);
                gpu::pool_release(pool,vec![ub]);
            }
            reduce_cols_call(&self.device,&self.queue,&self.reduce,pool,&dgp2,&bp.ln2_g.grad,rows,c);
            reduce_cols_call(&self.device,&self.queue,&self.reduce,pool,&dbp2,&bp.ln2_b.grad,rows,c);
            // residual: dx_attnin += incoming tmp_rc
            {
                let n=rows*c;
                let ub=make_uniform_u32(&self.device,&self.queue,&[n as u32,0,0,0]);
                gpu::dispatch(&self.device,&self.queue,&self.add,[gpu::ceil_div(n,256) as u32,1],&[
                    gpu::bind(0,&dx_attnin), gpu::bind(1,&self.tmp_rc),
                    gpu::bind(2,&self.tmp_4rc), gpu::bind(3,&ub)]);
                copy_buf(&self.device,&self.queue,&self.tmp_4rc,&dx_attnin,n*4);
            }
            // attention backward: c_proj
            // dy for c_proj = dx_attnin
            gemm_dwt_call(&self.device,&self.queue,&self.gemm_dwt,pool,
                &self.blk_attnout[i],&dx_attnin,&bp.proj_w.grad,rows,c,c);
            reduce_cols_call(&self.device,&self.queue,&self.reduce,pool,&dx_attnin,&bp.proj_b.grad,rows,c);
            // dconcat (rows,c) = dx_attnin @ proj_wT -> tmp_rc
            {
                let wt=bp.proj_w.data_t.as_ref().unwrap();
                gemm_fwd_call(&self.device,&self.queue,&self.gemm_fwd,pool,
                    &dx_attnin, wt, None, &self.tmp_rc, rows, c, c);
            }
            // attn core backward
            let ab=&self.attns[i];
            gpu::dispatch(&self.device,&self.queue,&self.split_dh,[gpu::ceil_div(rows*c,256) as u32,1],&[
                gpu::bind(0,&self.tmp_rc), gpu::bind(1,&ab.dh), gpu::bind(2,&ab.attn_dims)]);
            gpu::dispatch(&self.device,&self.queue,&self.attn_bwd_ds,[(self.b*self.nh) as u32,1],&[
                gpu::bind(0,&ab.dh), gpu::bind(1,&ab.v), gpu::bind(2,&ab.p),
                gpu::bind(3,&ab.ds), gpu::bind(4,&ab.dst), gpu::bind(5,&ab.attn_dims)]);
            let nsc=rows*c;
            let gem=|x:&wgpu::Buffer,w:&wgpu::Buffer,y:&wgpu::Buffer,dev:&wgpu::Device,queue:&wgpu::Queue,pipe:&wgpu::ComputePipeline,_pool:&RefCell<gpu::Pool>,dims:&wgpu::Buffer|{
                gpu::dispatch(dev,queue,pipe,[gpu::ceil_div(nsc,256) as u32,1],&[
                    gpu::bind(0,x),gpu::bind(1,w),gpu::bind(2,y),gpu::bind(3,dims)]);
            };
            gem(&ab.ds,&ab.k,&ab.dq,&self.device,&self.queue,&self.batched_gemm,pool,&ab.bg_dims);
            gem(&ab.dst,&ab.q,&ab.dk,&self.device,&self.queue,&self.batched_gemm,pool,&ab.bg_dims);
            gem(&ab.pt,&ab.dh,&ab.dv,&self.device,&self.queue,&self.batched_gemm,pool,&ab.bg_dims);
            let dqkv_out=&self.tmp_4rc; // sufficient
            gpu::dispatch(&self.device,&self.queue,&self.scatter,[gpu::ceil_div(nsc,256) as u32,1],&[
                gpu::bind(0,&ab.dq),gpu::bind(1,&ab.dk),gpu::bind(2,&ab.dv),
                gpu::bind(3,dqkv_out),gpu::bind(4,&ab.attn_dims)]);
            // cq backward
            gemm_dwt_call(&self.device,&self.queue,&self.gemm_dwt,pool,
                &self.blk_ln1out[i],dqkv_out,&bp.cq_w.grad,rows,c,3*c);
            reduce_cols_call(&self.device,&self.queue,&self.reduce,pool,dqkv_out,&bp.cq_b.grad,rows,3*c);
            // dln1 (rows,c) = dqkv (rows,3c) @ cq_wT
            let dln1=duplex(&self.device,rows*c);
            {
                let wt=bp.cq_w.data_t.as_ref().unwrap();
                gemm_fwd_call(&self.device,&self.queue,&self.gemm_fwd,pool,
                    dqkv_out,wt,None,&dln1,rows,3*c,c);
            }
            // ln1 backward cached x=x_stream[i], gy=dln1
            let dgp1=duplex(&self.device,rows*c);
            let dbp1=duplex(&self.device,rows*c);
            let dx_in=duplex(&self.device,rows*c);
            {
                let ub=gpu::pool_uniform_u32(pool,&self.device,&self.queue,&[rows as u32,c as u32,LN_EPS.to_bits(),0]);
                gpu::dispatch(&self.device,&self.queue,&self.ln_bwd,[rows as u32,1],&[
                    gpu::bind(0,&self.x_stream[i]), gpu::bind(1,&dln1),
                    gpu::bind(2,&bp.ln1_g.data), gpu::bind(3,&dx_in),
                    gpu::bind(4,&dgp1), gpu::bind(5,&dbp1), gpu::bind(6,&ub)]);
                gpu::pool_release(pool,vec![ub]);
            }
            reduce_cols_call(&self.device,&self.queue,&self.reduce,pool,&dgp1,&bp.ln1_g.grad,rows,c);
            reduce_cols_call(&self.device,&self.queue,&self.reduce,pool,&dbp1,&bp.ln1_b.grad,rows,c);
            // residual dx_in += dx_attnin (incoming)
            {
                let n=rows*c;
                let ub=make_uniform_u32(&self.device,&self.queue,&[n as u32,0,0,0]);
                gpu::dispatch(&self.device,&self.queue,&self.add,[gpu::ceil_div(n,256) as u32,1],&[
                    gpu::bind(0,&dx_in), gpu::bind(1,&dx_attnin),
                    gpu::bind(2,&self.tmp_4rc), gpu::bind(3,&ub)]);
                copy_buf(&self.device,&self.queue,&self.tmp_4rc,&dx_in,n*4);
            }
            // next incoming gradient
            copy_buf(&self.device,&self.queue,&dx_in,&self.tmp_rc,rows*c*4);
        }

        // embedding backward
        {
            let dims=[(v*c) as u32, rows as u32, c as u32, 0u32];
            let ub=make_uniform_u32(&self.device,&self.queue,&dims);
            gpu::dispatch(&self.device,&self.queue,&self.emb_bwd,[gpu::ceil_div(v*c,256) as u32,1],&[
                gpu::bind(0,&self.tmp_rc), gpu::bind(1,&self.token_ids_buf),
                gpu::bind(2,&self.tok_emb.grad), gpu::bind(3,&ub)]);
        }
        {
            let dims=[(self.t*c) as u32, rows as u32, c as u32, 0u32];
            let ub=make_uniform_u32(&self.device,&self.queue,&dims);
            gpu::dispatch(&self.device,&self.queue,&self.emb_bwd,[gpu::ceil_div(self.t*c,256) as u32,1],&[
                gpu::bind(0,&self.tmp_rc), gpu::bind(1,&self.pos_ids_buf),
                gpu::bind(2,&self.pos_emb.grad), gpu::bind(3,&ub)]);
        }

        // clip
        let mut total_sq=0.0f64;
        let collect=|buf:&wgpu::Buffer,n:usize,eng:&Self| sqsum_tensor(&eng.sqnorm,&eng.device,&eng.queue,pool,buf,n);
        total_sq += collect(&self.tok_emb.grad,self.tok_emb.n,self);
        total_sq += collect(&self.pos_emb.grad,self.pos_emb.n,self);
        for bp in &self.blocks {
            total_sq += collect(&bp.ln1_g.grad,bp.ln1_g.n,self);
            total_sq += collect(&bp.ln1_b.grad,bp.ln1_b.n,self);
            total_sq += collect(&bp.cq_w.grad,bp.cq_w.n,self);
            total_sq += collect(&bp.cq_b.grad,bp.cq_b.n,self);
            total_sq += collect(&bp.proj_w.grad,bp.proj_w.n,self);
            total_sq += collect(&bp.proj_b.grad,bp.proj_b.n,self);
            total_sq += collect(&bp.ln2_g.grad,bp.ln2_g.n,self);
            total_sq += collect(&bp.ln2_b.grad,bp.ln2_b.n,self);
            total_sq += collect(&bp.fc1_w.grad,bp.fc1_w.n,self);
            total_sq += collect(&bp.fc1_b.grad,bp.fc1_b.n,self);
            total_sq += collect(&bp.fc2_w.grad,bp.fc2_w.n,self);
            total_sq += collect(&bp.fc2_b.grad,bp.fc2_b.n,self);
        }
        total_sq += collect(&self.ln_f_g.grad,self.ln_f_g.n,self);
        total_sq += collect(&self.ln_f_b.grad,self.ln_f_b.n,self);
        total_sq += collect(&self.head_w.grad,self.head_w.n,self);
        total_sq += collect(&self.head_b.grad,self.head_b.n,self);

        let norm=(total_sq.sqrt()) as f32;
        if norm > clip && norm > 0.0 {
            let scale=clip/(norm+1e-6);
            let apply=|dev:&wgpu::Device,queue:&wgpu::Queue,gs:&wgpu::ComputePipeline,pool:&RefCell<gpu::Pool>,buf:&wgpu::Buffer,n:usize,scale:f32|{
                if scale==1.0||n==0 {return;}
                let dims=[scale.to_bits(), n as u32,0,0];
                let ub=gpu::pool_uniform_u32(pool,dev,queue,&dims);
                gpu::dispatch(dev,queue,gs,[gpu::ceil_div(n,256) as u32,1],&[gpu::bind(0,buf),gpu::bind(1,&ub)]);
                gpu::pool_release(pool,vec![ub]);
            };
            apply(&self.device,&self.queue,&self.grad_scale,pool,&self.tok_emb.grad,self.tok_emb.n,scale);
            apply(&self.device,&self.queue,&self.grad_scale,pool,&self.pos_emb.grad,self.pos_emb.n,scale);
            for i in 0..n_layer { let bp=&self.blocks[i];
                apply(&self.device,&self.queue,&self.grad_scale,pool,&bp.ln1_g.grad,bp.ln1_g.n,scale);
                apply(&self.device,&self.queue,&self.grad_scale,pool,&bp.ln1_b.grad,bp.ln1_b.n,scale);
                apply(&self.device,&self.queue,&self.grad_scale,pool,&bp.cq_w.grad,bp.cq_w.n,scale);
                apply(&self.device,&self.queue,&self.grad_scale,pool,&bp.cq_b.grad,bp.cq_b.n,scale);
                apply(&self.device,&self.queue,&self.grad_scale,pool,&bp.proj_w.grad,bp.proj_w.n,scale);
                apply(&self.device,&self.queue,&self.grad_scale,pool,&bp.proj_b.grad,bp.proj_b.n,scale);
                apply(&self.device,&self.queue,&self.grad_scale,pool,&bp.ln2_g.grad,bp.ln2_g.n,scale);
                apply(&self.device,&self.queue,&self.grad_scale,pool,&bp.ln2_b.grad,bp.ln2_b.n,scale);
                apply(&self.device,&self.queue,&self.grad_scale,pool,&bp.fc1_w.grad,bp.fc1_w.n,scale);
                apply(&self.device,&self.queue,&self.grad_scale,pool,&bp.fc1_b.grad,bp.fc1_b.n,scale);
                apply(&self.device,&self.queue,&self.grad_scale,pool,&bp.fc2_w.grad,bp.fc2_w.n,scale);
                apply(&self.device,&self.queue,&self.grad_scale,pool,&bp.fc2_b.grad,bp.fc2_b.n,scale);
            }
            apply(&self.device,&self.queue,&self.grad_scale,pool,&self.ln_f_g.grad,self.ln_f_g.n,scale);
            apply(&self.device,&self.queue,&self.grad_scale,pool,&self.ln_f_b.grad,self.ln_f_b.n,scale);
            apply(&self.device,&self.queue,&self.grad_scale,pool,&self.head_w.grad,self.head_w.n,scale);
            apply(&self.device,&self.queue,&self.grad_scale,pool,&self.head_b.grad,self.head_b.n,scale);
        }

        // adamw
        self.step += 1;
        let step=self.step;
        let b1=0.9f32; let b2=0.95f32; let eps=1e-8f32;
        // adamw tok/pos
        for param in [&mut self.tok_emb, &mut self.pos_emb] {
            let dims=[param.n as u32, lr.to_bits(), b1.to_bits(), b2.to_bits(),
                (1.0-b1.powf(step as f32)).to_bits(),
                (1.0-b2.powf(step as f32)).to_bits(),
                eps.to_bits(),
                if param.decay { wd } else { 0.0 }.to_bits()];
            let ub=gpu::pool_uniform_u32(pool,&self.device,&self.queue,&dims);
            gpu::dispatch(&self.device,&self.queue,&self.adamw,[gpu::ceil_div(param.n,256) as u32,1],&[
                gpu::bind(0,&param.grad),gpu::bind(1,&param.data),
                gpu::bind(2,&param.m),gpu::bind(3,&param.v),gpu::bind(4,&ub)]);
            gpu::pool_release(pool,vec![ub]);
            if let Some(dt)=&param.data_t {
                let ub=make_uniform_u32(&self.device,&self.queue,&[param.shape_in as u32,param.shape_out as u32,0,0]);
                gpu::dispatch(&self.device,&self.queue,&self.transpose_pipe,[gpu::ceil_div(param.n,256) as u32,1],&[
                    gpu::bind(0,&param.data), gpu::bind(1,dt), gpu::bind(2,&ub)]);
            }
        }
        // blocks
        for i in 0..n_layer {
            let bp=&mut self.blocks[i];
            for p in [&mut bp.ln1_g,&mut bp.ln1_b,&mut bp.cq_w,&mut bp.cq_b,&mut bp.proj_w,&mut bp.proj_b,&mut bp.ln2_g,&mut bp.ln2_b,&mut bp.fc1_w,&mut bp.fc1_b,&mut bp.fc2_w,&mut bp.fc2_b] {
                let dims=[p.n as u32, lr.to_bits(), b1.to_bits(), b2.to_bits(),
                    (1.0-b1.powf(step as f32)).to_bits(),
                    (1.0-b2.powf(step as f32)).to_bits(),
                    eps.to_bits(),
                    if p.decay { wd } else { 0.0 }.to_bits()];
                let ub=gpu::pool_uniform_u32(pool,&self.device,&self.queue,&dims);
                gpu::dispatch(&self.device,&self.queue,&self.adamw,[gpu::ceil_div(p.n,256) as u32,1],&[
                    gpu::bind(0,&p.grad),gpu::bind(1,&p.data),
                    gpu::bind(2,&p.m),gpu::bind(3,&p.v),gpu::bind(4,&ub)]);
                gpu::pool_release(pool,vec![ub]);
                if let Some(dt)=&p.data_t {
                    let ub=make_uniform_u32(&self.device,&self.queue,&[p.shape_in as u32,p.shape_out as u32,0,0]);
                    gpu::dispatch(&self.device,&self.queue,&self.transpose_pipe,[gpu::ceil_div(p.n,256) as u32,1],&[
                        gpu::bind(0,&p.data), gpu::bind(1,dt), gpu::bind(2,&ub)]);
                }
            }
        }
        // top
        for param in [&mut self.ln_f_g, &mut self.ln_f_b, &mut self.head_w, &mut self.head_b] {
            let dims=[param.n as u32, lr.to_bits(), b1.to_bits(), b2.to_bits(),
                (1.0-b1.powf(step as f32)).to_bits(),
                (1.0-b2.powf(step as f32)).to_bits(),
                eps.to_bits(),
                if param.decay { wd } else { 0.0 }.to_bits()];
            let ub=gpu::pool_uniform_u32(pool,&self.device,&self.queue,&dims);
            gpu::dispatch(&self.device,&self.queue,&self.adamw,[gpu::ceil_div(param.n,256) as u32,1],&[
                gpu::bind(0,&param.grad),gpu::bind(1,&param.data),
                gpu::bind(2,&param.m),gpu::bind(3,&param.v),gpu::bind(4,&ub)]);
            gpu::pool_release(pool,vec![ub]);
            if let Some(dt)=&param.data_t {
                let ub=make_uniform_u32(&self.device,&self.queue,&[param.shape_in as u32,param.shape_out as u32,0,0]);
                gpu::dispatch(&self.device,&self.queue,&self.transpose_pipe,[gpu::ceil_div(param.n,256) as u32,1],&[
                    gpu::bind(0,&param.data), gpu::bind(1,dt), gpu::bind(2,&ub)]);
            }
        }

        loss
    }

    pub fn to_gpt(&self) -> Gpt {
        // read params back host
        let mut rng=crate::rng::Rng::new(0);
        let mut gpt=Gpt::new(self.cfg, &mut rng);
        let p=GptParams {
            cfg: self.cfg,
            tok_emb: gpu::readback_f32(&self.device,&self.queue,&self.tok_emb.data,self.tok_emb.n),
            pos_emb: gpu::readback_f32(&self.device,&self.queue,&self.pos_emb.data,self.pos_emb.n),
            blocks: self.blocks.iter().map(|bp| crate::transformer::BlockParams {
                ln1_g: gpu::readback_f32(&self.device,&self.queue,&bp.ln1_g.data,bp.ln1_g.n),
                ln1_b: gpu::readback_f32(&self.device,&self.queue,&bp.ln1_b.data,bp.ln1_b.n),
                cq_w: gpu::readback_f32(&self.device,&self.queue,&bp.cq_w.data,bp.cq_w.n),
                cq_b: gpu::readback_f32(&self.device,&self.queue,&bp.cq_b.data,bp.cq_b.n),
                proj_w: gpu::readback_f32(&self.device,&self.queue,&bp.proj_w.data,bp.proj_w.n),
                proj_b: gpu::readback_f32(&self.device,&self.queue,&bp.proj_b.data,bp.proj_b.n),
                ln2_g: gpu::readback_f32(&self.device,&self.queue,&bp.ln2_g.data,bp.ln2_g.n),
                ln2_b: gpu::readback_f32(&self.device,&self.queue,&bp.ln2_b.data,bp.ln2_b.n),
                fc1_w: gpu::readback_f32(&self.device,&self.queue,&bp.fc1_w.data,bp.fc1_w.n),
                fc1_b: gpu::readback_f32(&self.device,&self.queue,&bp.fc1_b.data,bp.fc1_b.n),
                fc2_w: gpu::readback_f32(&self.device,&self.queue,&bp.fc2_w.data,bp.fc2_w.n),
                fc2_b: gpu::readback_f32(&self.device,&self.queue,&bp.fc2_b.data,bp.fc2_b.n),
            }).collect(),
            ln_f_g: gpu::readback_f32(&self.device,&self.queue,&self.ln_f_g.data,self.ln_f_g.n),
            ln_f_b: gpu::readback_f32(&self.device,&self.queue,&self.ln_f_b.data,self.ln_f_b.n),
            head_w: gpu::readback_f32(&self.device,&self.queue,&self.head_w.data,self.head_w.n),
            head_b: gpu::readback_f32(&self.device,&self.queue,&self.head_b.data,self.head_b.n),
        };
        gpt.import_params(&p);
        gpt
    }

    pub fn eval_loss(&mut self, ids:&[usize], targets:&[usize]) -> f32 {
        self.forward_pass(ids, targets)
    }
}

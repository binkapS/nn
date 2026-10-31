//! Multi-head causal self-attention with a fully hand-written backward pass.
//!
//! Forward per (batch, head):
//!   q_t = W_q x_t, k_t = W_k x_t, v_t = W_v x_t            (one fused QKV matmul)
//!   score_{t,s} = (q_t . k_s) / sqrt(d_head), s <= t        (causal mask)
//!   p_{t,s}   = softmax(score_{t,s})
//!   out_t     = Σ_s p_{t,s} v_s
//!   y_t       = W_o out_t
//!
//! Backward uses the well-known softmax+SDPA identities:
//!   ds_{t,s} = p_{t,s} * (ṗ_{t,s} - Σ_j ṗ_{t,j} p_{t,j}) / sqrt(d_head)
//!   with ṗ_{t,s} = (dout_t . v_s), and
//!   dq_t += Σ_s ds_{t,s} k_s ; dk_s += Σ_t ds_{t,s} q_t ; dv_s += Σ_t p_{t,s} dout_t.
//!
//! Parallelism: all heads of one batch strip `(bi, ·)` (disjoint slices of
//! `concat` / the fused-QKV gradient) run on one thread per batch element
//! via `std::thread::scope`; a single-batch call runs inline. Head scratch
//! buffers are flat `Vec<f32>`, grown on first use for a shape and reused
//! across steps (no per-step allocation churn).

use crate::modules::Linear;

use crate::rng::Rng;

pub struct Attention {
    n_head: usize,
    d_head: usize,

    pub(crate) c_qkv: Linear,  // (C, 3C): output columns ordered [q | k | v]
    pub(crate) c_proj: Linear, // (C, C)

    // Scratch, flat over the (batch, head) pairs (`bhh = bi * n_head + h`):
    q: Vec<f32>,        // (bh, T, d_head) queries
    k: Vec<f32>,        // (bh, T, d_head) keys
    v: Vec<f32>,        // (bh, T, d_head) values
    probs: Vec<f32>,    // (bh, T, T) row-stochastic over positions <= t
    qkv_flat: Vec<f32>, // (B*T, 3C) output of c_qkv before head-splitting
    concat: Vec<f32>,   // (B*T, C)  head outputs concatenated, input to c_proj
}

impl Attention {
    pub fn new(n_head: usize, embed_dim: usize, rng: &mut Rng) -> Self {
        assert_eq!(
            embed_dim % n_head,
            0,
            "embed_dim must split evenly among heads"
        );
        Self {
            n_head,
            d_head: embed_dim / n_head,
            c_qkv: Linear::new(embed_dim, 3 * embed_dim, 0.02, rng),
            c_proj: Linear::new(embed_dim, embed_dim, 0.02, rng),
            q: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            probs: Vec::new(),
            qkv_flat: Vec::new(),
            concat: Vec::new(),
        }
    }

    pub fn forward(&mut self, x: &[f32], b: usize, t: usize, c: usize) -> Vec<f32> {
        let bh = b * self.n_head;
        let d = self.d_head;

        self.qkv_flat = self.c_qkv.forward(x, b * t, c, 3 * c);

        // Grow (never shrink) the scratch buffers once per shape change.
        let q_len = bh * t * d;
        self.q.resize(q_len, 0.0);
        self.k.resize(q_len, 0.0);
        self.v.resize(q_len, 0.0);
        self.probs.resize(bh * t * t, 0.0);
        self.concat.resize(b * t * c, 0.0);

        let Attention {
            q,
            k,
            v,
            probs,
            concat,
            qkv_flat,
            ..
        } = self;
        let qkv_flat: &[f32] = qkv_flat;
        let per_head = t * d;
        let per_probs = t * t;
        let n_head = self.n_head;

        // Disjoint per-batch strips handed out via split_at_mut, so each
        // spawned worker owns its own slice of every scratch field.
        let mut q_rest: &mut [f32] = q;
        let mut k_rest: &mut [f32] = k;
        let mut v_rest: &mut [f32] = v;
        let mut probs_rest: &mut [f32] = probs;
        let mut concat_rest: &mut [f32] = concat;
        std::thread::scope(|s| {
            for bi in 0..b {
                let (qs, q_tail) = q_rest.split_at_mut(n_head * per_head);
                q_rest = q_tail;
                let (ks, k_tail) = k_rest.split_at_mut(n_head * per_head);
                k_rest = k_tail;
                let (vs, v_tail) = v_rest.split_at_mut(n_head * per_head);
                v_rest = v_tail;
                let (ps, p_tail) = probs_rest.split_at_mut(n_head * per_probs);
                probs_rest = p_tail;
                let (cs, c_tail) = concat_rest.split_at_mut(t * c);
                concat_rest = c_tail;
                if b == 1 {
                    strip_fwd(qkv_flat, qs, ks, vs, ps, cs, bi, t, c, d, n_head);
                } else {
                    s.spawn(move || strip_fwd(qkv_flat, qs, ks, vs, ps, cs, bi, t, c, d, n_head));
                }
            }
        });

        self.c_proj.forward(&self.concat, b * t, c, c)
    }

    pub fn backward(&mut self, dy: &[f32], b: usize, t: usize, c: usize) -> Vec<f32> {
        let d = self.d_head;

        // Gradients w.r.t. the concatenated head outputs (input of c_proj).
        let dconcat = self.c_proj.backward(dy, b * t, c, c);

        // Fused QKV gradient; each batch strip writes disjoint rows.
        let mut dqkv_flat = vec![0.0f32; b * t * 3 * c];

        {
            let Attention { q, k, v, probs, .. } = self;
            let per_head = t * d;
            let per_probs = t * t;
            let n_head = self.n_head;
            let mut q_rest: &[f32] = q;
            let mut k_rest: &[f32] = k;
            let mut v_rest: &[f32] = v;
            let mut probs_rest: &[f32] = probs;
            let mut dq_rest: &mut [f32] = &mut dqkv_flat;
            std::thread::scope(|s| {
                for bi in 0..b {
                    let (qs, q_tail) = q_rest.split_at(n_head * per_head);
                    q_rest = q_tail;
                    let (ks, k_tail) = k_rest.split_at(n_head * per_head);
                    k_rest = k_tail;
                    let (vs, v_tail) = v_rest.split_at(n_head * per_head);
                    v_rest = v_tail;
                    let (ps, p_tail) = probs_rest.split_at(n_head * per_probs);
                    probs_rest = p_tail;
                    let (dq_rows, dq_tail) =
                        dq_rest.split_at_mut(t * 3 * c);
                    dq_rest = dq_tail;
                    let dcs: &[f32] = &dconcat[bi * t * c..(bi + 1) * t * c];
                    if b == 1 {
                        strip_bwd(dcs, qs, ks, vs, ps, dq_rows, t, c, d, n_head);
                    } else {
                        s.spawn(move || strip_bwd(dcs, qs, ks, vs, ps, dq_rows, t, c, d, n_head));
                    }
                }
            });
        }

        self.c_qkv.backward(&dqkv_flat, b * t, c, 3 * c)
    }

    pub fn zero_grad(&mut self) {
        self.c_qkv.zero_grad();
        self.c_proj.zero_grad();
    }

    pub fn sq_sum(&self) -> f64 {
        self.c_qkv.sq_sum() + self.c_proj.sq_sum()
    }

    pub fn scale_grads(&mut self, s: f32) {
        self.c_qkv.scale_grads(s);
        self.c_proj.scale_grads(s);
    }

    pub fn adam_step(&mut self, lr: f32, wd: f32) {
        self.c_qkv.adam_step(lr, wd);
        self.c_proj.adam_step(lr, wd);
    }

    pub fn param_count(&self) -> usize {
        self.c_qkv.param_count() + self.c_proj.param_count()
    }
}

/// Forward of one batch strip: runs every head of batch row `bi` in order.
/// All scratch slices (`q…concat`) are this strip's exclusive slices.
#[allow(clippy::too_many_arguments)]
fn strip_fwd(
    qkv_flat: &[f32],
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    probs: &mut [f32],
    concat: &mut [f32],
    bi: usize,
    t: usize,
    c: usize,
    d: usize,
    n_head: usize,
) {
    for h in 0..n_head {
        let qh = &mut q[h * t * d..(h + 1) * t * d];
        let kh = &mut k[h * t * d..(h + 1) * t * d];
        let vh = &mut v[h * t * d..(h + 1) * t * d];
        let ph = &mut probs[h * t * t..(h + 1) * t * t];
        head_fwd(qkv_flat, qh, kh, vh, ph, concat, bi, h, t, c, d);
    }
}

/// Forward of one head: un-fuse q/k/v rows, causal softmax scores, weighted
/// value sum written into the head's `concat` columns for rows `bi*t..`.
///
/// `concat` column slots are written with `=` on the first accumulated term
/// (`s == 0`) so no zero-fill is needed before accumulation.
#[allow(clippy::too_many_arguments)]
fn head_fwd(
    qkv_flat: &[f32],
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    probs: &mut [f32],
    concat: &mut [f32],
    bi: usize,
    h: usize,
    t: usize,
    c: usize,
    d: usize,
) {
    // Copy this head's q, k, v rows out of the fused QKV output.
    for tt in 0..t {
        let src = (bi * t + tt) * 3 * c;
        for j in 0..d {
            q[tt * d + j] = qkv_flat[src + h * d + j];
            k[tt * d + j] = qkv_flat[src + c + h * d + j];
            v[tt * d + j] = qkv_flat[src + 2 * c + h * d + j];
        }
    }

    // Masked softmax over causal scores, row by row.
    let scale = 1.0 / (d as f32).sqrt();
    let (q, k) = (&*q, &*k);
    for i in 0..t {
        // scores s <= i; max is only over unmasked positions.
        let (used, _tail) = probs[i * t..(i + 1) * t].split_at_mut(i + 1);
        let mut mx = f32::NEG_INFINITY;
        for (s, ps) in used.iter_mut().enumerate() {
            let a: f32 = q[i * d..(i + 1) * d]
                .iter()
                .zip(&k[s * d..(s + 1) * d])
                .map(|(&qv, &kv)| qv * kv)
                .sum::<f32>()
                * scale;
            *ps = a;
            if a > mx {
                mx = a;
            }
        }
        let mut z = 0.0f32;
        for ps in used.iter_mut() {
            let e = (*ps - mx).exp();
            *ps = e;
            z += e;
        }
        for ps in used.iter_mut() {
            *ps /= z;
        }
    }

    // Weighted sum over values into the head's `concat` columns. Rows
    // bi*t..(bi+1)*t map to local slots 0..t of the strip slice; the column
    // offset of this head is	h*d.
    let dst_col = h * d;
    let v = &*v;
    for i in 0..t {
        let dst = i * c + dst_col;
        let p = &probs[i * t..i * t + t];
        for s in 0..=i {
            let w = p[s];
            let acc = &mut concat[dst..dst + d];
            let vs = &v[s * d..(s + 1) * d];
            if s == 0 {
                for (o, &vv) in acc.iter_mut().zip(vs) {
                    *o = w * vv;
                }
            } else {
                for (o, &vv) in acc.iter_mut().zip(vs) {
                    *o += w * vv;
                }
            }
        }
    }
}

/// Backward of one batch strip: every head's dq/dk/dv contributions land in
/// this strip's rows of the fused `dqkv_rows` gradient.
#[allow(clippy::too_many_arguments)]
fn strip_bwd(
    dconcat: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    probs: &[f32],
    dqkv_rows: &mut [f32],
    t: usize,
    c: usize,
    d: usize,
    n_head: usize,
) {
    for h in 0..n_head {
        let qh = &q[h * t * d..(h + 1) * t * d];
        let kh = &k[h * t * d..(h + 1) * t * d];
        let vh = &v[h * t * d..(h + 1) * t * d];
        let ph = &probs[h * t * t..(h + 1) * t * t];
        head_bwd(dconcat, qh, kh, vh, ph, dqkv_rows, h, t, c, d);
    }
}

/// Backward of one head: computes dq/dk/dv contributions of the strip's rows
/// and writes them into that strip's rows of the fused `dqkv_rows` gradient
/// (its column range h*d is exclusive to this head).
#[allow(clippy::too_many_arguments)]
fn head_bwd(
    dconcat: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    probs: &[f32],
    dqkv_rows: &mut [f32],
    h: usize,
    t: usize,
    c: usize,
    d: usize,
) {
    let scale = 1.0 / (d as f32).sqrt();

    // Head slice of row i of the strip (`dconcat` rows are strip-local):
    let out = |i: usize| &dconcat[i * c + h * d..i * c + (h + 1) * d];

    let mut dq = vec![0.0f32; t * d];
    let mut dk = vec![0.0f32; t * d];
    let mut dv = vec![0.0f32; t * d];
    let mut dp = vec![0.0f32; t * t];

    // dpi_{t,s} = (dout_t . v_s)
    for i in 0..t {
        let oi = out(i);
        for s in 0..=i {
            dp[i * t + s] = oi
                .iter()
                .zip(&v[s * d..(s + 1) * d])
                .map(|(&ov, &vv)| ov * vv)
                .sum();
        }
    }

    for i in 0..t {
        let oi = out(i);
        let mut dot = 0.0f32; // Σ_j dp_{t,j} p_{t,j}
        for s in 0..=i {
            dot += dp[i * t + s] * probs[i * t + s];
        }
        for s in 0..=i {
            let ds = probs[i * t + s] * (dp[i * t + s] - dot) * scale;
            for j in 0..d {
                dq[i * d + j] += ds * k[s * d + j];
                dk[s * d + j] += ds * q[i * d + j];
            }
            let pv = probs[i * t + s];
            for j in 0..d {
                dv[s * d + j] += pv * oi[j];
            }
        }
    }

    // Scatter into this strip's fused QKV gradient rows (columns h*d, c+h*d,
    // 2c+h*d are exclusive to this head).
    for tt in 0..t {
        let dst = tt * 3 * c;
        for j in 0..d {
            dqkv_rows[dst + h * d + j] = dq[tt * d + j];
            dqkv_rows[dst + c + h * d + j] = dk[tt * d + j];
            dqkv_rows[dst + 2 * c + h * d + j] = dv[tt * d + j];
        }
    }
}

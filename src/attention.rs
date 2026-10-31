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

use crate::modules::Linear;

use crate::rng::Rng;

pub struct Attention {
    n_head: usize,
    d_head: usize,

    pub(crate) c_qkv: Linear,  // (C, 3C): output columns ordered [q | k | v]
    pub(crate) c_proj: Linear, // (C, C)

    // Scratch saved for backward, indexed per (b, head) pair -> (b * H + h):
    q: Vec<Vec<f32>>,     // (T, d_head) queries
    k: Vec<Vec<f32>>,     // (T, d_head) keys
    v: Vec<Vec<f32>>,     // (T, d_head) values
    probs: Vec<Vec<f32>>, // (T, T) row-stochastic over positions <= t
    qkv_flat: Vec<f32>,   // (B*T, 3C) output of c_qkv before head-splitting
    concat: Vec<f32>,     // (B*T, C)  head outputs concatenated, input to c_proj
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

        self.q = vec![vec![0.0; t * d]; bh];
        self.k = vec![vec![0.0; t * d]; bh];
        self.v = vec![vec![0.0; t * d]; bh];
        self.probs = vec![vec![0.0; t * t]; bh];
        self.concat = vec![0.0; b * t * c];

        for bi in 0..b {
            for h in 0..self.n_head {
                let bhh = bi * self.n_head + h;

                // Copy this head's q, k, v out of the fused QKV output.
                for tt in 0..t {
                    let src = (bi * t + tt) * 3 * c;
                    for j in 0..d {
                        self.q[bhh][tt * d + j] = self.qkv_flat[src + h * d + j];
                        self.k[bhh][tt * d + j] = self.qkv_flat[src + c + h * d + j];
                        self.v[bhh][tt * d + j] = self.qkv_flat[src + 2 * c + h * d + j];
                    }
                }

                // Masked softmax over causal scores, row by row.
                let scale = 1.0 / (d as f32).sqrt();
                let (q, k) = (&self.q[bhh], &self.k[bhh]);
                for i in 0..t {
                    // scores s <= i; max is only over unmasked positions.
                    let pr = &mut self.probs[bhh][i * t..i * t + t];
                    let mut mx = f32::NEG_INFINITY;
                    for s in 0..=i {
                        let a: f32 = q[i * d..(i + 1) * d]
                            .iter()
                            .zip(&k[s * d..(s + 1) * d])
                            .map(|(&qv, &kv)| qv * kv)
                            .sum();
                        let a = a * scale;
                        pr[s] = a;
                        if a > mx {
                            mx = a;
                        }
                    }
                    let mut z = 0.0f32;
                    for ps in pr.iter_mut() {
                        let e = (*ps - mx).exp();
                        *ps = e;
                        z += e;
                    }
                    for ps in pr.iter_mut() {
                        *ps /= z;
                    }
                }

                // Weighted sum over values, written directly into the concat slots.
                let v = &self.v[bhh];
                for i in 0..t {
                    let dst = (bi * t + i) * c + h * d;
                    let p = &self.probs[bhh][i * t..i * t + t];
                    for s in 0..=i {
                        let w = p[s];
                        for j in 0..d {
                            self.concat[dst + j] += w * v[s * d + j];
                        }
                    }
                }
            }
        }

        self.c_proj.forward(&self.concat, b * t, c, c)
    }

    pub fn backward(&mut self, dy: &[f32], b: usize, t: usize, c: usize) -> Vec<f32> {
        let bh = b * self.n_head;
        let d = self.d_head;

        // Gradients w.r.t. the concatenated head outputs (input of c_proj).
        let dconcat = self.c_proj.backward(dy, b * t, c, c);

        let mut dq = vec![vec![0.0f32; t * d]; bh];
        let mut dk = vec![vec![0.0f32; t * d]; bh];
        let mut dv = vec![vec![0.0f32; t * d]; bh];
        let scale = 1.0 / (d as f32).sqrt();

        for bi in 0..b {
            for h in 0..self.n_head {
                let bhh = bi * self.n_head + h;
                let p = &self.probs[bhh];
                let (q, k, v) = (&self.q[bhh], &self.k[bhh], &self.v[bhh]);

                // Head slice of row i of dconcat:
                let out =
                    |i: usize| &dconcat[(bi * t + i) * c + h * d..(bi * t + i) * c + (h + 1) * d];

                // dp_{t,s} = (dout_t . v_s)
                let mut dp = vec![0.0f32; t * t];
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
                    let mut dot = 0.0f32; // Σ_j ṗ_{t,j} p_{t,j}
                    for s in 0..=i {
                        dot += dp[i * t + s] * p[i * t + s];
                    }
                    for s in 0..=i {
                        let ds = p[i * t + s] * (dp[i * t + s] - dot) * scale;
                        for j in 0..d {
                            dq[bhh][i * d + j] += ds * k[s * d + j];
                            dk[bhh][s * d + j] += ds * q[i * d + j];
                        }
                        let pv = p[i * t + s];
                        for j in 0..d {
                            dv[bhh][s * d + j] += pv * oi[j];
                        }
                    }
                }
            }
        }

        // Reassemble the gradient of the fused QKV output in [q | k | v] order.
        let mut dqkv_flat = vec![0.0f32; b * t * 3 * c];
        for bi in 0..b {
            for h in 0..self.n_head {
                let bhh = bi * self.n_head + h;
                for tt in 0..t {
                    let dst = (bi * t + tt) * 3 * c;
                    for j in 0..d {
                        dqkv_flat[dst + h * d + j] = dq[bhh][tt * d + j];
                        dqkv_flat[dst + c + h * d + j] = dk[bhh][tt * d + j];
                        dqkv_flat[dst + 2 * c + h * d + j] = dv[bhh][tt * d + j];
                    }
                }
            }
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

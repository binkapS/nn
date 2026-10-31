//! Trainable modules. Every module owns its parameters, its gradients, and
//! its AdamW moments, so no global borrow graph is needed.

use crate::ops::{self, matmul_bwd, matmul_fwd};
use crate::rng::Rng;

/// Per-tensor AdamW implementation.
#[derive(Clone)]
pub struct AdamState {
    m: Vec<f32>,
    v: Vec<f32>,
    step: u32,
    b1: f32,
    b2: f32,
    eps: f32,
}

impl AdamState {
    pub fn new_like(params: &[f32]) -> Self {
        Self {
            m: vec![0.0; params.len()],
            v: vec![0.0; params.len()],
            step: 0,
            b1: 0.9,
            b2: 0.95,
            eps: 1e-8,
        }
    }

    /// Decoupled weight decay (AdamW): `p -= lr * ((m̂/(√v̂+ε)) + wd·p)`.
    pub fn step(&mut self, params: &mut [f32], grads: &[f32], lr: f32) {
        self.step += 1;
        let t = self.step as f32;
        let bc1 = 1.0 - self.b1.powf(t);
        let bc2 = 1.0 - self.b2.powf(t);
        for i in 0..params.len() {
            let g = grads[i];
            let m = &mut self.m[i];
            let v = &mut self.v[i];
            *m = self.b1 * *m + (1.0 - self.b1) * g;
            *v = self.b2 * *v + (1.0 - self.b2) * g * g;
            let m_hat = *m / bc1;
            let v_hat = *v / bc2;
            params[i] -= lr * (m_hat / (v_hat.sqrt() + self.eps));
        }
    }
}

/// Common bookkeeping every parameter tensor needs.
#[derive(Clone)]
pub(crate) struct Tunable {
    pub(crate) params: Vec<f32>,
    pub(crate) grads: Vec<f32>,
    decay: bool,
    adam: AdamState,
}

impl Tunable {
    fn new(params: Vec<f32>, decay: bool) -> Self {
        let adam = AdamState::new_like(&params);
        Self {
            grads: vec![0.0; params.len()],
            params,
            decay,
            adam,
        }
    }

    fn zero_grad(&mut self) {
        self.grads.iter_mut().for_each(|g| *g = 0.0);
    }

    fn sq_sum(&self) -> f64 {
        self.grads.iter().map(|g| (*g as f64) * (*g as f64)).sum()
    }

    fn scale_grads(&mut self, s: f32) {
        if s != 1.0 {
            self.grads.iter_mut().for_each(|g| *g *= s);
        }
    }

    fn adam_step(&mut self, lr: f32, wd: f32) {
        let wd = if self.decay { wd } else { 0.0 };
        if wd > 0.0 {
            self.params.iter_mut().for_each(|p| *p *= 1.0 - lr * wd);
        }
        self.adam.step(&mut self.params, &self.grads, lr);
    }
    fn init_normal(t: usize, c: usize, std: f32, decay: bool, rng: &mut Rng) -> Self {
        let params: Vec<f32> = (0..t * c).map(|_| rng.normal(0.0, std)).collect();
        Self::new(params, decay)
    }
}

// ---------------------------------------------------------------------------
// Linear: y = x @ W + b, W stored transposed as (in, out).
// ---------------------------------------------------------------------------

pub struct Linear {
    pub(crate) w: Tunable,
    pub(crate) b: Tunable,
    cached_x: Vec<f32>,
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize, std: f32, rng: &mut Rng) -> Self {
        Self {
            w: Tunable::init_normal(in_dim, out_dim, std, true, rng),
            b: Tunable::new(vec![0.0; out_dim], false),
            cached_x: Vec::new(),
        }
    }

    pub fn forward(&mut self, x: &[f32], rows: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
        self.cached_x = x.to_vec();
        matmul_fwd(x, &self.w.params, Some(&self.b.params), rows, in_dim, out_dim)
    }

    pub fn backward(&mut self, dy: &[f32], rows: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
        let (dx, dwt) = matmul_bwd(&self.cached_x, &self.w.params, dy, rows, in_dim, out_dim);
        for (g, s) in self.w.grads.iter_mut().zip(dwt) {
            *g += s;
        }
        for cc in 0..out_dim {
            let mut acc = 0.0f32;
            for r in 0..rows {
                acc += dy[r * out_dim + cc];
            }
            self.b.grads[cc] += acc;
        }
        dx
    }

    pub fn zero_grad(&mut self) {
        self.w.zero_grad();
        self.b.zero_grad();
    }

    pub fn sq_sum(&self) -> f64 {
        self.w.sq_sum() + self.b.sq_sum()
    }

    pub fn scale_grads(&mut self, s: f32) {
        self.w.scale_grads(s);
        self.b.scale_grads(s);
    }

    pub fn adam_step(&mut self, lr: f32, wd: f32) {
        self.w.adam_step(lr, wd);
        self.b.adam_step(lr, wd);
    }

    pub fn param_count(&self) -> usize {
        self.w.params.len() + self.b.params.len()
    }
}

// ---------------------------------------------------------------------------
// LayerNorm over vector dimension.
// ---------------------------------------------------------------------------

pub struct LayerNorm {
    g: Tunable,
    b: Tunable,
    cached_x: Vec<f32>,
}

impl LayerNorm {
    pub fn new(dim: usize) -> Self {
        Self {
            g: Tunable::new(vec![1.0; dim], false),
            b: Tunable::new(vec![0.0; dim], false),
            cached_x: Vec::new(),
        }
    }

    pub fn forward(&mut self, x: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
        self.cached_x = x.to_vec();
        ops::layernorm_fwd(x, &self.g.params, &self.b.params, rows, dim, eps)
    }

    pub fn backward(&mut self, gy: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
        let (dx, dg, db) = ops::layernorm_bwd(&self.cached_x, gy, &self.g.params, rows, dim, eps);
        for (g, s) in self.g.grads.iter_mut().zip(dg) {
            *g += s;
        }
        for (g, s) in self.b.grads.iter_mut().zip(db) {
            *g += s;
        }
        dx
    }

    pub fn zero_grad(&mut self) {
        self.g.zero_grad();
        self.b.zero_grad();
    }

    pub fn sq_sum(&self) -> f64 {
        self.g.sq_sum() + self.b.sq_sum()
    }

    pub fn scale_grads(&mut self, s: f32) {
        self.g.scale_grads(s);
        self.b.scale_grads(s);
    }

    pub fn adam_step(&mut self, lr: f32, wd: f32) {
        self.g.adam_step(lr, wd);
        self.b.adam_step(lr, wd);
    }
}

// ---------------------------------------------------------------------------
// Embedding: rows of the table are gathered (and scattered on backward).
// ---------------------------------------------------------------------------

pub struct Embedding {
    pub(crate) w: Tunable,
}

impl Embedding {
    pub fn new(n: usize, dim: usize, std: f32, rng: &mut Rng) -> Self {
        Self {
            w: Tunable::init_normal(n, dim, std, true, rng),
        }
    }

    /// `ids` has `rows` entries; the output is `(rows, dim)`.
    pub fn forward(&self, ids: &[usize], dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; ids.len() * dim];
        for (r, &id) in ids.iter().enumerate() {
            let src = &self.w.params[id * dim..(id + 1) * dim];
            out[r * dim..(r + 1) * dim].copy_from_slice(src);
        }
        out
    }

    /// Scatter-adds `dx` into the table gradients, accumulating (here and in
    /// general: `+=`, so combined token+step-hit gradients add up properly).
    pub fn backward(&mut self, ids: &[usize], dx: &[f32], dim: usize) {
        for (r, &id) in ids.iter().enumerate() {
            let g = &mut self.w.grads[id * dim..(id + 1) * dim];
            let dr = &dx[r * dim..(r + 1) * dim];
            for (gv, &v) in g.iter_mut().zip(dr) {
                *gv += v;
            }
        }
    }

    pub fn w_len(&self) -> usize {
        self.w.params.len()
    }

    pub fn zero_grad(&mut self) {
        self.w.zero_grad();
    }

    pub fn sq_sum(&self) -> f64 {
        self.w.sq_sum()
    }

    pub fn scale_grads(&mut self, s: f32) {
        self.w.scale_grads(s);
    }

    pub fn adam_step(&mut self, lr: f32, wd: f32) {
        self.w.adam_step(lr, wd);
    }
}

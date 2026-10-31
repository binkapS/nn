//! Pre-LayerNorm Transformer block and the full decoder-only GPT model.

use crate::attention::Attention;
use crate::modules::{Embedding, LayerNorm, Linear};
use crate::ops::{self, cross_entropy};
use crate::rng::Rng;

pub const LN_EPS: f32 = 1e-5;

// ---------------------------------------------------------------------------
// MLP: y = W2 @ gelu(W1 @ x + b1) + b2
// ---------------------------------------------------------------------------

struct Mlp {
    fc1: Linear, // C -> 4C
    fc2: Linear, // 4C -> C
    hid: Vec<f32>,
}

impl Mlp {
    fn new(c: usize, rng: &mut Rng) -> Self {
        Self {
            fc1: Linear::new(c, 4 * c, 0.02, rng),
            fc2: Linear::new(4 * c, c, 0.02, rng),
            hid: Vec::new(),
        }
    }

    fn forward(&mut self, x: &[f32], rows: usize, c: usize) -> Vec<f32> {
        let h = self.fc1.forward(x, rows, c, 4 * c);
        let (g, saved) = ops::gelu_fwd(&h);
        self.hid = saved;
        self.fc2.forward(&g, rows, 4 * c, c)
    }

    fn backward(&mut self, dy: &[f32], rows: usize, c: usize) -> Vec<f32> {
        let dh = self.fc2.backward(dy, rows, 4 * c, c);
        let dg = ops::gelu_bwd(&self.hid, &dh);
        self.fc1.backward(&dg, rows, c, 4 * c)
    }

    fn zero_grad(&mut self) {
        self.fc1.zero_grad();
        self.fc2.zero_grad();
    }

    fn sq_sum(&self) -> f64 {
        self.fc1.sq_sum() + self.fc2.sq_sum()
    }

    fn scale_grads(&mut self, s: f32) {
        self.fc1.scale_grads(s);
        self.fc2.scale_grads(s);
    }

    fn adam_step(&mut self, lr: f32, wd: f32) {
        self.fc1.adam_step(lr, wd);
        self.fc2.adam_step(lr, wd);
    }

    fn param_count(&self) -> usize {
        self.fc1.param_count() + self.fc2.param_count()
    }
}

// ---------------------------------------------------------------------------
// Block: y = x2 + MLP(LN2(x2)), x2 = x + Attn(LN1(x))
// ---------------------------------------------------------------------------

struct Block {
    c: usize,
    ln1: LayerNorm,
    attn: Attention,
    ln2: LayerNorm,
    mlp: Mlp,

    // Activations saved by the last forward, needed by backward:
    ln1_out: Vec<f32>, // attention input
    x1: Vec<f32>,      // x + attn_out
    ln2_out: Vec<f32>, // mlp input
}

impl Block {
    fn new(n_head: usize, c: usize, rng: &mut Rng) -> Self {
        Self {
            c,
            ln1: LayerNorm::new(c),
            attn: Attention::new(n_head, c, rng),
            ln2: LayerNorm::new(c),
            mlp: Mlp::new(c, rng),
            ln1_out: Vec::new(),
            x1: Vec::new(),
            ln2_out: Vec::new(),
        }
    }

    fn forward(&mut self, x: &[f32], b: usize, t: usize, eps: f32) -> Vec<f32> {
        let rows = b * t;
        let c = self.c;
        let ln1_out = self.ln1.forward(x, rows, c, eps);
        self.ln1_out = ln1_out.clone();
        let attn_out = self.attn.forward(&ln1_out, b, t, c);

        self.x1 = x.iter().zip(&attn_out).map(|(&a, &o)| a + o).collect();

        let ln2_out = self.ln2.forward(&self.x1, rows, c, eps);
        self.ln2_out = ln2_out.clone();

        let mlp_out = self.mlp.forward(&ln2_out, rows, c);
        self.x1.iter().zip(&mlp_out).map(|(&a, &o)| a + o).collect()
    }

    fn backward(&mut self, dy: &[f32], b: usize, t: usize, eps: f32) -> Vec<f32> {
        let rows = b * t;
        let c = self.c;

        // y = x1 + mlp_out: gradient of y w.r.t. mlp branch is dy itself,
        // w.r.t. x1 (because attn_out terms of x1) it is also dy.
        let dln2_out = self.mlp.backward(dy, rows, c);
        let mut dx1 = self.ln2.backward(&dln2_out, rows, c, eps);
        for (g, &d) in dx1.iter_mut().zip(dy) {
            *g += d; // residual from y = x1 + mlp_out
        }

        // x1 = x + attn_out: dattn_out = dx1 and dx picks up dx1 residually.
        let dln1_out = self.attn.backward(&dx1, b, t, c);
        let mut dx = self.ln1.backward(&dln1_out, rows, c, eps);
        for (g, &d) in dx.iter_mut().zip(&dx1) {
            *g += d; // residual from x1 = x + attn_out
        }
        dx
    }

    fn zero_grad(&mut self) {
        self.ln1.zero_grad();
        self.attn.zero_grad();
        self.ln2.zero_grad();
        self.mlp.zero_grad();
    }

    fn sq_sum(&self) -> f64 {
        self.ln1.sq_sum() + self.attn.sq_sum() + self.ln2.sq_sum() + self.mlp.sq_sum()
    }

    fn scale_grads(&mut self, s: f32) {
        self.ln1.scale_grads(s);
        self.attn.scale_grads(s);
        self.ln2.scale_grads(s);
        self.mlp.scale_grads(s);
    }

    fn adam_step(&mut self, lr: f32, wd: f32) {
        self.ln1.adam_step(lr, wd);
        self.attn.adam_step(lr, wd);
        self.ln2.adam_step(lr, wd);
        self.mlp.adam_step(lr, wd);
    }

    fn param_count(&self) -> usize {
        self.attn.param_count() + self.mlp.param_count()
    }
}

// ---------------------------------------------------------------------------
// GPT
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct Config {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub block_size: usize,
    pub vocab_size: usize,
}

pub struct Gpt {
    pub cfg: Config,
    tok_emb: Embedding,
    pos_emb: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    head: Linear, // C -> V (untied from the embedding table)

    // Saved for backward:
    ids: Vec<usize>,   // (B*T) token ids of last forward
    emb: Vec<f32>,     // token+position embeddings (B*T, C)
    logits: Vec<f32>,  // (B*T, V) from last forward
    b: usize,
    t: usize,
}

impl Gpt {
    pub fn new(cfg: Config, rng: &mut Rng) -> Self {
        let blocks = (0..cfg.n_layer)
            .map(|_| Block::new(cfg.n_head, cfg.n_embd, rng))
            .collect();
        // Output projections of deeper layers scaled by 1/sqrt(2*n_layer)
        // (GPT-2 init): keeps residual stream variance stable at depth.
        Self {
            cfg,
            tok_emb: Embedding::new(cfg.vocab_size, cfg.n_embd, 0.02, rng),
            pos_emb: Embedding::new(cfg.block_size, cfg.n_embd, 0.02, rng),
            blocks,
            ln_f: LayerNorm::new(cfg.n_embd),
            head: Linear::new(cfg.n_embd, cfg.vocab_size, 0.02, rng),
            ids: Vec::new(),
            emb: Vec::new(),
            logits: Vec::new(),
            b: 0,
            t: cfg.block_size,
        }
    }

    /// Runs a full forward pass. `ids.len()` must be a multiple of block_size.
    /// Stores all activations needed by backward and returns `(B*T, V)` logits.
    pub fn forward(&mut self, ids: &[usize]) -> Vec<f32> {
        let cfg = self.cfg;
        let t = cfg.block_size;
        assert_eq!(ids.len() % t, 0, "input length must be a multiple of block_size");
        let b = ids.len() / t;
        let rows = b * t;
        let c = cfg.n_embd;

        let tok = self.tok_emb.forward(ids, c);

        let mut pos_ids = Vec::with_capacity(rows);
        for _ in 0..b {
            pos_ids.extend(0..t);
        }
        let pos = self.pos_emb.forward(&pos_ids, c);

        let mut h: Vec<f32> = tok.iter().zip(&pos).map(|(&a, &p)| a + p).collect();
        self.ids = ids.to_vec();
        self.emb = h.clone();
        self.b = b;
        self.t = t;

        for block in &mut self.blocks {
            h = block.forward(&h, b, t, LN_EPS);
        }
        let hf = self.ln_f.forward(&h, rows, c, LN_EPS);
        self.logits = self.head.forward(&hf, rows, c, cfg.vocab_size);
        self.logits.clone()
    }

    /// Cross-entropy loss of the last forward pass, plus full backward through
    /// the entire network into every parameter gradient. Returns the mean loss.
    pub fn backward_from_targets(&mut self, targets: &[usize]) -> f32 {
        let cfg = self.cfg;
        let rows = self.b * self.t;
        let (loss, dlogits, _p) = cross_entropy(&self.logits, targets, rows, cfg.vocab_size);

        let dhf = self.head.backward(&dlogits, rows, cfg.n_embd, cfg.vocab_size);
        let mut dh = self.ln_f.backward(&dhf, rows, cfg.n_embd, LN_EPS);

        for block in self.blocks.iter_mut().rev() {
            dh = block.backward(&dh, self.b, self.t, LN_EPS);
        }
        // dh is now the gradient of the combined (token + position) embedding.
        let demb = dh;

        let mut pos_ids = Vec::with_capacity(rows);
        for _ in 0..self.b {
            pos_ids.extend(0..self.t);
        }
        self.tok_emb.backward(&self.ids, &demb, cfg.n_embd);
        self.pos_emb.backward(&pos_ids, &demb, cfg.n_embd);

        loss
    }

    pub fn eval_loss(&mut self, ids: &[usize], targets: &[usize]) -> f32 {
        let cfg = self.cfg;
        let logits = self.forward(ids);
        let (loss, _, _) = cross_entropy(&logits, targets, self.b * self.t, cfg.vocab_size);
        loss
    }

    // -- training plumbing ---------------------------------------------------

    pub fn zero_grad(&mut self) {
        self.tok_emb.zero_grad();
        self.pos_emb.zero_grad();
        for block in &mut self.blocks {
            block.zero_grad();
        }
        self.ln_f.zero_grad();
        self.head.zero_grad();
    }

    pub fn total_grad_sq(&self) -> f64 {
        let mut s = self.tok_emb.sq_sum() + self.pos_emb.sq_sum() + self.ln_f.sq_sum() + self.head.sq_sum();
        for block in &self.blocks {
            s += block.sq_sum();
        }
        s
    }

    pub fn scale_grads(&mut self, s: f32) {
        self.tok_emb.scale_grads(s);
        self.pos_emb.scale_grads(s);
        for block in &mut self.blocks {
            block.scale_grads(s);
        }
        self.ln_f.scale_grads(s);
        self.head.scale_grads(s);
    }

    pub fn clip_grad_norm(&mut self, max_norm: f32) {
        let norm = self.total_grad_sq().sqrt() as f32;
        if norm < 1e-300 {
            return;
        }
        if norm > max_norm {
            self.scale_grads(max_norm / (norm + 1e-6));
        }
    }

    pub fn adam_step(&mut self, lr: f32, wd: f32) {
        self.tok_emb.adam_step(lr, wd);
        self.pos_emb.adam_step(lr, wd);
        for block in &mut self.blocks {
            block.adam_step(lr, wd);
        }
        self.ln_f.adam_step(lr, wd);
        self.head.adam_step(lr, wd);
    }

    /// One full training step: forward, CE loss + backward, global-norm
    /// gradient clipping, AdamW update. Returns the batch loss.
    pub fn train_step(&mut self, ids: &[usize], targets: &[usize], lr: f32, wd: f32, clip: f32) -> f32 {
        self.zero_grad();
        self.forward(ids);
        let loss = self.backward_from_targets(targets);
        self.clip_grad_norm(clip);
        self.adam_step(lr, wd);
        loss
    }

    pub fn param_count(&self) -> usize {
        let blocks: usize = self.blocks.iter().map(|b| b.param_count()).sum();
        self.tok_emb.w_len() + self.pos_emb.w_len() + blocks + self.head.param_count()
    }

    // -- sampling --------------------------------------------------------------

    /// Autoregressive sampling: grows `prompt` by `n_new` tokens, feeding only
    /// the last `block_size` tokens each step. Temperature 0 = greedy,
    /// top_k = 0 disables top-k. Returns the full sequence.
    pub fn generate(&mut self, prompt: &[usize], n_new: usize, temperature: f32, top_k: usize, rng: &mut Rng) -> Vec<usize> {
        assert!(!prompt.is_empty(), "prompt must contain at least one token");
        let mut seq = prompt.to_vec();
        for _ in 0..n_new {
            let pad = self.pad_context(&seq);
            let logits = self.forward(&pad);
            let row = logits.len() - self.cfg.vocab_size; // last token's row
            let next = sample_from_logits(&logits[row..], temperature, top_k, rng);
            seq.push(next);
        }
        seq
    }

    /// Builds exactly one row-block of tokens (`pad_context` returns a single
    /// sequence worth exactly `block_size`, i.e. batch 1):
    /// right-sequences are left-truncated or left-padded with the first token.
    fn pad_context(&self, seq: &[usize]) -> Vec<usize> {
        if seq.len() >= self.cfg.block_size {
            seq[seq.len() - self.cfg.block_size..].to_vec()
        } else {
            let pad = seq[0]; // soft left-pad with the first token
            let mut v = vec![pad; self.cfg.block_size - seq.len()];
            v.extend_from_slice(seq);
            v
        }
    }
}

/// Sample one class id from a single logits row.
fn sample_from_logits(row: &[f32], temperature: f32, top_k: usize, rng: &mut Rng) -> usize {
    assert!(!row.is_empty());
    if temperature <= 1e-6 {
        // Greedy: argmax.
        let (mut best_i, mut best_v) = (0, f32::NEG_INFINITY);
        for (i, &v) in row.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }
        return best_i;
    }
    let scale = 1.0 / temperature;
    let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut scores: Vec<f32> = row.iter().map(|&v| (v - mx) * scale).collect();
    if top_k > 0 && top_k < scores.len() {
        // Blank out all but the k largest entries, then renormalize via softmax.
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_unstable_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        let cutoff = scores[order[top_k - 1]];
        scores.iter_mut().for_each(|s| {
            if *s < cutoff {
                *s = f32::NEG_INFINITY;
            }
        });
    }
    let mut z = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s).exp();
        z += *s;
    }
    let u = rng.uniform01() * z;
    let mut acc = 0.0f32;
    for (i, &s) in scores.iter().enumerate() {
        acc += s;
        if u <= acc {
            return i;
        }
    }
    scores.len() - 1
}

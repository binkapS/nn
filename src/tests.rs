//! Gradient checks: every hand-derived backward pass is validated against
//! central finite differences of the corresponding loss, element-wise.

use crate::attention::Attention;
use crate::corpus::DEMO_TEXT;
use crate::data::{CharTokenizer, get_batch};
use crate::modules::Embedding;
use crate::ops::{
    cross_entropy, gelu_bwd, gelu_fwd, layernorm_bwd, layernorm_fwd, matmul_bwd, matmul_fwd,
};
use crate::rng::Rng;
use crate::transformer::Config;

/// Serial central-difference gradient check on `max_n` sampled indices.
/// Panics if any relative error exceeds `tol`. Returns (checked, max rel err).
fn check(
    f: &dyn Fn(&[f32]) -> f32,
    x: &[f32],
    analytic: &[f32],
    max_n: usize,
    eps: f32,
    tol: f64,
    rng: &mut Rng,
) -> (usize, f64) {
    let idxs: Vec<usize> = if x.len() <= max_n {
        (0..x.len()).collect()
    } else {
        (0..max_n).map(|_| rng.rand_index(x.len())).collect()
    };
    let mut worst = 0.0f64;
    for &i in &idxs {
        let h = eps.max(x[i].abs() * 1e-2);
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[i] += h;
        xm[i] -= h;
        let num = (f(&xp) - f(&xm)) / (2.0 * h);
        let ana = analytic[i];
        let denom = num.abs().max(ana.abs()).max(1e-2);
        let rel = (num - ana).abs() as f64 / denom as f64;
        assert!(
            rel < tol,
            "grad mismatch at index {}: numeric {:.6} analytic {:.6} (rel {:.4})",
            i,
            num,
            ana,
            rel
        );
        worst = worst.max(rel);
    }
    (idxs.len(), worst)
}

/// Lift an output vector to a scalar with the fixed random projection `r`.
fn with_r(out: &[f32], r: &[f32]) -> f32 {
    out.iter().zip(r).map(|(&o, &rr)| o * rr).sum()
}

#[test]
fn matmul_gradients() {
    let mut rng = Rng::new(0xC0FFEE);
    let (m, k, n) = (4usize, 6usize, 5usize);
    let x: Vec<f32> = (0..m * k).map(|_| rng.normal(0.0, 1.0)).collect();
    let wt: Vec<f32> = (0..k * n).map(|_| rng.normal(0.0, 1.0)).collect();
    let r: Vec<f32> = (0..m * n).map(|_| rng.normal(0.0, 1.0)).collect();

    let y = matmul_fwd(&x, &wt, None, m, k, n);
    let (dx, dwt) = matmul_bwd(&x, &wt, &r, m, k, n);
    assert_eq!(y.len(), m * n);

    let fx = |xq: &[f32]| with_r(&matmul_fwd(xq, &wt, None, m, k, n), &r);
    let fw = |wq: &[f32]| with_r(&matmul_fwd(&x, wq, None, m, k, n), &r);
    let (n1, w1) = check(&fx, &x, &dx, 6, 1e-2, 2e-2, &mut rng);
    let (n2, w2) = check(&fw, &wt, &dwt, 6, 1e-2, 2e-2, &mut rng);
    assert_eq!(n1 + n2, 12);
    assert!(w1 < 2e-2 && w2 < 2e-2);
}

#[test]
fn layernorm_gradients() {
    let mut rng = Rng::new(0xC0FFEE);
    let (rows, c) = (5usize, 8usize);
    let x: Vec<f32> = (0..rows * c).map(|_| rng.normal(0.0, 1.0)).collect();
    let g: Vec<f32> = (0..c).map(|_| rng.normal(1.0, 0.1)).collect();
    let b: Vec<f32> = (0..c).map(|_| rng.normal(0.0, 0.1)).collect();
    let r: Vec<f32> = (0..rows * c).map(|_| rng.normal(0.0, 1.0)).collect();

    let y = layernorm_fwd(&x, &g, &b, rows, c, 1e-5);
    assert_eq!(y.len(), rows * c);
    let (dx, dg, db) = layernorm_bwd(&x, &r, &g, rows, c, 1e-5);

    let fx = |xq: &[f32]| with_r(&layernorm_fwd(xq, &g, &b, rows, c, 1e-5), &r);
    let fg = |gq: &[f32]| with_r(&layernorm_fwd(&x, gq, &b, rows, c, 1e-5), &r);
    let fb = |bq: &[f32]| with_r(&layernorm_fwd(&x, &g, bq, rows, c, 1e-5), &r);
    let (n1, w1) = check(&fx, &x, &dx, 8, 1e-2, 2e-2, &mut rng);
    let (n2, w2) = check(&fg, &g, &dg, 8, 1e-2, 2e-2, &mut rng);
    let (n3, w3) = check(&fb, &b, &db, 8, 1e-2, 2e-2, &mut rng);
    assert_eq!(n1 + n2 + n3, 24);
    assert!(w1 < 2e-2 && w2 < 2e-2 && w3 < 2e-2);
}

#[test]
fn gelu_gradient() {
    let mut rng = Rng::new(0xC0FFEE);
    let x: Vec<f32> = (0..64).map(|_| rng.normal(0.0, 1.0)).collect();
    let r: Vec<f32> = (0..64).map(|_| rng.normal(0.0, 1.0)).collect();
    let (y, saved) = gelu_fwd(&x);
    assert_eq!(y.len(), x.len());
    let dx = gelu_bwd(&saved, &r);
    let f = |xq: &[f32]| with_r(&gelu_fwd(xq).0, &r);
    let (n, w) = check(&f, &x, &dx, 16, 1e-2, 2e-2, &mut rng);
    assert_eq!(n, 16);
    assert!(w < 2e-2);
}

#[test]
fn softmax_ce_gradients() {
    let mut rng = Rng::new(0xC0FFEE);
    let (rows, vocab) = (3usize, 9usize);
    let x: Vec<f32> = (0..rows * vocab).map(|_| rng.normal(0.0, 1.0)).collect();
    let targets: Vec<usize> = (0..rows).map(|_| rng.rand_index(vocab)).collect();

    // dL/dlogits for L = mean(-log p_target):
    let (_loss, dlogits, _probs) = cross_entropy(&x, &targets, rows, vocab);
    // The analytic gradient corresponds to the mean loss exactly:
    let f = |xq: &[f32]| {
        let (loss, _, _) = cross_entropy(xq, &targets, rows, vocab);
        loss
    };
    let (n, w) = check(&f, &x, &dlogits, 12, 1e-2, 2e-2, &mut rng);
    assert_eq!(n, 12);
    assert!(w < 2e-2);
}

#[test]
fn attention_gradients() {
    // Snapshot the freshly initialized parameters, rebuild an identical module
    // for each finite-difference probe (attention caches state on itself).
    let (b, t, c, h) = (2usize, 5usize, 8usize, 2usize);
    let mut rng = Rng::new(0xC0FFEE);
    let x: Vec<f32> = (0..b * t * c).map(|_| rng.normal(0.0, 1.0)).collect();
    let r: Vec<f32> = (0..b * t * c).map(|_| rng.normal(0.0, 1.0)).collect();

    let mut base = Attention::new(h, c, &mut rng);
    let params: Vec<Vec<f32>> = vec![
        base.c_qkv.w.params.clone(),
        base.c_qkv.b.params.clone(),
        base.c_proj.w.params.clone(),
        base.c_proj.b.params.clone(),
    ];
    let y = base.forward(&x, b, t, c);
    assert_eq!(y.len(), b * t * c);
    let dx = base.backward(&r, b, t, c); // L = Σ y·r

    let fwd = |xq: &[f32]| {
        let mut fresh = Attention::new(2, c, &mut Rng::new(0xDEAD_BEEF));
        build_with(&mut fresh, &params);
        fresh.forward(xq, b, t, c)
    };
    let f = |xq: &[f32]| with_r(&fwd(xq), &r);
    let (n, w) = check(&f, &x, &dx, 8, 1e-2, 2e-2, &mut rng);
    assert_eq!(n, 8);
    assert!(w < 2e-2);
}

fn build_with(a: &mut Attention, p: &[Vec<f32>]) {
    a.c_qkv.w.params.copy_from_slice(&p[0]);
    a.c_qkv.b.params.copy_from_slice(&p[1]);
    a.c_proj.w.params.copy_from_slice(&p[2]);
    a.c_proj.b.params.copy_from_slice(&p[3]);
}

#[test]
fn embedding_gradients() {
    let mut rng = Rng::new(0xC0FFEE);
    let (n, c) = (7usize, 5usize);
    let ids = vec![0usize, 3, 3, 6, 1, 3, 0]; // repeats exercise accumulation
    let table: Vec<f32> = (0..n * c).map(|_| rng.normal(0.0, 1.0)).collect();
    let r: Vec<f32> = (0..ids.len() * c).map(|_| rng.normal(0.0, 1.0)).collect();

    let mut emb = Embedding::new(n, c, 0.02, &mut rng);
    emb.w.params.copy_from_slice(&table);
    emb.zero_grad();
    let out = gather(&emb.w.params, &ids, c);
    assert_eq!(out.len(), ids.len() * c);
    emb.backward(&ids, &r, c);
    let dtable = emb.w.grads.clone();

    let f = |tq: &[f32]| with_r(&gather(tq, &ids, c), &r);
    let (cnt, w) = check(&f, &table, &dtable, 10, 1e-2, 2e-2, &mut rng);
    assert_eq!(cnt, 10);
    assert!(w < 2e-2);
}

fn gather(table: &[f32], ids: &[usize], c: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; ids.len() * c];
    for (r, &id) in ids.iter().enumerate() {
        out[r * c..(r + 1) * c].copy_from_slice(&table[id * c..(id + 1) * c]);
    }
    out
}

#[test]
fn schedule_edges() {
    // schedule(step, total, warmup, peak, min_lr) must stay finite and inside
    // [min_lr, peak] for pathological inputs too.
    let sched = |a: usize, t: usize, w: usize| crate::schedule(a, t, w, 1.5e-3, 1.5e-4);

    // total = 0 and total <= warmup must not divide by zero or truncate badly.
    for (total, warmup) in [(0usize, 0usize), (0, 5), (3, 5), (1, 1), (100, 0)] {
        for step in 0..(total.max(warmup) + 5) {
            let lr = sched(step, total, warmup);
            assert!(
                lr.is_finite(),
                "step {step} total {total} warmup {warmup} → {lr}"
            );
            assert!(
                (1.4e-4..=1.51e-3).contains(&lr),
                "lr {lr} out of band, step {step}/{total}/{warmup}"
            );
        }
    }
    // Sanity: full run reaches min_lr at the final step, peak at warmup end.
    let (total, warmup, peak, min) = (200usize, 10usize, 1.5e-3_f32, 1.5e-4_f32);
    assert_eq!(
        crate::schedule(0, total, warmup, peak, min),
        peak * 1.0 / warmup as f32
    );
    assert!((crate::schedule(total - 1, total, warmup, peak, min) - min).abs() < 1e-6);
}

#[test]
fn training_reduces_loss() {
    // Full-model smoke test: loss must drop meaningfully within a few dozen
    // steps on a slice of the demo corpus.
    let tok = CharTokenizer::build(DEMO_TEXT);
    let text: String = DEMO_TEXT.chars().take(400).collect();
    let ids = tok.encode(&text);
    let cfg = Config {
        n_layer: 2,
        n_head: 2,
        n_embd: 32,
        block_size: 16,
        vocab_size: tok.vocab_size(),
    };
    let mut rng = Rng::new(1);
    let mut model = crate::transformer::Gpt::new(cfg, &mut rng);

    let mut loss0 = f32::INFINITY;
    let mut loss_last = f32::INFINITY;
    for step in 0..60 {
        let (x, y) = get_batch(&mut rng, &ids, cfg.block_size, 2);
        let l = model.train_step(&x, &y, 3e-3, 0.0, 1.0);
        if step == 0 {
            loss0 = l;
        }
        loss_last = l;
    }
    assert!(
        loss_last < loss0 - 0.5,
        "training did not reduce loss: start {:.4}, end {:.4}",
        loss0,
        loss_last
    );
    // Gradients must have flowed (non-zero, finite).
    let sq = model.total_grad_sq();
    assert!(sq.is_finite() && sq > 0.0, "grads invalid: {sq}");
}

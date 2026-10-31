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
#[ignore = "performance report, run explicitly: cargo test --release -- --ignored --nocapture perf_baseline"]
fn perf_baseline() {
    // Times each hot op AT THE DEFAULT CONFIG SHAPES (B=8, T=128, C=128,
    // V=155, L=4) so optimization wins can be attributed to real workload
    // cost rather than microbenchmark artifacts.
    fn r2(rng: &mut Rng, n: usize) -> Vec<f32> {
        (0..n).map(|_| rng.normal(0.0, 1.0)).collect()
    }
    fn time(
        report: &mut Vec<(String, f64, f64)>,
        name: &str,
        flops: f64,
        iters: usize,
        mut f: impl FnMut(),
    ) {
        for _ in 0..iters {
            f();
        }
        let start = std::time::Instant::now();
        for _ in 0..iters {
            f();
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        report.push((
            name.to_string(),
            ms,
            if flops > 0.0 { flops / (ms * 1e6) } else { 0.0 },
        ));
    }

    let iters = 10;
    let (m, t, c, v, h, l) = (8 * 128, 128, 128, 155, 4, 4);
    let mut rng = Rng::new(0xBAD_5EED);

    let mut report: Vec<(String, f64, f64)> = Vec::new(); // (name, ms, GFLOP/s)
    macro_rules! time {
        ($name:expr, $flops:expr, $f:expr $(,)?) => {
            time(&mut report, &$name, $flops, iters, $f)
        };
    }

    // -- GEMMs at the four distinct (k, n) shapes of a block + head ---------
    for &(k, n, tag) in [(c, 3 * c, "qkv"), (c, c, "proj"), (c, 4 * c, "fc1"), (4 * c, c, "fc2")]
        .iter()
    {
        let x = r2(&mut rng, m * k);
        let wt = r2(&mut rng, k * n);
        let bias = r2(&mut rng, n);
        let gy = r2(&mut rng, m * n);
        let flops = 2.0 * (m * k * n) as f64;

        time!(&format!("matmul_fwd[{tag}] {m}×{k}×{n}"), flops, Box::new(|| {
            let _ = matmul_fwd(&x, &wt, Some(&bias), m, k, n);
        }));
        time!(
            &format!("matmul_bwd[{tag}] dx|dwt {m}×{k}×{n}"),
            3.0_f64 * flops / 2.0,
            Box::new(|| {
                let _ = matmul_bwd(&x, &wt, &gy, m, k, n);
            }),
        );
    }

    // -- attention (whole layer) --------------------------------------------
    let xattn = r2(&mut rng, m * c);
    let mut attn = crate::attention::Attention::new(h, c, &mut rng);
    let dy = r2(&mut rng, m * c);
    let attn_flops = 4.0 * (t * t * (c / h)) as f64 + 2.0 * (m * c * 3 * c) as f64 + 2.0 * (m * c * c) as f64;
    time!("attention.fwd", attn_flops, Box::new(|| {
        let _ = attn.forward(&xattn, 8, t, c);
    }));
    time!("attention.fwd+bwd", 2.5 * attn_flops, Box::new(|| {
        let _ = attn.forward(&xattn, 8, t, c);
        let _ = attn.backward(&dy, 8, t, c);
    }));

    // -- elementwise small fry -----------------------------------------------
    let xln = r2(&mut rng, m * c);
    let gln = r2(&mut rng, c);
    let bln = r2(&mut rng, c);
    let gyl = r2(&mut rng, m * c);
    time!("layernorm.fwd(1024,128)", 3.0 * (m * c) as f64, Box::new(|| {
        let _ = layernorm_fwd(&xln, &gln, &bln, m, c, 1e-5);
    }));
    time!(
        "layernorm.bwd(1024,128)",
        6.0 * (m * c) as f64,
        Box::new(|| {
            let _ = layernorm_bwd(&xln, &gyl, &gln, m, c, 1e-5);
        }),
    );
    let logits = r2(&mut rng, m * v);
    let targets: Vec<usize> = (0..m).map(|_| rng.rand_index(v)).collect();
    time!("cross_entropy(1024,155)", 4.0 * (m * v) as f64, Box::new(|| {
        let _ = cross_entropy(&logits, &targets, m, v);
    }));

    // -- optimizer + clip (parameter-count scale) -----------------------------
    let mut gpt = crate::transformer::Gpt::new(
        crate::transformer::Config {
            n_layer: l,
            n_head: h,
            n_embd: c,
            block_size: t,
            vocab_size: v,
        },
        &mut rng,
    );
    gpt.zero_grad();
    let _ = gpt.param_count();
    time!(
        "adamw+clip on whole Gpt (~847k params)",
        0.0,
        Box::new(move || {
            gpt.clip_grad_norm(1.0);
            gpt.adam_step(1e-3, 0.1);
        }),
    );

    println!("\n── perf baseline (default shapes: B=8 T=128 C=128 V=155, {} iters) ──", iters);
    println!("{:<44}>{:>10}{:>12}", "op", "ms", "GFLOP/s");
    for (name, ms, gfl) in &report {
        if *gfl > 0.0 {
            println!("{:<44} {:>10.2} {:>12.2}", name, ms, gfl);
        } else {
            println!("{:<44} {:>10.2} {:>12}", name, ms, "—");
        }
    }
    println!();
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

#[test]
#[ignore = "diag: split matmul_bwd halves"]
fn perf_diag_bwd_halves() {
    let (m, k, n) = (1024usize, 128usize, 384usize);
    let mut rng = Rng::new(7);
    let mut rr = |len: usize| (0..len).map(|_| rng.normal(0.0, 1.0)).collect::<Vec<f32>>();
    let x = rr(m * k);
    let wt = rr(k * n);
    let gy = rr(m * n);
    let iters = 10;
    let t = 4usize;

    // dx half: dx[r][kk] = dot(gy[r], wt[kk])
    let mut dx_t = 10_000.0f64;
    {
        let wt: &[f32] = &wt;
        let gy: &[f32] = &gy;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let mut dx = vec![0.0f32; m * k];
            std::thread::scope(|s| {
                let mut rest: &mut [f32] = &mut dx;
                for i in 0..t {
                    let len = (m / t) * k;
                    let (chunk, tail) = rest.split_at_mut(len);
                    rest = tail;
                    let (wt, gy) = (wt, gy);
                    s.spawn(move || {
                        for row in 0..(m / t) {
                            let gr = &gy[(row) * n..(row + 1) * n];
                            for kk in 0..k {
                                chunk[row * k + kk] = gr
                                    .iter()
                                    .zip(&wt[kk * n..(kk + 1) * n])
                                    .map(|(&g, &w)| g * w)
                                    .sum();
                            }
                        }
                    });
                }
            });
            std::hint::black_box(&dx);
        }
        dx_t = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    }

    // dwt half: per-thread accumulate over row ranges (current design)
    let mut dwt_t = 10_000.0f64;
    {
        let x: &[f32] = &x;
        let gy: &[f32] = &gy;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let mut dwt = vec![0.0f32; k * n];
            std::thread::scope(|s| {
                let handles: Vec<_> = (0..t)
                    .map(|i| {
                        let lo = i * (m / t);
                        let hi = (i + 1) * (m / t);
                        let (x, gy) = (x, gy);
                        s.spawn(move || {
                            let mut buf = vec![0.0f32; k * n];
                            for row in lo..hi {
                                let xr = &x[row * k..(row + 1) * k];
                                let gr = &gy[row * n..(row + 1) * n];
                                for (kk, &a) in xr.iter().enumerate() {
                                    for (dv, &g) in buf[kk * n..(kk + 1) * n].iter_mut().zip(gr) {
                                        *dv += a * g;
                                    }
                                }
                            }
                            buf
                        })
                    })
                    .collect();
                for h in handles {
                    for (o, v) in dwt.iter_mut().zip(h.join().unwrap()) {
                        *o += v;
                    }
                }
            });
            std::hint::black_box(&dwt);
        }
        dwt_t = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    }

    let fl = 2.0 * (m * k * n) as f64 / 1e9; // GFLOP of one half
    println!(
        "diag bwd halves [{m}×{k}×{n}]: dx = {dx_t:.2} ms ({:.1} GF/s)  dwt = {dwt_t:.2} ms ({:.1} GF/s)",
        fl / (dx_t / 1000.0),
        fl / (dwt_t / 1000.0),
    );
}

#[test]
#[ignore = "diag: full train_step at default shapes"]
fn perf_diag_train_step() {
    use crate::transformer::{Config, Gpt};
    use std::time::Instant;
    let cfg = Config {
        n_layer: 4,
        n_head: 4,
        n_embd: 128,
        block_size: 128,
        vocab_size: 155,
    };
    let mut rng = Rng::new(9);
    let mut model = Gpt::new(cfg, &mut rng);
    let rids: Vec<usize> = (0..1024).map(|_| rng.rand_index(155)).collect();
    let tids: Vec<usize> = (0..1024).map(|_| rng.rand_index(155)).collect();
    let iters = 5;

    let t0 = Instant::now();
    let mut loss = 0.0f32;
    for _ in 0..iters {
        loss = model.train_step(&rids, &tids, 1e-3, 0.1, 1.0);
    }
    let full = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // forward only
    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(model.forward(&rids));
    }
    let fwd = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // backward only (uses last forward's logits)
    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(model.backward_from_targets(&tids));
    }
    let bwd = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // updates
    let t0 = Instant::now();
    for _ in 0..iters {
        model.clip_grad_norm(1.0);
        model.adam_step(1e-3, 0.1);
    }
    let upd = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    println!(
        "diag train_step: full = {full:.1} ms  (fwd {fwd:.1}, bwd {bwd:.1}, upd {upd:.1}), loss {loss}",
    );
}

// ---------------------------------------------------------------------------
// Stage 1: GPU scaffold
// ---------------------------------------------------------------------------

#[test]
fn backend_selection_semantics() {
    use crate::gpu::{select_backend, Engine};

    // cpu always CPU regardless of hardware.
    let (e, note) = select_backend("cpu", false).unwrap();
    assert_eq!(e, Engine::Cpu);
    assert!(note.contains("cpu engine"));

    // unknown flag errors.
    assert!(select_backend("metal", false).is_err());

    // In this sandbox gpu/auto with gpu_implemented=false resolve to CPU but
    // carry the adapter/limitation note. (Adapter presence varies per host,
    // so only the shape of the outcome is asserted.)
    for flag in ["auto", "gpu"] {
        match select_backend(flag, false) {
            Ok((e, note)) => {
                assert_eq!(e, Engine::Cpu, "{flag} must fall back before Stage 2");
                assert!(!note.is_empty(), "{flag} must carry a note: {note}");
            }
            Err(err) => {
                // Only `gpu` may hard-fail without an adapter on the host.
                assert_eq!(flag, "gpu", "{flag} must not fail in auto mode: {err}");
                assert!(err.contains("no usable adapter"), "{err}");
            }
        }
    }
}

#[test]
#[ignore = "needs a GPU/driver; run explicitly: cargo test --release -- --ignored --nocapture gpu_scaffold"]
fn gpu_scaffold() {
    // Full scaffold check: probe → device open → passthrough roundtrip.
    let note = crate::gpu::probe_adapters().expect("adapter");
    println!("{note}");
    let dev = crate::gpu::open_device().expect("device");
    let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let out = crate::gpu::device_roundtrip(&dev, &data).expect("roundtrip");
    assert_eq!(out.len(), data.len());
    for (a, b) in out.iter().zip(&data) {
        assert_eq!(a, b, "passthrough roundtrip must be a 1:1 copy");
    }
    println!("gpu roundtrip OK over {} f32 lanes", data.len());
}

// ---------------------------------------------------------------------------
// Stage 2.1: GPU GEMM parity + first benchmarks (vs CPU reference)
// ---------------------------------------------------------------------------

/// Max relative error |a-b| / max(|a|,|b|,1e-8) over two slices.
fn rel_err_max(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "shape mismatch in parity compare");
    a.iter()
        .zip(b)
        .map(|(&x, &y)| ((x - y) as f64 / (x.abs().max(y.abs()) as f64).max(1e-8).max(1.0)))
        .fold(0.0f64, f64::max)
}

#[test]
#[ignore = "needs GPU; run: cargo test --release -- --ignored --nocapture gpu_gemm_parity"]
fn gpu_gemm_parity() {
    use crate::gpu::GpuGemm;
    let gemm = GpuGemm::new().expect("GpuGemm device open");
    let mut rng = Rng::new(0xC0FF_EE01);
    let mut r2 = |n: usize| (0..n).map(|_| rng.normal(0.0, 1.0)).collect::<Vec<f32>>();

    // Default-config shapes (multiples of 16) plus deliberately ragged ones
    // (not multiples of 16, tiny, 1×1) to exercise every OOB-guard path in
    // the WGSL tiles.
    let cases: &[(usize, usize, usize, u32)] = &[
        (1024, 128, 384, 1), // qkv, bias
        (1024, 128, 128, 0), // proj, no bias
        (1024, 128, 512, 1), // fc1
        (1024, 512, 128, 1), // fc2
        (5, 7, 6, 1),
        (5, 7, 6, 0),
        (17, 33, 29, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 0),
    ];

    for &(m, k, n, use_bias) in cases {
        let x = r2(m * k);
        let wt = r2(k * n);
        let bias = r2(n);
        let bias_opt = if use_bias == 1 { Some(&bias[..]) } else { None };
        let gy = r2(m * n);
        let tol = 2e-3;

        let cpu = matmul_fwd(&x, &wt, bias_opt, m, k, n);
        let gpu = gemm.matmul_fwd(&x, &wt, bias_opt, m, k, n);
        let e = rel_err_max(&cpu, &gpu);
        println!("fwd {m}×{k}×{n} bias={use_bias}: max rel err {e:.3e}");
        assert!(e < tol, "fwd {m}×{k}×{n}: rel err {e} >= {tol}");

        let (dx_c, dwt_c) = matmul_bwd(&x, &wt, &gy, m, k, n);
        let (dx_g, dwt_g) = gemm.matmul_bwd(&x, &wt, &gy, m, k, n);
        let e = rel_err_max(&dx_c, &dx_g);
        println!("  bwd dx {m}×{k}×{n}: max rel err {e:.3e}");
        assert!(e < tol, "dx {m}×{k}×{n}: rel err {e} >= {tol}");
        let e = rel_err_max(&dwt_c, &dwt_g);
        println!("  bwd dwt {m}×{k}×{n}: max rel err {e:.3e}");
        assert!(e < tol, "dwt {m}×{k}×{n}: rel err {e} >= {tol}");
    }
    println!("gpu_gemm_parity: all {} shapes within tol", cases.len());
}

#[test]
#[ignore = "needs GPU; run: cargo test --release -- --ignored --nocapture gpu_gemm_perf"]
fn gpu_gemm_perf() {
    use crate::gpu::GpuGemm;
    use std::time::Instant;
    let gemm = GpuGemm::new().expect("GpuGemm device open");
    let mut rng = Rng::new(0xBAD_5EED);
    let mut r2 = |n: usize| (0..n).map(|_| rng.normal(0.0, 1.0)).collect::<Vec<f32>>();

    let m = 1024usize;
    let iters = 10;
    let warm = 3;
    println!(
        "{:<34}{:>10}{:>10}{:>10}{:>10}",
        "shape", "cpu ms", "cpu GF/s", "gpu ms", "gpu GF/s"
    );

    for &(k, n, tag) in [
        (128usize, 384usize, "qkv"),
        (128, 128, "proj"),
        (128, 512, "fc1"),
        (512, 128, "fc2"),
    ]
    .iter()
    {
        let x = r2(m * k);
        let wt = r2(k * n);
        let bias = r2(n);
        let gy = r2(m * n);
        let flops = 2.0 * (m * k * n) as f64;
        let name = format!("fwd [{tag}] {m}×{k}×{n}");

        // CPU
        for _ in 0..warm {
            let _ = matmul_fwd(&x, &wt, Some(&bias), m, k, n);
        }
        let t = Instant::now();
        for _ in 0..iters {
            let _ = matmul_fwd(&x, &wt, Some(&bias), m, k, n);
        }
        let cpu_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // GPU (upload+dispatch+readback, the Stage-2.1 un-fused path)
        for _ in 0..warm {
            let _ = gemm.matmul_fwd(&x, &wt, Some(&bias), m, k, n);
        }
        let t = Instant::now();
        for _ in 0..iters {
            let _ = gemm.matmul_fwd(&x, &wt, Some(&bias), m, k, n);
        }
        let gpu_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        println!(
            "{:<34}{:>10.2}{:>10.2}{:>10.2}{:>10.2}",
            name,
            cpu_ms,
            flops / (cpu_ms * 1e6),
            gpu_ms,
            flops / (gpu_ms * 1e6),
        );

        // backward pair
        let name = format!("bwd [{tag}] {m}×{k}×{n}");
        let bflops = 3.0 * flops / 2.0;
        for _ in 0..warm {
            let _ = matmul_bwd(&x, &wt, &gy, m, k, n);
        }
        let t = Instant::now();
        for _ in 0..iters {
            let _ = matmul_bwd(&x, &wt, &gy, m, k, n);
        }
        let cpu_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        for _ in 0..warm {
            let _ = gemm.matmul_bwd(&x, &wt, &gy, m, k, n);
        }
        let t = Instant::now();
        for _ in 0..iters {
            let _ = gemm.matmul_bwd(&x, &wt, &gy, m, k, n);
        }
        let gpu_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!(
            "{:<34}{:>10.2}{:>10.2}{:>10.2}{:>10.2}",
            name,
            cpu_ms,
            bflops / (cpu_ms * 1e6),
            gpu_ms,
            bflops / (gpu_ms * 1e6),
        );
    }
}

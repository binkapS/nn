//! Primitive tensor math with explicit forward/backward passes.
//!
//! Layout conventions used throughout the crate:
//! * A 2-D matrix is row-major `Vec<f32>` with shape `(rows, cols)`.
//! * Linear layers compute `y = x @ W + b` with `W` stored transposed as
//!   `(in, out)`. This keeps the inner accumulation loops contiguous.
//! * The heavy paths (forward matmul and both backward matmuls, layernorm,
//!   gelu, softmax) are parallelized over disjoint row ranges with
//!   `std::thread::scope`; reductions (weight-gradient accumulation) use
//!   per-thread buffers that are summed after the threads join.

use std::thread;

/// Minimum number of rows before threading overhead is worth it.
const PAR_THRESHOLD: usize = 16;

fn thread_count() -> usize {
    thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Split `n` units of work into balanced `(lo, hi)` ranges, one per thread.
fn work_ranges(n: usize) -> Vec<(usize, usize)> {
    let t = thread_count().min(n.max(1));
    let base = n / t;
    let extra = n % t;
    let mut out = Vec::with_capacity(t);
    let mut start = 0;
    for i in 0..t {
        let len = base + usize::from(i < extra);
        out.push((start, start + len));
        start += len;
    }
    out
}

/// Run `f` over disjoint row slices of `dst` (rows × cols, row-major),
/// in parallel when `rows` is large enough. Each invocation receives its
/// exclusive `dst` slice and its `(lo, hi)` original row range.
fn par_rows<F>(dst: &mut [f32], rows: usize, cols: usize, f: &F)
where
    F: Fn(&mut [f32], (usize, usize)) + Sync,
{
    if rows < PAR_THRESHOLD || thread_count() == 1 {
        f(dst, (0, rows));
        return;
    }
    let ranges = work_ranges(rows);
    thread::scope(|s| {
        let mut rest = dst;
        for &(lo, hi) in &ranges {
            let len = (hi - lo) * cols;
            let (head, tail) = rest.split_at_mut(len);
            rest = tail;
            s.spawn(move || f(head, (lo, hi)));
        }
    });
}

/// Spawn threads that each accumulate into their own buffer, then sum them.
fn par_reduce<F>(n: usize, buf_len: usize, f: &F) -> Vec<f32>
where
    F: Fn(usize) -> Vec<f32> + Sync,
{
    let mut out = vec![0.0f32; buf_len];
    if n < PAR_THRESHOLD || thread_count() == 1 {
        for i in 0..n {
            for (o, s) in out.iter_mut().zip(f(i)) {
                *o += s;
            }
        }
        return out;
    }
    let ranges = work_ranges(n);
    let partials: Vec<Vec<f32>> = thread::scope(|s| {
        ranges
            .iter()
            .map(|&(lo, hi)| {
                s.spawn(move || {
                    let mut buf = vec![0.0f32; buf_len];
                    for i in lo..hi {
                        for (o, s) in buf.iter_mut().zip(f(i)) {
                            *o += s;
                        }
                    }
                    buf
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect()
    });
    for p in partials {
        for (o, s) in out.iter_mut().zip(p) {
            *o += s;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// matmul: y = x @ wt + bias, wt stored (in, out).
// ---------------------------------------------------------------------------

/// `y (M,N) = x (M,K) @ wt (K,N) + bias(N)`
pub fn matmul_fwd(x: &[f32], wt: &[f32], bias: Option<&[f32]>, m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(x.len(), m * k, "x shape");
    assert_eq!(wt.len(), k * n, "wt shape");
    if let Some(b) = bias {
        assert_eq!(b.len(), n, "bias shape");
    }
    let mut y = vec![0.0f32; m * n];
    if let Some(b) = bias {
        for row in y.chunks_exact_mut(n) {
            row.copy_from_slice(b);
        }
    }
    par_rows(&mut y, m, n, &|yr, (lo, _hi)| {
        for i in 0..(yr.len() / n) {
            let row = lo + i;
            let xr = &x[row * k..(row + 1) * k];
            let out = &mut yr[i * n..(i + 1) * n];
            for (kk, &a) in xr.iter().enumerate() {
                if a == 0.0 {
                    continue;
                }
                let wr = &wt[kk * n..(kk + 1) * n];
                for (ov, &wv) in out.iter_mut().zip(wr) {
                    *ov += a * wv;
                }
            }
        }
    });
    y
}

/// Backward through `y = x @ wt`: returns `(dx (M,K), dwt (K,N))` with
/// `dx = gy @ wtᵀ` and `dwt += xᵀ @ gy`.
pub fn matmul_bwd(x: &[f32], wt: &[f32], gy: &[f32], m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(gy.len(), m * n, "gy shape");
    let mut dx = vec![0.0f32; m * k];
    par_rows(&mut dx, m, k, &|dxr, (lo, _hi)| {
        for i in 0..(dxr.len() / k) {
            let row = lo + i;
            let gr = &gy[row * n..(row + 1) * n];
            let dx_row = &mut dxr[i * k..(i + 1) * k];
            for (kk, dv) in dx_row.iter_mut().enumerate() {
                let wr = &wt[kk * n..(kk + 1) * n];
                *dv = gr.iter().zip(wr).map(|(&g, &w)| g * w).sum();
            }
        }
    });

    // dwt: per-row outer products accumulated in per-thread buffers, then
    // reduced on the main thread.
    let dwt = par_reduce(m, k * n, &|row| {
        let xr = &x[row * k..(row + 1) * k];
        let gr = &gy[row * n..(row + 1) * n];
        let mut buf = vec![0.0f32; k * n];
        for (kk, &a) in xr.iter().enumerate() {
            if a == 0.0 {
                continue;
            }
            let drow = &mut buf[kk * n..(kk + 1) * n];
            for (dv, &g) in drow.iter_mut().zip(gr) {
                *dv += a * g;
            }
        }
        buf
    });
    (dx, dwt)
}

// ---------------------------------------------------------------------------
// LayerNorm over the last dimension.
// ---------------------------------------------------------------------------

/// Row-wise LayerNorm: `y = (x - mean)/sqrt(var + eps) * g + b`.
pub fn layernorm_fwd(x: &[f32], g: &[f32], b: &[f32], rows: usize, c: usize, eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), rows * c);
    let mut y = vec![0.0f32; rows * c];
    par_rows(&mut y, rows, c, &|yr, (lo, _hi)| {
        for i in 0..(yr.len() / c) {
            let row = lo + i;
            let xr = &x[row * c..(row + 1) * c];
            let mean = xr.iter().sum::<f32>() / c as f32;
            let var = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
            let rstd = 1.0 / (var + eps).sqrt();
            for (cc, (&xv, yv)) in xr.iter().zip(&mut yr[i * c..(i + 1) * c]).enumerate() {
                *yv = (xv - mean) * rstd * g[cc] + b[cc];
            }
        }
    });
    y
}

/// Backward of the above. Returns `(dx (rows,c), dg (c), db (c))`.
pub fn layernorm_bwd(x: &[f32], gy: &[f32], g: &[f32], rows: usize, c: usize, eps: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(x.len(), rows * c);
    assert_eq!(gy.len(), rows * c);
    let mut dx = vec![0.0f32; rows * c];
    let inv_c = 1.0 / c as f32; // retained for API symmetry; math below exact
    let _ = inv_c;

    if rows < PAR_THRESHOLD || thread_count() == 1 {
        let (dx_part, red) = layernorm_rows(x, gy, g, (0, rows), c, eps);
        dx.copy_from_slice(&dx_part);
        return (dx, red.0, red.1);
    }
    let ranges = work_ranges(rows);
    thread::scope(|s| {
        let mut rest = dx.as_mut_slice();
        let mut rest_x = x;
        let mut rest_gy = gy;
        let mut handles = Vec::new();
        for &(lo, hi) in &ranges {
            let len = (hi - lo) * c;
            let (_chunk, tail) = rest.split_at_mut(len);
            let (x_part, tail_x) = rest_x.split_at(len);
            let (gy_part, tail_gy) = rest_gy.split_at(len);
            rest = tail;
            rest_x = tail_x;
            rest_gy = tail_gy;
            handles.push(s.spawn(move || layernorm_rows(x_part, gy_part, g, (lo, hi), c, eps)));
        }
        let mut dg = vec![0.0f32; c];
        let mut db = vec![0.0f32; c];
        let mut base = 0usize;
        for h in handles {
            let (dx_part, (dg_p, db_p)) = h.join().unwrap();
            dx[base..base + dx_part.len()].copy_from_slice(&dx_part);
            base += dx_part.len();
            for (a, bp) in dg.iter_mut().zip(dg_p) {
                *a += bp;
            }
            for (a, bp) in db.iter_mut().zip(db_p) {
                *a += bp;
            }
        }
        (dx, dg, db)
    })
}

/// Worker for `layernorm_bwd`: computes dx for the given row range plus
/// partial dg/db reductions. Returns `(dx_snippet, (dg_partial, db_partial))`.
fn layernorm_rows(x: &[f32], gy: &[f32], g: &[f32], rows: (usize, usize), c: usize, eps: f32) -> (Vec<f32>, (Vec<f32>, Vec<f32>)) {
    let (lo, hi) = rows;
    let n_rows = hi - lo;
    let mut dx = vec![0.0f32; n_rows * c];
    let mut dg = vec![0.0f32; c];
    let mut db = vec![0.0f32; c];
    for i in 0..n_rows {
        let xr = &x[i * c..(i + 1) * c];
        let gr = &gy[i * c..(i + 1) * c];
        let dxr = &mut dx[i * c..(i + 1) * c];
        let mean = xr.iter().sum::<f32>() / c as f32;
        let var = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
        let rstd = 1.0 / (var + eps).sqrt();
        let mut m1 = 0.0f32; // Σ gy·g
        let mut m2 = 0.0f32; // Σ gy·g·xhat
        for (cc, &xv) in xr.iter().enumerate() {
            let xhat = (xv - mean) * rstd;
            let gyg = gr[cc] * g[cc];
            m1 += gyg;
            m2 += gyg * xhat;
            dg[cc] += gr[cc] * xhat;
            db[cc] += gr[cc];
        }
        for (cc, &xv) in xr.iter().enumerate() {
            let xhat = (xv - mean) * rstd;
            dxr[cc] = rstd * gr[cc] * g[cc] - rstd / c as f32 * (m1 + xhat * m2);
        }
    }
    (dx, (dg, db))
}

// ---------------------------------------------------------------------------
// GeLU (tanh approximation).
// ---------------------------------------------------------------------------

pub fn gelu_fwd(x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    const C0: f32 = 0.044_715;
    const C1: f32 = 0.797_884_6; // sqrt(2/π), rounded for f32
    let y = x
        .iter()
        .map(|&v| {
            let u = C1 * (v + C0 * v * v * v);
            0.5 * v * (1.0 + u.tanh())
        })
        .collect();
    (y, x.to_vec())
}

/// Backward of GeLU given its saved input.
pub fn gelu_bwd(x: &[f32], dy: &[f32]) -> Vec<f32> {
    const C0: f32 = 0.044_715;
    const C1: f32 = 0.797_884_6;
    let mut dx = vec![0.0f32; x.len()];
    for (i, (&v, &d)) in x.iter().zip(dy).enumerate() {
        let u = C1 * (v + C0 * v * v * v);
        let t = u.tanh();
        let sech2 = 1.0 - t * t;
        let dudx = C1 * (1.0 + 3.0 * C0 * v * v);
        dx[i] = d * (0.5 * (1.0 + t) + 0.5 * v * sech2 * dudx);
    }
    dx
}

// ---------------------------------------------------------------------------
// Softmax (rows) and softmax cross-entropy.
// ---------------------------------------------------------------------------

/// Numerically stable row softmax. Returns `(probs, saved_input)`.
pub fn softmax_fwd(x: &[f32], rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(x.len(), rows * cols);
    let mut p = vec![0.0f32; rows * cols];
    par_rows(&mut p, rows, cols, &|pr_all, (lo, _hi)| {
        for i in 0..(pr_all.len() / cols) {
            let row = lo + i;
            let xr = &x[row * cols..(row + 1) * cols];
            let pr = &mut pr_all[i * cols..(i + 1) * cols];
            let mx = xr.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut z = 0.0f32;
            for (pv, &xv) in pr.iter_mut().zip(xr) {
                let e = (xv - mx).exp();
                *pv = e;
                z += e;
            }
            for pv in pr.iter_mut() {
                *pv /= z;
            }
        }
    });
    (p, x.to_vec())
}

/// Backward of softmax: `dx = p * (dy - Σ_j dy_j p_j)` per row.
/// (Kept for API completeness; the equivalent update is fused into
/// `cross_entropy` for speed.)
#[allow(dead_code)]
pub fn softmax_bwd(p: &[f32], gy: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(p.len(), rows * cols);
    let mut dx = vec![0.0f32; rows * cols];
    par_rows(&mut dx, rows, cols, &|dxr_all, (lo, _hi)| {
        for i in 0..(dxr_all.len() / cols) {
            let row = lo + i;
            let pr = &p[row * cols..(row + 1) * cols];
            let gr = &gy[row * cols..(row + 1) * cols];
            let dot: f32 = pr.iter().zip(gr).map(|(&a, &b)| a * b).sum();
            let dxr = &mut dxr_all[i * cols..(i + 1) * cols];
            for (cc, (pv, &g)) in pr.iter().zip(gr).enumerate() {
                dxr[cc] = pv * (g - dot);
            }
        }
    });
    dx
}

/// Softmax cross-entropy over `(rows, vocab)` logits with integer targets.
/// Returns `(mean_loss, dlogits (rows,vocab), probs (rows,vocab))`.
pub fn cross_entropy(logits: &[f32], targets: &[usize], rows: usize, vocab: usize) -> (f32, Vec<f32>, Vec<f32>) {
    assert_eq!(logits.len(), rows * vocab, "logits shape");
    assert_eq!(targets.len(), rows, "targets shape");
    let (p, _saved) = softmax_fwd(logits, rows, vocab);
    let mut loss = 0.0f64;
    for r in 0..rows {
        loss += -p[r * vocab + targets[r]].max(f32::EPSILON).ln() as f64;
    }
    let loss = (loss / rows as f64) as f32;
    let mut dlogits = p.clone();
    let scale = 1.0 / rows as f32;
    for r in 0..rows {
        dlogits[r * vocab + targets[r]] -= 1.0;
    }
    for v in dlogits.iter_mut() {
        *v *= scale;
    }
    (loss, dlogits, p)
}

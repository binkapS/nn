# nn — Performance Plan: CPU Tuning First, Then wgpu

Status marker legend: `[ ]` todo · `[~]` in progress · `[x]` done.

## Why

The default run (5000 steps, 4 layers, d_model 128, ctx 128) takes ~1.5–2 h on
this 4-core laptop (i5-6300U, Intel HD 520 iGPU). The engine converts only a
small fraction of available throughput (~870 tokens/sec ≈ 8.5 GFLOP/s
effective). Two acceleration routes exist:

* **wgpu directly** — hand-written WGSL kernels, same "everything is ours"
  ethos as the CPU engine, numeric parity verified against the CPU reference.
* **Burn (framework)** — rejected: autodiff/backends become library code, the
  project's purpose (hand-written gradients) evaporates, and generic kernels
  may not even beat tuned CPU code at our tiny sizes (d=128, B·T=1024).

**Decision: wgpu route**, with a permanent CPU fallback. Honest caveat: our
matrices are small, so GPU gains come from *fusing* kernels, not raw GEMM
speed. Target hardware: HD 520 iGPU via Mesa Vulkan (ANV); lavapipe exists as
a diagnostic fallback only.

## Ground rules

* Every step keeps `cargo test` green and `clippy` warning-free.
* One commit per stage item (or small cluster), so each is independently
  revertable.
* CPU engine stays the reference forever: `--backend cpu|gpu|auto`, default
  `auto` prefers GPU but falls back to CPU with a warning on any adapter
  failure. `--backend cpu` must reproduce current behavior exactly.
* Numeric gates before integration: each GPU op passes a
  `gpu ≈ cpu` parity test (fp32, rel. tol ~1e-3) and the final full-model
  loss curves (CPU vs GPU, same seed) must track within noise.

## Stage 0 — CPU baseline worth having

* [x] **0.1 Profile & record baseline.** Reproduce current numbers as a table
      in this file: tokens/sec at default config, per-op share (measure a few
      isolated hot paths via `Instant` around matmul/attention/optimizer).
* [x] **0.2 Kill per-step allocations.** Audit `ops.rs`/`modules.rs`/
      `attention.rs` for `Vec::with_capacity`/clone-per-step; hoist scratch
      buffers into long-lived caches (threads or model-owned) where trivially
      profitable. No unsafe, no behavior change.
* [x] **0.3 GEMM loop order + register blocking.** Packed `k`-inner panels,
      4×4 register blocking, contiguous access on both operands; keep
      parallel row-chunk structure. Target: +1.5× on matmul microbenchmark.
* [x] **0.4 Re-baseline.** Re-run default-config benchmark; update the table
      here; commit. Gate: measurable win, tests green. (These wins persist
      regardless of what Stage 2 concludes.)

## Stage 1 — wgpu scaffold behind a `Backend` trait

* [x] **1.1 Add wgpu dep; adapter probe.** Startup lists adapters, selects
      `backend auto` (Vulkan/ANV preferred), warns + falls back to CPU on
      failure. Flag implemented and tested (adapter choice printed to stderr).
* [x] **1.2 Parity harness.** Test helper comparing a closure-computed CPU op
      vs a submitted GPU buffer (shape, rel-err, max-abs-err reporting). Used
      by every Stage 2 item.

## Stage 2 — Port ops one at a time (ordered by payoff)

Each item = WGSL kernel(s) + parity test + microbenchmark line here + commit.

* [x] **2.1 Matmul fwd + fused dgrad/wgrad backward** (biggest win; mirrored
      access patterns, transposed tiles).
* [x] **2.2 LayerNorm fwd/bwd, GeLU fwd/bwd** (elementwise, cheap).
* [x] **2.3 AdamW step kernel** (elementwise update, biases/LN scales
      excluded from decay as on CPU).
* [ ] **2.4 Cross-entropy softmax fwd/bwd.**
* [ ] **2.5 Causal attention fwd** (softmax + causal mask in one pass) **and
      backward** (`ds = p·(ṗ − Σṗp)/√d`).
* [ ] **2.6 Embedding scatter/gather** fwd/bwd.
* [ ] **2.7 Residency pass.** All activations/weights stay on device across
      steps; host transfers limited to the loss scalar and sampled tokens at
      generate time.

## Stage 3 — End-to-end gate

* [ ] **3.1 Training loop on GPU**: same schedule/init/seed; CPU-vs-GPU loss
      curves tracked within noise.
* [ ] **3.2 CLI wiring**: `--backend` in usage/help; graceful fallback tested.
* [ ] **3.3 Acceptance**: GPU end-to-end ≥ 2× Stage 0 baseline on the HD 520
      at default config; all tests pass on both backends. If the iGPU
      disappoints: bank Stage 0 wins, reassess together (options: iGPU
      overhead autopsy, bigger default model, or accept CPU-tuned status quo).

## Benchmark log

Shapes: B=8, T=128, C=128, V=155, L=4 (default config); microbench via
`cargo test --release -- --ignored --nocapture perf_baseline`.
Per-block means the cost of ONE transformer block (the model stacks L=4).

### Baseline (before Stage 0)

| Op (per block unless noted)   | ms | GFLOP/s | Notes |
|-------------------------------|------|-------|-------|
| matmul_fwd[qkv]   1024×128×384 | 7.87  | 12.8    | |
| matmul_bwd[qkv]               | 56.33 | 2.7     | per-row dwt alloc |
| matmul_fwd[proj]  1024×128×128 | 3.90  | 8.6     | |
| matmul_bwd[proj]              | 20.89 | 2.4     | |
| matmul_fwd[fc1]   1024×128×512 | 10.03 | 13.4    | |
| matmul_bwd[fc1]               | 77.05 | 2.6     | |
| matmul_fwd[fc2]   1024×512×128 | 13.94 | 9.6     | |
| matmul_bwd[fc2]              | 74.81 | 2.7     | |
| attention.fwd                 | 46.60 | 2.9     | serial b×h loop |
| attention.fwd+bwd             | 170.86| 2.0     | serial b×h loop |
| adamw+clip (whole model)      | 3.51  | —       | |

Step estimate from these numbers: ~1.75 s/step (~870 tokens/sec live).
Diagnosis: (1) `par_reduce` per-row `k*n` buffer allocs ≈ GB of zeroing per
step; (2) attention serialized its b×h loop on one thread; (3) forward GEMM
4× headroom via panel blocking; (4) per-step attention scratch allocs.

### After Stage 0 (same test, dedicated measurements)

| Op (per block unless noted)   | ms | GFLOP/s | Change |
|-------------------------------|------|-------|--------|
| matmul_fwd[qkv]               | 8.9  | 11.2   | 8-row panels (± noise vs 4-row) |
| matmul_bwd[qkv]               | 18.4 | 8.2    | dx = transposed-GEMM (was dot loop), dwt once-per-thread buffer |
| matmul_fwd[proj]              | 3.3  | 10.1   | |
| matmul_bwd[proj]              | 7.0  | 7.2    | |
| matmul_fwd[fc1]               | 11.9 | 11.3   | |
| matmul_bwd[fc1]               | 24.9 | 8.1    | |
| matmul_fwd[fc2]               | 9.5  | 14.1   | |
| matmul_bwd[fc2]              | 27.9 | 7.2    | |
| attention.fwd                 | ~19  | 7.1    | threaded by batch strip, reusable scratch |
| attention.fwd+bwd             | 65–90| 3.7–5.2 | |
| adamw+clip (whole model)      | ~4.4 | —      | |

Full `train_step` at default shapes (dedicated diag test):
**fwd 181 ms / bwd 316 ms / upd 4 ms → 513 ms/step.**

### Live confirmation (30-step c4 run, default config)

* Before: ~1.25 s/step, **~854 tokens/sec**
* After: ~0.49 s/step, **~2040 tokens/sec → 2.4× end-to-end**
* Default 5000-step run projected: **~17 min** (was ~70 min).

### Lessons learned (keep for Stage 2)

1. **Never trust a benchmark that ran on a stale binary** — `cargo test`
   rebuilds only test targets; `cargo build --release` the bin before
   believing live numbers (cost an hour of confusion: bench said 0.5 s/step,
   live 1.2 s/step because the binary predates the optimizations).
2. **`f32::mul_add` NO-GO on this target**: it lowers to scalar `fmaf`
   libcalls in these axpy loops (10× slower than plain mul+add); plain
   `mul + add` auto-vectorizes fine. If CPU-level FMA is ever needed again it
   must go through `core::arch` intrinsics with a scalar fallback — postponed
   to Stage 3 "reassess" options.
3. A stale background `nn` process (231% CPU) polluted live timings once;
   check `ps` before trusting deltas on a shared laptop.
4. 8-row GEMM panels ≈ 4-row panels within run noise; keep 8+4 (harmless).

### Stage 2.1 — GPU GEMM kernels on the HD 520 (un-fused path, per-call staging)

Note the workload-cost caveat: these GPU numbers include upload-dispatch-
readback per call; with residency (2.7) the transfers disappear, so the raw
kernel numbers are strictly better than shown. CPU numbers same-day run for
fairness (machine load makes CPU figures ±25% across runs).

| Op                    | CPU ms | CPU GF/s | GPU ms | GPU GF/s |
|-----------------------|--------|----------|--------|----------|
| fwd qkv  1024×128×384 | 11.6   | 8.7      | 11.7   | 8.6      |
| bwd qkv               | 27.5   | 5.5      | 26.5   | 5.7      |
| fwd proj 1024×128×128 | 7.8    | 4.3      | 9.6    | 3.5      |
| bwd proj              | 15.7   | 3.2      | 19.0   | 2.7      |
| fwd fc1  1024×128×512 | 19.2   | 7.0      | 15.3   | 8.8      |
| bwd fc1               | 42.3   | 4.8      | 33.4   | 6.0      |
| fwd fc2  1024×512×128 | 17.6   | 7.6      | 14.6   | 9.2      |
| bwd fc2               | 41.2   | 4.9      | 32.5   | 6.2      |

Parity (vs CPU, max rel err): fwd ≤ 3.3e-5, dx ≤ 1.6e-5, dwt ≤ 6.2e-5 —
all under the 2e-3 gate, including ragged 5×7×6 / 17×33×29 / 1×1×1 shapes
(with and without bias) that exercise every OOB guard lane.

Stage 2.1 lessons:
5. **WGSL workgroup staging trap**: a two-coordinate shared tile must not be
   staged via `lid.x` when the accumulation loop's chunk index has no thread
   dimension to live in — 16 threads with different `gid.y` write different
   values to the same `tile_x[lid.x]` slot (silent corruption, rel err 2.0).
   The dwt kernel dropped staging entirely (thread-accumulate from global,
   L1 absorbs the redundancy); the fwd kernel keeps staging because its
   chunk index `k` rides the *shared* x/w tile coordinates.
6. GPU ≈ CPU already on the big GEMM shapes *with* staging overhead —
   the residency pass (2.7) is likely worth more than kernel micro-tuning.
7. **Elementwise ops on GPU lose to CPU at these sizes (expected)**: at
   rows=1024×c=128 (~0.25 MB) the un-fused GPU path pays 1–2 buffer-create
   + upload roundtrips per call: ln.fwd 6.2 ms vs CPU 1.6 ms, ln.bwd 22.5 ms
   vs 1.9 ms (3 readbacks), gelu pair ~1.0×. This is why 2.2 kernels are
   single-pass and fused-ready; the payoff arrives when they chain on-DPU
   (fused blocks in 2.2's design) and with 2.7 residency.

### Stage 2.2 — LayerNorm + GeLU GPU kernels

Parity (vs CPU, max rel err within the 2e-3 gate) — shapes 1024×128,
17×33, 5×7, 1×128, 3×1024 (stride-loop lanes), GeLU over sign/magnitude
ranges incl. n=1000 ragged and n=1:

| op        | max rel err |
|-----------|-------------|
| ln.fwd    | 8.3e-7      |
| ln.bwd dx | 5.4e-7      |
| ln.bwd dg | 2.8e-5      |
| ln.bwd db | 4.8e-6      |
| gelu.fwd  | 2.9e-7      |
| gelu.bwd  | 3.3e-6      |

Kernel notes:
* `ln_fwd`/`ln_bwd`: one row per 256-thread workgroup; per-lane partials in
  workgroup memory + barrier tree-reduce; population variance clamped at 0
  (E[x²]−mean² can dip below zero in f32; CPU's Σ(x−m)² form can't).
* `ln_bwd` emits per-row dg/db partials into a (rows×c) buffer, then a
  generic `reduce_cols` kernel finishes them (WebGPU has no f32 atomics).
* Lesson #6 (this round): `global_invocation_id.x` is the GLOBAL thread
  index (row·WG + lane), not the workgroup index — one-row-per-workgroup
  kernels must use `@builtin(workgroup_id)`. Symptom: rel err 1.0 with
  garbage everywhere on rows ≥ 1 (the kernel read the wrong row).
* Lesson #7: WGSL float literals reject `_` separators mid-mantissa
  (`0.797_884_6` → parse error); write `0.7978846`.

### Stage 2.3 — AdamW + gradient-norm/clip GPU kernels

Parity (vs CPU, 5-step sequences incl. wd=0 and wd>0, ragged n=257/1000):

| check                | max rel err |
|----------------------|-------------|
| adamw params (steps 1–5, 3 configs) | ≤ 3.7e-9 |
| grad ‖·‖₂ (200k elems) | 0 (f64 host fold of f32 partials) |
| scale_grads ×0.1118  | 0 |

Un-fused perf (the 2.7 story in one number): full-model 847k-param AdamW
step = CPU 2.6 ms vs GPU 46.8 ms — because the host ships ~20 MB of
p/m/v/grad roundtrips per call across the staging path. The kernel itself
is a single ~12 MB on-device pass; with residency these transfers vanish
and the elementwise op becomes bandwidth-trivial even on the iGPU.
sqnorm reduction incl. staging: 6.3 ms.

Notes:
* AdamW needs *duplex* buffers (`STORAGE|COPY_DST|COPY_SRC`) for every
  in-out tensor — plain read-only uploads can't be copied back out
  (wgpu validation error caught by the parity test).
* The `powf` bias corrections stay host-side (no powf in WGSL core);
  `bc1/bc2` arrive in the uniform block.

### Stage 2.4 — softmax cross-entropy GPU kernels

Parity (vs CPU `ops::cross_entropy`, max rel err within the 2e-3 gate) —
shapes 1024×155 (default), 17×33, 5×7, 1×128, 3×1024, and ±30-magnitude
logits exercising max-shift stability + the underflow clamp:

| check    | max rel err |
|----------|-------------|
| probs    | 1.2e-7      |
| dlogits  | 1.9e-9      |
| mean loss| 0 (identical f64 fold; incl. EPS-clamped underflow rows) |

Kernel design:
* `ce_fwd`: one row per workgroup, four passes over the row (max → Σexp →
  probs → per-row loss written by lane 0 into a (rows,) buffer); host folds
  the mean in f64 mirroring the CPU exactly.
* `ce_bwd`: elementwise `(p − onehot)/rows` with u32 row/col decode from
  the flat index (div/mod).
* Targets upload as a u32 storage buffer (`upload_u32_storage`); the host
  API takes `&[usize]` like the CPU and converts.
* Lesson #8: the CPU clamps the target probability at `f32::EPSILON`
  before `ln`; the kernel must too — a ±30-logit sweep sends p_target to
  exact 0, and `−log(0) = +inf` made the mean loss inf while all 128k
  probs still agreed to 1e-7. Loss reduction paths are as numerically
  load-bearing as the forward math.

Un-fused perf at 1024×155: CPU 3.4 ms vs GPU 18.3 ms — two ~1.3 MB
readbacks (probs + dlogits) + one (rows,) loss dominate; residency (2.7)
turns this into a striped on-device pass.

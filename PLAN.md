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

* [ ] **0.1 Profile & record baseline.** Reproduce current numbers as a table
      in this file: tokens/sec at default config, per-op share (measure a few
      isolated hot paths via `Instant` around matmul/attention/optimizer).
* [ ] **0.2 Kill per-step allocations.** Audit `ops.rs`/`modules.rs`/
      `attention.rs` for `Vec::with_capacity`/clone-per-step; hoist scratch
      buffers into long-lived caches (threads or model-owned) where trivially
      profitable. No unsafe, no behavior change.
* [ ] **0.3 GEMM loop order + register blocking.** Packed `k`-inner panels,
      4×4 register blocking, contiguous access on both operands; keep
      parallel row-chunk structure. Target: +1.5× on matmul microbenchmark.
* [ ] **0.4 Re-baseline.** Re-run default-config benchmark; update the table
      here; commit. Gate: measurable win, tests green. (These wins persist
      regardless of what Stage 2 concludes.)

## Stage 1 — wgpu scaffold behind a `Backend` trait

* [ ] **1.1 Add wgpu dep; adapter probe.** Startup lists adapters, selects
      `backend auto` (Vulkan/ANV preferred), warns + falls back to CPU on
      failure. Flag implemented and tested (adapter choice printed to stderr).
* [ ] **1.2 Parity harness.** Test helper comparing a closure-computed CPU op
      vs a submitted GPU buffer (shape, rel-err, max-abs-err reporting). Used
      by every Stage 2 item.

## Stage 2 — Port ops one at a time (ordered by payoff)

Each item = WGSL kernel(s) + parity test + microbenchmark line here + commit.

* [ ] **2.1 Matmul fwd + fused dgrad/wgrad backward** (biggest win; mirrored
      access patterns, transposed tiles).
* [ ] **2.2 LayerNorm fwd/bwd, GeLU fwd/bwd** (elementwise, cheap).
* [ ] **2.3 AdamW step kernel** (elementwise update, biases/LN scales
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

| Config                     | Date       | Tokens/sec | Notes                |
|----------------------------|------------|------------|----------------------|
| default (5000/4/128/128) CPU | 2026-10-31 | ~870       | pre-optimization baseline |

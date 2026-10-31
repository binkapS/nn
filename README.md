# nn — a tiny GPT (decoder-only Transformer) in Rust

A complete, **working** neural network built on the Transformer design,
implemented from scratch with **zero dependencies**. Every component —
tokenization, embeddings, attention, all backward passes, AdamW — is
hand-written in this crate.

## What it is

A small GPT-style **decoder-only** language model trained with next-token
prediction on character level:

```
tokens → token + position embeddings
       → N × Transformer block (pre-LayerNorm):
     │     x₁ = x + CausalMultiHeadSelfAttention(LN(x))
     │     x₂ = x₁ + MLP(GeLU)(LN(x₁))
       → final LayerNorm
       → linear head → logits (next-char distribution)
       → softmax cross-entropy loss
```

Every backward pass (matmul, LayerNorm, GeLU, softmax/SDPA attention,
cross-entropy, embedding scatter) is implemented analytically and **verified
against central finite differences** by the test suite (`cargo test`).

## Usage

```bash
cargo run --release                      # train on the built-in demo corpus
cargo test  --release                    # run gradient checks + training smoke test
```

Train on your own text and tune hyperparameters:

```bash
cargo run --release -- \
  --data mybook.txt \
  --steps 2000 --layers 4 --heads 4 --embd 128 --block 128 \
  --batch 16 --lr 1e-3 --wd 0.1 \
  --temperature 0.8 --top-k 16 --prompt "once upon a "
```

| Flag             | Default   | Meaning                                    |
|------------------|-----------|--------------------------------------------|
| `--data`         | demo text | Text file to train on                       |
| `--steps`        | 600       | Training steps                              |
| `--layers`       | 3         | Transformer blocks                          |
| `--heads`        | 3         | Attention heads per block                   |
| `--embd`         | 96        | Model dimension `d_model`                   |
| `--block`        | 64        | Context length `T`                          |
| `--batch`        | 8         | Batch size `B`                              |
| `--lr`           | 1.5e-3    | Peak AdamW learning rate                    |
| `--wd`           | 0.1       | Decoupled weight decay (AdamW)              |
| `--temperature`  | 0.8       | Sampling temperature (0 = greedy)           |
| `--top-k`        | 12        | Top-k filtering (0 = off)                   |
| `--max-new`      | 320       | Characters to generate per sample           |
| `--prompt`       | "the "    | Prompt for the first generated sample       |
| `--seed`         | 42        | RNG seed (weights + batches + sampling)     |
| `--print-every`  | 20        | Progress interval                           |
| `--eval-every`   | 100       | Validation-loss interval                    |

## Example run

```
vocabulary    : 32 chars          parameters : 346 688
architecture  : 3 layers, 3 heads, d_model = 96, ctx = 64
step      loss (train/val)
   0            3.4564/3.4295
 300            1.8005/2.2555
 599            1.5100/2.3335
―― sample, prompt = "the river " (temp = 0.8) ――
the river wach wheresh as arur
ane thealed gale the the tolm lomeroned ...
```

(The bundled corpus is only ~3.2k characters, so the model learns its style
quickly and starts overfitting; point `--data` at a real book for better
results. As a rule of thumb, char-level models want ≥100k characters.)

## Implementation notes

* **Layouts** — activations are flat `(B·T, C)` row-major buffers; weights of
  linear layers are stored transposed `(in, out)` so all inner loops are
  contiguous.
* **Attention** — one fused QKV matmul per block; per-(b, head) causal softmax
  with the mask folded into the score loop; backward implements
  `ds = p·(ṗ − Σ ṗp)/√d` with `ṗ_{t,s} = dout_t·v_s`.
* **Parallelism** — matmuls, LayerNorm, softmax and GeLU run on all cores via
  `std::thread::scope` over disjoint row chunks (weight-gradient accumulation
  uses per-thread buffers reduced after join).
* **Optimizer** — AdamW (decoupled decay, bias-corrected moments, β₂ = 0.95),
  global-norm gradient clipping at 1.0, linear warmup (5 %) + cosine decay to
  10 % of peak. Each module owns its gradients and moments, so no global
  parameter registry is needed.
* **Initialization** — normal(0, 0.02) everywhere, with the residual output
  projections (attention `c_proj`, MLP `fc2`) scaled by `1/√(2·n_layers)`
  (GPT-2 style). Biases and LayerNorm scales are excluded from weight decay.
* **Reproducibility** — a single xorshift64* RNG seeds weight init, batch
  sampling and text generation; runs are deterministic per `--seed`.

## Source layout

| File                  | Contents                                        |
|-----------------------|-------------------------------------------------|
| `src/ops.rs`          | matmul/LayerNorm/GeLU/softmax/CE, fwd+bwd, threading |
| `src/modules.rs`      | Linear / LayerNorm / Embedding + AdamW state    |
| `src/attention.rs`    | causal multi-head self-attention + backward     |
| `src/transformer.rs`  | Transformer block, GPT model, training step, sampling |
| `src/data.rs`         | char tokenizer, train/val split, batch sampler  |
| `src/corpus.rs`       | built-in demo corpus                            |
| `src/rng.rs`          | xorshift64* PRNG + Box–Muller normal sampler    |
| `src/main.rs`         | CLI, training loop, schedule, evaluation, generation |
| `src/tests.rs`        | finite-difference gradient checks + smoke test  |

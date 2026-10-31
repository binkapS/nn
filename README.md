# nn — a tiny GPT (decoder-only Transformer) in Rust

A complete, **working** neural network built on the Transformer design,
implemented from scratch with **zero dependencies**. Every component —
tokenization, embeddings, attention, all backward passes, AdamW — is
hand-written in this crate. It trains on text pulled straight from a
**Hugging Face dataset** (default: the classic *TinyShakespeare*).

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
cargo run --release                      # train on Hugging Face TinyShakespeare
cargo test  --release                    # run gradient checks + training smoke test
```

Datasets are resolved in this order:

1. `--hf <alias|URL>` — fetch a Hugging Face dataset (see table below) and
   cache it in `data/`; subsequent runs reuse the cache offline.
2. `--data FILE` — train on any local text file.
3. Otherwise: the cached `data/tiny_shakespeare.txt` if present, else the
   tiny bundled demo corpus (fully offline fallback).

Train on your own dataset and tune hyperparameters:

```bash
cargo run --release -- \
  --hf shakespeare \
  --steps 2000 --layers 3 --heads 3 --embd 96 --block 64 \
  --batch 8 --lr 1.5e-3 --wd 0.1 \
  --temperature 0.8 --top-k 12 \
  --prompt "to be "
```

| Flag             | Default   | Meaning                                    |
|------------------|-----------|--------------------------------------------|
| `--hf`           | none      | HF dataset alias (`shakespeare`) or direct `https://…/resolve/main/…` URL, cached in `data/` |
| `--data`         | cached/defaulted | Local text file to train on instead (mutually exclusive with `--hf`) |
| `--help`, `-h`   | —         | Show full usage with defaults and exit     |
| `--steps`        | 2000      | Training steps                              |
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

## Available HF datasets

| Alias          | HF dataset                       | Contents                          |
|----------------|----------------------------------|-----------------------------------|
| `shakespeare`  | `SamPIngram/tinyshakespeare`     | 1.1 MB · canon tiny-shakespeare char-LM corpus (40 k lines of Shakespeare) |

Any direct HF resolve URL works too, e.g.:

```bash
cargo run --release -- --hf "https://huggingface.co/datasets/<user>/<repo>/resolve/main/<file>.txt"
```

### Fetch semantics (robust by design)

* **No panics on bad data sources.** `fetch_hf` returns a `Result`: a missing
  `curl`, network/DNS failure, HTTP error or empty download becomes a warning
  on stderr and a **graceful fallback** to a local dataset (valid cache →
  `--data` file → bundled demo corpus). Passing an unknown alias errors with
  the list of known aliases.
* **Validated cache.** Every successful download writes an integrity marker
  (`<file>.mini`) with the exact byte count. A cache file is reused only when
  the marker matches the content — truncated or unmarked files (e.g. an
  interrupted download) are silently re-fetched.
* **Deterministic cache names.** Direct URLs are slugged from their parsed
  `host/path` segments (sanitized, capped) plus an 8-hex FNV-1a hash of the
  full URL, so different URLs can never collide on one cache file.
* **CWD-independent paths.** `data/` is resolved against the CWD first, then
  relative to the executable (covers `cargo run` and `target/release`
  layouts), then created on demand — so the binary works from any directory.

## Sampling prompts

The generation showcase is dataset-agnostic: your `--prompt` goes first,
then up to three prompts are mined from corpus word frequency (e.g. for
Shakespeare the frequent-word prompts adapt automatically; the bundled demo
corpus yields its own). No Shakespeare-specific hardness.

## Example run

```
dataset       : data/tiny_shakespeare.txt (HuggingFace: SamPIngram/tinyshakespeare)
vocabulary    : 65 chars          parameters : 353 057
architecture  : 3 layers, 3 heads, d_model = 96, ctx = 64
step      loss (train/val)
   0            4.2131/4.2036
1000            2.2782/2.1450
1999            1.9266/1.9896
final: train 1.9361 · val 2.0648 · perplexity ≈ 7.88 · total time 595.4s

―― sample, prompt = "the " (temp = 0.8, top-k = 12) ――
the of
Orias ming the thour the in tis whull is thains.
Thou not ay tromer, thy, but men the chell the aspit
Anding ase oof die of but it be hom.

LEOS:
Sty, mart you forn sour thel bleass
The pare with maged his and the shhouldy wert the cring wome,
...
```

The model learns the *shape* of Shakespeare — speaker labels (`LEOS:`),
`thou/thee/thy/hath`, verse-and-blank structure, punctuation — within the
default 2000 steps (~10 min on a 4-core laptop). For cleaner text, raise
`--steps 5000`, `--block 128`, `--embd 128`, `--batch 16` (proportionally
more time; the default footprint deliberately fits a laptop CPU).

For reference: the bundled 3.2 k-character demo corpus (offline fallback)
converges in a few hundred steps by design.

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
* **Optimizer** — AdamW in the conventional form `θ ← θ − lr·(m̂/(√v̂+ε) + wd·θ)`
  (decoupled decay, bias-corrected moments, β₂ = 0.95), global-norm gradient
  clipping at 1.0, linear warmup (5 %) + cosine decay to 10 % of peak. Each
  module owns its gradients and moments, so no global parameter registry is
  needed.
* **Initialization** — normal(0, 0.02) everywhere, with the residual output
  projections (attention `c_proj`, MLP `fc2`) scaled by `1/√(2·n_layers)`
  (GPT-2 style). Biases and LayerNorm scales are excluded from weight decay.
* **Reproducibility** — a single xorshift64* RNG seeds weight init, batch
  sampling and text generation; runs are deterministic per `--seed`. The
  Box–Muller sampler caches the second variate of each draw pair, so no
  Gaussian samples are wasted.

## Source layout

| File                  | Contents                                        |
|-----------------------|-------------------------------------------------|
| `src/ops.rs`          | matmul/LayerNorm/GeLU/softmax/CE, fwd+bwd, threading |
| `src/modules.rs`      | Linear / LayerNorm / Embedding + AdamW state    |
| `src/attention.rs`    | causal multi-head self-attention + backward     |
| `src/transformer.rs`  | Transformer block, GPT model, training step, sampling |
| `src/data.rs`         | char tokenizer, train/val split, batch sampler, HF fetch + validated cache, prompt mining |
| `src/corpus.rs`       | built-in offline fallback demo corpus            |
| `src/rng.rs`          | xorshift64* PRNG + polar Box–Muller normal sampler (spare variate cached) |
| `src/main.rs`         | CLI, HF/`--data` loading, training loop, schedule, eval, generation |
| `src/tests.rs`        | finite-difference gradient checks + smoke test  |

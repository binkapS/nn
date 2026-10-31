//! A tiny dependency-free GPT (decoder-only Transformer) in Rust.
//!
//! Pipeline: character tokenizer -> token+position embeddings -> N pre-LN
//! transformer blocks (causal multi-head self-attention + GeLU MLP) -> final
//! LayerNorm -> linear head -> softmax cross-entropy. Everything, including
//! every backward pass, is implemented by hand in this crate.

mod attention;
mod corpus;
mod data;
mod gpu;
mod modules;
mod ops;
mod rng;
mod transformer;

#[cfg(test)]
mod tests;

use data::{CharTokenizer, get_batch, load_text, split};
use rng::Rng;
use std::time::Instant;
use transformer::{Config, Gpt};

struct Args {
    data: Option<String>,
    hf: Option<String>,
    /// True only when `--hf` was given on the command line (vs. the built-in
    /// default). Lets `--data` silently override the default dataset while
    /// still erroring on an explicit `--hf X --data Y` conflict.
    hf_explicit: bool,
    backend: String,
    steps: usize,
    layers: usize,
    heads: usize,
    embd: usize,
    block: usize,
    batch: usize,
    lr: f32,
    wd: f32,
    temperature: f32,
    top_k: usize,
    max_new: usize,
    prompt: String,
    seed: u64,
    print_every: usize,
    eval_every: usize,
}

fn default_args() -> Args {
    Args {
        data: None,
        // Default dataset: allenai/c4 (streaming sample, see data.rs).
        hf: Some(data::DEFAULT_HF_ALIAS.to_string()),
        hf_explicit: false,
        backend: "auto".to_string(),
        steps: 5000,
        layers: 4,
        heads: 4,
        embd: 128,
        block: 128,
        batch: 8,
        lr: 1.5e-3,
        wd: 0.1,
        temperature: 0.8,
        top_k: 12,
        max_new: 320,
        prompt: "the ".to_string(),
        seed: 42,
        print_every: 20,
        eval_every: 100,
    }
}

fn parse_args() -> Args {
    let mut a = default_args();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let usage = usage();
    let mut i = 0;
    let invalid = |flag: &str, usage: &str| -> ! {
        eprintln!("error: {flag}");
        eprintln!("{usage}");
        std::process::exit(2);
    };
    while i < argv.len() {
        let (flag, val) = (argv[i].as_str(), argv.get(i + 1).map(|s| s.as_str()));
        macro_rules! next {
            () => {
                match val {
                    Some(v) => {
                        i += 1;
                        v
                    }
                    None => invalid(&format!("flag {flag} needs a value"), &usage),
                }
            };
        }
        match flag {
            "--help" | "-h" => {
                print!("{usage}");
                std::process::exit(0);
            }
            "--data" => {
                a.data = Some(next!().to_string());
                // The default --hf is silent; an explicit one conflicts.
                if !a.hf_explicit {
                    a.hf = None;
                }
            }
            "--hf" => {
                a.hf = Some(next!().to_string());
                a.hf_explicit = true;
            }
            "--backend" => {
                a.backend = next!().to_string();
            }
            "--steps" => {
                a.steps = next!()
                    .parse()
                    .unwrap_or_else(|e| invalid(&format!("--steps: {e}"), &usage))
            }
            "--layers" => {
                a.layers = next!()
                    .parse()
                    .unwrap_or_else(|e| invalid(&format!("--layers: {e}"), &usage))
            }
            "--heads" => {
                a.heads = next!()
                    .parse()
                    .unwrap_or_else(|e| invalid(&format!("--heads: {e}"), &usage))
            }
            "--embd" => {
                a.embd = next!()
                    .parse()
                    .unwrap_or_else(|e| invalid(&format!("--embd: {e}"), &usage))
            }
            "--block" => {
                a.block = next!()
                    .parse()
                    .unwrap_or_else(|e| invalid(&format!("--block: {e}"), &usage))
            }
            "--batch" => {
                a.batch = next!()
                    .parse()
                    .unwrap_or_else(|e| invalid(&format!("--batch: {e}"), &usage))
            }
            "--lr" => {
                a.lr = next!()
                    .parse()
                    .unwrap_or_else(|e| invalid(&format!("--lr: {e}"), &usage))
            }
            "--wd" => {
                a.wd = next!()
                    .parse()
                    .unwrap_or_else(|e| invalid(&format!("--wd: {e}"), &usage))
            }
            "--temperature" => {
                a.temperature = next!()
                    .parse()
                    .unwrap_or_else(|e| invalid(&format!("--temperature: {e}"), &usage))
            }
            "--top-k" => {
                a.top_k = next!()
                    .parse()
                    .unwrap_or_else(|e| invalid(&format!("--top-k: {e}"), &usage))
            }
            "--max-new" => {
                a.max_new = next!()
                    .parse()
                    .unwrap_or_else(|e| invalid(&format!("--max-new: {e}"), &usage))
            }
            "--prompt" => a.prompt = next!().to_string(),
            "--seed" => {
                a.seed = next!()
                    .parse()
                    .unwrap_or_else(|e| invalid(&format!("--seed: {e}"), &usage))
            }
            "--print-every" => {
                a.print_every = next!()
                    .parse()
                    .unwrap_or_else(|e| invalid(&format!("--print-every: {e}"), &usage))
            }
            "--eval-every" => {
                a.eval_every = next!()
                    .parse()
                    .unwrap_or_else(|e| invalid(&format!("--eval-every: {e}"), &usage))
            }
            other => invalid(&format!("unknown flag: {other}"), &usage),
        }
        i += 1;
    }
    a
}

fn usage() -> String {
    let d = default_args();
    let aliases = data::HF_DATASETS
        .iter()
        .map(|ds| ds.alias)
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "tiny GPT — a dependency-free decoder-only Transformer in Rust

TRAINING DATA
  --hf <alias|URL>   Fetch a Hugging Face dataset (aliases: {aliases}) or a direct
                     https://.../resolve/main/... URL; cached in data/ and
                     reused offline. On failure, falls back locally (cached
                     file if valid, else the bundled demo corpus).
                     Default: `{def_hf}` (allenai/c4, a ~3 MB streamed sample).
  --data <FILE>      Train on a local text file instead (overrides --hf).

MODEL / TRAINING (defaults in parentheses)
  --backend <s>      Compute engine: cpu | gpu | auto ({})
                     GPU kernels land in Stage 2 (PLAN.md); until then all
                     backends train on CPU — auto/gpu still probe hardware.
  --steps <n>        Training steps ({})
  --layers <n>       Transformer blocks ({})
  --heads <n>        Attention heads per block ({})
  --embd <n>         Model dimension d_model ({})
  --block <n>        Context length T ({})
  --batch <n>        Batch size B ({})
  --lr <f>           Peak AdamW learning rate ({:e})
  --wd <f>           Decoupled weight decay ({})
  --seed <u64>       RNG seed for init/batches/sampling ({})

SAMPLING / OUTPUT
  --temperature <f>  Sampling temperature; 0 = greedy ({})
  --top-k <n>        Top-k filtering; 0 = off ({})
  --max-new <n>      Characters generated per sample ({})
  --prompt <s>       Your prompt for the first sample; the rest are mined
                     from corpus word frequency ('{}')
  --print-every <n>  Progress interval ({})
  --eval-every <n>   Validation-loss interval ({})
  -h, --help         Show this help and exit

The data/ directory is resolved against the CWD first, then relative to the
executable (cargo run / target/release layouts), and created on demand.
",
        d.steps,
        d.layers,
        d.heads,
        d.embd,
        d.block,
        d.batch,
        d.lr,
        d.wd,
        d.seed,
        d.temperature,
        d.top_k,
        d.max_new,
        d.prompt,
        d.print_every,
        d.eval_every,
        d.backend,
        def_hf = data::DEFAULT_HF_ALIAS,
    )
}

fn main() {
    let mut args = parse_args();

    if args.data.is_some() && args.hf.is_some() {
        eprintln!("error: --data and --hf are mutually exclusive — pick one dataset source");
        eprintln!("{}", usage());
        std::process::exit(2);
    }

    // Stage 1: resolve the compute engine. GPU kernels are not ported yet
    // (Stage 2), so `auto`/`gpu` probe the adapter and report; training stays
    // on CPU until kernels land.
    let (engine, adapter_note) = match gpu::select_backend(&args.backend, false) {
        Ok(pair) => pair,
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!("{}", usage());
            std::process::exit(2);
        }
    };
    let _ = engine; // used once GPU kernels land (Stage 2)
    eprintln!("[engine] {adapter_note}");

    // Resolution order: `--hf <alias|URL>` (with graceful local fallback) >
    // `--data FILE` > cached file in data/ > bundled demo corpus.
    let (text, source_label) = match &args.hf {
        Some(hf) => match data::fetch_hf(hf) {
            Ok(pair) => pair,
            Err(e) => {
                eprintln!("[hf] WARNING: {e}");
                eprintln!("[hf] falling back to a local dataset (cached/default/built-in).");
                load_text(&args.data)
            }
        },
        None => load_text(&args.data),
    };
    args.data = Some(source_label);

    let tokenizer = CharTokenizer::build(&text);
    let (train_text, val_text) = split(&text, 0.9);
    let train_ids = tokenizer.encode(&train_text);
    let val_ids = tokenizer.encode(&val_text);

    let cfg = Config {
        n_layer: args.layers,
        n_head: args.heads,
        n_embd: args.embd,
        block_size: args.block,
        vocab_size: tokenizer.vocab_size(),
    };

    let mut rng = Rng::new(args.seed);
    let mut model = Gpt::new(cfg, &mut rng);

    println!("―――――――――――――――――――――――――――――――――――――――――――――――");
    println!("tiny GPT · decoder-only Transformer · pure Rust");
    println!("―――――――――――――――――――――――――――――――――――――――――――――――");
    println!("dataset       : {}", args.data.as_deref().unwrap_or("?"));
    println!("vocabulary    : {} chars", cfg.vocab_size);
    println!("train tokens  : {}", train_ids.len());
    println!("val tokens    : {}", val_ids.len());
    println!(
        "architecture  : {} layers, {} heads, d_model = {}, ctx = {}",
        cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.block_size
    );
    println!("batch         : {} x {} tokens", args.batch, cfg.block_size);
    println!("parameters    : {}", model.param_count());
    println!("―――――――――――――――――――――――――――――――――――――――――――――――");

    let warmup = (0.05 * args.steps as f32) as usize;
    let min_lr = args.lr * 0.1;
    let peak_lr = args.lr;
    let before = Instant::now();
    let mut last_report = Instant::now();
    let mut nchars = 0usize;

    println!("step      loss (train/val)     lr            tokens/sec");
    println!("----      ----------------     -----------   ----------");
    for step in 0..args.steps {
        let (inputs, targets) = get_batch(&mut rng, &train_ids, cfg.block_size, args.batch);
        let lr = schedule(step, args.steps, warmup, peak_lr, min_lr);
        let loss = model.train_step(&inputs, &targets, lr, args.wd, 1.0);
        nchars += inputs.len();

        if step % args.print_every == 0 || step == args.steps - 1 {
            let elapsed = last_report.elapsed().as_secs_f32();
            last_report = Instant::now();
            let tps = nchars as f32 / elapsed.max(f32::EPSILON);
            nchars = 0;
            print!("{:>4}      {:>12.4}/", step, loss);
            if args.eval_every > 0 && (step % args.eval_every == 0 || step == args.steps - 1) {
                let (vi, vt) = get_batch(&mut rng, &val_ids, cfg.block_size, args.batch);
                let vl = model.eval_loss(&vi, &vt);
                print!("{:<12.4}", vl);
            } else {
                print!("{:<12}", "–");
            }
            print!("  {:<10.6}  {:>8.0}", lr, tps);
            println!("   ( {:>5.1}s elapsed )", before.elapsed().as_secs_f32());
        }
    }

    // Final averaged validation loss over a few batches for a stable figure.
    let eval_batches = 5;
    let mut vloss = 0.0f32;
    for _ in 0..eval_batches {
        let (vi, vt) = get_batch(&mut rng, &val_ids, cfg.block_size, args.batch);
        vloss += model.eval_loss(&vi, &vt) / eval_batches as f32;
    }
    let train_shots = 5;
    let mut tloss = 0.0f32;
    for _ in 0..train_shots {
        let (ti, tt) = get_batch(&mut rng, &train_ids, cfg.block_size, args.batch);
        tloss += model.eval_loss(&ti, &tt) / train_shots as f32;
    }
    println!("―――――――――――――――――――――――――――――――――――――――――――――――");
    println!(
        "final: train {:.4} · val {:.4} · perplexity ≈ {:.2} · total time {:.1}s",
        tloss,
        vloss,
        vloss.exp(),
        before.elapsed().as_secs_f32()
    );

    // Sampling showcase: the user's prompt first, then dataset-agnostic
    // prompts mined from corpus word frequency (see data::sample_prompts).
    let prompts = data::sample_prompts(&text, &args.prompt);
    for prompt in &prompts {
        let prompt_ids = tokenizer.encode(prompt);
        if prompt_ids.is_empty() {
            continue;
        }
        let seq = model.generate(
            &prompt_ids,
            args.max_new,
            args.temperature,
            args.top_k,
            &mut rng,
        );
        println!();
        println!(
            "―― sample, prompt = {prompt:?} (temp = {}, top-k = {}) ――",
            args.temperature, args.top_k
        );
        println!("{}", tokenizer.decode(&seq));
    }
}

/// Linear warmup to `peak`, then cosine decay to `min_lr`.
fn schedule(_step: usize, total: usize, warmup: usize, peak: f32, min_lr: f32) -> f32 {
    if _step < warmup {
        peak * (_step + 1) as f32 / warmup.max(1) as f32
    } else {
        let num = _step.saturating_sub(warmup);
        let denom = total.saturating_sub(warmup).max(1);
        let p = num as f32 / denom as f32;
        let p = p.min(1.0);
        min_lr + 0.5 * (peak - min_lr) * (1.0 + (std::f32::consts::PI * p).cos())
    }
}

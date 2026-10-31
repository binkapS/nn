//! A tiny dependency-free GPT (decoder-only Transformer) in Rust.
//!
//! Pipeline: character tokenizer -> token+position embeddings -> N pre-LN
//! transformer blocks (causal multi-head self-attention + GeLU MLP) -> final
//! LayerNorm -> linear head -> softmax cross-entropy. Everything, including
//! every backward pass, is implemented by hand in this crate.

mod attention;
mod corpus;
mod data;
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
        steps: 600,
        layers: 3,
        heads: 3,
        embd: 96,
        block: 64,
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
    let mut i = 0;
    while i < argv.len() {
        let (flag, val) = (argv[i].as_str(), argv.get(i + 1).map(|s| s.as_str()));
        macro_rules! next {
            () => {
                match val {
                    Some(v) => {
                        i += 1;
                        v
                    }
                    None => panic!("flag {flag} needs a value"),
                }
            };
        }
        match flag {
            "--data" => a.data = Some(next!().to_string()),
            "--steps" => a.steps = next!().parse().unwrap(),
            "--layers" => a.layers = next!().parse().unwrap(),
            "--heads" => a.heads = next!().parse().unwrap(),
            "--embd" => a.embd = next!().parse().unwrap(),
            "--block" => a.block = next!().parse().unwrap(),
            "--batch" => a.batch = next!().parse().unwrap(),
            "--lr" => a.lr = next!().parse().unwrap(),
            "--wd" => a.wd = next!().parse().unwrap(),
            "--temperature" => a.temperature = next!().parse().unwrap(),
            "--top-k" => a.top_k = next!().parse().unwrap(),
            "--max-new" => a.max_new = next!().parse().unwrap(),
            "--prompt" => a.prompt = next!().to_string(),
            "--seed" => a.seed = next!().parse().unwrap(),
            "--print-every" => a.print_every = next!().parse().unwrap(),
            "--eval-every" => a.eval_every = next!().parse().unwrap(),
            other => panic!("unknown flag: {other}"),
        }
        i += 1;
    }
    a
}

fn main() {
    let args = parse_args();

    let text = load_text(&args.data);
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
            print!(
                "{:>4}      {:>12.4}/", step, loss
            );
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

    // Sampling showcase from a few fixed prompts.
    for prompt in [args.prompt.as_str(), "tom ", "the river "] {
        let prompt_ids = tokenizer.encode(prompt);
        if prompt_ids.is_empty() {
            continue;
        }
        let seq = model.generate(&prompt_ids, args.max_new, args.temperature, args.top_k, &mut rng);
        println!();
        println!("―― sample, prompt = {prompt:?} (temp = {}, top-k = {}) ――", args.temperature, args.top_k);
        println!("{}", tokenizer.decode(&seq));
    }
}

/// Linear warmup to `peak`, then cosine decay to `min_lr`.
fn schedule(_step: usize, total: usize, warmup: usize, peak: f32, min_lr: f32) -> f32 {
    if _step < warmup {
        peak * (_step + 1) as f32 / warmup.max(1) as f32
    } else {
        let p = (_step - warmup) as f32 / (total - warmup).max(1) as f32;
        let p = p.min(1.0);
        min_lr + 0.5 * (peak - min_lr) * (1.0 + (std::f32::consts::PI * p).cos())
    }
}

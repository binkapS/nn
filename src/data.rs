//! Character-level tokenizer + batch sampling utilities.

use std::collections::HashMap;
use std::fs;

use crate::rng::Rng;

pub struct CharTokenizer {
    pub chars: Vec<char>,
    pub stoi: HashMap<char, usize>,
}

impl CharTokenizer {
    pub fn build(text: &str) -> Self {
        let mut buf: Vec<char> = text.chars().collect();
        buf.sort();
        buf.dedup();
        let stoi = buf.iter().cloned().enumerate().map(|(i, c)| (c, i)).collect();
        Self { chars: buf, stoi }
    }

    pub fn vocab_size(&self) -> usize {
        self.chars.len()
    }

    /// Encodes, silently dropping characters outside the vocabulary.
    pub fn encode(&self, s: &str) -> Vec<usize> {
        s.chars().filter_map(|c| self.stoi.get(&c).copied()).collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|&i| self.chars[i]).collect()
    }
}

/// Splits a string into (train, validation) parts at `train_frac`.
pub fn split(text: &str, train_frac: f64) -> (String, String) {
    let chars: Vec<char> = text.chars().collect();
    let cut = ((chars.len() as f64) * train_frac) as usize;
    let cut = cut.clamp(1, chars.len().saturating_sub(1));
    (chars[..cut].iter().collect(), chars[cut..].iter().collect())
}

/// Draws one batch of `(inputs, targets)` of shape (`batch`, `block_size`)
/// by sampling random windows. A target at position t is the token at t+1.
/// Inputs and targets are returned flattened (rows = batch * block_size).
pub fn get_batch(rng: &mut Rng, data: &[usize], block_size: usize, batch: usize) -> (Vec<usize>, Vec<usize>) {
    assert!(data.len() > block_size + 1, "dataset too small for block_size");
    let n = data.len() - block_size - 1;
    let mut inputs = Vec::with_capacity(batch * block_size);
    let mut targets = Vec::with_capacity(batch * block_size);
    for _ in 0..batch {
        let start = rng.rand_index(n);
        inputs.extend_from_slice(&data[start..start + block_size]);
        targets.extend_from_slice(&data[start + 1..start + block_size + 1]);
    }
    (inputs, targets)
}

/// Loads either a user-supplied file or (when `path` is None) the built-in
/// demo corpus, str or "".
pub fn load_text(path: &Option<String>) -> String {
    match path {
        Some(p) => fs::read_to_string(p).unwrap_or_else(|e| panic!("failed to read {p}: {e}")),
        None => crate::corpus::DEMO_TEXT.to_string(),
    }
}

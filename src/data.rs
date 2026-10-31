//! Character-level tokenizer + batch sampling utilities + HF dataset fetch.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

use crate::rng::Rng;

/// Hugging Face datasets this crate can pull as plain text at runtime,
/// keyed by a short alias. (`--hf <alias>`), plus any direct URL fallback.
pub const HF_DATASETS: &[(&str, &str, &str)] = &[
    // (alias, stored filename, resolve URL)
    (
        "shakespeare",
        "tiny_shakespeare.txt",
        "https://huggingface.co/datasets/SamPIngram/tinyshakespeare/resolve/main/input.txt",
    ),
];

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

/// Default training text: the cached Hugging Face dataset when present, the
/// bundled corpus otherwise. `--data FILE` always wins over both.
pub fn load_text(path: &Option<String>) -> String {
    match path {
        Some(p) => fs::read_to_string(p).unwrap_or_else(|e| panic!("failed to read {p}: {e}")),
        None => match fs::read_to_string("data/tiny_shakespeare.txt") {
            Ok(s) if !s.trim().is_empty() => s,
            _ => crate::corpus::DEMO_TEXT.to_string(),
        },
    }
}

/// Fetches a Hugging Face dataset as plain text into `data/<filename>` and
/// returns `(contents, local_path)`. `source` is either one of the
/// `HF_DATASETS` aliases (e.g. `shakespeare`) or a direct
/// `https://.../resolve/main/...` URL (saved as `data/hf_<slug>.txt`).
///
/// The runtime download shells out to `curl`, keeping the crate itself
/// dependency-free; fetched files are cached offline for later runs.
pub fn fetch_hf(source: &str) -> (String, String) {
    let (fname, url) = if source.starts_with("http://") || source.starts_with("https://") {
        let slug = direct_url_slug(source);
        (format!("hf_{slug}.txt"), source.to_string())
    } else {
        let found = HF_DATASETS
            .iter()
            .find(|(alias, _, _)| *alias == source)
            .unwrap_or_else(|| panic!("unknown HF dataset alias `{source}` (known: {:?}); pass a direct https URL instead",
                HF_DATASETS.iter().map(|(a, ..)| *a).collect::<Vec<_>>()));
        (found.1.to_string(), found.2.to_string())
    };

    let dir = PathBuf::from("data");
    fs::create_dir_all(&dir).expect("cannot create data/ directory");
    let local = dir.join(&fname);

    // Skip the download when a complete cache already exists.
    if let Ok(cached) = fs::read_to_string(&local)
        && !cached.trim().is_empty()
    {
        eprintln!("[hf] using cached dataset: {}", local.display());
        return (cached, local.display().to_string());
    }

    eprintln!("[hf] downloading {url}");
    let status = Command::new("curl")
        .arg("-sSLf")
        .arg("--max-time")
        .arg("300")
        .arg("-o")
        .arg(&local)
        .arg(&url)
        .status()
        .expect("failed to run curl (is it installed?) — or pass --data FILE directly");
    assert!(status.success(), "curl failed with {status}");
    let text = fs::read_to_string(&local)
        .unwrap_or_else(|e| panic!("downloaded file unreadable ({}): {e}", local.display()));
    let text = match text.strip_prefix('\u{feff}') {
        Some(t) => t.to_string(), // de-BOM
        None => text,
    };
    eprintln!("[hf] saved → {} ({} bytes)", local.display(), text.len());
    (text, local.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_alias_urls() {
        assert_eq!(HF_DATASETS[0].0, "shakespeare");
        assert!(HF_DATASETS[0].2.starts_with("https://huggingface.co/datasets/SamPIngram/tinyshakespeare/resolve/main/"));
    }

    #[test]
    fn direct_url_slug_works() {
        let url = "https://huggingface.co/datasets/SamPIngram/tinyshakespeare/resolve/main/input.txt";
        assert_eq!(direct_url_slug(url), "datasets_SamPIngram_tinyshakespeare_input.txt");
    }

    #[test]
    fn tokenizer_roundtrip_and_vocab() {
        let tok = CharTokenizer::build("abba the cat");
        assert_eq!(tok.vocab_size(), 7); // a b ␣ t h e c
        let ids = tok.encode("bb ab");
        assert_eq!(tok.decode(&ids), "bb ab");
        // unknown chars are dropped, not crashed on
        assert_eq!(tok.encode("a§b"), vec![tok.stoi[&'a'], tok.stoi[&'b']]);
    }
}

/// Builds a cache-file slug from a direct HF resolve URL, e.g.
/// `https://huggingface.co/datasets/X/Y/resolve/main/file.txt` → `X_Y_file.txt`.
fn direct_url_slug(url: &str) -> String {
    let core = url.split_once("https://").unwrap_or(("", url)).1;
    let parts: Vec<&str> = core.split('/').filter(|p| !p.is_empty()).collect();
    // Skip the scheme host (huggingface.co) and the "resolve"/"main" markers.
    let meaningful: Vec<&str> = parts
        .iter()
        .skip(1)
        .filter(|p| !p.contains(".co") && *p != &"resolve" && *p != &"main")
        .copied()
        .collect();
    meaningful.join("_")
}

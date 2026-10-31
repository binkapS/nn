//! Character-level tokenizer, batch sampling, Hugging Face dataset fetching
//! (with validated offline caching), and dataset-agnostic prompt mining.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::rng::Rng;

/// Hugging Face datasets this crate can pull as plain text at runtime,
/// keyed by short alias (`--hf <alias>`), plus any direct URL fallback.
/// The runtime download shells out to `curl`, keeping the crate dependency-free.
pub const HF_DATASETS: &[(&str, &str, &str)] = &[
    // (alias, stored filename, resolve URL)
    (
        "shakespeare",
        "tiny_shakespeare.txt",
        "https://huggingface.co/datasets/SamPIngram/tinyshakespeare/resolve/main/input.txt",
    ),
];

// ---------------------------------------------------------------------------
// Data directory resolution (not CWD-dependent, see README)
// ---------------------------------------------------------------------------

/// Locates (and does not yet create) the `data` directory. The first
/// existing candidate wins: `./data`, `<exe-dir>/data`,
/// `<exe-dir>/../../data` (i.e. the repo root when launched via
/// `cargo run`/`target/{debug,release}`), falling back to `./data` — which
/// is then created on demand — so runs stay consistent from any CWD.
pub fn default_data_dir() -> PathBuf {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(Path::to_path_buf));
    let candidates: Vec<PathBuf> = ["data"]
        .iter()
        .map(PathBuf::from)
        .chain(exe_dir.iter().flat_map(|d| {
            [
                d.join("data"),
                d.join("..").join("..").join("data").components().collect::<PathBuf>(),
            ]
        }))
        .collect();
    candidates
        .into_iter()
        .find(|p| p.is_dir())
        .unwrap_or_else(|| PathBuf::from("data"))
}

fn data_file(dir: &Path, fname: &str) -> PathBuf {
    dir.join(fname)
}

/// Integrity marker: after a successful download we write `<file>.mini`
/// (META INtegrity) holding the exact byte length of the content. A cache
/// file is "complete" only when the marker exists and matches, which catches
/// truncated or partially written downloads.
fn integrity_path(local: &Path) -> PathBuf {
    let mut name = local.file_name().unwrap_or_default().to_os_string();
    name.push(".mini");
    local.with_file_name(name)
}

/// Reads a cache file only if its integrity marker validates it.
fn load_cached_text(local: &Path) -> Option<String> {
    let expected: usize = fs::read_to_string(integrity_path(local)).ok()?.trim().parse().ok()?;
    let text = fs::read_to_string(local).ok()?;
    let text = de_bom(text);
    if text.len() == expected && !text.trim().is_empty() {
        Some(text)
    } else {
        None
    }
}

fn de_bom(s: String) -> String {
    match s.strip_prefix('\u{feff}') {
        Some(t) => t.to_string(),
        None => s,
    }
}

fn write_cache(local: &Path, text: &str) -> Result<(), String> {
    fs::write(local, text).map_err(|e| format!("cannot write {}: {e}", local.display()))?;
    fs::write(integrity_path(local), format!("{}\n", text.len()))
        .map_err(|e| format!("cannot write {}: {e}", integrity_path(local).display()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Dataset loading
// ---------------------------------------------------------------------------

/// Resolves the training text and a human-readable source label.
///
/// Resolution order: `--data FILE` (read directly, errors abort the run) >
/// cached + integrity-validated `data/tiny_shakespeare.txt` (HF dataset) >
/// bundled offline fallback corpus.
pub fn load_text(path: &Option<String>) -> (String, String) {
    if let Some(p) = path {
        let text = fs::read_to_string(p).unwrap_or_else(|e| panic!("failed to read {p}: {e}"));
        return (text, p.clone());
    }
    let cached = default_data_dir().join("tiny_shakespeare.txt");
    if let Some(text) = load_cached_text(&cached) {
        let label = std::fs::canonicalize(&cached)
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| cached.display().to_string());
        return (text, format!("{label} (HuggingFace: SamPIngram/tinyshakespeare)"));
    }
    (crate::corpus::DEMO_TEXT.to_string(), "built-in demo corpus (offline fallback)".to_string())
}

// ---------------------------------------------------------------------------
// Hugging Face fetch
// ---------------------------------------------------------------------------

/// Outcome of parsing `--hf`: a cache filename and the URL to fetch.
type Source = (String, String);

/// `source` is either an `HF_DATASETS` alias (e.g. `shakespeare`) or a direct
/// `https://.../resolve/main/...` URL (saved as `hf_<deterministic-hash>.txt`).
fn parse_hf_source(source: &str) -> Result<Source, String> {
    if source.starts_with("http://") || source.starts_with("https://") {
        Ok((format!("hf_{}.txt", direct_url_slug(source)), source.to_string()))
    } else {
        HF_DATASETS
            .iter()
            .find(|(alias, _, _)| *alias == source)
            .map(|&(_, name, url)| (name.to_string(), url.to_string()))
            .ok_or_else(|| format!(
                "unknown HF dataset alias `{source}` (known: {:?}) — or pass a direct https://... resolve URL",
                HF_DATASETS.iter().map(|(a, ..)| *a).collect::<Vec<_>>()
            ))
    }
}

/// Fetches a Hugging Face dataset as plain text into `<data-dir>/<file>` and
/// returns `(contents, local_path)` on success; a `Result` so that a bad URL,
/// missing `curl`, network failure or empty download degrades gracefully
/// instead of aborting the run (callers fall back locally; see `main.rs`).
///
/// Downloads are cached offline and reused only when validated by the
/// integrity marker (see `load_cached_text`).
pub fn fetch_hf(source: &str) -> Result<(String, String), String> {
    fetch_hf_into(&default_data_dir(), source)
}

/// As `fetch_hf`, but with an explicit data directory (used by tests).
pub fn fetch_hf_into(dir: &Path, source: &str) -> Result<(String, String), String> {
    let (fname, url) = parse_hf_source(source)?;
    fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let local = data_file(dir, &fname);

    // A validated cache means zero network traffic.
    if let Some(cached) = load_cached_text(&local) {
        eprintln!("[hf] using cached dataset: {}", local.display());
        return Ok((cached, local.display().to_string()));
    }

    eprintln!("[hf] downloading {url}");
    download_to(&local, &url)?;
    let text = fs::read_to_string(&local)
        .map_err(|e| format!("downloaded file unreadable ({}): {e}", local.display()))?;
    if text.trim().is_empty() {
        return Err(format!("download from {url} is empty — is the URL correct?"));
    }
    let text = de_bom(text);
    write_cache(&local, &text)?;
    eprintln!("[hf] saved → {} ({} bytes) [integrity marker written]", local.display(), text.len());
    Ok((text, local.display().to_string()))
}

/// Runs `curl`, translating every failure mode (binary missing, DNS error,
/// HTTP error, truncated transfer) into an `Err` instead of a panic.
fn download_to(local: &Path, url: &str) -> Result<(), String> {
    let output = Command::new("curl")
        .arg("-sSLf") // silent, follow redirects, fail on HTTP errors (4xx/5xx)
        .arg("--max-time")
        .arg("300")
        .arg("-o")
        .arg(local)
        .arg(url)
        .output()
        .map_err(|e| format!("failed to run curl (is it installed?) — or point --data at a local file: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "curl exited {} for {url} — dataset unreachable or URL wrong (stderr: {})",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(())
}

/// Deterministic cache-file slug from a direct HF resolve URL.
///
/// Parses the URL rather than filtering on magic substrings:
/// `scheme://host/p/a/t/h` → `host_p_a_t_h` with every segment sanitized to
/// `[A-Za-z0-9._-]`, capped to 96 chars, then suffixed with an 8-hex FNV-1a
/// hash of the full URL so different URLs can never collide on one cache
/// file. Example:
/// `https://huggingface.co/datasets/X/Y/resolve/main/input.txt`
/// → `huggingface.co_datasets_X_Y_resolve_main_input.txt-ab12cd34`.
fn direct_url_slug(url: &str) -> String {
    let core = url.split_once("://").map(|(_, c)| c).unwrap_or(url);
    let (host, path) = match core.split_once('/') {
        Some((h, p)) => (h, p),
        None => (core, ""),
    };
    let mut slug = sanitize_seg(host);
    for seg in path.split('/').filter(|s| !s.is_empty()) {
        slug.push('_');
        slug.push_str(&sanitize_seg(seg));
    }
    slug.truncate(96);
    slug.push('-');
    slug.push_str(&hash_hex8(url));
    slug
}

fn sanitize_seg(seg: &str) -> String {
    seg.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// 64-bit FNV-1a, rendered as 8 hex digits.
fn hash_hex8(s: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    format!("{:08x}", h as u32)
}

// ---------------------------------------------------------------------------
// Dataset-agnostic sample prompts (word-frequency mining)
// ---------------------------------------------------------------------------

/// Mining helper for the sampling showcase: the top-3 frequent words (len ≥ 2,
/// alphanumeric-only body) of the corpus, lowercased + trailing space, so
/// generation prompts adapt to whatever dataset is trained on. `user` is
/// always placed first; a corpus without suitable words falls back to "the ".
pub fn sample_prompts(text: &str, user: &str) -> Vec<String> {
    let mut counts: HashMap<String, (u32, usize)> = HashMap::new();
    let mut first_seen = 0usize;
    for w in text.split_whitespace() {
        let w = w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
        if w.chars().count() < 2 {
            first_seen += 1;
            continue;
        }
        let e = counts.entry(w).or_insert((0, first_seen));
        e.0 += 1;
        first_seen += 1;
    }
    let mut words: Vec<(String, u32, usize)> =
        counts.into_iter().map(|(w, (c, f))| (w, c, f)).collect();
    // frequency desc, first-appearance asc, then alphabetical *only* when
    // both are equal to keep the result deterministic.
    words.sort_unstable_by(|a, b| {
        b.1.cmp(&a.1).then(a.2.cmp(&b.2)).then(a.0.cmp(&b.0))
    });
    let mut prompts: Vec<String> = words
        .into_iter()
        .take(3)
        .map(|(w, ..)| format!("{w} "))
        .collect();
    if prompts.is_empty() {
        prompts.push("the ".to_string());
    }
    let mut out = vec![user.to_string()];
    out.extend(prompts);
    out
}

// ---------------------------------------------------------------------------
// Tokenizer / batching
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Tests (direct_url_slug is defined above; its doc example is asserted here)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_alias_urls() {
        assert_eq!(HF_DATASETS[0].0, "shakespeare");
        assert!(HF_DATASETS[0]
            .2
            .starts_with("https://huggingface.co/datasets/SamPIngram/tinyshakespeare/resolve/main/"));
    }

    #[test]
    fn direct_url_slug_works() {
        let url = "https://huggingface.co/datasets/SamPIngram/tinyshakespeare/resolve/main/input.txt";
        let slug = direct_url_slug(url);
        // deterministic: prefix is the sanitized parsed path, suffix is the
        // 8-hex FNV-1a of the full URL.
        assert_eq!(slug.split('-').count(), 2, "{slug}");
        assert!(slug.starts_with("huggingface.co_datasets_SamPIngram_tinyshakespeare"), "{slug}");
        assert!(slug.contains("input.txt-"), "{slug}");
        // hash suffix: dash + exactly 8 hex digits
        let tail = &slug[slug.len() - 9..];
        assert_eq!(&tail[..1], "-");
        assert!(tail[1..].chars().all(|c| c.is_ascii_hexdigit()), "{tail}");
        // different URLs never collide (hash differs)
        assert_ne!(direct_url_slug("https://x.com/a.txt"), direct_url_slug("https://x.com/b.txt"));
        assert_eq!(direct_url_slug("https://x.com/a.txt"), direct_url_slug("https://x.com/a.txt"));
    }

    #[test]
    fn unknown_alias_is_result_error() {
        let err = parse_hf_source("definitely_not_an_alias").unwrap_err();
        assert!(err.contains("unknown HF dataset alias"), "{err}");
        // direct URLs always parse
        assert!(parse_hf_source("https://example.com/data.txt").is_ok());
    }

    #[test]
    fn sample_prompts_uses_frequent_words() {
        let text = "the cat sat the cat the dog ran ran ran";
        let p = sample_prompts(text, "hello ");
        assert_eq!(p[0], "hello ");
        assert_eq!(p[1], "the ");
        assert!(p.contains(&"ran ".to_string()));
    }

    #[test]
    fn sample_prompts_falls_back() {
        assert_eq!(sample_prompts("", "x "), vec!["x ".to_string(), "the ".to_string()]);
    }

    #[test]
    fn integrity_marker_gate() {
        let dir = std::env::temp_dir().join(format!("nn_integrity_{}", std::process::id()));
        fs::remove_dir_all(&dir).ok();
        fs::create_dir_all(&dir).unwrap();

        // Use the exact (fname, url) the fetch path would derive for this URL.
        let (fname, url) = parse_hf_source("http://127.0.0.1:1/unreachable.txt").unwrap();
        let local = data_file(&dir, &fname);
        let text = "hello committed world\n";
        write_cache(&local, text).unwrap();

        // Validated cache returns the content without any network access.
        let got = fetch_hf_into(&dir, "http://127.0.0.1:1/unreachable.txt").unwrap();
        assert_eq!(got.0, text);
        assert_eq!(got.1, local.display().to_string());

        // Tamper with the file: cache must be invalidated (this then tries the
        // unreachable URL and must yield Err, not panic).
        fs::write(&local, "truncated").unwrap();
        let err = fetch_hf_into(&dir, &url).unwrap_err();
        assert!(err.contains("curl exited"), "{err}");

        // A cache file without a marker is equally rejected.
        fs::remove_file(integrity_path(&local)).unwrap();
        fs::write(&local, text).unwrap(); // byte-for-byte valid but unmarked
        assert!(fetch_hf_into(&dir, &url).is_err());

        fs::remove_dir_all(&dir).ok();
    }
}

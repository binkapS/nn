//! Character-level tokenizer, batch sampling, Hugging Face dataset fetching
//! (with validated offline caching), and dataset-agnostic prompt mining.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::rng::Rng;

/// How a known dataset is fetched.
///
/// * `Single` — one plain-text file fetched from a direct HF resolve URL
///   (downloaded via `curl`, cached, integrity-validated).
/// * `C4` — a *streaming* sample of the [allenai/c4] corpus via the HF
///   datasets-server `/rows` API: web-text documents are pulled in pages of
///   100 rows (no gzip decompression needed — the server emits plain JSON),
///   their `text` fields unescaped and joined with `\n` until the byte budget
///   is met. C4 is huge (364k+ validation rows, gigabytes of training data),
///   so this takes a small deterministic slice (`validation` split) sized for
///   a CPU-trainable demo.
///
/// [allenai/c4]: https://huggingface.co/datasets/allenai/c4
#[derive(Clone, Debug)]
pub enum Fetch {
    Single(Cow<'static, str>),
    C4 {
        config: &'static str,
        split: &'static str,
        /// Byte budget of the joined text (stop paging when reached).
        max_bytes: usize,
        /// Hard page cap so a pathological API response ends the loop.
        max_pages: usize,
    },
}

pub struct KnownDataset {
    pub alias: &'static str,
    pub filename: &'static str,
    pub spec: Fetch,
}

/// Hugging Face datasets this crate can pull, keyed by short alias
/// (`--hf <alias>`), plus any direct `https` resolve-URL fallback.
/// Runtime fetches shell out to `curl`, keeping the crate dependency-free.
pub const HF_DATASETS: &[KnownDataset] = &[
    KnownDataset {
        alias: "c4",
        filename: "c4_en_validation_sample.txt",
        spec: Fetch::C4 {
            config: "en",
            split: "validation",
            max_bytes: 3_000_000,
            max_pages: 60,
        },
    },
    KnownDataset {
        alias: "shakespeare",
        filename: "tiny_shakespeare.txt",
        spec: Fetch::Single(Cow::Borrowed(
            "https://huggingface.co/datasets/SamPIngram/tinyshakespeare/resolve/main/input.txt",
        )),
    },
];

/// Default dataset: [`allenai/c4`] alias `c4` (run by plain `cargo run`).
/// The default `--steps` (5000) and model size (4 layers, d_model 128,
/// ctx 128) in `main.rs` are tuned for this ~3 MB web-text slice.
///
/// [allenai/c4]: https://huggingface.co/datasets/allenai/c4
pub const DEFAULT_HF_ALIAS: &str = "c4";

const C4_ROWS_PAGE: usize = 100;

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
/// cached + integrity-validated dataset files in `data/` (checked in the
/// alias order of `HF_DATASETS`, c4 first) > bundled offline fallback corpus.
pub fn load_text(path: &Option<String>) -> (String, String) {
    if let Some(p) = path {
        let text = fs::read_to_string(p).unwrap_or_else(|e| panic!("failed to read {p}: {e}"));
        return (text, p.clone());
    }
    let dir = default_data_dir();
    for ds in HF_DATASETS {
        let local = dir.join(ds.filename);
        if let Some(text) = load_cached_text(&local) {
            let label = fs::canonicalize(&local)
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| local.display().to_string());
            return (text, format!("{label} (HuggingFace: {})", hf_label(ds)));
        }
    }
    (
        crate::corpus::DEMO_TEXT.to_string(),
        "built-in demo corpus (offline fallback)".to_string(),
    )
}

/// Human-readable repo label for the startup banner.
fn hf_label(ds: &KnownDataset) -> &'static str {
    match &ds.spec {
        Fetch::C4 { .. } => "allenai/c4",
        Fetch::Single(url) => match url.contains("tinyshakespeare") {
            true => "SamPIngram/tinyshakespeare",
            false => ds.alias,
        },
    }
}

// ---------------------------------------------------------------------------
// Hugging Face fetch
// ---------------------------------------------------------------------------

/// Resolves `--hf`: a known alias → (cache filename, spec), or any direct
/// `https://.../...` URL → deterministic single-file spec.
fn parse_hf_source(source: &str) -> Result<(String, Fetch), String> {
    if source.starts_with("http://") || source.starts_with("https://") {
        Ok((
            format!("hf_{}.txt", direct_url_slug(source)),
            Fetch::Single(Cow::Owned(source.to_string())),
        ))
    } else {
        HF_DATASETS
            .iter()
            .find(|ds| ds.alias == source)
            .map(|ds| (ds.filename.to_string(), ds.spec.clone()))
            .ok_or_else(|| format!(
                "unknown HF dataset alias `{source}` (known: {:?}) — or pass a direct https://... URL",
                HF_DATASETS.iter().map(|ds| ds.alias).collect::<Vec<_>>()
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
    let (fname, spec) = parse_hf_source(source)?;
    fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let local = data_file(dir, &fname);

    // A validated cache means zero network traffic.
    if let Some(cached) = load_cached_text(&local) {
        eprintln!("[hf] using cached dataset: {}", local.display());
        return Ok((cached, local.display().to_string()));
    }

    let text = match &spec {
        Fetch::Single(url) => {
            eprintln!("[hf] downloading {url}");
            download_to(&local, url)?;
            let raw = fs::read_to_string(&local)
                .map_err(|e| format!("downloaded file unreadable ({}): {e}", local.display()))?;
            if raw.trim().is_empty() {
                return Err(format!("download from {url} is empty — is the URL correct?"));
            }
            de_bom(raw)
        }
        Fetch::C4 { config, split, max_bytes, max_pages } => {
            eprintln!("[hf] streaming allenai/c4 (config {config}, split {split}, budget {max_bytes} bytes)");
            fetch_c4(config, split, *max_bytes, *max_pages)?
        }
    };

    write_cache(&local, &text)?;
    eprintln!("[hf] cached → {} ({} bytes) [integrity marker written]", local.display(), text.len());
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

/// Runs `curl` for one c4 page, capturing stdout (the JSON response).
fn download_body(url: &str) -> Result<String, String> {
    let output = Command::new("curl")
        .arg("-sSLf")
        .arg("--compressed") // transparent gzip of the HTTP transfer
        .arg("--max-time")
        .arg("120")
        .arg(url)
        .output()
        .map_err(|e| format!("failed to run curl (is it installed?) — or point --data at a local file: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "curl exited {} for {url} — dataset unreachable (stderr: {})",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// The datasets-server `/rows` page-URL builder: `C4_ROWS_PAGE` rows per page.
fn c4_page_url(config: &str, split: &str, offset: usize) -> String {
    format!(
        "https://datasets-server.huggingface.co/rows?dataset=allenai/c4&config={config}&split={split}&offset={offset}&length={}",
        C4_ROWS_PAGE
    )
}

/// Streams the `allenai/c4` slice: paged `/rows` requests, `text` fields
/// extracted and joined with `\n` until the byte budget or page cap is hit.
/// Real paging runs through its effect closure in `fetch_c4_with`; tests
/// inject canned pages.
pub fn fetch_c4(config: &str, split: &str, max_bytes: usize, max_pages: usize) -> Result<String, String> {
    fetch_c4_with(max_bytes, max_pages, |offset| {
        download_body(&c4_page_url(config, split, offset))
    })
}

/// Body of `fetch_c4`, parameterized over the page fetcher (unit-testable).
///
/// Loop invariants: `offset` always advances by the number of rows actually
/// received; a short page or an empty page at `offset > 0` ends the stream
/// cleanly, only an empty *first* page is an error.
fn fetch_c4_with(
    max_bytes: usize,
    max_pages: usize,
    mut fetch_page: impl FnMut(usize) -> Result<String, String>,
) -> Result<String, String> {
    let mut out = String::new();
    let mut offset = 0usize;
    for _page in 0..max_pages {
        let body = fetch_page(offset)?;
        let texts = extract_text_fields(&body);
        if texts.is_empty() {
            if offset == 0 {
                return Err(
                    "c4: the first page contained no text rows — API response malformed or split empty"
                        .to_string(),
                );
            }
            break; // end of split
        }
        for t in &texts {
            out.push_str(t);
            out.push('\n');
        }
        if out.len() >= max_bytes {
            // Clamp on a char boundary (String::truncate panics otherwise).
            let mut end = max_bytes.min(out.len());
            while !out.is_char_boundary(end) {
                end -= 1;
            }
            out.truncate(end);
            break;
        }
        if texts.len() < C4_ROWS_PAGE {
            break; // short page: end of split
        }
        offset += texts.len();
    }
    if out.is_empty() {
        return Err("c4: fetched no text documents — the API may be down or the split empty".to_string());
    }
    // Control characters would add junk vocabulary; keep \n and \t.
    Ok(out
        .chars()
        .filter(|&c| !c.is_control() || c == '\n' || c == '\t')
        .collect())
}

// ---------------------------------------------------------------------------
// Minimal JSON string-field extraction (datasets-server responses)
// ---------------------------------------------------------------------------

/// Extracts the value of every `"text"` string field from a (possibly large)
/// JSON body without a full JSON parser: scanning raw bytes is safe because
/// a literal `"text":` cannot appear *inside* a JSON string (quotes there
/// are escaped as `\"`). Unescapes the string content, including `\uXXXX`.
pub fn extract_text_fields(body: &str) -> Vec<String> {
    const KEY: &str = "\"text\":";
    let mut out = Vec::new();
    let bytes = body.as_bytes();
    let mut i = 0;
    while i + KEY.len() <= bytes.len() {
        if &bytes[i..i + KEY.len()] == KEY.as_bytes()
            && let Some(s) = json_string(&bytes[i + KEY.len()..])
        {
            out.push(s);
        }
        i += 1;
    }
    out
}

/// Parses a JSON string literal starting at `bytes` (must begin with `"`).
/// Returns the decoded content; malformed tails return whatever was decoded
/// so far (tolerant by design — c4 documents are arbitrary web text).
/// Decoding works on a byte buffer so raw multi-byte UTF-8 inside the string
/// is preserved verbatim.
fn json_string(bytes: &[u8]) -> Option<String> {
    if bytes.first() != Some(&b'"') {
        return None;
    }
    let mut out: Vec<u8> = Vec::new();
    let mut i = 1;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => return Some(String::from_utf8_lossy(&out).into_owned()),
            b'\\' => {
                i += 1;
                if i >= bytes.len() {
                    return Some(String::from_utf8_lossy(&out).into_owned());
                }
                match bytes[i] {
                    b'"' => out.push(b'"'),
                    b'\\' => out.push(b'\\'),
                    b'/' => out.push(b'/'),
                    b'b' => out.push(0x08),
                    b'f' => out.push(0x0C),
                    b'n' => out.push(b'\n'),
                    b'r' => out.push(b'\r'),
                    b't' => out.push(b'\t'),
                    b'u' => {
                        i = read_unicode_escape(bytes, i, &mut out);
                        continue; // i already advanced past the escape(s)
                    }
                    other => out.push(other),
                }
            }
            _ => out.push(bytes[i]), // raw UTF-8 continuation or ASCII
        }
        i += 1;
    }
    Some(String::from_utf8_lossy(&out).into_owned()) // unterminated: tolerant
}

/// Appends the UTF-8 encoding of `c` to `out`.
fn push_char(out: &mut Vec<u8>, c: char) {
    let mut buf = [0u8; 4];
    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
}

/// Reads one `\uXXXX` escape at `bytes[i] == 'u'` (i points at the `u`),
/// decoding surrogate pairs. Appends the result and returns the index of the
/// *next unread byte* after the escape (callers continue scanning there).
fn read_unicode_escape(bytes: &[u8], i: usize, out: &mut Vec<u8>) -> usize {
    let cp = hex4(bytes, i + 1);
    if (0xD800..=0xDBFF).contains(&cp) {
        // high surrogate: a well-formed pair is immediately followed by \uDC00..DFFF
        let is_pair = i + 10 < bytes.len()
            && bytes[i + 5] == b'\\'
            && bytes[i + 6] == b'u';
        let lo = if is_pair { hex4(bytes, i + 7) } else { 0 };
        if is_pair && (0xDC00..=0xDFFF).contains(&lo) {
            let combined = 0x1_0000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            push_char(out, char::from_u32(combined).unwrap_or('\u{FFFD}'));
            i + 11
        } else {
            push_char(out, '\u{FFFD}'); // lone high surrogate
            i + 5
        }
    } else if (0xDC00..=0xDFFF).contains(&cp) {
        push_char(out, '\u{FFFD}'); // lone low surrogate
        i + 5
    } else {
        push_char(out, char::from_u32(cp).unwrap_or('\u{FFFD}'));
        i + 5
    }
}

fn hex4(bytes: &[u8], start: usize) -> u32 {
    (start..start + 4)
        .map(|j| bytes.get(j).copied().unwrap_or(b'0'))
        .fold(0u32, |acc, b| acc * 16 + hex_digit(b))
}

fn hex_digit(b: u8) -> u32 {
    match b {
        b'0'..=b'9' => (b - b'0') as u32,
        b'a'..=b'f' => (b - b'a' + 10) as u32,
        b'A'..=b'F' => (b - b'A' + 10) as u32,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Deterministic cache-file slug for direct URLs
// ---------------------------------------------------------------------------

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
    words.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)).then(a.0.cmp(&b.0)));
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_aliases_and_labels() {
        let aliases: Vec<&str> = HF_DATASETS.iter().map(|d| d.alias).collect();
        assert_eq!(aliases, vec!["c4", "shakespeare"]);
        assert_eq!(DEFAULT_HF_ALIAS, "c4");
    }

    #[test]
    fn dataset_specs_resolve() {
        let (f1, s1) = parse_hf_source("c4").unwrap();
        assert_eq!(f1, "c4_en_validation_sample.txt");
        match s1 {
            Fetch::C4 { config, split, max_pages, .. } => {
                assert_eq!(config, "en");
                assert_eq!(split, "validation");
                assert!(max_pages >= 3);
            }
            _ => panic!("c4 must be a C4 spec"),
        }
        let (_, s2) = parse_hf_source("shakespeare").unwrap();
        assert!(matches!(s2, Fetch::Single(_)));
    }

    #[test]
    fn direct_url_slug_works() {
        let url = "https://huggingface.co/datasets/SamPIngram/tinyshakespeare/resolve/main/input.txt";
        let slug = direct_url_slug(url);
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
    fn json_text_extraction() {
        let body = r#"{"num_rows_total":2,"rows":[
            {"row_idx":0,"row":{"text":"Hello \"world\"\nLine2\ttab","timestamp":"x","url":"http://a/b"}},
            {"row_idx":1,"row":{"text":"Emoji éü and 😀 grin","timestamp":"y","url":"http://c/d"}}
        ]}"#;
        let texts = extract_text_fields(body);
        assert_eq!(texts.len(), 2);
        assert_eq!(texts[0], "Hello \"world\"\nLine2\ttab");
        assert_eq!(texts[1], "Emoji éü and 😀 grin");
        // escaped backslash-quotes *inside* a url string must not be mistaken
        // for a second text field
        let body2 = r#"{"rows":[{"row":{"url":"http://x.com/a?text=\"no\""},"text":"only-this"}]}"#;
        assert_eq!(extract_text_fields(body2).len(), 1);
        // surrogate pair decoding
        let body3 = r#"{"rows":[{"row":{"text":"A\ud83d\ude00B"}}]}"#;
        assert_eq!(extract_text_fields(body3)[0], "A😀B");
        // lone surrogates degrade to U+FFFD instead of panicking
        let body4 = r#"{"rows":[{"row":{"text":"A\ud83dB"}}]}"#;
        assert_eq!(extract_text_fields(body4)[0], "A\u{FFFD}B");
    }

    /// Builds a canned datasets-server page with `n` rows.
    fn mk_page(n: usize, body: impl Fn(usize) -> String) -> String {
        let rows: Vec<String> = (0..n)
            .map(|i| format!(r#"{{"row_idx":{},"row":{{"text":"{}",{},"url":"u{}"}}}}"#, i, body(i), r#""timestamp":"t""#, i))
            .collect();
        format!(r#"{{"num_rows_total":{},"rows":[{}]}}"#, n, rows.join(","))
    }

    #[test]
    fn c4_streaming_respects_byte_budget() {
        // Three full pages available, but the byte budget clamps mid-page-1.
        let page = mk_page(100, |i| format!("word{i} content text"));
        let mut calls = vec![];
        let text = fetch_c4_with(600, 10, |off| {
            calls.push(off);
            Ok(page.clone())
        })
        .unwrap();
        assert!(text.len() >= 500 && text.len() <= 604, "len={}", text.len());
        assert!(text.contains("word0 content text"));
        assert!(!text.contains("word100"), "budget clamp stopped before page 2");
        assert_eq!(calls, vec![0]); // stopped inside page 1
    }

    #[test]
    fn c4_short_page_ends_stream() {
        let pages = [mk_page(100, |i| format!("doc {i}")), mk_page(30, |i| format!("doc {}", 100 + i))];
        let mut calls = vec![];
        let text = fetch_c4_with(usize::MAX, 50, |off| {
            calls.push(off);
            Ok(pages[off / 100].clone())
        })
        .unwrap();
        assert_eq!(calls, vec![0, 100]);
        assert!(text.contains("doc 129"));
        assert!(!text.contains("doc 130"));
    }

    #[test]
    fn c4_empty_page_is_clean_end_only_past_first() {
        // First page empty → error.
        let err = fetch_c4_with(1000, 5, |_| Ok(r#"{"rows":[]}"#.to_string())).unwrap_err();
        assert!(err.contains("first page"), "{err}");
        // Second page empty (out-of-range offset) → clean end, no error.
        let first = mk_page(100, |i| format!("doc {i}"));
        let text = fetch_c4_with(usize::MAX, 5, |off| {
            if off == 0 {
                Ok(first.clone())
            } else {
                Ok(r#"{"rows":[]}"#.to_string())
            }
        })
        .unwrap();
        assert!(text.contains("doc 99"));
    }

    #[test]
    fn c4_control_chars_filtered() {
        let body = mk_page(1, |_| "ok\u{0007} char".to_string());
        let text = fetch_c4_with(1000, 10, |_| Ok(body.clone())).unwrap();
        assert!(text.contains("ok char"));
        assert!(!text.contains('\u{0007}'));
    }

    #[test]
    fn integrity_marker_gate() {
        let dir = std::env::temp_dir().join(format!("nn_integrity_{}", std::process::id()));
        fs::remove_dir_all(&dir).ok();
        fs::create_dir_all(&dir).unwrap();

        // Use the exact (fname, spec) the fetch path would derive for this URL.
        let (fname, spec) = parse_hf_source("http://127.0.0.1:1/unreachable.txt").unwrap();
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
        let Fetch::Single(url) = spec else { panic!() };
        let err = fetch_hf_into(&dir, &url).unwrap_err();
        assert!(err.contains("curl exited"), "{err}");

        // A cache file without a marker is equally rejected.
        fs::remove_file(integrity_path(&local)).unwrap();
        fs::write(&local, text).unwrap(); // byte-for-byte valid but unmarked
        assert!(fetch_hf_into(&dir, &url).is_err());

        fs::remove_dir_all(&dir).ok();
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
    fn tokenizer_roundtrip_and_vocab() {
        let tok = CharTokenizer::build("abba the cat");
        assert_eq!(tok.vocab_size(), 7); // a b ␣ t h e c
        let ids = tok.encode("bb ab");
        assert_eq!(tok.decode(&ids), "bb ab");
        // unknown chars are dropped, not crashed on
        assert_eq!(tok.encode("a§b"), vec![tok.stoi[&'a'], tok.stoi[&'b']]);
    }
}

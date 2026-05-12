//! User-supplied vocabulary overrides loaded from a JSON file.
//!
//! Two parts:
//!
//! * `rewrites`: ordered list of regex → template replacements applied
//!   to the raw input text *before* any pre-phonemize normalization.
//!   Capture groups (`$1`, `$2`, …) are supported via the standard
//!   `regex::Regex::replace_all` template syntax. Use this to rewrite
//!   abbreviations, fix idiosyncratic source spellings, or coerce
//!   numerical formats.
//!
//! * `pronunciations`: case-insensitive word → IPA map consulted as
//!   the very first step inside `phonemize_word`, ahead of gold and
//!   CMUdict. Use this to override the dictionary on a per-word basis
//!   (project-specific jargon, proper nouns, loanwords).
//!
//! JSON shape:
//! ```json
//! {
//!   "rewrites": [
//!     {"pattern": "Mr\\. (\\w+)", "replacement": "Mister $1"},
//!     {"pattern": "\\b(\\d+)x\\b", "replacement": "$1 times"}
//!   ],
//!   "pronunciations": {
//!     "kokoro": "kˈoʊkəɹoʊ",
//!     "github": "ɡˈɪthˌʌb"
//!   }
//! }
//! ```
//!
//! Wired in via `set_custom_vocab` from a CLI flag. See
//! `examples/loanword_vocab.json` for a worked example covering
//! French / German / Polish / Chinese loanwords.

use anyhow::{Context, Result};
use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

pub struct CustomVocab {
    rewrites: Vec<(Regex, String)>,
    pronunciations: HashMap<String, String>,
}

static VOCAB: OnceLock<CustomVocab> = OnceLock::new();

pub fn set(vocab: CustomVocab) -> Result<()> {
    VOCAB
        .set(vocab)
        .map_err(|_| anyhow::anyhow!("custom vocab already initialized"))
}

pub fn get() -> Option<&'static CustomVocab> {
    VOCAB.get()
}

impl CustomVocab {
    pub fn load(path: &Path) -> Result<Self> {
        let source = std::fs::read_to_string(path)
            .with_context(|| format!("reading custom vocab from {}", path.display()))?;
        let value: Value = serde_json::from_str(&source)
            .with_context(|| format!("parsing custom vocab json {}", path.display()))?;
        let obj = value
            .as_object()
            .context("custom vocab root must be a JSON object")?;

        let mut rewrites = Vec::new();
        if let Some(items) = obj.get("rewrites").and_then(|v| v.as_array()) {
            for item in items {
                let pattern = item
                    .get("pattern")
                    .and_then(|v| v.as_str())
                    .context("rewrite entry needs string 'pattern'")?;
                let replacement = item
                    .get("replacement")
                    .and_then(|v| v.as_str())
                    .context("rewrite entry needs string 'replacement'")?;
                let re = Regex::new(pattern)
                    .with_context(|| format!("compiling rewrite regex {pattern:?}"))?;
                rewrites.push((re, replacement.to_string()));
            }
        }

        let mut pronunciations = HashMap::new();
        if let Some(map) = obj.get("pronunciations").and_then(|v| v.as_object()) {
            for (key, value) in map {
                if let Some(ipa) = value.as_str() {
                    pronunciations.insert(key.to_ascii_lowercase(), ipa.to_string());
                }
            }
        }

        Ok(Self {
            rewrites,
            pronunciations,
        })
    }

    pub fn apply_rewrites(&self, text: &str) -> String {
        let mut out = text.to_string();
        for (re, replacement) in &self.rewrites {
            out = re.replace_all(&out, replacement.as_str()).into_owned();
        }
        out
    }

    pub fn lookup_pronunciation(&self, word: &str) -> Option<&str> {
        self.pronunciations
            .get(&word.to_ascii_lowercase())
            .map(String::as_str)
    }
}

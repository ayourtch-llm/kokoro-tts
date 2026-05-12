#![allow(dead_code)]

use serde_json::Value;
use std::collections::HashMap;
use std::sync::OnceLock;

pub struct MisakiGoldLexicon {
    entries: HashMap<String, String>,
}

static LEXICON: OnceLock<MisakiGoldLexicon> = OnceLock::new();

pub fn lexicon() -> &'static MisakiGoldLexicon {
    LEXICON.get_or_init(MisakiGoldLexicon::load)
}

impl MisakiGoldLexicon {
    fn load() -> Self {
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/data/misaki-us-gold.json"
        ));
        let value: Value = serde_json::from_str(source).expect("valid misaki gold json");
        let obj = value.as_object().expect("misaki gold json object");
        let mut entries = HashMap::with_capacity(obj.len() * 2);
        for (key, value) in obj {
            if let Some(ipa) = flatten_value(value) {
                entries.insert(key.clone(), ipa.clone());
                entries
                    .entry(key.to_ascii_lowercase())
                    .or_insert_with(|| ipa.clone());
            }
        }
        // Curated overrides for loanwords where the CMUdict / LTS
        // fallback gets it badly wrong (typically because a fragment
        // collides with a Greek-letter entry like "chi" → /kaɪ/).
        for (key, ipa) in CURATED_OVERRIDES {
            entries.insert((*key).to_string(), (*ipa).to_string());
            entries.insert(key.to_ascii_lowercase(), (*ipa).to_string());
        }
        Self { entries }
    }

    pub fn lookup(&self, word: &str) -> Option<&str> {
        self.entries.get(word).map(String::as_str).or_else(|| {
            self.entries
                .get(&word.to_ascii_lowercase())
                .map(String::as_str)
        })
    }
}

/// Hyphenated and other loanwords where neither gold's JSON nor the
/// CMUdict / LTS fallback gives a sensible English-loan pronunciation.
const CURATED_OVERRIDES: &[(&str, &str)] = &[
    ("tai-chi", "tˈaɪʧˌi"),
    ("taichi", "tˈaɪʧˌi"),
    ("qi-gong", "ʧˈiɡˌɔŋ"),
    ("qigong", "ʧˈiɡˌɔŋ"),
    ("kung-fu", "kˌʌŋfˈu"),
    ("jiu-jitsu", "ʤˌuʤˈɪtsu"),
    ("aikido", "aɪkˈido"),
];

fn flatten_value(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Object(map) => map
            .get("DEFAULT")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .or_else(|| map.values().find_map(|v| v.as_str().map(str::to_owned))),
        _ => None,
    }
}

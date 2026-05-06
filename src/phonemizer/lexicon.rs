#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::OnceLock;

pub struct Lexicon {
    entries: HashMap<String, Vec<&'static str>>,
}

static LEXICON: OnceLock<Lexicon> = OnceLock::new();

pub fn lexicon() -> &'static Lexicon {
    LEXICON.get_or_init(Lexicon::load)
}

impl Lexicon {
    fn load() -> Self {
        let source = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/data/cmudict-0.7b"));
        let mut entries = HashMap::new();
        for line in source.lines() {
            if line.is_empty() || line.starts_with(";;;") {
                continue;
            }
            let mut parts = line.split_whitespace();
            let Some(word) = parts.next() else {
                continue;
            };
            if word.contains('(') {
                continue;
            }
            let phones: Vec<&'static str> = parts.collect();
            if phones.is_empty() {
                continue;
            }
            entries.entry(word.to_ascii_lowercase()).or_insert(phones);
        }
        Self { entries }
    }

    pub fn lookup(&self, word: &str) -> Option<&[&'static str]> {
        self.entries
            .get(&word.to_ascii_lowercase())
            .map(|phones| phones.as_slice())
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

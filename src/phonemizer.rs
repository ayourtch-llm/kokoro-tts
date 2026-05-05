#![allow(dead_code)]

use anyhow::{bail, Result};

pub const MILESTONE_TEST_PHRASE: &str = "hello world";
pub const MILESTONE_TEST_PHONEMES: &str = "həlˈoʊ wˈɜɹld";

pub trait Phonemizer: Send + Sync {
    fn phonemize(&self, text: &str) -> Result<String>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct StubPhonemizer;

impl Phonemizer for StubPhonemizer {
    fn phonemize(&self, text: &str) -> Result<String> {
        if normalize_for_stub(text) == MILESTONE_TEST_PHRASE {
            Ok(MILESTONE_TEST_PHONEMES.to_string())
        } else {
            bail!(
                "stub phonemizer only supports the milestone phrase {:?}",
                MILESTONE_TEST_PHRASE
            )
        }
    }
}

#[cfg(feature = "espeak")]
#[derive(Debug, Default, Clone, Copy)]
pub struct EspeakPhonemizer;

#[cfg(feature = "espeak")]
impl Phonemizer for EspeakPhonemizer {
    fn phonemize(&self, _text: &str) -> Result<String> {
        bail!("espeak-ng phonemizer feature is declared but the FFI backend is not wired yet")
    }
}

fn normalize_for_stub(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_ascii_punctuation())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::{Phonemizer, StubPhonemizer, MILESTONE_TEST_PHONEMES};

    #[test]
    fn stub_returns_canned_ipa_for_milestone_phrase() {
        assert_eq!(
            StubPhonemizer.phonemize("Hello, world!").unwrap(),
            MILESTONE_TEST_PHONEMES
        );
    }

    #[test]
    fn stub_rejects_unknown_text() {
        assert!(StubPhonemizer.phonemize("different text").is_err());
    }
}

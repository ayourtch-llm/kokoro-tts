#![allow(dead_code)]

use anyhow::{bail, Result};

pub const MILESTONE_TEST_PHRASE: &str = "hello world";
pub const MILESTONE_TEST_PHONEMES: &str = "həlˈO wˈɜɹld";

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

mod arpabet;
mod lexicon;
mod misaki_gold;
mod sentence;

#[derive(Debug, Default, Clone, Copy)]
pub struct TwoTierPhonemizer;

impl Phonemizer for TwoTierPhonemizer {
    fn phonemize(&self, text: &str) -> Result<String> {
        let gold = misaki_gold::lexicon();
        let lexicon = lexicon::lexicon();
        let mut out = String::new();
        for sentence in sentence::split_sentences(text) {
            let sentence_out = phonemize_chunk(&sentence, gold, lexicon);
            if sentence_out.is_empty() {
                continue;
            }
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(&sentence_out);
        }
        Ok(out)
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

#[derive(Debug)]
enum Token {
    Word(String),
    Punct(char),
}

fn phonemize_chunk(
    text: &str,
    gold: &misaki_gold::MisakiGoldLexicon,
    lexicon: &lexicon::Lexicon,
) -> String {
    let mut out = String::new();
    for token in tokenize(text) {
        match token {
            Token::Word(word) => {
                if needs_space_before_word(&out) {
                    out.push(' ');
                }
                let ipa = gold
                    .lookup(&word)
                    .map(str::to_owned)
                    .or_else(|| lexicon.lookup(&word).map(arpabet::phones_to_ipa))
                    .unwrap_or_else(|| spell_out_word(&word));
                out.push_str(&ipa);
            }
            Token::Punct(punct) => out.push(punct),
        }
    }
    out
}

fn tokenize(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphabetic() || ch == '\'' || ch == '-' {
            current.push(ch);
        } else {
            if !current.is_empty() {
                tokens.push(Token::Word(std::mem::take(&mut current)));
            }
            if matches!(
                ch,
                ',' | '.' | '!' | '?' | ';' | ':' | '“' | '”' | '—' | '…'
            ) {
                tokens.push(Token::Punct(ch));
            }
        }
    }
    if !current.is_empty() {
        tokens.push(Token::Word(current));
    }
    tokens
}

fn needs_space_before_word(out: &str) -> bool {
    match out.chars().last() {
        None => false,
        Some(' ' | '(' | '[' | '{' | '“' | '‘' | '—' | '…') => false,
        Some(_) => true,
    }
}

fn spell_out_word(word: &str) -> String {
    let mut out = String::new();
    for ch in word.chars() {
        if !out.is_empty() {
            out.push(' ');
        }
        let letter = match ch.to_ascii_lowercase() {
            'a' => "eɪ",
            'b' => "bi",
            'c' => "si",
            'd' => "di",
            'e' => "i",
            'f' => "ɛf",
            'g' => "dʒi",
            'h' => "eɪʧ",
            'i' => "aɪ",
            'j' => "dʒeɪ",
            'k' => "keɪ",
            'l' => "ɛl",
            'm' => "ɛm",
            'n' => "ɛn",
            'o' => "oʊ",
            'p' => "pi",
            'q' => "kju",
            'r' => "ɑɹ",
            's' => "ɛs",
            't' => "ti",
            'u' => "ju",
            'v' => "vi",
            'w' => "dʌbəlju",
            'x' => "ɛks",
            'y' => "waɪ",
            'z' => "zi",
            _ => continue,
        };
        out.push_str(letter);
    }
    if out.is_empty() {
        word.to_string()
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::{Phonemizer, StubPhonemizer, TwoTierPhonemizer, MILESTONE_TEST_PHONEMES};

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

    #[test]
    fn two_tier_returns_canned_ipa_for_milestone_phrase() {
        assert_eq!(
            TwoTierPhonemizer.phonemize("hello world").unwrap(),
            MILESTONE_TEST_PHONEMES
        );
    }

    #[test]
    fn two_tier_keeps_sentence_punctuation() {
        assert_eq!(
            TwoTierPhonemizer
                .phonemize("hello world. hello world?")
                .unwrap(),
            "həlˈO wˈɜɹld. həlˈO wˈɜɹld?"
        );
    }
}

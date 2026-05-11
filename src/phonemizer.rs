#![allow(dead_code)]

use anyhow::{bail, Result};

pub const MILESTONE_TEST_PHRASE: &str = "hello world";
pub const MILESTONE_TEST_PHONEMES: &str = "həlˈO wˈɜɹld";

/// Phonemes string used by `tools/reference_*.py` to generate the per-stage
/// `.bin` oracles in `tmp/`. Must match Python's `DEFAULT_PHONEMES` verbatim
/// — its character count drives `style_index` selection in `*_check.rs`, and
/// changing it requires regenerating every reference under `tmp/`.
pub const REFERENCE_PHONEMES: &str = "həlˈoʊ wˈɜɹld";

pub trait Phonemizer: Send + Sync {
    fn phonemize(&self, text: &str) -> Result<String>;

    fn phonemize_chunks(&self, text: &str) -> Result<Vec<String>> {
        Ok(vec![self.phonemize(text)?])
    }
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
mod homograph;
mod lexicon;
mod lts;
mod misaki_gold;
mod normalize;
pub mod sentence;
#[allow(unused_imports)]
pub use normalize::normalize_abbreviations;
#[allow(unused_imports)]
pub use normalize::normalize_acronyms;
#[allow(unused_imports)]
pub use normalize::normalize_cardinals;
#[allow(unused_imports)]
pub use normalize::normalize_dates;
#[allow(unused_imports)]
pub use normalize::normalize_math;
#[allow(unused_imports)]
pub use normalize::normalize_money_time;
#[allow(unused_imports)]
pub use normalize::normalize_units;

#[derive(Debug, Default, Clone, Copy)]
pub struct TwoTierPhonemizer;

impl Phonemizer for TwoTierPhonemizer {
    fn phonemize(&self, text: &str) -> Result<String> {
        Ok(self.phonemize_chunks(text)?.join(" "))
    }

    fn phonemize_chunks(&self, text: &str) -> Result<Vec<String>> {
        let gold = misaki_gold::lexicon();
        let lexicon = lexicon::lexicon();
        let mut out = Vec::new();
        for sentence in sentence::split_sentences(text) {
            let sentence_out = phonemize_chunk(&sentence, gold, lexicon);
            if !sentence_out.is_empty() {
                out.push(sentence_out);
            }
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
    let text = normalize_cardinals(&normalize::normalize_acronyms(&normalize::normalize_units(
        &normalize::normalize_money_time(&normalize::normalize_math(&normalize::normalize_dates(
            &normalize::normalize_abbreviations(text),
        ))),
    )));
    let tokens = tokenize(text);
    let word_tokens: Vec<&str> = tokens
        .iter()
        .filter_map(|token| match token {
            Token::Word(word) => Some(word.as_str()),
            Token::Punct(_) => None,
        })
        .collect();
    let mut word_index = 0usize;
    let mut out = String::new();
    for token in &tokens {
        match token {
            Token::Word(word) => {
                if needs_space_before_word(&out) {
                    out.push(' ');
                }
                let ctx = homograph::WordContext::new(&word_tokens, word_index);
                let ipa = phonemize_word(word, &ctx, gold, lexicon);
                out.push_str(&ipa);
                word_index += 1;
            }
            Token::Punct(punct) => out.push(*punct),
        }
    }
    out
}

fn phonemize_word(
    word: &str,
    ctx: &homograph::WordContext<'_>,
    gold: &misaki_gold::MisakiGoldLexicon,
    lexicon: &lexicon::Lexicon,
) -> String {
    if let Some(ipa) = homograph::phonemize(word, ctx) {
        return ipa;
    }
    if let Some((base, possessive)) = split_possessive_acronym(word) {
        if normalize::is_pronounce_as_word_acronym(base) {
            let mut ipa = pronounce_or_spell_acronym(base, gold, lexicon);
            if possessive {
                ipa.push('z');
            }
            return ipa;
        }
        if normalize::is_all_caps_acronym(base) {
            // Same emphasis-vs-acronym heuristic as the non-possessive path:
            // gold-only lookup so CMUdict acronym spell-outs don't fire.
            if base.len() >= 3 {
                let lower = base.to_ascii_lowercase();
                if let Some(ipa) = gold.lookup(&lower) {
                    let mut ipa = ipa.to_owned();
                    if possessive {
                        ipa.push_str(possessive_phone_after(&ipa));
                    }
                    return ipa;
                }
            }
            let mut ipa = spell_out_word(base);
            if possessive {
                ipa.push('z');
            }
            return ipa;
        }
    }
    if normalize::is_all_caps_acronym(word) && normalize::is_pronounce_as_word_acronym(word) {
        return pronounce_or_spell_acronym(word, gold, lexicon);
    }
    if normalize::is_all_caps_acronym(word) {
        // Treat all-caps emphasis like a normal word when the lowercased form
        // is a real word in misaki-gold ("WAR" → "war", "STOP" → "stop",
        // "NEVER" → "never"). Only use the curated gold dict — CMUdict has
        // explicit entries for acronyms like "FBI" → "EH1 F B IY1 AY1"
        // which a CMU lookup would treat as a "word" and break spell-out.
        // 2-letter all-caps tokens are kept as letter spelling because
        // "IT"/"ID"/"AI"/"OK" are far more often acronyms than emphasis.
        if word.len() >= 3 {
            let lower = word.to_ascii_lowercase();
            if let Some(ipa) = gold.lookup(&lower) {
                return ipa.to_owned();
            }
        }
        return spell_out_word(word);
    }
    gold.lookup(&word)
        .map(str::to_owned)
        .or_else(|| lexicon.lookup(&word).map(arpabet::phones_to_ipa))
        .or_else(|| try_possessive(word, ctx, gold, lexicon))
        .or_else(|| try_hyphenated(word, ctx, gold, lexicon))
        .unwrap_or_else(|| lts::pronounce_oov(word))
}

/// Handle a hyphenated compound like "pre-war", "one-twelfth", "well-known"
/// when the whole token misses the lexicon. Split on '-', phonemize each
/// part via the full pipeline (so common parts like "one" get their
/// canonical /wʌn/ from gold rather than an LTS approximation), and join
/// with no separator — kokoro's prosody handles the syllable boundary.
/// Returns None for words without a hyphen or with empty fragments.
fn try_hyphenated(
    word: &str,
    ctx: &homograph::WordContext<'_>,
    gold: &misaki_gold::MisakiGoldLexicon,
    lexicon: &lexicon::Lexicon,
) -> Option<String> {
    if !word.contains('-') {
        return None;
    }
    let parts: Vec<&str> = word.split('-').filter(|p| !p.is_empty()).collect();
    if parts.len() < 2 {
        return None;
    }
    let mut out = String::new();
    for part in parts {
        out.push_str(&phonemize_word(part, ctx, gold, lexicon));
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Handle a possessive like "mark's" / "Germany's" / "country's" that
/// isn't in the lexicon as a contraction: phonemize the base and append
/// /s/, /z/, or /ɪz/ depending on the final phone of the base IPA.
fn try_possessive(
    word: &str,
    ctx: &homograph::WordContext<'_>,
    gold: &misaki_gold::MisakiGoldLexicon,
    lexicon: &lexicon::Lexicon,
) -> Option<String> {
    let base = word
        .strip_suffix("'s")
        .or_else(|| word.strip_suffix("'S"))
        .or_else(|| word.strip_suffix("\u{2019}s"))
        .or_else(|| word.strip_suffix("\u{2019}S"))?;
    if base.is_empty() {
        return None;
    }
    // Don't recurse infinitely: base must not itself end in 's.
    if base.ends_with('\'') || base.ends_with('\u{2019}') {
        return None;
    }
    let base_ipa = phonemize_word(base, ctx, gold, lexicon);
    if base_ipa.is_empty() {
        return None;
    }
    let suffix = possessive_phone_after(&base_ipa);
    Some(format!("{base_ipa}{suffix}"))
}

/// English possessive 's allomorph from the final phone of the base.
/// - sibilants (s, z, ʃ, ʒ, tʃ, dʒ) → /ɪz/
/// - voiceless stops/fricatives (p, t, k, f, θ) → /s/
/// - everything else (vowels, voiced consonants) → /z/
fn possessive_phone_after(ipa: &str) -> &'static str {
    let trimmed = ipa.trim_end_matches(|c: char| {
        c == 'ˈ' || c == 'ˌ' || c == 'ː' || c.is_whitespace()
    });
    let chars: Vec<char> = trimmed.chars().collect();
    if chars.len() >= 2 {
        let last_two: String = chars[chars.len() - 2..].iter().collect();
        if last_two == "tʃ" || last_two == "dʒ" {
            return "ɪz";
        }
    }
    match chars.last() {
        Some('s') | Some('z') | Some('ʃ') | Some('ʒ') => "ɪz",
        Some('p') | Some('t') | Some('k') | Some('f') | Some('θ') => "s",
        _ => "z",
    }
}

fn pronounce_or_spell_acronym(
    word: &str,
    gold: &misaki_gold::MisakiGoldLexicon,
    lexicon: &lexicon::Lexicon,
) -> String {
    let upper = word.to_ascii_uppercase();
    match upper.as_str() {
        "JSON" => "ˈdʒeɪsən".to_string(),
        "FAQ" => "fæk".to_string(),
        _ => gold
            .lookup(&upper)
            .map(str::to_owned)
            .or_else(|| lexicon.lookup(&upper).map(arpabet::phones_to_ipa))
            .unwrap_or_else(|| spell_out_word(word)),
    }
}

fn split_possessive_acronym(word: &str) -> Option<(&str, bool)> {
    if let Some(base) = word.strip_suffix("'s").or_else(|| word.strip_suffix("'S")) {
        if normalize::is_all_caps_acronym(base) {
            return Some((base, true));
        }
    }
    None
}

fn tokenize(text: String) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        // U+2019 RIGHT SINGLE QUOTATION MARK is used as the apostrophe in
        // most curly-quoted text (e.g. "mark’s"). Treat it as part of the
        // word like a straight apostrophe so possessives and contractions
        // stay together as one token.
        if ch.is_ascii_alphabetic() || ch == '\'' || ch == '\u{2019}' || ch == '-' {
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

    #[test]
    fn two_tier_handles_acronyms_and_possessives() {
        assert_eq!(TwoTierPhonemizer.phonemize("NASA's").unwrap(), "nˈæsəz");
        assert_eq!(TwoTierPhonemizer.phonemize("FBI's").unwrap(), "ɛf bi aɪz");
    }

    #[test]
    fn two_tier_disambiguates_homographs() {
        assert!(TwoTierPhonemizer
            .phonemize("I read the book yesterday.")
            .unwrap()
            .contains("ɹɛd"));
        assert!(TwoTierPhonemizer
            .phonemize("She will lead the meeting.")
            .unwrap()
            .contains("lˈid"));
    }
}

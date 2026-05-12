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
pub mod custom_vocab;
pub mod lexicon;
mod lts;
pub mod misaki_gold;
pub mod normalize;
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

pub fn lexicon_for_test(word: &str) -> Vec<String> {
    lexicon::lexicon()
        .lookup(word)
        .map(|phs| phs.iter().map(|s| (*s).to_string()).collect())
        .unwrap_or_default()
}

pub fn normalize_urls_for_test(s: &str) -> String {
    normalize::normalize_urls(s)
}

pub fn pre_phonemize_for_test(s: &str) -> String {
    pre_phonemize_normalize(s)
}

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
        // URL-normalize before sentence splitting so the splitter doesn't
        // cut a URL at its dots (e.g. "example.com/path").
        let url_normalized = normalize::normalize_urls_with(text, |w| {
            gold.lookup(w).is_some() || lexicon.lookup(w).is_some()
        });
        for sentence in sentence::split_sentences(&url_normalized) {
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

/// Run the full TTS pre-phonemize normalization cascade on text.
/// Exposed so the round-trip harness can normalize WER reference and
/// hypothesis identically to what the synthesis pipeline sees.
pub fn pre_phonemize_normalize(text: &str) -> String {
    let gold = misaki_gold::lexicon();
    let lexicon = lexicon::lexicon();
    // User-supplied regex rewrites: applied to raw input before any
    // built-in normalization so they can fix idiosyncratic source
    // spellings, expand custom abbreviations, etc.
    let rewritten;
    let text: &str = if let Some(vocab) = custom_vocab::get() {
        rewritten = vocab.apply_rewrites(text);
        &rewritten
    } else {
        text
    };
    let is_emphasis_word = |base: &str| -> bool {
        let lower = base.to_ascii_lowercase();
        let len = base.len();
        if len < 3 {
            return false;
        }
        if gold.lookup(&lower).is_some() {
            return true;
        }
        if len >= 6 && lexicon.lookup(&lower).is_some() {
            return true;
        }
        false
    };
    // En-dash and minus sign frequently appear in math/poker contexts
    // as the unary "negative" marker; fold them to ASCII hyphen so
    // normalize_math's "-" handling can route them to " minus ".
    // Editorial square brackets ('[B]efore', '[citation]') aren't
    // spoken markup; strip them so the wrapped content reads naturally.
    // En-dash '–' between digits is a range ("299–320" → "299 to 320");
    // elsewhere fold to ASCII '-' so normalize_math can route it to
    // " minus " when in unary-minus context.
    let chars: Vec<char> = text.chars().collect();
    let mut dashed = String::with_capacity(text.len());
    for (i, &c) in chars.iter().enumerate() {
        match c {
            '[' | ']' => continue,
            '–' => {
                let prev_digit = i > 0 && chars[i - 1].is_ascii_digit();
                let next_digit = chars.get(i + 1).is_some_and(|c| c.is_ascii_digit());
                if prev_digit && next_digit {
                    dashed.push_str(" to ");
                } else {
                    dashed.push('-');
                }
            }
            '−' => dashed.push('-'),
            _ => dashed.push(c),
        }
    }
    let greek = normalize::expand_greek_letters(&dashed);
    let supersub = normalize::fold_super_sub_digits(&greek);
    let cards = normalize::normalize_card_suits(&supersub);
    let foot_inch = normalize::expand_foot_inch_marks(&cards);
    let hex_expanded = normalize::expand_hex_literals(&foot_inch);
    let romanized = normalize::expand_roman_numerals(&hex_expanded, |w| {
        gold.lookup(w).is_some() || lexicon.lookup(w).is_some()
    });
    let folded = normalize::fold_diacritics(&romanized);
    let url_expanded = normalize::normalize_urls_with(&folded, |w| {
        gold.lookup(w).is_some() || lexicon.lookup(w).is_some()
    });
    normalize_cardinals(&normalize::normalize_acronyms_with(
        &normalize::lowercase_emphasis_function_words(&normalize::separate_digit_alpha_boundaries(
            &normalize::normalize_units(&normalize::normalize_money_time(&normalize::normalize_math(
                &normalize::normalize_dates(&normalize::normalize_abbreviations(&url_expanded)),
            ))),
        )),
        is_emphasis_word,
    ))
}

fn phonemize_chunk(
    text: &str,
    gold: &misaki_gold::MisakiGoldLexicon,
    lexicon: &lexicon::Lexicon,
) -> String {
    let text = pre_phonemize_normalize(text);
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
    // User-supplied per-word pronunciation override: highest priority.
    if let Some(vocab) = custom_vocab::get() {
        if let Some(ipa) = vocab.lookup_pronunciation(word) {
            return ipa.to_string();
        }
    }
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
            // Same emphasis-vs-acronym heuristic as the non-possessive path.
            let len = base.len();
            if len >= 3 {
                let lower = base.to_ascii_lowercase();
                let real_word = gold.lookup(&lower).map(str::to_owned).or_else(|| {
                    if len >= 6 {
                        lexicon.lookup(&lower).map(arpabet::phones_to_ipa)
                    } else {
                        None
                    }
                });
                if let Some(mut ipa) = real_word {
                    if possessive {
                        let suffix = possessive_phone_after(&ipa);
                        ipa.push_str(suffix);
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
        // "NEVER" → "never"). For length-3-5 stay gold-only — CMUdict has
        // explicit spelled-out entries for short acronyms like
        // "FBI" → "EH1 F B IY1 AY1" that a CMU lookup would mistake for a
        // "word" pronunciation. For length ≥ 6, fall back to CMUdict so
        // plural emphasis like "SYSTEMS", "PROBLEMS", "QUESTIONS",
        // "ACKNOWLEDGMENTS" (in CMUdict but not gold) reads as words.
        // 2-letter all-caps tokens stay letter-spelled because
        // "IT"/"ID"/"AI"/"OK" are usually initialisms, not emphasis.
        let len = word.len();
        if len >= 3 {
            let lower = word.to_ascii_lowercase();
            if let Some(ipa) = gold.lookup(&lower) {
                return ipa.to_owned();
            }
            if len >= 6 {
                if let Some(ipa) = lexicon.lookup(&lower).map(arpabet::phones_to_ipa) {
                    return ipa;
                }
            }
        }
        return spell_out_word(word);
    }
    gold.lookup(&word)
        .map(str::to_owned)
        .or_else(|| try_gold_plural(word, gold))
        .or_else(|| lexicon.lookup(&word).map(arpabet::phones_to_ipa))
        .or_else(|| try_possessive(word, ctx, gold, lexicon))
        .or_else(|| try_hyphenated(word, ctx, gold, lexicon))
        .or_else(|| try_british_digraph(word, gold, lexicon))
        .or_else(|| try_compound_prefix(word, gold, lexicon))
        .or_else(|| try_inflectional_suffix(word, gold, lexicon))
        .or_else(|| try_camel_case(word, ctx, gold, lexicon))
        .or_else(|| try_letter_spell_unpronounceable(word))
        .unwrap_or_else(|| lts::pronounce_oov(word))
}

/// Inflectional-suffix fallback for OOV verbs/nouns. Many words miss
/// gold and CMUdict in their inflected form ("hamstering",
/// "swansong'd", "factionalises") but the bare stem is in CMUdict.
/// LTS over the whole word places stress poorly; here we keep the
/// stem's authoritative pronunciation and append the suffix phones.
/// Only inflectional suffixes that change a final morpheme without
/// reassigning stress (-ing, -ly, -er, -ers, -ness) are listed —
/// -ed has voiceless/voiced/epenthetic allomorph complications that
/// try_gold_plural / possessive_phone_after handle for the *gold*
/// path; here we deliberately stay narrow.
fn try_inflectional_suffix(
    word: &str,
    gold: &misaki_gold::MisakiGoldLexicon,
    lexicon: &lexicon::Lexicon,
) -> Option<String> {
    const SUFFIXES: &[(&str, &str)] = &[
        ("ings", "ɪŋz"),
        ("ing", "ɪŋ"),
        ("ness", "nəs"),
        ("ers", "əɹz"),
        ("er", "əɹ"),
        ("ly", "li"),
        // Plural "-ies" of stems ending in 'y' (weirdies, beardies,
        // techies) — LTS over the whole word reads "ies" as /aɪs/
        // ("weirdies" → /wɛˈɪɹdaɪs/). Prefer stem + /iz/.
        ("ies", "iz"),
    ];
    let lower = word.to_ascii_lowercase();
    for &(suf, suf_ipa) in SUFFIXES {
        if let Some(stem) = lower.strip_suffix(suf) {
            if stem.len() < 3 {
                continue;
            }
            let stem_ipa = gold
                .lookup(stem)
                .map(str::to_owned)
                .or_else(|| lexicon.lookup(stem).map(arpabet::phones_to_ipa));
            if let Some(s) = stem_ipa {
                return Some(format!("{s}{suf_ipa}"));
            }
        }
    }
    None
}

/// Last-resort letter-spell for short OOV codes that LTS would render
/// as a vowel-less cluster ("std" → /std/, "btn" → /btn/) or that
/// happen to be common letter-spelled acronyms ("ieee" / "html" /
/// "css" / "sql"). Heuristic: word is ≤5 ASCII letters AND either has
/// no vowels at all, or it's lowercase but matches one of the common
/// known initialisms. Returns letter-spelled IPA.
fn try_letter_spell_unpronounceable(word: &str) -> Option<String> {
    let lower = word.to_ascii_lowercase();
    if !lower.chars().all(|c| c.is_ascii_lowercase()) {
        return None;
    }
    if lower.len() < 2 || lower.len() > 5 {
        return None;
    }
    let vowel_count = lower
        .chars()
        .filter(|c| matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y'))
        .count();
    const KNOWN_INITIALISMS: &[&str] = &[
        "ieee", "html", "css", "sql", "vhdl", "json", "xml", "yaml",
        "http", "https", "ftp", "ssh", "url", "ui", "ux", "api",
        "ci", "cd", "sdk", "ide", "ocr", "asr", "tts", "gui",
        "cpu", "gpu", "ram", "rom", "ssd", "hdd", "usb",
        "ai", "ml", "nlp", "ar", "vr", "iot",
    ];
    let known = KNOWN_INITIALISMS.contains(&lower.as_str());
    let unpronounceable = vowel_count == 0;
    if known || unpronounceable {
        return Some(spell_out_word(word));
    }
    None
}

/// Split a camelCase identifier ("NonCommutativeAdditiveSemigroup") and
/// phonemize each part through the full lookup chain so individual
/// words land their CMUdict entries instead of stretching LTS over the
/// whole compound. Common in technical prose.
fn try_camel_case(
    word: &str,
    ctx: &homograph::WordContext<'_>,
    gold: &misaki_gold::MisakiGoldLexicon,
    lexicon: &lexicon::Lexicon,
) -> Option<String> {
    let parts = lts::split_camel_case(word)?;
    if parts.len() < 2 {
        return None;
    }
    let mut out = String::new();
    for (idx, part) in parts.iter().enumerate() {
        if idx > 0 {
            out.push(' ');
        }
        out.push_str(&phonemize_word(part, ctx, gold, lexicon));
    }
    Some(out)
}

/// Last-resort: split OOV compound words at a known English prefix
/// boundary if the remainder is in gold/CMUdict.
/// "noncommutative" → non + commutative, "typename" → type + name,
/// "semigroup" → semi + group. Avoids LTS misreads of unhyphenated
/// technical compounds without hard-coding any specific word.
fn try_compound_prefix(
    word: &str,
    gold: &misaki_gold::MisakiGoldLexicon,
    lexicon: &lexicon::Lexicon,
) -> Option<String> {
    const PREFIXES: &[&str] = &[
        "anti", "astro", "auto", "bi", "bio", "co", "cyber", "de", "dis",
        "eco", "electro", "en", "ex", "extra", "geo", "hydro", "hyper",
        "inter", "intra", "macro", "mega", "micro", "mid", "mini", "mis",
        "mono", "multi", "nano", "neo", "neuro", "non", "ortho", "over",
        "photo", "poly", "post", "pre", "pro", "proto", "pseudo", "psycho",
        "quasi", "re", "retro", "self", "semi", "socio", "sub", "super",
        "tele", "thermo", "trans", "ultra", "un", "under", "uni",
    ];
    let lower = word.to_ascii_lowercase();
    for &pre in PREFIXES {
        if lower.len() <= pre.len() + 2 {
            continue;
        }
        if !lower.starts_with(pre) {
            continue;
        }
        let rest = &lower[pre.len()..];
        // Avoid splitting a real word that happens to start with a prefix-like
        // string ("under" + "stand" should NOT fire if "understand" is already
        // in gold — caller checks that). Require rest in gold or CMUdict.
        let Some(rest_ipa) = gold
            .lookup(rest)
            .map(str::to_owned)
            .or_else(|| lexicon.lookup(rest).map(arpabet::phones_to_ipa))
        else {
            continue;
        };
        let pre_ipa = gold
            .lookup(pre)
            .map(str::to_owned)
            .or_else(|| lexicon.lookup(pre).map(arpabet::phones_to_ipa))
            .unwrap_or_else(|| lts::pronounce_oov(pre));
        return Some(format!("{pre_ipa}{rest_ipa}"));
    }
    // Also try suffix split: split at common English noun-forming suffixes
    // like "-name", "-group", "-work", "-set" when the prefix is in gold.
    const SUFFIXES: &[&str] = &[
        "name", "group", "work", "set", "house", "board", "list",
        "mark", "point", "line", "case", "type", "side", "way",
        "time", "back", "down", "out", "off", "field", "space",
    ];
    for &suf in SUFFIXES {
        if lower.len() <= suf.len() + 2 {
            continue;
        }
        if !lower.ends_with(suf) {
            continue;
        }
        let head = &lower[..lower.len() - suf.len()];
        let Some(head_ipa) = gold
            .lookup(head)
            .map(str::to_owned)
            .or_else(|| lexicon.lookup(head).map(arpabet::phones_to_ipa))
        else {
            continue;
        };
        let suf_ipa = gold
            .lookup(suf)
            .map(str::to_owned)
            .or_else(|| lexicon.lookup(suf).map(arpabet::phones_to_ipa))
            .unwrap_or_else(|| lts::pronounce_oov(suf));
        return Some(format!("{head_ipa}{suf_ipa}"));
    }
    None
}

/// Last-resort fallback: collapse British "ae"/"oe" digraphs to "e" and
/// re-look-up. Catches Latin/Greek scientific terms like
/// "palaeolithic" → "paleolithic", "haemoglobin" → "hemoglobin",
/// "foetus" → "fetus", "oestrogen" → "estrogen" — words that LTS
/// otherwise reads literally as /pælæɛ.../ etc. Only fires after every
/// other lookup misses, so words that genuinely have an "ae"/"oe"
/// sequence with a distinct pronunciation (when they're in
/// gold/CMUdict) are untouched.
fn try_british_digraph(
    word: &str,
    gold: &misaki_gold::MisakiGoldLexicon,
    lexicon: &lexicon::Lexicon,
) -> Option<String> {
    let lower = word.to_ascii_lowercase();
    if !lower.contains("ae") && !lower.contains("oe") {
        return None;
    }
    let candidates = [
        lower.replace("ae", "e"),
        lower.replace("oe", "e"),
        lower.replace("ae", "e").replace("oe", "e"),
    ];
    for cand in &candidates {
        if cand == &lower {
            continue;
        }
        if let Some(ipa) = gold.lookup(cand) {
            return Some(ipa.to_owned());
        }
        if let Some(ipa) = lexicon.lookup(cand).map(arpabet::phones_to_ipa) {
            return Some(ipa);
        }
    }
    None
}

/// Derive a regular plural pronunciation from the gold singular when gold
/// has the base but not the plural ("centimes" → derive from "centime",
/// "krones" → derive from "krone"). This preempts CMUdict for words where
/// gold's curated singular pronunciation diverges from CMUdict's
/// English-anglicized plural (e.g. "centime" /sˈɑntˌim/ vs CMUdict
/// "CENTIMES" /sɛntaɪmz/). Only fires when gold has an exact match for
/// the bare singular, so irregular plurals (already in gold) are
/// unaffected.
fn try_gold_plural(word: &str, gold: &misaki_gold::MisakiGoldLexicon) -> Option<String> {
    if !word.ends_with('s') && !word.ends_with('S') {
        return None;
    }
    let lower = word.to_ascii_lowercase();
    // English plural: "y → ies" (babies/cities/parties). Try this
    // FIRST so we don't strip "es" and land on a nonsense gold stem
    // like "babi". Only when the y-form is in gold.
    if let Some(stem) = lower.strip_suffix("ies") {
        if !stem.is_empty() {
            let y_form = format!("{stem}y");
            if let Some(base_ipa) = gold.lookup(&y_form) {
                let suffix = possessive_phone_after(base_ipa);
                return Some(format!("{base_ipa}{suffix}"));
            }
        }
    }
    // Strip "s" first, then "es". Some words match both (e.g. "horses"
    // → "horse" via "s" strip), in which case the "s" strip wins —
    // that's what we want, since the singular is the one in gold.
    for trim in &[1usize, 2usize] {
        if lower.len() <= *trim {
            continue;
        }
        if *trim == 2 && !lower.ends_with("es") {
            continue;
        }
        let base = &lower[..lower.len() - trim];
        if let Some(base_ipa) = gold.lookup(base) {
            let suffix = possessive_phone_after(base_ipa);
            return Some(format!("{base_ipa}{suffix}"));
        }
    }
    None
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
    // First, try the de-hyphenated form in gold/CMUdict. Catches loanword
    // compounds like "tai-chi" where the dictionary stores "taichi" with
    // a /tʃi/ pronunciation that a part-by-part split would miss (the bare
    // "chi" is the Greek letter /kaɪ/).
    let dehyphenated: String = word.chars().filter(|c| *c != '-').collect();
    if !dehyphenated.is_empty() && dehyphenated != word {
        if let Some(ipa) = gold
            .lookup(&dehyphenated)
            .map(str::to_owned)
            .or_else(|| lexicon.lookup(&dehyphenated).map(arpabet::phones_to_ipa))
        {
            return Some(ipa);
        }
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
        // Sibilants (including the single-codepoint ligatures ʧ /tʃ/
        // and ʤ /dʒ/ that gold/arpabet emit) take the /ɪz/ allomorph —
        // "changes" → /tʃeɪndʒ/ + /ɪz/, not bare /z/.
        Some('s') | Some('z') | Some('ʃ') | Some('ʒ') | Some('ʧ') | Some('ʤ') => "ɪz",
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
            // Fold the right-single-quote U+2019 to ASCII apostrophe so
            // downstream lookups (CMUdict has "you're", not "you’re")
            // hit. The tokenizer already keeps them in the word; we
            // just need the spelling to match the dictionary.
            current.push(if ch == '\u{2019}' { '\'' } else { ch });
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
            .contains("ɹˈɛd"));
        assert!(TwoTierPhonemizer
            .phonemize("She will lead the meeting.")
            .unwrap()
            .contains("lˈid"));
    }

    #[test]
    fn two_tier_hyphenated_compounds_use_lexicon_parts() {
        // Each fragment of a hyphenated compound should be looked up in
        // gold/lexicon rather than dumped into LTS. "one" must come back as
        // its canonical /wʌn/, and the join is concatenation (no space)
        // so kokoro doesn't insert a word-boundary pause.
        let ipa = TwoTierPhonemizer.phonemize("one-twelfth").unwrap();
        assert!(
            ipa.contains("wˈʌn") || ipa.contains("wˈən"),
            "expected /wʌn/ from 'one' fragment, got {ipa:?}"
        );
        assert!(!ipa.contains(' '), "expected no space inside hyphenated compound, got {ipa:?}");

        let pre_war = TwoTierPhonemizer.phonemize("pre-war").unwrap();
        assert!(!pre_war.contains(' '), "expected no space inside 'pre-war', got {pre_war:?}");
    }
}

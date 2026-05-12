#![allow(dead_code)]

#[derive(Debug, Clone)]
struct Phone {
    ipa: String,
    vowel: bool,
}

pub fn pronounce_oov(word: &str) -> String {
    if let Some((base, possessive)) = split_possessive(word) {
        let mut ipa = pronounce_oov(base);
        if possessive {
            ipa.push('z');
        }
        return ipa;
    }

    if let Some(parts) = split_camel_case(word) {
        if parts.len() > 1 {
            let mut out = String::new();
            for (idx, part) in parts.iter().enumerate() {
                if idx > 0 {
                    out.push(' ');
                }
                out.push_str(&pronounce_oov(part));
            }
            return out;
        }
    }

    if word.contains('-') {
        let mut out = String::new();
        for (idx, part) in word.split('-').filter(|part| !part.is_empty()).enumerate() {
            if idx > 0 {
                out.push(' ');
            }
            out.push_str(&pronounce_oov(part));
        }
        return out;
    }

    let lower = word.to_ascii_lowercase();
    let mut phones = Vec::new();
    if let Some((prefix, rest)) = match_prefix(&lower) {
        phones.extend(prefix);
        phones.extend(transcribe_core(&rest));
    } else if let Some((stem, suffix)) = match_suffix(&lower) {
        phones.extend(transcribe_core(&stem));
        phones.extend(suffix);
    } else {
        phones.extend(transcribe_core(&lower));
    }

    if phones.is_empty() {
        return spell_out_word(word);
    }

    insert_stress_and_render(&phones)
}

fn insert_stress_and_render(phones: &[Phone]) -> String {
    let nuclei: Vec<usize> = phones
        .iter()
        .enumerate()
        .filter_map(|(idx, phone)| phone.vowel.then_some(idx))
        .collect();
    let stress_idx = match nuclei.len() {
        0 => None,
        1 | 2 => nuclei.first().copied(),
        n => nuclei.get(n - 2).copied(),
    };
    let mut out = String::new();
    for (idx, phone) in phones.iter().enumerate() {
        if Some(idx) == stress_idx {
            out.push('ˈ');
        }
        out.push_str(&phone.ipa);
    }
    out
}

fn transcribe_core(word: &str) -> Vec<Phone> {
    let chars: Vec<char> = word.chars().collect();
    if chars.is_empty() {
        return Vec::new();
    }
    let silent_e_vowel = silent_e_vowel_index(&chars);
    let mut phones = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        if let Some((phones_here, consumed)) = match_trigraphs(&chars, i) {
            phones.extend(phones_here);
            i += consumed;
            continue;
        }
        if let Some((phones_here, consumed)) = match_digraphs(&chars, i) {
            phones.extend(phones_here);
            i += consumed;
            continue;
        }
        let ch = chars[i];
        let next = chars.get(i + 1).copied();
        let prev = i.checked_sub(1).and_then(|j| chars.get(j).copied());
        if is_vowel(ch) {
            phones.extend(match ch {
                'a' => vowel_a(&chars, i, silent_e_vowel),
                'e' => vowel_e(&chars, i, silent_e_vowel),
                'i' => vowel_i(&chars, i, silent_e_vowel),
                'o' => vowel_o(&chars, i, silent_e_vowel),
                'u' => vowel_u(&chars, i, silent_e_vowel),
                'y' => vowel_y(&chars, i, silent_e_vowel),
                _ => Vec::new(),
            });
            i += 1;
            continue;
        }
        let mut pushed = true;
        match ch {
            'b' => phones.push(phone("b", false)),
            'c' => {
                if matches!(next, Some('e' | 'i' | 'y')) {
                    phones.push(phone("s", false));
                } else {
                    phones.push(phone("k", false));
                }
            }
            'd' => phones.push(phone("d", false)),
            'f' => phones.push(phone("f", false)),
            'g' => {
                if matches!(next, Some('e' | 'i' | 'y')) {
                    phones.push(phone("ʤ", false));
                } else {
                    phones.push(phone("ɡ", false));
                }
            }
            'h' => phones.push(phone("h", false)),
            'j' => phones.push(phone("ʤ", false)),
            'k' => phones.push(phone("k", false)),
            'l' => phones.push(phone("l", false)),
            'm' => phones.push(phone("m", false)),
            'n' => phones.push(phone("n", false)),
            'p' => phones.push(phone("p", false)),
            'q' => {
                phones.push(phone("k", false));
            }
            'r' => phones.push(phone("ɹ", false)),
            's' => phones.push(phone("s", false)),
            't' => phones.push(phone("t", false)),
            'v' => phones.push(phone("v", false)),
            'w' => phones.push(phone("w", false)),
            'x' => {
                phones.push(phone("k", false));
                phones.push(phone("s", false));
            }
            'z' => phones.push(phone("z", false)),
            'y' => {
                if i == 0 || matches!(prev, Some('-' | '\'')) {
                    phones.push(phone("j", false));
                } else if next.is_some() && is_vowel(next.unwrap_or(' ')) {
                    phones.push(phone("j", false));
                } else {
                    phones.push(phone("ɪ", true));
                }
            }
            _ => pushed = false,
        }
        if pushed {
            i += 1;
        } else {
            i += 1;
        }
    }
    phones
}

fn match_prefix(word: &str) -> Option<(Vec<Phone>, String)> {
    for (prefix, phones) in [
        (
            "under",
            vec![
                phone("ʌ", true),
                phone("n", false),
                phone("d", false),
                phone("ɹ", false),
            ],
        ),
        (
            "inter",
            vec![
                phone("ɪ", true),
                phone("n", false),
                phone("t", false),
                phone("ɹ", false),
            ],
        ),
        (
            "over",
            vec![phone("oʊ", true), phone("v", false), phone("ɚ", true)],
        ),
        (
            "trans",
            vec![
                phone("t", false),
                phone("ɹ", false),
                phone("æ", true),
                phone("n", false),
                phone("s", false),
            ],
        ),
        (
            "pre",
            vec![phone("p", false), phone("ɹ", false), phone("i", true)],
        ),
        ("re", vec![phone("ɹ", false), phone("i", true)]),
        (
            "dis",
            vec![phone("d", false), phone("ɪ", true), phone("s", false)],
        ),
        (
            "mis",
            vec![phone("m", false), phone("ɪ", true), phone("s", false)],
        ),
        (
            "non",
            vec![phone("n", false), phone("ɑ", true), phone("n", false)],
        ),
        ("un", vec![phone("ʌ", true), phone("n", false)]),
    ] {
        if let Some(rest) = word.strip_prefix(prefix) {
            if rest.len() >= 3 {
                return Some((phones, rest.to_string()));
            }
        }
    }
    None
}

fn match_suffix(word: &str) -> Option<(String, Vec<Phone>)> {
    for (suffix, phones) in [
        (
            "ization",
            vec![
                phone("ə", true),
                phone("z", false),
                phone("eɪ", true),
                phone("ʃ", false),
                phone("ən", true),
            ],
        ),
        (
            "ation",
            vec![phone("eɪ", true), phone("ʃ", false), phone("ən", true)],
        ),
        ("tion", vec![phone("ʃ", false), phone("ən", true)]),
        ("sion", vec![phone("ʒ", false), phone("ən", true)]),
        ("cian", vec![phone("ʃ", false), phone("ən", true)]),
        ("ing", vec![phone("ɪ", true), phone("ŋ", false)]),
        ("ed", vec![phone("ɪ", true), phone("d", false)]),
        ("ly", vec![phone("l", false), phone("i", true)]),
        ("er", vec![phone("əɹ", true)]),
        ("ous", vec![phone("ə", true), phone("s", false)]),
        (
            "able",
            vec![phone("ə", true), phone("b", false), phone("əl", true)],
        ),
        (
            "ible",
            vec![phone("ɪ", true), phone("b", false), phone("əl", true)],
        ),
        (
            "ment",
            vec![
                phone("m", false),
                phone("ə", true),
                phone("n", false),
                phone("t", false),
            ],
        ),
        (
            "ness",
            vec![phone("n", false), phone("ə", true), phone("s", false)],
        ),
        (
            "less",
            vec![phone("l", false), phone("ə", true), phone("s", false)],
        ),
        (
            "ful",
            vec![phone("f", false), phone("ə", true), phone("l", false)],
        ),
        (
            "ship",
            vec![phone("ʃ", false), phone("ɪ", true), phone("p", false)],
        ),
        (
            "hood",
            vec![phone("h", false), phone("ʊ", true), phone("d", false)],
        ),
        (
            "ist",
            vec![phone("ɪ", true), phone("s", false), phone("t", false)],
        ),
        (
            "ism",
            vec![phone("ɪ", true), phone("z", false), phone("əm", true)],
        ),
    ] {
        if let Some(stem) = word.strip_suffix(suffix) {
            if stem.len() >= 2 {
                return Some((stem.to_string(), phones));
            }
        }
    }
    None
}

fn match_trigraphs(chars: &[char], i: usize) -> Option<(Vec<Phone>, usize)> {
    let rem = &chars[i..];
    if starts_with(rem, "tch") {
        return Some((vec![phone("ʧ", false)], 3));
    }
    if starts_with(rem, "dge") {
        return Some((vec![phone("ʤ", false)], 3));
    }
    if starts_with(rem, "eigh") {
        return Some((vec![phone("eɪ", true)], 4));
    }
    if starts_with(rem, "igh") {
        return Some((vec![phone("aɪ", true)], 3));
    }
    if starts_with(rem, "sch") {
        return Some((vec![phone("s", false), phone("k", false)], 3));
    }
    if starts_with(rem, "qu") {
        return Some((vec![phone("k", false), phone("w", false)], 2));
    }
    if i == 0 && starts_with(rem, "ku") {
        return Some((vec![phone("k", false), phone("ju", true)], 2));
    }
    None
}

fn match_digraphs(chars: &[char], i: usize) -> Option<(Vec<Phone>, usize)> {
    let rem = &chars[i..];
    if starts_with(rem, "tion") {
        return Some((vec![phone("ʃ", false), phone("ən", true)], 4));
    }
    if starts_with(rem, "sion") {
        return Some((vec![phone("ʒ", false), phone("ən", true)], 4));
    }
    if starts_with(rem, "ph") {
        return Some((vec![phone("f", false)], 2));
    }
    if starts_with(rem, "ch") {
        return Some((vec![phone("ʧ", false)], 2));
    }
    if starts_with(rem, "sh") {
        return Some((vec![phone("ʃ", false)], 2));
    }
    if starts_with(rem, "th") {
        return Some((vec![phone("θ", false)], 2));
    }
    if starts_with(rem, "wh") {
        return Some((vec![phone("w", false)], 2));
    }
    if starts_with(rem, "ng") {
        return Some((vec![phone("ŋ", false)], 2));
    }
    if starts_with(rem, "ck") {
        return Some((vec![phone("k", false)], 2));
    }
    if starts_with(rem, "ee") || starts_with(rem, "ea") {
        return Some((vec![phone("i", true)], 2));
    }
    if starts_with(rem, "ar") {
        return Some((vec![phone("ɑɹ", true)], 2));
    }
    if starts_with(rem, "or") {
        return Some((vec![phone("ɔɹ", true)], 2));
    }
    if starts_with(rem, "ir") {
        return Some((vec![phone("ɪɹ", true)], 2));
    }
    if starts_with(rem, "ur") {
        return Some((vec![phone("ɜɹ", true)], 2));
    }
    if starts_with(rem, "ai") || starts_with(rem, "ay") {
        return Some((vec![phone("eɪ", true)], 2));
    }
    if starts_with(rem, "oa") || starts_with(rem, "oe") {
        return Some((vec![phone("oʊ", true)], 2));
    }
    if starts_with(rem, "oi") || starts_with(rem, "oy") {
        return Some((vec![phone("ɔɪ", true)], 2));
    }
    if starts_with(rem, "ow") {
        return Some((vec![phone("oʊ", true)], 2));
    }
    if starts_with(rem, "ou") {
        return Some((vec![phone("aʊ", true)], 2));
    }
    if starts_with(rem, "oo") {
        return Some((vec![phone("u", true)], 2));
    }
    if starts_with(rem, "ue") || starts_with(rem, "ew") {
        return Some((vec![phone("ju", true)], 2));
    }
    if starts_with(rem, "ie") {
        return Some((vec![phone("aɪ", true)], 2));
    }
    if starts_with(rem, "gn") {
        return Some((vec![phone("n", false)], 2));
    }
    if starts_with(rem, "kn") {
        return Some((vec![phone("n", false)], 2));
    }
    if starts_with(rem, "wr") {
        return Some((vec![phone("ɹ", false)], 2));
    }
    if starts_with(rem, "gh") {
        return Some((Vec::new(), 2));
    }
    if starts_with(rem, "ck") {
        return Some((vec![phone("k", false)], 2));
    }
    None
}

fn vowel_a(_chars: &[char], i: usize, silent_e_vowel: Option<usize>) -> Vec<Phone> {
    if Some(i) == silent_e_vowel {
        return vec![phone("eɪ", true)];
    }
    vec![phone("æ", true)]
}

fn vowel_e(chars: &[char], i: usize, silent_e_vowel: Option<usize>) -> Vec<Phone> {
    if Some(i) == silent_e_vowel {
        return vec![phone("i", true)];
    }
    if is_word_final_e(chars, i) {
        return Vec::new();
    }
    vec![phone("ɛ", true)]
}

fn vowel_i(_chars: &[char], i: usize, silent_e_vowel: Option<usize>) -> Vec<Phone> {
    if Some(i) == silent_e_vowel {
        return vec![phone("aɪ", true)];
    }
    vec![phone("ɪ", true)]
}

fn vowel_o(_chars: &[char], i: usize, silent_e_vowel: Option<usize>) -> Vec<Phone> {
    if Some(i) == silent_e_vowel {
        return vec![phone("oʊ", true)];
    }
    vec![phone("ɑ", true)]
}

fn vowel_u(_chars: &[char], i: usize, silent_e_vowel: Option<usize>) -> Vec<Phone> {
    if Some(i) == silent_e_vowel {
        return vec![phone("ju", true)];
    }
    vec![phone("ʌ", true)]
}

fn vowel_y(chars: &[char], i: usize, silent_e_vowel: Option<usize>) -> Vec<Phone> {
    if Some(i) == silent_e_vowel {
        return vec![phone("aɪ", true)];
    }
    if i == 0 {
        return vec![phone("j", false)];
    }
    if is_word_final(chars, i) {
        return vec![phone("aɪ", true)];
    }
    vec![phone("ɪ", true)]
}

fn is_word_final(chars: &[char], i: usize) -> bool {
    i + 1 == chars.len()
}

fn is_word_final_e(chars: &[char], i: usize) -> bool {
    is_word_final(chars, i) && chars[i] == 'e'
}

fn silent_e_vowel_index(chars: &[char]) -> Option<usize> {
    if chars.len() < 3 || chars.last().copied() != Some('e') {
        return None;
    }
    let mut last_vowel = None;
    for (idx, ch) in chars.iter().enumerate().take(chars.len() - 1) {
        if is_vowel(*ch) {
            last_vowel = Some(idx);
        }
    }
    let idx = last_vowel?;
    let trailing = chars[idx + 1..chars.len() - 1]
        .iter()
        .filter(|&&ch| !ch.is_ascii_alphabetic() || !is_vowel(ch))
        .count();
    if trailing <= 2 {
        Some(idx)
    } else {
        None
    }
}

fn split_possessive(word: &str) -> Option<(&str, bool)> {
    word.strip_suffix("'s")
        .or_else(|| word.strip_suffix("'S"))
        .map(|base| (base, true))
}

pub fn split_camel_case(word: &str) -> Option<Vec<String>> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut chars = word.chars().peekable();
    let mut prev_lower = false;
    while let Some(ch) = chars.next() {
        let is_boundary = prev_lower && ch.is_ascii_uppercase();
        if is_boundary && !current.is_empty() {
            parts.push(std::mem::take(&mut current));
        }
        current.push(ch);
        prev_lower = ch.is_ascii_lowercase();
    }
    if !current.is_empty() {
        parts.push(current);
    }
    if parts.len() > 1 {
        Some(parts)
    } else {
        None
    }
}

fn starts_with(chars: &[char], prefix: &str) -> bool {
    if chars.len() < prefix.len() {
        return false;
    }
    chars.iter().zip(prefix.chars()).all(|(a, b)| a == &b)
}

fn phone(ipa: &str, vowel: bool) -> Phone {
    Phone {
        ipa: ipa.to_string(),
        vowel,
    }
}

fn is_vowel(ch: char) -> bool {
    matches!(ch, 'a' | 'e' | 'i' | 'o' | 'u' | 'y')
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
    use super::pronounce_oov;

    #[test]
    fn handles_common_digraphs() {
        assert!(pronounce_oov("photon").contains("f"));
        assert!(pronounce_oov("church").contains("ʧ"));
        assert!(pronounce_oov("shoot").contains("ʃ"));
    }

    #[test]
    fn handles_silent_e_and_suffixes() {
        assert!(pronounce_oov("kite").contains("aɪ"));
        assert!(pronounce_oov("motion").contains("ʃ"));
        assert!(pronounce_oov("running").contains("ɪŋ"));
    }
}

#![allow(dead_code)]

pub fn phones_to_ipa(phones: &[&str]) -> String {
    let mut out = String::new();
    for phone in phones {
        if !out.is_empty() {
            // CMUdict tokens are concatenated per word, not separated by spaces.
        }
        out.push_str(&phone_to_ipa(phone));
    }
    out
}

fn phone_to_ipa(phone: &str) -> String {
    let (base, stress) = split_stress(phone);
    let stress_mark = match stress {
        1 => Some('ˈ'),
        2 => Some('ˌ'),
        _ => None,
    };
    let ipa = match base {
        "AA" => long_vowel("ɑ", stress),
        "AE" => "æ".to_string(),
        "AH" => ah(stress),
        "AO" => long_vowel("ɔ", stress),
        "AW" => diphthong("aʊ", stress_mark),
        "AY" => diphthong("aɪ", stress_mark),
        "B" => "b".to_string(),
        "CH" => affricate("ʧ", stress_mark),
        "D" => "d".to_string(),
        "DH" => "ð".to_string(),
        "EH" => "ɛ".to_string(),
        "ER" => er(stress),
        "EY" => diphthong("eɪ", stress_mark),
        "F" => "f".to_string(),
        "G" => "ɡ".to_string(),
        "HH" => "h".to_string(),
        "IH" => "ɪ".to_string(),
        "IY" => long_vowel("i", stress),
        "JH" => affricate("ʤ", stress_mark),
        "K" => "k".to_string(),
        "L" => "l".to_string(),
        "M" => "m".to_string(),
        "N" => "n".to_string(),
        "NG" => "ŋ".to_string(),
        "OW" => diphthong("oʊ", stress_mark),
        "OY" => diphthong("ɔɪ", stress_mark),
        "P" => "p".to_string(),
        "R" => "ɹ".to_string(),
        "S" => "s".to_string(),
        "SH" => "ʃ".to_string(),
        "T" => "t".to_string(),
        "TH" => "θ".to_string(),
        "UH" => "ʊ".to_string(),
        "UW" => long_vowel("u", stress),
        "V" => "v".to_string(),
        "W" => "w".to_string(),
        "Y" => "j".to_string(),
        "Z" => "z".to_string(),
        "ZH" => "ʒ".to_string(),
        "SIL" => " ".to_string(),
        other => panic!("unsupported ARPAbet symbol {other}"),
    };
    if stress_mark.is_some() && is_vowel(base) {
        let mut out = String::with_capacity(ipa.len() + 1);
        out.push(stress_mark.unwrap());
        out.push_str(&ipa);
        out
    } else {
        ipa
    }
}

fn split_stress(phone: &str) -> (&str, u8) {
    if let Some(stress) = phone.as_bytes().last().copied() {
        if (b'0'..=b'2').contains(&stress) {
            return (&phone[..phone.len() - 1], stress - b'0');
        }
    }
    (phone, 0)
}

fn is_vowel(base: &str) -> bool {
    matches!(base, "AA" | "AE" | "AH" | "AO" | "AW" | "AY" | "EH" | "ER" | "EY" | "IH" | "IY" | "OW" | "OY" | "UH" | "UW")
}

fn long_vowel(base: &str, stress: u8) -> String {
    match stress {
        1 | 2 => format!("{}ː", base),
        _ => base.to_string(),
    }
}

fn ah(stress: u8) -> String {
    match stress {
        0 => "ə".to_string(),
        1 => "ʌ".to_string(),
        2 => "ʌ".to_string(),
        _ => "ə".to_string(),
    }
}

fn er(stress: u8) -> String {
    match stress {
        0 => "ɚ".to_string(),
        1 => "ɜː".to_string(),
        2 => "ɜː".to_string(),
        _ => "ɚ".to_string(),
    }
}

fn diphthong(base: &str, stress_mark: Option<char>) -> String {
    let _ = stress_mark;
    base.to_string()
}

fn affricate(base: &str, stress_mark: Option<char>) -> String {
    diphthong(base, stress_mark)
}

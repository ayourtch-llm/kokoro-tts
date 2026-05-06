#![allow(dead_code)]

pub fn phones_to_ipa(phones: &[&str]) -> String {
    phones.iter().map(|phone| phone_to_ipa(phone)).collect()
}

fn phone_to_ipa(phone: &str) -> String {
    let (base, stress) = split_stress(phone);
    let stress_prefix = stress_prefix(stress);
    match base {
        "AA" => stressed_vowel("ɑ", &stress_prefix),
        "AE" => stressed_vowel("æ", &stress_prefix),
        "AH" => ah(stress, &stress_prefix),
        "AO" => stressed_vowel("ɔ", &stress_prefix),
        "AW" => stressed_diphthong("W", &stress_prefix),
        "AY" => stressed_diphthong("I", &stress_prefix),
        "B" => "b".to_string(),
        "CH" => "ʧ".to_string(),
        "D" => "d".to_string(),
        "DH" => "ð".to_string(),
        "EH" => "ɛ".to_string(),
        "ER" => er(stress, &stress_prefix),
        "EY" => stressed_diphthong("A", &stress_prefix),
        "F" => "f".to_string(),
        "G" => "ɡ".to_string(),
        "HH" => "h".to_string(),
        "IH" => "ɪ".to_string(),
        "IY" => stressed_vowel("i", &stress_prefix),
        "JH" => "ʤ".to_string(),
        "K" => "k".to_string(),
        "L" => "l".to_string(),
        "M" => "m".to_string(),
        "N" => "n".to_string(),
        "NG" => "ŋ".to_string(),
        "OW" => stressed_diphthong("O", &stress_prefix),
        "OY" => stressed_diphthong("Y", &stress_prefix),
        "P" => "p".to_string(),
        "R" => "ɹ".to_string(),
        "S" => "s".to_string(),
        "SH" => "ʃ".to_string(),
        "T" => "t".to_string(),
        "TH" => "θ".to_string(),
        "UH" => "ʊ".to_string(),
        "UW" => stressed_vowel("u", &stress_prefix),
        "V" => "v".to_string(),
        "W" => "w".to_string(),
        "Y" => "j".to_string(),
        "Z" => "z".to_string(),
        "ZH" => "ʒ".to_string(),
        "SIL" => " ".to_string(),
        other => panic!("unsupported ARPAbet symbol {other}"),
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

fn stress_prefix(stress: u8) -> &'static str {
    match stress {
        1 => "ˈ",
        2 => "ˌ",
        _ => "",
    }
}

fn stressed_vowel(base: &str, stress_prefix: &str) -> String {
    format!("{stress_prefix}{base}")
}

fn stressed_diphthong(base: &str, stress_prefix: &str) -> String {
    format!("{stress_prefix}{base}")
}

fn ah(stress: u8, stress_prefix: &str) -> String {
    let base = match stress {
        0 => "ə",
        1 | 2 => "ʌ",
        _ => "ə",
    };
    format!("{stress_prefix}{base}")
}

fn er(stress: u8, stress_prefix: &str) -> String {
    let base = match stress {
        0 => "əɹ",
        1 | 2 => "ɜɹ",
        _ => "əɹ",
    };
    format!("{stress_prefix}{base}")
}

#[cfg(test)]
mod tests {
    use super::phones_to_ipa;

    #[test]
    fn maps_misaki_style_vowels_and_diphthongs() {
        assert_eq!(phones_to_ipa(&["AA1"]), "ˈɑ");
        assert_eq!(phones_to_ipa(&["AO1"]), "ˈɔ");
        assert_eq!(phones_to_ipa(&["IY1"]), "ˈi");
        assert_eq!(phones_to_ipa(&["UW1"]), "ˈu");
        assert_eq!(phones_to_ipa(&["AW1"]), "ˈW");
        assert_eq!(phones_to_ipa(&["AY1"]), "ˈI");
        assert_eq!(phones_to_ipa(&["EY1"]), "ˈA");
        assert_eq!(phones_to_ipa(&["OW1"]), "ˈO");
        assert_eq!(phones_to_ipa(&["OY1"]), "ˈY");
    }

    #[test]
    fn maps_er_and_ah_conventions() {
        assert_eq!(phones_to_ipa(&["ER0"]), "əɹ");
        assert_eq!(phones_to_ipa(&["ER1"]), "ˈɜɹ");
        assert_eq!(phones_to_ipa(&["AH0"]), "ə");
        assert_eq!(phones_to_ipa(&["AH1"]), "ˈʌ");
    }
}

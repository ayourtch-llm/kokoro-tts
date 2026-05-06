#![allow(dead_code)]

pub fn normalize_cardinals(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if let Some((replacement, consumed)) = parse_token(&chars, i) {
            out.push_str(&replacement);
            i += consumed;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

pub fn normalize_abbreviations(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if let Some((replacement, consumed)) = match_abbreviation(&chars, i) {
            out.push_str(replacement);
            i += consumed;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

pub fn normalize_acronyms(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if let Some((replacement, consumed)) = match_acronym(&chars, i) {
            out.push_str(&replacement);
            i += consumed;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

pub fn normalize_money_time(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if let Some((replacement, consumed)) = match_money_prefix(&chars, i)
            .or_else(|| match_cents_suffix(&chars, i))
            .or_else(|| match_time(&chars, i))
        {
            out.push_str(&replacement);
            i += consumed;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

fn parse_token(chars: &[char], start: usize) -> Option<(String, usize)> {
    let mut i = start;
    let mut negative = false;
    if matches!(chars.get(i), Some('-' | '+')) {
        let sign = chars[i];
        let next = chars.get(i + 1)?;
        if !next.is_ascii_digit() {
            return None;
        }
        if !is_number_boundary(chars, start) {
            return None;
        }
        negative = sign == '-';
        i += 1;
    }

    if i >= chars.len() || !chars[i].is_ascii_digit() {
        return None;
    }

    let mut int_part = String::new();
    let mut frac_part = String::new();
    let mut decimal = false;
    let mut saw_comma = false;
    let mut saw_digit = false;
    while let Some(&ch) = chars.get(i) {
        if ch.is_ascii_digit() {
            saw_digit = true;
            if decimal {
                frac_part.push(ch);
            } else {
                int_part.push(ch);
            }
            i += 1;
            continue;
        }
        if ch == ',' && !decimal {
            let Some(next) = chars.get(i + 1) else {
                break;
            };
            if next.is_ascii_digit() {
                saw_comma = true;
                i += 1;
                continue;
            }
            break;
        }
        if ch == '.' && !decimal {
            let Some(next) = chars.get(i + 1) else {
                break;
            };
            if next.is_ascii_digit() {
                decimal = true;
                i += 1;
                continue;
            }
        }
        break;
    }
    if !saw_digit {
        return None;
    }

    let ord_suffix = ordinal_suffix(chars, i);
    if let Some((suffix, suffix_len)) = ord_suffix {
        if !decimal && is_number_boundary(chars, start) {
            let consumed = i - start + suffix_len;
            if token_ends_cleanly(chars, i + suffix_len) {
                return Some((ordinal_phrase(&int_part), consumed));
            }
        }
        if suffix != "th" && suffix != "st" && suffix != "nd" && suffix != "rd" {
            return None;
        }
    }

    if !is_number_boundary(chars, start) {
        return None;
    }

    if decimal {
        if frac_part.is_empty() {
            return None;
        }
        if !token_ends_cleanly(chars, i) {
            return None;
        }
        return Some((decimal_phrase(&int_part, negative, &frac_part), i - start));
    }

    let trimmed = int_part.trim_start_matches('0');
    let value = if trimmed.is_empty() { "0" } else { trimmed };
    if !saw_comma && value.len() == 4 {
        if let Ok(year) = value.parse::<u16>() {
            if (1000..=2099).contains(&year) && token_ends_cleanly(chars, i) {
                return Some((year_phrase(year), i - start));
            }
        }
    }

    if !token_ends_cleanly(chars, i) {
        return None;
    }
    Some((cardinal_phrase(&int_part, negative), i - start))
}

fn match_abbreviation(chars: &[char], start: usize) -> Option<(&'static str, usize)> {
    if start > 0 && chars[start - 1].is_ascii_alphanumeric() {
        return None;
    }
    const ABBREVIATIONS: &[(&str, &str)] = &[
        ("mrs.", "Missus"),
        ("mr.", "Mister"),
        ("ms.", "Miz"),
        ("dr.", "Doctor"),
        ("prof.", "Professor"),
        ("st.", "Saint"),
        ("jr.", "Junior"),
        ("sr.", "Senior"),
        ("e.g.", "for example"),
        ("i.e.", "that is"),
        ("etc.", "et cetera"),
        ("vs.", "versus"),
        ("cf.", "compare"),
        ("a.m.", "A M"),
        ("p.m.", "P M"),
    ];
    let tail = chars.get(start..)?;
    for (needle, replacement) in ABBREVIATIONS {
        if tail.len() < needle.len() {
            continue;
        }
        let matched = tail.get(..needle.len()).is_some_and(|prefix| {
            prefix
                .iter()
                .zip(needle.chars())
                .all(|(a, b)| a.eq_ignore_ascii_case(&b))
        });
        if matched {
            return Some((*replacement, needle.len()));
        }
    }
    None
}

fn match_acronym(chars: &[char], start: usize) -> Option<(String, usize)> {
    let tail = chars.get(start..)?;
    let mut end = 0usize;
    while let Some(ch) = tail.get(end) {
        if ch.is_ascii_alphabetic() {
            end += 1;
            continue;
        }
        if *ch == '\'' && matches!(tail.get(end + 1), Some('s' | 'S')) {
            end += 2;
            break;
        }
        break;
    }
    if end == 0 {
        return None;
    }
    let token: String = tail[..end].iter().collect();
    let Some((base, possessive)) = split_possessive(&token) else {
        return None;
    };
    if !is_all_caps_acronym(base) {
        return None;
    }
    if possessive || is_pronounce_as_word_acronym(base) {
        Some((token, end))
    } else {
        let replacement = spaced_letters(base);
        Some((replacement, end))
    }
}

fn split_possessive(token: &str) -> Option<(&str, bool)> {
    if token.len() > 2 && (token.ends_with("'s") || token.ends_with("'S")) {
        Some((&token[..token.len() - 2], true))
    } else {
        Some((token, false))
    }
}

fn match_money_prefix(chars: &[char], start: usize) -> Option<(String, usize)> {
    let symbol = *chars.get(start)?;
    let unit = match symbol {
        '$' => ("dollar", "dollars"),
        '€' => ("euro", "euros"),
        '£' => ("pound", "pounds"),
        '¥' => ("yen", "yen"),
        _ => return None,
    };
    let (int_part, frac_part, consumed) = scan_currency_amount(chars, start + 1)?;
    let trimmed = int_part.trim_start_matches('0');
    let value = if trimmed.is_empty() { "0" } else { trimmed };
    let mut out = String::new();
    out.push_str(&integer_to_words(value));
    out.push(' ');
    out.push_str(if value == "1" { unit.0 } else { unit.1 });
    if let Some(frac_part) = frac_part {
        if let Some(cents) = cents_words(&frac_part) {
            out.push(' ');
            out.push_str(&cents);
        }
    }
    Some((out, consumed + 1))
}

fn match_cents_suffix(chars: &[char], start: usize) -> Option<(String, usize)> {
    if !is_number_boundary(chars, start) {
        return None;
    }
    let (int_part, consumed) = scan_integer_span(chars, start)?;
    if !matches!(chars.get(start + consumed), Some('¢')) {
        return None;
    }
    let trimmed = int_part.trim_start_matches('0');
    let value = if trimmed.is_empty() { "0" } else { trimmed };
    let mut out = integer_to_words(value);
    out.push(' ');
    out.push_str(if value == "1" { "cent" } else { "cents" });
    Some((out, consumed + 1))
}

fn match_time(chars: &[char], start: usize) -> Option<(String, usize)> {
    if !matches!(chars.get(start), Some(ch) if ch.is_ascii_digit()) {
        return None;
    }
    if !is_number_boundary(chars, start) {
        return None;
    }
    let (hour, hour_len) = scan_integer_span(chars, start)?;
    let colon = start + hour_len;
    if !matches!(chars.get(colon), Some(':')) {
        return None;
    }
    let (minute, minute_len) = scan_integer_span(chars, colon + 1)?;
    if minute.is_empty() || matches!(chars.get(colon + 1 + minute_len), Some(':')) {
        return None;
    }
    let phrase = time_phrase(&hour, &minute);
    Some((phrase, hour_len + 1 + minute_len))
}

fn scan_currency_amount(chars: &[char], start: usize) -> Option<(String, Option<String>, usize)> {
    let mut i = start;
    let mut int_part = String::new();
    let mut frac_part = String::new();
    let mut decimal = false;
    let mut saw_digit = false;
    while let Some(&ch) = chars.get(i) {
        if ch.is_ascii_digit() {
            saw_digit = true;
            if decimal {
                frac_part.push(ch);
            } else {
                int_part.push(ch);
            }
            i += 1;
            continue;
        }
        if ch == ',' && !decimal {
            let Some(next) = chars.get(i + 1) else {
                break;
            };
            if next.is_ascii_digit() {
                i += 1;
                continue;
            }
            break;
        }
        if ch == '.' && !decimal {
            let Some(next) = chars.get(i + 1) else {
                break;
            };
            if next.is_ascii_digit() {
                decimal = true;
                i += 1;
                continue;
            }
        }
        break;
    }
    if !saw_digit {
        return None;
    }
    let frac = if decimal && !frac_part.is_empty() {
        Some(frac_part)
    } else {
        None
    };
    Some((int_part, frac, i - start))
}

fn scan_integer_span(chars: &[char], start: usize) -> Option<(String, usize)> {
    let mut i = start;
    let mut out = String::new();
    while let Some(&ch) = chars.get(i) {
        if ch.is_ascii_digit() {
            out.push(ch);
            i += 1;
        } else {
            break;
        }
    }
    if out.is_empty() {
        None
    } else {
        Some((out, i - start))
    }
}

fn cents_words(frac_part: &str) -> Option<String> {
    let mut digits = frac_part.chars().take(2).collect::<String>();
    if digits.is_empty() {
        return None;
    }
    if digits.len() == 1 {
        digits.push('0');
    }
    let trimmed = digits.trim_start_matches('0');
    let value = if trimmed.is_empty() { "0" } else { trimmed };
    let mut out = integer_to_words(value);
    out.push(' ');
    out.push_str(if value == "1" { "cent" } else { "cents" });
    Some(out)
}

fn time_phrase(hour: &str, minute: &str) -> String {
    let hour_words = integer_to_words(hour);
    let minute_trimmed = minute.trim_start_matches('0');
    if minute_trimmed.is_empty() {
        return hour_words;
    }
    if minute.len() == 2 && minute.starts_with('0') {
        return format!(
            "{hour_words} oh {}",
            digit_to_word(minute.chars().nth(1).unwrap())
        );
    }
    format!("{hour_words} {}", integer_to_words(minute))
}

pub fn is_pronounce_as_word_acronym(word: &str) -> bool {
    matches!(
        word,
        "NASA" | "NATO" | "RADAR" | "ASCII" | "JSON" | "ASAP" | "FAQ" | "PIN"
    )
}

pub fn is_all_caps_acronym(word: &str) -> bool {
    word.len() >= 2 && word.chars().all(|ch| ch.is_ascii_uppercase())
}

fn spaced_letters(word: &str) -> String {
    let mut out = String::new();
    for (idx, ch) in word.chars().enumerate() {
        if idx > 0 {
            out.push(' ');
        }
        out.push(ch);
    }
    out
}

fn is_number_boundary(chars: &[char], start: usize) -> bool {
    start == 0
        || !(chars[start - 1].is_ascii_alphanumeric() || matches!(chars[start - 1], ':' | '/'))
}

fn token_ends_cleanly(chars: &[char], end: usize) -> bool {
    match chars.get(end) {
        None => true,
        Some(ch) => !ch.is_ascii_alphabetic() && !matches!(ch, ':' | '/' | '%' | '$' | '°'),
    }
}

fn ordinal_suffix(chars: &[char], end: usize) -> Option<(&'static str, usize)> {
    let first = *chars.get(end)?;
    let second = chars.get(end + 1).copied()?;
    match (first.to_ascii_lowercase(), second.to_ascii_lowercase()) {
        ('s', 't') => Some(("st", 2)),
        ('n', 'd') => Some(("nd", 2)),
        ('r', 'd') => Some(("rd", 2)),
        ('t', 'h') => Some(("th", 2)),
        _ => None,
    }
}

fn cardinal_phrase(int_part: &str, negative: bool) -> String {
    let mut words = String::new();
    if negative {
        words.push_str("minus ");
    }
    words.push_str(&integer_to_words(int_part));
    words
}

fn decimal_phrase(int_part: &str, negative: bool, frac_part: &str) -> String {
    let mut words = cardinal_phrase(int_part, negative);
    words.push_str(" point");
    for ch in frac_part.chars() {
        words.push(' ');
        words.push_str(digit_to_word(ch));
    }
    words
}

fn ordinal_phrase(raw: &str) -> String {
    ordinalize_cardinal_phrase(&integer_to_words(raw))
}

fn ordinalize_cardinal_phrase(cardinal: &str) -> String {
    let mut parts: Vec<String> = cardinal.split_whitespace().map(str::to_owned).collect();
    if let Some(last) = parts.pop() {
        parts.push(match last.as_str() {
            "one" => "first".to_string(),
            "two" => "second".to_string(),
            "three" => "third".to_string(),
            "four" => "fourth".to_string(),
            "five" => "fifth".to_string(),
            "six" => "sixth".to_string(),
            "seven" => "seventh".to_string(),
            "eight" => "eighth".to_string(),
            "nine" => "ninth".to_string(),
            "ten" => "tenth".to_string(),
            "eleven" => "eleventh".to_string(),
            "twelve" => "twelfth".to_string(),
            "thirteen" => "thirteenth".to_string(),
            "fourteen" => "fourteenth".to_string(),
            "fifteen" => "fifteenth".to_string(),
            "sixteen" => "sixteenth".to_string(),
            "seventeen" => "seventeenth".to_string(),
            "eighteen" => "eighteenth".to_string(),
            "nineteen" => "nineteenth".to_string(),
            "twenty" => "twentieth".to_string(),
            "thirty" => "thirtieth".to_string(),
            "forty" => "fortieth".to_string(),
            "fifty" => "fiftieth".to_string(),
            "sixty" => "sixtieth".to_string(),
            "seventy" => "seventieth".to_string(),
            "eighty" => "eightieth".to_string(),
            "ninety" => "ninetieth".to_string(),
            "hundred" => "hundredth".to_string(),
            "thousand" => "thousandth".to_string(),
            "million" => "millionth".to_string(),
            "billion" => "billionth".to_string(),
            other => {
                let mut s = other.to_string();
                if s.ends_with('y') {
                    s.pop();
                    s.push_str("ieth");
                } else {
                    s.push_str("th");
                }
                s
            }
        });
    }
    parts.join(" ")
}

fn year_phrase(year: u16) -> String {
    match year {
        1000..=1099 => {
            let rest = year - 1000;
            if rest == 0 {
                "ten hundred".to_string()
            } else if rest < 10 {
                format!("ten oh {}", digit_to_word(char::from(b'0' + rest as u8)))
            } else {
                format!("ten {}", integer_to_words(&rest.to_string()))
            }
        }
        1100..=1999 => {
            let first_two = year / 100;
            let last_two = year % 100;
            if last_two == 0 {
                format!("{} hundred", integer_to_words(&first_two.to_string()))
            } else {
                format!(
                    "{} {}",
                    integer_to_words(&first_two.to_string()),
                    integer_to_words(&last_two.to_string())
                )
            }
        }
        2000 => "two thousand".to_string(),
        2001..=2009 => format!(
            "two thousand {}",
            integer_to_words(&(year - 2000).to_string())
        ),
        2010..=2099 => {
            let first_two = year / 100;
            let last_two = year % 100;
            format!(
                "{} {}",
                integer_to_words(&first_two.to_string()),
                integer_to_words(&last_two.to_string())
            )
        }
        _ => integer_to_words(&year.to_string()),
    }
}

fn integer_to_words(raw: &str) -> String {
    let trimmed = raw.trim_start_matches('0');
    let value = if trimmed.is_empty() { "0" } else { trimmed };
    let n: u128 = value.parse().unwrap_or(0);
    if n == 0 {
        return "zero".to_string();
    }
    const SCALES: [&str; 4] = ["", "thousand", "million", "billion"];
    let mut chunks = Vec::new();
    let mut remaining = n;
    let mut scale = 0usize;
    while remaining > 0 {
        let group = (remaining % 1000) as u16;
        if group != 0 {
            let mut part = convert_hundreds(group);
            if !SCALES[scale].is_empty() {
                part.push(' ');
                part.push_str(SCALES[scale]);
            }
            chunks.push(part);
        }
        remaining /= 1000;
        scale += 1;
    }
    chunks.reverse();
    chunks.join(" ")
}

fn convert_hundreds(n: u16) -> String {
    let hundreds = n / 100;
    let rem = n % 100;
    let mut out = String::new();
    if hundreds > 0 {
        out.push_str(UNITS[hundreds as usize]);
        out.push_str(" hundred");
        if rem > 0 {
            out.push(' ');
        }
    }
    if rem > 0 {
        if rem < 20 {
            out.push_str(UNITS[rem as usize]);
        } else {
            out.push_str(TENS[(rem / 10) as usize]);
            if rem % 10 != 0 {
                out.push(' ');
                out.push_str(UNITS[(rem % 10) as usize]);
            }
        }
    }
    out
}

fn digit_to_word(ch: char) -> &'static str {
    match ch {
        '0' => "zero",
        '1' => "one",
        '2' => "two",
        '3' => "three",
        '4' => "four",
        '5' => "five",
        '6' => "six",
        '7' => "seven",
        '8' => "eight",
        '9' => "nine",
        _ => "",
    }
}

const UNITS: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];

const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

#[cfg(test)]
mod tests {
    use super::{
        normalize_abbreviations, normalize_acronyms, normalize_cardinals, normalize_money_time,
    };

    #[test]
    fn normalizes_simple_integers() {
        assert_eq!(normalize_cardinals("82"), "eighty two");
        assert_eq!(
            normalize_cardinals("1,234"),
            "one thousand two hundred thirty four"
        );
    }

    #[test]
    fn normalizes_signed_and_decimal_numbers() {
        assert_eq!(normalize_cardinals("-42"), "minus forty two");
        assert_eq!(normalize_cardinals("3.14"), "three point one four");
        assert_eq!(normalize_cardinals("0.5"), "zero point five");
    }

    #[test]
    fn normalizes_ordinals() {
        assert_eq!(normalize_cardinals("1st"), "first");
        assert_eq!(normalize_cardinals("2nd"), "second");
        assert_eq!(normalize_cardinals("3rd"), "third");
        assert_eq!(normalize_cardinals("4th"), "fourth");
        assert_eq!(normalize_cardinals("21st"), "twenty first");
        assert_eq!(normalize_cardinals("100th"), "one hundredth");
        assert_eq!(normalize_cardinals("101st"), "one hundred first");
    }

    #[test]
    fn normalizes_years() {
        assert_eq!(normalize_cardinals("1234"), "twelve thirty four");
        assert_eq!(normalize_cardinals("1900"), "nineteen hundred");
        assert_eq!(normalize_cardinals("1999"), "nineteen ninety nine");
        assert_eq!(normalize_cardinals("2000"), "two thousand");
        assert_eq!(normalize_cardinals("2008"), "two thousand eight");
        assert_eq!(normalize_cardinals("2010"), "twenty ten");
        assert_eq!(normalize_cardinals("2026"), "twenty twenty six");
    }

    #[test]
    fn leaves_time_like_text_for_later_stages() {
        assert_eq!(normalize_cardinals("3:45"), "3:45");
    }

    #[test]
    fn normalizes_abbreviations_and_titles() {
        assert_eq!(
            normalize_abbreviations("Dr. Smith called Mr. Jones at 3 p.m."),
            "Doctor Smith called Mister Jones at 3 P M"
        );
        assert_eq!(
            normalize_abbreviations("Mrs. Lee met Ms. Kim."),
            "Missus Lee met Miz Kim."
        );
        assert_eq!(
            normalize_abbreviations("Prof. Adams lectured."),
            "Professor Adams lectured."
        );
    }

    #[test]
    fn normalizes_acronyms_for_spell_by_default_cases() {
        assert_eq!(normalize_acronyms("FBI"), "F B I");
        assert_eq!(normalize_acronyms("CIA"), "C I A");
        assert_eq!(normalize_acronyms("USA"), "U S A");
        assert_eq!(normalize_acronyms("SQL"), "S Q L");
        assert_eq!(normalize_acronyms("HTML"), "H T M L");
        assert_eq!(normalize_acronyms("USB"), "U S B");
    }

    #[test]
    fn keeps_pronounce_as_word_acronyms_unchanged() {
        assert_eq!(normalize_acronyms("NASA"), "NASA");
        assert_eq!(normalize_acronyms("NATO"), "NATO");
        assert_eq!(normalize_acronyms("RADAR"), "RADAR");
        assert_eq!(normalize_acronyms("JSON"), "JSON");
    }

    #[test]
    fn keeps_possessive_acronyms_intact_for_phonemizer() {
        assert_eq!(normalize_acronyms("NASA's"), "NASA's");
        assert_eq!(normalize_acronyms("FBI's"), "FBI's");
    }

    #[test]
    fn normalizes_money_and_time() {
        assert_eq!(normalize_money_time("$5"), "five dollars");
        assert_eq!(normalize_money_time("$1"), "one dollar");
        assert_eq!(normalize_money_time("$5.50"), "five dollars fifty cents");
        assert_eq!(
            normalize_money_time("$1,234.56"),
            "one thousand two hundred thirty four dollars fifty six cents"
        );
        assert_eq!(normalize_money_time("€5"), "five euros");
        assert_eq!(normalize_money_time("£1"), "one pound");
        assert_eq!(normalize_money_time("¥5"), "five yen");
        assert_eq!(normalize_money_time("5¢"), "five cents");
        assert_eq!(normalize_money_time("1¢"), "one cent");
        assert_eq!(normalize_money_time("3:45"), "three forty five");
        assert_eq!(normalize_money_time("3:00"), "three");
        assert_eq!(normalize_money_time("12:00"), "twelve");
    }
}

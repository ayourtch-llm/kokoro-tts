#![allow(dead_code)]

pub fn normalize_cardinals(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if let Some((replacement, consumed)) = parse_number(&chars, i) {
            out.push_str(&replacement);
            i += consumed;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

fn parse_number(chars: &[char], start: usize) -> Option<(String, usize)> {
    let mut i = start;
    let mut negative = false;
    if chars.get(i) == Some(&'-') {
        let Some(next) = chars.get(i + 1) else {
            return None;
        };
        if !next.is_ascii_digit() || start > 0 && chars[start - 1].is_ascii_alphabetic() {
            return None;
        }
        negative = true;
        i += 1;
    } else if chars.get(i) == Some(&'+') {
        let Some(next) = chars.get(i + 1) else {
            return None;
        };
        if !next.is_ascii_digit() {
            return None;
        }
        i += 1;
    }

    let mut int_part = String::new();
    let mut frac_part = String::new();
    let mut saw_digit = false;
    let mut decimal = false;
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
            i += 1;
            continue;
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
    if start > 0
        && (chars[start - 1].is_ascii_alphanumeric() || matches!(chars[start - 1], ':' | '/'))
    {
        return None;
    }
    if let Some(next) = chars.get(i) {
        if next.is_ascii_alphabetic() || matches!(next, ':' | '/' | '%' | '$' | '°') {
            return None;
        }
    }

    let mut words = String::new();
    if negative {
        words.push_str("minus ");
    }
    words.push_str(&integer_to_words(&int_part));
    if decimal {
        if frac_part.is_empty() {
            return None;
        }
        words.push_str(" point");
        for ch in frac_part.chars() {
            words.push(' ');
            words.push_str(digit_to_word(ch));
        }
    }
    Some((words, i - start))
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
    use super::normalize_cardinals;

    #[test]
    fn normalizes_simple_integers() {
        assert_eq!(normalize_cardinals("82"), "eighty two");
        assert_eq!(
            normalize_cardinals("1234"),
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
    fn leaves_ordinals_for_later_stages() {
        assert_eq!(normalize_cardinals("1st"), "1st");
        assert_eq!(normalize_cardinals("3:45"), "3:45");
    }
}

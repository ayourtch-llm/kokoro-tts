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
    normalize_acronyms_with(text, |_| false)
}

/// Fold common Latin diacritics to their ASCII bases so the tokenizer
/// (which treats only `is_ascii_alphabetic` chars as part of words)
/// doesn't split "fiancée" into "fianc" + "e" or drop "naïve" → "nave".
/// Targets English loanwords from French/German/Spanish — full Unicode
/// normalization isn't needed for the corpus this serves.
pub fn fold_diacritics(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        let folded = match ch {
            'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' | 'ă' | 'ą' => 'a',
            'À' | 'Á' | 'Â' | 'Ã' | 'Ä' | 'Å' | 'Ā' | 'Ă' | 'Ą' => 'A',
            'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ĕ' | 'ė' | 'ę' | 'ě' => 'e',
            'È' | 'É' | 'Ê' | 'Ë' | 'Ē' | 'Ĕ' | 'Ė' | 'Ę' | 'Ě' => 'E',
            'ì' | 'í' | 'î' | 'ï' | 'ī' | 'ĭ' | 'į' => 'i',
            'Ì' | 'Í' | 'Î' | 'Ï' | 'Ī' | 'Ĭ' | 'Į' => 'I',
            'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ō' | 'ŏ' | 'ő' => 'o',
            'Ò' | 'Ó' | 'Ô' | 'Õ' | 'Ö' | 'Ø' | 'Ō' | 'Ŏ' | 'Ő' => 'O',
            'ù' | 'ú' | 'û' | 'ü' | 'ū' | 'ŭ' | 'ů' | 'ű' | 'ų' => 'u',
            'Ù' | 'Ú' | 'Û' | 'Ü' | 'Ū' | 'Ŭ' | 'Ů' | 'Ű' | 'Ų' => 'U',
            'ñ' | 'ń' | 'ņ' | 'ň' => 'n',
            'Ñ' | 'Ń' | 'Ņ' | 'Ň' => 'N',
            'ç' | 'ć' | 'ĉ' | 'ċ' | 'č' => 'c',
            'Ç' | 'Ć' | 'Ĉ' | 'Ċ' | 'Č' => 'C',
            'ý' | 'ÿ' | 'ŷ' => 'y',
            'Ý' | 'Ÿ' | 'Ŷ' => 'Y',
            'ś' | 'ŝ' | 'ş' | 'š' => 's',
            'Ś' | 'Ŝ' | 'Ş' | 'Š' => 'S',
            'ź' | 'ż' | 'ž' => 'z',
            'Ź' | 'Ż' | 'Ž' => 'Z',
            'ĺ' | 'ļ' | 'ľ' | 'ł' => 'l',
            'Ĺ' | 'Ļ' | 'Ľ' | 'Ł' => 'L',
            'ŕ' | 'ŗ' | 'ř' => 'r',
            'Ŕ' | 'Ŗ' | 'Ř' => 'R',
            'ť' | 'ţ' => 't',
            'Ť' | 'Ţ' => 'T',
            'ď' | 'đ' => 'd',
            'Ď' | 'Đ' => 'D',
            'ğ' | 'ĝ' | 'ġ' | 'ģ' => 'g',
            'Ğ' | 'Ĝ' | 'Ġ' | 'Ģ' => 'G',
            'ĥ' | 'ħ' => 'h',
            'Ĥ' | 'Ħ' => 'H',
            'ĵ' => 'j',
            'Ĵ' => 'J',
            'ķ' => 'k',
            'Ķ' => 'K',
            'œ' => 'o', // Treat ligature as o; "oeuvre" → "ouvre" close enough.
            'Œ' => 'O',
            'æ' => 'a',
            'Æ' => 'A',
            'ß' => 's', // German eszett — drop to single s; not perfect but tokenizable.
            other => {
                out.push(other);
                continue;
            }
        };
        out.push(folded);
    }
    out
}

/// Lowercase 2-letter all-caps function words ("TO", "ON", "IN", …)
/// when adjacent to a 3+ letter all-caps token, so phrases like
/// "COME TO PARIS ON IMPORTANT BUSINESS" don't get the 2-letter
/// function words spelled out as "T O", "O N".
///
/// Isolated 2-letter all-caps tokens (UK, US, AS, IT in initialism
/// position) are left alone.
pub fn lowercase_emphasis_function_words(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    // Pass 1: collect alpha-token spans.
    let mut tokens: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_alphabetic() {
            let start = i;
            while i < chars.len() && chars[i].is_ascii_alphabetic() {
                i += 1;
            }
            tokens.push((start, i));
        } else {
            i += 1;
        }
    }
    // Identify 2-letter all-caps tokens to lowercase.
    let mut to_lower = vec![false; tokens.len()];
    for idx in 0..tokens.len() {
        let (s, e) = tokens[idx];
        if e - s != 2 {
            continue;
        }
        let tok: String = chars[s..e].iter().collect();
        if !is_all_caps_acronym(&tok) {
            continue;
        }
        let lower = tok.to_ascii_lowercase();
        if !is_emphasis_function_word(&lower) {
            continue;
        }
        let neighbor_is_caps_word = |nidx: usize| -> bool {
            let (ns, ne) = tokens[nidx];
            if ne - ns < 3 {
                return false;
            }
            let nt: String = chars[ns..ne].iter().collect();
            is_all_caps_acronym(&nt)
        };
        let prev_ok = idx > 0 && neighbor_is_caps_word(idx - 1);
        let next_ok = idx + 1 < tokens.len() && neighbor_is_caps_word(idx + 1);
        if prev_ok || next_ok {
            to_lower[idx] = true;
        }
    }
    // Pass 2: rebuild string.
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0;
    for (idx, &(s, e)) in tokens.iter().enumerate() {
        out.extend(chars[cursor..s].iter());
        if to_lower[idx] {
            for c in &chars[s..e] {
                out.push(c.to_ascii_lowercase());
            }
        } else {
            out.extend(chars[s..e].iter());
        }
        cursor = e;
    }
    out.extend(chars[cursor..].iter());
    out
}

fn is_emphasis_function_word(lower: &str) -> bool {
    matches!(
        lower,
        "to" | "in" | "on" | "at" | "of" | "or" | "as" | "by"
        | "is" | "be" | "an" | "no" | "so" | "we" | "he" | "my"
        | "me" | "do" | "if" | "it" | "up" | "am" | "go"
    )
}

/// Variant of `normalize_acronyms` that lets the caller suppress the
/// spell-out for tokens whose lowercased form is a real word (so
/// "BUT FIND LIVINGSTONE!" stays as words for the emphasis path in
/// `phonemize_word` to handle, instead of being pre-spelled to
/// "B U T F I N D L I V I N G S T O N E").
pub fn normalize_acronyms_with<F: Fn(&str) -> bool>(text: &str, is_real_word: F) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if let Some((replacement, consumed)) = match_acronym_with(&chars, i, &is_real_word) {
            out.push_str(&replacement);
            i += consumed;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

pub fn normalize_dates(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if let Some((replacement, consumed)) = match_iso_date(&chars, i)
            .or_else(|| match_slash_date(&chars, i))
            .or_else(|| match_hyphen_date(&chars, i))
            .or_else(|| match_month_date(&chars, i))
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

pub fn normalize_units(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if let Some((replacement, consumed)) = match_unit(&chars, i) {
            out.push_str(&replacement);
            i += consumed;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

pub fn normalize_math(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    let mut changed = false;
    while i < chars.len() {
        if let Some((replacement, consumed)) = match_math_operator(&chars, i) {
            out.push_str(&replacement);
            i += consumed;
            changed = true;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    if changed {
        collapse_whitespace(&out)
    } else {
        out
    }
}

fn match_math_operator(chars: &[char], start: usize) -> Option<(String, usize)> {
    let ch = *chars.get(start)?;
    match ch {
        '≤' => Some((" less than or equal to ".to_string(), 1)),
        '≥' => Some((" greater than or equal to ".to_string(), 1)),
        '≠' => Some((" not equal to ".to_string(), 1)),
        '×' => Some((" times ".to_string(), 1)),
        '÷' => Some((" divided by ".to_string(), 1)),
        '±' => Some((" plus or minus ".to_string(), 1)),
        '<' if matches!(chars.get(start + 1), Some('=')) => {
            if math_relational_context(chars, start, 2) {
                Some((" less than or equal to ".to_string(), 2))
            } else {
                None
            }
        }
        '>' if matches!(chars.get(start + 1), Some('=')) => {
            if math_relational_context(chars, start, 2) {
                Some((" greater than or equal to ".to_string(), 2))
            } else {
                None
            }
        }
        '+' => {
            if math_general_context(chars, start) {
                Some((" plus ".to_string(), 1))
            } else {
                None
            }
        }
        '=' => {
            if math_general_context(chars, start) {
                Some((" equals ".to_string(), 1))
            } else {
                None
            }
        }
        '-' => {
            if math_digit_context(chars, start) {
                Some((" minus ".to_string(), 1))
            } else {
                None
            }
        }
        '*' => {
            if math_digit_context(chars, start) {
                Some((" times ".to_string(), 1))
            } else {
                None
            }
        }
        '/' => {
            if math_slash_context(chars, start) {
                Some((" divided by ".to_string(), 1))
            } else {
                None
            }
        }
        '^' => {
            if math_exponent_context(chars, start) {
                Some((" to the power of ".to_string(), 1))
            } else {
                None
            }
        }
        '<' => {
            if math_relational_context(chars, start, 1) {
                Some((" less than ".to_string(), 1))
            } else {
                None
            }
        }
        '>' => {
            if math_relational_context(chars, start, 1) {
                Some((" greater than ".to_string(), 1))
            } else {
                None
            }
        }
        '%' => {
            if math_percent_context(chars, start) {
                Some((" percent ".to_string(), 1))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn math_general_context(chars: &[char], start: usize) -> bool {
    matches!(
        (
            prev_non_whitespace(chars, start),
            next_non_whitespace(chars, start + 1)
        ),
        (Some(left), Some(right)) if is_math_operand_char(left) && is_math_operand_char(right)
    )
}

fn math_relational_context(chars: &[char], start: usize, consumed: usize) -> bool {
    matches!(
        (
            prev_non_whitespace(chars, start),
            next_non_whitespace(chars, start + consumed)
        ),
        (Some(left), Some(right)) if is_math_operand_char(left) && is_math_operand_char(right)
    )
}

fn math_digit_context(chars: &[char], start: usize) -> bool {
    matches!(
        (
            prev_non_whitespace(chars, start),
            next_non_whitespace(chars, start + 1)
        ),
        (Some(left), Some(right)) if left.is_ascii_digit() && right.is_ascii_digit()
    )
}

fn math_exponent_context(chars: &[char], start: usize) -> bool {
    math_general_context(chars, start)
}

fn math_percent_context(chars: &[char], start: usize) -> bool {
    matches!(prev_non_whitespace(chars, start), Some(left) if left.is_ascii_digit())
}

fn math_slash_context(chars: &[char], start: usize) -> bool {
    let Some(left) = prev_non_whitespace(chars, start) else {
        return false;
    };
    let Some(right) = next_non_whitespace(chars, start + 1) else {
        return false;
    };
    if !left.is_ascii_digit() || !right.is_ascii_digit() {
        return false;
    }
    let left_space = start > 0 && chars[start - 1].is_ascii_whitespace();
    let right_space = matches!(chars.get(start + 1), Some(ch) if ch.is_ascii_whitespace());
    if left_space || right_space {
        return true;
    }
    digit_run_left(chars, start) > 1 || digit_run_right(chars, start + 1) > 1
}

fn prev_non_whitespace(chars: &[char], start: usize) -> Option<char> {
    if start == 0 {
        return None;
    }
    let mut i = start;
    while i > 0 {
        i -= 1;
        let ch = chars[i];
        if !ch.is_whitespace() {
            return Some(ch);
        }
    }
    None
}

fn next_non_whitespace(chars: &[char], start: usize) -> Option<char> {
    let mut i = start;
    while let Some(&ch) = chars.get(i) {
        if !ch.is_whitespace() {
            return Some(ch);
        }
        i += 1;
    }
    None
}

fn digit_run_left(chars: &[char], start: usize) -> usize {
    let mut count = 0usize;
    let mut i = start;
    while i > 0 {
        let ch = chars[i - 1];
        if ch.is_ascii_digit() {
            count += 1;
            i -= 1;
        } else {
            break;
        }
    }
    count
}

fn digit_run_right(chars: &[char], start: usize) -> usize {
    let mut count = 0usize;
    let mut i = start;
    while let Some(&ch) = chars.get(i) {
        if ch.is_ascii_digit() {
            count += 1;
            i += 1;
        } else {
            break;
        }
    }
    count
}

fn is_math_operand_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, ')' | ']' | '}')
}

fn collapse_whitespace(text: &str) -> String {
    let mut out = String::new();
    let mut last_was_space = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !last_was_space {
                out.push(' ');
                last_was_space = true;
            }
        } else {
            out.push(ch);
            last_was_space = false;
        }
    }
    out.trim().to_string()
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
            if (1000..=2099).contains(&year) {
                // Try decade-plural first ("1920s", "1920's", "1920’s") so the
                // bare-year branch doesn't strip the apostrophe and leave a
                // stray 's' for later passes.
                let plural_consumed = match (chars.get(i), chars.get(i + 1)) {
                    (Some(&'\''), Some(&'s')) | (Some(&'\''), Some(&'S')) => 2,
                    (Some(&'\u{2019}'), Some(&'s')) | (Some(&'\u{2019}'), Some(&'S')) => 2,
                    (Some(&'s'), _) | (Some(&'S'), _) => 1,
                    _ => 0,
                };
                if plural_consumed > 0 && token_ends_cleanly(chars, i + plural_consumed) {
                    return Some((
                        pluralize_year_phrase(year),
                        i + plural_consumed - start,
                    ));
                }
                if token_ends_cleanly(chars, i) {
                    return Some((year_phrase(year), i - start));
                }
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
    match_acronym_with(chars, start, &|_| false)
}

fn match_acronym_with<F: Fn(&str) -> bool>(
    chars: &[char],
    start: usize,
    is_real_word: &F,
) -> Option<(String, usize)> {
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
        return Some((token, end));
    }
    // Leave words-in-emphasis alone so phonemize_word's all-caps
    // emphasis branch reads them as words ("BUT" → /bʌt/, not "B U T").
    if is_real_word(base) {
        return Some((base.to_string(), end));
    }
    let replacement = spaced_letters(base);
    Some((replacement, end))
}

fn split_possessive(token: &str) -> Option<(&str, bool)> {
    if token.len() > 2 && (token.ends_with("'s") || token.ends_with("'S")) {
        Some((&token[..token.len() - 2], true))
    } else {
        Some((token, false))
    }
}

fn match_iso_date(chars: &[char], start: usize) -> Option<(String, usize)> {
    let (year, y_len) = scan_exact_digits(chars, start, 4)?;
    if !matches!(chars.get(start + y_len), Some('-')) {
        return None;
    }
    let (month, m_len) = scan_exact_digits(chars, start + y_len + 1, 2)?;
    if !matches!(chars.get(start + y_len + 1 + m_len), Some('-')) {
        return None;
    }
    let (day, d_len) = scan_day_token(chars, start + y_len + 1 + m_len + 1)?;
    let year_num = year.parse::<u16>().ok()?;
    let month_num = month.parse::<u8>().ok()?;
    let day_num = day.parse::<u8>().ok()?;
    let month_name = month_name_from_number(month_num)?;
    Some((
        format!(
            "{} {} {}",
            month_name,
            ordinal_phrase(&day_num.to_string()),
            year_phrase(year_num)
        ),
        y_len + 1 + m_len + 1 + d_len,
    ))
}

fn match_slash_date(chars: &[char], start: usize) -> Option<(String, usize)> {
    let (first, a_len) = scan_day_token(chars, start)?;
    if !matches!(chars.get(start + a_len), Some('/')) {
        return None;
    }
    let (second, b_len) = scan_day_token(chars, start + a_len + 1)?;
    if !matches!(chars.get(start + a_len + 1 + b_len), Some('/')) {
        return None;
    }
    let (year, c_len) = scan_exact_digits(chars, start + a_len + 1 + b_len + 1, 4)?;
    let month_num = first.parse::<u8>().ok()?;
    let day_num = second.parse::<u8>().ok()?;
    let year_num = year.parse::<u16>().ok()?;
    let month_name = month_name_from_number(month_num)?;
    Some((
        format!(
            "{} {} {}",
            month_name,
            ordinal_phrase(&day_num.to_string()),
            year_phrase(year_num)
        ),
        a_len + 1 + b_len + 1 + c_len,
    ))
}

fn match_hyphen_date(chars: &[char], start: usize) -> Option<(String, usize)> {
    let (first, a_len) = scan_day_token(chars, start)?;
    if !matches!(chars.get(start + a_len), Some('-')) {
        return None;
    }
    let (second, b_len) = scan_day_token(chars, start + a_len + 1)?;
    if !matches!(chars.get(start + a_len + 1 + b_len), Some('-')) {
        return None;
    }
    let (third, c_len) = scan_day_token(chars, start + a_len + 1 + b_len + 1)?;
    if first.len() == 4 && second.len() <= 2 && third.len() <= 2 {
        let year_num = first.parse::<u16>().ok()?;
        let month_num = second.parse::<u8>().ok()?;
        let day_num = third.parse::<u8>().ok()?;
        let month_name = month_name_from_number(month_num)?;
        return Some((
            format!(
                "{} {} {}",
                month_name,
                ordinal_phrase(&day_num.to_string()),
                year_phrase(year_num)
            ),
            a_len + 1 + b_len + 1 + c_len,
        ));
    }
    if first.len() <= 2 && second.len() <= 2 && third.len() == 4 {
        let month_num = first.parse::<u8>().ok()?;
        let day_num = second.parse::<u8>().ok()?;
        let year_num = third.parse::<u16>().ok()?;
        let month_name = month_name_from_number(month_num)?;
        return Some((
            format!(
                "{} {} {}",
                month_name,
                ordinal_phrase(&day_num.to_string()),
                year_phrase(year_num)
            ),
            a_len + 1 + b_len + 1 + c_len,
        ));
    }
    None
}

fn match_month_date(chars: &[char], start: usize) -> Option<(String, usize)> {
    let (month_raw, month_len, month_num) = scan_month_name(chars, start)?;
    let day_start = start + month_len;
    if !matches!(chars.get(day_start), Some(ch) if ch.is_ascii_whitespace()) {
        return None;
    }
    let mut i = day_start;
    while matches!(chars.get(i), Some(ch) if ch.is_ascii_whitespace()) {
        i += 1;
    }
    let (day, day_len) = scan_day_token(chars, i)?;
    let mut consumed = i - start + day_len;
    let mut year_part = None;
    let mut j = i + day_len;
    while matches!(chars.get(j), Some(ch) if ch.is_ascii_whitespace()) {
        j += 1;
        consumed += 1;
    }
    if matches!(chars.get(j), Some(',')) {
        j += 1;
        consumed += 1;
        while matches!(chars.get(j), Some(ch) if ch.is_ascii_whitespace()) {
            j += 1;
            consumed += 1;
        }
    }
    if let Some((year, year_len)) = scan_year_token(chars, j) {
        year_part = Some((year, year_len));
        consumed += year_len;
    }
    let day_num = day.parse::<u8>().ok()?;
    let month = month_name_from_number(month_num)?;
    let month_out = preserve_month_case(&month_raw, month);
    let mut out = format!("{} {}", month_out, ordinal_phrase(&day_num.to_string()));
    if let Some((year, _)) = year_part {
        let year_num = year.parse::<u16>().ok()?;
        out.push(' ');
        out.push_str(&year_phrase(year_num));
    }
    Some((out, consumed))
}

fn scan_month_name(chars: &[char], start: usize) -> Option<(String, usize, u8)> {
    let mut end = start;
    while let Some(ch) = chars.get(end) {
        if ch.is_ascii_alphabetic() {
            end += 1;
            continue;
        }
        break;
    }
    if end == start {
        return None;
    }
    let raw = chars.get(start..end)?.iter().collect::<String>();
    let raw_lower = raw.to_ascii_lowercase();
    let num = match raw_lower.as_str() {
        "january" | "jan" => 1,
        "february" | "feb" => 2,
        "march" | "mar" => 3,
        "april" | "apr" => 4,
        "may" => 5,
        "june" | "jun" => 6,
        "july" | "jul" => 7,
        "august" | "aug" => 8,
        "september" | "sep" | "sept" => 9,
        "october" | "oct" => 10,
        "november" | "nov" => 11,
        "december" | "dec" => 12,
        _ => return None,
    };
    let mut consumed = end - start;
    if matches!(chars.get(end), Some('.')) {
        consumed += 1;
    }
    Some((raw, consumed, num))
}

fn month_name_from_number(month: u8) -> Option<&'static str> {
    Some(match month {
        1 => "January",
        2 => "February",
        3 => "March",
        4 => "April",
        5 => "May",
        6 => "June",
        7 => "July",
        8 => "August",
        9 => "September",
        10 => "October",
        11 => "November",
        12 => "December",
        _ => return None,
    })
}

fn preserve_month_case(original: &str, canonical: &str) -> String {
    if original
        .chars()
        .all(|ch| ch.is_ascii_uppercase() || ch == '.')
    {
        canonical.to_ascii_uppercase()
    } else if original
        .chars()
        .next()
        .is_some_and(|ch| ch.is_ascii_uppercase())
    {
        canonical.to_string()
    } else {
        canonical.to_ascii_lowercase()
    }
}

fn scan_exact_digits(chars: &[char], start: usize, count: usize) -> Option<(String, usize)> {
    let mut out = String::new();
    for idx in 0..count {
        let ch = *chars.get(start + idx)?;
        if !ch.is_ascii_digit() {
            return None;
        }
        out.push(ch);
    }
    Some((out, count))
}

fn scan_day_token(chars: &[char], start: usize) -> Option<(String, usize)> {
    let mut i = start;
    let mut digits = String::new();
    while let Some(&ch) = chars.get(i) {
        if ch.is_ascii_digit() {
            digits.push(ch);
            i += 1;
        } else {
            break;
        }
    }
    if digits.is_empty() {
        return None;
    }
    if let Some((_, suffix_len)) = ordinal_suffix(chars, i) {
        i += suffix_len;
    }
    Some((digits, i - start))
}

fn scan_year_token(chars: &[char], start: usize) -> Option<(String, usize)> {
    let (digits, len) = scan_day_token(chars, start)?;
    if digits.len() == 4 {
        Some((digits, len))
    } else {
        None
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

fn match_unit(chars: &[char], start: usize) -> Option<(String, usize)> {
    let (number_raw, number_len) = scan_unit_number(chars, start)?;
    let mut idx = start + number_len;
    let had_space = matches!(chars.get(idx), Some(ch) if ch.is_whitespace());
    while matches!(chars.get(idx), Some(ch) if ch.is_whitespace()) {
        idx += 1;
    }
    let (singular, plural, unit_len, always_plural) = match_unit_suffix(chars, idx)?;
    if !unit_ends_cleanly(chars, idx + unit_len) {
        return None;
    }
    // Refuse "1920s" / "1990s" → "nineteen twenty seconds" / etc. When a
    // 4-digit year is glued directly to a single-letter unit (s, g, m, t),
    // it's almost always a decade plural; leave it for normalize_cardinals.
    if !had_space && unit_len == 1 && number_raw.len() == 4 {
        if let Ok(year) = number_raw.parse::<u16>() {
            if (1000..=2099).contains(&year) {
                return None;
            }
        }
    }
    let number_words = normalize_cardinals(&number_raw);
    let unit_word = if always_plural {
        if is_exact_one(&number_raw) {
            singular
        } else {
            plural
        }
    } else {
        singular
    };
    Some((
        format!("{number_words} {unit_word}"),
        idx + unit_len - start,
    ))
}

fn match_unit_suffix(
    chars: &[char],
    start: usize,
) -> Option<(&'static str, &'static str, usize, bool)> {
    let tail = chars.get(start..)?;
    let lower: String = tail.iter().take(5).collect::<String>().to_ascii_lowercase();
    if lower.starts_with("km/h") {
        return Some(("kilometers per hour", "kilometers per hour", 4, false));
    }
    if lower.starts_with("mph") {
        return Some(("miles per hour", "miles per hour", 3, false));
    }
    if lower.starts_with("kph") {
        return Some(("kilometers per hour", "kilometers per hour", 3, false));
    }
    if lower.starts_with("°c") {
        return Some(("degrees Celsius", "degrees Celsius", 2, false));
    }
    if lower.starts_with("°f") {
        return Some(("degrees Fahrenheit", "degrees Fahrenheit", 2, false));
    }
    if lower.starts_with("°k") {
        return Some(("degrees Kelvin", "degrees Kelvin", 2, false));
    }
    if lower.starts_with("mm") {
        return Some(("millimeter", "millimeters", 2, true));
    }
    if lower.starts_with("cm") {
        return Some(("centimeter", "centimeters", 2, true));
    }
    if lower.starts_with("km") {
        return Some(("kilometer", "kilometers", 2, true));
    }
    if lower.starts_with("in") {
        return Some(("inch", "inches", 2, true));
    }
    if lower.starts_with("ft") {
        return Some(("foot", "feet", 2, true));
    }
    if lower.starts_with("yd") {
        return Some(("yard", "yards", 2, true));
    }
    if lower.starts_with("mg") {
        return Some(("milligram", "milligrams", 2, true));
    }
    if lower.starts_with("g") {
        return Some(("gram", "grams", 1, true));
    }
    if lower.starts_with("kg") {
        return Some(("kilogram", "kilograms", 2, true));
    }
    if lower.starts_with("min") {
        return Some(("minute", "minutes", 3, true));
    }
    if lower.starts_with("mi") {
        return Some(("mile", "miles", 2, true));
    }
    if lower.starts_with("lb") {
        return Some(("pound", "pounds", 2, true));
    }
    if lower.starts_with("oz") {
        return Some(("ounce", "ounces", 2, true));
    }
    if lower.starts_with("hr") {
        return Some(("hour", "hours", 2, true));
    }
    if lower.starts_with("sec") {
        return Some(("second", "seconds", 3, true));
    }
    if lower.starts_with('s') {
        return Some(("second", "seconds", 1, true));
    }
    if lower.starts_with('m') {
        return Some(("meter", "meters", 1, true));
    }
    if lower.starts_with('t') {
        return Some(("ton", "tons", 1, true));
    }
    None
}

fn scan_unit_number(chars: &[char], start: usize) -> Option<(String, usize)> {
    if start > 0
        && (chars[start - 1].is_ascii_alphanumeric() || matches!(chars[start - 1], ':' | '/'))
    {
        return None;
    }
    let mut i = start;
    let mut out = String::new();
    if matches!(chars.get(i), Some('+' | '-')) {
        let sign = chars[i];
        let next = chars.get(i + 1)?;
        if !next.is_ascii_digit() {
            return None;
        }
        if sign == '-' {
            out.push(sign);
        }
        i += 1;
    }
    let mut saw_digit = false;
    while let Some(&ch) = chars.get(i) {
        if ch.is_ascii_digit() {
            saw_digit = true;
            out.push(ch);
            i += 1;
            continue;
        }
        if ch == ',' || ch == '.' {
            out.push(ch);
            i += 1;
            continue;
        }
        break;
    }
    if saw_digit {
        Some((out, i - start))
    } else {
        None
    }
}

fn unit_ends_cleanly(chars: &[char], end: usize) -> bool {
    match chars.get(end) {
        None => true,
        Some(ch) => !ch.is_ascii_alphabetic(),
    }
}

fn is_exact_one(raw: &str) -> bool {
    raw.trim_start_matches('0') == "1"
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

/// Decade-plural for "1920s" / "1920's" → "nineteen twenties".
/// Pluralizes the last word of the year phrase using English noun rules
/// for the words year_phrase actually emits.
fn pluralize_year_phrase(year: u16) -> String {
    let phrase = year_phrase(year);
    let mut parts: Vec<String> = phrase.split_whitespace().map(str::to_owned).collect();
    if let Some(last) = parts.pop() {
        let pluralized = match last.as_str() {
            "twenty" => "twenties".to_string(),
            "thirty" => "thirties".to_string(),
            "forty" => "forties".to_string(),
            "fifty" => "fifties".to_string(),
            "sixty" => "sixties".to_string(),
            "seventy" => "seventies".to_string(),
            "eighty" => "eighties".to_string(),
            "ninety" => "nineties".to_string(),
            "hundred" => "hundreds".to_string(),
            "thousand" => "thousands".to_string(),
            other => format!("{other}s"),
        };
        parts.push(pluralized);
    }
    parts.join(" ")
}

fn integer_to_words(raw: &str) -> String {
    let trimmed = raw.trim_start_matches('0');
    let value = if trimmed.is_empty() { "0" } else { trimmed };
    let n: u128 = value.parse().unwrap_or(0);
    if n == 0 {
        return "zero".to_string();
    }
    const SCALES: [&str; 10] = [
        "", "thousand", "million", "billion", "trillion",
        "quadrillion", "quintillion", "sextillion", "septillion", "octillion",
    ];
    let mut chunks = Vec::new();
    let mut remaining = n;
    let mut scale = 0usize;
    while remaining > 0 {
        let group = (remaining % 1000) as u16;
        if group != 0 {
            let mut part = convert_hundreds(group);
            let scale_name = if scale < SCALES.len() {
                SCALES[scale]
            } else {
                ""
            };
            if !scale_name.is_empty() {
                part.push(' ');
                part.push_str(scale_name);
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
        lowercase_emphasis_function_words, normalize_abbreviations, normalize_acronyms,
        normalize_cardinals, normalize_dates, normalize_math, normalize_money_time, normalize_units,
    };

    #[test]
    fn lowercases_two_letter_function_words_in_emphasis_runs() {
        assert_eq!(
            lowercase_emphasis_function_words("COME TO PARIS ON IMPORTANT BUSINESS"),
            "COME to PARIS on IMPORTANT BUSINESS"
        );
        assert_eq!(
            lowercase_emphasis_function_words("BUT FIND LIVINGSTONE"),
            "BUT FIND LIVINGSTONE"
        );
    }

    #[test]
    fn leaves_isolated_two_letter_caps_alone() {
        assert_eq!(lowercase_emphasis_function_words("the US economy"), "the US economy");
        assert_eq!(lowercase_emphasis_function_words("In IT we trust"), "In IT we trust");
    }

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
    fn normalizes_decade_plurals() {
        assert_eq!(normalize_cardinals("1920s"), "nineteen twenties");
        assert_eq!(normalize_cardinals("1990s"), "nineteen nineties");
        assert_eq!(normalize_cardinals("2010s"), "twenty tens");
        assert_eq!(normalize_cardinals("1900s"), "nineteen hundreds");
        assert_eq!(normalize_cardinals("2000s"), "two thousands");
        assert_eq!(normalize_cardinals("1920's"), "nineteen twenties");
        assert_eq!(
            normalize_cardinals("In the 1920s the world changed."),
            "In the nineteen twenties the world changed."
        );
    }

    #[test]
    fn decade_survives_full_pipeline() {
        // Mirror the order phonemizer.rs::phonemize uses, to catch regressions
        // where an earlier pass mangles "1920s" before normalize_cardinals runs.
        let pipeline = |text: &str| {
            normalize_cardinals(&normalize_acronyms(&normalize_units(
                &normalize_money_time(&normalize_math(&normalize_dates(
                    &normalize_abbreviations(text),
                ))),
            )))
        };
        assert_eq!(
            pipeline("In the 1920s the world changed."),
            "In the nineteen twenties the world changed."
        );
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

    #[test]
    fn normalizes_dates() {
        assert_eq!(normalize_dates("2026-05-06"), "May sixth twenty twenty six");
        assert_eq!(normalize_dates("5/6/2026"), "May sixth twenty twenty six");
        assert_eq!(normalize_dates("5-6-2026"), "May sixth twenty twenty six");
        assert_eq!(
            normalize_dates("May 6, 2026"),
            "May sixth twenty twenty six"
        );
        assert_eq!(
            normalize_dates("May 6th, 2026"),
            "May sixth twenty twenty six"
        );
        assert_eq!(normalize_dates("May 5"), "May fifth");
        assert_eq!(normalize_dates("Monday, May 6th"), "Monday, May sixth");
    }

    #[test]
    fn normalizes_units() {
        assert_eq!(normalize_units("5 kg"), "five kilograms");
        assert_eq!(normalize_units("1 kg"), "one kilogram");
        assert_eq!(normalize_units("0 kg"), "zero kilograms");
        assert_eq!(normalize_units("0.5 kg"), "zero point five kilograms");
        assert_eq!(normalize_units("5.5 kg"), "five point five kilograms");
        assert_eq!(normalize_units("5 km"), "five kilometers");
        assert_eq!(normalize_units("5 g"), "five grams");
        assert_eq!(normalize_units("5 mph"), "five miles per hour");
        assert_eq!(normalize_units("5 km/h"), "five kilometers per hour");
        assert_eq!(normalize_units("5°C"), "five degrees Celsius");
        assert_eq!(normalize_units("1 ft"), "one foot");
        assert_eq!(normalize_units("2 ft"), "two feet");
        assert_eq!(normalize_units("5 s"), "five seconds");
        assert_eq!(normalize_units("5 sec"), "five seconds");
        assert_eq!(normalize_units("5 min"), "five minutes");
        assert_eq!(normalize_units("5 hr"), "five hours");
    }

    #[test]
    fn normalizes_math_symbols() {
        assert_eq!(normalize_math("2 + 2 = 4"), "2 plus 2 equals 4");
        assert_eq!(normalize_math("5 - 3"), "5 minus 3");
        assert_eq!(normalize_math("2*3"), "2 times 3");
        assert_eq!(normalize_math("2 * 3"), "2 times 3");
        assert_eq!(normalize_math("10/2"), "10 divided by 2");
        assert_eq!(normalize_math("8 / 2"), "8 divided by 2");
        assert_eq!(normalize_math("x^2"), "x to the power of 2");
        assert_eq!(normalize_math("a <= b"), "a less than or equal to b");
        assert_eq!(normalize_math("a >= b"), "a greater than or equal to b");
        assert_eq!(normalize_math("a < b"), "a less than b");
        assert_eq!(normalize_math("a > b"), "a greater than b");
        assert_eq!(normalize_math("3×4"), "3 times 4");
        assert_eq!(normalize_math("8÷2"), "8 divided by 2");
        assert_eq!(normalize_math("x≠y"), "x not equal to y");
        assert_eq!(normalize_math("x≤y"), "x less than or equal to y");
        assert_eq!(normalize_math("x≥y"), "x greater than or equal to y");
        assert_eq!(normalize_math("10 ± 2"), "10 plus or minus 2");
        assert_eq!(normalize_math("50%"), "50 percent");
        assert_eq!(normalize_math("text-to-speech"), "text-to-speech");
        assert_eq!(normalize_math("**bold**"), "**bold**");
        assert_eq!(normalize_math("5/6"), "5/6");
    }
}

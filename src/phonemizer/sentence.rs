const ABBREVIATIONS: &[&str] = &[
    "mrs.", "mr.", "ms.", "dr.", "st.", "jr.", "sr.", "e.g.", "i.e.", "etc.", "vs.",
];

pub(crate) fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut i = 0;
    while i < text.len() {
        if let Some(len) = match_abbreviation(text, i) {
            current.push_str(&text[i..i + len]);
            i += len;
            continue;
        }

        if text[i..].starts_with("...") {
            current.push_str("...");
            i += 3;
            continue;
        }

        let ch = text[i..].chars().next().expect("valid utf-8 boundary");
        let ch_len = ch.len_utf8();

        if ch == '\n' {
            let next = text[i + ch_len..].chars().next();
            if next == Some('\n') {
                if !current.trim().is_empty() {
                    out.push(current.trim().to_string());
                }
                current.clear();
                i += ch_len;
                while i < text.len() {
                    let Some(next_ch) = text[i..].chars().next() else {
                        break;
                    };
                    if next_ch != '\n' {
                        break;
                    }
                    i += next_ch.len_utf8();
                }
                continue;
            } else {
                current.push(' ');
                i += ch_len;
                continue;
            }
        }

        current.push(ch);
        i += ch_len;

        if matches!(ch, '.' | '!' | '?') && should_end_sentence(text, i - ch_len, ch) {
            if !current.trim().is_empty() {
                out.push(current.trim().to_string());
            }
            current.clear();
        }
    }

    if !current.trim().is_empty() {
        out.push(current.trim().to_string());
    }
    out
}

fn match_abbreviation(text: &str, start: usize) -> Option<usize> {
    let tail = text.get(start..)?;
    for abbrev in ABBREVIATIONS {
        if tail.len() >= abbrev.len()
            && tail
                .get(..abbrev.len())
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case(abbrev))
        {
            return Some(abbrev.len());
        }
    }
    None
}

fn should_end_sentence(text: &str, dot_index: usize, ch: char) -> bool {
    if ch != '.' {
        return true;
    }

    if dot_index > 0 {
        let prev = text[..dot_index].chars().next_back();
        let next = text[dot_index + 1..].chars().next();
        if matches!((prev, next), (Some(p), Some(n)) if p.is_ascii_digit() && n.is_ascii_digit()) {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::split_sentences;

    #[test]
    fn keeps_abbreviations_inside_sentence() {
        assert_eq!(
            split_sentences("Mr. Smith arrived. He waited."),
            vec!["Mr. Smith arrived.", "He waited."]
        );
    }

    #[test]
    fn keeps_decimal_points_inside_sentence() {
        assert_eq!(
            split_sentences("She's 3.14 meters tall? Yes."),
            vec!["She's 3.14 meters tall?", "Yes."]
        );
    }

    #[test]
    fn splits_on_blank_lines() {
        assert_eq!(
            split_sentences("Hello.\n\nWorld."),
            vec!["Hello.", "World."]
        );
    }
}

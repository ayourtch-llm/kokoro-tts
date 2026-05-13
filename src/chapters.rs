//! Heuristic chapter splitter for plain-text books.
//!
//! Used by `speak --auto-split` to break a long input (e.g. a book) into
//! chapter-sized chunks so each is written to its own numbered audio
//! file. The detector is intentionally conservative: real e-books use a
//! wide range of formatting, so the rule is the most robust signal —
//! "the prose just ended and a heading is starting".

/// Splits a long text into chapter-sized chunks.
///
/// A chapter boundary is placed at every transition from a *body*
/// paragraph to a *heading* paragraph, where a paragraph is a run of
/// non-blank lines bounded by blank lines, and a paragraph is a
/// "heading" if it's short, has no terminal sentence punctuation, and
/// looks heading-like (chapter/part keyword, lone number, Roman
/// numeral, mostly-uppercase, or Title Case).
///
/// Returns the chapter slices in document order. Each slice preserves
/// the original whitespace (including internal blank lines, which the
/// downstream synthesizer renders as silence). The blank lines *between*
/// chapters are discarded.
///
/// If the text contains no body→heading transition, the entire input is
/// returned as a single chapter.
pub fn split_chapters(text: &str) -> Vec<String> {
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let segments = parse_segments(&normalized);
    if segments.is_empty() {
        return Vec::new();
    }

    let mut chapter_starts: Vec<usize> = vec![0];
    let mut last_kind = SegmentKind::None;
    for (idx, seg) in segments.iter().enumerate() {
        let seg_text = &normalized[seg.start..seg.end];
        let kind = classify(seg_text);
        if idx > 0
            && matches!(last_kind, SegmentKind::Body)
            && matches!(kind, SegmentKind::Heading)
        {
            chapter_starts.push(idx);
        }
        last_kind = kind;
    }

    let mut chapters = Vec::with_capacity(chapter_starts.len());
    for (i, &start_seg) in chapter_starts.iter().enumerate() {
        let start_byte = segments[start_seg].start;
        let end_seg = if i + 1 < chapter_starts.len() {
            chapter_starts[i + 1] - 1
        } else {
            segments.len() - 1
        };
        let end_byte = segments[end_seg].end;
        chapters.push(normalized[start_byte..end_byte].to_string());
    }
    chapters
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SegmentKind {
    None,
    Body,
    Heading,
}

struct Segment {
    start: usize,
    end: usize,
}

fn parse_segments(text: &str) -> Vec<Segment> {
    // Collect (byte_offset, line_content) for every line. split('\n')
    // strips the newline, so add 1 to each line's length when advancing
    // the offset.
    let mut lines: Vec<(usize, &str)> = Vec::new();
    let mut offset = 0usize;
    for line in text.split('\n') {
        lines.push((offset, line));
        offset += line.len() + 1;
    }

    let mut segments = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        while i < lines.len() && lines[i].1.trim().is_empty() {
            i += 1;
        }
        if i >= lines.len() {
            break;
        }
        let start = lines[i].0;
        let mut end = lines[i].0 + lines[i].1.len();
        i += 1;
        while i < lines.len() && !lines[i].1.trim().is_empty() {
            end = lines[i].0 + lines[i].1.len();
            i += 1;
        }
        segments.push(Segment { start, end });
    }
    segments
}

fn classify(text: &str) -> SegmentKind {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return SegmentKind::None;
    }
    if is_heading(trimmed) {
        SegmentKind::Heading
    } else {
        SegmentKind::Body
    }
}

const HEADING_KEYWORDS: &[&str] = &[
    "chapter",
    "part",
    "book",
    "section",
    "prologue",
    "epilogue",
    "interlude",
    "preface",
    "foreword",
    "afterword",
    "introduction",
];

fn is_heading(text: &str) -> bool {
    if text.chars().count() > 60 {
        return false;
    }
    // Reject anything that ends with a sentence terminator — a real
    // heading rarely does.
    if let Some(last) = text.chars().rev().find(|c| !c.is_whitespace()) {
        if matches!(last, '.' | '!' | '?') {
            return false;
        }
    }
    if !text.chars().any(|c| c.is_alphanumeric()) {
        return false;
    }
    // Em-dash / en-dash prefix is the conventional byline marker
    // ("—John Boyne") after a section. Treat as body so it doesn't
    // start a spurious one-line "chapter" of its own.
    if let Some(first) = text.chars().next() {
        if matches!(first, '—' | '–') {
            return false;
        }
    }

    let lower = text.to_ascii_lowercase();
    for kw in HEADING_KEYWORDS {
        if lower.starts_with(kw) {
            let after = lower[kw.len()..].chars().next();
            if !matches!(after, Some(c) if c.is_alphabetic()) {
                return true;
            }
        }
    }

    // A lone chapter number ("1", "23"). Allow up to 4 digits.
    if text.chars().all(|c| c.is_ascii_digit()) && text.len() <= 4 {
        return true;
    }
    if is_roman_numeral(text) {
        return true;
    }

    // Majority of the alphabetic characters are uppercase. Tolerates
    // typesetting oddities like "iNTRODUCTION" (caps + leading-cap
    // dropcap).
    let alpha: Vec<char> = text.chars().filter(|c| c.is_alphabetic()).collect();
    if !alpha.is_empty() {
        let upper = alpha.iter().filter(|c| c.is_uppercase()).count();
        if upper * 2 > alpha.len() {
            return true;
        }
    }

    // Multi-word Title Case ("The Beginning", "Logotherapy in a
    // Nutshell"). Single-word capitalization is too ambiguous (could
    // just be the start of an unpunctuated body sentence).
    //
    // Allow common lowercase "stopwords" (in, a, the, of, by, …) since
    // real book titles capitalize content words only.
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() >= 2 {
        let mut content_words = 0usize;
        let mut content_caps = 0usize;
        for w in &words {
            // Strip leading non-alphanumerics so "(Abridged)" still
            // counts as starting with "A".
            let first_alpha = w.chars().find(|c| c.is_alphabetic());
            let Some(first) = first_alpha else { continue };
            let lower = w.trim_matches(|c: char| !c.is_alphabetic()).to_ascii_lowercase();
            if is_title_stopword(&lower) {
                continue;
            }
            content_words += 1;
            if first.is_uppercase() {
                content_caps += 1;
            }
        }
        if content_words >= 1 && content_caps == content_words {
            return true;
        }
    }

    false
}

fn is_title_stopword(w: &str) -> bool {
    matches!(
        w,
        "a" | "an"
            | "and"
            | "as"
            | "at"
            | "but"
            | "by"
            | "for"
            | "from"
            | "in"
            | "into"
            | "of"
            | "on"
            | "or"
            | "the"
            | "to"
            | "with"
            | "via"
            | "vs"
    )
}

fn is_roman_numeral(s: &str) -> bool {
    if s.is_empty() || s.len() > 8 {
        return false;
    }
    s.chars()
        .all(|c| matches!(c, 'I' | 'V' | 'X' | 'L' | 'C' | 'D' | 'M'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_paragraph_is_one_chapter() {
        let chapters = split_chapters("Just one paragraph of body text.");
        assert_eq!(chapters.len(), 1);
    }

    #[test]
    fn empty_input_returns_empty() {
        assert!(split_chapters("").is_empty());
        assert!(split_chapters("\n\n\n").is_empty());
    }

    #[test]
    fn body_then_heading_then_body_splits_in_two() {
        let text = "\
The intro paragraph.

CHAPTER ONE

The first sentence of chapter one.
";
        let chapters = split_chapters(text);
        assert_eq!(chapters.len(), 2);
        assert!(chapters[0].contains("intro paragraph"));
        assert!(chapters[1].contains("CHAPTER ONE"));
        assert!(chapters[1].contains("first sentence"));
    }

    #[test]
    fn two_chapters_with_minimal_gap() {
        let text = "\
End of chapter one.

CHAPTER TWO

Beginning of chapter two.
";
        let chapters = split_chapters(text);
        assert_eq!(chapters.len(), 2);
        assert!(chapters[0].starts_with("End of"));
        assert!(chapters[1].contains("CHAPTER TWO"));
    }

    #[test]
    fn consecutive_heading_lines_stay_together() {
        // Mirrors the real-book layout: PART / TITLE / chapter-number /
        // chapter-name all belong to the SAME upcoming chapter.
        let text = "\
PART ONE:

YOUR BRAIN IS A WORK IN PROGRESS




1



iNTRODUCTION




We are in the early stages of a Brain Plasticity Revolution. That revolution begins with a clearer understanding that the brain's machinery is being continuously rewired.

Not too many years ago, the average citizen of the world was ignorant about the relationship between physical exercise and health.
";
        let chapters = split_chapters(text);
        assert_eq!(
            chapters.len(),
            1,
            "should not split between consecutive heading lines"
        );
        let only = &chapters[0];
        assert!(only.contains("PART ONE"));
        assert!(only.contains("iNTRODUCTION"));
        assert!(only.contains("Brain Plasticity"));
    }

    #[test]
    fn three_chapter_book() {
        let text = "\
Preamble.

CHAPTER 1

First body.

CHAPTER 2

Second body.
";
        let chapters = split_chapters(text);
        assert_eq!(chapters.len(), 3);
        assert!(chapters[0].starts_with("Preamble"));
        assert!(chapters[1].contains("CHAPTER 1"));
        assert!(chapters[2].contains("CHAPTER 2"));
    }

    #[test]
    fn paragraphs_inside_chapter_are_kept_together() {
        let text = "\
CHAPTER 1

First paragraph.

Second paragraph.

Third paragraph.
";
        let chapters = split_chapters(text);
        assert_eq!(chapters.len(), 1);
        assert!(chapters[0].contains("First paragraph"));
        assert!(chapters[0].contains("Second paragraph"));
        assert!(chapters[0].contains("Third paragraph"));
    }

    #[test]
    fn keyword_must_be_word_start_not_substring() {
        // "Chapterhouse" shouldn't accidentally trigger via the
        // "chapter" keyword. (And the period rejects it anyway.)
        assert!(!is_heading("Chapterhouse is a novel."));
        assert!(is_heading("Chapter 1"));
        assert!(is_heading("CHAPTER ONE"));
    }

    #[test]
    fn period_endings_are_not_headings() {
        assert!(!is_heading("This is a sentence."));
        assert!(!is_heading("Really?"));
        assert!(!is_heading("Bang!"));
    }

    #[test]
    fn em_dash_byline_is_not_a_heading() {
        // Bylines after a section look heading-like (short, Title Case,
        // no period) but should NOT split the chapter.
        assert!(!is_heading("—John Boyne"));
        assert!(!is_heading("–Viktor E. Frankl"));
    }

    #[test]
    fn title_case_with_stopwords_is_heading() {
        // The fix for misclassifying real TOC entries like
        // "Logotherapy in a Nutshell (Abridged)" as body.
        assert!(is_heading("Logotherapy in a Nutshell (Abridged)"));
        assert!(is_heading("Foreword by John Boyne"));
        assert!(is_heading("Preface to the 1992 Edition"));
    }

    #[test]
    fn lone_digits_and_roman_numerals_are_headings() {
        assert!(is_heading("1"));
        assert!(is_heading("42"));
        assert!(is_heading("XIV"));
        assert!(!is_heading("12345")); // too long
        assert!(!is_heading("Hello"));
    }

    #[test]
    fn majority_uppercase_is_heading() {
        assert!(is_heading("iNTRODUCTION"));
        assert!(is_heading("THE END"));
        // A plain sentence in body — period rejects it.
        assert!(!is_heading("Mixed sentence with normal casing."));
        // Title Case multi-word also counts.
        assert!(is_heading("The Beginning"));
    }
}

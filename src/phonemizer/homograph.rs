#![allow(dead_code)]

use super::arpabet;

pub struct WordContext<'a> {
    words: &'a [&'a str],
    index: usize,
}

impl<'a> WordContext<'a> {
    pub fn new(words: &'a [&'a str], index: usize) -> Self {
        Self { words, index }
    }

    fn prev(&self, n: usize) -> Option<&'a str> {
        self.index
            .checked_sub(n)
            .and_then(|idx| self.words.get(idx).copied())
    }

    fn next(&self, n: usize) -> Option<&'a str> {
        self.words.get(self.index + n).copied()
    }

    fn any_before(&self, pred: impl Fn(&str) -> bool) -> bool {
        self.words[..self.index].iter().copied().any(pred)
    }

    fn any_after(&self, pred: impl Fn(&str) -> bool) -> bool {
        self.words[self.index + 1..].iter().copied().any(pred)
    }
}

pub fn phonemize(word: &str, ctx: &WordContext<'_>) -> Option<String> {
    let lower = word.to_ascii_lowercase();
    match lower.as_str() {
        "read" => Some(if read_is_past(ctx) {
            ipa(&["R", "EH1", "D"])
        } else {
            ipa(&["R", "IY1", "D"])
        }),
        "lead" => Some(if is_noun_context(ctx) {
            ipa(&["L", "EH1", "D"])
        } else {
            ipa(&["L", "IY1", "D"])
        }),
        "live" => Some(if is_verb_context(ctx) || is_imperative_context(ctx) {
            ipa(&["L", "IH1", "V"])
        } else {
            ipa(&["L", "AY1", "V"])
        }),
        "wind" => Some(if is_verb_context(ctx) || is_imperative_context(ctx) {
            ipa(&["W", "AY1", "N", "D"])
        } else {
            ipa(&["W", "IH1", "N", "D"])
        }),
        "bow" => Some(if is_verb_context(ctx) || is_imperative_context(ctx) {
            ipa(&["B", "AW1"])
        } else {
            ipa(&["B", "OW1"])
        }),
        "tear" => Some(if is_verb_context(ctx) || is_imperative_context(ctx) {
            ipa(&["T", "EH1", "R"])
        } else {
            ipa(&["T", "IH1", "R"])
        }),
        "wound" => Some(if is_verb_context(ctx) || ctx.any_after(is_past_cue) {
            ipa(&["W", "AW1", "N", "D"])
        } else {
            ipa(&["W", "UW1", "N", "D"])
        }),
        "bass" => Some(if is_music_context(ctx) {
            ipa(&["B", "EY1", "S"])
        } else {
            ipa(&["B", "AE1", "S"])
        }),
        "close" => Some(if is_verb_context(ctx) || is_imperative_context(ctx) {
            ipa(&["K", "L", "OW1", "Z"])
        } else {
            ipa(&["K", "L", "OW1", "S"])
        }),
        "present" => Some(
            if is_verb_context(ctx) || is_imperative_context(ctx) || ctx.any_after(is_verb_cue) {
                ipa(&["P", "R", "IY0", "Z", "EH1", "N", "T"])
            } else {
                ipa(&["P", "R", "EH1", "Z", "AH0", "N", "T"])
            },
        ),
        "record" => Some(
            if is_verb_context(ctx) || is_imperative_context(ctx) || ctx.any_after(is_verb_cue) {
                ipa(&["R", "IH0", "K", "AO1", "R", "D"])
            } else {
                ipa(&["R", "EH1", "K", "ER0", "D"])
            },
        ),
        "object" => Some(
            if is_verb_context(ctx) || is_imperative_context(ctx) || ctx.any_after(is_verb_cue) {
                ipa(&["AH0", "B", "JH", "EH1", "K", "T"])
            } else {
                ipa(&["AA1", "B", "JH", "EH0", "K", "T"])
            },
        ),
        "produce" => Some(if is_noun_context(ctx) || is_adjective_context(ctx) {
            ipa(&["P", "R", "OW1", "D", "UW0", "S"])
        } else if is_verb_context(ctx) || is_imperative_context(ctx) {
            ipa(&["P", "R", "AH0", "D", "UW1", "S"])
        } else {
            ipa(&["P", "R", "AH0", "D", "UW1", "S"])
        }),
        "content" => Some(
            if is_noun_context(ctx) || ctx.next(1).is_some_and(is_copula) {
                ipa(&["K", "AA1", "N", "T", "EH0", "N", "T"])
            } else {
                ipa(&["K", "AH0", "N", "T", "EH1", "N", "T"])
            },
        ),
        "address" => Some(
            if is_noun_context(ctx) || ctx.next(1).is_some_and(is_copula) {
                ipa(&["AE1", "D", "R", "EH2", "S"])
            } else {
                ipa(&["AE0", "D", "R", "EH1", "S"])
            },
        ),
        "desert" => Some(if is_verb_context(ctx) || is_imperative_context(ctx) {
            ipa(&["D", "IH0", "Z", "ER1", "T"])
        } else {
            ipa(&["D", "EH1", "Z", "ER0", "T"])
        }),
        "contract" => Some(
            if is_verb_context(ctx) || is_imperative_context(ctx) || ctx.any_after(is_verb_cue) {
                ipa(&["K", "AH0", "N", "T", "R", "AE1", "K", "T"])
            } else {
                ipa(&["K", "AA1", "N", "T", "R", "AE2", "K", "T"])
            },
        ),
        "contest" => Some(
            if is_verb_context(ctx) || is_imperative_context(ctx) || ctx.any_after(is_verb_cue) {
                ipa(&["K", "AH0", "N", "T", "EH1", "S", "T"])
            } else {
                ipa(&["K", "AA1", "N", "T", "EH0", "S", "T"])
            },
        ),
        "conduct" => Some(
            if is_verb_context(ctx) || is_imperative_context(ctx) || ctx.any_after(is_verb_cue) {
                ipa(&["K", "AH0", "N", "D", "AH1", "K", "T"])
            } else {
                ipa(&["K", "AA1", "N", "D", "AH0", "K", "T"])
            },
        ),
        "conflict" => Some(
            if is_verb_context(ctx) || is_imperative_context(ctx) || ctx.any_after(is_verb_cue) {
                ipa(&["K", "AH0", "N", "F", "L", "EH1", "K", "T"])
            } else {
                ipa(&["K", "AA1", "N", "F", "L", "IH0", "K", "T"])
            },
        ),
        "convert" => Some(
            if is_verb_context(ctx) || is_imperative_context(ctx) || ctx.any_after(is_verb_cue) {
                ipa(&["K", "AH0", "N", "V", "ER1", "T"])
            } else {
                ipa(&["K", "AA1", "N", "V", "ER0", "T"])
            },
        ),
        "digest" => Some(if is_noun_context(ctx) || ctx.any_after(is_noun_cue) {
            ipa(&["D", "AY1", "JH", "EH0", "S", "T"])
        } else {
            ipa(&["D", "AY0", "JH", "EH1", "S", "T"])
        }),
        "insult" => Some(
            if is_verb_context(ctx) || is_imperative_context(ctx) || ctx.any_after(is_verb_cue) {
                ipa(&["IH1", "N", "S", "AH2", "L", "T"])
            } else {
                ipa(&["IH2", "N", "S", "AH1", "L", "T"])
            },
        ),
        "permit" => Some(
            if is_verb_context(ctx) || is_imperative_context(ctx) || ctx.any_after(is_verb_cue) {
                ipa(&["P", "ER1", "M", "IH2", "T"])
            } else {
                ipa(&["P", "ER0", "M", "IH1", "T"])
            },
        ),
        "project" => Some(
            if is_verb_context(ctx) || is_imperative_context(ctx) || ctx.any_after(is_verb_cue) {
                ipa(&["P", "R", "AA0", "JH", "EH1", "K", "T"])
            } else {
                ipa(&["P", "R", "AA1", "JH", "EH0", "K", "T"])
            },
        ),
        "progress" => Some(
            if is_verb_context(ctx) || is_imperative_context(ctx) || ctx.any_after(is_verb_cue) {
                ipa(&["P", "R", "AH0", "G", "R", "EH1", "S"])
            } else {
                ipa(&["P", "R", "AA1", "G", "R", "EH2", "S"])
            },
        ),
        "refuse" => Some(if is_noun_context(ctx) || ctx.any_after(is_noun_cue) {
            ipa(&["R", "EH1", "F", "Y", "UW2", "Z"])
        } else {
            ipa(&["R", "AH0", "F", "Y", "UW1", "Z"])
        }),
        "subject" => Some(
            if is_verb_context(ctx) || is_imperative_context(ctx) || ctx.any_after(is_verb_cue) {
                ipa(&["S", "AH1", "B", "JH", "IH0", "K", "T"])
            } else {
                ipa(&["S", "AH0", "B", "JH", "EH1", "K", "T"])
            },
        ),
        "suspect" => Some(
            if is_verb_context(ctx) || is_imperative_context(ctx) || ctx.any_after(is_verb_cue) {
                ipa(&["S", "AH1", "S", "P", "EH2", "K", "T"])
            } else {
                ipa(&["S", "AH0", "S", "P", "EH1", "K", "T"])
            },
        ),
        "invalid" => Some(
            if is_noun_context(ctx) || ctx.next(1).is_some_and(is_copula) {
                ipa(&["IH1", "N", "V", "AH0", "L", "AH0", "D"])
            } else {
                ipa(&["IH1", "N", "V", "AH0", "L", "IH0", "D"])
            },
        ),
        _ => None,
    }
}

fn ipa(phones: &[&str]) -> String {
    arpabet::phones_to_ipa(phones)
}

fn is_noun_context(ctx: &WordContext<'_>) -> bool {
    ctx.prev(1).is_some_and(is_noun_trigger) || ctx.prev(2).is_some_and(is_noun_trigger)
}

fn is_verb_context(ctx: &WordContext<'_>) -> bool {
    ctx.prev(1).is_some_and(is_verb_trigger) || ctx.prev(2).is_some_and(is_verb_trigger)
}

fn is_adjective_context(ctx: &WordContext<'_>) -> bool {
    ctx.prev(1).is_some_and(is_copula) || ctx.prev(2).is_some_and(is_copula)
}

fn is_imperative_context(ctx: &WordContext<'_>) -> bool {
    ctx.prev(1).is_none()
        && ctx.next(1).is_some_and(|word| {
            let lower = word.to_ascii_lowercase();
            is_determiner(&lower) || lower == "to"
        })
}

fn read_is_past(ctx: &WordContext<'_>) -> bool {
    if ctx.prev(1).is_some_and(is_future_modal) {
        return false;
    }
    // Infinitive marker "to" before "read" is decisive: "to read", "want to
    // read", "going to read" — always present-tense /riːd/, even when later
    // context contains words that look like past cues.
    if ctx.prev(1).is_some_and(|w| w.eq_ignore_ascii_case("to")) {
        return false;
    }
    if ctx.prev(1).is_some_and(is_past_cue) || ctx.prev(2).is_some_and(is_past_cue) {
        return true;
    }
    ctx.any_after(is_past_cue) || ctx.any_after(is_future_cue)
}

fn is_music_context(ctx: &WordContext<'_>) -> bool {
    ctx.prev(1).is_some_and(is_music_cue)
        || ctx.prev(2).is_some_and(is_music_cue)
        || ctx.next(1).is_some_and(is_music_head)
}

fn is_noun_trigger(word: &str) -> bool {
    let lower = word.to_ascii_lowercase();
    is_determiner(&lower) || matches!(lower.as_str(), "of" | "in" | "on" | "at" | "with" | "by")
}

fn is_verb_trigger(word: &str) -> bool {
    let lower = word.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "i" | "you"
            | "he"
            | "she"
            | "it"
            | "we"
            | "they"
            | "who"
            | "what"
            | "that"
            | "will"
            | "shall"
            | "would"
            | "can"
            | "could"
            | "should"
            | "may"
            | "might"
            | "must"
            | "to"
            | "do"
            | "does"
            | "did"
            | "don't"
            | "doesn't"
            | "didn't"
            | "not"
    )
}

fn is_future_modal(word: &str) -> bool {
    matches!(
        word.to_ascii_lowercase().as_str(),
        "will" | "shall" | "would"
    )
}

fn is_future_cue(word: &str) -> bool {
    matches!(
        word.to_ascii_lowercase().as_str(),
        "tomorrow" | "later" | "next" | "soon" | "tonight" | "afternoon" | "evening"
    )
}

fn is_past_cue(word: &str) -> bool {
    matches!(
        word.to_ascii_lowercase().as_str(),
        "yesterday"
            | "ago"
            | "last"
            | "before"
            | "earlier"
            | "previously"
            | "already"
            | "then"
            | "today"
            | "tonight"
            | "this"
            | "morning"
            | "afternoon"
            | "evening"
    )
}

fn is_noun_cue(word: &str) -> bool {
    matches!(
        word.to_ascii_lowercase().as_str(),
        "the" | "a" | "an" | "this" | "that" | "these" | "those" | "my" | "your" | "our" | "their"
    )
}

fn is_verb_cue(word: &str) -> bool {
    matches!(
        word.to_ascii_lowercase().as_str(),
        "will"
            | "shall"
            | "can"
            | "could"
            | "should"
            | "would"
            | "may"
            | "might"
            | "must"
            | "do"
            | "does"
            | "did"
            | "to"
            | "not"
            | "don't"
            | "please"
    )
}

fn is_copula(word: &str) -> bool {
    matches!(
        word.to_ascii_lowercase().as_str(),
        "am" | "is"
            | "are"
            | "was"
            | "were"
            | "be"
            | "been"
            | "being"
            | "seem"
            | "seems"
            | "seemed"
            | "become"
            | "becomes"
            | "became"
            | "look"
            | "looks"
            | "looked"
            | "sound"
            | "sounds"
            | "sounded"
            | "feel"
            | "feels"
            | "felt"
            | "appear"
            | "appears"
            | "appeared"
            | "remain"
            | "remains"
            | "remained"
            | "stay"
            | "stays"
            | "stayed"
    )
}

fn is_determiner(word: &str) -> bool {
    matches!(
        word,
        "the"
            | "a"
            | "an"
            | "this"
            | "that"
            | "these"
            | "those"
            | "my"
            | "your"
            | "our"
            | "their"
            | "his"
            | "her"
            | "its"
            | "some"
            | "any"
            | "each"
            | "every"
    )
}

fn is_music_cue(word: &str) -> bool {
    matches!(
        word.to_ascii_lowercase().as_str(),
        "play"
            | "plays"
            | "played"
            | "playing"
            | "hear"
            | "heard"
            | "listen"
            | "listened"
            | "bass"
            | "guitar"
            | "band"
            | "line"
            | "voice"
            | "part"
    )
}

fn is_music_head(word: &str) -> bool {
    matches!(
        word.to_ascii_lowercase().as_str(),
        "guitar" | "line" | "voice" | "part" | "clef" | "riff"
    )
}

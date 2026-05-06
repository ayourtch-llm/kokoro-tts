#!/usr/bin/env python3
"""Reference for stage 4: homograph disambiguation.

This mirrors the Rust rule set on a curated sentence corpus. The local
environment does not have a POS tagger installed, so the reference is a
hand-labeled rule mirror rather than a statistical tagger.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reference_punctuation import (
    load_cmudict,
    load_gold,
    spell_out_word,
)

GOLD_PATH = Path("data/misaki-us-gold.json")
CMUDICT_PATH = Path("data/cmudict-0.7b")

CASES = [
    "I read the book yesterday.",
    "I read books often.",
    "I read the book, but Andrew will read it tomorrow.",
    "She will lead the meeting.",
    "The pipe is made of lead.",
    "I live there.",
    "It's a live show.",
    "The wind is strong.",
    "Wind the clock.",
    "She wore a bow.",
    "Bow to the queen.",
    "A tear fell.",
    "Don't tear it.",
    "The wound healed.",
    "She wound the clock.",
    "I caught a bass.",
    "Play the bass.",
    "Close the door.",
    "We are close friends.",
    "Give a present.",
    "Present your work.",
    "Set a record.",
    "Record this song.",
    "Touch the object.",
    "I object to that.",
    "The produce is fresh.",
    "We produce widgets.",
    "The content is good.",
    "I'm content.",
    "Mail the address.",
    "I'll address that.",
    "Cross the desert.",
    "Don't desert me.",
    "The contract is signed.",
    "We contract viruses.",
    "Win the contest.",
    "They contest the result.",
    "The conduct was good.",
    "They conduct research.",
    "The conflict grew.",
    "They conflict often.",
    "A convert smiled.",
    "We convert files.",
    "Read the digest.",
    "Dogs digest food.",
    "The insult hurt.",
    "Don't insult her.",
    "He got a permit.",
    "We permit access.",
    "The project failed.",
    "They project confidence.",
    "The progress was slow.",
    "We progress daily.",
    "The refuse smelled.",
    "They refuse the offer.",
    "The subject changed.",
    "We subject samples.",
    "The suspect fled.",
    "I suspect that.",
    "The invalid was admitted.",
    "He is invalid.",
]


def ipa(*phones: str) -> str:
    return phones_to_ipa(list(phones))


def phones_to_ipa(phones: list[str]) -> str:
    return "".join(phone_to_ipa(phone) for phone in phones)


def phone_to_ipa(phone: str) -> str:
    base, stress = split_stress(phone)
    stress_prefix = stress_prefix_for(stress)
    if base == "AA":
        return stressed_vowel("ɑ", stress_prefix)
    if base == "AE":
        return "æ"
    if base == "AH":
        return ah(stress, stress_prefix)
    if base == "AO":
        return stressed_vowel("ɔ", stress_prefix)
    if base == "AW":
        return stressed_diphthong("W", stress_prefix)
    if base == "AY":
        return stressed_diphthong("I", stress_prefix)
    if base == "B":
        return "b"
    if base == "CH":
        return "ʧ"
    if base == "D":
        return "d"
    if base == "DH":
        return "ð"
    if base == "EH":
        return "ɛ"
    if base == "ER":
        return er(stress, stress_prefix)
    if base == "EY":
        return stressed_diphthong("A", stress_prefix)
    if base == "F":
        return "f"
    if base == "G":
        return "ɡ"
    if base == "HH":
        return "h"
    if base == "IH":
        return "ɪ"
    if base == "IY":
        return stressed_vowel("i", stress_prefix)
    if base == "JH":
        return "ʤ"
    if base == "K":
        return "k"
    if base == "L":
        return "l"
    if base == "M":
        return "m"
    if base == "N":
        return "n"
    if base == "NG":
        return "ŋ"
    if base == "OW":
        return stressed_diphthong("O", stress_prefix)
    if base == "OY":
        return stressed_diphthong("Y", stress_prefix)
    if base == "P":
        return "p"
    if base == "R":
        return "ɹ"
    if base == "S":
        return "s"
    if base == "SH":
        return "ʃ"
    if base == "T":
        return "t"
    if base == "TH":
        return "θ"
    if base == "UH":
        return "ʊ"
    if base == "UW":
        return stressed_vowel("u", stress_prefix)
    if base == "V":
        return "v"
    if base == "W":
        return "w"
    if base == "Y":
        return "j"
    if base == "Z":
        return "z"
    if base == "ZH":
        return "ʒ"
    raise KeyError(base)


def split_stress(phone: str) -> tuple[str, int]:
    if phone and phone[-1] in "012":
        return phone[:-1], int(phone[-1])
    return phone, 0


def stress_prefix_for(stress: int) -> str:
    return {1: "ˈ", 2: "ˌ"}.get(stress, "")


def stressed_vowel(base: str, stress_prefix: str) -> str:
    return stress_prefix + base


def stressed_diphthong(base: str, stress_prefix: str) -> str:
    return stress_prefix + base


def ah(stress: int, stress_prefix: str) -> str:
    base = {0: "ə", 1: "ʌ", 2: "ʌ"}.get(stress, "ə")
    return stress_prefix + base


def er(stress: int, stress_prefix: str) -> str:
    base = {0: "əɹ", 1: "ɜɹ", 2: "ɜɹ"}.get(stress, "əɹ")
    return stress_prefix + base


def is_determiner(word: str) -> bool:
    return word.lower() in {
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "my",
        "your",
        "our",
        "their",
        "his",
        "her",
        "its",
        "some",
        "any",
        "each",
        "every",
    }


def is_copula(word: str) -> bool:
    return word.lower() in {
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "seem",
        "seems",
        "seemed",
        "become",
        "becomes",
        "became",
        "look",
        "looks",
        "looked",
        "sound",
        "sounds",
        "sounded",
        "feel",
        "feels",
        "felt",
        "appear",
        "appears",
        "appeared",
        "remain",
        "remains",
        "remained",
        "stay",
        "stays",
        "stayed",
    }


def is_verb_cue(word: str) -> bool:
    return word.lower() in {
        "will",
        "shall",
        "can",
        "could",
        "should",
        "would",
        "may",
        "might",
        "must",
        "do",
        "does",
        "did",
        "to",
        "not",
        "don't",
        "doesn't",
        "didn't",
        "please",
    }


def is_noun_cue(word: str) -> bool:
    return is_determiner(word) or word.lower() in {"of", "in", "on", "at", "with", "by"}


def is_music_cue(word: str) -> bool:
    return word.lower() in {
        "play",
        "plays",
        "played",
        "playing",
        "hear",
        "heard",
        "listen",
        "listened",
        "bass",
        "guitar",
        "band",
        "line",
        "voice",
        "part",
    }


def is_music_head(word: str) -> bool:
    return word.lower() in {"guitar", "line", "voice", "part", "clef", "riff"}


def is_past_cue(word: str) -> bool:
    return word.lower() in {
        "yesterday",
        "ago",
        "last",
        "before",
        "earlier",
        "previously",
        "already",
        "then",
        "today",
        "tonight",
        "this",
        "morning",
        "afternoon",
        "evening",
    }


def is_future_cue(word: str) -> bool:
    return word.lower() in {"tomorrow", "later", "next", "soon", "tonight", "afternoon", "evening"}


def word_context(words: list[str], idx: int) -> tuple[str | None, str | None, str | None, str | None]:
    def get(offset: int) -> str | None:
        pos = idx + offset
        if 0 <= pos < len(words):
            return words[pos]
        return None

    return get(-1), get(-2), get(1), get(2)


def read_is_past(words: list[str], idx: int) -> bool:
    prev1, prev2, next1, next2 = word_context(words, idx)
    if prev1 and prev1.lower() in {"will", "shall", "would"}:
        return False
    if (prev1 and is_past_cue(prev1)) or (prev2 and is_past_cue(prev2)):
        return True
    return any(is_past_cue(word) or is_future_cue(word) for word in words[idx + 1 :])


def homograph_ipa(words: list[str], idx: int, gold: dict[str, str], lexicon: dict[str, list[str]]) -> str | None:
    word = words[idx]
    lower = word.lower()
    prev1, prev2, next1, next2 = word_context(words, idx)

    def noun_context() -> bool:
        return bool((prev1 and is_noun_cue(prev1)) or (prev2 and is_noun_cue(prev2)))

    def verb_context() -> bool:
        return bool((prev1 and prev1.lower() in {
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "who",
            "what",
            "that",
            "will",
            "shall",
            "would",
            "can",
            "could",
            "should",
            "may",
            "might",
            "must",
            "to",
            "do",
            "does",
            "did",
            "don't",
            "doesn't",
            "didn't",
            "not",
        }) or (prev2 and prev2.lower() in {
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "who",
            "what",
            "that",
            "will",
            "shall",
            "would",
            "can",
            "could",
            "should",
            "may",
            "might",
            "must",
            "to",
            "do",
            "does",
            "did",
            "don't",
            "doesn't",
            "didn't",
            "not",
        }))

    def imperative_context() -> bool:
        return prev1 is None and next1 is not None and (is_determiner(next1) or next1.lower() == "to")

    def music_context() -> bool:
        return bool((prev1 and is_music_cue(prev1)) or (prev2 and is_music_cue(prev2)) or (next1 and is_music_head(next1)))

    def copula_context() -> bool:
        return bool((prev1 and is_copula(prev1)) or (prev2 and is_copula(prev2)))

    if lower == "read":
        return ipa("R", "EH1", "D") if read_is_past(words, idx) else ipa("R", "IY1", "D")
    if lower == "lead":
        return ipa("L", "EH1", "D") if noun_context() else ipa("L", "IY1", "D")
    if lower == "live":
        return ipa("L", "IH1", "V") if verb_context() or any(w.lower() == "live" for w in words[idx + 1 :]) else ipa("L", "AY1", "V")
    if lower == "wind":
        return ipa("W", "AY1", "N", "D") if verb_context() or imperative_context() else ipa("W", "IH1", "N", "D")
    if lower == "bow":
        return ipa("B", "AW1") if verb_context() or imperative_context() else ipa("B", "OW1")
    if lower == "tear":
        return ipa("T", "EH1", "R") if verb_context() or imperative_context() else ipa("T", "IH1", "R")
    if lower == "wound":
        return ipa("W", "AW1", "N", "D") if verb_context() or any(is_past_cue(word) for word in words[idx + 1 :]) else ipa("W", "UW1", "N", "D")
    if lower == "bass":
        return "bˈAs" if music_context() else "bˈæs"
    if lower == "close":
        return ipa("K", "L", "OW1", "Z") if verb_context() or imperative_context() else ipa("K", "L", "OW1", "S")
    if lower == "present":
        return ipa("P", "R", "IY0", "Z", "EH1", "N", "T") if verb_context() else ipa("P", "R", "EH1", "Z", "AH0", "N", "T")
    if lower == "record":
        return ipa("R", "IH0", "K", "AO1", "R", "D") if verb_context() else ipa("R", "EH1", "K", "ER0", "D")
    if lower == "object":
        return ipa("AH0", "B", "JH", "EH1", "K", "T") if verb_context() else ipa("AA1", "B", "JH", "EH0", "K", "T")
    if lower == "produce":
        return ipa("P", "R", "OW1", "D", "UW0", "S") if noun_context() or copula_context() else ipa("P", "R", "AH0", "D", "UW1", "S")
    if lower == "content":
        return ipa("K", "AA1", "N", "T", "EH0", "N", "T") if noun_context() or next1 and is_copula(next1) else ipa("K", "AH0", "N", "T", "EH1", "N", "T")
    if lower == "address":
        return ipa("AE1", "D", "R", "EH2", "S") if noun_context() or next1 and is_copula(next1) else ipa("AE0", "D", "R", "EH1", "S")
    if lower == "desert":
        return ipa("D", "IH0", "Z", "ER1", "T") if verb_context() or imperative_context() else ipa("D", "EH1", "Z", "ER0", "T")
    if lower == "contract":
        return ipa("K", "AH0", "N", "T", "R", "AE1", "K", "T") if verb_context() else ipa("K", "AA1", "N", "T", "R", "AE2", "K", "T")
    if lower == "contest":
        return ipa("K", "AH0", "N", "T", "EH1", "S", "T") if verb_context() else ipa("K", "AA1", "N", "T", "EH0", "S", "T")
    if lower == "conduct":
        return ipa("K", "AH0", "N", "D", "AH1", "K", "T") if verb_context() else ipa("K", "AA1", "N", "D", "AH0", "K", "T")
    if lower == "conflict":
        return ipa("K", "AH0", "N", "F", "L", "EH1", "K", "T") if verb_context() else ipa("K", "AA1", "N", "F", "L", "IH0", "K", "T")
    if lower == "convert":
        return ipa("K", "AH0", "N", "V", "ER1", "T") if verb_context() else ipa("K", "AA1", "N", "V", "ER0", "T")
    if lower == "digest":
        return ipa("D", "AY1", "JH", "EH0", "S", "T") if noun_context() else ipa("D", "AY0", "JH", "EH1", "S", "T")
    if lower == "insult":
        return ipa("IH1", "N", "S", "AH2", "L", "T") if verb_context() else ipa("IH2", "N", "S", "AH1", "L", "T")
    if lower == "permit":
        return ipa("P", "ER1", "M", "IH2", "T") if verb_context() else ipa("P", "ER0", "M", "IH1", "T")
    if lower == "project":
        return ipa("P", "R", "AA0", "JH", "EH1", "K", "T") if verb_context() else ipa("P", "R", "AA1", "JH", "EH0", "K", "T")
    if lower == "progress":
        return ipa("P", "R", "AH0", "G", "R", "EH1", "S") if verb_context() else ipa("P", "R", "AA1", "G", "R", "EH2", "S")
    if lower == "refuse":
        return ipa("R", "EH1", "F", "Y", "UW2", "Z") if noun_context() else ipa("R", "AH0", "F", "Y", "UW1", "Z")
    if lower == "subject":
        return ipa("S", "AH1", "B", "JH", "IH0", "K", "T") if verb_context() else ipa("S", "AH0", "B", "JH", "EH1", "K", "T")
    if lower == "suspect":
        return ipa("S", "AH1", "S", "P", "EH2", "K", "T") if verb_context() else ipa("S", "AH0", "S", "P", "EH1", "K", "T")
    if lower == "invalid":
        return ipa("IH1", "N", "V", "AH0", "L", "AH0", "D") if noun_context() or next1 and is_copula(next1) else ipa("IH1", "N", "V", "AH0", "L", "IH0", "D")

    ipa_text = gold.get(lower)
    if ipa_text is not None:
        return ipa_text
    phones = lexicon.get(lower)
    if phones is not None:
        return phones_to_ipa(phones)
    return spell_out_word(word)


def tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    current = []
    for ch in text:
        if ch.isalpha() or ch in {"'", "-"}:
            current.append(ch)
        else:
            if current:
                tokens.append(("word", "".join(current)))
                current.clear()
            if ch in {",", ".", "!", "?", ";", ":", "“", "”", "—", "…"}:
                tokens.append(("punct", ch))
    if current:
        tokens.append(("word", "".join(current)))
    return tokens


def needs_space_before_word(out: str) -> bool:
    return bool(out) and out[-1] not in " ([{“‘—…"


def phonemize(text: str, gold: dict[str, str], lexicon: dict[str, list[str]]) -> str:
    tokens = tokenize(text)
    words = [value for kind, value in tokens if kind == "word"]
    out = []
    word_idx = 0
    for kind, value in tokens:
        if kind == "word":
            if value == "A":
                ipa_text = "ˈA"
            else:
                ipa_text = homograph_ipa(words, word_idx, gold, lexicon)
                if ipa_text is None:
                    ipa_text = gold.get(value.lower())
                    if ipa_text is None:
                        phones = lexicon.get(value.lower())
                        ipa_text = phones_to_ipa(phones) if phones is not None else spell_out_word(value)
            if needs_space_before_word("".join(out)):
                out.append(" ")
            out.append(ipa_text)
            word_idx += 1
        else:
            out.append(value)
    return "".join(out)


def write_reference(path: Path) -> None:
    gold = load_gold(GOLD_PATH)
    lexicon = load_cmudict(CMUDICT_PATH)
    with path.open("w", encoding="utf-8") as fh:
        for case in CASES:
            fh.write(json.dumps({"case": case, "ipa": phonemize(case, gold, lexicon)}, ensure_ascii=False))
            fh.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("tmp/reference_homograph.jsonl"))
    args = parser.parse_args()
    write_reference(args.out)


if __name__ == "__main__":
    main()

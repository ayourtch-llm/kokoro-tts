#!/usr/bin/env python3
"""Reference for stage 2: sentence-boundary-aware phonemization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reference_normalize import normalize_abbreviations

GOLD_PATH = Path("data/misaki-us-gold.json")
CMUDICT_PATH = Path("data/cmudict-0.7b")

CASES = [
    "Hello. How are you?",
    "Mr. Smith arrived. He waited.",
    "Dr. Adams said, \"Go.\"",
    "She paused... Then spoke.",
    "First line.\n\nSecond line.",
    "The house is there; the water is there: fine.",
    "It's fine, isn't it?",
    "e.g. examples are useful. i.e. they clarify.",
    "St. Louis is here.",
    "A quote: “Hello!”",
    "One? Two! Three.",
    "vs. is an abbreviation.",
]

ABBREVIATIONS = (
    "mrs.",
    "mr.",
    "ms.",
    "dr.",
    "prof.",
    "st.",
    "jr.",
    "sr.",
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "cf.",
    "a.m.",
    "p.m.",
)


def load_gold(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text())
    gold: dict[str, str] = {}
    for key, value in raw.items():
        ipa = flatten_value(value)
        if ipa is None:
            continue
        gold[key] = ipa
        gold.setdefault(key.lower(), ipa)
    return gold


def flatten_value(value) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("DEFAULT"), str):
            return value["DEFAULT"]
        for item in value.values():
            if isinstance(item, str):
                return item
    return None


def load_cmudict(path: Path) -> dict[str, list[str]]:
    lexicon: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(";;;"):
            continue
        parts = line.split()
        if not parts:
            continue
        word = parts[0]
        if "(" in word:
            continue
        lexicon.setdefault(word.lower(), parts[1:])
    return lexicon


def split_stress(phone: str) -> tuple[str, int]:
    if phone and phone[-1] in "012":
        return phone[:-1], int(phone[-1])
    return phone, 0


def stress_prefix(stress: int) -> str:
    return {1: "ˈ", 2: "ˌ"}.get(stress, "")


def long_vowel(base: str, stress: int) -> str:
    return stress_prefix(stress) + base


def simple(base: str):
    return lambda stress: stress_prefix(stress) + base


def ah(stress: int) -> str:
    return stress_prefix(stress) + {0: "ə", 1: "ʌ", 2: "ʌ"}.get(stress, "ə")


def er(stress: int) -> str:
    return stress_prefix(stress) + {0: "əɹ", 1: "ɜɹ", 2: "ɜɹ"}.get(stress, "əɹ")


PHONE_MAP = {
    "AA": lambda stress: long_vowel("ɑ", stress),
    "AE": simple("æ"),
    "AH": ah,
    "AO": lambda stress: long_vowel("ɔ", stress),
    "AW": simple("W"),
    "AY": simple("I"),
    "B": simple("b"),
    "CH": simple("ʧ"),
    "D": simple("d"),
    "DH": simple("ð"),
    "EH": simple("ɛ"),
    "ER": er,
    "EY": simple("A"),
    "F": simple("f"),
    "G": simple("ɡ"),
    "HH": simple("h"),
    "IH": simple("ɪ"),
    "IY": lambda stress: long_vowel("i", stress),
    "JH": simple("ʤ"),
    "K": simple("k"),
    "L": simple("l"),
    "M": simple("m"),
    "N": simple("n"),
    "NG": simple("ŋ"),
    "OW": simple("O"),
    "OY": simple("Y"),
    "P": simple("p"),
    "R": simple("ɹ"),
    "S": simple("s"),
    "SH": simple("ʃ"),
    "T": simple("t"),
    "TH": simple("θ"),
    "UH": simple("ʊ"),
    "UW": lambda stress: long_vowel("u", stress),
    "V": simple("v"),
    "W": simple("w"),
    "Y": simple("j"),
    "Z": simple("z"),
    "ZH": simple("ʒ"),
}


def phones_to_ipa(phones: list[str]) -> str:
    parts = []
    for phone in phones:
        base, stress = split_stress(phone)
        parts.append(PHONE_MAP[base](stress))
    return "".join(parts)


def spell_out_word(word: str) -> str:
    out = []
    for ch in word:
        letter = {
            "a": "eɪ",
            "b": "bi",
            "c": "si",
            "d": "di",
            "e": "i",
            "f": "ɛf",
            "g": "dʒi",
            "h": "eɪʧ",
            "i": "aɪ",
            "j": "dʒeɪ",
            "k": "keɪ",
            "l": "ɛl",
            "m": "ɛm",
            "n": "ɛn",
            "o": "oʊ",
            "p": "pi",
            "q": "kju",
            "r": "ɑɹ",
            "s": "ɛs",
            "t": "ti",
            "u": "ju",
            "v": "vi",
            "w": "dʌbəlju",
            "x": "ɛks",
            "y": "waɪ",
            "z": "zi",
        }.get(ch.lower())
        if letter is None:
            continue
        out.append(letter)
    return " ".join(out) if out else word


def match_abbreviation(text: str, start: int) -> int | None:
    tail = text[start:]
    for abbrev in ABBREVIATIONS:
        if tail[: len(abbrev)].lower() == abbrev:
            return len(abbrev)
    return None


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    current: list[str] = []
    i = 0
    while i < len(text):
        abbrev_len = match_abbreviation(text, i)
        if abbrev_len is not None:
            current.append(text[i : i + abbrev_len])
            i += abbrev_len
            continue

        if text.startswith("...", i):
            current.append("...")
            i += 3
            continue

        ch = text[i]
        if ch == "\n":
            if i + 1 < len(text) and text[i + 1] == "\n":
                sentence = "".join(current).strip()
                if sentence:
                    out.append(sentence)
                current.clear()
                while i < len(text) and text[i] == "\n":
                    i += 1
                continue
            current.append(" ")
            i += 1
            continue

        current.append(ch)
        i += 1
        if ch in ".!?" and _ends_sentence(text, i - 1, ch):
            sentence = "".join(current).strip()
            if sentence:
                out.append(sentence)
            current.clear()

    sentence = "".join(current).strip()
    if sentence:
        out.append(sentence)
    return out


def _ends_sentence(text: str, dot_index: int, ch: str) -> bool:
    if ch != ".":
        return True
    if dot_index > 0 and dot_index + 1 < len(text):
        prev = text[dot_index - 1]
        next_ = text[dot_index + 1]
        if prev.isdigit() and next_.isdigit():
            return False
    return True


def tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    current: list[str] = []
    for ch in text:
        if ch.isascii() and (ch.isalpha() or ch in {"'", "-"}):
            current.append(ch)
        else:
            if current:
                tokens.append(("word", "".join(current)))
                current.clear()
            if ch in ",.!?;:“”—…":
                tokens.append(("punct", ch))
    if current:
        tokens.append(("word", "".join(current)))
    return tokens


def phonemize_chunk(text: str, gold: dict[str, str], cmudict: dict[str, list[str]]) -> str:
    out: list[str] = []
    for kind, value in tokenize(text):
        if kind == "word":
            if out and out[-1] not in {" ", "(", "[", "{", "“", "‘", "—", "…"}:
                out.append(" ")
            ipa = gold.get(value) or gold.get(value.lower())
            if ipa is None and value.lower() in cmudict:
                ipa = phones_to_ipa(cmudict[value.lower()])
            if ipa is None:
                ipa = spell_out_word(value)
            out.append(ipa)
        else:
            out.append(value)
    return "".join(out)


def phonemize(text: str, gold: dict[str, str], cmudict: dict[str, list[str]]) -> str:
    text = normalize_abbreviations(text)
    parts = [phonemize_chunk(sentence, gold, cmudict) for sentence in split_sentences(text)]
    parts = [part for part in parts if part]
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tmp/reference_punctuation.jsonl")
    args = parser.parse_args()

    gold = load_gold(GOLD_PATH)
    cmudict = load_cmudict(CMUDICT_PATH)
    lines = []
    for case in CASES:
        ipa = phonemize(case, gold, cmudict)
        print(f"{case}: {ipa}")
        lines.append(json.dumps({"case": case, "ipa": ipa}, ensure_ascii=False))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

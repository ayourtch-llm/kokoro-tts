#!/usr/bin/env python3
"""Reference for the stage-1 CMUdict phonemizer."""

from __future__ import annotations

import argparse
from pathlib import Path


CASES = [
    "hello",
    "world",
    "speed",
    "father",
    "thought",
    "ready",
    "treaty",
    "about",
    "again",
    "before",
    "because",
    "change",
    "church",
    "children",
    "common",
    "different",
    "enough",
    "family",
    "friend",
    "house",
    "important",
    "know",
    "little",
    "mother",
    "number",
    "people",
    "question",
    "really",
    "school",
    "should",
    "through",
    "their",
    "there",
    "time",
    "under",
    "water",
    "what",
    "where",
    "woman",
    "year",
    "world",
    "hello world",
]

CMUDICT_PATH = Path("data/cmudict-0.7b")


def load_cmudict(path: Path) -> dict[str, list[list[str]]]:
    lexicon: dict[str, list[list[str]]] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith(";;;"):
            continue
        parts = line.split()
        if not parts:
            continue
        word = parts[0]
        if "(" in word:
            continue
        lexicon.setdefault(word.lower(), []).append(parts[1:])
    return lexicon


def phones_to_ipa(phones: list[str]) -> str:
    out = []
    for phone in phones:
        base, stress = split_stress(phone)
        ipa = PHONE_MAP[base](stress)
        out.append(ipa)
    return "".join(out)


def split_stress(phone: str) -> tuple[str, int]:
    if phone and phone[-1] in "012":
        return phone[:-1], int(phone[-1])
    return phone, 0


def long_vowel(base: str):
    return lambda stress: stress_prefix(stress) + base + ("ː" if stress in (1, 2) else "")


def simple(base: str):
    return lambda stress: stress_prefix(stress) + base


def stressed(base: str):
    return lambda stress: stress_prefix(stress) + base


def diphthong(base: str):
    return stressed(base)


def ah(stress: int) -> str:
    return stress_prefix(stress) + {0: "ə", 1: "ʌ", 2: "ʌ"}.get(stress, "ə")


def er(stress: int) -> str:
    return stress_prefix(stress) + {0: "ɚ", 1: "ɜː", 2: "ɜː"}.get(stress, "ɚ")


def stress_prefix(stress: int) -> str:
    return {1: "ˈ", 2: "ˌ"}.get(stress, "")


PHONE_MAP = {
    "AA": long_vowel("ɑ"),
    "AE": simple("æ"),
    "AH": ah,
    "AO": long_vowel("ɔ"),
    "AW": diphthong("aʊ"),
    "AY": diphthong("aɪ"),
    "B": simple("b"),
    "CH": diphthong("ʧ"),
    "D": simple("d"),
    "DH": simple("ð"),
    "EH": simple("ɛ"),
    "ER": er,
    "EY": diphthong("eɪ"),
    "F": simple("f"),
    "G": simple("ɡ"),
    "HH": simple("h"),
    "IH": simple("ɪ"),
    "IY": long_vowel("i"),
    "JH": diphthong("ʤ"),
    "K": simple("k"),
    "L": simple("l"),
    "M": simple("m"),
    "N": simple("n"),
    "NG": simple("ŋ"),
    "OW": diphthong("oʊ"),
    "OY": diphthong("ɔɪ"),
    "P": simple("p"),
    "R": simple("ɹ"),
    "S": simple("s"),
    "SH": simple("ʃ"),
    "T": simple("t"),
    "TH": simple("θ"),
    "UH": simple("ʊ"),
    "UW": long_vowel("u"),
    "V": simple("v"),
    "W": simple("w"),
    "Y": simple("j"),
    "Z": simple("z"),
    "ZH": simple("ʒ"),
}


def phonemize_word(lexicon: dict[str, list[list[str]]], word: str) -> str:
    phones = lexicon.get(word.lower())
    if not phones:
        raise KeyError(word)
    return phones_to_ipa(phones[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tmp/reference_lexicon.tsv")
    args = parser.parse_args()

    lexicon = load_cmudict(CMUDICT_PATH)
    lines = []
    for case in CASES:
        if " " in case:
            ipa = " ".join(phonemize_word(lexicon, word) for word in case.split())
        else:
            ipa = phonemize_word(lexicon, case)
        lines.append(f"{case}\t{ipa}")
        print(f"{case}: {ipa}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

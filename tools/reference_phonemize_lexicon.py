#!/usr/bin/env python3
"""Reference for the two-tier stage-1 phonemizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GOLD_PATH = Path("data/misaki-us-gold.json")
CMUDICT_PATH = Path("data/cmudict-0.7b")

GOLD_CASES = [
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
    "hello world",
]

FALLBACK_CASES = [
    "aaberg",
    "aaker",
    "ababa",
    "abaco",
    "abadi",
    "abided",
    "abated",
    "abounded",
    "abboud",
    "adjoins",
]


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


def long_vowel(base: str):
    return lambda stress: stress_prefix(stress) + base + ("ː" if stress in (1, 2) else "")


def simple(base: str):
    return lambda stress: stress_prefix(stress) + base


def ah(stress: int) -> str:
    return stress_prefix(stress) + {0: "ə", 1: "ʌ", 2: "ʌ"}.get(stress, "ə")


def er(stress: int) -> str:
    return stress_prefix(stress) + {0: "ɚ", 1: "ɜː", 2: "ɜː"}.get(stress, "ɚ")


PHONE_MAP = {
    "AA": long_vowel("ɑ"),
    "AE": simple("æ"),
    "AH": ah,
    "AO": long_vowel("ɔ"),
    "AW": simple("aʊ"),
    "AY": simple("aɪ"),
    "B": simple("b"),
    "CH": simple("ʧ"),
    "D": simple("d"),
    "DH": simple("ð"),
    "EH": simple("ɛ"),
    "ER": er,
    "EY": simple("eɪ"),
    "F": simple("f"),
    "G": simple("ɡ"),
    "HH": simple("h"),
    "IH": simple("ɪ"),
    "IY": long_vowel("i"),
    "JH": simple("ʤ"),
    "K": simple("k"),
    "L": simple("l"),
    "M": simple("m"),
    "N": simple("n"),
    "NG": simple("ŋ"),
    "OW": simple("oʊ"),
    "OY": simple("ɔɪ"),
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


def phones_to_ipa(phones: list[str]) -> str:
    return "".join(PHONE_MAP[base](stress) for base, stress in (split_stress(p) for p in phones))


def select_fallback_cases(cmudict: dict[str, list[str]], gold: dict[str, str], limit: int = 12) -> list[str]:
    cases = []
    for word in cmudict:
        if word in gold:
            continue
        if not word.isalpha():
            continue
        if len(word) < 4:
            continue
        phones = cmudict[word]
        if not can_convert_phones(phones):
            continue
        cases.append(word)
        if len(cases) == limit:
            break
    return cases


def dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def can_convert_phones(phones: list[str]) -> bool:
    return all((phone[:-1] if phone and phone[-1] in "012" else phone) in PHONE_MAP for phone in phones)


def resolve_case(gold: dict[str, str], cmudict: dict[str, list[str]], case: str) -> str | None:
    if " " in case:
        parts = [resolve_case(gold, cmudict, part) for part in case.split()]
        if any(part is None for part in parts):
            return None
        return " ".join(part for part in parts if part is not None)
    if case in gold:
        return gold[case]
    if case.lower() in gold:
        return gold[case.lower()]
    phones = cmudict[case.lower()][0]
    try:
        return phones_to_ipa(phones)
    except KeyError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tmp/reference_lexicon.tsv")
    args = parser.parse_args()

    gold = load_gold(GOLD_PATH)
    cmudict = load_cmudict(CMUDICT_PATH)
    cases = dedupe(GOLD_CASES + FALLBACK_CASES + select_fallback_cases(cmudict, gold))
    lines = []
    for case in cases:
        ipa = resolve_case(gold, cmudict, case)
        if ipa is None:
            continue
        print(f"{case}: {ipa}")
        lines.append(f"{case}\t{ipa}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

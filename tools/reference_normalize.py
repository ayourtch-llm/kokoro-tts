#!/usr/bin/env python3
"""Reference for stage 3.1: cardinal-number normalization."""

from __future__ import annotations

import argparse
from pathlib import Path

CASES = [
    "0",
    "5",
    "10",
    "11",
    "20",
    "21",
    "82",
    "100",
    "101",
    "110",
    "569",
    "1,234",
    "1,000,000",
    "-42",
    "3.14",
    "0.5",
    "12.05",
    "3:45",
    "1st",
    "I have 3 dogs.",
    "The total is 42 dollars.",
    "Room 007 is on the left.",
]

UNITS = [
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
]

TENS = [
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]


def normalize(text: str) -> str:
    chars = list(text)
    out: list[str] = []
    i = 0
    while i < len(chars):
        result = parse_number(chars, i)
        if result is None:
            out.append(chars[i])
            i += 1
        else:
            replacement, consumed = result
            out.append(replacement)
            i += consumed
    return "".join(out)


def parse_number(chars: list[str], start: int) -> tuple[str, int] | None:
    i = start
    negative = False
    if chars[i] in "+-":
        if i + 1 >= len(chars) or not chars[i + 1].isdigit():
            return None
        if start > 0 and (chars[start - 1].isalnum() or chars[start - 1] in ":/"):
            return None
        negative = chars[i] == "-"
        i += 1

    if i >= len(chars) or not chars[i].isdigit():
        return None

    int_part = []
    frac_part = []
    decimal = False
    saw_digit = False
    while i < len(chars):
        ch = chars[i]
        if ch.isdigit():
            saw_digit = True
            (frac_part if decimal else int_part).append(ch)
            i += 1
            continue
        if ch == "," and not decimal:
            i += 1
            continue
        if ch == "." and not decimal and i + 1 < len(chars) and chars[i + 1].isdigit():
            decimal = True
            i += 1
            continue
        break

    if not saw_digit:
        return None
    if start > 0 and (chars[start - 1].isalnum() or chars[start - 1] in ":/"):
        return None
    if i < len(chars) and (chars[i].isalpha() or chars[i] in ":/%$°"):
        return None

    words = []
    if negative:
        words.append("minus")
    words.append(integer_to_words("".join(int_part)))
    if decimal:
        if not frac_part:
            return None
        words.append("point")
        words.extend(digit_to_word(ch) for ch in frac_part)
    return " ".join(words), i - start


def integer_to_words(raw: str) -> str:
    raw = raw.lstrip("0") or "0"
    n = int(raw)
    if n == 0:
        return "zero"
    scales = ["", "thousand", "million", "billion"]
    chunks: list[str] = []
    scale = 0
    while n:
        group = n % 1000
        if group:
            part = convert_hundreds(group)
            if scales[scale]:
                part = f"{part} {scales[scale]}"
            chunks.append(part)
        n //= 1000
        scale += 1
    return " ".join(reversed(chunks))


def convert_hundreds(n: int) -> str:
    hundreds, rem = divmod(n, 100)
    parts: list[str] = []
    if hundreds:
        parts.append(f"{UNITS[hundreds]} hundred")
    if rem:
        if rem < 20:
            parts.append(UNITS[rem])
        else:
            tens, ones = divmod(rem, 10)
            if ones:
                parts.append(f"{TENS[tens]} {UNITS[ones]}")
            else:
                parts.append(TENS[tens])
    return " ".join(parts)


def digit_to_word(ch: str) -> str:
    return UNITS[int(ch)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tmp/reference_normalize.tsv")
    args = parser.parse_args()

    lines = []
    for case in CASES:
        normalized = normalize(case)
        print(f"{case}: {normalized}")
        lines.append(f"{case}\t{normalized}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

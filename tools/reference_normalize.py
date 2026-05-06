#!/usr/bin/env python3
"""Reference for stage 3.5: cardinal + ordinal + year + abbreviation + acronym + money/time normalization."""

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
    "1234",
    "1,234",
    "1,000,000",
    "-42",
    "3.14",
    "0.5",
    "12.05",
    "3:45",
    "1st",
    "2nd",
    "3rd",
    "4th",
    "21st",
    "22nd",
    "23rd",
    "24th",
    "100th",
    "101st",
    "1900",
    "1999",
    "2000",
    "2008",
    "2010",
    "2026",
    "I have 3 dogs.",
    "The total is 42 dollars.",
    "Room 007 is on the left.",
    "On July 4th, 1776, the colonies declared independence.",
    "The year 1900 was important.",
    "In 2008, things changed.",
    "Mr. Smith called Mrs. Jones.",
    "Ms. Lee met Dr. Brown.",
    "Prof. Adams lectured.",
    "St. Louis is in Missouri.",
    "Jr. and Sr. are suffixes.",
    "e.g. examples help.",
    "i.e. clarification follows.",
    "etc. is a shorthand.",
    "vs. is used in comparisons.",
    "cf. compare the notes.",
    "3 a.m.",
    "3 p.m.",
    "Dr. Smith called Mr. Jones at 3 p.m. on Monday.",
    "The meeting starts at 7 a.m. on Tue.",
    "A.M. and P.M. should be spoken as letters.",
    "NASA launched the FBI's new HTML parser by 3 p.m.",
    "NATO and RADAR are pronounceable acronyms.",
    "ASCII JSON ASAP FAQ PIN are common pronounce-as-word cases.",
    "FBI CIA USA SQL HTML CSS USB should spell out.",
    "NASA's mission succeeded.",
    "The SQL query used JSON and FAQ docs.",
    "$5",
    "$1",
    "$5.50",
    "$1,234.56",
    "€5",
    "£1",
    "¥5",
    "5¢",
    "1¢",
    "3:45",
    "3:00",
    "12:00",
    "The meeting starts at 3:45 PM and costs $25 per person.",
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
    text = normalize_abbreviations(text)
    text = normalize_money_time(text)
    text = normalize_acronyms(text)
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
    saw_comma = False
    saw_digit = False
    while i < len(chars):
        ch = chars[i]
        if ch.isdigit():
            saw_digit = True
            (frac_part if decimal else int_part).append(ch)
            i += 1
            continue
        if ch == "," and not decimal:
            if i + 1 < len(chars) and chars[i + 1].isdigit():
                saw_comma = True
                i += 1
                continue
            break
        if ch == "." and not decimal and i + 1 < len(chars) and chars[i + 1].isdigit():
            decimal = True
            i += 1
            continue
        break

    if not saw_digit:
        return None
    suffix = ordinal_suffix(chars, i)
    if suffix is not None and not decimal and is_number_boundary(chars, start):
        if token_ends_cleanly(chars, i + len(suffix)):
            return ordinal_phrase("".join(int_part)), i + len(suffix) - start

    if not is_number_boundary(chars, start):
        return None

    if decimal:
        if not frac_part:
            return None
        if not token_ends_cleanly(chars, i):
            return None
        return decimal_phrase("".join(int_part), negative, "".join(frac_part)), i - start

    trimmed = "".join(int_part).lstrip("0") or "0"
    if not saw_comma and len(trimmed) == 4:
        year = int(trimmed)
        if 1000 <= year <= 2099 and token_ends_cleanly(chars, i):
            return year_phrase(year), i - start

    if not token_ends_cleanly(chars, i):
        return None
    return cardinal_phrase("".join(int_part), negative), i - start


def is_number_boundary(chars: list[str], start: int) -> bool:
    return start == 0 or not (chars[start - 1].isalnum() or chars[start - 1] in ":/")


def token_ends_cleanly(chars: list[str], end: int) -> bool:
    if end >= len(chars):
        return True
    return not (chars[end].isalpha() or chars[end] in ":/%$°")


def ordinal_suffix(chars: list[str], end: int) -> str | None:
    if end + 1 >= len(chars):
        return None
    a = chars[end].lower()
    b = chars[end + 1].lower()
    if (a, b) in {("s", "t"), ("n", "d"), ("r", "d"), ("t", "h")}:
        return a + b
    return None


def cardinal_phrase(int_part: str, negative: bool) -> str:
    words = []
    if negative:
        words.append("minus")
    words.append(integer_to_words(int_part))
    return " ".join(words)


def decimal_phrase(int_part: str, negative: bool, frac_part: str) -> str:
    words = [cardinal_phrase(int_part, negative), "point"]
    words.extend(digit_to_word(ch) for ch in frac_part)
    return " ".join(words)


def ordinal_phrase(raw: str) -> str:
    return ordinalize_cardinal_phrase(integer_to_words(raw))


def ordinalize_cardinal_phrase(cardinal: str) -> str:
    parts = cardinal.split()
    if not parts:
        return cardinal
    last = parts.pop()
    parts.append(
        {
            "one": "first",
            "two": "second",
            "three": "third",
            "four": "fourth",
            "five": "fifth",
            "six": "sixth",
            "seven": "seventh",
            "eight": "eighth",
            "nine": "ninth",
            "ten": "tenth",
            "eleven": "eleventh",
            "twelve": "twelfth",
            "thirteen": "thirteenth",
            "fourteen": "fourteenth",
            "fifteen": "fifteenth",
            "sixteen": "sixteenth",
            "seventeen": "seventeenth",
            "eighteen": "eighteenth",
            "nineteen": "nineteenth",
            "twenty": "twentieth",
            "thirty": "thirtieth",
            "forty": "fortieth",
            "fifty": "fiftieth",
            "sixty": "sixtieth",
            "seventy": "seventieth",
            "eighty": "eightieth",
            "ninety": "ninetieth",
            "hundred": "hundredth",
            "thousand": "thousandth",
            "million": "millionth",
            "billion": "billionth",
        }.get(last, last + ("ieth" if last.endswith("y") else "th"))
    )
    return " ".join(parts)


def year_phrase(year: int) -> str:
    if 1000 <= year <= 1099:
        rest = year - 1000
        if rest == 0:
            return "ten hundred"
        if rest < 10:
            return f"ten oh {digit_to_word(str(rest))}"
        return f"ten {integer_to_words(str(rest))}"
    if 1100 <= year <= 1999:
        first_two = year // 100
        last_two = year % 100
        if last_two == 0:
            return f"{integer_to_words(str(first_two))} hundred"
        return f"{integer_to_words(str(first_two))} {integer_to_words(str(last_two))}"
    if year == 2000:
        return "two thousand"
    if 2001 <= year <= 2009:
        return f"two thousand {integer_to_words(str(year - 2000))}"
    if 2010 <= year <= 2099:
        first_two = year // 100
        last_two = year % 100
        return f"{integer_to_words(str(first_two))} {integer_to_words(str(last_two))}"
    return integer_to_words(str(year))


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


def normalize_abbreviations(text: str) -> str:
    abbrevs = [
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
    ]
    chars = list(text)
    out: list[str] = []
    i = 0
    while i < len(chars):
        if i > 0 and chars[i - 1].isalnum():
            out.append(chars[i])
            i += 1
            continue
        for needle, repl in abbrevs:
            if text[i : i + len(needle)].lower() == needle:
                out.append(repl)
                i += len(needle)
                break
        else:
            out.append(chars[i])
            i += 1
    return "".join(out)


def normalize_acronyms(text: str) -> str:
    pronounce = {"NASA", "NATO", "RADAR", "ASCII", "JSON", "ASAP", "FAQ", "PIN"}
    chars = list(text)
    out: list[str] = []
    i = 0
    while i < len(chars):
        j = i
        while j < len(chars) and chars[j].isalpha():
            j += 1
        if j - i >= 2:
            token = text[i:j]
            if token.isupper() and not (j + 1 < len(chars) and chars[j] == "'" and chars[j + 1] in {"s", "S"}):
                if token in pronounce:
                    out.append(token)
                else:
                    out.append(" ".join(token))
                i = j
                continue
        out.append(chars[i])
        i += 1
    return "".join(out)


def normalize_money_time(text: str) -> str:
    chars = list(text)
    out: list[str] = []
    i = 0
    while i < len(chars):
        for matcher in (match_money_prefix, match_cents_suffix, match_time):
            result = matcher(chars, i)
            if result is not None:
                replacement, consumed = result
                out.append(replacement)
                i += consumed
                break
        else:
            out.append(chars[i])
            i += 1
    return "".join(out)


def match_money_prefix(chars: list[str], start: int) -> tuple[str, int] | None:
    if start >= len(chars) or chars[start] not in "$€£¥":
        return None
    unit = {
        "$": ("dollar", "dollars"),
        "€": ("euro", "euros"),
        "£": ("pound", "pounds"),
        "¥": ("yen", "yen"),
    }[chars[start]]
    int_part, frac_part, consumed = scan_currency_amount(chars, start + 1)
    if not int_part:
        return None
    value = int_part.lstrip("0") or "0"
    words = [integer_to_words(value), unit[0] if value == "1" else unit[1]]
    if frac_part:
        cents = cents_words(frac_part)
        if cents:
            words.append(cents)
    return " ".join(words), consumed + 1


def match_cents_suffix(chars: list[str], start: int) -> tuple[str, int] | None:
    if start > 0 and (chars[start - 1].isalnum() or chars[start - 1] in ":/"):
        return None
    int_part, consumed = scan_integer_span(chars, start)
    if not int_part or start + consumed >= len(chars) or chars[start + consumed] != "¢":
        return None
    value = int_part.lstrip("0") or "0"
    unit = "cent" if value == "1" else "cents"
    return f"{integer_to_words(value)} {unit}", consumed + 1


def match_time(chars: list[str], start: int) -> tuple[str, int] | None:
    if start >= len(chars) or not chars[start].isdigit():
        return None
    if start > 0 and (chars[start - 1].isalnum() or chars[start - 1] in ":/"):
        return None
    hour, hour_len = scan_integer_span(chars, start)
    if not hour:
        return None
    colon = start + hour_len
    if colon >= len(chars) or chars[colon] != ":":
        return None
    minute, minute_len = scan_integer_span(chars, colon + 1)
    if not minute or (colon + 1 + minute_len < len(chars) and chars[colon + 1 + minute_len] == ":"):
        return None
    return time_phrase(hour, minute), hour_len + 1 + minute_len


def scan_currency_amount(chars: list[str], start: int) -> tuple[str, str, int]:
    i = start
    int_part: list[str] = []
    frac_part: list[str] = []
    decimal = False
    while i < len(chars):
        ch = chars[i]
        if ch.isdigit():
            (frac_part if decimal else int_part).append(ch)
            i += 1
            continue
        if ch == "," and not decimal and i + 1 < len(chars) and chars[i + 1].isdigit():
            i += 1
            continue
        if ch == "." and not decimal and i + 1 < len(chars) and chars[i + 1].isdigit():
            decimal = True
            i += 1
            continue
        break
    return "".join(int_part), "".join(frac_part), i - start


def scan_integer_span(chars: list[str], start: int) -> tuple[str, int]:
    i = start
    out: list[str] = []
    while i < len(chars) and chars[i].isdigit():
        out.append(chars[i])
        i += 1
    return "".join(out), i - start


def cents_words(frac_part: str) -> str | None:
    digits = frac_part[:2]
    if len(digits) == 1:
        digits += "0"
    if not digits:
        return None
    value = digits.lstrip("0") or "0"
    unit = "cent" if value == "1" else "cents"
    return f"{integer_to_words(value)} {unit}"


def time_phrase(hour: str, minute: str) -> str:
    hour_words = integer_to_words(hour)
    minute_trimmed = minute.lstrip("0")
    if not minute_trimmed:
        return hour_words
    if len(minute) == 2 and minute.startswith("0"):
        return f"{hour_words} oh {digit_to_word(minute[1])}"
    return f"{hour_words} {integer_to_words(minute)}"


def normalize_money_time(text: str) -> str:
    chars = list(text)
    out: list[str] = []
    i = 0
    while i < len(chars):
        for matcher in (match_money_prefix, match_cents_suffix, match_time):
            result = matcher(chars, i)
            if result is not None:
                replacement, consumed = result
                out.append(replacement)
                i += consumed
                break
        else:
            out.append(chars[i])
            i += 1
    return "".join(out)


def match_money_prefix(chars: list[str], start: int) -> tuple[str, int] | None:
    if start >= len(chars) or chars[start] not in "$€£¥":
        return None
    unit = {
        "$": ("dollar", "dollars"),
        "€": ("euro", "euros"),
        "£": ("pound", "pounds"),
        "¥": ("yen", "yen"),
    }[chars[start]]
    int_part, frac_part, consumed = scan_currency_amount(chars, start + 1)
    if not int_part:
        return None
    value = int_part.lstrip("0") or "0"
    words = [integer_to_words(value), unit[0] if value == "1" else unit[1]]
    if frac_part:
        cents = cents_words(frac_part)
        if cents:
            words.append(cents)
    return " ".join(words), consumed + 1


def match_cents_suffix(chars: list[str], start: int) -> tuple[str, int] | None:
    int_part, consumed = scan_integer_span(chars, start)
    if not int_part or start + consumed >= len(chars) or chars[start + consumed] != "¢":
        return None
    value = int_part.lstrip("0") or "0"
    unit = "cent" if value == "1" else "cents"
    return f"{integer_to_words(value)} {unit}", consumed + 1


def match_time(chars: list[str], start: int) -> tuple[str, int] | None:
    if start >= len(chars) or not chars[start].isdigit():
        return None
    if start > 0 and (chars[start - 1].isalnum() or chars[start - 1] in ":/"):
        return None
    hour, hour_len = scan_integer_span(chars, start)
    if not hour:
        return None
    colon = start + hour_len
    if colon >= len(chars) or chars[colon] != ":":
        return None
    minute, minute_len = scan_integer_span(chars, colon + 1)
    if not minute or (colon + 1 + minute_len < len(chars) and chars[colon + 1 + minute_len] == ":"):
        return None
    return time_phrase(hour, minute), hour_len + 1 + minute_len


def scan_currency_amount(chars: list[str], start: int) -> tuple[str, str, int]:
    i = start
    int_part: list[str] = []
    frac_part: list[str] = []
    decimal = False
    while i < len(chars):
        ch = chars[i]
        if ch.isdigit():
            (frac_part if decimal else int_part).append(ch)
            i += 1
            continue
        if ch == "," and not decimal and i + 1 < len(chars) and chars[i + 1].isdigit():
            i += 1
            continue
        if ch == "." and not decimal and i + 1 < len(chars) and chars[i + 1].isdigit():
            decimal = True
            i += 1
            continue
        break
    return "".join(int_part), "".join(frac_part), i - start


def scan_integer_span(chars: list[str], start: int) -> tuple[str, int]:
    i = start
    out: list[str] = []
    while i < len(chars) and chars[i].isdigit():
        out.append(chars[i])
        i += 1
    return "".join(out), i - start


def cents_words(frac_part: str) -> str | None:
    digits = frac_part[:2]
    if len(digits) == 1:
        digits += "0"
    if not digits:
        return None
    value = digits.lstrip("0") or "0"
    unit = "cent" if value == "1" else "cents"
    return f"{integer_to_words(value)} {unit}"


def time_phrase(hour: str, minute: str) -> str:
    hour_words = integer_to_words(hour)
    minute_trimmed = minute.lstrip("0")
    if not minute_trimmed:
        return hour_words
    if len(minute) == 2 and minute.startswith("0"):
        return f"{hour_words} oh {digit_to_word(minute[1])}"
    return f"{hour_words} {integer_to_words(minute)}"


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

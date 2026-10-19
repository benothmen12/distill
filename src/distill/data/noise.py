"""Realistic noise for operator messages.

Operators type fast, on phones, with gloves, in two languages. A model trained only
on clean sentences breaks on "mach M104 vibr bcp ticket meca stp". This module
turns clean messages into messy ones while keeping every value recoverable, so the
label does not change:

- typos from the operator's keyboard layout (AZERTY in French, QWERTY in English);
- missing accents, lowercase, missing punctuation;
- SMS abbreviations (stp, bcp, pls, tmrw...);
- identifiers written loosely: "M104", "m-104", "machine 104", "ligne 3", "ticket 412".

Digits inside words are never corrupted, so numbers stay correct.
"""

from __future__ import annotations

import random
import re
import unicodedata

KEYBOARD_ROWS = {
    "fr": ("azertyuiop", "qsdfghjklm", "wxcvbn"),
    "en": ("qwertyuiop", "asdfghjkl", "zxcvbnm"),
}

ABBREVIATIONS = {
    "fr": {
        "s'il te plaît": "stp",
        "s'il te plait": "stp",
        "s'il vous plaît": "svp",
        "s'il vous plait": "svp",
        "beaucoup": "bcp",
        "pour": "pr",
        "quelque": "qq",
        "maintenant": "mtn",
        "aujourd'hui": "auj",
        "toujours": "tjrs",
        "problème": "pb",
        "machine": "mach",
        "maintenance": "maint",
        "mécanique": "méca",
        "électrique": "élec",
    },
    "en": {
        "please": "pls",
        "you": "u",
        "tomorrow": "tmrw",
        "with": "w/",
        "because": "bc",
        "thanks": "thx",
        "machine": "mach",
        "maintenance": "maint",
        "mechanical": "mech",
        "electrical": "elec",
    },
}

MACHINE = re.compile(r"\bM-(\d{3})\b")
LINE = re.compile(r"\bL([1-6])\b")
PART = re.compile(r"\bP-(\d{4})\b")
TICKET = re.compile(r"\bTCK-(\d{5})\b")
WORD = re.compile(r"[^\W\d_]{3,}", re.UNICODE)


def _neighbors(layout: tuple[str, ...]) -> dict[str, str]:
    near: dict[str, str] = {}
    for row in layout:
        for i, char in enumerate(row):
            near[char] = row[max(i - 1, 0) : i] + row[i + 1 : i + 2]
    return near


NEIGHBORS = {lang: _neighbors(rows) for lang, rows in KEYBOARD_ROWS.items()}


def typo(word: str, language: str, rng: random.Random) -> str:
    """One keyboard mistake: swap, drop, double or hit a neighboring key."""
    i = rng.randrange(1, len(word))  # keep the first letter, as people usually do
    op = rng.choice(("swap", "drop", "double", "neighbor"))
    if op == "swap" and i < len(word) - 1:
        return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
    if op == "drop":
        return word[:i] + word[i + 1 :]
    if op == "double":
        return word[:i] + word[i] + word[i:]
    near = NEIGHBORS[language].get(word[i].lower(), "")
    if near:
        return word[:i] + rng.choice(near) + word[i + 1 :]
    return word[:i] + word[i + 1 :]


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def loosen_identifiers(text: str, language: str, rng: random.Random, p: float) -> str:
    """Write identifiers the way people actually type them."""
    fr = language == "fr"

    def pick(match: re.Match[str], variants: list[str]) -> str:
        return rng.choice(variants) if rng.random() < p else match.group(0)

    text = MACHINE.sub(
        lambda m: pick(m, [f"M{m[1]}", f"m-{m[1]}", f"m{m[1]}", f"M {m[1]}", f"machine {m[1]}"]),
        text,
    )
    text = LINE.sub(lambda m: pick(m, [f"l{m[1]}", f"{'ligne' if fr else 'line'} {m[1]}"]), text)
    text = PART.sub(
        lambda m: pick(m, [f"P{m[1]}", f"p-{m[1]}", f"{'pièce' if fr else 'part'} {m[1]}"]),
        text,
    )
    return TICKET.sub(lambda m: pick(m, [f"TCK{m[1]}", f"tck-{m[1]}", f"ticket {int(m[1])}"]), text)


def abbreviate(text: str, language: str, rng: random.Random, p: float) -> str:
    for full, short in ABBREVIATIONS[language].items():
        pattern = re.compile(rf"\b{re.escape(full)}\b", re.IGNORECASE)
        text = pattern.sub(lambda m, s=short: s if rng.random() < p else m.group(0), text)
    return text


def add_noise(text: str, language: str, rng: random.Random, strength: float = 1.0) -> str:
    """Return a messier version of `text`. `strength` scales every probability."""
    language = language if language in KEYBOARD_ROWS else "en"
    s = max(0.0, min(strength, 2.0))

    text = loosen_identifiers(text, language, rng, p=0.6 * s)
    text = abbreviate(text, language, rng, p=0.5 * s)
    text = WORD.sub(
        lambda m: typo(m.group(0), language, rng) if rng.random() < 0.12 * s else m.group(0),
        text,
    )
    if rng.random() < 0.5 * s:
        text = strip_accents(text)
    if rng.random() < 0.4 * s:
        text = text.lower()
    if rng.random() < 0.4 * s:
        text = re.sub(r"[.!?,;:]+(\s|$)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()

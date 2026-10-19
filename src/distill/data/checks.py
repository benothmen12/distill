"""Grounding checks: every value in a label must come from the operator's message.

A label that contains information absent from the message teaches the model to
invent it. These checks catch the teacher's most common slips:

- an identifier, number or time missing from the message;
- a free-text argument (a ticket note, a search keyword) that adds facts the
  operator never stated, or that just repeats the field name;
- an enum written in its raw technical form ("in_progress") instead of words.

They are deliberately tolerant of how people really write: "machine 104" counts
for M-104, "2 hours" for 120 minutes, "next week" for 168 hours.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from difflib import SequenceMatcher
from typing import Any, get_args

from distill.tools import ALL_TOOLS

NUMBER_WORDS = {
    1: ("one", "a", "an", "single", "un", "une"),
    2: ("two", "couple", "deux"),
    3: ("three", "trois"),
    4: ("four", "quatre"),
    5: ("five", "cinq"),
    6: ("six",),
    7: ("seven", "sept"),
    8: ("eight", "huit"),
    9: ("nine", "neuf"),
    10: ("ten", "dix"),
    11: ("eleven", "onze"),
    12: ("twelve", "dozen", "douze", "douzaine"),
    15: ("fifteen", "quinze"),
    20: ("twenty", "vingt"),
    30: ("thirty", "trente"),
    45: ("forty-five", "quarante-cinq"),
    50: ("fifty", "cinquante"),
}

# Unit words that stand for a whole value on their own: "a week" = 168 hours.
UNIT_WORDS = {
    ("hours", 24): ("day", "jour", "journee"),
    ("hours", 168): ("week", "semaine"),
    ("minutes", 30): ("half an hour", "half hour", "demi-heure", "demi heure"),
    ("minutes", 60): ("hour", "heure"),
    ("minutes", 1440): ("day", "jour", "journee", "24h"),
}

RELATIVE_DAY_WORDS = (
    "today", "yesterday", "tonight", "this morning", "last night", "last",
    "aujourd", "hier", "avant-hier", "ce matin", "cette nuit", "ce soir", "dernier", "derniere",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
)  # fmt: skip

RAW_ENUM_VALUES = sorted(
    {
        value
        for tool in ALL_TOOLS
        for info in tool.model_fields.values()
        for value in get_args(info.annotation)
        if isinstance(value, str) and "_" in value
    }
)

STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "have", "been", "please",
    "pour", "avec", "dans", "sans", "une", "des", "les", "sur", "est", "sont", "plus",
}  # fmt: skip


def fold(text: str) -> str:
    """Lowercase, without accents."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _ints(text: str) -> set[int]:
    return {int(run) for run in re.findall(r"\d+", text)}


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", text) is not None


# --- identifiers ---------------------------------------------------------------

_ID_PATTERNS = {
    "machine_id": r"\bm\s*-?\s*0*{n}\b|\bmach\w*\s*(?:n°|no\.?|#)?\s*0*{n}\b",
    "line_id": r"\bl\s*-?\s*{n}\b|\b(?:ligne|line)\s*(?:n°|no\.?|#)?\s*{n}\b",
    "part_number": r"\bp\s*-?\s*0*{n}\b|\b(?:piece|part|ref\w*)\s*(?:n°|no\.?|#)?\s*0*{n}\b",
    "ticket_id": r"\btck\s*-?\s*0*{n}\b|\bticket\s*(?:n°|no\.?|#)?\s*0*{n}\b",
}
_ANY_ID_PATTERNS = {
    "machine_id": r"\bm\s*-?\s*\d{3}\b|\bmach\w*\s*(?:n°|no\.?|#)?\s*\d{3}\b",
    "line_id": r"\bl\s*-?\s*[1-6]\b|\b(?:ligne|line)\s*[1-6]\b",
    "part_number": r"\bp\s*-?\s*\d{4}\b|\b(?:piece|part|ref\w*)\s*(?:n°|no\.?|#)?\s*\d{4}\b",
    "ticket_id": r"\btck\s*-?\s*\d{3,5}\b|\bticket\s*(?:n°|no\.?|#)?\s*\d{3,5}\b",
}


def identifier_in_message(field: str, value: str, message: str) -> bool:
    """True if the identifier is in the message, exactly or written loosely."""
    number = str(int(re.sub(r"\D", "", value)))
    pattern = _ID_PATTERNS[field].format(n=number)
    return re.search(pattern, fold(message)) is not None


def any_identifier_in_message(field: str, message: str) -> bool:
    """True if the message contains some identifier of this kind."""
    return re.search(_ANY_ID_PATTERNS[field], fold(message)) is not None


# --- numbers, dates, times -------------------------------------------------------


def number_in_message(field: str, value: int, message: str) -> bool:
    """True if the number can be read from the message, allowing exact conversions."""
    text = fold(message)
    candidates = {value}
    unit = "minutes" if field.endswith("_minutes") else "hours" if field.endswith("_hours") else ""
    if unit == "minutes" and value % 60 == 0:
        candidates.add(value // 60)
    if unit == "hours" and value % 24 == 0:
        candidates.add(value // 24)
    if candidates & _ints(text):
        return True
    for n in candidates:
        if any(_has_word(text, w) for w in NUMBER_WORDS.get(n, ())):
            return True
    return any(_has_word(text, w) for w in UNIT_WORDS.get((unit, value), ()))


def time_in_message(value: str | datetime, message: str) -> bool:
    """True if the hour of a datetime is in the message (14, 2 pm, midi...)."""
    moment = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    text = fold(message)
    hours = {moment.hour, moment.hour % 12 or 12}
    if hours & _ints(text):
        return True
    if moment.hour == 12 and any(_has_word(text, w) for w in ("noon", "midi")):
        return True
    return moment.hour == 0 and any(_has_word(text, w) for w in ("midnight", "minuit"))


def date_in_message(value: str | date, message: str) -> bool:
    """True if the day of a date is in the message, or a relative day word is."""
    day = value if isinstance(value, date) else date.fromisoformat(value)
    text = fold(message)
    return day.day in _ints(text) or any(w in text for w in RELATIVE_DAY_WORDS)


# --- free text -------------------------------------------------------------------


def _content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]{4,}", fold(text)) if w not in STOPWORDS]


def is_grounded(value: str, message: str) -> bool:
    """True if most content words of a free-text value appear in the message.

    Words match on their first four letters, so "vibrations" matches "vibre" and a
    typo at the end of a word still matches.
    """
    words = _content_words(value)
    if not words:
        return True
    stems = {w[:4] for w in _content_words(message)}
    found = sum(1 for w in words if w[:4] in stems)
    return found / len(words) >= 0.5


def is_placeholder(field: str, value: str) -> bool:
    """True if the value just repeats the field name, as in keyword='keywrd'."""
    folded = fold(value).strip(" .'\"")
    if folded in {"", "text", "n/a", "none", "null", "...", "todo"}:
        return True
    return SequenceMatcher(None, folded, field.replace("_", " ")).ratio() >= 0.7


def raw_enum_in_message(message: str) -> str | None:
    """Return a technical enum value written as is ("in_progress"), if any."""
    text = message.lower()
    return next((v for v in RAW_ENUM_VALUES if v in text), None)


def check_value(field: str, value: Any, message: str) -> str | None:
    """Return the rejection reason for one fixed argument, or None if it is grounded."""
    if field in _ID_PATTERNS:
        return None if identifier_in_message(field, value, message) else "identifier_missing"
    if isinstance(value, bool):
        return None  # urgency is phrased too freely to check reliably
    if isinstance(value, int):
        return None if number_in_message(field, value, message) else "number_missing"
    if field == "start_time":
        return None if time_in_message(value, message) else "time_missing"
    if field == "shift_date":
        return None if date_in_message(value, message) else "date_missing"
    return None

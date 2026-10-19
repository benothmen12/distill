"""Scenario sampling.

We do not ask the teacher to invent tool calls. We sample the expected calls
ourselves, inside the constraints of the schemas, and only ask the teacher to
write the operator message that leads to them. Labels are therefore correct by
construction, and we control the distribution: which tools, which categories,
which languages, how often defaults are left implicit.

Each spec is sampled from its own seeded RNG, so `sample_specs(1000)` starts with
exactly the same specs as `sample_specs(500)`. Growing the dataset later never
changes the examples already generated.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta
from typing import Any, Literal, get_args

from pydantic import BaseModel

from distill.tools import ALL_TOOLS, TOOLS_BY_NAME, Tool

Category = Literal["single", "parallel", "clarify", "out_of_scope"]

CATEGORY_WEIGHTS: dict[str, float] = {
    "single": 0.62,
    "parallel": 0.14,
    "clarify": 0.12,
    "out_of_scope": 0.12,
}
LANGUAGE_WEIGHTS: dict[str, float] = {"en": 0.6, "fr": 0.4}

STYLES = (
    "terse, like a quick chat message",
    "polite and complete",
    "stressed operator in the middle of an incident",
    "with a couple of small typos",
    "shift supervisor giving instructions",
    "uses shop-floor jargon",
    "spoken style, as if dictated by voice",
    "keywords only, not a full sentence, like a quick note between two tasks",
    "a 3 to 6 word fragment with no grammar at all",
)

OUT_OF_SCOPE_TOPICS = (
    "the canteen menu",
    "the weather this weekend",
    "a birthday message for a colleague",
    "last night's football results",
    "a personal leave request to HR",
    "translating a sentence for a friend",
    "a cooking recipe",
    "a general knowledge question",
    "help writing a personal email",
    "the price of a smartphone",
)

# Pairs of tools an operator plausibly asks for in one message. Every pair is
# independent: neither call needs the other's result, so they can run in parallel.
PARALLEL_PAIRS = (
    ("get_machine_status", "get_sensor_readings"),
    ("get_failure_risk", "search_maintenance_history"),
    ("create_maintenance_ticket", "notify_team"),
    ("create_maintenance_ticket", "schedule_downtime"),
    ("get_spare_part_stock", "get_spare_part_stock"),
    ("get_production_kpis", "get_production_kpis"),
    ("get_shift_report", "get_production_kpis"),
    ("update_ticket_status", "notify_team"),
    ("order_spare_part", "notify_team"),
    ("get_machine_status", "get_machine_status"),
)

IDENTIFIER_FIELDS = ("machine_id", "line_id", "ticket_id", "part_number")
FREE_TEXT_FIELDS = ("issue", "reason", "message", "note", "keyword")

MACHINE_IDS = [f"M-{n:03d}" for n in (*range(101, 161), *range(201, 241), *range(301, 321))]
LINE_IDS = [f"L{n}" for n in range(1, 7)]

NUMBER_POOLS: dict[str, list[int]] = {
    "window_minutes": [15, 30, 60, 120, 180, 240, 480, 720, 1440],
    "horizon_hours": [6, 8, 12, 24, 48, 72, 168],
    "duration_minutes": [15, 20, 30, 45, 60, 90, 120, 180, 240, 480],
    "quantity": [1, 1, 2, 2, 3, 4, 5, 6, 8, 10, 12, 20, 50],
    "limit": [3, 5, 10, 20],
}

OPTIONAL_INCLUDE_PROBABILITY = 0.5


class CallSpec(BaseModel):
    tool: str
    fixed: dict[str, Any]
    free_text: list[str]


class Spec(BaseModel):
    id: str
    category: Category
    language: Literal["en", "fr"]
    style: str
    now: datetime
    calls: list[CallSpec]
    missing: str | None = None
    topic: str | None = None


def _weighted(rng: random.Random, weights: dict[str, float]) -> str:
    return rng.choices(list(weights), weights=list(weights.values()), k=1)[0]


def _sample_now(rng: random.Random) -> datetime:
    day = date(2026, 1, 5) + timedelta(days=rng.randrange(0, 360))
    return datetime(day.year, day.month, day.day, rng.randrange(6, 22), rng.randrange(0, 60, 5))


def _sample_value(field: str, annotation: Any, rng: random.Random, now: datetime) -> Any:
    if field == "machine_id":
        return rng.choice(MACHINE_IDS)
    if field == "line_id":
        return rng.choice(LINE_IDS)
    if field == "ticket_id":
        return f"TCK-{rng.randint(100, 9999):05d}"
    if field == "part_number":
        return f"P-{rng.randint(1000, 9999)}"
    if field in NUMBER_POOLS:
        return rng.choice(NUMBER_POOLS[field])
    if field == "start_time":
        start = now + timedelta(hours=rng.randint(1, 72))
        start = start.replace(minute=rng.choice((0, 15, 30, 45)), second=0)
        return start.strftime("%Y-%m-%dT%H:%M")
    if field == "shift_date":
        return (now.date() - timedelta(days=rng.choice((0, 1, 1, 2, 3, 7)))).isoformat()
    if annotation is bool:
        return rng.random() < 0.4
    choices = get_args(annotation)
    if choices:
        return rng.choice(choices)
    raise ValueError(f"no sampler for field {field!r} ({annotation!r})")


def sample_call(tool: type[Tool], rng: random.Random, now: datetime) -> CallSpec:
    """Sample the arguments of one call. Optional arguments are sometimes left out,
    so the model also learns to rely on defaults."""
    fixed: dict[str, Any] = {}
    free_text: list[str] = []
    for field, info in tool.model_fields.items():
        if not info.is_required() and rng.random() >= OPTIONAL_INCLUDE_PROBABILITY:
            continue
        if field in FREE_TEXT_FIELDS:
            free_text.append(field)
        else:
            fixed[field] = _sample_value(field, info.annotation, rng, now)
    return CallSpec(tool=tool.name, fixed=fixed, free_text=free_text)


def _share_identifiers(first: CallSpec, second: CallSpec) -> None:
    """Two calls about the same machine or line should name the same one, unless
    both calls use the same tool (then they must differ to be distinct requests)."""
    if first.tool == second.tool:
        return
    for field in ("machine_id", "line_id", "part_number"):
        if field in first.fixed and field in second.fixed:
            second.fixed[field] = first.fixed[field]


def _sample_parallel(rng: random.Random, now: datetime) -> list[CallSpec]:
    first_name, second_name = rng.choice(PARALLEL_PAIRS)
    first = sample_call(TOOLS_BY_NAME[first_name], rng, now)
    second = sample_call(TOOLS_BY_NAME[second_name], rng, now)
    while first_name == second_name and second.fixed == first.fixed:
        second = sample_call(TOOLS_BY_NAME[second_name], rng, now)
    _share_identifiers(first, second)
    return [first, second]


def _sample_clarify(rng: random.Random, now: datetime) -> tuple[CallSpec, str]:
    """Pick a call and remove one required argument the operator forgets to give."""
    while True:
        call = sample_call(rng.choice(ALL_TOOLS), rng, now)
        required = [
            f
            for f, info in TOOLS_BY_NAME[call.tool].model_fields.items()
            if info.is_required() and f in call.fixed
        ]
        identifiers = [f for f in required if f in IDENTIFIER_FIELDS]
        # Mostly forgotten identifiers: they are the most common real-world gap, and
        # their absence can be checked automatically in the generated message.
        pool = identifiers if identifiers and rng.random() < 0.8 else required
        if pool:
            missing = rng.choice(pool)
            del call.fixed[missing]
            return call, missing


def sample_spec(seed: int, index: int) -> Spec:
    rng = random.Random(f"{seed}-{index}")
    category = _weighted(rng, CATEGORY_WEIGHTS)
    now = _sample_now(rng)
    common = {
        "id": f"s{seed}-{index:06d}",
        "category": category,
        "language": _weighted(rng, LANGUAGE_WEIGHTS),
        "style": rng.choice(STYLES),
        "now": now,
    }
    if category == "single":
        return Spec(**common, calls=[sample_call(rng.choice(ALL_TOOLS), rng, now)])
    if category == "parallel":
        return Spec(**common, calls=_sample_parallel(rng, now))
    if category == "clarify":
        call, missing = _sample_clarify(rng, now)
        return Spec(**common, calls=[call], missing=missing)
    return Spec(**common, calls=[], topic=rng.choice(OUT_OF_SCOPE_TOPICS))


def sample_specs(n: int, seed: int = 42) -> list[Spec]:
    return [sample_spec(seed, i) for i in range(n)]

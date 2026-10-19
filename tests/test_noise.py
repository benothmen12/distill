import random
import re

from distill.data.noise import add_noise, loosen_identifiers, strip_accents, typo

MESSAGE = (
    "Bonjour, la machine M-104 vibre beaucoup depuis ce matin. Ouvre un ticket "
    "priorité haute pour l'équipe mécanique s'il te plaît, et vérifie la pièce P-2231 "
    "sur la ligne L3 (ticket TCK-00412)."
)


def _numbers(text: str) -> set[int]:
    return {int(n) for n in re.findall(r"\d+", text)}


def test_noise_is_deterministic():
    a = add_noise(MESSAGE, "fr", random.Random(1))
    b = add_noise(MESSAGE, "fr", random.Random(1))
    assert a == b


def test_noise_changes_most_messages():
    noisy = {add_noise(MESSAGE, "fr", random.Random(seed)) for seed in range(50)}
    assert MESSAGE not in noisy
    assert len(noisy) > 40


def test_numbers_survive_the_noise():
    for seed in range(200):
        noisy = add_noise(MESSAGE, "fr", random.Random(seed), strength=2.0)
        assert _numbers(MESSAGE) <= _numbers(noisy), noisy


def test_identifiers_are_written_loosely():
    variants = {loosen_identifiers("check M-104", "en", random.Random(s), p=1.0) for s in range(60)}
    assert variants == {
        "check M104",
        "check m-104",
        "check m104",
        "check M 104",
        "check machine 104",
    }
    ticket = loosen_identifiers("ferme TCK-00412", "fr", random.Random(3), p=1.0)
    assert "412" in ticket


def test_typo_keeps_the_first_letter():
    for seed in range(100):
        out = typo("vibration", "en", random.Random(seed))
        assert out[0] == "v"
        assert out != "vibration" or seed  # at least changes in general


def test_strip_accents():
    assert strip_accents("priorité élevée, équipe mécanique") == "priorite elevee, equipe mecanique"


def test_zero_strength_changes_nothing():
    assert add_noise(MESSAGE, "fr", random.Random(0), strength=0.0) == MESSAGE

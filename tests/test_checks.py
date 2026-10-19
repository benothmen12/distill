"""Grounding checks, tested on real teacher outputs from the first generation run."""

import pytest

from distill.data.checks import (
    any_identifier_in_message,
    check_value,
    date_in_message,
    identifier_in_message,
    is_grounded,
    is_placeholder,
    number_in_message,
    raw_enum_in_message,
    time_in_message,
)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("machine_id", "M-104", "la M-104 vibre"),
        ("machine_id", "M-104", "machine 104 vibre bcp"),
        ("machine_id", "M-104", "m104 en alarme"),
        ("machine_id", "M-007", "check M-007"),
        ("line_id", "L3", "OEE de la ligne 3"),
        ("part_number", "P-2231", "stock pièce 2231 ?"),
        ("ticket_id", "TCK-00412", "ferme le ticket 412"),
        ("ticket_id", "TCK-02493", "TCK-02493 en cours"),
    ],
)
def test_identifiers_written_loosely_are_found(field, value, message):
    assert identifier_in_message(field, value, message)


def test_wrong_identifier_is_not_found():
    assert not identifier_in_message("machine_id", "M-104", "la M-105 vibre")
    assert not identifier_in_message("machine_id", "M-104", "commande 104 roulements")


def test_any_identifier_detects_a_leaked_value():
    assert any_identifier_in_message("machine_id", "la machine 238 est en panne")
    assert not any_identifier_in_message("machine_id", "la machine est en panne")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("horizon_hours", 24, "over the next 24 hours"),
        ("horizon_hours", 168, "risk for the coming week"),
        ("horizon_hours", 48, "dans les deux prochains jours"),
        ("duration_minutes", 120, "stop it for 2 hours"),
        ("duration_minutes", 30, "une demi-heure suffit"),
        ("quantity", 3, "commande trois roulements"),
        ("quantity", 1, "order a bearing"),
        ("limit", 10, "the last ten interventions"),
    ],
)
def test_numbers_are_found_with_conversions(field, value, message):
    assert number_in_message(field, value, message)


def test_missing_number_is_detected():
    # Real output: the limit (10) was never said.
    assert check_value("limit", 10, "M-123 intervntions keywrd") == "number_missing"


def test_times_and_dates():
    assert time_in_message("2026-10-21T14:00", "demain à 14h")
    assert time_in_message("2026-10-21T14:00", "tomorrow at 2 pm")
    assert not time_in_message("2026-10-21T14:00", "tomorrow morning")
    assert date_in_message("2026-10-19", "report of yesterday's night shift")
    assert date_in_message("2026-10-12", "rapport du 12 octobre")
    assert not date_in_message("2026-10-12", "rapport de l'équipe de nuit")


def test_ungrounded_note_is_detected():
    # Real output: the message never mentions a sensor.
    assert not is_grounded("Replaced faulty sensor", "Ticket TCK-04431 resolved.")


def test_grounded_free_text_passes():
    message = "Message critique à l'équipe électrique : vérifiez le tableau de distribution"
    assert is_grounded("vérifiez le tableau de distribution", message)
    assert is_grounded("Vibrations anormales", "la M-104 vibre de façon anormale")


def test_field_name_as_value_is_a_placeholder():
    # Real output: keyword="keywrd".
    assert is_placeholder("keyword", "keywrd")
    assert is_placeholder("note", "Note")
    assert not is_placeholder("keyword", "bearing")


def test_raw_enum_is_detected():
    # Real output: "TCK-02493 in_progress".
    assert raw_enum_in_message("TCK-02493 in_progress") == "in_progress"
    assert raw_enum_in_message("TCK-02493 en cours") is None

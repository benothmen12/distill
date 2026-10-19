"""Free-text values the teacher writes in another shape, and language labels."""

from datetime import datetime

import pytest

from distill.data.generate import Rejected, _check_language, _free_text_values, assemble_example
from distill.data.specs import CallSpec, Spec

NOW = datetime(2026, 1, 1, 10, 0)


def _spec(language, calls, category="single"):
    return Spec(id="t-1", category=category, language=language, style="test", now=NOW, calls=calls)


NOTIFY = CallSpec(tool="notify_team", fixed={"team": "electrical"}, free_text=["message"])
ORDER = CallSpec(
    tool="order_spare_part", fixed={"part_number": "P-1600", "quantity": 4}, free_text=[]
)


def test_flat_list_of_strings_is_read_in_field_order():
    spec = _spec("en", [ORDER, NOTIFY], category="parallel")
    per_call = _free_text_values(spec, ["lubricate the gearboxes"])
    assert per_call == [{}, {"message": "lubricate the gearboxes"}]


def test_value_given_under_the_wrong_call_index_is_still_found():
    spec = _spec("en", [ORDER, NOTIFY], category="parallel")
    per_call = _free_text_values(spec, [{"message": "lubricate the gearboxes"}])
    assert per_call[1] == {"message": "lubricate the gearboxes"}


def test_single_call_with_flat_string_is_accepted():
    spec = _spec("en", [NOTIFY])
    output = {
        "user_message": "Tell the electrical team the breaker tripped again",
        "free_text": ["the breaker tripped again"],
        "assistant_reply": None,
    }
    example = assemble_example(spec, output, "test")
    assert example.tool_calls[0].arguments["message"] == "the breaker tripped again"


def test_flat_string_that_is_not_in_the_message_is_still_rejected():
    spec = _spec("en", [NOTIFY])
    output = {
        "user_message": "Tell the electrical team about the breaker",
        "free_text": ["replace the fuse tomorrow morning"],
        "assistant_reply": None,
    }
    with pytest.raises(Rejected) as exc:
        assemble_example(spec, output, "test")
    assert exc.value.reason == "ungrounded_free_text"


def test_french_message_labelled_english_is_rejected():
    spec = _spec("en", [NOTIFY])
    output = {
        "user_message": "Équipe électrique : le disjoncteur saute encore",
        "free_text": [{"message": "le disjoncteur saute encore"}],
        "assistant_reply": None,
    }
    with pytest.raises(Rejected) as exc:
        assemble_example(spec, output, "test")
    assert exc.value.reason == "language_mismatch"


def test_english_message_labelled_french_is_rejected():
    with pytest.raises(Rejected) as exc:
        _check_language("Please check the stock of the part for me", "fr")
    assert exc.value.reason == "language_mismatch"


@pytest.mark.parametrize(
    "message, language",
    [
        ("P-5495 stock where", "en"),
        ("Can you check how many units of P-2790 we have?", "en"),
        ("Bonjour, pourriez-vous me fournir le stock de P-4043 ?", "fr"),
        ("M-238 état ?", "fr"),
    ],
)
def test_messages_in_their_own_language_are_accepted(message, language):
    _check_language(message, language)

"""Post-validator — checks on the model text (§8)."""

from groww_faq.query.validate import template_body, validate_answer

CONTEXT = (
    "HDFC Mutual Fund | HDFC Large Cap Fund | Direct Growth | "
    "expense_ratio: 0.74% | Exit load of 1% if redeemed within 1 year | "
    "Minimum SIP investment ₹100"
)


def test_valid_one_sentence_with_exact_number():
    ok, reason = validate_answer("The expense ratio is 0.74%.", CONTEXT)
    assert ok, reason


def test_four_sentences_rejected():
    text = (
        "This is sentence one. This is sentence two. This is sentence "
        "three. This is sentence four."
    )
    ok, reason = validate_answer(text, CONTEXT)
    assert not ok and reason == "too_many_sentences"


def test_short_retry_after_four_sentences_would_pass():
    # a short sentence on retry satisfies the validator (tested via pipeline)
    ok, _ = validate_answer("The expense ratio is 0.74%.", CONTEXT)
    assert ok


def test_url_in_answer_rejected():
    ok, reason = validate_answer(
        "See https://groww.in for the expense ratio.", CONTEXT
    )
    assert not ok and reason == "url_in_answer"


def test_advice_vocabulary_rejected():
    ok, reason = validate_answer("You should buy this fund.", CONTEXT)
    assert not ok and reason == "advice_vocabulary"


def test_invented_number_rejected():
    ok, reason = validate_answer("The expense ratio is 1.25%.", CONTEXT)
    assert not ok and reason == "invented_number"


def test_decimal_point_is_not_a_sentence_break():
    ok, reason = validate_answer(
        "The expense ratio is 0.74% and the exit load is 1% if redeemed "
        "within 1 year.", CONTEXT
    )
    assert ok, reason


def test_template_caps_to_two_sentences():
    long_text = "First sentence here. Second sentence here. Third one. Fourth."
    body = template_body(long_text)
    assert body == "First sentence here. Second sentence here."

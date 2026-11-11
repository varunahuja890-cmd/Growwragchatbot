"""Intent classifier — decide before retrieval (§8)."""

from groww_faq.query.intent import (
    ADVICE,
    COMPARISON,
    FACTUAL,
    PERFORMANCE,
    PORTFOLIO,
    PREDICTION,
    classify_intent,
)


def test_advice():
    assert classify_intent("Should I invest in HDFC Mid Cap?") == ADVICE


def test_comparison_explicit():
    assert classify_intent("Which is better, Large Cap or Mid Cap?") == COMPARISON


def test_two_scheme_names_is_comparison_even_if_mild():
    assert classify_intent("Tell me about Large Cap and Mid Cap") == COMPARISON


def test_prediction():
    assert classify_intent("Will this fund give 15% next year?") == PREDICTION


def test_portfolio():
    assert classify_intent("How much of my portfolio should be in ELSS?") == PORTFOLIO


def test_performance_redirect():
    assert (
        classify_intent("What were the 3-year returns of HDFC Flexi Cap?")
        == PERFORMANCE
    )


def test_factual():
    assert classify_intent("What is the expense ratio?") == FACTUAL
    assert classify_intent("What is the exit load of HDFC Large Cap?") == FACTUAL


def test_amc_plus_one_scheme_is_not_comparison():
    # "HDFC mutual fund" is the AMC page, not a second scheme.
    assert (
        classify_intent(
            "What is the expense ratio of the HDFC mutual fund large cap scheme?"
        )
        == FACTUAL
    )

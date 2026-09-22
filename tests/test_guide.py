"""The plain-English guide must stay complete, plain, honest and wired in."""

import re
from pathlib import Path

import pytest

from src import exporting as ex
from src import guide

APP_SOURCE = (Path(__file__).resolve().parents[1] / "app.py").read_text(
    encoding="utf-8"
)

METHODS = (
    "Equal Weight",
    "Minimum Variance",
    "Risk Parity",
    "Maximum Diversification",
    "Maximum Sharpe",
    "Diversified Maximum Sharpe",
    "Minimum CVaR",
    "Maximum Return / CVaR",
    "HRP Diversification",
)


def _words(text: str) -> int:
    return len(text.split())


def _every_text():
    for term in guide.GLOSSARY.values():
        yield f"term:{term.key}", term.plain
        yield f"example:{term.key}", term.example
        yield f"look_for:{term.key}", term.look_for
    for key, section in guide.SECTION_GUIDES.items():
        yield f"intro:{key}", section.intro
        yield f"what:{key}", section.what
        for item in (*section.look_for, *section.cannot):
            yield f"bullet:{key}", item
    for key, text in guide.CAPTIONS.items():
        yield f"caption:{key}", text
    for key, text in guide.HELP.items():
        yield f"help:{key}", text
    yield "landing", guide.LANDING_INTRO
    yield "landing-note", guide.LANDING_NOTE
    for _title, body in (*guide.LANDING_STEPS, *guide.LANDING_IDEAS):
        yield "landing-item", body


# ---------------------------------------------------------------- structure


def test_every_section_term_exists_once():
    for key, section in guide.SECTION_GUIDES.items():
        assert len(set(section.terms)) == len(section.terms), key
        for term in section.terms:
            assert term in guide.GLOSSARY, f"{key}: unknown term {term!r}"


def test_every_glossary_term_appears_on_some_page():
    used = {term for section in guide.SECTION_GUIDES.values() for term in section.terms}
    unused = set(guide.GLOSSARY) - used
    assert not unused, f"terms no page shows: {sorted(unused)}"


def test_six_pages_each_fully_described():
    assert list(guide.SECTION_GUIDES) == [
        "construction",
        "multi_period",
        "bootstrap",
        "walk_forward",
        "diversification",
        "summary",
    ]
    for key, section in guide.SECTION_GUIDES.items():
        assert section.intro and section.what, key
        assert len(section.look_for) == 3, key
        assert len(section.cannot) >= 2, f"{key} must say what it cannot tell you"


def test_all_nine_methods_have_a_blurb():
    assert tuple(guide.METHOD_BLURBS) == METHODS


# ------------------------------------------------------------- readability


def test_definitions_are_short_enough_to_read():
    for term in guide.GLOSSARY.values():
        assert 5 <= _words(term.plain) <= 60, term.key
        assert _words(term.example) <= 45, term.key
        assert _words(term.look_for) <= 45, term.key
        assert term.plain.endswith("."), term.key


def test_no_stray_whitespace_or_placeholders():
    for label, text in _every_text():
        assert text == text.strip(), label
        assert "  " not in text, label
        if label != "caption:cvar_frontier":
            assert "{" not in text and "}" not in text, label


def test_summaries_avoid_specialist_jargon():
    banned = (
        "covariance",
        "optimis",
        "herfindahl",
        "quantile",
        "look-ahead",
        "heteroskedastic",
    )
    plain_places = (
        *(s.intro for s in guide.SECTION_GUIDES.values()),
        *(s.what for s in guide.SECTION_GUIDES.values()),
        *(b for s in guide.SECTION_GUIDES.values() for b in (*s.look_for, *s.cannot)),
        *guide.CAPTIONS.values(),
        *guide.HELP.values(),
        guide.LANDING_INTRO,
    )
    for text in plain_places:
        lowered = text.lower()
        for word in banned:
            assert word not in lowered, f"{word!r} in: {text[:60]}"


# ---------------------------------------------------------------- honesty


def test_the_guide_describes_and_never_advises():
    forbidden = (
        r"\byou should\b",
        r"\bwe recommend\b",
        r"\bi recommend\b",
        r"\byou must\b",
        r"\bbest portfolio\b",
        r"\bwill outperform\b",
        r"\bguaranteed (return|profit|to)\b",
    )
    for label, text in _every_text():
        for pattern in forbidden:
            assert not re.search(pattern, text, re.I), f"{label}: {text[:70]}"


def test_forecast_terms_say_they_are_not_forecasts():
    for key in ("Expected return E(r)", "Bootstrap simulation", "Efficient frontier"):
        term = guide.GLOSSARY[key]
        assert re.search(
            r"not a forecast|not a prediction|shows the spread",
            term.look_for + term.plain,
            re.I,
        ) or ("past" in (term.look_for + term.plain).lower()), key


def test_the_landing_page_carries_the_disclaimer():
    note = guide.LANDING_NOTE.lower()
    assert "not" in note and "investment advice" in note
    assert "past results" in note


# ---------------------------------------------------------------- lookup


LABELS_THE_APP_PRINTS = [
    "CAGR",
    "Volatility",
    "Sharpe",
    "Sortino",
    "Max Drawdown",
    "CVaR 95%",
    "CVaR 99%",
    "Expected Return E(r)",
    "Expected Volatility",
    "Expected Sharpe",
    "Historical CAGR",
    "Bootstrap Median CAGR",
    "Bootstrap 5th Wealth",
    "OOS Median Return",
    "Beat Current Portfolio",
    "Actual ETFs",
    "Actual ETFs 3Y",
    "Effective Holdings",
    "Effective ETFs",
    "Effective Risk Bets",
    "Weighted Correlation",
    "Diversification Ratio",
    "Largest Risk Share",
    "Largest Risk Contributor",
    "Largest Weight 3Y",
    "Capital to Reallocate",
    "Capital Change 3Y vs Full",
    "Capital Change 5Y vs Full",
    "Δ E(r)",
    "Δ Risk Bets",
    "Δ DR",
    "Median Terminal Wealth",
    "5th Percentile Wealth",
    "Median CAGR",
    "5th Percentile CAGR",
    "P(Loss at Horizon)",
    "P(CAGR > 6%)",
    "Chance CAGR > 8%",
    "Median Max Drawdown",
    "5th Percentile Max Drawdown",
    "Median Return",
    "Worst Return",
    "Best Return",
    "Median Volatility",
    "Historical Volatility",
    "Portfolio Volatility",
    "Median Annualized Return",
    *METHODS,
]


@pytest.mark.parametrize("label", LABELS_THE_APP_PRINTS)
def test_every_label_the_app_prints_has_an_explanation(label):
    assert guide.find_term(label) is not None, label


def test_unknown_labels_are_not_invented():
    assert guide.find_term("Frobnication index") is None


# --------------------------------------------------------------- wiring


def test_every_caption_the_app_uses_exists_and_none_are_orphaned():
    used = set(re.findall(r'guide\.caption\(\s*"(\w+)"', APP_SOURCE))
    assert used <= set(guide.CAPTIONS), used - set(guide.CAPTIONS)
    assert set(guide.CAPTIONS) <= used, f"unused captions: {set(guide.CAPTIONS) - used}"


def test_every_help_text_the_app_uses_exists_and_none_are_orphaned():
    used = set(re.findall(r'guide\.HELP\["(\w+)"\]', APP_SOURCE))
    assert used <= set(guide.HELP), used - set(guide.HELP)
    assert set(guide.HELP) <= used, f"unused help: {set(guide.HELP) - used}"


def test_every_page_shows_its_summary_and_its_guide():
    for key in guide.SECTION_GUIDES:
        assert f'plain_english("{key}")' in APP_SOURCE, key
        assert f'glossary_expander("{key}")' in APP_SOURCE, key


def test_caption_placeholders_are_filled():
    assert "{cvar}" not in guide.caption("cvar_frontier", cvar="95%")
    assert "95%" in guide.caption("cvar_frontier", cvar="95%")


# ------------------------------------------------------------------ PDF


@pytest.mark.parametrize("section_key", list(guide.SECTION_GUIDES))
def test_a_section_pdf_carries_the_summary_and_the_terms(section_key):
    blocks = [
        ex.Heading("A page", 1),
        ex.Text("Opening paragraph."),
        *guide.pdf_intro(section_key),
        *guide.pdf_terms(section_key),
    ]
    pdf = ex.build_report_bytes(
        section_label="Guide test",
        blocks=blocks,
        parameters=[("Universe", "A, B")],
        subtitle="Guide test",
    )
    assert pdf.startswith(b"%PDF-")
    assert len(pdf) > 8_000


def test_the_full_report_glossary_lists_every_term_once():
    frame = guide._term_frame(guide.all_terms())
    assert len(frame) == len(guide.GLOSSARY)
    assert frame.index.is_unique
    assert list(frame.index) == sorted(frame.index, key=str.lower)


def test_the_glossary_pdf_builds():
    pdf = ex.build_report_bytes(
        section_label="Glossary",
        blocks=guide.pdf_glossary(),
        parameters=[("Universe", "A, B")],
        subtitle="Glossary",
    )
    assert pdf.startswith(b"%PDF-")
    # a page per few dozen terms, so a real glossary is several pages long
    assert len(re.findall(rb"/Type\s*/Page\b", pdf)) >= 3

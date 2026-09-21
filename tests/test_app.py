"""End-to-end smoke test for the Streamlit composition layer.

Runs the real ``app.py`` against a deterministic synthetic price history so it
exercises every section, chart and export without touching the network.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import data as data_module

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"

#: Matches ``config/universe.yaml`` plus enough history for a 36-month
#: training window and several out-of-sample holding blocks.
UNIVERSE = [
    "TMFC",
    "VWO",
    "ACWI",
    "IEFA",
    "SCHF",
    "SPDW",
    "VPL",
    "IPAC",
    "AVDE",
    "FLCA",
    "VEA",
]


def _synthetic_prices(tickers, start=None, end=None) -> pd.DataFrame:
    """Correlated random-walk prices, stable across runs."""
    symbols = [str(t).upper() for t in tickers]
    index = pd.date_range("2014-01-01", periods=2_600, freq="B")
    rng = np.random.default_rng(2024)

    market = rng.normal(0.0004, 0.009, len(index))
    frame = {}
    for offset, symbol in enumerate(symbols):
        idiosyncratic = rng.normal(0.0001, 0.006, len(index))
        drift = 0.00005 * (offset % 4)
        returns = 0.8 * market + idiosyncratic + drift
        frame[symbol] = 100.0 * np.exp(np.cumsum(returns))

    prices = pd.DataFrame(frame, index=index)
    prices.index.name = "Date"
    return prices


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(data_module, "download_prices", _synthetic_prices)


@pytest.fixture
def app(offline):
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    assert not at.exception, at.exception
    return at


def _run_research(at: AppTest) -> AppTest:
    button = next(b for b in at.button if "Run research" in b.label)
    button.set_value(True).run()
    assert not at.exception, at.exception
    return at


def test_landing_page_waits_for_a_run(app):
    assert not app.tabs
    assert any("Workflow" in md.value for md in app.markdown)


def test_full_run_renders_every_section(app):
    at = _run_research(app)

    assert [tab.label for tab in at.tabs] == [
        "Portfolio Construction",
        "Multi-Period Analysis",
        "Bootstrap Simulation",
        "Testing Without Hindsight",
        "Diversification & Overlap",
        "Summary of Findings",
    ]
    assert not at.error
    assert len(at.dataframe) >= 15
    # Streamlit renamed this element type between releases; count either name.
    charts = len(at.get("vega_lite_chart")) + len(at.get("arrow_vega_lite_chart"))
    assert charts >= 15


def test_every_section_registers_an_export(app):
    at = _run_research(app)
    registry = at.session_state["section_exports"]
    assert len(registry) == 6
    assert [label for label, _, _ in registry] == [
        "Section 1 — Portfolio construction",
        "Section 2 — Multi-period analysis",
        "Section 3 — Bootstrap simulation",
        "Section 4 — Testing without hindsight",
        "Section 5 — Diversification & overlap",
        "Section 6 — Summary of Findings",
    ]


def test_download_buttons_are_offered_for_each_section(app):
    at = _run_research(app)
    labels = [d.label for d in at.get("download_button")]
    # Six sections plus the combined report, each with a PDF and a workbook.
    assert labels.count("Download PDF") == 7
    assert labels.count("Download Excel") == 7


@pytest.mark.parametrize("index", range(6))
def test_section_pdf_builds_from_the_live_run(app, index):
    at = _run_research(app)
    _label, blocks_factory, _tables_factory = at.session_state["section_exports"][index]

    from src import exporting as ex

    payload = ex.build_report_bytes(
        section_label="Smoke test",
        blocks=blocks_factory(),
        parameters=[("Universe", ", ".join(UNIVERSE))],
        subtitle="Smoke test",
    )
    assert payload.startswith(b"%PDF-")
    assert len(payload) > 20_000


@pytest.mark.parametrize("index", range(6))
def test_section_workbook_builds_from_the_live_run(app, index):
    at = _run_research(app)
    _label, _blocks_factory, tables_factory = at.session_state["section_exports"][index]

    from src import tables as table_utils

    payload = table_utils.to_excel_bytes(tables_factory())
    assert payload[:2] == b"PK"
    assert len(payload) > 4_000


def test_full_report_export_builds(app):
    at = _run_research(app)
    pdf_builder, xlsx_builder = at.session_state["full_report_exports"]
    pdf = pdf_builder()
    assert pdf.startswith(b"%PDF-")
    assert len(pdf) > 100_000
    assert xlsx_builder()[:2] == b"PK"


def test_rerun_after_a_widget_change_stays_clean(app):
    at = _run_research(app)

    toggle = next(t for t in at.toggle if t.key == "growth_log")
    toggle.set_value(True).run()
    assert not at.exception, at.exception
    assert not at.error

    radio = next(r for r in at.radio if r.key == "frontier_estimator")
    radio.set_value("Historical E(r)").run()
    assert not at.exception, at.exception
    assert not at.error


def test_too_few_assets_is_reported_not_raised(app):
    universe = next(
        m for m in app.multiselect if m.label == "Curated research universe"
    )
    universe.set_value(["TMFC"])
    _run_research(app)
    assert any("at least two" in e.value.lower() for e in app.error)


def test_reference_weights_must_sum_to_one_hundred(app):
    weight = next(w for w in app.number_input if w.key == "ref_weight_TMFC")
    weight.set_value(10.0)
    _run_research(app)
    assert any("100%" in e.value for e in app.error)


def test_summary_of_findings_renders_scorecard(app):
    at = _run_research(app)
    text = " ".join(md.value for md in at.markdown)
    assert "Summary of Findings" in text
    assert "Decision lens" not in text


def test_unconvergeable_method_is_reported_not_silently_dropped(offline):
    """A 25% risk cap cannot hold on three ETFs (equal weight already puts 33% of
    the risk in each), so Diversified Maximum Sharpe has no feasible portfolio.
    The report has to say so instead of letting the row vanish."""
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    at.multiselect(key="control_selected_curated").set_value(["TMFC", "VWO", "ACWI"])
    at.slider(key="control_diversified_max_risk_share").set_value(25)
    at.slider(key="control_diversified_min_effective_assets").set_value(1.5)
    _run_research(at)

    warnings = [w.value for w in at.warning]
    assert any(
        "Did not converge" in text and "Diversified Maximum Sharpe" in text
        for text in warnings
    ), warnings
    # The remaining sections still render without the missing method.
    assert len(at.tabs) == 6
    assert not at.exception


def test_unfinished_month_is_excluded_and_disclosed(monkeypatch):
    def truncated(tickers, start=None, end=None):
        return _synthetic_prices(tickers).loc[:"2023-12-14"]

    monkeypatch.setattr(data_module, "download_prices", truncated)
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    # market data is memoised per (tickers, start, end, frequency) for the life of
    # the process; a distinct end date keeps this test off other tests' entries.
    at.date_input(key="control_end_date").set_value(date(2023, 12, 14))
    at = _run_research(at)

    captions = " ".join(c.value for c in at.caption)
    assert "Price data runs to 14 Dec 2023" in captions
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Sample end"] == "Nov 2023"


def test_masthead_carries_the_product_name(app):
    html = " ".join(md.value for md in app.markdown)
    assert "Portfolio Allocation Testing" in html
    assert "Sleeve" not in html

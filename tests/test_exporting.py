"""The PDF export must build from plain data, with no Streamlit runtime.

That is what lets the download button defer its work until the reader clicks,
so these tests exercise the block model and every figure renderer directly.
"""

import numpy as np
import pandas as pd
import pytest

from src import exporting as ex
from src import theme

TICKERS = ["AAA", "BBB", "CCC"]


def pdf_pages(payload: bytes) -> int:
    """Page count straight from the PDF trailer, without a parser dependency."""
    assert payload.startswith(b"%PDF-")
    assert payload.rstrip().endswith(b"%%EOF")
    return payload.count(b"/Type /Page\n") or payload.count(b"/Type /Page")


@pytest.fixture(scope="module")
def frame():
    return pd.DataFrame(
        {
            "CAGR": [0.201, 0.09, 0.137],
            "Volatility": [0.192, 0.162, 0.158],
            "Sharpe": [1.05, 0.55, 0.87],
            "Max Drawdown": [-0.309, -0.319, -0.257],
        },
        index=pd.Index(TICKERS, name="Ticker"),
    )


@pytest.fixture(scope="module")
def tidy():
    index = pd.date_range("2019-01-31", periods=48, freq="ME")
    rng = np.random.default_rng(4)
    wide = pd.DataFrame(rng.normal(0.007, 0.04, (48, 3)), index=index, columns=TICKERS)
    growth = (1 + wide).cumprod()
    return (
        growth.rename_axis("Date")
        .reset_index()
        .melt(id_vars="Date", var_name="Series", value_name="Growth")
    )


def test_fonts_register_a_unicode_capable_family():
    regular, bold, italic = ex.register_fonts()
    assert regular and bold and italic
    from reportlab.pdfbase import pdfmetrics

    # The report uses ≥, × and · in thresholds and captions.
    width = pdfmetrics.stringWidth("≥ 0.980× reference · 5%", regular, 8)
    assert width > 0


def test_minimal_report_builds(frame):
    payload = ex.build_report_bytes(
        section_label="Test section",
        blocks=[ex.Heading("Heading"), ex.Table(frame)],
        parameters=[("Universe", "AAA, BBB, CCC")],
        subtitle="Unit test",
    )
    assert pdf_pages(payload) >= 2  # cover plus at least one body page


def test_report_metadata_reaches_the_document(frame):
    payload = ex.build_report_bytes(
        section_label="Robustness analysis",
        blocks=[ex.Table(frame)],
        parameters=[("CVaR confidence", "95%")],
        subtitle="Unit test",
    )
    assert b"Robustness analysis" in payload or b"Robustness" in payload


def test_every_block_type_renders(frame, tidy):
    blocks = [
        ex.Heading("Level one", 1),
        ex.Heading("Level two", 2),
        ex.Heading("Level three", 3),
        ex.Text("Body copy with <b>markup</b>."),
        ex.Caption("A caption."),
        ex.Bullets(["First item", "Second item"]),
        ex.Callout("A callout.", tone="brass"),
        ex.StatusBanner("Engine status", "NO TRADE", "Detail line", "positive"),
        ex.KPIs([("Label", "Value", "Sub")] * 4),
        ex.Table(frame, note="A note.", index_label="Ticker"),
        ex.Spacer(),
        ex.PageBreak(),
        ex.Figure(
            "lines",
            {
                "data": tidy,
                "x": "Date",
                "y": "Growth",
                "series": "Series",
                "value_format": "plain",
            },
            title="A figure",
            note="A figure note.",
            height_in=2.2,
        ),
    ]
    payload = ex.build_report_bytes(
        section_label="All blocks",
        blocks=blocks,
        parameters=[("Sample", "2019–2024")],
        subtitle="Unit test",
        contents=("One", "Two"),
    )
    assert pdf_pages(payload) >= 3


def test_table_block_formats_by_column_name(frame):
    blocks = ex._table_flowable(ex.Table(frame), 400.0)
    flat = " ".join(
        cell.text
        for row in blocks[0]._cellvalues
        for cell in row
        if hasattr(cell, "text")
    )
    assert "20.1%" in flat
    assert "1.05" in flat


def test_table_block_honours_a_default_format():
    weights = pd.DataFrame(
        [[0.25, 0.75]], index=["Equal Weight"], columns=["AAA", "BBB"]
    )
    blocks = ex._table_flowable(ex.Table(weights, default_format="{:.1%}"), 400.0)
    flat = " ".join(
        cell.text
        for row in blocks[0]._cellvalues
        for cell in row
        if hasattr(cell, "text")
    )
    assert "25.0%" in flat and "75.0%" in flat


def test_table_block_truncates_long_frames():
    long_frame = pd.DataFrame({"CAGR": np.linspace(0, 1, 100)})
    blocks = ex._table_flowable(ex.Table(long_frame, max_rows=10), 400.0)
    assert len(blocks[0]._cellvalues) == 11  # header plus ten rows
    assert any("100 rows" in str(getattr(b, "text", "")) for b in blocks[1:])


def test_table_block_handles_an_empty_frame():
    blocks = ex._table_flowable(ex.Table(pd.DataFrame()), 400.0)
    assert "No rows" in blocks[0].text


def test_table_widths_never_overflow_the_frame():
    wide = pd.DataFrame(
        np.random.default_rng(0).normal(size=(3, 18)),
        columns=[f"Column number {i}" for i in range(18)],
    )
    table = ex._table_flowable(ex.Table(wide), ex.CONTENT_WIDTH)[0]
    assert sum(table._colWidths) <= ex.CONTENT_WIDTH + 1e-6


def _figure_specs(tidy, frame) -> dict[str, ex.Figure]:
    categories = pd.DataFrame(
        {
            "Ticker": TICKERS,
            "Value": [0.5, 0.3, 0.2],
            "Upper": [0.6, 0.4, 0.3],
            "Lower": [0.4, 0.2, 0.1],
            "Outer Low": [0.3, 0.1, 0.05],
            "Outer High": [0.7, 0.5, 0.4],
            "Extra": [0.45, 0.25, 0.15],
        }
    )
    stacked = pd.DataFrame(
        {
            "Method": ["A", "A", "A", "B", "B", "B"],
            "Ticker": TICKERS * 2,
            "Weight": [0.5, 0.3, 0.2, 0.2, 0.3, 0.5],
        }
    )
    grouped = pd.DataFrame(
        {
            "Ticker": TICKERS * 2,
            "Metric": ["One"] * 3 + ["Two"] * 3,
            "Value": [0.2, 0.4, 0.6, 0.3, 0.5, 0.7],
        }
    )
    scatter = frame.reset_index()
    frontier = pd.DataFrame(
        {
            "Volatility": np.linspace(0.05, 0.2, 20),
            "Expected Return": np.linspace(0.03, 0.12, 20),
        }
    )
    named = pd.DataFrame(
        {
            "Portfolio": ["Reference Portfolio", "Robust Consensus"],
            "Volatility": [0.14, 0.09],
            "Expected Return": [0.08, 0.06],
        }
    )
    fan = pd.DataFrame(
        {
            "Month": np.arange(1, 61),
            "5th": np.linspace(100_000, 120_000, 60),
            "25th": np.linspace(100_000, 150_000, 60),
            "Median": np.linspace(100_000, 180_000, 60),
            "75th": np.linspace(100_000, 220_000, 60),
            "95th": np.linspace(100_000, 300_000, 60),
        }
    )
    hist = pd.DataFrame(
        {
            "Series": ["Reference Portfolio"] * 20 + ["Robust Consensus"] * 20,
            "Value": list(np.linspace(50_000, 300_000, 20)) * 2,
            "Share": list(np.random.default_rng(1).random(40) / 20),
        }
    )
    area = pd.DataFrame(
        {
            "Date": list(pd.date_range("2020-01-31", periods=6, freq="ME")) * 3,
            "Ticker": sum(([t] * 6 for t in TICKERS), []),
            "Weight": list(np.random.default_rng(2).random(18)),
        }
    )

    return {
        "lines": ex.Figure(
            "lines",
            {
                "data": tidy,
                "x": "Date",
                "y": "Growth",
                "series": "Series",
                "value_format": "plain",
            },
        ),
        "lines_filled": ex.Figure(
            "lines",
            {
                "data": tidy.assign(Growth=lambda d: -d["Growth"] / 10),
                "x": "Date",
                "y": "Growth",
                "series": "Series",
                "fill": True,
            },
        ),
        "stacked_area": ex.Figure(
            "stacked_area",
            {"data": area, "x": "Date", "value": "Weight", "series": "Ticker"},
        ),
        "stacked_barh": ex.Figure(
            "stacked_barh",
            {
                "data": stacked,
                "cat": "Method",
                "value": "Weight",
                "series": "Ticker",
                "order": ["A", "B"],
            },
        ),
        "barh": ex.Figure(
            "barh", {"data": categories, "cat": "Ticker", "value": "Value"}
        ),
        "barh_threshold": ex.Figure(
            "barh",
            {
                "data": categories,
                "cat": "Ticker",
                "value": "Value",
                "threshold": 0.4,
                "marker": "Upper",
            },
        ),
        "barh_diverging": ex.Figure(
            "barh",
            {
                "data": categories.assign(Value=[0.1, -0.2, 0.05]),
                "cat": "Ticker",
                "value": "Value",
                "diverging": True,
                "annotate_format": "+.1%",
            },
        ),
        "barh_labelled": ex.Figure(
            "barh",
            {
                "data": categories.assign(Label=["a", "b", "c"]),
                "cat": "Ticker",
                "value": "Value",
                "annotate_column": "Label",
            },
        ),
        "grouped_bars": ex.Figure(
            "grouped_bars",
            {
                "data": grouped,
                "cat": "Ticker",
                "value": "Value",
                "series": "Metric",
                "y_max": 1.0,
                "rotation": 20,
            },
        ),
        "range_dot": ex.Figure(
            "range_dot",
            {
                "data": categories,
                "cat": "Ticker",
                "lo": "Lower",
                "hi": "Upper",
                "mid": "Value",
                "outer_lo": "Outer Low",
                "outer_hi": "Outer High",
                "point_a": "Extra",
                "point_b": "Value",
            },
        ),
        "scatter": ex.Figure(
            "scatter",
            {
                "data": scatter,
                "x": "Volatility",
                "y": "CAGR",
                "label": "Ticker",
                "size": "Sharpe",
            },
        ),
        "frontier": ex.Figure(
            "frontier",
            {
                "cloud": frontier,
                "frontier": frontier,
                "named": named,
                "x": "Volatility",
                "y": "Expected Return",
            },
        ),
        "multi_line": ex.Figure(
            "multi_line",
            {
                "data": pd.concat(
                    [
                        frontier.assign(Estimator="Raw historical"),
                        frontier.assign(Estimator="Market prior"),
                    ]
                ),
                "x": "Volatility",
                "y": "Expected Return",
                "series": "Estimator",
                "colors": {"Raw historical": theme.BRASS, "Market prior": theme.NAVY},
                "dashes": {"Raw historical": (0, (5, 3)), "Market prior": "-"},
            },
        ),
        "heatmap": ex.Figure(
            "heatmap",
            {
                "matrix": pd.DataFrame(
                    [[1.0, 0.3, -0.2], [0.3, 1.0, 0.1], [-0.2, 0.1, 1.0]],
                    index=TICKERS,
                    columns=TICKERS,
                ),
                "vmin": -1.0,
                "vmax": 1.0,
            },
        ),
        "fan": ex.Figure("fan", {"data": fan, "reference": 100_000.0}),
        "distribution": ex.Figure(
            "distribution", {"data": hist, "reference": 100_000.0}
        ),
    }


FIGURE_CASES = (
    "barh",
    "barh_diverging",
    "barh_labelled",
    "barh_threshold",
    "distribution",
    "fan",
    "frontier",
    "grouped_bars",
    "heatmap",
    "lines",
    "lines_filled",
    "multi_line",
    "range_dot",
    "scatter",
    "stacked_area",
    "stacked_barh",
)


@pytest.mark.parametrize("kind", FIGURE_CASES)
def test_figure_renderer_produces_a_png(kind, tidy, frame):
    spec = _figure_specs(tidy, frame)[kind]
    png = ex.render_figure(spec)
    assert png is not None, f"{kind} produced no image"
    assert png.startswith(b"\x89PNG"), f"{kind} is not a PNG"
    assert len(png) > 4_000, f"{kind} looks blank"


def test_every_declared_renderer_is_covered(tidy, frame):
    covered = {spec.kind for spec in _figure_specs(tidy, frame).values()}
    assert covered == set(ex._FIGURE_RENDERERS)


def test_unknown_figure_kind_is_skipped_not_raised():
    assert ex.render_figure(ex.Figure("no-such-kind", {})) is None


def test_broken_figure_payload_does_not_break_the_export():
    """One unplottable panel must not take down a whole section export."""
    assert ex.render_figure(ex.Figure("lines", {"data": pd.DataFrame()})) is None

    payload = ex.build_report_bytes(
        section_label="Resilience",
        blocks=[ex.Figure("lines", {"data": pd.DataFrame()}), ex.Heading("Still here")],
        parameters=[],
        subtitle="Unit test",
    )
    assert pdf_pages(payload) >= 2


def test_auto_decimals_keeps_tick_labels_distinct():
    assert ex._auto_decimals(np.linspace(0.0, 0.5, 10)) == 0
    assert ex._auto_decimals(np.linspace(0.118, 0.130, 10)) == 1
    assert ex._auto_decimals(np.linspace(0.1180, 0.1185, 10)) == 2


def test_file_stamp_is_filesystem_safe():
    from datetime import datetime

    name = ex.file_stamp(
        "Section 1 — Portfolio construction", datetime(2026, 9, 13, 2, 5)
    )
    assert (
        name
        == "portfolio-allocation-testing-section-1-portfolio-construction-20260913-0205"
    )
    assert not set(name) & set(' /\\:*?"<>|')

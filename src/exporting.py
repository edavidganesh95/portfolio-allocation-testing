"""Print-ready PDF and workbook export.

Each research section — and the full report — can be exported as a paginated
A4 landscape document: running header, page numbers, a parameter block that
records exactly how the numbers were produced, and a standing research
disclaimer.

The document is described declaratively. A section hands over a list of blocks
(:class:`Heading`, :class:`Table`, :class:`Figure`, …) holding plain data, and
this module lays them out with ReportLab and draws the figures with matplotlib
using the same :mod:`src.theme` tokens as the screen. Charts are therefore
recognisably the same charts on paper, and no Streamlit call is needed at build
time — which is what lets the download button render lazily, on click.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageTemplate,
    Paragraph,
    TableStyle,
)
from reportlab.platypus import (
    PageBreak as RLPageBreak,
)
from reportlab.platypus import (
    Spacer as RLSpacer,
)
from reportlab.platypus import (
    Table as RLTable,
)

from . import tables as table_utils
from . import theme

PAGE_SIZE = landscape(A4)
PAGE_WIDTH, PAGE_HEIGHT = PAGE_SIZE
MARGIN_X = 1.7 * cm
MARGIN_TOP = 2.05 * cm
MARGIN_BOTTOM = 1.4 * cm
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN_X

DISCLAIMER = (
    "Personal research project for educational purposes only — not financial advice. "
    "Historical, modelled and simulated results do not guarantee future outcomes."
)
FOOTER_NOTE = (
    "Personal research project · For educational purposes only · Not financial advice"
)


# ----------------------------------------------------------------------
# Fonts
# ----------------------------------------------------------------------

FONT = "Ledger"
FONT_BOLD = "Ledger-Bold"
FONT_ITALIC = "Ledger-Italic"


@lru_cache(maxsize=1)
def register_fonts() -> tuple[str, str, str]:
    """Register DejaVu Sans with ReportLab.

    ReportLab's built-in Type1 faces cannot encode the typographic characters
    this report uses (≥, ×, ·, en dashes). DejaVu ships with matplotlib, which
    is already a dependency, so the font is guaranteed present and the PDF
    needs no system font lookup.
    """
    try:
        from matplotlib import get_data_path

        ttf = f"{get_data_path()}/fonts/ttf"
        pdfmetrics.registerFont(TTFont(FONT, f"{ttf}/DejaVuSans.ttf"))
        pdfmetrics.registerFont(TTFont(FONT_BOLD, f"{ttf}/DejaVuSans-Bold.ttf"))
        pdfmetrics.registerFont(TTFont(FONT_ITALIC, f"{ttf}/DejaVuSans-Oblique.ttf"))
        pdfmetrics.registerFontFamily(
            FONT, normal=FONT, bold=FONT_BOLD, italic=FONT_ITALIC
        )
        return FONT, FONT_BOLD, FONT_ITALIC
    except Exception:  # pragma: no cover - fall back to core fonts
        return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


def _color(token: str) -> colors.Color:
    return colors.HexColor(token)


# ----------------------------------------------------------------------
# Block model
# ----------------------------------------------------------------------


@dataclass
class Heading:
    text: str
    level: int = 2


@dataclass
class Text:
    body: str
    measure: float = 0.62


@dataclass
class Caption:
    body: str


@dataclass
class Bullets:
    items: Sequence[str]


@dataclass
class Callout:
    body: str
    tone: str = "navy"


@dataclass
class StatusBanner:
    label: str
    value: str
    detail: str = ""
    tone: str = "navy"


@dataclass
class KPIs:
    items: Sequence[tuple[str, str, str]]


@dataclass
class Table:
    frame: pd.DataFrame
    note: str | None = None
    index_label: str | None = None
    formats: dict[str, str] | None = None
    default_format: str | None = None
    max_rows: int | None = 40
    font_size: float | None = None
    highlight: Sequence[str] = ()
    full_width: bool = False


@dataclass
class Figure:
    kind: str
    payload: dict[str, Any]
    title: str | None = None
    note: str | None = None
    height_in: float = 3.1


@dataclass
class Spacer:
    height: float = 0.35 * cm


@dataclass
class PageBreak:
    pass


Block = (
    Heading
    | Text
    | Caption
    | Bullets
    | Callout
    | StatusBanner
    | KPIs
    | Table
    | Figure
    | Spacer
    | PageBreak
)


@dataclass
class ReportMeta:
    title: str = "Portfolio Allocation Testing"
    subtitle: str = "Strategic ETF allocation research"
    section_label: str = ""
    generated_at: datetime = field(default_factory=datetime.now)
    parameters: Sequence[tuple[str, str]] = ()
    contents: Sequence[str] = ()
    footnote: str = DISCLAIMER


@dataclass
class Report:
    meta: ReportMeta
    blocks: Sequence[Block]


# ----------------------------------------------------------------------
# Paragraph styles
# ----------------------------------------------------------------------


@lru_cache(maxsize=1)
def _styles() -> dict[str, ParagraphStyle]:
    regular, bold, italic = register_fonts()
    base = ParagraphStyle(
        "body",
        fontName=regular,
        fontSize=8.6,
        leading=12.6,
        textColor=_color(theme.INK_SOFT),
        alignment=TA_LEFT,
        spaceAfter=4,
    )
    return {
        "cover_title": ParagraphStyle(
            "cover_title",
            parent=base,
            fontName=bold,
            fontSize=27,
            leading=31,
            textColor=_color(theme.NAVY_DEEP),
            spaceAfter=6,
        ),
        "cover_subtitle": ParagraphStyle(
            "cover_subtitle",
            parent=base,
            fontSize=12,
            leading=17,
            textColor=_color(theme.MUTED),
            spaceAfter=16,
        ),
        "kicker": ParagraphStyle(
            "kicker",
            parent=base,
            fontName=bold,
            fontSize=7.4,
            leading=10,
            textColor=_color(theme.BRASS),
            spaceAfter=3,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base,
            fontName=bold,
            fontSize=16,
            leading=20,
            textColor=_color(theme.NAVY_DEEP),
            spaceBefore=4,
            spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base,
            fontName=bold,
            fontSize=11.6,
            leading=15,
            textColor=_color(theme.INK),
            spaceBefore=10,
            spaceAfter=5,
        ),
        "h3": ParagraphStyle(
            "h3",
            parent=base,
            fontName=bold,
            fontSize=9.4,
            leading=13,
            textColor=_color(theme.NAVY),
            spaceBefore=8,
            spaceAfter=3,
        ),
        "body": base,
        "bullet": ParagraphStyle(
            "bullet",
            parent=base,
            leftIndent=11,
            bulletIndent=2,
            spaceAfter=2.5,
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base,
            fontName=italic,
            fontSize=7.4,
            leading=10.4,
            textColor=_color(theme.MUTED),
            spaceBefore=2,
            spaceAfter=6,
        ),
        "figure_title": ParagraphStyle(
            "figure_title",
            parent=base,
            fontName=bold,
            fontSize=8.8,
            leading=12,
            textColor=_color(theme.INK),
            spaceBefore=6,
            spaceAfter=3,
        ),
        "callout": ParagraphStyle(
            "callout",
            parent=base,
            fontSize=8.6,
            leading=12.8,
            textColor=_color(theme.INK),
        ),
        "th": ParagraphStyle(
            "th",
            parent=base,
            fontName=bold,
            fontSize=7.2,
            leading=9.2,
            textColor=_color("#FFFFFF"),
            spaceAfter=0,
        ),
        "td": ParagraphStyle(
            "td",
            parent=base,
            fontSize=7.4,
            leading=9.6,
            textColor=_color(theme.INK),
            spaceAfter=0,
        ),
        "kpi_label": ParagraphStyle(
            "kpi_label",
            parent=base,
            fontName=bold,
            fontSize=6.6,
            leading=8.6,
            textColor=_color(theme.MUTED),
            spaceAfter=1,
        ),
        "kpi_value": ParagraphStyle(
            "kpi_value",
            parent=base,
            fontName=bold,
            fontSize=12.4,
            leading=15,
            textColor=_color(theme.NAVY_DEEP),
            spaceAfter=1,
        ),
        "kpi_sub": ParagraphStyle(
            "kpi_sub",
            parent=base,
            fontSize=6.8,
            leading=9,
            textColor=_color(theme.MUTED),
            spaceAfter=0,
        ),
        "banner_label": ParagraphStyle(
            "banner_label",
            parent=base,
            fontName=bold,
            fontSize=6.8,
            leading=9,
            textColor=_color("#FFFFFF"),
            spaceAfter=1,
        ),
        "banner_value": ParagraphStyle(
            "banner_value",
            parent=base,
            fontName=bold,
            fontSize=15,
            leading=18,
            textColor=_color("#FFFFFF"),
            spaceAfter=1,
        ),
        "banner_detail": ParagraphStyle(
            "banner_detail",
            parent=base,
            fontSize=7.6,
            leading=10.4,
            textColor=_color("#FFFFFF"),
            spaceAfter=0,
        ),
    }


def _escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class HRule(Flowable):
    """Thin horizontal rule used to close a heading."""

    def __init__(
        self,
        width: float,
        thickness: float = 0.7,
        color: str = theme.BRASS,
        space_after: float = 4.0,
    ):
        super().__init__()
        self.width = width
        self.thickness = thickness
        self.color = color
        self.space_after = space_after
        self.height = thickness + space_after

    def draw(self) -> None:
        self.canv.setStrokeColor(_color(self.color))
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.space_after, self.width, self.space_after)


# ----------------------------------------------------------------------
# Matplotlib figure renderers
# ----------------------------------------------------------------------


def _new_figure(height_in: float, width_in: float | None = None):
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    theme.apply_matplotlib_style()
    width = width_in if width_in else CONTENT_WIDTH / 72.0
    fig, ax = plt.subplots(figsize=(width, height_in), layout="constrained")
    return fig, ax


def _finish(fig) -> bytes:
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches=None)
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def _auto_decimals(values) -> int:
    """Tick decimals that keep neighbouring percent labels distinguishable."""
    array = np.asarray(values, dtype=float).ravel()
    array = array[np.isfinite(array)]
    if array.size < 2:
        return 1
    span = float(array.max() - array.min())
    if span >= 0.08:
        return 0
    if span >= 0.008:
        return 1
    return 2


def _percent_axis(ax, which: str = "y", decimals: int = 0) -> None:
    from matplotlib.ticker import PercentFormatter

    formatter = PercentFormatter(xmax=1.0, decimals=decimals)
    if which == "y":
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


def _money_axis(ax, which: str = "y") -> None:
    from matplotlib.ticker import FuncFormatter

    def fmt(value, _pos):
        if abs(value) >= 1_000_000:
            return f"${value / 1_000_000:,.1f}M"
        if abs(value) >= 1_000:
            return f"${value / 1_000:,.0f}k"
        return f"${value:,.0f}"

    formatter = FuncFormatter(fmt)
    if which == "y":
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


def _date_axis(ax, values) -> bool:
    """Give a datetime x-axis readable, non-repeating tick labels."""
    if not pd.api.types.is_datetime64_any_dtype(pd.Series(values)):
        return False
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

    locator = AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator, show_offset=False))
    return True


def _legend_below(ax, handles=None, labels=None, columns: int | None = None) -> None:
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if not labels:
        return
    ncol = columns or min(6, max(2, len(labels)))
    offset = -0.26 if ax.get_xlabel() else -0.15
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, offset),
        ncol=ncol,
        frameon=False,
    )


def _series_colors(names: Sequence[str], payload: dict) -> list[str]:
    explicit = payload.get("colors")
    if isinstance(explicit, dict):
        return [
            explicit.get(name, theme.CATEGORICAL[i % len(theme.CATEGORICAL)])
            for i, name in enumerate(names)
        ]
    if isinstance(explicit, (list, tuple)) and len(explicit) >= len(names):
        return list(explicit[: len(names)])
    kind = payload.get("color_kind", "asset")
    if kind == "portfolio":
        return theme.portfolio_range(list(names))
    return [theme.CATEGORICAL[i % len(theme.CATEGORICAL)] for i in range(len(names))]


def _axis_value_setup(ax, payload: dict, which: str) -> None:
    value_format = payload.get("value_format", "percent")
    if value_format == "percent":
        _percent_axis(ax, which, payload.get("decimals", 0))
    elif value_format == "money":
        _money_axis(ax, which)


def _render_lines(payload: dict, height_in: float) -> bytes:
    fig, ax = _new_figure(height_in)
    frame = payload["data"]
    x, y, series = payload["x"], payload["y"], payload.get("series")
    fill = payload.get("fill", False)

    names = (
        list(dict.fromkeys(frame[series].tolist()))
        if series
        else [payload.get("label", y)]
    )
    palette = _series_colors(names, payload)

    for name, color in zip(names, palette, strict=True):
        part = frame[frame[series] == name] if series else frame
        part = part.sort_values(x)
        ax.plot(part[x], part[y], color=color, label=str(name), linewidth=1.4)
        if fill:
            ax.fill_between(part[x], part[y], 0.0, color=color, alpha=0.16, linewidth=0)

    if payload.get("reference") is not None:
        ax.axhline(
            float(payload["reference"]),
            color=theme.MUTED,
            linestyle=(0, (4, 3)),
            linewidth=0.9,
        )
    _axis_value_setup(ax, payload, "y")
    ax.set_ylabel(payload.get("y_title", ""))
    ax.set_xlabel(payload.get("x_title", ""))
    if payload.get("log", False):
        ax.set_yscale("log")
    if payload.get("date_axis", True):
        _date_axis(ax, frame[x])
    if len(names) > 1:
        _legend_below(ax)
    return _finish(fig)


def _render_stacked_area(payload: dict, height_in: float) -> bytes:
    fig, ax = _new_figure(height_in)
    frame = payload["data"]
    x, value, series = payload["x"], payload["value"], payload["series"]

    wide = (
        frame.pivot_table(index=x, columns=series, values=value, aggfunc="mean")
        .fillna(0.0)
        .sort_index()
    )
    totals = wide.sum(axis=1).replace(0.0, np.nan)
    wide = wide.div(totals, axis=0).fillna(0.0)
    names = sorted(wide.columns.tolist())
    palette = _series_colors(names, payload)

    ax.stackplot(
        wide.index,
        [wide[name].to_numpy() for name in names],
        labels=[str(n) for n in names],
        colors=palette,
        linewidth=0,
    )
    ax.set_ylim(0, 1)
    ax.margins(x=0)
    _percent_axis(ax, "y")
    ax.set_ylabel(payload.get("y_title", ""))
    ax.set_xlabel(payload.get("x_title", ""))
    # Grid lines would sit on top of a fully filled area, so drop them.
    ax.grid(visible=False)
    if payload.get("x_percent", False):
        _percent_axis(ax, "x", payload.get("x_decimals", _auto_decimals(wide.index)))
    elif payload.get("date_axis", True):
        _date_axis(ax, wide.index)
    _legend_below(ax, columns=min(6, len(names)))
    return _finish(fig)


def _render_stacked_barh(payload: dict, height_in: float) -> bytes:
    fig, ax = _new_figure(height_in)
    frame = payload["data"]
    cat, value, series = payload["cat"], payload["value"], payload["series"]
    order = payload.get("order") or list(dict.fromkeys(frame[cat].tolist()))

    wide = (
        frame.pivot_table(index=cat, columns=series, values=value, aggfunc="sum")
        .reindex(order)
        .fillna(0.0)
    )
    totals = wide.sum(axis=1).replace(0.0, np.nan)
    wide = wide.div(totals, axis=0).fillna(0.0)
    names = sorted(wide.columns.tolist())
    palette = _series_colors(names, payload)

    positions = np.arange(len(wide))[::-1]
    left = np.zeros(len(wide))
    for name, color in zip(names, palette, strict=True):
        widths = wide[name].to_numpy(float)
        ax.barh(
            positions,
            widths,
            left=left,
            height=0.68,
            color=color,
            label=str(name),
        )
        for pos, width, start in zip(positions, widths, left, strict=True):
            if width >= 0.055:
                ax.text(
                    start + width / 2,
                    pos,
                    f"{width:.0%}",
                    ha="center",
                    va="center",
                    fontsize=6.0,
                    color=theme.readable_text(color),
                )
        left += widths

    ax.set_yticks(positions)
    ax.set_yticklabels([str(v) for v in wide.index], fontsize=7.2)
    ax.set_xlim(0, 1)
    _percent_axis(ax, "x")
    ax.set_xlabel(payload.get("x_title", ""))
    ax.grid(visible=False)
    _legend_below(ax, columns=min(6, len(names)))
    return _finish(fig)


def _render_barh(payload: dict, height_in: float) -> bytes:
    fig, ax = _new_figure(height_in)
    frame = payload["data"].copy()
    cat, value = payload["cat"], payload["value"]
    if payload.get("sort", True):
        frame = frame.sort_values(value, ascending=True)

    labels = frame[cat].astype(str).tolist()
    values = frame[value].to_numpy(float)
    positions = np.arange(len(frame))

    color_spec = payload.get("color")
    if payload.get("diverging", False):
        bar_colors = [theme.POSITIVE if v > 0 else theme.NEGATIVE for v in values]
    elif payload.get("threshold") is not None:
        limit = float(payload["threshold"])
        bar_colors = [theme.POSITIVE if v >= limit else theme.NEGATIVE for v in values]
    elif isinstance(color_spec, dict):
        bar_colors = [color_spec.get(label, theme.NAVY) for label in labels]
    elif isinstance(color_spec, str):
        bar_colors = [color_spec] * len(values)
    else:
        bar_colors = _series_colors(labels, payload)

    ax.barh(positions, values, height=0.62, color=bar_colors)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=7.2)
    _axis_value_setup(ax, payload, "x")
    ax.set_xlabel(payload.get("x_title", ""))
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)

    if payload.get("threshold") is not None:
        ax.axvline(
            float(payload["threshold"]),
            color=theme.MUTED,
            linestyle=(0, (4, 3)),
            linewidth=0.9,
        )
    if payload.get("marker") is not None:
        ax.scatter(
            frame[payload["marker"]].to_numpy(float),
            positions,
            marker="|",
            s=110,
            linewidths=1.6,
            color=theme.INK,
            alpha=0.6,
            zorder=5,
        )
    if payload.get("annotate", True):
        spec = payload.get("annotate_format", ".1%")
        label_column = payload.get("annotate_column")
        texts = (
            frame[label_column].astype(str).tolist()
            if label_column
            else [format(value, spec) for value in values]
        )
        span = float(np.nanmax(np.abs(values))) if len(values) else 0.0
        pad = span * 0.015 if span else 0.0
        for pos, val, text in zip(positions, values, texts, strict=True):
            ax.text(
                val + (pad if val >= 0 else -pad),
                pos,
                text,
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=6.4,
                color=theme.INK_SOFT,
            )
        ax.margins(x=0.12)
    if payload.get("diverging", False):
        ax.axvline(0.0, color=theme.BORDER_STRONG, linewidth=0.8)
    return _finish(fig)


def _render_grouped_bars(payload: dict, height_in: float) -> bytes:
    fig, ax = _new_figure(height_in)
    frame = payload["data"]
    cat, value, series = payload["cat"], payload["value"], payload["series"]

    order = payload.get("order") or list(dict.fromkeys(frame[cat].tolist()))
    names = payload.get("series_order") or list(dict.fromkeys(frame[series].tolist()))
    wide = (
        frame.pivot_table(index=cat, columns=series, values=value, aggfunc="mean")
        .reindex(order)
        .reindex(columns=names)
    )
    palette = payload.get("colors")
    if not isinstance(palette, (list, tuple)):
        palette = [
            theme.ramp(theme.SEQUENTIAL, 0.35 + 0.62 * i / max(len(names) - 1, 1))
            for i in range(len(names))
        ]

    positions = np.arange(len(wide))
    width = 0.8 / max(len(names), 1)
    for i, (name, color) in enumerate(zip(names, palette, strict=True)):
        ax.bar(
            positions + i * width - 0.4 + width / 2,
            wide[name].to_numpy(float),
            width=width * 0.92,
            color=color,
            label=str(name),
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [str(v) for v in wide.index],
        fontsize=6.8,
        rotation=payload.get("rotation", 0),
        ha="center" if not payload.get("rotation") else "right",
    )
    _axis_value_setup(ax, payload, "y")
    if payload.get("y_max") is not None:
        ax.set_ylim(0, float(payload["y_max"]))
    ax.set_ylabel(payload.get("y_title", ""))
    _legend_below(ax, columns=min(4, len(names)))
    return _finish(fig)


def _render_range_dot(payload: dict, height_in: float) -> bytes:
    fig, ax = _new_figure(height_in)
    frame = payload["data"].copy()
    cat = payload["cat"]
    lo, hi, mid = payload["lo"], payload["hi"], payload["mid"]
    if payload.get("sort", True):
        frame = frame.sort_values(mid, ascending=True)

    labels = frame[cat].astype(str).tolist()
    positions = np.arange(len(frame))
    palette = payload.get("color")
    if isinstance(palette, dict):
        bar_colors = [palette.get(label, theme.NAVY) for label in labels]
    elif isinstance(palette, str):
        bar_colors = [palette] * len(labels)
    else:
        bar_colors = _series_colors(labels, payload)

    outer_lo, outer_hi = payload.get("outer_lo"), payload.get("outer_hi")
    if outer_lo and outer_hi:
        ax.hlines(
            positions,
            frame[outer_lo].to_numpy(float),
            frame[outer_hi].to_numpy(float),
            color=theme.BORDER_STRONG,
            linewidth=1.4,
            zorder=1,
        )
    ax.barh(
        positions,
        frame[hi].to_numpy(float) - frame[lo].to_numpy(float),
        left=frame[lo].to_numpy(float),
        height=0.34,
        color=bar_colors,
        alpha=payload.get("band_alpha", 0.85),
        zorder=2,
    )
    ax.scatter(
        frame[mid].to_numpy(float),
        positions,
        marker="|",
        s=190,
        linewidths=2.0,
        color=theme.INK,
        zorder=4,
    )
    for extra, marker, color in (
        (payload.get("point_a"), "o", theme.BRASS),
        (payload.get("point_b"), "D", theme.NAVY),
    ):
        if extra:
            ax.scatter(
                frame[extra].to_numpy(float),
                positions,
                marker=marker,
                s=42,
                color=color,
                edgecolors=theme.SURFACE,
                linewidths=0.7,
                zorder=5,
            )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=7.2)
    _axis_value_setup(ax, payload, "x")
    ax.set_xlabel(payload.get("x_title", ""))
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.06)
    return _finish(fig)


def _render_scatter(payload: dict, height_in: float) -> bytes:
    fig, ax = _new_figure(height_in)
    frame = payload["data"]
    x, y, label = payload["x"], payload["y"], payload.get("label")
    labels = frame[label].astype(str).tolist() if label else []
    palette = _series_colors(labels or [x], payload)

    sizes = 120.0
    if payload.get("size"):
        raw = frame[payload["size"]].to_numpy(float)
        span = np.nanmax(raw) - np.nanmin(raw)
        scaled = (raw - np.nanmin(raw)) / span if span > 0 else np.zeros_like(raw)
        sizes = 60 + 230 * scaled

    ax.scatter(
        frame[x].to_numpy(float),
        frame[y].to_numpy(float),
        s=sizes,
        c=palette[: len(frame)],
        alpha=0.9,
        edgecolors=theme.SURFACE,
        linewidths=0.7,
        zorder=3,
    )
    for i, text in enumerate(labels):
        ax.annotate(
            text,
            (float(frame[x].iloc[i]), float(frame[y].iloc[i])),
            textcoords="offset points",
            xytext=(8, 3),
            fontsize=6.8,
            color=theme.INK,
        )
    _percent_axis(ax, "x", payload.get("x_decimals", _auto_decimals(frame[x])))
    _percent_axis(ax, "y", payload.get("y_decimals", _auto_decimals(frame[y])))
    ax.set_xlabel(payload.get("x_title", ""))
    ax.set_ylabel(payload.get("y_title", ""))
    ax.grid(axis="both", visible=True)
    ax.margins(0.13)
    return _finish(fig)


def _render_frontier(payload: dict, height_in: float) -> bytes:
    fig, ax = _new_figure(height_in)
    cloud = payload.get("cloud")
    frontier = payload.get("frontier")
    named = payload.get("named")
    x, y = payload["x"], payload["y"]

    if cloud is not None and not cloud.empty:
        ax.scatter(
            cloud[x].to_numpy(float),
            cloud[y].to_numpy(float),
            s=2.4,
            color=theme.FAINT,
            alpha=0.30,
            linewidths=0,
            zorder=1,
        )
    if frontier is not None and not frontier.empty:
        ordered = frontier.sort_values(x)
        ax.plot(
            ordered[x].to_numpy(float),
            ordered[y].to_numpy(float),
            color=theme.NAVY,
            linewidth=1.9,
            zorder=3,
            label=payload.get("frontier_label", "Efficient frontier"),
        )
    if named is not None and not named.empty:
        for _, row in named.iterrows():
            name = str(row["Portfolio"])
            color = theme.portfolio_color(name)
            ax.scatter(
                float(row[x]),
                float(row[y]),
                s=58,
                color=color,
                marker="D" if name == "Reference Portfolio" else "o",
                edgecolors=theme.SURFACE,
                linewidths=0.8,
                zorder=5,
                label=name,
            )

    reference = frontier if frontier is not None and not frontier.empty else cloud
    if payload.get("x_percent", True):
        _percent_axis(ax, "x", payload.get("x_decimals", _auto_decimals(reference[x])))
    else:
        from matplotlib.ticker import FormatStrFormatter

        ax.xaxis.set_major_formatter(
            FormatStrFormatter(payload.get("x_format", "%.2f"))
        )
    if payload.get("y_percent", True):
        _percent_axis(ax, "y", payload.get("y_decimals", _auto_decimals(reference[y])))
    ax.set_xlabel(payload.get("x_title", ""))
    ax.set_ylabel(payload.get("y_title", ""))
    ax.grid(axis="both", visible=True)
    ax.margins(0.06)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        _legend_below(ax, handles, labels, columns=min(5, len(labels)))
    return _finish(fig)


def _render_multi_line(payload: dict, height_in: float) -> bytes:
    """Two frontier curves that share a covariance but differ in mean input."""
    fig, ax = _new_figure(height_in)
    frame = payload["data"]
    x, y, series = payload["x"], payload["y"], payload["series"]
    names = list(dict.fromkeys(frame[series].tolist()))
    styles = payload.get("dashes", {})
    palette = payload.get("colors", {})

    for i, name in enumerate(names):
        part = frame[frame[series] == name].sort_values(x)
        ax.plot(
            part[x],
            part[y],
            color=palette.get(name, theme.CATEGORICAL[i]),
            linestyle=styles.get(name, "-"),
            linewidth=1.9,
            label=str(name),
        )
    _percent_axis(ax, "x", payload.get("x_decimals", _auto_decimals(frame[x])))
    _percent_axis(ax, "y", payload.get("y_decimals", _auto_decimals(frame[y])))
    ax.set_xlabel(payload.get("x_title", ""))
    ax.set_ylabel(payload.get("y_title", ""))
    ax.grid(axis="both", visible=True)
    _legend_below(ax, columns=2)
    return _finish(fig)


def _render_heatmap(payload: dict, height_in: float) -> bytes:
    from matplotlib.colors import LinearSegmentedColormap

    fig, ax = _new_figure(height_in)
    matrix: pd.DataFrame = payload["matrix"]
    palette = payload.get("palette", theme.DIVERGING)
    cmap = LinearSegmentedColormap.from_list("ledger", list(palette))
    vmin = payload.get("vmin", float(np.nanmin(matrix.to_numpy(float))))
    vmax = payload.get("vmax", float(np.nanmax(matrix.to_numpy(float))))

    values = matrix.to_numpy(float)
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns.astype(str), fontsize=6.8)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index.astype(str), fontsize=6.8)
    ax.xaxis.set_ticks_position("top")
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color=theme.SURFACE, linewidth=1.1, linestyle="-")
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)

    spec = payload.get("cell_format", ".2f")
    threshold = payload.get("text_threshold", 0.62)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            span = max(abs(vmax), abs(vmin), 1e-9)
            dark = abs(value) / span > threshold
            ax.text(
                j,
                i,
                format(value, spec),
                ha="center",
                va="center",
                fontsize=5.9,
                color=theme.SURFACE if dark else theme.INK_SOFT,
            )

    bar = fig.colorbar(image, ax=ax, fraction=0.022, pad=0.015)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=6.2, length=0, colors=theme.MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return _finish(fig)


def _render_fan(payload: dict, height_in: float) -> bytes:
    fig, ax = _new_figure(height_in)
    fan = payload["data"]
    color = payload.get("color", theme.NAVY)
    months = fan["Month"].to_numpy(float)

    ax.fill_between(
        months,
        fan["5th"],
        fan["95th"],
        color=color,
        alpha=0.15,
        linewidth=0,
        label="5th – 95th percentile",
    )
    ax.fill_between(
        months,
        fan["25th"],
        fan["75th"],
        color=color,
        alpha=0.30,
        linewidth=0,
        label="25th – 75th percentile",
    )
    ax.plot(
        months,
        fan["Median"],
        color=color,
        linewidth=1.8,
        label="Median wealth trajectory",
    )
    if payload.get("reference") is not None:
        ax.axhline(
            float(payload["reference"]),
            color=theme.MUTED,
            linestyle=(0, (4, 3)),
            linewidth=0.9,
            label="Starting wealth",
        )
    _money_axis(ax, "y")
    ax.set_xlabel(payload.get("x_title", "Simulated month"))
    ax.set_ylabel(payload.get("y_title", "Portfolio wealth"))
    ax.margins(x=0)
    _legend_below(ax, columns=4)
    return _finish(fig)


def _render_distribution(payload: dict, height_in: float) -> bytes:
    fig, ax = _new_figure(height_in)
    hist = payload["data"]
    names = list(dict.fromkeys(hist["Series"].tolist()))
    palette = _series_colors(names, {**payload, "color_kind": "portfolio"})

    for name, color in zip(names, palette, strict=True):
        part = hist[hist["Series"] == name].sort_values("Value")
        ax.plot(
            part["Value"], part["Share"], color=color, linewidth=1.3, label=str(name)
        )
        ax.fill_between(
            part["Value"], part["Share"], 0.0, color=color, alpha=0.16, linewidth=0
        )
    if payload.get("reference") is not None:
        ax.axvline(
            float(payload["reference"]),
            color=theme.MUTED,
            linestyle=(0, (4, 3)),
            linewidth=0.9,
        )
    value_format = payload.get("value_format", "money")
    if value_format == "money":
        _money_axis(ax, "x")
    else:
        _percent_axis(ax, "x", payload.get("x_decimals", 0))
    _percent_axis(ax, "y", 1)
    ax.set_xlabel(payload.get("x_title", ""))
    ax.set_ylabel(payload.get("y_title", "Share of simulated paths"))
    ax.margins(x=0)
    _legend_below(ax, columns=min(5, len(names)))
    return _finish(fig)


_FIGURE_RENDERERS = {
    "lines": _render_lines,
    "stacked_area": _render_stacked_area,
    "stacked_barh": _render_stacked_barh,
    "barh": _render_barh,
    "grouped_bars": _render_grouped_bars,
    "range_dot": _render_range_dot,
    "scatter": _render_scatter,
    "frontier": _render_frontier,
    "multi_line": _render_multi_line,
    "heatmap": _render_heatmap,
    "fan": _render_fan,
    "distribution": _render_distribution,
}


def render_figure(spec: Figure) -> bytes | None:
    renderer = _FIGURE_RENDERERS.get(spec.kind)
    if renderer is None:
        return None
    try:
        return renderer(spec.payload, spec.height_in)
    except Exception:
        # A single unplottable panel must never take down an export.
        return None


# ----------------------------------------------------------------------
# Flowable builders
# ----------------------------------------------------------------------


def _format_cell(value: Any, spec: str | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and not np.isfinite(value):
        return "—"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%b %Y")
    if spec and isinstance(value, (int, float, np.integer, np.floating)):
        try:
            return spec.format(value)
        except (ValueError, TypeError):
            return str(value)
    if isinstance(value, (np.bool_, bool)):
        return "Yes" if bool(value) else "No"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.3f}"
    return str(value)


def _table_flowable(block: Table, width: float) -> list[Flowable]:
    styles = _styles()
    frame = block.frame
    if frame is None or frame.empty:
        return [Paragraph("No rows to report.", styles["caption"])]

    truncated = False
    if block.max_rows and len(frame) > block.max_rows:
        frame = frame.head(block.max_rows)
        truncated = True

    formats = dict(block.formats or {})
    for column in frame.columns:
        if column in formats:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            spec = table_utils.display_format(column) or block.default_format
            if spec:
                formats[column] = spec

    index_label = block.index_label or (frame.index.name or "")
    header = [index_label] + [str(c) for c in frame.columns]
    body = []
    for label, row in frame.iterrows():
        cells = [_format_cell(label, None)]
        for column in frame.columns:
            cells.append(_format_cell(row[column], formats.get(column)))
        body.append(cells)

    font_size = block.font_size or (
        7.4 if len(header) <= 9 else 6.8 if len(header) <= 14 else 6.1
    )
    numeric_columns = {
        i + 1
        for i, column in enumerate(frame.columns)
        if pd.api.types.is_numeric_dtype(frame[column])
    }

    header_left = ParagraphStyle(
        "th_left", parent=styles["th"], fontSize=font_size - 0.2
    )
    header_right = ParagraphStyle("th_right", parent=header_left, alignment=2)
    cell_left = ParagraphStyle("td_left", parent=styles["td"], fontSize=font_size)
    cell_right = ParagraphStyle("td_right", parent=cell_left, alignment=2)

    data = [
        [
            Paragraph(
                _escape(text), header_right if i in numeric_columns else header_left
            )
            for i, text in enumerate(header)
        ]
    ]
    data += [
        [
            Paragraph(_escape(text), cell_right if i in numeric_columns else cell_left)
            for i, text in enumerate(row)
        ]
        for row in body
    ]

    # Natural column widths: a six-column table should not be stretched across
    # the page just because the page is wide. Only scale down if the natural
    # width overflows the frame. Widths are measured with the real font metrics
    # rather than an average character width, because an all-caps value such as
    # "INCREASE" is far wider than the same number of digits.
    regular, bold, _ = register_fonts()
    padding = 9.0
    natural = []
    for index, name in enumerate(header):
        header_width = pdfmetrics.stringWidth(name, bold, font_size - 0.2)
        body_width = max(
            [pdfmetrics.stringWidth("0.00", regular, font_size)]
            + [
                pdfmetrics.stringWidth(str(row[index]), regular, font_size)
                for row in body
            ]
        )
        natural.append(max(header_width, body_width) + padding)

    total = float(sum(natural)) or 1.0
    if total > width or block.full_width:
        col_widths = [width * (value / total) for value in natural]
    else:
        col_widths = natural

    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), _color(theme.NAVY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), _color("#FFFFFF")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.4),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, _color(theme.NAVY_DEEP)),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, _color(theme.BORDER)),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, _color(theme.BORDER_STRONG)),
    ]
    for row_index in range(1, len(data)):
        if row_index % 2 == 0:
            commands.append(
                (
                    "BACKGROUND",
                    (0, row_index),
                    (-1, row_index),
                    _color(theme.SURFACE_ALT),
                )
            )
    highlight = {str(value) for value in block.highlight}
    if highlight:
        for row_index, label in enumerate(frame.index, start=1):
            if str(label) in highlight:
                commands.append(
                    (
                        "BACKGROUND",
                        (0, row_index),
                        (-1, row_index),
                        _color(theme.BRASS_SOFT),
                    )
                )

    table = RLTable(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle(commands))

    out: list[Flowable] = [table]
    notes = []
    if block.note:
        notes.append(block.note)
    if truncated:
        notes.append(f"Showing the first {block.max_rows} of {len(block.frame)} rows.")
    if notes:
        out.append(Paragraph(_escape(" ".join(notes)), styles["caption"]))
    return out


def _kpi_flowable(block: KPIs, width: float) -> Flowable:
    styles = _styles()
    items = list(block.items)
    if not items:
        return RLSpacer(1, 1)

    cells = []
    for label, value, sub in items:
        cells.append(
            [
                Paragraph(_escape(str(label)).upper(), styles["kpi_label"]),
                Paragraph(_escape(str(value)), styles["kpi_value"]),
                Paragraph(_escape(str(sub)), styles["kpi_sub"]),
            ]
        )

    inner = [
        RLTable([[c] for c in cell], colWidths=[width / len(items) - 8])
        for cell in cells
    ]
    for card in inner:
        card.setStyle(
            TableStyle(
                [
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )

    outer = RLTable([inner], colWidths=[width / len(items)] * len(items), hAlign="LEFT")
    outer.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (-1, -1), _color(theme.SURFACE)),
                ("BOX", (0, 0), (-1, -1), 0.5, _color(theme.BORDER)),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, _color(theme.BORDER)),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return outer


def _banner_flowable(block: StatusBanner, width: float) -> Flowable:
    styles = _styles()
    tone = {
        "navy": theme.NAVY,
        "positive": theme.POSITIVE,
        "negative": theme.NEGATIVE,
        "caution": theme.CAUTION,
        "muted": theme.MUTED,
    }.get(block.tone, theme.NAVY)

    cell = [
        Paragraph(_escape(block.label).upper(), styles["banner_label"]),
        Paragraph(_escape(block.value), styles["banner_value"]),
    ]
    if block.detail:
        cell.append(Paragraph(_escape(block.detail), styles["banner_detail"]))

    table = RLTable([[c] for c in cell], colWidths=[width], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), _color(tone)),
                ("LEFTPADDING", (0, 0), (-1, -1), 11),
                ("RIGHTPADDING", (0, 0), (-1, -1), 11),
                ("TOPPADDING", (0, 0), (0, 0), 8),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 9),
                ("TOPPADDING", (0, 1), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -2), 1),
            ]
        )
    )
    return table


def _callout_flowable(block: Callout, width: float) -> Flowable:
    styles = _styles()
    accent, background = {
        "navy": (theme.NAVY, theme.NAVY_SOFT),
        "brass": (theme.BRASS, theme.BRASS_SOFT),
        "positive": (theme.POSITIVE, theme.POSITIVE_SOFT),
        "caution": (theme.CAUTION, theme.CAUTION_SOFT),
        "negative": (theme.NEGATIVE, theme.NEGATIVE_SOFT),
    }.get(block.tone, (theme.NAVY, theme.NAVY_SOFT))

    table = RLTable(
        [[Paragraph(block.body, styles["callout"])]],
        colWidths=[width],
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), _color(background)),
                ("LINEBEFORE", (0, 0), (0, -1), 2.4, _color(accent)),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def _figure_flowable(block: Figure, width: float) -> list[Flowable]:
    styles = _styles()
    image_bytes = render_figure(block)
    out: list[Flowable] = []
    if block.title:
        out.append(Paragraph(_escape(block.title), styles["figure_title"]))
    if image_bytes is None:
        out.append(Paragraph("Chart unavailable for this run.", styles["caption"]))
        return out

    height = block.height_in * 72.0
    out.append(Image(io.BytesIO(image_bytes), width=width, height=height))
    if block.note:
        out.append(Paragraph(_escape(block.note), styles["caption"]))
    return [KeepTogether(out)]


def _blocks_to_flowables(blocks: Sequence[Block], width: float) -> list[Flowable]:
    styles = _styles()
    flowables: list[Flowable] = []

    for block in blocks:
        if isinstance(block, Heading):
            key = {1: "h1", 2: "h2", 3: "h3"}.get(block.level, "h2")
            flowables.append(Paragraph(_escape(block.text), styles[key]))
            if block.level <= 2:
                flowables.append(HRule(width, 0.7, theme.BRASS, 3.0))
        elif isinstance(block, Text):
            style = ParagraphStyle(
                "measured",
                parent=styles["body"],
                rightIndent=max(0.0, width * (1.0 - block.measure)),
            )
            flowables.append(Paragraph(block.body, style))
        elif isinstance(block, Caption):
            flowables.append(Paragraph(block.body, styles["caption"]))
        elif isinstance(block, Bullets):
            for item in block.items:
                flowables.append(Paragraph(item, styles["bullet"], bulletText="•"))
            flowables.append(RLSpacer(1, 3))
        elif isinstance(block, Callout):
            flowables.append(_callout_flowable(block, width))
            flowables.append(RLSpacer(1, 6))
        elif isinstance(block, StatusBanner):
            flowables.append(_banner_flowable(block, width))
            flowables.append(RLSpacer(1, 7))
        elif isinstance(block, KPIs):
            flowables.append(_kpi_flowable(block, width))
            flowables.append(RLSpacer(1, 8))
        elif isinstance(block, Table):
            flowables.extend(_table_flowable(block, width))
            flowables.append(RLSpacer(1, 7))
        elif isinstance(block, Figure):
            flowables.extend(_figure_flowable(block, width))
            flowables.append(RLSpacer(1, 5))
        elif isinstance(block, Spacer):
            flowables.append(RLSpacer(1, block.height))
        elif isinstance(block, PageBreak):
            flowables.append(RLPageBreak())

    return flowables


# ----------------------------------------------------------------------
# Document assembly
# ----------------------------------------------------------------------


class _ResearchDoc(BaseDocTemplate):
    def __init__(self, buffer, meta: ReportMeta):
        super().__init__(
            buffer,
            pagesize=PAGE_SIZE,
            leftMargin=MARGIN_X,
            rightMargin=MARGIN_X,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
            title=f"{meta.title} — {meta.section_label}".strip(" —"),
            author=meta.title,
            subject=meta.subtitle,
            creator=meta.title,
        )
        self.meta = meta
        regular, bold, _ = register_fonts()
        self._font = regular
        self._font_bold = bold

        frame = Frame(
            MARGIN_X,
            MARGIN_BOTTOM,
            CONTENT_WIDTH,
            PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM,
            id="content",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        cover = Frame(
            MARGIN_X,
            MARGIN_BOTTOM,
            CONTENT_WIDTH,
            PAGE_HEIGHT - 1.4 * cm - MARGIN_BOTTOM,
            id="cover",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(
            [
                PageTemplate(id="cover", frames=[cover], onPage=self._draw_cover),
                PageTemplate(id="body", frames=[frame], onPage=self._draw_chrome),
            ]
        )

    # -- page furniture ------------------------------------------------

    def _draw_cover(self, canvas, _doc) -> None:
        canvas.saveState()
        canvas.setFillColor(_color(theme.NAVY_DEEP))
        canvas.rect(0, PAGE_HEIGHT - 0.5 * cm, PAGE_WIDTH, 0.5 * cm, stroke=0, fill=1)
        canvas.setFillColor(_color(theme.BRASS))
        canvas.rect(0, PAGE_HEIGHT - 0.62 * cm, PAGE_WIDTH, 0.12 * cm, stroke=0, fill=1)
        self._draw_footer(canvas)
        canvas.restoreState()

    def _draw_chrome(self, canvas, _doc) -> None:
        canvas.saveState()
        top = PAGE_HEIGHT - 1.18 * cm
        canvas.setFont(self._font_bold, 8.2)
        canvas.setFillColor(_color(theme.NAVY_DEEP))
        canvas.drawString(MARGIN_X, top, self.meta.title)

        if self.meta.section_label:
            canvas.setFont(self._font, 8.2)
            canvas.setFillColor(_color(theme.MUTED))
            canvas.drawRightString(PAGE_WIDTH - MARGIN_X, top, self.meta.section_label)

        canvas.setStrokeColor(_color(theme.BORDER_STRONG))
        canvas.setLineWidth(0.6)
        canvas.line(
            MARGIN_X,
            top - 0.22 * cm,
            PAGE_WIDTH - MARGIN_X,
            top - 0.22 * cm,
        )
        canvas.setStrokeColor(_color(theme.BRASS))
        canvas.setLineWidth(1.1)
        canvas.line(MARGIN_X, top - 0.22 * cm, MARGIN_X + 2.4 * cm, top - 0.22 * cm)

        self._draw_footer(canvas)
        canvas.restoreState()

    def _draw_footer(self, canvas) -> None:
        bottom = 0.78 * cm
        canvas.setStrokeColor(_color(theme.BORDER))
        canvas.setLineWidth(0.5)
        canvas.line(
            MARGIN_X, bottom + 0.34 * cm, PAGE_WIDTH - MARGIN_X, bottom + 0.34 * cm
        )
        canvas.setFont(self._font, 6.4)
        canvas.setFillColor(_color(theme.MUTED))
        stamp = self.meta.generated_at.strftime("%d %b %Y %H:%M")
        canvas.drawString(MARGIN_X, bottom, f"Generated {stamp} · {FOOTER_NOTE}")
        canvas.setFont(self._font_bold, 6.6)
        canvas.drawRightString(
            PAGE_WIDTH - MARGIN_X, bottom, f"Page {canvas.getPageNumber()}"
        )


def _cover_flowables(meta: ReportMeta, width: float) -> list[Flowable]:
    styles = _styles()
    out: list[Flowable] = [RLSpacer(1, 1.9 * cm)]
    if meta.section_label:
        out.append(Paragraph(_escape(meta.section_label).upper(), styles["kicker"]))
    out.append(Paragraph(_escape(meta.title), styles["cover_title"]))
    out.append(Paragraph(_escape(meta.subtitle), styles["cover_subtitle"]))
    out.append(HRule(width * 0.34, 1.6, theme.BRASS, 12.0))

    if meta.parameters:
        rows = list(meta.parameters)
        half = (len(rows) + 1) // 2
        left, right = rows[:half], rows[half:]
        data = []
        for index in range(half):
            label_a, value_a = left[index]
            if index < len(right):
                label_b, value_b = right[index]
            else:
                label_b, value_b = "", ""
            data.append(
                [
                    Paragraph(_escape(label_a).upper(), styles["kpi_label"]),
                    Paragraph(_escape(value_a), styles["td"]),
                    Paragraph(_escape(label_b).upper(), styles["kpi_label"]),
                    Paragraph(_escape(value_b), styles["td"]),
                ]
            )
        column = width * 0.46
        table = RLTable(
            data,
            colWidths=[column * 0.42, column * 0.58, column * 0.42, column * 0.58],
            hAlign="LEFT",
        )
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                    (
                        "LINEBELOW",
                        (0, 0),
                        (-1, -2),
                        0.25,
                        _color(theme.BORDER),
                    ),
                ]
            )
        )
        out.append(Paragraph("Run parameters", styles["h3"]))
        out.append(table)

    if meta.contents:
        out.append(RLSpacer(1, 0.45 * cm))
        out.append(Paragraph("Contents", styles["h3"]))
        for index, entry in enumerate(meta.contents, start=1):
            out.append(Paragraph(f"{index}.  {_escape(entry)}", styles["bullet"]))

    out.append(RLSpacer(1, 0.65 * cm))
    out.append(
        _callout_flowable(
            Callout(
                "<b>How to read this report.</b> Sections 1 and 2 are in-sample "
                "diagnostics and describe the opportunity set. Sections 3 and 4 "
                "test whether an allocation survives resampled path risk and "
                "strictly out-of-sample construction. Section 5 diagnoses return-based "
                "diversification and overlap. Section 6 summarises the evidence and the "
                "trade-offs without selecting a portfolio for the reader.",
                tone="navy",
            ),
            width * 0.72,
        )
    )
    out.append(RLSpacer(1, 0.35 * cm))
    out.append(
        Paragraph(
            _escape(meta.footnote),
            ParagraphStyle(
                "cover_disclaimer",
                parent=styles["caption"],
                rightIndent=width * 0.28,
            ),
        )
    )
    return out


def build_pdf(report: Report) -> bytes:
    """Render a :class:`Report` to PDF bytes."""
    register_fonts()
    buffer = io.BytesIO()
    doc = _ResearchDoc(buffer, report.meta)

    story: list[Flowable] = []
    story.extend(_cover_flowables(report.meta, CONTENT_WIDTH))
    story.append(NextPageTemplate("body"))
    story.append(RLPageBreak())
    story.extend(_blocks_to_flowables(report.blocks, CONTENT_WIDTH))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def build_report_bytes(
    section_label: str,
    blocks: Sequence[Block],
    parameters: Sequence[tuple[str, str]],
    subtitle: str,
    title: str = "Portfolio Allocation Testing",
    generated_at: datetime | None = None,
    contents: Sequence[str] = (),
) -> bytes:
    """Convenience wrapper used by the app's download buttons."""
    meta = ReportMeta(
        title=title,
        subtitle=subtitle,
        section_label=section_label,
        generated_at=generated_at or datetime.now(),
        parameters=parameters,
        contents=contents,
    )
    return build_pdf(Report(meta=meta, blocks=list(blocks)))


def file_stamp(label: str, generated_at: datetime | None = None) -> str:
    """Filesystem-safe download name for a section export."""
    stamp = (generated_at or datetime.now()).strftime("%Y%m%d-%H%M")
    slug = "".join(char if char.isalnum() else "-" for char in label.lower()).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return f"portfolio-allocation-testing-{slug}-{stamp}"


__all__ = [
    "Bullets",
    "Callout",
    "Caption",
    "DISCLAIMER",
    "Figure",
    "Heading",
    "KPIs",
    "PageBreak",
    "Report",
    "ReportMeta",
    "Spacer",
    "StatusBanner",
    "Table",
    "Text",
    "build_pdf",
    "build_report_bytes",
    "file_stamp",
    "render_figure",
]

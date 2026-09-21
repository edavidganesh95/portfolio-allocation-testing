"""Design system for Portfolio Allocation Testing.

Single source of truth for colour, type and chart styling. Everything the user
sees — Streamlit chrome, Altair charts, exported PDF pages — resolves its
colours from the tokens in this module, so the screen and the printed research
note read as one system.

Design rationale
----------------
*Ledger* is built for dense numeric research read for long stretches and then
printed.

- **Warm paper ground** rather than clinical white: lower glare over long
  sessions, and the screen matches the tone of the exported page.
- **One authoritative navy** carries structure and interaction. Institutional,
  quiet, and it survives greyscale printing.
- **One warm brass accent**, reserved for the reference portfolio and for
  "this is the answer" moments, so highlight never competes with data.
- **Semantic colours** (deep green / brick / amber) are dark enough to read on
  paper, distinguishable under deuteranopia, and separable by lightness alone
  when printed in black and white.
- **Stable asset identity**: a ticker keeps the same colour in every chart and
  in the PDF, so the eye can track a holding across sections.
"""

from __future__ import annotations

from functools import lru_cache

# ----------------------------------------------------------------------
# Core tokens
# ----------------------------------------------------------------------

INK = "#10151C"
INK_SOFT = "#3C4652"
MUTED = "#6B7480"
FAINT = "#9AA1AB"

PAPER = "#FAF8F5"
SURFACE = "#FFFFFF"
SURFACE_ALT = "#F4F1EC"
BORDER = "#E3DED5"
BORDER_STRONG = "#CFC7BA"

NAVY = "#17395E"
NAVY_DEEP = "#0E2540"
NAVY_SOFT = "#E7EDF4"

BRASS = "#A9762F"
BRASS_SOFT = "#F7EFE2"

POSITIVE = "#237A5B"
POSITIVE_SOFT = "#E7F2EC"
NEGATIVE = "#9A2F28"
NEGATIVE_SOFT = "#FAEBE9"
CAUTION = "#B07A18"
CAUTION_SOFT = "#FBF2DF"

GRID = "#EAE5DC"

SANS = "Inter, Segoe UI, Helvetica Neue, Arial, sans-serif"

# ----------------------------------------------------------------------
# Categorical palette — stable, print-safe, deuteranopia-considerate
# ----------------------------------------------------------------------

CATEGORICAL: tuple[str, ...] = (
    "#17395E",  # navy
    "#C1762B",  # brass
    "#2E7D8F",  # teal
    "#7A5AA0",  # violet
    "#2F7A52",  # green
    "#A83A32",  # brick
    "#4C7FB8",  # azure
    "#8A7A2E",  # olive
    "#8C5A3C",  # umber
    "#9E4478",  # plum
    "#56707F",  # slate
    "#9C7D24",  # gold
)

#: Diverging ramp for correlation: teal = diversifying, brick = redundant.
DIVERGING: tuple[str, ...] = (
    "#1F5F6E",
    "#5C97A2",
    "#A8C6CC",
    "#F1EEE8",
    "#DCA9A4",
    "#C06C63",
    "#8E2C25",
)

#: Sequential navy ramp for magnitude-only heat (weights, drift).
SEQUENTIAL: tuple[str, ...] = (
    "#F5F7FA",
    "#DEE7F0",
    "#BCCCE0",
    "#8FAAC9",
    "#5C82AD",
    "#2F5A8B",
    "#17395E",
)

#: Named portfolios keep a fixed identity across every section and the PDF.
PORTFOLIO_COLORS: dict[str, str] = {
    "Reference Portfolio": BRASS,
    "Diversified Maximum Sharpe": NAVY,
    "Robust Consensus": NAVY,
    "Equal Weight": "#56707F",
    "Minimum Variance": "#2E7D8F",
    "Risk Parity": "#2F7A52",
    "Maximum Diversification": "#7A5AA0",
    "Maximum Sharpe": "#4C7FB8",
    "Minimum CVaR": "#A83A32",
    "Maximum Return / CVaR": "#8A7A2E",
    "HRP Diversification": "#1F5F6E",
}

# ----------------------------------------------------------------------
# Colour helpers
# ----------------------------------------------------------------------


def portfolio_color(name: str, fallback_index: int = 0) -> str:
    """Stable colour for a named portfolio."""
    return PORTFOLIO_COLORS.get(
        name,
        CATEGORICAL[fallback_index % len(CATEGORICAL)],
    )


def portfolio_range(names: list[str]) -> list[str]:
    """Colour range aligned to ``names`` for an Altair/matplotlib scale."""
    return [portfolio_color(name, i) for i, name in enumerate(names)]


@lru_cache(maxsize=64)
def _asset_palette(count: int) -> tuple[str, ...]:
    return tuple(CATEGORICAL[i % len(CATEGORICAL)] for i in range(count))


def asset_range(tickers: list[str]) -> list[str]:
    """Colour range aligned to a sorted ticker domain.

    Sorting the domain is what makes the mapping stable: the same universe
    yields the same ticker-to-colour pairing in every chart and in the PDF,
    regardless of the order a particular table happens to use.
    """
    return list(_asset_palette(len(tickers)))


def asset_color_map(tickers: list[str]) -> dict[str, str]:
    domain = sorted(set(tickers))
    return dict(zip(domain, asset_range(domain), strict=True))


def hex_to_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return (
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
    )


def mix(color_a: str, color_b: str, t: float) -> str:
    """Linear blend between two hex colours; ``t=0`` returns ``color_a``."""
    t = min(max(float(t), 0.0), 1.0)
    a = hex_to_rgb(color_a)
    b = hex_to_rgb(color_b)
    blended = [a[i] + (b[i] - a[i]) * t for i in range(3)]
    return "#" + "".join(f"{int(round(c * 255)):02X}" for c in blended)


def ramp(colors: tuple[str, ...] | list[str], t: float) -> str:
    """Sample a discrete colour ramp at ``t`` in [0, 1]."""
    stops = list(colors)
    if not stops:
        return SURFACE
    if len(stops) == 1:
        return stops[0]
    t = min(max(float(t), 0.0), 1.0)
    position = t * (len(stops) - 1)
    low = int(position)
    high = min(low + 1, len(stops) - 1)
    return mix(stops[low], stops[high], position - low)


def relative_luminance(color: str) -> float:
    """WCAG relative luminance of a hex colour."""

    def channel(value: float) -> float:
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    r, g, b = hex_to_rgb(color)
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG contrast ratio between two hex colours."""
    a = relative_luminance(foreground)
    b = relative_luminance(background)
    lighter, darker = max(a, b), min(a, b)
    return (lighter + 0.05) / (darker + 0.05)


def readable_text(background: str) -> str:
    """Ink or white, whichever reads better on ``background``.

    Chosen by contrast ratio rather than a lightness threshold, so a mid-tone
    heat-map cell never ends up with dark text on a dark wash.
    """
    return (
        INK
        if contrast_ratio(INK, background) >= contrast_ratio("#FFFFFF", background)
        else "#FFFFFF"
    )


# ----------------------------------------------------------------------
# Altair theme
# ----------------------------------------------------------------------

ALTAIR_THEME_NAME = "ledger"


def altair_theme() -> dict:
    """Vega-Lite config for the Ledger design system."""
    axis = {
        "labelFont": SANS,
        "labelFontSize": 11,
        "labelColor": MUTED,
        "labelPadding": 4,
        "titleFont": SANS,
        "titleFontSize": 11,
        "titleFontWeight": 600,
        "titleColor": INK_SOFT,
        "titlePadding": 8,
        "domainColor": BORDER_STRONG,
        "tickColor": BORDER_STRONG,
        "tickSize": 4,
        "gridColor": GRID,
        "gridDash": [2, 3],
    }
    return {
        "config": {
            "background": "transparent",
            "padding": {"left": 2, "right": 10, "top": 6, "bottom": 2},
            "font": SANS,
            "view": {"stroke": "transparent"},
            "axis": axis,
            "axisX": {"grid": False},
            "axisY": {"grid": True, "domain": False, "tickSize": 0},
            "legend": {
                "labelFont": SANS,
                "labelFontSize": 11,
                "labelColor": INK_SOFT,
                "titleFont": SANS,
                "titleFontSize": 11,
                "titleColor": MUTED,
                "titleFontWeight": 600,
                "symbolType": "circle",
                "symbolSize": 70,
                "orient": "bottom",
                "direction": "horizontal",
                "columns": 6,
                "offset": 10,
                "labelLimit": 220,
            },
            "title": {
                "font": SANS,
                "fontSize": 13,
                "fontWeight": 650,
                "color": INK,
                "anchor": "start",
                "offset": 10,
                "subtitleFont": SANS,
                "subtitleFontSize": 11,
                "subtitleColor": MUTED,
            },
            "range": {
                "category": list(CATEGORICAL),
                "diverging": list(DIVERGING),
                "heatmap": list(SEQUENTIAL),
                "ramp": list(SEQUENTIAL),
            },
            "line": {"strokeWidth": 2, "strokeCap": "round"},
            "area": {"opacity": 0.85, "line": False},
            "bar": {"cornerRadiusEnd": 2},
            "point": {"filled": True, "size": 80},
            "rule": {"strokeWidth": 2},
            "text": {"font": SANS, "fontSize": 11, "color": INK_SOFT},
            "circle": {"stroke": None},
        }
    }


def register_altair_theme(enable: bool = True) -> None:
    """Register (and optionally enable) the Ledger Altair theme."""
    import altair as alt

    try:
        alt.theme.register(ALTAIR_THEME_NAME, enable=enable)(altair_theme)
    except Exception:  # pragma: no cover - older Altair fallback
        alt.themes.register(ALTAIR_THEME_NAME, altair_theme)
        if enable:
            alt.themes.enable(ALTAIR_THEME_NAME)


# ----------------------------------------------------------------------
# Matplotlib style — used by the PDF export so print matches screen
# ----------------------------------------------------------------------


def matplotlib_rc() -> dict:
    from matplotlib.rcsetup import cycler

    return {
        "figure.facecolor": SURFACE,
        "figure.edgecolor": SURFACE,
        "figure.dpi": 190,
        "savefig.dpi": 190,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BORDER_STRONG,
        "axes.linewidth": 0.7,
        "axes.labelcolor": INK_SOFT,
        "axes.labelsize": 8.0,
        "axes.titlesize": 9.5,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 7.0,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": cycler(color=list(CATEGORICAL)),
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": (0, (1.6, 2.4)),
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "xtick.major.size": 2.6,
        "ytick.major.size": 0.0,
        "xtick.major.width": 0.7,
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "legend.frameon": False,
        "legend.fontsize": 7.4,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "lines.linewidth": 1.5,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0.0,
    }


def apply_matplotlib_style() -> None:
    import matplotlib

    matplotlib.rcParams.update(matplotlib_rc())

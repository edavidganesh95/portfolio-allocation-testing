import pytest

from src import theme


def _relative_luminance(hex_color: str) -> float:
    def channel(value: float) -> float:
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    r, g, b = theme.hex_to_rgb(hex_color)
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def _contrast(foreground: str, background: str) -> float:
    a = _relative_luminance(foreground)
    b = _relative_luminance(background)
    lighter, darker = max(a, b), min(a, b)
    return (lighter + 0.05) / (darker + 0.05)


def test_categorical_palette_has_no_duplicates():
    assert len(set(theme.CATEGORICAL)) == len(theme.CATEGORICAL)


@pytest.mark.parametrize("color", theme.CATEGORICAL)
def test_categorical_colors_are_legible_on_paper(color):
    # 3:1 is the WCAG threshold for graphical objects and large text.
    assert _contrast(color, theme.PAPER) >= 3.0


@pytest.mark.parametrize(
    "color",
    [theme.INK, theme.INK_SOFT, theme.NAVY, theme.NEGATIVE, theme.POSITIVE],
)
def test_body_colors_meet_text_contrast(color):
    assert _contrast(color, theme.PAPER) >= 4.5


@pytest.mark.parametrize("color", theme.PORTFOLIO_COLORS.values())
def test_portfolio_colors_are_legible(color):
    assert _contrast(color, theme.SURFACE) >= 3.0


def test_semantic_colors_are_distinguishable_in_greyscale():
    """Printed in black and white, pass/fail must not collapse to one grey."""
    lightness = sorted(
        _relative_luminance(c) for c in (theme.POSITIVE, theme.NEGATIVE, theme.CAUTION)
    )
    gaps = [b - a for a, b in zip(lightness, lightness[1:], strict=False)]
    assert min(gaps) > 0.03


def test_asset_color_map_is_stable_regardless_of_input_order():
    forward = theme.asset_color_map(["VWO", "TMFC", "ACWI"])
    backward = theme.asset_color_map(["ACWI", "VWO", "TMFC"])
    assert forward == backward
    assert len(set(forward.values())) == 3


def test_asset_color_map_cycles_beyond_the_palette():
    tickers = [f"T{i:02d}" for i in range(len(theme.CATEGORICAL) + 3)]
    mapping = theme.asset_color_map(tickers)
    assert len(mapping) == len(tickers)
    assert set(mapping.values()) <= set(theme.CATEGORICAL)


def test_portfolio_range_follows_the_requested_order():
    names = ["Robust Consensus", "Reference Portfolio"]
    assert theme.portfolio_range(names) == [theme.NAVY, theme.BRASS]


def test_portfolio_color_falls_back_for_unknown_names():
    assert theme.portfolio_color("Something New", 2) == theme.CATEGORICAL[2]


@pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_ramp_returns_a_valid_hex_color(t):
    value = theme.ramp(theme.SEQUENTIAL, t)
    assert value.startswith("#") and len(value) == 7
    theme.hex_to_rgb(value)


def test_ramp_endpoints_match_the_palette():
    assert theme.ramp(theme.SEQUENTIAL, 0.0) == theme.SEQUENTIAL[0].upper()
    assert theme.ramp(theme.SEQUENTIAL, 1.0) == theme.SEQUENTIAL[-1].upper()


def test_ramp_clamps_out_of_range_inputs():
    assert theme.ramp(theme.SEQUENTIAL, -5) == theme.ramp(theme.SEQUENTIAL, 0.0)
    assert theme.ramp(theme.SEQUENTIAL, 5) == theme.ramp(theme.SEQUENTIAL, 1.0)


def test_mix_is_symmetric_at_the_midpoint():
    assert theme.mix("#000000", "#FFFFFF", 0.5) == theme.mix("#FFFFFF", "#000000", 0.5)


def test_readable_text_flips_with_background_lightness():
    assert theme.readable_text(theme.NAVY_DEEP) == "#FFFFFF"
    assert theme.readable_text(theme.PAPER) == theme.INK


def test_altair_theme_registers_and_exposes_the_palette():
    import altair as alt

    theme.register_altair_theme()
    assert alt.theme.active == theme.ALTAIR_THEME_NAME
    config = theme.altair_theme()["config"]
    assert config["range"]["category"] == list(theme.CATEGORICAL)


def test_matplotlib_style_applies_the_same_palette():
    import matplotlib

    theme.apply_matplotlib_style()
    cycle = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
    assert cycle == list(theme.CATEGORICAL)

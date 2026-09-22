"""Portfolio Allocation Testing — Streamlit application.

Six analysis sections, driven from one sidebar of shared research controls:

1. **Portfolio Construction** — common-sample return diagnostics and candidate
   buy-and-hold starting allocations.
2. **Multi-Period Analysis** — all-portfolio 1Y / 3Y / 5Y / full-history realised metrics plus
   3Y / 5Y / full refits of every applicable construction rule.
3. **Bootstrap Simulation** — long-horizon buy-and-hold path risk with natural drift.
4. **Testing Without Hindsight** — independent out-of-sample buy-and-hold entry tests.
5. **Diversification & Overlap** — HRP and return-based concentration diagnostics.
6. **Summary of Findings** — a neutral comparison of return, risk, validation,
   diversification and implementation trade-offs.

The quantitative engine lives in ``src/``; the Ledger design system lives in
``src/theme.py`` and ``src/ui.py``.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st

from src import analytics, charts, guide, tables, theme
from src import exporting as ex
from src.compute import (
    BootstrapOutput,
    cached_all_lookback_weights,
    cached_benchmark_returns,
    cached_bootstrap,
    cached_construction,
    cached_covariance,
    cached_cvar_frontier,
    cached_cvar_opportunity_set,
    cached_efficient_frontier,
    cached_expected_return_diagnostics,
    cached_expected_returns,
    cached_feasible_cloud,
    cached_historical_cagr,
    cached_market_data,
    cached_monthly_returns,
    cached_multi_period_metrics,
    cached_return_diversification_frontier,
    cached_summary_table,
    cached_walk_forward,
    weights_frame,
)
from src.config import load_reference_example, load_universe, parse_ticker_input
from src.data import final_period_is_incomplete
from src.diversification import (
    HRP_NAME,
    cluster_membership_frame,
    correlation_clusters,
    diversification_profile,
    hierarchical_risk_parity,
)
from src.frontier import (
    named_cvar_points,
    named_portfolio_points,
    validate_weight_cap,
)
from src.metrics import buy_and_hold_portfolio_returns
from src.multiperiod import lookback_weight_stability
from src.ui import (
    apply_theme,
    block,
    callout,
    chart,
    export_bar,
    footer,
    glossary_expander,
    landing_guide,
    legend_key,
    methodology_strip,
    page_header,
    plain_english,
    section,
    status_row,
)
from src.walkforward import (
    block_win_counts,
    drift_summary,
    entry_outcome_summary,
    format_win_count,
)

st.set_page_config(
    page_title="Portfolio Allocation Testing",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()
charts.configure_altair()
page_header()

cfg = load_universe()
reference_example = load_reference_example()
all_tickers = list(cfg.assets)

MARKET_BENCHMARK_PRESETS = {
    "ACWI — Global equities": "ACWI",
    "VT — Global total market": "VT",
    "SPY — US large cap": "SPY",
    "VTI — US total market": "VTI",
    "EUSA — MSCI USA": "EUSA",
    "IEFA — Developed ex-US": "IEFA",
    "Custom Yahoo ticker": None,
}

SECTION_LABELS = [
    "Portfolio Construction",
    "Multi-Period Analysis",
    "Bootstrap Simulation",
    "Testing Without Hindsight",
    "Diversification & Overlap",
    "Summary of Findings",
]

MODEL_METHODOLOGIES = guide.METHOD_BLURBS

ROLLING_WINDOWS = {
    "monthly": (12, "12-month"),
    "weekly": (52, "52-week"),
    "daily": (252, "12-month"),
}


# ============================================================
# SIDEBAR — SHARED RESEARCH CONTROLS
# ============================================================


@st.fragment
def _research_controls_fragment() -> None:
    """Render controls without forcing the full analysis page to rerun.

    Streamlit normally re-executes the complete script whenever any widget
    changes. Keeping the sidebar inside a fragment means dropdowns, sliders and
    reference-weight edits only rerun this small control panel. The expensive
    research page is refreshed only when the reader explicitly presses
    ``Run research``.
    """
    st.markdown(
        '<div class="sidebar-title">Research controls</div>'
        '<div class="sidebar-note">Shared across all six sections. '
        "Adjust controls freely; the report updates only when you press Run research.</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Universe", expanded=True):
        selected_curated = st.multiselect(
            "Curated research universe",
            options=all_tickers,
            default=all_tickers,
            format_func=lambda t: f"{t} — {cfg.assets[t].role}",
            help=guide.HELP["universe_curated"],
            key="control_selected_curated",
        )
        custom_universe_raw = st.text_area(
            "Add Yahoo Finance tickers",
            value="",
            placeholder="e.g. QQQM, IJR, VXUS, 2800.HK",
            help=guide.HELP["universe_custom"],
            key="control_custom_universe",
        )
        custom_universe = parse_ticker_input(custom_universe_raw)
        selected = list(dict.fromkeys(selected_curated + custom_universe))

    with st.expander("Sample", expanded=True):
        start_date = st.date_input(
            "Requested history start", value=date(2018, 2, 1), key="control_start_date"
        )
        end_date = st.date_input(
            "History end", value=date.today(), key="control_end_date"
        )
        frequency = st.selectbox(
            "Return frequency",
            ["monthly", "weekly", "daily"],
            index=0,
            help=guide.HELP["frequency"],
            key="control_frequency",
        )

    with st.expander("Model", expanded=True):
        market_benchmark_choice = st.selectbox(
            "Market benchmark",
            options=list(MARKET_BENCHMARK_PRESETS),
            index=0,
            help=guide.HELP["benchmark"],
            key="control_market_benchmark_choice",
        )
        if MARKET_BENCHMARK_PRESETS[market_benchmark_choice] is None:
            market_benchmark = (
                st.text_input(
                    "Custom benchmark ticker",
                    value="",
                    placeholder="e.g. QQQ, IWM, ^GSPC",
                    help=guide.HELP["benchmark_custom"],
                    key="control_market_benchmark_custom",
                )
                .strip()
                .upper()
            )
        else:
            market_benchmark = MARKET_BENCHMARK_PRESETS[market_benchmark_choice]

        market_return_prior = (
            st.number_input(
                "Long-run market return assumption (%)",
                value=8.0,
                step=0.5,
                min_value=-20.0,
                max_value=30.0,
                help=guide.HELP["market_return"],
                key="control_market_return_prior",
            )
            / 100.0
        )
        shrinkage = st.slider(
            "Expected-return shrinkage",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.05,
            help=guide.HELP["shrinkage"],
            key="control_shrinkage",
        )
        max_weight = st.slider(
            "Maximum ETF weight",
            min_value=0.25,
            max_value=1.00,
            value=0.70,
            step=0.05,
            help=guide.HELP["max_weight"],
            key="control_max_weight",
        )
        risk_free_rate = (
            st.number_input(
                "Risk-free rate (%)",
                value=0.0,
                step=0.25,
                help=guide.HELP["risk_free"],
                key="control_risk_free_rate",
            )
            / 100.0
        )
        cvar_level = (
            st.selectbox(
                "CVaR confidence",
                options=[90, 95, 97, 99],
                index=1,
                format_func=lambda x: f"{x}%",
                help=guide.HELP["cvar"],
                key="control_cvar_level",
            )
            / 100.0
        )

    with st.expander("Diversified Max Sharpe", expanded=True):
        diversified_max_weight = (
            st.slider(
                "Maximum single ETF (%)",
                min_value=20,
                max_value=70,
                value=50,
                step=5,
                help=guide.HELP["div_max_weight"],
                key="control_diversified_max_weight",
            )
            / 100.0
        )
        diversified_min_effective_assets = st.slider(
            "Minimum effective ETF count",
            min_value=1.5,
            max_value=6.0,
            value=3.0,
            step=0.5,
            help=guide.HELP["div_min_effective"],
            key="control_diversified_min_effective_assets",
        )
        diversified_max_risk_share = (
            st.slider(
                "Maximum single risk contribution (%)",
                min_value=25,
                max_value=80,
                value=50,
                step=5,
                help=guide.HELP["div_max_risk"],
                key="control_diversified_max_risk_share",
            )
            / 100.0
        )

    with st.expander("Bootstrap simulation", expanded=True):
        bootstrap_horizon = st.selectbox(
            "Simulation horizon (years)",
            options=[5, 10, 15, 20],
            index=1,
            help=guide.HELP["boot_horizon"],
            key="control_bootstrap_horizon",
        )
        bootstrap_simulations = st.selectbox(
            "Simulation paths",
            options=[1_000, 5_000, 10_000, 25_000],
            index=2,
            help=guide.HELP["boot_paths"],
            key="control_bootstrap_simulations",
        )
        bootstrap_block = st.selectbox(
            "Block length (months)",
            options=[1, 3, 6, 12],
            index=1,
            help=guide.HELP["boot_block"],
            key="control_bootstrap_block",
        )
        bootstrap_seed = st.number_input(
            "Simulation seed",
            min_value=1,
            max_value=1_000_000,
            value=42,
            step=1,
            help=guide.HELP["boot_seed"],
            key="control_bootstrap_seed",
        )
        bootstrap_initial_wealth = st.number_input(
            "Starting wealth",
            min_value=1_000.0,
            value=100_000.0,
            step=10_000.0,
            help=guide.HELP["boot_wealth"],
            key="control_bootstrap_initial_wealth",
        )

    with st.expander("Testing without hindsight", expanded=True):
        wf_lookback_label = st.selectbox(
            "Training lookback",
            options=["36 months", "60 months", "Expanding"],
            index=0,
            help=guide.HELP["wf_lookback"],
            key="control_wf_lookback_label",
        )
        wf_holding_months = st.selectbox(
            "Out-of-sample holding period",
            options=[3, 6, 12],
            index=2,
            format_func=lambda x: f"{x} months",
            help=guide.HELP["wf_holding"],
            key="control_wf_holding_months",
        )

    with st.expander("Reference portfolio", expanded=True):
        enable_reference = st.checkbox(
            "Compare against a reference portfolio",
            value=True,
            help=guide.HELP["reference_enable"],
            key="control_enable_reference",
        )

        default_reference_tickers = [
            t for t in reference_example.weights if t in all_tickers
        ]
        reference_curated = st.multiselect(
            "Reference holdings — curated",
            options=all_tickers,
            default=default_reference_tickers,
            disabled=not enable_reference,
            format_func=lambda t: f"{t} — {cfg.assets[t].role}",
            key="control_reference_curated",
        )
        reference_custom_raw = st.text_area(
            "Reference holdings — custom tickers",
            value="",
            placeholder="e.g. VOO, QQQM",
            disabled=not enable_reference,
            help=guide.HELP["reference_custom"],
            key="control_reference_custom",
        )
        reference_custom = parse_ticker_input(reference_custom_raw)
        reference_tickers = list(dict.fromkeys(reference_curated + reference_custom))

        reference_weights_pct: dict[str, float] = {}
        if enable_reference:
            for ticker in reference_tickers:
                default_pct = 100.0 * reference_example.weights.get(ticker, 0.0)
                reference_weights_pct[ticker] = st.number_input(
                    f"{ticker} weight (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(round(default_pct, 2)),
                    step=0.5,
                    key=f"ref_weight_{ticker}",
                )

            total_reference_weight = sum(reference_weights_pct.values())
            st.caption(f"Total weight: {total_reference_weight:.2f}%")
            if reference_tickers and abs(total_reference_weight - 100.0) > 0.05:
                st.warning("Reference portfolio weights must sum to 100%.")

    controls = {
        "selected": selected,
        "start_date": start_date,
        "end_date": end_date,
        "frequency": frequency,
        "market_benchmark": market_benchmark,
        "market_benchmark_choice": market_benchmark_choice,
        "market_return_prior": market_return_prior,
        "shrinkage": shrinkage,
        "max_weight": max_weight,
        "risk_free_rate": risk_free_rate,
        "cvar_level": cvar_level,
        "diversified_max_weight": diversified_max_weight,
        "diversified_min_effective_assets": diversified_min_effective_assets,
        "diversified_max_risk_share": diversified_max_risk_share,
        "bootstrap_horizon": bootstrap_horizon,
        "bootstrap_simulations": bootstrap_simulations,
        "bootstrap_block": bootstrap_block,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_initial_wealth": bootstrap_initial_wealth,
        "wf_lookback_label": wf_lookback_label,
        "wf_holding_months": wf_holding_months,
        "enable_reference": enable_reference,
        "reference_tickers": reference_tickers,
        "reference_weights_pct": reference_weights_pct,
    }
    st.session_state["pending_controls"] = controls

    active_controls = st.session_state.get("run_config")
    if active_controls is not None and active_controls != controls:
        st.info("Controls changed — press Run research to apply them.")

    if st.button("Run research", width="stretch", type="primary", key="run_research"):
        st.session_state["run_config"] = controls
        # Explicitly request a full rerun only here. Ordinary dropdown/slider
        # interactions stay fragment-local and leave the report in place.
        st.rerun()


with st.sidebar:
    _research_controls_fragment()


# ------------------------------------------------------------
# Run snapshot
# ------------------------------------------------------------
# The sidebar fragment keeps edits local. The rest of the application always
# reads the last explicitly-run snapshot, so exploratory UI dropdowns never
# invalidate the analysis or bounce the reader back to the landing state.

sidebar_controls = st.session_state.get("pending_controls")
active = st.session_state.get("run_config")
controls_for_page = active or sidebar_controls

if controls_for_page is None:
    st.error("Could not initialise the research controls.")
    st.stop()

selected = controls_for_page["selected"]
start_date = controls_for_page["start_date"]
end_date = controls_for_page["end_date"]
frequency = controls_for_page["frequency"]
market_benchmark = controls_for_page["market_benchmark"]
market_benchmark_choice = controls_for_page["market_benchmark_choice"]
market_return_prior = controls_for_page["market_return_prior"]
shrinkage = controls_for_page["shrinkage"]
max_weight = controls_for_page["max_weight"]
risk_free_rate = controls_for_page["risk_free_rate"]
cvar_level = controls_for_page["cvar_level"]
diversified_max_weight = controls_for_page["diversified_max_weight"]
diversified_min_effective_assets = controls_for_page["diversified_min_effective_assets"]
diversified_max_risk_share = controls_for_page["diversified_max_risk_share"]
bootstrap_horizon = controls_for_page["bootstrap_horizon"]
bootstrap_simulations = controls_for_page["bootstrap_simulations"]
bootstrap_block = controls_for_page["bootstrap_block"]
bootstrap_seed = controls_for_page["bootstrap_seed"]
bootstrap_initial_wealth = controls_for_page["bootstrap_initial_wealth"]
wf_lookback_label = controls_for_page["wf_lookback_label"]
wf_holding_months = controls_for_page["wf_holding_months"]
enable_reference = controls_for_page["enable_reference"]
reference_tickers = controls_for_page["reference_tickers"]
reference_weights_pct = controls_for_page["reference_weights_pct"]


# ============================================================
# MASTHEAD / PROJECT BASIS
# ============================================================

section(
    "Mandate",
    "Long-term compounding without false precision",
    "A returns-only buy-and-hold allocation engine. The objective is to compare "
    "starting portfolios that balance expected return efficiency with diversification, "
    "then test them across multiple periods, simulated paths and out-of-sample entries.",
)

status_row(
    [
        ("Universe", f"{len(selected)} selected", "Curated + user-defined tickers"),
        ("Primary data", "Yahoo Finance", "Adjusted prices via yfinance"),
        ("Risk model", "Ledoit-Wolf", "Shrinkage covariance", "brass"),
        (
            "Tail risk",
            f"CVaR {int(cvar_level * 100)}%",
            f"Worst {int((1 - cvar_level) * 100)}% of periods",
            "brass",
        ),
    ]
)


def stop_with_error(message: str) -> None:
    """Report a configuration problem and clear the stale run."""
    st.error(message)
    st.session_state.pop("run_config", None)
    footer()
    st.stop()


if active is None:
    landing_guide()
    footer()
    st.stop()

if len(selected) < 2:
    stop_with_error("Select at least two candidate ETFs.")

if enable_reference:
    if not reference_tickers:
        stop_with_error("Add at least one reference holding or disable the comparison.")
    total_reference_weight = sum(reference_weights_pct.values())
    if abs(total_reference_weight - 100.0) > 0.05:
        stop_with_error("Reference portfolio weights must sum to 100% before running.")

analysis_tickers = list(selected)
if enable_reference:
    analysis_tickers = list(dict.fromkeys(analysis_tickers + reference_tickers))

with st.spinner("Downloading and aligning market data …"):
    try:
        market = cached_market_data(
            tuple(analysis_tickers),
            start_date.isoformat(),
            end_date.isoformat(),
            frequency,
        )
    except Exception as exc:
        stop_with_error(f"Could not build the return matrix: {exc}")

returns = market.returns

# The newest month/week is dropped from every return series while it is still in
# progress (see src.data.to_returns), so say what the data actually runs to.
DATA_THROUGH = market.prices.index.max()
PARTIAL_PERIOD_DROPPED = final_period_is_incomplete(market.prices, frequency)
optimizer_tickers = [t for t in selected if t in returns.columns]
if len(optimizer_tickers) < 2:
    stop_with_error(
        "Fewer than two candidate ETFs have a usable common return history."
    )

optimizer_returns = returns[optimizer_tickers].copy()

try:
    validate_weight_cap(len(optimizer_tickers), max_weight)
except ValueError as exc:
    stop_with_error(str(exc))

diversified_cap = min(max_weight, diversified_max_weight)
if len(optimizer_tickers) * diversified_cap < 1.0 - 1e-12:
    stop_with_error(
        "The Diversified Max Sharpe single-ETF cap is infeasible for the selected "
        f"universe. With {len(optimizer_tickers)} assets the cap must be at least "
        f"{1 / len(optimizer_tickers):.1%}."
    )
if diversified_min_effective_assets > len(optimizer_tickers):
    stop_with_error(
        "Minimum effective ETF count cannot exceed the number of selected ETFs."
    )

ASSET_COLORS = theme.asset_color_map(optimizer_tickers)

# ------------------------------------------------------------
# Shared model inputs (memoised — see src/compute.py)
# ------------------------------------------------------------

if not market_benchmark:
    stop_with_error("Choose a market benchmark or enter a custom Yahoo Finance ticker.")

with st.spinner(f"Loading expected-return benchmark ({market_benchmark}) …"):
    try:
        if market_benchmark in returns.columns:
            # Reuse the exact aligned series when the benchmark is already in the
            # downloaded matrix. This makes a benchmark ETF's self-beta exactly 1
            # and avoids a second Yahoo request.
            benchmark_returns = returns[market_benchmark].copy()
        else:
            benchmark_returns = cached_benchmark_returns(
                market_benchmark,
                start_date.isoformat(),
                end_date.isoformat(),
                frequency,
            )
    except Exception as exc:
        stop_with_error(f"Could not load market benchmark {market_benchmark}: {exc}")

try:
    mu = cached_expected_returns(
        optimizer_returns,
        benchmark_returns,
        market_return_prior,
        risk_free_rate,
        frequency,
        shrinkage,
    )
    expected_return_detail = cached_expected_return_diagnostics(
        optimizer_returns,
        benchmark_returns,
        market_return_prior,
        risk_free_rate,
        frequency,
        shrinkage,
    )
except Exception as exc:
    stop_with_error(f"Could not estimate market-anchored expected returns: {exc}")

raw_mu = cached_historical_cagr(optimizer_returns, frequency)
cov = cached_covariance(optimizer_returns, frequency)

with st.spinner("Fitting construction methods …"):
    weights, construction_status = cached_construction(
        optimizer_returns,
        benchmark_returns,
        market_return_prior,
        frequency,
        shrinkage,
        max_weight,
        risk_free_rate,
        cvar_level,
        diversified_max_weight,
        diversified_min_effective_assets,
        diversified_max_risk_share,
    )

if weights.empty:
    stop_with_error("No construction method converged for the selected universe.")

construction_methods = list(weights.index)
construction_portfolios = {name: weights.loc[name] for name in construction_methods}

# HRP is constructed once on the same full-sample covariance/correlation inputs.
# It is deliberately outside the eight return/risk optimisers because expected
# returns do not enter its construction; later sections use it as a transparent
# diversification benchmark.
base_corr = optimizer_returns.corr()
hrp_weights = hierarchical_risk_parity(cov, base_corr, max_weight=max_weight)

with st.spinner("Refitting construction methods across lookbacks …"):
    multi_period_weights, multi_period_status = cached_all_lookback_weights(
        optimizer_returns,
        benchmark_returns,
        market_return_prior,
        risk_free_rate,
        frequency,
        shrinkage,
        max_weight,
        cvar_level,
        min(max_weight, diversified_max_weight),
        diversified_min_effective_assets,
        diversified_max_risk_share,
    )
    multi_period_stability = lookback_weight_stability(multi_period_weights)


# Monthly returns drive simulation and validation regardless of the display
# frequency, so they are built once here rather than inside a tab.
monthly_all = cached_monthly_returns(market.prices)
try:
    if market_benchmark in monthly_all.columns:
        benchmark_monthly_returns = monthly_all[market_benchmark].copy()
    else:
        benchmark_monthly_returns = cached_benchmark_returns(
            market_benchmark,
            start_date.isoformat(),
            end_date.isoformat(),
            "monthly",
        )
except Exception as exc:
    stop_with_error(f"Could not load monthly benchmark returns for walk-forward: {exc}")

# Optional reference return series reused across sections.
reference_returns = None
reference_weights = None
if enable_reference:
    reference_weights = pd.Series(
        {t: w / 100.0 for t, w in reference_weights_pct.items()}, dtype=float
    )
    missing_reference = sorted(set(reference_weights.index) - set(returns.columns))
    if not missing_reference:
        reference_returns = buy_and_hold_portfolio_returns(
            returns[reference_weights.index], reference_weights
        )

# Full static portfolio set used for realised multi-period comparisons and all
# downstream evidence. Reference is held fixed; Equal Weight and HRP are included
# alongside every optimisation method.
all_static_portfolios: dict[str, pd.Series] = {}
if reference_weights is not None and reference_returns is not None:
    all_static_portfolios["Reference Portfolio"] = reference_weights
all_static_portfolios.update(construction_portfolios)
all_static_portfolios[HRP_NAME] = hrp_weights

multi_period_metrics = (
    cached_multi_period_metrics(
        optimizer_returns if reference_weights is None else returns,
        weights_frame(all_static_portfolios),
        frequency,
        risk_free_rate,
        cvar_level,
    )
    if all_static_portfolios
    else pd.DataFrame()
)

# ------------------------------------------------------------
# Walk-forward validation (shared by sections 4 and 5)
# ------------------------------------------------------------

lookback_map = {"36 months": 36, "60 months": 60, "Expanding": None}
wf_lookback_months = lookback_map[wf_lookback_label]
wf_min_train_months = 36 if wf_lookback_months is None else int(wf_lookback_months)
wf_reference = reference_weights if reference_returns is not None else None

wf_result = None
wf_error = None
with st.spinner("Running walk-forward validation …"):
    try:
        wf_result = cached_walk_forward(
            monthly_all,
            benchmark_monthly_returns,
            market_return_prior,
            tuple(optimizer_tickers),
            wf_lookback_months,
            wf_min_train_months,
            int(wf_holding_months),
            float(risk_free_rate),
            float(shrinkage),
            float(max_weight),
            float(cvar_level),
            wf_reference,
            float(min(max_weight, diversified_max_weight)),
            float(diversified_min_effective_assets),
            float(diversified_max_risk_share),
        )
    except Exception as exc:  # surfaced in section 4; other sections still render
        wf_error = str(exc)

wf_entry_stats = (
    entry_outcome_summary(wf_result) if wf_result is not None else pd.DataFrame()
)
wf_drift_stats = drift_summary(wf_result) if wf_result is not None else pd.DataFrame()

GENERATED_AT = datetime.now()
SAMPLE_LABEL = (
    f"{optimizer_returns.index.min():%b %Y} – {optimizer_returns.index.max():%b %Y}"
)
CVAR_COLUMN = f"CVaR {int(cvar_level * 100)}%"

RUN_PARAMETERS: list[tuple[str, str]] = [
    ("Universe", ", ".join(optimizer_tickers)),
    ("Return frequency", frequency.capitalize()),
    ("Common sample", f"{SAMPLE_LABEL} · {len(optimizer_returns)} observations"),
    (
        "Data through",
        f"{DATA_THROUGH:%d %b %Y}"
        + (" (unfinished period excluded)" if PARTIAL_PERIOD_DROPPED else ""),
    ),
    ("Market benchmark", market_benchmark),
    ("Market return assumption", f"{market_return_prior:.1%}"),
    ("Expected-return shrinkage", f"{shrinkage:.0%}"),
    ("Maximum ETF weight", f"{max_weight:.0%}"),
    ("Risk-free rate", f"{risk_free_rate:.2%}"),
    ("CVaR confidence", f"{cvar_level:.0%}"),
    ("Diversified max ETF", f"{min(max_weight, diversified_max_weight):.0%}"),
    ("Minimum effective ETFs", f"{diversified_min_effective_assets:.1f}"),
    ("Maximum risk contribution", f"{diversified_max_risk_share:.0%}"),
    ("Multi-period analysis", "1Y / 3Y / 5Y / Full metrics · 3Y / 5Y / Full refits"),
    (
        "Bootstrap",
        f"{bootstrap_simulations:,} paths · {bootstrap_horizon}y · "
        f"{bootstrap_block}-month blocks · seed {bootstrap_seed}",
    ),
    (
        "Walk-forward",
        f"{wf_lookback_label} training · {wf_holding_months}-month independent B&H holds",
    ),
    (
        "Reference portfolio",
        ", ".join(f"{t} {w:.1%}" for t, w in reference_weights.items() if w > 0)
        if reference_weights is not None
        else "Not enabled",
    ),
]


def pdf_builder(label: str, blocks_factory, contents: tuple[str, ...] = ()):
    """Defer PDF construction until the download button is actually clicked."""

    def build() -> bytes:
        return ex.build_report_bytes(
            section_label=label,
            blocks=blocks_factory(),
            parameters=RUN_PARAMETERS,
            subtitle="Strategic ETF allocation research",
            generated_at=GENERATED_AT,
            contents=contents,
        )

    return build


def excel_builder(tables_factory):
    def build() -> bytes:
        return tables.to_excel_bytes(tables_factory())

    return build


def stamp(label: str) -> str:
    return ex.file_stamp(label, GENERATED_AT)


def figure_height(rows: int, per_row: float = 0.26, base: float = 0.95) -> float:
    """Height in inches for a PDF chart with one row per category.

    Capped so a long universe cannot push a chart past the page and strand it
    on a sheet of its own.
    """
    return float(min(3.9, max(1.8, per_row * max(rows, 1) + base)))


def guided(section_key: str, blocks_factory):
    """Wrap a section's PDF blocks with its plain-English summary and key terms."""

    def build(with_terms: bool = True) -> list:
        blocks = list(blocks_factory())
        after = next((i for i, b in enumerate(blocks) if isinstance(b, ex.Text)), 0) + 1
        blocks[after:after] = guide.pdf_intro(section_key)
        if with_terms:
            blocks.extend(guide.pdf_terms(section_key))
        return blocks

    return build


#: Sections register their builders here as they render, so the combined
#: report contains exactly the sections this run was able to produce. It is
#: mirrored into session state so the export path can be exercised end to end
#: without a browser.
SECTION_EXPORTS: list[tuple[str, object, object]] = []
st.session_state["section_exports"] = SECTION_EXPORTS


(
    tab_construction,
    tab_robustness,
    tab_bootstrap,
    tab_walkforward,
    tab_diversification,
    tab_decision,
) = st.tabs(SECTION_LABELS)


# ============================================================
# SECTION 1 — PORTFOLIO CONSTRUCTION
# ============================================================

with tab_construction:
    section(
        "Section 1",
        "Portfolio construction",
        guide.SECTION_GUIDES["construction"].intro,
    )
    plain_english("construction")
    glossary_expander("construction")

    block("Common return sample", guide.caption("common_sample"))
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Candidate assets", len(optimizer_tickers))
    with d2:
        st.metric("Observations", len(optimizer_returns))
    with d3:
        st.metric("Sample start", optimizer_returns.index.min().strftime("%b %Y"))
    with d4:
        st.metric("Sample end", optimizer_returns.index.max().strftime("%b %Y"))

    if PARTIAL_PERIOD_DROPPED:
        period_word = "week" if frequency == "weekly" else "month"
        st.caption(
            f"Price data runs to {DATA_THROUGH:%d %b %Y}. That {period_word} is still in "
            f"progress, so it is excluded and every observation is a complete {period_word}."
        )

    excluded = sorted(set(selected) - set(optimizer_tickers))
    if excluded:
        st.warning(
            "Excluded because Yahoo Finance did not provide sufficient "
            "usable common history: " + ", ".join(excluded)
        )

    block("Return-stream diagnostics", guide.caption("diagnostics"))
    asset_stats = cached_summary_table(
        optimizer_returns, frequency, risk_free_rate, cvar_level
    )
    tables.render(
        tables.style(asset_stats, heat_columns=tables.metric_columns(asset_stats))
    )

    risk_return = charts.risk_return_scatter(asset_stats, CVAR_COLUMN)
    block("Risk and return", guide.caption("risk_return"))
    chart(risk_return)

    growth_long = analytics.growth_frame(optimizer_returns)
    block("Growth of $1", guide.caption("growth"))

    @st.fragment
    def _render_growth_chart() -> None:
        log_growth = st.toggle(
            "Logarithmic scale",
            value=False,
            key="growth_log",
            help=(
                "A log scale makes equal percentage moves equal distances, which "
                "is the fairer way to compare compounding over long samples. "
                "Changing it updates only this chart."
            ),
        )
        chart(charts.growth_chart(growth_long, optimizer_tickers, log_scale=log_growth))

    _render_growth_chart()

    drawdown_long = analytics.drawdown_frame(optimizer_returns)
    block("Drawdown profile", guide.caption("drawdown"))
    chart(charts.drawdown_chart(drawdown_long, optimizer_tickers))

    rolling_window, rolling_label = ROLLING_WINDOWS.get(frequency, (12, "12-period"))
    rolling_long = analytics.rolling_annual_return(optimizer_returns, rolling_window)
    if not rolling_long.empty:
        with st.expander(f"Rolling {rolling_label} returns", expanded=False):
            st.caption(
                "Compounded return over each trailing window. Useful for seeing "
                "whether a return stream's advantage is persistent or the result "
                "of one concentrated episode."
            )
            chart(
                charts.rolling_return_chart(
                    rolling_long, optimizer_tickers, rolling_label
                )
            )

    corr = optimizer_returns.corr()
    corr_order = analytics.cluster_order(corr)
    corr_long = (
        corr.rename_axis("Asset A")
        .reset_index()
        .melt(id_vars="Asset A", var_name="Asset B", value_name="Correlation")
    )
    block("Return correlation", guide.caption("correlation"))
    chart(charts.correlation_heatmap(corr_long, corr_order))

    with st.expander("Correlation matrix — numeric values", expanded=False):
        st.caption(
            "Pearson correlations over the exact sample used by the "
            "construction methods."
        )
        tables.render(
            tables.style(
                corr,
                heat_columns=list(corr.columns),
                palette=theme.DIVERGING,
                default_format=tables.RATIO2,
            )
        )

    with st.expander("Expected return E(r) — assumptions", expanded=False):
        st.caption(
            f"Benchmark: {market_benchmark} · long-run market return assumption: "
            f"{market_return_prior:.1%} · shrinkage weight: {shrinkage:.0%}. "
            "The shrunk E(r) blends historical CAGR with a beta-implied market anchor; "
            "the benchmark's realised historical CAGR is not automatically treated as a forecast."
        )
        expected_return_display = expected_return_detail.rename(
            columns={
                "Market Prior": "Beta-implied anchor",
                "Blended Expected Return": "Shrunk E(r)",
            }
        )
        tables.render(
            tables.style(
                expected_return_display,
                heat_columns=[
                    "Historical CAGR",
                    "Market Beta",
                    "Beta-implied anchor",
                    "Shrunk E(r)",
                ],
                precision_overrides={
                    "Historical CAGR": "{:.2%}",
                    "Market Beta": "{:.2f}",
                    "Beta-implied anchor": "{:.2%}",
                    "Shrunk E(r)": "{:.2%}",
                },
            )
        )

    block("Construction methods", guide.caption("methods"))
    methodology_strip(MODEL_METHODOLOGIES)

    # A method the solver could not converge on is dropped from every section, so
    # say so here rather than letting a row silently disappear from the tables.
    failed_construction = construction_status.loc[~construction_status["Success"]]
    if not failed_construction.empty:
        st.warning(
            "**Did not converge, so excluded from every section of this report:** "
            + ", ".join(failed_construction["Method"])
            + ". This usually means the constraints leave no feasible portfolio "
            "(for example a maximum risk contribution below 1 ÷ the number of ETFs) "
            "or the sample is too short for that method. Details are below."
        )
        with st.expander("Solver details for unavailable methods", expanded=False):
            tables.render(failed_construction.set_index("Method"))

    tables.render(
        tables.style(
            weights,
            heat_columns=list(weights.columns),
            default_format=tables.PCT1,
        ),
    )

    weights_long = analytics.weights_long(weights)
    block("Allocation by method", guide.caption("allocation"))
    chart(
        charts.weights_composition_chart(
            weights_long, optimizer_tickers, construction_methods
        )
    )

    construction_returns = pd.DataFrame(
        {
            name: buy_and_hold_portfolio_returns(optimizer_returns, weight_vector)
            for name, weight_vector in construction_portfolios.items()
        }
    )
    construction_stats = cached_summary_table(
        construction_returns, frequency, risk_free_rate, cvar_level
    )

    block("Construction-method return & risk metrics", guide.caption("method_metrics"))
    tables.render(
        tables.style(
            construction_stats,
            heat_columns=tables.metric_columns(construction_stats),
        )
    )

    block("The opportunity set & efficient frontier", guide.caption("frontier"))

    # Canonical export / downstream view uses Shrunk E(r). Interactive controls
    # below are display-only and run inside a fragment so changing them does not
    # refresh the full report.
    raw_frontier = cached_efficient_frontier(raw_mu, cov, max_weight, 55).assign(
        Estimator="Historical E(r)"
    )
    shrunk_frontier = cached_efficient_frontier(mu, cov, max_weight, 55).assign(
        Estimator="Shrunk E(r)"
    )
    sensitivity_frontier = pd.concat([raw_frontier, shrunk_frontier], ignore_index=True)

    frontier_portfolios = dict(construction_portfolios)
    if reference_returns is not None:
        frontier_portfolios = {
            "Reference Portfolio": reference_weights,
            **frontier_portfolios,
        }

    # Fixed canonical objects used by the PDF/export and the non-interactive
    # charts that follow.
    feasible_cloud = cached_feasible_cloud(
        mu, cov, max_weight, 5_000, risk_free_rate, 42
    )
    mv_frontier = shrunk_frontier.drop(columns=["Estimator"], errors="ignore")
    named_points = named_portfolio_points(
        frontier_portfolios, mu, cov, risk_free_rate=risk_free_rate
    )
    label_names = [
        name
        for name in [
            "Reference Portfolio",
            "Minimum Variance",
            "Maximum Sharpe",
            "Diversified Maximum Sharpe",
        ]
        if name in set(named_points["Portfolio"])
    ]
    cloud_plot = feasible_cloud[["Expected Return", "Volatility", "Sharpe"]]

    @st.fragment
    def _render_frontier_explorer() -> None:
        f1, f2 = st.columns([1.4, 1])
        with f1:
            estimator = st.radio(
                "Expected-return view",
                options=["Shrunk E(r)", "Historical E(r)", "Compare both"],
                index=0,
                horizontal=True,
                help=(
                    "Shrunk E(r) blends historical CAGR with a beta-implied market anchor. "
                    "Historical E(r) uses realised CAGR. Compare both overlays the two solved frontiers."
                ),
                key="frontier_estimator",
            )
        with f2:
            cloud_size = st.selectbox(
                "Simulated feasible portfolios",
                options=[1_000, 2_500, 5_000, 10_000],
                index=2,
                key="frontier_cloud_size",
                help="Changing this dropdown updates only the frontier panel.",
            )

        if estimator == "Compare both":
            chart(charts.sensitivity_frontier_chart(sensitivity_frontier))
            st.caption(
                "The two lines use identical covariance and portfolio constraints; only the "
                "expected-return estimate changes. Named coordinate cards use Shrunk E(r)."
            )
            display_named = named_points
        else:
            display_mu = raw_mu if estimator == "Historical E(r)" else mu
            display_cloud = cached_feasible_cloud(
                display_mu, cov, max_weight, int(cloud_size), risk_free_rate, 42
            )[["Expected Return", "Volatility", "Sharpe"]]
            display_frontier = cached_efficient_frontier(
                display_mu, cov, max_weight, 55
            )
            display_named = named_portfolio_points(
                frontier_portfolios,
                display_mu,
                cov,
                risk_free_rate=risk_free_rate,
            )
            display_labels = [
                name for name in label_names if name in set(display_named["Portfolio"])
            ]
            chart(
                charts.frontier_chart(
                    display_cloud, display_frontier, display_named, display_labels
                )
            )
            legend_key(
                [
                    ("Reference portfolio (diamond)", theme.BRASS),
                    ("Solved efficient frontier", theme.NAVY),
                    ("Simulated feasible portfolios", theme.FAINT),
                ]
            )

        gmv_row = display_named.loc[display_named["Portfolio"] == "Minimum Variance"]
        max_sharpe_row = display_named.loc[
            display_named["Portfolio"] == "Maximum Sharpe"
        ]
        p1, p2, p3 = st.columns(3)
        with p1:
            if not gmv_row.empty:
                st.metric(
                    "Global minimum variance",
                    f"{gmv_row.iloc[0]['Volatility']:.1%} vol",
                    f"{gmv_row.iloc[0]['Expected Return']:.1%} estimated return",
                    delta_color="off",
                )
        with p2:
            if not max_sharpe_row.empty:
                st.metric(
                    "Maximum Sharpe",
                    f"{max_sharpe_row.iloc[0]['Sharpe']:.2f}",
                    f"{max_sharpe_row.iloc[0]['Volatility']:.1%} vol",
                    delta_color="off",
                )
        with p3:
            if "Diversified Maximum Sharpe" in set(display_named["Portfolio"]):
                dm = display_named.loc[
                    display_named["Portfolio"] == "Diversified Maximum Sharpe"
                ].iloc[0]
                st.metric(
                    "Diversified Max Sharpe",
                    f"{dm['Sharpe']:.2f}",
                    f"{dm['Volatility']:.1%} vol",
                    delta_color="off",
                )

        with st.expander("Named portfolio coordinates", expanded=False):
            frontier_table = display_named.set_index("Portfolio").copy()
            tables.render(
                tables.style(
                    frontier_table,
                    heat_columns=["Expected Return", "Volatility", "Sharpe"],
                    precision_overrides={
                        "Expected Return": "{:.2%}",
                        "Volatility": "{:.2%}",
                        "Sharpe": "{:.2f}",
                    },
                )
            )

    _render_frontier_explorer()

    frontier_mix = analytics.frontier_composition(mv_frontier, "Volatility")
    block("How the frontier is built", guide.caption("frontier_build"))
    chart(charts.frontier_composition_chart(frontier_mix, optimizer_tickers))

    block(
        "Expected-return sensitivity: historical vs shrunk E(r)",
        guide.caption("sensitivity"),
    )
    chart(charts.sensitivity_frontier_chart(sensitivity_frontier))

    block(
        "Return–CVaR opportunity set",
        guide.caption("cvar_frontier", cvar=f"{cvar_level:.0%}"),
    )
    cvar_cloud = cached_cvar_opportunity_set(
        feasible_cloud, optimizer_returns, cvar_level
    )
    tail_frontier = cached_cvar_frontier(
        mu, optimizer_returns, max_weight, cvar_level, 40
    )
    named_tail = named_cvar_points(
        frontier_portfolios, mu, optimizer_returns, level=cvar_level
    )
    chart(
        charts.cvar_frontier_chart(
            cvar_cloud[["Expected Return", "CVaR"]],
            tail_frontier,
            named_tail,
            cvar_level,
        )
    )

    with st.expander("Frontier methodology & limitations", expanded=False):
        st.markdown(
            """
            **Portfolio cloud**

            Random long-only portfolios are generated subject to the same maximum
            single-asset weight used by the optimizers. They show the feasible opportunity
            set but do **not** define the frontier.

            **Mean–variance frontier**

            For a grid of expected-return targets, the engine separately minimizes
            portfolio variance subject to full investment, long-only weights and the
            configured maximum weight. The plotted frontier is therefore an optimized
            boundary, not the outer edge of random simulations.

            **Raw vs market-anchored expected returns**

            Historical CAGR can produce unstable mean–variance results. The shrunk E(r)
            estimate blends each ETF's historical return with a beta-implied return anchored
            to the selected broad-market benchmark and the user's long-run market return
            assumption. This avoids making the expected-return anchor depend on which ETFs
            happen to be in the candidate universe.

            **Return–CVaR frontier**

            This solves the corresponding historical minimum-CVaR problem for each
            expected-return target using a linear-programming formulation.

            **Interpretation**

            All frontier coordinates are in-sample estimates. They are useful for
            understanding the opportunity set and the behaviour of the construction
            methods; the Bootstrap, testing-without-hindsight and Summary of Findings sections remain
            the safeguards against treating the frontier as a forecast.
            """
        )

    # Reference-vs-candidate comparison is consolidated in Summary of Findings.

    # ---------------- Section 1 export ----------------

    def _section1_tables() -> dict[str, pd.DataFrame]:
        out = {
            "Asset diagnostics": asset_stats,
            "Correlation matrix": corr,
            "Method weights": weights,
            "Method solver status": construction_status.set_index("Method"),
            "Method metrics": construction_stats,
            "Frontier coordinates": named_points.set_index("Portfolio"),
            "Efficient frontier": mv_frontier,
        }
        return out

    def _section1_blocks() -> list:
        blocks = [
            ex.Heading("Portfolio construction", 1),
            ex.Text(
                "The common return sample, how each candidate ETF has behaved on "
                "it, and how eight construction philosophies allocate across the "
                "identical history. Everything on these pages is in-sample."
            ),
            ex.KPIs(
                [
                    ("Candidate assets", str(len(optimizer_tickers)), "Common sample"),
                    (
                        "Observations",
                        f"{len(optimizer_returns):,}",
                        frequency.capitalize(),
                    ),
                    (
                        "Sample start",
                        optimizer_returns.index.min().strftime("%b %Y"),
                        "First aligned period",
                    ),
                    (
                        "Sample end",
                        optimizer_returns.index.max().strftime("%b %Y"),
                        "Last aligned period",
                    ),
                ]
            ),
            ex.Heading("Return-stream diagnostics", 2),
            ex.Table(asset_stats, index_label="Ticker"),
            ex.Figure(
                "scatter",
                {
                    "data": asset_stats.reset_index(),
                    "x": "Volatility",
                    "y": "CAGR",
                    "label": "Ticker",
                    "colors": ASSET_COLORS,
                    "x_title": "Annualised volatility",
                    "y_title": "Realised CAGR",
                },
                title="Risk and return",
                note="Position is realised CAGR against annualised volatility.",
                height_in=2.8,
            ),
            ex.PageBreak(),
            ex.Figure(
                "lines",
                {
                    "data": growth_long,
                    "x": "Date",
                    "y": "Growth",
                    "series": "Series",
                    "colors": ASSET_COLORS,
                    "value_format": "plain",
                    "y_title": "Growth of $1",
                },
                title="Growth of $1",
                height_in=2.7,
            ),
            ex.Figure(
                "lines",
                {
                    "data": drawdown_long,
                    "x": "Date",
                    "y": "Drawdown",
                    "series": "Series",
                    "colors": ASSET_COLORS,
                    "fill": True,
                    "y_title": "Drawdown from prior peak",
                },
                title="Drawdown profile",
                height_in=2.5,
            ),
            ex.PageBreak(),
            ex.Heading("Return correlation", 2),
            ex.Figure(
                "heatmap",
                {
                    "matrix": corr.loc[corr_order, corr_order],
                    "palette": theme.DIVERGING,
                    "vmin": -1.0,
                    "vmax": 1.0,
                },
                note=(
                    "Ordered by hierarchical clustering. Brick indicates a "
                    "redundant pair; teal indicates a diversifying pair."
                ),
                height_in=3.4,
            ),
            ex.PageBreak(),
            ex.Heading("Construction methods", 2),
            ex.Table(weights, index_label="Method", default_format="{:.1%}"),
            ex.Figure(
                "stacked_barh",
                {
                    "data": weights_long,
                    "cat": "Method",
                    "value": "Weight",
                    "series": "Ticker",
                    "order": construction_methods,
                    "colors": ASSET_COLORS,
                    "x_title": "Allocation",
                },
                title="Allocation by method",
                height_in=2.9,
            ),
            ex.PageBreak(),
            ex.Heading("Construction-method return & risk metrics", 2),
            ex.Caption(
                "In-sample diagnostics: the same history estimates the weights "
                "and evaluates the resulting baskets. Maximum Sharpe refers to the "
                "model-implied Sharpe under the shrunk expected returns, not "
                "necessarily the highest realised historical Sharpe in this table."
            ),
            ex.Table(construction_stats, index_label="Method"),
            ex.Figure(
                "frontier",
                {
                    "cloud": cloud_plot,
                    "frontier": mv_frontier,
                    "named": named_points,
                    "x": "Volatility",
                    "y": "Expected Return",
                    "x_title": "Expected volatility",
                    "y_title": "Estimated expected return",
                },
                title="Opportunity set & efficient frontier (Shrunk E(r))",
                note=(
                    "Dots are randomly simulated feasible portfolios under the "
                    "same constraints; the line is solved exactly."
                ),
                height_in=3.4,
            ),
            ex.PageBreak(),
            ex.Figure(
                "stacked_area",
                {
                    "data": frontier_mix,
                    "x": "Volatility",
                    "value": "Weight",
                    "series": "Ticker",
                    "colors": ASSET_COLORS,
                    "x_percent": True,
                    "date_axis": False,
                    "x_title": "Expected volatility",
                    "y_title": "Frontier allocation",
                },
                title="How the frontier is built",
                height_in=2.6,
            ),
            ex.Figure(
                "multi_line",
                {
                    "data": sensitivity_frontier,
                    "x": "Volatility",
                    "y": "Expected Return",
                    "series": "Estimator",
                    "colors": {
                        "Historical E(r)": theme.BRASS,
                        "Shrunk E(r)": theme.NAVY,
                    },
                    "dashes": {"Historical E(r)": (0, (5, 3)), "Shrunk E(r)": "-"},
                    "x_title": "Expected volatility",
                    "y_title": "Estimated expected return",
                },
                title="Expected-return sensitivity: historical vs shrunk E(r)",
                note=(
                    "Same covariance and constraints; only the expected-return "
                    "estimate changes."
                ),
                height_in=2.7,
            ),
            ex.PageBreak(),
            ex.Figure(
                "frontier",
                {
                    "cloud": cvar_cloud[["Expected Return", "CVaR"]],
                    "frontier": tail_frontier,
                    "named": named_tail,
                    "x": "CVaR",
                    "y": "Expected Return",
                    "x_title": f"Historical CVaR {cvar_level:.0%} per return period",
                    "y_title": "Estimated expected return",
                    "frontier_label": "Minimum-CVaR frontier",
                },
                title="Return–CVaR opportunity set",
                height_in=3.3,
            ),
            ex.Table(
                named_points.set_index("Portfolio"),
                index_label="Portfolio",
                note="Named portfolio coordinates on the frontier chart.",
                formats={
                    "Expected Return": "{:.2%}",
                    "Volatility": "{:.2%}",
                    "Sharpe": "{:.2f}",
                },
            ),
        ]
        return blocks

    export_bar(
        pdf_builder(
            "Section 1 — Portfolio construction",
            guided("construction", _section1_blocks),
        ),
        stamp("section-1-construction"),
        excel_builder(_section1_tables),
        stamp("section-1-construction"),
        hint=(
            "Print-ready A4 landscape PDF of this section, or the underlying "
            "tables as a formatted workbook."
        ),
        key_prefix="s1",
    )
    SECTION_EXPORTS.append(
        (
            "Section 1 — Portfolio construction",
            guided("construction", _section1_blocks),
            _section1_tables,
        )
    )


# ============================================================
# SECTION 2 — MULTI-PERIOD ANALYSIS
# ============================================================

with tab_robustness:
    section(
        "Section 2",
        "Multi-period analysis",
        guide.SECTION_GUIDES["multi_period"].intro,
    )
    plain_english("multi_period")
    glossary_expander("multi_period")

    callout(
        "<b>How to read this:</b> 1Y is a realised-performance diagnostic only. "
        "It is deliberately <b>not</b> used to refit the strategic portfolios because "
        "roughly 12 monthly observations are too thin for a credible covariance/return estimate. "
        "The Reference Portfolio stays fixed; Equal Weight is mechanically unchanged when the "
        "universe is unchanged; HRP is rebuilt from each window's correlation/risk hierarchy."
    )

    block("Refit robustness — all construction methods", guide.caption("refit"))

    stability_display = multi_period_stability.copy()
    if reference_weights is not None and reference_returns is not None:
        ref = reference_weights.astype(float)
        ref = ref / float(ref.sum())
        ref_row = {
            "Largest Weight 3Y": float(ref.max()),
            "Actual ETFs 3Y": int((ref >= 0.0001).sum()),
            "Largest Weight 5Y": float(ref.max()),
            "Actual ETFs 5Y": int((ref >= 0.0001).sum()),
            "Largest Weight Full": float(ref.max()),
            "Actual ETFs Full": int((ref >= 0.0001).sum()),
            "Capital Change 3Y vs Full": 0.0,
            "Capital Change 5Y vs Full": 0.0,
        }
        stability_display = pd.concat(
            [pd.DataFrame([ref_row], index=["Reference Portfolio"]), stability_display]
        )

    if stability_display.empty:
        st.warning("No lookback refits converged for this run.")
    else:
        stability_cols = [
            "Capital Change 3Y vs Full",
            "Capital Change 5Y vs Full",
            "Largest Weight 3Y",
            "Largest Weight 5Y",
            "Largest Weight Full",
            "Actual ETFs 3Y",
            "Actual ETFs 5Y",
            "Actual ETFs Full",
        ]
        stability_display = stability_display.reindex(
            columns=[c for c in stability_cols if c in stability_display.columns]
        )
        tables.render(
            tables.style(
                stability_display,
                heat_columns=[
                    c
                    for c in ["Capital Change 3Y vs Full", "Capital Change 5Y vs Full"]
                    if c in stability_display.columns
                ],
                precision_overrides={
                    "Capital Change 3Y vs Full": "{:.1%}",
                    "Capital Change 5Y vs Full": "{:.1%}",
                    "Largest Weight 3Y": "{:.1%}",
                    "Largest Weight 5Y": "{:.1%}",
                    "Largest Weight Full": "{:.1%}",
                    "Actual ETFs 3Y": "{:.0f}",
                    "Actual ETFs 5Y": "{:.0f}",
                    "Actual ETFs Full": "{:.0f}",
                },
                highlight=["Reference Portfolio"],
            )
        )

    if not multi_period_weights.empty:
        refit_methods = list(
            dict.fromkeys(
                multi_period_weights.index.get_level_values("Method").tolist()
            )
        )
    else:
        refit_methods = []
    if reference_weights is not None and reference_returns is not None:
        refit_methods = ["Reference Portfolio", *refit_methods]

    if refit_methods:

        @st.fragment
        def _render_refit_weights() -> None:
            default_method = (
                "Diversified Maximum Sharpe"
                if "Diversified Maximum Sharpe" in refit_methods
                else refit_methods[0]
            )
            selected_method = st.selectbox(
                "Portfolio construction method to inspect",
                options=refit_methods,
                index=refit_methods.index(default_method),
                key="multi_period_refit_method",
                help="Changing this selector updates only this panel, not the full report.",
            )

            if selected_method == "Reference Portfolio":
                union_tickers = list(
                    dict.fromkeys(optimizer_tickers + list(reference_weights.index))
                )
                base = (
                    reference_weights.reindex(union_tickers).fillna(0.0).astype(float)
                )
                base = base / float(base.sum())
                method_weights = pd.DataFrame(
                    [base, base, base], index=["3Y", "5Y", "Full"]
                )
                st.caption(
                    "The Reference Portfolio is not re-optimised, so its starting weights are "
                    "intentionally identical across the three lookback labels."
                )
            else:
                try:
                    method_weights = (
                        multi_period_weights.xs(selected_method, level="Method")
                        .reindex(["3Y", "5Y", "Full"])
                        .dropna(how="all")
                    )
                except KeyError:
                    method_weights = pd.DataFrame()
                union_tickers = list(method_weights.columns)
                if selected_method == "Equal Weight":
                    st.caption(
                        "Equal Weight does not meaningfully refit when the ETF universe is unchanged; "
                        "its weights remain equal by construction."
                    )
                elif selected_method == HRP_NAME:
                    st.caption(
                        "HRP is rebuilt separately in every lookback from that window's covariance "
                        "and correlation hierarchy; expected returns are not used."
                    )

            if method_weights.empty:
                st.info("This method is unavailable on the selected lookbacks.")
                return

            tables.render(
                tables.style(
                    method_weights,
                    heat_columns=list(method_weights.columns),
                    default_format=tables.PCT1,
                )
            )
            chart(
                charts.weights_composition_chart(
                    analytics.weights_long(method_weights),
                    union_tickers,
                    list(method_weights.index),
                )
            )

        _render_refit_weights()

    if not multi_period_status.empty:
        failed_refits = multi_period_status.loc[~multi_period_status["Success"]]
        if not failed_refits.empty:
            with st.expander("Unavailable refits / solver details", expanded=False):
                tables.render(failed_refits)

    block(
        "Realised buy-and-hold metrics by period — all portfolios",
        guide.caption("realised_periods"),
    )

    if multi_period_metrics.empty:
        st.info("No multi-period metrics are available for this run.")
        mp_metrics_display = pd.DataFrame()
    else:
        mp_metrics_display = multi_period_metrics.copy()

        @st.fragment
        def _render_multi_period_metric_view() -> None:
            metric_options = [
                column
                for column in [
                    "CAGR",
                    "Volatility",
                    "Sharpe",
                    "Sortino",
                    "Max Drawdown",
                    CVAR_COLUMN,
                ]
                if column in mp_metrics_display.columns
            ]
            metric = st.selectbox(
                "Metric to compare across periods",
                options=metric_options,
                index=0,
                key="multi_period_metric",
                help="Changing this selector updates only this panel, not the full report.",
            )
            metric_table = mp_metrics_display.reset_index()[
                ["Lookback", "Portfolio", metric]
            ].pivot(index="Portfolio", columns="Lookback", values=metric)
            preferred = [
                x for x in ["1Y", "3Y", "5Y", "Full"] if x in metric_table.columns
            ]
            metric_table = metric_table.reindex(columns=preferred)
            portfolio_order = [
                name for name in all_static_portfolios if name in metric_table.index
            ]
            metric_table = metric_table.reindex(portfolio_order)
            ratio_metric = metric in {"Sharpe", "Sortino"}
            tables.render(
                tables.style(
                    metric_table,
                    heat_columns=list(metric_table.columns),
                    default_format=tables.RATIO2 if ratio_metric else tables.PCT1,
                    precision_overrides=(
                        dict.fromkeys(metric_table.columns, "{:.2f}")
                        if ratio_metric
                        else dict.fromkeys(metric_table.columns, "{:.1%}")
                    ),
                    highlight=["Reference Portfolio"],
                )
            )

        _render_multi_period_metric_view()

        with st.expander("Full multi-period metric table", expanded=False):
            tables.render(
                tables.style(
                    mp_metrics_display,
                    heat_columns=tables.metric_columns(mp_metrics_display),
                    highlight=["Reference Portfolio"],
                )
            )

    callout(
        "<b>Interpretation:</b> the refit test asks whether a construction rule changes "
        "materially when the estimation window changes. The realised-period test asks whether "
        "today's fixed allocation only looks attractive in one historical window. They are "
        "different questions, and neither is converted into a synthetic score."
    )

    def _section2_tables() -> dict[str, pd.DataFrame]:
        out = {
            "Refit stability summary": stability_display,
            "All refit weights": multi_period_weights,
            "Lookback fit status": multi_period_status,
        }
        if not multi_period_metrics.empty:
            out["B&H metrics by lookback"] = multi_period_metrics
        return out

    def _section2_blocks() -> list:
        blocks = [
            ex.Heading("Multi-period analysis", 1),
            ex.Text(
                "Every applicable construction method is refit on 3Y, 5Y and full "
                "history. Separately, realised buy-and-hold metrics are reported over "
                "1Y, 3Y, 5Y and full history for the complete portfolio set. The Reference "
                "stays fixed, and no synthetic robustness score or consensus portfolio is used."
            ),
            ex.Callout(
                "<b>1Y is diagnostic only.</b> It is not used to refit the strategic "
                "portfolios because the monthly sample is too small.",
                tone="brass",
            ),
        ]
        if not stability_display.empty:
            blocks.extend(
                [
                    ex.Heading("Refit robustness — all construction methods", 2),
                    ex.Text(
                        "Capital Change compares the 3Y or 5Y refit with the same method's "
                        "full-history starting allocation. 0% means identical weights."
                    ),
                    ex.Table(
                        stability_display,
                        index_label="Portfolio",
                        font_size=6.2,
                        full_width=True,
                    ),
                ]
            )
        if not multi_period_metrics.empty:
            blocks.extend(
                [
                    ex.Heading(
                        "Realised buy-and-hold metrics by period — all portfolios", 2
                    ),
                    ex.Table(
                        multi_period_metrics,
                        index_label="Lookback / portfolio",
                        font_size=5.7,
                        full_width=True,
                    ),
                ]
            )
        blocks.append(
            ex.Callout(
                "<b>Interpretation:</b> use the refit evidence to judge allocation stability "
                "and the realised-period evidence to judge historical window sensitivity. "
                "They answer different questions and are not combined into a single score.",
                tone="navy",
            )
        )
        return blocks

    export_bar(
        pdf_builder(
            "Section 2 — Multi-period analysis",
            guided("multi_period", _section2_blocks),
        ),
        stamp("section-2-multi-period"),
        excel_builder(_section2_tables),
        stamp("section-2-multi-period"),
        hint="All-method 3Y/5Y/full refits and all-portfolio 1Y/3Y/5Y/full B&H metrics.",
        key_prefix="s2",
    )
    SECTION_EXPORTS.append(
        (
            "Section 2 — Multi-period analysis",
            guided("multi_period", _section2_blocks),
            _section2_tables,
        )
    )


# ============================================================
# SECTION 3 — BOOTSTRAP SIMULATION
# ============================================================

bootstrap_result: BootstrapOutput | None = None
bootstrap_summary = pd.DataFrame()
bootstrap_probability_table = pd.DataFrame()

with tab_bootstrap:
    section(
        "Section 3",
        "Bootstrap simulation",
        guide.SECTION_GUIDES["bootstrap"].intro,
    )
    plain_english("bootstrap")
    glossary_expander("bootstrap")

    block("Simulation design", guide.caption("sim_design"))
    status_row(
        [
            (
                "Horizon",
                f"{bootstrap_horizon} years",
                f"{bootstrap_horizon * 12} simulated months",
            ),
            ("Paths", f"{bootstrap_simulations:,}", f"Seed {bootstrap_seed}"),
            (
                "Block length",
                f"{bootstrap_block} months",
                "1 = more reshuffling · 12 = more historical sequencing",
            ),
            (
                "Starting wealth",
                f"${bootstrap_initial_wealth:,.0f}",
                "Same for every portfolio",
                "brass",
            ),
        ]
    )
    st.caption(
        "Simulation always uses monthly returns. A 1-month block breaks the history into "
        "individual months; longer blocks preserve more of the historical order between "
        "neighbouring months. This explores path uncertainty — it is not a forecast and "
        "it is not out-of-sample validation."
    )

    # All available construction methods are simulated on every run.  The chart
    # selector below changes only what is displayed, not which methods are tested.
    simulation_portfolios = dict(all_static_portfolios)
    valid_portfolios: dict[str, pd.Series] = {}
    skipped_portfolios: list[str] = []
    for name, weight_vector in simulation_portfolios.items():
        missing_assets = [
            ticker
            for ticker, weight in weight_vector.items()
            if float(weight) > 1e-12 and ticker not in monthly_all.columns
        ]
        if missing_assets:
            skipped_portfolios.append(f"{name}: {', '.join(missing_assets)}")
        else:
            valid_portfolios[name] = weight_vector

    if skipped_portfolios:
        st.warning(
            "Skipped portfolios with holdings missing from the monthly common sample: "
            + " | ".join(skipped_portfolios)
        )

    chosen_portfolios = list(valid_portfolios)
    if not chosen_portfolios:
        st.info(
            "No complete portfolio could be simulated on the monthly common sample."
        )
    else:
        selected_frame = weights_frame(valid_portfolios)
        with st.spinner(
            f"Running {bootstrap_simulations:,} shared bootstrap paths for "
            f"{len(chosen_portfolios)} portfolios …"
        ):
            bootstrap_result = cached_bootstrap(
                monthly_all,
                selected_frame,
                int(bootstrap_simulations),
                int(bootstrap_horizon),
                int(bootstrap_block),
                int(bootstrap_seed),
                float(bootstrap_initial_wealth),
            )

    if bootstrap_result is not None:
        bootstrap_summary = bootstrap_result.summary

        block("Long-horizon outcome summary", guide.caption("boot_summary"))
        tables.render(
            tables.style(
                bootstrap_summary,
                heat_columns=[
                    "Median Terminal Wealth",
                    "5th Percentile Wealth",
                    "Median CAGR",
                    "P(Loss at Horizon)",
                ],
                highlight=["Reference Portfolio"],
            )
        )

        terminal_plot = bootstrap_summary.reset_index()[
            [
                "Portfolio",
                "5th Percentile Wealth",
                "25th Percentile Wealth",
                "Median Terminal Wealth",
                "75th Percentile Wealth",
                "95th Percentile Wealth",
            ]
        ]
        block(
            f"Where ${bootstrap_initial_wealth:,.0f} could end up after {bootstrap_horizon} years",
            guide.caption("boot_range"),
        )
        chart(charts.terminal_wealth_range_chart(terminal_plot, bootstrap_horizon))
        st.caption(
            "Simulated percentile ranges are not guaranteed confidence intervals. Future "
            "markets can contain environments that do not appear in the historical sample."
        )

        display_defaults = [
            name
            for name in [
                "Reference Portfolio",
                "Maximum Sharpe",
                "Diversified Maximum Sharpe",
                "Maximum Diversification",
                HRP_NAME,
            ]
            if name in chosen_portfolios
        ]

        @st.fragment
        def _render_bootstrap_distributions() -> None:
            display_portfolios = st.multiselect(
                "Portfolios to overlay in distribution charts",
                options=chosen_portfolios,
                default=display_defaults
                or chosen_portfolios[: min(5, len(chosen_portfolios))],
                help=(
                    "All portfolios are simulated regardless of this display filter. "
                    "Changing the selection updates only these charts."
                ),
                key="bootstrap_display_portfolios",
            )
            if not display_portfolios:
                display_portfolios = chosen_portfolios

            block("Terminal wealth distribution", guide.caption("boot_dist"))
            wealth_hist = analytics.histogram_frame(
                {
                    name: bootstrap_result.terminal_wealth[name]
                    for name in display_portfolios
                }
            )
            wealth_markers = analytics.percentile_markers(
                {
                    name: bootstrap_result.terminal_wealth[name]
                    for name in display_portfolios
                },
                percentiles=(5.0,),
            )
            chart(
                charts.distribution_chart(
                    wealth_hist,
                    wealth_markers,
                    value_format="$,.0f",
                    value_title="Terminal wealth",
                    reference=float(bootstrap_initial_wealth),
                    reference_label="Starting wealth",
                )
            )

            block("Annualised return distribution", guide.caption("boot_cagr"))
            cagr_hist = analytics.histogram_frame(
                {name: bootstrap_result.cagr[name] for name in display_portfolios}
            )
            chart(
                charts.distribution_chart(
                    cagr_hist,
                    value_format=".1%",
                    value_title="Annualised return over the horizon",
                    reference=0.0,
                    reference_label="0% CAGR",
                )
            )

        _render_bootstrap_distributions()

        block("Simulated maximum drawdown — all portfolios", guide.caption("boot_dd"))
        dd_hist = analytics.histogram_frame(
            {name: bootstrap_result.max_drawdown[name] for name in chosen_portfolios}
        )
        chart(
            charts.distribution_chart(
                dd_hist,
                value_format=".1%",
                value_title="Worst peak-to-trough decline",
            )
        )

        block("Path fan", guide.caption("fan"))

        export_fan_portfolio = (
            "Diversified Maximum Sharpe"
            if "Diversified Maximum Sharpe" in chosen_portfolios
            else chosen_portfolios[0]
        )
        export_fan = bootstrap_result.fans[export_fan_portfolio]

        @st.fragment
        def _render_bootstrap_fan() -> None:
            selected_fan_portfolio = st.selectbox(
                "Portfolio for path fan",
                options=chosen_portfolios,
                index=chosen_portfolios.index(export_fan_portfolio),
                key="bootstrap_fan_portfolio",
                help="Changing this dropdown updates only the path-fan panel.",
            )
            fan = bootstrap_result.fans[selected_fan_portfolio]
            chart(
                charts.fan_chart(
                    fan,
                    float(bootstrap_initial_wealth),
                    color=theme.portfolio_color(selected_fan_portfolio),
                )
            )

        _render_bootstrap_fan()

        prob_cols = [
            column
            for column in bootstrap_summary.columns
            if column == "P(Loss at Horizon)" or column.startswith("P(CAGR >")
        ]
        probability_labels = {
            "P(Loss at Horizon)": "Chance of Ending Below Start",
            **{
                col: col.replace("P(CAGR > ", "Chance CAGR > ").replace(")", "")
                for col in prob_cols
                if col.startswith("P(CAGR >")
            },
        }
        bootstrap_probability_table = bootstrap_summary[prob_cols].rename(
            columns=probability_labels
        )
        block("Outcome probabilities", guide.caption("probabilities"))
        tables.render(
            tables.style(
                bootstrap_probability_table,
                heat_columns=list(bootstrap_probability_table.columns),
                precision_overrides=dict.fromkeys(
                    bootstrap_probability_table.columns, "{:.1%}"
                ),
                highlight=["Reference Portfolio"],
            )
        )

        with st.expander("How to interpret bootstrap results", expanded=False):
            st.markdown(
                """
                **What it can tell you**

                Bootstrap analysis shows how the same starting portfolio behaves when the
                order of historically observed returns changes. It is useful for seeing
                sequence risk, downside ranges and how often long-horizon targets were met
                across many alternative paths.

                **What it cannot tell you**

                It does not know the future, cannot invent market regimes absent from the
                sample, and does not remove uncertainty about expected returns or covariance.
                It is also not out-of-sample testing — the next section addresses hindsight
                by repeatedly fitting portfolios only on information available at the time.
                """
            )

        def _section3_tables() -> dict[str, pd.DataFrame]:
            return {
                "Outcome summary": bootstrap_summary,
                "Outcome probabilities": bootstrap_probability_table,
            }

        def _section3_blocks() -> list:
            # The retail-facing PDF keeps the high-value evidence and removes
            # redundant statistical panels.  Detailed tables remain in Excel.
            return [
                ex.Heading("Bootstrap simulation", 1),
                ex.Text(
                    "Historical monthly return blocks are repeatedly rearranged to create "
                    "alternative long-term market sequences. Every portfolio is evaluated "
                    "on the same simulated paths. These are simulated outcomes, not forecasts."
                ),
                ex.KPIs(
                    [
                        (
                            "Horizon",
                            f"{bootstrap_horizon} years",
                            f"{bootstrap_horizon * 12} months",
                        ),
                        (
                            "Paths",
                            f"{bootstrap_simulations:,}",
                            f"Seed {bootstrap_seed}",
                        ),
                        (
                            "Block length",
                            f"{bootstrap_block} months",
                            "Contiguous historical blocks",
                        ),
                        (
                            "Starting wealth",
                            f"${bootstrap_initial_wealth:,.0f}",
                            "Same for every portfolio",
                        ),
                    ]
                ),
                ex.Heading("Long-horizon outcome summary", 2),
                ex.Table(
                    bootstrap_summary,
                    index_label="Portfolio",
                    highlight=["Reference Portfolio"],
                    font_size=5.8,
                    full_width=True,
                ),
                ex.Figure(
                    "range_dot",
                    {
                        "data": terminal_plot,
                        "cat": "Portfolio",
                        "lo": "25th Percentile Wealth",
                        "hi": "75th Percentile Wealth",
                        "mid": "Median Terminal Wealth",
                        "outer_lo": "5th Percentile Wealth",
                        "outer_hi": "95th Percentile Wealth",
                        "value_format": "money",
                        "color": {
                            name: theme.portfolio_color(name)
                            for name in terminal_plot["Portfolio"]
                        },
                        "x_title": f"Ending wealth after {bootstrap_horizon} years",
                    },
                    title=f"Where ${bootstrap_initial_wealth:,.0f} could end up after {bootstrap_horizon} years",
                    note="Median marker, middle 50% band and 5th–95th percentile range. Simulated outcomes, not forecasts.",
                    height_in=figure_height(len(terminal_plot), per_row=0.26),
                ),
                ex.PageBreak(),
                ex.Figure(
                    "distribution",
                    {
                        "data": dd_hist,
                        "value_format": "percent",
                        "x_title": "Worst peak-to-trough decline on the simulated path",
                    },
                    title="Simulated maximum drawdown — all portfolios",
                    height_in=2.8,
                ),
                ex.Figure(
                    "fan",
                    {
                        "data": export_fan,
                        "reference": float(bootstrap_initial_wealth),
                        "color": theme.portfolio_color(export_fan_portfolio),
                    },
                    title=f"Wealth path fan — {export_fan_portfolio}",
                    note="Central line = median wealth trajectory; shaded bands show wider simulated ranges.",
                    height_in=2.7,
                ),
                ex.Heading("Outcome probabilities", 2),
                ex.Table(
                    bootstrap_probability_table,
                    index_label="Portfolio",
                    highlight=["Reference Portfolio"],
                    formats=dict.fromkeys(
                        bootstrap_probability_table.columns, "{:.1%}"
                    ),
                    font_size=6.4,
                ),
                ex.Callout(
                    "<b>Interpretation:</b> bootstrap explores sequence and path risk using "
                    "the observed historical record. It is not a forecast and it is not "
                    "out-of-sample validation.",
                    tone="navy",
                ),
            ]

        export_bar(
            pdf_builder(
                "Section 3 — Bootstrap simulation",
                guided("bootstrap", _section3_blocks),
            ),
            stamp("section-3-bootstrap"),
            excel_builder(_section3_tables),
            stamp("section-3-bootstrap"),
            hint="Shared-path bootstrap outcomes for every available portfolio.",
            key_prefix="s3",
        )
        SECTION_EXPORTS.append(
            (
                "Section 3 — Bootstrap simulation",
                guided("bootstrap", _section3_blocks),
                _section3_tables,
            )
        )


# ============================================================
# SECTION 4 — TESTING WITHOUT HINDSIGHT
# ============================================================

wf_win_rates_numeric = pd.Series(dtype=float, name="Block Win Rate")
wf_win_counts = pd.DataFrame(columns=["Wins", "Tests", "Win Rate"], dtype=float)
wf_summary_display = pd.DataFrame()

with tab_walkforward:
    section(
        "Section 4",
        "Testing the portfolios without hindsight",
        guide.SECTION_GUIDES["walk_forward"].intro,
    )
    plain_english("walk_forward")
    glossary_expander("walk_forward")

    callout(
        f"<b>How the test works:</b> Look back {wf_lookback_label} → build the starting "
        f"portfolio → hold it for {wf_holding_months} months with no rebalancing → move "
        "to the next historical entry date and repeat. The Reference Portfolio is not "
        "re-optimised; it is simply held over the same test windows.",
        tone="navy",
    )

    status_row(
        [
            ("Lookback", wf_lookback_label, "Past information only"),
            (
                "Hold",
                f"{wf_holding_months} months",
                "Weights drift naturally",
            ),
            (
                "Rebalancing",
                "None",
                "Buy once at each historical entry",
                "brass",
            ),
            ("Validation data", "Monthly", "Strictly out of sample"),
        ]
    )

    if wf_result is None:
        st.error(f"Testing without hindsight could not run: {wf_error}")
    else:
        wf_names = [
            name
            for name in ["Reference Portfolio", *construction_methods, HRP_NAME]
            if name in wf_entry_stats.index
        ]
        wf_all_stats = wf_entry_stats.reindex(wf_names).dropna(how="all")
        first_oos = wf_result.returns.index.min()
        last_oos = wf_result.returns.index.max()
        completed_periods = len(wf_result.periods)

        if "Reference Portfolio" in set(wf_result.block_outcomes.get("Method", [])):
            # Counts, not just rates: a method that failed to fit in one fold has
            # fewer tests than the schedule, and the count has to be quoted
            # against the tests it actually took part in.
            wf_win_counts = (
                block_win_counts(wf_result, benchmark="Reference Portfolio")
                .reindex([name for name in wf_names if name != "Reference Portfolio"])
                .dropna()
            )
            wf_win_rates_numeric = wf_win_counts["Win Rate"].rename("Block Win Rate")

        rename_wf = {
            "Median Annualized Return": "Median Return",
            "Worst Annualized Return": "Worst Return",
            "Best Annualized Return": "Best Return",
            "Blocks": "Tests",
        }
        wf_summary_display = wf_all_stats.rename(columns=rename_wf).copy()
        if "Median Volatility" in wf_summary_display.columns:
            # Values are already decimal annualised volatility; the table formatter
            # now renders them as percentages rather than raw decimals.
            pass
        wf_summary_display["Beat Current Portfolio"] = "—"
        for name, tally in wf_win_counts.iterrows():
            wf_summary_display.loc[name, "Beat Current Portfolio"] = format_win_count(
                tally["Wins"], tally["Tests"]
            )

        block("Validation coverage", guide.caption("validation"))
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("First entry", first_oos.strftime("%b %Y"))
        with c2:
            st.metric("Last exit", last_oos.strftime("%b %Y"))
        with c3:
            st.metric("OOS months", len(wf_result.returns))
        with c4:
            st.metric("Independent tests", completed_periods)

        block(
            f"Independent {wf_holding_months}-month outcomes",
            guide.caption("wf_outcomes"),
        )
        if wf_holding_months < 12:
            st.caption(
                f"Returns are annualised from only {wf_holding_months} months of data per "
                "test, so best and worst figures swing widely. Compare portfolios with one "
                "another rather than reading any single number as a forecast."
            )
        if not wf_summary_display.empty:
            tables.render(
                tables.style(
                    wf_summary_display,
                    heat_columns=[
                        column
                        for column in [
                            "Median Return",
                            "Worst Return",
                            "Median Volatility",
                            "Median Max Drawdown",
                        ]
                        if column in wf_summary_display.columns
                    ],
                    precision_overrides={
                        "Median Return": "{:.1%}",
                        "Worst Return": "{:.1%}",
                        "Best Return": "{:.1%}",
                        "Median Volatility": "{:.1%}",
                        "Median Max Drawdown": "{:.1%}",
                        "Tests": "{:.0f}",
                    },
                    highlight=["Reference Portfolio"],
                )
            )

        # Keep audit detail accessible in the application without spending two
        # standalone pages on it in the retail-facing PDF.
        with st.expander(
            "Technical details — weight drift and test schedule", expanded=False
        ):
            wf_all_drift = wf_drift_stats.reindex(
                [name for name in wf_names if name in wf_drift_stats.index]
            ).dropna(how="all")
            if not wf_all_drift.empty:
                st.markdown("**Buy-and-hold weight drift**")
                st.caption(
                    "Weights are observed at the end of each holding period; nothing is traded "
                    "back to target."
                )
                tables.render(
                    tables.style(
                        wf_all_drift,
                        precision_overrides={"Blocks": "{:.0f}"},
                        highlight=["Reference Portfolio"],
                    )
                )

            st.markdown("**Training and holding windows**")
            schedule = wf_result.periods.copy()
            for col in ["Train Start", "Train End", "Test Start", "Test End"]:
                schedule[col] = pd.to_datetime(schedule[col]).dt.strftime("%b %Y")
            tables.render(schedule, hide_index=True)

            if not wf_result.weights.empty:
                available_methods = list(
                    dict.fromkeys(wf_result.weights["Method"].tolist())
                )

                @st.fragment
                def _render_walkforward_weights() -> None:
                    default_method = (
                        "Diversified Maximum Sharpe"
                        if "Diversified Maximum Sharpe" in available_methods
                        else available_methods[0]
                    )
                    selected_weight_method = st.selectbox(
                        "Starting allocations to inspect",
                        options=available_methods,
                        index=available_methods.index(default_method),
                        key="wf_weight_method",
                        help="Changing this dropdown updates only this technical-detail panel.",
                    )
                    wf_weight_view = wf_result.weights.loc[
                        wf_result.weights["Method"] == selected_weight_method
                    ].copy()
                    if not wf_weight_view.empty:
                        chart(
                            charts.weight_evolution_chart(
                                wf_weight_view,
                                optimizer_tickers,
                            )
                        )
                        st.caption(
                            "These are separate starting allocations at different historical dates. "
                            "Connecting them visually does not imply the strategy traded between tests."
                        )

                _render_walkforward_weights()

        failed_wf = wf_result.method_status.loc[~wf_result.method_status["Success"]]
        if not failed_wf.empty:
            with st.expander("Unavailable method runs", expanded=False):
                tables.render(
                    failed_wf[["Period", "Method", "Message"]], hide_index=True
                )

        def _section4_tables() -> dict[str, pd.DataFrame]:
            out = {
                "Out-of-sample summary": wf_all_stats,
                "Entry outcomes": wf_result.block_outcomes.set_index(
                    ["Period", "Method"]
                ),
                "Schedule": wf_result.periods.set_index("Period"),
            }
            if not wf_win_counts.empty:
                out["Win counts vs current"] = wf_win_counts
            return out

        def _section4_blocks() -> list:
            pdf_table = wf_all_stats.rename(columns=rename_wf).copy()
            pdf_table["Beat Current Portfolio"] = "—"
            for name, tally in wf_win_counts.iterrows():
                pdf_table.loc[name, "Beat Current Portfolio"] = format_win_count(
                    tally["Wins"], tally["Tests"]
                )
            return [
                ex.Heading("Testing the portfolios without hindsight", 1),
                ex.Text(
                    "At each historical entry date, the model uses only information that "
                    "was available then. It builds a starting allocation and holds it out "
                    "of sample without rebalancing. The Reference Portfolio is simply held "
                    "over the same periods rather than re-optimised."
                ),
                ex.Callout(
                    f"<b>{wf_lookback_label} lookback → build portfolio → "
                    f"{wf_holding_months}-month hold → repeat.</b> Holdings drift naturally "
                    "during each test.",
                    tone="navy",
                ),
                ex.KPIs(
                    [
                        (
                            "First entry",
                            first_oos.strftime("%b %Y"),
                            "First validated month",
                        ),
                        (
                            "Last exit",
                            last_oos.strftime("%b %Y"),
                            "Last validated month",
                        ),
                        (
                            "OOS months",
                            f"{len(wf_result.returns):,}",
                            "Strictly out of sample",
                        ),
                        (
                            "Independent tests",
                            str(completed_periods),
                            f"{wf_holding_months}-month holds",
                        ),
                    ]
                ),
                ex.Heading("Independent buy-and-hold outcomes", 2),
                ex.Table(
                    pdf_table,
                    index_label="Portfolio",
                    highlight=["Reference Portfolio"],
                    formats={
                        "Median Return": "{:.1%}",
                        "Worst Return": "{:.1%}",
                        "Best Return": "{:.1%}",
                        "Median Volatility": "{:.1%}",
                        "Median Max Drawdown": "{:.1%}",
                        "Tests": "{:.0f}",
                    },
                    font_size=5.7,
                    full_width=True,
                ),
                ex.Bullets(
                    [
                        "Median Return is the middle result across independent historical entry tests.",
                        "Worst and Best show sensitivity to the investor's starting date.",
                        "Median Volatility is the typical annualised amount of movement during a test.",
                        "Median Max Drawdown is the typical worst peak-to-trough fall inside a test.",
                        "Beat Current Portfolio counts how many tests outperformed the Reference Portfolio.",
                    ]
                ),
            ]

        export_bar(
            pdf_builder(
                "Section 4 — Testing without hindsight",
                guided("walk_forward", _section4_blocks),
            ),
            stamp("section-4-without-hindsight"),
            excel_builder(_section4_tables),
            stamp("section-4-without-hindsight"),
            hint="Independent out-of-sample buy-and-hold entry tests without hindsight.",
            key_prefix="s4",
        )
        SECTION_EXPORTS.append(
            (
                "Section 4 — Testing without hindsight",
                guided("walk_forward", _section4_blocks),
                _section4_tables,
            )
        )


# ============================================================
# SECTION 5 — DIVERSIFICATION & OVERLAP
# ============================================================

div_profile = pd.DataFrame()
diversification_portfolios: dict[str, pd.Series] = {}
return_div_named = pd.DataFrame()
return_div_frontier = pd.DataFrame()

with tab_diversification:
    section(
        "Section 5",
        "How diversified is each portfolio, really?",
        guide.SECTION_GUIDES["diversification"].intro,
    )
    plain_english("diversification")
    glossary_expander("diversification")

    div_corr = base_corr
    div_clusters = correlation_clusters(div_corr)

    reference_div_ready = reference_weights is not None and all(
        ticker in optimizer_tickers
        for ticker, weight in reference_weights.items()
        if float(weight) > 1e-12
    )

    if reference_weights is not None and not reference_div_ready:
        st.info(
            "The current/reference portfolio contains at least one holding outside the "
            "optimisation universe, so return-based diversification diagnostics are shown "
            "for the construction methods only."
        )

    if reference_div_ready:
        diversification_portfolios["Reference Portfolio"] = reference_weights
    diversification_portfolios.update(construction_portfolios)
    diversification_portfolios[HRP_NAME] = hrp_weights

    diversification_order = list(diversification_portfolios)
    div_weights = (
        pd.DataFrame(diversification_portfolios)
        .T.reindex(columns=optimizer_tickers)
        .fillna(0.0)
    )
    div_weights = div_weights.div(div_weights.sum(axis=1), axis=0)
    div_weights_long = analytics.weights_long(div_weights)

    div_profile = diversification_profile(
        diversification_portfolios, cov, div_corr, div_clusters
    )

    # Observed full-sample buy-and-hold volatility is added as a concrete risk
    # number next to the model-based concentration diagnostics.
    div_return_frame = pd.DataFrame(
        {
            name: buy_and_hold_portfolio_returns(
                optimizer_returns,
                pd.Series(weight_vector).reindex(optimizer_tickers).fillna(0.0),
            )
            for name, weight_vector in diversification_portfolios.items()
        }
    )
    div_hist_stats = cached_summary_table(
        div_return_frame, frequency, risk_free_rate, cvar_level
    )
    div_profile["Historical Volatility"] = div_hist_stats["Volatility"].reindex(
        div_profile.index
    )
    div_profile["Expected Return"] = [
        float(
            pd.Series(diversification_portfolios[name])
            .reindex(optimizer_tickers)
            .fillna(0.0)
            .to_numpy(float)
            @ mu.reindex(optimizer_tickers).to_numpy(float)
        )
        for name in div_profile.index
    ]

    div_risk_contrib = analytics.risk_contribution_frame(
        diversification_portfolios, cov
    )
    cluster_members = cluster_membership_frame(div_clusters)

    profile_columns = [
        "Actual ETFs",
        "Effective ETFs",
        "Historical Volatility",
        "Effective Risk Bets",
        "Weighted Correlation",
        "Diversification Ratio",
        "Largest Risk Contributor",
        "Largest Risk Share",
    ]
    profile_display = div_profile[profile_columns].rename(
        columns={"Effective ETFs": "Effective Holdings"}
    )

    block("Diversification profile", guide.caption("div_profile"))
    tables.render(
        tables.style(
            profile_display,
            heat_columns=[
                "Effective Holdings",
                "Historical Volatility",
                "Effective Risk Bets",
                "Weighted Correlation",
                "Diversification Ratio",
                "Largest Risk Share",
            ],
            precision_overrides={
                "Actual ETFs": "{:.0f}",
                "Effective Holdings": "{:.2f}",
                "Historical Volatility": "{:.1%}",
                "Effective Risk Bets": "{:.2f}",
                "Weighted Correlation": "{:.2f}",
                "Diversification Ratio": "{:.2f}",
                "Largest Risk Share": "{:.1%}",
            },
            highlight=["Reference Portfolio"],
        )
    )

    # ------------------------------------------------------------------
    # Return–Diversification Frontier
    # ------------------------------------------------------------------
    return_div_frontier = cached_return_diversification_frontier(
        mu, cov, max_weight, 55
    )
    return_div_named = div_profile.reset_index()[
        [
            "Portfolio",
            "Expected Return",
            "Diversification Ratio",
            "Effective ETFs",
            "Effective Risk Bets",
            "Largest Risk Share",
        ]
    ]
    label_names = [
        name
        for name in [
            "Reference Portfolio",
            "Maximum Sharpe",
            "Diversified Maximum Sharpe",
            "Maximum Diversification",
            HRP_NAME,
        ]
        if name in return_div_named["Portfolio"].tolist()
    ]

    block("Return–Diversification Frontier", guide.caption("div_frontier"))
    chart(
        charts.return_diversification_frontier_chart(
            return_div_frontier,
            return_div_named,
            label_names=label_names,
        )
    )
    callout(
        "<b>How to read it:</b> moving <b>up</b> means higher modelled expected return; "
        "moving <b>right</b> means a larger diversification ratio. The upper-right is "
        "attractive on these two dimensions, but the chart does not replace volatility, "
        "tail-risk, concentration or out-of-sample evidence. The curve is a trade-off "
        "boundary, not a recommendation.",
        tone="brass",
    )

    with st.expander(
        "ⓘ How the Return–Diversification Frontier is built", expanded=False
    ):
        st.markdown(
            r"""
            The frontier does **not** divide expected return by the diversification ratio.
            That would create an arbitrary score and can perversely penalise higher
            diversification.

            Instead, the engine chooses a series of minimum expected-return targets. For
            each target it solves:

            $$\max_w\; DR(w)$$

            subject to the portfolio expected return being at least the target, the weights
            summing to 100%, long-only weights and the normal maximum ETF weight.

            So each point answers: **"Among portfolios that can deliver at least this modelled
            expected return, which one gives the greatest diversification benefit?"**

            The Maximum Diversification portfolio is the diversification-first endpoint.
            As the required expected return rises, the optimiser progressively gives up
            diversification to meet the higher return target.
            """
        )

    with st.expander("ⓘ How HRP is constructed", expanded=False):
        st.markdown(
            """
            **Hierarchical Risk Parity (HRP)** is a diversification benchmark. It does not
            forecast which ETF will earn the highest return.

            1. The engine calculates historical correlations between the ETF return streams.
            2. ETFs that behave similarly are grouped into a hierarchy — like a family tree
               of closely and less-closely related return patterns.
            3. The hierarchy is repeatedly split into two branches.
            4. At each split, more capital is allocated to the lower-variance branch and less
               to the higher-variance branch.
            5. The process continues until every ETF has a weight, after which the normal
               maximum-weight constraint is applied if needed.

            No expected-return estimate enters those steps. The **E(r)** shown for HRP on
            the chart is calculated *after* HRP has been built, using the same expected-return
            assumptions as every other portfolio, purely so the methods can be compared on
            a common axis.
            """
        )
        tables.render(cluster_members)

    # The risk-contribution plot remains useful for interactive investigation,
    # but is intentionally collapsed and excluded from the retail-facing PDF so
    # it no longer consumes a standalone report page.
    with st.expander("Inspect where portfolio risk comes from", expanded=False):
        st.caption(
            "Capital weight and risk weight are not the same. This shows each ETF's share "
            "of portfolio volatility under the common covariance model."
        )
        chart(
            charts.risk_contribution_chart(
                div_risk_contrib, optimizer_tickers, diversification_order
            )
        )

    def _section5_tables() -> dict[str, pd.DataFrame]:
        return {
            "Diversification profile": profile_display,
            "Return-diversification frontier": return_div_frontier,
            "Starting allocations": div_weights,
            "Risk contribution": div_risk_contrib.pivot_table(
                index="Ticker", columns="Portfolio", values="Risk Share"
            ),
            "Correlation clusters": cluster_members,
        }

    def _section5_blocks() -> list:
        return [
            ex.Heading("How diversified is each portfolio, really?", 1),
            ex.Text(
                "The number of ETFs owned is only the starting point. These diagnostics "
                "separate literal holding count, capital concentration, risk concentration, "
                "return correlation and the volatility reduction achieved by combining assets."
            ),
            ex.Heading("Diversification profile", 2),
            ex.Table(
                profile_display,
                index_label="Portfolio",
                highlight=["Reference Portfolio"],
                formats={
                    "Actual ETFs": "{:.0f}",
                    "Effective Holdings": "{:.2f}",
                    "Historical Volatility": "{:.1%}",
                    "Effective Risk Bets": "{:.2f}",
                    "Weighted Correlation": "{:.2f}",
                    "Diversification Ratio": "{:.2f}",
                    "Largest Risk Share": "{:.1%}",
                },
                font_size=5.8,
                full_width=True,
            ),
            ex.Figure(
                "frontier",
                {
                    "frontier": return_div_frontier,
                    "named": return_div_named,
                    "x": "Diversification Ratio",
                    "y": "Expected Return",
                    "x_title": "Diversification ratio",
                    "y_title": "Expected return E(r)",
                    "x_percent": False,
                    "x_format": "%.2f",
                    "frontier_label": "Return–Diversification Frontier",
                },
                title="Return–Diversification Frontier",
                note=(
                    "For each minimum expected return, the line shows the maximum "
                    "diversification ratio available under the same long-only and weight-cap rules."
                ),
                height_in=3.2,
            ),
            ex.Bullets(
                [
                    "Actual ETFs is the literal holding count (weights above 0.01%; solver dust is ignored).",
                    "Effective Holdings measures how evenly capital is spread.",
                    "Effective Risk Bets measures how evenly portfolio variance is contributed.",
                    "Weighted Correlation measures return-stream overlap, not constituent overlap.",
                    "Diversification Ratio measures how much volatility is reduced by combining imperfectly correlated assets.",
                    "HRP is built from the correlation/risk hierarchy without using expected-return forecasts; its E(r) is calculated afterward for comparison only.",
                ]
            ),
        ]

    export_bar(
        pdf_builder(
            "Section 5 — Diversification & overlap",
            guided("diversification", _section5_blocks),
        ),
        stamp("section-5-diversification"),
        excel_builder(_section5_tables),
        stamp("section-5-diversification"),
        hint="Diversification diagnostics and the return–diversification frontier.",
        key_prefix="s5",
    )
    SECTION_EXPORTS.append(
        (
            "Section 5 — Diversification & overlap",
            guided("diversification", _section5_blocks),
            _section5_tables,
        )
    )


# ============================================================
# SECTION 6 — SUMMARY OF FINDINGS
# ============================================================

selection_scorecard = pd.DataFrame()
summary_deltas = pd.DataFrame()

with tab_decision:
    section(
        "Section 6",
        "Summary of Findings",
        guide.SECTION_GUIDES["summary"].intro,
    )
    plain_english("summary")
    glossary_expander("summary")

    selection_names = [
        name
        for name in [
            "Reference Portfolio",
            "Maximum Sharpe",
            "Diversified Maximum Sharpe",
            "Maximum Diversification",
            "Equal Weight",
            "Minimum Variance",
            "Risk Parity",
            "Minimum CVaR",
            "Maximum Return / CVaR",
            HRP_NAME,
        ]
        if name in all_static_portfolios
    ]
    selection_portfolios = {
        name: all_static_portfolios[name] for name in selection_names
    }

    if not selection_portfolios:
        st.info("No portfolios are available for the summary.")
    else:
        model_ready_names = [
            name
            for name, w in selection_portfolios.items()
            if all(
                ticker in optimizer_tickers
                for ticker, value in pd.Series(w).items()
                if float(value) > 1e-12
            )
        ]
        model_ready = {name: selection_portfolios[name] for name in model_ready_names}
        selection_model = (
            named_portfolio_points(
                model_ready, mu, cov, risk_free_rate=risk_free_rate
            ).set_index("Portfolio")
            if model_ready
            else pd.DataFrame()
        )

        # Historical buy-and-hold evidence uses the actual common return sample.
        selection_hist_series: dict[str, pd.Series] = {}
        for name, w in selection_portfolios.items():
            active_tickers = [
                ticker
                for ticker, value in pd.Series(w).items()
                if float(value) > 1e-12 and ticker in returns.columns
            ]
            if not active_tickers:
                continue
            selection_hist_series[name] = buy_and_hold_portfolio_returns(
                returns[active_tickers],
                pd.Series(w).reindex(active_tickers).fillna(0.0),
            )
        selection_hist_returns = pd.DataFrame(selection_hist_series)
        selection_hist = (
            cached_summary_table(
                selection_hist_returns, frequency, risk_free_rate, cvar_level
            )
            if not selection_hist_returns.empty
            else pd.DataFrame()
        )

        # Reuse the shared all-portfolio bootstrap from Section 3 where possible.
        if not bootstrap_summary.empty:
            selection_bootstrap = bootstrap_summary.reindex(
                [name for name in selection_names if name in bootstrap_summary.index]
            ).copy()
        else:
            selection_bootstrap = cached_bootstrap(
                monthly_all,
                weights_frame(selection_portfolios),
                int(bootstrap_simulations),
                int(bootstrap_horizon),
                int(bootstrap_block),
                int(bootstrap_seed),
                float(bootstrap_initial_wealth),
            ).summary

        selection_wf = (
            wf_entry_stats.reindex(
                [name for name in selection_names if name in wf_entry_stats.index]
            ).copy()
            if not wf_entry_stats.empty
            else pd.DataFrame()
        )
        selection_div = (
            div_profile.reindex(
                [name for name in selection_names if name in div_profile.index]
            ).copy()
            if not div_profile.empty
            else pd.DataFrame()
        )

        rows = []
        for name in selection_names:
            current = pd.Series(selection_portfolios[name], dtype=float)
            row: dict[str, object] = {"Portfolio": name}

            if name in selection_model.index:
                row.update(
                    {
                        "Expected Return E(r)": float(
                            selection_model.loc[name, "Expected Return"]
                        ),
                        "Expected Volatility": float(
                            selection_model.loc[name, "Volatility"]
                        ),
                        "Expected Sharpe": float(selection_model.loc[name, "Sharpe"]),
                    }
                )
            if name in selection_hist.index:
                row["Historical CAGR"] = float(selection_hist.loc[name, "CAGR"])
            if name in selection_bootstrap.index:
                row.update(
                    {
                        "Bootstrap Median CAGR": float(
                            selection_bootstrap.loc[name, "Median CAGR"]
                        ),
                        "Bootstrap 5th Wealth": float(
                            selection_bootstrap.loc[name, "5th Percentile Wealth"]
                        ),
                    }
                )
            if name in selection_wf.index:
                row["OOS Median Return"] = float(
                    selection_wf.loc[name, "Median Annualized Return"]
                )

            if name == "Reference Portfolio":
                row["OOS Win Rate vs Current"] = np.nan
                row["Beat Current Portfolio"] = "—"
            else:
                rate = float(wf_win_rates_numeric.get(name, np.nan))
                row["OOS Win Rate vs Current"] = rate
                if name in wf_win_counts.index and np.isfinite(rate):
                    tally = wf_win_counts.loc[name]
                    row["Beat Current Portfolio"] = format_win_count(
                        tally["Wins"], tally["Tests"]
                    )
                else:
                    row["Beat Current Portfolio"] = "—"

            if name in selection_div.index:
                row.update(
                    {
                        "Actual ETFs": float(selection_div.loc[name, "Actual ETFs"]),
                        "Effective Holdings": float(
                            selection_div.loc[name, "Effective ETFs"]
                        ),
                        "Effective Risk Bets": float(
                            selection_div.loc[name, "Effective Risk Bets"]
                        ),
                        "Diversification Ratio": float(
                            selection_div.loc[name, "Diversification Ratio"]
                        ),
                        "Largest Risk Share": float(
                            selection_div.loc[name, "Largest Risk Share"]
                        ),
                    }
                )

            if reference_weights is not None:
                universe = sorted(set(reference_weights.index) | set(current.index))
                a = reference_weights.reindex(universe).fillna(0.0).astype(float)
                b = current.reindex(universe).fillna(0.0).astype(float)
                if float(a.sum()) > 0 and float(b.sum()) > 0:
                    a = a / float(a.sum())
                    b = b / float(b.sum())
                    row["Capital to Reallocate"] = 0.5 * float(np.abs(a - b).sum())
            rows.append(row)

        selection_scorecard = pd.DataFrame(rows).set_index("Portfolio")

        return_columns = [
            column
            for column in [
                "Expected Return E(r)",
                "Expected Volatility",
                "Expected Sharpe",
                "Historical CAGR",
                "Bootstrap Median CAGR",
                "Bootstrap 5th Wealth",
                "OOS Median Return",
                "Beat Current Portfolio",
            ]
            if column in selection_scorecard.columns
        ]
        diversification_columns = [
            column
            for column in [
                "Actual ETFs",
                "Effective Holdings",
                "Effective Risk Bets",
                "Diversification Ratio",
                "Largest Risk Share",
                "Capital to Reallocate",
            ]
            if column in selection_scorecard.columns
        ]

        block("Return, risk and validation", guide.caption("summary_return"))
        tables.render(
            tables.style(
                selection_scorecard[return_columns],
                heat_columns=[
                    column
                    for column in [
                        "Expected Return E(r)",
                        "Expected Sharpe",
                        "Historical CAGR",
                        "Bootstrap Median CAGR",
                        "OOS Median Return",
                    ]
                    if column in return_columns
                ],
                precision_overrides={
                    "Expected Return E(r)": "{:.2%}",
                    "Expected Volatility": "{:.2%}",
                    "Expected Sharpe": "{:.2f}",
                    "Historical CAGR": "{:.1%}",
                    "Bootstrap Median CAGR": "{:.1%}",
                    "Bootstrap 5th Wealth": "${:,.0f}",
                    "OOS Median Return": "{:.1%}",
                },
                highlight=["Reference Portfolio"],
            )
        )

        block("Diversification and implementation", guide.caption("summary_div"))
        tables.render(
            tables.style(
                selection_scorecard[diversification_columns],
                heat_columns=[
                    column
                    for column in [
                        "Effective Holdings",
                        "Effective Risk Bets",
                        "Diversification Ratio",
                        "Largest Risk Share",
                        "Capital to Reallocate",
                    ]
                    if column in diversification_columns
                ],
                precision_overrides={
                    "Actual ETFs": "{:.0f}",
                    "Effective Holdings": "{:.2f}",
                    "Effective Risk Bets": "{:.2f}",
                    "Diversification Ratio": "{:.2f}",
                    "Largest Risk Share": "{:.1%}",
                    "Capital to Reallocate": "{:.1%}",
                },
                highlight=["Reference Portfolio"],
            )
        )

        # Explicit deltas make the Reference useful as an anchor without prose
        # that argues for or against retaining it.
        if "Reference Portfolio" in selection_scorecard.index:
            ref = selection_scorecard.loc["Reference Portfolio"]
            delta_specs = [
                ("Expected Return E(r)", "Δ E(r)"),
                ("Expected Volatility", "Δ Expected Volatility"),
                ("Effective Risk Bets", "Δ Risk Bets"),
                ("Diversification Ratio", "Δ DR"),
                ("Largest Risk Share", "Δ Largest Risk Share"),
            ]
            delta_data = {}
            for source, label in delta_specs:
                if source in selection_scorecard.columns and pd.notna(ref.get(source)):
                    delta_data[label] = pd.to_numeric(
                        selection_scorecard[source], errors="coerce"
                    ) - float(ref[source])
            if "Capital to Reallocate" in selection_scorecard.columns:
                delta_data["Capital to Reallocate"] = pd.to_numeric(
                    selection_scorecard["Capital to Reallocate"], errors="coerce"
                )
            summary_deltas = pd.DataFrame(delta_data, index=selection_scorecard.index)

            with st.expander(
                "Compare each portfolio with the current allocation", expanded=False
            ):
                st.caption(
                    "Positive and negative signs are simple differences versus the current/reference "
                    "portfolio; they are not scores."
                )
                tables.render(
                    tables.style(
                        summary_deltas,
                        precision_overrides={
                            "Δ E(r)": "{:+.2%}",
                            "Δ Expected Volatility": "{:+.2%}",
                            "Δ Risk Bets": "{:+.2f}",
                            "Δ DR": "{:+.2f}",
                            "Δ Largest Risk Share": "{:+.1%}",
                            "Capital to Reallocate": "{:.1%}",
                        },
                        highlight=["Reference Portfolio"],
                    )
                )

        block("Starting allocation comparison", guide.caption("alloc_compare"))
        allocation_defaults = [
            name
            for name in [
                "Reference Portfolio",
                "Maximum Sharpe",
                "Diversified Maximum Sharpe",
                "Maximum Diversification",
            ]
            if name in selection_names
        ]
        export_allocation_view = (
            allocation_defaults or selection_names[: min(4, len(selection_names))]
        )
        selection_weights = (
            pd.DataFrame(
                {name: selection_portfolios[name] for name in export_allocation_view}
            )
            .T.reindex(columns=optimizer_tickers)
            .fillna(0.0)
        )

        @st.fragment
        def _render_summary_allocations() -> None:
            allocation_view = st.multiselect(
                "Portfolios to compare",
                options=selection_names,
                default=export_allocation_view,
                key="summary_allocation_view",
                help="Changing this selection updates only the allocation chart.",
            )
            if not allocation_view:
                allocation_view = export_allocation_view or selection_names[:1]
            display_weights = (
                pd.DataFrame(
                    {name: selection_portfolios[name] for name in allocation_view}
                )
                .T.reindex(columns=optimizer_tickers)
                .fillna(0.0)
            )
            chart(
                charts.weights_composition_chart(
                    analytics.weights_long(display_weights),
                    optimizer_tickers,
                    allocation_view,
                    category="Method",
                    height=max(230, 44 * len(allocation_view)),
                )
            )

        _render_summary_allocations()

        with st.expander("ⓘ How to use this summary", expanded=False):
            st.markdown(
                """
                There is deliberately no combined score and no automatically selected winner.
                Expected-return efficiency, historical performance, bootstrap downside,
                out-of-sample evidence, diversification and implementation distance answer
                different questions. Use the earlier sections when a number here needs context,
                then decide which trade-offs matter for your own research objective.
                """
            )

        def _section6_tables() -> dict[str, pd.DataFrame]:
            out = {
                "Summary of findings": selection_scorecard,
                "Starting allocations shown": selection_weights,
                # Detailed evidence is retained in the workbook as an audit appendix
                # but no longer consumes a standalone main-report page.
                "Appendix - Bootstrap evidence": selection_bootstrap,
            }
            if not selection_wf.empty:
                out["Appendix - Walk-forward evidence"] = selection_wf
            if not selection_div.empty:
                out["Appendix - Diversification evidence"] = selection_div
            if not summary_deltas.empty:
                out["Comparison vs current"] = summary_deltas
            return out

        def _section6_blocks() -> list:
            return [
                ex.Heading("Summary of Findings", 1),
                ex.Text(
                    "This page brings together the main findings from portfolio construction, "
                    "historical analysis, simulations, out-of-sample testing and diversification. "
                    "It summarises trade-offs; it does not select a portfolio for the reader."
                ),
                ex.Heading("Return, risk and validation", 2),
                ex.Table(
                    selection_scorecard[return_columns],
                    index_label="Portfolio",
                    highlight=["Reference Portfolio"],
                    formats={
                        "Expected Return E(r)": "{:.2%}",
                        "Expected Volatility": "{:.2%}",
                        "Expected Sharpe": "{:.2f}",
                        "Historical CAGR": "{:.1%}",
                        "Bootstrap Median CAGR": "{:.1%}",
                        "Bootstrap 5th Wealth": "${:,.0f}",
                        "OOS Median Return": "{:.1%}",
                    },
                    font_size=5.4,
                    full_width=True,
                ),
                ex.Heading("Diversification and implementation", 2),
                ex.Table(
                    selection_scorecard[diversification_columns],
                    index_label="Portfolio",
                    highlight=["Reference Portfolio"],
                    formats={
                        "Actual ETFs": "{:.0f}",
                        "Effective Holdings": "{:.2f}",
                        "Effective Risk Bets": "{:.2f}",
                        "Diversification Ratio": "{:.2f}",
                        "Largest Risk Share": "{:.1%}",
                        "Capital to Reallocate": "{:.1%}",
                    },
                    font_size=5.8,
                    full_width=True,
                ),
                ex.Heading("Starting allocation comparison", 2),
                ex.Figure(
                    "stacked_barh",
                    {
                        "data": analytics.weights_long(selection_weights),
                        "cat": "Method",
                        "value": "Weight",
                        "series": "Ticker",
                        "order": export_allocation_view,
                        "x_title": "Starting allocation",
                    },
                    note="Starting weights only; buy-and-hold weights are allowed to drift afterward.",
                    height_in=figure_height(len(export_allocation_view), per_row=0.31),
                ),
            ]

        export_bar(
            pdf_builder(
                "Section 6 — Summary of Findings", guided("summary", _section6_blocks)
            ),
            stamp("section-6-summary-of-findings"),
            excel_builder(_section6_tables),
            stamp("section-6-summary-of-findings"),
            hint="Neutral summary of the evidence, trade-offs and starting allocations.",
            key_prefix="s6",
        )
        SECTION_EXPORTS.append(
            (
                "Section 6 — Summary of Findings",
                guided("summary", _section6_blocks),
                _section6_tables,
            )
        )


# ============================================================
# COMPLETE RESEARCH REPORT
# ============================================================

st.markdown("---")
section(
    "Export",
    "Complete research report",
    "All six sections in one paginated, print-ready document, with the run "
    "parameters recorded on the cover so the output is reproducible.",
)


def _full_report_blocks() -> list:
    blocks: list = []
    for index, (_, factory, _unused) in enumerate(SECTION_EXPORTS):
        if index:
            blocks.append(ex.PageBreak())
        blocks.extend(factory(with_terms=False))
    blocks.extend(guide.pdf_glossary())
    return blocks


def _full_report_tables() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for label, _unused, factory in SECTION_EXPORTS:
        prefix = label.split("—")[0].strip().replace("Section ", "S")
        for name, frame in factory().items():
            out[f"{prefix} {name}"] = frame
    return out


_full_report_pdf = pdf_builder(
    "Full research report",
    _full_report_blocks,
    contents=tuple(label for label, _, _ in SECTION_EXPORTS),
)
_full_report_xlsx = excel_builder(_full_report_tables)
st.session_state["full_report_exports"] = (_full_report_pdf, _full_report_xlsx)

export_bar(
    _full_report_pdf,
    stamp("full-research-report"),
    _full_report_xlsx,
    stamp("full-research-report"),
    hint=(
        f"{len(SECTION_EXPORTS)} of 6 sections are available for this run. "
        "The PDF is built when you click, so it always reflects what is on screen."
    ),
    key_prefix="full",
)

footer(f"Run generated {GENERATED_AT:%d %b %Y %H:%M} from the parameters shown above.")

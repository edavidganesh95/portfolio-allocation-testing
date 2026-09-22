"""Streamlit UI components for the Ledger design system.

Base colours come from ``.streamlit/config.toml`` so Streamlit's own widgets
are themed natively; the CSS here is component polish on top of that, and every
value resolves from :mod:`src.theme`.
"""

from __future__ import annotations

import html
from collections.abc import Callable, Iterable, Sequence

import streamlit as st

from . import guide, theme

PRIMARY = theme.NAVY
ACCENT = theme.BRASS
TEXT = theme.INK
MUTED = theme.MUTED
BORDER = theme.BORDER
SURFACE = theme.SURFACE


def apply_theme() -> None:
    """Inject component CSS. Safe to call once per script run."""
    st.markdown(
        f"""
        <style>
        :root {{
            --ink: {theme.INK};
            --ink-soft: {theme.INK_SOFT};
            --muted: {theme.MUTED};
            --paper: {theme.PAPER};
            --surface: {theme.SURFACE};
            --surface-alt: {theme.SURFACE_ALT};
            --border: {theme.BORDER};
            --border-strong: {theme.BORDER_STRONG};
            --navy: {theme.NAVY};
            --navy-deep: {theme.NAVY_DEEP};
            --navy-soft: {theme.NAVY_SOFT};
            --brass: {theme.BRASS};
            --brass-soft: {theme.BRASS_SOFT};
            --positive: {theme.POSITIVE};
            --negative: {theme.NEGATIVE};
            --caution: {theme.CAUTION};
            /* Streamlit's toolbar strip, which overlays the top of the page. */
            --app-toolbar-height: 3.75rem;
        }}

        .block-container {{
            max-width: 1560px;
            /* The toolbar overlays the top of the page, so the first block has
               to clear it or the masthead sits underneath. */
            padding-top: calc(var(--app-toolbar-height) + 1rem);
            padding-bottom: 4.5rem;
        }}

        /* ---------- masthead ---------- */
        .masthead {{
            display: flex;
            align-items: flex-start;
            gap: .95rem;
            padding-bottom: .85rem;
            border-bottom: 1px solid var(--border);
            margin-bottom: 1.35rem;
        }}
        .masthead-mark {{
            flex: 0 0 auto;
            width: 42px;
            height: 42px;
            border-radius: 11px;
            background: var(--navy-deep);
            color: #fff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: -.02em;
            box-shadow: inset 0 -3px 0 0 var(--brass);
        }}
        .masthead-text {{ min-width: 0; }}
        .page-title {{
            font-size: clamp(1.55rem, 2.3vw, 2.15rem);
            line-height: 1.1;
            letter-spacing: -.032em;
            font-weight: 750;
            margin: 0 0 .16rem;
            color: var(--ink);
        }}
        .page-subtitle {{
            color: var(--muted);
            font-size: .93rem;
            line-height: 1.5;
            margin: 0;
            max-width: 74ch;
        }}

        /* ---------- section headers ---------- */
        .section-kicker {{
            color: var(--brass);
            font-size: .67rem;
            font-weight: 750;
            letter-spacing: .13em;
            text-transform: uppercase;
            margin: .2rem 0 .3rem;
        }}
        .section-title {{
            font-size: 1.5rem;
            font-weight: 730;
            letter-spacing: -.028em;
            margin: 0 0 .3rem;
            color: var(--ink);
        }}
        .section-copy {{
            color: var(--ink-soft);
            max-width: 92ch;
            line-height: 1.58;
            font-size: .92rem;
            margin-bottom: .55rem;
        }}
        .section-rule {{
            height: 2px;
            width: 54px;
            background: var(--brass);
            border-radius: 2px;
            margin: 0 0 1.15rem;
        }}

        /* ---------- block headings ---------- */
        .block-title {{
            font-size: 1.02rem;
            font-weight: 700;
            letter-spacing: -.012em;
            color: var(--ink);
            margin: 1.5rem 0 .2rem;
            padding-top: .2rem;
        }}
        .block-caption {{
            color: var(--muted);
            font-size: .815rem;
            line-height: 1.55;
            max-width: 100ch;
            margin: 0 0 .6rem;
        }}

        /* ---------- status cards ---------- */
        .status-card {{
            border: 1px solid var(--border);
            border-top: 2px solid var(--card-accent, var(--navy));
            background: var(--surface);
            padding: .82rem .92rem .88rem;
            min-height: 98px;
            border-radius: 10px;
        }}
        .status-card .label {{
            font-size: .64rem;
            letter-spacing: .09em;
            color: var(--muted);
            text-transform: uppercase;
            font-weight: 750;
        }}
        .status-card .value {{
            font-size: 1.16rem;
            font-weight: 720;
            line-height: 1.25;
            margin-top: .3rem;
            color: var(--ink);
            word-break: break-word;
        }}
        .status-card .sub {{
            margin-top: .22rem;
            color: var(--muted);
            font-size: .745rem;
            line-height: 1.4;
        }}

        div[data-testid="stMetric"] {{
            border: 1px solid var(--border);
            border-top: 2px solid var(--navy);
            background: var(--surface);
            padding: .7rem .85rem .75rem;
            border-radius: 10px;
        }}
        div[data-testid="stMetric"] label p {{
            font-size: .64rem !important;
            letter-spacing: .09em;
            text-transform: uppercase;
            font-weight: 750;
            color: var(--muted) !important;
            white-space: normal;
            overflow: visible;
            text-overflow: clip;
            line-height: 1.25;
        }}
        div[data-testid="stMetric"] label {{ overflow: visible; }}
        div[data-testid="stMetricValue"] div {{
            white-space: normal;
            overflow: visible;
            text-overflow: clip;
        }}
        div[data-testid="stMetricValue"] {{
            font-size: 1.24rem !important;
            font-weight: 720;
            color: var(--ink);
        }}

        /* ---------- plain-English guide ---------- */
        .plain-card {{
            border: 1px solid var(--border);
            border-top: 3px solid var(--brass);
            background: var(--surface);
            border-radius: 10px;
            padding: .85rem 1.05rem 1rem;
            margin: .35rem 0 .8rem;
        }}
        .plain-head {{
            font-size: .66rem;
            letter-spacing: .13em;
            text-transform: uppercase;
            font-weight: 750;
            color: var(--brass);
            margin-bottom: .55rem;
        }}
        .plain-grid {{
            display: grid;
            grid-template-columns: 1.15fr 1.1fr 1fr;
            gap: 1.35rem;
        }}
        .plain-label {{
            font-size: .74rem;
            font-weight: 720;
            color: var(--navy-deep);
            margin-bottom: .25rem;
        }}
        .plain-card p, .plain-card li {{
            font-size: .85rem;
            line-height: 1.55;
            color: var(--ink-soft);
            margin: 0;
        }}
        .plain-card ul {{ margin: 0; padding-left: 1.05rem; }}
        .plain-card li {{ margin-bottom: .3rem; }}
        @media (max-width: 1050px) {{
            .plain-grid {{ grid-template-columns: 1fr; gap: .9rem; }}
        }}

        .glossary {{ margin: 0; }}
        .glossary dt {{
            font-weight: 720;
            color: var(--navy-deep);
            font-size: .9rem;
            margin-top: .95rem;
        }}
        .glossary dt:first-child {{ margin-top: .2rem; }}
        .glossary dd {{
            margin: .15rem 0 0;
            font-size: .85rem;
            line-height: 1.55;
            color: var(--ink-soft);
            max-width: 98ch;
        }}
        .glossary dd.ex, .glossary dd.lf {{ color: var(--muted); }}
        .glossary dd.fm {{
            font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
            font-size: .77rem;
            color: var(--muted);
        }}

        /* ---------- start here (landing page) ---------- */
        .start-intro {{
            font-size: 1rem;
            line-height: 1.62;
            color: var(--ink-soft);
            max-width: 88ch;
            margin: .2rem 0 1.15rem;
        }}
        .start-label {{
            font-size: .66rem;
            letter-spacing: .13em;
            text-transform: uppercase;
            font-weight: 750;
            color: var(--brass);
            margin: 1.2rem 0 .55rem;
        }}
        .start-steps, .idea-strip {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: .8rem;
        }}
        .start-step, .idea {{
            border: 1px solid var(--border);
            background: var(--surface);
            border-radius: 10px;
            padding: .8rem .9rem .9rem;
        }}
        .start-step .n {{
            width: 1.5rem;
            height: 1.5rem;
            border-radius: 50%;
            background: var(--navy);
            color: #fff;
            font-weight: 700;
            font-size: .8rem;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: .45rem;
        }}
        .start-step b, .idea b {{
            display: block;
            color: var(--navy-deep);
            font-size: .9rem;
            margin-bottom: .2rem;
        }}
        .start-step p, .idea span {{
            font-size: .83rem;
            line-height: 1.5;
            color: var(--ink-soft);
            margin: 0;
        }}
        .idea {{ border-top: 3px solid var(--brass); }}
        .pages-list {{
            border: 1px solid var(--border);
            border-radius: 10px;
            background: var(--surface);
            overflow: hidden;
        }}
        .pages-list div {{
            display: grid;
            grid-template-columns: 4.2rem 15rem 1fr;
            gap: .6rem;
            padding: .55rem .9rem;
            border-bottom: 1px solid var(--border);
            font-size: .84rem;
            line-height: 1.45;
            color: var(--ink-soft);
        }}
        .pages-list div:last-child {{ border-bottom: 0; }}
        .pages-list b {{ color: var(--navy-deep); }}
        .pages-list .pg {{ color: var(--brass); font-weight: 700; }}
        @media (max-width: 1050px) {{
            .start-steps, .idea-strip {{ grid-template-columns: 1fr 1fr; }}
            .pages-list div {{ grid-template-columns: 3.5rem 1fr; }}
            .pages-list div span:last-child {{ grid-column: 1 / -1; }}
        }}

        /* ---------- callout ---------- */
        .soft-callout {{
            border: 1px solid var(--callout-border, #D7E1EE);
            border-left: 3px solid var(--callout-accent, var(--navy));
            border-radius: 9px;
            background: var(--callout-bg, var(--navy-soft));
            padding: .82rem .95rem;
            margin: .7rem 0 1rem;
            color: var(--ink);
            font-size: .875rem;
            line-height: 1.56;
            max-width: 108ch;
        }}

        /* ---------- methodology chips ---------- */
        .methodology-strip {{
            display: flex;
            flex-wrap: wrap;
            gap: .42rem;
            margin: .3rem 0 1rem;
        }}
        .methodology-chip {{
            display: inline-flex;
            align-items: center;
            gap: .4rem;
            border: 1px solid var(--border);
            background: var(--surface);
            border-radius: 999px;
            padding: .3rem .58rem .3rem .68rem;
            font-size: .755rem;
            font-weight: 620;
            color: var(--ink-soft);
        }}
        .methodology-chip:hover {{ border-color: var(--border-strong); }}
        .info-dot {{
            width: 16px;
            height: 16px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            background: var(--navy-soft);
            color: var(--navy);
            font-size: .63rem;
            font-weight: 800;
            cursor: help;
        }}

        /* ---------- legend key ---------- */
        .legend-key {{
            display: flex;
            flex-wrap: wrap;
            gap: .2rem 1rem;
            margin: .1rem 0 .7rem;
            font-size: .765rem;
            color: var(--ink-soft);
        }}
        .legend-key span.item {{ display: inline-flex; align-items: center; gap: .36rem; }}
        .legend-key i {{
            width: 11px;
            height: 11px;
            border-radius: 3px;
            display: inline-block;
        }}

        /* ---------- tables & charts ---------- */
        [data-testid="stDataFrame"] {{
            border: 1px solid var(--border);
            border-radius: 9px;
            overflow: hidden;
        }}
        div[data-testid="stAlert"] {{ border-radius: 9px; }}

        /* ---------- tabs ---------- */
        div[data-baseweb="tab-list"] {{
            gap: .12rem;
            border-bottom: 1px solid var(--border);
            background: var(--paper);
            padding-top: .2rem;
            overflow-x: auto;
            flex-wrap: nowrap;
            scrollbar-width: thin;
        }}

        /* Keep the section tabs reachable from anywhere in a long section.
           The tab list's own wrapper is exactly as tall as the tabs, so a
           sticky element there has nowhere to travel; the wrapper one level up
           spans the whole tab group and is the element that has to stick. It
           is offset by the toolbar height so the tabs park below it rather
           than underneath it. If Streamlit's DOM changes, :has() simply stops
           matching and the tabs scroll normally. */
        div[data-testid="stTabs"] div:has(> div[data-baseweb="tab-list"]) {{
            position: sticky;
            top: var(--app-toolbar-height);
            z-index: 20;
            background: var(--paper);
        }}
        button[data-baseweb="tab"] {{
            font-weight: 650;
            font-size: .875rem;
            letter-spacing: -.005em;
            padding: .55rem .9rem;
            color: var(--muted);
            white-space: nowrap;
            flex: 0 0 auto;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{ color: var(--navy); }}

        /* ---------- sidebar ---------- */
        section[data-testid="stSidebar"] .block-container {{ padding-top: 1.1rem; }}
        .sidebar-title {{
            font-size: .95rem;
            font-weight: 730;
            letter-spacing: -.015em;
            color: var(--ink);
            margin-bottom: .1rem;
        }}
        .sidebar-note {{
            font-size: .745rem;
            color: var(--muted);
            line-height: 1.45;
            margin-bottom: .7rem;
        }}
        section[data-testid="stSidebar"] summary p {{
            font-weight: 650;
            font-size: .845rem;
        }}

        /* ---------- export bar ---------- */
        .export-bar {{
            border: 1px solid var(--border);
            border-left: 3px solid var(--brass);
            background: var(--surface);
            border-radius: 9px;
            padding: .72rem .9rem .3rem;
            margin: 1.4rem 0 .5rem;
        }}
        .export-bar .title {{
            font-size: .67rem;
            letter-spacing: .11em;
            text-transform: uppercase;
            font-weight: 750;
            color: var(--muted);
            margin-bottom: .1rem;
        }}
        .export-bar .hint {{
            font-size: .765rem;
            color: var(--muted);
            line-height: 1.45;
            margin-bottom: .55rem;
        }}
        div[data-testid="stDownloadButton"] button p {{
            white-space: nowrap;
        }}

        /* ---------- footer ---------- */
        .app-footer {{
            margin-top: 2.6rem;
            padding-top: .9rem;
            border-top: 1px solid var(--border);
            color: var(--muted);
            font-size: .745rem;
            line-height: 1.55;
            max-width: 108ch;
        }}

        hr {{ border-color: var(--border) !important; margin: 1.8rem 0 !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _esc(value) -> str:
    return html.escape(str(value))


# ----------------------------------------------------------------------
# Structure
# ----------------------------------------------------------------------


def page_header(
    title: str = "Portfolio Allocation Testing",
    subtitle: str = (
        "A returns-only portfolio research engine for robust, buy-and-hold "
        "strategic ETF allocation."
    ),
    mark: str = "PA",
) -> None:
    st.markdown(
        f"""
        <div class="masthead">
            <div class="masthead-mark">{_esc(mark)}</div>
            <div class="masthead-text">
                <div class="page-title">{_esc(title)}</div>
                <p class="page-subtitle">{_esc(subtitle)}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(kicker: str, title: str, copy: str | None = None) -> None:
    parts = [
        f'<div class="section-kicker">{_esc(kicker)}</div>',
        f'<div class="section-title">{_esc(title)}</div>',
    ]
    if copy:
        parts.append(f'<div class="section-copy">{_esc(copy)}</div>')
    parts.append('<div class="section-rule"></div>')
    st.markdown("".join(parts), unsafe_allow_html=True)


def block(title: str, caption: str | None = None) -> None:
    """Sub-heading inside a section, with an optional explanatory line."""
    st.markdown(
        f'<div class="block-title">{_esc(title)}</div>',
        unsafe_allow_html=True,
    )
    if caption:
        st.markdown(
            f'<p class="block-caption">{_esc(caption)}</p>',
            unsafe_allow_html=True,
        )


def status_card(label, value, sub: str = "", tone: str = "navy") -> None:
    accent = {
        "navy": theme.NAVY,
        "brass": theme.BRASS,
        "positive": theme.POSITIVE,
        "negative": theme.NEGATIVE,
        "caution": theme.CAUTION,
        "muted": theme.BORDER_STRONG,
    }.get(tone, theme.NAVY)
    st.markdown(
        f"""
        <div class="status-card" style="--card-accent: {accent};">
            <div class="label">{_esc(label)}</div>
            <div class="value">{_esc(value)}</div>
            <div class="sub">{_esc(sub)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_row(cards: Sequence[tuple]) -> None:
    """Render a row of status cards from ``(label, value, sub[, tone])``."""
    if not cards:
        return
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards, strict=True):
        label, value, sub = card[0], card[1], card[2] if len(card) > 2 else ""
        tone = card[3] if len(card) > 3 else "navy"
        with column:
            status_card(label, value, sub, tone)


def callout(html_text: str, tone: str = "navy") -> None:
    accent, background, border = {
        "navy": (theme.NAVY, theme.NAVY_SOFT, "#D7E1EE"),
        "brass": (theme.BRASS, theme.BRASS_SOFT, "#EBDCC4"),
        "positive": (theme.POSITIVE, theme.POSITIVE_SOFT, "#C9E2D6"),
        "caution": (theme.CAUTION, theme.CAUTION_SOFT, "#EBDCB4"),
        "negative": (theme.NEGATIVE, theme.NEGATIVE_SOFT, "#EFCFCB"),
    }.get(tone, (theme.NAVY, theme.NAVY_SOFT, "#D7E1EE"))
    st.markdown(
        f'<div class="soft-callout" style="--callout-accent:{accent};'
        f'--callout-bg:{background};--callout-border:{border};">{html_text}</div>',
        unsafe_allow_html=True,
    )


def plain_english(section_key: str) -> None:
    """Always-visible summary at the top of a page, in ordinary language."""
    guide_for_page = guide.SECTION_GUIDES[section_key]
    look = "".join(f"<li>{_esc(item)}</li>" for item in guide_for_page.look_for)
    cannot = "".join(f"<li>{_esc(item)}</li>" for item in guide_for_page.cannot)
    st.markdown(
        '<div class="plain-card">'
        '<div class="plain-head">In plain English</div>'
        '<div class="plain-grid">'
        '<div><div class="plain-label">What this page does</div>'
        f"<p>{_esc(guide_for_page.what)}</p></div>"
        '<div><div class="plain-label">What to look for</div>'
        f"<ul>{look}</ul></div>"
        '<div><div class="plain-label">What it cannot tell you</div>'
        f"<ul>{cannot}</ul></div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def _glossary_html(terms: Sequence[guide.Term]) -> str:
    parts = ['<dl class="glossary">']
    for term in terms:
        parts.append(f"<dt>{_esc(term.heading)}</dt>")
        parts.append(f"<dd>{_esc(term.plain)}</dd>")
        if term.example:
            parts.append(f'<dd class="ex"><b>Example.</b> {_esc(term.example)}</dd>')
        if term.look_for:
            parts.append(
                f'<dd class="lf"><b>What to look for.</b> {_esc(term.look_for)}</dd>'
            )
        if term.formula:
            parts.append(f'<dd class="fm">{_esc(term.formula)}</dd>')
    parts.append("</dl>")
    return "".join(parts)


def glossary_expander(section_key: str) -> None:
    """Collapsed guide to every term a page uses."""
    terms = guide.terms_for(section_key)
    with st.expander(
        f"Plain-English guide to the terms on this page ({len(terms)})",
        expanded=False,
    ):
        st.markdown(_glossary_html(terms), unsafe_allow_html=True)


def landing_guide() -> None:
    """The 'start here' panel shown before the first run."""
    steps = "".join(
        '<div class="start-step">'
        f'<div class="n">{number}</div><b>{_esc(title)}</b><p>{_esc(body)}</p></div>'
        for number, (title, body) in enumerate(guide.LANDING_STEPS, start=1)
    )
    ideas = "".join(
        f'<div class="idea"><b>{_esc(name)}</b><span>{_esc(body)}</span></div>'
        for name, body in guide.LANDING_IDEAS
    )
    pages = "".join(
        f'<div><span class="pg">Page {number}</span>'
        f"<span><b>{_esc(page.title)}</b></span><span>{_esc(page.intro)}</span></div>"
        for number, page in enumerate(guide.SECTION_GUIDES.values(), start=1)
    )
    st.markdown(
        f'<div class="start-intro">{_esc(guide.LANDING_INTRO)}</div>'
        '<div class="start-label">Start here</div>'
        f'<div class="start-steps">{steps}</div>'
        '<div class="start-label">Four ideas run through every page</div>'
        f'<div class="idea-strip">{ideas}</div>'
        '<div class="start-label">The six pages at a glance</div>'
        f'<div class="pages-list">{pages}</div>',
        unsafe_allow_html=True,
    )
    core = [guide.GLOSSARY[key] for key in guide.LANDING_CORE_TERMS]
    with st.expander("The four numbers you will see most often", expanded=True):
        st.markdown(_glossary_html(core), unsafe_allow_html=True)
    callout(_esc(guide.LANDING_NOTE), tone="brass")


def methodology_strip(methodologies: dict[str, str]) -> None:
    """Compact model labels with browser-native hover explanations."""
    chips = [
        '<span class="methodology-chip">'
        f"{_esc(name)} "
        f'<span class="info-dot" title="{html.escape(explanation, quote=True)}">i</span>'
        "</span>"
        for name, explanation in methodologies.items()
    ]
    st.markdown(
        '<div class="methodology-strip">' + "".join(chips) + "</div>",
        unsafe_allow_html=True,
    )


def legend_key(items: Iterable[tuple[str, str]]) -> None:
    """Small inline key for encodings a chart legend cannot express."""
    parts = [
        f'<span class="item"><i style="background:{color};"></i>{_esc(label)}</span>'
        for label, color in items
    ]
    st.markdown(
        '<div class="legend-key">' + "".join(parts) + "</div>",
        unsafe_allow_html=True,
    )


def chart(altair_chart, key: str | None = None) -> None:
    """Render an Altair chart with the Ledger theme rather than Streamlit's."""
    st.altair_chart(altair_chart, width="stretch", theme=None, key=key)


def footer(extra: str = "") -> None:
    st.markdown(
        f"""
        <div class="app-footer">
            <b>Personal research project</b> · For educational purposes only · Not financial advice.
            {_esc(extra)}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Exports
# ----------------------------------------------------------------------


def export_bar(
    pdf_builder: Callable[[], bytes],
    pdf_name: str,
    excel_builder: Callable[[], bytes] | None = None,
    excel_name: str | None = None,
    hint: str = "",
    key_prefix: str = "export",
) -> None:
    """Download row for a section.

    The builders are passed to ``st.download_button`` as callables, so nothing
    is rendered or computed until the reader actually clicks — a section export
    costs zero on an ordinary rerun.
    """
    st.markdown(
        '<div class="export-bar">'
        '<div class="title">Export</div>'
        f'<div class="hint">{_esc(hint)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    columns = st.columns([1, 1, 3])
    with columns[0]:
        st.download_button(
            "Download PDF",
            data=pdf_builder,
            file_name=f"{pdf_name}.pdf",
            mime="application/pdf",
            type="primary",
            width="stretch",
            key=f"{key_prefix}_pdf",
            icon=":material/picture_as_pdf:",
        )
    if excel_builder is not None:
        with columns[1]:
            st.download_button(
                "Download Excel",
                data=excel_builder,
                file_name=f"{excel_name or pdf_name}.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                ),
                width="stretch",
                key=f"{key_prefix}_xlsx",
                icon=":material/table_view:",
            )


def fmt_pct(value, decimals: int = 1) -> str:
    if value is None:
        return "—"
    try:
        if value != value:
            return "—"
    except Exception:
        pass
    return f"{float(value):.{decimals}%}"


def fmt_money(value) -> str:
    if value is None:
        return "—"
    try:
        if value != value:
            return "—"
    except Exception:
        pass
    return f"${float(value):,.0f}"

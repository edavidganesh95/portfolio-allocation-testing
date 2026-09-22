"""Plain-English guide: what each page does and what every number means.

One source for all explanatory copy, so the on-screen pages, the sidebar tooltips
and the printed PDFs always say the same thing. Nothing here imports Streamlit.

Writing rules, enforced by ``tests/test_guide.py``:

* Write for someone who has never read a fund factsheet: short sentences, a
  concrete example wherever a number could be abstract.
* Describe and interpret; never advise. Nothing here tells a reader what to buy.
* Be honest about limits. Every page says what it cannot tell you.
"""

from __future__ import annotations

from dataclasses import dataclass
from xml.sax.saxutils import escape

import pandas as pd

from . import exporting as ex


@dataclass(frozen=True)
class Term:
    key: str
    plain: str
    example: str = ""
    look_for: str = ""
    formula: str = ""
    title: str = ""

    @property
    def heading(self) -> str:
        return self.title or self.key


def _t(key, plain, example="", look_for="", formula="", title=""):
    return Term(key, plain, example, look_for, formula, title)


# ----------------------------------------------------------------------
# Glossary
# ----------------------------------------------------------------------

_TERMS: tuple[Term, ...] = (
    # -- the basics -----------------------------------------------------
    _t(
        "Buy-and-hold",
        "Buy your chosen mix once and leave it alone, with no regular rebalancing. "
        "Holdings that grow faster take up a larger share as time passes.",
        "Start 50/50 in funds A and B. If A doubles and B stays flat, you now hold "
        "about 67% in A without having traded.",
        "Every result in this tool assumes buy-and-hold, so a mix's shares will drift.",
    ),
    _t(
        "Reference portfolio",
        "The portfolio you hold today, or an example one. Every other mix is compared "
        "with it, so you can see what changing would mean.",
        look_for="It is a yardstick, not a recommendation.",
    ),
    _t(
        "Universe",
        "The list of funds the tool is allowed to choose from. You can add or remove funds.",
    ),
    _t(
        "Common sample",
        "To compare funds fairly the tool only uses dates when every fund existed. "
        "The newest fund sets the start date for everyone.",
        "Adding a fund launched in 2019 means all results start in late 2019, even "
        "if the other funds have a longer history.",
        "Fewer months of history means less reliable results.",
        title="Common sample (observations)",
    ),
    _t(
        "In-sample vs out-of-sample",
        "In-sample means judging a mix on the same history used to build it, which "
        "flatters it. Out-of-sample means judging it on data it had not seen.",
        look_for="Trust out-of-sample results more than in-sample ones.",
    ),
    _t(
        "Risk-free rate",
        "What a safe, cash-like investment earns each year. Return above this is "
        "the reward for taking risk.",
        look_for="Used when measuring return per unit of risk (Sharpe and Sortino).",
    ),
    _t(
        "Market benchmark",
        "A broad stock-market fund used as the yardstick for 'the market' "
        "(the default is a world stock market fund). Each fund's beta is measured against it.",
    ),
    _t(
        "Long-run market return assumption",
        "Your own estimate of what the broad market earns each year over the long run. "
        "The default is 8%. It is an assumption you can change, not a prediction.",
        look_for="Changing it shifts every fund's expected return.",
    ),
    # -- return and risk statistics -------------------------------------
    _t(
        "CAGR",
        "The steady yearly growth rate that would turn the starting amount into the "
        "ending amount. Short for compound annual growth rate.",
        "$10,000 growing at 8% a year becomes about $21,600 after 10 years.",
        "Higher is better, but read it next to volatility and drawdown.",
        title="CAGR (yearly growth rate)",
    ),
    _t(
        "Volatility",
        "How much returns bounce around, measured per year. Higher volatility means a bumpier ride.",
        "A fund with 16% volatility has typically moved within roughly 16 points "
        "either side of its average yearly return.",
        "Lower is smoother. Compare funds that earned similar returns.",
    ),
    _t(
        "Sharpe",
        "Return earned for each unit of bumpiness, after subtracting what cash would "
        "earn. A higher number means more return for the risk taken.",
        "Two funds both return 10% a year. One has 10% volatility (Sharpe 1.0), the "
        "other 20% (Sharpe 0.5). The smoother one is more efficient.",
        "Above about 1 is often called good and below 0.5 weak. Only a rough guide "
        "over a short history.",
        "(CAGR - risk-free rate) / volatility",
        title="Sharpe ratio",
    ),
    _t(
        "Sortino",
        "Like Sharpe, but only counts the downward moves as risk, since upward surprises "
        "do not worry investors.",
        look_for="Higher is better. If it is much higher than Sharpe, most of the "
        "bumpiness was upward.",
        title="Sortino ratio",
    ),
    _t(
        "Max Drawdown",
        "The biggest fall from a peak to the next low over the period. It is what "
        "someone who bought at the top would have lived through.",
        "A drawdown of -30% turns $10,000 into $7,000 at the low point.",
        "Closer to zero is milder. Ask yourself whether you could sit through that fall.",
        title="Max drawdown (worst fall)",
    ),
    _t(
        "CVaR",
        "The average loss in the worst 5% of months. It describes how bad the bad months "
        "were, not just how often they happen.",
        "A CVaR of 9% means that in the worst months the fund lost about 9% on average.",
        "Lower is better. With about seven years of data it rests on only around four "
        "months, so treat it as rough.",
        title="CVaR (size of the worst months)",
    ),
    _t(
        "Correlation",
        "How closely two funds move together, from -1 to +1. Near +1 they rise and fall "
        "together. Near 0 they are unrelated. Below 0 they tend to move in opposite directions.",
        look_for="The lower the correlation between holdings, the more they offset each other.",
    ),
    _t(
        "Beta",
        "How strongly a fund moves with the broad market.",
        "A beta of 1.2 means the fund has tended to move about 1.2% when the market moves 1%.",
        "Above 1 amplifies the market's moves; below 1 dampens them.",
        title="Beta (market sensitivity)",
    ),
    _t(
        "Growth of $1",
        "What $1 invested at the start would have become, with dividends reinvested.",
        look_for="A steeper line means faster growth; dips show the falls along the way.",
    ),
    _t(
        "Drawdown profile",
        "For each fund, how far below its previous high it sat at every date.",
        look_for="Deep or long dips are the stretches an investor would have had to sit through.",
    ),
    _t(
        "Log scale",
        "A chart setting that gives equal percentage moves equal space, which is fairer "
        "for comparing growth over long periods.",
    ),
    _t(
        "Rolling returns",
        "The return over every 12-month window, not just calendar years.",
        look_for="Shows whether good results were steady or came from one lucky stretch.",
    ),
    # -- expected returns and the risk model ----------------------------
    _t(
        "Expected return E(r)",
        "The model's estimate of a fund's typical yearly return going forward. It blends "
        "the fund's own past return with a market-based estimate.",
        look_for="It is an assumption used to build the mixes, not a forecast.",
        title="Expected return, E(r)",
    ),
    _t(
        "Historical E(r)",
        "Uses each fund's own past return as it happened.",
        look_for="Simple, but a short history can flatter or punish a fund by luck.",
    ),
    _t(
        "Shrunk E(r)",
        "Pulls each fund's past return part of the way towards a market-based figure, "
        "because a short record can be misleading.",
        look_for="This is the default view because it is steadier than raw past returns.",
    ),
    _t(
        "Shrinkage",
        "A dial from 0% to 100%. At 0% the tool trusts each fund's own past return "
        "completely; at 100% it uses only the market-based estimate. The default is 50%.",
    ),
    _t(
        "Beta-implied anchor",
        "The return a fund would 'deserve' given how strongly it moves with the market: "
        "cash rate plus beta times the extra return you assume for the market.",
        formula="risk-free rate + beta x (market return assumption - risk-free rate)",
    ),
    _t(
        "Expected volatility",
        "The model's estimate of a mix's bumpiness, worked out from how its funds have "
        "moved together in the past. Compare with Historical volatility, which is what "
        "the mix actually did.",
    ),
    _t(
        "Expected Sharpe",
        "The model's estimate of return per unit of risk for a mix, from the expected "
        "return and expected volatility. It is not what the mix achieved in the past.",
    ),
    _t(
        "Ledoit-Wolf risk model",
        "A statistical method for estimating how funds move together from a short history "
        "without overreacting to noise. It pulls extreme estimates towards more sensible ones.",
        title="Ledoit-Wolf risk model",
    ),
    # -- frontiers -------------------------------------------------------
    _t(
        "Efficient frontier",
        "For each level of risk, the highest expected return you could get by mixing these "
        "funds, according to the model. A mix below the line takes more risk for less reward.",
        look_for="It is built from past data, so it shows what was possible, not what will be.",
    ),
    _t(
        "Opportunity set",
        "The grey dots: thousands of random mixes of your funds, drawn to show the range of "
        "risk and return combinations that were possible.",
    ),
    _t(
        "Global minimum variance",
        "The mix with the smoothest ride that can be built from these funds.",
        look_for="It is not necessarily the mix with the best return.",
    ),
    _t(
        "Return-CVaR frontier",
        "The same idea as the efficient frontier, but risk is the size of the worst months "
        "rather than overall bumpiness. Further left is better.",
        title="Return-CVaR frontier",
    ),
    _t(
        "Maximum ETF weight",
        "A cap on how much of a mix any single fund can take. The default is 70%.",
        look_for="Lower caps force the money to be spread out more.",
    ),
    # -- the nine ways of splitting money -------------------------------
    _t(
        "Equal Weight",
        "The same amount in every fund. It needs no forecasts, which makes it a useful yardstick.",
        look_for="Simple, but it ignores that some funds are far riskier or overlap heavily.",
    ),
    _t(
        "Minimum Variance",
        "The mix that has had the smoothest ride, meaning the lowest volatility.",
        look_for="It often ends up concentrated in a few calm funds and may give up growth.",
    ),
    _t(
        "Risk Parity",
        "Gives each fund an equal share of the risk, so jumpier funds get less money.",
        look_for="Spreads risk evenly, but does not consider which funds might earn more.",
    ),
    _t(
        "Maximum Diversification",
        "Looks for the mix where the holdings offset each other most, meaning the biggest "
        "reduction in bumpiness from combining funds.",
        look_for="It ignores return, and can lean on the few funds that behave differently "
        "from the rest.",
    ),
    _t(
        "Maximum Sharpe",
        "The mix with the best estimated return per unit of risk, using the expected returns.",
        look_for="Very sensitive to the return estimates; it often piles into one or two funds.",
    ),
    _t(
        "Diversified Maximum Sharpe",
        "The same goal as Maximum Sharpe, but with guardrails: a smaller cap per fund, a "
        "minimum spread of the money, and a limit on how much of the risk one fund can cause.",
        look_for="Trades some of the best-case return for a more spread-out mix.",
    ),
    _t(
        "Minimum CVaR",
        "The mix that has had the mildest worst months.",
        look_for="It depends on a handful of bad months, so it can change a lot with the data.",
    ),
    _t(
        "Maximum Return / CVaR",
        "The best estimated return for each unit of loss in the worst months.",
        look_for="Also sensitive to the return estimates and to just a few bad months.",
    ),
    _t(
        "HRP Diversification",
        "Sorts funds into families that move alike, then spreads money across the families, "
        "giving less to the jumpier ones. It uses no return forecasts.",
        look_for="A rule-based way to spread money without predicting winners.",
        title="HRP (hierarchical risk parity)",
    ),
    # -- page 2: multi-period -------------------------------------------
    _t(
        "Lookback window",
        "How much past history is used to build a mix: the last 3 years, the last 5 years, "
        "or everything available.",
    ),
    _t(
        "Refit",
        "Rebuilding a mix from scratch using a different stretch of history.",
        look_for="If a mix changes a lot when refitted, its answer depends on the window.",
    ),
    _t(
        "Capital Change vs Full",
        "The share of your money that would move if you used a 3-year or 5-year window "
        "instead of all history. 0% means identical mixes; 100% means completely different.",
        look_for="Lower means a steadier method, which suits a buy-and-hold plan.",
        title="Capital change vs full history",
    ),
    _t(
        "Largest Weight",
        "The biggest single holding in a mix.",
        look_for="A large number means the mix leans heavily on one fund.",
    ),
    _t(
        "Actual ETFs",
        "How many funds are really held, ignoring tiny leftovers below 0.01%.",
    ),
    _t(
        "Realised metrics by period",
        "How each of today's mixes would have performed if bought at the start of the "
        "last 1, 3 or 5 years (or all history) and left alone. The mixes are not rebuilt for each window.",
        look_for="Whether a mix looks good in every window or only in one.",
    ),
    # -- page 3: bootstrap ----------------------------------------------
    _t(
        "Bootstrap simulation",
        "Shuffling blocks of past months into thousands of alternative histories to see "
        "the range of outcomes a mix could have produced. Think of the past months as a "
        "deck of cards: shuffle it, deal ten years, note where the money ends up, repeat.",
        look_for="It shows the spread of results, not a forecast.",
    ),
    _t(
        "Simulation paths",
        "How many alternative histories were created. More paths give smoother, more repeatable numbers.",
    ),
    _t(
        "Block length",
        "How many consecutive months are kept together when history is reshuffled. "
        "Longer blocks keep more of the way good and bad periods cluster.",
    ),
    _t(
        "Seed",
        "A number that fixes the shuffling so results can be repeated exactly. "
        "Changing it shows how much the answer depends on the luck of the shuffle.",
    ),
    _t(
        "Median terminal wealth",
        "The middle outcome: half the simulated runs ended above this, half below.",
        "If it reads $400,000 on a $100,000 start, half the runs ended above $400,000.",
        title="Median ending wealth",
    ),
    _t(
        "5th Percentile Wealth",
        "A poor but plausible outcome: only 5 out of every 100 simulated runs ended lower.",
        look_for="It is a bad case, not a guaranteed floor. Some runs did worse.",
        title="5th-percentile wealth (a poor outcome)",
    ),
    _t(
        "Median CAGR",
        "The typical yearly growth rate across the simulated runs.",
        title="Median CAGR (typical yearly growth)",
    ),
    _t(
        "P(Loss at Horizon)",
        "How often a simulated run ended below the amount it started with.",
        "1.5% means about 3 in 200 runs lost money over the period.",
        "Lower is better. It counts nominal losses, before inflation.",
        title="Chance of ending below the start",
    ),
    _t(
        "P(CAGR > x%)",
        "How often a simulated run grew faster than 6%, 8% or 10% a year.",
        title="Chance of beating 6%, 8% or 10% a year",
    ),
    _t(
        "Simulated max drawdown",
        "The deepest fall from a peak at any point along each simulated run. "
        "The median is the typical run; the 5th percentile is a bad case that only 5% of runs fell below.",
        look_for="Closer to zero means milder falls.",
    ),
    _t(
        "Path fan",
        "A chart of how the range of outcomes widens over time for one portfolio. "
        "The line is the middle outcome; the dark band holds the middle 50% of runs and the light band about 90%.",
    ),
    _t(
        "Terminal wealth distribution",
        "A histogram of where the simulated runs ended: the taller the curve, the more runs finished at that amount.",
        title="Ending-wealth distribution",
    ),
    # -- page 4: testing without hindsight -------------------------------
    _t(
        "Testing without hindsight",
        "Goes back to a past date, builds each mix using only what was known then, holds it "
        "for a fixed time with no changes, records the result, then moves forward and repeats.",
        look_for="This is the fairest test in the tool, because the mixes cannot peek at the future.",
    ),
    _t(
        "Training lookback",
        "How much history the model may use before each test starts. Only data from before "
        "the start date is used.",
    ),
    _t(
        "Holding period",
        "How long each mix is held, untouched, before that test ends.",
    ),
    _t(
        "Independent tests",
        "The number of separate start dates that were tested.",
        look_for="With only a few tests, one good or bad year can dominate. Treat the results as hints, not proof.",
    ),
    _t(
        "Median Return",
        "The middle yearly return across the separate tests.",
        title="Median return (across tests)",
    ),
    _t(
        "Worst Return",
        "The weakest yearly return in any single test.",
        look_for="Shows how much the start date mattered.",
        title="Worst and best return",
    ),
    _t(
        "Median Volatility",
        "The typical bumpiness experienced during a test.",
        title="Median volatility (across tests)",
    ),
    _t(
        "Median Max Drawdown",
        "The typical worst fall inside a single test.",
        title="Median max drawdown (across tests)",
    ),
    _t(
        "Beat Current Portfolio",
        "In how many of the separate tests the mix finished ahead of your reference portfolio.",
        "'2 of 4 (50%)' means it came out ahead in two of the four tests.",
        "With only four tests, a 2-2 split says very little.",
        title="Beat current portfolio",
    ),
    _t(
        "Weight drift",
        "How far holdings moved from their starting mix by the end of a hold, just because "
        "some rose faster than others.",
    ),
    # -- page 5: diversification ----------------------------------------
    _t(
        "Effective Holdings",
        "The number of equal-sized holdings your money is behaving like. It looks past how "
        "many funds you own to how evenly the money is divided.",
        "Five funds with 70% in one behave like about two equal holdings.",
        "Well below 'Actual ETFs' means most of the money sits in a few funds.",
        "1 / (sum of each weight squared)",
        title="Effective holdings",
    ),
    _t(
        "Effective Risk Bets",
        "The same idea applied to risk: how many equal-sized sources of risk the portfolio "
        "behaves like. One fund causing most of the ups and downs pulls it towards 1.",
        look_for="Higher means the risk is spread across more independent sources.",
        title="Effective risk bets",
    ),
    _t(
        "Weighted Correlation",
        "The average correlation between the holdings, counting pairs that hold more money "
        "more heavily. It measures how alike the holdings' returns are.",
        look_for="Lower means the holdings offset each other more. It does not compare what the funds hold inside.",
        title="Weighted correlation",
    ),
    _t(
        "Diversification Ratio",
        "The average volatility of the individual funds divided by the volatility of the "
        "combined portfolio. A value of 1.0 means combining the funds removed no bumpiness.",
        "1.15 means the portfolio is about 13% less bumpy than the average of its funds.",
        "Higher means combining the funds reduced the bumpiness more.",
        title="Diversification ratio",
    ),
    _t(
        "Largest Risk Share",
        "The share of the portfolio's ups and downs caused by its single biggest source of risk.",
        look_for="If one fund causes most of the risk, the rest are along for the ride.",
        title="Largest risk share",
    ),
    _t(
        "Historical Volatility",
        "How bumpy the mix actually was over the sample, as opposed to the model's estimate.",
        title="Historical volatility",
    ),
    _t(
        "Risk contribution",
        "How much of the mix's total bumpiness each holding causes. It can be very different "
        "from how much money the holding takes.",
        "A fund with 20% of the money can cause 50% of the risk if it is much jumpier than the rest.",
    ),
    _t(
        "Return-Diversification Frontier",
        "For each minimum level of expected return, the most diversified mix that can be built. "
        "Going up means more expected return; going right means more diversified.",
        look_for="It shows what you give up in diversification to get more return, and the other way round.",
        title="Return-diversification frontier",
    ),
    _t(
        "Correlation clusters",
        "Groups of funds that behave alike, found by sorting them by how closely their returns move together.",
    ),
    # -- page 6: summary --------------------------------------------------
    _t(
        "Historical CAGR",
        "The yearly growth rate the mix actually delivered over the sample, if bought at the start and left alone.",
        title="Historical CAGR (what happened)",
    ),
    _t(
        "Bootstrap Median CAGR",
        "The typical yearly growth rate across the simulated futures.",
        title="Simulated median CAGR",
    ),
    _t(
        "Bootstrap 5th Wealth",
        "A poor but plausible ending amount from the simulations: only about 5% of runs ended lower.",
        look_for="It is not a guaranteed floor.",
        title="Simulated 5th-percentile wealth",
    ),
    _t(
        "OOS Median Return",
        "The middle yearly return across the separate tests without hindsight.",
        title="Median return in tests without hindsight",
    ),
    _t(
        "Capital to Reallocate",
        "The share of your current portfolio that would have to move into different holdings "
        "to reach the mix.",
        "40% means 40 cents of every current dollar would move.",
        "A smaller number means an easier switch. It is a one-time distance, not a "
        "promise that you would keep trading afterwards.",
        "half the sum of the absolute weight differences",
    ),
    _t(
        "Difference vs current",
        "How each mix differs from your reference portfolio on a measure. A plus sign means higher "
        "than the reference, a minus sign lower. These are simple differences, not scores.",
        title="Difference vs your current portfolio",
    ),
)

GLOSSARY: dict[str, Term] = {term.key: term for term in _TERMS}

#: Labels the app prints that map to a differently named glossary entry.
ALIASES: dict[str, str] = {
    "Expected Return E(r)": "Expected return E(r)",
    "Expected Volatility": "Expected volatility",
    "Expected Sharpe": "Expected Sharpe",
    "Effective ETFs": "Effective Holdings",
    "Portfolio Volatility": "Historical Volatility",
    "Median Terminal Wealth": "Median terminal wealth",
    "5th Percentile Max Drawdown": "Simulated max drawdown",
    "5th Percentile CAGR": "Median CAGR",
    "Chance of Ending Below Start": "P(Loss at Horizon)",
    "Median Annualized Return": "Median Return",
    "Worst Annualized Return": "Worst Return",
    "Best Annualized Return": "Worst Return",
    "Capital Change 3Y vs Full": "Capital Change vs Full",
    "Capital Change 5Y vs Full": "Capital Change vs Full",
    "Largest Risk Contributor": "Largest Risk Share",
    "Δ E(r)": "Difference vs current",
    "Δ DR": "Difference vs current",
}


def find_term(label: str) -> Term | None:
    """Resolve a label the app prints to its glossary entry, if it has one."""
    label = str(label).strip()
    if label in GLOSSARY:
        return GLOSSARY[label]
    if label in ALIASES:
        return GLOSSARY.get(ALIASES[label])
    if label.startswith("CVaR"):
        return GLOSSARY["CVaR"]
    if label.startswith("P(CAGR"):
        return GLOSSARY["P(CAGR > x%)"]
    if label.startswith("Chance CAGR"):
        return GLOSSARY["P(CAGR > x%)"]
    if label.startswith("Δ"):
        return GLOSSARY["Difference vs current"]
    for prefix, key in (
        ("Largest Weight", "Largest Weight"),
        ("Actual ETFs", "Actual ETFs"),
        ("Median Terminal Wealth", "Median terminal wealth"),
        ("Best Return", "Worst Return"),
    ):
        if label.startswith(prefix):
            return GLOSSARY[key]
    return None


# ----------------------------------------------------------------------
# Section guides
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SectionGuide:
    key: str
    title: str
    intro: str
    what: str
    look_for: tuple[str, ...]
    cannot: tuple[str, ...]
    terms: tuple[str, ...]


SECTION_GUIDES: dict[str, SectionGuide] = {
    "construction": SectionGuide(
        key="construction",
        title="Portfolio construction",
        intro="Meet the funds, see how they have behaved, and compare nine ways of splitting money between them.",
        what=(
            "This page lines up the funds you are considering and shows how each one behaved "
            "over the same stretch of history. It then builds nine example mixes, each following "
            "a different idea (split the money equally, aim for the smoothest ride, spread the "
            "risk evenly, and so on) so you can see what each idea produces."
        ),
        look_for=(
            "Which funds moved most alike on the correlation map. Funds that move together do not spread your risk much.",
            "How different the nine mixes look. Some spread the money widely; others put most of it in one or two funds.",
            "Whether a mix's past growth came with a bumpy ride (volatility) or deep falls (max drawdown).",
        ),
        cannot=(
            "Every number here uses the same history the mixes were built from, which flatters them. Pages 3 and 4 test them more fairly.",
            "It shows how the funds' returns moved together, not what the funds hold inside.",
        ),
        terms=(
            "Common sample",
            "Universe",
            "Reference portfolio",
            "Market benchmark",
            "Long-run market return assumption",
            "Risk-free rate",
            "CAGR",
            "Volatility",
            "Sharpe",
            "Sortino",
            "Max Drawdown",
            "CVaR",
            "Growth of $1",
            "Log scale",
            "Drawdown profile",
            "Rolling returns",
            "Correlation",
            "Beta",
            "Expected return E(r)",
            "Historical E(r)",
            "Shrunk E(r)",
            "Shrinkage",
            "Beta-implied anchor",
            "Ledoit-Wolf risk model",
            "Maximum ETF weight",
            "Equal Weight",
            "Minimum Variance",
            "Risk Parity",
            "Maximum Diversification",
            "Maximum Sharpe",
            "Diversified Maximum Sharpe",
            "Minimum CVaR",
            "Maximum Return / CVaR",
            "HRP Diversification",
            "Efficient frontier",
            "Opportunity set",
            "Global minimum variance",
            "Return-CVaR frontier",
            "In-sample vs out-of-sample",
        ),
    ),
    "multi_period": SectionGuide(
        key="multi_period",
        title="Multi-period analysis",
        intro="Does the answer change depending on which stretch of history you use?",
        what=(
            "This page asks whether a method's answer depends on which stretch of history you "
            "feed it. It rebuilds each mix using only the last 3 years, only the last 5 years and "
            "all available history, and shows how much of your money would move between versions. "
            "It also shows how today's mixes would have done if bought at the start of the last "
            "1, 3 or 5 years and left alone."
        ),
        look_for=(
            "The 'capital change' figures. A low number means the method gives a similar answer whatever window you use, which suits a buy-and-hold plan.",
            "Whether a mix looks good in every window or in only one.",
            "The 1-year column is only a snapshot of recent performance; it is not used to build anything.",
        ),
        cannot=(
            "A steady method is not necessarily a good one. It can be steadily wrong.",
            "The windows overlap, so they are not three independent checks.",
        ),
        terms=(
            "Buy-and-hold",
            "Lookback window",
            "Refit",
            "Capital Change vs Full",
            "Largest Weight",
            "Actual ETFs",
            "Realised metrics by period",
            "CAGR",
            "Volatility",
            "Sharpe",
            "Max Drawdown",
        ),
    ),
    "bootstrap": SectionGuide(
        key="bootstrap",
        title="Bootstrap simulation",
        intro="Shuffle history thousands of times to see the range of outcomes each mix could produce.",
        what=(
            "This page replays history in thousands of shuffled orders to see the range of "
            "outcomes each mix could have produced. Think of the past months as a deck of cards: "
            "shuffle the deck, deal ten years, see where the money ends up, and repeat ten thousand "
            "times. Every mix gets exactly the same shuffled decks, so differences come from the mix "
            "and not from luck."
        ),
        look_for=(
            "The gap between the middle outcome and the poor-but-plausible one (5th percentile). A narrow gap means a less uncertain ride.",
            "The chance of ending below the starting amount: how often ten years of investing lost money in the simulations.",
            "The simulated worst fall: how deep a bad stretch could get along the way.",
        ),
        cannot=(
            "It can only reshuffle months that really happened, so a crisis unlike anything in the sample cannot appear.",
            "The mixes were designed using this same history, so they get a friendly test. Treat any comparison with your reference portfolio with care.",
            "The results are simulated outcomes, not forecasts.",
        ),
        terms=(
            "Bootstrap simulation",
            "Simulation paths",
            "Block length",
            "Seed",
            "Median terminal wealth",
            "5th Percentile Wealth",
            "Median CAGR",
            "P(Loss at Horizon)",
            "P(CAGR > x%)",
            "Simulated max drawdown",
            "Path fan",
            "Terminal wealth distribution",
        ),
    ),
    "walk_forward": SectionGuide(
        key="walk_forward",
        title="Testing without hindsight",
        intro="Pretend it is a past date, build each mix with only what was known then, and see how it did.",
        what=(
            "This page tries to remove hindsight. It goes back to a past date, pretends that is "
            "today, builds each mix using only the previous few years of data, holds it for a year "
            "with no changes, and records what happened. Then it moves forward a year and repeats. "
            "Each test is a separate 'what if I had started then' experiment."
        ),
        look_for=(
            "How many tests there are. With only a few, a single good or bad year can dominate, so treat the results as hints, not proof.",
            "The 'beat current portfolio' counts (such as 2 of 4) and the worst-year return.",
            "Whether a mix's falls during the holds were milder or deeper than your reference portfolio's.",
        ),
        cannot=(
            "The period tested may contain no major market crash, so a calm run says little about how a mix behaves in one.",
            "Your reference portfolio is not rebuilt each year. It is simply held, so this compares a rule against a fixed choice.",
        ),
        terms=(
            "Testing without hindsight",
            "In-sample vs out-of-sample",
            "Training lookback",
            "Holding period",
            "Independent tests",
            "Median Return",
            "Worst Return",
            "Median Volatility",
            "Median Max Drawdown",
            "Beat Current Portfolio",
            "Weight drift",
        ),
    ),
    "diversification": SectionGuide(
        key="diversification",
        title="Diversification and overlap",
        intro="Owning many funds does not guarantee your money is really spread out. Check how evenly money and risk are divided.",
        what=(
            "Owning many funds does not guarantee your money is spread out. This page checks how "
            "evenly your money, and your risk, is really divided, and how much the holdings move "
            "together. It also compares each mix with a simple family-tree approach called HRP."
        ),
        look_for=(
            "'Effective holdings' far below 'Actual ETFs' means most of the money sits in a few funds.",
            "'Largest risk share': if one fund causes most of the ups and downs, the rest are along for the ride.",
            "A higher diversification ratio and a lower weighted correlation mean the holdings offset each other more.",
        ),
        cannot=(
            "It measures how the funds' returns moved together, not what they hold inside. Two funds can hold different stocks and still move together, and funds can share holdings without moving identically.",
            "Funds tend to move together more in sell-offs than in calm periods, so these numbers can flatter diversification in a crisis.",
        ),
        terms=(
            "Actual ETFs",
            "Effective Holdings",
            "Effective Risk Bets",
            "Weighted Correlation",
            "Diversification Ratio",
            "Largest Risk Share",
            "Historical Volatility",
            "Risk contribution",
            "Return-Diversification Frontier",
            "Correlation clusters",
            "HRP Diversification",
            "Correlation",
        ),
    ),
    "summary": SectionGuide(
        key="summary",
        title="Summary of findings",
        intro="Everything side by side, with no score and no winner.",
        what=(
            "This page puts every mix side by side: the return the model assumes, the actual past "
            "record, the simulated range, the test without hindsight, how diversified it is, and "
            "how much of your money would have to move to get there. There is deliberately no "
            "combined score and no winner."
        ),
        look_for=(
            "Trade-offs. Mixes with higher return numbers usually come with more risk or less diversification.",
            "Where the numbers disagree, for example strong past return but a weak test without hindsight. Disagreements are informative.",
            "'Capital to reallocate': the size of the switch from your current holdings.",
        ),
        cannot=(
            "It cannot say which mix suits you. That depends on your goals, other assets, tax situation and how you would cope with a fall.",
            "It does not see what is inside the funds, and it does not model trading costs or taxes.",
        ),
        terms=(
            "Expected return E(r)",
            "Expected volatility",
            "Expected Sharpe",
            "Historical CAGR",
            "Bootstrap Median CAGR",
            "Bootstrap 5th Wealth",
            "OOS Median Return",
            "Beat Current Portfolio",
            "Actual ETFs",
            "Effective Holdings",
            "Effective Risk Bets",
            "Diversification Ratio",
            "Largest Risk Share",
            "Capital to Reallocate",
            "Difference vs current",
        ),
    ),
}


def terms_for(section_key: str) -> list[Term]:
    """The glossary entries for one page, in reading order."""
    return [GLOSSARY[key] for key in SECTION_GUIDES[section_key].terms]


def all_terms() -> list[Term]:
    return sorted(GLOSSARY.values(), key=lambda term: term.heading.lower())


# ----------------------------------------------------------------------
# Landing page
# ----------------------------------------------------------------------

LANDING_INTRO = (
    "This tool helps you compare different ways of splitting money across ETFs for a "
    "long-term, buy-and-hold approach. It tests each mix against the past, against thousands "
    "of shuffled versions of the past, and against a 'what if I had chosen it back then, "
    "without knowing what came next' check. It does not pick a winner and it does not tell "
    "you what to buy."
)

LANDING_STEPS: tuple[tuple[str, str], ...] = (
    (
        "Choose the funds to consider",
        "Use the sidebar's Universe box. The default list is only a starting point; "
        "you can untick funds or add any ticker Yahoo Finance knows.",
    ),
    (
        "Describe what you hold today",
        "In the sidebar's Reference portfolio box, enter your current holdings so every "
        "result can be compared with them.",
    ),
    (
        "Press Run research",
        "Prices are downloaded and the tests run, which takes around half a minute. "
        "Changing a setting does nothing until you press it again.",
    ),
    (
        "Read the six pages in order",
        "Page 1 shows the funds and the mixes. Pages 2 to 4 test whether a mix holds up. "
        "Page 5 checks whether it is really spread out. Page 6 lines everything up.",
    ),
)

LANDING_IDEAS: tuple[tuple[str, str], ...] = (
    ("Return", "How much your money grew."),
    ("Risk", "How bumpy the ride was and how deep the falls went."),
    ("Diversification", "Whether your holdings behave differently from one another."),
    (
        "Evidence",
        "How far to trust a result. The past is not a forecast, and a short history means low confidence.",
    ),
)

LANDING_CORE_TERMS: tuple[str, ...] = ("CAGR", "Volatility", "Max Drawdown", "Sharpe")

LANDING_NOTE = (
    "Educational research tool. Nothing here is investment advice, and past results do not "
    "predict future ones. Fund fees are already reflected in the prices used, but trading "
    "costs and taxes are not modelled."
)


# ----------------------------------------------------------------------
# Sidebar help text
# ----------------------------------------------------------------------

HELP: dict[str, str] = {
    "universe_curated": "The funds the tool can choose from. Untick any you do not want to consider.",
    "universe_custom": (
        "Add any fund or stock ticker that Yahoo Finance recognises, for example VXUS or 2800.HK. "
        "Separate them with commas, spaces or new lines. A newer fund shortens the shared "
        "history of every fund."
    ),
    "frequency": (
        "How often prices are sampled. Monthly suits a long-term buy-and-hold study; weekly "
        "and daily add detail but also noise, and run more slowly."
    ),
    "benchmark": (
        "A broad stock-market fund used as the yardstick for 'the market'. Each fund's beta "
        "(how strongly it moves with the market) is measured against it. It does not have to "
        "be one of the funds you are considering."
    ),
    "benchmark_custom": "Any Yahoo Finance ticker with enough overlapping history.",
    "market_return": (
        "Your own long-run estimate of what the broad market earns each year. The default is "
        "8%. It is an assumption you can change, not a prediction, and it sets the "
        "market-based part of each fund's expected return."
    ),
    "shrinkage": (
        "How much to trust each fund's own past return. At 0% the tool uses only the fund's "
        "past return; at 100% only a market-based estimate. 50% blends the two, which is "
        "steadier when the history is short."
    ),
    "max_weight": (
        "The most any single fund can take in a mix. Lower caps force the money to be spread out more."
    ),
    "risk_free": (
        "What a safe, cash-like investment earns each year. It is subtracted when measuring "
        "return per unit of risk. 0% is a simple default; enter a current cash rate to be stricter."
    ),
    "cvar": (
        "Which slice of bad months to average. 95% means the worst 5% of months; 99% means "
        "the worst 1%, which is very few months and so less reliable."
    ),
    "div_max_weight": "A tighter cap on any one fund, used only by the Diversified Maximum Sharpe mix.",
    "div_min_effective": (
        "The money must be at least as spread out as this many equal holdings. At 3.0 the mix "
        "has to behave like three equal holdings or more, which stops one or two funds dominating."
    ),
    "div_max_risk": (
        "The most of the mix's ups and downs any one fund may cause. At 50%, no fund can "
        "drive more than half of the risk."
    ),
    "boot_horizon": "How many years each simulated future runs for.",
    "boot_paths": (
        "How many alternative futures to simulate. More paths give smoother, more repeatable "
        "numbers but take longer."
    ),
    "boot_block": (
        "How many consecutive months stay together when history is reshuffled. 1 shuffles "
        "every month independently; 12 keeps whole years intact, which preserves more of the "
        "way good and bad periods cluster."
    ),
    "boot_seed": (
        "A number that fixes the random shuffling so you get the same results each time. "
        "Change it to see how much the answer depends on the luck of the shuffle."
    ),
    "boot_wealth": "The starting amount for the simulations. It only scales the dollar figures; percentages do not change.",
    "wf_lookback": (
        "How much past history the model may use to build each mix before a test starts. "
        "Only data from before the start date is used. 'Expanding' uses everything available up to that date."
    ),
    "wf_holding": (
        "How long each mix is held, untouched, before that test ends. Shorter holds give more "
        "tests, but each one is noisier."
    ),
    "reference_enable": "Compare every mix with a portfolio you choose, for example what you hold today.",
    "reference_custom": "Any Yahoo Finance ticker can be used.",
}


# ----------------------------------------------------------------------
# Chart and table captions
# ----------------------------------------------------------------------

CAPTIONS: dict[str, str] = {
    "common_sample": (
        "Every fund is compared over the same dates: the period when all of them existed. "
        "The newest fund sets the start date, so adding a young fund shortens everyone's "
        "history and makes results less reliable."
    ),
    "diagnostics": (
        "One row per fund, all measured over the same dates so the comparison is fair. "
        "Higher return is better; higher volatility, a deeper drawdown and a larger CVaR mean "
        "a rougher ride. The guide above explains each column."
    ),
    "risk_return": (
        "Each dot is a fund. Higher means more return; further right means a bumpier ride. "
        "Funds towards the upper left gave more return for less bumpiness. Hover a dot for more numbers."
    ),
    "growth": (
        "What $1 put into each fund at the start would have become, with dividends reinvested. "
        "Click a fund in the legend to isolate it."
    ),
    "drawdown": (
        "How far each fund sat below its previous high at every date. Deep or long dips are "
        "the stretches an investor would have had to sit through."
    ),
    "correlation": (
        "Which funds moved together. Red squares are pairs that rose and fell almost in step "
        "(little diversification between them); teal squares are pairs that behaved more "
        "independently. Funds that behave alike are grouped side by side."
    ),
    "methods": (
        "Several different ideas for splitting money across the funds, all built from the same "
        "data. Hover the small 'i' beside a name for a one-line explanation."
    ),
    "allocation": (
        "Each bar is one complete portfolio split by fund. The colours show where the money "
        "goes, so you can see which methods spread it widely and which concentrate it."
    ),
    "method_metrics": (
        "How each mix would have performed if bought at the start of the sample and left alone. "
        "This is a flattering test, because the mixes were designed using the same history. "
        "'Maximum Sharpe' means the best expected return per unit of risk in the model, so its "
        "past Sharpe need not be the highest."
    ),
    "frontier": (
        "Each grey dot is a random mix of your funds. The blue line is the best trade-off the "
        "model found: for each level of risk, the most expected return you could get. Dots well "
        "below the line take more risk for less reward. The coloured markers show where the "
        "named mixes sit."
    ),
    "frontier_build": (
        "Moving along the blue line from low risk to high risk, this shows which funds the "
        "model leans on. It reveals which funds it uses to 'buy' extra return."
    ),
    "sensitivity": (
        "The same chart drawn twice: once using each fund's own past return, once using the "
        "blended estimate. If the two lines are far apart, the answer depends heavily on what "
        "you assume about returns, which is a warning against trusting any single number."
    ),
    "cvar_frontier": (
        "The same idea, but risk is measured as how bad the worst months were (CVaR at {cvar}) "
        "instead of overall bumpiness. Further left is better. The line marks the mildest "
        "worst-months profile available at each level of expected return."
    ),
    "refit": (
        "Each method is rebuilt using only the last 3 years, only the last 5, and all history. "
        "'Capital change' is the share of your money that would move between versions. Low "
        "means the method gives a similar answer whatever window you use, which suits a "
        "buy-and-hold plan; high means the answer depends on the window."
    ),
    "realised_periods": (
        "How today's mixes would have done if bought at the start of the last 1, 3 or 5 years "
        "(or all history) and left alone. They are not rebuilt for each window. Use the "
        "selector to compare one measure at a time. The 1-year column is only a recent snapshot."
    ),
    "sim_design": (
        "The settings behind the simulation. Every portfolio receives identical shuffled "
        "histories, so differences come from the portfolio itself rather than luck."
    ),
    "boot_summary": (
        "What ending wealth could look like in the simulations. The median is the middle "
        "outcome; the 5th percentile is a poor but plausible one that only 5 out of 100 runs "
        "did worse than. These are simulated outcomes, not forecasts."
    ),
    "boot_range": (
        "Each row is one portfolio. The dark tick is the middle (median) outcome, the coloured "
        "bar covers the middle 50% of runs, and the thin line runs from a poor outcome (5th "
        "percentile) to a strong one (95th percentile). A wider bar means a less predictable result."
    ),
    "boot_dist": (
        "The same simulations as a histogram: the higher the curve, the more runs ended at that "
        "amount. Curves that overlap heavily mean the portfolios were hard to tell apart."
    ),
    "boot_cagr": (
        "The typical yearly growth rate (CAGR) each simulated run achieved. The vertical line "
        "marks 0%; anything to its left is a run that lost money."
    ),
    "boot_dd": (
        "For each simulated run, the deepest fall from a peak at any point along the way. "
        "Curves further right (closer to 0%) mean milder falls."
    ),
    "fan": (
        "How the range of outcomes widens over time for one portfolio. The line is the middle "
        "outcome, the darker band holds the middle 50% of runs and the lighter band about 90%. "
        "The dashed line is your starting amount."
    ),
    "probabilities": (
        "Simple counts from the simulations: how often a run ended below its start, and how "
        "often it grew faster than 6%, 8% and 10% a year. They count what happened in the "
        "simulated runs; they are not predictions."
    ),
    "validation": (
        "The dates covered by the tests, and how many separate start dates there were. With "
        "only a few, treat the results as hints, not proof."
    ),
    "wf_outcomes": (
        "Each row summarises a set of separate 'what if I had started then' experiments, not "
        "one continuous strategy. The question: if the model had built this mix using only "
        "what was known at the time, how did it do afterwards?"
    ),
    "div_profile": (
        "How spread out each mix really is, from several angles: how many funds are held, how "
        "evenly money and risk are divided, how similarly the holdings move, and how much the "
        "combination smooths the ride. The guide above explains every column."
    ),
    "div_frontier": (
        "Each point on the line is the most diversified mix that still delivers at least that "
        "much expected return. Going up means more expected return; going right means more "
        "diversified. The line shows what you give up in diversification to get more return."
    ),
    "summary_return": (
        "Model estimates, actual past results, simulated outcomes and the test without "
        "hindsight, side by side. They are different lenses, so read them together. There is "
        "deliberately no combined score."
    ),
    "summary_div": (
        "How spread out each mix is, and how much of your current portfolio would have to move "
        "to reach it (capital to reallocate)."
    ),
    "alloc_compare": (
        "Pick the mixes you want to compare. These are the starting splits; under buy-and-hold "
        "the weights will drift as prices move."
    ),
}


def caption(key: str, **values: str) -> str:
    """Caption text for a chart or table, with any ``{placeholders}`` filled in."""
    return CAPTIONS[key].format(**values) if values else CAPTIONS[key]


#: Hover text for the construction-method chips: one line per method.
METHOD_BLURBS: dict[str, str] = {
    name: GLOSSARY[name].plain
    for name in (
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
}


# ----------------------------------------------------------------------
# PDF blocks
# ----------------------------------------------------------------------


def _term_frame(terms: list[Term]) -> pd.DataFrame:
    rows = {}
    for term in terms:
        meaning = term.plain
        if term.example:
            meaning += f" Example: {term.example}"
        rows[term.heading] = {
            "In plain English": meaning,
            "What to look for": term.look_for or "",
        }
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "Term"
    return frame


def pdf_intro(section_key: str) -> list[ex.Block]:
    """The in-plain-English summary that opens a section's PDF."""
    guide = SECTION_GUIDES[section_key]
    return [
        ex.Callout(f"<b>In plain English.</b> {escape(guide.what)}", tone="brass"),
        ex.Heading("What to look for", 3),
        ex.Bullets([escape(item) for item in guide.look_for]),
        ex.Heading("What this page cannot tell you", 3),
        ex.Bullets([escape(item) for item in guide.cannot]),
    ]


def pdf_terms(section_key: str) -> list[ex.Block]:
    """The key-terms table that closes a section's PDF."""
    frame = _term_frame(terms_for(section_key))
    return [
        ex.Heading("Key terms on this page", 2),
        ex.Table(frame, font_size=6.3, full_width=True, max_rows=None),
    ]


def pdf_glossary() -> list[ex.Block]:
    """One alphabetical glossary for the complete report."""
    return [
        ex.PageBreak(),
        ex.Heading("Glossary", 1),
        ex.Text(
            "Every measure in this report in plain English, in alphabetical order. "
            "The past is not a forecast, and nothing here is investment advice."
        ),
        ex.Table(
            _term_frame(all_terms()), font_size=6.3, full_width=True, max_rows=None
        ),
    ]

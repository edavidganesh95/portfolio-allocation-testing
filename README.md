# Portfolio Allocation Testing

A **returns-only strategic portfolio research engine** for long-term, buy-and-hold
ETF allocation, delivered as a Streamlit application with print-ready PDF and Excel
export.

The project asks a practical question:

> If I am free to choose a new starting allocation today, what buy-and-hold ETF mix
> preserves strong expected return efficiency while avoiding an allocation dominated
> by one capital or risk bet?

It does **not** require institutional holdings data. Everything is reproducible from
public adjusted-price history.

---

## Design philosophy

The app deliberately separates three ideas that are often mixed together:

1. **Return efficiency** — what allocation has the strongest model-implied Sharpe?
2. **Diversification** — how concentrated are capital, variance risk and return streams?
3. **Validation** — do the candidate starting allocations survive different periods,
   simulated paths and out-of-sample entry dates?

The workflow runs in six sections:

| Section | Question it answers |
| --- | --- |
| 1 · Portfolio Construction | What does the common return sample look like, and what starting allocations do different construction philosophies produce? |
| 2 · Multi-Period Analysis | How stable are all applicable construction rules across 3Y / 5Y / full refits, and how do all portfolios behave over 1Y / 3Y / 5Y / full realised buy-and-hold windows? |
| 3 · Bootstrap Simulation | How do the candidate starting allocations behave across thousands of alternative sequences of the observed return record? |
| 4 · Testing Without Hindsight | Would those starting-allocation rules have worked when built using only information available at each historical entry date? |
| 5 · Diversification & Overlap | How concentrated are the portfolios in capital and risk, and what expected-return/diversification trade-off is available? |
| 6 · Summary of Findings | How do all portfolio methods compare across expected return, risk, historical evidence, simulation, out-of-sample testing, diversification and implementation distance? |

There is **no automated trade gate or selected winner**. The current/reference portfolio is simply the comparison anchor used throughout the analysis.

---

## What the engine does

### Data and expected returns

- Yahoo Finance adjusted-price ingestion via `yfinance`
- Curated research universe plus user-defined Yahoo Finance tickers
- Common-sample return alignment; monthly / weekly / daily return support
- User-selectable broad-market benchmark (ACWI, VT, SPY, VTI, MSCI USA proxy,
  developed ex-US or custom Yahoo ticker)
- Separate long-run market-return assumption
- Beta-implied expected-return anchor:
  `rf + beta × (market return assumption − rf)`
- User-controlled shrinkage between historical CAGR and the beta-implied anchor
- Historical E(r), shrunk E(r), or side-by-side frontier comparison

### Diagnostics

- CAGR, volatility, Sharpe, Sortino, max drawdown and historical CVaR
- Growth-of-$1 and drawdown views
- Correlation matrix ordered by hierarchical clustering
- Ledoit-Wolf shrinkage covariance

### Portfolio construction

- Equal Weight
- Minimum Variance
- Risk Parity
- Maximum Diversification
- Maximum Sharpe using market-anchored expected returns
- **Diversified Maximum Sharpe** — the same expected-Sharpe objective, but subject
  to explicit diversification guardrails:
  - tighter maximum single-ETF weight;
  - minimum effective ETF count;
  - maximum single variance-risk contribution share
- Minimum CVaR
- Maximum Return / CVaR
- Exact mean-variance frontier and return-CVaR opportunity set
- Historical-E(r) vs shrunk-E(r) frontier sensitivity

### Multi-period analysis

The old mixed-objective “robust consensus” layer has been removed.

Instead the app shows:

- **Every applicable construction rule refitted on 3Y, 5Y and full history**, including HRP; the Reference Portfolio remains fixed and Equal Weight is mechanically unchanged when the universe is unchanged
- **Realised buy-and-hold metrics over 1Y, 3Y, 5Y and full history** for the complete portfolio set
- 1Y is diagnostic only; it is not used to optimise the strategic portfolio because
  roughly 12 monthly observations are too thin for an 11-ETF strategic optimisation

The point is transparency: if weights swing wildly across lookbacks, the user can see
that directly rather than relying on a synthetic stability score.

### Interaction model

Research controls and in-page portfolio/metric selectors use Streamlit fragments. Changing a dropdown, slider or display filter updates only the relevant control panel rather than re-running and visually refreshing the full research page. The full analysis is recomputed only when **Run research** is pressed.

### Bootstrap simulation

- Circular moving-block bootstrap of monthly **asset** returns
- Shared sampled sequences across portfolios for fair comparison
- Starting weights are set once and then drift naturally
- No periodic rebalance inside the simulated horizon
- Terminal wealth, CAGR, loss probability and maximum-drawdown distributions

The bootstrap tests **path uncertainty**. It does not re-estimate a new optimizer inside
each bootstrap path, so it is not a parameter-bootstrap or nested optimisation test.

### Testing without hindsight

- Rolling or expanding training windows
- No look-ahead: each historical entry portfolio is fitted only on prior returns
- Each fold is an **independent buy-and-hold entry experiment**
- Starting weights drift naturally throughout the holding block
- No recurring turnover or transaction-cost assumption
- Out-of-sample return, drawdown, block win-rate and weight-drift diagnostics

### Diversification & overlap

- Every construction method plus the current/reference portfolio and HRP
- Actual ETF count (ignoring only numerical solver dust) and observed annualised portfolio volatility
- Effective Holdings — capital concentration via `1 / sum(w²)`
- Effective Risk Bets — concentration of portfolio variance contribution
- Weighted portfolio correlation — return-stream overlap
- Diversification Ratio and largest single risk contributor/share
- Hierarchical Risk Parity (HRP) as a diversification-first counterfactual built from correlation clustering and recursive risk allocation, with no expected-return forecast in its construction
- **Return–Diversification Frontier**: for a grid of minimum expected-return targets, maximise Diversification Ratio under the same long-only and maximum-weight constraints, then overlay all named portfolios
- Optional advanced risk-contribution and correlation-cluster detail

This is **economic return overlap**, not constituent-level holdings overlap. DR is one diversification lens, so it is shown alongside capital concentration and risk-contribution diagnostics rather than treated as a standalone score.

### Summary of Findings

The final investor-facing section is a neutral comparison rather than an automated portfolio-selection verdict. It includes **all portfolio construction methods** plus HRP and the current/reference allocation.

The summary brings together:

- expected return E(r), expected volatility and expected Sharpe;
- realised historical buy-and-hold CAGR;
- bootstrap median CAGR and 5th-percentile terminal wealth;
- out-of-sample median return and a plain-English win count versus the current portfolio;
- Actual ETFs, Effective Holdings, Effective Risk Bets, Diversification Ratio and Largest Risk Share;
- **Capital to Reallocate**, calculated as `0.5 × sum(abs(w − w_current))`.

The starting-allocation chart is user-selectable so ten methods do not have to be shown at once. Detailed bootstrap, walk-forward and diversification evidence remains in the Excel appendix rather than consuming a final standalone report page. No combined score or winner is produced.
---

## Reference portfolio

The app supports an optional **user-defined reference portfolio**, evaluated over the
same historical sample as the candidate allocations.

`config/examples.yaml` contains a sample TMFC / VWO / ACWI portfolio purely as a
starting example. Nothing in the engine requires those holdings.

---

## Reporting

Every section has an **Export** panel offering:

- a print-ready A4 landscape PDF with run parameters, running header, page numbers
  and research disclaimer;
- a formatted Excel workbook with real numeric cells and consistent formats.

A **Complete research report** bundles all six sections into one document. The retail-facing PDF ends on **Summary of Findings**; detailed evidence remains available in the workbook appendix. Every PDF page carries the footer `Personal research project · For educational purposes only · Not financial advice`.

---

## Visual system — *Ledger*

The design system lives in `src/theme.py` and is shared by the Streamlit UI, Altair
charts and PDF renderer.

| Token | Value | Role |
| --- | --- | --- |
| Paper | `#FAF8F5` | Warm off-white ground |
| Ink | `#10151C` | Primary text |
| Navy | `#17395E` | Structure and primary brand |
| Brass | `#A9762F` | Reference / answer emphasis |
| Positive / Negative / Caution | `#237A5B` / `#9A2F28` / `#B07A18` | Semantic states |

A ticker retains the same colour across the app and exported report.

---

## Repository structure

```text
.
├── app.py                  # composition layer — controls, layout, exports
├── src/
│   ├── config.py           # universe / example-portfolio loading
│   ├── data.py             # price download, alignment, return conversion
│   ├── metrics.py          # B&H returns, CAGR, volatility, drawdown, CVaR
│   ├── expected_returns.py # benchmark betas and expected-return shrinkage
│   ├── risk.py             # Ledoit-Wolf covariance, risk contributions
│   ├── optimization.py     # construction methods + diversified max Sharpe
│   ├── frontier.py         # mean-variance, CVaR and return-diversification frontiers
│   ├── multiperiod.py      # 1Y/3Y/5Y/full sensitivity analysis
│   ├── bootstrap.py        # moving-block B&H simulation
│   ├── walkforward.py      # independent OOS B&H entry tests
│   ├── diversification.py  # HRP and concentration diagnostics
│   ├── compute.py          # memoised wrappers around the engine
│   ├── analytics.py        # presentation-layer derived views
│   ├── theme.py            # Ledger design tokens
│   ├── charts.py           # on-screen Altair charts
│   ├── tables.py           # formatting and workbook export
│   ├── exporting.py        # print-ready PDF engine
│   └── ui.py               # Streamlit components / CSS
├── config/
│   ├── universe.yaml
│   └── examples.yaml
└── tests/
```

---

## Run locally

```bash
python -m venv .venv
```

```bash
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
streamlit run app.py
```

Configure the universe, expected-return assumptions, diversification guardrails and optional reference
portfolio, then press **Run research**.

## Run tests

```bash
pip install -r requirements-dev.txt
pytest
```

The test suite is designed to run offline by stubbing market data with deterministic
synthetic prices.

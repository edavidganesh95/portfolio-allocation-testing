# Changelog

Newest first. The entries record *why* the methodology changed, not just what moved.

## v5.2 — review fixes and rename

**Renamed** to *Portfolio Allocation Testing* (page title, masthead, PDF cover and header,
README, package name and export filenames).

**Data**
- The unfinished current month (or week) is no longer counted as a full period.
  Resampling labels the last bucket with its period end however few days it holds, so a
  21 September price became a "30 September" month and fed every monthly statistic, the
  bootstrap sample and the final out-of-sample fold. `to_returns` now drops a trailing
  period that is still in progress; month-ends that fall on a weekend or market holiday
  (for example Good Friday) are recognised as complete. `drop_incomplete=False` restores
  the old behaviour.
- Section 1 states the last price date and that the unfinished period is excluded; the
  run parameters record *Data through*.

**Out-of-sample testing**
- "Beat Current Portfolio" counts are quoted against the tests a method actually took
  part in. Previously the denominator was the total number of folds, so a method that
  failed to fit in one fold could print `5 of 5 (100%)` for a true `4 of 4`.
- The Section 4 heading follows the selected holding period; for holds under 12 months a
  note explains that returns are annualised from very little data.

**Construction**
- A method the solver could not converge on is no longer dropped silently: Section 1
  names it, explains the usual cause and lists the solver messages.

**Bootstrap**
- Memory is bounded. The simulation is streamed in chunks and only as many portfolio
  paths as fit a fixed budget are held at once, instead of a `paths × months × assets`
  tensor. The largest selectable run (25,000 paths, 20 years) falls from about 1.8 GB to
  about 0.3 GB; the default run from about 360 MB to about 170 MB. Results equal the
  reference implementation to floating-point precision and do not depend on the budget.
- The percentile fan takes all five quantiles in one pass, so the section is also faster
  at every setting (default run about 4.0 s → 2.1 s).

**Repository**
- Removed the retired robustness and decision engines (`robustness.py`, `decision.py`),
  their charts and tests, the unused peer-mean shrinkage helper, and the empty
  `notebooks/` folder. Lint and format checks are clean.

## v5.1 — final-review interaction and multi-period update

- Multi-period analysis refits **every** construction method on 3Y, 5Y and full history
  (HRP is rebuilt from each window's covariance and correlation hierarchy) and reports how
  much capital each shorter-window fit moves relative to the full-history fit. The
  Reference Portfolio stays fixed; Equal Weight is mechanically unchanged.
- Realised 1Y / 3Y / 5Y / full buy-and-hold metrics cover the complete portfolio set, with
  a method selector and a metric selector.
- Sidebar controls and in-page selectors run in Streamlit fragments: editing a control
  updates only its own panel, and the full report reruns only when **Run research** is
  pressed.

## v5 — retail-research and diversification rebuild

- Reworked copy for a retail-investor-first reading layer; formulas and methodology moved
  into optional detail panels. The final section is **Summary of Findings** and no longer
  implies a selected or defended portfolio.
- Expected returns are shown as **E(r)**, with the historical, shrunk or both views. The
  shrinkage anchor is unchanged: a benchmark-beta market prior, so the anchor does not
  depend on which ETFs are in the candidate universe.
- Bootstrap covers every static portfolio, including HRP and the current allocation.
  Walk-forward tests every construction method plus HRP and reports win rates as counts
  such as `2 of 4 (50%)`.
- Diversification profile adds **Actual ETFs** and observed **Historical Volatility**
  alongside Effective Holdings, Effective Risk Bets, Weighted Correlation, Diversification
  Ratio and largest risk contribution. HRP is a correlation-clustering, recursive-risk
  benchmark that uses no expected-return forecast.
- New **Return–Diversification Frontier**: for successive minimum expected-return targets,
  the engine maximises the Diversification Ratio under the long-only, full-investment and
  maximum-weight rules.
- Summary of Findings adds **Capital to Reallocate** = ½ Σ|w − w_current|, an optional
  delta view against the current portfolio, and no combined score or winner.
- Every PDF page carries: *Personal research project · For educational purposes only ·
  Not financial advice.*

## v4 — buy-and-hold portfolio-selection methodology

- Removed the *Robustness Analysis / Robust Consensus* workflow and the automated
  NO TRADE / robust-band decision engine. A consensus of unrelated objectives is not an
  optimum of any of them, and an automated trade gate is the wrong shape for a one-time
  buy-and-hold choice.
- Added **Multi-Period Analysis**: 1Y / 3Y / 5Y / Full realised buy-and-hold metrics, and
  3Y / 5Y / Full refits of one consistent allocation rule. 1Y is diagnostic only and is
  never used for optimisation.
- Added **Diversified Maximum Sharpe**: maximises market-prior expected Sharpe subject to a
  maximum single-ETF weight, a minimum effective ETF count and a maximum single
  variance-risk contribution.
- Added **Effective Risk Bets** to the diversification analysis.
- Bootstrap remains a path-risk test rather than a nested parameter bootstrap;
  walk-forward remains independent historical buy-and-hold entry testing with natural
  drift.

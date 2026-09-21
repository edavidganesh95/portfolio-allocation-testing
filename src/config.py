from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIVERSE_PATH = ROOT / "config" / "universe.yaml"
DEFAULT_EXAMPLES_PATH = ROOT / "config" / "examples.yaml"


@dataclass(frozen=True)
class AssetMeta:
    ticker: str
    name: str
    role: str


@dataclass(frozen=True)
class UniverseConfig:
    benchmark: str
    assets: dict[str, AssetMeta]


@dataclass(frozen=True)
class ReferencePortfolioExample:
    name: str
    weights: dict[str, float]


def load_universe(path: Path | str = DEFAULT_UNIVERSE_PATH) -> UniverseConfig:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assets = {
        ticker: AssetMeta(
            ticker=ticker,
            name=meta["name"],
            role=meta["role"],
        )
        for ticker, meta in payload["assets"].items()
    }

    return UniverseConfig(
        benchmark=payload["benchmark"],
        assets=assets,
    )


def load_reference_example(
    path: Path | str = DEFAULT_EXAMPLES_PATH,
) -> ReferencePortfolioExample:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    ref = payload["reference_portfolio"]
    return ReferencePortfolioExample(
        name=str(ref.get("name", "Example reference portfolio")),
        weights={str(k): float(v) for k, v in ref["weights"].items()},
    )


def parse_ticker_input(raw: str) -> list[str]:
    """
    Parse a user-entered list of Yahoo Finance tickers.

    Accepts comma, semicolon, whitespace or newline separators.
    Preserves first-seen order and normalizes to uppercase.
    """
    if not raw:
        return []

    normalized = raw.replace(",", " ").replace(";", " ")
    tickers = []
    seen = set()

    for token in normalized.split():
        ticker = token.strip().upper()
        if ticker and ticker not in seen:
            tickers.append(ticker)
            seen.add(ticker)

    return tickers

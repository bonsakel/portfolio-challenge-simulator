"""Download ETF and VIX price data and create aligned, auditable datasets."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import pandas as pd
import yfinance as yf
import yaml


ROOT = Path(__file__).resolve().parents[2]
CBOE_VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")


def load_config(path: Path = ROOT / "configs/experiment.yaml") -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")
    return config


def configured_date_range(
    config: dict[str, Any], warmup_trading_days: int
) -> tuple[date, date]:
    periods = (
        period
        for event in config["events"].values()
        for period in event.values()
    )
    starts: list[date] = []
    ends: list[date] = []
    for start, end in periods:
        starts.append(date.fromisoformat(start))
        ends.append(date.fromisoformat(end))
    if not starts:
        raise ValueError("No experiment periods are configured")
    calendar_buffer = warmup_trading_days * 7 // 5 + 45
    return (
        min(starts) - timedelta(days=calendar_buffer),
        max(ends) + timedelta(days=1),
    )


def output_paths(root: Path, tickers: list[str]) -> list[Path]:
    return [
        *(root / "data/raw/etf_prices" / f"{ticker}.csv" for ticker in tickers),
        root / "data/raw/vix/VIX_History.csv",
        root / "data/processed/daily_returns/adjusted_close.csv",
        root / "data/processed/daily_returns/daily_returns.csv",
        root / "data/processed/daily_returns/vix.csv",
        root / "data/processed/metadata.json",
    ]


def ensure_output_paths_available(paths: list[Path], force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        names = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Output files already exist; pass --force to replace them:\n" + names
        )


def normalized_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    index = pd.DatetimeIndex(pd.to_datetime(result.index))
    if index.tz is not None:
        index = index.tz_convert(None)
    result.index = index.normalize()
    result.index.name = "Date"
    return result.sort_index()


def extract_price_columns(download: pd.DataFrame, ticker: str) -> pd.DataFrame:
    selected: dict[str, pd.Series] = {}
    for field in PRICE_COLUMNS:
        matches = [
            column
            for column in download.columns
            if field.lower() in (
                str(part).lower()
                for part in (column if isinstance(column, tuple) else (column,))
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {field!r} column for {ticker}; "
                f"found {len(matches)}"
            )
        selected[field] = download[matches[0]]
    return normalized_frame(pd.DataFrame(selected, index=download.index))


def download_etfs(
    tickers: list[str], start: date, end: date, raw_dir: Path
) -> tuple[dict[str, pd.Series], dict[str, dict[str, Any]]]:
    prices: dict[str, pd.Series] = {}
    coverage: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        download = yf.download(
            ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
        )
        if download.empty:
            raise RuntimeError(f"Yahoo Finance returned no data for {ticker}")
        frame = extract_price_columns(download, ticker)
        if frame.index.has_duplicates:
            raise ValueError(f"Yahoo Finance returned duplicate dates for {ticker}")
        if (frame["Adj Close"].dropna() <= 0).any():
            raise ValueError(f"Adjusted close contains nonpositive values for {ticker}")
        if frame["Adj Close"].notna().sum() == 0:
            raise ValueError(f"Adjusted close has no valid values for {ticker}")
        frame.to_csv(raw_dir / f"{ticker}.csv", index_label="Date")
        series = frame["Adj Close"].rename(ticker)
        prices[ticker] = series
        coverage[ticker] = {
            "source": f"Yahoo Finance ({ticker}) via yfinance",
            "first_date": series.first_valid_index().date().isoformat(),
            "last_date": series.last_valid_index().date().isoformat(),
            "missing_adjusted_close_rows": int(series.isna().sum()),
            "raw_file": f"data/raw/etf_prices/{ticker}.csv",
        }
    return prices, coverage


def download_vix(raw_path: Path, start: date, end: date) -> tuple[pd.Series, dict[str, Any]]:
    with urlopen(CBOE_VIX_URL, timeout=30) as response:
        raw_path.write_bytes(response.read())

    frame = pd.read_csv(raw_path)
    columns = {str(column).strip().upper(): column for column in frame.columns}
    if "DATE" not in columns or "CLOSE" not in columns:
        raise ValueError(
            f"Unexpected CBOE VIX columns: {', '.join(map(str, frame.columns))}"
        )
    vix = pd.DataFrame(
        {
            "VIX": pd.to_numeric(frame[columns["CLOSE"]], errors="coerce").to_numpy()
        },
        index=pd.to_datetime(frame[columns["DATE"]], errors="coerce"),
    )
    vix = normalized_frame(vix).dropna()
    vix = vix.loc[(vix.index >= pd.Timestamp(start)) & (vix.index < pd.Timestamp(end))]
    if vix.empty:
        raise RuntimeError("CBOE returned no VIX observations in the requested date range")
    if vix.index.has_duplicates or (vix["VIX"] <= 0).any():
        raise ValueError("CBOE VIX data contains duplicate dates or invalid values")
    series = vix["VIX"].rename("VIX")
    return series, {
        "first_date": series.index.min().date().isoformat(),
        "last_date": series.index.max().date().isoformat(),
        "missing_close_rows_excluded": int(frame[columns["CLOSE"]].isna().sum()),
        "raw_file": "data/raw/vix/VIX_History.csv",
        "source_url": CBOE_VIX_URL,
    }


def prepare_aligned_data(
    prices: dict[str, pd.Series], vix: pd.Series, tickers: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    asset_prices = pd.concat(
        [prices[ticker] for ticker in tickers], axis=1, join="outer", sort=True
    ).sort_index()
    combined = pd.concat(
        [asset_prices, vix], axis=1, join="outer", sort=True
    ).sort_index()
    aligned = combined.dropna()
    returns = asset_prices.pct_change(fill_method=None)
    returns = returns.loc[aligned.index].dropna()
    return combined, aligned, returns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/experiment.yaml"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing raw and processed data files",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    tickers = config["assets"]
    warmup_days = int(config["data"]["warmup_trading_days"])
    start, end = configured_date_range(config, warmup_days)
    output_files = output_paths(ROOT, tickers)
    ensure_output_paths_available(output_files, args.force)

    etf_raw_dir = ROOT / "data/raw/etf_prices"
    vix_raw_dir = ROOT / "data/raw/vix"
    processed_dir = ROOT / "data/processed/daily_returns"
    for directory in (etf_raw_dir, vix_raw_dir, processed_dir):
        directory.mkdir(parents=True, exist_ok=True)

    prices, etf_coverage = download_etfs(tickers, start, end, etf_raw_dir)
    vix, vix_coverage = download_vix(
        vix_raw_dir / "VIX_History.csv", start, end
    )
    combined, aligned, returns = prepare_aligned_data(prices, vix, tickers)
    if aligned.empty:
        raise ValueError("No dates have valid observations for every ETF and VIX")
    if returns.empty:
        raise ValueError("Not enough aligned observations to calculate daily returns")

    first_training_start = min(
        date.fromisoformat(event["training"][0])
        for event in config["events"].values()
    )
    warmup_returns = returns.loc[returns.index < pd.Timestamp(first_training_start)]
    if len(warmup_returns) < warmup_days:
        raise ValueError(
            f"Need {warmup_days} pre-training daily returns for rolling-window "
            f"warm-up; found {len(warmup_returns)}"
        )

    period_coverage: dict[str, dict[str, dict[str, Any]]] = {}
    for event_name, event in config["events"].items():
        period_coverage[event_name] = {}
        for period_name, (period_start, period_end) in event.items():
            period_rows = aligned.loc[period_start:period_end]
            if period_rows.empty:
                raise ValueError(
                    f"No aligned observations for {event_name} {period_name} "
                    f"({period_start} to {period_end})"
                )
            period_coverage[event_name][period_name] = {
                "first_observation": period_rows.index.min().date().isoformat(),
                "last_observation": period_rows.index.max().date().isoformat(),
                "rows": len(period_rows),
            }

    adjusted_close_path = processed_dir / "adjusted_close.csv"
    returns_path = processed_dir / "daily_returns.csv"
    vix_path = processed_dir / "vix.csv"
    aligned[tickers].to_csv(adjusted_close_path, index_label="Date")
    returns.to_csv(returns_path, index_label="Date")
    aligned[["VIX"]].to_csv(vix_path, index_label="Date")

    metadata = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_date_range": {
            "start_inclusive": start.isoformat(),
            "end_exclusive": end.isoformat(),
        },
        "warmup": {
            "required_trading_days": warmup_days,
            "available_daily_returns_before_first_training_start": len(warmup_returns),
            "purpose": "Initialize rolling windows and observations without lookahead.",
        },
        "sources": {
            "etf_prices": "Yahoo Finance via yfinance",
            "vix": "CBOE",
        },
        "package_versions": {
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scipy": version("scipy"),
            "yfinance": version("yfinance"),
            "PyPortfolioOpt": version("PyPortfolioOpt"),
            "gymnasium": version("gymnasium"),
            "stable-baselines3": version("stable-baselines3"),
            "matplotlib": version("matplotlib"),
            "PyYAML": version("PyYAML"),
            "pytest": version("pytest"),
        },
        "adjustment": (
            "Adjusted Close is used for ETF return calculations. Raw downloads "
            "retain OHLC, adjusted close, and volume."
        ),
        "missing_data_treatment": (
            "No forward-filling or interpolation. Aligned outputs exclude "
            "dates without valid observations for every ETF and VIX. Returns "
            "use adjacent source observations; rows with missing current or "
            "previous ETF prices are excluded to avoid treating multi-day "
            "price changes as daily returns."
        ),
        "etfs": etf_coverage,
        "vix": vix_coverage,
        "configured_period_coverage": period_coverage,
        "aligned_data": {
            "first_date": aligned.index.min().date().isoformat(),
            "last_date": aligned.index.max().date().isoformat(),
            "rows": len(aligned),
            "rows_excluded_for_missing_observations": int(combined.isna().any(axis=1).sum()),
            "daily_return_rows": len(returns),
            "daily_return_first_date": returns.index.min().date().isoformat(),
            "daily_return_last_date": returns.index.max().date().isoformat(),
            "assets": tickers,
            "files": {
                "adjusted_close": "data/processed/daily_returns/adjusted_close.csv",
                "daily_returns": "data/processed/daily_returns/daily_returns.csv",
                "vix": "data/processed/daily_returns/vix.csv",
            },
        },
    }
    with (ROOT / "data/processed/metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    print(
        f"Saved {len(aligned)} aligned observations from "
        f"{aligned.index.min().date()} through {aligned.index.max().date()}."
    )
    print(f"Metadata: {ROOT / 'data/processed/metadata.json'}")


if __name__ == "__main__":
    main()

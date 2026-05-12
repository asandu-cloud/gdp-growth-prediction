"""
layer2_preprocessing.py -- clean daily yields, compute spreads,
build quarterly panel for forecasting models.

Usage:
    python thesis_pipeline/layer2_preprocessing.py
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from thesis_pipeline.config import (
    DATA_PROCESSED, DATA_RAW, GDP_SERIES_ID, MATURITY_IDS,
)


# ── Loading ────────────────────────────────────────────

def load_raw_csv(series_id: str) -> pd.Series:
    path = DATA_RAW / f"{series_id}.csv"
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df.iloc[:, 0]


def load_yield_panel() -> pd.DataFrame:
    series = {sid: load_raw_csv(sid) for sid in MATURITY_IDS}
    return pd.DataFrame(series)


# ── Daily series ───────────────────────────────────────

def build_daily_yields(raw_panel: pd.DataFrame) -> pd.DataFrame:
    n_before = len(raw_panel)
    clean = raw_panel.dropna(how="any").sort_index()
    n_after = len(clean)
    print(f"  Raw dates: {n_before}")
    print(f"  Clean dates: {n_after} ({n_before - n_after} dropped — missing maturities)")
    return clean


def build_daily_term_spread(daily_yields: pd.DataFrame) -> pd.Series:
    spread = daily_yields["DGS10"] - daily_yields["DGS2"]
    spread.name = "term_spread_10Y_2Y"
    return spread


# ── Quarterly aggregation ──────────────────────────────

def build_quarterly_panel(daily_yields: pd.DataFrame,
                          daily_term_spread: pd.Series) -> pd.DataFrame:
    # Yields: last business day of quarter
    q_yields = daily_yields.resample("QE").last()

    # Term spread: last business day of quarter
    q_term_spread = daily_term_spread.resample("QE").last()

    # Fed funds rate (daily, 7-day freq): last value in quarter
    dff = load_raw_csv("DFF").dropna()
    q_dff = dff.resample("QE").last()
    q_dff.name = "DFF"

    # Credit spread: BAA - AAA (monthly), last month of quarter
    baa = load_raw_csv("BAA").dropna()
    aaa = load_raw_csv("AAA").dropna()
    credit_spread = baa - aaa
    credit_spread.name = "credit_spread_Baa_Aaa"
    q_credit_spread = credit_spread.resample("QE").last()

    # S&P 500: quarterly return (% change between end-of-quarter closes)
    sp500 = load_raw_csv("SP500").dropna()
    sp500_eoq = sp500.resample("QE").last()
    q_sp500_ret = sp500_eoq.pct_change() * 100
    q_sp500_ret.name = "SP500_return"

    # Initial claims 4-week MA (weekly): end-of-quarter value
    ic4wsa = load_raw_csv("IC4WSA").dropna()
    q_ic4wsa = ic4wsa.resample("QE").last()
    q_ic4wsa.name = "IC4WSA"

    # NAPM / ISM Manufacturing PMI (monthly diffusion index): last month of quarter
    napm = load_raw_csv("NAPM").dropna()
    q_napm = napm.resample("QE").last()
    q_napm.name = "NAPM"

    # GDP: already quarterly, re-index to quarter-end
    gdp = load_raw_csv(GDP_SERIES_ID).dropna()
    q_gdp = gdp.resample("QE").last()
    q_gdp.name = "GDP_growth"

    # Lagged GDP
    q_gdp_lag = q_gdp.shift(1)
    q_gdp_lag.name = "GDP_growth_lag1"

    # Assemble
    quarterly = pd.DataFrame({
        "GDP_growth": q_gdp,
        "GDP_growth_lag1": q_gdp_lag,
        "term_spread_10Y_2Y": q_term_spread,
        "DFF": q_dff,
        "credit_spread_Baa_Aaa": q_credit_spread,
        "SP500_return": q_sp500_ret,
        "IC4WSA": q_ic4wsa,
        "NAPM": q_napm,
    })

    for col in MATURITY_IDS:
        quarterly[col] = q_yields[col]

    quarterly.index.name = "date"

    # Trim to sample period: start at 1982, drop quarters without GDP
    quarterly = quarterly.loc["1982":]
    quarterly = quarterly.loc[quarterly["GDP_growth"].notna()]

    return quarterly


# ── Summary ────────────────────────────────────────────

def print_summary(daily_yields: pd.DataFrame,
                  daily_term_spread: pd.Series,
                  quarterly: pd.DataFrame):
    print(f"\n--- Daily Yields ---")
    print(f"  Date range: {daily_yields.index.min().date()} to {daily_yields.index.max().date()}")
    print(f"  Observations: {len(daily_yields):,}")
    print(f"  Maturities: {', '.join(daily_yields.columns)}")

    print(f"\n--- Daily Term Spread (10Y - 2Y) ---")
    print(f"  Date range: {daily_term_spread.index.min().date()} to {daily_term_spread.index.max().date()}")
    print(f"  Observations: {len(daily_term_spread):,}")
    print(f"  Min: {daily_term_spread.min():.2f}  Max: {daily_term_spread.max():.2f}  Mean: {daily_term_spread.mean():.2f}")

    inversions = (daily_term_spread < 0).sum()
    print(f"  Yield curve inversions (spread < 0): {inversions:,} days")

    print(f"\n--- Quarterly Panel ---")
    print(f"  Date range: {quarterly.index.min().date()} to {quarterly.index.max().date()}")
    print(f"  Observations: {len(quarterly)}")
    print(f"  Columns: {len(quarterly.columns)}")

    nan_counts = quarterly.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if len(cols_with_nan) > 0:
        print(f"  NaN counts:")
        for col, count in cols_with_nan.items():
            print(f"    {col}: {count}")
    else:
        print(f"  NaN counts: none")

    # Anomaly checks
    neg_yields = (daily_yields < 0).any(axis=1).sum()
    if neg_yields > 0:
        print(f"\n  ANOMALY: Negative yields on {neg_yields} dates")

    neg_credit = (quarterly["credit_spread_Baa_Aaa"] < 0).sum()
    if neg_credit > 0:
        print(f"  ANOMALY: Negative credit spread on {neg_credit} quarters")


# ── Main ───────────────────────────────────────────────

def main():
    print("=" * 60)
    print("LAYER 2: Yield Curve Preprocessing")
    print("=" * 60)

    # Build daily series
    raw_panel = load_yield_panel()
    daily_yields = build_daily_yields(raw_panel)
    daily_term_spread = build_daily_term_spread(daily_yields)

    # Build quarterly panel
    quarterly = build_quarterly_panel(daily_yields, daily_term_spread)

    # Save outputs
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    daily_yields.index.name = "date"
    daily_yields.to_csv(DATA_PROCESSED / "daily_yields.csv")

    daily_term_spread.to_frame().to_csv(DATA_PROCESSED / "daily_term_spread.csv")

    quarterly.to_csv(DATA_PROCESSED / "quarterly_panel.csv")

    # Summary
    print_summary(daily_yields, daily_term_spread, quarterly)

    print(f"\n{'=' * 60}")
    print(f"Layer 2 complete. Output saved to {DATA_PROCESSED}/")
    print(f"  daily_yields.csv")
    print(f"  daily_term_spread.csv")
    print(f"  quarterly_panel.csv")


if __name__ == "__main__":
    main()

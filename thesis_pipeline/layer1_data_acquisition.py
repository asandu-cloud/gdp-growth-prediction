"""
layer1_data_acquisition.py -- download all raw FRED series for the thesis.

Usage:
    python thesis_pipeline/layer1_data_acquisition.py
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# Allow running as a standalone script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thesis_pipeline.config import ALL_SERIES, DATA_RAW, START_DATE
from thesis_pipeline.utils import get_fred_client, save_series_csv, write_metadata


def download_fred_series(fred, series_id: str, label: str,
                         max_retries: int = 3, backoff: int = 5) -> pd.Series | None:
    """Download a single series from FRED with retry on transient errors."""
    print(f"  Downloading {series_id} ({label})...")
    for attempt in range(1, max_retries + 1):
        try:
            data = fred.get_series(series_id, observation_start=START_DATE)
            print(f"    {len(data)} obs, {data.index.min().date()} to {data.index.max().date()}")
            return data
        except Exception as e:
            if attempt < max_retries and "Internal Server Error" in str(e):
                wait = backoff * attempt
                print(f"    Attempt {attempt} failed (server error), retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    FAILED: {e}", file=sys.stderr)
                return None


def download_sp500_yfinance() -> pd.Series | None:
    """Fallback: download S&P 500 daily close from Yahoo Finance."""
    try:
        import yfinance as yf
        print("  Downloading ^GSPC from Yahoo Finance (fallback)...")
        ticker = yf.Ticker("^GSPC")
        df = ticker.history(start=START_DATE, auto_adjust=True)
        if df.empty:
            print("    FAILED: yfinance returned empty data", file=sys.stderr)
            return None
        series = df["Close"].rename("SP500")
        series.index = series.index.tz_localize(None)
        print(f"    {len(series)} obs, {series.index.min().date()} to {series.index.max().date()}")
        return series
    except Exception as e:
        print(f"    FAILED: {e}", file=sys.stderr)
        return None


def download_all_series() -> dict:
    """
    Download every series in ALL_SERIES from FRED.
    Falls back to yfinance for SP500 if FRED data is truncated.
    """
    fred = get_fred_client()
    downloaded = {}
    sp500_source = "FRED"

    for series_id, label in ALL_SERIES.items():
        data = download_fred_series(fred, series_id, label)
        downloaded[series_id] = data

    # SP500 fallback: FRED often restricts this to ~10 years
    sp500 = downloaded.get("SP500")
    if sp500 is not None and sp500.index.min().year > 1990:
        print(f"\n  SP500 from FRED starts at {sp500.index.min().date()} (truncated).")
        yf_data = download_sp500_yfinance()
        if yf_data is not None:
            downloaded["SP500"] = yf_data
            sp500_source = "yfinance"
        else:
            print("  WARNING: yfinance fallback also failed. SP500 will be truncated.")
    elif sp500 is None:
        yf_data = download_sp500_yfinance()
        if yf_data is not None:
            downloaded["SP500"] = yf_data
            sp500_source = "yfinance"

    return downloaded, sp500_source


def build_metadata(downloaded: dict, sp500_source: str) -> dict:
    """Build metadata dictionary for all downloaded series."""
    meta = {
        "download_timestamp": datetime.now().isoformat(),
        "start_date_requested": START_DATE,
        "sp500_source": sp500_source,
        "series": {},
    }

    for series_id, data in downloaded.items():
        label = ALL_SERIES[series_id]
        if data is not None and len(data) > 0:
            non_null = data.dropna()
            meta["series"][series_id] = {
                "label": label,
                "total_observations": len(data),
                "non_null_observations": len(non_null),
                "first_date": str(data.index.min().date()),
                "last_date": str(data.index.max().date()),
            }
        else:
            meta["series"][series_id] = {
                "label": label,
                "error": "Download failed or returned empty data",
            }
    return meta


def main():
    print("=" * 60)
    print("LAYER 1: FRED Data Acquisition")
    print("=" * 60)

    downloaded, sp500_source = download_all_series()

    for series_id, data in downloaded.items():
        if data is not None:
            save_series_csv(data, series_id, DATA_RAW)

    meta = build_metadata(downloaded, sp500_source)
    meta_path = write_metadata(meta, DATA_RAW)

    success = sum(1 for v in downloaded.values() if v is not None)
    failed = sum(1 for v in downloaded.values() if v is None)

    print(f"\n{'=' * 60}")
    print(f"Done. {success}/{success + failed} series saved to {DATA_RAW}")
    if failed:
        print(f"WARNING: {failed} series failed to download.")
    print(f"SP500 source: {sp500_source}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()

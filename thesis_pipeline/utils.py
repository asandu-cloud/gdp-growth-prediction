"""
utils.py -- shared helper functions for the thesis pipeline.
"""
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred


def get_fred_client() -> Fred:
    """
    Build an authenticated fredapi.Fred client.
    Loads the API key from .env in the project root.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY not found. Set it in the .env file at the project root."
        )
    return Fred(api_key=api_key)


def save_series_csv(series: pd.Series, series_id: str, output_dir: Path) -> Path:
    """Save a pandas Series to CSV in the given directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{series_id}.csv"
    df = series.to_frame(name=series_id)
    df.index.name = "date"
    df.to_csv(path)
    return path


def write_metadata(metadata: dict, output_dir: Path,
                   filename: str = "download_metadata.json") -> Path:
    """Write a metadata dictionary to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    return path

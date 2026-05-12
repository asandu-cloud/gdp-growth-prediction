"""
layer3_entropy.py -- compute Shannon and Tsallis entropy of the yield curve
under two normalisation methods (absolute yields and yield gradients).

Usage:
    python thesis_pipeline/layer3_entropy.py
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from thesis_pipeline.config import (
    DATA_ENTROPY, DATA_PROCESSED, EPSILON, TSALLIS_Q_VALUES,
)


# ── Loading ────────────────────────────────────────────

def load_daily_yields() -> pd.DataFrame:
    path = DATA_PROCESSED / "daily_yields.csv"
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


# ── Normalisation ──────────────────────────────────────

def normalise_method_a(yields: pd.DataFrame) -> pd.DataFrame:
    """Absolute yield normalisation: p_i = y_i / sum(y), with shift if min <= 0."""
    shifted = yields.copy()
    row_min = yields.min(axis=1)
    needs_shift = row_min <= 0
    n_shifted = needs_shift.sum()

    if n_shifted > 0:
        shifted.loc[needs_shift] = (
            yields.loc[needs_shift].sub(row_min[needs_shift], axis=0) + EPSILON
        )
        print(f"  Shift correction applied on {n_shifted} dates")
    else:
        print(f"  No shift correction needed")

    row_sums = shifted.sum(axis=1)
    probs = shifted.div(row_sums, axis=0)
    return probs


def normalise_method_b(yields: pd.DataFrame) -> pd.DataFrame:
    """Yield gradient normalisation: p_i = |y_{i+1} - y_i| / sum(|diffs|)."""
    abs_diffs = yields.diff(axis=1).iloc[:, 1:].abs()
    abs_diffs.columns = [
        "3M-6M", "6M-1Y", "1Y-2Y", "2Y-3Y", "3Y-5Y", "5Y-7Y", "7Y-10Y"
    ]

    row_sums = abs_diffs.sum(axis=1)
    zero_rows = (row_sums == 0).sum()
    if zero_rows > 0:
        print(f"  WARNING: {zero_rows} dates with zero gradient sum — set to uniform")
        row_sums = row_sums.replace(0, 1)
        abs_diffs.loc[row_sums == 0] = 1.0 / 7.0

    probs = abs_diffs.div(row_sums, axis=0)
    print(f"  {len(probs)} days, {probs.shape[1]} bins")
    return probs


# ── Entropy computation ────────────────────────────────

def compute_shannon(probs: pd.DataFrame) -> pd.Series:
    """H = -sum(p_i * ln(p_i)), with 0*ln(0) = 0."""
    with np.errstate(divide="ignore"):
        log_probs = np.where(probs > 0, np.log(probs.values), 0.0)
    H = -(probs.values * log_probs).sum(axis=1)
    return pd.Series(H, index=probs.index)


def compute_tsallis(probs: pd.DataFrame, q: float) -> pd.Series:
    """S_q = (1/(q-1)) * (1 - sum(p_i^q))."""
    S_q = (1.0 / (q - 1.0)) * (1.0 - (probs.values ** q).sum(axis=1))
    return pd.Series(S_q, index=probs.index)


def compute_all_entropy(probs_a: pd.DataFrame,
                        probs_b: pd.DataFrame) -> dict[str, pd.Series]:
    results = {}

    results["shannon_A"] = compute_shannon(probs_a)
    results["shannon_B"] = compute_shannon(probs_b)

    for q in TSALLIS_Q_VALUES:
        results[f"tsallis_A_q{q}"] = compute_tsallis(probs_a, q)
        results[f"tsallis_B_q{q}"] = compute_tsallis(probs_b, q)

    print(f"  Computed {len(results)} entropy series")
    return results


# ── Quarterly ──────────────────────────────────────────

def build_quarterly_entropy(daily_entropy: dict[str, pd.Series]) -> pd.DataFrame:
    """Last business day of each quarter — never within-quarter average."""
    # Align to quarterly panel date range
    qp = pd.read_csv(DATA_PROCESSED / "quarterly_panel.csv",
                      parse_dates=["date"], index_col="date")

    quarterly = {}
    for name, series in daily_entropy.items():
        quarterly[name] = series.resample("QE").last()

    df = pd.DataFrame(quarterly)
    df = df.reindex(qp.index)
    df.index.name = "date"
    return df


# ── Verification ───────────────────────────────────────

def verify_bounds(daily_entropy: dict[str, pd.Series]) -> bool:
    bounds = {
        "shannon_A": np.log(8),
        "shannon_B": np.log(7),
    }
    all_ok = True
    tol = 1e-10

    for name, series in daily_entropy.items():
        lo = series.min()
        if lo < -tol:
            print(f"  FAIL: {name} has negative values (min={lo:.6f})")
            all_ok = False

        if name in bounds:
            hi = series.max()
            ub = bounds[name]
            if hi > ub + tol:
                print(f"  FAIL: {name} exceeds upper bound (max={hi:.6f}, bound={ub:.6f})")
                all_ok = False
            else:
                print(f"  OK: {name} in [0, {ub:.4f}], actual [{lo:.4f}, {hi:.4f}]")

    if all_ok:
        print(f"  All bounds verified.")
    return all_ok


# ── Summary ────────────────────────────────────────────

def print_summary(daily_entropy: dict[str, pd.Series],
                  quarterly_entropy: pd.DataFrame):
    print(f"\n--- Daily Entropy Series ---")
    for name, series in daily_entropy.items():
        print(f"  {name}: {len(series):,} obs, "
              f"min={series.min():.4f}, max={series.max():.4f}, mean={series.mean():.4f}")

    print(f"\n--- Quarterly Entropy ---")
    print(f"  Date range: {quarterly_entropy.index.min().date()} to {quarterly_entropy.index.max().date()}")
    print(f"  Observations: {len(quarterly_entropy)}")
    print(f"  Columns: {list(quarterly_entropy.columns)}")

    nan_counts = quarterly_entropy.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if len(cols_with_nan) > 0:
        print(f"  NaN counts:")
        for col, count in cols_with_nan.items():
            print(f"    {col}: {count}")
    else:
        print(f"  NaN counts: none")


# ── Save ───────────────────────────────────────────────

def save_outputs(daily_entropy: dict[str, pd.Series],
                 quarterly_entropy: pd.DataFrame):
    DATA_ENTROPY.mkdir(parents=True, exist_ok=True)

    for name, series in daily_entropy.items():
        series.name = name
        series.index.name = "date"
        series.to_csv(DATA_ENTROPY / f"daily_{name}.csv")

    quarterly_entropy.to_csv(DATA_ENTROPY / "quarterly_entropy.csv")


# ── Main ───────────────────────────────────────────────

def main():
    print("=" * 60)
    print("LAYER 3: Entropy Computation")
    print("=" * 60)

    daily_yields = load_daily_yields()

    print("\n--- Method A: Absolute Yield Normalisation ---")
    probs_a = normalise_method_a(daily_yields)

    print("\n--- Method B: Yield Gradient Normalisation ---")
    probs_b = normalise_method_b(daily_yields)

    print("\n--- Computing entropy measures ---")
    daily_entropy = compute_all_entropy(probs_a, probs_b)

    print("\n--- Verification ---")
    verify_bounds(daily_entropy)

    print("\n--- Quarterly entropy (last business day of quarter) ---")
    quarterly_entropy = build_quarterly_entropy(daily_entropy)

    save_outputs(daily_entropy, quarterly_entropy)
    print_summary(daily_entropy, quarterly_entropy)

    print(f"\n{'=' * 60}")
    print(f"Layer 3 complete. Output saved to {DATA_ENTROPY}/")
    for name in daily_entropy:
        print(f"  daily_{name}.csv")
    print(f"  quarterly_entropy.csv")


if __name__ == "__main__":
    main()

"""
layer4_derived_features.py -- construct derived entropy features,
dependent variables, and the master quarterly panel.

Usage:
    python thesis_pipeline/layer4_derived_features.py
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from thesis_pipeline.config import (
    DATA_ENTROPY, DATA_FEATURES, DATA_PROCESSED, EPSILON,
    TABLES_DIR, TSALLIS_Q_VALUES,
)


# ── Helpers: subset entropy (mirrors Layer 3 logic) ───

def _normalise_subset_a(yields_subset: pd.DataFrame) -> pd.DataFrame:
    shifted = yields_subset.copy()
    row_min = yields_subset.min(axis=1)
    needs_shift = row_min <= 0
    if needs_shift.any():
        shifted.loc[needs_shift] = (
            yields_subset.loc[needs_shift].sub(row_min[needs_shift], axis=0) + EPSILON
        )
    row_sums = shifted.sum(axis=1)
    return shifted.div(row_sums, axis=0)


def _normalise_subset_b(yields_subset: pd.DataFrame) -> pd.DataFrame:
    abs_diffs = yields_subset.diff(axis=1).iloc[:, 1:].abs()
    row_sums = abs_diffs.sum(axis=1)
    n_bins = abs_diffs.shape[1]
    zero_mask = row_sums == 0
    if zero_mask.any():
        abs_diffs.loc[zero_mask] = 1.0 / n_bins
        row_sums = row_sums.replace(0, 1)
    return abs_diffs.div(row_sums, axis=0)


def _shannon(probs: pd.DataFrame) -> pd.Series:
    with np.errstate(divide="ignore"):
        log_probs = np.where(probs > 0, np.log(probs.values), 0.0)
    H = -(probs.values * log_probs).sum(axis=1)
    return pd.Series(H, index=probs.index)


# ── Loading ────────────────────────────────────────────

def load_quarterly_panel() -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / "quarterly_panel.csv",
                       parse_dates=["date"], index_col="date")


def load_quarterly_entropy() -> pd.DataFrame:
    return pd.read_csv(DATA_ENTROPY / "quarterly_entropy.csv",
                       parse_dates=["date"], index_col="date")


def load_daily_shannon(method: str) -> pd.Series:
    df = pd.read_csv(DATA_ENTROPY / f"daily_shannon_{method}.csv",
                     parse_dates=["date"], index_col="date")
    return df.iloc[:, 0]


def load_daily_yields() -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / "daily_yields.csv",
                       parse_dates=["date"], index_col="date")


def load_daily_term_spread() -> pd.Series:
    df = pd.read_csv(DATA_PROCESSED / "daily_term_spread.csv",
                     parse_dates=["date"], index_col="date")
    return df.iloc[:, 0]


# ── Derived features ──────────────────────────────────

def compute_entropy_momentum(quarterly_shannon: pd.Series) -> pd.Series:
    return quarterly_shannon.diff(1)


def compute_entropy_acceleration(quarterly_shannon: pd.Series) -> pd.Series:
    return quarterly_shannon.diff(1).diff(1)


def compute_entropy_zscore(quarterly_shannon: pd.Series) -> pd.Series:
    shifted = quarterly_shannon.shift(1)
    roll_mean = shifted.rolling(window=20, min_periods=20).mean()
    roll_std = shifted.rolling(window=20, min_periods=20).std()
    return (quarterly_shannon - roll_mean) / roll_std


def compute_entropy_volatility(daily_shannon: pd.Series,
                               quarterly_index: pd.DatetimeIndex) -> pd.Series:
    vol = daily_shannon.resample("QE").std()
    return vol.reindex(quarterly_index)


def compute_subset_entropy(daily_yields: pd.DataFrame,
                           columns: list[str],
                           method: str,
                           quarterly_index: pd.DatetimeIndex) -> pd.Series:
    subset = daily_yields[columns]
    if method == "A":
        probs = _normalise_subset_a(subset)
    else:
        probs = _normalise_subset_b(subset)
    daily_H = _shannon(probs)
    quarterly_H = daily_H.resample("QE").last()
    return quarterly_H.reindex(quarterly_index)


SHORT_END_COLS = ["DGS3MO", "DGS6MO", "DGS1", "DGS2"]
LONG_END_COLS = ["DGS5", "DGS7", "DGS10"]


def compute_all_derived(quarterly_entropy: pd.DataFrame,
                        daily_yields: pd.DataFrame,
                        daily_shannon_a: pd.Series,
                        daily_shannon_b: pd.Series,
                        quarterly_index: pd.DatetimeIndex) -> pd.DataFrame:
    features = {}

    for method in ["A", "B"]:
        shannon = quarterly_entropy[f"shannon_{method}"]
        daily_sh = daily_shannon_a if method == "A" else daily_shannon_b

        features[f"entropy_momentum_{method}"] = compute_entropy_momentum(shannon)
        features[f"entropy_accel_{method}"] = compute_entropy_acceleration(shannon)
        features[f"entropy_zscore_{method}"] = compute_entropy_zscore(shannon)
        features[f"entropy_vol_{method}"] = compute_entropy_volatility(daily_sh, quarterly_index)

        features[f"entropy_short_{method}"] = compute_subset_entropy(
            daily_yields, SHORT_END_COLS, method, quarterly_index)
        features[f"entropy_long_{method}"] = compute_subset_entropy(
            daily_yields, LONG_END_COLS, method, quarterly_index)
        features[f"entropy_gap_{method}"] = (
            features[f"entropy_short_{method}"] - features[f"entropy_long_{method}"]
        )

    return pd.DataFrame(features, index=quarterly_index)


# ── Dependent variables ────────────────────────────────

def compute_gdp_cumulative_averages(gdp: pd.Series) -> pd.DataFrame:
    result = {}
    for h in [1, 2, 4]:
        result[f"GDP_growth_h{h}"] = sum(gdp.shift(-j) for j in range(1, h + 1)) / h
    return pd.DataFrame(result, index=gdp.index)


def compute_below_trend(gdp: pd.Series) -> pd.Series:
    expanding_mean = gdp.shift(1).expanding(min_periods=1).mean()
    return (gdp < expanding_mean).astype(int)


# ── Assembly ───────────────────────────────────────────

def build_quarterly_full_panel() -> pd.DataFrame:
    qp = load_quarterly_panel()
    qe = load_quarterly_entropy()
    daily_yields = load_daily_yields()
    daily_sh_a = load_daily_shannon("A")
    daily_sh_b = load_daily_shannon("B")

    qi = qp.index

    # Derived features
    derived = compute_all_derived(qe, daily_yields, daily_sh_a, daily_sh_b, qi)

    # Dependent variables
    gdp_targets = compute_gdp_cumulative_averages(qp["GDP_growth"])
    below_trend = compute_below_trend(qp["GDP_growth"])

    # Merge
    panel = qp.copy()
    panel = panel.join(gdp_targets)
    panel["below_trend"] = below_trend
    panel = panel.join(qe)
    panel = panel.join(derived)

    return panel


def build_daily_midas_file() -> pd.DataFrame:
    daily_sh_a = load_daily_shannon("A")
    daily_sh_b = load_daily_shannon("B")
    daily_ts = load_daily_term_spread()

    midas = pd.DataFrame({
        "shannon_A": daily_sh_a,
        "shannon_B": daily_sh_b,
        "term_spread_10Y_2Y": daily_ts,
    })
    midas.index.name = "date"
    return midas


# ── Descriptive statistics ─────────────────────────────

def build_descriptive_statistics(panel: pd.DataFrame) -> pd.DataFrame:
    variables = [
        ("GDP growth", "GDP_growth"),
        ("Term spread (10Y-2Y)", "term_spread_10Y_2Y"),
        ("Federal funds rate", "DFF"),
        ("Credit spread (Baa-Aaa)", "credit_spread_Baa_Aaa"),
        ("S&P 500 return", "SP500_return"),
        ("Initial claims (4-wk MA)", "IC4WSA"),
        ("ISM Manufacturing PMI", "NAPM"),
        ("Lagged GDP growth", "GDP_growth_lag1"),
        ("Shannon entropy (A)", "shannon_A"),
        ("Shannon entropy (B)", "shannon_B"),
        ("Tsallis q=0.5 (A)", "tsallis_A_q0.5"),
        ("Tsallis q=0.5 (B)", "tsallis_B_q0.5"),
        ("Tsallis q=1.5 (A)", "tsallis_A_q1.5"),
        ("Tsallis q=1.5 (B)", "tsallis_B_q1.5"),
        ("Tsallis q=2.0 (A)", "tsallis_A_q2.0"),
        ("Tsallis q=2.0 (B)", "tsallis_B_q2.0"),
        ("Entropy momentum (A)", "entropy_momentum_A"),
        ("Entropy momentum (B)", "entropy_momentum_B"),
        ("Entropy acceleration (A)", "entropy_accel_A"),
        ("Entropy acceleration (B)", "entropy_accel_B"),
        ("Entropy z-score (A)", "entropy_zscore_A"),
        ("Entropy z-score (B)", "entropy_zscore_B"),
        ("Entropy volatility (A)", "entropy_vol_A"),
        ("Entropy volatility (B)", "entropy_vol_B"),
        ("Short-end entropy (A)", "entropy_short_A"),
        ("Short-end entropy (B)", "entropy_short_B"),
        ("Long-end entropy (A)", "entropy_long_A"),
        ("Long-end entropy (B)", "entropy_long_B"),
        ("Entropy gap (A)", "entropy_gap_A"),
        ("Entropy gap (B)", "entropy_gap_B"),
    ]

    rows = []
    for label, col in variables:
        s = panel[col]
        first_valid = s.first_valid_index()
        first_q = (f"{first_valid.year}Q{(first_valid.month - 1) // 3 + 1}"
                   if first_valid is not None else "N/A")
        rows.append({
            "Variable": label,
            "N": int(s.count()),
            "Mean": s.mean(),
            "Std": s.std(),
            "Min": s.min(),
            "25%": s.quantile(0.25),
            "Median": s.median(),
            "75%": s.quantile(0.75),
            "Max": s.max(),
            "First quarter": first_q,
        })

    return pd.DataFrame(rows).set_index("Variable")


def save_descriptive_statistics(desc: pd.DataFrame):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    desc.to_csv(TABLES_DIR / "descriptive_statistics.csv")

    numeric_cols = [c for c in desc.columns if c != "First quarter"]
    formatters = {c: "{:.3f}".format for c in numeric_cols}
    formatters["N"] = "{:.0f}".format

    latex = desc.to_latex(
        formatters=formatters,
        caption="Descriptive statistics for the quarterly panel.",
        label="tab:descriptive_statistics",
        position="htbp",
    )
    with open(TABLES_DIR / "descriptive_statistics.tex", "w") as f:
        f.write(latex)


# ── Verification ───────────────────────────────────────

def verify_and_print_summary(panel: pd.DataFrame, midas: pd.DataFrame):
    print(f"\n--- Quarterly Full Panel ---")
    print(f"  Date range: {panel.index.min().date()} to {panel.index.max().date()}")
    print(f"  Observations: {len(panel)}")
    print(f"  Columns: {len(panel.columns)}")

    nan_counts = panel.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if len(cols_with_nan) > 0:
        print(f"  NaN counts:")
        for col, count in cols_with_nan.items():
            print(f"    {col}: {count}")

    # Verify subset entropy bounds
    bounds = {
        "entropy_short_A": np.log(4),
        "entropy_short_B": np.log(3),
        "entropy_long_A": np.log(3),
        "entropy_long_B": np.log(2),
    }
    print(f"\n--- Subset entropy bounds ---")
    tol = 1e-10
    for col, ub in bounds.items():
        hi = panel[col].max()
        lo = panel[col].min()
        ok = hi <= ub + tol and lo >= -tol
        status = "OK" if ok else "FAIL"
        print(f"  {status}: {col} in [0, {ub:.4f}], actual [{lo:.4f}, {hi:.4f}]")

    # Verify below-trend is binary
    bt_vals = panel["below_trend"].dropna().unique()
    print(f"\n--- Below-trend indicator ---")
    print(f"  Unique values: {sorted(bt_vals)}")
    print(f"  Below-trend quarters: {int(panel['below_trend'].sum())} / {int(panel['below_trend'].count())}")

    # Verify first-valid dates
    print(f"\n--- Feature first-valid dates ---")
    for col in ["entropy_momentum_A", "entropy_accel_A", "entropy_zscore_A"]:
        fv = panel[col].first_valid_index()
        print(f"  {col}: {fv.date() if fv else 'N/A'}")

    print(f"\n--- Daily MIDAS File ---")
    print(f"  Observations: {len(midas):,}")
    print(f"  Columns: {list(midas.columns)}")


# ── Main ───────────────────────────────────────────────

def main():
    print("=" * 60)
    print("LAYER 4: Derived Feature Construction")
    print("=" * 60)

    print("\n--- Building quarterly full panel ---")
    panel = build_quarterly_full_panel()

    print("\n--- Building daily MIDAS file ---")
    midas = build_daily_midas_file()

    # Save outputs
    DATA_FEATURES.mkdir(parents=True, exist_ok=True)
    panel.to_csv(DATA_FEATURES / "quarterly_full_panel.csv")
    midas.to_csv(DATA_FEATURES / "daily_entropy_for_midas.csv")

    # Descriptive statistics
    print("\n--- Descriptive statistics ---")
    desc = build_descriptive_statistics(panel)
    save_descriptive_statistics(desc)
    print(desc.to_string())

    # Verification
    verify_and_print_summary(panel, midas)

    print(f"\n{'=' * 60}")
    print(f"Layer 4 complete. Output saved to:")
    print(f"  {DATA_FEATURES / 'quarterly_full_panel.csv'}")
    print(f"  {DATA_FEATURES / 'daily_entropy_for_midas.csv'}")
    print(f"  {TABLES_DIR / 'descriptive_statistics.csv'}")
    print(f"  {TABLES_DIR / 'descriptive_statistics.tex'}")


if __name__ == "__main__":
    main()

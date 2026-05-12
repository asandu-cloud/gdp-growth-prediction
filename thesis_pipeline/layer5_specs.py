"""
layer5_specs.py -- shared constants, specification definitions,
and expanding-window generator for Layer 5 models.
"""
import numpy as np
import pandas as pd

# ── Train / OOS split ──────────────────────────────────
TRAIN_END = pd.Timestamp("1999-12-31")
OOS_START = pd.Timestamp("2000-03-31")
COVID_QUARTER = pd.Timestamp("2020-06-30")

# ── Forecast horizons ──────────────────────────────────
HORIZONS = [1, 2, 4]

# ── Quantile regression ────────────────────────────────
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
FOCAL_QUANTILE = 0.10

# ── Random forest ──────────────────────────────────────
RF_N_TREES = 500
RF_MIN_SAMPLES_LEAF = 5
RF_REESTIMATE_EVERY = 4

# ── MIDAS ──────────────────────────────────────────────
MIDAS_MAX_LAG = 126
MIDAS_CUTOFFS = [15, 30, 45, 60]
MIDAS_N_STARTS = 10

# ── Specification column lists ─────────────────────────

BENCHMARK_COLS = [
    "GDP_growth_lag1",
    "term_spread_10Y_2Y",
    "DFF",
    "credit_spread_Baa_Aaa",
    "SP500_return",
    "IC4WSA",
    "NAPM",
]

DERIVED_COLS = {
    "A": [
        "entropy_momentum_A", "entropy_accel_A", "entropy_zscore_A",
        "entropy_vol_A", "entropy_short_A", "entropy_long_A", "entropy_gap_A",
    ],
    "B": [
        "entropy_momentum_B", "entropy_accel_B", "entropy_zscore_B",
        "entropy_vol_B", "entropy_short_B", "entropy_long_B", "entropy_gap_B",
    ],
}


def get_specs(method: str) -> dict[str, list[str]]:
    """
    Return {spec_name: column_list} for a given method ('A' or 'B').

    Specs:
      spec1         — 7 benchmarks (identical for A and B)
      spec2         — benchmarks + Shannon entropy
      spec3         — benchmarks + Shannon entropy + 7 derived features
      spec4_q{q}    — benchmarks + Tsallis entropy at q
    """
    m = method
    specs = {
        "spec1": BENCHMARK_COLS.copy(),
        "spec2": BENCHMARK_COLS + [f"shannon_{m}"],
        "spec3": BENCHMARK_COLS + [f"shannon_{m}"] + DERIVED_COLS[m],
    }
    for q in [0.5, 1.5, 2.0]:
        specs[f"spec4_q{q}"] = BENCHMARK_COLS + [f"tsallis_{m}_q{q}"]
    return specs


# ── Expanding-window generator ─────────────────────────

def expanding_window_quarters(panel: pd.DataFrame, reestimate_every: int = 1):
    """
    Yield (train_end_idx, oos_idx, oos_count, refit) for each OOS quarter.

    train_end_idx:  integer index of the last training row
    oos_idx:        integer index of the OOS row to forecast
    oos_count:      0-based counter in the OOS period
    refit:          True if the model should be re-estimated this quarter
    """
    oos_mask = panel.index > TRAIN_END
    oos_positions = np.where(oos_mask)[0]

    for count, oos_idx in enumerate(oos_positions):
        train_end_idx = oos_idx - 1
        refit = (count % reestimate_every == 0)
        yield train_end_idx, oos_idx, count, refit

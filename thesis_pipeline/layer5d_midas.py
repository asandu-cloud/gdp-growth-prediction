"""
layer5d_midas.py -- MIDAS nowcasting with Beta polynomial weighting.

Usage:
    python thesis_pipeline/layer5d_midas.py
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from thesis_pipeline.config import DATA_FEATURES, DATA_RESULTS
from thesis_pipeline.layer5_specs import (
    BENCHMARK_COLS, MIDAS_MAX_LAG, MIDAS_CUTOFFS,
    MIDAS_N_STARTS, TRAIN_END, COVID_QUARTER,
)

K = MIDAS_MAX_LAG
VALIDATION_ONLY = False

HF_VARS = ["shannon_A", "shannon_B", "term_spread_10Y_2Y"]


def beta_weights(K: int, theta1: float, theta2: float) -> np.ndarray:
    """Normalized Beta polynomial weights. Index 0 = most recent lag."""
    k = np.arange(1, K + 1)
    s = k / (K + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = s ** (theta1 - 1) * (1 - s) ** (theta2 - 1)
    raw = np.nan_to_num(raw, nan=0.0, posinf=1e10, neginf=0.0)
    total = raw.sum()
    if total <= 0 or not np.isfinite(total):
        return np.ones(K) / K
    return raw / total


def get_quarter_start(qdate: pd.Timestamp) -> pd.Timestamp:
    month = ((qdate.month - 1) // 3) * 3 + 1
    return pd.Timestamp(year=qdate.year, month=month, day=1)


def build_lag_matrix(daily_hf: pd.Series, quarterly_dates: pd.DatetimeIndex,
                     cutoff: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (n_quarters, K) matrix of daily HF lags for each quarter.
    Column 0 = most recent (day d), column K-1 = oldest (day d-K+1).
    """
    daily_vals = daily_hf.values
    daily_idx = daily_hf.index
    n = len(quarterly_dates)
    lag_matrix = np.full((n, K), np.nan)
    valid = np.ones(n, dtype=bool)

    for i, qdate in enumerate(quarterly_dates):
        q_start = get_quarter_start(qdate)
        q_positions = np.where((daily_idx >= q_start) & (daily_idx <= qdate))[0]

        if len(q_positions) < cutoff:
            valid[i] = False
            continue

        end_pos = q_positions[cutoff - 1]
        start_pos = end_pos - K + 1

        if start_pos < 0:
            valid[i] = False
            continue

        window = daily_vals[start_pos:end_pos + 1]
        if len(window) != K or np.any(np.isnan(window)):
            valid[i] = False
            continue

        lag_matrix[i, :] = window[::-1]

    return lag_matrix, valid


def concentrated_residuals(theta, lag_matrix, y, Z):
    """Residuals from concentrated NLS: OLS given Beta weights from theta."""
    theta1, theta2 = theta
    w = beta_weights(lag_matrix.shape[1], theta1, theta2)
    x_midas = lag_matrix @ w
    X = np.column_stack([np.ones(len(y)), x_midas, Z])
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta_hat


def estimate_midas(lag_matrix, y, Z, n_starts=10, seed=42):
    """Estimate MIDAS via concentrated NLS with multiple random starts."""
    rng = np.random.default_rng(seed)

    best_result = None
    best_rss = np.inf
    all_rss = []

    starts = rng.uniform(0.5, 10.0, size=(n_starts, 2))
    starts[0] = [1.0, 5.0]
    starts[1] = [1.0, 1.0]
    if n_starts > 2:
        starts[2] = [2.0, 2.0]

    for s in starts:
        try:
            result = least_squares(
                concentrated_residuals, x0=s,
                args=(lag_matrix, y, Z),
                bounds=([0.01, 0.01], [200.0, 200.0]),
                method="trf",
                max_nfev=500,
            )
            rss = (result.fun ** 2).sum()
            all_rss.append(rss)
            if rss < best_rss:
                best_rss = rss
                best_result = result
        except Exception:
            all_rss.append(np.inf)

    if best_result is None:
        return None

    theta1, theta2 = best_result.x
    w = beta_weights(lag_matrix.shape[1], theta1, theta2)
    x_midas = lag_matrix @ w
    X = np.column_stack([np.ones(len(y)), x_midas, Z])
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]

    return {
        "theta1": theta1, "theta2": theta2,
        "beta_hat": beta_hat, "weights": w,
        "rss": best_rss, "all_rss": all_rss,
    }


def validate_single_quarter(daily_data, panel):
    """Validate MIDAS on 2000Q1 at cutoff=30, Shannon A."""
    print("\n" + "=" * 60)
    print("VALIDATION: 2000Q1, cutoff d=30, Shannon A")
    print("=" * 60)

    target_col = "GDP_growth"
    hf_series = daily_data["shannon_A"]

    lag_matrix, valid = build_lag_matrix(hf_series, panel.index, 30, K)

    Z_all = panel[BENCHMARK_COLS].shift(1)
    z_valid = ~Z_all.isna().any(axis=1).values
    y_valid = panel[target_col].notna().values

    train_mask = (panel.index <= TRAIN_END) & valid & z_valid & y_valid
    train_idx = np.where(train_mask)[0]

    y_train = panel[target_col].values[train_idx]
    Z_train = Z_all.values[train_idx]
    lag_train = lag_matrix[train_idx]

    print(f"\n  Training quarters: {len(train_idx)}")
    print(f"  Period: {panel.index[train_idx[0]].strftime('%Y-%m')} – "
          f"{panel.index[train_idx[-1]].strftime('%Y-%m')}")

    result = estimate_midas(lag_train, y_train, Z_train, n_starts=MIDAS_N_STARTS)

    if result is None:
        print("  FAILED: no valid estimation")
        return None

    print(f"\n  θ₁ = {result['theta1']:.4f}")
    print(f"  θ₂ = {result['theta2']:.4f}")
    print(f"  Both positive: "
          f"{'YES' if result['theta1'] > 0 and result['theta2'] > 0 else 'NO'}")

    w = result["weights"]
    print(f"\n  Weight vector w(k):")
    print(f"    Sum             = {w.sum():.6f}")
    print(f"    w[0]   (recent) = {w[0]:.6f}")
    print(f"    w[31]  (~1 mo)  = {w[31]:.6f}")
    print(f"    w[62]  (~3 mo)  = {w[62]:.6f}")
    print(f"    w[93]  (~5 mo)  = {w[93]:.6f}")
    print(f"    w[125] (oldest) = {w[125]:.6f}")

    peak = np.argmax(w)
    if peak < K // 4:
        shape = "DECAYING (recent lags dominate)"
    elif peak > 3 * K // 4:
        shape = "REVERSE (old lags dominate)"
    else:
        shape = f"HUMPED (peak at lag {peak})"
    print(f"    Shape: {shape}")

    valid_rss = [r for r in result["all_rss"] if np.isfinite(r)]
    print(f"\n  RSS across {len(valid_rss)} valid starts:")
    print(f"    Best  = {min(valid_rss):.2f}")
    print(f"    Worst = {max(valid_rss):.2f}")
    print(f"    Spread = {max(valid_rss) - min(valid_rss):.4f}")

    b = result["beta_hat"]
    print(f"\n  Coefficients:")
    print(f"    α (intercept)                = {b[0]:8.4f}")
    print(f"    β (daily Shannon A)          = {b[1]:8.4f}")
    for j, col in enumerate(BENCHMARK_COLS):
        print(f"    γ_{col:<28s} = {b[2 + j]:8.4f}")

    oos_idx = np.where(panel.index > TRAIN_END)[0][0]
    if valid[oos_idx] and z_valid[oos_idx]:
        x_midas_oos = lag_matrix[oos_idx] @ w
        Z_oos = Z_all.values[oos_idx]
        X_oos = np.concatenate([[1.0], [x_midas_oos], Z_oos])
        nowcast = X_oos @ b
        actual = panel[target_col].values[oos_idx]
        print(f"\n  Nowcast for 2000Q1:")
        print(f"    Predicted = {nowcast:.4f}")
        print(f"    Actual    = {actual:.4f}")
        print(f"    Error     = {nowcast - actual:.4f}")
        plausible = -10 < nowcast < 15
        print(f"    Plausible (-10 to 15)? {'YES' if plausible else 'NO'}")

    return result


# ── Full expanding-window estimation ─────────────────────


def run_midas_expanding(daily_data, panel, hf_col, cutoff, n_starts=10):
    """Expanding-window MIDAS for one HF variable at one cutoff."""
    target_col = "GDP_growth"
    hf_series = daily_data[hf_col]

    lag_matrix, valid = build_lag_matrix(hf_series, panel.index, cutoff, K)
    Z_all = panel[BENCHMARK_COLS].shift(1)
    z_valid = ~Z_all.isna().any(axis=1).values
    y_valid = panel[target_col].notna().values

    oos_positions = np.where(panel.index > TRAIN_END)[0]

    oos_dates, oos_preds, oos_actuals = [], [], []
    param_records = []

    for oos_idx in oos_positions:
        train_idx = np.where(
            (np.arange(len(panel)) < oos_idx) & valid & z_valid & y_valid
        )[0]

        oos_date = panel.index[oos_idx]
        actual = panel[target_col].values[oos_idx]

        if len(train_idx) < 10 or not valid[oos_idx] or not z_valid[oos_idx]:
            oos_dates.append(oos_date)
            oos_preds.append(np.nan)
            oos_actuals.append(actual)
            continue

        y_train = panel[target_col].values[train_idx]
        Z_train = Z_all.values[train_idx]
        lag_train = lag_matrix[train_idx]

        result = estimate_midas(lag_train, y_train, Z_train, n_starts=n_starts)

        if result is None:
            oos_dates.append(oos_date)
            oos_preds.append(np.nan)
            oos_actuals.append(actual)
            continue

        w = result["weights"]
        x_midas_oos = lag_matrix[oos_idx] @ w
        Z_oos = Z_all.values[oos_idx]
        X_oos = np.concatenate([[1.0], [x_midas_oos], Z_oos])
        nowcast = X_oos @ result["beta_hat"]

        oos_dates.append(oos_date)
        oos_preds.append(nowcast)
        oos_actuals.append(actual)

        weight_peak = int(np.argmax(result["weights"]))
        param_records.append({
            "date": oos_date,
            "hf_var": hf_col, "cutoff": cutoff,
            "theta1": result["theta1"], "theta2": result["theta2"],
            "beta_hf": result["beta_hat"][1], "rss": result["rss"],
            "weight_peak_lag": weight_peak,
        })

    return {
        "oos_preds": pd.Series(oos_preds, index=oos_dates, name="prediction"),
        "actuals": pd.Series(oos_actuals, index=oos_dates, name="actual"),
        "params": pd.DataFrame(param_records),
    }


def compute_midas_metrics(results: dict) -> pd.DataFrame:
    rows = []
    for (hf_var, cutoff), res in results.items():
        p = res["oos_preds"]
        a = res["actuals"]
        vmask = p.notna() & a.notna()
        pv, av = p[vmask], a[vmask]
        if len(pv) == 0:
            continue

        rmse = np.sqrt(((av - pv) ** 2).mean())
        mae = (av - pv).abs().mean()

        nc = pv.index != COVID_QUARTER
        pnc, anc = pv[nc], av[nc]
        rmse_nc = np.sqrt(((anc - pnc) ** 2).mean())
        mae_nc = (anc - pnc).abs().mean()

        rows.append({
            "hf_var": hf_var, "cutoff": cutoff,
            "n_oos": len(pv),
            "RMSE": rmse, "MAE": mae,
            "RMSE_exCOVID": rmse_nc, "MAE_exCOVID": mae_nc,
        })

    return pd.DataFrame(rows)


def print_theta_diagnostics(results: dict):
    """Print diagnostics on θ₁, θ₂ distributions across OOS quarters."""
    print("\n" + "=" * 60)
    print("THETA DIAGNOSTICS")
    print("=" * 60)

    BOUND = 200.0
    BOUND_TOL = 0.5

    for hf_col in HF_VARS:
        print(f"\n  ── {hf_col} ──")
        for d in MIDAS_CUTOFFS:
            key = (hf_col, d)
            if key not in results:
                continue
            params = results[key]["params"]
            if params.empty:
                print(f"    d={d}: no valid estimations")
                continue

            t1 = params["theta1"]
            t2 = params["theta2"]
            n = len(t1)
            n_bound = (t1 >= BOUND - BOUND_TOL).sum()
            pct_bound = 100 * n_bound / n

            print(f"    d={d:>2}: n={n}")
            print(f"      θ₁  median={t1.median():7.2f}  "
                  f"IQR=[{t1.quantile(0.25):6.2f}, {t1.quantile(0.75):6.2f}]  "
                  f"hit bound: {n_bound}/{n} ({pct_bound:.0f}%)"
                  f"{'  ⚠ FLAG' if pct_bound > 10 else ''}")
            print(f"      θ₂  median={t2.median():7.2f}  "
                  f"IQR=[{t2.quantile(0.25):6.2f}, {t2.quantile(0.75):6.2f}]")

            peak = params["weight_peak_lag"]
            print(f"      Peak lag  median={peak.median():5.0f}  "
                  f"IQR=[{peak.quantile(0.25):4.0f}, {peak.quantile(0.75):4.0f}]")

    # Weight profile evolution over time
    print(f"\n  ── Weight profile evolution (Shannon A, d=30) ──")
    key = ("shannon_A", 30)
    if key in results and not results[key]["params"].empty:
        params = results[key]["params"]
        n = len(params)
        thirds = [params.iloc[:n // 3], params.iloc[n // 3:2 * n // 3],
                  params.iloc[2 * n // 3:]]
        labels = ["Early OOS (2000–2008)", "Mid OOS (2008–2016)",
                  "Late OOS (2016–2025)"]
        for label, chunk in zip(labels, thirds):
            if chunk.empty:
                continue
            print(f"    {label}:")
            print(f"      θ₁ median={chunk['theta1'].median():.2f}  "
                  f"peak lag median={chunk['weight_peak_lag'].median():.0f}")


def main():
    print("=" * 60)
    print("LAYER 5D: MIDAS Nowcasting")
    print("=" * 60)

    panel = pd.read_csv(
        DATA_FEATURES / "quarterly_full_panel.csv",
        parse_dates=["date"], index_col="date",
    )
    daily_data = pd.read_csv(
        DATA_FEATURES / "daily_entropy_for_midas.csv",
        parse_dates=["date"], index_col="date",
    )

    print(f"\n  Panel: {len(panel)} quarters")
    print(f"  Daily: {len(daily_data)} days, columns: {list(daily_data.columns)}")

    # ── Step 1: Single-quarter validation ──
    validate_single_quarter(daily_data, panel)

    if VALIDATION_ONLY:
        print("\n" + "=" * 60)
        print("VALIDATION COMPLETE — inspect results above.")
        print("Set VALIDATION_ONLY = False and re-run for full estimation.")
        print("=" * 60)
        return

    # ── Step 2: Full expanding-window estimation ──
    print("\n" + "=" * 60)
    print("FULL EXPANDING-WINDOW MIDAS ESTIMATION")
    print("=" * 60)

    n_starts = MIDAS_N_STARTS
    results = {}
    first_run = True

    for hf_col in HF_VARS:
        for d in MIDAS_CUTOFFS:
            key = (hf_col, d)
            print(f"\n  MIDAS {hf_col} d={d}...", end="", flush=True)

            t0 = time.time()
            res = run_midas_expanding(
                daily_data, panel, hf_col, d, n_starts=n_starts,
            )
            elapsed = time.time() - t0
            n_valid = res["oos_preds"].notna().sum()
            print(f" {elapsed:.1f}s, {n_valid} valid nowcasts")

            results[key] = res

            if first_run:
                n_total_runs = len(HF_VARS) * len(MIDAS_CUTOFFS)
                est_total = elapsed * n_total_runs
                print(f"\n    Extrapolated total: ~{est_total / 60:.1f} min "
                      f"({n_total_runs} combinations)")
                if elapsed > 60:
                    print("    WARNING: first run > 60s — check optimisation")
                first_run = False

    # ── Metrics ──
    metrics = compute_midas_metrics(results)

    print(f"\n{'─' * 70}")
    print(f"{'HF var':<22} {'d':<6} {'N':<6} "
          f"{'RMSE':>7} {'MAE':>7} {'RMSE*':>7} {'MAE*':>7}")
    print(f"{'─' * 70}")
    for _, row in metrics.iterrows():
        print(f"{row['hf_var']:<22} {int(row['cutoff']):<6} "
              f"{int(row['n_oos']):<6} "
              f"{row['RMSE']:7.3f} {row['MAE']:7.3f} "
              f"{row['RMSE_exCOVID']:7.3f} {row['MAE_exCOVID']:7.3f}")

    # ── Theta diagnostics ──
    print_theta_diagnostics(results)

    # ── Save ──
    DATA_RESULTS.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(DATA_RESULTS / "midas_metrics.csv", index=False)

    pred_frames = []
    for (hf_var, cutoff), res in results.items():
        df = pd.DataFrame({
            "hf_var": hf_var, "cutoff": cutoff,
            "date": res["oos_preds"].index,
            "prediction": res["oos_preds"].values,
            "actual": res["actuals"].values,
        })
        pred_frames.append(df)
    all_preds = pd.concat(pred_frames, ignore_index=True)
    all_preds.to_csv(DATA_RESULTS / "midas_predictions.csv", index=False)

    param_frames = []
    for (_, _), res in results.items():
        if not res["params"].empty:
            param_frames.append(res["params"])
    if param_frames:
        all_params = pd.concat(param_frames, ignore_index=True)
        all_params.to_csv(DATA_RESULTS / "midas_parameters.csv", index=False)

    print(f"\nSaved to {DATA_RESULTS}/")
    print(f"  midas_metrics.csv")
    print(f"  midas_predictions.csv")
    print(f"  midas_parameters.csv")


if __name__ == "__main__":
    main()

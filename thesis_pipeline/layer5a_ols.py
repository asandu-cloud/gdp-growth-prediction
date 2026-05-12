"""
layer5a_ols.py -- OLS expanding-window estimation with Newey-West HAC.

Usage:
    python thesis_pipeline/layer5a_ols.py
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import statsmodels.api as sm

from thesis_pipeline.config import DATA_FEATURES, DATA_RESULTS
from thesis_pipeline.layer5_specs import (
    HORIZONS, expanding_window_quarters, get_specs,
)


def run_ols_single(panel: pd.DataFrame, feature_cols: list[str],
                   target_col: str, h: int) -> dict:
    """
    Run OLS with expanding window for one spec/horizon combination.

    Returns dict with:
      oos_preds:   Series (index=OOS dates, values=predictions)
      actuals:     Series (index=OOS dates, values=realised targets)
      last_model:  final fitted model (for coefficient reporting)
      coef_last:   params from full-sample estimation
      bse_last:    Newey-West std errors from full-sample
      pval_last:   p-values from full-sample
      r2_last:     R² from full-sample
      train_means: Series of expanding training-set means (for ΔMSE)
    """
    oos_dates = []
    oos_preds = []
    oos_actuals = []
    train_means = []

    for train_end, oos_idx, _, _ in expanding_window_quarters(panel):
        train = panel.iloc[:train_end + 1]
        oos_row = panel.iloc[oos_idx]

        y_train = train[target_col]
        X_train = train[feature_cols]
        valid = pd.concat([y_train, X_train], axis=1).dropna()
        if len(valid) < len(feature_cols) + 2:
            oos_dates.append(panel.index[oos_idx])
            oos_preds.append(np.nan)
            oos_actuals.append(oos_row[target_col])
            train_means.append(np.nan)
            continue

        y_fit = valid[target_col]
        X_fit = sm.add_constant(valid[feature_cols])

        model = sm.OLS(y_fit, X_fit).fit(cov_type="HAC", cov_kwds={"maxlags": h})

        X_oos = oos_row[feature_cols]
        if X_oos.isna().any():
            pred = np.nan
        else:
            X_oos_c = np.concatenate([[1.0], X_oos.values])
            pred = model.predict(X_oos_c.reshape(1, -1))[0]

        oos_dates.append(panel.index[oos_idx])
        oos_preds.append(pred)
        oos_actuals.append(oos_row[target_col])
        train_means.append(y_fit.mean())

    # Full-sample estimation for coefficient reporting
    y_full = panel[target_col]
    X_full = panel[feature_cols]
    valid_full = pd.concat([y_full, X_full], axis=1).dropna()
    y_fs = valid_full[target_col]
    X_fs = sm.add_constant(valid_full[feature_cols])
    full_model = sm.OLS(y_fs, X_fs).fit(cov_type="HAC", cov_kwds={"maxlags": h})

    return {
        "oos_preds": pd.Series(oos_preds, index=oos_dates, name="prediction"),
        "actuals": pd.Series(oos_actuals, index=oos_dates, name="actual"),
        "train_means": pd.Series(train_means, index=oos_dates, name="train_mean"),
        "coef_last": full_model.params,
        "bse_last": full_model.bse,
        "pval_last": full_model.pvalues,
        "r2_last": full_model.rsquared,
    }


def run_ols_all(panel: pd.DataFrame) -> dict:
    """
    Run OLS for all specs × horizons × methods.
    Returns nested dict keyed by (method, spec_name, horizon).
    """
    results = {}
    spec1_cache = {}

    for method in ["A", "B"]:
        specs = get_specs(method)
        for spec_name, feature_cols in specs.items():
            for h in HORIZONS:
                target_col = f"GDP_growth_h{h}"
                key = (method, spec_name, h)

                # Spec 1 is identical for both methods — reuse
                if spec_name == "spec1" and method == "B":
                    results[key] = results[("A", "spec1", h)]
                    continue
                if spec_name == "spec1":
                    print(f"  OLS {spec_name} h={h} (shared)...")
                else:
                    print(f"  OLS {spec_name}_{method} h={h}...")

                res = run_ols_single(panel, feature_cols, target_col, h)
                results[key] = res

                if spec_name == "spec1":
                    spec1_cache[(spec_name, h)] = res

    return results


def compute_ols_metrics(results: dict) -> pd.DataFrame:
    """Compute RMSE, MAE for each OLS result. Returns a summary DataFrame."""
    from thesis_pipeline.layer5_specs import COVID_QUARTER

    rows = []
    for (method, spec_name, h), res in results.items():
        preds = res["oos_preds"]
        actuals = res["actuals"]

        valid = preds.notna() & actuals.notna()
        p, a = preds[valid], actuals[valid]

        if len(p) == 0:
            continue

        rmse = np.sqrt(((a - p) ** 2).mean())
        mae = (a - p).abs().mean()

        # Excluding COVID
        mask = p.index != COVID_QUARTER
        p_nc, a_nc = p[mask], a[mask]
        rmse_nc = np.sqrt(((a_nc - p_nc) ** 2).mean())
        mae_nc = (a_nc - p_nc).abs().mean()

        rows.append({
            "method": method, "spec": spec_name, "h": h,
            "n_oos": len(p),
            "RMSE": rmse, "MAE": mae,
            "RMSE_exCOVID": rmse_nc, "MAE_exCOVID": mae_nc,
        })

    return pd.DataFrame(rows)


def print_ols_summary(metrics: pd.DataFrame):
    """Print OLS results in a readable format."""
    print(f"\n{'─' * 70}")
    print(f"{'Method':<8} {'Spec':<12} {'h':<4} {'N':<6} "
          f"{'RMSE':>7} {'MAE':>7} {'RMSE*':>7} {'MAE*':>7}")
    print(f"{'─' * 70}")
    print(f"{'':>46} (* = excl. 2020Q2)")
    for _, row in metrics.iterrows():
        print(f"{row['method']:<8} {row['spec']:<12} {row['h']:<4} "
              f"{row['n_oos']:<6} "
              f"{row['RMSE']:7.3f} {row['MAE']:7.3f} "
              f"{row['RMSE_exCOVID']:7.3f} {row['MAE_exCOVID']:7.3f}")


def main():
    print("=" * 60)
    print("LAYER 5A: OLS Expanding-Window Estimation")
    print("=" * 60)

    panel = pd.read_csv(
        DATA_FEATURES / "quarterly_full_panel.csv",
        parse_dates=["date"], index_col="date",
    )

    print("\n--- Running OLS regressions ---")
    results = run_ols_all(panel)

    print(f"\n--- OLS complete: {len(results)} spec/horizon combinations ---")

    metrics = compute_ols_metrics(results)
    print_ols_summary(metrics)

    # Save
    DATA_RESULTS.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(DATA_RESULTS / "ols_metrics.csv", index=False)

    # Save OOS predictions for downstream evaluation
    pred_frames = []
    for (method, spec_name, h), res in results.items():
        df = pd.DataFrame({
            "method": method, "spec": spec_name, "h": h,
            "date": res["oos_preds"].index,
            "prediction": res["oos_preds"].values,
            "actual": res["actuals"].values,
            "train_mean": res["train_means"].values,
        })
        pred_frames.append(df)
    all_preds = pd.concat(pred_frames, ignore_index=True)
    all_preds.to_csv(DATA_RESULTS / "ols_predictions.csv", index=False)

    print(f"\nSaved to {DATA_RESULTS}/")
    print(f"  ols_metrics.csv")
    print(f"  ols_predictions.csv")


if __name__ == "__main__":
    main()

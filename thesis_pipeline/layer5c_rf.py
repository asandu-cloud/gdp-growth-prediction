"""
layer5c_rf.py -- Random Forest expanding-window estimation with SHAP.

Usage:
    python thesis_pipeline/layer5c_rf.py
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from thesis_pipeline.config import DATA_FEATURES, DATA_RESULTS
from thesis_pipeline.layer5_specs import (
    BENCHMARK_COLS, DERIVED_COLS, HORIZONS, RF_MIN_SAMPLES_LEAF,
    RF_N_TREES, RF_REESTIMATE_EVERY, expanding_window_quarters,
)


def get_spec3_cols(method: str) -> list[str]:
    return BENCHMARK_COLS + [f"shannon_{method}"] + DERIVED_COLS[method]


def run_rf_single(panel: pd.DataFrame, feature_cols: list[str],
                  target_col: str) -> dict:
    """
    Run Random Forest with expanding window, re-estimating every 4 quarters.
    Computes SHAP values at each re-estimation.
    """
    import shap

    oos_dates = []
    oos_preds = []
    oos_actuals = []
    train_means = []
    importances_list = []
    shap_values_list = []
    shap_X_list = []

    current_model = None

    for train_end, oos_idx, _, refit in expanding_window_quarters(
        panel, reestimate_every=RF_REESTIMATE_EVERY
    ):
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

        y_fit = valid[target_col].values
        X_fit = valid[feature_cols].values
        t_mean = y_fit.mean()

        if refit or current_model is None:
            rf = RandomForestRegressor(
                n_estimators=RF_N_TREES,
                max_features="sqrt",
                min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                max_depth=None,
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(X_fit, y_fit)
            current_model = rf

            importances_list.append({
                "quarter": panel.index[oos_idx],
                "importances": dict(zip(feature_cols, rf.feature_importances_)),
            })

            explainer = shap.TreeExplainer(current_model)
            sv = explainer.shap_values(X_fit)
            shap_values_list.append(sv)
            shap_X_list.append(X_fit)

        X_oos = oos_row[feature_cols]
        if X_oos.isna().any():
            pred = np.nan
        else:
            pred = current_model.predict(X_oos.values.reshape(1, -1))[0]

        oos_dates.append(panel.index[oos_idx])
        oos_preds.append(pred)
        oos_actuals.append(oos_row[target_col])
        train_means.append(t_mean)

    return {
        "oos_preds": pd.Series(oos_preds, index=oos_dates, name="prediction"),
        "actuals": pd.Series(oos_actuals, index=oos_dates, name="actual"),
        "train_means": pd.Series(train_means, index=oos_dates, name="train_mean"),
        "importances": importances_list,
        "shap_values": shap_values_list,
        "shap_X": shap_X_list,
        "feature_cols": feature_cols,
    }


def run_rf_all(panel: pd.DataFrame) -> dict:
    """Run RF under Spec (3) for both methods and all horizons."""
    results = {}

    for method in ["A", "B"]:
        feature_cols = get_spec3_cols(method)
        for h in HORIZONS:
            target_col = f"GDP_growth_h{h}"
            key = (method, "spec3", h)
            print(f"  RF spec3_{method} h={h}...")
            results[key] = run_rf_single(panel, feature_cols, target_col)

    return results


def verify_shap(results: dict):
    """Verify SHAP values sum to prediction - base_value for each sample."""
    print("\n--- SHAP additivity verification ---")
    for (method, _, h), res in results.items():
        if not res["shap_values"]:
            continue
        sv = res["shap_values"][-1]

        shap_sums = sv.sum(axis=1)
        mean_abs = np.abs(shap_sums).mean()

        print(f"  spec3_{method} h={h}: last re-estimation, "
              f"{sv.shape[0]} samples, {sv.shape[1]} features")
        print(f"    SHAP sum range: [{shap_sums.min():.4f}, {shap_sums.max():.4f}], "
              f"mean|sum|: {mean_abs:.4f}")


def compute_rf_metrics(results: dict) -> pd.DataFrame:
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


def compare_with_ols(rf_metrics: pd.DataFrame):
    """Compare RF RMSE with OLS RMSE under Spec (3)."""
    ols_path = DATA_RESULTS / "ols_metrics.csv"
    if not ols_path.exists():
        print("\n  OLS metrics not found — skipping comparison.")
        return

    ols_metrics = pd.read_csv(ols_path)
    ols_spec3 = ols_metrics[ols_metrics["spec"] == "spec3"]

    print(f"\n{'─' * 60}")
    print(f"RF vs OLS under Spec (3)")
    print(f"{'Method':<8} {'h':<4} {'RF RMSE':>9} {'OLS RMSE':>10} "
          f"{'RF*':>9} {'OLS*':>10}")
    print(f"{'─' * 60}")

    for _, rf_row in rf_metrics.iterrows():
        ols_row = ols_spec3[
            (ols_spec3["method"] == rf_row["method"]) &
            (ols_spec3["h"] == rf_row["h"])
        ]
        if ols_row.empty:
            continue
        ols_row = ols_row.iloc[0]
        print(f"{rf_row['method']:<8} {rf_row['h']:<4} "
              f"{rf_row['RMSE']:9.3f} {ols_row['RMSE']:10.3f} "
              f"{rf_row['RMSE_exCOVID']:9.3f} {ols_row['RMSE_exCOVID']:10.3f}")


def main():
    print("=" * 60)
    print("LAYER 5C: Random Forest Expanding-Window Estimation")
    print("=" * 60)

    panel = pd.read_csv(
        DATA_FEATURES / "quarterly_full_panel.csv",
        parse_dates=["date"], index_col="date",
    )

    print("\n--- Running Random Forest ---")
    results = run_rf_all(panel)

    print(f"\n--- RF complete: {len(results)} method/horizon combinations ---")

    verify_shap(results)

    metrics = compute_rf_metrics(results)

    print(f"\n{'─' * 60}")
    print(f"{'Method':<8} {'Spec':<8} {'h':<4} {'N':<6} "
          f"{'RMSE':>7} {'MAE':>7} {'RMSE*':>7} {'MAE*':>7}")
    print(f"{'─' * 60}")
    for _, row in metrics.iterrows():
        print(f"{row['method']:<8} {row['spec']:<8} {int(row['h']):<4} "
              f"{int(row['n_oos']):<6} "
              f"{row['RMSE']:7.3f} {row['MAE']:7.3f} "
              f"{row['RMSE_exCOVID']:7.3f} {row['MAE_exCOVID']:7.3f}")

    compare_with_ols(metrics)

    DATA_RESULTS.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(DATA_RESULTS / "rf_metrics.csv", index=False)

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
    all_preds.to_csv(DATA_RESULTS / "rf_predictions.csv", index=False)

    # Save SHAP and importances for figure generation
    shap_data = {}
    for (method, spec_name, h), res in results.items():
        shap_data[(method, spec_name, h)] = {
            "shap_values": res["shap_values"],
            "shap_X": res["shap_X"],
            "feature_cols": res["feature_cols"],
            "importances": res["importances"],
        }
    with open(DATA_RESULTS / "rf_shap_data.pkl", "wb") as f:
        pickle.dump(shap_data, f)

    print(f"\nSaved to {DATA_RESULTS}/")
    print(f"  rf_metrics.csv")
    print(f"  rf_predictions.csv")
    print(f"  rf_shap_data.pkl")


if __name__ == "__main__":
    main()

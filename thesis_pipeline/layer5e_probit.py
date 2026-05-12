"""
layer5e_probit.py -- Probit expanding-window estimation for below-trend indicator.

Usage:
    python thesis_pipeline/layer5e_probit.py
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


def run_probit_single(panel: pd.DataFrame, feature_cols: list[str],
                      h: int) -> dict:
    """
    Run probit with expanding window for one spec/horizon.
    Target: below_trend shifted by -h (predict h quarters ahead).
    """
    target = panel["below_trend"].shift(-h)

    oos_dates = []
    oos_probs = []
    oos_actuals = []

    for train_end, oos_idx, _, _ in expanding_window_quarters(panel):
        train = panel.iloc[:train_end + 1]
        oos_row = panel.iloc[oos_idx]

        y_train = target.iloc[:train_end + 1]
        X_train = train[feature_cols]
        valid = pd.concat([y_train, X_train], axis=1).dropna()

        if len(valid) < len(feature_cols) + 2:
            oos_dates.append(panel.index[oos_idx])
            oos_probs.append(np.nan)
            oos_actuals.append(target.iloc[oos_idx])
            continue

        y_fit = valid.iloc[:, 0]
        X_fit = sm.add_constant(valid[feature_cols])

        try:
            model = sm.Probit(y_fit, X_fit).fit(disp=0, maxiter=100)
        except Exception:
            oos_dates.append(panel.index[oos_idx])
            oos_probs.append(np.nan)
            oos_actuals.append(target.iloc[oos_idx])
            continue

        X_oos = oos_row[feature_cols]
        if X_oos.isna().any():
            prob = np.nan
        else:
            X_oos_c = np.concatenate([[1.0], X_oos.values])
            prob = model.predict(X_oos_c.reshape(1, -1))[0]

        oos_dates.append(panel.index[oos_idx])
        oos_probs.append(prob)
        oos_actuals.append(target.iloc[oos_idx])

    return {
        "oos_probs": pd.Series(oos_probs, index=oos_dates, name="prob"),
        "actuals": pd.Series(oos_actuals, index=oos_dates, name="actual"),
    }


def run_probit_all(panel: pd.DataFrame) -> dict:
    """
    Run probit for specs (1) and (2), both methods, all horizons.
    Returns dict keyed by (method, spec_name, horizon).
    """
    results = {}

    for method in ["A", "B"]:
        specs = get_specs(method)
        for spec_name in ["spec1", "spec2"]:
            feature_cols = specs[spec_name]
            for h in HORIZONS:
                key = (method, spec_name, h)

                if spec_name == "spec1" and method == "B":
                    results[key] = results[("A", "spec1", h)]
                    continue

                if spec_name == "spec1":
                    print(f"  Probit {spec_name} h={h} (shared)...")
                else:
                    print(f"  Probit {spec_name}_{method} h={h}...")

                results[key] = run_probit_single(panel, feature_cols, h)

    return results


def print_probit_summary(results: dict):
    from sklearn.metrics import roc_auc_score
    from thesis_pipeline.layer5_specs import COVID_QUARTER

    print(f"\n{'─' * 70}")
    print(f"{'Method':<8} {'Spec':<8} {'h':<4} {'N':<6} "
          f"{'QPS':>7} {'AUROC':>7} {'QPS*':>7} {'AUROC*':>7}")
    print(f"{'─' * 70}")
    print(f"{'':>44} (* = excl. 2020Q2)")

    rows = []
    for (method, spec_name, h), res in results.items():
        probs = res["oos_probs"]
        actuals = res["actuals"]
        valid = probs.notna() & actuals.notna()
        p, a = probs[valid], actuals[valid]
        if len(p) == 0:
            continue

        qps = (2 * (a - p) ** 2).mean()
        try:
            auroc = roc_auc_score(a, p)
        except ValueError:
            auroc = np.nan

        mask = p.index != COVID_QUARTER
        p_nc, a_nc = p[mask], a[mask]
        qps_nc = (2 * (a_nc - p_nc) ** 2).mean()
        try:
            auroc_nc = roc_auc_score(a_nc, p_nc)
        except ValueError:
            auroc_nc = np.nan

        print(f"{method:<8} {spec_name:<8} {h:<4} {len(p):<6} "
              f"{qps:7.4f} {auroc:7.4f} {qps_nc:7.4f} {auroc_nc:7.4f}")

        rows.append({
            "method": method, "spec": spec_name, "h": h,
            "n_oos": len(p), "QPS": qps, "AUROC": auroc,
            "QPS_exCOVID": qps_nc, "AUROC_exCOVID": auroc_nc,
        })

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("LAYER 5E: Probit Expanding-Window Estimation")
    print("=" * 60)

    panel = pd.read_csv(
        DATA_FEATURES / "quarterly_full_panel.csv",
        parse_dates=["date"], index_col="date",
    )

    print("\n--- Running probit regressions ---")
    results = run_probit_all(panel)

    print(f"\n--- Probit complete: {len(results)} spec/horizon combinations ---")

    # Verification: probabilities in [0, 1]
    print("\n--- Verification ---")
    for (method, spec_name, h), res in results.items():
        probs = res["oos_probs"].dropna()
        lo, hi = probs.min(), probs.max()
        ok = lo >= 0 and hi <= 1
        status = "OK" if ok else "FAIL"
        print(f"  {status}: {spec_name}_{method} h={h}: "
              f"prob range [{lo:.4f}, {hi:.4f}]")

    metrics = print_probit_summary(results)

    DATA_RESULTS.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(DATA_RESULTS / "probit_metrics.csv", index=False)

    pred_frames = []
    for (method, spec_name, h), res in results.items():
        df = pd.DataFrame({
            "method": method, "spec": spec_name, "h": h,
            "date": res["oos_probs"].index,
            "prob": res["oos_probs"].values,
            "actual": res["actuals"].values,
        })
        pred_frames.append(df)
    all_preds = pd.concat(pred_frames, ignore_index=True)
    all_preds.to_csv(DATA_RESULTS / "probit_predictions.csv", index=False)

    print(f"\nSaved to {DATA_RESULTS}/")
    print(f"  probit_metrics.csv")
    print(f"  probit_predictions.csv")


if __name__ == "__main__":
    main()

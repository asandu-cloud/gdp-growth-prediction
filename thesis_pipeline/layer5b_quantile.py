"""
layer5b_quantile.py -- Quantile regression expanding-window estimation.

Usage:
    python thesis_pipeline/layer5b_quantile.py
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

warnings.filterwarnings("ignore", message="Maximum number of iterations")

from thesis_pipeline.config import DATA_FEATURES, DATA_RESULTS
from thesis_pipeline.layer5_specs import (
    FOCAL_QUANTILE, HORIZONS, QUANTILES, expanding_window_quarters, get_specs,
)


def run_quantile_single(panel: pd.DataFrame, feature_cols: list[str],
                        target_col: str, tau: float) -> dict:
    """
    Run quantile regression with expanding window for one spec/horizon/tau.
    """
    oos_dates = []
    oos_preds = []
    oos_actuals = []

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
            continue

        y_fit = valid[target_col]
        X_fit = sm.add_constant(valid[feature_cols])

        try:
            model = QuantReg(y_fit, X_fit).fit(q=tau, max_iter=1000)
        except Exception:
            oos_dates.append(panel.index[oos_idx])
            oos_preds.append(np.nan)
            oos_actuals.append(oos_row[target_col])
            continue

        X_oos = oos_row[feature_cols]
        if X_oos.isna().any():
            pred = np.nan
        else:
            X_oos_c = np.concatenate([[1.0], X_oos.values])
            pred = model.predict(X_oos_c.reshape(1, -1))[0]

        oos_dates.append(panel.index[oos_idx])
        oos_preds.append(pred)
        oos_actuals.append(oos_row[target_col])

    return {
        "oos_preds": pd.Series(oos_preds, index=oos_dates, name="prediction"),
        "actuals": pd.Series(oos_actuals, index=oos_dates, name="actual"),
    }


def pinball_loss(actual: pd.Series, predicted: pd.Series, tau: float) -> float:
    """Average pinball (quantile) loss: L_τ = mean(ρ_τ(y − q_hat))."""
    u = actual - predicted
    return (u * (tau - (u < 0).astype(float))).mean()


def run_quantile_all(panel: pd.DataFrame) -> dict:
    """
    Run quantile regression for specs (1), (2), (3) × all horizons × all τ × both methods.
    Returns dict keyed by (method, spec_name, h, tau).
    """
    results = {}

    for method in ["A", "B"]:
        specs = get_specs(method)
        # Only specs 1, 2, 3 — not spec4
        qr_specs = {k: v for k, v in specs.items() if not k.startswith("spec4")}

        for spec_name, feature_cols in qr_specs.items():
            for h in HORIZONS:
                target_col = f"GDP_growth_h{h}"
                for tau in QUANTILES:
                    key = (method, spec_name, h, tau)

                    if spec_name == "spec1" and method == "B":
                        results[key] = results[("A", "spec1", h, tau)]
                        continue

                    if spec_name == "spec1":
                        label = f"spec1 h={h} τ={tau:.2f} (shared)"
                    else:
                        label = f"{spec_name}_{method} h={h} τ={tau:.2f}"

                    # Only print for focal quantile to reduce noise
                    if tau == FOCAL_QUANTILE:
                        print(f"  QR {label} (+ other τ)...")

                    res = run_quantile_single(panel, feature_cols, target_col, tau)
                    results[key] = res

    return results


def compute_quantile_metrics(results: dict) -> pd.DataFrame:
    """Compute pinball loss for each result."""
    from thesis_pipeline.layer5_specs import COVID_QUARTER

    rows = []
    for (method, spec_name, h, tau), res in results.items():
        preds = res["oos_preds"]
        actuals = res["actuals"]
        valid = preds.notna() & actuals.notna()
        p, a = preds[valid], actuals[valid]
        if len(p) == 0:
            continue

        loss = pinball_loss(a, p, tau)

        mask = p.index != COVID_QUARTER
        loss_nc = pinball_loss(a[mask], p[mask], tau)

        rows.append({
            "method": method, "spec": spec_name, "h": h, "tau": tau,
            "n_oos": len(p), "pinball": loss, "pinball_exCOVID": loss_nc,
        })

    return pd.DataFrame(rows)


def verify_quantile_ordering(results: dict):
    """Check that τ=0.10 predictions < τ=0.50 predictions."""
    print("\n--- Quantile ordering verification (τ=0.10 < τ=0.50) ---")

    for method in ["A", "B"]:
        for spec_name in ["spec1", "spec2", "spec3"]:
            for h in HORIZONS:
                key_10 = (method, spec_name, h, 0.10)
                key_50 = (method, spec_name, h, 0.50)
                if key_10 not in results or key_50 not in results:
                    continue

                p10 = results[key_10]["oos_preds"]
                p50 = results[key_50]["oos_preds"]
                valid = p10.notna() & p50.notna()
                violations = (p10[valid] > p50[valid]).sum()
                total = valid.sum()
                status = "OK" if violations == 0 else f"WARN ({violations} crossings)"
                if spec_name == "spec1" and method == "B":
                    continue
                label = f"{spec_name}" if spec_name == "spec1" else f"{spec_name}_{method}"
                print(f"  {status}: {label} h={h} — "
                      f"{violations}/{total} quarters with τ=0.10 > τ=0.50")


def print_quantile_summary(metrics: pd.DataFrame):
    """Print pinball loss summary for focal quantile."""
    focal = metrics[metrics["tau"] == FOCAL_QUANTILE]
    print(f"\n{'─' * 60}")
    print(f"Pinball loss at τ = {FOCAL_QUANTILE}")
    print(f"{'Method':<8} {'Spec':<8} {'h':<4} {'N':<6} "
          f"{'Loss':>8} {'Loss*':>8}")
    print(f"{'─' * 60}")
    for _, row in focal.iterrows():
        print(f"{row['method']:<8} {row['spec']:<8} {int(row['h']):<4} "
              f"{int(row['n_oos']):<6} "
              f"{row['pinball']:8.4f} {row['pinball_exCOVID']:8.4f}")


def main():
    print("=" * 60)
    print("LAYER 5B: Quantile Regression Expanding-Window Estimation")
    print("=" * 60)

    panel = pd.read_csv(
        DATA_FEATURES / "quarterly_full_panel.csv",
        parse_dates=["date"], index_col="date",
    )

    print("\n--- Running quantile regressions ---")
    results = run_quantile_all(panel)

    n_unique = len([k for k in results
                    if not (k[1] == "spec1" and k[0] == "B")])
    print(f"\n--- Quantile complete: {len(results)} total "
          f"({n_unique} unique) combinations ---")

    verify_quantile_ordering(results)

    metrics = compute_quantile_metrics(results)
    print_quantile_summary(metrics)

    DATA_RESULTS.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(DATA_RESULTS / "quantile_metrics.csv", index=False)

    pred_frames = []
    for (method, spec_name, h, tau), res in results.items():
        df = pd.DataFrame({
            "method": method, "spec": spec_name, "h": h, "tau": tau,
            "date": res["oos_preds"].index,
            "prediction": res["oos_preds"].values,
            "actual": res["actuals"].values,
        })
        pred_frames.append(df)
    all_preds = pd.concat(pred_frames, ignore_index=True)
    all_preds.to_csv(DATA_RESULTS / "quantile_predictions.csv", index=False)

    print(f"\nSaved to {DATA_RESULTS}/")
    print(f"  quantile_metrics.csv")
    print(f"  quantile_predictions.csv")


if __name__ == "__main__":
    main()

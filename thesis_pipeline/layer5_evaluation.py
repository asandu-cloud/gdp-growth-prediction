"""
layer5_evaluation.py -- Metrics, statistical tests, LaTeX tables, and figures.

Usage:
    python thesis_pipeline/layer5_evaluation.py
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.quantile_regression import QuantReg

from thesis_pipeline.config import (
    DATA_FEATURES, DATA_RESULTS, FIGURES_DIR, TABLES_DIR,
)
from thesis_pipeline.layer5_specs import (
    BENCHMARK_COLS, DERIVED_COLS, FOCAL_QUANTILE, HORIZONS,
    QUANTILES, TRAIN_END, COVID_QUARTER, get_specs,
)

warnings.filterwarnings("ignore", message="Maximum number of iterations")

RECESSIONS = [
    ("1981-07-01", "1982-11-01"),
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]

VAR_LABELS = {
    "const": "Intercept",
    "GDP_growth_lag1": r"GDP$_{t-1}$",
    "term_spread_10Y_2Y": "Term Spread",
    "DFF": "Fed Funds Rate",
    "credit_spread_Baa_Aaa": "Credit Spread",
    "SP500_return": r"S\&P 500 Return",
    "IC4WSA": "Initial Claims",
    "NAPM": "ISM PMI",
    "shannon_A": r"Shannon $H^A$",
    "shannon_B": r"Shannon $H^B$",
    "entropy_momentum_A": r"$\Delta H^A$",
    "entropy_accel_A": r"$\Delta^2 H^A$",
    "entropy_zscore_A": r"$z^A_H$",
    "entropy_vol_A": r"$\sigma^A_H$",
    "entropy_short_A": r"$H^A_{\mathrm{short}}$",
    "entropy_long_A": r"$H^A_{\mathrm{long}}$",
    "entropy_gap_A": r"$G^A_H$",
}
for q in [0.5, 1.5, 2.0]:
    VAR_LABELS[f"tsallis_A_q{q}"] = f"Tsallis $S^A_{{q={q}}}$"


def stars(p):
    if np.isnan(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# ═══════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════

def clark_west(actual, pred_r, pred_u, h=1):
    """Clark-West test for nested models. One-sided: H₁ = unrestricted improves."""
    e1 = actual - pred_r
    e2 = actual - pred_u
    f = e1**2 - (e2**2 - (pred_r - pred_u)**2)
    valid = np.isfinite(f)
    f = f[valid]
    T = len(f)
    if T < 5:
        return np.nan, np.nan
    se = f.std(ddof=1) / np.sqrt(T)
    if se == 0:
        return np.nan, np.nan
    t = f.mean() / se
    p = 1 - stats.t.cdf(t, df=T - 1)
    return t, p


def diebold_mariano(actual, pred1, pred2, h=1):
    """DM test with HLN small-sample correction. Two-sided."""
    d = (actual - pred1)**2 - (actual - pred2)**2
    valid = np.isfinite(d)
    d = d[valid]
    T = len(d)
    if T < 5:
        return np.nan, np.nan
    d_bar = d.mean()
    gamma_0 = np.var(d, ddof=1)
    hac_var = gamma_0
    for k in range(1, h):
        if k >= T:
            break
        gamma_k = np.cov(d[k:], d[:-k], ddof=1)[0, 1]
        hac_var += 2 * gamma_k
    if hac_var <= 0:
        return np.nan, np.nan
    dm_raw = d_bar / np.sqrt(hac_var / T)
    hln = np.sqrt((T + 1 - 2*h + h*(h-1)/T) / T)
    dm = dm_raw * hln
    p = 2 * (1 - stats.t.cdf(abs(dm), df=T - 1))
    return dm, p


def clark_west_qps(actual, prob_r, prob_u, h=1):
    """Clark-West on QPS differentials for probit. One-sided."""
    e1_sq = (actual - prob_r)**2
    e2_sq = (actual - prob_u)**2
    f = e1_sq - (e2_sq - (prob_r - prob_u)**2)
    valid = np.isfinite(f)
    f = f[valid]
    T = len(f)
    if T < 5:
        return np.nan, np.nan
    se = f.std(ddof=1) / np.sqrt(T)
    if se == 0:
        return np.nan, np.nan
    t = f.mean() / se
    p = 1 - stats.t.cdf(t, df=T - 1)
    return t, p


# ═══════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════

def load_predictions():
    preds = {}
    for name in ["ols", "rf", "quantile", "probit", "midas"]:
        preds[name] = pd.read_csv(
            DATA_RESULTS / f"{name}_predictions.csv", parse_dates=["date"],
        )
    return preds


def get_series(df, date_col="date", val_col="prediction", **filters):
    mask = pd.Series(True, index=df.index)
    for k, v in filters.items():
        mask &= df[k] == v
    sub = df[mask].sort_values(date_col)
    return sub


# ═══════════════════════════════════════════════════════════
# 37 PRE-SPECIFIED TESTS
# ═══════════════════════════════════════════════════════════

def run_all_tests(preds):
    ols = preds["ols"]
    rf = preds["rf"]
    probit = preds["probit"]
    midas = preds["midas"]

    rows = []
    tid = 0

    def _ols(method, spec, h):
        s = get_series(ols, method=method, spec=spec, h=h)
        return s["prediction"].values, s["actual"].values

    # 1-6: Spec(1) vs Spec(2), CW, both methods × 3h
    for method in ["A", "B"]:
        for h in HORIZONS:
            tid += 1
            p1, a = _ols(method, "spec1", h)
            p2, _ = _ols(method, "spec2", h)
            t, p = clark_west(a, p1, p2, h=h)
            rows.append(dict(
                id=tid, category="Forecasting", test="CW",
                pair=f"Spec(1) vs Spec(2)$_{{{method}}}$",
                detail=f"h={h}", method=method,
                stat=t, p_value=p,
            ))

    # 7-12: Spec(2) vs Spec(3), CW, both methods × 3h
    for method in ["A", "B"]:
        for h in HORIZONS:
            tid += 1
            p1, a = _ols(method, "spec2", h)
            p2, _ = _ols(method, "spec3", h)
            t, p = clark_west(a, p1, p2, h=h)
            rows.append(dict(
                id=tid, category="Forecasting", test="CW",
                pair=f"Spec(2) vs Spec(3)$_{{{method}}}$",
                detail=f"h={h}", method=method,
                stat=t, p_value=p,
            ))

    # 13-21: Spec(1) vs Spec(4) at each q, CW, Method A × 3h
    for q in [0.5, 1.5, 2.0]:
        for h in HORIZONS:
            tid += 1
            p1, a = _ols("A", "spec1", h)
            p2, _ = _ols("A", f"spec4_q{q}", h)
            t, p = clark_west(a, p1, p2, h=h)
            rows.append(dict(
                id=tid, category="Forecasting", test="CW",
                pair=f"Spec(1) vs Spec(4)$_{{q={q}}}$",
                detail=f"h={h}", method="A",
                stat=t, p_value=p,
            ))

    # 22-24: Spec(2) Method A vs Method B, DM × 3h
    for h in HORIZONS:
        tid += 1
        pA, a = _ols("A", "spec2", h)
        pB, _ = _ols("B", "spec2", h)
        t, p = diebold_mariano(a, pA, pB, h=h)
        rows.append(dict(
            id=tid, category="Forecasting", test="DM",
            pair="Spec(2)$_A$ vs Spec(2)$_B$",
            detail=f"h={h}", method="A vs B",
            stat=t, p_value=p,
        ))

    # 25-27: OLS vs RF under Spec(3), DM, Method A × 3h
    for h in HORIZONS:
        tid += 1
        p_ols, a = _ols("A", "spec3", h)
        rf_s = get_series(rf, method="A", spec="spec3", h=h)
        p_rf = rf_s["prediction"].values
        t, p = diebold_mariano(a, p_ols, p_rf, h=h)
        rows.append(dict(
            id=tid, category="Forecasting", test="DM",
            pair="OLS vs RF, Spec(3)$_A$",
            detail=f"h={h}", method="A",
            stat=t, p_value=p,
        ))

    # 28-31: Entropy MIDAS vs spread MIDAS, DM × 4 cutoffs
    for d in [15, 30, 45, 60]:
        tid += 1
        ent = get_series(midas, hf_var="shannon_A", cutoff=d)
        spr = get_series(midas, hf_var="term_spread_10Y_2Y", cutoff=d)
        t, p = diebold_mariano(
            ent["actual"].values, ent["prediction"].values,
            spr["prediction"].values, h=1,
        )
        rows.append(dict(
            id=tid, category="Nowcasting", test="DM",
            pair="Entropy$_A$ vs Spread MIDAS",
            detail=f"d={d}", method="A",
            stat=t, p_value=p,
        ))

    # 32-37: Probit Spec(1) vs Spec(2), CW on QPS, both methods × 3h
    for method in ["A", "B"]:
        for h in HORIZONS:
            tid += 1
            s1 = get_series(probit, method=method, spec="spec1", h=h)
            s2 = get_series(probit, method=method, spec="spec2", h=h)
            t, p = clark_west_qps(
                s1["actual"].values, s1["prob"].values,
                s2["prob"].values, h=h,
            )
            rows.append(dict(
                id=tid, category="Binary", test="CW",
                pair=f"Probit(1) vs Probit(2)$_{{{method}}}$",
                detail=f"h={h}", method=method,
                stat=t, p_value=p,
            ))

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
# FULL-SAMPLE RE-ESTIMATION (for coefficient tables)
# ═══════════════════════════════════════════════════════════

def estimate_full_sample_ols(panel, method="A"):
    specs = get_specs(method)
    results = {}
    for spec_name, feature_cols in specs.items():
        for h in HORIZONS:
            target = f"GDP_growth_h{h}"
            y = panel[target]
            X = panel[feature_cols]
            valid = pd.concat([y, X], axis=1).dropna()
            y_fs = valid[target]
            X_fs = sm.add_constant(valid[feature_cols])
            model = sm.OLS(y_fs, X_fs).fit(
                cov_type="HAC", cov_kwds={"maxlags": h},
            )
            results[(spec_name, h)] = model
    return results


def estimate_full_sample_qr(panel, method="A", tau=0.10):
    specs = get_specs(method)
    qr_specs = {k: v for k, v in specs.items() if not k.startswith("spec4")}
    results = {}
    for spec_name, feature_cols in qr_specs.items():
        for h in HORIZONS:
            target = f"GDP_growth_h{h}"
            y = panel[target]
            X = panel[feature_cols]
            valid = pd.concat([y, X], axis=1).dropna()
            y_fs = valid[target]
            X_fs = sm.add_constant(valid[feature_cols])
            try:
                model = QuantReg(y_fs, X_fs).fit(q=tau, max_iter=1000)
                results[(spec_name, h)] = model
            except Exception:
                pass
    return results


# ═══════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════

def compute_ols_oos_metrics(ols_df):
    rows = []
    for (method, spec, h), grp in ols_df.groupby(["method", "spec", "h"]):
        g = grp.dropna(subset=["prediction", "actual", "train_mean"])
        if len(g) == 0:
            continue
        a, p, m = g["actual"].values, g["prediction"].values, g["train_mean"].values
        mse_model = ((a - p)**2).sum()
        mse_bench = ((a - m)**2).sum()
        rmse = np.sqrt(mse_model / len(a))
        mae = np.abs(a - p).mean()
        delta = 1 - mse_model / mse_bench if mse_bench > 0 else np.nan

        nc = g["date"] != COVID_QUARTER
        a_nc, p_nc, m_nc = a[nc.values], p[nc.values], m[nc.values]
        mse_nc = ((a_nc - p_nc)**2).sum()
        mse_b_nc = ((a_nc - m_nc)**2).sum()
        rmse_nc = np.sqrt(mse_nc / len(a_nc))
        mae_nc = np.abs(a_nc - p_nc).mean()
        delta_nc = 1 - mse_nc / mse_b_nc if mse_b_nc > 0 else np.nan

        rows.append(dict(
            method=method, spec=spec, h=h, n=len(a),
            RMSE=rmse, MAE=mae, dMSE=delta,
            RMSE_nc=rmse_nc, MAE_nc=mae_nc, dMSE_nc=delta_nc,
        ))
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
# TABLE GENERATORS
# ═══════════════════════════════════════════════════════════

def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"  Saved {path.name}")


def generate_test_results_table(test_df):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Diebold--Mariano and Clark--West Test Results}",
        r"\label{tab:test_results}",
        r"\footnotesize",
        r"\begin{tabular}{rllllrrl}",
        r"\toprule",
        r"\# & Category & Test & Comparison & & Stat & $p$ & \\",
        r"\midrule",
    ]
    for _, row in test_df.iterrows():
        s = stars(row["p_value"])
        stat_str = f"{row['stat']:.3f}" if np.isfinite(row["stat"]) else "--"
        p_str = f"{row['p_value']:.3f}" if np.isfinite(row["p_value"]) else "--"
        lines.append(
            f"  {int(row['id'])} & {row['category']} & {row['test']} & "
            f"{row['pair']} & {row['detail']} & "
            f"{stat_str} & {p_str} & "
            f"{'$^{' + s + '}$' if s else ''} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write(TABLES_DIR / "test_results.tex", "\n".join(lines))
    test_df.to_csv(TABLES_DIR / "test_results.csv", index=False)


def generate_ols_coef_tables(ols_models, method="A"):
    spec_order = ["spec1", "spec2", "spec3",
                  "spec4_q0.5", "spec4_q1.5", "spec4_q2.0"]
    spec_headers = ["(1)", "(2)", "(3)",
                    r"(4) $q\!=\!0.5$", r"(4) $q\!=\!1.5$", r"(4) $q\!=\!2.0$"]

    specs_def = get_specs(method)
    all_vars = (["const"] + BENCHMARK_COLS
                + [f"shannon_{method}"] + DERIVED_COLS[method]
                + [f"tsallis_{method}_q{q}" for q in [0.5, 1.5, 2.0]])

    for h in HORIZONS:
        ncols = len(spec_order)
        col_spec = "l" + "c" * ncols
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{OLS Estimates, Method {method}, $h = {h}$}}",
            f"\\label{{tab:ols_coef_{method}_h{h}}}",
            r"\footnotesize",
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            " & ".join([""] + spec_headers) + r" \\",
            r"\midrule",
        ]

        for var in all_vars:
            label = VAR_LABELS.get(var, var.replace("_", r"\_"))
            coef_cells = []
            se_cells = []
            for spec_name in spec_order:
                key = (spec_name, h)
                if key not in ols_models:
                    coef_cells.append("")
                    se_cells.append("")
                    continue
                model = ols_models[key]
                if var in model.params.index:
                    c = model.params[var]
                    s = model.bse[var]
                    pv = model.pvalues[var]
                    st = stars(pv)
                    sup = f"$^{{{st}}}$" if st else ""
                    coef_cells.append(f"{c:.3f}{sup}")
                    se_cells.append(f"({s:.3f})")
                else:
                    coef_cells.append("")
                    se_cells.append("")

            if all(c == "" for c in coef_cells):
                continue
            lines.append(f"  {label} & " + " & ".join(coef_cells) + r" \\")
            lines.append(f"  & " + " & ".join(se_cells) + r" \\[2pt]")

        lines.append(r"\midrule")

        n_cells, r2_cells = [], []
        for spec_name in spec_order:
            key = (spec_name, h)
            if key in ols_models:
                m = ols_models[key]
                n_cells.append(f"{int(m.nobs)}")
                r2_cells.append(f"{m.rsquared:.3f}")
            else:
                n_cells.append("")
                r2_cells.append("")
        lines.append(f"  $N$ & " + " & ".join(n_cells) + r" \\")
        lines.append(f"  $R^2$ & " + " & ".join(r2_cells) + r" \\")

        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        _write(TABLES_DIR / f"ols_coefficients_h{h}.tex", "\n".join(lines))


def generate_ols_oos_table(ols_df):
    metrics = compute_ols_oos_metrics(ols_df)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{OLS Out-of-Sample Performance}",
        r"\label{tab:ols_oos}",
        r"\footnotesize",
        r"\begin{tabular}{llrccccccc}",
        r"\toprule",
        r"Method & Spec & $h$ & $N$ & RMSE & MAE & $\Delta$MSE"
        r" & RMSE$^*$ & MAE$^*$ & $\Delta$MSE$^*$ \\",
        r"\midrule",
    ]
    for _, r in metrics.iterrows():
        lines.append(
            f"  {r['method']} & {r['spec']} & {int(r['h'])} & {int(r['n'])} & "
            f"{r['RMSE']:.3f} & {r['MAE']:.3f} & {r['dMSE']:.3f} & "
            f"{r['RMSE_nc']:.3f} & {r['MAE_nc']:.3f} & {r['dMSE_nc']:.3f} \\\\"
        )
    lines += [r"\bottomrule",
              r"\multicolumn{10}{l}{\footnotesize $^*$ Excluding 2020Q2.} \\",
              r"\end{tabular}", r"\end{table}"]
    _write(TABLES_DIR / "ols_oos_performance.tex", "\n".join(lines))
    metrics.to_csv(TABLES_DIR / "ols_oos_performance.csv", index=False)


def generate_qr_coef_table(qr_models, method="A"):
    spec_order = ["spec1", "spec2", "spec3"]
    spec_headers = ["(1)", "(2)", "(3)"]
    specs_def = get_specs(method)
    all_vars = (["const"] + BENCHMARK_COLS
                + [f"shannon_{method}"] + DERIVED_COLS[method])

    for h in HORIZONS:
        ncols = len(spec_order)
        col_spec = "l" + "c" * ncols
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{Quantile Regression ($\\tau = {FOCAL_QUANTILE}$), "
            f"Method {method}, $h = {h}$}}",
            f"\\label{{tab:qr_coef_{method}_h{h}}}",
            r"\footnotesize",
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            " & ".join([""] + spec_headers) + r" \\",
            r"\midrule",
        ]
        for var in all_vars:
            label = VAR_LABELS.get(var, var.replace("_", r"\_"))
            coef_cells, se_cells = [], []
            for spec_name in spec_order:
                key = (spec_name, h)
                if key not in qr_models:
                    coef_cells.append("")
                    se_cells.append("")
                    continue
                model = qr_models[key]
                if var in model.params.index:
                    c = model.params[var]
                    s = model.bse[var]
                    pv = model.pvalues[var]
                    st = stars(pv)
                    sup = f"$^{{{st}}}$" if st else ""
                    coef_cells.append(f"{c:.3f}{sup}")
                    se_cells.append(f"({s:.3f})")
                else:
                    coef_cells.append("")
                    se_cells.append("")
            if all(c == "" for c in coef_cells):
                continue
            lines.append(f"  {label} & " + " & ".join(coef_cells) + r" \\")
            lines.append(f"  & " + " & ".join(se_cells) + r" \\[2pt]")

        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        _write(TABLES_DIR / f"qr_coefficients_h{h}.tex", "\n".join(lines))


def generate_qr_pinball_table():
    metrics = pd.read_csv(DATA_RESULTS / "quantile_metrics.csv")
    focal = metrics[metrics["tau"] == FOCAL_QUANTILE]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{Quantile Regression Pinball Loss ($\\tau = {FOCAL_QUANTILE}$)}}",
        r"\label{tab:qr_pinball}",
        r"\footnotesize",
        r"\begin{tabular}{llrccc}",
        r"\toprule",
        r"Method & Spec & $h$ & $N$ & Pinball & Pinball$^*$ \\",
        r"\midrule",
    ]
    for _, r in focal.iterrows():
        lines.append(
            f"  {r['method']} & {r['spec']} & {int(r['h'])} & "
            f"{int(r['n_oos'])} & {r['pinball']:.4f} & "
            f"{r['pinball_exCOVID']:.4f} \\\\"
        )
    lines += [r"\bottomrule",
              r"\multicolumn{6}{l}{\footnotesize $^*$ Excluding 2020Q2.} \\",
              r"\end{tabular}", r"\end{table}"]
    _write(TABLES_DIR / "qr_pinball_loss.tex", "\n".join(lines))


def generate_rf_comparison_table():
    ols_m = pd.read_csv(DATA_RESULTS / "ols_metrics.csv")
    rf_m = pd.read_csv(DATA_RESULTS / "rf_metrics.csv")
    ols3 = ols_m[ols_m["spec"] == "spec3"]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Random Forest vs.\ OLS under Specification (3)}",
        r"\label{tab:rf_vs_ols}",
        r"\footnotesize",
        r"\begin{tabular}{llrcccc}",
        r"\toprule",
        r"Method & $h$ & $N$ & OLS RMSE & RF RMSE & OLS RMSE$^*$ & RF RMSE$^*$ \\",
        r"\midrule",
    ]
    for _, rf_row in rf_m.iterrows():
        ols_row = ols3[
            (ols3["method"] == rf_row["method"]) & (ols3["h"] == rf_row["h"])
        ].iloc[0]
        lines.append(
            f"  {rf_row['method']} & {int(rf_row['h'])} & "
            f"{int(rf_row['n_oos'])} & "
            f"{ols_row['RMSE']:.3f} & {rf_row['RMSE']:.3f} & "
            f"{ols_row['RMSE_exCOVID']:.3f} & {rf_row['RMSE_exCOVID']:.3f} \\\\"
        )
    lines += [r"\bottomrule",
              r"\multicolumn{7}{l}{\footnotesize $^*$ Excluding 2020Q2.} \\",
              r"\end{tabular}", r"\end{table}"]
    _write(TABLES_DIR / "rf_vs_ols.tex", "\n".join(lines))


def generate_midas_table():
    metrics = pd.read_csv(DATA_RESULTS / "midas_metrics.csv")
    label_map = {
        "shannon_A": r"Shannon $H^A$",
        "shannon_B": r"Shannon $H^B$",
        "term_spread_10Y_2Y": "Term Spread",
    }
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{MIDAS Nowcasting Performance}",
        r"\label{tab:midas}",
        r"\footnotesize",
        r"\begin{tabular}{lrcccc}",
        r"\toprule",
        r"HF Variable & $d$ & $N$ & RMSE & MAE & RMSE$^*$ \\",
        r"\midrule",
    ]
    for _, r in metrics.iterrows():
        lab = label_map.get(r["hf_var"], r["hf_var"])
        lines.append(
            f"  {lab} & {int(r['cutoff'])} & {int(r['n_oos'])} & "
            f"{r['RMSE']:.3f} & {r['MAE']:.3f} & {r['RMSE_exCOVID']:.3f} \\\\"
        )
    lines += [r"\bottomrule",
              r"\multicolumn{6}{l}{\footnotesize $^*$ Excluding 2020Q2.} \\",
              r"\end{tabular}", r"\end{table}"]
    _write(TABLES_DIR / "midas_nowcasting.tex", "\n".join(lines))


def generate_probit_table():
    metrics = pd.read_csv(DATA_RESULTS / "probit_metrics.csv")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Probit Out-of-Sample Performance}",
        r"\label{tab:probit}",
        r"\footnotesize",
        r"\begin{tabular}{llrccccc}",
        r"\toprule",
        r"Method & Spec & $h$ & $N$ & QPS & AUROC & QPS$^*$ & AUROC$^*$ \\",
        r"\midrule",
    ]
    for _, r in metrics.iterrows():
        lines.append(
            f"  {r['method']} & {r['spec']} & {int(r['h'])} & "
            f"{int(r['n_oos'])} & {r['QPS']:.4f} & {r['AUROC']:.4f} & "
            f"{r['QPS_exCOVID']:.4f} & {r['AUROC_exCOVID']:.4f} \\\\"
        )
    lines += [r"\bottomrule",
              r"\multicolumn{8}{l}{\footnotesize $^*$ Excluding 2020Q2.} \\",
              r"\end{tabular}", r"\end{table}"]
    _write(TABLES_DIR / "probit_results.tex", "\n".join(lines))


# ═══════════════════════════════════════════════════════════
# FIGURE GENERATORS
# ═══════════════════════════════════════════════════════════

def generate_shap_figure():
    import shap

    with open(DATA_RESULTS / "rf_shap_data.pkl", "rb") as f:
        shap_data = pickle.load(f)

    key = ("A", "spec3", 1)
    if key not in shap_data:
        print("  SHAP data not found for Method A / Spec(3) / h=1")
        return
    sd = shap_data[key]
    sv = sd["shap_values"][-1]
    X_data = sd["shap_X"][-1]
    feature_cols = sd["feature_cols"]

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        sv, X_data, feature_names=feature_cols,
        show=False, plot_size=None,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.pdf", bbox_inches="tight", dpi=150)
    plt.close("all")
    print("  Saved shap_summary.pdf")


def generate_midas_rmse_figure():
    metrics = pd.read_csv(DATA_RESULTS / "midas_metrics.csv")
    cutoffs = sorted(metrics["cutoff"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))

    for hf_var, label, ls in [
        ("shannon_A", "Shannon Entropy (A)", "-"),
        ("shannon_B", "Shannon Entropy (B)", "--"),
        ("term_spread_10Y_2Y", "Term Spread", "-."),
    ]:
        sub = metrics[metrics["hf_var"] == hf_var].sort_values("cutoff")
        ax.plot(sub["cutoff"], sub["RMSE_exCOVID"], ls, marker="o",
                label=label, linewidth=1.5)

    ax.set_xlabel("Within-Quarter Cutoff (trading days)")
    ax.set_ylabel("RMSE (excl. COVID)")
    ax.set_xticks(cutoffs)
    ax.legend(frameon=False)
    ax.set_title("MIDAS Nowcasting: RMSE Trajectory")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "midas_rmse_trajectory.pdf",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved midas_rmse_trajectory.pdf")


def generate_entropy_timeseries_figure():
    daily = pd.read_csv(
        DATA_FEATURES / "daily_entropy_for_midas.csv",
        parse_dates=["date"], index_col="date",
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily.index, daily["shannon_A"], linewidth=0.3, color="steelblue")
    for start, end in RECESSIONS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                    alpha=0.20, color="grey", zorder=0)
    ax.set_ylabel("Shannon Entropy (Method A)")
    ax.set_title("Daily Yield Curve Entropy with NBER Recession Shading")
    ax.set_xlim(daily.index[0], daily.index[-1])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "entropy_timeseries.pdf",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved entropy_timeseries.pdf")


def generate_quantile_fan_figure():
    qr = pd.read_csv(
        DATA_RESULTS / "quantile_predictions.csv", parse_dates=["date"],
    )
    method, spec, h = "A", "spec2", 1
    sub = qr[(qr["method"] == method) & (qr["spec"] == spec) & (qr["h"] == h)]

    pivot = sub.pivot_table(
        index="date", columns="tau", values="prediction",
    ).sort_index()
    actuals = sub.drop_duplicates("date").set_index("date")["actual"].sort_index()

    if pivot.empty:
        print("  Quantile fan: no data for spec2_A h=1")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    dates = pivot.index

    if 0.10 in pivot.columns and 0.90 in pivot.columns:
        ax.fill_between(dates, pivot[0.10], pivot[0.90],
                        alpha=0.15, color="steelblue", label="10%--90%")
    if 0.25 in pivot.columns and 0.75 in pivot.columns:
        ax.fill_between(dates, pivot[0.25], pivot[0.75],
                        alpha=0.25, color="steelblue", label="25%--75%")
    if 0.50 in pivot.columns:
        ax.plot(dates, pivot[0.50], color="navy", linewidth=1, label="Median")

    ax.plot(actuals.index, actuals.values, "k.", markersize=2.5, label="Actual",
            zorder=5)

    for start, end in RECESSIONS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                    alpha=0.15, color="grey", zorder=0)

    ax.set_ylabel("GDP Growth (%)")
    ax.set_title(f"Quantile Fan Chart — Spec (2)$_A$, $h = {h}$")
    ax.legend(frameon=False, loc="lower left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "quantile_fan.pdf", bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved quantile_fan.pdf")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("LAYER 5 EVALUATION: Tests, Tables, Figures")
    print("=" * 60)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print("\n--- Loading predictions ---")
    preds = load_predictions()
    for k, v in preds.items():
        print(f"  {k}: {len(v)} rows")

    panel = pd.read_csv(
        DATA_FEATURES / "quarterly_full_panel.csv",
        parse_dates=["date"], index_col="date",
    )

    # ── 37 statistical tests ──
    print("\n--- Running 37 pre-specified tests ---")
    test_df = run_all_tests(preds)

    n_sig = (test_df["p_value"] < 0.10).sum()
    n_sig5 = (test_df["p_value"] < 0.05).sum()
    print(f"  {len(test_df)} tests completed")
    print(f"  Significant at 10%: {n_sig}")
    print(f"  Significant at  5%: {n_sig5}")

    print(f"\n{'─' * 75}")
    print(f"{'#':>3}  {'Cat':<12} {'Test':<4} {'Comparison':<35} "
          f"{'':>5} {'Stat':>7} {'p':>7}")
    print(f"{'─' * 75}")
    for _, r in test_df.iterrows():
        s = stars(r["p_value"])
        stat_s = f"{r['stat']:7.3f}" if np.isfinite(r["stat"]) else "     --"
        p_s = f"{r['p_value']:7.3f}" if np.isfinite(r["p_value"]) else "     --"
        pair_clean = r["pair"].replace("$", "").replace("{", "").replace("}", "")
        pair_clean = pair_clean.replace("_", "").replace("\\", "")
        print(f"{int(r['id']):>3}  {r['category']:<12} {r['test']:<4} "
              f"{pair_clean:<35} {r['detail']:>5} {stat_s} {p_s} {s}")

    # ── Full-sample OLS re-estimation ──
    print("\n--- Full-sample OLS re-estimation (Method A) ---")
    ols_models = estimate_full_sample_ols(panel, method="A")
    print(f"  {len(ols_models)} models estimated")

    # ── Full-sample QR re-estimation ──
    print("--- Full-sample QR re-estimation (τ=0.10, Method A) ---")
    qr_models = estimate_full_sample_qr(panel, method="A", tau=FOCAL_QUANTILE)
    print(f"  {len(qr_models)} models estimated")

    # ── Generate tables ──
    print("\n--- Generating LaTeX tables ---")
    generate_test_results_table(test_df)
    generate_ols_coef_tables(ols_models, method="A")
    generate_ols_oos_table(preds["ols"])
    generate_qr_coef_table(qr_models, method="A")
    generate_qr_pinball_table()
    generate_rf_comparison_table()
    generate_midas_table()
    generate_probit_table()

    # ── Generate figures ──
    print("\n--- Generating figures ---")
    generate_shap_figure()
    generate_midas_rmse_figure()
    generate_entropy_timeseries_figure()
    generate_quantile_fan_figure()

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print(f"  Tables: {TABLES_DIR}/")
    print(f"  Figures: {FIGURES_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

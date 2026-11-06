"""Calibration plots and horizon metrics for saved RSF bootstrap models.

Optimized for Reviewer revision:
- Targets specific evaluation horizons (12, 36, 60 months) directly.
- Completely avoids slow full-grid evaluations and unnecessary Dykstra iterations.
- Generates publication-ready figures (PDF & PNG) and metrics summary tables.

Usage:
  python Calibration_plot.py --ratio 1 --settings tra75 --n-rep 100
"""

import argparse
import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

# Add candidate project root paths dynamically relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for path_entry in (
    SCRIPT_DIR,
    os.path.abspath("."),
    os.path.join(SCRIPT_DIR, ".."),
    os.path.join(SCRIPT_DIR, "../test"),
    os.path.join(SCRIPT_DIR, "../.."),
):
    if os.path.isdir(path_entry) and path_entry not in sys.path:
        sys.path.append(path_entry)

# Import or fallback compute_ipcw_targets_and_weights
try:
    from src.RSF_ISR_diff2 import compute_ipcw_targets_and_weights
except ImportError:
    try:
        from RSF_ISR_diff2 import compute_ipcw_targets_and_weights
    except ImportError:
        from sksurv.nonparametric import kaplan_meier_estimator

        def compute_ipcw_targets_and_weights(y, time_grid):
            time_col = y["time"]
            event_col = y["event"]
            km_t, km_p = kaplan_meier_estimator(~event_col, time_col)

            def G(t):
                idx = np.searchsorted(km_t, t, side="right") - 1
                if idx >= 0:
                    return max(km_p[idx], 1e-4)
                return 1.0

            n = len(time_col)
            T = len(time_grid)
            weights = np.zeros((n, T))
            y_surv_matrix = np.zeros((n, T))
            for j, t in enumerate(time_grid):
                for i in range(n):
                    if time_col[i] <= t and event_col[i] == 1:
                        y_surv_matrix[i, j] = 0.0
                        weights[i, j] = 1.0 / G(time_col[i])
                    elif time_col[i] > t:
                        y_surv_matrix[i, j] = 1.0
                        weights[i, j] = 1.0 / G(t)
                    else:
                        y_surv_matrix[i, j] = 0.0
                        weights[i, j] = 0.0
            return y_surv_matrix, weights


def load_rsf_model(path, setting):
    """Load a fitted RSF model, remapping legacy __main__ class references."""
    compat_class = None
    try:
        if setting == "tra75":
            from src.RSF_ISR_diff3_75 import RSF_Calibrated2 as compat_class
        else:
            from src.RSF_ISR_diff2 import RSF_Calibrated2 as compat_class
    except ImportError:
        try:
            if setting == "tra75":
                from RSF_ISR_diff3_75 import RSF_Calibrated2 as compat_class
            else:
                from RSF_ISR_diff2 import RSF_Calibrated2 as compat_class
        except ImportError:
            pass

    class RSFCompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if (module == "__main__" or "RSF_ISR_diff" in module) and name == "RSF_Calibrated2":
                if compat_class is not None:
                    return compat_class
            return super().find_class(module, name)

    with open(path, "rb") as model_file:
        return RSFCompatUnpickler(model_file).load()


SETTINGS = ("tra75", "over75calib")
HORIZONS = (12, 36, 60)


def predict_rsf_and_isr_horizons(model, X, horizons=HORIZONS):
    """Extract RSF and RSF-ISR survival probabilities only for targeted horizons.

    Completely bypasses Dykstra iterations and full-grid evaluations.
    Takes ~1-2 seconds per model repetition on 100k samples instead of tens of minutes.
    """
    # 1. RSF survival function prediction
    surv_funcs = model.base_model.predict_survival_function(X)
    # Evaluate StepFunction only at the targeted horizons: shape (n_samples, len(horizons))
    S_rsf = np.asarray([[fn(h) for h in horizons] for fn in surv_funcs], dtype=float)

    # 2. RW-ISR transformation for targeted horizons
    time_grid = list(getattr(model, "time_grid_", np.arange(1, 72)))
    S_iso = np.empty_like(S_rsf)

    for col_idx, h in enumerate(horizons):
        if h in time_grid:
            t_idx = time_grid.index(h)
        else:
            t_idx = h - 1
        iso_model = model.iso_models_[t_idx]
        S_iso[:, col_idx] = iso_model.transform(S_rsf[:, col_idx])

    return {"RSF": S_rsf, "RSF_ISR": S_iso}


def calibration_curve(risk_boot, y, horizon, n_bins=10):
    """Return decile calibration points and bootstrap prediction 95% CIs.

    Parameters
    ----------
    risk_boot : np.ndarray, shape (n_rep, n_samples)
        Predicted risks for each bootstrap repetition.
    y : structured array
        Survival labels with 'time' and 'event'.
    horizon : int
        Time horizon in months.
    n_bins : int
        Number of risk quantile bins.
    """
    risk_mean = risk_boot.mean(axis=0)
    frame = pd.DataFrame({
        "risk": risk_mean,
        "time": y["time"],
        "event": y["event"],
    })
    frame["bin"] = pd.qcut(frame["risk"], q=n_bins, duplicates="drop")

    rows = []
    for bin_number, (_, group) in enumerate(frame.groupby("bin", observed=True), start=1):
        indices = group.index.to_numpy()
        km = KaplanMeierFitter().fit(group["time"], event_observed=group["event"])
        observed = float(1.0 - km.predict(horizon))
        bin_boot = risk_boot[:, indices].mean(axis=1)
        rows.append({
            "horizon_months": horizon,
            "risk_decile": bin_number,
            "n": len(group),
            "predicted": float(risk_mean[indices].mean()),
            "predicted_lower": float(np.percentile(bin_boot, 2.5)),
            "predicted_upper": float(np.percentile(bin_boot, 97.5)),
            "observed_km": observed,
        })
    return pd.DataFrame(rows)


def _weighted_logistic_calibration(y_binary, risk, weights):
    """Fit weighted y ~ intercept + slope*logit(risk) by Newton iterations."""
    eps = 1e-8
    clipped = np.clip(risk, eps, 1.0 - eps)
    z = np.log(clipped / (1.0 - clipped))
    design = np.column_stack((np.ones_like(z), z))
    beta = np.zeros(2, dtype=float)
    for _ in range(100):
        eta = np.clip(design @ beta, -30, 30)
        fitted = 1.0 / (1.0 + np.exp(-eta))
        variance = np.maximum(fitted * (1.0 - fitted), eps)
        gradient = design.T @ (weights * (y_binary - fitted))
        hessian = design.T @ ((weights * variance)[:, None] * design)
        step = np.linalg.solve(hessian + np.eye(2) * 1e-10, gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(beta[0]), float(beta[1])


def horizon_metrics(risk_single, y, horizon, ipcw=None):
    """Estimate IPCW calibration-in-large, slope, O:E, and weighted risks."""
    targets, all_weights = ipcw if ipcw is not None else compute_ipcw_targets_and_weights(y, [horizon])
    outcome = 1.0 - targets[:, 0]
    weights = all_weights[:, 0]
    risk = np.clip(risk_single, 1e-8, 1.0 - 1e-8)
    valid = weights > 0
    outcome, weights, risk = outcome[valid], weights[valid], risk[valid]
    if not len(outcome) or outcome.sum() == 0 or outcome.sum() == len(outcome):
        return {
            "calibration_in_large": np.nan,
            "calibration_slope": np.nan,
            "observed_risk_ipcw": np.nan,
            "predicted_risk_ipcw": np.nan,
            "observed_expected_ratio": np.nan,
        }

    intercept, slope = _weighted_logistic_calibration(outcome, risk, weights)
    observed = float(np.average(outcome, weights=weights))
    predicted = float(np.average(risk, weights=weights))

    # Calibration-in-the-large (offset logit risk, fit intercept)
    offset = np.log(risk / (1.0 - risk))
    cil = 0.0
    for _ in range(100):
        fitted = 1.0 / (1.0 + np.exp(-np.clip(offset + cil, -30, 30)))
        gradient = np.sum(weights * (outcome - fitted))
        hessian = np.sum(weights * fitted * (1.0 - fitted))
        step = gradient / max(hessian, 1e-12)
        cil += step
        if abs(step) < 1e-8:
            break
    oe = float(np.sum(weights * outcome) / np.sum(weights * risk))
    return {
        "calibration_in_large": float(cil),
        "calibration_slope": slope,
        "observed_risk_ipcw": observed,
        "predicted_risk_ipcw": predicted,
        "observed_expected_ratio": oe,
    }


def _model_features(model, X, setting):
    X_model = X.copy()
    if setting == "tra75":
        X_model = X_model.loc[:, ~X_model.columns.str.startswith("insurer")]
    expected = getattr(getattr(model, "base_model", None), "feature_names_in_", None)
    if expected is not None:
        X_model = X_model.reindex(columns=list(expected), fill_value=0)
    return X_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratio", type=int, default=1, help="Case-control ratio (default: 1)")
    parser.add_argument("--n-rep", type=int, default=100, help="Number of repetitions to evaluate (default: 100)")
    parser.add_argument("--settings", nargs="+", choices=SETTINGS, default=["tra75"],
                        help="Calibration strategies to evaluate (default: tra75)")
    default_root = os.environ.get("DESC_DATA_ROOT", "./data")
    parser.add_argument("--root", default=default_root,
                        help="Root directory containing models and evaluation data (default: ./data or $DESC_DATA_ROOT)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: same as ratio directory)")
    parser.add_argument("--n-bins", type=int, default=10, help="Number of risk deciles (default: 10)")
    args = parser.parse_args()

    ratio_dir = os.path.join(args.root, f"ratio_{args.ratio}")
    output_dir = args.output_dir or ratio_dir
    os.makedirs(output_dir, exist_ok=True)

    eval_data_path = os.path.join(args.root, "calib_eval_data_v2.pkl")
    print(f"Loading evaluation data from: {eval_data_path}")
    with open(eval_data_path, "rb") as f:
        _, _, X_eval, y_eval = pickle.load(f)

    age_mask = X_eval["age"].to_numpy() >= 75
    X_eval = X_eval.loc[age_mask]
    y_eval = y_eval[age_mask]
    n_samples = len(y_eval)
    n_events = int(y_eval["event"].sum())
    print(f"Evaluation records (age >= 75): {n_samples:,}; events: {n_events:,}")

    # Precompute IPCW targets and weights for targeted horizons
    ipcw_by_horizon = {
        h: compute_ipcw_targets_and_weights(y_eval, [h])
        for h in HORIZONS
    }

    summary_rows = []

    for setting in args.settings:
        print(f"\n==========================================")
        print(f"Evaluating strategy: {setting} (Ratio: {args.ratio})")
        print(f"==========================================")

        predictions = {"RSF": [], "RSF_ISR": []}
        t0_total = time.time()

        for rep in range(args.n_rep):
            model_path = os.path.join(ratio_dir, f"Model_{rep}_x1_{setting}.pkl")
            if not os.path.exists(model_path):
                print(f"  [Rep {rep:02d}] Model not found at {model_path}, skipping.")
                continue

            t0_rep = time.time()
            model = load_rsf_model(model_path, setting)
            X_model = _model_features(model, X_eval, setting)

            # Fast targeted horizon prediction (bypassing Dykstra)
            res = predict_rsf_and_isr_horizons(model, X_model, HORIZONS)
            for name in predictions:
                predictions[name].append(res[name])

            rep_elapsed = time.time() - t0_rep
            if (rep + 1) % 10 == 0 or rep == 0 or (rep + 1) == args.n_rep:
                print(f"  Processed rep {rep + 1}/{args.n_rep} ({rep_elapsed:.2f} s/rep)")

        actual_n_rep = len(predictions["RSF"])
        if actual_n_rep == 0:
            print(f"Error: No models found for setting {setting} in {ratio_dir}.")
            continue

        total_elapsed = time.time() - t0_total
        print(f"Completed {actual_n_rep} reps in {total_elapsed:.1f} s ({total_elapsed / actual_n_rep:.2f} s/rep).")

        # Stack into shape: (actual_n_rep, n_samples, len(HORIZONS))
        predictions = {name: np.stack(values, axis=0) for name, values in predictions.items()}

        # ----------------------------------------------------
        # Plot calibration curves & compute horizon metrics
        # ----------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        curve_rows = []

        for ax_idx, (ax, horizon) in enumerate(zip(axes, HORIZONS)):
            h_idx = HORIZONS.index(horizon)
            years = horizon // 12
            all_curves = {}

            # First compute metrics to display slope/CIL in legend
            model_summary = {}
            for name in ("RSF", "RSF_ISR"):
                # risk_boot shape: (actual_n_rep, n_samples)
                risk_boot = 1.0 - predictions[name][:, :, h_idx]

                slopes, cils = [], []
                for rep_idx in range(actual_n_rep):
                    metric = horizon_metrics(risk_boot[rep_idx], y_eval, horizon, ipcw_by_horizon[horizon])
                    summary_rows.append({
                        "rep": rep_idx,
                        "training_calibration_strategy": setting,
                        "model": name,
                        "horizon_months": horizon,
                        **metric,
                    })
                    slopes.append(metric["calibration_slope"])
                    cils.append(metric["calibration_in_large"])

                model_summary[name] = {
                    "mean_slope": np.nanmean(slopes),
                    "mean_cil": np.nanmean(cils),
                }

                # Decile curve
                curve = calibration_curve(risk_boot, y_eval, horizon, args.n_bins)
                curve.insert(0, "model", name)
                curve.insert(0, "training_calibration_strategy", setting)
                curve_rows.extend(curve.to_dict("records"))
                all_curves[name] = curve

            # Plot both models
            color_map = {"RSF": "#195C8C", "RSF_ISR": "#a22020"}
            marker_map = {"RSF": "s", "RSF_ISR": "o"}

            for name in ("RSF", "RSF_ISR"):
                curve = all_curves[name]
                s = model_summary[name]
                label_str = f"{name} (Slope: {s['mean_slope']:.2f}, CIL: {s['mean_cil']:+.2f})"
                c = color_map[name]
                m = marker_map[name]

                ax.plot(curve["predicted"], curve["observed_km"], f"{m}-", color=c,
                        label=label_str, linewidth=1.8, markersize=5)
                ax.errorbar(
                    curve["predicted"], curve["observed_km"],
                    xerr=[curve["predicted"] - curve["predicted_lower"],
                          curve["predicted_upper"] - curve["predicted"]],
                    fmt="none", ecolor=c, alpha=0.5, capsize=2,
                )

            # Determine axis limits for this horizon
            max_p = max(c["predicted"].max() for c in all_curves.values())
            max_o = max(c["observed_km"].max() for c in all_curves.values())
            axis_max = min(1.0, max(max_p, max_o) * 1.15)
            axis_max = max(axis_max, 0.02)

            ax.plot([0, axis_max], [0, axis_max], "--", color="black", alpha=0.6, label="Ideal")
            ax.set_xlim(0, axis_max)
            ax.set_ylim(0, axis_max)
            ax.set_title(f"{years}-Year Horizon ({horizon} Months)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Predicted cumulative risk", fontsize=11)
            if ax_idx == 0:
                ax.set_ylabel("Observed risk (Kaplan-Meier)", fontsize=11)
            ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
            ax.grid(True, linestyle=":", alpha=0.5)

        fig.suptitle(f"Hip Fracture Risk Calibration in Older Adults (Aged >=75) [{setting}]",
                     fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()

        # Save plot in both PDF and PNG
        pdf_path = os.path.join(output_dir, f"Calibration_RSFI_ratio{args.ratio}_{setting}.pdf")
        png_path = os.path.join(output_dir, f"Calibration_RSFI_ratio{args.ratio}_{setting}.png")
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved calibration plots:\n  -> {pdf_path}\n  -> {png_path}")

        # Save deciles CSV
        deciles_path = os.path.join(output_dir, f"Calibration_deciles_ratio{args.ratio}_{setting}.csv")
        pd.DataFrame(curve_rows).to_csv(deciles_path, index=False)
        print(f"Saved deciles CSV: {deciles_path}")

    # Save complete metrics CSV
    metrics_path = os.path.join(output_dir, f"Calibration_metrics_ratio{args.ratio}.csv")
    df_metrics = pd.DataFrame(summary_rows)
    df_metrics.to_csv(metrics_path, index=False)
    print(f"\nSaved raw bootstrap metrics CSV: {metrics_path}")

    # Generate summary table (Mean, 95% CI)
    summary_table_rows = []
    for (setting, model_name, horizon), group in df_metrics.groupby(
        ["training_calibration_strategy", "model", "horizon_months"], observed=True
    ):
        years = horizon // 12
        row = {
            "Strategy": setting,
            "Model": model_name,
            "Horizon": f"{years}-year ({horizon}m)",
            "Slope_mean": group["calibration_slope"].mean(),
            "Slope_2.5%": group["calibration_slope"].quantile(0.025),
            "Slope_97.5%": group["calibration_slope"].quantile(0.975),
            "CIL_mean": group["calibration_in_large"].mean(),
            "CIL_2.5%": group["calibration_in_large"].quantile(0.025),
            "CIL_97.5%": group["calibration_in_large"].quantile(0.975),
            "OE_mean": group["observed_expected_ratio"].mean(),
            "OE_2.5%": group["observed_expected_ratio"].quantile(0.025),
            "OE_97.5%": group["observed_expected_ratio"].quantile(0.975),
        }
        summary_table_rows.append(row)

    summary_df = pd.DataFrame(summary_table_rows)
    summary_csv_path = os.path.join(output_dir, f"Calibration_summary_table_ratio{args.ratio}.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved summary table CSV: {summary_csv_path}")

    print("\nSummary Metrics Table:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

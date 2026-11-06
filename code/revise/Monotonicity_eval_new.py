"""Quantify survival-curve monotonicity violations from saved RSF models.

Optimized for Reviewer revision:
- Bypasses unnecessary Dykstra iterations.
- Evaluates full 71-month survival curves for RSF and RSF-ISR in seconds per rep.
- Quantifies temporal monotonicity violations (curve level, step level, milestone horizons).
- Outputs raw repetition metrics and aggregated summary tables (Mean, 95% CI).

Usage:
  python Monotonicity_eval.py --ratio 1 --settings tra75 --n-rep 100
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

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
MODEL_NAMES = ("RSF", "RSF_ISR")
MILESTONES = (12, 36, 60)


def predict_rsf_and_isr_full(model, X):
    """Predict survival curves across all time points for RSF and RSF-ISR.

    Bypasses Dykstra projection entirely, evaluating RSF survival curves
    and applying fitted isotonic transformations in a single vectorized pass (~1-2 sec/rep).
    """
    time_grid = np.asarray(getattr(model, "time_grid_", np.arange(1, 72)))
    surv_funcs = model.base_model.predict_survival_function(X)

    # Evaluate step functions on the full time grid: shape (n_samples, len(time_grid))
    S_rsf = np.asarray([fn(time_grid) for fn in surv_funcs], dtype=float)

    results = {"RSF": S_rsf}

    # Apply 1D isotonic transform at each monthly horizon
    if getattr(model, "use_isotonic", True) and hasattr(model, "iso_models_"):
        S_iso = np.empty_like(S_rsf)
        for t_idx in range(len(time_grid)):
            S_iso[:, t_idx] = model.iso_models_[t_idx].transform(S_rsf[:, t_idx])
        results["RSF_ISR"] = S_iso

    return results


def monotonicity_summary(survival, tol=1e-10):
    """Summarize per-curve, per-step, and clinical-horizon violations."""
    survival = np.asarray(survival, dtype=float)
    if survival.ndim != 2 or survival.shape[1] < 2:
        raise ValueError("survival must have shape (n_samples, n_times>=2)")
    # Survival should be non-increasing: increase = S(t+1) - S(t) > 0 is a violation
    increase = np.diff(survival, axis=1)
    violating = increase > tol
    max_rise_by_curve = np.maximum(increase, 0).max(axis=1)
    result = {
        "n_curves": int(len(survival)),
        "curves_with_any_violation": int(violating.any(axis=1).sum()),
        "curve_violation_rate": float(violating.any(axis=1).mean()),
        "violating_adjacent_steps": int(violating.sum()),
        "adjacent_step_violation_rate": float(violating.mean()),
        "mean_max_increase": float(max_rise_by_curve.mean()),
        "maximum_increase": float(max(0.0, increase.max())),
    }
    # Column h-2 is the interval ending at month h on a 1-based monthly grid.
    for horizon in MILESTONES:
        if horizon - 2 < increase.shape[1]:
            step = increase[:, horizon - 2]
            result[f"month_{horizon}_increase_n"] = int((step > tol).sum())
            result[f"month_{horizon}_increase_rate"] = float((step > tol).mean())
            result[f"month_{horizon}_max_increase"] = float(max(0.0, step.max()))
    return result


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
    parser.add_argument("--n-rep", type=int, default=100, help="Number of repetitions (default: 100)")
    parser.add_argument("--settings", nargs="+", choices=SETTINGS, default=["tra75"],
                        help="Calibration strategies (default: tra75)")
    default_root = os.environ.get("DESC_DATA_ROOT", "./data")
    parser.add_argument("--root", default=default_root,
                        help="Root directory containing models and data (default: ./data or $DESC_DATA_ROOT)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
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

    rows = []
    for setting in args.settings:
        print(f"\n==========================================")
        print(f"Evaluating Monotonicity: {setting} (Ratio: {args.ratio})")
        print(f"==========================================")
        t0_setting = time.time()

        for rep in range(args.n_rep):
            model_path = os.path.join(ratio_dir, f"Model_{rep}_x1_{setting}.pkl")
            if not os.path.exists(model_path):
                print(f"  [Rep {rep:02d}] Model not found, skipping.")
                continue

            t0_rep = time.time()
            model = load_rsf_model(model_path, setting)
            X_model = _model_features(model, X_eval, setting)

            # Fast prediction without Dykstra
            result = predict_rsf_and_isr_full(model, X_model)

            for model_name in MODEL_NAMES:
                if model_name not in result:
                    continue
                rows.append({
                    "rep": rep,
                    "training_calibration_strategy": setting,
                    "model": model_name,
                    "n_events_eval": n_events,
                    **monotonicity_summary(result[model_name]),
                })

            rep_elapsed = time.time() - t0_rep
            if (rep + 1) % 10 == 0 or rep == 0 or (rep + 1) == args.n_rep:
                print(f"  Processed rep {rep + 1}/{args.n_rep} ({rep_elapsed:.2f} s/rep)")

        total_elapsed = time.time() - t0_setting
        print(f"Completed {setting} in {total_elapsed:.1f} s.")

    output_path = os.path.join(output_dir, f"Monotonicity_ratio{args.ratio}.csv")
    df_raw = pd.DataFrame(rows)
    df_raw.to_csv(output_path, index=False)
    print(f"\nSaved raw monotonicity metrics: {output_path}")

    # Generate aggregated summary table
    summary_rows = []
    for (setting, model_name), group in df_raw.groupby(
        ["training_calibration_strategy", "model"], observed=True
    ):
        summary_rows.append({
            "Strategy": setting,
            "Model": model_name,
            "Curve_Violation_Rate_mean": group["curve_violation_rate"].mean(),
            "Curve_Violation_Rate_2.5%": group["curve_violation_rate"].quantile(0.025),
            "Curve_Violation_Rate_97.5%": group["curve_violation_rate"].quantile(0.975),
            "Step_Violation_Rate_mean": group["adjacent_step_violation_rate"].mean(),
            "Mean_Max_Increase_mean": group["mean_max_increase"].mean(),
            "Maximum_Increase_max": group["maximum_increase"].max(),
            "Month12_Viol_Rate_mean": group["month_12_increase_rate"].mean(),
            "Month36_Viol_Rate_mean": group["month_36_increase_rate"].mean(),
            "Month60_Viol_Rate_mean": group["month_60_increase_rate"].mean(),
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, f"Monotonicity_summary_ratio{args.ratio}.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved monotonicity summary table: {summary_path}")

    print("\nMonotonicity Summary Table:")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()

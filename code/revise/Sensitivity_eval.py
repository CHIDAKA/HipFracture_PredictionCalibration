"""Evaluate first/latest eligible checkup records using saved RSF models.

Run this in the remote analysis environment, for example:
    python Sensitivity_eval.py --ratio 1

No model fitting is performed. Outputs are written separately for the two
existing training/calibration strategies.
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
# Dynamically discover project import directories without hardcoded user paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for candidate in (
    SCRIPT_DIR,
    os.path.abspath("."),
    os.path.join(SCRIPT_DIR, ".."),
    os.path.join(SCRIPT_DIR, "../test"),
    os.path.join(SCRIPT_DIR, "../.."),
):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.append(candidate)
from rsf_pickle_compat import load_rsf_model  # noqa: E402
from src.RSF_ISR_diff2 import (  # noqa: E402
    integrated_brier_skill_score,
    load_csv_by_ids,
    risk_from_surv,
    Xy_from_df,
)


NOT_USE = [
    "exam_month", "observable_start_ym", "observable_end_ym", "onset",
    "exam_ymd", "onset_ym", "end_ym", "start_ym", "birth_ym",
    "kenshin_times_before_onset", "observation_start_age",
    "months_until_first_kenshin", "duration_disease", "distance", "weights",
    "subclass", "observation_months",
]
SETTINGS = ("over75calib", "tra75")
MODEL_NAMES = ("RSF", "RSF_ISR", "RSF_dykstra", "RSF_ISR_dykstra")


def select_one_record_per_person(df, position, id_col="kojin_id", date_col="exam_ymd"):
    """Select the first or last eligible checkup by its actual date."""
    if position not in {"first", "last"}:
        raise ValueError("position must be 'first' or 'last'")
    missing = {id_col, date_col}.difference(df.columns)
    if missing:
        raise ValueError(f"Required columns are missing: {sorted(missing)}")
    selected = df.copy()
    selected[date_col] = pd.to_datetime(selected[date_col], errors="raise")
    selected = selected.sort_values([id_col, date_col], kind="mergesort")
    return selected.drop_duplicates(id_col, keep=position)


def model_features(model, X, setting):
    """Match feature names and ordering to the fitted bootstrap model."""
    X_model = X.copy()
    if setting == "tra75":
        X_model = X_model.loc[:, ~X_model.columns.str.startswith("insurer")]
    expected = getattr(getattr(model, "base_model", None), "feature_names_in_", None)
    if expected is not None:
        X_model = X_model.reindex(columns=list(expected), fill_value=0)
    return X_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratio", type=int, default=1)
    parser.add_argument("--n-rep", type=int, default=100)
    default_root = os.environ.get("DESC_DATA_ROOT", "./data")
    parser.add_argument("--root", default=default_root,
                        help="Data root directory (default: ./data or $DESC_DATA_ROOT)")
    parser.add_argument("--disease-csv", default=None,
                        help="Path to disease.csv (default: <root>/disease.csv)")
    args = parser.parse_args()

    ratio_dir = os.path.join(args.root, f"ratio_{args.ratio}")
    with open(os.path.join(args.root, "calib_eval_data_v2.pkl"), "rb") as f:
        X_calib, y_calib, _, _ = pickle.load(f)
    calib_mask = X_calib["age"].to_numpy() >= 75
    X_calib = X_calib.loc[calib_mask]
    y_calib = y_calib[calib_mask]

    ids = pd.read_csv(os.path.join(args.root, "used_IDs.csv"))
    eval_ids = ids.loc[ids["split"] == "eval", "kojin_id"].to_numpy()
    disease_path = args.disease_csv or os.path.join(args.root, "disease.csv")
    if not os.path.exists(disease_path):
        candidate_alt = os.path.join(os.path.dirname(args.root), "disease.csv")
        if os.path.exists(candidate_alt):
            disease_path = candidate_alt
    eval_records = load_csv_by_ids(disease_path, eval_ids)
    # Match the manuscript's >=75 evaluation population before selecting one
    # record per person; this gives first/latest analyses the same ID set.
    eval_records = eval_records.loc[eval_records["age"] >= 75]
    records = {
        "first": select_one_record_per_person(eval_records, "first"),
        "last": select_one_record_per_person(eval_records, "last"),
    }
    eval_data = {
        position: Xy_from_df(frame, NOT_USE)
        for position, frame in records.items()
    }
    counts = {position: len(y) for position, (_, y) in eval_data.items()}
    ids_first = set(records["first"]["kojin_id"])
    ids_last = set(records["last"]["kojin_id"])
    if ids_first != ids_last:
        raise ValueError("First/latest analyses do not contain the same eligible individuals")
    print(f"Eligible evaluation records (age >=75): {counts}")

    times = np.arange(1, 72)
    for setting in args.settings:
        metrics_path = os.path.join(ratio_dir, f"metrics_sensitivity_{setting}.csv")
        rows = []
        for rep in range(args.n_rep):
            model_path = os.path.join(ratio_dir, f"Model_{rep}_x1_{setting}.pkl")
            model = load_rsf_model(model_path, setting)
            for position, (X, y_eval) in eval_data.items():
                X_model = model_features(model, X, setting)
                predictions = model.predict_all(X_model)
                for model_name in MODEL_NAMES:
                    if model_name not in predictions:
                        continue
                    survival = predictions[model_name]
                    ibss = integrated_brier_skill_score(survival, y_calib, y_eval, times)
                    cindex = concordance_index_censored(
                        y_eval["event"], y_eval["time"], risk_from_surv(survival, times)
                    )[0]
                    rows.append({
                        "rep": rep,
                        "training_calibration_strategy": setting,
                        "record_position": position,
                        "n_eval": len(y_eval),
                        "n_events": int(y_eval["event"].sum()),
                        "model": model_name,
                        "cindex": cindex,
                        "IBSS": ibss,
                    })
        pd.DataFrame(rows).to_csv(metrics_path, index=False)
        print(f"Saved {metrics_path}")


if __name__ == "__main__":
    main()

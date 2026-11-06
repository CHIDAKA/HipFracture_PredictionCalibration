"""Fit age/sex and clinical-only Cox-ISR benchmarks for the >=75 cohort.

The script follows COX_ISR.py: ID-level case-control bootstrap, 50,000-row
training samples, age >=75 training/calibration/evaluation, IPCW isotonic
calibration, C-index and IBSS. Both models use the same bootstrap sample on
each repetition. The clinical model contains age, sex, and all non-ICD
predictors (including health-checkup variables); diagnosis-code features are
excluded. No RSF models are refit.

Example (remote server):
    python Cox_Baselines.py --ratio 1
"""

import argparse
import os
import pickle
import re
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
from src.RSF_ISR_diff2 import (  # noqa: E402
    integrated_brier_skill_score,
    risk_from_surv,
)
from src.RSF_ISR_diff3_75 import (  # noqa: E402
    bootstrap_train_ids,
    load_csv_by_ids,
    trim_rows,
    Xy_from_df,
)
from COX_ISR import Cox_Calibrated2  # noqa: E402


NOT_USE = [
    "exam_month", "observable_start_ym", "observable_end_ym", "onset",
    "exam_ymd", "onset_ym", "end_ym", "start_ym", "birth_ym",
    "kenshin_times_before_onset", "observation_start_age",
    "months_until_first_kenshin", "duration_disease", "distance", "weights",
    "subclass", "observation_months",
]
ICD3_PATTERN = re.compile(r"^[A-Z][0-9]{2}$")
EXPECTED_ICD_FEATURES = 395


def select_baseline_feature_sets(columns, expected_icd_count=EXPECTED_ICD_FEATURES):
    """Return age/sex and age/sex+clinical features, excluding ICD-10 3-char codes."""
    columns = list(columns)
    required = {"age", "sex_code"}
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"Required demographic features are missing: {sorted(missing)}")

    diagnosis = [name for name in columns if ICD3_PATTERN.fullmatch(str(name))]
    if len(diagnosis) != expected_icd_count:
        raise ValueError(
            f"Detected {len(diagnosis)} three-character ICD features; expected "
            f"{expected_icd_count}. Review ICD feature naming before fitting."
        )
    age_sex = ["age", "sex_code"]
    clinical = [name for name in columns if name not in set(diagnosis)]
    return {"age_sex": age_sex, "clinical": clinical}, diagnosis


def restrict_age75(X, y):
    mask = X["age"].to_numpy() >= 75
    return X.loc[mask], y[mask]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratio", type=int, default=1)
    parser.add_argument("--n-rep", type=int, default=100)
    parser.add_argument("--target-n-rows", type=int, default=500000)
    parser.add_argument("--train-size", type=int, default=50000)
    default_data_root = os.environ.get("DESC_DATA_ROOT", "./data")
    default_manuscript_dir = os.environ.get("DESC_MANUSCRIPT_DIR", os.path.join(default_data_root, "Manuscript") if os.path.isdir(os.path.join(default_data_root, "Manuscript")) else default_data_root)
    default_summary = os.environ.get("DESC_SUMMARY_PATH", os.path.join(default_data_root, "patient_matching_data.csv"))
    parser.add_argument("--data-root", default=default_data_root, help="Data root directory (default: ./data or $DESC_DATA_ROOT)")
    parser.add_argument("--manuscript-dir", default=default_manuscript_dir, help="Manuscript data directory")
    parser.add_argument(
        "--summary-path",
        default=default_summary,
        help="Patient-level matching summary path (default: <data-root>/patient_matching_data.csv)",
    )
    args = parser.parse_args()

    ratio_dir = os.path.join(args.manuscript_dir, f"ratio_{args.ratio}")
    output_dir = os.path.join(ratio_dir, "cox_baselines_tra75")
    os.makedirs(output_dir, exist_ok=True)

    summary_df = pd.read_csv(args.summary_path)
    case_df = pd.read_csv(os.path.join(args.data_root, "pid_S720-S722.csv"))
    summary_df["case"] = summary_df["kojin_id"].isin(case_df["kojin_id"])
    split = pd.read_csv(os.path.join(args.manuscript_dir, "used_IDs.csv"))
    train_ids = split.loc[split["split"] == "train", "kojin_id"].to_numpy()
    del case_df, split

    with open(os.path.join(args.manuscript_dir, "calib_eval_data_v2.pkl"), "rb") as f:
        X_calib, y_calib, X_eval, y_eval = pickle.load(f)
    X_calib, y_calib = restrict_age75(X_calib, y_calib)
    X_eval, y_eval = restrict_age75(X_eval, y_eval)
    for X in (X_calib, X_eval):
        X.drop(columns=[c for c in X if str(c).startswith("insurer")], inplace=True, errors="ignore")

    feature_sets, diagnosis_features = select_baseline_feature_sets(
        X_calib.columns, args.expected_icd_features
    )
    missing_eval = {
        feature_set: sorted(set(columns).difference(X_eval.columns))
        for feature_set, columns in feature_sets.items()
        if set(columns).difference(X_eval.columns)
    }
    if missing_eval:
        raise ValueError(f"Calibration/evaluation feature mismatch: {missing_eval}")

    data_path = os.path.join(args.manuscript_dir, "disease.csv")
    times = np.arange(1, 72)
    metrics_path = os.path.join(output_dir, "metrics_Cox_baselines_tra75.csv")
    feature_manifest_path = os.path.join(output_dir, "feature_manifest.txt")
    rows = []

    with open(feature_manifest_path, "w", encoding="utf-8") as f:
        f.write(f"Three-character ICD diagnosis features excluded ({len(diagnosis_features)}):\n")
        f.write(", ".join(diagnosis_features) + "\n\n")
        for feature_set, features in feature_sets.items():
            f.write(f"{feature_set} ({len(features)} predictors):\n")
            f.write(", ".join(features) + "\n\n")
    print(f"ICD features detected: {len(diagnosis_features)}")
    print(f"Age/sex predictors: {feature_sets['age_sex']}")
    print(f"Clinical predictors: {len(feature_sets['clinical'])} total, ICD codes excluded")

    for rep in range(args.n_rep):
        print(f"Bootstrap {rep + 1}/{args.n_rep}")
        rng = np.random.default_rng(rep)
        sampled_ids = bootstrap_train_ids(
            train_ids,
            summary_df[["kojin_id", "case", "kenshin_times"]],
            ratio=args.ratio,
            target_n_rows=args.target_n_rows,
            rng=rng,
        )
        train_df = load_csv_by_ids(data_path, sampled_ids)
        train_df = train_df.loc[train_df["age"] >= 75]
        train_df = train_df.loc[:, ~train_df.columns.str.startswith("insurer")]
        train_df = trim_rows(train_df, args.train_size, rng)
        X_train_full, y_train = Xy_from_df(train_df, NOT_USE)
        train_feature_sets, train_diagnosis_features = select_baseline_feature_sets(
            X_train_full.columns, args.expected_icd_features
        )
        if train_diagnosis_features != diagnosis_features or train_feature_sets != feature_sets:
            raise ValueError("Training and calibration feature definitions differ")

        for feature_set, columns in feature_sets.items():
            absent_train = set(columns).difference(X_train_full.columns)
            absent_calib = set(columns).difference(X_calib.columns)
            absent_eval = set(columns).difference(X_eval.columns)
            if absent_train or absent_calib or absent_eval:
                raise ValueError(
                    f"Feature mismatch for {feature_set}: train={sorted(absent_train)[:5]}, "
                    f"calibration={sorted(absent_calib)[:5]}, evaluation={sorted(absent_eval)[:5]}"
                )

            model = Cox_Calibrated2()
            model.fit(
                X_train_full.loc[:, columns], y_train,
                X_calib.loc[:, columns], y_calib,
                times,
            )
            model_path = os.path.join(output_dir, f"Model_{rep}_{feature_set}_tra75.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            predictions = model.predict_all(X_eval.loc[:, columns])
            for prediction_name, survival in predictions.items():
                ibss = integrated_brier_skill_score(survival, y_calib, y_eval, times)
                cindex = concordance_index_censored(
                    y_eval["event"], y_eval["time"], risk_from_surv(survival, times)
                )[0]
                rows.append({
                    "rep": rep,
                    "ratio_control_case": args.ratio,
                    "train_size": len(y_train),
                    "feature_set": feature_set,
                    "model": prediction_name,
                    "n_calibration": len(y_calib),
                    "n_evaluation": len(y_eval),
                    "n_events_evaluation": int(y_eval["event"].sum()),
                    "cindex": cindex,
                    "IBSS": ibss,
                    "model_path": model_path,
                })

        pd.DataFrame(rows).to_csv(metrics_path, index=False)
        print(f"Saved interim metrics through repetition {rep + 1}: {metrics_path}")


if __name__ == "__main__":
    main()

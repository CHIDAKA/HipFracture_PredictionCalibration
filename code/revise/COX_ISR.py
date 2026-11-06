from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
import argparse
from sksurv.linear_model import CoxnetSurvivalAnalysis
import os
import pandas as pd
import numpy as np
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
from src.RSF_ISR_diff2 import RSF_Calibrated2, logit, sigmoid, compute_ipcw_targets_and_weights, recalibrate_model_full_ipcw, integrated_brier_skill_score, risk_from_surv
from src.RSF_ISR_diff3_75 import bootstrap_train_ids, trim_rows, Xy_from_df, load_csv_by_ids

from sksurv.functions import StepFunction
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import (concordance_index_censored, 
                            integrated_brier_score, 
                            cumulative_dynamic_auc)
from sklearn.pipeline import make_pipeline

import pickle
from sklearn.impute import SimpleImputer

class Cox_Calibrated2(BaseEstimator):

    def __init__(
        self,
        use_isotonic=True,
        l1_ratio=0.5,
        alpha_min_ratio=0.01
    ):

        self.use_isotonic = use_isotonic

        self.base_model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            CoxnetSurvivalAnalysis(
            l1_ratio=l1_ratio,
            alpha_min_ratio=alpha_min_ratio,
            fit_baseline_model=True
            )
        )

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        time_grid
    ):

        self.time_grid_ = time_grid

        self.base_model.fit(
            X_train,
            y_train
        )

        surv_funcs = (
            self.base_model
            .predict_survival_function(
                X_val
            )
        )

        S_val = np.asarray([
            [fn(t) for t in time_grid]
            for fn in surv_funcs
        ])

        y_surv_matrix, weights = (
            compute_ipcw_targets_and_weights(
                y_val,
                time_grid
            )
        )

        self.iso_models_ = []

        for t_idx in range(
            len(time_grid)
        ):

            ir = IsotonicRegression(
                out_of_bounds="clip"
            )

            ir.fit(
                S_val[:, t_idx],
                y_surv_matrix[:, t_idx],
                sample_weight=weights[:, t_idx]
            )

            self.iso_models_.append(ir)

        return self

    def predict_all(self, X):

        surv_funcs = (
            self.base_model
            .predict_survival_function(X)
        )

        S_cox = np.asarray([
            [fn(t) for t in self.time_grid_]
            for fn in surv_funcs
        ])

        results = {
            "Cox": S_cox
        }

        if self.use_isotonic:

            S_iso = np.copy(S_cox)

            for t_idx in range(
                len(self.time_grid_)
            ):

                S_iso[:, t_idx] = (
                    self.iso_models_[t_idx]
                    .transform(
                        S_iso[:, t_idx]
                    )
                )

            results["Cox_ISR"] = S_iso

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RSF with different case-control ratios")
    parser.add_argument("--ratio", type=int, default=5, help="Case-control ratio (default: 5)")
    default_root = os.environ.get("DESC_DATA_ROOT", "./data")
    parser.add_argument("--root", default=default_root, help="Data root directory (default: ./data or $DESC_DATA_ROOT)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()
    ratio = args.ratio
    output_dir = args.output_dir or os.path.join(args.root, f"ratio_{ratio}")
    os.makedirs(output_dir, exist_ok=True)
    

    n_rep = 100
    TARGET_N = 500000 #目的より大きいサイズ
    True_TARGET_N = 50000

    not_use = [
        "exam_month","observable_start_ym","observable_end_ym","onset",
        "exam_ymd","onset_ym","end_ym","start_ym","birth_ym",
        "kenshin_times_before_onset","observation_start_age",
        "months_until_first_kenshin","duration_disease", 'distance', 'weights', 'subclass', 'observation_months'
    ]


# data load
    summary_path = os.path.join(args.root, "patient_matching_data.csv")
    if not os.path.exists(summary_path):
        summary_path = os.path.join(os.path.dirname(args.root), "patient_matching_data.csv")
    summary_df = pd.read_csv(summary_path)

    case_path = os.path.join(args.root, "pid_S720-S722.csv")
    if not os.path.exists(case_path):
        case_path = os.path.join(os.path.dirname(args.root), "pid_S720-S722.csv")
    case_df = pd.read_csv(case_path)
    summary_df["case"] = summary_df["kojin_id"].isin(case_df.kojin_id)

    used_ids_path = os.path.join(args.root, "used_IDs.csv")
    df = pd.read_csv(used_ids_path)
    train_ids = df[df["split"] == "train"]["kojin_id"].values
    calib_ids = df[df["split"] == "calib"]["kojin_id"].values
    eval_ids = df[df["split"] == "eval"]["kojin_id"].values
    del case_df,df

    filepath = os.path.join(args.root, "disease.csv")
    with open(os.path.join(args.root, "calib_eval_data_v2.pkl"), "rb") as f:
            X_calib, y_calib, X_eval, y_eval = pickle.load(f) 


    #########最初から絞ると、評価側のIDが変わってしまう　ー＞　比較しにくい

    mask_ov75 = X_calib["age"] >= 75
    X_calib = X_calib.loc[mask_ov75,~X_calib.columns.str.startswith("insurer")]
    y_calib = y_calib[mask_ov75]

    mask_ov75 = X_eval["age"] >= 75
    X_eval = X_eval.loc[mask_ov75,~X_eval.columns.str.startswith("insurer")]
    y_eval = y_eval[mask_ov75]


    for i in range(n_rep):
        print(i)
        rng_i = np.random.default_rng(i)

        # ✅ trainのみbootstrap + ratio制御
        train_ids_bs = bootstrap_train_ids(
            train_ids,
            summary_df[["kojin_id", "case", "kenshin_times"]],
            ratio=ratio,
            target_n_rows=TARGET_N,
            rng=rng_i
        )
        
        # ✅ trainのみロード
        train_df = load_csv_by_ids(filepath, train_ids_bs)

    #########最初から絞ると、評価側のIDが変わってしまう　ー＞　比較しにくい
        mask_ov75 = train_df["age"] >= 75
        train_df = train_df.loc[mask_ov75,~train_df.columns.str.startswith("insurer")]
        

        # ✅ size調整（trainのみ）
        # ここだとpreprocessで減るtrain_df = trim_rows(train_df, 50000, rng_i) # サイズを同じにしたい
        train_df = trim_rows(train_df, True_TARGET_N, rng_i)
        X_train, y_train = Xy_from_df(train_df, not_use)
        # なぜかここでは適用されなかった ~train_df.columns.str.startswith("insurer")]
        print("Train loaded:", len(y_train))
        

        time_grid = np.arange(1, 72)
        
        time_grid = np.arange(1,72)

        cox_model = Cox_Calibrated2()

        cox_model.fit(
            X_train,
            y_train,
            X_calib,
            y_calib,
            time_grid
        )

        with open(f"{output_dir}Model_{i}_COX_tra75.pkl", "wb") as f:
                pickle.dump(cox_model, f)
        result = cox_model.predict_all(X_eval)


        for name in ["Cox","Cox_ISR"]:

    

            IBSS = integrated_brier_skill_score(
                result[name], y_calib, y_eval, time_grid
                )
            
            cindex,_,_,_,_ =concordance_index_censored(
                y_eval["event"], y_eval["time"], 
                risk_from_surv(result[name], time_grid)
                )



            met_path = f"{output_dir}metrics_diff_COX_tra75.txt"
            exist = os.path.exists(met_path)
            mode = "a"
            write_header = not exist
            with open(met_path, mode) as f:
                pd.DataFrame([{
                    "rep": i,
                    "train_size": len(y_train),
                    "model": name,
                    "cindex": cindex,
                    "IBS": IBSS,
                    "model_path": f"{output_dir}Model_{i}_COX_tra75.pkl",
                }]).to_csv(f, header=write_header, index=False)
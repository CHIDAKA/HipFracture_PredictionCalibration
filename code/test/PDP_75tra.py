import sys
import argparse
import os
import numpy as np
import pandas as pd
import pickle

from sklearn.isotonic import IsotonicRegression
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.functions import StepFunction
import matplotlib.pyplot as plt

sys.path.append("./test")
from src.RSF_ISR_diff2 import RSF_Calibrated2, logit, sigmoid, compute_ipcw_targets_and_weights, recalibrate_model_full_ipcw, integrated_brier_skill_score, risk_from_surv

def compute_pdp_iso(model, X, feature, grid_values, time_grid=None):
    

    results = {
        "RSF": [],
        "RSF_ISR": [],
    }

    for val in grid_values:

        X_tmp = X.copy()
        X_tmp[feature] = val

        result =  model.predict_iso(X_tmp)

        for key in ["RSF", "RSF_ISR"]:
            # S_safe = np.clip(S, 1e-8, 1.0)

            # risk = np.trapezoid(
            #     -np.log(S_safe),
            #     axis=1
            # )

            results[key].append(risk_from_surv(result[key], time_grid).mean())

    # shape: (grid,)
    for k in results:
        results[k] = np.array(results[k])

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute PDP for RSF models")
    parser.add_argument("--ratio", type=int, default=5)
    args = parser.parse_args()
    ratio = args.ratio

    rng = np.random.default_rng(0)
    n_rep = 100# テスト
    n_sub = 50000  # 調整可能


    with open(f"./data/calib_eval_data_v2.pkl", "rb") as f:
            X_calib, y_calib, X_eval, y_eval = pickle.load(f)
    #     # remove age under 75
    # mask_ov75 = X_calib["age"] >= 75
    # X_calib = X_calib[mask_ov75]
    # y_calib = y_calib[mask_ov75]

    mask_ov75 = X_eval["age"] >= 75
    X_eval = X_eval.loc[mask_ov75,~X_eval.columns.str.startswith("insurer")]
    y_eval = y_eval[mask_ov75]


    idx = rng.choice(len(X_eval), n_sub, replace=False)
    X_sub = X_eval.iloc[idx].copy()

    time_grid = np.arange(1, 72)




    features = ["age", "bmi", "ALT", "AST", "SBP", "e_gfr"]
    for feature in features:
        grid_values = np.linspace(
            X_eval[feature].quantile(0.05),
            X_eval[feature].quantile(0.95),
            50  # ← grid_resolution
        )
        
        all_results = {
        "RSF": [],
        "RSF_ISR": [],
        }


        for i in range(n_rep):
            model_path = f"./data/ratio_{ratio}/Model_{i}_x1_tra75.pkl"
            if not os.path.exists(model_path):
                continue

            with open(model_path, "rb") as f:
                model = pickle.load(f)


            pdp = compute_pdp_iso(
                model,
                X_sub,
                feature=feature,
                grid_values=grid_values
            )

            for k in all_results:
                all_results[k].append(pdp[k])


        for k in all_results:
            all_results[k] = np.array(all_results[k])
            # shape: (n_bootstrap, n_grid)

        pickle_path = f"./data/ratio_{ratio}/x1_tra75_{feature}_pdp.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(all_results, f)

        summary = {}

        for k in all_results:

            summary[k] = {
                "mean": all_results[k].mean(axis=0),
                "lower": np.percentile(all_results[k], 2.5, axis=0),
                "upper": np.percentile(all_results[k], 97.5, axis=0)
            }

        for k in summary:

            plt.figure()

            plt.plot(grid_values, summary[k]["mean"], label=k)

            plt.fill_between(
                grid_values,
                summary[k]["lower"],
                summary[k]["upper"],
                alpha=0.3
            )

            plt.title(f"PDP with CI: {k}")
            plt.xlabel(feature)
            plt.ylabel("risk")
            plt.legend()

            plt.show()
            plt.savefig(f"./data/ratio_{ratio}/x1_tra75_{k}_{feature}_pdp.pdf")


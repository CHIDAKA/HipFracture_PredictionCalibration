import argparse
import os
import numpy as np
import pandas as pd
import pickle

from sklearn.isotonic import IsotonicRegression
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.functions import StepFunction


import sys
sys.path.append("./test")
from src.RSF_ISR_diff2 import RSF_Calibrated2, logit, sigmoid, compute_ipcw_targets_and_weights, recalibrate_model_full_ipcw, integrated_brier_skill_score, risk_from_surv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RSF with different case-control ratios")
    parser.add_argument("--ratio", type=int, default=5, help="Case-control ratio (default: 5)")
    args = parser.parse_args()
    ratio = args.ratio
    output_dir = f"./results/ratio_{ratio}/"
    os.makedirs(output_dir, exist_ok=True)

# data load

    with open(f"./data/calib_eval_data_v2.pkl", "rb") as f:
            X_calib, y_calib, X_eval, y_eval = pickle.load(f) 
    
    # remove age under 75
    mask_ov75 = X_calib["age"] >= 75
    X_calib = X_calib[mask_ov75]
    y_calib = y_calib[mask_ov75]

    mask_ov75 = X_eval["age"] >= 75
    X_eval = X_eval[mask_ov75]
    y_eval = y_eval[mask_ov75]
    print(f"Number of calibration samples (age >= 75): {len(y_calib)}")
    print(f"Number of evaluation samples (age >= 75): {len(y_eval)}")

    n_rep = 100
    time_grid = np.arange(1, 72)
    for i in range(n_rep):
        print(i)
        rng_i = np.random.default_rng(i)
        
        with open(f"{output_dir}Model_{i}_x1.pkl", "rb") as f:
            model = pickle.load(f)
        
        recalibrate_model_full_ipcw(model, X_calib, y_calib, time_grid)


        with open(f"{output_dir}Model_{i}_x1_over75calib.pkl", "wb") as f:
             pickle.dump(model, f)

        
        result = model.predict_all(X_eval)


        for name in ["RSF","RSF_ISR","RSF_dykstra","RSF_ISR_dykstra"]:

    

            IBSS = integrated_brier_skill_score(
                result[name], y_calib, y_eval, time_grid
                )
            
            cindex,_,_,_,_ =concordance_index_censored(
                y_eval["event"], y_eval["time"], 
                risk_from_surv(result[name], time_grid)
                )



            met_path = f"{output_dir}metrics_diff_x1_over75calib.txt"
            exist = os.path.exists(met_path)
            mode = "a"
            write_header = not exist
            with open(met_path, mode) as f:
                pd.DataFrame([{
                    "rep": i,
                    "model": name,
                    "cindex": cindex,
                    "IBS": IBSS,
                    "model_path": f"{output_dir}Model_{i}_x1_over75calib.pkl",
                }]).to_csv(f, header=write_header, index=False)

                
        


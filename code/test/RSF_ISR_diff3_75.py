import os
import re
from sys import prefix
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.functions import StepFunction
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import (concordance_index_censored, 
                            integrated_brier_score, 
                            cumulative_dynamic_auc)
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.inspection import permutation_importance
from datetime import datetime
from sksurv.util import Surv
from sklearn.model_selection import KFold
from scipy.stats import rankdata
import optuna

from sklearn.isotonic import IsotonicRegression
from sklearn.base import BaseEstimator
from collections import Counter

# ratio = 1 argparseで指定
n_rep = 100
TARGET_N = 500000 #目的より大きいサイズ
True_TARGET_N = 50000

not_use = [
    "exam_month","observable_start_ym","observable_end_ym","onset",
    "exam_ymd","onset_ym","end_ym","start_ym","birth_ym",
    "kenshin_times_before_onset","observation_start_age",
    "months_until_first_kenshin","duration_disease", 'distance', 'weights', 'subclass', 'observation_months'
]

def split_ids_stratified(
    id_label_df,
    split=(0.8, 0.1, 0.1),
    id_col="id",
    label_col="event",
    random_state=42
):
    from sklearn.model_selection import train_test_split

    ids = id_label_df[id_col]
    y = id_label_df[label_col]

    train_ids, temp_ids = train_test_split(
        ids,
        test_size=(1 - split[0]),
        stratify=y,
        random_state=random_state
    )

    y_temp = id_label_df.set_index(id_col).loc[temp_ids, label_col]

    calib_ids, eval_ids = train_test_split(
        temp_ids,
        test_size=split[2] / (split[1] + split[2]),
        stratify=y_temp,
        random_state=random_state
    )

    return (
        train_ids.values,
        calib_ids.values,
        eval_ids.values
    )

def bootstrap_train_ids(
    train_ids,
    id_label_df,
    ratio,
    target_n_rows,
    rng,
    id_col="kojin_id",
    label_col="case",
    record_col="kenshin_times"
):
    avg_rows_per_id = id_label_df[record_col].mean()
    df = id_label_df.set_index(id_col).reindex(train_ids)

    case_ids = df[df[label_col] == 1].index.values
    control_ids = df[df[label_col] == 0].index.values

    # --- ID数計算（row数ベース）
    target_n_ids = int(target_n_rows / avg_rows_per_id)

    n_case = int(round(target_n_ids / (1 + ratio)))
    n_case = max(n_case, 1)

    n_control = n_case * ratio

    # --- bootstrap
    sampled_case = rng.choice(case_ids, size=n_case, replace=True)
    sampled_control = rng.choice(control_ids, size=n_control, replace=True)

    train_ids_bs = np.concatenate([sampled_case, sampled_control])

    return train_ids_bs



def load_csv_by_ids(
    filepath,
    target_ids,
    id_col="kojin_id",
    chunksize=100000
):
    import pandas as pd

    id_counts = Counter(target_ids)  # ← ここが重要
    target_set = set(id_counts.keys())

    result = []

    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        sub = chunk[chunk[id_col].isin(target_set)]
        if not sub.empty:
            
            # ✅ IDごとに繰り返し
            repeated = []
            for id_val, group in sub.groupby(id_col):
                k = id_counts[id_val]
                for _ in range(k):
                    repeated.append(group.copy())

            result.append(pd.concat(repeated, ignore_index=True))

    return pd.concat(result, ignore_index=True)


def trim_rows(df, target_n, rng):
    if len(df) > target_n:
        return df.sample(
            n=target_n,
            replace=False,
            random_state=rng.integers(1e9)
        )
    return df

def Xy_from_df(df, not_use):
    X = df.drop(columns=['kojin_id', 'event', "time", "S72"] + not_use, errors='ignore')
    X = X.join(pd.get_dummies(X["insurer_shubetsu"], prefix="insurer")).drop(columns=["insurer_shubetsu"], errors='ignore')
    y = Surv.from_dataframe('event', "time", df)
    return X, y


def logit(p):
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_ipcw_weights_full(y, time_grid):

    time = y["time"]
    event = y["event"]

    km_t, km_p = kaplan_meier_estimator(~event, time)

    def G(t):
        idx = np.searchsorted(km_t, t, side="right") - 1
        if idx >= 0:
            return max(km_p[idx], 1e-4)
        return 1.0

    n = len(time)
    T = len(time_grid)

    weights = np.zeros((n, T))

    for j, t in enumerate(time_grid):
        for i in range(n):
            ti = min(time[i], t)
            weights[i, j] = 1.0 / G(ti)

    return weights

def compute_ipcw_targets_and_weights(y, time_grid):
    time = y["time"]
    event = y["event"]

    km_t, km_p = kaplan_meier_estimator(~event, time)

    def G(t):
        idx = np.searchsorted(km_t, t, side="right") - 1
        if idx >= 0:
            return max(km_p[idx], 1e-4)
        return 1.0

    n = len(time)
    T = len(time_grid)

    weights = np.zeros((n, T))
    y_surv_matrix = np.zeros((n, T))

    for j, t in enumerate(time_grid):
        for i in range(n):
            if time[i] <= t and event[i] == 1:
                # 時点 t までにイベント発生
                y_surv_matrix[i, j] = 0.0
                weights[i, j] = 1.0 / G(time[i])
            elif time[i] > t:
                # 時点 t を超えて確実に生存
                y_surv_matrix[i, j] = 1.0
                weights[i, j] = 1.0 / G(t)
            else:
                # 時点 t 以前に打ち切り（重みを0にして学習から除外）
                y_surv_matrix[i, j] = 0.0
                weights[i, j] = 0.0 

    return y_surv_matrix, weights

# ---------------------------------
# recalibration（核心）
# ---------------------------------
def recalibrate_model_full_ipcw(model, X_calib, y_calib, time_grid):

    S_val = model.base_model.predict_survival_function(X_calib)
    S_val = np.asarray([[fn(t) for t in time_grid] for fn in S_val])
    y_surv_matrix, weights = compute_ipcw_targets_and_weights(y_calib, time_grid)
    new_iso_models = []

    for t_idx, t in enumerate(time_grid):
        
        ir = IsotonicRegression(out_of_bounds="clip")

        # ★修正：打ち切りを考慮した y_surv_matrix を使用
        ir.fit(
            S_val[:, t_idx],
            y_surv_matrix[:, t_idx],
            sample_weight=weights[:, t_idx]
        )

        new_iso_models.append(ir)

    model.iso_models_ = new_iso_models
# ---------------------------------
# メインクラス
# ---------------------------------
class RSF_Calibrated2(BaseEstimator):

    def __init__(self, base_model,
                 use_isotonic=True,
                 use_dykstra=False,
                 dykstra_iter=5):

        self.base_model = base_model
        self.use_isotonic = use_isotonic
        self.use_dykstra = use_dykstra
        self.dykstra_iter = dykstra_iter

    # =========================
    # fit
    # =========================
    def fit(self, X_train, y_train,
            X_val=None, y_val=None,
            time_grid=None):

        self.time_grid_ = time_grid

        # -------------------------
        # RSF学習
        # -------------------------
        self.base_model.fit(X_train, y_train)

        # -------------------------
        # isotonic（validation）
        # -------------------------
        S_val = self.base_model.predict_survival_function(X_val)
        S_val = np.asarray([[fn(t) for t in time_grid] for fn in S_val])

        y_surv_matrix, weights = compute_ipcw_targets_and_weights(y_val, time_grid)
        self.iso_models_ = []

        for t_idx, t in enumerate(time_grid):
            
            ir = IsotonicRegression(out_of_bounds="clip")

            # ★修正：打ち切りを考慮した y_surv_matrix を使用
            ir.fit(
                S_val[:, t_idx],
                y_surv_matrix[:, t_idx],
                sample_weight=weights[:, t_idx]
            )

            

        # for t_idx, t in enumerate(time_grid):

        #     y_t = (y_val["time"] > t).astype(float)

        #     ir = IsotonicRegression(out_of_bounds="clip")

        #     ir.fit(
        #         S_val[:, t_idx],
        #         y_t,
        #         sample_weight=weights[:, t_idx]
        #     )
            self.iso_models_.append(ir)

        return self

    # =========================
    # predict
    # =========================
    def predict(self, X, return_survival_function=False, return_area = True):

        S = self.base_model.predict_survival_function(X)
        S = np.array([
            fn(self.time_grid_)
            for fn in S
        ])


        # -------------------------
        # isotonic
        # -------------------------
        if self.use_isotonic:

            S_cal = np.zeros_like(S)

            for t_idx in range(len(self.time_grid_)):
                S_cal[:, t_idx] = self.iso_models_[t_idx].transform(
                    S[:, t_idx]
                )

            S = S_cal

        # 単調性
        #S = np.minimum.accumulate(S, axis=1) どうするか

        # -------------------------
        # Dykstra
        # -------------------------
        if self.use_dykstra:

            Z = logit(np.clip(S, 1e-4, 1 - 1e-4))
            risk = -np.log(S[:, -1] + 1e-4)

            Z = self._dykstra(Z, risk)
            S = sigmoid(Z)

        S_safe = np.clip(S, 1e-8, 1.0)

        risk_last = -np.log(S_safe[:, -1]) # 真のハザードではないが、実用上の変換
        risk_int = np.trapezoid(
            -np.log(S_safe),
            x=self.time_grid_,
            axis=1
        )


        # risk_last  = -np.log(S[:, -1]) # 真のハザードではないが、実用上の変換
        # risk_int   = np.trapezoid(-np.log(S), axis=1)
        if return_survival_function:
            return S
        elif return_area:
            return risk_int
        else:
            return risk_last


    # =========================
    # Dykstra
    # =========================
    def _dykstra(self, Z, risk):

        S = Z.copy()
        n, T = S.shape

        p = np.zeros_like(S)
        q = np.zeros_like(S)

        order = np.argsort(risk)

        for _ in range(self.dykstra_iter):

            # risk方向
            U = S + p
            for t in range(T):
                ir = IsotonicRegression(increasing=False)
                vals = ir.fit_transform(
                    np.arange(n),
                    U[order, t]
                )
                S[order, t] = vals
            p = U - S

            # time方向
            V = S + q
            for i in range(n):
                ir = IsotonicRegression(increasing=False)
                S[i, :] = ir.fit_transform(
                    np.arange(T),
                    V[i, :]
                )
            q = V - S

        return S
    
    def predict_all(self, X):
        # -------------------------
        # ① RSF
        # -------------------------
        surv_funcs = self.base_model.predict_survival_function(X)

        S_rsf = np.array([
            fn(self.time_grid_) for fn in surv_funcs
        ])

        # # 単調補正（安全）
        # S_rsf = np.minimum.accumulate(S_rsf, axis=1)

        results = {
            "RSF": S_rsf
        }

        # -------------------------
        # ② ISR
        # -------------------------
        if self.use_isotonic:

            S_iso = np.copy(S_rsf)

            for t_idx in range(len(self.time_grid_)):
                S_iso[:, t_idx] = self.iso_models_[t_idx].transform(
                    S_iso[:, t_idx]
                )

            #S_iso = np.minimum.accumulate(S_iso, axis=1) # 単調性はある

            results["RSF_ISR"] = S_iso

        # -------------------------
        # ③ Dykstra（RSFベース）
        # -------------------------
        if self.use_dykstra:

            Z = logit(np.clip(S_rsf, 1e-4, 1-1e-4))
            risk = -np.log(S_rsf[:, -1] + 1e-4)

            Z_dyk = self._dykstra(Z, risk)
            S_dyk = sigmoid(Z_dyk)

            results["RSF_dykstra"] = S_dyk

        # -------------------------
        # ④ ISR + Dykstra
        # -------------------------
        if self.use_isotonic and self.use_dykstra:

            Z = logit(np.clip(S_iso, 1e-4, 1-1e-4))
            risk = -np.log(S_iso[:, -1] + 1e-4)

            Z_dyk2 = self._dykstra(Z, risk)
            S_iso_dyk = sigmoid(Z_dyk2)

            results["RSF_ISR_dykstra"] = S_iso_dyk

        return results


def Xy_from_df(df, not_use):
    X = df.drop(columns=['kojin_id', 'event', "time", "S72"] + not_use, errors='ignore')
    if "insurer_shubetsu" in X.columns:
        X = X.join(pd.get_dummies(X["insurer_shubetsu"], prefix="insurer")).drop(columns=["insurer_shubetsu"], errors='ignore')
    y = Surv.from_dataframe('event', "time", df)
    return X, y

def integrated_brier_skill_score(ypred_rsf, survival_train, survival_test, times):
    # predictがあるか
    ibs_rsf = integrated_brier_score(
        survival_train, survival_test, ypred_rsf, times
    )

    # 3. KM（ベースライン）の予測値を作成
    # 学習データ全体からカプラン・マイヤー曲線を推定
    times_km, prob_km = kaplan_meier_estimator(
        survival_train["event"], survival_train["time"]
    )
    km_step_func = StepFunction(times_km, prob_km)
    # 指定した評価時点（times）における KM の生存確率を取得 (1次元配列)
    preds_km_single = km_step_func(times)

    # テストデータのサンプル数分だけ複製して (n_samples_test, n_times) の行列にする
    n_samples_test = len(survival_test)
    ypred_km = np.tile(preds_km_single, (n_samples_test, 1))

    # 4. KM の IBS を計算
    ibs_km = integrated_brier_score(
        survival_train, survival_test, ypred_km, times
    )

    # 5. 統合ブライアスキルスコア (IBSS) の算出
    ibss = 1.0 - (ibs_rsf / ibs_km)
    print(f"Evaluation time: {times.min()} to {times.max()}")
    print(f"Integrated Brier Score (KM): {ibs_km:.4f}")
    print(f"Integrated Brier Score (RSF): {ibs_rsf:.4f}" )
    print(f"Integrated Brier Skill Score (IBSS): {ibss:.4f}")
    return ibss

def risk_from_surv(S, time_grid_):
            S_safe = np.clip(S, 1e-8, 1.0)
            risk_last = -np.log(S_safe[:, -1]) # 真のハザードではないが、実用上の変換
            risk_int = np.trapezoid(
                -np.log(S_safe),
                x=time_grid_,
                axis=1
            )
            return risk_int


def calculate_1d_ale(X, predict_fn, feature_name, K=50):
    """
    1次元/多次元(生存関数等)出力に対応したALEを計算する関数 (厳密化版)
    
    引数:
        X (pd.DataFrame): 特徴量のデータフレーム
        predict_fn (callable): 特徴量Xを受け取り、予測結果のnumpy配列を返す関数。
                               ・生存関数を渡すと、リスク上昇時にプロットは下落（負）になります。
                               ・リスクスコアを渡すと、プロットは上昇（正）になります。
        feature_name (str): ALEを計算したい特徴量の名前
        K (int): ビンの数 (グリッド分割数)
        
    戻り値:
        z_bounds (np.array): ビンの境界値 (長さ K+1)
        ale_values (np.array): センタリングされたALE値 (予測出力の次元に合わせた配列)
    """
    
    #x_target = X[feature_name].values
    
    # --- 修正点: 対象特徴量の欠損値を除外して計算ベースを作成 ---
    valid_mask = ~pd.isna(X[feature_name])
    X_valid = X[valid_mask].copy()
    x_target = X_valid[feature_name].values
    
    if len(x_target) == 0:
        raise ValueError(f"特徴量 '{feature_name}' はすべて欠損値です。")
    
    # K個の区間(ビン)を作成するための分位点を計算
    quantiles = np.linspace(0, 1, K + 1)
    z_bounds = np.quantile(x_target, quantiles)
    # 重複する境界を削除してユニークにする
    z_bounds = np.unique(z_bounds) 
    actual_K = len(z_bounds) - 1
    
    # 予測出力の形状(次元)を確認
    test_pred = np.array(predict_fn(X.iloc[0:1]))
    is_multidim = len(test_pred.shape) > 1 and test_pred.shape[1] > 1
    
    if is_multidim:
        n_outputs = test_pred.shape[1]
        local_effects = np.zeros((actual_K, n_outputs))
    else:
        local_effects = np.zeros(actual_K)
        
    # 各ビンについて局所効果（Local Effect）を計算
    for k in range(actual_K):
        lower_bound = z_bounds[k]
        upper_bound = z_bounds[k+1]
        
        # 1. ビン内のデータ取得 (境界の重複カウントを排除するための修正)
        if k == 0:
            in_bin_idx = np.where((x_target >= lower_bound) & (x_target <= upper_bound))[0]
        else:
            in_bin_idx = np.where((x_target > lower_bound) & (x_target <= upper_bound))[0]
        
        if len(in_bin_idx) == 0:
            if is_multidim:
                local_effects[k, :] = 0
            else:
                local_effects[k] = 0
            continue
            
        # 2. ビン内の実際のデータを取り出す
        X_k = X.iloc[in_bin_idx].copy()
        
        # 3. 特徴量値の置き換え
        X_lower = X_k.copy()
        X_lower[feature_name] = lower_bound
        X_upper = X_k.copy()
        X_upper[feature_name] = upper_bound
        
        # 4. モデルによる予測値の差分（局所的な勾配）を計算し、平均をとる
        pred_upper = np.array(predict_fn(X_upper))
        pred_lower = np.array(predict_fn(X_lower))
        
        if is_multidim:
            local_effects[k, :] = np.mean(pred_upper - pred_lower, axis=0)
        else:
            local_effects[k] = np.mean(pred_upper - pred_lower)
            
    # 5. 局所効果を累積（積分）する
    # 修正: ALE関数は境界点(z_bounds)で定義されるため、配列サイズは actual_K + 1 になる
    if is_multidim:
        accumulated_effects = np.zeros((actual_K + 1, n_outputs))
        accumulated_effects[1:, :] = np.cumsum(local_effects, axis=0)
    else:
        accumulated_effects = np.zeros(actual_K + 1)
        accumulated_effects[1:] = np.cumsum(local_effects)
    
    # 6. センタリング（全体の平均効果をゼロにする）の厳密化
    # 各データポイントがどのビンに属するかを判定 (0 から actual_K-1)
    bin_indices = np.digitize(x_target, z_bounds[1:], right=False)
    bin_indices = np.clip(bin_indices, 0, actual_K - 1) 
    
    # 各データポイントにおける未センタリングのALE値を推定
    # (ビンの下限と上限の累積効果の平均を使用)
    if is_multidim:
        uncentered_ale = (accumulated_effects[bin_indices] + accumulated_effects[bin_indices + 1]) / 2.0
        mean_effect = np.mean(uncentered_ale, axis=0)
    else:
        uncentered_ale = (accumulated_effects[bin_indices] + accumulated_effects[bin_indices + 1]) / 2.0
        mean_effect = np.mean(uncentered_ale)
    
    ale_values = accumulated_effects - mean_effect
    
    # 修正: X軸の値としてビンの中心ではなく、境界値 (z_bounds) を返す
    return z_bounds, ale_values


# ratioの引数があれば取得、なければデフォルト値を使用
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RSF with different case-control ratios")
    parser.add_argument("--ratio", type=int, default=5, help="Case-control ratio (default: 5)")
    args = parser.parse_args()
    ratio = args.ratio
    output_dir = f"./data/ratio_{ratio}/"
    os.makedirs(output_dir, exist_ok=True)

# data load
    summary_df = pd.read_csv('./data/patient_matching_data.csv')
    # case label
    case_df = pd.read_csv('./data/pid_S720-S722.csv')
    summary_df["case"] = summary_df["kojin_id"].isin(case_df.kojin_id)

    if not os.path.exists(f"./data/used_IDs.csv"): # 固定のIDを使用
        #. ID split
        train_ids, calib_ids, eval_ids = split_ids_stratified(
            summary_df[["kojin_id", "case", "kenshin_times"]],
            id_col="kojin_id",
            label_col="case",
            split = (0.8, 0.1, 0.1),
            )

        print(
            f"train_ids: {len(train_ids)},calib_ids: {len(calib_ids)},eval_ids: {len(eval_ids)}"
            )

        df = pd.DataFrame({
            "split": ["train"] * len(train_ids) + ["calib"] * len(calib_ids) + ["eval"] * len(eval_ids),
            "kojin_id": np.concatenate([train_ids, calib_ids, eval_ids])
        })
        df.to_csv(f"./data/used_IDs.csv", index=False)
        
        del case_df, df
    else:
        df = pd.read_csv(f"./data/used_IDs.csv")
        train_ids = df[df["split"] == "train"]["kojin_id"].values
        calib_ids = df[df["split"] == "calib"]["kojin_id"].values
        eval_ids = df[df["split"] == "eval"]["kojin_id"].values
        del case_df, df

    filepath = "./data/disease.csv"

    if not os.path.exists(f"./data/calib_eval_data_v2.pkl"):
        # ✅ calib / eval は最初に固定読み込み（重要）
        X_calib, y_calib = Xy_from_df(
                                load_csv_by_ids(filepath, calib_ids), not_use)
        
        print("Calib loaded:", len(y_calib))

        X_eval, y_eval = Xy_from_df(
                                load_csv_by_ids(filepath, eval_ids), not_use)
        
        print("Eval loaded:", len(y_eval))
        with open(f"./data/calib_eval_data_v2.pkl", "wb") as f:
            pickle.dump((X_calib, y_calib, X_eval, y_eval), f)
    else:
        with open(f"./data/calib_eval_data_v2.pkl", "rb") as f:
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
        
        rsf = RandomSurvivalForest(
                n_estimators=500,
                min_samples_leaf=250,
                #max_features=best_params["max_features"],
                max_depth=15,
                random_state=0,
                n_jobs=-1
            )

        final_model = RSF_Calibrated2(
            base_model=rsf,
            use_isotonic=True, #iso,
            use_dykstra=True #dyk
        )
        final_model.fit(
            X_train, y_train,
            X_calib, y_calib,
            time_grid
        )            
        with open(f"{output_dir}Model_{i}_x1_tra75.pkl", "wb") as f:
                pickle.dump(final_model, f)
        result = final_model.predict_all(X_eval)


        for name in ["RSF","RSF_ISR","RSF_dykstra","RSF_ISR_dykstra"]:

    

            IBSS = integrated_brier_skill_score(
                result[name], y_calib, y_eval, time_grid
                )
            
            cindex,_,_,_,_ =concordance_index_censored(
                y_eval["event"], y_eval["time"], 
                risk_from_surv(result[name], time_grid)
                )



            met_path = f"{output_dir}metrics_diff_x1_tra75.txt"
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
                    "model_path": f"{output_dir}Model_{i}_x1_tra75.pkl",
                }]).to_csv(f, header=write_header, index=False)

                
        


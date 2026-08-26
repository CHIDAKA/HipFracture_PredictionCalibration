import pandas as pd
import numpy as np
import re
import os




def average_variable(df, prefix):
    # prefixで始まる列を検索（数値列は除外）
    target_cols = [col for col in df.columns if col.startswith(prefix)]
    if len(target_cols) == 0:
        print(f"No columns found starting with '{prefix}'")
        return
    return df[target_cols].mean(axis=1, skipna=True)

def preprocessing_hc(df):
    
    remove_col =[] 
    for prefix in ["systolic_blood_pressure", "diastolic_blood_pressure",
                   "chusei_shibou_", "hdl_", "ldl_", "got_", "gpt_", "gamma_gt_",
                   "kuufukuji_ketto_", "zuiji_ketto_", "hba1c_ngsp_", "hba1c_jdsh_",
                   "nyoutou_kikai_yomitori_code", "nyoutou_mokushihou_code",
                   "nyoutanpaku_kikai_yomitori_code", "nyoutanpaku_mokushihou_code",
                   "nyousenketsu_kikai_yomitori_code", "nyousenketsu_mokushihou_code",
                   "kessei_cr_", "kessei_nyousan_"]:
        remove_col.extend([col for col in df.columns if col.startswith(prefix)])
    df_temp = df.copy()
    df_list =[df_temp]
    df_list.append(average_variable(df,"systolic_blood_pressure").rename("SBP"))
    df_list.append(average_variable(df,"diastolic_blood_pressure").rename("DBP"))
    df_list.append(average_variable(df,"chusei_shibou_").rename("TG"))
    df_list.append(average_variable(df,"hdl_").rename("HDL_C"))
    df_list.append(average_variable(df,"ldl_").rename("LDL_C"))
    df_list.append(average_variable(df,"got_").rename("AST"))
    df_list.append(average_variable(df,"gpt_").rename("ALT"))
    df_list.append(average_variable(df,"gamma_gt_").rename("gGT"))
    df_list.append(average_variable(df,"kuufukuji_ketto_").rename("fasting_BG"))
    df_list.append(average_variable(df,"zuiji_ketto_").rename("spot_BG"))

    df_list.append(
        pd.concat([average_variable(df, "hba1c_ngsp_"), average_variable(df, "hba1c_jdsh_")],
                   axis=1, keys=["hba1c_ngsp","hba1c_jdsh"]).apply(
                       lambda x: np.round(x["hba1c_ngsp"],1) if not pd.isnull(x["hba1c_ngsp"]) else np.round(1.02*x["hba1c_jdsh"]+0.25,1), 
                       axis=1).rename("hba1c"))
    
    df_list.append(df.apply(
        lambda x: x["nyoutou_kikai_yomitori_code"] if pd.notnull(x["nyoutou_kikai_yomitori_code"]) else x["nyoutou_mokushihou_code"], 
                       axis=1).rename("urinary_glucose"))
    
    df_list.append(df.apply(
        lambda x: x["nyoutanpaku_kikai_yomitori_code"] if pd.notnull(x["nyoutanpaku_kikai_yomitori_code"]) else x["nyoutanpaku_mokushihou_code"], 
                       axis=1).rename("urinary_protein"))
    
    df_list.append(df.apply(
        lambda x: x["nyousenketsu_kikai_yomitori_code"] if pd.notnull(x["nyousenketsu_kikai_yomitori_code"]) else x["nyousenketsu_mokushihou_code"], 
                       axis=1).rename("urinary_blood"))
    
    df_list.append(average_variable(df,"kessei_cr_").rename("sCr"))
    df_list.append(average_variable(df,"kessei_nyousan_").rename("sUA"))
    
    df_temp = pd.concat(df_list, axis=1)

    
    

    df_null = pd.read_csv("./data/null_prevalence_checkup.csv")
    remove_col.extend(df_null[df_null.null_ratio >= 0.5].column_name.tolist())
    df_temp = df_temp.drop(columns=remove_col, errors='ignore')
    # df_temp = df_temp.loc[:,df_temp.isnull().sum()/len(df_temp) < 0.5].reset_index(drop=True)
    # これやるとchunkごとに列が変わるので、後で結合できなくなってしまうので、あとで実行
    df_temp = df_temp[df_temp.time > 0].reset_index(drop=True)

    df_null_ = pd.DataFrame(
        index =df_temp.columns,
        data =df_temp.isnull().sum()/len(df_temp),
        columns = ["null_ratio"]        
        )
    df_null_.sort_values(by="null_ratio", ascending=False, inplace=True)

    print("Null value proportions:")
    print(df_null_.head(10))

    f = lambda x: re.match(r"^[A-Z][0-9]{2}$|^999$|(0|^[0-9]{7}).0$|\d.",str(x))==None
    labtest_col = df_temp.columns[list(map(f,df_temp.columns))]
    for col in labtest_col:
        if df_temp[col].isnull().sum() > 0:
            df_temp[col+"_NA"] = df_temp[col].isna().astype(int)
    print("Length after filtering:", len(df_temp))
    


    return df_temp


def preprocess(df):
    print("Preprocessing..., Original length:", len(df))

    df["exam_month"] = pd.to_datetime(df["exam_month"], format="%Y-%m-%d %H:%M:%S")
    YM_cols = ["birth_ym", "observable_start_ym", "observable_end_ym"]
    f = lambda x: pd.to_datetime(df[x], format="%Y/%m")
    for x in YM_cols:
        df[x] = f(x)

    # 3. 年の差と月から年齢を計算
    df['age'] = (df["exam_month"].dt.year - df["birth_ym"].dt.year) - (df["exam_month"].dt.month < df["birth_ym"].dt.month).astype(int)

    onset = pd.to_datetime(df["kojin_id"].map(onset_dict), format="%Y-%m-%d")
    df["event"] = (~onset.isna()).astype(int)
    onset = onset.fillna(df["observable_end_ym"])
    df["onset"] = pd.to_datetime(onset, unit='s')
    df['time'] = (df["onset"].dt.year- df["exam_month"].dt.year) * 12 + (df["onset"].dt.month - df["exam_month"].dt.month)
    return df

if __name__ == "__main__":
    chunksize = 10000  # メモリに応じて調整（例: 1万〜10万）
    input_path = "./data/disease_n2.csv"
    output_path = "./data/disease.csv"
    first = True  # ヘッダー制御用
    case_df = pd.read_csv('./data/pid_S720-S722.csv')
    onset_dict = case_df.set_index('kojin_id')['onset_month'].to_dict()

    for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunksize)):
        print(f"Processing chunk {i}...")

        # 前処理
        chunk = preprocess(chunk)
        chunk = preprocessing_hc(chunk)

        # 書き出し（最初だけheader=True）
        chunk.to_csv(
            output_path,
            mode="w" if first else "a",
            header=first,
            index=False
        )
        first = False

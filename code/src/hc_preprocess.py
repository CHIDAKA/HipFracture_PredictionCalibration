import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Calculate start and end ym
def calculate_observation_s_e(x):
    start = datetime.strptime(x["first_exam_ymd"],"%Y-%m-%d") - relativedelta(months = x["months_until_first_kenshin"])
    end = start + relativedelta(months = x["observation_months"])
    return int(start.strftime("%Y%m")),int(end.strftime("%Y%m"))

# Calculate age at exam
def yyyymm_int(x, frmt):
    return int(datetime.strptime(x,frmt).strftime("%Y%m"))

def age_calc(x):
    # We define age based on month
   age = (yyyymm_int(x["exam_ymd"],"%Y-%m-%d") - yyyymm_int(x["birth_ym"],"%Y/%m"))//100
   return age

# Check if the exam date is after the onset date
def time_diff(x):
    case_col = x["is_disease"]
    if not case_col:
        return False
    elif case_col:
        return bool(yyyymm_int(x["exam_ymd"],"%Y-%m-%d") > yyyymm_int(x["onset_ym"],"%Y/%m"))
    else:
        return Exception("case is not 0 or 1")
    
# NaN ratio
def nan_ratio(df, title="NaN ratio in s32 dataset including after onset"):
    plt.figure(figsize=(6,16))
    plt.barh(y=df.columns,
    width=df.isnull().sum()/df.shape[0]*100)
    plt.xticks(rotation=90, fontsize=8)
    plt.title(title)
    plt.tight_layout()


def plot_density_by_prefix(df, prefix, title_name, xlabel, xlim_range=None, case_col='case'):
    """
    指定したprefixで始まる列の密度分布をcase別に描画する汎用関数
    
    Parameters:
    -----------
    df : pandas.DataFrame
        分析対象のデータフレーム
    prefix : str
        検索対象の列名のprefix（例: 'systolic_blood_pressure', 'hdl_', 'bmi'）
    title_name : str
        グラフのタイトルに使用する名前（例: 'Blood Pressure', 'HDL Cholesterol'）
    xlabel : str
        X軸のラベル（例: 'Pressure (mmHg)', 'Cholesterol (mg/dL)'）
    xlim_range : tuple, optional
        X軸の範囲 (min, max)
    case_col : str, default='case'
        case/controlを判定する列名
    """
    
    # prefixで始まる列を検索
    target_cols = [col for col in df.columns if col.startswith(prefix)]
    
    if len(target_cols) == 0:
        print(f"No columns found starting with '{prefix}'")
        return
    
    print(f"Found {len(target_cols)} columns starting with '{prefix}': {target_cols}")
    
    # 1x2のサブプロットを作成（case=True と case=False で分ける）
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{title_name} Density Distributions', fontsize=16)
    
    # 色の設定（列数に応じて色を生成）
    colors = plt.cm.Set1(np.linspace(0, 1, len(target_cols)))
    case_labels = ['Case=False (Control)', 'Case=True (Fracture)']
    
    for case_idx, case_value in enumerate([False, True]):
        ax = axes[case_idx]
        
        # 各列のdensity curveを描画
        for i, col in enumerate(target_cols):
            data = df[df[case_col] == case_value][col].dropna()
            
            if len(data) > 0:
                # 列名を短縮してラベルに使用
                short_label = col.replace(prefix, '').replace('_', '').strip()
                if short_label == '' or short_label == 'other':
                    short_label = f'{prefix.rstrip("_")}{i+1}' if short_label == '' else 'other'
                
                # seabornのkdeplotを使用してスムージングされたdensity curveを描画
                sns.kdeplot(data=data, ax=ax, color=colors[i], 
                           label=short_label, linewidth=2, alpha=0.8)
        
        ax.set_title(f'{case_labels[case_idx]}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # X軸の範囲設定
        if xlim_range:
            ax.set_xlim(xlim_range)
    
    plt.tight_layout()
    plt.show()
    
    # データの基本統計を表示
    print(f"\nBasic statistics for {title_name.lower()} measures:")
    for case_value in [False, True]:
        print(f"\n=== Case={case_value} ===")
        for col in target_cols:
            data = df[df[case_col] == case_value][col].dropna()
            if len(data) > 0:
                print(f"{col}: n={len(data)}, mean={data.mean():.1f}, std={data.std():.1f}")
            else:
                print(f"{col}: No data")


def plot_missing_by_time_prefix(df, prefix, title_name, date_col='exam_ymd', no_count=True):
    """
    指定したprefixで始まる列の欠測割合の時系列変化を描画する関数
    
    Parameters:
    -----------
    df : pandas.DataFrame
        分析対象のデータフレーム
    prefix : str
        検索対象の列名のprefix
    title_name : str
        グラフのタイトルに使用する名前
    date_col : str, default='exam_ymd'
        日付列名
    """
    
    # prefixで始まる列を検索
    target_cols = [col for col in df.columns if col.startswith(prefix)]
    
    if len(target_cols) == 0:
        print(f"No columns found starting with '{prefix}'")
        return
    
    print(f"Found {len(target_cols)} columns starting with '{prefix}': {target_cols}")
    
    # 日付を月単位に変換
    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col])
    df_temp['year_month'] = df_temp[date_col].dt.to_period('M')
    
    # 月ごとの欠測割合を計算
    missing_by_month = []
    
    for ym in sorted(df_temp['year_month'].dropna().unique()):
        month_data = df_temp[df_temp['year_month'] == ym]
        if len(month_data) > 0:
            missing_rates = {}
            missing_rates['year_month'] = ym
            missing_rates['total_records'] = len(month_data)
            
            for col in target_cols:
                missing_rate = month_data[col].isnull().sum() / len(month_data) * 100
                missing_rates[col] = missing_rate
            
            missing_by_month.append(missing_rates)
    
    # DataFrameに変換
    missing_df = pd.DataFrame(missing_by_month)
    
    if no_count:
        fig, axes = plt.subplots(1, 1, figsize=(16, 8))

        ax1 = axes
        colors = plt.cm.Set1(np.linspace(0, 1, len(target_cols)))
        
        for i, col in enumerate(target_cols):
            short_label = col.replace(prefix, '').replace('_', '').strip()
            if short_label == '' or short_label == 'other':
                short_label = f'{prefix.rstrip("_")}{i+1}' if short_label == '' else 'other'
            
            ax1.plot(missing_df['year_month'].astype(str), missing_df[col], 
                    color=colors[i], marker='o', linewidth=2, alpha=0.8, 
                    label=short_label, markersize=4)
        
        ax1.set_title(f'{title_name} - Missing Data Rate Over Time', fontsize=14)
        ax1.set_xlabel('Year-Month')
        ax1.set_ylabel('Missing Rate (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

    else:
        # グラフ描画
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # 上のグラフ: 欠測割合の推移
        ax1 = axes[0]
        colors = plt.cm.Set1(np.linspace(0, 1, len(target_cols)))
        
        for i, col in enumerate(target_cols):
            short_label = col.replace(prefix, '').replace('_', '').strip()
            if short_label == '' or short_label == 'other':
                short_label = f'{prefix.rstrip("_")}{i+1}' if short_label == '' else 'other'
            
            ax1.plot(missing_df['year_month'].astype(str), missing_df[col], 
                    color=colors[i], marker='o', linewidth=2, alpha=0.8, 
                    label=short_label, markersize=4)
        
        ax1.set_title(f'{title_name} - Missing Data Rate Over Time', fontsize=14)
        ax1.set_xlabel('Year-Month')
        ax1.set_ylabel('Missing Rate (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 下のグラフ: 月ごとの総レコード数
        ax2 = axes[1]
        ax2.bar(missing_df['year_month'].astype(str), missing_df['total_records'], 
            alpha=0.6, color='gray')
        ax2.set_title('Total Records per Month', fontsize=14)
        ax2.set_xlabel('Year-Month')
        ax2.set_ylabel('Number of Records')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # 統計サマリー
    print(f"\nMissing data statistics for {title_name.lower()}:")
    print("=" * 60)
    
    for col in target_cols:
        overall_missing = df[col].isnull().sum() / len(df) * 100
        monthly_missing_stats = missing_df[col].describe()
        
        print(f"\n{col}:")
        print(f"  Overall missing rate: {overall_missing:.1f}%")
        print(f"  Monthly missing rate - Mean: {monthly_missing_stats['mean']:.1f}%, Std: {monthly_missing_stats['std']:.1f}%")
        print(f"  Monthly missing rate - Min: {monthly_missing_stats['min']:.1f}%, Max: {monthly_missing_stats['max']:.1f}%")
    
    return missing_df


def plot_categorical_by_prefix(df, prefix, title_name, case_col='case', max_categories=20):
    """
    指定したprefixで始まるカテゴリカル列の分布をcase別に描画する汎用関数
    
    Parameters:
    -----------
    df : pandas.DataFrame
        分析対象のデータフレーム
    prefix : str
        検索対象の列名のprefix
    title_name : str
        グラフのタイトルに使用する名前
    case_col : str, default='case'
        case/controlを判定する列名
    max_categories : int, default=20
        表示する最大カテゴリ数（多すぎる場合は上位N個のみ表示）
    """
    
    # prefixで始まる列を検索（数値列は除外）
    target_cols = [col for col in df.columns if col.startswith(prefix)]
    
    if len(target_cols) == 0:
        print(f"No columns found starting with '{prefix}'")
        return
    
    print(f"Found {len(target_cols)} columns starting with '{prefix}': {target_cols}")
    
    # グリッドサイズを決定
    n_cols = len(target_cols)
    if n_cols == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        axes = [ax]
    else:
        n_rows = (n_cols + 1) // 2  # 2列レイアウト
        fig, axes_grid = plt.subplots(n_rows, 2, figsize=(16, 5*n_rows))
        
        # axesを1次元配列に変換
        if n_rows == 1:
            axes = axes_grid if isinstance(axes_grid, np.ndarray) else [axes_grid]
        else:
            axes = axes_grid.flatten()
    
    fig.suptitle(f'{title_name} - Categorical Distributions by Case Status', fontsize=16)
    
    case_labels = ['Case=False (Control)', 'Case=True (Fracture)']
    colors = ['lightblue', 'salmon']
    
    for idx, col in enumerate(target_cols):
        # 該当する列のデータを取得
        col_data = df[col].dropna()
        
        if len(col_data) == 0:
            # データがない場合
            ax = axes[idx]
            ax.text(0.5, 0.5, f'No data available\nfor {col}', 
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title(col.replace(prefix, '').replace('_', ' ').strip().title())
            continue
        
        # カテゴリごとの集計
        crosstab = pd.crosstab(df[col], df[case_col], normalize='columns') * 100
        
        # カテゴリが多すぎる場合は上位のみ表示
        if len(crosstab) > max_categories:
            # 全体での出現頻度順に並べて上位を取得
            total_counts = df[col].value_counts()
            top_categories = total_counts.head(max_categories).index
            crosstab = crosstab.loc[top_categories]
            print(f"Note: {col} has {len(total_counts)} categories. Showing top {max_categories}.")
        
        # バープロット描画
        ax = axes[idx]
        # case別のバーを並べて描画
        x_pos = np.arange(len(crosstab.index))
        width = 0.35
        
        for case_idx, case_value in enumerate([False, True]):
            if case_value in crosstab.columns:
                ax.bar(x_pos + (case_idx - 0.5) * width, 
                      crosstab[case_value], 
                      width, 
                      label=case_labels[case_idx],
                      color=colors[case_idx],
                      alpha=0.8)
        
        # グラフの設定
        ax.set_title(col.replace(prefix, '').replace('_', ' ').strip().title())
        ax.set_xlabel('Categories')
        ax.set_ylabel('Percentage (%)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(crosstab.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 余ったサブプロットを非表示
    for idx in range(len(target_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # 統計サマリー
    print(f"\nCategorical statistics for {title_name.lower()}:")
    print("=" * 60)
    
    for col in target_cols:
        print(f"\n{col}:")
        
        # 全体の分布
        total_counts = df[col].value_counts()
        print(f"  Total categories: {len(total_counts)}")
        print(f"  Most common: {total_counts.index[0]} ({total_counts.iloc[0]} records)")
        
        # case別の分布
        crosstab_counts = pd.crosstab(df[col], df[case_col])
        if len(crosstab_counts) <= 10:  # カテゴリが少ない場合は詳細表示
            print("  Case-control crosstab:")
            print(crosstab_counts.to_string())
        
        # 欠測率
        missing_rate = df[col].isnull().sum() / len(df) * 100
        print(f"  Missing rate: {missing_rate:.1f}%")
    
    return df[target_cols].describe(include='all')

def average_variable(df, prefix):
    # prefixで始まる列を検索（数値列は除外）
    target_cols = [col for col in df.columns if col.startswith(prefix)]
    if len(target_cols) == 0:
        print(f"No columns found starting with '{prefix}'")
        return
    return df[target_cols].mean(axis=1, skipna=True)


def preprocessing_hc(df):
    

    keep_col = ["kojin_id","exam_ymd","height","weight","fukui", #bmiは計算できるため除外
                "bmi", #250911追加
                "naizo_shibou_menseki","non_hdl","hematocrit_value","kesshikisoryou_hb",
                "sekkekyuusuu", "kesshoubansuu", "e_gfr", # hoken_shidou_level_codeは血圧、脂質、血糖から判断され、実際には階層があるので除外
                "seikatsu_shuukan_kaizen_code"] 
    df_temp = df.copy()[keep_col]

    transform_cols1 = ["shindenzu_code",
                       "fukuyaku1_ketsuatsu_code", "fukuyaku2_kettou_code", "fukuyaku3_shishitsu_code",
                       "kioureki1_noukekkan_code","kioureki2_shinkekkan_code","kioureki3_zinfuzen_code",
                       "hinketsu_code","kitsuen_code","undou_shuukan_30pun_code","hokou_or_shintai_katsudou_code",
                       "taijuu_henka_last_year_code","tabekata2_shuushinmae_code","tabekata3_yashoku_kanshoku_code",
                       "suimin_code","hoken_shidou_kibou_code"]
    for col in transform_cols1:
        if col in df.columns:
            df_temp[col+"r"] = df[col].replace({2:0})
    df_temp["taijuu_zouka_20sai_code"] = df.taijuu_henka_20sai_code.replace({2:0})
    df_temp["gantei_kensa_wm_code_r"] = df.gantei_kensa_wm_code.replace({1:0,2:1,3:2,4:3})
    df_temp["gantei_kensa_kaihen_davis_code_r"] = df.gantei_kensa_kaihen_davis_code.replace({1:0,2:1,3:2,4:3})
    df_temp["hokou_hayai_code"] = df.hokou_sokudo_code.replace({2:0})
    df_temp["tabekata1_hayagui_code_r"] = df.tabekata1_hayagui_code.replace({2:0,3:1})
    df_temp["tabekata3_kanshoku_code_r"] = df.tabekata3_kanshoku_code.replace({3:0,2:1,1:2})
    df_temp["no_breakfast_code"] = df.shokushuukan_code.replace({2:0})
    df_temp["alcohol_freq"] = df.inshu_code.replace({3:0,2:1,1:2})
    df_temp["alcohol_amount"] = df.inshuryou_code.replace({1:0,2:1,3:2,4:3})

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

    disease_posi = np.where(df.columns.str.startswith("A"))
    df_list.append(df.copy()[df.iloc[:,disease_posi[0][0]:].columns.tolist()])
    df_temp = pd.concat(df_list, axis=1)

    

    return df_temp
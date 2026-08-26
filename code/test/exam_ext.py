
import duckdb
import pandas as pd

# Using ICD10 that was diagnosed in >1% of all patients
df_dis = pd.read_csv("./data/prevalence_icd10_sub.csv")
dis_feature = df_dis[df_dis["prevalence"] > 0.01]["icd10"].tolist()
print(len(dis_feature))

# Using checkup items without over 50% null values across all records
####Not now####
df_null = pd.read_csv("./data/null_prevalence_checkup.csv")
# checkup_columns = df_null[df_null["null_ratio"] < 0.5]["column_name"].tolist()
checkup_columns = df_null["column_name"].tolist()
print(len(checkup_columns))

con = duckdb.connect()
data_dir = "./data/"

dis_feature_str = ",".join([f"'{x}'" for x in dis_feature])
coalesce_cols = ",\n".join([
    f"COALESCE(p.{c}, 0) AS {c}"
    for c in dis_feature
])

checkup_feature_str = ",\n".join([f"{x}" for x in checkup_columns if x not in ["kojin_id", "exam_ymd"]])
checkup_cols = ",\n".join([
    f"e.{x}" 
    for x in checkup_columns if x not in ["kojin_id", "exam_ymd"]
    ])

query = f"""
COPY (
    WITH 

    exam AS (
        SELECT
            kojin_id,
            {checkup_feature_str},
            DATE_TRUNC('month', CAST(exam_ymd AS DATE)) AS exam_month
        FROM read_csv_auto('{data_dir}exam_interview.csv')
    ),
    
    
    dict AS (
        SELECT *
        FROM read_csv_auto('{data_dir}m_icd10.csv')
    ),

    receipt AS (
        SELECT
            b.kojin_id,
            DATE_TRUNC(
                'month',
                STRPTIME(b.receipt_ym, '%Y/%m')
            ) AS receipt_month,

            d.icd10_sub_code AS icd10
        FROM read_csv_auto('{data_dir}receipt_diseases.csv') AS b
        LEFT JOIN dict d
        ON b.diseases_code = d.diseases_code
        WHERE utagai_flg != 1
            AND d.icd10_sub_code IN ({dis_feature_str})
        
    ),

    base AS (
    SELECT  
        e.kojin_id,
        e.exam_month,
        r.icd10,
        COUNT(*) AS cnt
    FROM exam e
    JOIN receipt r
        ON e.kojin_id = r.kojin_id
        AND r.receipt_month >= e.exam_month - INTERVAL '6 months'
        AND r.receipt_month <  e.exam_month  
    GROUP BY
        e.kojin_id,
        e.exam_month,
        r.icd10
        
    ),

pivoted AS (
    SELECT *
    FROM base
    PIVOT(
        SUM(cnt)
        FOR icd10 IN ({dis_feature_str})
    )
),


attr AS (
    SELECT
        kojin_id,
        birth_ym,
        sex_code,
        insurer_shubetsu,
        observable_start_ym,
        observable_end_ym
    FROM read_csv_auto('{data_dir}tekiyo.csv')
)


    SELECT
        p.kojin_id,
        p.exam_month,
        {coalesce_cols},
        {checkup_cols},
        a.birth_ym,
        a.sex_code,
        a.insurer_shubetsu,
        a.observable_start_ym,
        a.observable_end_ym
    FROM pivoted p

    LEFT JOIN exam e
        ON p.kojin_id = e.kojin_id
        AND p.exam_month = e.exam_month

    LEFT JOIN attr a
        ON p.kojin_id = a.kojin_id
)


TO './result/disease.csv'
(FORMAT CSV, HEADER);

"""
con.execute(query)
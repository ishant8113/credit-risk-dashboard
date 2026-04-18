import duckdb
import pandas as pd
import os
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DB_PATH  = 'data/credit_risk.db'
CSV_PATH = 'Data/Give me some credit.csv'

def setup():
    print("Setting up database...")

    # Load CSV
    df = pd.read_csv(CSV_PATH)
    df.rename(columns={
        'SeriousDlqin2yrs'                        : 'is_default',
        'RevolvingUtilizationOfUnsecuredLines'     : 'revolving_util',
        'Age'                                     : 'age',
        'NumberOfTime30-59DaysPastDueNotWorse'    : 'late_30_59',
        'DebtRatio'                               : 'debt_ratio',
        'MonthlyIncome'                           : 'monthly_income',
        'NumberOfOpenCreditLinesAndLoans'         : 'open_credit_lines',
        'NumberOfTimes90DaysLate'                 : 'late_90_plus',
        'NumberRealEstateLoansOrLines'            : 'real_estate_loans',
        'NumberOfTime60-89DaysPastDueNotWorse'    : 'late_60_89',
        'NumberOfDependents'                      : 'num_dependents'
    }, inplace=True)

    con = duckdb.connect(DB_PATH)

    # raw table
    con.execute("DROP TABLE IF EXISTS raw_loans")
    con.execute("CREATE TABLE raw_loans AS SELECT * FROM df")

    # clean_loans
    con.execute("""
    CREATE OR REPLACE TABLE clean_loans AS
    SELECT
        is_default,
        CASE WHEN age < 18 OR age > 100 THEN NULL ELSE age END AS age,
        COALESCE(monthly_income, 3500)              AS monthly_income,
        COALESCE(num_dependents, 0)                 AS num_dependents,
        LEAST(GREATEST(revolving_util, 0), 1.0)     AS revolving_util,
        LEAST(debt_ratio, 5.0)                      AS debt_ratio,
        LEAST(open_credit_lines, 50)                AS open_credit_lines,
        LEAST(real_estate_loans, 10)                AS real_estate_loans,
        LEAST(late_30_59,  15)                      AS late_30_59,
        LEAST(late_60_89,  15)                      AS late_60_89,
        LEAST(late_90_plus, 15)                     AS late_90_plus
    FROM raw_loans
    WHERE age > 0 AND age IS NOT NULL
    """)

    # feature_engineering
    con.execute("""
    CREATE OR REPLACE TABLE feature_engineering AS
    SELECT *,
        CASE WHEN age < 25 THEN '18-25' WHEN age < 35 THEN '25-35'
             WHEN age < 45 THEN '35-45' WHEN age < 55 THEN '45-55'
             WHEN age < 65 THEN '55-65' ELSE '65+' END AS age_band,
        CASE WHEN monthly_income < 2000  THEN '<$2k'
             WHEN monthly_income < 4000  THEN '$2-4k'
             WHEN monthly_income < 6000  THEN '$4-6k'
             WHEN monthly_income < 10000 THEN '$6-10k'
             ELSE '>$10k' END AS income_band,
        (late_30_59 + late_60_89 + late_90_plus)       AS total_late_payments,
        CASE WHEN late_90_plus >= 1 THEN 1 ELSE 0 END  AS ever_seriously_delinquent,
        CASE WHEN revolving_util > 0.8 THEN 1 ELSE 0 END AS high_util_flag,
        CASE WHEN debt_ratio > 0.5 THEN 1 ELSE 0 END   AS high_debt_flag,
        CASE WHEN num_dependents = 0 THEN monthly_income
             ELSE ROUND(monthly_income / num_dependents, 2)
        END AS income_per_dependent,
        ROUND(monthly_income * debt_ratio, 2)           AS estimated_monthly_debt,
        ROW_NUMBER() OVER (ORDER BY (SELECT NULL))      AS row_num
    FROM clean_loans
    """)

    # risk_segments
    con.execute("""
    CREATE OR REPLACE TABLE risk_segments AS
    SELECT *,
        CASE
            WHEN late_90_plus >= 1 OR revolving_util > 0.9 THEN 'High Risk'
            WHEN late_30_59   >= 2 OR debt_ratio     > 0.7 THEN 'Medium Risk'
            WHEN late_30_59   >= 1 OR revolving_util > 0.6 THEN 'Medium Risk'
            ELSE 'Low Risk'
        END AS risk_segment,
        ROUND(
            (LEAST(late_90_plus, 5) * 6.0) +
            (LEAST(late_60_89,   5) * 3.0) +
            (LEAST(late_30_59,   5) * 1.0) +
            (CASE WHEN revolving_util > 0.9 THEN 20
                  WHEN revolving_util > 0.7 THEN 12
                  WHEN revolving_util > 0.5 THEN 6 ELSE 0 END) +
            (CASE WHEN debt_ratio > 1.0 THEN 15
                  WHEN debt_ratio > 0.5 THEN 8
                  WHEN debt_ratio > 0.3 THEN 3 ELSE 0 END) +
            (CASE WHEN age < 25 THEN 10 WHEN age < 35 THEN 5
                  WHEN age > 60 THEN -5 ELSE 0 END) +
            (CASE WHEN monthly_income < 2000 THEN 5
                  WHEN monthly_income < 4000 THEN 2 ELSE 0 END)
        , 1) AS rule_risk_score
    FROM feature_engineering
    """)

    # portfolio_summary
    con.execute("""
    CREATE OR REPLACE TABLE portfolio_summary AS
    SELECT 'age_band' AS dimension, age_band AS segment,
        COUNT(*) AS total_customers, SUM(is_default) AS total_defaults,
        ROUND(AVG(is_default)*100,2) AS default_rate_pct,
        ROUND(AVG(monthly_income),0) AS avg_income,
        ROUND(AVG(debt_ratio),3) AS avg_debt_ratio,
        ROUND(AVG(revolving_util),3) AS avg_util
    FROM risk_segments WHERE age_band IS NOT NULL GROUP BY age_band
    UNION ALL
    SELECT 'income_band', income_band,
        COUNT(*), SUM(is_default),
        ROUND(AVG(is_default)*100,2),
        ROUND(AVG(monthly_income),0),
        ROUND(AVG(debt_ratio),3),
        ROUND(AVG(revolving_util),3)
    FROM risk_segments WHERE income_band IS NOT NULL GROUP BY income_band
    UNION ALL
    SELECT 'risk_segment', risk_segment,
        COUNT(*), SUM(is_default),
        ROUND(AVG(is_default)*100,2),
        ROUND(AVG(monthly_income),0),
        ROUND(AVG(debt_ratio),3),
        ROUND(AVG(revolving_util),3)
    FROM risk_segments GROUP BY risk_segment
    """)

    print("✅ Database ready")

    # Train models
    FEATURES = [
        'age','monthly_income','num_dependents','revolving_util',
        'debt_ratio','open_credit_lines','real_estate_loans',
        'late_30_59','late_60_89','late_90_plus','total_late_payments',
        'ever_seriously_delinquent','high_util_flag','high_debt_flag',
        'income_per_dependent','estimated_monthly_debt'
    ]

    df_model = con.execute("SELECT * FROM risk_segments").df()
    df_model = df_model[FEATURES + ['is_default']].dropna()

    X = df_model[FEATURES]
    y = df_model['is_default']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)

    ratio = (y_train==0).sum() / (y_train==1).sum()
    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=ratio, n_estimators=200, max_depth=4,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='auc', verbosity=0
    )
    xgb_model.fit(X_train, y_train)

    # Save scores
    df_scores = X_test.copy()
    df_scores['is_default']   = y_test.values
    df_scores['pd_score_lr']  = lr.predict_proba(X_test_scaled)[:, 1]
    df_scores['pd_score_xgb'] = xgb_model.predict_proba(X_test)[:, 1]
    df_scores['ml_risk_segment'] = pd.cut(
        df_scores['pd_score_xgb'],
        bins=[0, 0.1, 0.3, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    con.execute("DROP TABLE IF EXISTS model_scores")
    con.execute("CREATE TABLE model_scores AS SELECT * FROM df_scores")

    import pickle
    os.makedirs('app', exist_ok=True)
    with open('app/xgb_model.pkl', 'wb') as f: pickle.dump(xgb_model, f)
    with open('app/lr_model.pkl',  'wb') as f: pickle.dump(lr, f)
    with open('app/scaler.pkl',    'wb') as f: pickle.dump(scaler, f)
    with open('app/features.pkl',  'wb') as f: pickle.dump(FEATURES, f)

    con.close()
    print("✅ Models trained and saved")

if __name__ == "__main__":
    setup()
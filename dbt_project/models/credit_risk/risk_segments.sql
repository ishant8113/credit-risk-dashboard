/*
  Layer 3: Feature Engineering → Business Risk Segments
  Rule-based segmentation — used directly in the dashboard
  before ML scores are added
  Depends on: feature_engineering
*/

WITH base AS (
    SELECT * FROM {{ ref('feature_engineering') }}
),

segmented AS (
    SELECT
        *,

        -- RULE-BASED RISK SCORE (0-100 points)
        -- Used as the pre-ML risk indicator in dashboard
        ROUND(
            -- Late payments are the strongest signal (max 50 pts)
            (LEAST(late_90_plus,  5) * 6.0) +     -- up to 30 pts
            (LEAST(late_60_89,    5) * 3.0) +     -- up to 15 pts
            (LEAST(late_30_59,    5) * 1.0) +     -- up to 5 pts

            -- High utilization (max 20 pts)
            (CASE 
                WHEN revolving_util > 0.9 THEN 20
                WHEN revolving_util > 0.7 THEN 12
                WHEN revolving_util > 0.5 THEN 6
                ELSE 0
            END) +

            -- High debt ratio (max 15 pts)
            (CASE 
                WHEN debt_ratio > 1.0 THEN 15
                WHEN debt_ratio > 0.5 THEN 8
                WHEN debt_ratio > 0.3 THEN 3
                ELSE 0
            END) +

            -- Age factor (younger = higher risk, max 10 pts)
            (CASE 
                WHEN age < 25 THEN 10
                WHEN age < 35 THEN 5
                WHEN age > 60 THEN -5   -- seniors are lower risk
                ELSE 0
            END) +

            -- Low income (max 5 pts)
            (CASE 
                WHEN monthly_income < 2000 THEN 5
                WHEN monthly_income < 4000 THEN 2
                ELSE 0
            END)
        , 1) AS rule_risk_score,

        -- RISK SEGMENT (for dashboard traffic light)
        CASE
            WHEN late_90_plus >= 2                          THEN 'High Risk'
            WHEN late_90_plus >= 1 OR revolving_util > 0.9 THEN 'High Risk'
            WHEN late_30_59   >= 2 OR debt_ratio     > 0.7 THEN 'Medium Risk'
            WHEN late_30_59   >= 1 OR revolving_util > 0.6 THEN 'Medium Risk'
            ELSE                                                 'Low Risk'
        END AS risk_segment

    FROM base
)

SELECT * FROM segmented
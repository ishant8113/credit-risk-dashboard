/*
  Aggregated view for Portfolio Overview page in Streamlit
  No ML needed — pure SQL analytics
  Depends on: risk_segments
*/

-- Summary by age band
WITH by_age AS (
    SELECT
        'age_band'                          AS dimension,
        age_band                            AS segment,
        COUNT(*)                            AS total_customers,
        SUM(is_default)                     AS total_defaults,
        ROUND(AVG(is_default) * 100, 2)    AS default_rate_pct,
        ROUND(AVG(monthly_income), 0)      AS avg_income,
        ROUND(AVG(debt_ratio), 3)          AS avg_debt_ratio,
        ROUND(AVG(revolving_util), 3)      AS avg_util
    FROM {{ ref('risk_segments') }}
    WHERE age_band IS NOT NULL
    GROUP BY age_band
),

-- Summary by income band
by_income AS (
    SELECT
        'income_band'                       AS dimension,
        income_band                         AS segment,
        COUNT(*)                            AS total_customers,
        SUM(is_default)                     AS total_defaults,
        ROUND(AVG(is_default) * 100, 2)    AS default_rate_pct,
        ROUND(AVG(monthly_income), 0)      AS avg_income,
        ROUND(AVG(debt_ratio), 3)          AS avg_debt_ratio,
        ROUND(AVG(revolving_util), 3)      AS avg_util
    FROM {{ ref('risk_segments') }}
    WHERE income_band IS NOT NULL
    GROUP BY income_band
),

-- Summary by risk segment
by_risk AS (
    SELECT
        'risk_segment'                      AS dimension,
        risk_segment                        AS segment,
        COUNT(*)                            AS total_customers,
        SUM(is_default)                     AS total_defaults,
        ROUND(AVG(is_default) * 100, 2)    AS default_rate_pct,
        ROUND(AVG(monthly_income), 0)      AS avg_income,
        ROUND(AVG(debt_ratio), 3)          AS avg_debt_ratio,
        ROUND(AVG(revolving_util), 3)      AS avg_util
    FROM {{ ref('risk_segments') }}
    GROUP BY risk_segment
)

SELECT * FROM by_age
UNION ALL
SELECT * FROM by_income
UNION ALL
SELECT * FROM by_risk
ORDER BY dimension, default_rate_pct DESC
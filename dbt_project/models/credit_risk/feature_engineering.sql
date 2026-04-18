/*
  Layer 2: Clean → Engineered Features
  Creates derived columns used by the ML model and dashboard
  Depends on: clean_loans
*/

WITH base AS (
    SELECT * FROM {{ ref('clean_loans') }}
),

engineered AS (
    SELECT
        *,

        -- AGE BANDS (for dashboard grouping)
        CASE
            WHEN age < 25  THEN '18-25'
            WHEN age < 35  THEN '25-35'
            WHEN age < 45  THEN '35-45'
            WHEN age < 55  THEN '45-55'
            WHEN age < 65  THEN '55-65'
            ELSE                '65+'
        END AS age_band,

        -- INCOME BANDS
        CASE
            WHEN monthly_income < 2000  THEN '<$2k'
            WHEN monthly_income < 4000  THEN '$2-4k'
            WHEN monthly_income < 6000  THEN '$4-6k'
            WHEN monthly_income < 10000 THEN '$6-10k'
            ELSE                             '>$10k'
        END AS income_band,

        -- TOTAL LATE PAYMENTS (combined signal)
        (late_30_59 + late_60_89 + late_90_plus) AS total_late_payments,

        -- EVER SERIOUSLY DELINQUENT (90+ days even once)
        CASE WHEN late_90_plus >= 1 THEN 1 ELSE 0 END AS ever_seriously_delinquent,

        -- HIGH UTILIZATION FLAG (>80% of credit used)
        CASE WHEN revolving_util > 0.8 THEN 1 ELSE 0 END AS high_util_flag,

        -- HIGH DEBT RATIO FLAG
        CASE WHEN debt_ratio > 0.5 THEN 1 ELSE 0 END AS high_debt_flag,

        -- INCOME PER DEPENDENT (financial stress indicator)
        CASE 
            WHEN num_dependents = 0 THEN monthly_income
            ELSE ROUND(monthly_income / num_dependents, 2)
        END AS income_per_dependent,

        -- DEBT MONTHLY BURDEN (estimated absolute debt)
        ROUND(monthly_income * debt_ratio, 2) AS estimated_monthly_debt

    FROM base
    WHERE age IS NOT NULL    -- final filter: remove rows where age was invalid
),

-- Add a simple row number for splitting train/test in dbt
numbered AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS row_num
    FROM engineered
)

SELECT * FROM numbered
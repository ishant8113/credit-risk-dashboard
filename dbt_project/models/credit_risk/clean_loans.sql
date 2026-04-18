/*
  Layer 1: Raw → Clean
  Handles: missing values, outliers, invalid ages, invalid financial ratios
*/

WITH base AS (
    SELECT * FROM main.raw_loans
),

cleaned AS (
    SELECT

        -- Target
        is_default,

        -- Age: remove invalid values
        CASE 
            WHEN age < 18 OR age > 100 THEN NULL 
            ELSE age 
        END AS age,

        -- Income: fill ~19.8% missing with median
        -- Median is safer than mean here (right-skewed distribution)
        COALESCE(monthly_income, 3500)              AS monthly_income,

        -- Dependents: fill 2.6% missing with 0
        COALESCE(num_dependents, 0)                 AS num_dependents,

        -- Revolving utilization: valid range 0-1
        -- Values > 1 are data entry errors
        LEAST(GREATEST(revolving_util, 0), 1.0)     AS revolving_util,

        -- Debt ratio: cap at 5 (extreme outliers are data errors)
        LEAST(debt_ratio, 5.0)                      AS debt_ratio,

        -- Credit lines (reasonable max = 50)
        LEAST(open_credit_lines, 50)                AS open_credit_lines,

        -- Real estate loans (reasonable max = 10)
        LEAST(real_estate_loans, 10)                AS real_estate_loans,

        -- Late payment flags — cap at 15 (beyond that, signal is the same)
        LEAST(late_30_59,  15)                      AS late_30_59,
        LEAST(late_60_89,  15)                      AS late_60_89,
        LEAST(late_90_plus, 15)                     AS late_90_plus

    FROM base
    WHERE age > 0              -- remove 0-age invalid rows
      AND age IS NOT NULL
)

SELECT * FROM cleaned
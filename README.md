# 💳 Credit Risk Analytics Dashboard

An end-to-end credit risk analytics project built for a data analyst portfolio.
This project ingests raw loan data, builds a SQL transformation pipeline using dbt,
trains a credit risk model achieving a **Gini coefficient of 0.72**, and surfaces
insights through an interactive Streamlit dashboard.

🔗 **[Live Dashboard →](https://ishant8113-credit-risk-dashboard.streamlit.app)**

---

## 📸 Dashboard Preview

| Portfolio Overview | Risk Segmentation |
|---|---|
| Default rates by age, income | Customer risk tiers |

| Model Performance | Customer Lookup |
|---|---|
| ROC curve, feature importance | Live PD score calculator |

---

## 🏗️ Project Architecture
Raw CSV (150k rows)
↓
DuckDB + dbt          ← SQL transformation pipeline
↓
XGBoost Model         ← Probability of Default scoring
↓
Streamlit App         ← Interactive 4-page dashboard
↓
Streamlit Cloud       ← Live deployment

---

## 📊 Key Results

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| AUC | 0.8488 | 0.8604 |
| Gini Coefficient | 0.6976 | 0.7208 |

> A Gini coefficient above 0.60 is considered good for credit risk models.
> Industry benchmark for retail credit is typically 0.65–0.75.

---

## 🗂️ Project Structure
credit-risk-dashboard/
├── Data/
│   └── Give me some credit.csv     ← Kaggle dataset (150k rows)
├── notebooks/
│   ├── eda.ipynb                   ← Exploratory data analysis
│   ├── 02_duckdb_setup.ipynb       ← Database setup
│   └── 03_modeling.ipynb           ← ML model training
├── dbt_project/
│   ├── dbt_project.yml             ← dbt configuration
│   └── models/credit_risk/
│       ├── clean_loans.sql         ← Layer 1: cleaning
│       ├── feature_engineering.sql ← Layer 2: feature creation
│       ├── risk_segments.sql       ← Layer 3: risk scoring
│       └── portfolio_summary.sql   ← Layer 4: aggregations
├── app/
│   ├── streamlit_app.py            ← Main dashboard
│   ├── setup_db.py                 ← Auto database builder
│   ├── xgb_model.pkl               ← Trained XGBoost model
│   ├── lr_model.pkl                ← Trained Logistic Regression
│   ├── scaler.pkl                  ← Feature scaler
│   └── features.pkl                ← Feature column order
└── requirements.txt

---

## 🔄 SQL Pipeline (dbt + DuckDB)

Four layered SQL models transform raw data into analytics-ready tables:

**Layer 1 — `clean_loans.sql`**
Handles missing values, outlier capping, and invalid age removal.
- `monthly_income` nulls filled with median ($3,500)
- `revolving_util` capped at 1.0 (values > 1 are data errors)
- `debt_ratio` capped at 5.0

**Layer 2 — `feature_engineering.sql`**
Creates derived features used by the ML model.
- Age bands and income bands for segmentation
- Total late payments combined signal
- High utilization and high debt ratio flags
- Income per dependent (financial stress indicator)

**Layer 3 — `risk_segments.sql`**
Applies rule-based risk scoring (0–100 points) and segments
customers into Low / Medium / High risk tiers.

**Layer 4 — `portfolio_summary.sql`**
Aggregates default rates by age band, income band, and risk segment
for the portfolio overview dashboard page.

---

## 🤖 Machine Learning

**Dataset:** Give Me Some Credit (Kaggle) — 150,000 loan applicants

**Target:** `SeriousDlqin2yrs` — whether the borrower experienced 90+ days
delinquency within 2 years (6.7% base rate — highly imbalanced)

**Features used (16 total):**
- Demographics: age, monthly income, number of dependents
- Credit behavior: revolving utilization, debt ratio
- Delinquency history: 30-59, 60-89, 90+ days late payments
- Engineered: total late payments, ever seriously delinquent flag,
  high utilization flag, income per dependent, estimated monthly debt

**Models trained:**
- Logistic Regression with `class_weight='balanced'`
- XGBoost with `scale_pos_weight` for class imbalance

**Top predictive features:**
1. `late_90_plus` — times 90+ days late
2. `revolving_util` — revolving credit utilization
3. `ever_seriously_delinquent` — binary flag
4. `late_30_59` — times 30-59 days late
5. `debt_ratio` — monthly debt obligations / income

---

## 📱 Dashboard Pages

### 📊 Portfolio Overview
- 5 KPI cards (total customers, defaults, default rate, avg income, avg debt ratio)
- Default rate by age group and income band
- Monthly income and debt ratio distributions
- Default rate heatmap across age × income segments

### 🔴 Risk Segmentation
- Customer distribution across Low / Medium / High risk tiers
- Default rate per risk segment
- Rule-based risk score distribution overlaid by default status
- Average risk indicators per segment

### 🤖 Model Performance
- ROC curves for both models side by side
- Confusion matrices at 0.5 threshold
- XGBoost feature importance chart
- PD score distribution for defaulters vs non-defaulters
- Model comparison against industry benchmarks

### 🔍 Customer Lookup
- Input any customer profile using sliders
- Live Probability of Default gauge meter
- Risk segment badge (Low / Medium / High)
- Risk flags (high utilization, late payments, low income)
- Expected Loss calculation using Basel III formula:
  `EL = PD × LGD × EAD`

---

## 🚀 Run Locally

### 1 — Clone the repo
```bash
git clone https://github.com/ishant8113/credit-risk-dashboard.git
cd credit-risk-dashboard
```

### 2 — Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### 4 — Run the app
```bash
streamlit run app/streamlit_app.py
```

The app will automatically build the database and train models on first run
(takes 2–3 minutes). Subsequent runs load instantly from cache.

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Data ingestion | Pandas, DuckDB |
| SQL pipeline | dbt-core + dbt-duckdb |
| ML modeling | Scikit-learn, XGBoost |
| Dashboard | Streamlit, Plotly |
| Database | DuckDB |
| Deployment | Streamlit Cloud |
| Version control | GitHub |

---

## 📚 Dataset

**Give Me Some Credit** — Kaggle Competition Dataset  
150,000 loan applicants with 10 financial features and a binary default label.  
[View on Kaggle →](https://www.kaggle.com/c/GiveMeSomeCredit)

---

## 👤 Author

**Ishant**  
Data Analyst Portfolio Project  
[GitHub →](https://github.com/ishant8113)
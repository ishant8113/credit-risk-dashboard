import os
import sys
import pickle
import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.metrics import roc_curve, confusion_matrix

# ─────────────────────────────────────────
# PATHS — works locally and on Streamlit Cloud
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, 'data', 'credit_risk.db')
APP_PATH = os.path.join(BASE_DIR, 'app')

# ─────────────────────────────────────────
# AUTO SETUP — builds DB if it doesn't exist
# ─────────────────────────────────────────
if not os.path.exists(DB_PATH):
    st.info("Setting up database for first time — this takes 2-3 minutes...")
    os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
    sys.path.insert(0, BASE_DIR)
    from app.setup_db import setup
    setup()
    st.success("Setup complete! Reloading...")
    st.rerun()
st.set_page_config(
    page_title = "Credit Risk Dashboard",
    page_icon  = "💳",
    layout     = "wide"
)

# ─────────────────────────────────────────
# LOAD MODELS & DATA (cached)
# ─────────────────────────────────────────
@st.cache_resource
def load_models():
    with open(os.path.join(APP_PATH, 'xgb_model.pkl'), 'rb') as f:
        xgb_model = pickle.load(f)
    with open(os.path.join(APP_PATH, 'lr_model.pkl'), 'rb') as f:
        lr_model = pickle.load(f)
    with open(os.path.join(APP_PATH, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(APP_PATH, 'features.pkl'), 'rb') as f:
        features = pickle.load(f)
    return xgb_model, lr_model, scaler, features

@st.cache_data
def load_data():
    con = duckdb.connect(DB_PATH, read_only=True)
    risk_df      = con.execute("SELECT * FROM risk_segments").df()
    portfolio_df = con.execute("SELECT * FROM portfolio_summary").df()
    scores_df    = con.execute("SELECT * FROM model_scores").df()
    con.close()
    return risk_df, portfolio_df, scores_df

xgb_model, lr_model, scaler, FEATURES = load_models()
risk_df, portfolio_df, scores_df       = load_data()

# ─────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/bank-card-back-side.png", width=60)
st.sidebar.title("Credit Risk Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Portfolio Overview",
        "🔴 Risk Segmentation",
        "🤖 Model Performance",
        "🔍 Customer Lookup"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Dataset:** Give Me Some Credit  
    **Rows:** 149,986  
    **Model:** XGBoost  
    **Gini:** 0.72  
    **AUC:** 0.86  
    """
)

# ─────────────────────────────────────────
# HELPER — KPI CARD
# ─────────────────────────────────────────
def kpi(label, value, delta=None, color="#5C9BD4"):
    st.markdown(
        f"""
        <div style="background:#f8f9fa;border-left:5px solid {color};
                    padding:16px 20px;border-radius:8px;margin-bottom:8px">
            <div style="font-size:13px;color:#666">{label}</div>
            <div style="font-size:28px;font-weight:700;color:#222">{value}</div>
            {"<div style='font-size:12px;color:#888'>"+delta+"</div>" if delta else ""}
        </div>
        """,
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════
# PAGE 1 — PORTFOLIO OVERVIEW
# ═══════════════════════════════════════════════════════
if page == "📊 Portfolio Overview":

    st.title("📊 Portfolio Overview")
    st.markdown("High level summary of the entire loan portfolio and default patterns.")
    st.markdown("---")

    # ── KPI Row ──
    total     = len(risk_df)
    defaults  = risk_df['is_default'].sum()
    def_rate  = risk_df['is_default'].mean() * 100
    avg_income= risk_df['monthly_income'].mean()
    avg_debt  = risk_df['debt_ratio'].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: kpi("Total Customers",    f"{total:,}",           color="#5C9BD4")
    with col2: kpi("Total Defaults",     f"{defaults:,}",        color="#E07B54")
    with col3: kpi("Default Rate",       f"{def_rate:.2f}%",     color="#E07B54")
    with col4: kpi("Avg Monthly Income", f"${avg_income:,.0f}",  color="#5CB85C")
    with col5: kpi("Avg Debt Ratio",     f"{avg_debt:.3f}",      color="#F0AD4E")

    st.markdown("---")

    # ── Row 1: Default Rate by Age Band + Income Band ──
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Default Rate by Age Group")
        age_data = (
            portfolio_df[portfolio_df['dimension'] == 'age_band']
            .sort_values('default_rate_pct', ascending=False)
        )
        fig = px.bar(
            age_data, x='segment', y='default_rate_pct',
            color='default_rate_pct',
            color_continuous_scale='RdYlGn_r',
            text='default_rate_pct',
            labels={'segment': 'Age Group', 'default_rate_pct': 'Default Rate %'}
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Default Rate by Income Band")
        inc_data = (
            portfolio_df[portfolio_df['dimension'] == 'income_band']
            .sort_values('default_rate_pct', ascending=False)
        )
        fig = px.bar(
            inc_data, x='segment', y='default_rate_pct',
            color='default_rate_pct',
            color_continuous_scale='RdYlGn_r',
            text='default_rate_pct',
            labels={'segment': 'Income Band', 'default_rate_pct': 'Default Rate %'}
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Income Distribution + Debt Ratio Distribution ──
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Income Distribution")
        fig = px.histogram(
            risk_df, x='monthly_income',
            nbins=60,
            range_x=[0, 20000],
            color_discrete_sequence=['#5C9BD4'],
            labels={'monthly_income': 'Monthly Income ($)'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Debt Ratio Distribution")
        fig = px.histogram(
            risk_df, x='debt_ratio',
            nbins=60,
            range_x=[0, 2],
            color_discrete_sequence=['#E07B54'],
            labels={'debt_ratio': 'Debt Ratio'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Default Rate Heatmap ──
    st.subheader("Default Rate Heatmap — Age Group vs Income Band")
    heat_df = (
        risk_df[risk_df['age_band'].notna() & risk_df['income_band'].notna()]
        .groupby(['age_band', 'income_band'])['is_default']
        .mean()
        .reset_index()
    )
    heat_df['default_rate_pct'] = (heat_df['is_default'] * 100).round(2)
    pivot = heat_df.pivot(
        index='age_band',
        columns='income_band',
        values='default_rate_pct'
    )
    fig = px.imshow(
        pivot,
        color_continuous_scale='RdYlGn_r',
        text_auto=True,
        aspect='auto',
        labels={'color': 'Default Rate %'}
    )
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════
# PAGE 2 — RISK SEGMENTATION
# ═══════════════════════════════════════════════════════
elif page == "🔴 Risk Segmentation":

    st.title("🔴 Risk Segmentation")
    st.markdown("Customer distribution across rule-based and ML-based risk tiers.")
    st.markdown("---")

    # ── KPI Row ──
    seg_counts = risk_df['risk_segment'].value_counts()
    col1, col2, col3 = st.columns(3)
    with col1:
        n = seg_counts.get('Low Risk', 0)
        kpi("Low Risk Customers",    f"{n:,}",
            f"{n/len(risk_df)*100:.1f}% of portfolio", color="#5CB85C")
    with col2:
        n = seg_counts.get('Medium Risk', 0)
        kpi("Medium Risk Customers", f"{n:,}",
            f"{n/len(risk_df)*100:.1f}% of portfolio", color="#F0AD4E")
    with col3:
        n = seg_counts.get('High Risk', 0)
        kpi("High Risk Customers",   f"{n:,}",
            f"{n/len(risk_df)*100:.1f}% of portfolio", color="#E07B54")

    st.markdown("---")

    # ── Row 1: Pie chart + Default rate by segment ──
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Distribution by Risk Segment")
        seg_df = risk_df['risk_segment'].value_counts().reset_index()
        seg_df.columns = ['Risk Segment', 'Count']
        fig = px.pie(
            seg_df,
            names='Risk Segment',
            values='Count',
            color='Risk Segment',
            color_discrete_map={
                'Low Risk'   : '#5CB85C',
                'Medium Risk': '#F0AD4E',
                'High Risk'  : '#E07B54'
            },
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Default Rate by Risk Segment")
        seg_default = (
            risk_df.groupby('risk_segment')['is_default']
            .mean()
            .reset_index()
        )
        seg_default['default_rate_pct'] = (seg_default['is_default'] * 100).round(2)
        fig = px.bar(
            seg_default,
            x='risk_segment', y='default_rate_pct',
            color='risk_segment',
            color_discrete_map={
                'Low Risk'   : '#5CB85C',
                'Medium Risk': '#F0AD4E',
                'High Risk'  : '#E07B54'
            },
            text='default_rate_pct',
            labels={
                'risk_segment'     : 'Risk Segment',
                'default_rate_pct' : 'Default Rate %'
            }
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Rule risk score distribution ──
    st.subheader("Rule-Based Risk Score Distribution by Default Status")
    fig = px.histogram(
        risk_df, x='rule_risk_score',
        color='is_default',
        nbins=50,
        barmode='overlay',
        opacity=0.7,
        color_discrete_map={0: '#5C9BD4', 1: '#E07B54'},
        labels={
            'rule_risk_score': 'Risk Score (0-100)',
            'is_default'     : 'Defaulted'
        }
    )
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Avg features per segment ──
    st.subheader("Average Risk Indicators by Segment")
    seg_stats = risk_df.groupby('risk_segment').agg(
        avg_util        = ('revolving_util',   'mean'),
        avg_debt_ratio  = ('debt_ratio',        'mean'),
        avg_late_90     = ('late_90_plus',      'mean'),
        avg_income      = ('monthly_income',    'mean')
    ).round(3).reset_index()

    st.dataframe(
    seg_stats,
    use_container_width=True,
    hide_index=True
)

# ═══════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════
elif page == "🤖 Model Performance":

    st.title("🤖 Model Performance")
    st.markdown("Evaluation metrics for Logistic Regression and XGBoost credit risk models.")
    st.markdown("---")

    # ── KPI Row ──
    col1, col2, col3, col4 = st.columns(4)
    with col1: kpi("XGBoost AUC",  "0.8604", color="#5C9BD4")
    with col2: kpi("XGBoost Gini", "0.7208", color="#5C9BD4")
    with col3: kpi("LR AUC",       "0.8488", color="#E07B54")
    with col4: kpi("LR Gini",      "0.6976", color="#E07B54")

    st.markdown("---")

    # ── Row 1: ROC Curve ──
    st.subheader("ROC Curve — Both Models")

    y_true     = scores_df['is_default']
    xgb_probs  = scores_df['pd_score_xgb']
    lr_probs   = scores_df['pd_score_lr']

    xgb_fpr, xgb_tpr, _ = roc_curve(y_true, xgb_probs)
    lr_fpr,  lr_tpr,  _ = roc_curve(y_true, lr_probs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xgb_fpr, y=xgb_tpr,
        name='XGBoost (AUC=0.860)',
        line=dict(color='#5C9BD4', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=lr_fpr, y=lr_tpr,
        name='Logistic Regression (AUC=0.849)',
        line=dict(color='#E07B54', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random Baseline',
        line=dict(color='gray', width=1, dash='dash')
    ))
    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=420,
        legend=dict(x=0.6, y=0.1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Confusion Matrices ──
    col1, col2 = st.columns(2)

    for col, probs, title, color in zip(
        [col1, col2],
        [xgb_probs, lr_probs],
        ['XGBoost', 'Logistic Regression'],
        ['#5C9BD4', '#E07B54']
    ):
        with col:
            st.subheader(f"{title} — Confusion Matrix")
            preds = (probs >= 0.5).astype(int)
            cm    = confusion_matrix(y_true, preds)
            cm_df = pd.DataFrame(
                cm,
                index=['Actual: No Default', 'Actual: Default'],
                columns=['Pred: No Default', 'Pred: Default']
            )
            fig = px.imshow(
                cm_df,
                text_auto=True,
                color_continuous_scale='Blues',
                aspect='auto'
            )
            fig.update_layout(height=320, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Feature Importance ──
    st.subheader("XGBoost Feature Importance")
    imp_df = pd.DataFrame({
        'Feature'   : FEATURES,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=True)

    fig = px.bar(
        imp_df, x='Importance', y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Blues',
        labels={'Importance': 'Importance Score', 'Feature': ''}
    )
    fig.update_layout(height=500, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: PD Score Distribution ──
    st.subheader("PD Score Distribution — Defaulters vs Non-Defaulters")
    fig = px.histogram(
        scores_df, x='pd_score_xgb',
        color='is_default',
        nbins=60,
        barmode='overlay',
        opacity=0.7,
        color_discrete_map={0: '#5C9BD4', 1: '#E07B54'},
        labels={
            'pd_score_xgb': 'Probability of Default (XGBoost)',
            'is_default'  : 'Defaulted'
        }
    )
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    # ── Row 5: Gini Comparison Table ──
    st.subheader("Model Comparison — Industry Benchmarks")
    benchmark_df = pd.DataFrame({
        'Model'               : ['Logistic Regression', 'XGBoost', 'Industry Minimum', 'Industry Good', 'Industry Excellent'],
        'AUC'                 : [0.8488, 0.8604, 0.70, 0.80, 0.90],
        'Gini'                : [0.6976, 0.7208, 0.40, 0.60, 0.80],
        'Assessment'          : ['✅ Strong', '✅ Strong', '⚠️ Minimum', '✅ Good', '🏆 Excellent']
    })
    st.dataframe(benchmark_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════
# PAGE 4 — CUSTOMER LOOKUP
# ═══════════════════════════════════════════════════════
elif page == "🔍 Customer Lookup":

    st.title("🔍 Customer PD Score Lookup")
    st.markdown("Enter a customer profile to get their live Probability of Default score.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Customer Profile")

        age            = st.slider("Age",                        18, 100, 35)
        monthly_income = st.number_input("Monthly Income ($)",   0, 100000, 4000, step=500)
        num_dependents = st.slider("Number of Dependents",       0, 10, 1)
        revolving_util = st.slider("Revolving Utilization",      0.0, 1.0, 0.3, step=0.01)
        debt_ratio     = st.slider("Debt Ratio",                 0.0, 5.0, 0.3, step=0.01)
        open_credit    = st.slider("Open Credit Lines",          0, 50, 5)
        real_estate    = st.slider("Real Estate Loans",          0, 10, 1)
        late_30_59     = st.slider("Times 30-59 Days Late",      0, 15, 0)
        late_60_89     = st.slider("Times 60-89 Days Late",      0, 15, 0)
        late_90_plus   = st.slider("Times 90+ Days Late",        0, 15, 0)

    with col2:
        st.subheader("Risk Assessment")

        # Derived features (same as feature_engineering.sql)
        total_late              = late_30_59 + late_60_89 + late_90_plus
        ever_seriously_delin    = 1 if late_90_plus >= 1 else 0
        high_util_flag          = 1 if revolving_util > 0.8 else 0
        high_debt_flag          = 1 if debt_ratio > 0.5 else 0
        income_per_dep          = monthly_income if num_dependents == 0 else round(monthly_income / num_dependents, 2)
        est_monthly_debt        = round(monthly_income * debt_ratio, 2)

        # Build input dataframe in exact feature order
        input_data = pd.DataFrame([[
            age, monthly_income, num_dependents,
            revolving_util, debt_ratio, open_credit,
            real_estate, late_30_59, late_60_89, late_90_plus,
            total_late, ever_seriously_delin,
            high_util_flag, high_debt_flag,
            income_per_dep, est_monthly_debt
        ]], columns=FEATURES)

        # Get PD scores
        pd_xgb = xgb_model.predict_proba(input_data)[0][1]
        pd_lr  = lr_model.predict_proba(
            scaler.transform(input_data)
        )[0][1]

        # Risk segment
        if pd_xgb >= 0.3:
            segment = "🔴 High Risk"
            seg_color = "#E07B54"
        elif pd_xgb >= 0.1:
            segment = "🟡 Medium Risk"
            seg_color = "#F0AD4E"
        else:
            segment = "🟢 Low Risk"
            seg_color = "#5CB85C"

        # Display PD Score gauge
        fig = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = pd_xgb * 100,
            title = {'text': "Probability of Default (XGBoost)", 'font': {'size': 16}},
            number= {'suffix': "%", 'font': {'size': 40}},
            delta = {'reference': 6.7, 'suffix': '%'},
            gauge = {
                'axis' : {'range': [0, 100]},
                'bar'  : {'color': seg_color},
                'steps': [
                    {'range': [0,  10], 'color': '#d4edda'},
                    {'range': [10, 30], 'color': '#fff3cd'},
                    {'range': [30, 100],'color': '#f8d7da'}
                ],
                'threshold': {
                    'line' : {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': 30
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Risk segment badge
        st.markdown(
            f"""
            <div style="background:{seg_color};color:white;
                        text-align:center;padding:14px;
                        border-radius:8px;font-size:22px;
                        font-weight:700;margin-bottom:16px">
                {segment}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Score breakdown
        st.markdown("**Score Breakdown**")
        score_df = pd.DataFrame({
            'Model'                : ['XGBoost', 'Logistic Regression'],
            'PD Score'             : [f"{pd_xgb*100:.2f}%", f"{pd_lr*100:.2f}%"],
            'vs Portfolio Avg 6.7%': [
                f"{'▲' if pd_xgb > 0.067 else '▼'} {abs(pd_xgb - 0.067)*100:.2f}%",
                f"{'▲' if pd_lr  > 0.067 else '▼'} {abs(pd_lr  - 0.067)*100:.2f}%"
            ]
        })
        st.dataframe(score_df, use_container_width=True, hide_index=True)

        # Key risk flags
        st.markdown("**Risk Flags**")
        flags = []
        if late_90_plus >= 1:   flags.append("🔴 Has 90+ day late payments")
        if revolving_util > 0.8:flags.append("🔴 High revolving utilization (>80%)")
        if debt_ratio > 0.5:    flags.append("🟡 High debt ratio (>0.5)")
        if late_30_59 >= 2:     flags.append("🟡 Multiple 30-59 day late payments")
        if monthly_income < 2000:flags.append("🟡 Low monthly income (<$2,000)")
        if age < 25:            flags.append("🟡 Young borrower (higher risk group)")
        if not flags:           flags.append("🟢 No significant risk flags detected")

        for flag in flags:
            st.markdown(f"- {flag}")

        # Expected loss estimate
        st.markdown("**Expected Loss Estimate**")
        lgd = 0.45   # industry standard Loss Given Default
        ead = monthly_income * 12
        el  = pd_xgb * lgd * ead
        st.markdown(
            f"""
            - **PD** (Probability of Default): `{pd_xgb*100:.2f}%`
            - **LGD** (Loss Given Default): `45%` *(industry standard)*
            - **EAD** (Exposure at Default): `${ead:,.0f}` *(annual income)*
            - **Expected Loss = PD × LGD × EAD**: `${el:,.0f}`
            """
        )
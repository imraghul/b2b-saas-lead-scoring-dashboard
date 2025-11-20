import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
import altair as alt

st.set_page_config(
    page_title="B2B SaaS AI Lead Scoring Dashboard",
    page_icon="🚀",
    layout="wide",
)

st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top, #1f2a44 0%, #0b1120 55%, #05070f 100%);
            padding: 0 1.5rem 2.5rem 1.5rem;
            }
        [data-testid="stHeader"] {background: transparent;}
        .hero {
            background: rgba(15, 23, 42, 0.55);
            border-radius: 25px;
            padding: 3rem 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
            margin: 1.5rem auto 2.75rem auto;
            max-width: 1100px;
        }
        .hero h1 {font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem;}
        .hero p {font-size: 1.2rem; opacity: 0.95;}
        .upload-card {
            background: rgba(15, 23, 42, 0.8);
            border-radius: 20px;
            padding: 2.5rem;
            text-align: center;
            box-shadow: 0 20px 55px rgba(3, 7, 18, 0.65);
            margin: 2.25rem auto;
            max-width: 1100px;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .metrics-wrapper {margin: 1.5rem 0;}
        .metric-card {
            background: rgba(15, 23, 42, 0.85);
            border-radius: 18px;
            padding: 1.75rem;
            box-shadow: 0 12px 45px rgba(2, 6, 23, 0.45);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .metric-label {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.8rem;
            color: rgba(226, 232, 240, 0.8);
            margin-bottom: 0.4rem;
        }
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            color: #a5b4fc;
        }
        .metric-delta {color: #48bb78; font-weight: 600;}
        .glass-panel {
            background: rgba(15, 23, 42, 0.85);
            border-radius: 22px;
            padding: 2rem;
            box-shadow: 0 25px 55px rgba(2, 6, 23, 0.55);
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .filters-card {
            background: rgba(15, 23, 42, 0.95);
            border-radius: 24px;
            padding: 1.5rem 1.5rem 0.5rem 1.5rem;
            box-shadow: 0 25px 55px rgba(2, 6, 23, 0.55);
            margin-bottom: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .insight-card {
            border-radius: 18px;
            padding: 1.75rem;
            color: white;
        }
        .insight-card.positive {
            background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        }
        .insight-card.warning {
            background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
        }
        .dev-card {
            background: rgba(15, 23, 42, 0.92);
            border-radius: 24px;
            padding: 1.75rem;
            margin: 2.75rem auto 2.25rem auto;
            display: flex;
            align-items: center;
            gap: 1.25rem;
            box-shadow: 0 25px 55px rgba(2, 6, 23, 0.55);
            border: 1px solid rgba(255, 255, 255, 0.08);
            max-width: 1100px;
        }
        .dev-icon {
            width: 70px;
            height: 70px;
            border-radius: 20px;
            background: radial-gradient(circle at 30% 30%, #facc15, #f97316);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            color: #0f172a;
            font-weight: 800;
            box-shadow: inset 0 0 12px rgba(0,0,0,0.25);
        }
        .dev-card h4 {color: #f8fafc; margin: 0; font-size: 1.2rem;}
        .dev-card p {color: rgba(226, 232, 240, 0.85); margin: 0.1rem 0 0 0;}
        div[data-testid="stFileUploader"] section {text-align: center;}
        div[data-testid="stFileUploader"] label {color: #e2e8f0; font-weight: 600; font-size: 1rem;}
        div[data-testid="stFileUploader"] button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 12px;
            padding: 0.6rem 1.2rem;
            border: none;
        }
        .filters-card h3 {color: white; margin-bottom: 1.2rem;}
        .filters-card label span {color: white !important;}
        .filters-card [data-baseweb="slider"] div {background-color: #a5b4fc50;}
    </style>
    """,
    unsafe_allow_html=True,
)

HERO_TITLE = "AI Lead Scoring Dashboard"
DEV_CARD_HTML = """
<div class="dev-card">
    <div class="dev-icon">RG</div>
    <div>
        <h4>Developed by Raghul G</h4>
        <p>MBA 24-26 &middot; ID: M042-24</p>
    </div>
</div>
"""
REQUIRED_COLUMNS = [
    "Lead_ID",
    "Industry",
    "Company_Size",
    "Region",
    "Tech_Maturity",
    "Website_Visits",
    "Pricing_Page_Views",
    "Product_Page_Depth",
    "Email_Engagement",
    "Demo_Requests",
    "Webinar_Attendance",
    "CRM_Interactions",
    "Lead_Response_Time",
    "Feature_Breadth",
    "Integrations_Count",
    "DAU_MAU_Ratio",
    "Past_Revenue",
    "Contract_Length",
    "Renewal_Count",
    "Lead_Source",
    "Opportunity_Stage",
    "Sales_Cycle_Length",
    "Engagement_Score",
    "Recency_Score",
    "Intent_Score",
    "Converted",
    "Churned",
    "CLV",
]
CATEGORICAL_COLS = ["Industry", "Region", "Lead_Source", "Opportunity_Stage"]


def make_sample_data(rows: int = 60) -> pd.DataFrame:
    """Generate a small demo dataset that mirrors the expected schema."""

    rng = np.random.default_rng(42)
    industries = ["Technology", "Healthcare", "Finance", "Manufacturing", "Retail"]
    regions = ["North America", "Europe", "Asia Pacific", "Latin America"]
    sources = ["Inbound", "Outbound", "Partner", "Event", "Referral"]
    stages = ["Prospect", "Qualification", "Proposal", "Negotiation", "Closed Won", "Closed Lost"]

    def choice(options, size):
        return rng.choice(options, size=size)

    df = pd.DataFrame(
        {
            "Lead_ID": [f"LEAD_{1000 + i}" for i in range(rows)],
            "Industry": choice(industries, rows),
            "Company_Size": rng.integers(50, 5000, size=rows),
            "Region": choice(regions, rows),
            "Tech_Maturity": rng.integers(1, 5, size=rows),
            "Website_Visits": rng.integers(200, 5000, size=rows),
            "Pricing_Page_Views": rng.integers(20, 600, size=rows),
            "Product_Page_Depth": rng.integers(1, 12, size=rows),
            "Email_Engagement": rng.integers(10, 100, size=rows),
            "Demo_Requests": rng.integers(0, 12, size=rows),
            "Webinar_Attendance": rng.integers(0, 8, size=rows),
            "CRM_Interactions": rng.integers(5, 50, size=rows),
            "Lead_Response_Time": rng.integers(1, 48, size=rows),
            "Feature_Breadth": rng.integers(1, 10, size=rows),
            "Integrations_Count": rng.integers(0, 25, size=rows),
            "DAU_MAU_Ratio": rng.uniform(0.05, 0.75, size=rows),
            "Past_Revenue": rng.integers(20_000, 750_000, size=rows),
            "Contract_Length": rng.integers(6, 36, size=rows),
            "Renewal_Count": rng.integers(0, 7, size=rows),
            "Lead_Source": choice(sources, rows),
            "Opportunity_Stage": choice(stages, rows),
            "Sales_Cycle_Length": rng.integers(15, 180, size=rows),
            "Engagement_Score": rng.integers(10, 100, size=rows),
            "Recency_Score": rng.integers(1, 10, size=rows),
            "Intent_Score": rng.integers(1, 10, size=rows),
            "Converted": rng.integers(0, 2, size=rows),
            "Churned": rng.integers(0, 2, size=rows),
            "CLV": rng.integers(30_000, 300_000, size=rows),
        }
    )

    # Add light correlation: higher engagement + intent improve conversion; churn inversely.
    engagement_factor = (df["Engagement_Score"] + df["Intent_Score"]) / 200
    df["Converted"] = (rng.random(rows) < (0.25 + engagement_factor)).astype(int)
    df["Churned"] = (rng.random(rows) < (0.15 + (1 - engagement_factor) / 2)).astype(int)

    return df

# -------------------------------------------------
# Hero / intro section
# -------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>🚀 {HERO_TITLE}</h1>
        <p>Intelligent B2B SaaS lead prioritization powered by machine learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(DEV_CARD_HTML, unsafe_allow_html=True)

with st.container():
    st.markdown(
        """
        <div class="upload-card">
            <h2 style="color:#f8fafc;">👋 Welcome!</h2>
            <p style="font-size:1.05rem;color:#f8fafc;opacity:0.9;">Upload your B2B SaaS dataset to unlock AI-powered insights and lead scoring.</p>
            <p style="color:#ffffff;opacity:0.85;margin-bottom:1.2rem;">The dashboard will automatically train conversion, churn, and CLV models.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "📁 Upload Dataset (CSV)",
        type=["csv"],
        label_visibility="collapsed",
        help="Include all required columns such as Lead_ID, Industry, Region, Opportunity_Stage, Converted, Churned, and CLV.",
    )
    st.caption("Or try the demo dataset if you don't have a file handy.")
    _col_pad, sample_col = st.columns([1, 1])
    with sample_col:
        if st.button("Show Dashboard with Sample Data", use_container_width=True):
            st.session_state["use_sample_data"] = True

if uploaded is not None:
    st.session_state["use_sample_data"] = False

use_sample = st.session_state.get("use_sample_data", False)

if uploaded is None and not use_sample:
    st.stop()

@st.cache_data(show_spinner=False)
def load_data(file):
    return pd.read_csv(file)

if use_sample:
    df = make_sample_data()
    st.success("Loaded sample dataset. All charts and tables now use demo data.")
else:
    df = load_data(uploaded)

missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    st.error("Your file is missing these required columns: {}".format(", ".join(missing)))
    st.stop()

with st.expander("Raw data preview", expanded=False):
    st.dataframe(df.head(25))

# -------------------------------------------------
# Feature engineering + model training
# -------------------------------------------------
df_encoded = df.copy()
encoders = {}
for col in CATEGORICAL_COLS:
    encoder = LabelEncoder()
    df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
    encoders[col] = encoder

X_conv = df_encoded.drop(["Lead_ID", "Converted", "Churned", "CLV"], axis=1)
y_conv = df_encoded["Converted"]
conv_model = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)
conv_model.fit(X_conv, y_conv)
df_encoded["Pred_Convert"] = conv_model.predict_proba(X_conv)[:, 1]

X_churn = df_encoded.drop(["Lead_ID", "Converted", "Churned", "CLV", "Pred_Convert"], axis=1)
y_churn = df_encoded["Churned"]
churn_model = LogisticRegression(max_iter=1500)
churn_model.fit(X_churn, y_churn)
df_encoded["Pred_Churn"] = churn_model.predict_proba(X_churn)[:, 1]

X_clv = df_encoded.drop(
    ["Lead_ID", "Converted", "Churned", "CLV", "Pred_Convert", "Pred_Churn"], axis=1
)
y_clv = df_encoded["CLV"]
clv_model = XGBRegressor(
    n_estimators=350,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
)
clv_model.fit(X_clv, y_clv)
df_encoded["Pred_CLV"] = clv_model.predict(X_clv)

# Consolidate outputs
result_df = df.join(
    df_encoded[["Pred_Convert", "Pred_Churn", "Pred_CLV"]]
)
result_df["Expected_Revenue"] = (
    result_df["Pred_Convert"] * result_df["Pred_CLV"] * (1 - result_df["Pred_Churn"])
)

# -------------------------------------------------
# Filter deck (inline)
# -------------------------------------------------
industries = sorted(result_df["Industry"].dropna().unique())
regions = sorted(result_df["Region"].dropna().unique())
stages = sorted(result_df["Opportunity_Stage"].dropna().unique())

with st.container():
    st.markdown('<div class="filters-card"><h3>🎯 Filters</h3>', unsafe_allow_html=True)
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    selected_industries = row1_col1.multiselect("Industry", industries, default=industries)
    selected_regions = row1_col2.multiselect("Region", regions, default=regions)
    selected_stages = row1_col3.multiselect("Opportunity Stage", stages, default=stages)

    size_min, size_max = int(result_df["Company_Size"].min()), int(result_df["Company_Size"].max())
    row2_col1, row2_col2, row2_col3 = st.columns(3)
    company_size_range = row2_col1.slider(
        "Company size range",
        min_value=size_min,
        max_value=size_max,
        value=(size_min, size_max),
    )
    min_conv_pct = row2_col2.slider(
        "🎯 Min conversion probability (%)",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
    )
    min_conv = min_conv_pct / 100
    min_expected_revenue = row2_col3.number_input(
        "💰 Min expected revenue",
        min_value=0.0,
        value=0.0,
        step=5000.0,
    )
    st.markdown('</div>', unsafe_allow_html=True)

filtered_df = result_df[
    (result_df["Industry"].isin(selected_industries))
    & (result_df["Region"].isin(selected_regions))
    & (result_df["Opportunity_Stage"].isin(selected_stages))
    & (result_df["Company_Size"].between(company_size_range[0], company_size_range[1]))
    & (result_df["Pred_Convert"] >= min_conv)
    & (result_df["Expected_Revenue"] >= min_expected_revenue)
]

# -------------------------------------------------
# KPI cards
# -------------------------------------------------
st.markdown("<h2 style='color:white;margin-top:1rem;'>📊 Key Metrics</h2>", unsafe_allow_html=True)

total_leads = len(filtered_df)
share_of_total = (
    f"↑ {total_leads / len(result_df):.1%} of dataset"
    if len(result_df) > 0 and total_leads > 0
    else "No leads in view"
)
avg_conv = filtered_df["Pred_Convert"].mean() if total_leads else 0
avg_churn = filtered_df["Pred_Churn"].mean() if total_leads else 0
total_rev = filtered_df["Expected_Revenue"].sum() if total_leads else 0
rev_per_lead = total_rev / total_leads if total_leads else 0

metric_data = [
    ("Total Leads", f"{total_leads:,}", share_of_total, "#48bb78"),
    ("Avg Conversion", f"{avg_conv * 100:.1f}%", "High Priority", "#48bb78"),
    ("Avg Churn Risk", f"{avg_churn * 100:.1f}%", "↑ Needs Attention", "#f56565"),
    ("Expected Revenue", f"${total_rev:,.0f}", f"${rev_per_lead:,.0f} per lead" if total_leads else "$0 per lead", "#48bb78"),
]

metric_cols = st.columns(4)
for col, (label, value, delta, delta_color) in zip(metric_cols, metric_data):
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta" style="color:{delta_color};">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------
# Content sections (single page)
# -------------------------------------------------
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown("### 🏆 Top 15 High-Value Leads")
if filtered_df.empty:
    st.info("No leads match the current filter criteria.")
else:
    cols_to_show = [
        "Lead_ID",
        "Industry",
        "Region",
        "Company_Size",
        "Opportunity_Stage",
        "Pred_Convert",
        "Pred_Churn",
        "Pred_CLV",
        "Expected_Revenue",
    ]
    styled = filtered_df.sort_values("Expected_Revenue", ascending=False).head(15)[cols_to_show]
    styled = styled.style.format(
        {
            "Pred_Convert": "{:.1%}",
            "Pred_Churn": "{:.1%}",
            "Pred_CLV": "${:,.0f}",
            "Expected_Revenue": "${:,.0f}",
        }
    )
    st.dataframe(styled, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown("### 📈 Model Output Distributions & Relationships")
if filtered_df.empty:
    st.info("No data available for visualizations. Adjust the filters above.")
else:
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        chart = (
            alt.Chart(filtered_df)
            .mark_bar(color="#6366f1")
            .encode(x=alt.X("Pred_Convert", bin=alt.Bin(maxbins=20)), y="count()")
            .properties(height=250)
        )
        st.altair_chart(chart, use_container_width=True)
    with chart_col2:
        chart = (
            alt.Chart(filtered_df)
            .mark_bar(color="#f97316")
            .encode(x=alt.X("Pred_Churn", bin=alt.Bin(maxbins=20)), y="count()")
            .properties(height=250)
        )
        st.altair_chart(chart, use_container_width=True)
    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        chart = (
            alt.Chart(filtered_df)
            .mark_bar(color="#a855f7")
            .encode(x=alt.X("Pred_CLV", bin=alt.Bin(maxbins=20)), y="count()")
            .properties(height=250)
        )
        st.altair_chart(chart, use_container_width=True)
    with chart_col4:
        scatter = (
            alt.Chart(filtered_df)
            .mark_circle(size=65, opacity=0.8)
            .encode(
                x=alt.X("Pred_CLV", title="Predicted CLV"),
                y=alt.Y("Expected_Revenue", title="Expected Revenue"),
                color=alt.Color("Industry", legend=alt.Legend(title="Industry")),
                tooltip=[
                    "Lead_ID",
                    "Industry",
                    "Region",
                    "Company_Size",
                    "Pred_Convert",
                    "Pred_Churn",
                    "Pred_CLV",
                    "Expected_Revenue",
                ],
            )
            .interactive()
        )
        st.altair_chart(scatter, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown("### 🔍 Lead Explorer")
st.caption("Full dataset with model outputs. Use the filters above to focus on the accounts that matter.")
st.dataframe(
    filtered_df[
        [
            "Lead_ID",
            "Industry",
            "Region",
            "Company_Size",
            "Opportunity_Stage",
            "Pred_Convert",
            "Pred_Churn",
            "Pred_CLV",
            "Expected_Revenue",
            "Converted",
            "Churned",
            "CLV",
        ]
    ],
    use_container_width=True,
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown("### 🎨 Strategic Insights")
if filtered_df.empty:
    st.info("No insights to surface. Please broaden your filters.")
else:
    high_priority = filtered_df[
        (filtered_df["Pred_Convert"] >= 0.7)
        & (filtered_df["Expected_Revenue"] >= filtered_df["Expected_Revenue"].median())
    ]
    at_risk = filtered_df[
        (filtered_df["Pred_Churn"] >= 0.4)
        & (filtered_df["Pred_CLV"] >= filtered_df["Pred_CLV"].median())
    ]
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"""
            <div class="insight-card positive">
                <h3>🎯 High Priority Leads</h3>
                <div style="font-size: 3rem; font-weight: 700;">{len(high_priority)}</div>
                <p>High conversion probability & revenue potential</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f"""
            <div class="insight-card warning">
                <h3>⚠️ At-Risk Accounts</h3>
                <div style="font-size: 3rem; font-weight: 700;">{len(at_risk)}</div>
                <p>High churn risk with significant CLV</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("#### Expected Revenue by Industry")
    industry_rev = (
        filtered_df.groupby("Industry")["Expected_Revenue"].sum().reset_index().sort_values(
            "Expected_Revenue", ascending=False
        )
    )
    bar = (
        alt.Chart(industry_rev)
        .mark_bar(color="#6366f1")
        .encode(
            x=alt.X("Expected_Revenue", title="Expected Revenue"),
            y=alt.Y("Industry", sort="-x"),
            tooltip=["Industry", alt.Tooltip("Expected_Revenue", format=",.0f")],
        )
    )
    st.altair_chart(bar, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# Download block
# -------------------------------------------------
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.subheader("💾 Export Results")
st.caption(
    "Download the complete dataset with all model predictions for further analysis."
)
st.download_button(
    "📥 Download CSV",
    data=result_df.to_csv(index=False).encode("utf-8"),
    file_name="B2B_SaaS_Model_Output.csv",
    mime="text/csv",
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align:center;color:white;margin-top:2rem;">
        <p><strong>💡 Pro Tip:</strong> Focus on leads with high conversion probability and expected revenue for maximum ROI.</p>
        <p style="opacity:0.85;">Powered by Random Forest, Logistic Regression & XGBoost</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(DEV_CARD_HTML, unsafe_allow_html=True)

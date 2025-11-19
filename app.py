import streamlit as st
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
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 0 1.5rem 2rem 1.5rem;
        }
        [data-testid="stHeader"] {background: transparent;}
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        section[data-testid="stSidebar"] * {color: white !important;}
        .hero {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 25px;
            padding: 3rem 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
            margin-bottom: 2.5rem;
        }
        .hero h1 {font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem;}
        .hero p {font-size: 1.2rem; opacity: 0.95;}
        .upload-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 2.5rem;
            text-align: center;
            box-shadow: 0 20px 40px rgba(15, 23, 42, 0.15);
            margin-bottom: 2rem;
        }
        .metrics-wrapper {margin: 1.5rem 0;}
        .metric-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 18px;
            padding: 1.75rem;
            box-shadow: 0 12px 35px rgba(15, 23, 42, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.6);
        }
        .metric-label {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.8rem;
            color: #4a5568;
            margin-bottom: 0.4rem;
        }
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            color: #4c51bf;
        }
        .metric-delta {color: #48bb78; font-weight: 600;}
        .glass-panel {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 20px 45px rgba(15, 23, 42, 0.15);
            margin-bottom: 1.5rem;
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
        div[data-testid="stFileUploader"] section {text-align: center;}
        div[data-testid="stFileUploader"] label {color: #4a5568; font-weight: 600; font-size: 1rem;}
        div[data-testid="stFileUploader"] button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 12px;
            padding: 0.6rem 1.2rem;
            border: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

HERO_TITLE = "AI Lead Scoring Dashboard"
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

with st.container():
    st.markdown(
        """
        <div class="upload-card">
            <h2 style="color:#2d3748;">👋 Welcome!</h2>
            <p style="font-size:1.05rem;color:#4a5568;">Upload your B2B SaaS dataset to unlock AI-powered insights and lead scoring.</p>
            <p style="color:#718096;margin-bottom:1.2rem;">The dashboard will automatically train conversion, churn, and CLV models.</p>
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

if not uploaded:
    st.stop()

@st.cache_data(show_spinner=False)
def load_data(file):
    return pd.read_csv(file)

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
# Sidebar filters
# -------------------------------------------------
st.sidebar.header("🎯 Filters")
st.sidebar.caption("Fine-tune industries, regions, and thresholds to mirror the glossy mock UI.")

industries = sorted(result_df["Industry"].dropna().unique())
regions = sorted(result_df["Region"].dropna().unique())
stages = sorted(result_df["Opportunity_Stage"].dropna().unique())

selected_industries = st.sidebar.multiselect("Industry", industries, default=industries)
selected_regions = st.sidebar.multiselect("Region", regions, default=regions)
selected_stages = st.sidebar.multiselect("Opportunity Stage", stages, default=stages)

size_min, size_max = int(result_df["Company_Size"].min()), int(result_df["Company_Size"].max())
company_size_range = st.sidebar.slider(
    "Company size range",
    min_value=size_min,
    max_value=size_max,
    value=(size_min, size_max),
)

min_conv_pct = st.sidebar.slider(
    "🎯 Min conversion probability (%)",
    min_value=0,
    max_value=100,
    value=0,
    step=5,
)
min_conv = min_conv_pct / 100
min_expected_revenue = st.sidebar.number_input(
    "💰 Min expected revenue",
    min_value=0.0,
    value=0.0,
    step=5000.0,
)

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
# Tabs
# -------------------------------------------------
tab_top_leads, tab_analytics, tab_explorer, tab_insights = st.tabs(
    ["🏆 Top Leads", "📈 Analytics", "🔍 Lead Explorer", "🎨 Insights"]
)

with tab_top_leads:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### Top 15 High-Value Leads")
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

with tab_analytics:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### Model Output Distributions")
    if filtered_df.empty:
        st.info("No data available for visualizations. Adjust the sidebar filters.")
    else:
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            chart = (
                alt.Chart(filtered_df)
                .mark_bar(color="#667eea")
                .encode(x=alt.X("Pred_Convert", bin=alt.Bin(maxbins=20)), y="count()")
                .properties(height=250)
            )
            st.altair_chart(chart, use_container_width=True)
        with chart_col2:
            chart = (
                alt.Chart(filtered_df)
                .mark_bar(color="#f56565")
                .encode(x=alt.X("Pred_Churn", bin=alt.Bin(maxbins=20)), y="count()")
                .properties(height=250)
            )
            st.altair_chart(chart, use_container_width=True)
        chart_col3, chart_col4 = st.columns(2)
        with chart_col3:
            chart = (
                alt.Chart(filtered_df)
                .mark_bar(color="#764ba2")
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

with tab_explorer:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### Lead Explorer")
    st.caption("Full dataset with model outputs. Use sidebar filters to narrow down accounts.")
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

with tab_insights:
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
            .mark_bar(color="#667eea")
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

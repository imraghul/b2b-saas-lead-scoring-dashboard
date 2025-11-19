
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
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(0,0,0,0.05);
        box-shadow: 0px 8px 32px rgba(102, 126, 234, 0.15);
    }
    .insight-card {
        border-radius: 16px;
        padding: 1.5rem;
        color: white;
    }
    .insight-card.positive {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
    }
    .insight-card.warning {
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

HERO_TITLE = "B2B SaaS AI Lead Scoring & CLV Dashboard"
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
with st.container():
    st.markdown(
        f"""
        <div class="hero">
            <h1>🚀 {HERO_TITLE}</h1>
            <p style="font-size: 1.1rem; opacity: 0.9;">
                Intelligent lead prioritization powered by Random Forest conversion scoring, logistic churn modeling and XGBoost CLV predictions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.sidebar.header("Filters")
st.sidebar.caption("Use these knobs to narrow down high-impact accounts.")

uploaded = st.sidebar.file_uploader("Upload your B2B SaaS dataset (CSV)", type=["csv"])

if not uploaded:
    st.info(
        "⬆️ Upload **B2B_SaaS_AI_Dataset.csv** or a similar file to unlock the dashboard. "
        "Required columns: Lead_ID, Industry, Company_Size, Region, Opportunity_Stage, Converted, Churned, CLV, etc."
    )
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

min_conv = st.sidebar.slider("Minimum conversion probability", 0.0, 1.0, 0.0, 0.05)
max_churn = st.sidebar.slider("Maximum churn probability", 0.0, 1.0, 1.0, 0.05)
min_expected_revenue = st.sidebar.number_input(
    "Minimum expected revenue",
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
    & (result_df["Pred_Churn"] <= max_churn)
    & (result_df["Expected_Revenue"] >= min_expected_revenue)
]

# -------------------------------------------------
# KPI cards
# -------------------------------------------------
st.subheader("Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Leads in view", len(filtered_df))
col2.metric("Average Pred_Convert", f"{filtered_df['Pred_Convert'].mean():.2%}" if not filtered_df.empty else "0.00%")
col3.metric(
    "Average Pred_Churn",
    f"{filtered_df['Pred_Churn'].mean():.2%}" if not filtered_df.empty else "0.00%",
)
col4.metric(
    "Total Expected Revenue",
    f"${filtered_df['Expected_Revenue'].sum():,.0f}" if not filtered_df.empty else "$0",
)

# -------------------------------------------------
# Tabs
# -------------------------------------------------
tab_top_leads, tab_analytics, tab_explorer, tab_insights = st.tabs(
    ["🏆 Top Leads", "📈 Analytics", "🔍 Lead Explorer", "🎨 Insights"]
)

with tab_top_leads:
    st.markdown("### Top 15 high-value leads")
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
        st.dataframe(
            filtered_df.sort_values("Expected_Revenue", ascending=False).head(15)[cols_to_show]
        )

with tab_analytics:
    st.markdown("### Model output distributions")
    if filtered_df.empty:
        st.info("No data available for visualizations. Adjust the sidebar filters.")
    else:
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            chart = (
                alt.Chart(filtered_df)
                .mark_bar()
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

with tab_explorer:
    st.markdown("### Lead explorer")
    st.caption("Full dataset with model outputs. Apply filters from the sidebar to focus on a segment.")
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
        ]
    )

with tab_insights:
    st.markdown("### Strategic insights")
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
                    <h3>🎯 High priority leads</h3>
                    <div style="font-size: 3rem; font-weight: 700;">{len(high_priority)}</div>
                    <p>High conversion probability and high revenue potential.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(
                f"""
                <div class="insight-card warning">
                    <h3>⚠️ At-risk accounts</h3>
                    <div style="font-size: 3rem; font-weight: 700;">{len(at_risk)}</div>
                    <p>High churn probability but strong CLV signals.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("#### Industry revenue spotlight")
        industry_rev = (
            filtered_df.groupby("Industry")["Expected_Revenue"].sum().reset_index().sort_values(
                "Expected_Revenue", ascending=False
            )
        )
        bar = (
            alt.Chart(industry_rev)
            .mark_bar(color="#667eea")
            .encode(
                x=alt.X("Expected_Revenue", title="Expected revenue"),
                y=alt.Y("Industry", sort="-x"),
                tooltip=["Industry", alt.Tooltip("Expected_Revenue", format=",.0f")],
            )
        )
        st.altair_chart(bar, use_container_width=True)

# -------------------------------------------------
# Download block
# -------------------------------------------------
st.subheader("Download model outputs")
st.caption(
    "Export the enriched dataset with conversion, churn and CLV predictions for sharing or deeper analysis."
)
st.download_button(
    "📥 Download CSV",
    data=result_df.to_csv(index=False).encode("utf-8"),
    file_name="B2B_SaaS_Model_Output.csv",
    mime="text/csv",
)

st.success(
    "Pro tip: Combine high Pred_Convert with high Expected_Revenue for fast wins. High Pred_Churn + high CLV marks accounts for proactive retention."
)

import uuid
import sys
from datetime import date, datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database.connection import SessionLocal, create_tables
from database.models import Transaction
from src.services.transaction_service import process_transaction

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="FinGuard | Fraud Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = ROOT / "models" / "fraud_detection_model.joblib"
DATA_PATH = ROOT / "data" / "processed" / "processed_transactions.csv"


@st.cache_resource
def initialize_database():
    """Create the transaction table once when the dashboard starts."""
    create_tables()
    return True


initialize_database()


# ---------------- LOAD EXTERNAL CSS ----------------

CSS_PATH = Path(__file__).parent / "style.css"

with open(CSS_PATH, encoding="utf-8") as css_file:
    st.markdown(
        f"<style>{css_file.read()}</style>",
        unsafe_allow_html=True
    )


# ---------------- DATABASE DATA ----------------

@st.cache_data(ttl=10)
def load_database_transactions():
    """Load saved transactions from the FinGuard database."""
    db = SessionLocal()

    try:
        records = db.query(Transaction).order_by(
            Transaction.created_at.desc()
        ).all()

        transactions = [
            {
                "Transaction ID": row.transaction_id,
                "Customer ID": row.customer_id,
                "Amount": row.transaction_amount,
                "Location": row.location,
                "Prediction": (
                    "Fraudulent" if row.predicted_fraud else "Genuine"
                ),
                "Fraud Probability": row.fraud_probability,
                "Created At": row.created_at
            }
            for row in records
        ]

        return pd.DataFrame(transactions)

    finally:
        db.close()


# ---------------- LOAD DATA AND MODEL ----------------

if not MODEL_PATH.exists():
    st.error("Model file not found. Check the models folder.")
    st.stop()

if not DATA_PATH.exists():
    st.error("Processed dataset not found. Check the data folder.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

saved = load_model()
df = load_data()
database_df = load_database_transactions()

model = saved["model"]
feature_columns = saved["feature_columns"]
threshold = float(saved["threshold"])

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("# 🛡️ FinGuard")
    st.caption("FINANCIAL INTELLIGENCE")
    st.divider()

    page = st.radio(
    "NAVIGATION",
    [
        "Home",
        "Dataset Overview",
        "EDA",
        "Feature Engineering",
        "Executive Overview",
        "Fraud Analytics",
        "Transaction Explorer",
        "Fraud Prediction",
        "Model Information"
    ]
)

    st.divider()
    st.caption("Machine Learning: Random Forest")
    st.caption(f"Detection threshold: {threshold:.3f}")
    st.caption("Dataset: Financial Transactions")

# ---------------- COMMON METRICS ----------------

total = len(df)
fraud = int(df["Fraudulent"].sum())
genuine = total - fraud
fraud_rate = (fraud / total * 100) if total else 0

def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------- HOME PAGE ----------------

if page == "Home":

    st.title("🛡️ Welcome to FinGuard")
    st.subheader("Financial Fraud Detection & Analytics System")

    # Project Overview
    st.markdown("## Project Overview")

    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #142b3a, #163c43);
        padding: 22px;
        border-radius: 12px;
        border-left: 5px solid #2dd4bf;
        margin-bottom: 15px;
    ">
        <h3 style="color:#2dd4bf; margin-top:0;">
            FinGuard – Financial Intelligence
        </h3>
        <p style="color:#ffffff; font-size:16px;">
            FinGuard is a machine learning-based financial fraud
            detection and analytics system designed to identify
            suspicious transactions, analyze fraud patterns and
            support financial risk analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Project Objectives
    st.markdown("## Project Objectives")

    objectives = [
        ("01", "Fraud Identification",
         "Identify potentially fraudulent financial transactions."),
        ("02", "Pattern Analysis",
         "Analyze genuine and fraudulent transaction patterns."),
        ("03", "Machine Learning",
         "Apply machine learning for fraud prediction."),
        ("04", "Data Visualization",
         "Visualize transaction data through interactive charts."),
        ("05", "Risk Analysis",
         "Support better financial risk analysis.")
    ]

    for start in range(0, len(objectives), 2):
        cols = st.columns(2)

        for col, item in zip(cols, objectives[start:start + 2]):
            number, heading, description = item

            with col:
                st.markdown(
                    f"""
                    <div style="
                        background:#142b3a;
                        border:1px solid #28505a;
                        border-radius:10px;
                        padding:17px;
                        margin-bottom:12px;
                        min-height:105px;
                    ">
                        <span style="color:#2dd4bf;
                        font-size:13px;">OBJECTIVE {number}</span>
                        <h4 style="color:white;margin:7px 0;">
                            {heading}
                        </h4>
                        <p style="color:#cbd5e1;margin:0;">
                            {description}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.divider()

    # Key Features
    st.markdown("## Key Features")

    features = [
        ("📊", "Data Analytics",
         "Explore transaction patterns and fraud distribution."),
        ("🤖", "Fraud Prediction",
         "Predict potential fraud using a trained Random Forest model."),
        ("🔍", "Transaction Explorer",
         "Filter, inspect and download transaction records.")
    ]

    cols = st.columns(3)

    for col, (emoji, heading, description) in zip(cols, features):
        with col:
            st.markdown(
                f"""
                <div style="
                    background:#142b3a;
                    border:1px solid #28505a;
                    border-radius:12px;
                    padding:20px;
                    min-height:155px;
                ">
                    <div style="font-size:28px;">{emoji}</div>
                    <h4 style="color:#2dd4bf;">{heading}</h4>
                    <p style="color:#cbd5e1;">{description}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    # Technical Stack
    st.markdown("## Technical Stack")

    stack = [
        ("Programming Language", "Python"),
        ("Data Processing", "Pandas, NumPy"),
        ("Machine Learning", "Scikit-learn, Random Forest"),
        ("Dashboard", "Streamlit"),
        ("Data Visualization", "Plotly"),
        ("Database", "SQLite"),
        ("Model Storage", "Joblib"),
        ("Configuration", "YAML"),
        ("Dataset Format", "CSV")
    ]

    stack_df = pd.DataFrame(
        stack,
        columns=["Technology Category", "Tools & Technologies"]
    )

    st.dataframe(
        stack_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Project Workflow
    st.markdown("## Project Workflow")

    workflow = [
        ("01", "Data Collection", "Collect financial transaction data."),
        ("02", "Data Preprocessing", "Clean and prepare transaction data."),
        ("03", "Feature Engineering", "Create useful input features."),
        ("04", "Model Training", "Use the trained Random Forest model."),
        ("05", "Fraud Prediction", "Predict suspicious transactions."),
        ("06", "Dashboard Analytics", "Display insights and analytics.")
    ]

    workflow_df = pd.DataFrame(
        workflow,
        columns=["Step", "Process", "Description"]
    )

    st.dataframe(
        workflow_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Project Dataset
    st.markdown("## Project Dataset")

    genuine = total - fraud
    fraud_rate = (fraud / total * 100) if total else 0

    a, b, c, d = st.columns(4)

    a.metric("Total Transactions", f"{total:,}")
    b.metric("Genuine Transactions", f"{genuine:,}")
    c.metric("Fraud Cases", f"{fraud:,}")
    d.metric("Fraud Rate", f"{fraud_rate:.2f}%")

    st.info(
        "This is an academic demonstration project using a supplied "
        "financial transaction dataset. It is not connected to live "
        "banking or payment systems."
    )

    st.divider()

    # Fraud Distribution
    st.markdown("## Fraud Distribution")

    fraud_chart_data = pd.DataFrame({
        "Transaction Type": ["Genuine", "Fraudulent"],
        "Count": [genuine, fraud]
    })

    fig = px.pie(
        fraud_chart_data,
        names="Transaction Type",
        values="Count",
        hole=0.55,
        title="Genuine vs Fraudulent Transactions"
    )

    fig.update_traces(
        textinfo="label+percent",
        hovertemplate="%{label}<br>Count: %{value:,}<extra></extra>"
    )

    fig.update_layout(
        template="plotly_dark",
        height=420
    )

    chart_col, summary_col = st.columns([1.5, 1])

    with chart_col:
        st.plotly_chart(fig, use_container_width=True)

    with summary_col:
        st.markdown("### Fraud Summary")
        st.metric("Fraud Percentage", f"{fraud_rate:.2f}%")
        st.metric("Genuine Percentage", f"{100 - fraud_rate:.2f}%")
        st.metric("Total Fraud Cases", f"{fraud:,}")
        st.metric("Total Genuine Cases", f"{genuine:,}")


# ---------------- DATASET OVERVIEW ----------------

elif page == "Dataset Overview":

    raw_data_path = ROOT / "data" / "raw" / "financial_fraud_detection_dataset.csv"
    raw_df = pd.read_csv(raw_data_path)

    st.title("Dataset Overview")
    st.caption("Explore the original financial transaction dataset.")

    total_records = len(raw_df)
    total_columns = len(raw_df.columns)
    fraud_count = int(raw_df["Fraudulent"].sum())
    genuine_count = total_records - fraud_count
    fraud_percentage = (fraud_count / total_records) * 100

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Total Transactions", f"{total_records:,}")
    with c2:
        st.metric("Total Features", total_columns)
    with c3:
        st.metric("Fraudulent", f"{fraud_count:,}")
    with c4:
        st.metric("Fraud Rate", f"{fraud_percentage:.2f}%")

    st.markdown("### Dataset Information")

    info_col, quality_col = st.columns(2)

    with info_col:
        st.write("**Dataset Shape**")
        st.write(f"Rows: {total_records:,}")
        st.write(f"Columns: {total_columns}")
        st.write("**Dataset File:**")
        st.code("data/raw/financial_fraud_detection_dataset.csv")

    with quality_col:
        st.write("**Data Quality**")
        st.metric("Missing Values", int(raw_df.isnull().sum().sum()))
        st.metric("Duplicate Rows", int(raw_df.duplicated().sum()))
        st.metric("Unique Transactions", raw_df["Transaction_ID"].nunique())

    st.markdown("### Fraud Distribution")

    fraud_col, genuine_col = st.columns(2)

    with fraud_col:
        st.metric("Fraudulent Transactions", f"{fraud_count:,}")

    with genuine_col:
        st.metric("Genuine Transactions", f"{genuine_count:,}")

    st.markdown("### Original Dataset Preview")
    st.dataframe(raw_df.head(20), width="stretch")

    st.markdown("### Dataset Columns")
    columns_df = pd.DataFrame({
        "Column Name": raw_df.columns,
        "Data Type": raw_df.dtypes.astype(str).values,
        "Missing Values": raw_df.isnull().sum().values
    })
    st.dataframe(columns_df, width="stretch")


# ---------------- EDA PAGE ----------------

elif page == "EDA":

    raw_data_path = ROOT / "data" / "raw" / "financial_fraud_detection_dataset.csv"
    raw_df = pd.read_csv(raw_data_path)

    st.title("Exploratory Data Analysis")
    st.caption("Analyze transaction patterns and identify fraud trends.")

    # Fraud distribution
    st.markdown("### 1. Fraud Distribution")

    fraud_counts = raw_df["Fraudulent"].value_counts().rename(
        index={0: "Genuine", 1: "Fraudulent"}
    ).reset_index()

    fraud_counts.columns = ["Transaction Type", "Count"]

    fig1 = px.pie(
        fraud_counts,
        names="Transaction Type",
        values="Count",
        title="Genuine vs Fraudulent Transactions",
        template="plotly_dark",
        hole=0.4
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Transaction amount analysis
    st.markdown("### 2. Transaction Amount Analysis")

    fig2 = px.histogram(
        raw_df,
        x="Transaction_Amount",
        color="Fraudulent",
        nbins=30,
        title="Transaction Amount Distribution",
        template="plotly_dark",
        barmode="overlay"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Fraud by merchant category
    st.markdown("### 3. Fraud by Merchant Category")

    merchant_data = (
        raw_df.groupby("Merchant_Category")["Fraudulent"]
        .sum()
        .reset_index()
        .sort_values("Fraudulent", ascending=False)
    )

    fig3 = px.bar(
        merchant_data,
        x="Merchant_Category",
        y="Fraudulent",
        title="Fraud Cases by Merchant Category",
        template="plotly_dark"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Fraud by payment method
    st.markdown("### 4. Fraud by Payment Method")

    payment_data = (
        raw_df.groupby("Payment_Method")["Fraudulent"]
        .sum()
        .reset_index()
        .sort_values("Fraudulent", ascending=False)
    )

    fig4 = px.bar(
        payment_data,
        x="Payment_Method",
        y="Fraudulent",
        title="Fraud Cases by Payment Method",
        template="plotly_dark"
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Fraud by device type
    st.markdown("### 5. Fraud by Device Type")

    device_data = (
        raw_df.groupby("Device_Type")["Fraudulent"]
        .sum()
        .reset_index()
        .sort_values("Fraudulent", ascending=False)
    )

    fig5 = px.bar(
        device_data,
        x="Device_Type",
        y="Fraudulent",
        title="Fraud Cases by Device Type",
        template="plotly_dark"
    )
    st.plotly_chart(fig5, use_container_width=True)

    # Fraud by location
    st.markdown("### 6. Fraud by Location")

    location_data = (
        raw_df.groupby("Location")["Fraudulent"]
        .sum()
        .reset_index()
        .sort_values("Fraudulent", ascending=False)
    )

    fig6 = px.bar(
        location_data,
        x="Location",
        y="Fraudulent",
        title="Fraud Cases by Location",
        template="plotly_dark"
    )
    st.plotly_chart(fig6, use_container_width=True)


# ---------------- FEATURE ENGINEERING PAGE ----------------

elif page == "Feature Engineering":

    st.title("Feature Engineering")
    st.caption("Understand how raw transaction data is transformed into model-ready features.")

    raw_data_path = ROOT / "data" / "raw" / "financial_fraud_detection_dataset.csv"
    raw_df = pd.read_csv(raw_data_path)

    processed_df = df.copy()

    st.markdown("### 1. Raw vs Processed Dataset")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Raw Columns", len(raw_df.columns))

    with col2:
        st.metric("Processed Columns", len(processed_df.columns))

    with col3:
        st.metric("Model Input Features", len(feature_columns))

    st.markdown("### 2. Feature Engineering Steps")

    steps = [
        ("Date Feature Extraction", "Transaction date is represented through hour, day, month and weekday features."),
        ("Categorical Encoding", "Categorical fields are converted into numeric one-hot encoded columns."),
        ("Binary Feature", "Suspicious keyword information is represented as a numeric feature."),
        ("Identifier Removal", "Transaction and customer identifiers are not used as model input features."),
        ("Target Separation", "Fraudulent is the target column used to identify genuine and fraudulent transactions.")
    ]

    for title, description in steps:
        with st.expander(title):
            st.write(description)

    st.markdown("### 3. Engineered Date Features")

    date_features = [
        "Transaction_Hour",
        "Transaction_Day",
        "Transaction_Month",
        "Transaction_Weekday"
    ]

    for feature in date_features:
        if feature in processed_df.columns:
            st.write(f"✓ {feature}")

    st.markdown("### 4. Processed Feature Categories")

    numeric_features = [
        "Transaction_Amount",
        "Is_International",
        "Previous_Transactions",
        "Average_Spend",
        "Account_Age_Days",
        "Suspicious_Keyword",
        "Transaction_Hour",
        "Transaction_Day",
        "Transaction_Month",
        "Transaction_Weekday"
    ]

    categorical_prefixes = [
        "Merchant_Category_",
        "Payment_Method_",
        "Device_Type_",
        "Location_"
    ]

    numeric_count = sum(
        1 for feature in numeric_features
        if feature in feature_columns
    )

    categorical_count = sum(
        1 for feature in feature_columns
        if any(feature.startswith(prefix) for prefix in categorical_prefixes)
    )

    feature_summary = pd.DataFrame({
        "Feature Group": ["Numeric / Engineered", "One-Hot Encoded"],
        "Number of Features": [numeric_count, categorical_count]
    })

    fig = px.bar(
        feature_summary,
        x="Feature Group",
        y="Number of Features",
        title="Model Input Feature Groups",
        template="plotly_dark",
        text="Number of Features"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 5. Final Model Features")

    st.write(f"Total model input features: **{len(feature_columns)}**")

    feature_df = pd.DataFrame({
        "Feature Name": feature_columns
    })

    st.dataframe(feature_df, width="stretch")

    st.markdown("### 6. Processed Dataset Preview")

    st.dataframe(processed_df.head(20), width="stretch")

    st.markdown("### 7. Data Quality Check")

    quality_col1, quality_col2, quality_col3 = st.columns(3)

    with quality_col1:
        st.metric("Missing Values", int(processed_df.isnull().sum().sum()))

    with quality_col2:
        st.metric("Duplicate Rows", int(processed_df.duplicated().sum()))

    with quality_col3:
        st.metric("Total Records", f"{len(processed_df):,}")



# ---------------- EXECUTIVE OVERVIEW ----------------

if page == "Executive Overview":

    st.title("Financial Risk Overview")
    st.markdown(
        '<p class="section-note">Monitor transaction activity and fraud patterns.</p>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Total Transactions", f"{total:,}")
    with c2:
        metric_card("Fraudulent", f"{fraud:,}")
    with c3:
        metric_card("Genuine", f"{genuine:,}")
    with c4:
        metric_card("Fraud Rate", f"{fraud_rate:.2f}%")

    st.write("")
    left, right = st.columns(2)

    with left:
        st.subheader("Transaction Distribution")

        distribution = pd.DataFrame({
            "Type": ["Genuine", "Fraudulent"],
            "Transactions": [genuine, fraud]
        })

        fig = px.pie(
            distribution,
            names="Type",
            values="Transactions",
            hole=0.65,
            color="Type",
            color_discrete_map={
                "Genuine": "#2dd4bf",
                "Fraudulent": "#f87171"
            }
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            legend_title_text=""
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Fraud vs Genuine")

        fig = px.bar(
            distribution,
            x="Type",
            y="Transactions",
            color="Type",
            color_discrete_map={
                "Genuine": "#2dd4bf",
                "Fraudulent": "#f87171"
            }
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Dataset Records")
    st.dataframe(df.head(10), use_container_width=True)

    st.divider()
    st.subheader("Database Transactions")

    db_col1, db_col2 = st.columns(2)

    with db_col1:
        st.metric("Saved Transactions", len(database_df))

    with db_col2:
        db_fraud_count = (
            int(database_df["Prediction"].eq("Fraudulent").sum())
            if not database_df.empty else 0
        )
        st.metric("Predicted Fraud", db_fraud_count)

    if database_df.empty:
        st.info("No transactions saved in the database yet.")
    else:
        st.dataframe(database_df, use_container_width=True)


# ---------------- FRAUD ANALYTICS ----------------

elif page == "Fraud Analytics":

    st.title("Fraud Analytics")
    st.caption("Explore fraud patterns across transaction categories.")

    fraud_df = df[df["Fraudulent"] == 1].copy()

    if fraud_df.empty:
        st.info("No fraudulent records found.")

    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Fraud by Payment Method")

            payment_cols = [
                c for c in fraud_df.columns
                if c.startswith("Payment_Method_")
            ]

            if payment_cols:
                counts = fraud_df[payment_cols].sum().sort_values(
                    ascending=False
                )

                chart = pd.DataFrame({
                    "Payment Method": [
                        c.replace("Payment_Method_", "")
                        for c in counts.index
                    ],
                    "Count": counts.values
                })

                fig = px.bar(
                    chart,
                    x="Payment Method",
                    y="Count",
                    color="Count",
                    color_continuous_scale="Teal",
                    text="Count"
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Payment method data is not available.")

        with col2:
            st.subheader("Fraud by Merchant")

            merchant_cols = [
                c for c in fraud_df.columns
                if c.startswith("Merchant_Category_")
            ]

            if merchant_cols:
                counts = fraud_df[merchant_cols].sum().sort_values(
                    ascending=False
                )

                chart = pd.DataFrame({
                    "Merchant": [
                        c.replace("Merchant_Category_", "")
                        for c in counts.index
                    ],
                    "Count": counts.values
                })

                fig = px.bar(
                    chart, x="Merchant", y="Count", color="Count"
                )
                st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Fraud by Device")

            device_cols = [
                c for c in fraud_df.columns
                if c.startswith("Device_Type_")
            ]

            if device_cols:
                counts = fraud_df[device_cols].sum()

                chart = pd.DataFrame({
                    "Device": [
                        c.replace("Device_Type_", "")
                        for c in counts.index
                    ],
                    "Count": counts.values
                })

                fig = px.bar(chart, x="Device", y="Count")
                st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.subheader("International Fraud")

            if "Is_International" in fraud_df.columns:
                chart = fraud_df["Is_International"].value_counts().reset_index()
                chart.columns = ["International", "Count"]
                chart["International"] = chart["International"].map(
                    {0: "Domestic", 1: "International"}
                )

                fig = px.pie(
                    chart,
                    names="International",
                    values="Count",
                    hole=0.5
                )

                st.plotly_chart(fig, use_container_width=True)

# 
# ---------------- TRANSACTION EXPLORER ----------------

elif page == "Transaction Explorer":

    st.title("Transaction Explorer")
    st.caption(
        "Search, filter and inspect dataset and saved database transactions."
    )

    st.subheader("Dataset Transactions")

    if "Fraudulent" in df.columns:

        choice = st.selectbox(
            "Dataset Transaction Status",
            ["All", "Genuine", "Fraudulent"]
        )

        filtered = df.copy()

        if choice == "Genuine":
            filtered = filtered[filtered["Fraudulent"] == 0]

        elif choice == "Fraudulent":
            filtered = filtered[filtered["Fraudulent"] == 1]

        st.write(f"Dataset Records Found: {len(filtered):,}")

        st.dataframe(
            filtered,
            use_container_width=True
        )

        csv = filtered.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Dataset Transactions",
            data=csv,
            file_name="finguard_dataset_transactions.csv",
            mime="text/csv"
        )

    st.divider()

    st.subheader("Saved Database Transactions")

    explorer_db = load_database_transactions()

    if explorer_db.empty:

        st.info("No saved database transactions found.")

    else:

        db_choice = st.selectbox(
            "Saved Transaction Status",
            ["All", "Genuine", "Fraudulent"],
            key="database_status"
        )

        db_filtered = explorer_db.copy()

        if db_choice == "Genuine":
            db_filtered = db_filtered[
                db_filtered["Prediction"] == "Genuine"
            ]

        elif db_choice == "Fraudulent":
            db_filtered = db_filtered[
                db_filtered["Prediction"] == "Fraudulent"
            ]

        st.write(
            f"Saved Records Found: {len(db_filtered)}"
        )

        st.dataframe(
            db_filtered,
            use_container_width=True
        )

        db_csv = db_filtered.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Saved Transactions",
            data=db_csv,
            file_name="finguard_saved_transactions.csv",
            mime="text/csv"
        )

# ---------------- FRAUD PREDICTION ----------------

elif page == "Fraud Prediction":

    st.title("Fraud Prediction")
    st.caption(
        "Enter transaction details to estimate fraud risk "
        "and save the result to the FinGuard database."
    )

    with st.form("prediction_form"):

        st.subheader("Customer Information")

        customer_id = st.text_input(
            "Customer ID",
            placeholder="e.g. CUST1001"
        )

        st.subheader("Transaction Details")

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.01,
            value=1000.0
        )

        selected_date = st.date_input(
            "Transaction Date",
            value=date.today()
        )

        selected_time = st.time_input(
            "Transaction Time",
            value=datetime.now().time().replace(
                second=0, microsecond=0
            )
        )

        previous = st.number_input(
            "Previous Transactions",
            min_value=0,
            value=5
        )

        average = st.number_input(
            "Average Spend",
            min_value=0.0,
            value=1500.0
        )

        account_age = st.number_input(
            "Account Age (Days)",
            min_value=0,
            value=365
        )

        merchant = st.selectbox(
            "Merchant Category",
            [
                "Food", "Electronics", "Entertainment",
                "Grocery", "Fashion", "Health",
                "Travel", "Utilities"
            ]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "NetBanking", "PayPal", "Debit Card",
                "Credit Card", "UPI"
            ]
        )

        device = st.selectbox(
            "Device Type",
            ["Desktop", "POS", "Mobile"]
        )

        location = st.selectbox(
            "Location",
            [
                "Delhi", "Mumbai", "Hyderabad", "Pune",
                "Kolkata", "Bengaluru", "Chennai"
            ]
        )

        international = st.selectbox(
            "International Transaction",
            ["No", "Yes"]
        )

        keyword = st.selectbox(
            "Suspicious Keyword",
            ["No", "Yes"]
        )

        submitted = st.form_submit_button(
            "Analyze & Save Transaction",
            use_container_width=True
        )

    if submitted:

        if not customer_id.strip():
            st.error("Please enter a Customer ID.")

        else:

            transaction_id = (
                f"FG-{uuid.uuid4().hex[:10].upper()}"
            )

            transaction_data = {
                "Transaction_ID": transaction_id,
                "Customer_ID": customer_id.strip(),

                "Transaction_Date": (
                    selected_date.strftime("%d-%m-%Y")
                    + " "
                    + selected_time.strftime("%H:%M")
                ),

                "Transaction_Amount": float(amount),
                "Merchant_Category": merchant,
                "Payment_Method": payment,
                "Device_Type": device,
                "Location": location,

                "Is_International": (
                    1 if international == "Yes" else 0
                ),

                "Previous_Transactions": int(previous),
                "Average_Spend": float(average),
                "Account_Age_Days": int(account_age),
                "Suspicious_Keyword": keyword
            }

            try:

                result = process_transaction(transaction_data)

                load_database_transactions.clear()

                st.divider()
                st.subheader("Prediction Result")

                st.success("Transaction analyzed and saved successfully.")

                st.write(
                    f"**Transaction ID:** {result['transaction_id']}"
                )

                st.write(
                    f"**Customer ID:** {customer_id.strip()}"
                )

                st.metric(
                    "Estimated Fraud Probability",
                    f"{result['fraud_probability'] * 100:.2f}%"
                )

                if result["predicted_fraud"]:
                    st.error(
                        "High Risk: This transaction is classified "
                        "as potentially fraudulent."
                    )
                else:
                    st.success(
                        "Low Risk: This transaction is classified "
                        "as genuine."
                    )

                st.caption(
                    f"Classification threshold: "
                    f"{threshold:.3f}. "
                    "This is a model estimate, not a confirmed "
                    "fraud decision."
                )

            except Exception as error:
                st.error(f"Transaction processing failed: {error}")


# ---------------- MODEL INFORMATION ----------------

elif page == "Model Information":

    st.title("Model Information")
    st.caption("Details of the trained machine learning model.")

    st.info("Algorithm: Random Forest Classifier")
    st.info(f"Number of input features: {len(feature_columns)}")
    st.info(f"Saved classification threshold: {threshold:.3f}")

    st.subheader("Verified Test Evaluation Results")

    st.caption(
        "Results from the saved Random Forest model "
        "evaluated on 1,000 test transactions."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Accuracy", "85.90%")
    c2.metric("ROC-AUC", "0.8951")
    c3.metric("Fraud Precision", "35.48%")

    c4, c5, c6 = st.columns(3)

    c4.metric("Fraud Recall", "57.29%")
    c5.metric("Fraud F1-Score", "43.82%")
    c6.metric("Threshold", "0.395")

    st.subheader("Confusion Matrix")

    confusion_df = pd.DataFrame(
        {
            "Predicted Genuine": [804, 41],
            "Predicted Fraudulent": [100, 55]
        },
        index=["Actual Genuine", "Actual Fraudulent"]
    )

    st.dataframe(confusion_df, use_container_width=True)

    st.warning(
        "Fraud Recall is 57.29%. The model missed 41 out of "
        "96 fraudulent transactions in the test set. "
        "This model is for academic demonstration and should "
        "not be used as a standalone real-world fraud prevention system."
    )

    

# ---------------- FOOTER ----------------

st.divider()
st.caption(
    "FinGuard | Financial Fraud Detection & Analytics System | "
    "Academic project and demonstration"
)

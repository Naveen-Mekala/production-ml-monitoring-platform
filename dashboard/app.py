
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st


# Configure API URL using an environment variable.
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8000",
).rstrip("/")

FEATURES = [
    "income",
    "credit_score",
    "debt_to_income",
    "loan_amount",
    "employment_years",
    "interest_rate",
]


def api_get(path):
    """Fetch JSON data from the FastAPI backend."""
    request = Request(
        f"{API_BASE_URL}{path}",
        headers={"Accept": "application/json"},
        method="GET",
    )

    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


st.set_page_config(
    page_title="ML Model Monitoring",
    page_icon="📊",
    layout="wide",
)

st.title("Production ML & Model Monitoring Platform")
st.caption(f"Monitoring API: {API_BASE_URL}")

# Retrieve monitoring information from the API.
try:
    summary = api_get("/monitoring/summary")
except HTTPError as exc:
    st.error(
        f"The monitoring API returned HTTP {exc.code}. "
        "Check the API logs and endpoint."
    )
    st.stop()
except (URLError, TimeoutError, OSError, ValueError) as exc:
    st.error(
        "Cannot connect to the monitoring API. "
        "Check that the API is running and API_BASE_URL is correct."
    )
    st.caption(str(exc))
    st.stop()

records = summary.get("predictions", [])
metrics = summary.get("metrics", {})
drift_report = summary.get("drift", {})

# Prediction overview.
st.header("Prediction Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Predictions",
    summary.get("total_predictions", len(records)),
)
col2.metric(
    "Model Version",
    summary.get("model_version", "Unknown"),
)

if records:
    predictions_df = pd.DataFrame(records)
    risk_counts = (
        predictions_df["risk_level"]
        .value_counts()
        .reindex(["low", "medium", "high"], fill_value=0)
    )
    high_risk_count = int(risk_counts.get("high", 0))
else:
    predictions_df = pd.DataFrame()
    risk_counts = pd.Series(
        {"low": 0, "medium": 0, "high": 0}
    )
    high_risk_count = 0

col3.metric("High-Risk Predictions", high_risk_count)

if not records:
    st.info("No prediction records found. Submit a prediction to the API first.")
else:
    # Parse timestamps and order records chronologically for charts.
    predictions_df["timestamp"] = pd.to_datetime(
        predictions_df["timestamp"],
        errors="coerce",
        utc=True,
    )
    predictions_df = predictions_df.dropna(subset=["timestamp"])
    predictions_df = predictions_df.sort_values("timestamp")

    st.subheader("Risk Distribution")
    st.bar_chart(risk_counts)

    st.subheader("Prediction Activity")
    if not predictions_df.empty:
        activity = (
            predictions_df.set_index("timestamp")
            .resample("h")
            .size()
            .rename("prediction_count")
        )
        st.line_chart(activity)

    st.subheader("Input Feature Distributions")
    feature_records = [
        record.get("features", {})
        for record in records
        if isinstance(record.get("features"), dict)
    ]

    feature_df = pd.DataFrame(feature_records)

    available_features = [
        feature for feature in FEATURES
        if feature in feature_df.columns
    ]

    if available_features:
        selected_feature = st.selectbox(
            "Select a feature",
            options=available_features,
        )
        values = pd.to_numeric(
            feature_df[selected_feature],
            errors="coerce",
        ).dropna()

        if not values.empty:
            st.bar_chart(values.value_counts().sort_index())
        else:
            st.info("No valid values are available for this feature.")
    else:
        st.info("No feature data is available in the prediction history.")

    st.subheader("Prediction History")

    history_columns = [
        column
        for column in [
            "timestamp",
            "prediction",
            "default_probability",
            "risk_level",
            "model_version",
        ]
        if column in predictions_df.columns
    ]

    st.dataframe(
        predictions_df[history_columns].sort_values(
            "timestamp",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Model Evaluation Metrics")

if metrics:
    metric_cols = st.columns(min(len(metrics), 4))
    for index, (name, value) in enumerate(metrics.items()):
        with metric_cols[index % len(metric_cols)]:
            if isinstance(value, (int, float)):
                st.metric(name.replace("_", " ").title(), f"{value:.4f}")
            else:
                st.metric(name.replace("_", " ").title(), str(value))
else:
    st.info("Model metrics are not available from the API yet.")

with st.expander("Full model metrics"):
    st.json(metrics)

# Data drift monitoring.
st.divider()
st.header("Data Drift Monitoring")

if drift_report and isinstance(drift_report.get("features"), dict):
    feature_results = drift_report["features"]
    drift_df = pd.DataFrame.from_dict(
        feature_results,
        orient="index",
    )
    drift_df.index.name = "feature"
    drift_df = drift_df.reset_index()

    flagged = int(
        (drift_df["drift_detected"] == True).sum()
    ) if "drift_detected" in drift_df.columns else 0

    drift_col1, drift_col2, drift_col3 = st.columns(3)

    drift_col1.metric("Features Analyzed", len(drift_df))
    drift_col2.metric("Features Flagged", flagged)
    drift_col3.metric(
        "Current Sample Size",
        drift_report.get("current_rows", 0),
    )

    st.caption(
        f"Method: {drift_report.get('method', 'Unknown')} | "
        f"Significance level: "
        f"{drift_report.get('significance_level', 'Unknown')}"
    )

    display_columns = [
        column
        for column in [
            "feature",
            "ks_statistic",
            "p_value",
            "drift_detected",
            "status",
        ]
        if column in drift_df.columns
    ]

    st.dataframe(
        drift_df[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    if drift_report.get("current_rows", 0) < 100:
        st.warning(
            "The current sample is small. Treat drift results as "
            "provisional until more predictions are collected."
        )

    st.caption(
        drift_report.get(
            "interpretation",
            "Review drift results before drawing conclusions.",
        )
    )
else:
    st.info(
        "No drift report is available yet. The monitoring API must "
        "generate or provide a drift report before it can be displayed."
    )

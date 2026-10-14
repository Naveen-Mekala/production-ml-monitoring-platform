import json
from pathlib import Path

import pandas as pd
import streamlit as st

LOG_PATH = Path("data/prediction_logs.jsonl")
METRICS_PATH = Path("data/model_metrics.json")
DRIFT_PATH = Path("data/drift_report.json")

st.set_page_config(
    page_title="ML Model Monitoring",
    page_icon="📊",
    layout="wide",
)

st.title("Production ML Model Monitoring")
st.caption("Loan Default Prediction | Model Version 1.0.0")

if LOG_PATH.exists():
    with LOG_PATH.open("r", encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    predictions = pd.DataFrame(records)

    if not predictions.empty:
        predictions["timestamp"] = pd.to_datetime(
            predictions["timestamp"], utc=True
        )
        predictions["default_probability"] = pd.to_numeric(
            predictions["default_probability"]
        )
        predictions["prediction"] = pd.to_numeric(
            predictions["prediction"]
        )

        st.subheader("Prediction Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Predictions", len(predictions))
        col2.metric(
            "Predicted Default Rate",
            f"{predictions['prediction'].mean():.1%}",
        )
        col3.metric(
            "Average Default Probability",
            f"{predictions['default_probability'].mean():.1%}",
        )
        col4.metric(
            "High-Risk Predictions",
            int((predictions["risk_level"] == "high").sum()),
        )

        st.subheader("Risk Distribution")
        risk_counts = predictions["risk_level"].value_counts().reindex(
            ["low", "medium", "high"], fill_value=0
        )
        st.bar_chart(risk_counts)

        st.subheader("Prediction Activity")
        activity = (
            predictions.set_index("timestamp")
            .resample("h")
            .size()
            .rename("prediction_count")
        )
        st.line_chart(activity)

        st.subheader("Input Feature Distributions")
        feature_df = pd.DataFrame(
            [record["features"] for record in records]
        )
        selected_feature = st.selectbox(
            "Select a feature",
            options=list(feature_df.columns),
        )
        st.bar_chart(feature_df[selected_feature].value_counts().sort_index())

        st.subheader("Prediction History")
        st.dataframe(
            predictions[
                [
                    "timestamp",
                    "prediction",
                    "default_probability",
                    "risk_level",
                    "model_version",
                ]
            ].sort_values("timestamp", ascending=False),
            use_container_width=True,
        )

        with st.expander("Model Evaluation Metrics"):
            if METRICS_PATH.exists():
                with METRICS_PATH.open("r", encoding="utf-8") as file:
                    st.json(json.load(file))
            else:
                st.info("Model metrics file not found.")

    else:
        st.info("No prediction records found. Call POST /predict first.")
else:
    st.info("Prediction log not found. Make a prediction through the API first.")

st.divider()
st.header("Data Drift Monitoring")

if DRIFT_PATH.exists():
    with DRIFT_PATH.open("r", encoding="utf-8") as file:
        drift_report = json.load(file)

    feature_results = drift_report["features"]
    drift_df = pd.DataFrame.from_dict(feature_results, orient="index")
    drift_df.index.name = "feature"
    drift_df = drift_df.reset_index()

    flagged = int(
        (drift_df["drift_detected"] == True).sum()
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Features Analyzed", len(drift_df))
    col2.metric("Features Flagged", flagged)
    col3.metric(
        "Current Sample Size",
        drift_report["current_rows"],
    )

    st.caption(
        f"Method: {drift_report['method']} | "
        f"Significance level: {drift_report['significance_level']}"
    )

    st.dataframe(
        drift_df[
            ["feature", "ks_statistic", "p_value", "drift_detected", "status"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    if drift_report["current_rows"] < 100:
        st.warning(
            "The current sample is very small. Treat these results as "
            "provisional; collect more predictions before drawing conclusions."
        )

    st.caption(drift_report["interpretation"])
else:
    st.info(
        "No drift report found. Run `python monitoring/drift_detector.py` "
        "to generate one."
    )

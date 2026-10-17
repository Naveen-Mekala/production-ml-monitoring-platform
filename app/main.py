
import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from app.schemas.prediction import LoanApplication, PredictionResponse
from monitoring.prediction_logger import log_prediction


# Resolve paths from the project root, regardless of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "app/models/loan_default_model.joblib"
LOG_PATH = PROJECT_ROOT / "data/prediction_logs.jsonl"
METRICS_PATH = PROJECT_ROOT / "data/model_metrics.json"
DRIFT_PATH = PROJECT_ROOT / "data/drift_report.json"

FEATURES = [
    "income",
    "credit_score",
    "debt_to_income",
    "loan_amount",
    "employment_years",
    "interest_rate",
]

MODEL_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model when the API starts."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run: python src/train.py"
        )

    app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="Production ML & Model Monitoring API",
    description=(
        "Loan default prediction API with prediction logging "
        "and monitoring."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Check whether the API is running."""
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
    }


@app.get("/model/info")
def model_info():
    """Return model details and expected input features."""
    return {
        "model_name": "RandomForestClassifier",
        "model_version": MODEL_VERSION,
        "features": FEATURES,
    }


@app.get("/monitoring/summary")
def monitoring_summary(
    limit: int = Query(default=100, ge=1, le=1000),
):
    """Return recent predictions, model metrics, and saved drift results."""
    predictions = []

    # Read prediction history, ignoring malformed individual log lines.
    if LOG_PATH.exists():
        with LOG_PATH.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    predictions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Read evaluation metrics.
    metrics = {}
    if METRICS_PATH.exists():
        with METRICS_PATH.open("r", encoding="utf-8") as file:
            metrics = json.load(file)

    # Read the latest saved drift report, if one exists.
    drift = {}
    if DRIFT_PATH.exists():
        with DRIFT_PATH.open("r", encoding="utf-8") as file:
            drift = json.load(file)

    return {
        "model_version": MODEL_VERSION,
        "total_predictions": len(predictions),
        "predictions": predictions[-limit:][::-1],
        "metrics": metrics,
        "drift": drift,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    """Predict loan default risk and log the successful prediction."""
    try:
        features = application.model_dump()
        row = pd.DataFrame([features], columns=FEATURES)

        model = app.state.model
        prediction = int(model.predict(row)[0])
        probability = float(model.predict_proba(row)[0][1])

        if probability >= 0.70:
            risk_level = "high"
        elif probability >= 0.35:
            risk_level = "medium"
        else:
            risk_level = "low"

        response = PredictionResponse(
            prediction=prediction,
            default_probability=round(probability, 4),
            risk_level=risk_level,
            model_version=MODEL_VERSION,
        )

        log_prediction(
            features=features,
            prediction=prediction,
            default_probability=response.default_probability,
            risk_level=risk_level,
            model_version=MODEL_VERSION,
        )

        return response

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc

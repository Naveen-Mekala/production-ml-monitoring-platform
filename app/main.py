from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas.prediction import LoanApplication, PredictionResponse
from monitoring.prediction_logger import log_prediction

MODEL_PATH = Path("app/models/loan_default_model.joblib")
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
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run: python src/train.py"
        )
    app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="Production ML & Model Monitoring API",
    description="Loan default prediction API with prediction logging.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "healthy", "model_version": MODEL_VERSION}


@app.get("/model/info")
def model_info():
    return {
        "model_name": "RandomForestClassifier",
        "model_version": MODEL_VERSION,
        "features": FEATURES,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
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

import json
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("data/prediction_logs.jsonl")


def log_prediction(
    features: dict,
    prediction: int,
    default_probability: float,
    risk_level: str,
    model_version: str,
) -> None:
    """Append one prediction event to a JSON Lines log."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "prediction": prediction,
        "default_probability": default_probability,
        "risk_level": risk_level,
        "model_version": model_version,
    }

    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event) + "\n")

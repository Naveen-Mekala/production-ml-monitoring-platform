
from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_model_info_endpoint():
    with TestClient(app) as client:
        response = client.get("/model/info")

    assert response.status_code == 200
    assert response.json()["model_version"] == "1.0.0"
    assert len(response.json()["features"]) == 6


def test_prediction_endpoint():
    payload = {
        "income": 50000,
        "credit_score": 650,
        "debt_to_income": 0.3,
        "loan_amount": 20000,
        "employment_years": 5,
        "interest_rate": 8.5,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in [0, 1]
    assert 0 <= result["default_probability"] <= 1
    assert result["risk_level"] in ["low", "medium", "high"]
    assert result["model_version"] == "1.0.0"


def test_invalid_credit_score_is_rejected():
    payload = {
        "income": 50000,
        "credit_score": 950,
        "debt_to_income": 0.3,
        "loan_amount": 20000,
        "employment_years": 5,
        "interest_rate": 8.5,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_unknown_input_field_is_rejected():
    payload = {
        "income": 50000,
        "credit_score": 650,
        "debt_to_income": 0.3,
        "loan_amount": 20000,
        "employment_years": 5,
        "interest_rate": 8.5,
        "unexpected_field": 123,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_monitoring_summary_endpoint():
    with TestClient(app) as client:
        response = client.get("/monitoring/summary")

    assert response.status_code == 200

    result = response.json()
    assert result["model_version"] == "1.0.0"
    assert "total_predictions" in result
    assert "predictions" in result
    assert "metrics" in result
    assert "drift" in result
    assert isinstance(result["predictions"], list)

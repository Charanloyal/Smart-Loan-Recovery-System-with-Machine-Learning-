import pytest
from fastapi import status

@pytest.mark.asyncio
async def test_serve_dashboard(client):
    response = await client.get("/")
    assert response.status_code == status.HTTP_200_OK
    assert "text/html" in response.headers["content-type"]
    assert "SmartRecovery" in response.text

@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    response = await client.get("/api/v1/metrics")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "classification" in data
    assert "regression" in data
    assert "best_classification" in data
    assert "best_regression" in data

@pytest.mark.asyncio
async def test_predict_endpoint_success(client):
    payload = {
        "credit_score": 720,
        "annual_income": 80000.0,
        "loan_amount": 10000.0,
        "interest_rate": 8.5,
        "debt_to_income": 0.15,
        "employment_length": 6,
        "home_ownership": "MORTGAGE",
        "loan_purpose": "credit_card"
    }
    response = await client.post("/api/v1/predict", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "success"
    assert "default_probability" in data
    assert "risk_tier" in data
    assert "recommended_action" in data
    assert "predicted_recovery_rate" in data
    assert "expected_recovery_amount" in data

@pytest.mark.asyncio
async def test_predict_endpoint_validation_error(client):
    # Invalid credit score (outside FICO range 300-850)
    payload = {
        "credit_score": 900,
        "annual_income": 80000.0,
        "loan_amount": 10000.0,
        "interest_rate": 8.5,
        "debt_to_income": 0.15,
        "employment_length": 6,
        "home_ownership": "MORTGAGE",
        "loan_purpose": "credit_card"
    }
    response = await client.post("/api/v1/predict", json=payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

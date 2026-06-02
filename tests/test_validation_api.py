from fastapi.testclient import TestClient

from oracle.validation_api import app


def test_oracle_health_endpoint():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}


def test_oracle_validate_endpoint_accepts_strong_synthetic_returns():
    client = TestClient(app)
    returns = [0.001 + ((i % 5) - 2) * 0.0001 for i in range(260)]
    response = client.post(
        "/validate",
        json={
            "model_state": {"w": [1, 2, 3]},
            "returns": returns,
            "min_sharpe": 0.1,
            "max_drawdown": -0.5,
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["validated"] is True
    assert body["validation_criteria"]["sharpe_significant"] is True
    assert body["model_hash"].startswith("0x")

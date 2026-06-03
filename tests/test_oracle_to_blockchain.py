import json

from fastapi.testclient import TestClient

from oracle.validation_api import app


def test_oracle_to_blockchain_anchor_file_event(tmp_path, monkeypatch):
    monkeypatch.setenv("FEDALPHA_REPORT_DIR", str(tmp_path))
    client = TestClient(app)
    returns = [0.001 + ((idx % 5) - 2) * 0.0001 for idx in range(260)]

    response = client.post(
        "/validate",
        json={
            "model_state": {"coef": [0.1, 0.2]},
            "returns": returns,
            "min_sharpe": 0.1,
            "max_drawdown": -0.5,
            "round_id": 9,
            "anchor_blockchain": True,
        },
    )

    body = response.json()
    events = json.loads((tmp_path / "blockchain_events.json").read_text(encoding="utf-8"))
    assert response.status_code == 200
    assert body["blockchain_anchor"]["round_id"] == 9
    assert events[-1]["model_hash"] == body["model_hash"]
    assert events[-1]["validated"] is True

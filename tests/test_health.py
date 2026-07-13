from fastapi.testclient import TestClient

from app.main import app


def test_health(monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test")
    monkeypatch.setenv("ROBOFLOW_WORKSPACE", "harsh-chakravarti")
    monkeypatch.setenv("ROBOFLOW_WORKFLOW_ID", "helmet-detection-safety-monitoring-1783929349435")

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["runtime"] == "roboflow-workflows"

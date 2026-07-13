"""
Smoke test for RoboflowWorkflowClient.predict().

This mocks InferenceHTTPClient.run_workflow with a payload shaped exactly
like the real response captured from the live "Helmet Detection Safety
Monitoring" workflow (workspace: harsh-chakravarti, workflow_id:
helmet-detection-safety-monitoring-1783929349435), via Roboflow's
workflows_run tool. The live capture returned zero detections (the sample
photo had no people/helmets in frame), so one synthetic "no helmet"
detection with a SAM3-style polygon `points` list is added here to exercise
the bbox-from-polygon path and confirm raw points are never leaked.

This test runs fully offline. A separate, optional live check against the
real Roboflow endpoint is in scripts/live_smoke_test.py — see its docstring
for why that one isn't part of this automated test file.
"""

from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.config import Settings
from app.roboflow_client import (
    RoboflowWorkflowClient,
    RoboflowWorkflowUnauthorizedError,
    save_annotated_image,
)

# Real output shape confirmed via Roboflow workflows_run, with one synthetic
# detection appended (see module docstring).
FAKE_OUTPUT_IMAGE_B64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBD"  # truncated stand-in bytes

REAL_WORKFLOW_RESPONSE = [
    {
        "output_image": FAKE_OUTPUT_IMAGE_B64,
        "predictions": {
            "image": {"width": 612, "height": 408},
            "predictions": [
                {
                    "class": "no helmet",
                    "confidence": 0.87,
                    "points": [
                        {"x": 100.0, "y": 50.0},
                        {"x": 160.0, "y": 55.0},
                        {"x": 158.0, "y": 120.0},
                        {"x": 98.0, "y": 118.0},
                    ],
                }
            ],
        },
        "compliance_summary": {
            "persons": 1,
            "helmet": 0,
            "no_helmet": 1,
            "violations": 1,
            "unknown": 0,
            "status": "violation",
        },
        "person_count": 1,
        "helmet_count": 0,
        "no_helmet_count": 1,
        "violation_count": 1,
        "safety_status": "violation",
        "vision_events_error_status": False,
        "vision_events_message": "Vision event sent successfully",
    }
]


def _make_client() -> RoboflowWorkflowClient:
    settings = Settings(
        roboflow_api_key="test-key",
        roboflow_workspace="harsh-chakravarti",
        roboflow_workflow_id="helmet-detection-safety-monitoring-1783929349435",
    )
    return RoboflowWorkflowClient(settings)


def test_predict_returns_expected_output_keys():
    client = _make_client()
    sample_image = Image.new("RGB", (612, 408), color=(120, 120, 120))

    with patch.object(client.client, "run_workflow", return_value=REAL_WORKFLOW_RESPONSE) as mocked:
        result = client.predict(sample_image, include_annotated_image=True)

    mocked.assert_called_once()

    # Expected top-level keys, per the confirmed workflow output.
    for key in (
        "image",
        "latency_ms",
        "detections",
        "compliance",
        "safety_status",
        "output_image_base64",
        "vision_events",
    ):
        assert key in result, f"missing expected key: {key}"

    assert result["safety_status"] == "violation"
    assert result["compliance"]["violations"] == 1
    assert result["vision_events"]["message"] == "Vision event sent successfully"
    assert result["output_image_base64"] == FAKE_OUTPUT_IMAGE_B64


def test_predict_derives_bbox_from_polygon_and_drops_raw_points():
    client = _make_client()
    sample_image = Image.new("RGB", (612, 408), color=(120, 120, 120))

    with patch.object(client.client, "run_workflow", return_value=REAL_WORKFLOW_RESPONSE):
        result = client.predict(sample_image)

    assert len(result["detections"]) == 1
    detection = result["detections"][0]

    assert detection["class"] == "no helmet"
    assert detection["confidence"] == 0.87

    bbox = detection["bbox"]
    assert bbox["x1"] == 98.0
    assert bbox["y1"] == 50.0
    assert bbox["x2"] == 160.0
    assert bbox["y2"] == 120.0

    # The raw polygon must never be forwarded in the response.
    assert "points" not in detection
    assert "raw" not in detection


def test_predict_retries_then_succeeds(monkeypatch):
    client = _make_client()
    client.settings.roboflow_retry_backoff_seconds = 0  # keep the test fast
    sample_image = Image.new("RGB", (10, 10))

    calls = {"count": 0}

    def flaky_run_workflow(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] < 2:
            raise ConnectionError("simulated transient network error")
        return REAL_WORKFLOW_RESPONSE

    with patch.object(client.client, "run_workflow", side_effect=flaky_run_workflow):
        result = client.predict(sample_image)

    assert calls["count"] == 2
    assert result["safety_status"] == "violation"


def test_predict_fails_fast_on_unauthorized_api_key():
    client = _make_client()
    client.settings.roboflow_retry_backoff_seconds = 0
    sample_image = Image.new("RGB", (10, 10))

    class UnauthorizedError(Exception):
        status_code = 401

    with patch.object(client.client, "run_workflow", side_effect=UnauthorizedError("unauthorized")):
        try:
            client.predict(sample_image)
        except RoboflowWorkflowUnauthorizedError as exc:
            assert "not authorized" in str(exc)
        else:
            raise AssertionError("expected RoboflowWorkflowUnauthorizedError")


def test_save_annotated_image_writes_bytes_to_disk(tmp_path: Path):
    import base64

    payload = base64.b64encode(b"not-a-real-jpeg-but-thats-fine-for-this-test")
    destination = tmp_path / "annotated" / "out.jpg"

    result_path = save_annotated_image(payload.decode("ascii"), destination)

    assert result_path == destination
    assert destination.exists()
    assert base64.b64decode(payload) == destination.read_bytes()

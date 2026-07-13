"""
Client for the "Helmet Detection Safety Monitoring" Roboflow Workflow.

Workflow ground truth (confirmed via Roboflow's workflows_get / workflows_run,
not assumed):

- Single declared input: `image`. No runtime parameters are defined on this
  workflow, so nothing needs to be passed via `parameters=`.
- The detector step (`sam3_helmet_detector`, a SAM3 block with class_names
  ["helmet", "no helmet", "person"]) runs with output_format="polygons".
  That means each entry in `predictions.predictions` carries a `points`
  polygon, NOT a native x/y/width/height box. `_bbox_from_prediction` derives
  an axis-aligned box from those points and intentionally drops the raw
  point list afterwards (a single detection's polygon can be large, and nothing
  downstream in this API needs to render arbitrary polygons).
- Output keys returned per image: output_image, predictions, compliance_summary,
  person_count, helmet_count, no_helmet_count, violation_count, safety_status,
  vision_events_error_status, vision_events_message.
- `output_image` is a raw base64 string (no "data:image/...;base64," prefix).
"""

import base64
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Optional

from inference_sdk import InferenceHTTPClient
from PIL import Image

from app.config import Settings

logger = logging.getLogger(__name__)


class RoboflowWorkflowError(Exception):
    """Raised when the Roboflow workflow call fails after retries are exhausted."""


class RoboflowWorkflowTimeoutError(RoboflowWorkflowError):
    """Raised when every attempt at the Roboflow workflow call exceeded the configured timeout."""


class RoboflowWorkflowUnauthorizedError(RoboflowWorkflowError):
    """Raised when Roboflow rejects the request because the API key is unauthorized."""


def _as_output_dict(result: Any) -> dict[str, Any]:
    """
    workflows_run/run_workflow returns a list with one output dict per input
    image. This normalizes a single-image call into that first output dict.
    """
    if isinstance(result, list) and result:
        if isinstance(result[0], dict):
            return result[0]
    if isinstance(result, dict):
        return result
    return {}


def _extract_base64_image(output_image: Any) -> Optional[str]:
    """
    Handles both shapes seen from workflow image outputs:
    - output_image: "base64..."                              (confirmed shape for this workflow)
    - output_image: {"type": "base64", "value": "base64..."}  (defensive fallback)
    """
    if isinstance(output_image, str):
        return output_image
    if isinstance(output_image, dict):
        value = output_image.get("value")
        if isinstance(value, str):
            return value
    return None


def _bbox_from_points(points: Any) -> dict[str, Optional[float]]:
    """Derive an axis-aligned bounding box from a SAM3 polygon point list.

    Deliberately does NOT retain the raw points afterwards: a single
    detection's polygon can carry hundreds of coordinate pairs, and nothing
    downstream in this API renders arbitrary polygons, so keeping them would
    only bloat every response.
    """
    xs: list[float] = []
    ys: list[float] = []

    for p in points or []:
        if isinstance(p, dict):
            x, y = p.get("x"), p.get("y")
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            x, y = p[0], p[1]
        else:
            continue
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))

    if not xs or not ys:
        return {"x1": None, "y1": None, "x2": None, "y2": None, "width": None, "height": None}

    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": x2 - x1, "height": y2 - y1}


def _bbox_from_prediction(pred: dict[str, Any]) -> dict[str, Optional[float]]:
    # Primary path: this workflow's detector runs output_format="polygons".
    points = pred.get("points")
    if points:
        return _bbox_from_points(points)

    # Fallback, in case the detector step is ever swapped for a native
    # bounding-box model (center x/y/width/height).
    x, y, width, height = pred.get("x"), pred.get("y"), pred.get("width"), pred.get("height")
    if all(v is not None for v in [x, y, width, height]):
        x1 = float(x) - float(width) / 2
        y1 = float(y) - float(height) / 2
        return {
            "x1": x1,
            "y1": y1,
            "x2": x1 + float(width),
            "y2": y1 + float(height),
            "width": float(width),
            "height": float(height),
        }

    return {"x1": None, "y1": None, "x2": None, "y2": None, "width": None, "height": None}


def _extract_detections(predictions_output: Any) -> list[dict[str, Any]]:
    if not isinstance(predictions_output, dict):
        return []

    raw_predictions = predictions_output.get("predictions", [])
    detections: list[dict[str, Any]] = []

    for pred in raw_predictions:
        if not isinstance(pred, dict):
            continue

        class_name = pred.get("class") or pred.get("class_name") or "unknown"
        confidence = float(pred.get("confidence", 0.0))
        bbox = _bbox_from_prediction(pred)

        detections.append({"class": class_name, "confidence": confidence, "bbox": bbox})

    return detections


def _is_unauthorized_error(exc: BaseException) -> bool:
    """Detect a Roboflow auth failure without depending on the SDK's exception type."""
    seen: set[int] = set()
    current: BaseException | None = exc

    while current is not None and id(current) not in seen:
        seen.add(id(current))

        status_code = getattr(current, "status_code", None)
        if status_code == 401:
            return True

        message = str(current)
        if "Unauthorized api_key" in message or "status_code=401" in message or "401 Client Error" in message:
            return True

        current = current.__cause__ or current.__context__

    return False


def save_annotated_image(base64_image: str, destination: Path) -> Path:
    """Decode a base64 output_image and write it straight to disk.

    Callers should use this instead of holding the decoded bytes in memory
    any longer than needed, and should never log `base64_image` itself.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    image_bytes = base64.b64decode(base64_image)
    destination.write_bytes(image_bytes)
    return destination


class RoboflowWorkflowClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = InferenceHTTPClient(
            api_url=settings.roboflow_api_url,
            api_key=settings.roboflow_api_key,
        )
        # One worker is enough: this executor exists solely to give the
        # (otherwise timeout-less) SDK call a hard deadline per attempt.
        self._executor = ThreadPoolExecutor(max_workers=1)

    def close(self) -> None:
        self._executor.shutdown(wait=False)

    def _call_workflow_once(self, image: Image.Image) -> Any:
        return self.client.run_workflow(
            workspace_name=self.settings.roboflow_workspace,
            workflow_id=self.settings.roboflow_workflow_id,
            images={"image": image},
            use_cache=self.settings.roboflow_use_cache,
        )

    def _run_workflow_with_retries(self, image: Image.Image) -> Any:
        max_attempts = self.settings.roboflow_max_retries + 1
        last_exc: Optional[BaseException] = None
        timed_out = False

        for attempt in range(1, max_attempts + 1):
            future = self._executor.submit(self._call_workflow_once, image)
            try:
                return future.result(timeout=self.settings.roboflow_request_timeout_seconds)
            except FutureTimeoutError as exc:
                timed_out = True
                last_exc = exc
                future.cancel()
                logger.warning(
                    "Roboflow workflow call timed out after %.1fs (attempt %s/%s)",
                    self.settings.roboflow_request_timeout_seconds,
                    attempt,
                    max_attempts,
                )
            except Exception as exc:  # noqa: BLE001 - normalized into RoboflowWorkflowError below
                timed_out = False
                last_exc = exc
                if _is_unauthorized_error(exc):
                    raise RoboflowWorkflowUnauthorizedError(
                        "Roboflow API key is not authorized for serverless inference."
                    ) from exc
                logger.warning(
                    "Roboflow workflow call failed (attempt %s/%s): %s",
                    attempt,
                    max_attempts,
                    exc,
                )

            if attempt < max_attempts:
                time.sleep(self.settings.roboflow_retry_backoff_seconds * attempt)

        error_cls = RoboflowWorkflowTimeoutError if timed_out else RoboflowWorkflowError
        raise error_cls(
            f"Roboflow workflow call failed after {max_attempts} attempt(s): {last_exc}"
        ) from last_exc

    def predict(self, image: Image.Image, include_annotated_image: bool = False) -> dict[str, Any]:
        started = time.perf_counter()

        result = self._run_workflow_with_retries(image)

        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        output = _as_output_dict(result)

        predictions_output = output.get("predictions", {})
        detections = _extract_detections(predictions_output)

        compliance = output.get("compliance_summary") or {
            "persons": int(output.get("person_count", 0) or 0),
            "helmet": int(output.get("helmet_count", 0) or 0),
            "no_helmet": int(output.get("no_helmet_count", 0) or 0),
            "violations": int(output.get("violation_count", 0) or 0),
            "unknown": 0,
            "status": output.get("safety_status", "clear"),
        }

        output_image_base64 = None
        if include_annotated_image:
            output_image_base64 = _extract_base64_image(output.get("output_image"))

        image_info = predictions_output.get("image") if isinstance(predictions_output, dict) else {}
        if not image_info:
            image_info = {"width": image.width, "height": image.height}

        return {
            "image": image_info,
            "latency_ms": latency_ms,
            "detections": detections,
            "compliance": compliance,
            "safety_status": output.get("safety_status", compliance.get("status", "clear")),
            "output_image_base64": output_image_base64,
            "vision_events": {
                "error_status": output.get("vision_events_error_status"),
                "message": output.get("vision_events_message"),
            },
        }

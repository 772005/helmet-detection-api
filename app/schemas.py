from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    """Axis-aligned box. Derived from the workflow's polygon `points` output
    (SAM3 runs with output_format="polygons"), or from native x/y/width/height
    if the detector step is ever swapped for a bounding-box model.
    """

    x1: Optional[float] = None
    y1: Optional[float] = None
    x2: Optional[float] = None
    y2: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None


class Detection(BaseModel):
    class_name: str = Field(alias="class")
    confidence: float
    bbox: BoundingBox

    model_config = ConfigDict(populate_by_name=True)


class ComplianceSummary(BaseModel):
    persons: int = 0
    helmet: int = 0
    no_helmet: int = 0
    violations: int = 0
    unknown: int = 0
    status: str = "clear"


class VisionEventsStatus(BaseModel):
    error_status: Optional[bool] = None
    message: Optional[str] = None


class PredictResponse(BaseModel):
    image: dict[str, Any]
    latency_ms: float
    detections: list[Detection]
    compliance: ComplianceSummary
    safety_status: str
    output_image_base64: Optional[str] = None
    vision_events: VisionEventsStatus = VisionEventsStatus()

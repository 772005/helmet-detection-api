"""
Optional LIVE smoke test against the real Roboflow endpoint.

Why this isn't in tests/ and isn't run automatically:
- It needs a real ROBOFLOW_API_KEY with access to the harsh-chakravarti
  workspace.
- The sandbox this integration was built in has no network egress to
  serverless.roboflow.com, so this script has NOT been executed by the
  assistant that wrote it. tests/test_roboflow_client.py covers the parsing
  logic offline against a real captured response instead; run this script
  yourself once you have credentials and network access.

Usage:
    export ROBOFLOW_API_KEY=your_real_key
    python scripts/live_smoke_test.py path/to/sample.jpg
"""

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.roboflow_client import RoboflowWorkflowClient, save_annotated_image  # noqa: E402

EXPECTED_KEYS = {
    "image",
    "latency_ms",
    "detections",
    "compliance",
    "safety_status",
    "output_image_base64",
    "vision_events",
}


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: python {sys.argv[0]} path/to/sample.jpg")
        raise SystemExit(1)

    image_path = Path(sys.argv[1])
    image = Image.open(image_path).convert("RGB")

    settings = get_settings()
    client = RoboflowWorkflowClient(settings)

    result = client.predict(image, include_annotated_image=True)

    missing = EXPECTED_KEYS - result.keys()
    assert not missing, f"missing expected output keys: {missing}"

    print("safety_status:", result["safety_status"])
    print("compliance:", result["compliance"])
    print("detections found:", len(result["detections"]))
    print("vision_events:", result["vision_events"])

    if result["output_image_base64"]:
        out_path = save_annotated_image(
            result["output_image_base64"], Path("tmp_outputs") / "annotated.jpg"
        )
        print("annotated image written to:", out_path)

    client.close()
    print("OK: all expected keys present.")


if __name__ == "__main__":
    main()

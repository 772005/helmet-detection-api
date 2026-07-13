# Helmet Detection Safety API

FastAPI service that wraps the **Helmet Detection Safety Monitoring** Roboflow Workflow.

- Workspace: `harsh-chakravarti`
- Workflow ID: `helmet-detection-safety-monitoring-1783929349435`
- Detector: SAM3 zero-shot, prompted with classes `helmet`, `no helmet`, `person`, run with `output_format="polygons"`

This README, the workflow ID/output keys, and the response example below are grounded in the workflow's real definition and a real captured run (via Roboflow's `workflows_get` / `workflows_run`), not assumed from documentation alone.

## What the workflow returns

Per image, the workflow returns:

```json
{
  "output_image": "<base64 JPEG, no data:... prefix>",
  "predictions": {
    "image": { "width": 612, "height": 408 },
    "predictions": [
      { "class": "no helmet", "confidence": 0.87, "points": [ /* polygon */ ] }
    ]
  },
  "compliance_summary": { "persons": 1, "helmet": 0, "no_helmet": 1, "violations": 1, "unknown": 0, "status": "violation" },
  "person_count": 1,
  "helmet_count": 0,
  "no_helmet_count": 1,
  "violation_count": 1,
  "safety_status": "violation",
  "vision_events_error_status": false,
  "vision_events_message": "Vision event sent successfully"
}
```

Important detail: the SAM3 detector step runs with `output_format="polygons"`, so each detection carries a `points` polygon rather than a native bounding box. `app/roboflow_client.py` derives an axis-aligned bounding box from those points and **does not forward the raw polygon** — a single detection's point list can be large, and nothing in this API renders arbitrary polygons.

## Architecture

```text
Image Upload
   ↓
FastAPI /predict
   ↓
Roboflow Workflow (SAM3: helmet / no helmet / person, polygon output)
   ↓
Compliance summary + annotated image + Vision Events
   ↓
JSON response (bounding boxes derived from polygons; raw points dropped)
```

## Reliability

The `inference-sdk` client has no first-class per-call HTTP timeout, so `RoboflowWorkflowClient` enforces one itself: each call runs in a worker thread with a hard deadline (`ROBOFLOW_REQUEST_TIMEOUT_SECONDS`), retried up to `ROBOFLOW_MAX_RETRIES` times with linear backoff (`ROBOFLOW_RETRY_BACKOFF_SECONDS`). Failures are raised as typed errors — `RoboflowWorkflowTimeoutError` or `RoboflowWorkflowError` — which `/predict` maps to `504`/`502` respectively, instead of leaking raw SDK exceptions.

By default `ROBOFLOW_USE_CACHE=false`, since this workflow performs a live safety check (and can trigger a Vision Events alert) — you generally don't want a stale cached result for a fresh frame.

## Local setup

```bash
git clone <your-repo-url>
cd helmet-detection-api

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set your real key (get one at app.roboflow.com/settings/api):

```
ROBOFLOW_API_KEY=YOUR_REAL_ROBOFLOW_API_KEY
```

Run:

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000, or check health:

```bash
curl http://localhost:8000/health
```

Run a prediction:

```bash
curl -X POST "http://localhost:8000/predict?annotate=true" \
  -F "file=@sample.jpg"
```

## Docker

```bash
docker build -t helmet-detection-api .
docker run --rm -p 8080:8080 \
  -e ROBOFLOW_API_KEY=YOUR_REAL_ROBOFLOW_API_KEY \
  -e ROBOFLOW_WORKSPACE=harsh-chakravarti \
  -e ROBOFLOW_WORKFLOW_ID=helmet-detection-safety-monitoring-1783929349435 \
  helmet-detection-api
```

Open http://localhost:8080.

## Environment variables

| Name | Description |
|---|---|
| `ROBOFLOW_API_KEY` | Your Roboflow API key |
| `ROBOFLOW_WORKSPACE` | Roboflow workspace slug |
| `ROBOFLOW_WORKFLOW_ID` | Workflow URL slug |
| `ROBOFLOW_API_URL` | `https://serverless.roboflow.com` by default |
| `ROBOFLOW_REQUEST_TIMEOUT_SECONDS` | Per-attempt hard timeout for the workflow call (default `15`) |
| `ROBOFLOW_MAX_RETRIES` | Retries after the first attempt (default `2`) |
| `ROBOFLOW_RETRY_BACKOFF_SECONDS` | Linear backoff base, in seconds (default `1.0`) |
| `ROBOFLOW_USE_CACHE` | Whether to allow Roboflow's own result cache (default `false`) |
| `MAX_UPLOAD_MB` | Max upload size |
| `ALLOWED_IMAGE_TYPES` | Comma-separated content types |

## Deployment notes

### Render

Use Docker deployment. Set the environment variables above (at minimum `ROBOFLOW_API_KEY`, `ROBOFLOW_WORKSPACE`, `ROBOFLOW_WORKFLOW_ID`). Health check path: `/health`.

### Google Cloud Run

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/helmet-detection-api

gcloud run deploy helmet-detection-api \
  --image gcr.io/YOUR_PROJECT_ID/helmet-detection-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars ROBOFLOW_WORKSPACE=harsh-chakravarti,ROBOFLOW_WORKFLOW_ID=helmet-detection-safety-monitoring-1783929349435,ROBOFLOW_API_URL=https://serverless.roboflow.com
```

Set `ROBOFLOW_API_KEY` securely through Cloud Run environment variables or Secret Manager — never commit it.

## Security

- Never commit `.env`.
- Use GitHub Actions secrets for any deployment credentials.
- If using a restricted Roboflow key, make sure it can run the workflow and (if you want Vision Events alerts logged) write to Vision Events.

## Testing / verification

- `tests/test_roboflow_client.py` and `tests/test_health.py` run fully offline and are part of CI. They mock the Roboflow SDK call with a payload shaped exactly like a real captured response from this workflow (plus one synthetic polygon detection, since the live capture used for grounding had zero detections in frame) to verify: expected output keys, polygon→bbox derivation, that raw `points` are never leaked, and retry-then-succeed behavior.
- `scripts/live_smoke_test.py` hits the **real** Roboflow endpoint with a real image and a real API key. It has **not** been run by whoever/whatever built this integration, because that was done in a sandbox with no network route to `serverless.roboflow.com` and no real API key for this workspace. Run it yourself once you have both:
  ```bash
  export ROBOFLOW_API_KEY=your_real_key
  python scripts/live_smoke_test.py path/to/sample.jpg
  ```

## API response example

```json
{
  "image": { "width": 612, "height": 408 },
  "latency_ms": 1090.2,
  "detections": [
    {
      "class": "no helmet",
      "confidence": 0.87,
      "bbox": { "x1": 98.0, "y1": 50.0, "x2": 160.0, "y2": 120.0, "width": 62.0, "height": 70.0 }
    }
  ],
  "compliance": { "persons": 1, "helmet": 0, "no_helmet": 1, "violations": 1, "unknown": 0, "status": "violation" },
  "safety_status": "violation",
  "output_image_base64": "<base64>",
  "vision_events": { "error_status": false, "message": "Vision event sent successfully" }
}
```

## Known, deliberately-left-alone items

- A `StarletteDeprecationWarning` about `httpx`/`httpx2` shows up under this exact `fastapi==0.115.6` + `httpx==0.28.1` pin combination when running tests. It's cosmetic (`TestClient` internals), doesn't affect runtime behavior, and fixing it would mean bumping pinned versions — flagging it here rather than changing dependency pins without asking.

## Future improvements

- Replace the SAM3 zero-shot workflow with a trained YOLOv8n ONNX model for faster CPU inference.
- Add video frame ingestion (note: this workflow's image-based path is different from Roboflow's WebRTC/video stream path — treat that as a separate integration).
- Add false-positive review storage.
- Add a latency dashboard.
- Add GitHub Actions deployment to Render or Cloud Run.

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial helmet detection FastAPI service"

git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/helmet-detection-api.git
git push -u origin main
```

Or with the GitHub CLI:

```bash
gh repo create helmet-detection-api --public --source=. --remote=origin --push
```

After pushing, add your real `ROBOFLOW_API_KEY` as a GitHub Actions secret if you extend CI/CD. Your local `.env` should never be committed.

from contextlib import asynccontextmanager
from functools import lru_cache
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

from app.config import get_settings
from app.roboflow_client import (
    RoboflowWorkflowClient,
    RoboflowWorkflowError,
    RoboflowWorkflowUnauthorizedError,
    RoboflowWorkflowTimeoutError,
)
from app.schemas import PredictResponse


@lru_cache
def get_client() -> RoboflowWorkflowClient:
    return RoboflowWorkflowClient(get_settings())


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    if get_client.cache_info().currsize:
        get_client().close()
    get_client.cache_clear()


app = FastAPI(
    title="Helmet Detection Safety API",
    description="FastAPI wrapper around a Roboflow helmet safety monitoring Workflow.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "workspace": settings.roboflow_workspace,
        "workflow_id": settings.roboflow_workflow_id,
        "runtime": "roboflow-workflows",
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    annotate: bool = Query(default=False, description="Include base64 annotated output image."),
):
    settings = get_settings()

    if file.content_type not in settings.allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type {file.content_type}. Allowed: {sorted(settings.allowed_types)}",
        )

    contents = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(contents) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Max size is {settings.max_upload_mb} MB.")

    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    client = get_client()

    try:
        return client.predict(image=image, include_annotated_image=annotate)
    except RoboflowWorkflowUnauthorizedError as exc:
        raise HTTPException(
            status_code=401,
            detail=f"Roboflow API key is not authorized for serverless inference: {exc}",
        ) from exc
    except RoboflowWorkflowTimeoutError as exc:
        raise HTTPException(status_code=504, detail=f"Roboflow workflow request timed out: {exc}") from exc
    except RoboflowWorkflowError as exc:
        raise HTTPException(status_code=502, detail=f"Roboflow workflow request failed: {exc}") from exc


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html>
  <head>
    <title>Helmet Detection Demo</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      :root {
        --bg: #08111f;
        --bg-soft: #101c33;
        --panel: rgba(10, 16, 30, 0.86);
        --panel-strong: #0d1729;
        --text: #eef3ff;
        --muted: #97a3bb;
        --accent: #6ee7ff;
        --accent-2: #8b5cf6;
        --success: #34d399;
        --danger: #fb7185;
        --border: rgba(148, 163, 184, 0.2);
        --shadow: 0 24px 80px rgba(0, 0, 0, 0.42);
      }

      * { box-sizing: border-box; }

      body {
        margin: 0;
        min-height: 100vh;
        color: var(--text);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background:
          radial-gradient(circle at top left, rgba(110, 231, 255, 0.14), transparent 28%),
          radial-gradient(circle at bottom right, rgba(139, 92, 246, 0.20), transparent 24%),
          linear-gradient(180deg, #060b16 0%, #09111f 52%, #060b16 100%);
      }

      .shell {
        max-width: 1320px;
        margin: 0 auto;
        padding: 32px 20px 40px;
      }

      .hero {
        display: grid;
        grid-template-columns: 1.4fr 0.9fr;
        gap: 20px;
        align-items: stretch;
      }

      .hero-card,
      .panel {
        background: linear-gradient(180deg, rgba(14, 22, 40, 0.92), rgba(7, 12, 22, 0.92));
        border: 1px solid var(--border);
        border-radius: 24px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(18px);
      }

      .hero-card {
        padding: 28px;
        overflow: hidden;
        position: relative;
      }

      .hero-card::after {
        content: "";
        position: absolute;
        inset: auto -30% -40% auto;
        width: 280px;
        height: 280px;
        background: radial-gradient(circle, rgba(110, 231, 255, 0.18), transparent 65%);
        pointer-events: none;
      }

      .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(110, 231, 255, 0.12);
        color: var(--accent);
        border: 1px solid rgba(110, 231, 255, 0.25);
        font-size: 12px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }

      h1 {
        margin: 16px 0 12px;
        font-size: clamp(32px, 5vw, 54px);
        line-height: 0.96;
        letter-spacing: -0.04em;
      }

      .lede {
        max-width: 62ch;
        color: var(--muted);
        font-size: 15px;
        line-height: 1.7;
      }

      .controls {
        margin-top: 22px;
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        align-items: center;
      }

      .file-input {
        flex: 1 1 260px;
        padding: 14px 16px;
        border-radius: 16px;
        border: 1px dashed rgba(148, 163, 184, 0.32);
        background: rgba(255, 255, 255, 0.03);
        color: var(--muted);
      }

      .button {
        border: 0;
        padding: 14px 18px;
        border-radius: 16px;
        background: linear-gradient(135deg, var(--accent), #9b8cff);
        color: #04111c;
        font-weight: 700;
        cursor: pointer;
        transition: transform 0.2s ease, filter 0.2s ease;
      }

      .button:hover { transform: translateY(-1px); filter: brightness(1.05); }

      .stats {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        margin-top: 24px;
      }

      .stat {
        padding: 16px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(148, 163, 184, 0.16);
      }

      .stat-label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }
      .stat-value { font-size: 24px; font-weight: 700; margin-top: 8px; }

      .grid {
        margin-top: 20px;
        display: grid;
        grid-template-columns: 1.25fr 0.95fr;
        gap: 20px;
        align-items: start;
      }

      .panel { padding: 18px; }

      .panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 16px;
      }

      .panel-title {
        margin: 0;
        font-size: 18px;
        letter-spacing: -0.02em;
      }

      .badge {
        padding: 8px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        background: rgba(148, 163, 184, 0.12);
        color: var(--muted);
      }

      .badge.ok { background: rgba(52, 211, 153, 0.12); color: var(--success); }
      .badge.violation { background: rgba(251, 113, 133, 0.14); color: #ff8ea1; }

      .viewer {
        position: relative;
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.14);
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      }

      .viewer-canvas {
        display: block;
        width: 100%;
        height: auto;
        aspect-ratio: 4 / 3;
        background: #08111f;
      }

      .empty-state {
        min-height: 480px;
        display: grid;
        place-items: center;
        color: var(--muted);
        text-align: center;
        padding: 32px;
      }

      .empty-state strong { display: block; color: var(--text); margin-bottom: 8px; font-size: 18px; }

      .detection-list {
        display: grid;
        gap: 12px;
      }

      .detection-item {
        padding: 14px;
        border-radius: 16px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(148, 163, 184, 0.14);
      }

      .detection-top {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        align-items: center;
        margin-bottom: 10px;
      }

      .class-name { font-size: 16px; font-weight: 700; text-transform: capitalize; }
      .confidence { color: var(--accent); font-weight: 700; }

      .bbox-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
        color: var(--muted);
        font-size: 12px;
      }

      .bbox-grid span {
        display: block;
        margin-top: 4px;
        color: var(--text);
        font-size: 13px;
        font-variant-numeric: tabular-nums;
      }

      .json-box {
        margin-top: 20px;
        padding: 18px;
        border-radius: 20px;
        background: #050916;
        border: 1px solid rgba(148, 163, 184, 0.14);
        min-height: 480px;
        max-height: 480px;
        overflow: auto;
      }

      .json-tools {
        display: flex;
        justify-content: flex-end;
        margin-top: 10px;
      }

      .ghost-button {
        border: 1px solid rgba(148, 163, 184, 0.24);
        background: rgba(148, 163, 184, 0.08);
        color: var(--text);
        border-radius: 12px;
        padding: 8px 12px;
        font-size: 12px;
        cursor: pointer;
      }

      pre {
        margin: 0;
        color: #dbeafe;
        font-size: 12px;
        line-height: 1.6;
        white-space: pre-wrap;
        word-break: break-word;
      }

      .footer-note {
        margin-top: 16px;
        color: var(--muted);
        font-size: 12px;
      }

      @media (max-width: 980px) {
        .hero, .grid { grid-template-columns: 1fr; }
      }

      @media (max-width: 640px) {
        .shell { padding: 16px 12px 24px; }
        .hero-card, .panel { border-radius: 20px; }
        .stats { grid-template-columns: 1fr; }
        .bbox-grid { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <section class="hero">
        <div class="hero-card">
          <div class="eyebrow">Live helmet safety monitor</div>
          <h1>Readable detection UI with image preview and bounding boxes.</h1>
          <p class="lede">
            Upload a frame, inspect the returned detections, and see each box drawn directly over the image.
            The right-hand panel lists the exact bounding box values so you can inspect confidence and placement.
          </p>

          <div class="controls">
            <input id="file" class="file-input" type="file" accept="image/png,image/jpeg,image/webp" />
            <button class="button" onclick="run()">Run Detection</button>
          </div>

          <div class="stats">
            <div class="stat">
              <div class="stat-label">Safety status</div>
              <div class="stat-value" id="statusValue">Idle</div>
            </div>
            <div class="stat">
              <div class="stat-label">Detections</div>
              <div class="stat-value" id="countValue">0</div>
            </div>
          </div>
        </div>

        <aside class="panel">
          <div class="panel-header">
            <h2 class="panel-title">Detection summary</h2>
            <span class="badge" id="summaryBadge">Waiting</span>
          </div>
          <div class="detection-list" id="detectionList">
            <div class="detection-item">
              <div class="class-name">No image loaded</div>
              <div class="footer-note">Choose a photo to see bounding boxes and confidence scores here.</div>
            </div>
          </div>
        </aside>
      </section>

      <section class="grid">
        <div class="panel">
          <div class="panel-header">
            <h2 class="panel-title">Image with bounding boxes</h2>
            <span class="badge" id="imageBadge">Preview</span>
          </div>
          <div class="viewer" id="viewer">
            <canvas id="canvas" class="viewer-canvas"></canvas>
            <div class="empty-state" id="emptyState">
              <div>
                <strong>Preview will appear here</strong>
                The uploaded image and detected boxes are rendered on the canvas after you run the model.
              </div>
            </div>
          </div>
          <div class="footer-note">Bounding boxes are scaled to the rendered image so the overlay stays accurate on desktop and mobile.</div>
        </div>

        <div class="panel">
          <div class="panel-header">
            <h2 class="panel-title">JSON response</h2>
            <span class="badge">Raw output</span>
          </div>
          <div class="json-box">
            <pre id="json">Run a detection to see the compact response payload.</pre>
          </div>
          <div class="json-tools">
            <button id="toggleJson" class="ghost-button" type="button">Show full payload</button>
          </div>
          <div class="footer-note">Default view hides large base64 data so this panel stays the same height as the image panel.</div>
        </div>
      </section>
    </div>

    <script>
      const canvas = document.getElementById("canvas");
      const ctx = canvas.getContext("2d");
      const emptyState = document.getElementById("emptyState");
      const statusValue = document.getElementById("statusValue");
      const countValue = document.getElementById("countValue");
      const summaryBadge = document.getElementById("summaryBadge");
      const imageBadge = document.getElementById("imageBadge");
      const detectionList = document.getElementById("detectionList");
      const toggleJsonButton = document.getElementById("toggleJson");

      let showFullPayload = false;
      let latestResponse = null;

      function buildCompactResponse(data) {
        return {
          image: data.image || null,
          latency_ms: data.latency_ms,
          safety_status: data.safety_status,
          compliance: data.compliance || null,
          detections_count: Array.isArray(data.detections) ? data.detections.length : 0,
          detections: Array.isArray(data.detections)
            ? data.detections.map((detection) => ({
                class: detection.class,
                confidence: detection.confidence,
                bbox: detection.bbox,
              }))
            : [],
          vision_events: data.vision_events || null,
          output_image_base64: data.output_image_base64 ? "<hidden in compact view>" : null,
        };
      }

      function renderJsonPayload() {
        const jsonEl = document.getElementById("json");
        if (!latestResponse) {
          jsonEl.textContent = "Run a detection to see the compact response payload.";
          return;
        }

        const payload = showFullPayload ? latestResponse : buildCompactResponse(latestResponse);
        jsonEl.textContent = JSON.stringify(payload, null, 2);
        toggleJsonButton.textContent = showFullPayload ? "Show compact payload" : "Show full payload";
      }

      toggleJsonButton.addEventListener("click", () => {
        showFullPayload = !showFullPayload;
        renderJsonPayload();
      });

      function formatNumber(value) {
        if (value === null || value === undefined || Number.isNaN(Number(value))) {
          return "-";
        }
        return Number(value).toFixed(1);
      }

      function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
      }

      function overlaps(a, b) {
        return !(a.right < b.left || a.left > b.right || a.bottom < b.top || a.top > b.bottom);
      }

      function placeLabel(box, labelWidth, labelHeight, occupied) {
        const candidates = [
          { x: box.x, y: box.y - labelHeight - 8 },
          { x: box.x, y: box.y + 8 },
          { x: box.x, y: box.y - labelHeight - 8 - (labelHeight + 6) },
          { x: box.x, y: box.y + 8 + (labelHeight + 6) },
        ];

        for (const candidate of candidates) {
          const placed = {
            left: candidate.x,
            top: candidate.y,
            right: candidate.x + labelWidth,
            bottom: candidate.y + labelHeight,
          };

          if (placed.top >= 0 && placed.bottom <= box.canvasHeight && placed.left >= 0 && placed.right <= box.canvasWidth) {
            if (!occupied.some((item) => overlaps(item, placed))) {
              occupied.push(placed);
              return candidate;
            }
          }
        }

        const fallback = {
          x: clamp(box.x, 0, Math.max(0, box.canvasWidth - labelWidth)),
          y: clamp(box.y + 8, 0, Math.max(0, box.canvasHeight - labelHeight)),
        };
        occupied.push({
          left: fallback.x,
          top: fallback.y,
          right: fallback.x + labelWidth,
          bottom: fallback.y + labelHeight,
        });
        return fallback;
      }

      function drawOverlay(imageSrc, detections) {
        return new Promise((resolve, reject) => {
          const image = new Image();
          image.onload = () => {
            const maxWidth = 860;
            const scale = Math.min(1, maxWidth / image.width);
            const width = Math.round(image.width * scale);
            const height = Math.round(image.height * scale);

            canvas.width = width;
            canvas.height = height;
            canvas.style.width = width + "px";
            canvas.style.height = height + "px";

            ctx.clearRect(0, 0, width, height);
            ctx.drawImage(image, 0, 0, width, height);

            const occupiedLabels = [];

            detections.forEach((detection, index) => {
              const bbox = detection.bbox || {};
              const x1 = bbox.x1;
              const y1 = bbox.y1;
              const x2 = bbox.x2;
              const y2 = bbox.y2;

              if ([x1, y1, x2, y2].some((value) => value === null || value === undefined)) {
                return;
              }

              const boxX = x1 * scale;
              const boxY = y1 * scale;
              const boxW = (x2 - x1) * scale;
              const boxH = (y2 - y1) * scale;
              const palette = ["#6ee7ff", "#8b5cf6", "#34d399", "#fb7185"];
              const color = palette[index % palette.length];

              ctx.lineWidth = Math.max(2, Math.round(3 * scale));
              ctx.strokeStyle = color;
              ctx.fillStyle = `${color}20`;
              ctx.fillRect(boxX, boxY, boxW, boxH);
              ctx.shadowColor = "rgba(0, 0, 0, 0.35)";
              ctx.shadowBlur = 10;
              ctx.strokeRect(boxX, boxY, boxW, boxH);
              ctx.shadowBlur = 0;

              const label = `${detection.class} ${(detection.confidence * 100).toFixed(0)}%`;
              ctx.font = "700 13px Inter, system-ui, sans-serif";
              const labelWidth = ctx.measureText(label).width + 18;
              const labelHeight = 24;
              const labelPosition = placeLabel(
                { x: boxX, y: boxY, canvasWidth: width, canvasHeight: height },
                labelWidth,
                labelHeight,
                occupiedLabels
              );

              ctx.fillStyle = color;
              ctx.fillRect(labelPosition.x, labelPosition.y, labelWidth, labelHeight);
              ctx.fillStyle = "#04111c";
              ctx.fillText(label, labelPosition.x + 9, labelPosition.y + 16);
            });

            resolve();
          };
          image.onerror = reject;
          image.src = imageSrc;
        });
      }

      function renderDetectionList(detections) {
        if (!detections.length) {
          detectionList.innerHTML = `
            <div class="detection-item">
              <div class="class-name">No detections</div>
              <div class="footer-note">The current image did not return any detections.</div>
            </div>
          `;
          return;
        }

        detectionList.innerHTML = detections.map((detection, index) => {
          const bbox = detection.bbox || {};
          return `
            <div class="detection-item">
              <div class="detection-top">
                <div class="class-name">${index + 1}. ${detection.class}</div>
                <div class="confidence">${(detection.confidence * 100).toFixed(1)}%</div>
              </div>
              <div class="bbox-grid">
                <div>x1<span>${formatNumber(bbox.x1)}</span></div>
                <div>y1<span>${formatNumber(bbox.y1)}</span></div>
                <div>x2<span>${formatNumber(bbox.x2)}</span></div>
                <div>y2<span>${formatNumber(bbox.y2)}</span></div>
                <div>width<span>${formatNumber(bbox.width)}</span></div>
                <div>height<span>${formatNumber(bbox.height)}</span></div>
              </div>
            </div>
          `;
        }).join("");
      }

      async function run() {
        const fileInput = document.getElementById("file");

        if (!fileInput.files.length) {
          alert("Choose an image first.");
          return;
        }

        const form = new FormData();
        form.append("file", fileInput.files[0]);

        const jsonEl = document.getElementById("json");
        jsonEl.textContent = "Running...";
        statusValue.textContent = "Running";
        countValue.textContent = "0";
        summaryBadge.textContent = "Working";
        summaryBadge.className = "badge";
        imageBadge.textContent = "Rendering";
        emptyState.style.display = "grid";
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        detectionList.innerHTML = `
          <div class="detection-item">
            <div class="class-name">Running detection</div>
            <div class="footer-note">Please wait while the model returns the image and bounding boxes.</div>
          </div>
        `;

        const res = await fetch("/predict?annotate=true", {
          method: "POST",
          body: form
        });

        const data = await res.json();
        latestResponse = data;
        renderJsonPayload();

        const detections = Array.isArray(data.detections) ? data.detections : [];
        renderDetectionList(detections);
        countValue.textContent = String(detections.length);

        const status = data.safety_status || data.compliance?.status || "unknown";
        statusValue.textContent = status;
        summaryBadge.textContent = status;
        summaryBadge.className = `badge ${status === "violation" ? "violation" : "ok"}`;
        imageBadge.textContent = detections.length ? `${detections.length} boxes` : "No boxes";

        if (data.output_image_base64) {
          emptyState.style.display = "none";
          const imageSrc = "data:image/jpeg;base64," + data.output_image_base64;
          await drawOverlay(imageSrc, detections);
        }
      }
    </script>
  </body>
</html>
"""

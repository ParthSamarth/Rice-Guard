"""
server/main.py -- RiceGuard AI local API server (the "Application Layer").

Wraps the already-validated pipeline.inference_pipeline.RiceDiseasePipeline
behind a small local-network HTTP API for the Android client. This file
contains NO detection/classification/Grad-CAM/recommendation logic of its
own -- see server/pipeline_service.py for the one place that calls the
pipeline, and PROJECT_README.md / android/README.md for the full picture.

Run (from the project root):
    server/run_server.ps1
or manually:
    cd server
    ../.venv/Scripts/python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000

(module-relative imports below assume `server/` is the working directory,
matching the run command above and android/README.md's setup instructions)
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.discovery import RiceGuardAdvertiser  # noqa: E402
from server.pipeline_service import PipelineService, ModelsNotReadyError, TMP_ROOT  # noqa: E402
from server.security import LocalNetworkOnlyMiddleware, resolve_result_file  # noqa: E402
from server.schemas import HealthResponse, PredictResponse, ErrorResponse  # noqa: E402

# Single source of truth for the port, shared with run_server.ps1 (which sets
# RICEGUARD_PORT before launching uvicorn) so the mDNS advertisement below
# always matches the port uvicorn is actually bound to -- nothing here can
# introspect uvicorn's own bind port at runtime, so both sides read the same
# env var instead of duplicating the number.
SERVER_PORT = int(os.environ.get("RICEGUARD_PORT", "8000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("riceguard.server")

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB -- generous for a phone-camera JPEG, rejects anything absurd
MIN_DIMENSION_PX = 200               # below this, the image is unusable for detection/classification either way
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
CLEANUP_INTERVAL_SECONDS = 5 * 60

service = PipelineService()
advertiser = RiceGuardAdvertiser(port=SERVER_PORT, version="1.1.0")


async def _cleanup_loop():
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        try:
            removed = await asyncio.to_thread(service.sweep_expired_results)
            if removed:
                logger.info("cleanup: removed %d expired result folder(s)", removed)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cleanup sweep failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading RiceDiseasePipeline (YOLOv8n @ 640 + ResNet50) ...")
    service.load_pipeline()  # raises + aborts startup if checkpoints are missing/broken -- fail fast, not on first scan
    logger.info("Models loaded. RiceGuard AI server ready.")
    cleanup_task = asyncio.create_task(_cleanup_loop())
    await advertiser.start()
    try:
        yield
    finally:
        cleanup_task.cancel()
        await advertiser.stop()


app = FastAPI(
    title="RiceGuard AI",
    version="1.1.0",
    description="Developed by: Parth Samarth",
    lifespan=lifespan,
)
app.add_middleware(LocalNetworkOnlyMiddleware)


@app.get("/health", response_model=HealthResponse)
async def health():
    info = service.health_info()
    return HealthResponse(
        status="ok" if info["models_ready"] else "degraded",
        models_ready=info["models_ready"],
        yolo=info["yolo"],
        resnet50=info["resnet50"],
        device=str(info["device"]) if info["device"] is not None else None,
        queue_depth=info["queue_depth"],
    )


def _validate_upload(content_type: str | None, data: bytes) -> None:
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{content_type}'. "
                                                      f"Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}.")
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400,
                             detail=f"Image too large ({len(data) / 1_048_576:.1f} MB). "
                                    f"Maximum is {MAX_UPLOAD_BYTES / 1_048_576:.0f} MB.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:  # verify() invalidates the handle; reopen to read size
            w, h = im.size
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="This image could not be read.")
    if w < MIN_DIMENSION_PX or h < MIN_DIMENSION_PX:
        raise HTTPException(status_code=400,
                             detail=f"Image resolution too low ({w}x{h}). Minimum is "
                                    f"{MIN_DIMENSION_PX}x{MIN_DIMENSION_PX}.")


def _to_url(request_id: str, file_path: str | None) -> str | None:
    if not file_path:
        return None
    return f"/results/{request_id}/{Path(file_path).name}"


def _build_response(request_id: str, result: dict, server_t0: float) -> PredictResponse:
    for region in result.get("regions", []):
        region["gradcam_url"] = _to_url(request_id, region.pop("gradcam", None))

    return PredictResponse(
        request_id=request_id,
        status=result["status"],
        decision_path=result["decision_path"],
        final_disease=result["final_disease"],
        final_confidence=result["final_confidence"],
        yolo_prediction=result["yolo_prediction"],
        yolo_confidence=result["yolo_confidence"],
        cnn_prediction=result["cnn_prediction"],
        cnn_confidence=result["cnn_confidence"],
        agreement=result["agreement"],
        agreement_detail=result["agreement_detail"],
        detections=result["detections"],
        regions=result["regions"],
        gradcam_url=_to_url(request_id, result.get("gradcam_path")),
        detection_url=_to_url(request_id, result.get("detection_image_path")),
        recommendation=result["recommendation"],
        thresholds_used=result["thresholds_used"],
        pipeline_processing_time_sec=result["processing_time_sec"],
        server_processing_time_sec=round(time.time() - server_t0, 3),
    )


@app.post("/predict", response_model=PredictResponse, responses={400: {"model": ErrorResponse},
                                                                   503: {"model": ErrorResponse},
                                                                   500: {"model": ErrorResponse}})
async def predict(request: Request, image: UploadFile = File(...)):
    server_t0 = time.time()
    client_ip = request.client.host if request.client else "unknown"
    # These two log lines are pure observability (a request-console started/
    # finished pair the optional riceguard_console.py can parse for its live
    # dashboard) -- they add no branching, no new state, nothing that
    # touches the pipeline itself.
    logger.info("predict_start ip=%s", client_ip)

    if not service.models_ready:
        raise HTTPException(status_code=503, detail="AI models are not ready yet. Try again shortly.")

    data = await image.read()
    _validate_upload(image.content_type, data)

    request_id = service.new_request_id()
    upload_path = await service.save_upload(request_id, data, image.filename)

    try:
        result = await service.run_prediction(request_id, upload_path)
    except ModelsNotReadyError:
        logger.info("predict_done ip=%s status=503 duration=%.2fs", client_ip, time.time() - server_t0)
        raise HTTPException(status_code=503, detail="AI models are not ready yet. Try again shortly.")
    except Exception as exc:  # noqa: BLE001
        # Full detail stays in the server log only -- the Android app never
        # sees a raw Python traceback (project brief section 21).
        logger.exception("request %s: pipeline failed", request_id)
        logger.info("predict_done ip=%s status=500 duration=%.2fs", client_ip, time.time() - server_t0)
        raise HTTPException(status_code=500,
                             detail=f"The AI server could not complete the analysis (request {request_id}).")
    finally:
        service.cleanup_upload(request_id, upload_path)

    response = _build_response(request_id, result, server_t0)
    result_str = f"{response.final_disease} {response.final_confidence * 100:.0f}%" if response.final_disease and response.final_confidence is not None else "n/a"
    logger.info("predict_done ip=%s status=200 duration=%.2fs result=%s", client_ip, time.time() - server_t0, result_str)
    return response


@app.get("/results/{request_id}/{filename}")
async def get_result_file(request_id: str, filename: str):
    path = resolve_result_file(TMP_ROOT, request_id, filename)
    if path is None:
        raise HTTPException(status_code=404, detail="Result not found (it may have expired).")
    return FileResponse(path)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"status": "error", "message": exc.detail})

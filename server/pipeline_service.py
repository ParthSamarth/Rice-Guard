"""
server/pipeline_service.py

Owns the ONE RiceDiseasePipeline instance for the server's lifetime (model
loading is expensive -- this must happen once at startup, never per
request), the single-GPU request queue (project brief section 23: one RTX
4050, so inference jobs are serialized through an asyncio.Lock rather than
run concurrently), and the temp-file lifecycle for each request's uploaded
image + generated result images (project brief sections 7, 22).

This module deliberately contains NO detection/classification/Grad-CAM/
recommendation logic of its own -- it only calls
pipeline.inference_pipeline.RiceDiseasePipeline.run() (already validated,
already the project's single source of truth for that logic) and reshapes
its output for HTTP.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config  # noqa: E402
from pipeline.inference_pipeline import RiceDiseasePipeline  # noqa: E402

from server.security import RequestIdGenerator, sanitize_upload_filename  # noqa: E402
from server.visualize import render_detection_image  # noqa: E402

logger = logging.getLogger("riceguard.pipeline_service")

TMP_ROOT = Path(__file__).resolve().parent / "tmp"
RESULT_TTL_SECONDS = 30 * 60  # how long a request's result images stay fetchable before cleanup sweeps them


class ModelsNotReadyError(RuntimeError):
    pass


class PipelineService:
    def __init__(self):
        self.config = load_config()
        self.yolo_weights = self.config.path("paths", "models") / "yolo_yolov8n" / "best.pt"
        self.classifier_checkpoint = self.config.path("paths", "models") / "classifier_resnet50" / "best.pt"

        self._pipeline: Optional[RiceDiseasePipeline] = None
        self._gpu_lock = asyncio.Lock()
        self._queue_depth = 0
        self._request_ids = RequestIdGenerator()
        self._load_error: Optional[str] = None

        TMP_ROOT.mkdir(parents=True, exist_ok=True)

    # ---- lifecycle -----------------------------------------------------

    def verify_model_files(self) -> None:
        """Project brief section 4: "Before wiring them into the API, verify
        these files exist." Raises with a specific, actionable message
        rather than letting a missing-file error surface later as a cryptic
        torch.load failure deep inside model construction."""
        missing = []
        if not self.yolo_weights.exists():
            missing.append(str(self.yolo_weights))
        if not self.classifier_checkpoint.exists():
            missing.append(str(self.classifier_checkpoint))
        if missing:
            raise FileNotFoundError(
                "RiceGuard AI server cannot start: required model checkpoint(s) not found:\n  "
                + "\n  ".join(missing)
                + "\nThis server never substitutes a different checkpoint -- verify the final models "
                  "(YOLOv8n @ 640, ResNet50) are in place before retrying."
            )

    def load_pipeline(self) -> None:
        """Called once, synchronously, at server startup (see main.py's
        lifespan handler) -- NOT lazily on first request, so a broken
        checkpoint fails the server startup loudly instead of the first
        user's scan."""
        self.verify_model_files()
        try:
            self._pipeline = RiceDiseasePipeline(
                str(self.yolo_weights), str(self.classifier_checkpoint),
                config=self.config, gradcam_dir=TMP_ROOT, save_gradcam=True,
            )
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            raise

    @property
    def models_ready(self) -> bool:
        return self._pipeline is not None

    @property
    def queue_depth(self) -> int:
        return self._queue_depth

    def health_info(self) -> dict:
        # Deliberately does NOT include the checkpoints' filesystem paths --
        # project brief section 22: never expose server filesystem paths to
        # the client, even informationally. yolo/resnet50 booleans are enough
        # to answer "are the final models loaded and ready."
        return {
            "models_ready": self.models_ready,
            "yolo": self.models_ready,
            "resnet50": self.models_ready,
            "device": getattr(self._pipeline.yolo, "device", None) if self.models_ready else None,
            "queue_depth": self._queue_depth,
            "load_error": self._load_error,
        }

    # ---- request handling -----------------------------------------------

    def new_request_id(self) -> str:
        return self._request_ids.next_id()

    def request_dir(self, request_id: str) -> Path:
        return TMP_ROOT / request_id

    async def save_upload(self, request_id: str, data: bytes, original_filename: str | None) -> Path:
        req_dir = self.request_dir(request_id)
        req_dir.mkdir(parents=True, exist_ok=True)
        name = sanitize_upload_filename(original_filename)
        path = req_dir / name
        await asyncio.to_thread(path.write_bytes, data)
        return path

    async def run_prediction(self, request_id: str, image_path: Path) -> dict:
        """Serialized through self._gpu_lock so only one image is ever being
        processed by the GPU at a time; additional concurrent requests await
        the lock (a queue, per project brief section 23) rather than
        launching a second inference job. The blocking pipeline.run() call
        itself runs in a worker thread (run_in_executor) so the event loop
        stays free to accept /health checks and queue further requests while
        one is in flight."""
        if not self.models_ready:
            raise ModelsNotReadyError("Models are not loaded.")

        self._queue_depth += 1
        try:
            async with self._gpu_lock:
                self._queue_depth -= 1
                loop = asyncio.get_running_loop()
                t0 = time.time()
                result = await loop.run_in_executor(None, self._pipeline.run, str(image_path))
                logger.info("request %s: status=%s decision_path=%s gpu_time=%.2fs",
                            request_id, result.get("status"), result.get("decision_path"), time.time() - t0)
                # Relocated BEFORE releasing the lock: the pipeline's Grad-CAM
                # writer uses one shared, fixed filename per region index
                # (TMP_ROOT/upload_regionN.png -- see _relocate_gradcam_files'
                # own docstring), so the next queued request must not be able
                # to start writing to that same shared path until this
                # request's files are safely moved into its own subfolder.
                self._relocate_gradcam_files(request_id, result)
        finally:
            self._queue_depth = max(0, self._queue_depth)

        detection_path = self._render_detection_image(request_id, image_path, result)
        if detection_path:
            result["detection_image_path"] = str(detection_path)
        return result

    def _relocate_gradcam_files(self, request_id: str, result: dict) -> None:
        """RiceDiseasePipeline.gradcam_dir is fixed once at construction time
        (this service passes TMP_ROOT, since a per-request value can't be
        threaded through a shared, already-constructed pipeline instance) --
        so every request's Grad-CAM files land directly in TMP_ROOT under a
        FIXED filename (the sanitized upload name's stem), not in this
        request's own subfolder. Left alone, two requests in a row would
        silently overwrite each other's Grad-CAM images on disk before the
        first client ever fetches theirs. This moves each file this request
        actually produced into TMP_ROOT/{request_id}/ right after the GPU
        call returns (i.e. before the lock is released for the next request)
        and rewrites the result dict's path fields to match -- so the
        request-scoped isolation the URL scheme (GET /results/{request_id}/...)
        assumes is actually true on disk, not just in the URL."""
        moved: dict[str, str] = {}
        req_dir = self.request_dir(request_id)
        req_dir.mkdir(parents=True, exist_ok=True)

        def _move(src_str: str | None) -> str | None:
            if not src_str:
                return src_str
            if src_str in moved:
                return moved[src_str]
            src = Path(src_str)
            if not src.exists() or src.parent != TMP_ROOT:
                return src_str  # already somewhere else / already moved -- leave as-is
            dest = req_dir / src.name
            try:
                src.replace(dest)
            except OSError as exc:
                logger.warning("request %s: could not relocate gradcam file %s: %s", request_id, src, exc)
                return src_str
            moved[src_str] = str(dest)
            return str(dest)

        for region in result.get("regions", []):
            region["gradcam"] = _move(region.get("gradcam"))
        result["gradcam"] = _move(result.get("gradcam"))
        result["gradcam_path"] = _move(result.get("gradcam_path"))

    def _render_detection_image(self, request_id: str, image_path: Path, result: dict) -> Optional[Path]:
        try:
            out_path = self.request_dir(request_id) / "detection.jpg"
            detections = result.get("detections") or []
            return render_detection_image(image_path, detections, out_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("request %s: failed to render detection image: %s", request_id, exc)
            return None

    def cleanup_upload(self, request_id: str, upload_path: Path) -> None:
        """Deletes ONLY the raw uploaded image right after it's been used --
        project brief section 22: "never permanently store incoming user
        images on the server by default." The derived result images
        (Grad-CAM, detection viz) are left in place for RESULT_TTL_SECONDS so
        the phone can fetch them via GET /results/... after the JSON
        response returns (see sweep_expired_results for when those go too)."""
        try:
            if upload_path.exists():
                upload_path.unlink()
        except OSError as exc:
            logger.warning("request %s: could not remove uploaded image: %s", request_id, exc)

    def sweep_expired_results(self) -> int:
        """Removes any request folder older than RESULT_TTL_SECONDS. Called
        periodically by a background task (see main.py) -- this is what
        makes 'temporary processing files' (project brief section 2) actually
        temporary, without racing the Android app's own follow-up GETs for
        the same request's Grad-CAM/detection images right after /predict
        returns."""
        removed = 0
        now = time.time()
        if not TMP_ROOT.exists():
            return 0
        for child in TMP_ROOT.iterdir():
            if not child.is_dir():
                continue
            try:
                age = now - child.stat().st_mtime
                if age > RESULT_TTL_SECONDS:
                    shutil.rmtree(child, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
        return removed

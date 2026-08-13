"""Local browser dashboard for camera preview and batched edge inference."""

from __future__ import annotations

import base64
import binascii
import logging
import threading
import time
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from datetime import date as Date
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import EdgeConfig
from .decision import FramePrediction, majority_vote
from .inference import DinoOnnxClassifier
from .service import EdgeService
from .sources import create_camera_source
from .storage import PredictionStore

UI_DIRECTORY = Path(__file__).with_name("ui")
MAX_IMAGE_BYTES = 6 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000
logger = logging.getLogger(__name__)


class _SharedCameraSource:
    """Expose runtime-locked captures to the background edge service."""

    def __init__(self, runtime: DashboardRuntime) -> None:
        self.runtime = runtime

    def start(self) -> None:
        self.runtime.start_camera()

    def capture(self) -> Image.Image:
        return self.runtime.capture_pillow_frame()

    def close(self) -> None:
        return None


class _LockedClassifier:
    """Serialize browser and background inference through one ONNX session."""

    def __init__(self, runtime: DashboardRuntime) -> None:
        self.runtime = runtime

    def classify(self, images: Sequence[Image.Image]) -> list[FramePrediction]:
        with self.runtime._inference_lock:
            return self.runtime.classifier.classify(images)


class DashboardRuntime:
    """Own the dashboard classifier and persist browser-captured bursts."""

    def __init__(
        self,
        config: EdgeConfig,
        *,
        classifier: Any | None = None,
        store: PredictionStore | None = None,
        source: Any | None = None,
    ) -> None:
        self.config = config
        self.classifier = classifier or DinoOnnxClassifier(
            config.model.path,
            metadata_path=config.model.metadata_path,
            intra_op_threads=config.model.intra_op_threads,
            inter_op_threads=config.model.inter_op_threads,
            execution_provider=config.model.execution_provider,
        )
        self.store = store or PredictionStore(
            config.storage.database_path,
            image_directory=config.storage.image_directory,
            retain_images=config.storage.retain_images,
            retention_days=config.storage.retention_days,
            jpeg_quality=config.storage.jpeg_quality,
        )
        self._owns_store = store is None
        self._inference_lock = threading.Lock()
        self.source = source
        if self.source is None and config.camera.backend == "picamera2":
            self.source = create_camera_source(config.camera)
        self._camera_lock = threading.Lock()
        self._camera_started = False
        self._background_stop = threading.Event()
        self._background_thread: threading.Thread | None = None

    def public_config(self) -> dict[str, object]:
        service = self.config.service
        return {
            "burst_size": service.burst_size,
            "capture_spacing_seconds": service.capture_spacing_seconds,
            "interval_seconds": service.interval_seconds,
            "camera_width": self.config.camera.width,
            "camera_height": self.config.camera.height,
            "capture_mode": "server" if self.source is not None else "browser",
        }

    def start_camera(self) -> None:
        if self.source is None:
            raise RuntimeError("No server-managed camera is configured")
        with self._camera_lock:
            if not self._camera_started:
                self.source.start()
                self._camera_started = True

    def capture_camera_frame(self) -> dict[str, object]:
        image = self.capture_pillow_frame()
        image.thumbnail((1280, 720), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=self.config.storage.jpeg_quality,
        )
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {
            "image": f"data:image/jpeg;base64,{encoded}",
            "width": image.width,
            "height": image.height,
        }

    def capture_pillow_frame(self) -> Image.Image:
        if self.source is None:
            raise RuntimeError("No server-managed camera is configured")
        with self._camera_lock:
            if not self._camera_started:
                raise RuntimeError("The server-managed camera is not started")
            return self.source.capture().convert("RGB")

    def stop_camera(self) -> None:
        if self.source is None:
            return
        if self._background_thread is not None and self._background_thread.is_alive():
            return
        with self._camera_lock:
            if self._camera_started:
                self.source.close()
                self._camera_started = False

    def start_background_capture(self) -> None:
        """Start the server-owned 24/7 observation loop when a camera exists."""

        if self.source is None or self._background_thread is not None:
            return
        self.start_camera()
        self._background_stop.clear()
        self._background_thread = threading.Thread(
            target=self._run_background_capture,
            name="operai-eye-capture",
            daemon=True,
        )
        self._background_thread.start()

    def _run_background_capture(self) -> None:
        background_store = PredictionStore(
            self.config.storage.database_path,
            image_directory=self.config.storage.image_directory,
            retain_images=self.config.storage.retain_images,
            retention_days=self.config.storage.retention_days,
            jpeg_quality=self.config.storage.jpeg_quality,
        )
        service = EdgeService(
            source=_SharedCameraSource(self),
            classifier=_LockedClassifier(self),
            store=background_store,
            service_config=self.config.service,
            decision_config=self.config.decision,
        )
        try:
            if not self.config.service.run_immediately and self._background_stop.wait(
                self.config.service.interval_seconds
            ):
                return
            next_cycle = time.monotonic()
            while not self._background_stop.is_set():
                delay = next_cycle - time.monotonic()
                if delay > 0 and self._background_stop.wait(delay):
                    break
                try:
                    service.run_once()
                except Exception:
                    logger.exception("Background camera observation failed")
                next_cycle += self.config.service.interval_seconds
                while next_cycle <= time.monotonic():
                    next_cycle += self.config.service.interval_seconds
        finally:
            background_store.close()

    def classify_encoded_images(
        self, encoded_images: Sequence[str]
    ) -> dict[str, object]:
        expected = self.config.service.burst_size
        if len(encoded_images) != expected:
            raise ValueError(
                f"Expected exactly {expected} images, got {len(encoded_images)}"
            )

        images = [_decode_browser_image(value) for value in encoded_images]
        started_at = datetime.now(UTC)
        inference_start = time.monotonic()
        with self._inference_lock:
            predictions: list[FramePrediction] = self.classifier.classify(images)
        inference_ms = (time.monotonic() - inference_start) * 1000
        if len(predictions) != len(images):
            raise RuntimeError(
                f"Classifier returned {len(predictions)} predictions for "
                f"{len(images)} images"
            )

        vote = majority_vote(
            predictions,
            minimum_phase_confidence=self.config.decision.minimum_phase_confidence,
            minimum_vote_fraction=self.config.decision.minimum_vote_fraction,
        )
        completed_at = datetime.now(UTC)
        burst_id = self.store.record_success(
            started_at=started_at,
            completed_at=completed_at,
            captured_at=[started_at] * len(images),
            images=images,
            predictions=predictions,
            vote=vote,
            captures_expected=expected,
            inference_ms=inference_ms,
        )
        return {
            "burst_id": burst_id,
            "completed_at": completed_at.isoformat(),
            "inference_ms": inference_ms,
            "predictions": [asdict(prediction) for prediction in predictions],
            "vote": asdict(vote),
        }

    def close(self) -> None:
        self._background_stop.set()
        if self._background_thread is not None:
            self._background_thread.join(timeout=20)
            self._background_thread = None
        with self._camera_lock:
            if self.source is not None and self._camera_started:
                self.source.close()
                self._camera_started = False
        if self._owns_store:
            self.store.close()


def _decode_browser_image(value: str) -> Image.Image:
    if not isinstance(value, str):
        raise TypeError("Each image must be a base64 data URL")
    header, separator, payload = value.partition(",")
    if (
        separator != ","
        or not header.startswith("data:image/")
        or ";base64" not in header
    ):
        raise ValueError("Each image must be a base64 image data URL")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("An image contains invalid base64 data") from exc
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Each encoded image must be between 1 byte and {MAX_IMAGE_BYTES} bytes"
        )
    try:
        with Image.open(BytesIO(raw)) as opened:
            opened.load()
            if opened.width * opened.height > MAX_IMAGE_PIXELS:
                raise ValueError("An image exceeds the maximum pixel count")
            return ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("An uploaded value is not a supported image") from exc


def create_dashboard_app(
    config: EdgeConfig,
    *,
    classifier: Any | None = None,
    store: PredictionStore | None = None,
    source: Any | None = None,
    serve_ui: bool = True,
):
    """Build the FastAPI dashboard and optional Pi-native camera endpoints."""

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    runtime = DashboardRuntime(
        config,
        classifier=classifier,
        store=store,
        source=source,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        runtime.start_background_capture()
        try:
            yield
        finally:
            runtime.close()

    app = FastAPI(title="OperAI-EYE live camera", lifespan=lifespan)
    app.state.dashboard_runtime = runtime
    if serve_ui:
        app.mount("/assets", StaticFiles(directory=UI_DIRECTORY), name="edge-ui-assets")

        @app.get("/", include_in_schema=False)
        async def dashboard():
            return FileResponse(UI_DIRECTORY / "index.html")

    @app.get("/api/config")
    async def dashboard_config():
        return runtime.public_config()

    @app.get("/api/health/live")
    async def health_live():
        return {"status": "ok"}

    @app.get("/api/health/ready")
    async def health_ready():
        checker = getattr(runtime.classifier, "is_ready", None)
        if checker is not None and not checker():
            raise HTTPException(
                status_code=503, detail="Inference backend is not ready"
            )
        return {"status": "ready"}

    @app.get("/api/history")
    async def observation_history(hours: int = 24, limit: int = 20):
        if not 1 <= hours <= 24 * 366:
            raise HTTPException(status_code=400, detail="hours must be in [1, 8784]")
        if not 1 <= limit <= 100:
            raise HTTPException(status_code=400, detail="limit must be in [1, 100]")
        return runtime.store.dashboard(hours=hours, limit=limit)

    @app.get("/api/history/day")
    async def daily_observation_history(
        date: str,
        timezone_offset_minutes: int = 0,
        limit: int = 20,
    ):
        try:
            selected_day = Date.fromisoformat(date)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="date must use YYYY-MM-DD"
            ) from exc
        try:
            return runtime.store.daily_timeline(
                selected_day,
                timezone_offset_minutes=timezone_offset_minutes,
                recent_limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/camera/start")
    def camera_start():
        try:
            runtime.start_camera()
            return {"status": "started"}
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Camera start failed: {exc}"
            ) from exc

    @app.get("/api/camera/frame")
    def camera_frame():
        try:
            return runtime.capture_camera_frame()
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Camera capture failed: {exc}"
            ) from exc

    @app.post("/api/camera/stop")
    def camera_stop():
        try:
            runtime.stop_camera()
            return {"status": "stopped"}
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Camera stop failed: {exc}"
            ) from exc

    @app.post("/api/classify")
    async def classify(payload: dict[str, list[str]]):
        try:
            return runtime.classify_encoded_images(payload.get("images", []))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Inference failed: {exc}"
            ) from exc

    return app


def run_dashboard(
    config: EdgeConfig,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    """Start the local dashboard server and optionally open the system browser."""

    import webbrowser

    import uvicorn

    app = create_dashboard_app(config)
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{display_host}:{port}"
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"OperAI-EYE camera dashboard: {url}")
    uvicorn.run(app, host=host, port=port, log_level="info")

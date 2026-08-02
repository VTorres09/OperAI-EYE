"""Local browser dashboard for camera preview and batched edge inference."""

from __future__ import annotations

import base64
import binascii
import threading
import time
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import EdgeConfig
from .decision import FramePrediction, majority_vote
from .inference import DinoOnnxClassifier
from .storage import PredictionStore

UI_DIRECTORY = Path(__file__).with_name("ui")
MAX_IMAGE_BYTES = 6 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000


class DashboardRuntime:
    """Own the dashboard classifier and persist browser-captured bursts."""

    def __init__(
        self,
        config: EdgeConfig,
        *,
        classifier: Any | None = None,
        store: PredictionStore | None = None,
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

    def public_config(self) -> dict[str, object]:
        service = self.config.service
        return {
            "burst_size": service.burst_size,
            "capture_spacing_seconds": service.capture_spacing_seconds,
            "interval_seconds": service.interval_seconds,
            "camera_width": self.config.camera.width,
            "camera_height": self.config.camera.height,
        }

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
    serve_ui: bool = True,
):
    """Build the FastAPI dashboard without touching the operating-system camera."""

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    runtime = DashboardRuntime(config, classifier=classifier, store=store)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
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

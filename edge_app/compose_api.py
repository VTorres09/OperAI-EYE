"""Container entrypoint for the Triton-backed OperAI-EYE API."""

from __future__ import annotations

import os
from pathlib import Path

from .config import load_config
from .dashboard import create_dashboard_app
from .triton_inference import TritonDinoClassifier


def build_app():
    config_path = Path(os.environ.get("OPERAI_CONFIG_PATH", "/app/config.toml"))
    config = load_config(config_path)
    classifier = TritonDinoClassifier(
        os.environ.get("TRITON_URL", "http://triton:8000"),
        model_name=os.environ.get("TRITON_MODEL_NAME", "operai_eye_dinov3"),
        model_version=os.environ.get("TRITON_MODEL_VERSION", ""),
        timeout_seconds=float(os.environ.get("TRITON_TIMEOUT_SECONDS", "120")),
    )
    return create_dashboard_app(config, classifier=classifier, serve_ui=False)


app = build_app()

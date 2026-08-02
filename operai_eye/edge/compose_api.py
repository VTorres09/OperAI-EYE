"""Container entrypoint for the self-contained OperAI-EYE application."""

from __future__ import annotations

from pathlib import Path

from .config import load_config
from .dashboard import create_dashboard_app


def build_app():
    config_path = Path("/app/config.toml")
    config = load_config(config_path)
    return create_dashboard_app(config)


app = build_app()

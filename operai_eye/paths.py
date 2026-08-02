"""Shared paths for repository-backed application workflows."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("OPERAI_PROJECT_ROOT", Path.cwd())).resolve()
DATA_DIR = Path("data/exocentric_rgb")
OUTPUT_DIR = Path("output")
FRONTEND_DIR = PROJECT_ROOT / "frontend"
STATIC_DIR = PROJECT_ROOT / "static"
PROMPTS_DIR = Path("prompts")
MODELS_DIR = Path("models")

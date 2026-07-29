"""Configuration loading for the edge service."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, TypeVar


@dataclass(frozen=True)
class ServiceConfig:
    interval_seconds: float = 60.0
    burst_size: int = 5
    capture_spacing_seconds: float = 1.0
    minimum_captures: int = 3
    run_immediately: bool = True
    restart_after_failures: int = 5


@dataclass(frozen=True)
class CameraConfig:
    backend: str = "picamera2"
    width: int = 1920
    height: int = 1080
    rotation: int = 0
    device: str = "0"
    warmup_seconds: float = 2.0


@dataclass(frozen=True)
class ModelConfig:
    path: Path = Path("/opt/operai-eye/models/operai_eye_dinov3.onnx")
    metadata_path: Path | None = None
    intra_op_threads: int = 4
    inter_op_threads: int = 1


@dataclass(frozen=True)
class DecisionConfig:
    minimum_phase_confidence: float = 0.0
    minimum_vote_fraction: float = 0.6


@dataclass(frozen=True)
class StorageConfig:
    database_path: Path = Path("/var/lib/operai-eye/predictions.sqlite3")
    image_directory: Path = Path("/var/lib/operai-eye/images")
    retain_images: str = "none"
    retention_days: int = 7
    jpeg_quality: int = 85


@dataclass(frozen=True)
class EdgeConfig:
    service: ServiceConfig = ServiceConfig()
    camera: CameraConfig = CameraConfig()
    model: ModelConfig = ModelConfig()
    decision: DecisionConfig = DecisionConfig()
    storage: StorageConfig = StorageConfig()


_T = TypeVar("_T")


def _section(cls: type[_T], values: dict[str, Any] | None) -> _T:
    values = dict(values or {})
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} setting(s): {', '.join(unknown)}")
    return cls(**values)


def _expand_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser()


def load_config(path: Path | None = None) -> EdgeConfig:
    """Load TOML configuration and apply supported environment overrides."""

    data: dict[str, Any] = {}
    if path is not None:
        with path.open("rb") as handle:
            data = tomllib.load(handle)

    unknown_sections = sorted(
        set(data) - {"service", "camera", "model", "decision", "storage"}
    )
    if unknown_sections:
        raise ValueError(f"Unknown config section(s): {', '.join(unknown_sections)}")

    model = _section(ModelConfig, data.get("model"))
    model = replace(
        model,
        path=_expand_path(model.path),  # type: ignore[arg-type]
        metadata_path=_expand_path(model.metadata_path),
    )
    storage = _section(StorageConfig, data.get("storage"))
    storage = replace(
        storage,
        database_path=_expand_path(storage.database_path),  # type: ignore[arg-type]
        image_directory=_expand_path(storage.image_directory),  # type: ignore[arg-type]
    )
    config = EdgeConfig(
        service=_section(ServiceConfig, data.get("service")),
        camera=_section(CameraConfig, data.get("camera")),
        model=model,
        decision=_section(DecisionConfig, data.get("decision")),
        storage=storage,
    )

    model_path = os.environ.get("OPERAI_MODEL_PATH")
    database_path = os.environ.get("OPERAI_DATABASE_PATH")
    camera_backend = os.environ.get("OPERAI_CAMERA_BACKEND")
    if model_path:
        config = replace(config, model=replace(config.model, path=Path(model_path)))
    if database_path:
        config = replace(
            config,
            storage=replace(config.storage, database_path=Path(database_path)),
        )
    if camera_backend:
        config = replace(
            config,
            camera=replace(config.camera, backend=camera_backend),
        )

    validate_config(config)
    return config


def validate_config(config: EdgeConfig) -> None:
    service = config.service
    if service.interval_seconds <= 0:
        raise ValueError("service.interval_seconds must be positive")
    if service.burst_size <= 0:
        raise ValueError("service.burst_size must be positive")
    if service.capture_spacing_seconds < 0:
        raise ValueError("service.capture_spacing_seconds cannot be negative")
    if not 1 <= service.minimum_captures <= service.burst_size:
        raise ValueError("service.minimum_captures must be between 1 and burst_size")
    if service.restart_after_failures <= 0:
        raise ValueError("service.restart_after_failures must be positive")
    if (
        service.capture_spacing_seconds * max(0, service.burst_size - 1)
        >= service.interval_seconds
    ):
        raise ValueError("the configured capture burst must fit inside one interval")
    if config.camera.width <= 0 or config.camera.height <= 0:
        raise ValueError("camera width and height must be positive")
    if config.camera.rotation not in {0, 90, 180, 270}:
        raise ValueError("camera.rotation must be 0, 90, 180, or 270")
    if config.model.intra_op_threads <= 0 or config.model.inter_op_threads <= 0:
        raise ValueError("model thread counts must be positive")
    if not 0 <= config.decision.minimum_phase_confidence <= 1:
        raise ValueError("decision.minimum_phase_confidence must be in [0, 1]")
    if not 0 <= config.decision.minimum_vote_fraction <= 1:
        raise ValueError("decision.minimum_vote_fraction must be in [0, 1]")
    if config.storage.retain_images not in {"none", "uncertain", "all"}:
        raise ValueError("storage.retain_images must be none, uncertain, or all")
    if config.storage.retention_days < 0:
        raise ValueError("storage.retention_days cannot be negative")
    if not 1 <= config.storage.jpeg_quality <= 100:
        raise ValueError("storage.jpeg_quality must be in [1, 100]")

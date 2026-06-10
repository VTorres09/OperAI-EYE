"""Resolve the private OperAI-EYE dataset directly from Hugging Face."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent

HF_DATASET_REPO_ID = "OperAI-Research/operai-eye-exocentric-rgb-test"
HF_DATASET_REVISION = "b363f3b89449b9d3367e19719178e1199d0b324d"
HF_DATASET_SPLIT = "test"
HF_LABELS_FILE = "labels/test_labels.csv"
EXPECTED_TEST_IMAGES = 23_000


class DatasetPreparationError(RuntimeError):
    """Raised when the private HF dataset cannot be resolved."""


@dataclass(frozen=True)
class HFDatasetPaths:
    repo_id: str
    revision: str
    snapshot_path: Path
    split_path: Path
    labels_path: Path
    metadata_path: Path
    image_count: int


def count_pngs(path: Path) -> int:
    return sum(1 for _ in path.rglob("*.png")) if path.exists() else 0


def _token() -> str | bool:
    load_dotenv(ROOT / ".env")
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or True


@lru_cache(maxsize=8)
def get_hf_dataset_paths(
    *,
    repo_id: str = HF_DATASET_REPO_ID,
    revision: str = HF_DATASET_REVISION,
    split: str = HF_DATASET_SPLIT,
    expected_images: int = EXPECTED_TEST_IMAGES,
    max_workers: int = 8,
    local_files_only: bool = True,
) -> HFDatasetPaths:
    """Return validated paths inside the Hugging Face snapshot cache."""

    try:
        snapshot_path = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                allow_patterns=["README.md", HF_LABELS_FILE, f"{split}/**"],
                max_workers=max_workers,
                token=_token(),
                local_files_only=local_files_only,
            )
        )
    except Exception as exc:
        mode = "cached" if local_files_only else "private"
        raise DatasetPreparationError(
            f"Could not resolve the {mode} Hugging Face dataset {repo_id}@{revision}. "
            f"Hugging Face error: {exc}"
        ) from exc

    split_path = snapshot_path / split
    labels_path = snapshot_path / HF_LABELS_FILE
    metadata_path = split_path / "metadata.csv"
    missing = [
        str(path.relative_to(snapshot_path))
        for path in (split_path, labels_path, metadata_path)
        if not path.exists()
    ]
    if missing:
        raise DatasetPreparationError(
            f"Hugging Face snapshot is missing {', '.join(missing)}: {snapshot_path}"
        )

    image_count = count_pngs(split_path)
    if image_count != expected_images:
        raise DatasetPreparationError(
            f"Hugging Face snapshot has {image_count} PNGs, expected {expected_images}: {split_path}"
        )

    return HFDatasetPaths(
        repo_id=repo_id,
        revision=revision,
        snapshot_path=snapshot_path,
        split_path=split_path,
        labels_path=labels_path,
        metadata_path=metadata_path,
        image_count=image_count,
    )


def _status(paths: HFDatasetPaths) -> dict[str, Any]:
    return {
        "ready": True,
        "images_available": True,
        "labels_available": True,
        "metadata_available": True,
        "image_count": paths.image_count,
        "expected_images": EXPECTED_TEST_IMAGES,
        "snapshot_path": str(paths.snapshot_path),
        "split_path": str(paths.split_path),
        "labels_path": str(paths.labels_path),
        "metadata_path": str(paths.metadata_path),
        "repo_id": paths.repo_id,
        "revision": paths.revision,
        "error": None,
    }


def dataset_status() -> dict[str, Any]:
    """Report whether the pinned dataset is available in the HF cache."""

    try:
        return _status(get_hf_dataset_paths(local_files_only=True))
    except DatasetPreparationError as exc:
        return {
            "ready": False,
            "images_available": False,
            "labels_available": False,
            "metadata_available": False,
            "image_count": 0,
            "expected_images": EXPECTED_TEST_IMAGES,
            "snapshot_path": None,
            "split_path": None,
            "labels_path": None,
            "metadata_path": None,
            "repo_id": HF_DATASET_REPO_ID,
            "revision": HF_DATASET_REVISION,
            "error": str(exc),
        }


def prepare_hf_dataset(
    *,
    repo_id: str = HF_DATASET_REPO_ID,
    revision: str = HF_DATASET_REVISION,
    split: str = HF_DATASET_SPLIT,
    expected_images: int = EXPECTED_TEST_IMAGES,
    max_workers: int = 8,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Download and validate the dataset, returning its HF cache paths."""

    paths = get_hf_dataset_paths(
        repo_id=repo_id,
        revision=revision,
        split=split,
        expected_images=expected_images,
        max_workers=max_workers,
        local_files_only=local_files_only,
    )
    return _status(paths)

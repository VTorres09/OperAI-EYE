"""Utilities for using the private OperAI-EYE Hugging Face dataset cache."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent

HF_DATASET_REPO_ID = "OperAI-Research/operai-eye-exocentric-rgb-test"
HF_DATASET_REVISION = "b363f3b89449b9d3367e19719178e1199d0b324d"
HF_DATASET_SPLIT = "test"
EXPECTED_TEST_IMAGES = 23_000

DATA_DIR = ROOT / "data" / "exocentric_rgb"
LABELS_PATH = ROOT / "output" / "test_labels.csv"


class DatasetPreparationError(RuntimeError):
    """Raised when the private HF dataset cannot be prepared locally."""


def count_pngs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*.png"))


def dataset_status(
    *,
    data_dir: Path = DATA_DIR,
    labels_path: Path = LABELS_PATH,
    split: str = HF_DATASET_SPLIT,
    expected_images: int = EXPECTED_TEST_IMAGES,
) -> dict[str, Any]:
    split_path = data_dir / split
    image_count = count_pngs(split_path)
    labels_available = labels_path.exists()
    metadata_available = (split_path / "metadata.csv").exists()
    split_available = image_count == expected_images

    return {
        "ready": split_available and labels_available,
        "images_available": split_available,
        "labels_available": labels_available,
        "metadata_available": metadata_available,
        "image_count": image_count,
        "expected_images": expected_images,
        "data_dir": str(data_dir),
        "split_path": str(split_path),
        "split_is_symlink": split_path.is_symlink(),
        "split_target": str(split_path.resolve()) if split_path.exists() else None,
        "labels_path": str(labels_path),
        "labels_is_symlink": labels_path.is_symlink(),
        "labels_target": str(labels_path.resolve()) if labels_path.exists() else None,
        "repo_id": HF_DATASET_REPO_ID,
        "revision": HF_DATASET_REVISION,
    }


def _token() -> str | bool:
    load_dotenv(ROOT / ".env")
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or True


def _replace_with_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.is_symlink() or link_path.is_file():
        link_path.unlink()
    elif link_path.exists():
        shutil.rmtree(link_path)

    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target_path, target_is_directory=target_path.is_dir())


def _link_labels(snapshot_path: Path, labels_path: Path, overwrite: bool) -> None:
    source_labels = snapshot_path / "labels" / "test_labels.csv"
    if not source_labels.exists():
        raise DatasetPreparationError(f"HF snapshot is missing labels/test_labels.csv: {snapshot_path}")

    if (
        labels_path.is_symlink()
        and labels_path.exists()
        and labels_path.resolve() == source_labels.resolve()
    ):
        return

    if labels_path.exists() and not overwrite:
        return

    _replace_with_symlink(labels_path, source_labels)


def prepare_hf_dataset(
    *,
    repo_id: str = HF_DATASET_REPO_ID,
    revision: str = HF_DATASET_REVISION,
    data_dir: Path = DATA_DIR,
    labels_path: Path = LABELS_PATH,
    split: str = HF_DATASET_SPLIT,
    expected_images: int = EXPECTED_TEST_IMAGES,
    max_workers: int = 8,
    local_files_only: bool = False,
    overwrite_labels: bool = False,
) -> dict[str, Any]:
    """Download the private dataset into the HF cache and point local paths at it.

    Images and labels remain in the Hugging Face cache. Local paths become
    symlinks to the cached snapshot so fresh machines and local explorers use
    Hugging Face as the single source of truth.
    """

    try:
        snapshot_path = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                allow_patterns=[
                    "README.md",
                    "labels/test_labels.csv",
                    f"{split}/**",
                ],
                max_workers=max_workers,
                token=_token(),
                local_files_only=local_files_only,
            )
        )
    except Exception as exc:
        raise DatasetPreparationError(
            "Could not prepare the private Hugging Face dataset. Set HF_TOKEN "
            "with access to OperAI-Research/operai-eye-exocentric-rgb-test, "
            f"then retry. Hugging Face error: {exc}"
        ) from exc

    cached_split = snapshot_path / split
    cached_metadata = cached_split / "metadata.csv"
    if not cached_split.is_dir() or not cached_metadata.exists():
        raise DatasetPreparationError(f"HF snapshot is missing {split}/metadata.csv: {snapshot_path}")

    image_count = count_pngs(cached_split)
    if image_count != expected_images:
        raise DatasetPreparationError(
            f"HF snapshot has {image_count} PNGs, expected {expected_images}: {cached_split}"
        )

    _link_labels(snapshot_path, labels_path, overwrite=overwrite_labels)
    split_path = data_dir / split
    if not split_path.exists() or not split_path.is_symlink() or split_path.resolve() != cached_split.resolve():
        _replace_with_symlink(split_path, cached_split)

    return dataset_status(
        data_dir=data_dir,
        labels_path=labels_path,
        split=split,
        expected_images=expected_images,
    )

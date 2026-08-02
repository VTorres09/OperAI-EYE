"""Resolve, extract, correct, and publish the private SFT Hugging Face dataset."""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

from operai_eye.paths import OUTPUT_DIR, PROJECT_ROOT
from operai_eye.pipeline.prepare_sft_dataset import (
    CORE_PHASES,
    DEFAULT_REPO_ID,
    DEFAULT_REVISION_PATH,
)

DEFAULT_WORK_DIR = OUTPUT_DIR / "hf_sft_dataset"
DEFAULT_CORRECTIONS_PATH = OUTPUT_DIR / "sft_label_corrections.csv"
DEFAULT_CORRECTED_STAGE_DIR = OUTPUT_DIR / "sft_dataset_corrected"
SFT_SPLITS = ("train", "validation")
CORE_PHASE_SET = set(CORE_PHASES)
CORRECTION_FIELDNAMES = [
    "path",
    "split",
    "original_phase",
    "corrected_phase",
    "note",
    "reviewed",
    "updated_at",
]


class SFTDatasetPreparationError(RuntimeError):
    """Raised when the private SFT dataset cannot be resolved."""


@dataclass(frozen=True)
class HFSFTDatasetPaths:
    repo_id: str
    revision: str
    snapshot_path: Path
    work_dir: Path
    extracted_dir: Path
    labels_paths: dict[str, Path]
    split_dirs: dict[str, Path]
    image_count: int


def _token() -> str | bool:
    load_dotenv(PROJECT_ROOT / ".env")
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or True


def resolve_revision(
    revision: str | None = None,
    revision_file: Path = DEFAULT_REVISION_PATH,
) -> str:
    if revision:
        return revision
    if revision_file.exists():
        value = json.loads(revision_file.read_text(encoding="utf-8"))
        return value["revision"]
    return "main"


def _safe_revision_dir(repo_id: str, revision: str) -> str:
    safe_repo = repo_id.replace("/", "__")
    safe_revision = revision.replace("/", "_")
    return f"{safe_repo}__{safe_revision}"


def _extract_archives(snapshot_path: Path, extracted_dir: Path) -> None:
    extracted_dir.mkdir(parents=True, exist_ok=True)
    for split in SFT_SPLITS:
        archive_path = snapshot_path / f"{split}.zip"
        if not archive_path.exists():
            raise SFTDatasetPreparationError(f"Missing archive: {archive_path}")
        split_dir = extracted_dir / split
        metadata_path = split_dir / "metadata.csv"
        if metadata_path.exists():
            continue
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(split_dir)


@lru_cache(maxsize=8)
def get_hf_sft_dataset_paths(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    revision_file: str = str(DEFAULT_REVISION_PATH),
    work_dir: str = str(DEFAULT_WORK_DIR),
    max_workers: int = 8,
    local_files_only: bool = True,
) -> HFSFTDatasetPaths:
    resolved_revision = resolve_revision(revision, Path(revision_file))
    try:
        snapshot_path = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=resolved_revision,
                allow_patterns=[
                    "README.md",
                    "analysis_manifest.json",
                    "labels/*.csv",
                    "train.zip",
                    "validation.zip",
                ],
                max_workers=max_workers,
                token=_token(),
                local_files_only=local_files_only,
            )
        )
    except Exception as exc:
        mode = "cached" if local_files_only else "private"
        raise SFTDatasetPreparationError(
            f"Could not resolve the {mode} Hugging Face SFT dataset "
            f"{repo_id}@{resolved_revision}. Hugging Face error: {exc}"
        ) from exc

    base_work_dir = Path(work_dir) / _safe_revision_dir(repo_id, resolved_revision)
    extracted_dir = base_work_dir / "extracted"
    _extract_archives(snapshot_path, extracted_dir)

    labels_paths = {
        split: snapshot_path / "labels" / f"{split}_labels.csv" for split in SFT_SPLITS
    }
    split_dirs = {split: extracted_dir / split for split in SFT_SPLITS}
    missing = [
        str(path)
        for path in [*labels_paths.values(), *split_dirs.values()]
        if not path.exists()
    ]
    if missing:
        raise SFTDatasetPreparationError(
            f"SFT dataset is missing required files: {', '.join(missing)}"
        )

    image_count = sum(
        1 for split_dir in split_dirs.values() for _ in split_dir.rglob("*.png")
    )
    return HFSFTDatasetPaths(
        repo_id=repo_id,
        revision=resolved_revision,
        snapshot_path=snapshot_path,
        work_dir=base_work_dir,
        extracted_dir=extracted_dir,
        labels_paths=labels_paths,
        split_dirs=split_dirs,
        image_count=image_count,
    )


def prepare_hf_sft_dataset(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    revision_file: Path = DEFAULT_REVISION_PATH,
    work_dir: Path = DEFAULT_WORK_DIR,
    max_workers: int = 8,
    local_files_only: bool = False,
) -> HFSFTDatasetPaths:
    get_hf_sft_dataset_paths.cache_clear()
    return get_hf_sft_dataset_paths(
        repo_id=repo_id,
        revision=revision,
        revision_file=str(revision_file),
        work_dir=str(work_dir),
        max_workers=max_workers,
        local_files_only=local_files_only,
    )


def hf_sft_dataset_status(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    revision_file: Path = DEFAULT_REVISION_PATH,
    work_dir: Path = DEFAULT_WORK_DIR,
) -> dict[str, Any]:
    try:
        paths = get_hf_sft_dataset_paths(
            repo_id=repo_id,
            revision=revision,
            revision_file=str(revision_file),
            work_dir=str(work_dir),
            local_files_only=True,
        )
    except SFTDatasetPreparationError as exc:
        error = str(exc)
        cache_missing = "Cannot find an appropriate cached snapshot" in error
        return {
            "ready": False,
            "repo_id": repo_id,
            "revision": resolve_revision(revision, revision_file),
            "snapshot_path": None,
            "extracted_dir": None,
            "image_count": 0,
            "cache_missing": cache_missing,
            "error": error,
        }
    return {
        "ready": True,
        "repo_id": paths.repo_id,
        "revision": paths.revision,
        "snapshot_path": str(paths.snapshot_path),
        "extracted_dir": str(paths.extracted_dir),
        "image_count": paths.image_count,
        "cache_missing": False,
        "error": None,
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_label_rows(dataset: HFSFTDatasetPaths) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in SFT_SPLITS:
        rows.extend(read_csv_rows(dataset.labels_paths[split]))
    return rows


def read_corrections(
    corrections_path: Path = DEFAULT_CORRECTIONS_PATH,
) -> dict[str, dict[str, str]]:
    if not corrections_path.exists():
        return {}
    with corrections_path.open(newline="", encoding="utf-8") as handle:
        return {row["path"]: row for row in csv.DictReader(handle) if row.get("path")}


def write_corrections(
    corrections: dict[str, dict[str, str]],
    corrections_path: Path = DEFAULT_CORRECTIONS_PATH,
) -> None:
    corrections_path.parent.mkdir(parents=True, exist_ok=True)
    with corrections_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CORRECTION_FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            corrections[path]
            for path in sorted(corrections)
            if corrections[path].get("reviewed") == "true"
            or corrections[path].get("corrected_phase")
        )


def apply_corrections_to_rows(
    rows: Iterable[dict[str, str]],
    corrections: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    corrected_rows = []
    for row in rows:
        copied = dict(row)
        path = copied.get("path") or copied.get("source_path") or ""
        corrected_phase = corrections.get(path, {}).get("corrected_phase", "")
        if corrected_phase:
            copied["phase"] = corrected_phase
            if "label" in copied:
                copied["label"] = corrected_phase
        corrected_rows.append(copied)
    return corrected_rows


def upsert_correction(
    *,
    path: str,
    split: str,
    original_phase: str,
    corrected_phase: str = "",
    note: str = "",
    reviewed: bool = True,
    corrections_path: Path = DEFAULT_CORRECTIONS_PATH,
) -> dict[str, str]:
    if corrected_phase and corrected_phase not in {*CORE_PHASE_SET, "UNKNOWN"}:
        raise ValueError(f"Invalid corrected phase: {corrected_phase}")
    corrections = read_corrections(corrections_path)
    row = {
        "path": path,
        "split": split,
        "original_phase": original_phase,
        "corrected_phase": corrected_phase,
        "note": note,
        "reviewed": "true" if reviewed else "false",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    corrections[path] = row
    write_corrections(corrections, corrections_path)
    return row


def source_to_archive_path(source_path: str) -> Path:
    parts = Path(source_path).parts
    if parts and parts[0] in SFT_SPLITS:
        return Path(*parts[1:])
    return Path(source_path)


def image_path_for_source(dataset: HFSFTDatasetPaths, source_path: str) -> Path | None:
    parts = Path(source_path).parts
    if not parts or parts[0] not in SFT_SPLITS:
        return None
    path = dataset.split_dirs[parts[0]] / source_to_archive_path(source_path)
    return path if path.is_file() else None


def _write_csv_to_string(rows: list[dict[str, str]], fieldnames: list[str]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _read_metadata(dataset: HFSFTDatasetPaths, split: str) -> list[dict[str, str]]:
    return read_csv_rows(dataset.split_dirs[split] / "metadata.csv")


def build_corrected_stage(
    *,
    dataset: HFSFTDatasetPaths,
    corrections_path: Path = DEFAULT_CORRECTIONS_PATH,
    stage_dir: Path = DEFAULT_CORRECTED_STAGE_DIR,
) -> dict[str, Any]:
    corrections = read_corrections(corrections_path)
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    (stage_dir / "labels").mkdir()

    split_summary: dict[str, Any] = {}
    for split in SFT_SPLITS:
        labels = apply_corrections_to_rows(
            read_csv_rows(dataset.labels_paths[split]),
            corrections,
        )
        if labels:
            labels_fieldnames = list(labels[0])
            labels_text = _write_csv_to_string(labels, labels_fieldnames)
            (stage_dir / "labels" / f"{split}_labels.csv").write_text(
                labels_text,
                encoding="utf-8",
            )

        metadata_rows = apply_corrections_to_rows(
            _read_metadata(dataset, split),
            corrections,
        )
        staged_rows = [
            row for row in metadata_rows if row.get("phase") in CORE_PHASE_SET
        ]
        if not staged_rows:
            raise SFTDatasetPreparationError(f"No trainable rows for {split}")
        metadata_fieldnames = list(staged_rows[0])
        metadata_text = _write_csv_to_string(staged_rows, metadata_fieldnames)

        archive_path = stage_dir / f"{split}.zip"
        with zipfile.ZipFile(
            archive_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            archive.writestr("metadata.csv", metadata_text)
            for row in staged_rows:
                file_name = row["file_name"]
                source_file = dataset.split_dirs[split] / file_name
                if not source_file.exists():
                    raise SFTDatasetPreparationError(
                        f"Missing image while staging {split}: {source_file}"
                    )
                archive.write(source_file, arcname=file_name)

        phase_counts = Counter(row.get("phase", "") for row in labels)
        staged_counts = Counter(row.get("phase", "") for row in staged_rows)
        split_summary[split] = {
            "labels": len(labels),
            "staged": len(staged_rows),
            "ignored_unknown": phase_counts.get("UNKNOWN", 0),
            "label_phase_counts": dict(sorted(phase_counts.items())),
            "staged_phase_counts": dict(sorted(staged_counts.items())),
        }

    readme = dataset.snapshot_path / "README.md"
    if readme.exists():
        shutil.copy2(readme, stage_dir / "README.md")
    else:
        (stage_dir / "README.md").write_text(
            "# OperAI EYE Exocentric RGB SFT\n",
            encoding="utf-8",
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_repo_id": dataset.repo_id,
        "source_revision": dataset.revision,
        "corrections_path": str(corrections_path),
        "corrections_applied": sum(
            1 for row in corrections.values() if row.get("corrected_phase")
        ),
        "reviewed_without_change": sum(
            1
            for row in corrections.values()
            if row.get("reviewed") == "true" and not row.get("corrected_phase")
        ),
        "splits": split_summary,
    }
    (stage_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"stage_dir": str(stage_dir), **manifest}


def publish_corrected_stage(
    *,
    stage_dir: Path = DEFAULT_CORRECTED_STAGE_DIR,
    repo_id: str = DEFAULT_REPO_ID,
    revision_output: Path = DEFAULT_REVISION_PATH,
    workers: int = 8,
) -> dict[str, Any]:
    token = _token()
    if token is True:
        raise SFTDatasetPreparationError(
            "HF_TOKEN or HUGGINGFACE_HUB_TOKEN is required to publish corrections"
        )
    if not stage_dir.exists():
        raise SFTDatasetPreparationError(f"Corrected stage not found: {stage_dir}")

    api = HfApi(token=token)
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=stage_dir,
        num_workers=workers,
    )
    revision = api.repo_info(repo_id=repo_id, repo_type="dataset").sha
    result = {
        "repo_id": repo_id,
        "revision": revision,
        "stage_dir": str(stage_dir),
        "published_at": datetime.now(timezone.utc).isoformat(),
    }
    revision_output.parent.mkdir(parents=True, exist_ok=True)
    revision_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    get_hf_sft_dataset_paths.cache_clear()
    return result

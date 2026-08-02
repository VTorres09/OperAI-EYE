from __future__ import annotations

from pathlib import Path
from threading import Lock, Thread
from typing import Any, Optional

from operai_eye.pipeline.audit_labels import LabelRecord, audit_records
from operai_eye.pipeline.hf_sft_dataset import (
    CORE_PHASE_SET,
    DEFAULT_CORRECTED_STAGE_DIR,
    DEFAULT_CORRECTIONS_PATH,
    HFSFTDatasetPaths,
    SFTDatasetPreparationError,
    apply_corrections_to_rows,
    build_corrected_stage,
    get_hf_sft_dataset_paths,
    hf_sft_dataset_status,
    image_path_for_source,
    prepare_hf_sft_dataset,
    publish_corrected_stage,
    read_corrections,
    read_label_rows,
    upsert_correction,
)
from operai_eye.pipeline.prepare_sft_dataset import (
    DEFAULT_REPO_ID,
    DEFAULT_REVISION_PATH,
)

_audit_prepare_lock = Lock()
_audit_prepare_state = {"state": "idle", "message": "", "error": None}
_publish_lock = Lock()
_publish_state = {
    "state": "idle",
    "message": "",
    "error": None,
    "revision": None,
    "published_at": None,
}
_candidate_cache_lock = Lock()
_candidate_cache: dict[str, Any] = {}


def _set_prepare_state(state: str, message: str = "", error: str | None = None) -> None:
    with _audit_prepare_lock:
        _audit_prepare_state.update(
            {"state": state, "message": message, "error": error}
        )


def _get_prepare_state() -> dict[str, Any]:
    with _audit_prepare_lock:
        return dict(_audit_prepare_state)


def _set_publish_state(
    state: str,
    message: str = "",
    error: str | None = None,
    revision: str | None = None,
    published_at: str | None = None,
) -> None:
    with _publish_lock:
        _publish_state.update(
            {
                "state": state,
                "message": message,
                "error": error,
                "revision": revision,
                "published_at": published_at,
            }
        )


def _get_publish_state() -> dict[str, Any]:
    with _publish_lock:
        return dict(_publish_state)


def _prepare_job() -> None:
    _set_prepare_state("running", "Downloading SFT dataset from Hugging Face")
    try:
        paths = prepare_hf_sft_dataset()
    except Exception as exc:
        _set_prepare_state("error", "SFT dataset preparation failed", str(exc))
        return
    _set_prepare_state(
        "complete",
        f"SFT dataset ready with {paths.image_count} extracted images",
    )


def _publish_job() -> None:
    _set_publish_state("running", "Uploading corrected dataset to Hugging Face")
    try:
        result = publish_corrected_stage(
            stage_dir=DEFAULT_CORRECTED_STAGE_DIR,
            repo_id=DEFAULT_REPO_ID,
            revision_output=DEFAULT_REVISION_PATH,
        )
    except Exception as exc:
        _set_publish_state("error", "Publish failed", str(exc))
        return
    _set_publish_state(
        "complete",
        f"Published revision {result['revision'][:10]}",
        revision=result["revision"],
        published_at=result["published_at"],
    )


def get_audit_status() -> dict[str, Any]:
    status = hf_sft_dataset_status()
    status["prepare"] = _get_prepare_state()
    status["publish"] = _get_publish_state()
    status["corrections_path"] = str(DEFAULT_CORRECTIONS_PATH)
    status["corrected_stage_dir"] = str(DEFAULT_CORRECTED_STAGE_DIR)
    return status


def prepare_audit_dataset() -> dict[str, Any]:
    status = hf_sft_dataset_status()
    prepare = _get_prepare_state()
    if status["ready"] or prepare["state"] == "running":
        return {**status, "prepare": prepare}

    _set_prepare_state("running", "Starting SFT dataset download")
    Thread(target=_prepare_job, daemon=True).start()
    status = hf_sft_dataset_status()
    status["prepare"] = _get_prepare_state()
    return status


def get_dataset() -> HFSFTDatasetPaths:
    return get_hf_sft_dataset_paths(local_files_only=True)


def get_audit_image_path(source_path: Path) -> Optional[Path]:
    try:
        dataset = get_dataset()
    except SFTDatasetPreparationError:
        return None
    return image_path_for_source(dataset, str(source_path))


def _records_from_dataset(
    dataset: HFSFTDatasetPaths,
    corrections_path: Path = DEFAULT_CORRECTIONS_PATH,
) -> list[LabelRecord]:
    corrections = read_corrections(corrections_path)
    corrected_rows = apply_corrections_to_rows(read_label_rows(dataset), corrections)
    records: list[LabelRecord] = []
    for row in corrected_rows:
        path = (row.get("path") or "").strip()
        phase = (row.get("phase") or "").strip()
        if not path or not phase:
            continue
        try:
            frame_id = int((row.get("frame_id") or "").strip())
        except ValueError:
            continue
        try:
            confidence = float(row.get("confidence") or "")
        except ValueError:
            confidence = None
        records.append(
            LabelRecord(
                path=path,
                split=(row.get("split") or "").strip(),
                surgery_type=(row.get("surgery_type") or "").strip(),
                procedure_id=(row.get("procedure_id") or "").strip(),
                take_id=(row.get("take_id") or "").strip(),
                camera=(row.get("camera") or "").strip(),
                frame_id=frame_id,
                phase=phase,
                confidence=confidence,
                row=row,
            )
        )
    return records


def _candidate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_priority: dict[str, int] = {}
    by_split: dict[str, int] = {}
    for row in rows:
        by_priority[row["priority"]] = by_priority.get(row["priority"], 0) + 1
        by_split[row["split"]] = by_split.get(row["split"], 0) + 1
    return {
        "total": len(rows),
        "by_priority": dict(sorted(by_priority.items())),
        "by_split": dict(sorted(by_split.items())),
    }


def _cached_suspects(dataset: HFSFTDatasetPaths) -> list[dict[str, Any]]:
    corrections_mtime = (
        DEFAULT_CORRECTIONS_PATH.stat().st_mtime
        if DEFAULT_CORRECTIONS_PATH.exists()
        else 0
    )
    cache_key = f"{dataset.repo_id}@{dataset.revision}:{corrections_mtime}"
    with _candidate_cache_lock:
        if _candidate_cache.get("key") == cache_key:
            return list(_candidate_cache["suspects"])

    records = _records_from_dataset(dataset)
    suspects = audit_records(
        records,
        data_dir=dataset.extracted_dir,
        max_neighbor_gap=45,
        window=90,
        minimum_neighbors=3,
        majority_threshold=0.75,
        low_confidence=0.55,
        hash_distance=6,
        image_hash=True,
    )
    with _candidate_cache_lock:
        _candidate_cache.clear()
        _candidate_cache.update({"key": cache_key, "suspects": suspects})
    return list(suspects)


def get_audit_candidates(
    *,
    page: int = 1,
    page_size: int = 24,
    priority: Optional[str] = None,
    split: Optional[str] = None,
    reason: Optional[str] = None,
    reviewed: Optional[bool] = None,
) -> dict[str, Any]:
    dataset = get_dataset()
    corrections = read_corrections()
    suspects = _cached_suspects(dataset)

    filtered = []
    for row in suspects:
        correction = corrections.get(row["path"], {})
        is_reviewed = correction.get("reviewed") == "true"
        if priority and row["priority"] != priority:
            continue
        if split and row["split"] != split:
            continue
        if reason and reason not in row["reasons"]:
            continue
        if reviewed is not None and is_reviewed != reviewed:
            continue
        enriched = dict(row)
        enriched["image_url"] = f"/sft-images/{row['path']}"
        enriched["correction"] = correction or None
        enriched["reviewed"] = is_reviewed
        filtered.append(enriched)

    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "items": filtered[start:end],
        "summary": _candidate_summary(suspects),
        "filtered_summary": _candidate_summary(filtered),
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": (total + page_size - 1) // page_size,
        "dataset": {
            "repo_id": dataset.repo_id,
            "revision": dataset.revision,
            "image_count": dataset.image_count,
        },
    }


def save_audit_correction(
    *,
    path: str,
    split: str,
    original_phase: str,
    corrected_phase: str = "",
    note: str = "",
    reviewed: bool = True,
) -> dict[str, str]:
    result = upsert_correction(
        path=path,
        split=split,
        original_phase=original_phase,
        corrected_phase=corrected_phase,
        note=note,
        reviewed=reviewed,
    )
    with _candidate_cache_lock:
        _candidate_cache.clear()
    return result


def stage_audit_corrections() -> dict[str, Any]:
    dataset = get_dataset()
    return build_corrected_stage(dataset=dataset)


def publish_audit_corrections() -> dict[str, Any]:
    state = _get_publish_state()
    if state["state"] == "running":
        return state
    _set_publish_state("running", "Starting Hugging Face upload")
    Thread(target=_publish_job, daemon=True).start()
    return _get_publish_state()


def audit_filter_options() -> dict[str, Any]:
    return {
        "priorities": ["high", "medium", "low"],
        "splits": ["train", "validation"],
        "reasons": [
            "isolated_temporal_island",
            "local_majority_mismatch",
            "low_confidence",
            "visual_near_duplicate_disagreement",
        ],
        "phases": sorted([*CORE_PHASE_SET, "UNKNOWN"]),
    }

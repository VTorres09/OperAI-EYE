from pathlib import Path
from typing import Optional

import pandas as pd

from hf_dataset import DatasetPreparationError, get_hf_dataset_paths
from hf_sft_dataset import (
    SFTDatasetPreparationError,
    get_hf_sft_dataset_paths,
    image_path_for_source,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
LABEL_COLUMNS = [
    "path",
    "split",
    "surgery_type",
    "procedure_id",
    "take_id",
    "camera",
    "frame_id",
    "phase",
    "confidence",
    "key_visual_cues",
]
SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}

_df: Optional[pd.DataFrame] = None


def reset_cache() -> None:
    global _df
    _df = None


def get_df() -> pd.DataFrame:
    global _df
    if _df is not None:
        return _df

    frames = []
    try:
        test_paths = get_hf_dataset_paths(local_files_only=True)
    except DatasetPreparationError:
        pass
    else:
        test_df = pd.read_csv(test_paths.labels_path, dtype={"frame_id": str})
        test_df = test_df[test_df["path"] != "path"].copy()
        test_df["split"] = test_df["split"].fillna("test")
        frames.append(test_df)

    try:
        sft_paths = get_hf_sft_dataset_paths(local_files_only=True)
    except SFTDatasetPreparationError:
        pass
    else:
        sft_frames = []
        for split, labels_path in sft_paths.labels_paths.items():
            split_df = pd.read_csv(labels_path, dtype={"frame_id": str})
            split_df = split_df[split_df["path"] != "path"].copy()
            split_df["split"] = split
            split_df = split_df[
                split_df["path"].map(
                    lambda path: image_path_for_source(sft_paths, str(path))
                    is not None
                )
            ]
            sft_frames.append(split_df)
        if sft_frames:
            frames.append(pd.concat(sft_frames, ignore_index=True))

    if frames:
        _df = pd.concat(frames, ignore_index=True)
        for column in LABEL_COLUMNS:
            if column not in _df.columns:
                _df[column] = None
        _df = _df[LABEL_COLUMNS]
        _df["confidence"] = pd.to_numeric(_df["confidence"], errors="coerce")
        _df = _df.reset_index(drop=True)
    else:
        _df = pd.DataFrame(columns=LABEL_COLUMNS)
    return _df


def get_image_path(relative_path: Path) -> Optional[Path]:
    parts = relative_path.parts
    if parts and parts[0] in {"train", "validation"}:
        try:
            sft_paths = get_hf_sft_dataset_paths(local_files_only=True)
        except SFTDatasetPreparationError:
            return None
        return image_path_for_source(sft_paths, str(relative_path))

    try:
        snapshot_path = get_hf_dataset_paths(local_files_only=True).snapshot_path
    except DatasetPreparationError:
        return None
    image_path = snapshot_path / relative_path
    return image_path if image_path.is_file() else None


def _ordered_values(values: list) -> list:
    return sorted(
        values,
        key=lambda value: (
            SPLIT_ORDER.get(str(value), 100),
            str(value),
        ),
    )


def _apply_filters(
    df: pd.DataFrame,
    split: Optional[str] = None,
    phase: Optional[str] = None,
    surgery_type: Optional[str] = None,
    camera: Optional[str] = None,
    procedure_id: Optional[int] = None,
    take_id: Optional[int] = None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if split:
        mask &= df["split"] == split
    if phase:
        mask &= df["phase"] == phase
    if surgery_type:
        mask &= df["surgery_type"] == surgery_type
    if camera:
        mask &= df["camera"] == camera
    if procedure_id is not None:
        mask &= df["procedure_id"] == procedure_id
    if take_id is not None:
        mask &= df["take_id"] == take_id
    return df[mask]


def get_filter_options() -> dict:
    df = get_df()
    return {
        "splits": _ordered_values(df["split"].dropna().unique().tolist()),
        "phases": sorted(df["phase"].dropna().unique().tolist()),
        "surgery_types": sorted(df["surgery_type"].dropna().unique().tolist()),
        "cameras": sorted(df["camera"].dropna().unique().tolist()),
        "procedure_ids": sorted(df["procedure_id"].dropna().unique().tolist()),
        "take_ids": sorted(df["take_id"].dropna().unique().tolist()),
    }


def get_model_predictions(model_id: str) -> Optional[pd.DataFrame]:
    from .eval_data import get_metadata
    metadata = get_metadata()
    if model_id not in metadata.get("models", {}):
        return None
    info = metadata["models"][model_id]
    results_file = OUTPUT_DIR / info["results_file"]
    if not results_file.exists():
        return None
    return pd.read_csv(results_file)


def get_stats(
    split: Optional[str] = None,
    phase: Optional[str] = None,
    surgery_type: Optional[str] = None,
    camera: Optional[str] = None,
    procedure_id: Optional[int] = None,
    take_id: Optional[int] = None,
) -> dict:
    df = get_df()
    filtered = _apply_filters(
        df, split, phase, surgery_type, camera, procedure_id, take_id
    )
    total = len(filtered)

    def counts(col: str) -> list[dict]:
        vc = filtered[col].value_counts()
        return [
            {"label": str(k), "count": int(v), "pct": round(int(v) / total * 100, 1) if total else 0}
            for k, v in vc.items()
        ]

    return {
        "total": total,
        "by_split": counts("split"),
        "by_phase": counts("phase"),
        "by_camera": counts("camera"),
        "by_surgery_type": counts("surgery_type"),
    }


def get_images(
    page: int = 1,
    page_size: int = 20,
    split: Optional[str] = None,
    phase: Optional[str] = None,
    surgery_type: Optional[str] = None,
    camera: Optional[str] = None,
    procedure_id: Optional[int] = None,
    take_id: Optional[int] = None,
    model_id: Optional[str] = None,
) -> dict:
    df = get_df()
    filtered = _apply_filters(
        df, split, phase, surgery_type, camera, procedure_id, take_id
    )
    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    page_df = filtered.iloc[start:end]

    pred_df = None
    if model_id:
        pred_df = get_model_predictions(model_id)
        if pred_df is not None:
            pred_map = dict(zip(pred_df["path"], pred_df["predicted"]))
        else:
            pred_map = {}
    else:
        pred_map = {}

    items = []
    for _, row in page_df.iterrows():
        path = str(row["path"])
        items.append({
            "path": path,
            "image_url": f"/images/{path}",
            "split": str(row["split"]) if pd.notna(row["split"]) else None,
            "surgery_type": str(row["surgery_type"]) if pd.notna(row["surgery_type"]) else None,
            "procedure_id": int(row["procedure_id"]) if pd.notna(row["procedure_id"]) else None,
            "take_id": int(row["take_id"]) if pd.notna(row["take_id"]) else None,
            "camera": str(row["camera"]) if pd.notna(row["camera"]) else None,
            "frame_id": str(row["frame_id"]),
            "phase": str(row["phase"]) if pd.notna(row["phase"]) else None,
            "confidence": float(row["confidence"]) if pd.notna(row["confidence"]) else None,
            "key_visual_cues": str(row["key_visual_cues"]) if pd.notna(row.get("key_visual_cues")) else "",
            "model_predicted": pred_map.get(path),
        })

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }

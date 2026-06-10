from pathlib import Path
from typing import Optional

import pandas as pd

from hf_dataset import DatasetPreparationError, get_hf_dataset_paths

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

_df: Optional[pd.DataFrame] = None


def reset_cache() -> None:
    global _df
    _df = None


def get_df() -> pd.DataFrame:
    global _df
    try:
        labels_path = get_hf_dataset_paths(local_files_only=True).labels_path
    except DatasetPreparationError:
        return pd.DataFrame(columns=LABEL_COLUMNS)
    if _df is None:
        _df = pd.read_csv(labels_path, dtype={"frame_id": str})
        _df["confidence"] = pd.to_numeric(_df["confidence"], errors="coerce")
        _df = _df[_df["path"] != "path"].reset_index(drop=True)
    return _df


def get_image_path(relative_path: Path) -> Optional[Path]:
    try:
        snapshot_path = get_hf_dataset_paths(local_files_only=True).snapshot_path
    except DatasetPreparationError:
        return None
    image_path = snapshot_path / relative_path
    return image_path if image_path.is_file() else None


def _apply_filters(
    df: pd.DataFrame,
    phase: Optional[str] = None,
    surgery_type: Optional[str] = None,
    camera: Optional[str] = None,
    procedure_id: Optional[int] = None,
    take_id: Optional[int] = None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
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
    phase: Optional[str] = None,
    surgery_type: Optional[str] = None,
    camera: Optional[str] = None,
    procedure_id: Optional[int] = None,
    take_id: Optional[int] = None,
) -> dict:
    df = get_df()
    filtered = _apply_filters(
        df, phase, surgery_type, camera, procedure_id, take_id
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
        "by_phase": counts("phase"),
        "by_camera": counts("camera"),
        "by_surgery_type": counts("surgery_type"),
    }


def get_images(
    page: int = 1,
    page_size: int = 20,
    phase: Optional[str] = None,
    surgery_type: Optional[str] = None,
    camera: Optional[str] = None,
    procedure_id: Optional[int] = None,
    take_id: Optional[int] = None,
    model_id: Optional[str] = None,
) -> dict:
    df = get_df()
    filtered = _apply_filters(
        df, phase, surgery_type, camera, procedure_id, take_id
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

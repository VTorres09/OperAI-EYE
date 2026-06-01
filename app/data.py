from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "exocentric_rgb"
CSV_PATH = Path(__file__).resolve().parent.parent / "output" / "test_labels.csv"

_df: Optional[pd.DataFrame] = None


def get_df() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(CSV_PATH, dtype={"frame_id": str})
        _df["confidence"] = pd.to_numeric(_df["confidence"], errors="coerce")
        _df = _df[_df["path"] != "path"].reset_index(drop=True)
    return _df


def _apply_filters(
    df: pd.DataFrame,
    phase: Optional[str] = None,
    surgery_type: Optional[str] = None,
    camera: Optional[str] = None,
    procedure_id: Optional[int] = None,
    take_id: Optional[int] = None,
    confidence_min: Optional[float] = None,
    confidence_max: Optional[float] = None,
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
    if confidence_min is not None:
        mask &= df["confidence"] >= confidence_min
    if confidence_max is not None:
        mask &= df["confidence"] <= confidence_max
    return df[mask]


def get_filter_options() -> dict:
    df = get_df()
    return {
        "phases": sorted(df["phase"].dropna().unique().tolist()),
        "surgery_types": sorted(df["surgery_type"].dropna().unique().tolist()),
        "cameras": sorted(df["camera"].dropna().unique().tolist()),
        "procedure_ids": sorted(df["procedure_id"].dropna().unique().tolist()),
        "take_ids": sorted(df["take_id"].dropna().unique().tolist()),
        "confidence_range": [
            float(df["confidence"].min()),
            float(df["confidence"].max()),
        ],
    }


def get_stats(
    phase: Optional[str] = None,
    surgery_type: Optional[str] = None,
    camera: Optional[str] = None,
    procedure_id: Optional[int] = None,
    take_id: Optional[int] = None,
    confidence_min: Optional[float] = None,
    confidence_max: Optional[float] = None,
) -> dict:
    df = get_df()
    filtered = _apply_filters(
        df, phase, surgery_type, camera, procedure_id, take_id, confidence_min, confidence_max
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
        "avg_confidence": round(float(filtered["confidence"].mean()), 3) if total else None,
    }


def get_images(
    page: int = 1,
    page_size: int = 20,
    phase: Optional[str] = None,
    surgery_type: Optional[str] = None,
    camera: Optional[str] = None,
    procedure_id: Optional[int] = None,
    take_id: Optional[int] = None,
    confidence_min: Optional[float] = None,
    confidence_max: Optional[float] = None,
) -> dict:
    df = get_df()
    filtered = _apply_filters(
        df, phase, surgery_type, camera, procedure_id, take_id, confidence_min, confidence_max
    )
    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    page_df = filtered.iloc[start:end]

    items = []
    for _, row in page_df.iterrows():
        items.append({
            "path": str(row["path"]),
            "image_url": f"/images/{row['path']}",
            "split": str(row["split"]) if pd.notna(row["split"]) else None,
            "surgery_type": str(row["surgery_type"]) if pd.notna(row["surgery_type"]) else None,
            "procedure_id": int(row["procedure_id"]) if pd.notna(row["procedure_id"]) else None,
            "take_id": int(row["take_id"]) if pd.notna(row["take_id"]) else None,
            "camera": str(row["camera"]) if pd.notna(row["camera"]) else None,
            "frame_id": str(row["frame_id"]),
            "phase": str(row["phase"]) if pd.notna(row["phase"]) else None,
            "confidence": float(row["confidence"]) if pd.notna(row["confidence"]) else None,
            "key_visual_cues": str(row["key_visual_cues"]) if pd.notna(row.get("key_visual_cues")) else "",
        })

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }

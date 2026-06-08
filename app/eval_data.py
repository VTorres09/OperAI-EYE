import json
from pathlib import Path
from typing import Optional

import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "exocentric_rgb"
LABELS_CSV = OUTPUT_DIR / "test_labels.csv"
METADATA_FILE = OUTPUT_DIR / "eval_metadata.json"


def get_metadata() -> dict:
    if not METADATA_FILE.exists():
        return {"models": {}}
    with open(METADATA_FILE) as f:
        return json.load(f)


def save_metadata(metadata: dict):
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)


def register_model(
    model_id: str,
    model_name: str,
    prompt_file: str,
    description: str = "",
    labels_file: str = "test_labels.csv",
) -> dict:
    metadata = get_metadata()
    metadata["models"][model_id] = {
        "model_name": model_name,
        "prompt_file": prompt_file,
        "description": description,
        "labels_file": labels_file,
        "results_file": f"eval_results_{model_id}.csv",
    }
    save_metadata(metadata)
    return metadata["models"][model_id]


def list_models() -> list[dict]:
    metadata = get_metadata()
    models = []
    for model_id, info in metadata.get("models", {}).items():
        results_file = OUTPUT_DIR / info["results_file"]
        if results_file.exists():
            df = pd.read_csv(results_file)
            info["total_evaluated"] = len(df)
            info["accuracy"] = float((df["predicted"] == df["ground_truth"]).mean())
        else:
            info["total_evaluated"] = 0
            info["accuracy"] = None
        models.append({"model_id": model_id, **info})
    return sorted(models, key=lambda x: x.get("total_evaluated", 0), reverse=True)


def _load_labels_df() -> Optional[pd.DataFrame]:
    if not LABELS_CSV.exists():
        return None
    df = pd.read_csv(LABELS_CSV, dtype={"frame_id": str})
    df = df[df["path"] != "path"].reset_index(drop=True)
    return df


def _apply_eval_filters(
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


def get_eval_filter_options() -> dict:
    df = _load_labels_df()
    if df is None:
        return {
            "phases": [],
            "surgery_types": [],
            "cameras": [],
            "procedure_ids": [],
            "take_ids": [],
        }
    return {
        "phases": sorted(df["phase"].dropna().unique().tolist()),
        "surgery_types": sorted(df["surgery_type"].dropna().unique().tolist()),
        "cameras": sorted(df["camera"].dropna().unique().tolist()),
        "procedure_ids": sorted(df["procedure_id"].dropna().unique().tolist()),
        "take_ids": sorted(df["take_id"].dropna().unique().tolist()),
    }


def get_eval_results(
    model_id: str,
    phase: Optional[str] = None,
    surgery_type: Optional[str] = None,
    camera: Optional[str] = None,
    procedure_id: Optional[int] = None,
    take_id: Optional[int] = None,
) -> Optional[dict]:
    metadata = get_metadata()
    if model_id not in metadata.get("models", {}):
        return None

    info = metadata["models"][model_id]
    results_file = OUTPUT_DIR / info["results_file"]
    if not results_file.exists():
        return None

    df = pd.read_csv(results_file)

    # Join with labels to get metadata for filtering
    labels_df = _load_labels_df()
    if labels_df is not None:
        meta_cols = ["path", "surgery_type", "camera", "procedure_id", "take_id"]
        df = df.merge(labels_df[meta_cols], on="path", how="left")

    # Apply filters
    if phase or surgery_type or camera or procedure_id is not None or take_id is not None:
        df = _apply_eval_filters(
            df,
            phase=phase,
            surgery_type=surgery_type,
            camera=camera,
            procedure_id=procedure_id,
            take_id=take_id,
        )

    if len(df) == 0:
        return {
            "model_id": model_id,
            "model_info": info,
            "total": 0,
            "correct": 0,
            "accuracy": 0,
            "per_class": {},
            "confusion_matrix": {},
        }

    total = len(df)
    correct = (df["predicted"] == df["ground_truth"]).sum()
    accuracy = correct / total if total > 0 else 0

    classes = sorted(set(df["ground_truth"].dropna().unique()) | set(df["predicted"].dropna().unique()))
    per_class = {}
    for cls in classes:
        tp = ((df["predicted"] == cls) & (df["ground_truth"] == cls)).sum()
        fp = ((df["predicted"] == cls) & (df["ground_truth"] != cls)).sum()
        fn = ((df["predicted"] != cls) & (df["ground_truth"] == cls)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        per_class[cls] = {
            "precision": round(float(precision), 3),
            "recall": round(float(recall), 3),
            "f1": round(float(f1), 3),
            "support": int((df["ground_truth"] == cls).sum()),
        }

    confusion = {}
    for gt in classes:
        confusion[gt] = {}
        for pred in classes:
            confusion[gt][pred] = int(((df["ground_truth"] == gt) & (df["predicted"] == pred)).sum())

    return {
        "model_id": model_id,
        "model_info": info,
        "total": int(total),
        "correct": int(correct),
        "accuracy": round(float(accuracy), 3),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def compare_models(
    model_ids: list[str],
    phase: Optional[str] = None,
    surgery_type: Optional[str] = None,
    camera: Optional[str] = None,
    procedure_id: Optional[int] = None,
    take_id: Optional[int] = None,
) -> Optional[dict]:
    results = []
    for model_id in model_ids:
        result = get_eval_results(
            model_id,
            phase=phase,
            surgery_type=surgery_type,
            camera=camera,
            procedure_id=procedure_id,
            take_id=take_id,
        )
        if result:
            results.append(result)

    if not results:
        return None

    all_classes = set()
    for r in results:
        all_classes.update(r["per_class"].keys())
    all_classes = sorted(all_classes)

    comparison = {
        "models": [],
        "classes": all_classes,
    }

    for r in results:
        model_data = {
            "model_id": r["model_id"],
            "model_name": r["model_info"]["model_name"],
            "accuracy": r["accuracy"],
            "total": r["total"],
            "per_class": {},
        }
        for cls in all_classes:
            model_data["per_class"][cls] = r["per_class"].get(cls, {
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "support": 0,
            })
        comparison["models"].append(model_data)

    return comparison

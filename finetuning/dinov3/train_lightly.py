#!/usr/bin/env python3
"""Prepare and launch LightlyTrain DINOv3 multilabel image classification."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hf_sft_dataset import (
    DEFAULT_WORK_DIR as DEFAULT_HF_SFT_WORK_DIR,
    get_hf_sft_dataset_paths,
    prepare_hf_sft_dataset,
    read_csv_rows,
    resolve_revision,
)
from label_data import DATA_DIR, OUTPUT_DIR
from prepare_sft_dataset import DEFAULT_REPO_ID, DEFAULT_REVISION_PATH

DEFAULT_WORK_DIR = OUTPUT_DIR / "lightly_dinov3"
DEFAULT_OUT_DIR = DEFAULT_WORK_DIR / "runs" / "dinov3_vitb16_multilabel"
DEFAULT_MODEL = "dinov3/vitb16"
IMAGE_COLUMN = "image_path"
LABEL_COLUMN = "label"
CLASSES = {
    0: "idle",
    1: "people_in_room",
    2: "surgery_inactive",
    3: "surgery_active",
}
PHASE_TO_LABELS = {
    "IDLE": ("idle",),
    "PATIENT_IN_ROOM": ("people_in_room", "surgery_inactive"),
    "SURGERY_ACTIVE": ("people_in_room", "surgery_active"),
}


def parse_auto_or_int(value: str) -> int | str:
    if value == "auto":
        return value
    try:
        return int(value)
    except ValueError:
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["prepare", "train"],
        help="prepare writes Lightly CSVs; train prepares then calls LightlyTrain.",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--source",
        choices=["hf", "local"],
        default="hf",
        help="Use the Hugging Face SFT dataset by default; local uses label CSVs.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision")
    parser.add_argument("--revision-file", type=Path, default=DEFAULT_REVISION_PATH)
    parser.add_argument("--sft-work-dir", type=Path, default=DEFAULT_HF_SFT_WORK_DIR)
    parser.add_argument("--dataset-local-files-only", action="store_true")
    parser.add_argument("--dataset-max-workers", type=int, default=8)
    parser.add_argument("--train-labels", type=Path, default=OUTPUT_DIR / "train_labels.csv")
    parser.add_argument(
        "--validation-labels",
        type=Path,
        default=OUTPUT_DIR / "validation_labels.csv",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--steps", default="auto")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--num-workers", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--val-every-num-steps", type=int)
    parser.add_argument("--save-every-num-steps", type=int)
    parser.add_argument("--gradient-accumulation-steps", default="auto")
    parser.add_argument("--resume-interrupted", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_label_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Label CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def phase_labels(phase: str) -> tuple[str, ...] | None:
    return PHASE_TO_LABELS.get(phase.strip().upper())


def convert_rows(
    label_rows: list[dict[str, str]],
    data_dir: Path,
) -> tuple[list[dict[str, str]], Counter[str]]:
    converted: list[dict[str, str]] = []
    skipped: Counter[str] = Counter()
    for row in label_rows:
        if (row.get("error") or "").strip():
            skipped["error"] += 1
            continue
        phase = (row.get("phase") or "").strip().upper()
        labels = phase_labels(phase)
        if labels is None:
            skipped[phase or "missing_phase"] += 1
            continue
        source_path = (row.get("path") or "").strip()
        image_path = data_dir / source_path
        if not image_path.exists():
            skipped["missing_image"] += 1
            continue
        converted.append(
            {
                IMAGE_COLUMN: str(image_path.resolve()),
                LABEL_COLUMN: ",".join(labels),
                "phase": phase,
                "source_path": source_path,
                "confidence": row.get("confidence", ""),
                "surgery_type": row.get("surgery_type", ""),
                "procedure_id": row.get("procedure_id", ""),
                "take_id": row.get("take_id", ""),
                "camera": row.get("camera", ""),
                "frame_id": row.get("frame_id", ""),
            }
        )
    return converted, skipped


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def class_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row[LABEL_COLUMN].split(","))
    return dict(sorted(counts.items()))


def phase_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(row["phase"] for row in rows).items()))


def prepare_lightly_data(
    *,
    train_labels: Path,
    validation_labels: Path,
    data_dir: Path,
    work_dir: Path,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    train_rows, train_skipped = convert_rows(read_label_rows(train_labels), data_dir)
    validation_rows, validation_skipped = convert_rows(
        read_label_rows(validation_labels),
        data_dir,
    )

    train_csv = work_dir / "train.csv"
    validation_csv = work_dir / "validation.csv"
    write_rows(train_csv, train_rows)
    write_rows(validation_csv, validation_rows)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classification_task": "multilabel",
        "classes": CLASSES,
        "phase_to_labels": PHASE_TO_LABELS,
        "csv": {
            "train_csv": str(train_csv),
            "val_csv": str(validation_csv),
            "csv_image_column": IMAGE_COLUMN,
            "csv_label_column": LABEL_COLUMN,
            "csv_label_type": "name",
            "label_delimiter": ",",
        },
        "source": {
            "data_dir": str(data_dir),
            "train_labels": str(train_labels),
            "validation_labels": str(validation_labels),
        },
        "splits": {
            "train": {
                "rows": len(train_rows),
                "phase_counts": phase_counts(train_rows),
                "class_counts": class_counts(train_rows),
                "skipped": dict(sorted(train_skipped.items())),
            },
            "validation": {
                "rows": len(validation_rows),
                "phase_counts": phase_counts(validation_rows),
                "class_counts": class_counts(validation_rows),
                "skipped": dict(sorted(validation_skipped.items())),
            },
        },
    }
    target_manifest = manifest_path or work_dir / "manifest.json"
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    target_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["manifest_path"] = str(target_manifest)
    return manifest


def prepare_lightly_data_from_hf(
    *,
    repo_id: str,
    revision: str | None,
    revision_file: Path,
    sft_work_dir: Path,
    local_files_only: bool,
    max_workers: int,
    work_dir: Path,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    if local_files_only:
        dataset = get_hf_sft_dataset_paths(
            repo_id=repo_id,
            revision=revision,
            revision_file=str(revision_file),
            work_dir=str(sft_work_dir),
            max_workers=max_workers,
            local_files_only=True,
        )
    else:
        dataset = prepare_hf_sft_dataset(
            repo_id=repo_id,
            revision=revision,
            revision_file=revision_file,
            work_dir=sft_work_dir,
            max_workers=max_workers,
            local_files_only=False,
        )

    train_rows, train_skipped = convert_rows(
        read_csv_rows(dataset.labels_paths["train"]),
        dataset.extracted_dir,
    )
    validation_rows, validation_skipped = convert_rows(
        read_csv_rows(dataset.labels_paths["validation"]),
        dataset.extracted_dir,
    )

    train_csv = work_dir / "train.csv"
    validation_csv = work_dir / "validation.csv"
    write_rows(train_csv, train_rows)
    write_rows(validation_csv, validation_rows)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classification_task": "multilabel",
        "classes": CLASSES,
        "phase_to_labels": PHASE_TO_LABELS,
        "csv": {
            "train_csv": str(train_csv),
            "val_csv": str(validation_csv),
            "csv_image_column": IMAGE_COLUMN,
            "csv_label_column": LABEL_COLUMN,
            "csv_label_type": "name",
            "label_delimiter": ",",
        },
        "source": {
            "kind": "huggingface",
            "repo_id": dataset.repo_id,
            "revision": dataset.revision,
            "requested_revision": resolve_revision(revision, revision_file),
            "snapshot_path": str(dataset.snapshot_path),
            "extracted_dir": str(dataset.extracted_dir),
        },
        "splits": {
            "train": {
                "rows": len(train_rows),
                "phase_counts": phase_counts(train_rows),
                "class_counts": class_counts(train_rows),
                "skipped": dict(sorted(train_skipped.items())),
            },
            "validation": {
                "rows": len(validation_rows),
                "phase_counts": phase_counts(validation_rows),
                "class_counts": class_counts(validation_rows),
                "skipped": dict(sorted(validation_skipped.items())),
            },
        },
    }
    target_manifest = manifest_path or work_dir / "manifest.json"
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    target_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["manifest_path"] = str(target_manifest)
    return manifest


def data_config(manifest: dict[str, Any]) -> dict[str, Any]:
    csv_config = manifest["csv"]
    return {
        "train_csv": csv_config["train_csv"],
        "val_csv": csv_config["val_csv"],
        "classes": CLASSES,
        "csv_image_column": csv_config["csv_image_column"],
        "csv_label_column": csv_config["csv_label_column"],
        "csv_label_type": csv_config["csv_label_type"],
        "label_delimiter": csv_config["label_delimiter"],
    }


def train(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    try:
        import lightly_train
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "lightly_train is not installed. Run with "
            "`uv run --with lightly-train python -m "
            "finetuning.dinov3.train_lightly train ...` or add lightly-train "
            "to your environment."
        ) from exc

    transform_args = None
    if args.image_size:
        transform_args = {"image_size": (args.image_size, args.image_size)}

    logger_args = None
    if args.val_every_num_steps:
        logger_args = {"val_every_num_steps": args.val_every_num_steps}

    save_checkpoint_args = None
    if args.save_every_num_steps:
        save_checkpoint_args = {
            "save_every_num_steps": args.save_every_num_steps,
        }

    lightly_train.train_image_classification(
        out=args.out,
        data=data_config(manifest),
        model=args.model,
        classification_task="multilabel",
        steps=parse_auto_or_int(args.steps),
        batch_size=parse_auto_or_int(args.batch_size),
        num_workers=parse_auto_or_int(args.num_workers),
        devices=parse_auto_or_int(args.devices),
        accelerator=args.accelerator,
        precision=args.precision,
        seed=args.seed,
        resume_interrupted=args.resume_interrupted,
        overwrite=args.overwrite,
        transform_args=transform_args,
        logger_args=logger_args,
        metric_args={"classwise": True},
        save_checkpoint_args=save_checkpoint_args,
        gradient_accumulation_steps=parse_auto_or_int(args.gradient_accumulation_steps),
    )


def main() -> int:
    args = parse_args()
    if args.source == "hf":
        manifest = prepare_lightly_data_from_hf(
            repo_id=args.repo_id,
            revision=args.revision,
            revision_file=args.revision_file,
            sft_work_dir=args.sft_work_dir,
            local_files_only=args.dataset_local_files_only,
            max_workers=args.dataset_max_workers,
            work_dir=args.work_dir,
            manifest_path=args.manifest,
        )
    else:
        manifest = prepare_lightly_data(
            train_labels=args.train_labels,
            validation_labels=args.validation_labels,
            data_dir=args.data_dir,
            work_dir=args.work_dir,
            manifest_path=args.manifest,
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.command == "train":
        train(args, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

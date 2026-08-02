"""Evaluate a LightlyTrain DINOv3 multilabel export on the test split."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from operai_eye.paths import OUTPUT_DIR
from operai_eye.pipeline.hf_dataset import get_hf_dataset_paths
from operai_eye.web.eval_data import register_model

DEFAULT_EXPORT = (
    OUTPUT_DIR
    / "lightly_dinov3"
    / "runs"
    / "dinov3_vitb16_multilabel"
    / "exported_models"
    / "exported_best.pt"
)
EVALUATION_PHASES = {"IDLE", "PATIENT_IN_ROOM", "SURGERY_ACTIVE"}
CLASS_NAMES = ["idle", "people_in_room", "surgery_inactive", "surgery_active"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--model-id", default="dinov3_vitb16_lightly_best")
    parser.add_argument("--model-name", default="DINOv3 ViT-B/16 Lightly best")
    parser.add_argument(
        "--description",
        default=(
            "LightlyTrain DINOv3 ViT-B/16 multilabel best checkpoint evaluated "
            "on the pinned private test split."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--dataset-local-files-only", action="store_true")
    parser.add_argument("--dataset-max-workers", type=int, default=8)
    return parser.parse_args()


def existing_paths(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["path"] for row in csv.DictReader(handle)}


def load_rows(labels_path: Path, limit: int | None) -> list[dict[str, str]]:
    with labels_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if (row.get("phase") or "").strip() in EVALUATION_PHASES
            and not (row.get("error") or "").strip()
        ]
    return rows[:limit] if limit else rows


class ImageRows(Dataset):
    def __init__(self, rows: list[dict[str, str]], snapshot_path: Path) -> None:
        from torchvision import transforms

        self.rows = rows
        self.snapshot_path = snapshot_path
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = self.snapshot_path / row["path"]
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {
            "path": row["path"],
            "ground_truth": row["phase"],
            "image": tensor,
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path": [item["path"] for item in batch],
        "ground_truth": [item["ground_truth"] for item in batch],
        "image": torch.stack([item["image"] for item in batch]),
    }


def load_model(export_path: Path, device: torch.device) -> Any:
    from lightly_train._task_models.image_classification.task_model import (
        ImageClassification,
    )

    checkpoint = torch.load(export_path, map_location="cpu", weights_only=False)
    init_args = dict(checkpoint["model_init_args"])
    init_args.pop("model_name", None)
    init_args["load_weights"] = False
    model = ImageClassification(**init_args)
    incompatible = model.load_state_dict(
        normalize_train_model_state(checkpoint["train_model"])
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Could not load the LightlyTrain export cleanly. "
            f"Missing keys: {incompatible.missing_keys[:5]}; "
            f"unexpected keys: {incompatible.unexpected_keys[:5]}"
        )
    model.to(device)
    model.eval()
    return model


def normalize_train_model_state(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert exported LightningModule keys to the task model keyspace."""

    return {key.removeprefix("model."): value for key, value in state_dict.items()}


def phase_from_probs(probs: torch.Tensor) -> tuple[str, float, dict[str, float]]:
    values = {
        name: float(probs[index].item()) for index, name in enumerate(CLASS_NAMES)
    }
    phase_scores = {
        "IDLE": values["idle"],
        "PATIENT_IN_ROOM": values["people_in_room"] * values["surgery_inactive"],
        "SURGERY_ACTIVE": values["people_in_room"] * values["surgery_active"],
    }
    phase = max(phase_scores, key=phase_scores.get)
    return phase, phase_scores[phase], values


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    correct = sum(row["ground_truth"] == row["predicted"] for row in results)
    class_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "total": 0}
    )
    for row in results:
        gt = row["ground_truth"]
        pred = row["predicted"]
        class_stats[gt]["total"] += 1
        if gt == pred:
            class_stats[gt]["tp"] += 1
        else:
            class_stats[gt]["fn"] += 1
            class_stats[pred]["fp"] += 1
    per_class = {}
    for phase in sorted(EVALUATION_PHASES):
        stats = class_stats[phase]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_class[phase] = {
            "total": stats["total"],
            "correct": tp,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total if total else 0.0, 4),
        "per_class": per_class,
    }


def main() -> int:
    args = parse_args()
    output_path = args.output or OUTPUT_DIR / f"eval_results_{args.model_id}.csv"
    metrics_path = (
        args.metrics_output or OUTPUT_DIR / f"eval_metrics_{args.model_id}.json"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = get_hf_dataset_paths(
        max_workers=args.dataset_max_workers,
        local_files_only=args.dataset_local_files_only,
    )
    rows = load_rows(dataset.labels_path, args.limit)
    done = existing_paths(output_path)
    rows = [row for row in rows if row["path"] not in done]

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    torch.set_float32_matmul_precision("high")
    model = load_model(args.export, device)

    fieldnames = [
        "path",
        "ground_truth",
        "predicted",
        "confidence",
        "key_visual_cues",
        "correct",
        "prob_idle",
        "prob_people_in_room",
        "prob_surgery_inactive",
        "prob_surgery_active",
    ]
    file_exists = output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        loader = DataLoader(
            ImageRows(rows, dataset.snapshot_path),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate,
            prefetch_factor=1 if args.num_workers else None,
            persistent_workers=args.num_workers > 0,
        )
        with torch.inference_mode():
            for batch in tqdm(loader, desc="Evaluating DINOv3"):
                images = batch["image"].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    logits = model.forward_backend(images)
                probs = torch.sigmoid(logits.float()).cpu()
                for index, image_path in enumerate(batch["path"]):
                    predicted, confidence, values = phase_from_probs(probs[index])
                    ground_truth = batch["ground_truth"][index]
                    writer.writerow(
                        {
                            "path": image_path,
                            "ground_truth": ground_truth,
                            "predicted": predicted,
                            "confidence": round(confidence, 6),
                            "key_visual_cues": "|".join(
                                f"p_{name}={values[name]:.4f}" for name in CLASS_NAMES
                            ),
                            "correct": ground_truth == predicted,
                            "prob_idle": round(values["idle"], 6),
                            "prob_people_in_room": round(values["people_in_room"], 6),
                            "prob_surgery_inactive": round(
                                values["surgery_inactive"], 6
                            ),
                            "prob_surgery_active": round(values["surgery_active"], 6),
                        }
                    )
                handle.flush()

    with output_path.open(newline="", encoding="utf-8") as handle:
        results = list(csv.DictReader(handle))
    metrics = compute_metrics(results)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    register_model(
        args.model_id,
        args.model_name,
        prompt_file="finetuning/dinov3/evaluate_lightly.py",
        description=args.description,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate a Moondream Cloud fine-tune on the private test split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from operai_eye.paths import OUTPUT_DIR, PROJECT_ROOT
from operai_eye.pipeline.hf_dataset import get_hf_dataset_paths
from operai_eye.pipeline.prepare_sft_dataset import CORE_PHASES, QUESTION
from operai_eye.training.moondream.finetune_moondream import parse_phase
from operai_eye.web.eval_data import register_model

DEFAULT_MODEL = "moondream3-preview/01KV38VJ6DYECNHFCFSK2B96SN@3751"
EVALUATION_PHASES = set(CORE_PHASES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--model-id", default="moondream3_sft_10k_best")
    parser.add_argument("--model-name", default="Moondream3 SFT 10k best")
    parser.add_argument(
        "--description",
        default=(
            "Moondream Cloud fine-tune best checkpoint evaluated on the pinned "
            "private test split."
        ),
    )
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--dataset-local-files-only", action="store_true")
    parser.add_argument("--dataset-max-workers", type=int, default=8)
    return parser.parse_args()


def successful_paths(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["path"]
            for row in csv.DictReader(handle)
            if not (row.get("key_visual_cues") or "").startswith("ERROR:")
        }


def load_rows(labels_path: Path, limit: int | None) -> list[dict[str, str]]:
    with labels_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if (row.get("phase") or "").strip() in EVALUATION_PHASES
            and not (row.get("error") or "").strip()
        ]
    return rows[:limit] if limit else rows


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
    for phase in sorted(EVALUATION_PHASES | {"UNKNOWN"}):
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


def evaluate_one(
    *,
    model: Any,
    snapshot_path: Path,
    row: dict[str, str],
    max_retries: int,
) -> dict[str, Any]:
    image_path = snapshot_path / row["path"]
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            with Image.open(image_path) as image:
                result = model.query(
                    image=image.convert("RGB"),
                    question=QUESTION,
                    settings={"temperature": 0.0, "max_tokens": 16},
                )
            answer = str(result.get("answer", ""))
            predicted = parse_phase(answer)
            return {
                "path": row["path"],
                "ground_truth": row["phase"],
                "predicted": predicted,
                "confidence": 1.0 if predicted != "UNKNOWN" else 0.0,
                "key_visual_cues": answer,
                "correct": row["phase"] == predicted,
            }
        except Exception as exc:  # noqa: BLE001 - retry API/network failures
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(2**attempt, 8))
    return {
        "path": row["path"],
        "ground_truth": row["phase"],
        "predicted": "UNKNOWN",
        "confidence": 0.0,
        "key_visual_cues": f"ERROR: {last_error}",
        "correct": False,
    }


def main() -> int:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.getenv("MOONDREAM_API_KEY")
    if not api_key:
        raise RuntimeError("MOONDREAM_API_KEY is required")

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
    done = successful_paths(output_path)
    rows = [row for row in rows if row["path"] not in done]

    import moondream as md

    model = md.vl(api_key=api_key, model=args.model)
    fieldnames = [
        "path",
        "ground_truth",
        "predicted",
        "confidence",
        "key_visual_cues",
        "correct",
    ]
    file_exists = output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(
                    evaluate_one,
                    model=model,
                    snapshot_path=dataset.snapshot_path,
                    row=row,
                    max_retries=args.max_retries,
                )
                for row in rows
            ]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Evaluating Moondream"
            ):
                writer.writerow(future.result())
                handle.flush()

    with output_path.open(newline="", encoding="utf-8") as handle:
        results = list(csv.DictReader(handle))
    metrics = compute_metrics(results)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    register_model(
        args.model_id,
        args.model_name,
        prompt_file="prepare_sft_dataset.QUESTION",
        description=args.description,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

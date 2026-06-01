#!/usr/bin/env python3
"""Evaluate Moondream against labeled dataset.

Usage:
    python evaluate_moondream.py --labels output/validation_labels.csv
    python evaluate_moondream.py --labels output/validation_labels.csv --output eval_results.csv
    python evaluate_moondream.py --labels output/validation_labels.csv --limit 100
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_PROMPT_PATH = Path("prompts/or_phase.txt")
DATA_DIR = Path("data/exocentric_rgb")
OUTPUT_DIR = Path("output")
VALID_PHASES = {"IDLE", "TURNOVER", "PATIENT_IN_ROOM", "SURGERY_ACTIVE", "UNKNOWN"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Moondream on labeled OR images.")
    parser.add_argument("--labels", type=Path, required=True, help="Path to labels CSV")
    parser.add_argument("--output", type=Path, help="Output CSV path (default: output/eval_results.csv)")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH, help="Prompt file path")
    parser.add_argument("--model", default="vikhyatk/moondream2", help="Moondream model name")
    parser.add_argument("--revision", default="2025-01-09", help="Model revision")
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], help="Device to use")
    parser.add_argument("--limit", type=int, help="Limit number of images to evaluate")
    parser.add_argument("--compile", action="store_true", help="Compile model for speed")
    return parser.parse_args()


def get_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_name: str, revision: str, device: str, compile_model: bool) -> Any:
    logger.info("Loading model %s on %s...", model_name, device)
    dtype = torch.bfloat16 if device == "cuda" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
        dtype=dtype,
        device_map=device,
    )
    if compile_model:
        logger.info("Compiling model...")
        model.compile()
    return model


def load_prompt(prompt_path: Path) -> str:
    if not prompt_path.exists():
        logger.error("Prompt file not found: %s", prompt_path)
        sys.exit(1)
    return prompt_path.read_text().strip()


def load_labels(labels_path: Path) -> list[dict[str, str]]:
    if not labels_path.exists():
        logger.error("Labels file not found: %s", labels_path)
        sys.exit(1)
    rows = []
    with open(labels_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    logger.info("Loaded %d labels from %s", len(rows), labels_path)
    return rows


def parse_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"phase": "UNKNOWN", "confidence": 0.0, "key_visual_cues": ["parse_error"]}


def evaluate_image(model: Any, image_path: Path, prompt: str) -> dict[str, Any]:
    image = Image.open(image_path)
    encoded = model.encode_image(image)
    result = model.query(encoded, prompt, settings={"temperature": 0.1, "max_tokens": 500})
    answer = result.get("answer", "")
    return parse_response(answer)


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for r in results if r["ground_truth"] == r["predicted"])
    accuracy = correct / total if total > 0 else 0.0
    class_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "total": 0})
    for r in results:
        gt = r["ground_truth"]
        pred = r["predicted"]
        class_stats[gt]["total"] += 1
        if gt == pred:
            class_stats[gt]["tp"] += 1
        else:
            class_stats[gt]["fn"] += 1
            class_stats[pred]["fp"] += 1
    per_class = {}
    for phase in VALID_PHASES:
        stats = class_stats[phase]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
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
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
    }


def main() -> int:
    args = parse_args()
    device = get_device(args.device)
    prompt = load_prompt(args.prompt)
    labels = load_labels(args.labels)
    if args.limit:
        labels = labels[: args.limit]
    model = load_model(args.model, args.revision, device, args.compile)
    output_path = args.output or OUTPUT_DIR / "eval_results.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    errors = 0
    for row in tqdm(labels, desc="Evaluating"):
        image_rel_path = row["path"]
        image_path = DATA_DIR / image_rel_path
        if not image_path.exists():
            logger.warning("Image not found: %s", image_path)
            errors += 1
            continue
        ground_truth = row["phase"]
        try:
            prediction = evaluate_image(model, image_path, prompt)
        except Exception as e:
            logger.warning("Error evaluating %s: %s", image_path, e)
            prediction = {"phase": "ERROR", "confidence": 0.0, "key_visual_cues": [str(e)]}
            errors += 1
        predicted_phase = prediction.get("phase", "UNKNOWN")
        if predicted_phase not in VALID_PHASES:
            predicted_phase = "UNKNOWN"
        results.append({
            "path": image_rel_path,
            "ground_truth": ground_truth,
            "predicted": predicted_phase,
            "confidence": prediction.get("confidence", 0.0),
            "key_visual_cues": "|".join(prediction.get("key_visual_cues", [])),
            "correct": ground_truth == predicted_phase,
        })
    if results:
        fieldnames = ["path", "ground_truth", "predicted", "confidence", "key_visual_cues", "correct"]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        logger.info("Wrote %d results to %s", len(results), output_path)
    metrics = compute_metrics(results)
    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info("Total: %d | Correct: %d | Accuracy: %.2f%%", metrics["total"], metrics["correct"], metrics["accuracy"] * 100)
    logger.info("Errors: %d", errors)
    logger.info("-" * 50)
    logger.info("Per-class metrics:")
    for phase in ["IDLE", "TURNOVER", "PATIENT_IN_ROOM", "SURGERY_ACTIVE", "UNKNOWN"]:
        stats = metrics["per_class"].get(phase, {"total": 0, "correct": 0, "precision": 0, "recall": 0, "f1": 0})
        logger.info(
            "  %s: n=%d, acc=%.2f%%, P=%.2f, R=%.2f, F1=%.2f",
            phase.ljust(16),
            stats["total"],
            (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0,
            stats["precision"],
            stats["recall"],
            stats["f1"],
        )
    logger.info("=" * 50)
    metrics_path = OUTPUT_DIR / "eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Saved metrics to %s", metrics_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

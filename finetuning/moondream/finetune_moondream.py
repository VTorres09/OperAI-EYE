#!/usr/bin/env python3
"""Run resumable Moondream supervised fine-tuning for OR phase classification."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prepare_sft_dataset import (  # noqa: E402
    CORE_PHASES,
    DEFAULT_REPO_ID,
    DEFAULT_REVISION_PATH,
    QUESTION,
    VALID_PHASES,
)

DEFAULT_STATE_PATH = Path("output/moondream_sft_state.json")
DEFAULT_RESULTS_PATH = Path("output/moondream_sft_results.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision")
    parser.add_argument("--revision-file", type=Path, default=DEFAULT_REVISION_PATH)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--name")
    parser.add_argument("--rank", type=int, choices=[8, 16, 24, 32], default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-concurrency", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_phase(answer: str) -> str:
    normalized = answer.strip().upper()
    if normalized in VALID_PHASES:
        return normalized
    matches = re.findall(
        r"\b(?:IDLE|PATIENT_IN_ROOM|SURGERY_ACTIVE|UNKNOWN)\b",
        normalized,
    )
    return matches[0] if len(set(matches)) == 1 else "UNKNOWN"


def compute_metrics(
    ground_truth: Sequence[str],
    predictions: Sequence[str],
) -> dict[str, Any]:
    if len(ground_truth) != len(predictions):
        raise ValueError("Ground truth and prediction lengths differ")
    total = len(ground_truth)
    correct = sum(expected == predicted for expected, predicted in zip(ground_truth, predictions))
    per_class = {}
    for phase in sorted(VALID_PHASES):
        tp = sum(
            expected == phase and predicted == phase
            for expected, predicted in zip(ground_truth, predictions)
        )
        fp = sum(
            expected != phase and predicted == phase
            for expected, predicted in zip(ground_truth, predictions)
        )
        fn = sum(
            expected == phase and predicted != phase
            for expected, predicted in zip(ground_truth, predictions)
        )
        support = sum(expected == phase for expected in ground_truth)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[phase] = {
            "support": support,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    macro_f1 = sum(per_class[phase]["f1"] for phase in CORE_PHASES) / len(CORE_PHASES)
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total if total else 0.0, 6),
        "core_macro_f1": round(macro_f1, 6),
        "per_class": per_class,
        "prediction_counts": dict(sorted(Counter(predictions).items())),
    }


def sft_group(example: dict[str, Any]) -> dict[str, Any]:
    phase = str(example["phase"]).strip().upper()
    if phase not in VALID_PHASES:
        raise ValueError(f"Invalid phase: {phase}")
    return {
        "mode": "sft",
        "request": {
            "skill": "query",
            "image": example["image"],
            "question": example.get("question") or QUESTION,
        },
        "target": {"answer": phase},
    }


def epoch_order(length: int, epoch: int, seed: int) -> list[int]:
    indices = list(range(length))
    random.Random(seed + epoch).shuffle(indices)
    return indices


def iter_batches(values: Sequence[int], batch_size: int) -> Iterable[tuple[int, Sequence[int]]]:
    for offset in range(0, len(values), batch_size):
        yield offset // batch_size, values[offset : offset + batch_size]


def train_epoch(
    ft: Any,
    train_data: Any,
    *,
    epoch: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    start_batch: int = 0,
    on_progress: Callable[[int, dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any] | None, list[float]]:
    order = epoch_order(len(train_data), epoch, seed)
    last_step = None
    losses = []
    for batch_index, indices in iter_batches(order, batch_size):
        if batch_index < start_batch:
            continue
        groups = [sft_group(train_data[index]) for index in indices]
        step = ft.train_step(groups, lr=learning_rate)
        last_step = dict(step)
        loss = step.get("sft_loss")
        if loss is not None:
            losses.append(float(loss))
        if on_progress:
            on_progress(batch_index + 1, last_step)
    return last_step, losses


def evaluate(ft: Any, validation_data: Any, concurrency: int) -> dict[str, Any]:
    requests = (
        (
            str(example["phase"]).strip().upper(),
            {
                "skill": "query",
                "image": example["image"],
                "question": example.get("question") or QUESTION,
                "num_rollouts": 1,
                "settings": {"temperature": 0.0, "max_tokens": 16},
            },
        )
        for example in validation_data
    )
    ground_truth = []
    predictions = []
    for expected, response in ft.rollout_stream(
        requests,
        max_concurrency=concurrency,
        buffer_size=max(concurrency * 2, 1),
    ):
        answer = response["rollouts"][0]["output"]["answer"]
        ground_truth.append(expected)
        predictions.append(parse_phase(answer))
    return compute_metrics(ground_truth, predictions)


def early_stopping_update(
    best_score: float,
    score: float,
    minimum_improvement: float = 0.01,
) -> tuple[float, bool, bool]:
    improved = score >= best_score + minimum_improvement
    if improved:
        return score, True, False
    return best_score, False, True


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def resolve_revision(args: argparse.Namespace) -> str:
    if args.revision:
        return args.revision
    if not args.revision_file.exists():
        raise RuntimeError(
            f"Dataset revision file not found: {args.revision_file}; publish the dataset first"
        )
    value = json.loads(args.revision_file.read_text(encoding="utf-8"))
    return value["revision"]


def initial_state(args: argparse.Namespace, revision: str, ft: Any) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "revision": revision,
        "finetune_id": ft.finetune_id,
        "finetune_name": ft.name,
        "rank": ft.rank,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "maximum_epochs": args.epochs,
        "next_epoch": 0,
        "next_batch": 0,
        "last_step": None,
        "epoch_losses": [],
        "best_core_macro_f1": None,
        "best_model_id": None,
        "metrics": [],
        "checkpoints": [],
        "complete": False,
        "stopped_early": False,
    }


def validate_resume_state(
    state: dict[str, Any],
    args: argparse.Namespace,
    revision: str,
) -> None:
    expected = {
        "repo_id": args.repo_id,
        "revision": revision,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }
    mismatches = {
        key: (state.get(key), value)
        for key, value in expected.items()
        if state.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"State does not match requested run: {mismatches}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(".env", override=False)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN or HUGGINGFACE_HUB_TOKEN is required")
    api_key = os.getenv("MOONDREAM_API_KEY")
    if not api_key and not args.dry_run:
        raise RuntimeError("MOONDREAM_API_KEY is required")

    revision = resolve_revision(args)
    from datasets import load_dataset

    dataset = load_dataset(
        args.repo_id,
        revision=revision,
        token=token,
        data_files={
            "train": "train.zip",
            "validation": "validation.zip",
        },
    )
    train_data = dataset["train"]
    validation_data = dataset["validation"]
    if args.dry_run:
        example_group = sft_group(train_data[0])
        return {
            "dry_run": True,
            "repo_id": args.repo_id,
            "revision": revision,
            "train_count": len(train_data),
            "validation_count": len(validation_data),
            "example_request": {
                "mode": example_group["mode"],
                "skill": example_group["request"]["skill"],
                "question": example_group["request"]["question"],
                "target": example_group["target"],
                "image_size": example_group["request"]["image"].size,
            },
        }

    import moondream as md

    if args.state.exists():
        state = json.loads(args.state.read_text(encoding="utf-8"))
        validate_resume_state(state, args, revision)
        ft = md.ft(api_key=api_key, finetune_id=state["finetune_id"])
    else:
        name = args.name or f"operai-eye-or-phase-{int(time.time())}"
        ft = md.ft(api_key=api_key, name=name, rank=args.rank)
        state = initial_state(args, revision, ft)
        write_json(args.state, state)

    if state["complete"]:
        return state

    if not state["metrics"]:
        baseline = evaluate(ft, validation_data, args.eval_concurrency)
        baseline_record = {
            "epoch": -1,
            "step": 0,
            "kind": "baseline",
            **baseline,
        }
        state["metrics"].append(baseline_record)
        state["best_core_macro_f1"] = baseline["core_macro_f1"]
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(args.state, state)
        ft.log_metrics(
            0,
            {
                "eval/accuracy": baseline["accuracy"],
                "eval/core_macro_f1": baseline["core_macro_f1"],
            },
        )

    for epoch in range(state["next_epoch"], args.epochs):
        start_batch = state["next_batch"] if epoch == state["next_epoch"] else 0
        accumulated_losses = (
            list(state.get("epoch_losses", [])) if start_batch else []
        )

        def record_progress(next_batch: int, step: dict[str, Any]) -> None:
            loss = step.get("sft_loss")
            if loss is not None:
                accumulated_losses.append(float(loss))
            state["next_epoch"] = epoch
            state["next_batch"] = next_batch
            state["last_step"] = step
            state["epoch_losses"] = accumulated_losses
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_json(args.state, state)

        last_step, _ = train_epoch(
            ft,
            train_data,
            epoch=epoch,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            start_batch=start_batch,
            on_progress=record_progress,
        )
        if last_step is None:
            last_step = state.get("last_step") or {"step": 0}

        checkpoint = ft.save_checkpoint()["checkpoint"]
        model_id = ft.model(checkpoint["step"])
        metrics = evaluate(ft, validation_data, args.eval_concurrency)
        epoch_record = {
            "epoch": epoch,
            "step": last_step.get("step"),
            "kind": "epoch",
            "mean_sft_loss": mean(accumulated_losses),
            **metrics,
        }
        state["metrics"].append(epoch_record)
        state["checkpoints"].append({
            **checkpoint,
            "model_id": model_id,
            "epoch": epoch,
        })

        best_score, improved, stop = early_stopping_update(
            float(state["best_core_macro_f1"]),
            metrics["core_macro_f1"],
        )
        state["best_core_macro_f1"] = best_score
        if improved:
            state["best_model_id"] = model_id
        state["next_epoch"] = epoch + 1
        state["next_batch"] = 0
        state["epoch_losses"] = []
        state["stopped_early"] = stop
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(args.state, state)

        log_values = {
            "eval/accuracy": metrics["accuracy"],
            "eval/core_macro_f1": metrics["core_macro_f1"],
        }
        if epoch_record["mean_sft_loss"] is not None:
            log_values["train/sft_loss"] = epoch_record["mean_sft_loss"]
        for phase, phase_metrics in metrics["per_class"].items():
            for metric in ("precision", "recall", "f1"):
                log_values[f"eval/{phase.lower()}_{metric}"] = phase_metrics[metric]
        ft.log_metrics(int(last_step.get("step") or checkpoint["step"]), log_values)

        print(
            f"epoch={epoch + 1} accuracy={metrics['accuracy']:.4f} "
            f"core_macro_f1={metrics['core_macro_f1']:.4f} "
            f"improved={improved} model={model_id}",
            flush=True,
        )
        if stop:
            break

    state["complete"] = True
    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(args.state, state)
    write_json(args.results, state)
    return state


def main() -> int:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

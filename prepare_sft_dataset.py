#!/usr/bin/env python3
"""Analyze, stage, publish, and verify the OperAI-EYE SFT dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from huggingface_hub import HfApi

from label_data import DATA_DIR, OUTPUT_DIR, VALID_PHASES, deterministic_sample

DEFAULT_REPO_ID = "OperAI-Research/operai-eye-exocentric-rgb-sft"
DEFAULT_STAGE_DIR = OUTPUT_DIR / "sft_dataset"
DEFAULT_ANALYSIS_PATH = OUTPUT_DIR / "sft_analysis.json"
DEFAULT_REVISION_PATH = OUTPUT_DIR / "sft_dataset_revision.json"
DEFAULT_SEED = 42
QUESTION = (
    "Classify the operating room phase. Respond with exactly one of: "
    "IDLE, PATIENT_IN_ROOM, SURGERY_ACTIVE."
)
CORE_PHASES = ("IDLE", "PATIENT_IN_ROOM", "SURGERY_ACTIVE")
TRAINING_PHASES = set(CORE_PHASES)
SPLIT_RULES = {
    "train": {
        "initial": 10_000,
        "step": 1_000,
        "maximum": 10_000,
        "minimum_per_core_class": 200,
        "metadata_tolerance": 0.03,
    },
    "validation": {
        "initial": 2_000,
        "step": 500,
        "maximum": 3_000,
        "minimum_per_core_class": 30,
        "metadata_tolerance": 0.05,
    },
}
METADATA_FIELDS = ("procedure", "take", "camera")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze labels and recommend sample sizes")
    analyze.add_argument("--seed", type=int, default=DEFAULT_SEED)
    analyze.add_argument("--output", type=Path, default=DEFAULT_ANALYSIS_PATH)
    analyze.add_argument("--json", action="store_true")

    stage = subparsers.add_parser("stage", help="Build an ImageFolder-compatible upload directory")
    stage.add_argument("--seed", type=int, default=DEFAULT_SEED)
    stage.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS_PATH)
    stage.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    stage.add_argument("--prompt", type=Path, default=Path("prompts/or_phase.txt"))
    stage.add_argument("--model", default=os.getenv("MODEL_NAME", "kimi-k2.6"))
    stage.add_argument("--allow-unready", action="store_true")

    publish = subparsers.add_parser("publish", help="Upload and verify the staged dataset")
    publish.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    publish.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    publish.add_argument("--revision-output", type=Path, default=DEFAULT_REVISION_PATH)
    publish.add_argument("--workers", type=int, default=8)

    return parser.parse_args()


def image_metadata(path: Path) -> dict[str, str]:
    parts = path.relative_to(DATA_DIR).parts
    return {
        "procedure": f"{parts[1]}/{parts[2]}",
        "take": f"{parts[1]}/{parts[2]}/{parts[3]}",
        "camera": parts[4],
    }


def distribution(values: Iterable[str]) -> dict[str, float]:
    counts = Counter(values)
    total = sum(counts.values())
    if not total:
        return {}
    return {key: value / total for key, value in sorted(counts.items())}


def maximum_distribution_gap(
    full: dict[str, float],
    sample: dict[str, float],
) -> float:
    keys = set(full) | set(sample)
    return max((abs(full.get(key, 0.0) - sample.get(key, 0.0)) for key in keys), default=0.0)


def read_successful_labels(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    labels: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rel_path = (row.get("path") or "").strip()
            phase = (row.get("phase") or "").strip()
            error = (row.get("error") or "").strip()
            if rel_path and not error and phase in VALID_PHASES:
                labels[rel_path] = row
    return labels


def sampled_paths(split: str, seed: int) -> tuple[list[Path], list[Path]]:
    full = sorted((DATA_DIR / split).rglob("*.png"))
    return full, deterministic_sample(full, None, seed)


def completed_milestone(
    shuffled: list[Path],
    labels: dict[str, dict[str, str]],
    split: str,
) -> tuple[int, int]:
    completed_prefix = 0
    for path in shuffled:
        rel_path = str(path.relative_to(DATA_DIR))
        if rel_path not in labels:
            break
        completed_prefix += 1

    rule = SPLIT_RULES[split]
    if completed_prefix < rule["initial"]:
        return completed_prefix, rule["initial"]
    milestone = min(
        rule["maximum"],
        rule["initial"]
        + ((completed_prefix - rule["initial"]) // rule["step"]) * rule["step"],
    )
    return completed_prefix, milestone


def confidence_summary(rows: list[dict[str, str]]) -> dict[str, float | int | None]:
    values = []
    for row in rows:
        try:
            values.append(float(row.get("confidence") or 0.0))
        except ValueError:
            continue
    if not values:
        return {"count": 0, "minimum": None, "median": None, "mean": None, "maximum": None}
    return {
        "count": len(values),
        "minimum": round(min(values), 4),
        "median": round(statistics.median(values), 4),
        "mean": round(statistics.mean(values), 4),
        "maximum": round(max(values), 4),
    }


def analyze_split(
    split: str,
    full_paths: list[Path],
    shuffled_paths: list[Path],
    labels: dict[str, dict[str, str]],
) -> dict[str, Any]:
    rule = SPLIT_RULES[split]
    completed_prefix, selected_count = completed_milestone(shuffled_paths, labels, split)
    selected_paths = shuffled_paths[:selected_count]
    selected_rows = [
        labels[str(path.relative_to(DATA_DIR))]
        for path in selected_paths
        if str(path.relative_to(DATA_DIR)) in labels
    ]
    training_rows = [
        row for row in selected_rows if row.get("phase") in TRAINING_PHASES
    ]
    class_counts = Counter(row["phase"] for row in selected_rows)
    training_class_counts = Counter(row["phase"] for row in training_rows)

    metadata_gaps = {}
    for field in METADATA_FIELDS:
        full_dist = distribution(image_metadata(path)[field] for path in full_paths)
        sample_dist = distribution(image_metadata(path)[field] for path in selected_paths)
        metadata_gaps[field] = round(maximum_distribution_gap(full_dist, sample_dist), 6)

    class_minimums_met = all(
        class_counts.get(phase, 0) >= rule["minimum_per_core_class"]
        for phase in CORE_PHASES
    )
    metadata_representative = all(
        gap <= rule["metadata_tolerance"] for gap in metadata_gaps.values()
    )

    stability_gap = None
    class_distribution_stable = True
    if split == "train" and selected_count >= 1_000 and len(selected_rows) == selected_count:
        block = rule["step"]
        previous = selected_rows[selected_count - 2 * block : selected_count - block]
        latest = selected_rows[selected_count - block : selected_count]
        previous_dist = distribution(
            row["phase"] for row in previous if row["phase"] in CORE_PHASES
        )
        latest_dist = distribution(
            row["phase"] for row in latest if row["phase"] in CORE_PHASES
        )
        stability_gap = maximum_distribution_gap(previous_dist, latest_dist)
        class_distribution_stable = stability_gap <= 0.075

    sample_complete = completed_prefix >= selected_count
    criteria_met = (
        sample_complete
        and class_minimums_met
        and metadata_representative
        and class_distribution_stable
    )
    exhausted = selected_count >= rule["maximum"] and sample_complete
    if not sample_complete:
        recommended_count = selected_count
    elif criteria_met or exhausted:
        recommended_count = selected_count
    else:
        recommended_count = min(selected_count + rule["step"], rule["maximum"])

    return {
        "split": split,
        "full_image_count": len(full_paths),
        "completed_prefix": completed_prefix,
        "selected_count": selected_count,
        "recommended_count": recommended_count,
        "sample_complete": sample_complete,
        "criteria_met": criteria_met,
        "exhausted": exhausted,
        "needs_labeling": completed_prefix < recommended_count,
        "class_counts": {
            phase: class_counts.get(phase, 0) for phase in sorted(VALID_PHASES)
        },
        "training_count": len(training_rows),
        "ignored_unknown_count": class_counts.get("UNKNOWN", 0),
        "training_class_counts": {
            phase: training_class_counts.get(phase, 0) for phase in CORE_PHASES
        },
        "minimum_per_core_class": rule["minimum_per_core_class"],
        "class_minimums_met": class_minimums_met,
        "metadata_max_absolute_gaps": metadata_gaps,
        "metadata_tolerance": rule["metadata_tolerance"],
        "metadata_representative": metadata_representative,
        "class_stability_max_absolute_gap": (
            round(stability_gap, 6) if stability_gap is not None else None
        ),
        "class_stability_tolerance": 0.075 if split == "train" else None,
        "class_distribution_stable": class_distribution_stable,
        "confidence": confidence_summary(selected_rows),
    }


def analyze_dataset(seed: int) -> dict[str, Any]:
    splits = {}
    for split in SPLIT_RULES:
        full, shuffled = sampled_paths(split, seed)
        labels = read_successful_labels(OUTPUT_DIR / f"{split}_labels.csv")
        splits[split] = analyze_split(split, full, shuffled, labels)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "taxonomy": sorted(VALID_PHASES),
        "splits": splits,
        "ready_for_staging": all(
            (result["criteria_met"] or result["exhausted"])
            and result["sample_complete"]
            for result in splits.values()
        ),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_analysis(seed: int, output: Path, as_json: bool) -> int:
    analysis = analyze_dataset(seed)
    write_json(output, analysis)
    if as_json:
        print(json.dumps(analysis, indent=2, sort_keys=True))
    else:
        for split, result in analysis["splits"].items():
            print(
                f"{split}: selected={result['selected_count']} "
                f"completed={result['completed_prefix']} "
                f"recommended={result['recommended_count']} "
                f"criteria_met={result['criteria_met']}"
            )
        print(f"Analysis: {output}")
    return 0


def dataset_card(analysis: dict[str, Any], model: str, prompt_hash: str) -> str:
    train_count = analysis["splits"]["train"].get(
        "training_count", analysis["splits"]["train"]["selected_count"]
    )
    validation_count = analysis["splits"]["validation"].get(
        "training_count", analysis["splits"]["validation"]["selected_count"]
    )
    return f"""---
license: apache-2.0
task_categories:
- image-classification
tags:
- medical
- operating-room
- surgical-phase-recognition
- moondream
- sft
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: train.zip
  - split: validation
    path: validation.zip
---

# OperAI EYE Exocentric RGB SFT

Kimi-labeled exocentric operating-room frames derived from
[ardamamur/EgoExOR](https://huggingface.co/datasets/ardamamur/EgoExOR).

- Train examples: {train_count}
- Validation examples: {validation_count}
- Labeling model: `{model}`
- Sampling seed: `{analysis['seed']}`
- Prompt SHA-256: `{prompt_hash}`
- Classes: `IDLE`, `PATIENT_IN_ROOM`, `SURGERY_ACTIVE`

Images are stored as an ImageFolder dataset with split-local `metadata.csv` files.
The Kimi confidence values are retained for analysis and are not used as filters.
Rows labeled `UNKNOWN` are retained in the source label CSVs for auditing, but
are omitted from the training and validation archives.

## License and attribution

The source EgoExOR dataset is released under Apache-2.0. Cite the original
EgoExOR publication and dataset when using this derivative dataset.
"""


def stage_dataset(
    analysis_path: Path,
    stage_dir: Path,
    prompt_path: Path,
    model: str,
    seed: int,
    allow_unready: bool,
) -> dict[str, Any]:
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if analysis["seed"] != seed:
        raise ValueError(
            f"Analysis seed {analysis['seed']} does not match requested seed {seed}"
        )
    if not analysis["ready_for_staging"] and not allow_unready:
        raise RuntimeError(
            "Analysis recommends more labeling; rerun with --allow-unready only to override"
        )

    prompt = prompt_path.read_text(encoding="utf-8").strip()
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    sampled_manifest: dict[str, list[str]] = {}
    staged_manifest: dict[str, list[str]] = {}
    ignored_manifest: dict[str, list[str]] = {}
    for split, result in analysis["splits"].items():
        _, shuffled = sampled_paths(split, seed)
        selected = shuffled[: result["selected_count"]]
        labels = read_successful_labels(OUTPUT_DIR / f"{split}_labels.csv")
        split_dir = stage_dir / split
        split_dir.mkdir(parents=True)
        metadata_rows = []
        label_rows = []
        sampled_manifest[split] = []
        staged_manifest[split] = []
        ignored_manifest[split] = []

        for source in selected:
            source_rel = source.relative_to(DATA_DIR)
            source_key = str(source_rel)
            if source_key not in labels:
                raise RuntimeError(f"Missing successful label for {source_key}")
            row = labels[source_key]
            sampled_manifest[split].append(source_key)
            label_rows.append(row)
            if row["phase"] not in TRAINING_PHASES:
                ignored_manifest[split].append(source_key)
                continue
            image_rel = Path(*source_rel.parts[1:])
            destination = split_dir / image_rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            staged_manifest[split].append(source_key)
            metadata_rows.append({
                "file_name": str(image_rel),
                "label": row["phase"],
                "phase": row["phase"],
                "question": QUESTION,
                "confidence": row.get("confidence", ""),
                "key_visual_cues": row.get("key_visual_cues", ""),
                "surgery_type": row.get("surgery_type", ""),
                "procedure_id": row.get("procedure_id", ""),
                "take_id": row.get("take_id", ""),
                "camera": row.get("camera", ""),
                "frame_id": row.get("frame_id", ""),
                "source_path": source_key,
                "labeling_model": model,
                "sampling_seed": seed,
                "prompt_sha256": prompt_hash,
            })

        if not metadata_rows:
            raise RuntimeError(f"No trainable rows were staged for {split}")

        with (split_dir / "metadata.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0]))
            writer.writeheader()
            writer.writerows(metadata_rows)

        labels_dir = stage_dir / "labels"
        labels_dir.mkdir(exist_ok=True)
        with (labels_dir / f"{split}_labels.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(label_rows[0]))
            writer.writeheader()
            writer.writerows(label_rows)

        archive_path = stage_dir / f"{split}.zip"
        with zipfile.ZipFile(
            archive_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for path in sorted(split_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=str(path.relative_to(split_dir)))
        shutil.rmtree(split_dir)

    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "ardamamur/EgoExOR",
        "license": "apache-2.0",
        "labeling_model": model,
        "prompt_path": str(prompt_path),
        "prompt_sha256": prompt_hash,
        "question": QUESTION,
        "sampling_seed": seed,
        "analysis": analysis,
        "sampled_paths": sampled_manifest,
        "staged_paths": staged_manifest,
        "ignored_unknown_paths": ignored_manifest,
    }
    write_json(stage_dir / "analysis_manifest.json", provenance)
    (stage_dir / "README.md").write_text(
        dataset_card(analysis, model, prompt_hash),
        encoding="utf-8",
    )
    return provenance


def publish_dataset(
    repo_id: str,
    stage_dir: Path,
    revision_output: Path,
    workers: int,
) -> dict[str, Any]:
    load_dotenv(".env", override=False)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN or HUGGINGFACE_HUB_TOKEN is required")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=stage_dir,
        num_workers=workers,
    )
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    for obsolete_dir in SPLIT_RULES:
        if any(path.startswith(f"{obsolete_dir}/") for path in repo_files):
            api.delete_folder(
                path_in_repo=obsolete_dir,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Replace {obsolete_dir} files with ImageFolder archive",
            )
    revision = api.repo_info(repo_id=repo_id, repo_type="dataset").sha

    from datasets import load_dataset

    expected = json.loads(
        (stage_dir / "analysis_manifest.json").read_text(encoding="utf-8")
    )["analysis"]["splits"]
    verified = {}
    for split in SPLIT_RULES:
        dataset = load_dataset(
            repo_id,
            split=split,
            revision=revision,
            token=token,
            data_files={split: f"{split}.zip"},
        )
        expected_count = expected[split].get(
            "training_count",
            expected[split]["selected_count"],
        )
        if len(dataset) != expected_count:
            raise RuntimeError(
                f"{split} has {len(dataset)} rows, expected {expected_count}"
            )
        if len(dataset) and not hasattr(dataset[0]["image"], "size"):
            raise RuntimeError(f"{split} image column did not decode to a PIL image")
        if len(set(dataset["source_path"])) != len(dataset):
            raise RuntimeError(f"{split} contains duplicate source paths")
        verified[split] = len(dataset)

    result = {
        "repo_id": repo_id,
        "revision": revision,
        "verified_counts": verified,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(revision_output, result)
    return result


def main() -> int:
    load_dotenv(".env", override=False)
    args = parse_args()
    if args.command == "analyze":
        return run_analysis(args.seed, args.output, args.json)
    if args.command == "stage":
        provenance = stage_dataset(
            args.analysis,
            args.stage_dir,
            args.prompt,
            args.model,
            args.seed,
            args.allow_unready,
        )
        print(json.dumps({
            "stage_dir": str(args.stage_dir),
            "counts": {
                split: len(paths)
                for split, paths in provenance["sampled_paths"].items()
            },
        }, indent=2))
        return 0
    if args.command == "publish":
        print(json.dumps(
            publish_dataset(
                args.repo_id,
                args.stage_dir,
                args.revision_output,
                args.workers,
            ),
            indent=2,
        ))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

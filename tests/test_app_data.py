"""Tests for Explorer data loading and split filtering."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from hf_dataset import HFDatasetPaths
from hf_sft_dataset import HFSFTDatasetPaths, SFTDatasetPreparationError

from app import data


FIELDNAMES = [
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


def write_labels(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def label_row(split: str, frame_id: str, phase: str) -> dict[str, str]:
    return {
        "path": f"{split}/MISS/1/take_1/external_1/frame_{frame_id}.png",
        "split": split,
        "surgery_type": "MISS",
        "procedure_id": "1",
        "take_id": "1",
        "camera": "external_1",
        "frame_id": frame_id,
        "phase": phase,
        "confidence": "0.9",
        "key_visual_cues": "",
    }


class ExplorerDataTest(unittest.TestCase):
    def tearDown(self) -> None:
        data.reset_cache()

    def test_combines_test_and_sft_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            test_snapshot = root / "test_snapshot"
            test_labels = test_snapshot / "labels" / "test_labels.csv"
            test_image = test_snapshot / "test/MISS/1/take_1/external_1/frame_000001.png"
            test_image.parent.mkdir(parents=True)
            test_image.write_bytes(b"png")
            write_labels(test_labels, [label_row("test", "000001", "IDLE")])

            sft_snapshot = root / "sft_snapshot"
            sft_extracted = root / "sft_extracted"
            labels_paths = {}
            split_dirs = {}
            for split, phase in (
                ("train", "PATIENT_IN_ROOM"),
                ("validation", "SURGERY_ACTIVE"),
            ):
                labels_paths[split] = sft_snapshot / "labels" / f"{split}_labels.csv"
                split_dirs[split] = sft_extracted / split
                row = label_row(split, "000002", phase)
                image = split_dirs[split] / Path(*Path(row["path"]).parts[1:])
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(b"png")
                write_labels(labels_paths[split], [row])

            test_paths = HFDatasetPaths(
                repo_id="org/test",
                revision="test-sha",
                snapshot_path=test_snapshot,
                split_path=test_snapshot / "test",
                labels_path=test_labels,
                metadata_path=test_snapshot / "test" / "metadata.csv",
                image_count=1,
            )
            sft_paths = HFSFTDatasetPaths(
                repo_id="org/sft",
                revision="sft-sha",
                snapshot_path=sft_snapshot,
                work_dir=root,
                extracted_dir=sft_extracted,
                labels_paths=labels_paths,
                split_dirs=split_dirs,
                image_count=2,
            )

            with patch.object(data, "get_hf_dataset_paths", return_value=test_paths), patch.object(
                data,
                "get_hf_sft_dataset_paths",
                return_value=sft_paths,
            ):
                data.reset_cache()
                options = data.get_filter_options()
                train_images = data.get_images(split="train")
                validation_stats = data.get_stats(split="validation")
                train_path = data.get_image_path(
                    Path("train/MISS/1/take_1/external_1/frame_000002.png")
                )

            self.assertEqual(options["splits"], ["train", "validation", "test"])
            self.assertEqual(train_images["total"], 1)
            self.assertEqual(train_images["items"][0]["split"], "train")
            self.assertEqual(
                train_images["items"][0]["image_url"],
                "/images/train/MISS/1/take_1/external_1/frame_000002.png",
            )
            self.assertEqual(validation_stats["total"], 1)
            self.assertEqual(train_path, split_dirs["train"] / "MISS/1/take_1/external_1/frame_000002.png")

    def test_prediction_filter_limits_explorer_to_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            test_snapshot = root / "test_snapshot"
            test_labels = test_snapshot / "labels" / "test_labels.csv"
            rows = [
                label_row("test", "000001", "IDLE"),
                label_row("test", "000002", "SURGERY_ACTIVE"),
            ]
            for row in rows:
                image = test_snapshot / row["path"]
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(b"png")
            write_labels(test_labels, rows)

            test_paths = HFDatasetPaths(
                repo_id="org/test",
                revision="test-sha",
                snapshot_path=test_snapshot,
                split_path=test_snapshot / "test",
                labels_path=test_labels,
                metadata_path=test_snapshot / "test" / "metadata.csv",
                image_count=2,
            )
            predictions = pd.DataFrame(
                [
                    {"path": rows[0]["path"], "predicted": "IDLE"},
                    {"path": rows[1]["path"], "predicted": "PATIENT_IN_ROOM"},
                ]
            )

            with patch.object(data, "get_hf_dataset_paths", return_value=test_paths), patch.object(
                data,
                "get_hf_sft_dataset_paths",
                side_effect=SFTDatasetPreparationError("no sft cache"),
            ), patch.object(data, "get_model_predictions", return_value=predictions):
                data.reset_cache()
                incorrect = data.get_images(model_id="model", prediction="incorrect")
                correct_stats = data.get_stats(model_id="model", prediction="correct")

            self.assertEqual(incorrect["total"], 1)
            self.assertEqual(incorrect["items"][0]["frame_id"], "000002")
            self.assertFalse(incorrect["items"][0]["model_correct"])
            self.assertEqual(correct_stats["total"], 1)


if __name__ == "__main__":
    unittest.main()

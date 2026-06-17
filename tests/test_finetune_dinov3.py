"""Tests for LightlyTrain DINOv3 data preparation."""

import csv
import tempfile
import unittest
from pathlib import Path

from finetuning.dinov3 import train_lightly


class LightlyDinov3PreparationTest(unittest.TestCase):
    def test_multilabel_csv_filters_unknown_and_maps_hierarchy(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_dir = root / "data"
            work_dir = root / "lightly"
            train_labels = root / "train_labels.csv"
            validation_labels = root / "validation_labels.csv"

            rows = [
                {
                    "path": "train/MISS/1/take_1/external_1/frame_000001.png",
                    "split": "train",
                    "surgery_type": "MISS",
                    "procedure_id": "1",
                    "take_id": "1",
                    "camera": "external_1",
                    "frame_id": "000001",
                    "phase": "IDLE",
                    "confidence": "0.9",
                    "key_visual_cues": "",
                    "error": "",
                },
                {
                    "path": "train/MISS/1/take_1/external_1/frame_000002.png",
                    "split": "train",
                    "surgery_type": "MISS",
                    "procedure_id": "1",
                    "take_id": "1",
                    "camera": "external_1",
                    "frame_id": "000002",
                    "phase": "PATIENT_IN_ROOM",
                    "confidence": "0.9",
                    "key_visual_cues": "",
                    "error": "",
                },
                {
                    "path": "train/MISS/1/take_1/external_1/frame_000003.png",
                    "split": "train",
                    "surgery_type": "MISS",
                    "procedure_id": "1",
                    "take_id": "1",
                    "camera": "external_1",
                    "frame_id": "000003",
                    "phase": "SURGERY_ACTIVE",
                    "confidence": "0.9",
                    "key_visual_cues": "",
                    "error": "",
                },
                {
                    "path": "train/MISS/1/take_1/external_1/frame_000004.png",
                    "split": "train",
                    "surgery_type": "MISS",
                    "procedure_id": "1",
                    "take_id": "1",
                    "camera": "external_1",
                    "frame_id": "000004",
                    "phase": "UNKNOWN",
                    "confidence": "0.9",
                    "key_visual_cues": "",
                    "error": "",
                },
            ]

            for row in rows:
                image = data_dir / row["path"]
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(b"png")

            for labels_path in (train_labels, validation_labels):
                with labels_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)

            manifest = train_lightly.prepare_lightly_data(
                train_labels=train_labels,
                validation_labels=validation_labels,
                data_dir=data_dir,
                work_dir=work_dir,
            )

            self.assertEqual(manifest["splits"]["train"]["rows"], 3)
            self.assertEqual(manifest["splits"]["train"]["skipped"], {"UNKNOWN": 1})

            with (work_dir / "train.csv").open(newline="", encoding="utf-8") as handle:
                prepared = list(csv.DictReader(handle))

            self.assertEqual(
                [row["label"] for row in prepared],
                [
                    "idle",
                    "people_in_room,surgery_inactive",
                    "people_in_room,surgery_active",
                ],
            )

            config = train_lightly.data_config(manifest)
            self.assertEqual(config["classes"], train_lightly.CLASSES)
            self.assertEqual(config["csv_label_type"], "name")


if __name__ == "__main__":
    unittest.main()

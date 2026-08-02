"""Tests for Hugging Face SFT dataset correction staging."""

from __future__ import annotations

import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

from operai_eye.pipeline.hf_sft_dataset import (
    HFSFTDatasetPaths,
    build_corrected_stage,
    write_corrections,
)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class HFSFTDatasetTest(unittest.TestCase):
    def test_corrected_stage_keeps_labels_and_filters_unknown_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / "snapshot"
            extracted = root / "extracted"
            stage = root / "stage"
            corrections_path = root / "corrections.csv"
            labels_dir = snapshot / "labels"
            labels_paths: dict[str, Path] = {}
            split_dirs: dict[str, Path] = {}

            for split in ("train", "validation"):
                split_dir = extracted / split
                split_dirs[split] = split_dir
                labels_paths[split] = labels_dir / f"{split}_labels.csv"
                rows = []
                metadata_rows = []
                for frame_id, phase in (
                    ("000001", "IDLE"),
                    ("000002", "PATIENT_IN_ROOM"),
                    ("000003", "UNKNOWN"),
                ):
                    file_name = f"MISS/1/take_1/external_1/frame_{frame_id}.png"
                    source_path = f"{split}/{file_name}"
                    image_path = split_dir / file_name
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image_path.write_bytes(b"png")
                    row = {
                        "path": source_path,
                        "split": split,
                        "surgery_type": "MISS",
                        "procedure_id": "1",
                        "take_id": "1",
                        "camera": "external_1",
                        "frame_id": frame_id,
                        "phase": phase,
                        "confidence": "0.9",
                        "key_visual_cues": "",
                        "error": "",
                    }
                    rows.append(row)
                    metadata_rows.append(
                        {
                            "file_name": file_name,
                            "phase": phase,
                            "label": phase,
                            "source_path": source_path,
                        }
                    )
                write_csv(labels_paths[split], rows)
                write_csv(split_dir / "metadata.csv", metadata_rows)

            write_corrections(
                {
                    "train/MISS/1/take_1/external_1/frame_000001.png": {
                        "path": "train/MISS/1/take_1/external_1/frame_000001.png",
                        "split": "train",
                        "original_phase": "IDLE",
                        "corrected_phase": "UNKNOWN",
                        "note": "hide from training",
                        "reviewed": "true",
                        "updated_at": "2026-06-16T00:00:00+00:00",
                    },
                    "train/MISS/1/take_1/external_1/frame_000002.png": {
                        "path": "train/MISS/1/take_1/external_1/frame_000002.png",
                        "split": "train",
                        "original_phase": "PATIENT_IN_ROOM",
                        "corrected_phase": "SURGERY_ACTIVE",
                        "note": "active",
                        "reviewed": "true",
                        "updated_at": "2026-06-16T00:00:00+00:00",
                    },
                },
                corrections_path,
            )
            dataset = HFSFTDatasetPaths(
                repo_id="org/dataset",
                revision="abc",
                snapshot_path=snapshot,
                work_dir=root,
                extracted_dir=extracted,
                labels_paths=labels_paths,
                split_dirs=split_dirs,
                image_count=6,
            )

            result = build_corrected_stage(
                dataset=dataset,
                corrections_path=corrections_path,
                stage_dir=stage,
            )

            self.assertEqual(result["splits"]["train"]["labels"], 3)
            self.assertEqual(result["splits"]["train"]["staged"], 1)
            self.assertEqual(result["splits"]["train"]["ignored_unknown"], 2)

            with (stage / "labels" / "train_labels.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                staged_labels = list(csv.DictReader(handle))
            self.assertEqual(
                [row["phase"] for row in staged_labels],
                ["UNKNOWN", "SURGERY_ACTIVE", "UNKNOWN"],
            )

            with zipfile.ZipFile(stage / "train.zip") as archive:
                metadata = archive.read("metadata.csv").decode("utf-8")
                names = archive.namelist()

            self.assertIn("SURGERY_ACTIVE", metadata)
            self.assertNotIn("frame_000001.png", metadata)
            self.assertNotIn("frame_000003.png", metadata)
            self.assertEqual(
                [name for name in names if name.endswith(".png")],
                ["MISS/1/take_1/external_1/frame_000002.png"],
            )


if __name__ == "__main__":
    unittest.main()

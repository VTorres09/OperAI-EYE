"""Tests for SFT dataset analysis and staging."""

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import prepare_sft_dataset as prepare


def make_path(split: str, index: int, camera: str = "external_1") -> Path:
    return (
        prepare.DATA_DIR
        / split
        / "MISS"
        / str(index % 4)
        / f"take_{index % 8}"
        / camera
        / f"frame_{index:06d}.png"
    )


def make_labels(
    paths: list[Path],
    phases: list[str],
) -> dict[str, dict[str, str]]:
    return {
        str(path.relative_to(prepare.DATA_DIR)): {
            "path": str(path.relative_to(prepare.DATA_DIR)),
            "phase": phases[index % len(phases)],
            "confidence": "0.9",
            "error": "",
        }
        for index, path in enumerate(paths)
    }


class AdaptiveAnalysisTest(unittest.TestCase):
    def test_balanced_representative_train_sample_is_ready(self) -> None:
        paths = [
            make_path("train", index, f"external_{index % 5}")
            for index in range(10_000)
        ]
        labels = make_labels(paths, list(prepare.CORE_PHASES))

        result = prepare.analyze_split("train", paths, paths, labels)

        self.assertTrue(result["criteria_met"])
        self.assertEqual(result["recommended_count"], 10_000)

    def test_missing_core_class_recommends_expansion(self) -> None:
        paths = [make_path("train", index) for index in range(10_000)]
        labels = make_labels(paths, ["IDLE", "PATIENT_IN_ROOM"])

        result = prepare.analyze_split("train", paths, paths, labels)

        self.assertFalse(result["class_minimums_met"])
        self.assertEqual(result["recommended_count"], 10_000)

    def test_metadata_skew_recommends_expansion(self) -> None:
        full = [
            make_path("validation", index, f"external_{index % 2}")
            for index in range(3_000)
        ]
        sample = [
            make_path("validation", index, "external_1")
            for index in range(2_000)
        ] + full[2_000:]
        labels = make_labels(sample[:2_000], list(prepare.CORE_PHASES))

        result = prepare.analyze_split("validation", full, sample, labels)

        self.assertFalse(result["metadata_representative"])
        self.assertEqual(result["recommended_count"], 2_500)


class StageTest(unittest.TestCase):
    def test_stage_writes_imagefolder_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_dir = root / "data"
            output_dir = root / "output"
            stage_dir = root / "stage"
            prompt = root / "prompt.txt"
            prompt.write_text("classify", encoding="utf-8")
            analysis_path = root / "analysis.json"
            analysis = {
                "seed": 42,
                "ready_for_staging": True,
                "splits": {
                    "train": {"selected_count": 1},
                    "validation": {"selected_count": 1},
                },
            }
            prepare.write_json(analysis_path, analysis)

            labels_by_split = {}
            shuffled_by_split = {}
            for split in ("train", "validation"):
                image = (
                    data_dir
                    / split
                    / "MISS"
                    / "1"
                    / "take_1"
                    / "external_1"
                    / "frame_000001.png"
                )
                image.parent.mkdir(parents=True)
                image.write_bytes(b"png")
                shuffled_by_split[split] = [image]
                labels_by_split[split] = {
                    str(image.relative_to(data_dir)): {
                        "path": str(image.relative_to(data_dir)),
                        "split": split,
                        "surgery_type": "MISS",
                        "procedure_id": "1",
                        "take_id": "1",
                        "camera": "external_1",
                        "frame_id": "000001",
                        "phase": "IDLE",
                        "confidence": "0.9",
                        "key_visual_cues": "empty, quiet",
                        "error": "",
                    }
                }

            def fake_sampled(split: str, seed: int):
                paths = shuffled_by_split[split]
                return paths, paths

            def fake_labels(path: Path):
                return labels_by_split[path.stem.replace("_labels", "")]

            with patch.object(prepare, "DATA_DIR", data_dir), patch.object(
                prepare, "OUTPUT_DIR", output_dir
            ), patch.object(
                prepare, "sampled_paths", side_effect=fake_sampled
            ), patch.object(
                prepare, "read_successful_labels", side_effect=fake_labels
            ):
                result = prepare.stage_dataset(
                    analysis_path,
                    stage_dir,
                    prompt,
                    "kimi-k2.6",
                    42,
                    False,
                )

            self.assertEqual(len(result["sampled_paths"]["train"]), 1)
            with zipfile.ZipFile(stage_dir / "train.zip") as archive:
                metadata = archive.read("metadata.csv").decode("utf-8")
            self.assertIn("question", metadata)
            self.assertIn('"empty, quiet"', metadata)

    def test_stage_omits_unknown_rows_from_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_dir = root / "data"
            output_dir = root / "output"
            stage_dir = root / "stage"
            prompt = root / "prompt.txt"
            prompt.write_text("classify", encoding="utf-8")
            analysis_path = root / "analysis.json"
            analysis = {
                "seed": 42,
                "ready_for_staging": True,
                "splits": {
                    "train": {"selected_count": 2, "training_count": 1},
                    "validation": {"selected_count": 2, "training_count": 1},
                },
            }
            prepare.write_json(analysis_path, analysis)

            labels_by_split = {}
            shuffled_by_split = {}
            for split in ("train", "validation"):
                paths = []
                labels_by_split[split] = {}
                for index, phase in enumerate(("IDLE", "UNKNOWN"), start=1):
                    image = (
                        data_dir
                        / split
                        / "MISS"
                        / "1"
                        / "take_1"
                        / "external_1"
                        / f"frame_{index:06d}.png"
                    )
                    image.parent.mkdir(parents=True, exist_ok=True)
                    image.write_bytes(b"png")
                    paths.append(image)
                    labels_by_split[split][str(image.relative_to(data_dir))] = {
                        "path": str(image.relative_to(data_dir)),
                        "split": split,
                        "surgery_type": "MISS",
                        "procedure_id": "1",
                        "take_id": "1",
                        "camera": "external_1",
                        "frame_id": f"{index:06d}",
                        "phase": phase,
                        "confidence": "0.9",
                        "key_visual_cues": "",
                        "error": "",
                    }
                shuffled_by_split[split] = paths

            def fake_sampled(split: str, seed: int):
                paths = shuffled_by_split[split]
                return paths, paths

            def fake_labels(path: Path):
                return labels_by_split[path.stem.replace("_labels", "")]

            with patch.object(prepare, "DATA_DIR", data_dir), patch.object(
                prepare, "OUTPUT_DIR", output_dir
            ), patch.object(
                prepare, "sampled_paths", side_effect=fake_sampled
            ), patch.object(
                prepare, "read_successful_labels", side_effect=fake_labels
            ):
                result = prepare.stage_dataset(
                    analysis_path,
                    stage_dir,
                    prompt,
                    "kimi-k2.6",
                    42,
                    False,
                )

            self.assertEqual(len(result["sampled_paths"]["train"]), 2)
            self.assertEqual(len(result["staged_paths"]["train"]), 1)
            self.assertEqual(len(result["ignored_unknown_paths"]["train"]), 1)
            with zipfile.ZipFile(stage_dir / "train.zip") as archive:
                self.assertIn("metadata.csv", archive.namelist())
                self.assertEqual(
                    [
                        name
                        for name in archive.namelist()
                        if name.endswith(".png")
                    ],
                    ["MISS/1/take_1/external_1/frame_000001.png"],
                )


if __name__ == "__main__":
    unittest.main()

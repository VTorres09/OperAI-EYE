"""Tests for deterministic labeling, CSV resume, and retry behavior."""

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from operai_eye.pipeline import label_data


class SamplingTest(unittest.TestCase):
    def test_larger_sample_contains_smaller_sample_in_order(self) -> None:
        paths = [Path(f"frame_{index:03d}.png") for index in range(100)]

        small = label_data.deterministic_sample(paths, 10, seed=42)
        large = label_data.deterministic_sample(paths, 25, seed=42)

        self.assertEqual(small, large[:10])
        self.assertNotEqual(small, sorted(paths)[:10])


class LabelCsvTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "labels.csv"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_rows(self, rows: list[dict[str, str]]) -> None:
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=label_data.FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

    def test_existing_labels_ignore_errors_and_parse_quoted_fields(self) -> None:
        self.write_rows(
            [
                {
                    "path": "train/a.png",
                    "phase": "IDLE",
                    "key_visual_cues": "staff, equipment",
                    "error": "",
                },
                {
                    "path": "train/b.png",
                    "phase": "ERROR",
                    "key_visual_cues": "",
                    "error": "timeout, retry later",
                },
            ]
        )

        self.assertEqual(
            label_data.load_existing_labels(self.csv_path),
            {"train/a.png"},
        )

    def test_upsert_replaces_failed_row(self) -> None:
        self.write_rows(
            [
                {
                    "path": "train/a.png",
                    "phase": "ERROR",
                    "error": "timeout, retry later",
                }
            ]
        )

        label_data.update_csv_rows(
            [
                {
                    "path": "train/a.png",
                    "split": "train",
                    "surgery_type": "MISS",
                    "procedure_id": "1",
                    "take_id": "1",
                    "camera": "external_1",
                    "frame_id": "000001",
                    "phase": "IDLE",
                    "confidence": 0.9,
                    "key_visual_cues": "empty, quiet",
                    "error": "",
                }
            ],
            self.csv_path,
        )

        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["phase"], "IDLE")
        self.assertEqual(rows[0]["key_visual_cues"], "empty, quiet")


class RetryTest(unittest.IsolatedAsyncioTestCase):
    async def test_transient_error_is_retried(self) -> None:
        image = Path(
            "data/exocentric_rgb/train/MISS/1/take_1/external_1/frame_000001.png"
        )
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"phase":"IDLE","confidence":0.9,"key_visual_cues":[]}'
                    )
                )
            ]
        )
        create = AsyncMock(side_effect=[RuntimeError("503 unavailable"), completion])
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        with (
            patch.object(label_data, "image_to_base64", return_value="encoded"),
            patch("operai_eye.pipeline.label_data.asyncio.sleep", new=AsyncMock()),
        ):
            result = await label_data.label_image(
                client,
                image,
                "prompt",
                "kimi-k2.6",
                label_data.asyncio.Semaphore(1),
            )

        self.assertEqual(result["phase"], "IDLE")
        self.assertEqual(create.await_count, 2)


if __name__ == "__main__":
    unittest.main()

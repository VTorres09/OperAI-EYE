import csv
import tempfile
import unittest
from pathlib import Path

from hf_dataset import DatasetPreparationError, validate_labeled_images


class ValidateLabeledImagesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.snapshot_path = Path(self.temp_dir.name)
        self.labels_path = self.snapshot_path / "labels.csv"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_labels(self, paths: list[str]) -> None:
        with self.labels_path.open("w", newline="", encoding="utf-8") as labels_file:
            writer = csv.DictWriter(labels_file, fieldnames=["path", "phase"])
            writer.writeheader()
            writer.writerows({"path": path, "phase": "IDLE"} for path in paths)

    def create_image(self, relative_path: str) -> None:
        image_path = self.snapshot_path / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.touch()

    def test_counts_only_images_referenced_by_labels(self) -> None:
        labeled_paths = ["test/camera/frame_000.png", "test/camera/frame_001.png"]
        for path in labeled_paths:
            self.create_image(path)
        self.create_image("test/camera/unlabeled_extra.png")
        self.write_labels(labeled_paths)

        self.assertEqual(
            validate_labeled_images(self.snapshot_path, self.labels_path, 2),
            2,
        )

    def test_rejects_missing_labeled_image(self) -> None:
        self.write_labels(["test/camera/missing.png"])

        with self.assertRaisesRegex(DatasetPreparationError, "missing 1 labeled PNG"):
            validate_labeled_images(self.snapshot_path, self.labels_path, 1)

    def test_rejects_duplicate_labeled_path(self) -> None:
        path = "test/camera/frame_000.png"
        self.create_image(path)
        self.write_labels([path, path])

        with self.assertRaisesRegex(DatasetPreparationError, "1 duplicate PNG"):
            validate_labeled_images(self.snapshot_path, self.labels_path, 2)


if __name__ == "__main__":
    unittest.main()

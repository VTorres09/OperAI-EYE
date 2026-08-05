"""Tests for the Raspberry Pi edge capture and inference pipeline."""

from __future__ import annotations

import base64
import tempfile
import threading
import unittest
from contextlib import ExitStack
from dataclasses import replace
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from operai_eye.edge.config import (
    CameraConfig,
    DecisionConfig,
    EdgeConfig,
    ServiceConfig,
    StorageConfig,
    load_config,
)
from operai_eye.edge.dashboard import (
    DashboardRuntime,
    _decode_browser_image,
    create_dashboard_app,
)
from operai_eye.edge.decision import (
    FramePrediction,
    majority_vote,
    prediction_from_probabilities,
)
from operai_eye.edge.inference import (
    preprocess_image,
    select_execution_providers,
    sigmoid,
)
from operai_eye.edge.service import EdgeService
from operai_eye.edge.sources import DirectorySource
from operai_eye.edge.storage import PredictionStore


def make_prediction(phase: str, confidence: float = 0.8) -> FramePrediction:
    scores = {
        "IDLE": 0.1,
        "PATIENT_IN_ROOM": 0.1,
        "SURGERY_ACTIVE": 0.1,
    }
    scores[phase] = confidence
    return FramePrediction(
        phase=phase,
        confidence=confidence,
        class_probabilities={
            "idle": 0.1,
            "people_in_room": 0.9,
            "surgery_inactive": 0.1,
            "surgery_active": 0.9,
        },
        phase_scores=scores,
    )


class FakeSource:
    def __init__(self, count: int = 5) -> None:
        self.count = count
        self.started = False
        self.captures = 0

    def start(self) -> None:
        self.started = True

    def capture(self) -> Image.Image:
        if self.captures >= self.count:
            raise RuntimeError("camera failure")
        self.captures += 1
        return Image.new("RGB", (32, 24), (self.captures, 0, 0))

    def close(self) -> None:
        self.started = False


class FakeClassifier:
    def __init__(self, predictions: list[FramePrediction]) -> None:
        self.predictions = predictions
        self.batch_sizes: list[int] = []

    def classify(self, images: list[Image.Image]) -> list[FramePrediction]:
        self.batch_sizes.append(len(images))
        return self.predictions[: len(images)]


def image_data_url(color: tuple[int, int, int] = (12, 34, 56)) -> str:
    buffer = BytesIO()
    Image.new("RGB", (32, 24), color).save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class DecisionTest(unittest.TestCase):
    def test_phase_mapping_matches_dinov3_evaluation(self) -> None:
        prediction = prediction_from_probabilities(
            {
                "idle": 0.1,
                "people_in_room": 0.9,
                "surgery_inactive": 0.2,
                "surgery_active": 0.8,
            }
        )
        self.assertEqual(prediction.phase, "SURGERY_ACTIVE")
        self.assertAlmostEqual(prediction.confidence, 0.72)

    def test_majority_vote(self) -> None:
        result = majority_vote(
            [
                make_prediction("IDLE", 0.7),
                make_prediction("SURGERY_ACTIVE", 0.8),
                make_prediction("SURGERY_ACTIVE", 0.9),
                make_prediction("PATIENT_IN_ROOM", 0.6),
                make_prediction("SURGERY_ACTIVE", 1.0),
            ]
        )
        self.assertEqual(result.phase, "SURGERY_ACTIVE")
        self.assertEqual(result.votes["SURGERY_ACTIVE"], 3)
        self.assertAlmostEqual(result.vote_fraction, 0.6)
        self.assertAlmostEqual(result.confidence, 0.9)

    def test_two_two_one_split_is_unknown(self) -> None:
        result = majority_vote(
            [
                make_prediction("IDLE", 0.7),
                make_prediction("IDLE", 0.7),
                make_prediction("SURGERY_ACTIVE", 0.9),
                make_prediction("SURGERY_ACTIVE", 0.9),
                make_prediction("PATIENT_IN_ROOM", 0.5),
            ],
            minimum_vote_fraction=0.6,
        )
        self.assertEqual(result.phase, "UNKNOWN")
        self.assertEqual(result.uncertain_reason, "vote_fraction")


class PreprocessingTest(unittest.TestCase):
    def test_preprocess_image_has_expected_shape_and_dtype(self) -> None:
        image = Image.new("RGB", (640, 360), (255, 0, 0))
        tensor = preprocess_image(image)
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertAlmostEqual(float(tensor[0, 0, 0]), (1.0 - 0.485) / 0.229)
        self.assertAlmostEqual(float(tensor[1, 0, 0]), -0.456 / 0.224, places=6)

    def test_provider_selection_prefers_coreml_only_when_available(self) -> None:
        self.assertEqual(
            select_execution_providers(
                "auto", ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            ),
            ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        self.assertEqual(
            select_execution_providers("auto", ["CPUExecutionProvider"]),
            ["CPUExecutionProvider"],
        )
        self.assertEqual(
            select_execution_providers(
                "cpu", ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            ),
            ["CPUExecutionProvider"],
        )

    def test_explicit_coreml_requires_provider(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "CoreMLExecutionProvider"):
            select_execution_providers("coreml", ["CPUExecutionProvider"])

    def test_wide_image_crop_matches_torchvision_half_pixel_rounding(self) -> None:
        columns = np.arange(640, dtype=np.uint16) % 256
        array = np.repeat(columns[np.newaxis, :, np.newaxis], 360, axis=0)
        array = np.repeat(array, 3, axis=2).astype(np.uint8)
        image = Image.fromarray(array, mode="RGB")
        resized = image.resize((455, 256), resample=Image.Resampling.BICUBIC)
        expected_crop = resized.crop((116, 16, 340, 240))
        expected = np.asarray(expected_crop, dtype=np.float32) / 255.0
        expected = (
            expected - np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
        ) / np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
        expected = np.transpose(expected, (2, 0, 1))

        np.testing.assert_array_equal(preprocess_image(image), expected)

    def test_sigmoid_is_stable_for_large_values(self) -> None:
        values = sigmoid(np.asarray([-1000.0, 0.0, 1000.0], dtype=np.float32))
        np.testing.assert_allclose(values, [0.0, 0.5, 1.0], atol=1e-7)


class ConfigurationTest(unittest.TestCase):
    def test_loads_toml_and_environment_independent_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "edge.toml"
            path.write_text(
                """
[service]
interval_seconds = 30
burst_size = 5
capture_spacing_seconds = 1
minimum_captures = 3

[model]
path = "./model.onnx"

[storage]
database_path = "./state/results.sqlite3"
image_directory = "./state/images"
""",
                encoding="utf-8",
            )
            config = load_config(path)
        self.assertEqual(config.service.interval_seconds, 30)
        self.assertEqual(config.model.path, Path("model.onnx"))
        self.assertEqual(config.storage.retain_images, "none")

    def test_rejects_burst_that_does_not_fit_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "edge.toml"
            path.write_text(
                """
[service]
interval_seconds = 4
burst_size = 5
capture_spacing_seconds = 1
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "fit inside one interval"):
                load_config(path)


class StorageAndServiceTest(unittest.TestCase):
    def test_service_batches_five_images_and_persists_vote(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = PredictionStore(
                root / "predictions.sqlite3",
                image_directory=root / "images",
                retain_images="none",
            )
            source = FakeSource()
            classifier = FakeClassifier(
                [
                    make_prediction("PATIENT_IN_ROOM"),
                    make_prediction("PATIENT_IN_ROOM"),
                    make_prediction("SURGERY_ACTIVE"),
                    make_prediction("PATIENT_IN_ROOM"),
                    make_prediction("IDLE"),
                ]
            )
            service = EdgeService(
                source=source,
                classifier=classifier,
                store=store,
                service_config=replace(ServiceConfig(), capture_spacing_seconds=0.0),
                decision_config=DecisionConfig(),
                monotonic=lambda: 0.0,
                utcnow=lambda: datetime(2026, 7, 25, tzinfo=UTC),
            )

            result = service.run_once()
            latest = store.latest()
            frame_count = store.connection.execute(
                "SELECT COUNT(*) FROM frames"
            ).fetchone()[0]
            store.close()

            self.assertEqual(classifier.batch_sizes, [5])
            self.assertEqual(result.vote.phase, "PATIENT_IN_ROOM")
            self.assertEqual(frame_count, 5)
            self.assertEqual(latest["phase"], "PATIENT_IN_ROOM")
            self.assertFalse((root / "images").exists())

    def test_failed_capture_burst_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = PredictionStore(
                root / "predictions.sqlite3",
                image_directory=root / "images",
            )
            service = EdgeService(
                source=FakeSource(count=2),
                classifier=FakeClassifier([]),
                store=store,
                service_config=replace(ServiceConfig(), capture_spacing_seconds=0.0),
                decision_config=DecisionConfig(),
            )
            with self.assertRaisesRegex(RuntimeError, "captured 2/5"):
                service.run_once()
            latest = store.latest()
            store.close()
            self.assertEqual(latest["phase"], "ERROR")
            self.assertEqual(latest["captures_succeeded"], 2)

    def test_offline_loop_stops_at_maximum_cycles(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = PredictionStore(
                root / "predictions.sqlite3",
                image_directory=root / "images",
            )
            source = FakeSource(count=10)
            classifier = FakeClassifier([make_prediction("IDLE") for _ in range(5)])
            service = EdgeService(
                source=source,
                classifier=classifier,
                store=store,
                service_config=replace(ServiceConfig(), capture_spacing_seconds=0.0),
                decision_config=DecisionConfig(),
            )
            cycles = service.run_forever(
                threading.Event(), maximum_cycles=2, offline=True
            )
            store.close()
            self.assertEqual(cycles, 2)
            self.assertEqual(classifier.batch_sizes, [5, 5])


class DashboardTest(unittest.TestCase):
    def test_decodes_browser_image_data_url(self) -> None:
        image = _decode_browser_image(image_data_url())
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (32, 24))

    def test_rejects_non_image_data_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "base64 image data URL"):
            _decode_browser_image("hello")

    def test_classifies_one_browser_burst_and_persists_vote(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = EdgeConfig(
                service=replace(ServiceConfig(), capture_spacing_seconds=0.0),
                storage=StorageConfig(
                    database_path=root / "predictions.sqlite3",
                    image_directory=root / "images",
                ),
            )
            store = PredictionStore(
                config.storage.database_path,
                image_directory=config.storage.image_directory,
            )
            classifier = FakeClassifier(
                [
                    make_prediction("PATIENT_IN_ROOM", 0.8),
                    make_prediction("PATIENT_IN_ROOM", 0.9),
                    make_prediction("SURGERY_ACTIVE", 0.6),
                    make_prediction("PATIENT_IN_ROOM", 0.7),
                    make_prediction("SURGERY_ACTIVE", 0.6),
                ]
            )
            runtime = DashboardRuntime(config, classifier=classifier, store=store)
            result = runtime.classify_encoded_images([image_data_url()] * 5)
            latest = store.latest()
            store.close()

            self.assertEqual(classifier.batch_sizes, [5])
            self.assertEqual(result["vote"]["phase"], "PATIENT_IN_ROOM")
            self.assertEqual(result["vote"]["votes"]["PATIENT_IN_ROOM"], 3)
            self.assertEqual(len(result["predictions"]), 5)
            self.assertEqual(latest["phase"], "PATIENT_IN_ROOM")

    def test_requires_the_configured_number_of_browser_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = EdgeConfig(
                storage=StorageConfig(
                    database_path=root / "predictions.sqlite3",
                    image_directory=root / "images",
                )
            )
            store = PredictionStore(
                config.storage.database_path,
                image_directory=config.storage.image_directory,
            )
            runtime = DashboardRuntime(
                config,
                classifier=FakeClassifier([]),
                store=store,
            )
            with self.assertRaisesRegex(ValueError, "exactly 5 images"):
                runtime.classify_encoded_images([image_data_url()] * 4)
            store.close()

    def test_classify_endpoint_declares_a_json_request_body(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = EdgeConfig(
                storage=StorageConfig(
                    database_path=root / "predictions.sqlite3",
                    image_directory=root / "images",
                )
            )
            store = PredictionStore(
                config.storage.database_path,
                image_directory=config.storage.image_directory,
            )
            app = create_dashboard_app(
                config,
                classifier=FakeClassifier([]),
                store=store,
            )
            operation = app.openapi()["paths"]["/api/classify"]["post"]
            store.close()

            self.assertIn("requestBody", operation)
            self.assertNotIn("parameters", operation)

    def test_server_camera_lifecycle_returns_a_browser_ready_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = EdgeConfig(
                storage=StorageConfig(
                    database_path=root / "predictions.sqlite3",
                    image_directory=root / "images",
                )
            )
            store = PredictionStore(
                config.storage.database_path,
                image_directory=config.storage.image_directory,
            )
            source = FakeSource(count=1)
            runtime = DashboardRuntime(
                config,
                classifier=FakeClassifier([]),
                store=store,
                source=source,
            )

            self.assertEqual(runtime.public_config()["capture_mode"], "server")
            runtime.start_camera()
            frame = runtime.capture_camera_frame()
            decoded = _decode_browser_image(frame["image"])
            self.assertEqual(decoded.size, (32, 24))
            self.assertEqual((frame["width"], frame["height"]), (32, 24))
            runtime.stop_camera()

            self.assertFalse(source.started)
            store.close()

    def test_opencv_dashboard_keeps_browser_camera_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = EdgeConfig(
                camera=CameraConfig(backend="opencv"),
                storage=StorageConfig(
                    database_path=root / "predictions.sqlite3",
                    image_directory=root / "images",
                ),
            )
            store = PredictionStore(
                config.storage.database_path,
                image_directory=config.storage.image_directory,
            )
            runtime = DashboardRuntime(
                config,
                classifier=FakeClassifier([]),
                store=store,
            )

            self.assertEqual(runtime.public_config()["capture_mode"], "browser")
            store.close()

    def test_camera_endpoints_are_exposed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = EdgeConfig(
                storage=StorageConfig(
                    database_path=root / "predictions.sqlite3",
                    image_directory=root / "images",
                )
            )
            store = PredictionStore(
                config.storage.database_path,
                image_directory=config.storage.image_directory,
            )
            app = create_dashboard_app(
                config,
                classifier=FakeClassifier([]),
                store=store,
                source=FakeSource(),
            )
            paths = app.openapi()["paths"]
            store.close()

            self.assertIn("/api/camera/start", paths)
            self.assertIn("/api/camera/frame", paths)
            self.assertIn("/api/camera/stop", paths)


class DirectorySourceTest(unittest.TestCase):
    def test_natural_order_and_frame_step(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, red in [
                ("frame_10.png", 10),
                ("frame_2.png", 2),
                ("frame_1.png", 1),
            ]:
                Image.new("RGB", (2, 2), (red, 0, 0)).save(root / name)
            source = DirectorySource(root, frame_step=2)
            source.start()
            first = source.capture()
            second = source.capture()
            source.close()
            self.assertEqual(first.getpixel((0, 0))[0], 1)
            self.assertEqual(second.getpixel((0, 0))[0], 10)


class OpenCvSourceTest(unittest.TestCase):
    def test_macos_camera_uses_avfoundation_and_reports_permission_help(self) -> None:
        from unittest.mock import MagicMock, patch

        from operai_eye.edge.sources import OpenCvSource

        capture = MagicMock()
        capture.isOpened.return_value = False
        fake_cv2 = MagicMock(CAP_AVFOUNDATION=1200)
        fake_cv2.VideoCapture.return_value = capture
        source = OpenCvSource("0")

        with ExitStack() as stack:
            stack.enter_context(patch.dict("sys.modules", {"cv2": fake_cv2}))
            stack.enter_context(patch("operai_eye.edge.sources.sys.platform", "darwin"))
            with self.assertRaisesRegex(RuntimeError, "Privacy & Security > Camera"):
                source.start()

        fake_cv2.VideoCapture.assert_called_once_with(0, 1200)
        capture.release.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

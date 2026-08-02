"""ONNX Runtime inference for the exported OperAI-EYE DINOv3 model."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .decision import CLASS_NAMES, FramePrediction, prediction_from_probabilities

DEFAULT_IMAGE_SIZE = 224
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def select_execution_providers(requested: str, available: Sequence[str]) -> list[str]:
    """Select a portable ONNX Runtime provider chain."""

    available_set = set(available)
    if requested == "cpu":
        return ["CPUExecutionProvider"]
    if requested == "coreml":
        if "CoreMLExecutionProvider" not in available_set:
            raise RuntimeError("CoreMLExecutionProvider is not available")
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    if requested != "auto":
        raise ValueError(f"Unsupported execution provider: {requested}")
    if "CoreMLExecutionProvider" in available_set:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _resize_short_side(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    if width <= height:
        resized_size = (size, int(size * height / width))
    else:
        resized_size = (int(size * width / height), size)
    resized = image.resize(
        resized_size,
        resample=Image.Resampling.BICUBIC,
    )
    return resized


def preprocess_image(
    image: Image.Image,
    *,
    size: int = DEFAULT_IMAGE_SIZE,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> np.ndarray:
    """Match the Resize(256), CenterCrop(224), ImageNet normalization pipeline."""

    resize_short_side = round(size * 256 / 224)
    resized = _resize_short_side(image, resize_short_side)
    # torchvision CenterCrop rounds half-pixel offsets instead of flooring them.
    left = round((resized.width - size) / 2)
    top = round((resized.height - size) / 2)
    cropped = resized.crop((left, top, left + size, top + size))
    array = np.asarray(cropped, dtype=np.float32) / np.float32(255.0)
    array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(
        std, dtype=np.float32
    )
    return np.transpose(array, (2, 0, 1))


def sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float32)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


class DinoOnnxClassifier:
    """Load the edge ONNX artifact once and classify a complete burst in one call."""

    def __init__(
        self,
        model_path: Path,
        *,
        metadata_path: Path | None = None,
        intra_op_threads: int = 4,
        inter_op_threads: int = 1,
        execution_provider: str = "auto",
    ) -> None:
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "onnxruntime is required for edge inference; install the edge extra"
            ) from exc

        if not model_path.is_file():
            raise FileNotFoundError(f"DINO ONNX model not found: {model_path}")
        inferred_metadata = model_path.with_suffix(".json")
        metadata_file = metadata_path or inferred_metadata
        self.metadata: dict[str, Any] = {}
        if metadata_file.is_file():
            self.metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = inter_op_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=select_execution_providers(
                execution_provider, ort.get_available_providers()
            ),
        )
        self.input_name = self.metadata.get(
            "input_name", self.session.get_inputs()[0].name
        )
        self.output_name = self.metadata.get(
            "output_name", self.session.get_outputs()[0].name
        )
        self.image_size = int(self.metadata.get("image_size", DEFAULT_IMAGE_SIZE))
        self.mean = tuple(self.metadata.get("mean", DEFAULT_MEAN))
        self.std = tuple(self.metadata.get("std", DEFAULT_STD))
        if (
            len(self.mean) != 3
            or len(self.std) != 3
            or any(float(value) <= 0 for value in self.std)
        ):
            raise ValueError(
                "Model normalization metadata must contain three valid channels"
            )
        self.class_names = tuple(self.metadata.get("class_names", CLASS_NAMES))
        if self.class_names != CLASS_NAMES:
            raise ValueError(
                f"Model classes {self.class_names!r} do not match {CLASS_NAMES!r}"
            )

    def classify(self, images: Sequence[Image.Image]) -> list[FramePrediction]:
        if not images:
            raise ValueError("Cannot classify an empty image batch")
        batch = np.stack(
            [
                preprocess_image(
                    image,
                    size=self.image_size,
                    mean=self.mean,
                    std=self.std,
                )
                for image in images
            ]
        )
        outputs = self.session.run([self.output_name], {self.input_name: batch})
        logits = np.asarray(outputs[0], dtype=np.float32)
        if logits.shape != (len(images), len(CLASS_NAMES)):
            raise RuntimeError(
                f"Unexpected model output shape {logits.shape}; "
                f"expected ({len(images)}, {len(CLASS_NAMES)})"
            )
        if not np.isfinite(logits).all():
            raise RuntimeError("Model returned non-finite logits")
        probabilities = sigmoid(logits)
        return [
            prediction_from_probabilities(
                {
                    name: float(probabilities[row, column])
                    for column, name in enumerate(CLASS_NAMES)
                }
            )
            for row in range(len(images))
        ]

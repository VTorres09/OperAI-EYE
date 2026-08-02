"""Triton HTTP inference client for the exported OperAI-EYE DINOv3 model."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from PIL import Image

from .decision import CLASS_NAMES, FramePrediction, prediction_from_probabilities
from .inference import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MEAN,
    DEFAULT_STD,
    preprocess_image,
    sigmoid,
)

TRITON_HEADER_LENGTH = "Inference-Header-Content-Length"


class TritonDinoClassifier:
    """Classify a complete image burst through Triton's binary HTTP protocol."""

    def __init__(
        self,
        url: str,
        *,
        model_name: str = "operai_eye_dinov3",
        model_version: str = "",
        input_name: str = "images",
        output_name: str = "logits",
        image_size: int = DEFAULT_IMAGE_SIZE,
        mean: Sequence[float] = DEFAULT_MEAN,
        std: Sequence[float] = DEFAULT_STD,
        timeout_seconds: float = 120.0,
        urlopen: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        self.url = url.rstrip("/")
        self.model_name = model_name
        self.model_version = model_version
        self.input_name = input_name
        self.output_name = output_name
        self.image_size = image_size
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.timeout_seconds = timeout_seconds
        self._urlopen = urlopen

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
        ).astype("<f4", copy=False)
        logits = self._infer(batch)
        if logits.shape != (len(images), len(CLASS_NAMES)):
            raise RuntimeError(
                f"Unexpected Triton output shape {logits.shape}; "
                f"expected ({len(images)}, {len(CLASS_NAMES)})"
            )
        if not np.isfinite(logits).all():
            raise RuntimeError("Triton returned non-finite logits")
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

    def is_live(self) -> bool:
        return self._health("/v2/health/live")

    def is_ready(self) -> bool:
        return self._health(f"{self._model_path()}/ready")

    def _health(self, path: str) -> bool:
        request = urllib.request.Request(f"{self.url}{path}", method="GET")
        try:
            with self._urlopen(request, timeout=5.0) as response:
                return 200 <= int(response.status) < 300
        except (OSError, urllib.error.URLError):
            return False

    def _model_path(self) -> str:
        path = f"/v2/models/{self.model_name}"
        if self.model_version:
            path += f"/versions/{self.model_version}"
        return path

    def _infer(self, batch: np.ndarray) -> np.ndarray:
        input_bytes = batch.tobytes(order="C")
        header = {
            "inputs": [
                {
                    "name": self.input_name,
                    "shape": list(batch.shape),
                    "datatype": "FP32",
                    "parameters": {"binary_data_size": len(input_bytes)},
                }
            ],
            "outputs": [
                {
                    "name": self.output_name,
                    "parameters": {"binary_data": True},
                }
            ],
        }
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            f"{self.url}{self._model_path()}/infer",
            data=header_bytes + input_bytes,
            headers={
                "Content-Type": "application/octet-stream",
                TRITON_HEADER_LENGTH: str(len(header_bytes)),
            },
            method="POST",
        )
        try:
            with self._urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read()
                response_header_length = response.headers.get(TRITON_HEADER_LENGTH)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Triton inference returned HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Triton at {self.url}: {exc.reason}"
            ) from exc

        if response_header_length is None:
            raise RuntimeError("Triton response did not include a binary header length")
        try:
            header_length = int(response_header_length)
            metadata = json.loads(payload[:header_length])
            output = next(
                item for item in metadata["outputs"] if item["name"] == self.output_name
            )
            shape = tuple(int(value) for value in output["shape"])
            binary_size = int(output["parameters"]["binary_data_size"])
        except (
            KeyError,
            StopIteration,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise RuntimeError("Triton returned malformed output metadata") from exc
        binary = payload[header_length : header_length + binary_size]
        expected_size = int(np.prod(shape)) * np.dtype("<f4").itemsize
        if binary_size != expected_size or len(binary) != expected_size:
            raise RuntimeError(
                f"Triton returned {len(binary)} output bytes; expected {expected_size}"
            )
        return np.frombuffer(binary, dtype="<f4").reshape(shape).copy()

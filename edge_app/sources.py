"""Camera and offline replay sources."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .config import CameraConfig


class EndOfSource(EOFError):
    """Raised when an offline replay source has no frames left."""


class ImageSource(Protocol):
    def start(self) -> None: ...

    def capture(self) -> Image.Image: ...

    def close(self) -> None: ...


def _rotate(image: Image.Image, degrees: int) -> Image.Image:
    if not degrees:
        return image
    return image.rotate(degrees, expand=True)


class Picamera2Source:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.camera = None

    def start(self) -> None:
        try:
            from picamera2 import Picamera2
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Picamera2 is not available. Install the Raspberry Pi OS "
                "python3-picamera2 package and expose system site packages."
            ) from exc
        self.camera = Picamera2()
        camera_config = self.camera.create_still_configuration(
            main={
                "size": (self.config.width, self.config.height),
                "format": "RGB888",
            },
            buffer_count=2,
        )
        self.camera.configure(camera_config)
        self.camera.start()
        if self.config.warmup_seconds:
            time.sleep(self.config.warmup_seconds)

    def capture(self) -> Image.Image:
        if self.camera is None:
            raise RuntimeError("Camera has not been started")
        array = self.camera.capture_array("main")
        return _rotate(Image.fromarray(array, mode="RGB"), self.config.rotation)

    def close(self) -> None:
        if self.camera is not None:
            self.camera.stop()
            self.camera.close()
            self.camera = None


class OpenCvSource:
    """USB camera or conventional video-file source."""

    def __init__(
        self,
        device: str,
        *,
        width: int | None = None,
        height: int | None = None,
        rotation: int = 0,
        frame_step: int = 1,
    ) -> None:
        if frame_step <= 0:
            raise ValueError("frame_step must be positive")
        self.device = device
        self.width = width
        self.height = height
        self.rotation = rotation
        self.frame_step = frame_step
        self.capture_handle = None

    def start(self) -> None:
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "opencv-python-headless is required for USB cameras and video replay"
            ) from exc
        source: str | int = int(self.device) if self.device.isdecimal() else self.device
        self.capture_handle = cv2.VideoCapture(source)
        if self.width:
            self.capture_handle.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.capture_handle.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.capture_handle.isOpened():
            self.close()
            raise RuntimeError(f"Could not open camera/video source: {self.device}")

    def capture(self) -> Image.Image:
        if self.capture_handle is None:
            raise RuntimeError("Source has not been started")
        import cv2

        ok, bgr = self.capture_handle.read()
        if not ok:
            raise EndOfSource(f"Reached the end of {self.device}")
        for _ in range(self.frame_step - 1):
            if not self.capture_handle.grab():
                break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return _rotate(Image.fromarray(rgb), self.rotation)

    def close(self) -> None:
        if self.capture_handle is not None:
            self.capture_handle.release()
            self.capture_handle = None


def _natural_key(path: Path) -> list[str | int]:
    return [
        int(part) if part.isdecimal() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


class DirectorySource:
    """Replay a naturally sorted directory of extracted dataset frames."""

    def __init__(
        self,
        path: Path,
        *,
        pattern: str = "*.png",
        frame_step: int = 1,
        rotation: int = 0,
    ) -> None:
        if frame_step <= 0:
            raise ValueError("frame_step must be positive")
        self.path = path
        self.pattern = pattern
        self.frame_step = frame_step
        self.rotation = rotation
        self.paths: list[Path] = []
        self.index = 0

    def start(self) -> None:
        self.index = 0
        self.paths = sorted(self.path.glob(self.pattern), key=_natural_key)
        if not self.paths:
            raise FileNotFoundError(
                f"No images matching {self.pattern!r} in {self.path}"
            )

    def capture(self) -> Image.Image:
        if self.index >= len(self.paths):
            raise EndOfSource(f"Reached the end of {self.path}")
        path = self.paths[self.index]
        self.index += self.frame_step
        with Image.open(path) as image:
            return _rotate(image.convert("RGB"), self.rotation)

    def close(self) -> None:
        return None


class Hdf5Source:
    """Replay one exocentric camera directly from an EgoExOR HDF5 take."""

    def __init__(
        self,
        path: Path,
        *,
        take_path: str,
        camera: str,
        frame_step: int = 1,
        rotation: int = 0,
    ) -> None:
        if frame_step <= 0:
            raise ValueError("frame_step must be positive")
        self.path = path
        self.take_path = take_path.rstrip("/")
        self.camera_name = camera
        self.frame_step = frame_step
        self.rotation = rotation
        self.handle = None
        self.frames = None
        self.camera_index: int | None = None
        self.index = 0

    def start(self) -> None:
        import h5py

        self.index = 0
        self.camera_index = None
        self.handle = h5py.File(self.path, "r")
        sources_path = f"{self.take_path}/sources"
        frames_path = f"{self.take_path}/frames/rgb"
        if sources_path not in self.handle or frames_path not in self.handle:
            self.close()
            raise KeyError(
                f"{self.take_path!r} does not contain sources and frames/rgb"
            )
        sources = self.handle[sources_path]
        source_count = int(sources.attrs["source_count"])
        for index in range(source_count):
            name = sources.attrs.get(f"source_{index}")
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            if name == self.camera_name:
                self.camera_index = index
                break
        if self.camera_index is None:
            self.close()
            raise KeyError(
                f"Camera {self.camera_name!r} not present under {self.take_path}"
            )
        self.frames = self.handle[frames_path]

    def capture(self) -> Image.Image:
        if self.frames is None or self.camera_index is None:
            raise RuntimeError("HDF5 source has not been started")
        if self.index >= self.frames.shape[0]:
            raise EndOfSource(f"Reached the end of {self.path}:{self.take_path}")
        array = np.asarray(self.frames[self.index, self.camera_index], dtype=np.uint8)
        self.index += self.frame_step
        return _rotate(Image.fromarray(array, mode="RGB"), self.rotation)

    def close(self) -> None:
        self.frames = None
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def create_camera_source(config: CameraConfig) -> ImageSource:
    if config.backend == "picamera2":
        return Picamera2Source(config)
    if config.backend == "opencv":
        return OpenCvSource(
            config.device,
            width=config.width,
            height=config.height,
            rotation=config.rotation,
            frame_step=1,
        )
    raise ValueError(f"Unsupported camera backend: {config.backend}")

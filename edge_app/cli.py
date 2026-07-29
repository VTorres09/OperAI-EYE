"""Command-line interface for the OperAI-EYE edge service."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

from .config import EdgeConfig, load_config
from .inference import DinoOnnxClassifier
from .service import EdgeService
from .sources import (
    DirectorySource,
    Hdf5Source,
    ImageSource,
    OpenCvSource,
    create_camera_source,
)
from .storage import PredictionStore

DEFAULT_CONFIG = Path("/etc/operai-eye/config.toml")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help=f"TOML configuration file (production default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("run", help="Run the Raspberry Pi camera service forever.")
    subparsers.add_parser("once", help="Capture and classify one camera burst.")

    status = subparsers.add_parser("status", help="Print local service status as JSON.")
    status.add_argument("--hours", type=int, default=24)

    directory = subparsers.add_parser(
        "replay-directory",
        help="Replay extracted EgoExOR frames from a directory.",
    )
    directory.add_argument("path", type=Path)
    directory.add_argument("--pattern", default="*.png")
    _add_replay_options(directory, default_frame_step=1)

    video = subparsers.add_parser(
        "replay-video", help="Replay a conventional video through the burst pipeline."
    )
    video.add_argument("path", type=Path)
    _add_replay_options(video, default_frame_step=30)

    hdf5 = subparsers.add_parser(
        "replay-hdf5", help="Replay one EgoExOR HDF5 take and camera."
    )
    hdf5.add_argument("path", type=Path)
    hdf5.add_argument(
        "--take-path",
        required=True,
        help="Example: data/MISS/1/take/2",
    )
    hdf5.add_argument("--camera", default="external_1")
    _add_replay_options(hdf5, default_frame_step=1)
    return parser.parse_args(argv)


def _add_replay_options(
    parser: argparse.ArgumentParser, *, default_frame_step: int
) -> None:
    parser.add_argument("--frame-step", type=int, default=default_frame_step)
    parser.add_argument("--max-bursts", type=int)
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Keep the configured one-second spacing within each burst.",
    )


def _load(args: argparse.Namespace) -> EdgeConfig:
    config_path = args.config
    if config_path is None and DEFAULT_CONFIG.is_file():
        config_path = DEFAULT_CONFIG
    return load_config(config_path)


def _classifier(config: EdgeConfig) -> DinoOnnxClassifier:
    return DinoOnnxClassifier(
        config.model.path,
        metadata_path=config.model.metadata_path,
        intra_op_threads=config.model.intra_op_threads,
        inter_op_threads=config.model.inter_op_threads,
    )


def _store(config: EdgeConfig) -> PredictionStore:
    return PredictionStore(
        config.storage.database_path,
        image_directory=config.storage.image_directory,
        retain_images=config.storage.retain_images,
        retention_days=config.storage.retention_days,
        jpeg_quality=config.storage.jpeg_quality,
    )


def _service(
    config: EdgeConfig, source: ImageSource, store: PredictionStore
) -> EdgeService:
    return EdgeService(
        source=source,
        classifier=_classifier(config),
        store=store,
        service_config=config.service,
        decision_config=config.decision,
    )


def _install_signal_handlers(stop_event: threading.Event) -> None:
    def stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)


def _print_cycle(result: object) -> None:
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = _load(args)
    store = _store(config)
    try:
        if args.command == "status":
            print(json.dumps(store.summary(hours=args.hours), indent=2, sort_keys=True))
            return 0

        source: ImageSource
        offline = False
        maximum_cycles = None
        if args.command in {"run", "once"}:
            source = create_camera_source(config.camera)
        elif args.command == "replay-directory":
            source = DirectorySource(
                args.path,
                pattern=args.pattern,
                frame_step=args.frame_step,
                rotation=config.camera.rotation,
            )
            offline = True
            maximum_cycles = args.max_bursts
        elif args.command == "replay-video":
            source = OpenCvSource(
                str(args.path),
                rotation=config.camera.rotation,
                frame_step=args.frame_step,
            )
            offline = True
            maximum_cycles = args.max_bursts
        elif args.command == "replay-hdf5":
            source = Hdf5Source(
                args.path,
                take_path=args.take_path,
                camera=args.camera,
                frame_step=args.frame_step,
                rotation=config.camera.rotation,
            )
            offline = True
            maximum_cycles = args.max_bursts
        else:  # pragma: no cover - guarded by argparse
            raise AssertionError(args.command)

        if offline and not args.realtime:
            config = replace(
                config,
                service=replace(config.service, capture_spacing_seconds=0.0),
            )
        service = _service(config, source, store)

        if args.command == "once":
            try:
                source.start()
                _print_cycle(service.run_once())
            finally:
                source.close()
            return 0

        stop_event = threading.Event()
        _install_signal_handlers(stop_event)
        cycles = service.run_forever(
            stop_event,
            maximum_cycles=maximum_cycles,
            offline=offline,
        )
        logging.getLogger(__name__).info("Completed %d burst(s)", cycles)
        store.prune_images()
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())

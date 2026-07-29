"""Continuous five-frame burst capture and batched classification loop."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from PIL import Image

from .config import DecisionConfig, ServiceConfig
from .decision import FramePrediction, VoteResult, majority_vote
from .sources import EndOfSource, ImageSource
from .storage import PredictionStore

logger = logging.getLogger(__name__)


class Classifier(Protocol):
    def classify(self, images: Sequence[Image.Image]) -> list[FramePrediction]: ...


@dataclass(frozen=True)
class CycleResult:
    burst_id: str
    vote: VoteResult
    captures_succeeded: int
    inference_ms: float


class EdgeService:
    def __init__(
        self,
        *,
        source: ImageSource,
        classifier: Classifier,
        store: PredictionStore,
        service_config: ServiceConfig,
        decision_config: DecisionConfig,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        utcnow: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.source = source
        self.classifier = classifier
        self.store = store
        self.service_config = service_config
        self.decision_config = decision_config
        self.sleep = sleep
        self.monotonic = monotonic
        self.utcnow = utcnow

    def run_once(self) -> CycleResult:
        started_at = self.utcnow()
        images: list[Image.Image] = []
        captured_at: list[datetime] = []
        capture_positions: list[int] = []
        capture_errors: list[str] = []
        burst_start = self.monotonic()

        for index in range(self.service_config.burst_size):
            target = burst_start + index * self.service_config.capture_spacing_seconds
            delay = target - self.monotonic()
            if delay > 0:
                self.sleep(delay)
            try:
                images.append(self.source.capture())
                captured_at.append(self.utcnow())
                capture_positions.append(index)
            except EndOfSource:
                raise
            except Exception as exc:
                capture_errors.append(f"frame {index + 1}: {type(exc).__name__}: {exc}")
                logger.exception("Capture %d failed", index + 1)

        if len(images) < self.service_config.minimum_captures:
            message = "; ".join(capture_errors) or "insufficient successful captures"
            burst_id = self.store.record_failure(
                started_at=started_at,
                completed_at=self.utcnow(),
                captures_expected=self.service_config.burst_size,
                captures_succeeded=len(images),
                error=message,
            )
            raise RuntimeError(
                f"Burst {burst_id} captured {len(images)}/"
                f"{self.service_config.burst_size} frames: {message}"
            )

        inference_start = self.monotonic()
        try:
            predictions = self.classifier.classify(images)
            if len(predictions) != len(images):
                raise RuntimeError(
                    f"Classifier returned {len(predictions)} predictions "
                    f"for {len(images)} images"
                )
        except Exception as exc:
            burst_id = self.store.record_failure(
                started_at=started_at,
                completed_at=self.utcnow(),
                captures_expected=self.service_config.burst_size,
                captures_succeeded=len(images),
                error=f"inference: {type(exc).__name__}: {exc}",
            )
            raise RuntimeError(f"Burst {burst_id} inference failed") from exc
        inference_ms = (self.monotonic() - inference_start) * 1000
        vote = majority_vote(
            predictions,
            minimum_phase_confidence=self.decision_config.minimum_phase_confidence,
            minimum_vote_fraction=self.decision_config.minimum_vote_fraction,
        )
        burst_id = self.store.record_success(
            started_at=started_at,
            completed_at=self.utcnow(),
            captured_at=captured_at,
            images=images,
            predictions=predictions,
            vote=vote,
            captures_expected=self.service_config.burst_size,
            inference_ms=inference_ms,
            capture_positions=capture_positions,
            warning="; ".join(capture_errors) or None,
        )
        logger.info(
            "burst=%s phase=%s confidence=%.4f votes=%s inference_ms=%.1f",
            burst_id,
            vote.phase,
            vote.confidence,
            dict(vote.votes),
            inference_ms,
        )
        return CycleResult(
            burst_id=burst_id,
            vote=vote,
            captures_succeeded=len(images),
            inference_ms=inference_ms,
        )

    def run_forever(
        self,
        stop_event: threading.Event,
        *,
        maximum_cycles: int | None = None,
        offline: bool = False,
    ) -> int:
        cycles = 0
        consecutive_failures = 0
        try:
            self.source.start()
            if (
                not self.service_config.run_immediately
                and not offline
                and stop_event.wait(self.service_config.interval_seconds)
            ):
                return cycles
            next_cycle = self.monotonic()
            while not stop_event.is_set():
                if not offline:
                    delay = next_cycle - self.monotonic()
                    if delay > 0 and stop_event.wait(delay):
                        break
                try:
                    self.run_once()
                    consecutive_failures = 0
                except EndOfSource:
                    logger.info("Offline replay source exhausted")
                    break
                except Exception:
                    logger.exception("Burst failed; service will continue")
                    consecutive_failures += 1
                    if (
                        not offline
                        and consecutive_failures
                        >= self.service_config.restart_after_failures
                    ):
                        raise RuntimeError(
                            f"{consecutive_failures} consecutive bursts failed"
                        )
                cycles += 1
                if cycles % 60 == 0:
                    try:
                        removed = self.store.prune_images()
                        if removed:
                            logger.info("Pruned %d expired image(s)", removed)
                    except Exception:
                        logger.exception("Image retention cleanup failed")
                if maximum_cycles is not None and cycles >= maximum_cycles:
                    break
                if offline:
                    continue
                next_cycle += self.service_config.interval_seconds
                now = self.monotonic()
                skipped_intervals = 0
                while next_cycle <= now:
                    next_cycle += self.service_config.interval_seconds
                    skipped_intervals += 1
                if skipped_intervals:
                    logger.warning(
                        "Skipped %d interval(s) because processing exceeded the schedule",
                        skipped_intervals,
                    )
            return cycles
        finally:
            self.source.close()

"""Local, durable prediction storage with bounded optional image retention."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from PIL import Image

from .decision import FramePrediction, VoteResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS bursts (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    phase TEXT NOT NULL,
    confidence REAL,
    vote_fraction REAL,
    votes_json TEXT NOT NULL,
    mean_phase_scores_json TEXT NOT NULL,
    captures_expected INTEGER NOT NULL,
    captures_succeeded INTEGER NOT NULL,
    inference_ms REAL,
    uncertain_reason TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_bursts_started_at ON bursts(started_at DESC);

CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    burst_id TEXT NOT NULL REFERENCES bursts(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    captured_at TEXT NOT NULL,
    phase TEXT NOT NULL,
    confidence REAL NOT NULL,
    class_probabilities_json TEXT NOT NULL,
    phase_scores_json TEXT NOT NULL,
    image_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_frames_burst_id ON frames(burst_id);
"""


class PredictionStore:
    def __init__(
        self,
        database_path: Path,
        *,
        image_directory: Path,
        retain_images: str = "none",
        retention_days: int = 7,
        jpeg_quality: int = 85,
    ) -> None:
        self.database_path = database_path
        self.image_directory = image_directory
        self.retain_images = retain_images
        self.retention_days = retention_days
        self.jpeg_quality = jpeg_quality
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(SCHEMA)

    def record_success(
        self,
        *,
        started_at: datetime,
        completed_at: datetime,
        captured_at: Sequence[datetime],
        images: Sequence[Image.Image],
        predictions: Sequence[FramePrediction],
        vote: VoteResult,
        captures_expected: int,
        inference_ms: float,
        capture_positions: Sequence[int] | None = None,
        warning: str | None = None,
    ) -> str:
        if not (len(captured_at) == len(images) == len(predictions)):
            raise ValueError(
                "captured_at, images, and predictions must have equal length"
            )
        positions = list(capture_positions or range(len(images)))
        if len(positions) != len(images):
            raise ValueError("capture_positions and images must have equal length")
        burst_id = _burst_id(started_at)
        keep_images = self.retain_images == "all" or (
            self.retain_images == "uncertain" and vote.phase == "UNKNOWN"
        )
        image_paths: list[str | None] = [None] * len(images)
        if keep_images:
            image_paths = self._save_images(burst_id, images)

        with self.connection:
            self.connection.execute(
                """
                INSERT INTO bursts (
                    id, started_at, completed_at, phase, confidence, vote_fraction,
                    votes_json, mean_phase_scores_json, captures_expected,
                    captures_succeeded, inference_ms, uncertain_reason, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    burst_id,
                    _iso(started_at),
                    _iso(completed_at),
                    vote.phase,
                    vote.confidence,
                    vote.vote_fraction,
                    json.dumps(vote.votes, sort_keys=True),
                    json.dumps(vote.mean_phase_scores, sort_keys=True),
                    captures_expected,
                    len(predictions),
                    inference_ms,
                    vote.uncertain_reason,
                    warning,
                ),
            )
            self.connection.executemany(
                """
                INSERT INTO frames (
                    burst_id, position, captured_at, phase, confidence,
                    class_probabilities_json, phase_scores_json, image_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        burst_id,
                        positions[position],
                        _iso(captured_at[position]),
                        prediction.phase,
                        prediction.confidence,
                        json.dumps(prediction.class_probabilities, sort_keys=True),
                        json.dumps(prediction.phase_scores, sort_keys=True),
                        image_paths[position],
                    )
                    for position, prediction in enumerate(predictions)
                ],
            )
        return burst_id

    def record_failure(
        self,
        *,
        started_at: datetime,
        completed_at: datetime,
        captures_expected: int,
        captures_succeeded: int,
        error: str,
    ) -> str:
        burst_id = _burst_id(started_at)
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO bursts (
                    id, started_at, completed_at, phase, confidence, vote_fraction,
                    votes_json, mean_phase_scores_json, captures_expected,
                    captures_succeeded, inference_ms, uncertain_reason, error
                ) VALUES (?, ?, ?, 'ERROR', NULL, NULL, '{}', '{}', ?, ?, NULL, NULL, ?)
                """,
                (
                    burst_id,
                    _iso(started_at),
                    _iso(completed_at),
                    captures_expected,
                    captures_succeeded,
                    error,
                ),
            )
        return burst_id

    def latest(self) -> dict[str, object] | None:
        row = self.connection.execute(
            "SELECT * FROM bursts ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return _decode_row(row) if row else None

    def summary(self, *, hours: int = 24) -> dict[str, object]:
        since = datetime.now(UTC) - timedelta(hours=hours)
        rows = self.connection.execute(
            """
            SELECT phase, COUNT(*) AS count
            FROM bursts
            WHERE started_at >= ?
            GROUP BY phase
            ORDER BY phase
            """,
            (_iso(since),),
        ).fetchall()
        return {
            "database_path": str(self.database_path),
            "latest": self.latest(),
            "window_hours": hours,
            "phase_counts": {row["phase"]: row["count"] for row in rows},
        }

    def prune_images(self, *, now: datetime | None = None) -> int:
        if not self.image_directory.exists():
            return 0
        cutoff = (now or datetime.now(UTC)) - timedelta(days=self.retention_days)
        removed = 0
        paths = self.connection.execute(
            """
            SELECT id, image_path FROM frames
            WHERE image_path IS NOT NULL AND captured_at < ?
            """,
            (_iso(cutoff),),
        ).fetchall()
        with self.connection:
            for row in paths:
                path = Path(row["image_path"])
                try:
                    path.unlink()
                    removed += 1
                except FileNotFoundError:
                    pass
                self.connection.execute(
                    "UPDATE frames SET image_path = NULL WHERE id = ?", (row["id"],)
                )
        for directory in sorted(self.image_directory.glob("*"), reverse=True):
            if directory.is_dir():
                try:
                    directory.rmdir()
                except OSError:
                    pass
        return removed

    def close(self) -> None:
        self.connection.close()

    def _save_images(self, burst_id: str, images: Sequence[Image.Image]) -> list[str]:
        directory = self.image_directory / burst_id
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for index, image in enumerate(images):
            path = directory / f"frame_{index + 1:02d}.jpg"
            temporary = path.with_suffix(".jpg.tmp")
            image.save(temporary, format="JPEG", quality=self.jpeg_quality)
            temporary.replace(path)
            paths.append(str(path))
        return paths


def _burst_id(started_at: datetime) -> str:
    timestamp = started_at.astimezone(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _decode_row(row: sqlite3.Row) -> dict[str, object]:
    decoded = dict(row)
    decoded["votes"] = json.loads(str(decoded.pop("votes_json")))
    decoded["mean_phase_scores"] = json.loads(
        str(decoded.pop("mean_phase_scores_json"))
    )
    return decoded

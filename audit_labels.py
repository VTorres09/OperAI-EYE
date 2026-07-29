#!/usr/bin/env python3
"""Audit frame labels for temporal and confidence inconsistencies."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from label_data import DATA_DIR, OUTPUT_DIR, VALID_PHASES

CORE_PHASES = ("IDLE", "PATIENT_IN_ROOM", "SURGERY_ACTIVE")
CORE_PHASE_SET = set(CORE_PHASES)
DEFAULT_LABELS = (
    OUTPUT_DIR / "train_labels.csv",
    OUTPUT_DIR / "validation_labels.csv",
)
FIELDNAMES = [
    "priority",
    "score",
    "split",
    "path",
    "suggested_phase",
    "phase",
    "confidence",
    "reasons",
    "neighbor_vote_counts",
    "group_key",
    "frame_id",
    "prev_path",
    "prev_phase",
    "prev_gap",
    "next_path",
    "next_phase",
    "next_gap",
]


@dataclass(frozen=True)
class LabelRecord:
    path: str
    split: str
    surgery_type: str
    procedure_id: str
    take_id: str
    camera: str
    frame_id: int
    phase: str
    confidence: float | None
    row: dict[str, str]

    @property
    def group_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.split,
            self.surgery_type,
            self.procedure_id,
            self.take_id,
            self.camera,
        )

    @property
    def group_label(self) -> str:
        return "/".join(self.group_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        nargs="+",
        default=[path for path in DEFAULT_LABELS if path.exists()],
        help="Label CSV(s) to audit; defaults to train and validation outputs.",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "label_audit_suspects.csv",
        help="CSV file for review candidates.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=OUTPUT_DIR / "label_audit_summary.json",
        help="JSON summary file.",
    )
    parser.add_argument(
        "--max-neighbor-gap",
        type=int,
        default=45,
        help="Maximum frame gap for immediate-neighbor temporal checks.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=90,
        help="Frame window for local majority voting.",
    )
    parser.add_argument(
        "--minimum-neighbors",
        type=int,
        default=3,
        help="Minimum labeled core-phase neighbors needed for majority voting.",
    )
    parser.add_argument(
        "--majority-threshold",
        type=float,
        default=0.75,
        help="Minimum neighbor share required for a majority mismatch signal.",
    )
    parser.add_argument(
        "--low-confidence",
        type=float,
        default=0.55,
        help="Confidence at or below this value is flagged.",
    )
    parser.add_argument(
        "--hash-distance",
        type=int,
        default=6,
        help="Average-hash distance at or below this value marks similar images.",
    )
    parser.add_argument(
        "--no-image-hash",
        action="store_true",
        help="Disable visual near-duplicate checks.",
    )
    parser.add_argument("--json", action="store_true", help="Print summary JSON.")
    return parser.parse_args()


def parse_confidence(value: str | None) -> float | None:
    try:
        return float(value or "")
    except ValueError:
        return None


def read_labels(paths: Iterable[Path]) -> list[LabelRecord]:
    records: list[LabelRecord] = []
    for labels_path in paths:
        with labels_path.open(newline="", encoding="utf-8-sig") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                phase = (row.get("phase") or "").strip()
                error = (row.get("error") or "").strip()
                rel_path = (row.get("path") or "").strip()
                if error or not rel_path or phase not in VALID_PHASES:
                    continue
                try:
                    frame_id = int((row.get("frame_id") or "").strip())
                except ValueError:
                    raise ValueError(
                        f"Invalid frame_id on row {row_number} in {labels_path}: "
                        f"{row.get('frame_id')!r}"
                    ) from None
                records.append(
                    LabelRecord(
                        path=rel_path,
                        split=(row.get("split") or "").strip(),
                        surgery_type=(row.get("surgery_type") or "").strip(),
                        procedure_id=(row.get("procedure_id") or "").strip(),
                        take_id=(row.get("take_id") or "").strip(),
                        camera=(row.get("camera") or "").strip(),
                        frame_id=frame_id,
                        phase=phase,
                        confidence=parse_confidence(row.get("confidence")),
                        row=row,
                    )
                )
    return records


def nearest_core(
    records: list[LabelRecord],
    index: int,
    direction: int,
) -> tuple[LabelRecord, int] | None:
    current = records[index]
    probe = index + direction
    while 0 <= probe < len(records):
        candidate = records[probe]
        if candidate.phase in CORE_PHASE_SET:
            return candidate, abs(current.frame_id - candidate.frame_id)
        probe += direction
    return None


def local_votes(
    records: list[LabelRecord],
    index: int,
    window: int,
) -> Counter[str]:
    current = records[index]
    votes: Counter[str] = Counter()
    for candidate in records:
        if candidate is current or candidate.phase not in CORE_PHASE_SET:
            continue
        if abs(candidate.frame_id - current.frame_id) <= window:
            votes[candidate.phase] += 1
    return votes


@lru_cache(maxsize=8192)
def average_hash(path: str) -> tuple[int, ...]:
    with Image.open(path) as image:
        gray = image.convert("L").resize((8, 8))
        pixels = list(gray.getdata())
    mean = sum(pixels) / len(pixels)
    return tuple(1 if value >= mean else 0 for value in pixels)


def hamming(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return sum(a != b for a, b in zip(left, right))


def visual_disagreement(
    current: LabelRecord,
    neighbor: LabelRecord | None,
    data_dir: Path,
    max_gap: int,
    hash_distance: int,
) -> bool:
    if neighbor is None:
        return False
    if current.phase == neighbor.phase:
        return False
    if abs(current.frame_id - neighbor.frame_id) > max_gap:
        return False
    current_path = data_dir / current.path
    neighbor_path = data_dir / neighbor.path
    if not current_path.exists() or not neighbor_path.exists():
        return False
    distance = hamming(average_hash(str(current_path)), average_hash(str(neighbor_path)))
    return distance <= hash_distance


def priority(score: int) -> str:
    if score >= 7:
        return "high"
    if score >= 3:
        return "medium"
    return "low"


def best_vote(votes: Counter[str]) -> tuple[str | None, int, float]:
    total = sum(votes.values())
    if not total:
        return None, 0, 0.0
    phase, count = votes.most_common(1)[0]
    return phase, count, count / total


def audit_records(
    records: list[LabelRecord],
    *,
    data_dir: Path,
    max_neighbor_gap: int,
    window: int,
    minimum_neighbors: int,
    majority_threshold: float,
    low_confidence: float,
    hash_distance: int,
    image_hash: bool,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[LabelRecord]] = defaultdict(list)
    for record in records:
        grouped[record.group_key].append(record)

    suspects: list[dict[str, Any]] = []
    for group_records in grouped.values():
        group_records.sort(key=lambda record: record.frame_id)
        for index, record in enumerate(group_records):
            if record.phase not in CORE_PHASE_SET:
                continue

            prev_pair = nearest_core(group_records, index, -1)
            next_pair = nearest_core(group_records, index, 1)
            prev = prev_pair[0] if prev_pair else None
            prev_gap = prev_pair[1] if prev_pair else None
            next_ = next_pair[0] if next_pair else None
            next_gap = next_pair[1] if next_pair else None
            votes = local_votes(group_records, index, window)
            voted_phase, voted_count, voted_share = best_vote(votes)

            reasons: list[str] = []
            score = 0
            suggested_phase: str | None = None

            if (
                prev is not None
                and next_ is not None
                and prev_gap is not None
                and next_gap is not None
                and prev_gap <= max_neighbor_gap
                and next_gap <= max_neighbor_gap
                and prev.phase == next_.phase
                and record.phase != prev.phase
            ):
                reasons.append("isolated_temporal_island")
                score += 4
                suggested_phase = prev.phase

            if (
                voted_phase is not None
                and voted_phase != record.phase
                and sum(votes.values()) >= minimum_neighbors
                and voted_share >= majority_threshold
            ):
                reasons.append(
                    f"local_majority_mismatch:{voted_phase}:{voted_count}/"
                    f"{sum(votes.values())}"
                )
                score += 2
                suggested_phase = suggested_phase or voted_phase

            if record.confidence is not None and record.confidence <= low_confidence:
                reasons.append("low_confidence")
                score += 1

            if image_hash and reasons:
                visual_prev = visual_disagreement(
                    record,
                    prev,
                    data_dir,
                    max_neighbor_gap,
                    hash_distance,
                )
                visual_next = visual_disagreement(
                    record,
                    next_,
                    data_dir,
                    max_neighbor_gap,
                    hash_distance,
                )
                if visual_prev or visual_next:
                    reasons.append("visual_near_duplicate_disagreement")
                    score += 1
                    if suggested_phase is None:
                        if visual_prev and prev is not None:
                            suggested_phase = prev.phase
                        elif visual_next and next_ is not None:
                            suggested_phase = next_.phase

            if not reasons:
                continue

            suspects.append(
                {
                    "priority": priority(score),
                    "score": score,
                    "split": record.split,
                    "path": record.path,
                    "suggested_phase": suggested_phase or "",
                    "phase": record.phase,
                    "confidence": (
                        "" if record.confidence is None else f"{record.confidence:.4f}"
                    ),
                    "reasons": "|".join(reasons),
                    "neighbor_vote_counts": json.dumps(
                        dict(sorted(votes.items())),
                        sort_keys=True,
                    ),
                    "group_key": record.group_label,
                    "frame_id": record.frame_id,
                    "prev_path": prev.path if prev else "",
                    "prev_phase": prev.phase if prev else "",
                    "prev_gap": "" if prev_gap is None else prev_gap,
                    "next_path": next_.path if next_ else "",
                    "next_phase": next_.phase if next_ else "",
                    "next_gap": "" if next_gap is None else next_gap,
                }
            )

    suspects.sort(
        key=lambda row: (
            {"high": 0, "medium": 1, "low": 2}[row["priority"]],
            -int(row["score"]),
            row["split"],
            row["group_key"],
            int(row["frame_id"]),
        )
    )
    return suspects


def summarize(
    records: list[LabelRecord],
    suspects: list[dict[str, Any]],
    labels: list[Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        by_split[record.split][record.phase] += 1

    suspect_priorities = Counter(row["priority"] for row in suspects)
    suspect_reasons: Counter[str] = Counter()
    for row in suspects:
        suspect_reasons.update(
            reason.split(":", 1)[0] for reason in str(row["reasons"]).split("|")
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "labels": [str(path) for path in labels],
        "settings": {
            "max_neighbor_gap": args.max_neighbor_gap,
            "window": args.window,
            "minimum_neighbors": args.minimum_neighbors,
            "majority_threshold": args.majority_threshold,
            "low_confidence": args.low_confidence,
            "image_hash": not args.no_image_hash,
            "hash_distance": args.hash_distance,
        },
        "records": {
            "total": len(records),
            "by_split_phase": {
                split: dict(sorted(counts.items()))
                for split, counts in sorted(by_split.items())
            },
            "trainable_core_rows": sum(
                1 for record in records if record.phase in CORE_PHASE_SET
            ),
            "ignored_unknown_rows": sum(
                1 for record in records if record.phase == "UNKNOWN"
            ),
        },
        "suspects": {
            "total": len(suspects),
            "by_priority": dict(sorted(suspect_priorities.items())),
            "by_reason": dict(sorted(suspect_reasons.items())),
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if not args.labels:
        raise SystemExit("No label CSV files found; pass --labels explicitly.")
    records = read_labels(args.labels)
    suspects = audit_records(
        records,
        data_dir=args.data_dir,
        max_neighbor_gap=args.max_neighbor_gap,
        window=args.window,
        minimum_neighbors=args.minimum_neighbors,
        majority_threshold=args.majority_threshold,
        low_confidence=args.low_confidence,
        hash_distance=args.hash_distance,
        image_hash=not args.no_image_hash,
    )
    summary = summarize(records, suspects, args.labels, args)
    write_csv(args.output, suspects)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            f"Audited {len(records)} rows; wrote {len(suspects)} suspects to "
            f"{args.output}"
        )
        print(f"Summary: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

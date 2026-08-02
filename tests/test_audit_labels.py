"""Tests for temporal label-audit heuristics."""

import unittest

from operai_eye.pipeline.audit_labels import LabelRecord, audit_records


def record(frame_id: int, phase: str, confidence: float = 0.9) -> LabelRecord:
    return LabelRecord(
        path=f"train/MISS/1/take_1/external_1/frame_{frame_id:06d}.png",
        split="train",
        surgery_type="MISS",
        procedure_id="1",
        take_id="1",
        camera="external_1",
        frame_id=frame_id,
        phase=phase,
        confidence=confidence,
        row={},
    )


class AuditLabelsTest(unittest.TestCase):
    def test_flags_isolated_temporal_island(self) -> None:
        suspects = audit_records(
            [
                record(1, "PATIENT_IN_ROOM"),
                record(2, "SURGERY_ACTIVE"),
                record(3, "PATIENT_IN_ROOM"),
            ],
            data_dir=".",
            max_neighbor_gap=5,
            window=10,
            minimum_neighbors=2,
            majority_threshold=0.75,
            low_confidence=0.55,
            hash_distance=6,
            image_hash=False,
        )

        self.assertEqual(len(suspects), 1)
        self.assertEqual(suspects[0]["path"], record(2, "SURGERY_ACTIVE").path)
        self.assertEqual(suspects[0]["suggested_phase"], "PATIENT_IN_ROOM")
        self.assertIn("isolated_temporal_island", suspects[0]["reasons"])

    def test_does_not_flag_sustained_transition(self) -> None:
        suspects = audit_records(
            [
                record(1, "PATIENT_IN_ROOM"),
                record(2, "SURGERY_ACTIVE"),
                record(3, "SURGERY_ACTIVE"),
                record(4, "SURGERY_ACTIVE"),
            ],
            data_dir=".",
            max_neighbor_gap=5,
            window=1,
            minimum_neighbors=3,
            majority_threshold=0.75,
            low_confidence=0.55,
            hash_distance=6,
            image_hash=False,
        )

        self.assertEqual(suspects, [])


if __name__ == "__main__":
    unittest.main()

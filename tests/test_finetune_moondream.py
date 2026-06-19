"""Tests for Moondream SFT request shaping, metrics, and resume behavior."""

import unittest
import csv
import tempfile
from pathlib import Path

from PIL import Image

from finetuning.moondream import evaluate_finetune
from finetuning.moondream import finetune_moondream as finetune


def example(phase: str) -> dict:
    return {
        "image": Image.new("RGB", (4, 4)),
        "phase": phase,
        "question": finetune.QUESTION,
    }


class FakeFinetune:
    def __init__(self) -> None:
        self.groups = []
        self.steps = 0

    def train_step(self, groups, lr):
        self.groups.append((groups, lr))
        self.steps += 1
        return {"step": self.steps, "sft_loss": 1.0 / self.steps}

    def rollout_stream(self, requests, max_concurrency, buffer_size):
        for expected, request in requests:
            self.groups.append((request, max_concurrency, buffer_size))
            yield expected, {"rollouts": [{"output": {"answer": expected}}]}


class SftShapeTest(unittest.TestCase):
    def test_sft_group_uses_query_target(self) -> None:
        group = finetune.sft_group(example("SURGERY_ACTIVE"))

        self.assertEqual(group["mode"], "sft")
        self.assertEqual(group["request"]["skill"], "query")
        self.assertEqual(group["target"], {"answer": "SURGERY_ACTIVE"})

    def test_train_epoch_batches_and_resumes(self) -> None:
        model = FakeFinetune()
        data = [
            example("IDLE"),
            example("PATIENT_IN_ROOM"),
            example("SURGERY_ACTIVE"),
            example("PATIENT_IN_ROOM"),
            example("IDLE"),
        ]
        progress = []

        last_step, losses = finetune.train_epoch(
            model,
            data,
            epoch=0,
            batch_size=2,
            learning_rate=2e-4,
            seed=42,
            start_batch=1,
            on_progress=lambda batch, step: progress.append((batch, step["step"])),
        )

        self.assertEqual(len(model.groups), 2)
        self.assertEqual([len(groups) for groups, _ in model.groups], [2, 1])
        self.assertEqual(progress, [(2, 1), (3, 2)])
        self.assertEqual(last_step["step"], 2)
        self.assertEqual(losses, [1.0, 0.5])

    def test_epoch_order_is_deterministic_and_changes_by_epoch(self) -> None:
        first = finetune.epoch_order(20, 0, 42)
        repeated = finetune.epoch_order(20, 0, 42)
        second = finetune.epoch_order(20, 1, 42)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, second)

    def test_evaluation_ignores_unknown_labels(self) -> None:
        model = FakeFinetune()
        data = [
            example(phase)
            for phase in ("IDLE", "PATIENT_IN_ROOM", "SURGERY_ACTIVE", "UNKNOWN")
        ]

        metrics = finetune.evaluate(model, data, concurrency=2)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["total"], 3)
        self.assertEqual(metrics["per_class"]["PATIENT_IN_ROOM"]["recall"], 1.0)
        self.assertEqual(model.groups[0][0]["settings"]["max_tokens"], 16)

    def test_early_stopping_requires_one_percent_improvement(self) -> None:
        best, improved, stop = finetune.early_stopping_update(0.50, 0.509)
        self.assertEqual(best, 0.50)
        self.assertFalse(improved)
        self.assertTrue(stop)

        best, improved, stop = finetune.early_stopping_update(0.50, 0.51)
        self.assertEqual(best, 0.51)
        self.assertTrue(improved)
        self.assertFalse(stop)


class MetricsTest(unittest.TestCase):
    def test_metrics_include_core_macro_f1(self) -> None:
        metrics = finetune.compute_metrics(
            ["IDLE", "PATIENT_IN_ROOM", "SURGERY_ACTIVE", "UNKNOWN"],
            ["IDLE", "PATIENT_IN_ROOM", "UNKNOWN", "UNKNOWN"],
        )

        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertIn("core_macro_f1", metrics)
        self.assertEqual(metrics["per_class"]["SURGERY_ACTIVE"]["recall"], 0.0)

    def test_parser_rejects_ambiguous_outputs(self) -> None:
        self.assertEqual(finetune.parse_phase("IDLE"), "IDLE")
        self.assertEqual(
            finetune.parse_phase("It could be IDLE or UNKNOWN"),
            "UNKNOWN",
        )

    def test_cloud_evaluation_resume_retries_error_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "results.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "path",
                        "ground_truth",
                        "predicted",
                        "confidence",
                        "key_visual_cues",
                        "correct",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "path": "ok.png",
                        "ground_truth": "IDLE",
                        "predicted": "IDLE",
                        "confidence": 1.0,
                        "key_visual_cues": "IDLE",
                        "correct": True,
                    }
                )
                writer.writerow(
                    {
                        "path": "retry.png",
                        "ground_truth": "IDLE",
                        "predicted": "UNKNOWN",
                        "confidence": 0.0,
                        "key_visual_cues": "ERROR: HTTP Error 429: Too Many Requests",
                        "correct": False,
                    }
                )

            self.assertEqual(evaluate_finetune.successful_paths(path), {"ok.png"})


if __name__ == "__main__":
    unittest.main()

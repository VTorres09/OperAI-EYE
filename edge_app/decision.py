"""Convert DINO multilabel outputs into OR phases and aggregate a burst."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

PHASES = ("IDLE", "PATIENT_IN_ROOM", "SURGERY_ACTIVE")
CLASS_NAMES = ("idle", "people_in_room", "surgery_inactive", "surgery_active")


@dataclass(frozen=True)
class FramePrediction:
    phase: str
    confidence: float
    class_probabilities: Mapping[str, float]
    phase_scores: Mapping[str, float]


@dataclass(frozen=True)
class VoteResult:
    phase: str
    confidence: float
    vote_fraction: float
    votes: Mapping[str, int]
    mean_phase_scores: Mapping[str, float]
    uncertain_reason: str | None = None


def prediction_from_probabilities(
    probabilities: Mapping[str, float],
) -> FramePrediction:
    """Apply the phase mapping used by the DINOv3 evaluation pipeline."""

    missing = set(CLASS_NAMES) - set(probabilities)
    if missing:
        raise ValueError(f"Missing class probabilities: {', '.join(sorted(missing))}")
    phase_scores = {
        "IDLE": float(probabilities["idle"]),
        "PATIENT_IN_ROOM": float(probabilities["people_in_room"])
        * float(probabilities["surgery_inactive"]),
        "SURGERY_ACTIVE": float(probabilities["people_in_room"])
        * float(probabilities["surgery_active"]),
    }
    phase = max(PHASES, key=lambda item: phase_scores[item])
    return FramePrediction(
        phase=phase,
        confidence=phase_scores[phase],
        class_probabilities=dict(probabilities),
        phase_scores=phase_scores,
    )


def majority_vote(
    predictions: Sequence[FramePrediction],
    *,
    minimum_phase_confidence: float = 0.0,
    minimum_vote_fraction: float = 0.6,
) -> VoteResult:
    """Vote over frames, breaking ties with the mean phase score."""

    if not predictions:
        raise ValueError("Cannot vote over an empty prediction batch")
    invalid = [item.phase for item in predictions if item.phase not in PHASES]
    if invalid:
        raise ValueError(f"Invalid predicted phase: {invalid[0]}")

    votes = Counter(item.phase for item in predictions)
    max_votes = max(votes.values())
    tied = [phase for phase in PHASES if votes[phase] == max_votes]
    mean_scores = {
        phase: sum(float(item.phase_scores.get(phase, 0.0)) for item in predictions)
        / len(predictions)
        for phase in PHASES
    }
    winner = max(tied, key=lambda phase: mean_scores[phase])
    vote_fraction = votes[winner] / len(predictions)
    winner_predictions = [item for item in predictions if item.phase == winner]
    confidence = sum(item.confidence for item in winner_predictions) / len(
        winner_predictions
    )

    uncertain_reason = None
    phase = winner
    if vote_fraction < minimum_vote_fraction:
        phase = "UNKNOWN"
        uncertain_reason = "vote_fraction"
    elif confidence < minimum_phase_confidence:
        phase = "UNKNOWN"
        uncertain_reason = "phase_confidence"

    return VoteResult(
        phase=phase,
        confidence=confidence,
        vote_fraction=vote_fraction,
        votes={name: votes[name] for name in PHASES},
        mean_phase_scores=mean_scores,
        uncertain_reason=uncertain_reason,
    )

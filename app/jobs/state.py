"""Job state machine (Phase 11).

Strongly typed lifecycle with validated transitions. No arbitrary invalid
state transitions are allowed — every transition is checked against an explicit
allowed-successor map.

Lifecycle:
    DRAFT
    -> PLANNING -> PLANNED
    -> SCENE_RESOLUTION -> SCENES_READY
    -> VIDEO_GENERATION -> VIDEO_GENERATED
    -> AUDIO_GENERATION -> AUDIO_READY
    -> ASSEMBLY -> QUALITY_CONTROL -> COMPLETED

Failure: ANY_STATE -> FAILED
Retry: FAILED -> RETRYING -> PREVIOUS_VALID_STATE (or FAILED again)
Cancel: ANY_ACTIVE_STATE -> CANCEL_REQUESTED -> CANCELED
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..core.errors import TypedErrorCode, VideoError


class JobState(str, Enum):
    DRAFT = "DRAFT"
    PLANNING = "PLANNING"
    PLANNED = "PLANNED"
    SCENE_RESOLUTION = "SCENE_RESOLUTION"
    SCENES_READY = "SCENES_READY"
    VIDEO_GENERATION = "VIDEO_GENERATION"
    VIDEO_GENERATED = "VIDEO_GENERATED"
    AUDIO_GENERATION = "AUDIO_GENERATION"
    AUDIO_READY = "AUDIO_READY"
    ASSEMBLY = "ASSEMBLY"
    QUALITY_CONTROL = "QUALITY_CONTROL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELED = "CANCELED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"


# Valid forward transitions (linear pipeline). Terminal states have no out.
FORWARD: dict[JobState, set[JobState]] = {
    JobState.DRAFT: {JobState.PLANNING, JobState.QUEUED, JobState.CANCEL_REQUESTED, JobState.FAILED},
    JobState.PLANNING: {JobState.PLANNED, JobState.FAILED, JobState.CANCEL_REQUESTED},
    JobState.PLANNED: {JobState.SCENE_RESOLUTION, JobState.QUEUED, JobState.FAILED, JobState.CANCEL_REQUESTED},
    JobState.SCENE_RESOLUTION: {JobState.SCENES_READY, JobState.FAILED, JobState.CANCEL_REQUESTED},
    JobState.SCENES_READY: {JobState.VIDEO_GENERATION, JobState.QUEUED, JobState.FAILED, JobState.CANCEL_REQUESTED},
    JobState.VIDEO_GENERATION: {JobState.VIDEO_GENERATED, JobState.FAILED, JobState.CANCEL_REQUESTED},
    JobState.VIDEO_GENERATED: {JobState.AUDIO_GENERATION, JobState.QUEUED, JobState.ASSEMBLY, JobState.FAILED, JobState.CANCEL_REQUESTED},
    JobState.AUDIO_GENERATION: {JobState.AUDIO_READY, JobState.FAILED, JobState.CANCEL_REQUESTED},
    JobState.AUDIO_READY: {JobState.ASSEMBLY, JobState.QUEUED, JobState.FAILED, JobState.CANCEL_REQUESTED},
    JobState.ASSEMBLY: {JobState.QUALITY_CONTROL, JobState.FAILED, JobState.CANCEL_REQUESTED},
    JobState.QUALITY_CONTROL: {JobState.COMPLETED, JobState.FAILED},
    JobState.COMPLETED: set(),
    JobState.FAILED: {JobState.RETRYING, JobState.CANCELED},
    JobState.RETRYING: {JobState.QUEUED, JobState.FAILED},
    JobState.QUEUED: {JobState.RUNNING, JobState.FAILED, JobState.CANCEL_REQUESTED},
    JobState.RUNNING: {JobState.COMPLETED, JobState.FAILED, JobState.CANCEL_REQUESTED},
    JobState.CANCEL_REQUESTED: {JobState.CANCELED, JobState.FAILED},
    JobState.CANCELED: set(),
}

# States considered terminal (no further transitions except via retry).
TERMINAL = {JobState.COMPLETED, JobState.FAILED, JobState.CANCELED}

# States considered active (in-progress, eligible for cancellation/recovery).
ACTIVE = {
    JobState.PLANNING, JobState.SCENE_RESOLUTION, JobState.VIDEO_GENERATION,
    JobState.AUDIO_GENERATION, JobState.ASSEMBLY, JobState.QUALITY_CONTROL,
    JobState.QUEUED, JobState.RUNNING, JobState.RETRYING, JobState.CANCEL_REQUESTED,
}

# Valid "previous valid" states a RETRYING job can return to.
RETRY_TARGETS = {
    JobState.DRAFT, JobState.PLANNING, JobState.PLANNED, JobState.SCENE_RESOLUTION,
    JobState.SCENES_READY, JobState.VIDEO_GENERATION, JobState.VIDEO_GENERATED,
    JobState.AUDIO_GENERATION, JobState.AUDIO_READY, JobState.ASSEMBLY,
    JobState.QUALITY_CONTROL, JobState.QUEUED, JobState.RUNNING,
}


def is_valid_transition(from_state: JobState, to_state: JobState) -> bool:
    """True if transitioning from_state -> to_state is allowed."""
    if from_state == to_state:
        return True  # idempotent / no-op
    return to_state in FORWARD.get(from_state, set())


def assert_transition(from_state: JobState, to_state: JobState) -> None:
    """Raise INVALID_TRANSITION if the transition is not allowed."""
    if not is_valid_transition(from_state, to_state):
        raise VideoError(
            TypedErrorCode.INVALID_TRANSITION,
            f"Invalid job state transition: {from_state.value} -> {to_state.value}.",
            context={"from": from_state.value, "to": to_state.value,
                     "allowed": [s.value for s in FORWARD.get(from_state, set())]},
        )


def can_retry(state: JobState) -> bool:
    return state is JobState.FAILED


def can_cancel(state: JobState) -> bool:
    return state in ACTIVE and state not in (JobState.CANCEL_REQUESTED, JobState.CANCELED)


def is_terminal(state: JobState) -> bool:
    return state in TERMINAL


def is_active(state: JobState) -> bool:
    return state in ACTIVE

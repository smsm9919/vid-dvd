"""Job identity, types, and the Job record model (Phase 11)."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from ..core.errors import TypedErrorCode, VideoError
from .state import JobState


class JobType(str, Enum):
    CONTENT_PLAN = "CONTENT_PLAN"
    AD_VARIANTS = "AD_VARIANTS"
    SCENE_RESOLUTION = "SCENE_RESOLUTION"
    VIDEO_SCENE = "VIDEO_SCENE"
    VOICEOVER = "VOICEOVER"
    MUSIC = "MUSIC"
    SFX = "SFX"
    CAPTIONS = "CAPTIONS"
    ASSEMBLY = "ASSEMBLY"
    FINAL_QC = "FINAL_QC"


# Typed error codes that are RETRYABLE (transient).
RETRYABLE_CODES = {
    TypedErrorCode.COMFYUI_UNREACHABLE.value,
    TypedErrorCode.GENERATION_TIMEOUT.value,
    TypedErrorCode.FFMPEG_ERROR.value,
    TypedErrorCode.CONCURRENCY_LIMIT.value,
}

# Typed error codes that are NON-RETRYABLE (deterministic).
NON_RETRYABLE_CODES = {
    TypedErrorCode.WORKFLOW_INVALID.value,
    TypedErrorCode.WORKFLOW_REJECTED.value,
    TypedErrorCode.WORKFLOW_NOT_FOUND.value,
    TypedErrorCode.MODEL_NOT_FOUND.value,
    TypedErrorCode.INVALID_REFERENCE.value,
    TypedErrorCode.INVALID_MP4.value,
    TypedErrorCode.INVALID_AUDIO.value,
    TypedErrorCode.INVALID_TIMELINE.value,
    TypedErrorCode.UNSUPPORTED_PROFILE.value,
    TypedErrorCode.TRANSITION_ERROR.value,
    TypedErrorCode.CAPTION_RENDER_ERROR.value,
    TypedErrorCode.NON_RETRYABLE.value,
    TypedErrorCode.INVALID_TRANSITION.value,
    TypedErrorCode.DEPENDENCY_FAILED.value,
    TypedErrorCode.DEPENDENCY_MISSING.value,
    TypedErrorCode.JOB_NOT_FOUND.value,
}


def is_retryable(code: Optional[str]) -> bool:
    if code is None:
        return False
    if code in NON_RETRYABLE_CODES:
        return False
    if code in RETRYABLE_CODES:
        return True
    # Unknown codes default to non-retryable (fail safe).
    return False


def is_non_retryable(code: Optional[str]) -> bool:
    if code is None:
        return False
    return code in NON_RETRYABLE_CODES


@dataclass
class Job:
    """A single orchestration job record."""

    job_id: str
    project_id: str
    job_type: JobType
    state: JobState = JobState.DRAFT
    parent_job_id: Optional[str] = None
    scene_index: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    provider: Optional[str] = None
    model: Optional[str] = None
    workflow: Optional[str] = None
    seed: Optional[int] = None
    input_fingerprint: Optional[str] = None
    output_fingerprint: Optional[str] = None
    output_path: Optional[str] = None
    error_code: Optional[str] = None
    error_detail: Optional[str] = None
    cancel_requested: bool = False
    provider_canceled: Optional[bool] = None  # None = unknown / not applicable
    # Free-form structured inputs/outputs (for re-execution + provenance).
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    # Dependency job_ids that must be COMPLETED before this job can run.
    depends_on: list[str] = field(default_factory=list)
    # Last backoff delay used (seconds), for observability.
    last_backoff: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["job_type"] = self.job_type.value
        d["state"] = self.state.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Job":
        d = dict(d)
        d["job_type"] = JobType(d["job_type"])
        d["state"] = JobState(d["state"])
        return cls(**d)

    @property
    def duration_s(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


def new_job_id() -> str:
    return "job_" + uuid.uuid4().hex[:16]


def new_project_id() -> str:
    return "proj_" + uuid.uuid4().hex[:12]


def fingerprint(inputs: dict[str, Any]) -> str:
    """Stable SHA-256 fingerprint of normalized job inputs.

    Used for idempotency: identical normalized inputs => identical fingerprint.
    Keys are sorted; values are JSON-serialized with sort_keys for determinism.
    """
    blob = json.dumps(inputs, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def file_fingerprint(path: Path) -> Optional[str]:
    """A fingerprint of a file's content (size + sha256) for cache validation."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    size = p.stat().st_size
    h.update(str(size).encode())
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:32]


def make_job(
    project_id: str,
    job_type: JobType,
    *,
    parent_job_id: Optional[str] = None,
    scene_index: Optional[int] = None,
    inputs: Optional[dict[str, Any]] = None,
    depends_on: Optional[list[str]] = None,
    max_retries: int = 3,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    workflow: Optional[str] = None,
    seed: Optional[int] = None,
) -> Job:
    """Construct a new Job with stable identity + input fingerprint."""
    jid = new_job_id()
    inputs = inputs or {}
    return Job(
        job_id=jid, project_id=project_id, job_type=job_type, state=JobState.DRAFT,
        parent_job_id=parent_job_id, scene_index=scene_index,
        inputs=inputs, depends_on=depends_on or [], max_retries=max_retries,
        provider=provider, model=model, workflow=workflow, seed=seed,
        input_fingerprint=fingerprint(inputs),
    )

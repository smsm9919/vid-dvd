"""Dependency graph, progress, and concurrency control (Phase 11).

Explicit dependencies: a job is only eligible to run when all its
`depends_on` jobs are COMPLETED. A failed dependency blocks downstream jobs
with a typed DEPENDENCY_FAILED error. Progress is derived from actual job
states — never fake percentages. Concurrency is bounded for scene-level jobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..core.errors import TypedErrorCode, VideoError
from .models import Job, JobType
from .state import JobState, is_terminal


@dataclass
class DependencyStatus:
    ready: bool
    failed_deps: list[str] = None  # type: ignore[assignment]
    pending_deps: list[str] = None  # type: ignore[assignment]
    missing_deps: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.failed_deps is None:
            self.failed_deps = []
        if self.pending_deps is None:
            self.pending_deps = []
        if self.missing_deps is None:
            self.missing_deps = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "failed_deps": self.failed_deps,
            "pending_deps": self.pending_deps,
            "missing_deps": self.missing_deps,
        }


def check_dependencies(job: Job, all_jobs: dict[str, Job]) -> DependencyStatus:
    """Check whether a job's dependencies are satisfied.

    A dependency is satisfied only if it is COMPLETED. A FAILED dependency
    blocks the job (DEPENDENCY_FAILED). A non-terminal dependency is pending.
    A missing dependency id is a DEPENDENCY_MISSING error.
    """
    status = DependencyStatus(ready=True)
    for dep_id in job.depends_on:
        dep = all_jobs.get(dep_id)
        if dep is None:
            status.missing_deps.append(dep_id)
            status.ready = False
            continue
        if dep.state is JobState.FAILED:
            status.failed_deps.append(dep_id)
            status.ready = False
        elif dep.state is not JobState.COMPLETED:
            status.pending_deps.append(dep_id)
            status.ready = False
    return status


def assert_dependencies(job: Job, all_jobs: dict[str, Job]) -> None:
    """Raise a typed error if dependencies are not satisfied."""
    status = check_dependencies(job, all_jobs)
    if status.missing_deps:
        raise VideoError(
            TypedErrorCode.DEPENDENCY_MISSING,
            f"Job {job.job_id} has missing dependencies: {status.missing_deps}.",
            context={"job_id": job.job_id, "missing_deps": status.missing_deps,
                     "depends_on": job.depends_on})
    if status.failed_deps:
        raise VideoError(
            TypedErrorCode.DEPENDENCY_FAILED,
            f"Job {job.job_id} cannot proceed: dependencies failed: {status.failed_deps}.",
            context={"job_id": job.job_id, "failed_deps": status.failed_deps,
                     "depends_on": job.depends_on})
    if status.pending_deps:
        raise VideoError(
            TypedErrorCode.DEPENDENCY_MISSING,
            f"Job {job.job_id} pending dependencies: {status.pending_deps}.",
            context={"job_id": job.job_id, "pending_deps": status.pending_deps})


# ---------------------------------------------------------------- progress
@dataclass
class Progress:
    overall_progress: int  # 0-100, derived from actual job states
    stage: str
    scene: Optional[str]  # e.g. "3 / 5" or None
    current_job: Optional[str]
    total_jobs: int
    completed_jobs: int
    failed_jobs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_progress": self.overall_progress,
            "stage": self.stage,
            "scene": self.scene,
            "current_job": self.current_job,
            "total_jobs": self.total_jobs,
            "completed_jobs": self.completed_jobs,
            "failed_jobs": self.failed_jobs,
        }


def compute_progress(jobs: list[Job]) -> Progress:
    """Derive progress from actual job states — no fake percentages.

    overall_progress = (completed_jobs / total_jobs) * 100, floored.
    stage = the highest-priority in-progress or next-pending job's stage.
    """
    if not jobs:
        return Progress(0, "IDLE", None, None, 0, 0, 0)
    total = len(jobs)
    completed = sum(1 for j in jobs if j.state is JobState.COMPLETED)
    failed = sum(1 for j in jobs if j.state is JobState.FAILED)
    pct = int((completed / total) * 100) if total else 0
    # Determine current stage from the first non-terminal job in pipeline order.
    order = [
        JobType.CONTENT_PLAN, JobType.AD_VARIANTS, JobType.SCENE_RESOLUTION,
        JobType.VIDEO_SCENE, JobType.VOICEOVER, JobType.MUSIC, JobType.SFX,
        JobType.CAPTIONS, JobType.ASSEMBLY, JobType.FINAL_QC,
    ]
    stage = "COMPLETED" if completed == total else ("FAILED" if failed else "IDLE")
    current = None
    for jt in order:
        for j in jobs:
            if j.job_type is jt and not is_terminal(j.state):
                stage = j.state.value
                current = j.job_id
                break
        if current:
            break
    # Scene progress for VIDEO_SCENE jobs.
    scene_jobs = [j for j in jobs if j.job_type is JobType.VIDEO_SCENE]
    scene_str = None
    if scene_jobs:
        done = sum(1 for j in scene_jobs if j.state is JobState.COMPLETED)
        scene_str = f"{done} / {len(scene_jobs)}"
    return Progress(pct, stage, scene_str, current, total, completed, failed)


# ---------------------------------------------------------------- concurrency
class ConcurrencyLimiter:
    """Bounded concurrency for independent scene jobs.

    Configurable via MAX_CONCURRENT_VIDEO_JOBS / MAX_CONCURRENT_AUDIO_JOBS.
    Defaults are conservative to avoid exhausting GPU/CPU/RAM.
    """

    DEFAULT_VIDEO = 2
    DEFAULT_AUDIO = 4

    def __init__(self, max_video: int = DEFAULT_VIDEO, max_audio: int = DEFAULT_AUDIO) -> None:
        self.max_video = max(1, max_video)
        self.max_audio = max(1, max_audio)
        self._running_video = 0
        self._running_audio = 0

    def try_acquire(self, job_type: JobType) -> bool:
        """Try to acquire a slot. Returns True if acquired, else CONCURRENCY_LIMIT."""
        if job_type is JobType.VIDEO_SCENE:
            if self._running_video < self.max_video:
                self._running_video += 1
                return True
            return False
        if job_type in (JobType.VOICEOVER, JobType.MUSIC, JobType.SFX):
            if self._running_audio < self.max_audio:
                self._running_audio += 1
                return True
            return False
        # Non-scene jobs: always allowed (single instance).
        return True

    def release(self, job_type: JobType) -> None:
        if job_type is JobType.VIDEO_SCENE:
            self._running_video = max(0, self._running_video - 1)
        elif job_type in (JobType.VOICEOVER, JobType.MUSIC, JobType.SFX):
            self._running_audio = max(0, self._running_audio - 1)

    @property
    def running_video(self) -> int:
        return self._running_video

    @property
    def running_audio(self) -> int:
        return self._running_audio

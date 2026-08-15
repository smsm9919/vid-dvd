"""Comprehensive tests for Phase 11: Production Job Orchestration.

Covers: valid/invalid state transitions, dependency handling, job creation,
idempotency, fingerprints, cache hits/invalid, retries, retry limits,
retryable/non-retryable errors, exponential backoff, progress, concurrency
limits, cancellation, persistence, restart recovery, failure propagation,
output validation, provider unavailable, completed-job resume, serialization.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from app.brain.models import ContentBrief
from app.core.errors import TypedErrorCode, VideoError
from app.jobs.cache import CacheStore
from app.jobs.dependencies import (
    ConcurrencyLimiter, assert_dependencies, check_dependencies, compute_progress,
)
from app.jobs.models import (
    Job, JobType, file_fingerprint, fingerprint, is_non_retryable, is_retryable,
    make_job, new_job_id, new_project_id,
)
from app.jobs.orchestrator import Orchestrator
from app.jobs.persistence import JobStore, ProjectRegistry
from app.jobs.retry import (
    DEFAULT_POLICY, RetryPolicy, classify_error, next_delay, record_retry,
    should_retry,
)
from app.jobs.state import (
    JobState, FORWARD, TERMINAL, ACTIVE, assert_transition, can_cancel,
    can_retry, is_active, is_terminal, is_valid_transition,
)
from app.media import verify_mp4


# --------------------------------------------------------------- fixtures
@pytest.fixture()
def tmp_root():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def orch(tmp_root):
    return Orchestrator(root=tmp_root)


def _make_scene_video(path: Path, duration: float = 3.0, color: str = "red"):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=1080x1920:d={duration}:r=30",
         "-t", str(duration), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    verify_mp4(path)
    return path


# ===============================================================
# 1. STATE MACHINE
# ===============================================================
class TestStateMachine:
    def test_valid_forward_transitions(self):
        assert is_valid_transition(JobState.DRAFT, JobState.PLANNING)
        assert is_valid_transition(JobState.PLANNING, JobState.PLANNED)
        assert is_valid_transition(JobState.PLANNED, JobState.SCENE_RESOLUTION)
        assert is_valid_transition(JobState.SCENE_RESOLUTION, JobState.SCENES_READY)
        assert is_valid_transition(JobState.SCENES_READY, JobState.VIDEO_GENERATION)
        assert is_valid_transition(JobState.VIDEO_GENERATION, JobState.VIDEO_GENERATED)
        assert is_valid_transition(JobState.VIDEO_GENERATED, JobState.AUDIO_GENERATION)
        assert is_valid_transition(JobState.AUDIO_GENERATION, JobState.AUDIO_READY)
        assert is_valid_transition(JobState.AUDIO_READY, JobState.ASSEMBLY)
        assert is_valid_transition(JobState.ASSEMBLY, JobState.QUALITY_CONTROL)
        assert is_valid_transition(JobState.QUALITY_CONTROL, JobState.COMPLETED)

    def test_invalid_transitions_rejected(self):
        assert not is_valid_transition(JobState.COMPLETED, JobState.PLANNING)
        assert not is_valid_transition(JobState.COMPLETED, JobState.RUNNING)
        assert not is_valid_transition(JobState.CANCELED, JobState.COMPLETED)
        assert not is_valid_transition(JobState.DRAFT, JobState.COMPLETED)
        assert not is_valid_transition(JobState.DRAFT, JobState.QUALITY_CONTROL)

    def test_assert_transition_raises_typed_error(self):
        with pytest.raises(VideoError) as ei:
            assert_transition(JobState.COMPLETED, JobState.RUNNING)
        assert ei.value.code is TypedErrorCode.INVALID_TRANSITION
        assert "from" in ei.value.context

    def test_any_state_can_fail(self):
        for s in JobState:
            if s in (JobState.COMPLETED, JobState.CANCELED):
                continue
            assert is_valid_transition(s, JobState.FAILED), f"{s} -> FAILED should be allowed"

    def test_failed_can_retry(self):
        assert is_valid_transition(JobState.FAILED, JobState.RETRYING)
        assert is_valid_transition(JobState.RETRYING, JobState.QUEUED)

    def test_cancel_path(self):
        assert is_valid_transition(JobState.RUNNING, JobState.CANCEL_REQUESTED)
        assert is_valid_transition(JobState.CANCEL_REQUESTED, JobState.CANCELED)

    def test_terminal_states_have_no_forward(self):
        assert FORWARD[JobState.COMPLETED] == set()
        assert FORWARD[JobState.CANCELED] == set()

    def test_is_terminal(self):
        assert is_terminal(JobState.COMPLETED)
        assert is_terminal(JobState.FAILED)
        assert is_terminal(JobState.CANCELED)
        assert not is_terminal(JobState.RUNNING)
        assert not is_terminal(JobState.QUEUED)

    def test_is_active(self):
        assert is_active(JobState.RUNNING)
        assert is_active(JobState.QUEUED)
        assert not is_active(JobState.COMPLETED)
        assert not is_active(JobState.FAILED)

    def test_can_retry_only_from_failed(self):
        assert can_retry(JobState.FAILED)
        assert not can_retry(JobState.COMPLETED)
        assert not can_retry(JobState.RUNNING)

    def test_can_cancel_only_active(self):
        assert can_cancel(JobState.RUNNING)
        assert can_cancel(JobState.QUEUED)
        assert not can_cancel(JobState.COMPLETED)
        assert not can_cancel(JobState.FAILED)
        assert not can_cancel(JobState.CANCELED)

    def test_idempotent_same_state_transition(self):
        assert is_valid_transition(JobState.RUNNING, JobState.RUNNING)


# ===============================================================
# 2. JOB IDENTITY + TYPES
# ===============================================================
class TestJobIdentity:
    def test_job_types_complete(self):
        expected = {"CONTENT_PLAN", "AD_VARIANTS", "SCENE_RESOLUTION", "VIDEO_SCENE",
                    "VOICEOVER", "MUSIC", "SFX", "CAPTIONS", "ASSEMBLY", "FINAL_QC"}
        assert {t.value for t in JobType} == expected

    def test_new_job_id_unique(self):
        ids = {new_job_id() for _ in range(100)}
        assert len(ids) == 100

    def test_new_project_id_unique(self):
        pids = {new_project_id() for _ in range(100)}
        assert len(pids) == 100

    def test_make_job_has_fingerprint(self):
        job = make_job("p1", JobType.CONTENT_PLAN,
                       inputs={"brief": {"idea": "lion", "duration_seconds": 12}})
        assert job.input_fingerprint is not None
        assert len(job.input_fingerprint) == 32
        assert job.state is JobState.DRAFT
        assert job.job_id.startswith("job_")

    def test_job_serialization_roundtrip(self):
        job = make_job("p1", JobType.VIDEO_SCENE, scene_index=2,
                       inputs={"prompt": "cat"}, depends_on=["job_a", "job_b"],
                       provider="wan", model="wan2.2", seed=42)
        d = job.to_dict()
        assert d["job_type"] == "VIDEO_SCENE"
        assert d["state"] == "DRAFT"
        assert d["scene_index"] == 2
        assert d["seed"] == 42
        job2 = Job.from_dict(d)
        assert job2.job_type is JobType.VIDEO_SCENE
        assert job2.state is JobState.DRAFT
        assert job2.depends_on == ["job_a", "job_b"]
        assert job2.input_fingerprint == job.input_fingerprint


# ===============================================================
# 3. FINGERPRINTS
# ===============================================================
class TestFingerprints:
    def test_fingerprint_order_stable(self):
        f1 = fingerprint({"a": 1, "b": 2})
        f2 = fingerprint({"b": 2, "a": 1})
        assert f1 == f2

    def test_fingerprint_different_inputs(self):
        f1 = fingerprint({"a": 1})
        f2 = fingerprint({"a": 2})
        assert f1 != f2

    def test_fingerprint_deterministic(self):
        inputs = {"prompt": "lion", "seed": 42, "duration": 4.0}
        assert fingerprint(inputs) == fingerprint(inputs)

    def test_file_fingerprint_missing_file(self):
        assert file_fingerprint(Path("/nonexistent")) is None

    def test_file_fingerprint_stable(self, tmp_path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"hello world")
        fp1 = file_fingerprint(p)
        fp2 = file_fingerprint(p)
        assert fp1 == fp2
        assert len(fp1) == 32

    def test_file_fingerprint_changes_with_content(self, tmp_path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"hello")
        fp1 = file_fingerprint(p)
        p.write_bytes(b"hello world")
        fp2 = file_fingerprint(p)
        assert fp1 != fp2


# ===============================================================
# 4. RETRY POLICY + BACKOFF
# ===============================================================
class TestRetryPolicy:
    def test_retryable_codes(self):
        assert is_retryable(TypedErrorCode.COMFYUI_UNREACHABLE.value)
        assert is_retryable(TypedErrorCode.GENERATION_TIMEOUT.value)
        assert is_retryable(TypedErrorCode.FFMPEG_ERROR.value)

    def test_non_retryable_codes(self):
        assert is_non_retryable(TypedErrorCode.MODEL_NOT_FOUND.value)
        assert is_non_retryable(TypedErrorCode.WORKFLOW_INVALID.value)
        assert is_non_retryable(TypedErrorCode.INVALID_REFERENCE.value)
        assert is_non_retryable(TypedErrorCode.UNSUPPORTED_PROFILE.value)

    def test_unknown_code_defaults_non_retryable(self):
        assert not is_retryable("UNKNOWN_CODE")
        assert not is_non_retryable("UNKNOWN_CODE")

    def test_classify_error(self):
        assert classify_error("COMFYUI_UNREACHABLE") == "RETRYABLE"
        assert classify_error("MODEL_NOT_FOUND") == "NON_RETRYABLE"
        assert classify_error("UNKNOWN") == "UNKNOWN"
        assert classify_error(None) == "UNKNOWN"

    def test_backoff_is_bounded(self):
        policy = RetryPolicy(base_delay=1.0, max_delay=30.0)
        for attempt in range(10):
            delay = policy.compute_delay(attempt)
            assert 0 <= delay <= 30.0 + 30.0 * 0.2  # max + jitter

    def test_backoff_grows_exponentially(self):
        policy = RetryPolicy(base_delay=1.0, max_delay=100.0, jitter=0.0)
        assert policy.compute_delay(0) == 1.0
        assert policy.compute_delay(1) == 2.0
        assert policy.compute_delay(2) == 4.0
        assert policy.compute_delay(3) == 8.0

    def test_should_retry_retryable_with_retries_left(self):
        job = make_job("p", JobType.VIDEO_SCENE, inputs={"x": 1})
        job.error_code = TypedErrorCode.COMFYUI_UNREACHABLE.value
        job.retry_count = 0
        assert should_retry(job)

    def test_should_retry_exhausted(self):
        job = make_job("p", JobType.VIDEO_SCENE, inputs={"x": 1})
        job.error_code = TypedErrorCode.COMFYUI_UNREACHABLE.value
        job.retry_count = 3
        job.max_retries = 3
        assert not should_retry(job)

    def test_should_not_retry_non_retryable(self):
        job = make_job("p", JobType.VIDEO_SCENE, inputs={"x": 1})
        job.error_code = TypedErrorCode.MODEL_NOT_FOUND.value
        assert not should_retry(job)

    def test_record_retry_increments_count(self):
        job = make_job("p", JobType.VIDEO_SCENE, inputs={"x": 1})
        job.error_code = TypedErrorCode.COMFYUI_UNREACHABLE.value
        delay = record_retry(job, RetryPolicy(base_delay=1.0, jitter=0.0))
        assert job.retry_count == 1
        assert delay == 1.0
        assert job.last_backoff == 1.0

    def test_record_retry_raises_for_non_retryable(self):
        job = make_job("p", JobType.VIDEO_SCENE, inputs={"x": 1})
        job.error_code = TypedErrorCode.MODEL_NOT_FOUND.value
        with pytest.raises(VideoError) as ei:
            record_retry(job)
        assert ei.value.code is TypedErrorCode.NON_RETRYABLE

    def test_record_retry_raises_when_exhausted(self):
        job = make_job("p", JobType.VIDEO_SCENE, inputs={"x": 1})
        job.error_code = TypedErrorCode.COMFYUI_UNREACHABLE.value
        job.retry_count = 3
        job.max_retries = 3
        with pytest.raises(VideoError) as ei:
            record_retry(job)
        assert ei.value.code is TypedErrorCode.RETRY_EXHAUSTED

    def test_next_delay_returns_bounded_value(self):
        job = make_job("p", JobType.VIDEO_SCENE, inputs={"x": 1})
        job.error_code = TypedErrorCode.FFMPEG_ERROR.value
        job.retry_count = 2
        delay = next_delay(job, RetryPolicy(base_delay=1.0, max_delay=30.0, jitter=0.0))
        assert delay == 4.0


# ===============================================================
# 5. DEPENDENCIES
# ===============================================================
class TestDependencies:
    def test_ready_when_all_deps_completed(self):
        j1 = make_job("p", JobType.CONTENT_PLAN, inputs={"a": 1})
        j1.state = JobState.COMPLETED
        j2 = make_job("p", JobType.SCENE_RESOLUTION, inputs={"b": 2}, depends_on=[j1.job_id])
        status = check_dependencies(j2, {j1.job_id: j1})
        assert status.ready

    def test_not_ready_when_dep_pending(self):
        j1 = make_job("p", JobType.CONTENT_PLAN, inputs={"a": 1})
        j1.state = JobState.RUNNING
        j2 = make_job("p", JobType.SCENE_RESOLUTION, inputs={"b": 2}, depends_on=[j1.job_id])
        status = check_dependencies(j2, {j1.job_id: j1})
        assert not status.ready
        assert j1.job_id in status.pending_deps

    def test_failed_dep_blocks(self):
        j1 = make_job("p", JobType.VIDEO_SCENE, inputs={"a": 1})
        j1.state = JobState.FAILED
        j2 = make_job("p", JobType.ASSEMBLY, inputs={"b": 2}, depends_on=[j1.job_id])
        status = check_dependencies(j2, {j1.job_id: j1})
        assert not status.ready
        assert j1.job_id in status.failed_deps

    def test_missing_dep_blocks(self):
        j2 = make_job("p", JobType.ASSEMBLY, inputs={"b": 2}, depends_on=["nonexistent"])
        status = check_dependencies(j2, {})
        assert not status.ready
        assert "nonexistent" in status.missing_deps

    def test_assert_dependencies_raises_on_failed(self):
        j1 = make_job("p", JobType.VIDEO_SCENE, inputs={"a": 1})
        j1.state = JobState.FAILED
        j2 = make_job("p", JobType.ASSEMBLY, inputs={"b": 2}, depends_on=[j1.job_id])
        with pytest.raises(VideoError) as ei:
            assert_dependencies(j2, {j1.job_id: j1})
        assert ei.value.code is TypedErrorCode.DEPENDENCY_FAILED

    def test_assert_dependencies_raises_on_missing(self):
        j2 = make_job("p", JobType.ASSEMBLY, inputs={"b": 2}, depends_on=["ghost"])
        with pytest.raises(VideoError) as ei:
            assert_dependencies(j2, {})
        assert ei.value.code is TypedErrorCode.DEPENDENCY_MISSING


# ===============================================================
# 6. PROGRESS
# ===============================================================
class TestProgress:
    def test_empty_progress(self):
        p = compute_progress([])
        assert p.overall_progress == 0
        assert p.stage == "IDLE"
        assert p.total_jobs == 0

    def test_progress_from_states(self):
        jobs = [
            make_job("p", JobType.CONTENT_PLAN, inputs={}),
            make_job("p", JobType.SCENE_RESOLUTION, inputs={}),
            make_job("p", JobType.VIDEO_SCENE, inputs={}),
        ]
        jobs[0].state = JobState.COMPLETED
        jobs[1].state = JobState.RUNNING
        p = compute_progress(jobs)
        assert p.total_jobs == 3
        assert p.completed_jobs == 1
        assert p.overall_progress == 33  # 1/3
        assert p.stage == JobState.RUNNING.value

    def test_progress_100_when_all_completed(self):
        jobs = [make_job("p", JobType.CONTENT_PLAN, inputs={})]
        jobs[0].state = JobState.COMPLETED
        p = compute_progress(jobs)
        assert p.overall_progress == 100
        assert p.stage == "COMPLETED"

    def test_progress_shows_failed(self):
        jobs = [make_job("p", JobType.VIDEO_SCENE, inputs={})]
        jobs[0].state = JobState.FAILED
        p = compute_progress(jobs)
        assert p.failed_jobs == 1
        assert p.stage == "FAILED"

    def test_scene_progress(self):
        jobs = [
            make_job("p", JobType.VIDEO_SCENE, scene_index=1, inputs={}),
            make_job("p", JobType.VIDEO_SCENE, scene_index=2, inputs={}),
            make_job("p", JobType.VIDEO_SCENE, scene_index=3, inputs={}),
        ]
        jobs[0].state = JobState.COMPLETED
        jobs[1].state = JobState.COMPLETED
        jobs[2].state = JobState.RUNNING
        p = compute_progress(jobs)
        assert p.scene == "2 / 3"


# ===============================================================
# 7. CONCURRENCY
# ===============================================================
class TestConcurrency:
    def test_video_concurrency_limit(self):
        lim = ConcurrencyLimiter(max_video=2)
        assert lim.try_acquire(JobType.VIDEO_SCENE)
        assert lim.try_acquire(JobType.VIDEO_SCENE)
        assert not lim.try_acquire(JobType.VIDEO_SCENE)
        assert lim.running_video == 2

    def test_release_frees_slot(self):
        lim = ConcurrencyLimiter(max_video=1)
        assert lim.try_acquire(JobType.VIDEO_SCENE)
        lim.release(JobType.VIDEO_SCENE)
        assert lim.try_acquire(JobType.VIDEO_SCENE)

    def test_audio_concurrency_limit(self):
        lim = ConcurrencyLimiter(max_audio=2)
        assert lim.try_acquire(JobType.VOICEOVER)
        assert lim.try_acquire(JobType.MUSIC)
        assert not lim.try_acquire(JobType.SFX)
        lim.release(JobType.VOICEOVER)
        assert lim.try_acquire(JobType.SFX)

    def test_non_scene_jobs_always_allowed(self):
        lim = ConcurrencyLimiter(max_video=1, max_audio=1)
        assert lim.try_acquire(JobType.CONTENT_PLAN)
        assert lim.try_acquire(JobType.ASSEMBLY)

    def test_defaults_conservative(self):
        lim = ConcurrencyLimiter()
        assert lim.max_video == ConcurrencyLimiter.DEFAULT_VIDEO
        assert lim.max_audio == ConcurrencyLimiter.DEFAULT_AUDIO


# ===============================================================
# 8. CACHE + IDEMPOTENCY
# ===============================================================
class TestCache:
    def test_cache_store_and_lookup(self, tmp_path):
        cache = CacheStore(tmp_path / "cache.json")
        p = tmp_path / "out.txt"
        p.write_text("hello")
        cache.store("CONTENT_PLAN", "fp1", p)
        entry = cache.lookup("CONTENT_PLAN", "fp1")
        assert entry is not None
        assert entry.output_path == str(p)

    def test_cache_miss(self, tmp_path):
        cache = CacheStore(tmp_path / "cache.json")
        assert cache.lookup("CONTENT_PLAN", "nonexistent") is None

    def test_cache_invalid_when_file_missing(self, tmp_path):
        cache = CacheStore(tmp_path / "cache.json")
        p = tmp_path / "out.txt"
        p.write_text("hello")
        cache.store("CONTENT_PLAN", "fp1", p)
        p.unlink()
        assert cache.lookup("CONTENT_PLAN", "fp1") is None
        # entry should be evicted
        assert len(cache) == 0

    def test_cache_invalid_when_file_changed(self, tmp_path):
        cache = CacheStore(tmp_path / "cache.json")
        p = tmp_path / "out.txt"
        p.write_text("hello")
        cache.store("CONTENT_PLAN", "fp1", p)
        p.write_text("different content")
        assert cache.lookup("CONTENT_PLAN", "fp1") is None

    def test_cache_invalid_when_empty_file(self, tmp_path):
        cache = CacheStore(tmp_path / "cache.json")
        p = tmp_path / "empty.txt"
        p.write_text("")
        with pytest.raises(VideoError) as ei:
            cache.store("CONTENT_PLAN", "fp1", p)
        assert ei.value.code is TypedErrorCode.CACHE_INVALID

    def test_cache_video_validation(self, tmp_path):
        cache = CacheStore(tmp_path / "cache.json")
        mp4 = tmp_path / "video.mp4"
        _make_scene_video(mp4, duration=2.0)
        cache.store("VIDEO_SCENE", "fp1", mp4, validate="video")
        entry = cache.lookup("VIDEO_SCENE", "fp1", validate="video")
        assert entry is not None

    def test_cache_video_invalid_corrupt(self, tmp_path):
        cache = CacheStore(tmp_path / "cache.json")
        mp4 = tmp_path / "corrupt.mp4"
        mp4.write_bytes(b"not a real mp4")
        with pytest.raises(VideoError):
            cache.store("VIDEO_SCENE", "fp1", mp4, validate="video")

    def test_cache_invalidate(self, tmp_path):
        cache = CacheStore(tmp_path / "cache.json")
        p = tmp_path / "out.txt"
        p.write_text("hello")
        cache.store("CONTENT_PLAN", "fp1", p)
        assert cache.invalidate("CONTENT_PLAN", "fp1")
        assert cache.lookup("CONTENT_PLAN", "fp1") is None
        assert not cache.invalidate("CONTENT_PLAN", "fp1")

    def test_cache_persists_to_disk(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        cache = CacheStore(cache_path)
        p = tmp_path / "out.txt"
        p.write_text("hello")
        cache.store("CONTENT_PLAN", "fp1", p)
        # Reload from disk
        cache2 = CacheStore(cache_path)
        assert cache2.lookup("CONTENT_PLAN", "fp1") is not None


# ===============================================================
# 9. PERSISTENCE
# ===============================================================
class TestPersistence:
    def test_job_store_put_get(self, tmp_path):
        store = JobStore("proj1", root=tmp_path)
        job = make_job("proj1", JobType.CONTENT_PLAN, inputs={"a": 1})
        store.put(job)
        assert store.get(job.job_id) is not None
        assert store.require(job.job_id).job_type is JobType.CONTENT_PLAN

    def test_job_store_require_not_found(self, tmp_path):
        store = JobStore("proj1", root=tmp_path)
        with pytest.raises(VideoError) as ei:
            store.require("nonexistent")
        assert ei.value.code is TypedErrorCode.JOB_NOT_FOUND

    def test_job_store_by_type(self, tmp_path):
        store = JobStore("proj1", root=tmp_path)
        store.put(make_job("proj1", JobType.CONTENT_PLAN, inputs={}))
        store.put(make_job("proj1", JobType.VIDEO_SCENE, inputs={}))
        store.put(make_job("proj1", JobType.VIDEO_SCENE, inputs={}))
        assert len(store.by_type("VIDEO_SCENE")) == 2
        assert len(store.by_type("CONTENT_PLAN")) == 1

    def test_job_store_by_scene(self, tmp_path):
        store = JobStore("proj1", root=tmp_path)
        store.put(make_job("proj1", JobType.VIDEO_SCENE, scene_index=3, inputs={}))
        store.put(make_job("proj1", JobType.VIDEO_SCENE, scene_index=5, inputs={}))
        assert len(store.by_scene(3)) == 1

    def test_job_store_persists_across_instances(self, tmp_path):
        store1 = JobStore("proj1", root=tmp_path)
        job = make_job("proj1", JobType.CONTENT_PLAN, inputs={"a": 1})
        store1.put(job)
        # New instance loads from disk
        store2 = JobStore("proj1", root=tmp_path)
        assert store2.get(job.job_id) is not None
        assert store2.get(job.job_id).inputs == {"a": 1}

    def test_job_store_remove(self, tmp_path):
        store = JobStore("proj1", root=tmp_path)
        job = make_job("proj1", JobType.CONTENT_PLAN, inputs={})
        store.put(job)
        assert store.remove(job.job_id)
        assert store.get(job.job_id) is None

    def test_project_registry_lists(self, tmp_path):
        (tmp_path / "p1").mkdir()
        (tmp_path / "p1" / "jobs.json").write_text("{}")
        (tmp_path / "p2").mkdir()
        (tmp_path / "p2" / "jobs.json").write_text("{}")
        reg = ProjectRegistry(root=tmp_path)
        assert set(reg.list_projects()) == {"p1", "p2"}

    def test_corrupt_jobs_json_starts_fresh(self, tmp_path):
        d = tmp_path / "proj1"
        d.mkdir()
        (d / "jobs.json").write_text("not valid json{{{")
        store = JobStore("proj1", root=tmp_path)
        assert len(store) == 0  # does not crash


# ===============================================================
# 10. ORCHESTRATOR — PROJECT + JOB MANAGEMENT
# ===============================================================
class TestOrchestratorManagement:
    def test_create_project(self, orch):
        proj = orch.create_project()
        assert proj.project_id.startswith("proj_")
        assert proj.assets_dir.exists()

    def test_get_project_rehydrates(self, orch, tmp_root):
        proj = orch.create_project("rehydrate_test")
        # Drop from memory; get_project should rehydrate from disk.
        orch.projects.pop("rehydrate_test")
        proj2 = orch.get_project("rehydrate_test")
        assert proj2.project_id == "rehydrate_test"

    def test_get_project_not_found(self, orch):
        with pytest.raises(VideoError) as ei:
            orch.get_project("nonexistent")
        assert ei.value.code is TypedErrorCode.JOB_NOT_FOUND

    def test_create_job(self, orch):
        orch.create_project("p1")
        job = orch.create_job("p1", JobType.CONTENT_PLAN, inputs={"a": 1})
        assert job.job_id.startswith("job_")
        assert orch.get_job("p1", job.job_id).job_type is JobType.CONTENT_PLAN

    def test_list_jobs(self, orch):
        orch.create_project("p1")
        orch.create_job("p1", JobType.CONTENT_PLAN, inputs={})
        orch.create_job("p1", JobType.VIDEO_SCENE, inputs={})
        assert len(orch.list_jobs("p1")) == 2

    def test_transition_valid(self, orch):
        orch.create_project("p1")
        job = orch.create_job("p1", JobType.CONTENT_PLAN, inputs={})
        orch.transition("p1", job.job_id, JobState.QUEUED)
        orch.transition("p1", job.job_id, JobState.RUNNING)
        assert orch.get_job("p1", job.job_id).state is JobState.RUNNING
        assert orch.get_job("p1", job.job_id).started_at is not None

    def test_transition_invalid_raises(self, orch):
        orch.create_project("p1")
        job = orch.create_job("p1", JobType.CONTENT_PLAN, inputs={})
        with pytest.raises(VideoError) as ei:
            orch.transition("p1", job.job_id, JobState.COMPLETED)
        assert ei.value.code is TypedErrorCode.INVALID_TRANSITION

    def test_cancel_job(self, orch):
        orch.create_project("p1")
        job = orch.create_job("p1", JobType.CONTENT_PLAN, inputs={})
        orch.transition("p1", job.job_id, JobState.QUEUED)
        canceled = orch.cancel_job("p1", job.job_id)
        assert canceled.state is JobState.CANCELED
        assert canceled.cancel_requested is True

    def test_cancel_terminal_raises(self, orch):
        orch.create_project("p1")
        job = orch.create_job("p1", JobType.CONTENT_PLAN, inputs={})
        orch.transition("p1", job.job_id, JobState.QUEUED)
        orch.transition("p1", job.job_id, JobState.RUNNING)
        orch.transition("p1", job.job_id, JobState.COMPLETED)
        with pytest.raises(VideoError) as ei:
            orch.cancel_job("p1", job.job_id)
        assert ei.value.code is TypedErrorCode.CANCEL_NOT_ALLOWED

    def test_retry_job(self, orch):
        orch.create_project("p1")
        job = orch.create_job("p1", JobType.VIDEO_SCENE, inputs={})
        orch.transition("p1", job.job_id, JobState.QUEUED)
        orch.transition("p1", job.job_id, JobState.RUNNING)
        orch.transition("p1", job.job_id, JobState.FAILED, error_code="COMFYUI_UNREACHABLE",
                        error_detail="temp")
        retried = orch.retry_job("p1", job.job_id)
        assert retried.state is JobState.QUEUED
        assert retried.retry_count == 1

    def test_retry_non_retryable_raises(self, orch):
        orch.create_project("p1")
        job = orch.create_job("p1", JobType.VIDEO_SCENE, inputs={})
        orch.transition("p1", job.job_id, JobState.QUEUED)
        orch.transition("p1", job.job_id, JobState.RUNNING)
        orch.transition("p1", job.job_id, JobState.FAILED, error_code="MODEL_NOT_FOUND",
                        error_detail="no model")
        with pytest.raises(VideoError) as ei:
            orch.retry_job("p1", job.job_id)
        assert ei.value.code is TypedErrorCode.NON_RETRYABLE

    def test_retry_not_from_non_failed_raises(self, orch):
        orch.create_project("p1")
        job = orch.create_job("p1", JobType.CONTENT_PLAN, inputs={})
        with pytest.raises(VideoError):
            orch.retry_job("p1", job.job_id)


# ===============================================================
# 11. ORCHESTRATOR — STAGE EXECUTION (real FFmpeg)
# ===============================================================
class TestOrchestratorExecution:
    def test_content_plan_runs(self, orch):
        orch.create_project("p1")
        job = orch.create_job("p1", JobType.CONTENT_PLAN,
                              inputs={"brief": {"idea": "lion hunt",
                                                "duration_seconds": 12, "mode": "cinematic"}})
        result = orch.run_content_plan("p1", job.job_id)
        assert result.state is JobState.COMPLETED
        assert result.output_path is not None
        assert Path(result.output_path).exists()
        assert result.outputs.get("scene_count", 0) > 0

    def test_scene_resolution_runs(self, orch):
        orch.create_project("p1")
        cp = orch.create_job("p1", JobType.CONTENT_PLAN,
                             inputs={"brief": {"idea": "lion", "duration_seconds": 12, "mode": "cinematic"}})
        orch.run_content_plan("p1", cp.job_id)
        sr = orch.create_job("p1", JobType.SCENE_RESOLUTION, depends_on=[cp.job_id])
        result = orch.run_scene_resolution("p1", sr.job_id)
        assert result.state is JobState.COMPLETED
        assert result.outputs.get("resolved_scenes", 0) > 0

    def test_assembly_runs_with_real_ffmpeg(self, orch):
        proj = orch.create_project("p1")
        cp = orch.create_job("p1", JobType.CONTENT_PLAN,
                             inputs={"brief": {"idea": "lion", "duration_seconds": 12, "mode": "cinematic"}})
        orch.run_content_plan("p1", cp.job_id)
        sr = orch.create_job("p1", JobType.SCENE_RESOLUTION, depends_on=[cp.job_id])
        orch.run_scene_resolution("p1", sr.job_id)
        # Create real test MP4s for each scene.
        video_jobs = []
        for i, scene in enumerate(proj.plan.scenes, start=1):
            mp4 = proj.assets_dir / f"scene_{i}.mp4"
            _make_scene_video(mp4, duration=float(scene.duration))
            vj = orch.create_job("p1", JobType.VIDEO_SCENE, scene_index=i,
                                 inputs={"scene_index": i, "duration": float(scene.duration)},
                                 depends_on=[sr.job_id])
            orch.transition("p1", vj.job_id, JobState.QUEUED)
            orch.transition("p1", vj.job_id, JobState.RUNNING)
            vj.output_path = str(mp4)
            vj.output_fingerprint = file_fingerprint(mp4)
            proj.store.put(vj)
            orch.transition("p1", vj.job_id, JobState.COMPLETED)
            video_jobs.append(vj)
        asm = orch.create_job("p1", JobType.ASSEMBLY,
                              inputs={"profile_name": "TIKTOK", "quality": "high", "silent": True},
                              depends_on=[j.job_id for j in video_jobs])
        result = orch.run_assembly("p1", asm.job_id)
        assert result.state is JobState.COMPLETED
        assert result.outputs.get("qc", {}).get("ok") is True
        assert Path(result.output_path).exists()
        # ffprobe the final output.
        verify_mp4(Path(result.output_path))

    def test_idempotency_cache_reuse(self, orch):
        orch.create_project("p1")
        inputs = {"brief": {"idea": "lion hunt", "duration_seconds": 12, "mode": "cinematic"}}
        cp1 = orch.create_job("p1", JobType.CONTENT_PLAN, inputs=inputs)
        orch.run_content_plan("p1", cp1.job_id)
        # Second job with identical inputs should be a cache hit.
        cp2 = orch.create_job("p1", JobType.CONTENT_PLAN, inputs=inputs)
        result = orch.run_content_plan("p1", cp2.job_id)
        assert result.state is JobState.COMPLETED
        assert cp2.retry_count == 0  # no retries needed for cache hit

    def test_dependency_failure_blocks_assembly(self, orch):
        proj = orch.create_project("p1")
        bad = orch.create_job("p1", JobType.VIDEO_SCENE, scene_index=99,
                              inputs={"scene_index": 99, "duration": 4.0})
        orch.transition("p1", bad.job_id, JobState.QUEUED)
        orch.transition("p1", bad.job_id, JobState.RUNNING)
        orch.transition("p1", bad.job_id, JobState.FAILED, error_code="MODEL_NOT_FOUND",
                        error_detail="no model")
        asm = orch.create_job("p1", JobType.ASSEMBLY,
                              inputs={"profile_name": "TIKTOK", "silent": True},
                              depends_on=[bad.job_id])
        result = orch.run_assembly("p1", asm.job_id)
        assert result.state is JobState.FAILED
        assert result.error_code == "DEPENDENCY_FAILED"

    def test_assembly_with_no_video_jobs_fails(self, orch):
        orch.create_project("p1")
        asm = orch.create_job("p1", JobType.ASSEMBLY,
                              inputs={"profile_name": "TIKTOK", "silent": True})
        result = orch.run_assembly("p1", asm.job_id)
        assert result.state is JobState.FAILED
        assert result.error_code == "DEPENDENCY_MISSING"

    def test_voiceover_fails_no_provider(self, orch):
        """TTS must FAIL with NO_PROVIDER — never fake success."""
        orch.create_project("p1")
        vo = orch.create_job("p1", JobType.VOICEOVER, scene_index=1,
                             inputs={"scene_index": 1, "text": "hello", "language": "en"})
        result = orch.run_voiceover("p1", vo.job_id)
        assert result.state is JobState.FAILED
        assert result.error_code == "NO_PROVIDER"


# ===============================================================
# 12. RECOVERY + RESUMABILITY
# ===============================================================
class TestRecovery:
    def test_recover_running_job_no_output(self, orch, tmp_root):
        proj = orch.create_project("p1")
        job = orch.create_job("p1", JobType.ASSEMBLY, inputs={"profile_name": "TIKTOK", "silent": True})
        orch.transition("p1", job.job_id, JobState.QUEUED)
        orch.transition("p1", job.job_id, JobState.RUNNING)
        # Simulate restart: new orchestrator instance.
        orch2 = Orchestrator(root=tmp_root)
        recovered = orch2.recover_project("p1")
        assert len(recovered) == 1
        assert recovered[0].state is JobState.FAILED
        assert recovered[0].error_code == "NO_OUTPUT"

    def test_recover_running_job_with_valid_output(self, orch, tmp_root):
        proj = orch.create_project("p1")
        # Create a real MP4 as a "completed" output.
        mp4 = proj.assets_dir / "scene.mp4"
        _make_scene_video(mp4, duration=2.0)
        job = orch.create_job("p1", JobType.VIDEO_SCENE, scene_index=1,
                              inputs={"scene_index": 1, "duration": 2.0})
        orch.transition("p1", job.job_id, JobState.QUEUED)
        orch.transition("p1", job.job_id, JobState.RUNNING)
        job.output_path = str(mp4)
        job.output_fingerprint = file_fingerprint(mp4)
        proj.store.put(job)
        # Simulate restart.
        orch2 = Orchestrator(root=tmp_root)
        recovered = orch2.recover_project("p1")
        assert len(recovered) == 1
        assert recovered[0].state is JobState.COMPLETED

    def test_recover_does_not_mark_abandoned_successful(self, orch, tmp_root):
        """A job left RUNNING with NO output must NOT be marked COMPLETED."""
        proj = orch.create_project("p1")
        job = orch.create_job("p1", JobType.VIDEO_SCENE, scene_index=1, inputs={})
        orch.transition("p1", job.job_id, JobState.QUEUED)
        orch.transition("p1", job.job_id, JobState.RUNNING)
        orch2 = Orchestrator(root=tmp_root)
        recovered = orch2.recover_project("p1")
        assert recovered[0].state is not JobState.COMPLETED
        assert recovered[0].state is JobState.FAILED

    def test_resumability_assembly_after_restart(self, orch, tmp_root):
        """Assembly after restart should reuse cached video outputs."""
        proj = orch.create_project("p1")
        cp = orch.create_job("p1", JobType.CONTENT_PLAN,
                             inputs={"brief": {"idea": "lion", "duration_seconds": 8, "mode": "cinematic"}})
        orch.run_content_plan("p1", cp.job_id)
        sr = orch.create_job("p1", JobType.SCENE_RESOLUTION, depends_on=[cp.job_id])
        orch.run_scene_resolution("p1", sr.job_id)
        video_jobs = []
        for i, scene in enumerate(proj.plan.scenes, start=1):
            mp4 = proj.assets_dir / f"scene_{i}.mp4"
            _make_scene_video(mp4, duration=float(scene.duration))
            vj = orch.create_job("p1", JobType.VIDEO_SCENE, scene_index=i,
                                 inputs={"scene_index": i, "duration": float(scene.duration)},
                                 depends_on=[sr.job_id])
            orch.transition("p1", vj.job_id, JobState.QUEUED)
            orch.transition("p1", vj.job_id, JobState.RUNNING)
            vj.output_path = str(mp4)
            vj.output_fingerprint = file_fingerprint(mp4)
            proj.store.put(vj)
            orch.transition("p1", vj.job_id, JobState.COMPLETED)
            video_jobs.append(vj)
        # Restart.
        orch2 = Orchestrator(root=tmp_root)
        asm = orch2.create_job("p1", JobType.ASSEMBLY,
                               inputs={"profile_name": "TIKTOK", "quality": "high", "silent": True},
                               depends_on=[j.job_id for j in video_jobs])
        result = orch2.run_assembly("p1", asm.job_id)
        assert result.state is JobState.COMPLETED
        assert Path(result.output_path).exists()

    def test_recover_all(self, orch, tmp_root):
        orch.create_project("p1")
        orch.create_project("p2")
        recovered = orch.recover_all()
        # No RUNNING jobs, so nothing to recover.
        assert "p1" in recovered
        assert "p2" in recovered


# ===============================================================
# 13. OUTPUT INTEGRITY VALIDATION
# ===============================================================
class TestOutputIntegrity:
    def test_video_output_validated(self, orch):
        """VIDEO_SCENE output must pass verify_mp4 before COMPLETED."""
        proj = orch.create_project("p1")
        # A corrupt file should not pass validation.
        corrupt = proj.assets_dir / "corrupt.mp4"
        corrupt.write_bytes(b"not an mp4")
        job = orch.create_job("p1", JobType.VIDEO_SCENE, scene_index=1,
                              inputs={"scene_index": 1, "duration": 2.0})
        orch.transition("p1", job.job_id, JobState.QUEUED)
        orch.transition("p1", job.job_id, JobState.RUNNING)
        job.output_path = str(corrupt)
        proj.store.put(job)
        # Transition to COMPLETED would skip validation here (validation is in
        # _run_with_retry). But _validate_output directly should fail.
        with pytest.raises(VideoError):
            orch._validate_output(job, str(corrupt))

    def test_valid_video_passes_validation(self, orch):
        proj = orch.create_project("p1")
        mp4 = proj.assets_dir / "good.mp4"
        _make_scene_video(mp4, duration=2.0)
        job = orch.create_job("p1", JobType.VIDEO_SCENE, scene_index=1, inputs={})
        # Should not raise.
        orch._validate_output(job, str(mp4))


# ===============================================================
# 14. PROVIDER UNAVAILABLE
# ===============================================================
class TestProviderUnavailable:
    def test_video_scene_fails_no_provider(self, orch):
        """VIDEO_SCENE must FAIL with NO_PROVIDER when no real provider exists.
        Never fake a successful generation."""
        proj = orch.create_project("p1")
        job = orch.create_job("p1", JobType.VIDEO_SCENE, scene_index=1,
                              inputs={"scene_index": 1, "duration": 4.0, "prompt": "lion"})
        result = orch.run_video_scene("p1", job.job_id)
        assert result.state is JobState.FAILED
        assert result.error_code == "NO_PROVIDER"

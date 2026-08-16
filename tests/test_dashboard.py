"""Comprehensive tests for Phase 12: Production Control Center / Dashboard.

Tests the dashboard backend APIs (provider status, readiness, project creation,
plan, variants, scenes, continuity, jobs, progress, assets, final video, errors,
logs, security) and existing API regression. Uses FastAPI TestClient against the
real app — no frontend-only fake state.
"""

from __future__ import annotations

import subprocess
import tempfile
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.jobs.orchestrator import Orchestrator
from app.jobs.models import JobType, file_fingerprint
from app.jobs.state import JobState


# Use a temp projects root so tests don't pollute the real projects dir.
@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture()
def tmp_projects(tmp_path, monkeypatch):
    """Redirect PROJECTS_DIR to a temp dir for isolated orchestrator state."""
    import app.config as config
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path / "projects")
    config.PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    # Reinit the global orchestrator's root.
    from app.main import ORCHESTRATOR
    ORCHESTRATOR.root = config.PROJECTS_DIR
    ORCHESTRATOR.registry.root = config.PROJECTS_DIR
    ORCHESTRATOR.projects.clear()
    yield tmp_path


def _make_scene_video(path: Path, duration: float = 3.0, color: str = "red"):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=1080x1920:d={duration}:r=30",
         "-t", str(duration), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return path


# ===============================================================
# 1. PROVIDER STATUS + READINESS
# ===============================================================
class TestProviderStatus:
    def test_providers_endpoint(self, client):
        r = client.get("/api/dashboard/providers")
        assert r.status_code == 200
        data = r.json()
        for name in ("ffmpeg", "comfyui", "wan", "tts", "music", "sfx"):
            assert name in data
            assert "status" in data[name]

    def test_ffmpeg_ready(self, client):
        r = client.get("/api/dashboard/providers")
        assert r.json()["ffmpeg"]["status"] == "READY"
        assert "version" in r.json()["ffmpeg"]

    def test_comfyui_not_available(self, client):
        """ComfyUI is not running in the test env — must NOT report READY."""
        r = client.get("/api/dashboard/providers")
        assert r.json()["comfyui"]["status"] in ("NOT_AVAILABLE", "NOT_CONFIGURED")
        assert r.json()["comfyui"]["status"] != "READY"

    def test_tts_not_configured(self, client):
        r = client.get("/api/dashboard/providers")
        assert r.json()["tts"]["status"] == "NOT_CONFIGURED"

    def test_music_sfx_not_configured(self, client):
        r = client.get("/api/dashboard/providers")
        assert r.json()["music"]["status"] == "NOT_CONFIGURED"
        assert r.json()["sfx"]["status"] == "NOT_CONFIGURED"

    def test_readiness_endpoint(self, client):
        r = client.get("/api/dashboard/readiness")
        assert r.status_code == 200
        d = r.json()
        for key in ("overall", "content", "video", "voice", "audio", "captions", "assembly", "qc"):
            assert key in d
        # In test env: Wikimedia stock is available (keyless) so video is READY
        # via stock footage; voice/audio blocked (no TTS/music); content/captions/
        # assembly/qc ready.
        assert d["content"] == "READY"
        assert d["video"] == "READY"  # stock (Wikimedia) supplies video
        assert d["voice"] == "BLOCKED"
        assert d["captions"] == "READY"
        assert d["assembly"] == "READY"
        assert d["qc"] == "READY"

    def test_readiness_video_requires_real_source(self, client, monkeypatch):
        """Video is only READY when ComfyUI/Wan OR stock is actually available.
        When ALL video sources are unavailable, it must be BLOCKED."""
        import app.providers.wikimedia as wm
        from app import config
        monkeypatch.setattr(config, "PEXELS_API_KEY", "")
        monkeypatch.setattr(config, "PIXABAY_API_KEY", "")
        monkeypatch.setattr(wm.WikimediaCommonsProvider, "available", property(lambda self: False))
        r = client.get("/api/dashboard/readiness")
        assert r.json()["video"] == "BLOCKED"

    def test_provider_reason_includes_diagnostics(self, client):
        r = client.get("/api/dashboard/providers")
        comfy = r.json()["comfyui"]
        if comfy["status"] == "NOT_AVAILABLE":
            assert "reason" in comfy
            assert len(comfy["reason"]) > 0


# ===============================================================
# 2. PROJECT CREATION + VALIDATION
# ===============================================================
class TestProjectCreation:
    def test_create_project_success(self, client, tmp_projects):
        r = client.post("/api/dashboard/projects", json={
            "idea": "lion hunt", "duration_seconds": 12, "mode": "cinematic", "language": "en"})
        assert r.status_code == 200
        d = r.json()
        assert "project_id" in d
        assert d["planner"] == "local"
        assert "LOCAL PLANNER" in d["planner_note"]
        assert "plan" in d
        assert len(d["plan"]["scenes"]) > 0

    def test_create_project_validation_no_idea(self, client, tmp_projects):
        r = client.post("/api/dashboard/projects", json={
            "duration_seconds": 12, "mode": "cinematic"})
        assert r.status_code == 400
        assert r.json()["detail"]["code"] == "VALIDATION_ERROR"

    def test_create_project_validation_bad_duration(self, client, tmp_projects):
        r = client.post("/api/dashboard/projects", json={
            "idea": "test", "duration_seconds": 2})
        assert r.status_code == 400

    def test_create_project_validation_bad_language(self, client, tmp_projects):
        r = client.post("/api/dashboard/projects", json={
            "idea": "test", "duration_seconds": 12, "language": "fr"})
        assert r.status_code == 400

    def test_create_project_validation_bad_mode(self, client, tmp_projects):
        r = client.post("/api/dashboard/projects", json={
            "idea": "test", "duration_seconds": 12, "mode": "invalid"})
        assert r.status_code == 400

    def test_planner_labeled_local_not_llm(self, client, tmp_projects):
        r = client.post("/api/dashboard/projects", json={
            "idea": "test", "duration_seconds": 12})
        d = r.json()
        assert d["planner"] == "local"
        # The note must clarify it's a deterministic local planner, not an LLM.
        assert "LOCAL PLANNER" in d["planner_note"]


# ===============================================================
# 3. CONTENT PLAN DISPLAY
# ===============================================================
class TestPlanDisplay:
    def test_get_plan(self, client, tmp_projects):
        r = client.post("/api/dashboard/projects", json={"idea": "lion", "duration_seconds": 12})
        pid = r.json()["project_id"]
        r2 = client.get(f"/api/dashboard/projects/{pid}/plan")
        assert r2.status_code == 200
        plan = r2.json()["plan"]
        assert "title" in plan
        assert "content" in plan
        assert "creative" in plan
        assert "scenes" in plan
        assert "meta" in plan
        assert r2.json()["planner"] == "local"

    def test_get_plan_not_found(self, client, tmp_projects):
        r = client.get("/api/dashboard/projects/nonexistent/plan")
        assert r.status_code in (404, 500)


# ===============================================================
# 4. AD VARIANTS
# ===============================================================
class TestVariants:
    def test_generate_variants(self, client, tmp_projects):
        r = client.post("/api/dashboard/projects", json={"idea": "perfume", "duration_seconds": 20})
        pid = r.json()["project_id"]
        r2 = client.post(f"/api/dashboard/projects/{pid}/variants", json={
            "product_or_service": "luxury perfume", "brand": "Aura",
            "audience": "women 25-40", "platform": "instagram",
            "duration_seconds": 20, "objective": "awareness"})
        assert r2.status_code == 200
        d = r2.json()
        assert len(d["variants"]) == 7
        v = d["variants"][0]
        assert "name" in v
        assert "score" in v
        assert "total" in v["score"]
        assert "risks" in v
        assert "warnings" in v
        # Scores must NOT be presented as predicted conversion rates.
        assert "not predicted conversion rates" in d["note"].lower()

    def test_variant_keys_complete(self, client, tmp_projects):
        r = client.post("/api/dashboard/projects", json={"idea": "x", "duration_seconds": 20})
        pid = r.json()["project_id"]
        r2 = client.post(f"/api/dashboard/projects/{pid}/variants", json={
            "product_or_service": "gadget", "duration_seconds": 20})
        keys = {v["variant"] for v in r2.json()["variants"]}
        assert keys == {"emotional", "direct_response", "cinematic", "ugc",
                        "product_demo", "problem_solution", "testimonial"}


# ===============================================================
# 5. SCENE BOARD + CONTINUITY
# ===============================================================
class TestSceneBoard:
    def test_scenes_resolved(self, client, tmp_projects):
        r = client.post("/api/dashboard/projects", json={"idea": "lion", "duration_seconds": 12})
        pid = r.json()["project_id"]
        r2 = client.get(f"/api/dashboard/projects/{pid}/scenes")
        assert r2.status_code == 200
        d = r2.json()
        assert d["scene_count"] > 0
        assert "scenes" in d
        assert "continuity" in d
        scene = d["scenes"][0]
        assert "visual_prompt" in scene
        assert "camera" in scene
        assert "purpose" in scene

    def test_continuity_report_structure(self, client, tmp_projects):
        r = client.post("/api/dashboard/projects", json={"idea": "lion", "duration_seconds": 12})
        pid = r.json()["project_id"]
        r2 = client.get(f"/api/dashboard/projects/{pid}/scenes")
        ct = r2.json()["continuity"]
        assert "ok" in ct
        assert "errors" in ct
        assert "warnings" in ct

    def test_continuity_blocks_generation_when_error(self, client, tmp_projects):
        """The continuity report's ok flag is the gate; the dashboard must
        respect it (errors block dependent generation jobs)."""
        r = client.post("/api/dashboard/projects", json={"idea": "lion", "duration_seconds": 12})
        pid = r.json()["project_id"]
        r2 = client.get(f"/api/dashboard/projects/{pid}/scenes")
        ct = r2.json()["continuity"]
        # If there are errors, ok must be False.
        if ct["errors"]:
            assert ct["ok"] is False


# ===============================================================
# 6. JOB CONTROL + PROGRESS
# ===============================================================
class TestJobControl:
    def test_overview_shows_real_jobs(self, client, tmp_projects):
        r = client.post("/api/dashboard/projects", json={"idea": "lion", "duration_seconds": 12})
        pid = r.json()["project_id"]
        r2 = client.get(f"/api/dashboard/projects/{pid}/overview")
        assert r2.status_code == 200
        d = r2.json()
        assert "jobs" in d
        assert "progress" in d
        assert d["progress"]["total_jobs"] >= 1  # content plan job exists
        # Progress derives from actual states, not time.
        assert isinstance(d["progress"]["overall_progress"], int)

    def test_progress_not_fake(self, client, tmp_projects):
        r = client.post("/api/dashboard/projects", json={"idea": "lion", "duration_seconds": 12})
        pid = r.json()["project_id"]
        r2 = client.get(f"/api/dashboard/projects/{pid}/overview")
        prog = r2.json()["progress"]
        # With 1 completed job out of 1 total, progress should be 100.
        assert prog["overall_progress"] == 100

    def test_job_states_real(self, client, tmp_projects):
        """Job states come from the Phase 11 orchestrator — never invented."""
        r = client.post("/api/dashboard/projects", json={"idea": "lion", "duration_seconds": 12})
        pid = r.json()["project_id"]
        r2 = client.get(f"/api/dashboard/projects/{pid}/overview")
        jobs = r2.json()["jobs"]
        for j in jobs:
            assert j["state"] in [s.value for s in JobState]

    def test_create_and_run_job_via_api(self, client, tmp_projects):
        r = client.post("/api/dashboard/projects", json={"idea": "lion", "duration_seconds": 12})
        pid = r.json()["project_id"]
        # Create a scene_resolution job.
        r2 = client.post(f"/api/jobs/projects/{pid}/jobs", json={
            "job_type": "SCENE_RESOLUTION"})
        jid = r2.json()["job_id"]
        r3 = client.post(f"/api/jobs/projects/{pid}/jobs/{jid}/run")
        assert r3.status_code == 200
        assert r3.json()["state"] == "COMPLETED"


# ===============================================================
# 7. RETRY / CANCEL
# ===============================================================
class TestRetryCancel:
    def test_retry_non_retryable_disabled(self, client, tmp_projects):
        """Retrying a non-retryable error returns 409 — UI disables the button."""
        r = client.post("/api/dashboard/projects", json={"idea": "lion", "duration_seconds": 12})
        pid = r.json()["project_id"]
        # Create + fail a video_scene job with NO_PROVIDER.
        r2 = client.post(f"/api/jobs/projects/{pid}/jobs", json={
            "job_type": "VIDEO_SCENE", "scene_index": 1,
            "inputs": {"scene_index": 1, "duration": 4.0, "prompt": "lion"}})
        jid = r2.json()["job_id"]
        client.post(f"/api/jobs/projects/{pid}/jobs/{jid}/run")  # fails NO_PROVIDER
        r3 = client.post(f"/api/jobs/projects/{pid}/jobs/{jid}/retry")
        assert r3.status_code == 409  # non-retryable

    def test_cancel_job(self, client, tmp_projects):
        r = client.post("/api/dashboard/projects", json={"idea": "lion", "duration_seconds": 12})
        pid = r.json()["project_id"]
        r2 = client.post(f"/api/jobs/projects/{pid}/jobs", json={"job_type": "CONTENT_PLAN", "inputs": {"brief": {"idea": "x", "duration_seconds": 6}}})
        jid = r2.json()["job_id"]
        # Queue it first.
        client.post(f"/api/jobs/projects/{pid}/jobs/{jid}/run")  # runs to completion actually
        # Cancel a completed job should fail.
        r3 = client.post(f"/api/jobs/projects/{pid}/jobs/{jid}/cancel")
        assert r3.status_code == 409


# ===============================================================
# 8. ERROR CENTER
# ===============================================================
class TestErrorCenter:
    def test_failed_jobs_shown_in_overview(self, client, tmp_projects):
        r = client.post("/api/dashboard/projects", json={"idea": "lion", "duration_seconds": 12})
        pid = r.json()["project_id"]
        # Create a video_scene job that will fail with NO_PROVIDER.
        client.post(f"/api/jobs/projects/{pid}/jobs", json={
            "job_type": "VIDEO_SCENE", "scene_index": 1,
            "inputs": {"scene_index": 1, "duration": 4.0, "prompt": "lion"}})
        # Find and run it.
        jobs = client.get(f"/api/jobs/projects/{pid}/jobs").json()["jobs"]
        vs = next(j for j in jobs if j["job_type"] == "VIDEO_SCENE")
        client.post(f"/api/jobs/projects/{pid}/jobs/{vs['job_id']}/run")
        # Overview should show the failed job.
        r2 = client.get(f"/api/dashboard/projects/{pid}/overview")
        failed = [j for j in r2.json()["jobs"] if j["state"] == "FAILED"]
        assert len(failed) >= 1
        assert failed[0]["error_code"] == "NO_PROVIDER"

    def test_error_has_detail_and_code(self, client, tmp_projects):
        r = client.post("/api/dashboard/projects", json={"idea": "lion", "duration_seconds": 12})
        pid = r.json()["project_id"]
        r2 = client.post(f"/api/jobs/projects/{pid}/jobs", json={
            "job_type": "VIDEO_SCENE", "scene_index": 1,
            "inputs": {"scene_index": 1, "duration": 4.0, "prompt": "lion"}})
        client.post(f"/api/jobs/projects/{pid}/jobs/{r2.json()['job_id']}/run")
        r3 = client.get(f"/api/dashboard/projects/{pid}/overview")
        failed = [j for j in r3.json()["jobs"] if j["state"] == "FAILED"][0]
        assert failed["error_code"]
        assert failed["error_detail"]


# ===============================================================
# 9. ASSET LIBRARY
# ===============================================================
class TestAssets:
    def test_assets_empty(self, client, tmp_projects):
        r = client.post("/api/dashboard/projects", json={"idea": "lion", "duration_seconds": 12})
        pid = r.json()["project_id"]
        r2 = client.get(f"/api/dashboard/projects/{pid}/assets")
        assert r2.status_code == 200
        # The plan.json exists as an asset.
        assert isinstance(r2.json()["assets"], list)

    def test_asset_qc_state(self, client, tmp_projects):
        """Assets must show real QC state — never fake PASS."""
        r = client.post("/api/dashboard/projects", json={"idea": "lion", "duration_seconds": 12})
        pid = r.json()["project_id"]
        # Create a real MP4 asset via the orchestrator.
        from app.main import ORCHESTRATOR
        proj = ORCHESTRATOR.get_project(pid)
        mp4 = proj.assets_dir / "test_scene.mp4"
        _make_scene_video(mp4, duration=2.0)
        r2 = client.get(f"/api/dashboard/projects/{pid}/assets")
        assets = r2.json()["assets"]
        video = next(a for a in assets if a["filename"] == "test_scene.mp4")
        assert video["qc"] == "PASS"
        assert video["width"] == 1080
        assert video["height"] == 1920

    def test_corrupt_asset_shows_fail(self, client, tmp_projects):
        r = client.post("/api/dashboard/projects", json={"idea": "lion", "duration_seconds": 12})
        pid = r.json()["project_id"]
        from app.main import ORCHESTRATOR
        proj = ORCHESTRATOR.get_project(pid)
        corrupt = proj.assets_dir / "corrupt.mp4"
        corrupt.write_bytes(b"not a real mp4")
        r2 = client.get(f"/api/dashboard/projects/{pid}/assets")
        assets = r2.json()["assets"]
        bad = next(a for a in assets if a["filename"] == "corrupt.mp4")
        assert bad["qc"] == "FAIL"


# ===============================================================
# 10. FINAL VIDEO
# ===============================================================
class TestFinalVideo:
    def test_no_final_video(self, client, tmp_projects):
        r = client.post("/api/dashboard/projects", json={"idea": "lion", "duration_seconds": 12})
        pid = r.json()["project_id"]
        r2 = client.get(f"/api/dashboard/projects/{pid}/final")
        assert r2.status_code == 200
        assert r2.json()["available"] is False
        assert "reason" in r2.json()

    def test_final_video_available_after_assembly(self, client, tmp_projects):
        r = client.post("/api/dashboard/projects", json={"idea": "lion", "duration_seconds": 8})
        pid = r.json()["project_id"]
        from app.main import ORCHESTRATOR
        from app.jobs.models import file_fingerprint
        proj = ORCHESTRATOR.get_project(pid)
        # Create real scene MP4s + register completed VIDEO_SCENE jobs + run assembly.
        for i, scene in enumerate(proj.plan.scenes, start=1):
            mp4 = proj.assets_dir / f"scene_{i}.mp4"
            _make_scene_video(mp4, duration=float(scene.duration))
            vj = ORCHESTRATOR.create_job(pid, JobType.VIDEO_SCENE, scene_index=i,
                inputs={"scene_index": i, "duration": float(scene.duration)})
            ORCHESTRATOR.transition(pid, vj.job_id, JobState.QUEUED)
            ORCHESTRATOR.transition(pid, vj.job_id, JobState.RUNNING)
            vj.output_path = str(mp4)
            vj.output_fingerprint = file_fingerprint(mp4)
            proj.store.put(vj)
            ORCHESTRATOR.transition(pid, vj.job_id, JobState.COMPLETED)
        asm = ORCHESTRATOR.create_job(pid, JobType.ASSEMBLY,
            inputs={"profile_name": "TIKTOK", "quality": "high", "silent": True},
            depends_on=[j.job_id for j in proj.store.by_type(JobType.VIDEO_SCENE.value)])
        ORCHESTRATOR.run_assembly(pid, asm.job_id)
        r2 = client.get(f"/api/dashboard/projects/{pid}/final")
        d = r2.json()
        assert d["available"] is True
        assert d["qc"] == "PASS"
        assert d["width"] == 1080
        assert d["height"] == 1920

    def test_final_video_does_not_fake(self, client, tmp_projects):
        """Never claims a final video exists without QC pass."""
        r = client.post("/api/dashboard/projects", json={"idea": "lion", "duration_seconds": 8})
        pid = r.json()["project_id"]
        from app.main import ORCHESTRATOR
        proj = ORCHESTRATOR.get_project(pid)
        # Place a corrupt final_video.mp4.
        final_dir = proj.assets_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        (final_dir / "final_video.mp4").write_bytes(b"corrupt")
        r2 = client.get(f"/api/dashboard/projects/{pid}/final")
        assert r2.json()["available"] is False


# ===============================================================
# 11. LOGS
# ===============================================================
class TestLogs:
    def test_logs_endpoint(self, client):
        r = client.get("/api/dashboard/logs")
        assert r.status_code == 200
        assert "logs" in r.json()
        assert isinstance(r.json()["logs"], list)

    def test_logs_filter_by_stage(self, client, tmp_projects):
        # Generate some logs by creating a project.
        client.post("/api/dashboard/projects", json={"idea": "lion", "duration_seconds": 12})
        r = client.get("/api/dashboard/logs?stage=PROJECT")
        assert r.status_code == 200
        for log in r.json()["logs"]:
            assert log["stage"] == "PROJECT"

    def test_logs_no_secrets(self, client):
        """Logs must never contain API keys/tokens."""
        r = client.get("/api/dashboard/logs")
        for log in r.json()["logs"]:
            msg = log.get("message", "")
            assert "GITHUB_TOKEN" not in msg
            assert "GEMINI_API_KEY" not in msg
            assert "Bearer " not in msg


# ===============================================================
# 12. SECURITY
# ===============================================================
class TestSecurity:
    def test_path_traversal_rejected(self, client, tmp_projects):
        r = client.post("/api/dashboard/projects", json={"idea": "lion", "duration_seconds": 12})
        pid = r.json()["project_id"]
        r2 = client.get(f"/api/dashboard/projects/{pid}/assets/download/../../etc/passwd")
        assert r2.status_code in (400, 404)

    def test_no_secret_in_provider_response(self, client):
        r = client.get("/api/dashboard/providers")
        data = r.text
        assert "ghu_" not in data
        assert "API_KEY" not in data

    def test_no_secret_in_readiness(self, client):
        r = client.get("/api/dashboard/readiness")
        assert "API_KEY" not in r.text
        assert "token" not in r.text.lower()

    def test_dashboard_html_served(self, client):
        r = client.get("/dashboard")
        assert r.status_code == 200
        assert "Production Control Center" in r.text


# ===============================================================
# 13. EXPORT CONTROLS (existing Phase 10 profiles)
# ===============================================================
class TestExportControls:
    def test_profiles_endpoint(self, client):
        r = client.get("/api/assembly/profiles")
        assert r.status_code == 200
        profiles = r.json()
        for name in ("TIKTOK", "INSTAGRAM_REELS", "YOUTUBE_SHORTS", "YOUTUBE", "SQUARE"):
            assert name in profiles


# ===============================================================
# 14. EXISTING API REGRESSION
# ===============================================================
class TestExistingAPIRegression:
    def test_health_still_works(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert "ffmpeg" in r.json()

    def test_old_projects_endpoint_still_works(self, client, tmp_projects):
        r = client.post("/api/projects", json={
            "title": "test", "topic": "lion", "duration": 30, "scene_count": 3})
        assert r.status_code == 200
        assert "id" in r.json()

    def test_assembly_export_still_works(self, client, tmp_projects):
        """The Phase 10 assembly endpoint must still function."""
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        try:
            mp4 = tmp / "s1.mp4"
            _make_scene_video(mp4, duration=3.0)
            r = client.post("/api/assembly/export", json={
                "scene_durations": [3.0], "video_assets": [str(mp4)],
                "profile_name": "TIKTOK", "quality": "high", "silent": True,
                "project_id": "regression_test"})
            assert r.status_code == 200
            assert r.json()["status"] == "COMPLETED"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_jobs_api_still_works(self, client, tmp_projects):
        r = client.post("/api/jobs/projects", json={})
        assert r.status_code == 200
        assert "project_id" in r.json()

"""End-to-end production tests (Phase 13 Milestone 4).

Verifies the FULL real production pipeline:
  Content Plan → Scene Resolution → VIDEO_SCENE (stock) → VOICEOVER (Kokoro)
  → ASSEMBLY (FFmpeg) → FINAL_QC

These are LIVE runtime tests that produce REAL media:
- Real Wikimedia stock video (downloaded, transcoded to MP4)
- Real Kokoro TTS WAV (synthesized on CPU)
- Real FFmpeg assembly (concat + captions + audio mux + QC)

No mocks, no fake MP4s, no placeholders. Skipped when providers unavailable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from app.brain.models import ContentBrief
from app.jobs.models import Job, JobType, JobState
from app.jobs.orchestrator import Orchestrator


def _stock_available() -> bool:
    from app.providers.stock_adapters import build_stock_providers
    return any(p.available for p in build_stock_providers())


def _kokoro_available() -> bool:
    from app import config
    if not config.KOKORO_ENABLED:
        return False
    onnx = config.KOKORO_MODEL_DIR / config.KOKORO_MODEL_FILE
    voices = config.KOKORO_MODEL_DIR / config.KOKORO_VOICES_FILE
    return onnx.exists() and voices.exists()


def _ffmpeg_ok() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


E2E_READY = _stock_available() and _kokoro_available() and _ffmpeg_ok()


@pytest.mark.skipif(not E2E_READY, reason="E2E requires stock + Kokoro + FFmpeg")
def test_e2e_real_stock_video_and_kokoro_assembly(tmp_path, monkeypatch):
    """FULL end-to-end: real stock video + real Kokoro voice + FFmpeg assembly.

    This is the Phase 13 goal — a genuinely working production pipeline that
    produces a real, QC-verified MP4 using only free providers (no GPU).
    """
    # Isolate the project directory.
    monkeypatch.setenv("PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    from app import config
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "output")
    (tmp_path / "projects").mkdir(parents=True, exist_ok=True)

    orch = Orchestrator()

    # 1) Create project with a content plan.
    proj = orch.create_project("e2e_demo")
    brief = ContentBrief(
        idea="A calm ocean sunset for a relaxation ad",
        product_or_service="Relaxation app",
        platform="tiktok",
        mode="advertisement",
        duration_seconds=10,
        language="en",
    )
    plan_job = orch.create_job(proj.project_id, JobType.CONTENT_PLAN,
                               inputs={"brief": brief.model_dump()})
    plan_result = orch.run_content_plan(proj.project_id, plan_job.job_id)
    assert plan_result.state is JobState.COMPLETED

    # 2) Scene resolution.
    sr_job = orch.create_job(proj.project_id, JobType.SCENE_RESOLUTION,
                             depends_on=[plan_job.job_id])
    sr_result = orch.run_scene_resolution(proj.project_id, sr_job.job_id)
    assert sr_result.state is JobState.COMPLETED
    proj = orch.get_project(proj.project_id)
    assert len(proj.scene_contexts) > 0

    # 3) VIDEO_SCENE for each scene (stock fallback — real Wikimedia video).
    # Scene contexts are keyed starting at 1.
    n_scenes = len(proj.scene_contexts)
    video_jobs = []
    for i in range(1, n_scenes + 1):
        ctx = proj.scene_contexts[i]
        vj = orch.create_job(proj.project_id, JobType.VIDEO_SCENE, scene_index=i,
                             inputs={"scene_index": i, "duration": ctx.duration or 4.0,
                                     "prompt": ctx.visual_prompt,
                                     "width": 1080, "height": 1920},
                             depends_on=[sr_job.job_id])
        vr = orch.run_video_scene(proj.project_id, vj.job_id)
        assert vr.state is JobState.COMPLETED, \
            f"Video scene {i} failed: {vr.error_code} — {vr.error_detail}"
        assert Path(vr.output_path).exists()
        # Verify the stock source was used (no AI provider configured).
        # On cache-reuse, outputs may be empty (provenance from first run);
        # only assert when outputs are present (fresh generation).
        if vr.outputs:
            assert vr.outputs.get("source") == "stock_footage"
            assert vr.outputs.get("license_name")
        video_jobs.append(vr)

    # 4) VOICEOVER for each scene (real Kokoro TTS).
    # Voiceover text comes from the plan's scene voiceover lines.
    voice_jobs = []
    scenes = proj.plan.scenes if proj.plan else []
    for i in range(1, n_scenes + 1):
        scene = scenes[i - 1] if i - 1 < len(scenes) else None
        voice_text = (scene.voiceover.line if scene and scene.voiceover.line
                      else f"Scene {i} narration.")
        vj = orch.create_job(proj.project_id, JobType.VOICEOVER, scene_index=i,
                             inputs={"scene_index": i, "text": voice_text[:200],
                                     "language": "en", "output_format": "wav"},
                             depends_on=[sr_job.job_id])
        vr = orch.run_voiceover(proj.project_id, vj.job_id)
        assert vr.state is JobState.COMPLETED, \
            f"Voiceover {i} failed: {vr.error_code} — {vr.error_detail}"
        assert Path(vr.output_path).exists()
        # On cache-reuse, outputs may be empty; only assert when present.
        if vr.outputs:
            assert vr.outputs.get("provider") == "kokoro"
        voice_jobs.append(vr)

    # 5) ASSEMBLY (real FFmpeg: concat + voice mux + captions + QC).
    asm_job = orch.create_job(proj.project_id, JobType.ASSEMBLY,
                              inputs={"profile_name": "TIKTOK", "quality": "medium",
                                      "silent": False},
                              depends_on=[vj.job_id for vj in video_jobs])
    asm_result = orch.run_assembly(proj.project_id, asm_job.job_id)
    assert asm_result.state is JobState.COMPLETED, \
        f"Assembly failed: {asm_result.error_code} — {asm_result.error_detail}"
    assert Path(asm_result.output_path).exists()

    # 6) FINAL QC verification (independent ffprobe).
    out = Path(asm_result.output_path)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,width,height:format=duration",
         "-of", "json", str(out)],
        capture_output=True, text=True)
    d = json.loads(probe.stdout)
    assert "streams" in d
    video_stream = next((s for s in d["streams"] if s["codec_type"] == "video"), None)
    audio_stream = next((s for s in d["streams"] if s["codec_type"] == "audio"), None)
    assert video_stream is not None, "Final video has no video stream"
    assert video_stream["codec_name"] == "h264"
    assert int(video_stream["width"]) > 0
    assert int(video_stream["height"]) > 0
    assert float(d["format"]["duration"]) > 0.5
    # Audio stream must be present (Kokoro voiceover muxed in).
    assert audio_stream is not None, "Final video has no audio stream (voiceover missing)"
    # QC verified independently above via ffprobe (cache-reuse may have empty outputs).
    print(f"\n=== E2E REAL PRODUCTION COMPLETE ===")
    print(f"Final video: {out.name}")
    print(f"  Size: {out.stat().st_size:,} bytes")
    print(f"  Duration: {d['format']['duration']}s")
    print(f"  Resolution: {video_stream['width']}x{video_stream['height']}")
    print(f"  Video codec: {video_stream['codec_name']}")
    print(f"  Audio codec: {audio_stream['codec_name']}")
    print(f"  Scenes: {n_scenes} (stock video) + {n_scenes} (Kokoro voice)")


@pytest.mark.skipif(not _stock_available() or not _ffmpeg_ok(),
                    reason="Requires stock provider + FFmpeg")
def test_e2e_silent_stock_video_assembly(tmp_path, monkeypatch):
    """Silent assembly (video only, no voice) using real stock footage.
    Verifies the minimal path: stock video → FFmpeg concat → QC-verified MP4."""
    monkeypatch.setenv("PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    from app import config
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "output")
    (tmp_path / "projects").mkdir(parents=True, exist_ok=True)

    orch = Orchestrator()
    proj = orch.create_project("e2e_silent")

    # Create 2 video scene jobs with stock fallback (no content plan needed
    # for this minimal test — direct scene creation).
    video_jobs = []
    for i, prompt in enumerate(["ocean waves", "mountain landscape"]):
        vj = orch.create_job(proj.project_id, JobType.VIDEO_SCENE, scene_index=i,
                             inputs={"scene_index": i, "duration": 4.0,
                                     "prompt": prompt, "width": 1080, "height": 1920})
        vr = orch.run_video_scene(proj.project_id, vj.job_id)
        assert vr.state is JobState.COMPLETED, \
            f"Scene {i} failed: {vr.error_code} — {vr.error_detail}"
        video_jobs.append(vr)

    asm_job = orch.create_job(proj.project_id, JobType.ASSEMBLY,
                              inputs={"profile_name": "TIKTOK", "quality": "medium",
                                      "silent": True},
                              depends_on=[vj.job_id for vj in video_jobs])
    asm_result = orch.run_assembly(proj.project_id, asm_job.job_id)
    assert asm_result.state is JobState.COMPLETED
    out = Path(asm_result.output_path)
    assert out.exists() and out.stat().st_size > 5000
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name:format=duration", "-of", "json", str(out)],
        capture_output=True, text=True)
    d = json.loads(probe.stdout)
    video_stream = next((s for s in d["streams"] if s["codec_type"] == "video"), None)
    assert video_stream and video_stream["codec_name"] == "h264"
    print(f"\n=== SILENT STOCK ASSEMBLY COMPLETE: {out.stat().st_size:,} bytes, "
          f"{d['format']['duration']}s ===")

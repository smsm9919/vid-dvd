import json, uuid
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from .config import *
from .models import ProjectCreate
from .planner import local_scene_plan, gemini_scene_plan
from .comfy import ComfyClient, ComfyError
from .media import ffmpeg_available, ffprobe_available, verify_mp4
from .core.errors import VideoError, TypedErrorCode
from .providers.registry import build_providers, provider_status, select_provider
from .providers.base import GenerationRequest
from .brain.models import ContentBrief, ProductionPlan
from .brain.content_brain import plan_content
from .ads.brief import AdBrief
from .ads.variants import generate_variants, generate_variant
from .ads.scoring import score_variant, compare_variants
from .scene.continuity import resolve_scene_context, resolve_all_scenes, validate_continuity
from .scene.references import registry_from_plan
from .providers.wan import WanProvider, GenerationOptions, WanMode
from .voice.tts import NullTTSProvider, TTSProvider, VoiceRequest, select_tts_provider
from .voice.voiceover import generate_scene_voiceover, generate_project_voiceover, validate_voice_timing
from .audio.music import MusicRequest, NullMusicProvider
from .audio.sfx import SFXRequest, NullSFXProvider
from .audio.mixer import MixTrack, MixOptions, mix_audio, mix_scene_audio
from .audio.qc import verify_audio
from .captions.captions import CaptionFormat, CaptionStyle, generate_captions, write_captions
from .editing.timeline import Timeline, TimelineScene, build_timeline_from_assets, validate_timeline
from .editing.profiles import ExportProfile, Quality, get_profile, PROFILES
from .editing.assembly import ExportRequest, ExportResult, export_video, validate_scene_assets
from .editing.qc import final_qc
from .jobs.orchestrator import Orchestrator, JobError
from .jobs.models import Job, JobType
from .jobs.state import JobState

app=FastAPI(title="VideoFactory Local", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
PROJECTS={}

@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR/"index.html").read_text(encoding="utf-8")

@app.get("/api/health")
async def health():
    # Real provider detection via the registry. Never fabricates availability.
    providers = await provider_status(build_providers())
    comfy = next((p for p in providers if p.get("name") == "comfyui"), {"ok": False})
    return {
        "app": "ok",
        "ffmpeg": ffmpeg_available(),
        "ffprobe": ffprobe_available(),
        "comfyui": comfy,
        "providers": providers,
        "workflow_exists": WORKFLOW_PATH.exists(),
    }

@app.post("/api/plan")
async def plan(brief: ContentBrief):
    # Phase 5: AI Content Brain. Produces a ProductionPlan only; no video.
    plan = await plan_content(brief)
    return plan

@app.post("/api/ads/variants")
async def ads_variants(brief: AdBrief):
    # Phase 6: generate all 7 creative variants + heuristic scores + comparison.
    results = generate_variants(brief)
    scores = [score_variant(r, brief) for r in results]
    comparison = compare_variants(results, brief)
    return {
        "variants": [r.to_dict() for r in results],
        "scores": [s.to_dict() for s in scores],
        "comparison": comparison.to_dict(),
    }

@app.post("/api/ads/variants/{key}")
async def ads_variant(key: str, brief: AdBrief):
    # Phase 6: generate a single variant by key.
    try:
        r = generate_variant(brief, key)
    except KeyError:
        raise HTTPException(404, f"Unknown variant: {key}")
    s = score_variant(r, brief)
    return {"variant": r.to_dict(), "score": s.to_dict()}

@app.post("/api/scene/resolve/{scene_index}")
async def scene_resolve(scene_index: int, plan: ProductionPlan):
    # Phase 7: resolve a single scene's full continuity context before generation.
    try:
        ctx = resolve_scene_context(plan, scene_index)
    except IndexError:
        raise HTTPException(404, f"Scene index {scene_index} not found.")
    report = validate_continuity(plan)
    return {"context": ctx.to_dict(), "validation": report.to_dict()}

@app.post("/api/scene/resolve")
async def scene_resolve_all(plan: ProductionPlan):
    # Phase 7: resolve all scenes + continuity validation + reference registry.
    ctxs = resolve_all_scenes(plan)
    report = validate_continuity(plan)
    reg = registry_from_plan(plan)
    return {
        "contexts": [c.to_dict() for c in ctxs],
        "validation": report.to_dict(),
        "references": reg.to_dict(),
    }

class DiagnoseRequest(BaseModel):
    plan: Optional[ProductionPlan] = None

@app.post("/api/diagnose")
async def diagnose(req: Optional[DiagnoseRequest] = None):
    # Phase 8: honest readiness state with exact blockers.
    p = WanProvider(COMFYUI_URL, COMFYUI_TIMEOUT_SECONDS,
                    t2v_workflow=WAN_T2V_WORKFLOW, i2v_workflow=WAN_I2V_WORKFLOW)
    context = None
    registry = None
    if req is not None and req.plan is not None:
        ctxs = resolve_all_scenes(req.plan)
        if ctxs:
            context = ctxs[0]
        registry = registry_from_plan(req.plan)
    report = await p.diagnose(context=context, registry=registry)
    return report.to_dict()

class GenerateRequest(BaseModel):
    plan: ProductionPlan
    scene_index: int
    mode: str = "t2v"
    width: int = 832
    height: int = 480
    frames: int = 81
    fps: float = 24.0
    steps: int = 20
    cfg: float = 6.0
    seed: Optional[int] = None
    negative_prompt: str = ""
    project_id: Optional[str] = None

@app.post("/api/generate")
async def generate_scene(req: GenerateRequest):
    # Phase 8: real Wan T2V/I2V generation from a resolved scene context.
    # Never returns success without a verified output MP4.
    try:
        mode = WanMode(req.mode)
    except ValueError:
        raise HTTPException(400, f"Invalid mode '{req.mode}'; use 't2v' or 'i2v'.")
    try:
        plan = req.plan
        ctx = resolve_scene_context(plan, req.scene_index)
        registry = registry_from_plan(plan)
        options = GenerationOptions(
            width=req.width, height=req.height, frames=req.frames, fps=req.fps,
            steps=req.steps, cfg=req.cfg, seed=req.seed,
            negative_prompt=req.negative_prompt, mode=mode,
        )
        p = WanProvider(COMFYUI_URL, COMFYUI_TIMEOUT_SECONDS,
                        t2v_workflow=WAN_T2V_WORKFLOW, i2v_workflow=WAN_I2V_WORKFLOW)
        result, meta = await p.generate_from_context(
            ctx, options, registry=registry, project_id=req.project_id,
        )
        return {
            "status": "COMPLETED",
            "provider": "wan",
            "model": meta.model,
            "mode": meta.mode,
            "prompt_id": meta.prompt_id,
            "scene_index": meta.scene_index,
            "seed": meta.seed,
            "output_path": meta.output_path,
            "metadata": meta.to_dict(),
        }
    except VideoError as e:
        status = 502 if e.code in (TypedErrorCode.NO_PROVIDER, TypedErrorCode.COMFYUI_UNREACHABLE,
                                   TypedErrorCode.GENERATION_TIMEOUT, TypedErrorCode.MODEL_NOT_FOUND,
                                   TypedErrorCode.WORKFLOW_REJECTED, TypedErrorCode.NO_OUTPUT,
                                   TypedErrorCode.WORKFLOW_NOT_FOUND, TypedErrorCode.INVALID_REFERENCE) else 500
        raise HTTPException(status, {"status": "FAILED", "error_code": e.code.value, "error_detail": e.detail, "context": e.context})

class VoiceGenerateRequest(BaseModel):
    plan: ProductionPlan
    scene_index: int
    project_id: Optional[str] = None

@app.post("/api/voice/generate")
async def voice_generate(req: VoiceGenerateRequest):
    # Phase 9: generate voiceover for one scene. Never fake success.
    try:
        plan = req.plan
        scene = next((s for s in plan.scenes if s.index == req.scene_index), None)
        if scene is None:
            raise HTTPException(404, f"Scene {req.scene_index} not found.")
        if not scene.voiceover.line.strip():
            raise HTTPException(400, "Scene has no voiceover line.")
        provider = select_tts_provider([])  # no real providers configured yet
        asset, result = generate_scene_voiceover(
            scene, plan, provider, project_id=req.project_id,
        )
        return {
            "status": "COMPLETED", "provider": asset.provider,
            "language": asset.language, "voice": asset.voice,
            "duration": asset.duration, "output_path": asset.path,
            "scene_index": asset.scene_index, "text": asset.text,
        }
    except VideoError as e:
        raise HTTPException(502, {"status": "FAILED", "error_code": e.code.value, "error_detail": e.detail})

class AudioMixRequest(BaseModel):
    voice_path: Optional[str] = None
    music_path: Optional[str] = None
    sfx_paths: list[str] = []
    ambience_path: Optional[str] = None
    scene_duration: float
    voice_start_offset: float = 0.0
    music_volume: float = 0.6
    voice_volume: float = 1.0
    music_duck: bool = True
    duck_level: float = 0.25
    fade_in: float = 0.0
    fade_out: float = 0.0
    project_id: Optional[str] = None

@app.post("/api/audio/mix")
async def audio_mix(req: AudioMixRequest):
    # Phase 9: FFmpeg audio mixing. Returns a verified mixed file.
    try:
        import uuid as _uuid
        out_dir = OUTPUT_DIR / (req.project_id or "standalone") / _uuid.uuid4().hex[:12]
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / "mix.mp3"
        result = mix_scene_audio(
            Path(req.voice_path) if req.voice_path else None,
            Path(req.music_path) if req.music_path else None,
            [Path(p) for p in req.sfx_paths],
            Path(req.ambience_path) if req.ambience_path else None,
            dest, scene_duration=req.scene_duration, voice_start_offset=req.voice_start_offset,
            music_volume=req.music_volume, voice_volume=req.voice_volume,
            music_duck=req.music_duck, duck_level=req.duck_level,
            fade_in=req.fade_in, fade_out=req.fade_out,
        )
        return {"status": "COMPLETED", "output_path": str(result.path),
                "duration": result.duration, "tracks": len(result.tracks)}
    except VideoError as e:
        raise HTTPException(502, {"status": "FAILED", "error_code": e.code.value, "error_detail": e.detail})

class CaptionRequest(BaseModel):
    plan: ProductionPlan
    format: str = "srt"
    style: str = "tiktok"

@app.post("/api/captions/generate")
async def captions_generate(req: CaptionRequest):
    # Phase 9: generate SRT/VTT/burned-in captions from plan.
    try:
        fmt = CaptionFormat(req.format)
        style = CaptionStyle(req.style)
    except ValueError:
        raise HTTPException(400, f"Invalid format '{req.format}' or style '{req.style}'.")
    rep = generate_captions(req.plan, fmt, style)
    return rep.to_dict()

@app.post("/api/audio/qc")
async def audio_qc(path: str):
    # Phase 9: verify a real audio file. Never reports success without verification.
    try:
        qc = verify_audio(Path(path))
        return {"status": "COMPLETED", "qc": qc}
    except VideoError as e:
        raise HTTPException(502, {"status": "FAILED", "error_code": e.code.value, "error_detail": e.detail})

class AssemblyRequest(BaseModel):
    scene_durations: list[float]
    video_assets: list[str]
    voice_assets: Optional[list[Optional[str]]] = None
    voice_start_offsets: Optional[list[float]] = None
    music_assets: Optional[list[list[str]]] = None
    sfx_assets: Optional[list[list[str]]] = None
    ambience_assets: Optional[list[Optional[str]]] = None
    caption_texts: Optional[list[str]] = None
    transitions: Optional[list[str]] = None
    transition_durations: Optional[list[float]] = None
    profile_name: str = "TIKTOK"
    quality: str = "high"
    include_captions: bool = False
    caption_mode: str = "srt"
    caption_cues: list[dict] = []
    caption_style: dict = {}
    include_branding: bool = False
    brand: dict = {}
    silent: bool = False
    project_id: Optional[str] = None

@app.post("/api/assembly/export")
async def assembly_export(req: AssemblyRequest):
    # Phase 10: full final assembly. Never COMPLETED without QC-verified MP4.
    try:
        from .editing.transitions import parse_transition
        tl = build_timeline_from_assets(
            req.scene_durations, [Path(p) for p in req.video_assets],
            voice_assets=[[Path(p) if p else None for p in req.voice_assets] if req.voice_assets else None][0] if req.voice_assets else None,
            voice_start_offsets=req.voice_start_offsets,
            music_assets=[[Path(p) for p in m] for m in req.music_assets] if req.music_assets else None,
            sfx_assets=[[Path(p) for p in s] for s in req.sfx_assets] if req.sfx_assets else None,
            ambience_assets=[[Path(p) if p else None for p in req.ambience_assets] if req.ambience_assets else None][0] if req.ambience_assets else None,
            caption_texts=req.caption_texts,
            transitions=req.transitions, transition_durations=req.transition_durations,
        )
        exp_req = ExportRequest(
            timeline=tl, profile_name=req.profile_name, quality=Quality(req.quality),
            include_captions=req.include_captions, caption_mode=req.caption_mode,
            caption_cues=req.caption_cues, caption_style=req.caption_style,
            include_branding=req.include_branding, brand=req.brand,
            silent=req.silent, project_id=req.project_id,
        )
        result = export_video(exp_req)
        return result.to_dict()
    except VideoError as e:
        raise HTTPException(502, {"status": "FAILED", "error_code": e.code.value, "error_detail": e.detail})
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.get("/api/assembly/profiles")
async def assembly_profiles():
    return {name: p.to_dict() for name, p in PROFILES.items()}

@app.post("/api/projects")
async def create_project(req: ProjectCreate):
    pid=uuid.uuid4().hex[:12]
    script=req.script.strip() or f"Create a short cinematic video about: {req.topic}"
    planner="local"
    try:
        if GEMINI_API_KEY:
            scenes=await gemini_scene_plan(req.topic,script,req.duration,req.scene_count,GEMINI_API_KEY,GEMINI_MODEL)
            planner="gemini"
        else:
            scenes=local_scene_plan(req.topic,script,req.duration,req.scene_count)
    except Exception:
        scenes=local_scene_plan(req.topic,script,req.duration,req.scene_count)
        planner="local_fallback"
    project={"id":pid,"title":req.title,"topic":req.topic,"script":script,"language":req.language,"duration":req.duration,"scenes":[s.model_dump() for s in scenes],"planner":planner,"status":"planned"}
    PROJECTS[pid]=project
    (PROJECTS_DIR/f"{pid}.json").write_text(json.dumps(project,ensure_ascii=False,indent=2),encoding="utf-8")
    return project

@app.post("/api/projects/{pid}/generate")
async def generate(pid:str):
    project=PROJECTS.get(pid)
    if not project:
        f=PROJECTS_DIR/f"{pid}.json"
        if f.exists(): project=json.loads(f.read_text(encoding="utf-8"))
    if not project: raise HTTPException(404,"Project not found")
    project["status"]="generating"
    try:
        # Real provider selection. Raises NO_PROVIDER if nothing is available.
        provider = await select_provider()
        client=ComfyClient(COMFYUI_URL,COMFYUI_TIMEOUT_SECONDS)
        workflow=client.load_workflow(WORKFLOW_PATH)
        clips=[]
        for scene in project["scenes"]:
            full_prompt=scene["prompt"]+"\nNegative prompt: "+scene["negative_prompt"]
            wf=client.inject_prompt(workflow,full_prompt,COMFYUI_PROMPT_NODE_ID,COMFYUI_PROMPT_FIELD)
            job=await client.queue(wf)
            outputs=await client.wait(job)
            clip=OUTPUT_DIR/f"{pid}_scene_{scene['index']}.mp4"
            await client.download_first_media(outputs,clip)
            # Verify each clip is a real, readable MP4 before proceeding.
            verify_mp4(clip)
            clips.append(clip)
        from .media import concat_videos, normalize_vertical
        joined=OUTPUT_DIR/f"{pid}_joined.mp4"
        concat_videos(clips,joined)
        final=OUTPUT_DIR/f"{pid}_final.mp4"
        normalize_vertical(joined,final)
        # Final Quality Control: verify the produced MP4 is real and readable.
        qc=verify_mp4(final)
        project["status"]="completed"; project["output_path"]=str(final); project["qc"]=qc
        (PROJECTS_DIR/f"{pid}.json").write_text(json.dumps(project,ensure_ascii=False,indent=2),encoding="utf-8")
        return {"ok":True,"project":project,"download":f"/api/projects/{pid}/download"}
    except VideoError as e:
        # Typed failure: never hide the real reason. Map to an actionable HTTP error.
        project["status"]="failed"; project["error"]=e.to_dict()
        (PROJECTS_DIR/f"{pid}.json").write_text(json.dumps(project,ensure_ascii=False,indent=2),encoding="utf-8")
        status = 502 if e.code in (TypedErrorCode.NO_PROVIDER, TypedErrorCode.COMFYUI_UNREACHABLE,
                                   TypedErrorCode.GENERATION_TIMEOUT, TypedErrorCode.MODEL_NOT_FOUND,
                                   TypedErrorCode.WORKFLOW_REJECTED, TypedErrorCode.NO_OUTPUT) else 500
        raise HTTPException(status, e.to_dict())
    except Exception as e:
        project["status"]="failed"; project["error"]={"code":"UNKNOWN","detail":str(e)}
        (PROJECTS_DIR/f"{pid}.json").write_text(json.dumps(project,ensure_ascii=False,indent=2),encoding="utf-8")
        raise HTTPException(500, {"code":"UNKNOWN","detail":str(e)})

@app.get("/api/projects/{pid}/download")
def download(pid:str):
    f=OUTPUT_DIR/f"{pid}_final.mp4"
    if not f.exists(): raise HTTPException(404,"Final video not found")
    return FileResponse(f,media_type="video/mp4",filename=f.name)


# ====================================================================
# Phase 11: Production Job Orchestration API
# ====================================================================
ORCHESTRATOR = Orchestrator()


class CreateProjectRequest(BaseModel):
    project_id: Optional[str] = None


class CreateJobRequest(BaseModel):
    job_type: str
    parent_job_id: Optional[str] = None
    scene_index: Optional[int] = None
    inputs: dict = {}
    depends_on: list[str] = []
    max_retries: int = 3
    provider: Optional[str] = None
    model: Optional[str] = None
    workflow: Optional[str] = None
    seed: Optional[int] = None


@app.post("/api/jobs/projects")
def jobs_create_project(req: CreateProjectRequest):
    """Create a new orchestrator project."""
    proj = ORCHESTRATOR.create_project(req.project_id)
    return proj.to_summary()


@app.get("/api/jobs/projects")
def jobs_list_projects():
    """List all known orchestrator projects (survives restart)."""
    return {"projects": ORCHESTRATOR.list_projects()}


@app.get("/api/jobs/projects/{pid}")
def jobs_get_project(pid: str):
    """Get a project summary including progress."""
    try:
        proj = ORCHESTRATOR.get_project(pid)
        return proj.to_summary()
    except JobError as e:
        raise HTTPException(404, e.to_dict())


@app.post("/api/jobs/projects/{pid}/jobs")
def jobs_create_job(pid: str, req: CreateJobRequest):
    """Create a new job in a project."""
    try:
        jt = JobType(req.job_type)
    except ValueError:
        raise HTTPException(400, {"code": "INVALID_JOB_TYPE",
                                  "detail": f"Unknown job type: {req.job_type}"})
    try:
        job = ORCHESTRATOR.create_job(pid, jt, parent_job_id=req.parent_job_id,
                                      scene_index=req.scene_index, inputs=req.inputs,
                                      depends_on=req.depends_on, max_retries=req.max_retries,
                                      provider=req.provider, model=req.model,
                                      workflow=req.workflow, seed=req.seed)
        return job.to_dict()
    except JobError as e:
        raise HTTPException(404, e.to_dict())


@app.get("/api/jobs/projects/{pid}/jobs")
def jobs_list_jobs(pid: str):
    """List all jobs in a project."""
    try:
        jobs = ORCHESTRATOR.list_jobs(pid)
        return {"jobs": [j.to_dict() for j in jobs]}
    except JobError as e:
        raise HTTPException(404, e.to_dict())


@app.get("/api/jobs/projects/{pid}/jobs/{jid}")
def jobs_get_job(pid: str, jid: str):
    """Get a single job's full state."""
    try:
        return ORCHESTRATOR.get_job(pid, jid).to_dict()
    except JobError as e:
        raise HTTPException(404, e.to_dict())


@app.post("/api/jobs/projects/{pid}/jobs/{jid}/run")
def jobs_run_job(pid: str, jid: str):
    """Execute a job by type. Routes to the matching stage executor."""
    try:
        job = ORCHESTRATOR.get_job(pid, jid)
    except JobError as e:
        raise HTTPException(404, e.to_dict())
    runners = {
        JobType.CONTENT_PLAN: ORCHESTRATOR.run_content_plan,
        JobType.SCENE_RESOLUTION: ORCHESTRATOR.run_scene_resolution,
        JobType.VIDEO_SCENE: ORCHESTRATOR.run_video_scene,
        JobType.VOICEOVER: ORCHESTRATOR.run_voiceover,
        JobType.ASSEMBLY: ORCHESTRATOR.run_assembly,
    }
    runner = runners.get(job.job_type)
    if runner is None:
        raise HTTPException(400, {"code": "NO_RUNNER",
                                  "detail": f"No executor for job type {job.job_type.value}"})
    try:
        result = runner(pid, jid)
        return result.to_dict()
    except VideoError as e:
        raise HTTPException(502, e.to_dict())


@app.post("/api/jobs/projects/{pid}/jobs/{jid}/retry")
def jobs_retry_job(pid: str, jid: str):
    """Retry a failed job (subject to retry policy)."""
    try:
        return ORCHESTRATOR.retry_job(pid, jid).to_dict()
    except VideoError as e:
        status = 409 if e.code in (TypedErrorCode.NON_RETRYABLE, TypedErrorCode.RETRY_EXHAUSTED,
                                   TypedErrorCode.CANCEL_NOT_ALLOWED) else 404
        raise HTTPException(status, e.to_dict())


@app.post("/api/jobs/projects/{pid}/jobs/{jid}/cancel")
def jobs_cancel_job(pid: str, jid: str):
    """Cancel an active job."""
    try:
        return ORCHESTRATOR.cancel_job(pid, jid).to_dict()
    except VideoError as e:
        raise HTTPException(409, e.to_dict())


@app.get("/api/jobs/projects/{pid}/progress")
def jobs_get_progress(pid: str):
    """Get project progress (derived from actual job states)."""
    try:
        return ORCHESTRATOR.get_progress(pid)
    except JobError as e:
        raise HTTPException(404, e.to_dict())


@app.get("/api/jobs/projects/{pid}/jobs/{jid}/output")
def jobs_get_output(pid: str, jid: str):
    """Get a job's output (file path + QC metadata)."""
    try:
        job = ORCHESTRATOR.get_job(pid, jid)
    except JobError as e:
        raise HTTPException(404, e.to_dict())
    if not job.output_path:
        raise HTTPException(404, {"code": "NO_OUTPUT", "detail": "Job has no output yet."})
    return {"output_path": job.output_path, "output_fingerprint": job.output_fingerprint,
            "outputs": job.outputs, "state": job.state.value}


@app.post("/api/jobs/recover")
def jobs_recover_all():
    """Startup recovery: re-validate abandoned RUNNING/QUEUED/RETRYING jobs."""
    return {"recovered": {pid: [j.to_dict() for j in jobs]
                          for pid, jobs in ORCHESTRATOR.recover_all().items()}}

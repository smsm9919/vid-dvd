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

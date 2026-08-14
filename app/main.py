import json, uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from .config import *
from .models import ProjectCreate
from .planner import local_scene_plan, gemini_scene_plan
from .comfy import ComfyClient, ComfyError
from .media import ffmpeg_available

app=FastAPI(title="VideoFactory Local", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
PROJECTS={}

@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR/"index.html").read_text(encoding="utf-8")

@app.get("/api/health")
async def health():
    comfy={"ok":False}
    try: comfy={"ok":True,"data":await ComfyClient(COMFYUI_URL,5).health()}
    except Exception as e: comfy={"ok":False,"error":str(e)}
    return {"app":"ok","ffmpeg":ffmpeg_available(),"comfyui":comfy,"workflow_exists":WORKFLOW_PATH.exists()}

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
    client=ComfyClient(COMFYUI_URL,COMFYUI_TIMEOUT_SECONDS)
    try:
        workflow=client.load_workflow(WORKFLOW_PATH)
        clips=[]
        for scene in project["scenes"]:
            full_prompt=scene["prompt"]+"\nNegative prompt: "+scene["negative_prompt"]
            wf=client.inject_prompt(workflow,full_prompt,COMFYUI_PROMPT_NODE_ID,COMFYUI_PROMPT_FIELD)
            job=await client.queue(wf)
            outputs=await client.wait(job)
            clip=OUTPUT_DIR/f"{pid}_scene_{scene['index']}.mp4"
            await client.download_first_media(outputs,clip)
            clips.append(clip)
        from .media import concat_videos, normalize_vertical
        joined=OUTPUT_DIR/f"{pid}_joined.mp4"
        concat_videos(clips,joined)
        final=OUTPUT_DIR/f"{pid}_final.mp4"
        normalize_vertical(joined,final)
        project["status"]="completed"; project["output_path"]=str(final)
        (PROJECTS_DIR/f"{pid}.json").write_text(json.dumps(project,ensure_ascii=False,indent=2),encoding="utf-8")
        return {"ok":True,"project":project,"download":f"/api/projects/{pid}/download"}
    except ComfyError as e:
        project["status"]="failed"
        raise HTTPException(502,str(e))
    except Exception as e:
        project["status"]="failed"
        raise HTTPException(500,str(e))

@app.get("/api/projects/{pid}/download")
def download(pid:str):
    f=OUTPUT_DIR/f"{pid}_final.mp4"
    if not f.exists(): raise HTTPException(404,"Final video not found")
    return FileResponse(f,media_type="video/mp4",filename=f.name)

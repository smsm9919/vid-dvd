import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", "8090"))
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
COMFYUI_WORKFLOW = os.getenv("COMFYUI_WORKFLOW", "workflows/wan22_ti2v_api.json")
COMFYUI_PROMPT_NODE_ID = os.getenv("COMFYUI_PROMPT_NODE_ID", "").strip()
COMFYUI_PROMPT_FIELD = os.getenv("COMFYUI_PROMPT_FIELD", "text").strip()
COMFYUI_TIMEOUT_SECONDS = int(os.getenv("COMFYUI_TIMEOUT_SECONDS", "1800"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
VIDEO_PROVIDERS = os.getenv("VIDEO_PROVIDERS", "comfyui").strip()
WORKFLOW_PATH = ROOT / COMFYUI_WORKFLOW
OUTPUT_DIR = ROOT / "output"
PROJECTS_DIR = ROOT / "projects"
STATIC_DIR = ROOT / "static"
OUTPUT_DIR.mkdir(exist_ok=True)
PROJECTS_DIR.mkdir(exist_ok=True)

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
# Wan 2.2 (Phase 8)
WAN_T2V_WORKFLOW = ROOT / os.getenv("WAN_T2V_WORKFLOW", "workflows/wan22_t2v_api.json")
WAN_I2V_WORKFLOW = ROOT / os.getenv("WAN_I2V_WORKFLOW", "workflows/wan22_i2v_api.json")
WAN_REQUIRED_MODEL = os.getenv("WAN_REQUIRED_MODEL", "wan2.2").strip()
OUTPUT_DIR = ROOT / "output"
PROJECTS_DIR = ROOT / "projects"
STATIC_DIR = ROOT / "static"
OUTPUT_DIR.mkdir(exist_ok=True)
PROJECTS_DIR.mkdir(exist_ok=True)

# ---- Phase 13: free-first multi-provider hub ----
# When true (default), paid providers are never selected automatically. A paid
# provider may only run when ALLOW_PAID_PROVIDERS=true AND the per-call cost is
# within MAX_PAID_COST_USD.
FREE_FIRST = os.getenv("FREE_FIRST", "true").strip().lower() in ("1", "true", "yes", "on")
ALLOW_PAID_PROVIDERS = os.getenv("ALLOW_PAID_PROVIDERS", "false").strip().lower() in ("1", "true", "yes", "on")
try:
    MAX_PAID_COST_USD = float(os.getenv("MAX_PAID_COST_USD", "0"))
except ValueError:
    MAX_PAID_COST_USD = 0.0

# Stock provider API keys (free tiers; keys stored only in env, never logged).
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "").strip()
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "").strip()

# Local asset cache for downloaded stock media (never committed to Git).
ASSET_CACHE_DIR = ROOT / os.getenv("ASSET_CACHE_DIR", "assets_cache")
ASSET_LIBRARY_DIR = ROOT / os.getenv("ASSET_LIBRARY_DIR", "assets_library")
ASSET_CACHE_DIR.mkdir(exist_ok=True)
ASSET_LIBRARY_DIR.mkdir(exist_ok=True)

# Per-provider HTTP settings (timeouts in seconds).
STOCK_HTTP_TIMEOUT = int(os.getenv("STOCK_HTTP_TIMEOUT", "30"))
STOCK_MAX_DOWNLOAD_BYTES = int(os.getenv("STOCK_MAX_DOWNLOAD_BYTES", str(200 * 1024 * 1024)))  # 200 MB cap

# Piper TTS (local, CPU, GPL-3.0 — optional adapter, voices downloaded on demand).
PIPER_ENABLED = os.getenv("PIPER_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
PIPER_VOICES_DIR = ROOT / os.getenv("PIPER_VOICES_DIR", "models/piper_voices")
PIPER_DEFAULT_VOICE_EN = os.getenv("PIPER_DEFAULT_VOICE_EN", "en_US-lessac-medium").strip()
PIPER_DEFAULT_VOICE_DE = os.getenv("PIPER_DEFAULT_VOICE_DE", "de_DE-thorsten-medium").strip()
PIPER_DEFAULT_VOICE_AR = os.getenv("PIPER_DEFAULT_VOICE_AR", "").strip()

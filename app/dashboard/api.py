"""Dashboard backend endpoints (Phase 12).

Thin API layer backed by existing domain logic. The frontend never duplicates
business logic — every dashboard view derives from real backend state:
- provider readiness (real detection, never fabricated)
- production readiness (CONTENT/VIDEO/VOICE/AUDIO/CAPTIONS/ASSEMBLY/QC)
- ad variants (real generation + heuristic scoring, never conversion rates)
- scene board + continuity (real resolution + validation)
- job control center (Phase 11 orchestrator as source of truth)
- asset library (real file introspection + QC)
- final video (real QC-verified MP4 or honest NOT AVAILABLE)
- logs (structured events, no secrets)

No fake state. No secret exposure.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Optional

from ..ads.brief import AdBrief
from ..ads.scoring import compare_variants, score_variant
from ..ads.variants import generate_variants
from ..audio.music import NullMusicProvider
from ..audio.qc import probe_audio, verify_audio
from ..audio.sfx import NullSFXProvider
from ..brain.content_brain import local_content_plan
from ..brain.models import ContentBrief
from ..core.errors import VideoError
from ..editing.profiles import PROFILES, Quality, get_profile
from ..media import ffmpeg_available, ffprobe_available, verify_mp4
from ..scene.continuity import resolve_all_scenes, validate_continuity
from ..voice.tts import NullTTSProvider, select_tts_provider


# ---------------------------------------------------------------- provider status
def _ffmpeg_status() -> dict[str, Any]:
    if not ffmpeg_available():
        return {"status": "NOT_AVAILABLE", "reason": "ffmpeg binary not found on PATH."}
    try:
        out = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=True, timeout=5)
        version = out.stdout.split("\n")[0] if out.stdout else "unknown"
        return {"status": "READY", "version": version}
    except Exception as e:
        return {"status": "ERROR", "reason": str(e)}


async def _comfyui_status() -> dict[str, Any]:
    from ..providers.registry import build_providers, provider_status
    providers = build_providers()
    statuses = await provider_status(providers)

    def _normalize(p: Optional[dict]) -> dict[str, Any]:
        if p is None:
            return {"status": "NOT_CONFIGURED", "reason": "Provider not registered."}
        if p.get("ok"):
            return {"status": "READY", "name": p.get("name"), "url": p.get("url")}
        err = p.get("error", {})
        return {"status": "NOT_AVAILABLE", "name": p.get("name"), "url": p.get("url"),
                "reason": err.get("detail", "Provider unreachable."),
                "error_code": err.get("code")}

    comfy = next((p for p in statuses if p.get("name") == "comfyui"), None)
    wan = next((p for p in statuses if p.get("name") == "wan"), None)
    return {"comfyui": _normalize(comfy), "wan": _normalize(wan)}


def _tts_status() -> dict[str, Any]:
    from ..providers.router import build_tts_providers
    provider = select_tts_provider(build_tts_providers())
    if isinstance(provider, NullTTSProvider):
        # Distinguish Piper disabled vs uninstalled for actionable diagnostics.
        from .. import config
        if config.PIPER_ENABLED:
            import shutil
            if shutil.which("piper"):
                return {"status": "NOT_CONFIGURED",
                        "reason": "Piper enabled but no voice models found for the requested language."}
            return {"status": "BLOCKED", "reason": "PIPER_ENABLED=true but piper binary not on PATH. Install: pip install piper-tts"}
        return {"status": "NOT_CONFIGURED",
                "reason": "No TTS provider configured. Enable Piper (PIPER_ENABLED=true, GPL-3.0) for free local TTS."}
    meta = getattr(provider, "meta", None)
    license_info = meta().license.to_dict() if callable(meta) else None
    return {"status": "READY", "provider": getattr(provider, "name", type(provider).__name__),
            "license": license_info}


def _music_status() -> dict[str, Any]:
    return {"status": "NOT_CONFIGURED", "reason": "No real music provider registered (only NullMusicProvider)."}


def _sfx_status() -> dict[str, Any]:
    return {"status": "NOT_CONFIGURED", "reason": "No real SFX provider registered (only NullSFXProvider)."}


async def provider_panel() -> dict[str, Any]:
    """Real provider readiness panel. Never claims ready merely because config exists."""
    comfy = await _comfyui_status()
    from ..providers.router import build_default_router
    from ..providers.stock_adapters import build_stock_providers
    router = build_default_router()
    stock = []
    for p in build_stock_providers():
        m = p.meta()
        stock.append({
            "name": p.name, "available": p.available,
            "license": m.license.to_dict(), "cost": m.cost.to_dict(),
        })
    return {
        "ffmpeg": _ffmpeg_status(),
        "comfyui": comfy["comfyui"],
        "wan": comfy["wan"],
        "tts": _tts_status(),
        "music": _music_status(),
        "sfx": _sfx_status(),
        # Phase 13 additions
        "stock": stock,
        "router": router.status(),
        "free_first": _free_first_status(),
    }


def _free_first_status() -> dict[str, Any]:
    """Honest free-first policy + cost-guard summary."""
    from .. import config
    return {
        "free_first": config.FREE_FIRST,
        "allow_paid_providers": config.ALLOW_PAID_PROVIDERS,
        "max_paid_cost_usd": config.MAX_PAID_COST_USD,
        "policy": ("Paid providers are blocked (FREE_FIRST). "
                   if config.FREE_FIRST and not config.ALLOW_PAID_PROVIDERS
                   else "Paid providers allowed within budget. "
                   if config.ALLOW_PAID_PROVIDERS
                   else "Free-first off."),
    }


# ---------------------------------------------------------------- readiness
async def production_readiness() -> dict[str, Any]:
    """Production readiness summary derived from real backend diagnostics.

    CONTENT: ready if local planner available (always; Gemini is optional).
    VIDEO: ready if a real video provider (ComfyUI/Wan) is healthy.
    VOICE: ready if a real (non-Null) TTS provider is configured.
    AUDIO: ready if music AND sfx providers are configured (mixing needs FFmpeg too).
    CAPTIONS: ready if FFmpeg (for burned-in) is available.
    ASSEMBLY: ready if FFmpeg is available.
    QC: ready if FFmpeg + ffprobe are available.
    """
    panel = await provider_panel()

    def ok(name: str) -> bool:
        return panel.get(name, {}).get("status") == "READY"

    content = "READY"
    # VIDEO: AI generation (ComfyUI/Wan) OR stock footage can supply video.
    stock_any = any(s.get("available") for s in panel.get("stock", []))
    video = "READY" if (ok("comfyui") or ok("wan") or stock_any) else "BLOCKED"
    stock_stage = "READY" if stock_any else "BLOCKED"
    voice = "READY" if ok("tts") else "BLOCKED"
    audio = "READY" if ok("music") and ok("sfx") and ok("ffmpeg") else "BLOCKED"
    captions = "READY" if ok("ffmpeg") else "BLOCKED"
    assembly = "READY" if ok("ffmpeg") else "BLOCKED"
    qc = "READY" if ok("ffmpeg") and ffprobe_available() else "BLOCKED"
    # Overall is READY only when a full free-first pipeline is possible without AI generation:
    # content + (stock OR video-gen) + voice + assembly + QC. Audio (music/sfx) is optional
    # for a minimal ad (silent or voice-only is still a valid MP4 after assembly).
    core = [content, video, voice, captions, assembly, qc]
    overall = "READY" if all(s == "READY" for s in core) else "NOT_READY"
    return {
        "overall": overall,
        "content": content, "video": video, "stock": stock_stage, "voice": voice,
        "audio": audio, "captions": captions, "assembly": assembly, "qc": qc,
        "providers": panel,
    }


# ---------------------------------------------------------------- variants
def build_ad_variants(brief: AdBrief) -> dict[str, Any]:
    """Generate all 7 ad variants with heuristic scores + claim checks.

    Scores are creative-quality heuristics, NOT predicted conversion rates.
    """
    results = generate_variants(brief)
    scored = []
    for r in results:
        score = score_variant(r, brief)
        scored.append({**r.to_dict(), "score": score.to_dict()})
    comparison = compare_variants(results, brief)
    return {
        "variants": scored,
        "comparison": comparison.to_dict() if hasattr(comparison, "to_dict") else {},
        "note": "Scores are heuristic creative-quality indicators, not predicted conversion rates.",
    }


# ---------------------------------------------------------------- scene board + continuity
def scene_board(plan) -> dict[str, Any]:
    """Resolve all scenes + validate continuity. Returns the scene board + report."""
    contexts = resolve_all_scenes(plan)
    report = validate_continuity(plan)
    return {
        "scenes": [c.to_dict() for c in contexts],
        "continuity": report.to_dict(),
        "scene_count": len(contexts),
    }


# ---------------------------------------------------------------- asset introspection
def _probe_video(path: Path) -> dict[str, Any]:
    """Probe a verified MP4 for dashboard display (duration/res/fps/codecs)."""
    try:
        report = verify_mp4(path)
        return {"duration": report.get("duration"), "width": report.get("width"),
                "height": report.get("height"), "fps": report.get("fps"),
                "video_codec": report.get("video_codec"), "audio_codec": report.get("audio_codec")}
    except Exception:
        return {}


def list_assets(project_dir: Path) -> list[dict[str, Any]]:
    """List real assets in a project's assets dir with QC state.

    Never exposes an asset as valid if it fails its relevant QC.
    """
    assets: list[dict[str, Any]] = []
    if not project_dir.exists():
        return assets
    for p in sorted(project_dir.rglob("*")):
        if not p.is_file() or p.suffix not in (".mp4", ".mp3", ".wav", ".srt", ".vtt", ".ass", ".json", ".png", ".jpg"):
            continue
        if p.name in ("jobs.json", "cache.json"):
            continue
        size = p.stat().st_size
        entry: dict[str, Any] = {
            "filename": p.name, "type": p.suffix.lstrip("."), "size": size, "qc": "UNKNOWN",
        }
        try:
            if p.suffix == ".mp4":
                verify_mp4(p)
                entry["qc"] = "PASS"
                entry.update(_probe_video(p))
            elif p.suffix in (".mp3", ".wav"):
                verify_audio(p)
                entry["qc"] = "PASS"
                info = probe_audio(p)
                entry["duration"] = info.get("duration")
                entry["audio_codec"] = info.get("codec_name")
            elif p.suffix in (".srt", ".vtt", ".ass"):
                content = p.read_text(encoding="utf-8")
                entry["qc"] = "PASS" if len(content.strip()) > 0 else "FAIL"
                entry["lines"] = content.count("\n")
            else:
                entry["qc"] = "PASS" if size > 0 else "FAIL"
        except VideoError as e:
            entry["qc"] = "FAIL"
            entry["qc_error"] = e.code.value
        except Exception as e:
            entry["qc"] = "FAIL"
            entry["qc_error"] = str(e)[:120]
        assets.append(entry)
    return assets


# ---------------------------------------------------------------- final video
def final_video_info(project_dir: Path) -> dict[str, Any]:
    """Find a real QC-verified final MP4, or report NOT AVAILABLE with the blocker.

    Never shows a fake preview. Never claims a final video exists without QC pass.
    """
    candidates: list[Path] = []
    if project_dir.exists():
        for p in project_dir.rglob("final_video.mp4"):
            candidates.append(p)
    if not candidates:
        return {"available": False, "reason": "No final video has been produced yet."}
    path = max(candidates, key=lambda p: p.stat().st_mtime)
    if not path.exists() or path.stat().st_size <= 0:
        return {"available": False, "reason": f"Final video file is empty or missing: {path.name}"}
    try:
        verify_mp4(path)
    except VideoError as e:
        return {"available": False, "reason": f"Final video failed QC: {e.code.value} — {e.detail}",
                "filename": path.name}
    probe = _probe_video(path)
    return {
        "available": True, "filename": path.name, "size": path.stat().st_size,
        "qc": "PASS", **probe,
    }


# ---------------------------------------------------------------- logs
class _MemoryLogHandler(logging.Handler):
    """Captures the last N log records for the dashboard log viewer."""

    def __init__(self, capacity: int = 500) -> None:
        super().__init__()
        self.capacity = capacity
        self.records: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        stage = ""
        body = msg
        if msg.startswith("[") and "]" in msg:
            stage = msg[1:msg.index("]")]
            body = msg[msg.index("]") + 1:].strip()
        self.records.append({
            "stage": stage, "message": body,
            "level": record.levelname, "time": record.created,
        })
        if len(self.records) > self.capacity:
            self.records = self.records[-self.capacity:]


def get_logs() -> list[dict[str, Any]]:
    """Return recent structured log events. No secrets/tokens are logged."""
    logger = logging.getLogger("videofactory")
    for h in logger.handlers:
        if isinstance(h, _MemoryLogHandler):
            return list(h.records[-200:])
    return []


_INSTALLED = False


def install_log_capture() -> None:
    """Attach the in-memory log handler (idempotent). Call at app startup."""
    global _INSTALLED
    if _INSTALLED:
        return
    logger = logging.getLogger("videofactory")
    if not any(isinstance(h, _MemoryLogHandler) for h in logger.handlers):
        logger.addHandler(_MemoryLogHandler())
    _INSTALLED = True

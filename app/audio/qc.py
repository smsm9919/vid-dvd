"""Audio quality control (Phase 9).

Verifies real audio files exist, are non-empty, and are decodable by FFmpeg.
Never reports audio generation as successful without a real, verified output.

Mirrors the rigor of :mod:`app.media` (video QC) but for audio assets (voice,
music, SFX, ambience, final mixes).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Optional

from ..core.errors import TypedErrorCode, VideoError
from ..media import ffprobe_available


class AudioError(VideoError):
    """Typed audio failure (INVALID_AUDIO / FFMPEG_ERROR)."""

    def __init__(self, code: TypedErrorCode, detail: str = "", *, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(code, detail, context=context)


def probe_audio(path: Path) -> dict[str, Any]:
    """Probe an audio file and return a structured QC report.

    Raises AudioError(INVALID_AUDIO) if the file is missing/empty/has no audio
    stream, or AudioError(FFMPEG_ERROR) if ffprobe is unavailable.
    """
    path = Path(path)
    if not path.exists():
        raise AudioError(
            TypedErrorCode.INVALID_AUDIO,
            f"Audio file does not exist: {path}",
            context={"path": str(path)},
        )
    size = path.stat().st_size
    if size <= 0:
        raise AudioError(
            TypedErrorCode.INVALID_AUDIO,
            f"Audio file is empty (0 bytes): {path}",
            context={"path": str(path), "size": size},
        )
    if not ffprobe_available():
        raise AudioError(
            TypedErrorCode.FFMPEG_ERROR,
            "ffprobe not found on PATH; cannot verify audio.",
            context={"path": str(path)},
        )
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams",
         "-of", "json", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if p.returncode != 0:
        raise AudioError(
            TypedErrorCode.INVALID_AUDIO,
            f"ffprobe cannot decode audio: {p.stderr[-1000:].strip()}",
            context={"path": str(path), "stderr": p.stderr[-500:]},
        )
    try:
        data = json.loads(p.stdout or "{}")
    except json.JSONDecodeError as e:
        raise AudioError(
            TypedErrorCode.INVALID_AUDIO,
            f"ffprobe returned non-JSON: {e}",
            context={"path": str(path)},
        )
    streams = data.get("streams", [])
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise AudioError(
            TypedErrorCode.INVALID_AUDIO,
            "No audio stream found.",
            context={"path": str(path), "streams": streams},
        )
    fmt = data.get("format", {})
    return {
        "path": str(path),
        "size": size,
        "ok": True,
        "codec": audio.get("codec_name"),
        "sample_rate": int(audio.get("sample_rate") or 0),
        "channels": int(audio.get("channels") or 0),
        "duration": float(fmt.get("duration") or audio.get("duration") or 0),
        "bit_rate": int(fmt.get("bit_rate") or 0),
    }


def verify_audio(path: Path) -> dict[str, Any]:
    """Verify an audio file is real and decodable. Returns a QC report.

    Raises AudioError on any failure. Never reports success without a verified
    real audio file.
    """
    return probe_audio(path)


def generate_silent_audio(path: Path, duration: float, *, sample_rate: int = 44100) -> Path:
    """Generate a deterministic silent audio file with FFmpeg.

    Used for testing the audio pipeline and as a safe placeholder when a TTS
    provider is unavailable — NOT a substitute for real TTS output. The caller
    must never present this as real voiceover.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"anullsrc=channel_layout=stereo:sample_rate={sample_rate}",
         "-t", str(duration), "-c:a", "libmp3lame", "-b:a", "192k",
         str(path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    return path


def generate_tone_audio(path: Path, duration: float, *, freq: float = 440.0,
                        sample_rate: int = 44100) -> Path:
    """Generate a deterministic sine-tone audio file with FFmpeg.

    For deterministic music/SFX test fixtures only — never reported as real
    provider output.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"sine=frequency={freq}:sample_rate={sample_rate}",
         "-t", str(duration), "-c:a", "libmp3lame", "-b:a", "192k",
         str(path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    return path

"""Piper TTS provider — local, CPU-friendly (Phase 13).

Piper is a neural TTS that runs in real time on CPU (verified on this machine:
~3.4s WAV synthesized from text, no GPU). It is the free-first TTS for the hub.

LICENSING (critical): the maintained ``piper-tts`` PyPI package (OHF-Voice/
piper1-gpl, v1.7.0) is **GPL-3.0**. The original rhasspy/piper (MIT) was
archived Oct 2025. Because this project's code is not GPL, Piper is an OPTIONAL
adapter: it is only loaded when PIPER_ENABLED=true (explicit opt-in), the
license is recorded on every generated asset, and Piper voice models are never
bundled into the repository (downloaded on demand to models/piper_voices).

Runtime contract: ``synthesize`` returns a real, non-empty, FFmpeg-decodable
WAV. It never fakes synthesis. If Piper is disabled, the missing model, or
synthesis fails, it raises a typed error (NO_PROVIDER / MODEL_NOT_FOUND /
NO_OUTPUT).
"""

from __future__ import annotations

import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Optional

from .. import config
from ..core.errors import TypedErrorCode, VideoError
from ..core.logging import log
from .tts import SUPPORTED_LANGUAGES, TTSProvider, VoiceRequest, VoiceResult
from ..providers.contracts import (
    Capability,
    ProviderCapability,
    ProviderCost,
    ProviderKind,
    ProviderLicense,
    ProviderMeta,
    ProviderRuntime,
)

# Official Piper voice model repository (Hugging Face, rhasspy/piper-voices).
_PIPER_VOICE_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# GPL-3.0 — recorded on every asset so downstream licensing is explicit.
_PIPER_LICENSE = ProviderLicense(
    name="Piper (piper-tts, GPL-3.0)",
    spdx="GPL-3.0-only",
    commercial_use="restricted",
    attribution_required=True,
    attribution_text="Voice synthesis by Piper (piper-tts, GPL-3.0). "
                     "Commercial use requires GPL-3.0 compliance.",
    source_url="https://github.com/OHF-Voice/piper1-gpl",
)

# Default voice model names per language (medium quality, 22.05 kHz).
_DEFAULT_VOICES = {
    "en": config.PIPER_DEFAULT_VOICE_EN,
    "de": config.PIPER_DEFAULT_VOICE_DE,
    "ar": config.PIPER_DEFAULT_VOICE_AR,
}


def _voice_relpath(voice: str) -> tuple[str, str]:
    """Map a voice name like ``en_US-lessac-medium`` to its HF path + .onnx."""
    # Format: <lang>_<region>-<speaker>-<quality>
    lang_region, _, _ = voice.partition("-")
    lang, _, region = lang_region.partition("_")
    return f"{lang}/{lang}_{region}/{voice.partition('-')[2].split('-')[0]}/{voice}", voice


def _voice_url(voice: str) -> tuple[str, str]:
    rel, name = _voice_relpath(voice)
    onnx = f"{_PIPER_VOICE_BASE}/{rel}/{name}.onnx"
    jsonc = f"{_PIPER_VOICE_BASE}/{rel}/{name}.onnx.json"
    return onnx, jsonc


class PiperProvider(TTSProvider):
    """Local CPU Piper TTS (GPL-3.0, opt-in)."""

    def __init__(self, voices_dir: Optional[Path] = None) -> None:
        self._voices_dir = Path(voices_dir) if voices_dir else config.PIPER_VOICES_DIR

    @property
    def name(self) -> str:
        return "piper"

    @property
    def available(self) -> bool:
        """True only when explicitly enabled AND the piper binary is installed."""
        if not config.PIPER_ENABLED:
            return False
        return shutil.which("piper") is not None

    def meta(self) -> ProviderMeta:
        return ProviderMeta(
            name=self.name,
            kind=ProviderKind.LOCAL,
            version="1.7.0",
            description="Piper neural TTS — local CPU, real-time, GPL-3.0 (opt-in)",
            cost=ProviderCost(is_paid=False, unit="free",
                              note="Free CPU inference; GPL-3.0 license obligation applies."),
            license=_PIPER_LICENSE,
            runtime=ProviderRuntime(
                requires_gpu=False, requires_ram_gb=1.0,
                requires_api_key=False, requires_network=True,  # network only for first voice download
                cpu_fallback=True,
            ),
            capability=ProviderCapability(
                capabilities=[Capability.TEXT_TO_SPEECH],
                languages=[l for l in SUPPORTED_LANGUAGES if _DEFAULT_VOICES.get(l)],
            ),
        )

    def license(self) -> ProviderLicense:
        return _PIPER_LICENSE

    # -- voice management ----------------------------------------------------
    def _voice_path(self, voice: str) -> tuple[Path, Path]:
        onnx = self._voices_dir / f"{voice}.onnx"
        jsonc = self._voices_dir / f"{voice}.onnx.json"
        return onnx, jsonc

    def _ensure_voice(self, voice: str) -> tuple[Path, Path]:
        """Download a voice model on demand if not already present."""
        onnx, jsonc = self._voice_path(voice)
        if onnx.exists() and jsonc.exists() and onnx.stat().st_size > 0:
            return onnx, jsonc
        self._voices_dir.mkdir(parents=True, exist_ok=True)
        onnx_url, json_url = _voice_url(voice)
        log("PIPER", f"downloading voice {voice}", voices_dir=str(self._voices_dir))
        try:
            urllib.request.urlretrieve(onnx_url, onnx)
            urllib.request.urlretrieve(json_url, jsonc)
        except Exception as e:  # noqa: BLE001
            onnx.unlink(missing_ok=True)
            jsonc.unlink(missing_ok=True)
            raise VideoError(
                TypedErrorCode.MODEL_NOT_FOUND,
                f"Failed to download Piper voice '{voice}': {e}",
                context={"voice": voice, "url": onnx_url},
            ) from e
        if not onnx.exists() or onnx.stat().st_size == 0:
            raise VideoError(
                TypedErrorCode.MODEL_NOT_FOUND,
                f"Piper voice '{voice}' downloaded as empty.",
                context={"voice": voice},
            )
        return onnx, jsonc

    def _resolve_voice(self, request: VoiceRequest) -> str:
        if request.voice:
            return request.voice
        v = _DEFAULT_VOICES.get(request.language)
        if not v:
            raise VideoError(
                TypedErrorCode.CAPABILITY_UNSUPPORTED,
                f"Piper has no default voice for language '{request.language}'. "
                f"Set PIPER_DEFAULT_VOICE_{request.language.upper()} or request.voice.",
                context={"language": request.language},
            )
        return v

    def list_voices(self, language: str = "en") -> list[str]:
        v = _DEFAULT_VOICES.get(language, "")
        return [v] if v else []

    # -- synthesis -----------------------------------------------------------
    def synthesize(self, request: VoiceRequest, destination: Path) -> VoiceResult:
        request.validate()
        if not self.available:
            raise VideoError(
                TypedErrorCode.NO_PROVIDER,
                "Piper TTS is not available. Set PIPER_ENABLED=true and install piper-tts.",
                context={"piper_enabled": config.PIPER_ENABLED, "piper_on_path": shutil.which("piper") is not None},
            )
        voice = self._resolve_voice(request)
        onnx, jsonc = self._ensure_voice(voice)
        # Piper writes WAV; if mp3 requested, synthesize WAV then transcode with ffmpeg.
        wav_dest = destination if request.output_format == "wav" else destination.with_suffix(".wav")
        wav_dest.parent.mkdir(parents=True, exist_ok=True)
        # length-scale controls rate (1.0 = normal; higher = slower). Piper uses inverse.
        length_scale = 1.0 / request.rate if request.rate and request.rate > 0 else 1.0
        cmd = [
            "piper", "-m", str(onnx), "-c", str(jsonc),
            "-f", str(wav_dest), "--length-scale", str(length_scale),
        ]
        log("PIPER", f"synthesize voice={voice} lang={request.language}", text_len=len(request.text))
        try:
            proc = subprocess.run(cmd, input=request.text.encode("utf-8"),
                                  capture_output=True, timeout=120)
        except subprocess.TimeoutExpired as e:
            raise VideoError(
                TypedErrorCode.GENERATION_TIMEOUT,
                f"Piper synthesis timed out for voice '{voice}'.",
                context={"voice": voice},
            ) from e
        except FileNotFoundError as e:
            raise VideoError(
                TypedErrorCode.NO_PROVIDER,
                "Piper binary not found on PATH.",
                context={"voice": voice},
            ) from e
        if proc.returncode != 0:
            raise VideoError(
                TypedErrorCode.NO_OUTPUT,
                f"Piper synthesis failed (exit {proc.returncode}): "
                f"{proc.stderr.decode('utf-8', 'replace')[:500]}",
                context={"voice": voice, "returncode": proc.returncode},
            )
        if not wav_dest.exists() or wav_dest.stat().st_size == 0:
            raise VideoError(
                TypedErrorCode.NO_OUTPUT,
                f"Piper produced no output file for voice '{voice}'.",
                context={"voice": voice, "path": str(wav_dest)},
            )
        # Verify the WAV is real and decodable.
        from ..audio.qc import verify_audio
        report = verify_audio(wav_dest)
        if not report.get("ok"):
            raise VideoError(
                TypedErrorCode.INVALID_AUDIO,
                f"Piper output failed audio QC: {report.get('error', 'failed')}",
                context={"voice": voice, "path": str(wav_dest)},
            )
        duration = float(report.get("duration", 0.0))
        sample_rate = int(report.get("sample_rate", 22050))
        # Transcode to mp3 if requested.
        final_path = wav_dest
        if request.output_format == "mp3" and wav_dest != destination:
            self._transcode_mp3(wav_dest, destination)
            final_path = destination
            wav_dest.unlink(missing_ok=True)
        return VoiceResult(
            path=final_path, duration=duration, provider=self.name,
            language=request.language, voice=voice, text=request.text,
            sample_rate=sample_rate,
        )

    @staticmethod
    def _transcode_mp3(wav: Path, mp3: Path) -> None:
        if not shutil.which("ffmpeg"):
            raise VideoError(
                TypedErrorCode.FFMPEG_ERROR,
                "ffmpeg not available to transcode Piper WAV to MP3.",
            )
        proc = subprocess.run([
            "ffmpeg", "-y", "-i", str(wav), "-c:a", "libmp3lame", "-q:a", "4", str(mp3),
        ], capture_output=True)
        if proc.returncode != 0 or not mp3.exists():
            raise VideoError(
                TypedErrorCode.FFMPEG_ERROR,
                f"MP3 transcode failed: {proc.stderr.decode('utf-8', 'replace')[:500]}",
            )


def build_piper_provider() -> PiperProvider:
    return PiperProvider()

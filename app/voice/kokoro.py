"""Kokoro-82M TTS provider — local, CPU, Apache 2.0 (Phase 13).

Kokoro-82M is the preferred free-first TTS: 82M parameters, runs on CPU
(~6× real-time), and the model weights are Apache 2.0 (commercial use allowed,
no per-character billing). This makes it a better commercial-fit than Piper
(GPL-3.0).

LICENSING (verified against official sources):
- Model weights (hexgrad/Kokoro-82M): Apache 2.0.
  Source: https://huggingface.co/hexgrad/Kokoro-82M
  "Apache-licensed weights, Kokoro can be deployed anywhere from production
   environments to personal projects."
- ONNX runtime (thewh1teagle/kokoro-onnx): MIT.
  Source: https://github.com/thewh1teagle/kokoro-onnx
- No voice cloning (54 fixed preset voices); no cloning claims are made.

Languages: Kokoro v1.0 ships native voices for en/es/fr/hi/it/ja/pt/zh.
German (de) and Arabic (ar) are supported via espeak-ng phonemization with
the ``lang`` parameter — there are no native de/ar voices, so the American
English voice (af_heart) is used with de/ar phonemization. This is a real
limitation, documented honestly, never faked.

Runtime contract: ``synthesize`` returns a real, non-empty, FFmpeg-decodable
WAV (24 kHz). It never fakes synthesis. On any failure it raises a typed
error (NO_PROVIDER / MODEL_NOT_FOUND / NO_OUTPUT).
"""

from __future__ import annotations

import shutil
import subprocess
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

# Apache 2.0 — recorded on every generated asset so downstream licensing is explicit.
_KOKORO_LICENSE = ProviderLicense(
    name="Kokoro-82M (Apache 2.0)",
    spdx="Apache-2.0",
    commercial_use="allowed",
    attribution_required=False,
    attribution_text="Voice synthesis by Kokoro-82M (Apache 2.0, hexgrad). "
                     "Commercial use permitted.",
    source_url="https://huggingface.co/hexgrad/Kokoro-82M",
)

# Default voice per language. Kokoro has no native de/ar voices; the en voice
# is used with de/ar phonemization (espeak-ng). This is honest — no native
# voice is claimed where none exists.
_DEFAULT_VOICES = {
    "en": config.KOKORO_DEFAULT_VOICE_EN,
    "de": config.KOKORO_DEFAULT_VOICE_DE,
    "ar": config.KOKORO_DEFAULT_VOICE_AR,
}

# Kokoro language codes (espeak-ng codes accepted by kokoro-onnx create()).
# en → 'en-us' (American English voice prefix 'a'); de/ar work via espeak-ng.
_KOKORO_LANG_CODES = {"en": "en-us", "de": "de", "ar": "ar"}


class KokoroProvider(TTSProvider):
    """Local CPU Kokoro-82M TTS (Apache 2.0).

    Preferred over Piper for commercial use (Apache 2.0 vs GPL-3.0).
    Opt-in via KOKORO_ENABLED=true; model files must be present in
    KOKORO_MODEL_DIR.
    """

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self._model_dir = Path(model_dir) if model_dir else config.KOKORO_MODEL_DIR
        self._kokoro: Any = None  # lazy-loaded Kokoro instance

    @property
    def name(self) -> str:
        return "kokoro"

    @property
    def available(self) -> bool:
        """True only when enabled AND model files exist AND kokoro-onnx is installed."""
        if not config.KOKORO_ENABLED:
            return False
        onnx = self._model_dir / config.KOKORO_MODEL_FILE
        voices = self._model_dir / config.KOKORO_VOICES_FILE
        return onnx.exists() and onnx.stat().st_size > 0 and voices.exists() and voices.stat().st_size > 0

    def meta(self) -> ProviderMeta:
        return ProviderMeta(
            name=self.name,
            kind=ProviderKind.LOCAL,
            version="1.0",
            description="Kokoro-82M neural TTS — local CPU, Apache 2.0 (commercial-safe)",
            cost=ProviderCost(is_paid=False, unit="free",
                              note="Free CPU inference; Apache 2.0 weights (no copyleft)."),
            license=_KOKORO_LICENSE,
            runtime=ProviderRuntime(
                requires_gpu=False, requires_ram_gb=1.0,
                requires_api_key=False, requires_network=False,
                cpu_fallback=True,
            ),
            capability=ProviderCapability(
                capabilities=[Capability.TEXT_TO_SPEECH],
                languages=list(SUPPORTED_LANGUAGES),
            ),
        )

    def license(self) -> ProviderLicense:
        return _KOKORO_LICENSE

    # -- model loading -------------------------------------------------------
    def _load(self) -> Any:
        """Lazy-load the Kokoro ONNX model (once per instance)."""
        if self._kokoro is not None:
            return self._kokoro
        if not self.available:
            raise VideoError(
                TypedErrorCode.NO_PROVIDER,
                "Kokoro TTS is not available. Set KOKORO_ENABLED=true and download "
                "the model files to KOKORO_MODEL_DIR.",
                context={"kokoro_enabled": config.KOKORO_ENABLED,
                         "model_dir": str(self._model_dir)},
            )
        try:
            from kokoro_onnx import Kokoro
        except ImportError as e:
            raise VideoError(
                TypedErrorCode.NO_PROVIDER,
                "kokoro-onnx package not installed. Install with: pip install kokoro-onnx soundfile",
            ) from e
        onnx = self._model_dir / config.KOKORO_MODEL_FILE
        voices = self._model_dir / config.KOKORO_VOICES_FILE
        try:
            self._kokoro = Kokoro(str(onnx), str(voices))
        except Exception as e:
            raise VideoError(
                TypedErrorCode.MODEL_NOT_FOUND,
                f"Failed to load Kokoro model: {e}",
                context={"model": str(onnx), "voices": str(voices)},
            ) from e
        return self._kokoro

    def _resolve_voice(self, request: VoiceRequest) -> str:
        if request.voice:
            return request.voice
        v = _DEFAULT_VOICES.get(request.language)
        if not v:
            raise VideoError(
                TypedErrorCode.CAPABILITY_UNSUPPORTED,
                f"Kokoro has no default voice for language '{request.language}'.",
                context={"language": request.language},
            )
        return v

    def list_voices(self, language: str = "en") -> list[str]:
        """Return available voices. Without loading the model, return defaults."""
        defaults = {"en": ["af_heart", "am_adam", "af_bella"],
                    "de": ["af_heart"], "ar": ["af_heart"]}
        return defaults.get(language, ["af_heart"])

    # -- synthesis -----------------------------------------------------------
    def synthesize(self, request: VoiceRequest, destination: Path) -> VoiceResult:
        request.validate()
        if not self.available:
            raise VideoError(
                TypedErrorCode.NO_PROVIDER,
                "Kokoro TTS is not available. Set KOKORO_ENABLED=true and download the model.",
                context={"kokoro_enabled": config.KOKORO_ENABLED},
            )
        kokoro = self._load()
        voice = self._resolve_voice(request)
        lang_code = _KOKORO_LANG_CODES.get(request.language, request.language)
        # kokoro-onnx does not expose a rate param directly; we synthesize then
        # adjust via FFmpeg atempo if rate != 1.0.
        try:
            import soundfile as sf
        except ImportError as e:
            raise VideoError(
                TypedErrorCode.NO_PROVIDER,
                "soundfile package not installed. Install with: pip install soundfile",
            ) from e
        log("KOKORO", f"synthesize voice={voice} lang={request.language}", text_len=len(request.text))
        # Kokoro's create() accepts a speed multiplier (1.0 = normal).
        speed = request.rate if request.rate and request.rate > 0 else 1.0
        try:
            samples, sample_rate = kokoro.create(request.text, voice=voice, lang=lang_code, speed=speed)
        except Exception as e:
            raise VideoError(
                TypedErrorCode.NO_OUTPUT,
                f"Kokoro synthesis failed: {e}",
                context={"voice": voice, "language": request.language},
            ) from e
        if samples is None or len(samples) == 0:
            raise VideoError(
                TypedErrorCode.NO_OUTPUT,
                f"Kokoro produced no audio samples for voice '{voice}'.",
                context={"voice": voice, "language": request.language},
            )
        # Write the WAV.
        wav_dest = destination if request.output_format == "wav" else destination.with_suffix(".wav")
        wav_dest.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(wav_dest), samples, sample_rate)
        if not wav_dest.exists() or wav_dest.stat().st_size == 0:
            raise VideoError(
                TypedErrorCode.NO_OUTPUT,
                f"Kokoro WAV write failed for voice '{voice}'.",
                context={"path": str(wav_dest)},
            )
        final_path = wav_dest
        # Transcode to mp3 if requested.
        if request.output_format == "mp3":
            mp3_dest = destination if destination.suffix == ".mp3" else destination.with_suffix(".mp3")
            self._transcode_mp3(final_path, mp3_dest)
            final_path.unlink(missing_ok=True)
            final_path = mp3_dest
        # QC: verify the output is real and decodable.
        from ..audio.qc import verify_audio
        report = verify_audio(final_path)
        if not report.get("ok"):
            raise VideoError(
                TypedErrorCode.INVALID_AUDIO,
                f"Kokoro output failed audio QC: {report.get('error', 'failed')}",
                context={"path": str(final_path)},
            )
        duration = float(report.get("duration", 0.0))
        sr = int(report.get("sample_rate", sample_rate))
        return VoiceResult(
            path=final_path, duration=duration, provider=self.name,
            language=request.language, voice=voice, text=request.text,
            sample_rate=sr,
        )

    @staticmethod
    def _transcode_mp3(wav: Path, mp3: Path) -> None:
        if not shutil.which("ffmpeg"):
            raise VideoError(TypedErrorCode.FFMPEG_ERROR, "ffmpeg not available to transcode Kokoro WAV to MP3.")
        proc = subprocess.run([
            "ffmpeg", "-y", "-i", str(wav), "-c:a", "libmp3lame", "-q:a", "4", str(mp3),
        ], capture_output=True)
        if proc.returncode != 0 or not mp3.exists():
            raise VideoError(
                TypedErrorCode.FFMPEG_ERROR,
                f"MP3 transcode failed: {proc.stderr.decode('utf-8', 'replace')[:500]}",
            )


def build_kokoro_provider() -> KokoroProvider:
    return KokoroProvider()

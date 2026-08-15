"""TTS provider abstraction (Phase 9).

A provider-agnostic Text-to-Speech layer supporting multiple languages
(English, German, Arabic), voice selection, gender, speaking rate, pitch,
emotion/style, pronunciation direction, and output format. No TTS vendor is
hard-coded; future providers implement :class:`TTSProvider`.

Contract guarantees (mirroring video providers):
- ``synthesize`` MUST return a real, existing, non-empty, FFmpeg-decodable
  audio file. It MUST NOT simulate success.
- If no TTS provider is configured/reachable, generation raises a typed
  ``NO_PROVIDER`` error — never a fake audio file.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from ..core.errors import TypedErrorCode, VideoError


SUPPORTED_LANGUAGES = ("en", "de", "ar")


class TTSError(VideoError):
    """Typed TTS failure."""

    def __init__(self, code: TypedErrorCode, detail: str = "", *, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(code, detail, context=context)


@dataclass
class VoiceRequest:
    """A single TTS synthesis request.

    Attributes:
        text: the text to synthesize.
        language: ISO code (en/de/ar).
        voice: provider voice identifier; None = provider default.
        gender: "male"/"female" where the provider supports it.
        rate: speaking rate multiplier (1.0 = normal).
        pitch: pitch multiplier where supported (1.0 = normal).
        emotion/style: e.g. "narrative", "excited", "calm".
        direction: pronunciation/delivery direction text.
        output_format: e.g. "mp3", "wav".
    """

    text: str
    language: str = "en"
    voice: Optional[str] = None
    gender: Optional[str] = None
    rate: float = 1.0
    pitch: float = 1.0
    emotion: Optional[str] = None
    style: Optional[str] = None
    direction: Optional[str] = None
    output_format: str = "mp3"

    def validate(self) -> None:
        if not self.text.strip():
            raise TTSError(TypedErrorCode.WORKFLOW_INVALID, "TTS text is empty.")
        if self.language not in SUPPORTED_LANGUAGES:
            raise TTSError(
                TypedErrorCode.WORKFLOW_INVALID,
                f"Unsupported language '{self.language}'. Supported: {list(SUPPORTED_LANGUAGES)}.",
                context={"language": self.language, "supported": list(SUPPORTED_LANGUAGES)},
            )
        if self.rate <= 0 or self.rate > 4.0:
            raise TTSError(TypedErrorCode.WORKFLOW_INVALID, f"rate {self.rate} out of range (0,4].")
        if self.pitch <= 0 or self.pitch > 4.0:
            raise TTSError(TypedErrorCode.WORKFLOW_INVALID, f"pitch {self.pitch} out of range (0,4].")
        if self.gender is not None and self.gender not in ("male", "female"):
            raise TTSError(TypedErrorCode.WORKFLOW_INVALID, f"gender '{self.gender}' must be male/female or None.")
        if self.output_format not in ("mp3", "wav"):
            raise TTSError(TypedErrorCode.WORKFLOW_INVALID, f"output_format '{self.output_format}' must be mp3/wav.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text, "language": self.language, "voice": self.voice,
            "gender": self.gender, "rate": self.rate, "pitch": self.pitch,
            "emotion": self.emotion, "style": self.style, "direction": self.direction,
            "output_format": self.output_format,
        }


@dataclass
class VoiceResult:
    """Result of a successful TTS synthesis.

    ``path`` is guaranteed to exist, be non-empty, and be FFmpeg-decodable.
    """

    path: Path
    duration: float
    provider: str
    language: str
    voice: Optional[str]
    text: str
    sample_rate: int = 44100

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path), "duration": self.duration, "provider": self.provider,
            "language": self.language, "voice": self.voice, "text": self.text,
            "sample_rate": self.sample_rate,
        }


class TTSProvider(abc.ABC):
    """Abstract base class for all TTS providers."""

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def available(self) -> bool:
        """True if the provider is configured and reachable right now."""

    @abc.abstractmethod
    def list_voices(self, language: str = "en") -> list[str]:
        """Return voice identifiers available for a language."""

    @abc.abstractmethod
    def synthesize(self, request: VoiceRequest, destination: Path) -> VoiceResult:
        """Synthesize speech to ``destination``. Returns a VoiceResult.

        On failure raises TTSError. Never returns a path that does not exist or
        is not a real, decodable audio file.
        """


class NullTTSProvider(TTSProvider):
    """Default no-op provider used when no real TTS is configured.

    ALWAYS raises NO_PROVIDER — it never synthesizes. This makes the absence of
    a TTS provider an explicit, typed failure rather than a silent fake.
    """

    @property
    def name(self) -> str:
        return "null"

    @property
    def available(self) -> bool:
        return False

    def list_voices(self, language: str = "en") -> list[str]:
        return []

    def synthesize(self, request: VoiceRequest, destination: Path) -> VoiceResult:
        raise TTSError(
            TypedErrorCode.NO_PROVIDER,
            "No TTS provider configured. Configure a real TTS provider to generate voiceover.",
            context={"requested_language": request.language},
        )


def select_tts_provider(providers: list[TTSProvider]) -> TTSProvider:
    """Return the first available provider; NO_PROVIDER if none."""
    for p in providers:
        if p.available:
            return p
    return NullTTSProvider()

"""Music provider abstraction (Phase 9).

A provider-agnostic layer for background music driven by
:class:`~app.brain.models.AudioPlan.music_direction`. Supports mood categories
(cinematic, emotional, energetic, dark, luxury, corporate, suspense,
documentary). Never claims music was generated unless a real audio asset exists.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..core.errors import TypedErrorCode, VideoError


class MusicError(VideoError):
    def __init__(self, code: TypedErrorCode, detail: str = "", *, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(code, detail, context=context)


# Recognized mood categories. Direction text is matched against these.
MUSIC_MOODS = (
    "cinematic", "emotional", "energetic", "dark", "luxury",
    "corporate", "suspense", "documentary",
)


@dataclass
class MusicRequest:
    """A music generation/selection request."""

    mood: str  # one of MUSIC_MOODS
    duration: float
    direction: Optional[str] = None
    fade_in: float = 0.0
    fade_out: float = 0.0
    output_format: str = "mp3"

    def validate(self) -> None:
        if self.mood not in MUSIC_MOODS:
            raise MusicError(
                TypedErrorCode.WORKFLOW_INVALID,
                f"Unknown music mood '{self.mood}'. Supported: {list(MUSIC_MOODS)}.",
                context={"mood": self.mood, "supported": list(MUSIC_MOODS)},
            )
        if self.duration <= 0:
            raise MusicError(TypedErrorCode.WORKFLOW_INVALID, f"duration {self.duration} must be > 0.")
        if self.fade_in < 0 or self.fade_out < 0:
            raise MusicError(TypedErrorCode.WORKFLOW_INVALID, "fades must be >= 0.")
        if self.fade_in + self.fade_out > self.duration:
            raise MusicError(TypedErrorCode.WORKFLOW_INVALID, "fade_in + fade_out exceeds duration.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mood": self.mood, "duration": self.duration, "direction": self.direction,
            "fade_in": self.fade_in, "fade_out": self.fade_out, "output_format": self.output_format,
        }


@dataclass
class MusicResult:
    path: Path
    duration: float
    mood: str
    provider: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "duration": self.duration, "mood": self.mood, "provider": self.provider}


class MusicProvider(abc.ABC):
    """Abstract base class for all music providers."""

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def available(self) -> bool: ...

    @abc.abstractmethod
    def generate(self, request: MusicRequest, destination: Path) -> MusicResult:
        """Generate/select music to ``destination``. Never fake success."""


class NullMusicProvider(MusicProvider):
    """No-op provider; raises NO_PROVIDER."""

    @property
    def name(self) -> str:
        return "null"

    @property
    def available(self) -> bool:
        return False

    def generate(self, request: MusicRequest, destination: Path) -> MusicResult:
        request.validate()
        raise MusicError(
            TypedErrorCode.NO_PROVIDER,
            "No music provider configured. Configure a real music provider.",
            context={"mood": request.mood},
        )


def parse_mood(music_direction: str) -> str:
    """Extract a supported mood from a free-text music_direction string."""
    text = (music_direction or "").lower()
    for mood in MUSIC_MOODS:
        if mood in text:
            return mood
    return "cinematic"  # safe default

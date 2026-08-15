"""SFX provider abstraction (Phase 9).

A provider-agnostic layer for scene-level sound effects. Effects are mapped to
scene timing. Never claims SFX generation succeeded unless a real audio asset
exists.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..core.errors import TypedErrorCode, VideoError


class SFXError(VideoError):
    def __init__(self, code: TypedErrorCode, detail: str = "", *, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(code, detail, context=context)


# Recognized SFX categories.
SFX_CATEGORIES = (
    "whoosh", "impact", "footsteps", "rain", "wind", "jungle_ambience",
    "traffic", "product_sounds", "cinematic_hits", "ambient",
)


@dataclass
class SFXRequest:
    category: str
    duration: float
    scene_index: Optional[int] = None
    start_offset: float = 0.0
    direction: Optional[str] = None
    output_format: str = "mp3"

    def validate(self) -> None:
        if self.category not in SFX_CATEGORIES:
            raise SFXError(
                TypedErrorCode.WORKFLOW_INVALID,
                f"Unknown SFX category '{self.category}'. Supported: {list(SFX_CATEGORIES)}.",
                context={"category": self.category, "supported": list(SFX_CATEGORIES)},
            )
        if self.duration <= 0:
            raise SFXError(TypedErrorCode.WORKFLOW_INVALID, f"duration {self.duration} must be > 0.")
        if self.start_offset < 0:
            raise SFXError(TypedErrorCode.WORKFLOW_INVALID, "start_offset must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category, "duration": self.duration,
            "scene_index": self.scene_index, "start_offset": self.start_offset,
            "direction": self.direction, "output_format": self.output_format,
        }


@dataclass
class SFXResult:
    path: Path
    duration: float
    category: str
    provider: str
    scene_index: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path), "duration": self.duration, "category": self.category,
            "provider": self.provider, "scene_index": self.scene_index,
        }


class SFXProvider(abc.ABC):
    """Abstract base class for all SFX providers."""

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def available(self) -> bool: ...

    @abc.abstractmethod
    def generate(self, request: SFXRequest, destination: Path) -> SFXResult:
        """Generate an SFX clip to ``destination``. Never fake success."""


class NullSFXProvider(SFXProvider):
    """No-op provider; raises NO_PROVIDER."""

    @property
    def name(self) -> str:
        return "null"

    @property
    def available(self) -> bool:
        return False

    def generate(self, request: SFXRequest, destination: Path) -> SFXResult:
        request.validate()
        raise SFXError(
            TypedErrorCode.NO_PROVIDER,
            "No SFX provider configured. Configure a real SFX provider.",
            context={"category": request.category},
        )


def parse_sfx_categories(sfx_text: str) -> list[str]:
    """Extract recognized SFX categories from a free-text direction string."""
    text = (sfx_text or "").lower()
    found: list[str] = []
    for cat in SFX_CATEGORIES:
        if cat in text:
            found.append(cat)
    return found

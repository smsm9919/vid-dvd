"""Export profiles (Phase 10).

Configurable platform export profiles. Platform-specific assumptions are NOT
hard-coded in the core editor — the editor reads width/height/fps/codec/etc.
from the profile object. Profiles can be added/overridden at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .timeline import EditingError
from ..core.errors import TypedErrorCode


class AspectRatio(str, Enum):
    VERTICAL_9_16 = "9:16"
    HORIZONTAL_16_9 = "16:9"
    SQUARE_1_1 = "1:1"


class Quality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ExportProfile:
    """A configurable export profile.

    Fields are read by the compositor; nothing platform-specific lives in the
    core editor logic.
    """

    name: str
    width: int
    height: int
    fps: int
    video_codec: str = "libx264"
    pixel_format: str = "yuv420p"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    audio_sample_rate: int = 44100
    crf: int = 20  # only for libx264
    preset: str = "medium"
    aspect: str = "16:9"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "width": self.width, "height": self.height,
            "fps": self.fps, "video_codec": self.video_codec,
            "pixel_format": self.pixel_format, "audio_codec": self.audio_codec,
            "audio_bitrate": self.audio_bitrate, "audio_sample_rate": self.audio_sample_rate,
            "crf": self.crf, "preset": self.preset, "aspect": self.aspect,
        }

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)


# Built-in profiles. Configurable: callers may register/override via
# PROFILES dict or pass a custom ExportProfile directly to the exporter.
TIKTOK = ExportProfile("TIKTOK", 1080, 1920, fps=30, aspect="9:16")
INSTAGRAM_REELS = ExportProfile("INSTAGRAM_REELS", 1080, 1920, fps=30, aspect="9:16")
YOUTUBE_SHORTS = ExportProfile("YOUTUBE_SHORTS", 1080, 1920, fps=30, aspect="9:16")
YOUTUBE = ExportProfile("YOUTUBE", 1920, 1080, fps=30, aspect="16:9")
SQUARE = ExportProfile("SQUARE", 1080, 1080, fps=30, aspect="1:1")

PROFILES: dict[str, ExportProfile] = {
    "TIKTOK": TIKTOK,
    "INSTAGRAM_REELS": INSTAGRAM_REELS,
    "YOUTUBE_SHORTS": YOUTUBE_SHORTS,
    "YOUTUBE": YOUTUBE,
    "SQUARE": SQUARE,
}

# Quality → CRF/preset mapping.
QUALITY_CRF = {Quality.LOW: 26, Quality.MEDIUM: 23, Quality.HIGH: 20}
QUALITY_PRESET = {Quality.LOW: "veryfast", Quality.MEDIUM: "medium", Quality.HIGH: "medium"}


def get_profile(name: str, *, quality: Quality = Quality.HIGH) -> ExportProfile:
    """Look up a profile by name, applying quality settings. Raises on unknown."""
    if name not in PROFILES:
        raise EditingError(
            TypedErrorCode.UNSUPPORTED_PROFILE,
            f"Unknown export profile '{name}'. Supported: {list(PROFILES)}.",
            context={"profile": name, "supported": list(PROFILES)},
        )
    base = PROFILES[name]
    # Apply quality overrides for libx264.
    crf = QUALITY_CRF[quality]
    preset = QUALITY_PRESET[quality]
    return ExportProfile(
        name=base.name, width=base.width, height=base.height, fps=base.fps,
        video_codec=base.video_codec, pixel_format=base.pixel_format,
        audio_codec=base.audio_codec, audio_bitrate=base.audio_bitrate,
        audio_sample_rate=base.audio_sample_rate, crf=crf, preset=preset,
        aspect=base.aspect,
    )


def register_profile(profile: ExportProfile) -> None:
    """Register or override a profile at runtime."""
    PROFILES[profile.name] = profile

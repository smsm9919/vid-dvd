"""Caption generation (Phase 9).

Supports SRT, WebVTT, and burned-in captions. Consumes the actual scene
voice/caption data from a :class:`~app.brain.models.ProductionPlan`. Styles are
configurable for TikTok / Instagram Reels / YouTube Shorts / YouTube. No single
caption style is hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from ..brain.models import CaptionPlan, ProductionPlan, Scene


class CaptionFormat(str, Enum):
    SRT = "srt"
    VTT = "vtt"
    BURNED_IN = "burned_in"


class CaptionStyle(str, Enum):
    TIKTOK = "tiktok"
    REELS = "reels"
    SHORTS = "shorts"
    YOUTUBE = "youtube"


# Per-style styling hints (for burned-in rendering / subtitle styling).
STYLE_PROFILES: dict[CaptionStyle, dict[str, Any]] = {
    CaptionStyle.TIKTOK: {"font_size": 48, "font_color": "white", "stroke": "black",
                          "stroke_width": 3, "position": "bottom", "bold": True, "max_chars": 42},
    CaptionStyle.REELS: {"font_size": 44, "font_color": "white", "stroke": "black",
                         "stroke_width": 2, "position": "center", "bold": True, "max_chars": 50},
    CaptionStyle.SHORTS: {"font_size": 46, "font_color": "white", "stroke": "black",
                          "stroke_width": 3, "position": "bottom", "bold": True, "max_chars": 40},
    CaptionStyle.YOUTUBE: {"font_size": 36, "font_color": "white", "bg_color": "rgba(0,0,0,0.75)",
                           "position": "bottom", "bold": False, "max_chars": 60},
}


class CaptionValidationError(Exception):
    """Structured caption timing validation error."""

    def __init__(self, code: str, message: str, *, context: Optional[dict[str, Any]] = None) -> None:
        self.code = code
        self.message = message
        self.context = context or {}
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": self.context}


@dataclass
class CaptionCue:
    """A single timed caption entry."""

    index: int
    start: float  # seconds, absolute project time
    end: float
    text: str
    scene_index: int

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "start": self.start, "end": self.end,
                "text": self.text, "scene_index": self.scene_index}


@dataclass
class CaptionReport:
    cues: list[CaptionCue]
    format: CaptionFormat
    style: CaptionStyle
    content: str  # serialized SRT/VTT text (empty for burned_in)
    errors: list[CaptionValidationError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cues": [c.to_dict() for c in self.cues], "format": self.format.value,
            "style": self.style.value, "content": self.content,
            "errors": [e.to_dict() for e in self.errors], "warnings": self.warnings,
            "ok": not self.errors,
        }


def _fmt_ts(seconds: float, *, vtt: bool = False) -> str:
    """Format seconds as HH:MM:SS,mmm (SRT) or HH:MM:SS.mmm (VTT)."""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:  # rounding overflow
        s += 1
        ms = 0
    sep = "." if vtt else ","
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def validate_caption_timing(cues: list[CaptionCue], project_duration: float) -> list[CaptionValidationError]:
    """Validate caption timing. Returns a list of structured errors.

    Detects: negative timestamps, overlapping entries, out-of-range timestamps,
    end before start.
    """
    errors: list[CaptionValidationError] = []
    for c in cues:
        if c.start < 0:
            errors.append(CaptionValidationError("NEGATIVE_START",
                f"Cue {c.index} has negative start {c.start}.", context={"cue": c.index, "start": c.start}))
        if c.end < c.start:
            errors.append(CaptionValidationError("END_BEFORE_START",
                f"Cue {c.index} ends before it starts.", context={"cue": c.index, "start": c.start, "end": c.end}))
        if c.end > project_duration + 0.5:
            errors.append(CaptionValidationError("OUT_OF_RANGE",
                f"Cue {c.index} end {c.end} exceeds project duration {project_duration}.",
                context={"cue": c.index, "end": c.end, "project_duration": project_duration}))
        if not c.text.strip():
            errors.append(CaptionValidationError("EMPTY_TEXT",
                f"Cue {c.index} has empty text.", context={"cue": c.index}))
    # Overlap detection.
    sorted_cues = sorted(cues, key=lambda x: x.start)
    for a, b in zip(sorted_cues, sorted_cues[1:]):
        if a.end > b.start + 0.01:
            errors.append(CaptionValidationError("OVERLAP",
                f"cue {a.index} overlaps cue {b.index}.",
                context={"cue_a": a.index, "cue_b": b.index, "a_end": a.end, "b_start": b.start}))
    return errors


def build_cues_from_plan(plan: ProductionPlan) -> list[CaptionCue]:
    """Build caption cues from scene caption/voice data.

    Each scene's caption text (falling back to the voiceover line) becomes a cue
    spanning the scene, adjusted by the voiceover start_offset when present.
    """
    cues: list[CaptionCue] = []
    abs_time = 0.0
    idx = 1
    for scene in plan.scenes:
        text = scene.caption.text or scene.voiceover.line
        if not text.strip():
            abs_time += scene.duration
            continue
        start = abs_time + scene.voiceover.start_offset
        end = abs_time + scene.duration
        if end <= start:
            end = start + 0.5
        cues.append(CaptionCue(index=idx, start=start, end=end, text=text, scene_index=scene.index))
        idx += 1
        abs_time += scene.duration
    return cues


def render_srt(cues: list[CaptionCue]) -> str:
    """Render cues as SRT text."""
    lines: list[str] = []
    for c in cues:
        lines.append(str(c.index))
        lines.append(f"{_fmt_ts(c.start)} --> {_fmt_ts(c.end)}")
        lines.append(c.text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_vtt(cues: list[CaptionCue]) -> str:
    """Render cues as WebVTT text."""
    lines: list[str] = ["WEBVTT", ""]
    for c in cues:
        lines.append(f"{_fmt_ts(c.start, vtt=True)} --> {_fmt_ts(c.end, vtt=True)}")
        lines.append(c.text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def generate_captions(
    plan: ProductionPlan,
    fmt: CaptionFormat = CaptionFormat.SRT,
    style: CaptionStyle = CaptionStyle.TIKTOK,
    *,
    validate: bool = True,
) -> CaptionReport:
    """Generate captions from a plan.

    Returns a CaptionReport with cues, serialized content (SRT/VTT), and any
    validation errors. For burned_in, content is empty (rendering happens in a
    later FFmpeg compositing phase) but cues + style profile are returned.
    """
    cues = build_cues_from_plan(plan)
    project_duration = sum(s.duration for s in plan.scenes)
    errors: list[CaptionValidationError] = []
    warnings: list[str] = []
    if validate:
        errors = validate_caption_timing(cues, project_duration)
    content = ""
    if fmt == CaptionFormat.SRT:
        content = render_srt(cues)
    elif fmt == CaptionFormat.VTT:
        content = render_vtt(cues)
    # burned_in: no text content, style profile carried for the compositing phase.
    if not cues:
        warnings.append("No caption cues generated (scenes have no caption/voice text).")
    return CaptionReport(cues=cues, format=fmt, style=style, content=content,
                         errors=errors, warnings=warnings)


def style_profile(style: CaptionStyle) -> dict[str, Any]:
    """Return the styling parameters for a caption style."""
    return STYLE_PROFILES[style]


def write_captions(report: CaptionReport, destination: Path) -> Path:
    """Write SRT/VTT content to a file. Returns the path."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ext = {"srt": ".srt", "vtt": ".vtt", "burned_in": ".ass"}[report.format.value]
    if destination.suffix != ext:
        destination = destination.with_suffix(ext)
    destination.write_text(report.content, encoding="utf-8")
    return destination

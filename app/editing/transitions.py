"""Transition abstraction (Phase 10).

Deterministic transitions: cut, fade, crossfade, dip_to_black, wipe.
Transitions respect scene durations and are only applied when the
ProductionPlan/scene transition intent requests them — never automatically
everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .timeline import EditingError
from ..core.errors import TypedErrorCode


class TransitionType(str, Enum):
    CUT = "cut"
    FADE = "fade"
    CROSSFADE = "crossfade"
    DIP_TO_BLACK = "dip_to_black"
    WIPE = "wipe"


SUPPORTED_TRANSITIONS = {t.value for t in TransitionType}


def parse_transition(intent: Optional[str]) -> tuple[TransitionType, float]:
    """Parse a transition intent string into (type, default_duration).

    Recognizes keywords like "fade", "crossfade", "dip to black", "wipe".
    Default durations: cut=0, fade=0.5, crossfade=0.5, dip_to_black=0.7, wipe=0.5.
    """
    if not intent:
        return TransitionType.CUT, 0.0
    text = intent.lower().strip()
    if "crossfade" in text or "cross fade" in text:
        return TransitionType.CROSSFADE, 0.5
    if "dip" in text and "black" in text:
        return TransitionType.DIP_TO_BLACK, 0.7
    if "wipe" in text:
        return TransitionType.WIPE, 0.5
    if "fade" in text:
        return TransitionType.FADE, 0.5
    return TransitionType.CUT, 0.0


@dataclass
class TransitionSpec:
    """A transition applied between scene N-1 and scene N."""

    type: TransitionType
    duration: float = 0.0

    def validate(self, scene_duration: float) -> None:
        type_val = self.type.value if hasattr(self.type, "value") else self.type
        if type_val not in SUPPORTED_TRANSITIONS:
            raise EditingError(
                TypedErrorCode.TRANSITION_ERROR,
                f"Unsupported transition '{self.type}'.",
                context={"transition": str(self.type)},
            )
        if self.duration < 0:
            raise EditingError(
                TypedErrorCode.TRANSITION_ERROR,
                f"Negative transition duration {self.duration}.",
                context={"duration": self.duration},
            )
        # Crossfade/wipe need two clips to overlap; require duration <= scene.
        if self.type in (TransitionType.CROSSFADE, TransitionType.WIPE) and self.duration > scene_duration:
            raise EditingError(
                TypedErrorCode.TRANSITION_ERROR,
                f"{self.type} duration {self.duration} exceeds scene duration {scene_duration}.",
                context={"duration": self.duration, "scene_duration": scene_duration},
            )

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type.value, "duration": self.duration}


def transition_filter(spec: TransitionSpec, prev_label: str, cur_label: str,
                     out_label: str) -> Optional[str]:
    """Return an FFmpeg filtergraph snippet implementing a transition.

    Returns None for cut (handled by simple concatenation).
    """
    if spec.type == TransitionType.CUT or spec.duration <= 0:
        return None
    d = spec.duration
    if spec.type == TransitionType.FADE:
        # Fade out prev, fade in cur at the boundary — approximated by a fade
        # on the current clip's first `d` seconds.
        return None  # fades applied per-clip in compositor
    if spec.type == TransitionType.CROSSFADE:
        return (f"{prev_label}{cur_label}xfade=transition=fade:duration={d}[{out_label}]")
    if spec.type == TransitionType.DIP_TO_BLACK:
        return (f"{prev_label}{cur_label}xfade=transition=fadeblack:duration={d}[{out_label}]")
    if spec.type == TransitionType.WIPE:
        return (f"{prev_label}{cur_label}xfade=transition=wipeleft:duration={d}[{out_label}]")
    return None

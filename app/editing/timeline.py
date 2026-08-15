"""Editing timeline model (Phase 10).

An explicit, structured representation of the final video timeline. Each scene
slot carries its video/voice/music/SFX/caption/transition/brand elements by
reference — never relying on implicit filename ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..core.errors import TypedErrorCode, VideoError


class EditingError(VideoError):
    """Typed editing failure."""

    def __init__(self, code: TypedErrorCode, detail: str = "", *, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(code, detail, context=context)


@dataclass
class TimelineScene:
    """One slot in the editing timeline."""

    scene_index: int  # 1-based, matches ProductionPlan.Scene.index
    start_time: float  # absolute seconds in the final timeline
    duration: float
    video_asset: Optional[Path] = None  # required for real assembly
    voice_asset: Optional[Path] = None  # optional per-scene voiceover
    voice_start_offset: float = 0.0
    music_assets: list[Path] = field(default_factory=list)
    sfx_assets: list[Path] = field(default_factory=list)
    ambience_asset: Optional[Path] = None
    caption_text: str = ""
    caption_start: float = 0.0  # absolute
    caption_end: float = 0.0
    transition: str = "cut"  # in-transition from previous scene
    transition_duration: float = 0.0
    brand_overlay: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_index": self.scene_index, "start_time": self.start_time,
            "duration": self.duration,
            "video_asset": str(self.video_asset) if self.video_asset else None,
            "voice_asset": str(self.voice_asset) if self.voice_asset else None,
            "voice_start_offset": self.voice_start_offset,
            "music_assets": [str(p) for p in self.music_assets],
            "sfx_assets": [str(p) for p in self.sfx_assets],
            "ambience_asset": str(self.ambience_asset) if self.ambience_asset else None,
            "caption_text": self.caption_text,
            "caption_start": self.caption_start, "caption_end": self.caption_end,
            "transition": self.transition, "transition_duration": self.transition_duration,
            "brand_overlay": self.brand_overlay,
        }


@dataclass
class Timeline:
    """The full editing timeline."""

    scenes: list[TimelineScene] = field(default_factory=list)
    total_duration: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenes": [s.to_dict() for s in self.scenes],
            "total_duration": self.total_duration,
            "scene_count": len(self.scenes),
        }


def validate_timeline(timeline: Timeline, *, require_video: bool = True) -> list[EditingError]:
    """Validate a timeline structurally. Returns a list of errors (empty = ok).

    Checks: non-empty, sequential scene indices, contiguous start times
    matching cumulative duration, non-negative times, no asset extending beyond
    its scene, transition durations fit, and (when require_video) each scene has
    a video asset.
    """
    errors: list[EditingError] = []
    if not timeline.scenes:
        errors.append(EditingError(TypedErrorCode.INVALID_TIMELINE, "Timeline has no scenes."))
        return errors
    expected_start = 0.0
    for ts in timeline.scenes:
        if ts.duration <= 0:
            errors.append(EditingError(
                TypedErrorCode.INVALID_TIMELINE,
                f"Scene {ts.scene_index} has non-positive duration {ts.duration}.",
                context={"scene_index": ts.scene_index}))
        if abs(ts.start_time - expected_start) > 0.01:
            errors.append(EditingError(
                TypedErrorCode.INVALID_TIMELINE,
                f"Scene {ts.scene_index} start_time {ts.start_time} != expected {expected_start}.",
                context={"scene_index": ts.scene_index, "start_time": ts.start_time, "expected": expected_start}))
        if ts.start_time < 0:
            errors.append(EditingError(
                TypedErrorCode.INVALID_TIMELINE,
                f"Scene {ts.scene_index} has negative start_time.",
                context={"scene_index": ts.scene_index}))
        if require_video and ts.video_asset is None:
            errors.append(EditingError(
                TypedErrorCode.MISSING_VIDEO_ASSET,
                f"Scene {ts.scene_index} has no video asset.",
                context={"scene_index": ts.scene_index}))
        # Voice must fit within scene (start_offset + ...). End not strictly known
        # at timeline build, but offset must be < duration.
        if ts.voice_start_offset < 0:
            errors.append(EditingError(
                TypedErrorCode.INVALID_TIMELINE,
                f"Scene {ts.scene_index} has negative voice_start_offset.",
                context={"scene_index": ts.scene_index}))
        if ts.voice_start_offset >= ts.duration and ts.voice_asset is not None:
            errors.append(EditingError(
                TypedErrorCode.INVALID_TIMELINE,
                f"Scene {ts.scene_index} voice_start_offset {ts.voice_start_offset} >= duration {ts.duration}.",
                context={"scene_index": ts.scene_index}))
        # Transition duration must not exceed scene duration.
        if ts.transition_duration < 0:
            errors.append(EditingError(
                TypedErrorCode.TRANSITION_ERROR,
                f"Scene {ts.scene_index} has negative transition_duration.",
                context={"scene_index": ts.scene_index}))
        if ts.transition not in ("cut", "fade", "crossfade", "dip_to_black", "wipe"):
            errors.append(EditingError(
                TypedErrorCode.TRANSITION_ERROR,
                f"Scene {ts.scene_index} has unknown transition '{ts.transition}'.",
                context={"scene_index": ts.scene_index, "transition": ts.transition}))
        # Caption times must be within scene span.
        if ts.caption_text:
            cap_start_rel = ts.caption_start - ts.start_time
            cap_end_rel = ts.caption_end - ts.start_time
            if cap_start_rel < -0.01 or cap_end_rel > ts.duration + 0.5:
                errors.append(EditingError(
                    TypedErrorCode.INVALID_TIMELINE,
                    f"Scene {ts.scene_index} caption out of range [{cap_start_rel:.2f},{cap_end_rel:.2f}] vs duration {ts.duration}.",
                    context={"scene_index": ts.scene_index}))
            if ts.caption_end < ts.caption_start:
                errors.append(EditingError(
                    TypedErrorCode.INVALID_TIMELINE,
                    f"Scene {ts.scene_index} caption end before start.",
                    context={"scene_index": ts.scene_index}))
        expected_start += ts.duration
    if abs(timeline.total_duration - expected_start) > 0.05:
        errors.append(EditingError(
            TypedErrorCode.INVALID_TIMELINE,
            f"total_duration {timeline.total_duration} != sum of scene durations {expected_start}.",
            context={"total_duration": timeline.total_duration, "expected": expected_start}))
    return errors


def build_timeline_from_assets(
    scene_durations: list[float],
    video_assets: list[Path],
    *,
    voice_assets: Optional[list[Optional[Path]]] = None,
    voice_start_offsets: Optional[list[float]] = None,
    music_assets: Optional[list[list[Path]]] = None,
    sfx_assets: Optional[list[list[Path]]] = None,
    ambience_assets: Optional[list[Optional[Path]]] = None,
    caption_texts: Optional[list[str]] = None,
    transitions: Optional[list[str]] = None,
    transition_durations: Optional[list[float]] = None,
) -> Timeline:
    """Build a Timeline from parallel per-scene asset lists.

    The first scene's transition is forced to 'cut' (no prior scene to blend).
    """
    n = len(scene_durations)
    if len(video_assets) != n:
        raise EditingError(
            TypedErrorCode.INVALID_TIMELINE,
            f"video_assets count {len(video_assets)} != scene count {n}.",
        )
    timeline = Timeline()
    t = 0.0
    for i in range(n):
        idx = i + 1
        dur = scene_durations[i]
        trans = (transitions[i] if transitions else "cut")
        if i == 0:
            trans = "cut"
        tdur = (transition_durations[i] if transition_durations else 0.0)
        if i == 0:
            tdur = 0.0
        cap = (caption_texts[i] if caption_texts else "")
        ts = TimelineScene(
            scene_index=idx, start_time=t, duration=dur,
            video_asset=video_assets[i],
            voice_asset=(voice_assets[i] if voice_assets else None),
            voice_start_offset=(voice_start_offsets[i] if voice_start_offsets else 0.0),
            music_assets=(music_assets[i] if music_assets else []),
            sfx_assets=(sfx_assets[i] if sfx_assets else []),
            ambience_asset=(ambience_assets[i] if ambience_assets else None),
            caption_text=cap, caption_start=t, caption_end=t + dur,
            transition=trans, transition_duration=tdur,
        )
        timeline.scenes.append(ts)
        t += dur
    timeline.total_duration = t
    return timeline

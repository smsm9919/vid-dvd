"""Scene assembly + final export engine (Phase 10).

Orchestrates the compositor: validates each scene before assembly, normalizes
and concatenates video, integrates mixed audio, optionally burns captions and
branding, runs final QC. Never silently skips a scene; never overwrites source
assets; never reports COMPLETED before QC passes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..audio.mixer import MixTrack, MixOptions, mix_audio
from ..audio.qc import verify_audio
from ..brain.models import ProductionPlan
from ..config import OUTPUT_DIR
from ..core.errors import TypedErrorCode, VideoError
from ..core.logging import log
from ..media import verify_mp4
from .compositor import (
    _apply_branding, _burn_captions, _build_ass_from_cues, _concat_scenes,
    _mux_audio, _normalize_scene_video, _silent_audio, captions_renderable, font_available,
)
from .profiles import ExportProfile, Quality, get_profile
from .qc import final_qc
from .timeline import EditingError, Timeline, TimelineScene, validate_timeline
from .transitions import TransitionSpec, TransitionType, parse_transition


@dataclass
class ExportRequest:
    """An explicit export request."""

    timeline: Timeline
    profile_name: str = "TIKTOK"
    quality: Quality = Quality.HIGH
    include_captions: bool = False
    caption_mode: str = "srt"  # "srt", "vtt", "burned_in"
    caption_cues: list[dict[str, Any]] = field(default_factory=list)
    caption_style: dict[str, Any] = field(default_factory=dict)
    include_branding: bool = False
    brand: dict[str, Any] = field(default_factory=dict)
    audio_path: Optional[Path] = None  # pre-mixed audio; None = silent
    silent: bool = False
    project_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name, "quality": self.quality.value,
            "include_captions": self.include_captions,
            "caption_mode": self.caption_mode,
            "include_branding": self.include_branding,
            "silent": self.silent, "project_id": self.project_id,
            "timeline": self.timeline.to_dict(),
        }


@dataclass
class ExportResult:
    status: str  # COMPLETED | FAILED
    output_path: Optional[str]
    profile: Optional[str]
    duration: float
    resolution: Optional[tuple[int, int]]
    video_codec: Optional[str]
    audio_codec: Optional[str]
    qc: dict[str, Any]
    error_code: Optional[str]
    error_detail: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status, "output_path": self.output_path,
            "profile": self.profile, "duration": self.duration,
            "resolution": list(self.resolution) if self.resolution else None,
            "video_codec": self.video_codec, "audio_codec": self.audio_codec,
            "qc": self.qc, "error_code": self.error_code,
            "error_detail": self.error_detail,
        }


def validate_scene_assets(scene: TimelineScene, *, require_audio: bool = False) -> list[EditingError]:
    """Validate one scene's assets exist and are readable before assembly."""
    errors: list[EditingError] = []
    if scene.video_asset is None:
        errors.append(EditingError(
            TypedErrorCode.MISSING_VIDEO_ASSET,
            f"Scene {scene.scene_index} has no video asset.",
            context={"scene_index": scene.scene_index}))
    elif not Path(scene.video_asset).exists():
        errors.append(EditingError(
            TypedErrorCode.MISSING_VIDEO_ASSET,
            f"Scene {scene.scene_index} video missing: {scene.video_asset}",
            context={"scene_index": scene.scene_index, "path": str(scene.video_asset)}))
    else:
        try:
            verify_mp4(scene.video_asset)
        except VideoError as e:
            errors.append(EditingError(
                TypedErrorCode.MISSING_VIDEO_ASSET,
                f"Scene {scene.scene_index} video not readable: {e.detail}",
                context={"scene_index": scene.scene_index, "code": e.code.value}))
    # Voice asset (optional but if present must exist/readable).
    if scene.voice_asset is not None:
        if not Path(scene.voice_asset).exists():
            errors.append(EditingError(
                TypedErrorCode.MISSING_AUDIO_ASSET,
                f"Scene {scene.scene_index} voice missing: {scene.voice_asset}",
                context={"scene_index": scene.scene_index}))
        else:
            try:
                verify_audio(scene.voice_asset)
            except VideoError as e:
                errors.append(EditingError(
                    TypedErrorCode.MISSING_AUDIO_ASSET,
                    f"Scene {scene.scene_index} voice not readable: {e.detail}",
                    context={"scene_index": scene.scene_index, "code": e.code.value}))
    if require_audio and scene.voice_asset is None:
        errors.append(EditingError(
            TypedErrorCode.MISSING_AUDIO_ASSET,
            f"Scene {scene.scene_index} requires a voice asset.",
            context={"scene_index": scene.scene_index}))
    return errors


def _make_audio_track(timeline: Timeline, profile: ExportProfile, work_dir: Path,
                      *, silent: bool) -> Optional[Path]:
    """Build the final mixed audio track from timeline per-scene audio.

    If silent or no audio assets, return None (caller adds silent track).
    """
    if silent:
        return None
    tracks: list[MixTrack] = []
    has_any = False
    for ts in timeline.scenes:
        if ts.voice_asset is not None:
            tracks.append(MixTrack(path=Path(ts.voice_asset), kind="voice",
                                   volume=1.0, start_offset=ts.start_time,
                                   duck_when_voice=False))
            has_any = True
        for m in ts.music_assets:
            tracks.append(MixTrack(path=Path(m), kind="music", volume=0.6,
                                   start_offset=ts.start_time, duck_when_voice=True))
            has_any = True
        for s in ts.sfx_assets:
            tracks.append(MixTrack(path=Path(s), kind="sfx", volume=0.5,
                                   start_offset=ts.start_time))
            has_any = True
        if ts.ambience_asset is not None:
            tracks.append(MixTrack(path=Path(ts.ambience_asset), kind="ambience",
                                   volume=0.3, start_offset=ts.start_time))
            has_any = True
    if not has_any:
        return None
    dest = work_dir / "mixed_audio.mp3"
    opts = MixOptions(target_duration=timeline.total_duration, music_duck_level=0.25)
    mix_audio(tracks, dest, opts)
    return dest


def export_video(req: ExportRequest, output_dir: Optional[Path] = None) -> ExportResult:
    """Run the full assembly + export pipeline. Returns an ExportResult.

    Never returns COMPLETED without a real QC-verified MP4.
    """
    try:
        # 1. Validate timeline + per-scene assets.
        timeline_errors = validate_timeline(req.timeline)
        scene_errors: list[EditingError] = []
        for ts in req.timeline.scenes:
            scene_errors += validate_scene_assets(ts)
        all_errors = timeline_errors + scene_errors
        if all_errors:
            err = all_errors[0]
            return ExportResult(
                status="FAILED", output_path=None, profile=req.profile_name,
                duration=0.0, resolution=None, video_codec=None, audio_codec=None,
                qc={"ok": False, "errors": [e.to_dict() for e in all_errors]},
                error_code=err.code.value, error_detail=err.detail,
            )
        profile = get_profile(req.profile_name, quality=req.quality)
        # 2. Work directory.
        work = Path(output_dir) if output_dir else (OUTPUT_DIR / (req.project_id or "standalone") / uuid.uuid4().hex[:12])
        work.mkdir(parents=True, exist_ok=True)
        # 3. Normalize each scene to the profile.
        normalized: list[Path] = []
        for i, ts in enumerate(req.timeline.scenes):
            trans_spec = TransitionSpec(
                TransitionType(ts.transition) if ts.transition in
                {t.value for t in TransitionType} else TransitionType.CUT,
                ts.transition_duration,
            )
            norm = work / f"scene_{ts.scene_index}_norm.mp4"
            _normalize_scene_video(Path(ts.video_asset), norm, profile, transition=trans_spec)
            normalized.append(norm)
        # 4. Concatenate.
        concat_dest = work / "concat.mp4"
        _concat_scenes(normalized, concat_dest, profile)
        current = concat_dest
        # 5. Burn captions if requested.
        if req.include_captions and req.caption_mode == "burned_in":
            if not captions_renderable():
                raise EditingError(
                    TypedErrorCode.CAPTION_RENDER_ERROR,
                    "FFmpeg lacks libass 'subtitles' filter; cannot burn captions.",
                )
            ass = work / "captions.ass"
            ass.write_text(_build_ass_from_cues(req.caption_cues, req.caption_style,
                                                 profile.width, profile.height), encoding="utf-8")
            burned = work / "burned.mp4"
            _burn_captions(current, burned, ass, profile)
            current = burned
        # 6. Branding if requested.
        if req.include_branding and req.brand:
            branded = work / "branded.mp4"
            _apply_branding(current, branded, req.brand, profile)
            current = branded
        # 7. Audio integration.
        audio_path = _make_audio_track(req.timeline, profile, work, silent=req.silent)
        final_video = work / "final_video.mp4"
        if audio_path is not None:
            _mux_audio(current, audio_path, final_video, profile)
        else:
            # Add silent audio so output has an audio stream (unless silent=True
            # still wants a track). Spec: video must contain audio stream unless
            # silent explicitly requested. We always include a track.
            _silent_audio(current, final_video, profile)
        # 8. Final QC.
        qc = final_qc(final_video, profile=profile, require_audio=not req.silent)
        if not qc["ok"]:
            return ExportResult(
                status="FAILED", output_path=str(final_video), profile=req.profile_name,
                duration=qc.get("duration", 0.0), resolution=(qc.get("width", 0), qc.get("height", 0)),
                video_codec=qc.get("video_codec"), audio_codec=qc.get("audio_codec"),
                qc=qc, error_code=TypedErrorCode.FINAL_QC_FAILED.value,
                error_detail=qc.get("error", "Final QC failed."),
            )
        log("EDIT", "export COMPLETED", path=str(final_video))
        return ExportResult(
            status="COMPLETED", output_path=str(final_video), profile=req.profile_name,
            duration=qc["duration"], resolution=(qc["width"], qc["height"]),
            video_codec=qc["video_codec"], audio_codec=qc["audio_codec"], qc=qc,
            error_code=None, error_detail=None,
        )
    except VideoError as e:
        return ExportResult(
            status="FAILED", output_path=None, profile=req.profile_name,
            duration=0.0, resolution=None, video_codec=None, audio_codec=None,
            qc={"ok": False}, error_code=e.code.value, error_detail=e.detail,
        )

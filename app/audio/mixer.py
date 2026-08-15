"""Audio mixing engine (Phase 9).

Deterministic FFmpeg-based final audio mixing supporting:

    VOICE + MUSIC + SFX + AMBIENCE

with independent volume levels, music ducking under voice, fades, normalization,
scene transitions, silence handling, and clipping prevention.

Never overwrites source assets — the mixed output is a new file. Never reports
success without a verified, FFmpeg-decodable output.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..config import OUTPUT_DIR
from ..core.errors import TypedErrorCode, VideoError
from ..core.logging import log
from .qc import AudioError, verify_audio
import subprocess


class MixError(VideoError):
    def __init__(self, code: TypedErrorCode, detail: str = "", *, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(code, detail, context=context)


@dataclass
class MixTrack:
    """One track in a mix."""

    path: Path
    kind: str  # "voice" | "music" | "sfx" | "ambience"
    volume: float = 1.0  # linear gain multiplier
    start_offset: float = 0.0  # seconds into the final mix
    duration: Optional[float] = None  # None = use full source
    fade_in: float = 0.0
    fade_out: float = 0.0
    duck_when_voice: bool = False  # music only: lower volume while voice plays

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path), "kind": self.kind, "volume": self.volume,
            "start_offset": self.start_offset, "duration": self.duration,
            "fade_in": self.fade_in, "fade_out": self.fade_out,
            "duck_when_voice": self.duck_when_voice,
        }


@dataclass
class MixOptions:
    """Global mix options."""

    target_duration: Optional[float] = None  # None = max of tracks
    music_duck_level: float = 0.25  # music volume when ducked (0-1)
    normalize: bool = True
    fade_in: float = 0.0
    fade_out: float = 0.0
    output_format: str = "mp3"
    sample_rate: int = 44100

    def validate(self) -> None:
        if not (0.0 <= self.music_duck_level <= 1.0):
            raise MixError(TypedErrorCode.WORKFLOW_INVALID, f"music_duck_level {self.music_duck_level} out of [0,1].")
        if self.target_duration is not None and self.target_duration <= 0:
            raise MixError(TypedErrorCode.WORKFLOW_INVALID, "target_duration must be > 0.")
        if self.sample_rate <= 0:
            raise MixError(TypedErrorCode.WORKFLOW_INVALID, "sample_rate must be > 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_duration": self.target_duration, "music_duck_level": self.music_duck_level,
            "normalize": self.normalize, "fade_in": self.fade_in, "fade_out": self.fade_out,
            "output_format": self.output_format, "sample_rate": self.sample_rate,
        }


@dataclass
class MixResult:
    path: Path
    duration: float
    tracks: list[MixTrack]
    sample_rate: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path), "duration": self.duration,
            "sample_rate": self.sample_rate, "tracks": [t.to_dict() for t in self.tracks],
        }


def _build_filter(tracks: list[MixTrack], options: MixOptions, voice_duration: float) -> str:
    """Build the FFmpeg filter_complex for mixing.

    Implements per-track volume, fades, start offsets, music ducking, and a
    final normalization/clipping-prevention stage.
    """
    chains: list[str] = []
    mix_labels: list[str] = []
    for i, t in enumerate(tracks):
        # Each input: delay to start_offset, apply volume + fades.
        parts: list[str] = []
        # Ensure consistent sample rate / channel layout.
        parts.append(f"[{i}:a]aresample={options.sample_rate},aformat=channel_layouts=stereo")
        label = f"a{i}"
        if t.start_offset > 0:
            parts.append(f"adelay={int(t.start_offset*1000)}|{int(t.start_offset*1000)}")
        vol = t.volume
        if t.duck_when_voice and voice_duration > 0:
            # Duck music while voice plays, restore after.
            duck = options.music_duck_level
            # volume expression: duck while t<voice_duration else full.
            vol_expr = f"volume='if(lt(t\\,{voice_duration}),{duck}*{vol},{vol})':eval=frame"
            parts.append(vol_expr)
        else:
            parts.append(f"volume={vol}")
        if t.fade_in > 0:
            parts.append(f"afade=t=in:st=0:d={t.fade_in}")
        if t.fade_out > 0 and (t.duration or 0) > 0:
            st = (t.duration or 0) - t.fade_out
            if st > 0:
                parts.append(f"afade=t=out:st={st}:d={t.fade_out}")
        chains.append(",".join(parts) + f"[{label}]")
        mix_labels.append(f"[{label}]")
    # Mix all tracks.
    n = len(mix_labels)
    mix_in = "".join(mix_labels)
    chains.append(f"{mix_in}amix=inputs={n}:normalize=0:duration=longest[mixed]")
    # Global fades + normalization + clipping prevention (all on [mixed] -> [out]).
    # A chain starting with an input label connects directly to the first filter
    # (no comma after [mixed]); subsequent filters ARE comma-separated.
    post_filters: list[str] = []
    if options.fade_in > 0:
        post_filters.append(f"afade=t=in:st=0:d={options.fade_in}")
    if options.fade_out > 0 and options.target_duration:
        st = options.target_duration - options.fade_out
        if st > 0:
            post_filters.append(f"afade=t=out:st={st}:d={options.fade_out}")
    if options.normalize:
        post_filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
    # Hard limiter for clipping prevention.
    post_filters.append("alimiter=limit=0.95")
    chains.append("[mixed]" + ",".join(post_filters) + "[out]")
    return ";".join(chains)


def mix_audio(tracks: list[MixTrack], destination: Path,
              options: Optional[MixOptions] = None) -> MixResult:
    """Mix multiple audio tracks into ``destination`` with FFmpeg.

    Implements independent volumes, music ducking under voice, fades,
    normalization, and clipping prevention. Never overwrites source assets.
    Raises MixError on any FFmpeg failure; verifies the output is decodable.
    """
    if not tracks:
        raise MixError(TypedErrorCode.WORKFLOW_INVALID, "No tracks supplied to mix.")
    options = options or MixOptions()
    options.validate()
    for t in tracks:
        if not Path(t.path).exists():
            raise MixError(TypedErrorCode.INVALID_AUDIO, f"Track missing: {t.path}", context={"path": str(t.path)})

    # Determine voice duration for ducking.
    voice_tracks = [t for t in tracks if t.kind == "voice"]
    voice_duration = max((t.duration or 0.0) for t in voice_tracks) if voice_tracks else 0.0

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    inputs: list[str] = []
    for t in tracks:
        inputs += ["-i", str(t.path)]
    filt = _build_filter(tracks, options, voice_duration)
    codec = "libmp3lame" if options.output_format == "mp3" else "pcm_s16le"
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", filt,
           "-map", "[out]", "-c:a", codec, "-ar", str(options.sample_rate),
           "-t", str(options.target_duration or 9999), str(destination)]
    log("AUDIO", f"mixing {len(tracks)} tracks -> {destination}")
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise MixError(
            TypedErrorCode.FFMPEG_ERROR,
            f"FFmpeg mix failed (exit {p.returncode}): {p.stderr[-1000:].strip()}",
            context={"cmd": [str(c) for c in cmd[:6]], "stderr": p.stderr[-500:]},
        )
    # Verify the mixed output is real and decodable.
    qc = verify_audio(destination)
    log("AUDIO", "mix verified", path=str(destination), duration=qc["duration"])
    return MixResult(path=destination, duration=qc["duration"], tracks=tracks,
                     sample_rate=qc["sample_rate"])


def mix_scene_audio(
    voice: Optional[Path], music: Optional[Path], sfx: list[Path],
    ambience: Optional[Path], destination: Path, *,
    scene_duration: float, voice_start_offset: float = 0.0,
    music_volume: float = 0.6, voice_volume: float = 1.0,
    sfx_volume: float = 0.5, ambience_volume: float = 0.3,
    music_duck: bool = True, duck_level: float = 0.25,
    fade_in: float = 0.0, fade_out: float = 0.0,
) -> MixResult:
    """Convenience: mix one scene's audio tracks with sensible defaults."""
    tracks: list[MixTrack] = []
    if voice is not None:
        tracks.append(MixTrack(path=voice, kind="voice", volume=voice_volume,
                               start_offset=voice_start_offset, duck_when_voice=False))
    if music is not None:
        tracks.append(MixTrack(path=music, kind="music", volume=music_volume,
                               duck_when_voice=music_duck, fade_in=fade_in, fade_out=fade_out))
    for s in sfx:
        tracks.append(MixTrack(path=s, kind="sfx", volume=sfx_volume))
    if ambience is not None:
        tracks.append(MixTrack(path=ambience, kind="ambience", volume=ambience_volume))
    if not tracks:
        raise MixError(TypedErrorCode.WORKFLOW_INVALID, "No audio tracks to mix for scene.")
    options = MixOptions(target_duration=scene_duration, music_duck_level=duck_level,
                         fade_in=fade_in, fade_out=fade_out)
    return mix_audio(tracks, destination, options)


def _mix_dir(project_id: Optional[str], scene_index: Optional[int]) -> Path:
    run_id = uuid.uuid4().hex[:12]
    return Path(OUTPUT_DIR) / (project_id or "standalone") / f"scene_{scene_index or 0}" / run_id / "mix"

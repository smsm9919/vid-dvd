"""FFmpeg compositor (Phase 10).

Deterministic FFmpeg post-production: normalize each scene to the export
profile, concatenate, integrate mixed audio, and optionally burn in captions
and brand overlays. Never overwrites source assets.

Burned-in captions use the FFmpeg `subtitles` filter (libass). If libass/font
rendering support is unavailable, the compositor raises a typed
CAPTION_RENDER_ERROR rather than silently producing broken output (especially
important for Arabic RTL, which requires proper shaping support).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..core.errors import TypedErrorCode, VideoError
from ..core.logging import log
from ..media import ffprobe_available, probe, verify_mp4
from .profiles import ExportProfile
from .timeline import EditingError, Timeline, TimelineScene
from .transitions import TransitionSpec, TransitionType, transition_filter


def _ffmpeg_has_filter(name: str) -> bool:
    """Check whether an FFmpeg filter is available."""
    p = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        return False
    for line in p.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-1] == name:
            return True
        # format: " ... name  description"
        for tok in parts:
            if tok == name:
                return True
    return False


def captions_renderable() -> bool:
    """True if this FFmpeg can render burned-in subtitles (libass subtitles filter)."""
    return _ffmpeg_has_filter("subtitles")


def font_available(font_name: str) -> bool:
    """Check a font is resolvable via fontconfig."""
    try:
        p = subprocess.run(["fc-list", f":family={font_name}"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return bool(p.stdout.strip())
    except FileNotFoundError:
        return False


def _normalize_scene_video(src: Path, dest: Path, profile: ExportProfile,
                           *, transition: TransitionSpec) -> Path:
    """Normalize one scene clip to the profile resolution/fps/pixfmt.

    Uses scale + pad (preserve aspect ratio, no stretching) and sets fps.
    Applies an in-fade when the scene uses a fade transition (handled by
    per-clip fade since xfade needs overlap that concat doesn't give).
    """
    w, h = profile.width, profile.height
    # Scale preserving aspect, then pad to exact target. Avoids stretching.
    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"fps={profile.fps},format={profile.pixel_format}"
    )
    if transition.type == TransitionType.FADE and transition.duration > 0:
        # Video fade in for the scene (transition fade).
        vf += f",fade=t=in:st=0:d={transition.duration}"
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", vf,
        "-c:v", profile.video_codec, "-preset", profile.preset,
        "-crf", str(profile.crf), "-pix_fmt", profile.pixel_format,
        "-an",  # drop source audio; we mix our own
        "-t", _probe_duration_safe(src),
        str(dest),
    ]
    log("EDIT", f"normalize scene {src.name} -> {dest.name} ({w}x{h}@{profile.fps})")
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise EditingError(
            TypedErrorCode.FFMPEG_ERROR,
            f"Scene normalization failed (exit {p.returncode}): {p.stderr[-800:].strip()}",
            context={"src": str(src), "stderr": p.stderr[-400:]},
        )
    return dest


def _probe_duration_safe(path: Path) -> str:
    """Best-effort duration string; falls back to a large value."""
    try:
        data = probe(path)
        d = float(data.get("format", {}).get("duration") or 0)
        return f"{d:.3f}" if d > 0 else "9999"
    except VideoError:
        return "9999"


def _concat_scenes(normalized_paths: list[Path], dest: Path, profile: ExportProfile) -> Path:
    """Concatenate normalized scene clips. Uses re-encoding for safety."""
    if len(normalized_paths) == 1:
        # Re-mux single clip to target container/codec.
        cmd = ["ffmpeg", "-y", "-i", str(normalized_paths[0]),
               "-c:v", profile.video_codec, "-preset", profile.preset,
               "-crf", str(profile.crf), "-pix_fmt", profile.pixel_format,
               "-an", str(dest)]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            raise EditingError(
                TypedErrorCode.EXPORT_FAILED,
                f"Single-scene export failed: {p.stderr[-600:].strip()}",
                context={"stderr": p.stderr[-300:]})
        return dest
    listing = dest.with_suffix(".concat.txt")
    listing.write_text("\n".join(f"file '{p.resolve().as_posix()}'" for p in normalized_paths),
                       encoding="utf-8")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
        "-c:v", profile.video_codec, "-preset", profile.preset,
        "-crf", str(profile.crf), "-pix_fmt", profile.pixel_format,
        "-an", str(dest),
    ]
    log("EDIT", f"concat {len(normalized_paths)} scenes -> {dest.name}")
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    listing.unlink(missing_ok=True)
    if p.returncode != 0:
        raise EditingError(
            TypedErrorCode.EXPORT_FAILED,
            f"Concatenation failed (exit {p.returncode}): {p.stderr[-800:].strip()}",
            context={"stderr": p.stderr[-400:]},
        )
    return dest


def _build_ass_from_cues(cues: list[dict[str, Any]], style: dict[str, Any],
                         width: int, height: int) -> str:
    """Build an ASS subtitle script from caption cues dicts.

    Style parameters come from the Phase 9 caption style profile. Positions map
    to ASS alignment. Uses a DejaVu Sans fallback font that supports Latin and
    basic glyphs; for Arabic, the compositor validates shaping support exists.
    """
    font = style.get("font", "DejaVu Sans")
    fs = style.get("font_size", 36)
    color = _hex_to_ass_color(style.get("font_color", "white"))
    stroke = _hex_to_ass_color(style.get("stroke", "black"))
    sw = style.get("stroke_width", 2)
    pos = style.get("position", "bottom")
    # ASS alignment: 2=bottom-center, 5=center, 8=top-center
    align = {"bottom": 2, "center": 5, "top": 8}.get(pos, 2)
    bold = "-1" if style.get("bold") else "0"
    margin_v = int(height * 0.06)
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font},{fs},{color},{stroke},&H80000000,{bold},0,0,0,100,100,0,0,1,{sw},0,{align},20,20,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = [header]
    for c in cues:
        start = _ass_time(c["start"])
        end = _ass_time(c["end"])
        text = c["text"].replace("\n", "\\N")
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")
    return "".join(lines)


def _hex_to_ass_color(name: str) -> str:
    """Convert a color name/hex to ASS &HBBGGRR format."""
    named = {
        "white": "&H00FFFFFF", "black": "&H00000000", "yellow": "&H0000FFFF",
        "red": "&H000000FF", "blue": "&H00FF0000", "green": "&H0000FF00",
    }
    return named.get(name.lower(), name if name.startswith("&H") else "&H00FFFFFF")


def _ass_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs == 100:
        s += 1
        cs = 0
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _burn_captions(video_path: Path, dest: Path, ass_script: Path,
                   profile: ExportProfile) -> Path:
    """Burn an ASS script into the video via the subtitles filter."""
    if not captions_renderable():
        raise EditingError(
            TypedErrorCode.CAPTION_RENDER_ERROR,
            "FFmpeg lacks the 'subtitles' filter (libass); cannot burn captions. "
            "Install ffmpeg with libass/fontconfig support.",
            context={"filter": "subtitles"},
        )
    # Escape colons/backslashes in the path for the filter.
    esc = str(ass_script).replace("\\", "\\\\").replace(":", "\\:")
    vf = f"subtitles='{esc}'"
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vf", vf,
           "-c:v", profile.video_codec, "-preset", profile.preset,
           "-crf", str(profile.crf), "-pix_fmt", profile.pixel_format,
           "-an", str(dest)]
    log("EDIT", f"burn captions -> {dest.name}")
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise EditingError(
            TypedErrorCode.CAPTION_RENDER_ERROR,
            f"Caption burn-in failed: {p.stderr[-800:].strip()}",
            context={"stderr": p.stderr[-400:]},
        )
    return dest


def _apply_branding(video_path: Path, dest: Path, brand: dict[str, Any],
                   profile: ExportProfile) -> Path:
    """Apply optional brand overlay (logo + watermark/CTA text).

    Branding is optional and applied only during final composition — never to
    source assets. Uses overlay for a logo and drawtext for CTA/watermark.
    """
    filters: list[str] = []
    inputs: list[str] = []
    n_extra = 0
    logo = brand.get("logo_path")
    if logo and Path(logo).exists():
        inputs += ["-i", str(logo)]
        # Scale logo, position bottom-right.
        filters.append(f"[0:v][{1}]overlay=W-w-30:H-h-30[bg]")
        # Note: with one extra input, the main video is [0], logo is [1].
        n_extra = 1
    cta = brand.get("cta")
    watermark = brand.get("watermark")
    texts = []
    if cta:
        texts.append(("drawtext", f"text='{cta}':fontcolor=white:fontsize={int(profile.height*0.04)}:"
                                   f"x=(w-text_w)/2:y=h-text_h-40:box=1:boxcolor=black@0.5"))
    if watermark:
        texts.append(("drawtext", f"text='{watermark}':fontcolor=white@0.6:fontsize={int(profile.height*0.03)}:"
                                  f"x=20:y=20"))
    if logo and texts:
        # Apply drawtext after overlay.
        post = "[bg]" + ",".join(f"{t}={expr}" for t, expr in texts) + "[out]"
        filters.append(post)
    elif not logo and texts:
        filters.append("[0:v]" + ",".join(f"{t}={expr}" for t, expr in texts) + "[out]")
    elif logo and not texts:
        pass  # overlay already produces [bg]; rename
    else:
        # No branding elements; just copy.
        import shutil
        shutil.copy2(video_path, dest)
        return dest

    if logo and not texts:
        filt = filters[0].replace("[bg]", "[out]")
        filt_chain = filt
    else:
        filt_chain = ";".join(filters)
    cmd = ["ffmpeg", "-y", "-i", str(video_path)] + inputs + [
        "-filter_complex", filt_chain, "-map", "[out]",
        "-c:v", profile.video_codec, "-preset", profile.preset,
        "-crf", str(profile.crf), "-pix_fmt", profile.pixel_format,
        "-an", str(dest),
    ]
    log("EDIT", f"apply branding -> {dest.name}")
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise EditingError(
            TypedErrorCode.EXPORT_FAILED,
            f"Branding failed: {p.stderr[-800:].strip()}",
            context={"stderr": p.stderr[-400:]},
        )
    return dest


def _mux_audio(video_path: Path, audio_path: Path, dest: Path,
               profile: ExportProfile) -> Path:
    """Mux a mixed audio track onto the video.

    Audio/video sync is explicit: the audio is padded with silence if shorter
    than the video and trimmed if longer, so the final container matches the
    video duration (no silent truncation of the video stream).
    """
    v_dur = float(_probe_duration_safe(video_path))
    # Pad short audio (apad) and trim long audio (atrim) to exactly video length.
    afilt = f"[1:a]apad,atrim=0:{v_dur:.3f},asetpts=N/SR/TB[aout]"
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path),
           "-filter_complex", afilt,
           "-map", "0:v:0", "-map", "[aout]",
           "-c:v", "copy", "-c:a", profile.audio_codec,
           "-b:a", profile.audio_bitrate, "-ar", str(profile.audio_sample_rate),
           "-t", f"{v_dur:.3f}", str(dest)]
    log("EDIT", f"mux audio -> {dest.name}")
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise EditingError(
            TypedErrorCode.EXPORT_FAILED,
            f"Audio mux failed: {p.stderr[-800:].strip()}",
            context={"stderr": p.stderr[-400:]},
        )
    return dest


def _silent_audio(video_path: Path, dest: Path, profile: ExportProfile) -> Path:
    """Add a silent audio track so the output has both streams."""
    duration = _probe_duration_safe(video_path)
    cmd = ["ffmpeg", "-y", "-i", str(video_path),
           "-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate={profile.audio_sample_rate}",
           "-c:v", "copy", "-c:a", profile.audio_codec, "-b:a", profile.audio_bitrate,
           "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(dest)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise EditingError(
            TypedErrorCode.EXPORT_FAILED,
            f"Silent audio mux failed: {p.stderr[-600:].strip()}",
            context={"stderr": p.stderr[-300:]},
        )
    return dest

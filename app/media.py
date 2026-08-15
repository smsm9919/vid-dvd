import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .core.errors import (
    TypedErrorCode,
    VideoError,
)


class MediaError(VideoError):
    """Typed media/FFmpeg error.

    Backwards-compatible with the legacy ``MediaError(RuntimeError)`` name, but
    now carries a :class:`~app.core.errors.TypedErrorCode`. By default media
    failures map to ``FFMPEG_ERROR``; output-integrity failures use
    ``INVALID_MP4``.
    """

    def __init__(self, detail: str, *, code: TypedErrorCode = TypedErrorCode.FFMPEG_ERROR, **context: Any) -> None:
        super().__init__(code, detail, context=context or None)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def run(cmd):
    """Run a subprocess command, raising a typed FFMPEG_ERROR on failure."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        raise ffmpeg_error(
            f"Command failed (exit {p.returncode}): {' '.join(map(str, cmd[:3]))}…",
            returncode=p.returncode,
            stderr=p.stderr[-5000:],
            cmd=[str(c) for c in cmd],
        )
    return p


def concat_videos(paths, output):
    """Concatenate clips into ``output``.

    Uses re-encoding (``-c:v libx264``) instead of ``-c copy`` so that clips
    with differing codecs, resolutions, or timebases concatenate safely. The
    legacy ``-c copy`` concat demuxer silently produced broken MP4s when source
    clips had mismatched parameters — a common case with ComfyUI scene outputs.
    """
    paths = [Path(p) for p in paths]
    if not paths:
        raise MediaError("No clips supplied.")
    if len(paths) == 1:
        shutil.copy2(paths[0], output)
        return output
    listing = Path(output).with_suffix(".txt")
    listing.write_text("\n".join(f"file '{p.resolve().as_posix()}'" for p in paths), encoding="utf-8")
    try:
        run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(output),
        ])
    finally:
        listing.unlink(missing_ok=True)
    return output


def normalize_vertical(input_path, output_path):
    """Normalize to 1080x1920 vertical (9:16) with H.264/AAC."""
    vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
    run([
        "ffmpeg", "-y", "-i", str(input_path), "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(output_path),
    ])
    return output_path


def probe(path) -> dict[str, Any]:
    """Probe a media file with ffprobe and return parsed JSON.

    Raises MediaError(FFMPEG_ERROR) if ffprobe is missing or fails,
    MediaError(INVALID_MP4) if the file has no video stream.
    """
    path = Path(path)
    if not ffprobe_available():
        raise MediaError("ffprobe not found on PATH; cannot probe media.", code=TypedErrorCode.FFMPEG_ERROR, path=str(path))
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if p.returncode != 0:
        raise MediaError(
            f"ffprobe cannot read file: {p.stderr[-1000:].strip()}",
            code=TypedErrorCode.INVALID_MP4,
            path=str(path), stderr=p.stderr[-500:],
        )
    try:
        data = json.loads(p.stdout or "{}")
    except json.JSONDecodeError as e:
        raise MediaError(f"ffprobe returned non-JSON: {e}", code=TypedErrorCode.INVALID_MP4, path=str(path))
    return data


def verify_mp4(path) -> dict[str, Any]:
    """Verify an MP4 is real and readable.

    Checks (in order): file exists, size > 0, ffprobe-readable, has a video
    stream, has a codec, and (when present) reports duration/resolution/fps.
    Returns a structured QC report. Raises MediaError(INVALID_MP4) /
    MediaError(FFMPEG_ERROR) on any failure. Never reports success for a
    missing or corrupt file.
    """
    path = Path(path)
    if not path.exists():
        raise MediaError(f"File does not exist: {path}", code=TypedErrorCode.INVALID_MP4, path=str(path))
    size = path.stat().st_size
    if size <= 0:
        raise MediaError(f"File is empty (0 bytes): {path}", code=TypedErrorCode.INVALID_MP4, path=str(path), size=size)

    data = probe(path)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError("No video stream found.", code=TypedErrorCode.INVALID_MP4, path=str(path), streams=streams)
    codec = video.get("codec_name")
    if not codec:
        raise MediaError("Video stream has no codec.", code=TypedErrorCode.INVALID_MP4, path=str(path))

    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    report = {
        "path": str(path),
        "size": size,
        "duration": float(data.get("format", {}).get("duration") or video.get("duration") or 0),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "fps": _parse_fps(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        "video_codec": codec,
        "audio_codec": audio.get("codec_name") if audio else None,
        "has_audio": audio is not None,
        "ok": True,
    }
    return report


def _parse_fps(rate: str) -> float:
    if not rate or rate == "0/0":
        return 0.0
    if "/" in rate:
        try:
            num, den = rate.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        except ValueError:
            return 0.0
    try:
        return float(rate)
    except ValueError:
        return 0.0

"""Tests for the media/FFmpeg layer (Phase 3).

These tests are RUNTIME tests: they require a real ``ffmpeg``/``ffprobe`` on
PATH and generate real tiny MP4s with FFmpeg's lavfi source. They verify:
  - concat re-encodes safely across differing source codecs
  - verify_mp4 accepts a real MP4 and rejects corrupt/empty/missing files
  - MediaError / typed FFMPEG_ERROR / INVALID_MP4 are raised correctly
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from app.media import (
    MediaError,
    concat_videos,
    ffmpeg_available,
    ffprobe_available,
    normalize_vertical,
    probe,
    verify_mp4,
)
from app.core.errors import TypedErrorCode

needs_ffmpeg = pytest.mark.skipif(
    not (ffmpeg_available() and ffprobe_available()),
    reason="ffmpeg/ffprobe not available on PATH",
)


def _make_clip(path: Path, duration: float = 1.0, color: str = "red", w: int = 320, h: int = 240, codec: str = "libx264"):
    """Generate a real, tiny MP4 using FFmpeg's lavfi test source."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s={w}x{h}:d={duration}",
            "-c:v", codec, "-pix_fmt", "yuv420p", "-t", str(duration), str(path),
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )


@needs_ffmpeg
def test_verify_mp4_accepts_real_mp4(tmp_path):
    clip = tmp_path / "clip.mp4"
    _make_clip(clip, color="blue")
    report = verify_mp4(clip)
    assert report["ok"] is True
    assert report["video_codec"] == "h264"
    assert report["width"] == 320 and report["height"] == 240
    assert report["size"] > 0
    assert report["duration"] > 0


@needs_ffmpeg
def test_verify_mp4_rejects_missing(tmp_path):
    with pytest.raises(MediaError) as e:
        verify_mp4(tmp_path / "nope.mp4")
    assert e.value.code is TypedErrorCode.INVALID_MP4


@needs_ffmpeg
def test_verify_mp4_rejects_empty(tmp_path):
    f = tmp_path / "empty.mp4"
    f.write_bytes(b"")
    with pytest.raises(MediaError) as e:
        verify_mp4(f)
    assert e.value.code is TypedErrorCode.INVALID_MP4


@needs_ffmpeg
def test_verify_mp4_rejects_corrupt(tmp_path):
    f = tmp_path / "corrupt.mp4"
    f.write_bytes(b"definitely not an mp4 file")
    with pytest.raises(MediaError) as e:
        verify_mp4(f)
    assert e.value.code is TypedErrorCode.INVALID_MP4


@needs_ffmpeg
def test_concat_reencodes_mismatched_codecs(tmp_path):
    """Two clips with different resolutions must concatenate safely (re-encode)."""
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    _make_clip(a, color="red", w=320, h=240)
    _make_clip(b, color="green", w=480, h=360)
    out = tmp_path / "out.mp4"
    concat_videos([a, b], out)
    # The legacy -c copy path would have failed or produced a broken file here.
    report = verify_mp4(out)
    assert report["ok"] is True
    assert report["video_codec"] == "h264"


@needs_ffmpeg
def test_concat_single_clip(tmp_path):
    a = tmp_path / "a.mp4"
    _make_clip(a, color="red")
    out = tmp_path / "out.mp4"
    concat_videos([a], out)
    assert out.exists() and out.stat().st_size > 0


def test_concat_empty_raises():
    with pytest.raises(MediaError) as e:
        concat_videos([], Path("/tmp/x.mp4"))
    assert e.value.code is TypedErrorCode.FFMPEG_ERROR


@needs_ffmpeg
def test_normalize_vertical(tmp_path):
    src = tmp_path / "src.mp4"
    _make_clip(src, color="gray", w=640, h=480)
    out = tmp_path / "v.mp4"
    normalize_vertical(src, out)
    report = verify_mp4(out)
    assert report["ok"] is True
    assert report["width"] == 1080 and report["height"] == 1920


@needs_ffmpeg
def test_probe_returns_streams(tmp_path):
    clip = tmp_path / "clip.mp4"
    _make_clip(clip, color="red")
    data = probe(clip)
    assert any(s.get("codec_type") == "video" for s in data.get("streams", []))

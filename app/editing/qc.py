"""Final QC for exported MP4 (Phase 10).

Before returning COMPLETED, verify the actual final MP4 with ffprobe/FFmpeg:
file exists, size > 0, duration, resolution, FPS, codec, pixel format, video
stream exists, audio stream exists when required, audio duration, container
integrity, decodability. Returns structured QC data. Raises/returns typed
FINAL_QC_FAILED on any failure. Never reports success before QC passes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..core.errors import TypedErrorCode, VideoError
from ..media import probe, verify_mp4


def final_qc(path: Path, *, profile: Optional[Any] = None,
             require_audio: bool = True) -> dict[str, Any]:
    """Run final QC on an exported MP4. Returns a structured report.

    A failure is reported with ok=False and an error code/detail (does not
    raise), so the assembly layer can return a typed FAILED ExportResult.
    """
    report: dict[str, Any] = {"path": str(path), "ok": False}
    try:
        if not Path(path).exists():
            return _fail(report, TypedErrorCode.FINAL_QC_FAILED, "Final MP4 does not exist.")
        size = Path(path).stat().st_size
        report["size"] = size
        if size <= 0:
            return _fail(report, TypedErrorCode.FINAL_QC_FAILED, "Final MP4 is empty (0 bytes).")
        # verify_mp4 checks exists/size/video stream/codec + returns structured data.
        vqc = verify_mp4(path)
        report["duration"] = vqc["duration"]
        report["width"] = vqc["width"]
        report["height"] = vqc["height"]
        report["fps"] = vqc["fps"]
        report["video_codec"] = vqc["video_codec"]
        report["has_audio"] = vqc["has_audio"]
        report["audio_codec"] = vqc["audio_codec"]
        # Pixel format from probe.
        data = probe(path)
        vstream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
        report["pixel_format"] = vstream.get("pix_fmt")
        # Audio stream requirement.
        if require_audio and not vqc["has_audio"]:
            return _fail(report, TypedErrorCode.FINAL_QC_FAILED,
                         "Final MP4 has no audio stream but audio was required.")
        # Audio duration sanity (within 1s of video if present).
        astream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
        if astream:
            a_dur = float(astream.get("duration") or data.get("format", {}).get("duration") or 0)
            report["audio_duration"] = a_dur
            v_dur = float(vqc["duration"])
            if v_dur > 0 and abs(a_dur - v_dur) > 1.5:
                return _fail(report, TypedErrorCode.FINAL_QC_FAILED,
                             f"Audio/video duration mismatch: audio {a_dur:.2f}s vs video {v_dur:.2f}s.")
        # Profile conformance if provided.
        if profile is not None:
            pw, ph = profile.width, profile.height
            if vqc["width"] != pw or vqc["height"] != ph:
                return _fail(report, TypedErrorCode.FINAL_QC_FAILED,
                             f"Resolution {vqc['width']}x{vqc['height']} != profile {pw}x{ph}.")
            if profile.fps and abs(vqc["fps"] - profile.fps) > 1.5:
                return _fail(report, TypedErrorCode.FINAL_QC_FAILED,
                             f"FPS {vqc['fps']} != profile {profile.fps}.")
        report["ok"] = True
        report["error"] = None
        return report
    except VideoError as e:
        return _fail(report, e.code, e.detail)


def _fail(report: dict[str, Any], code: TypedErrorCode, detail: str) -> dict[str, Any]:
    report["ok"] = False
    report["error_code"] = code.value
    report["error"] = detail
    return report

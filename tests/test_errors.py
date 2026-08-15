"""Tests for the typed error architecture (Phase 4)."""

from app.core.errors import (
    TypedErrorCode,
    VideoError,
    ffmpeg_error,
    invalid_mp4,
)


def test_all_required_typed_codes_present():
    required = {
        "COMFYUI_UNREACHABLE",
        "MODEL_NOT_FOUND",
        "WORKFLOW_INVALID",
        "WORKFLOW_REJECTED",
        "GENERATION_TIMEOUT",
        "NO_OUTPUT",
        "INVALID_MP4",
        "FFMPEG_ERROR",
        "NO_PROVIDER",
    }
    present = {c.value for c in TypedErrorCode}
    assert required.issubset(present), f"Missing codes: {required - present}"


def test_video_error_is_runtime_error():
    e = VideoError(TypedErrorCode.NO_PROVIDER, "none available")
    assert isinstance(e, RuntimeError)
    assert e.code is TypedErrorCode.NO_PROVIDER
    assert e.detail == "none available"
    assert e.context == {}


def test_video_error_carries_context():
    e = VideoError(TypedErrorCode.GENERATION_TIMEOUT, "slow", context={"prompt_id": "abc"})
    assert e.context == {"prompt_id": "abc"}
    assert "abc" in str(e)
    assert "[GENERATION_TIMEOUT]" in str(e)


def test_to_dict_roundtrip():
    e = VideoError(TypedErrorCode.INVALID_MP4, "bad", context={"path": "/x.mp4"})
    d = e.to_dict()
    assert d == {"code": "INVALID_MP4", "detail": "bad", "context": {"path": "/x.mp4"}}


def test_convenience_constructors():
    assert ffmpeg_error("boom").code is TypedErrorCode.FFMPEG_ERROR
    assert invalid_mp4("nope").code is TypedErrorCode.INVALID_MP4
    assert ffmpeg_error("boom", cmd=["ffmpeg"]).context == {"cmd": ["ffmpeg"]}


def test_code_from_string():
    # Wire-friendly: codes are str enums, constructable from their value.
    e = VideoError("NO_OUTPUT", "nothing")
    assert e.code is TypedErrorCode.NO_OUTPUT

"""Typed error codes and exceptions for the video production pipeline.

Every failure in a critical generation/media path must raise a :class:`VideoError`
carrying one of the :class:`TypedErrorCode` values. This makes failures machine-
readable and prevents the system from hiding a real failure behind a generic
``RuntimeError``.

The pipeline MUST NEVER simulate successful generation. If a real output MP4 is
missing, unreadable, or fails verification, raise the matching typed error instead.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional


class TypedErrorCode(str, Enum):
    """Stable, wire-friendly typed failure codes."""

    # Provider selection
    NO_PROVIDER = "NO_PROVIDER"

    # ComfyUI transport / workflow
    COMFYUI_UNREACHABLE = "COMFYUI_UNREACHABLE"
    WORKFLOW_INVALID = "WORKFLOW_INVALID"
    WORKFLOW_REJECTED = "WORKFLOW_REJECTED"
    WORKFLOW_NOT_FOUND = "WORKFLOW_NOT_FOUND"

    # Generation runtime
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    GENERATION_TIMEOUT = "GENERATION_TIMEOUT"
    NO_OUTPUT = "NO_OUTPUT"
    INVALID_REFERENCE = "INVALID_REFERENCE"

    # Editing / final assembly (Phase 10)
    MISSING_VIDEO_ASSET = "MISSING_VIDEO_ASSET"
    MISSING_AUDIO_ASSET = "MISSING_AUDIO_ASSET"
    INVALID_TIMELINE = "INVALID_TIMELINE"
    UNSUPPORTED_PROFILE = "UNSUPPORTED_PROFILE"
    TRANSITION_ERROR = "TRANSITION_ERROR"
    CAPTION_RENDER_ERROR = "CAPTION_RENDER_ERROR"
    EXPORT_FAILED = "EXPORT_FAILED"
    FINAL_QC_FAILED = "FINAL_QC_FAILED"

    # Output / media integrity
    INVALID_MP4 = "INVALID_MP4"
    INVALID_AUDIO = "INVALID_AUDIO"
    FFMPEG_ERROR = "FFMPEG_ERROR"

    # Job orchestration (Phase 11)
    INVALID_TRANSITION = "INVALID_TRANSITION"
    DEPENDENCY_FAILED = "DEPENDENCY_FAILED"
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    CACHE_INVALID = "CACHE_INVALID"
    RETRY_EXHAUSTED = "RETRY_EXHAUSTED"
    NON_RETRYABLE = "NON_RETRYABLE"
    CANCEL_NOT_ALLOWED = "CANCEL_NOT_ALLOWED"
    CONCURRENCY_LIMIT = "CONCURRENCY_LIMIT"
    PERSISTENCE_ERROR = "PERSISTENCE_ERROR"

    # Multi-provider hub (Phase 13) — additive
    STOCK_SEARCH_FAILED = "STOCK_SEARCH_FAILED"
    STOCK_DOWNLOAD_FAILED = "STOCK_DOWNLOAD_FAILED"
    PROVIDER_DISABLED = "PROVIDER_DISABLED"
    PAID_PROVIDER_BLOCKED = "PAID_PROVIDER_BLOCKED"
    CAPABILITY_UNSUPPORTED = "CAPABILITY_UNSUPPORTED"
    LICENSE_UNKNOWN = "LICENSE_UNKNOWN"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    INVALID_ASSET = "INVALID_ASSET"
    ASSET_NOT_FOUND = "ASSET_NOT_FOUND"


class VideoError(RuntimeError):
    """A typed pipeline failure.

    Attributes:
        code: a :class:`TypedErrorCode` describing the failure class.
        detail: human/actionable diagnostic text.
        context: optional dict of extra structured diagnostics (prompt_id, path, ...).
    """

    def __init__(
        self,
        code: TypedErrorCode,
        detail: str = "",
        *,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        self.code: TypedErrorCode = TypedErrorCode(code)
        self.detail: str = detail or self.code.value
        self.context: dict[str, Any] = dict(context or {})
        super().__init__(self.__str__())

    def __str__(self) -> str:  # noqa: D401
        ctx = f" context={self.context}" if self.context else ""
        return f"[{self.code.value}] {self.detail}{ctx}"

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code.value, "detail": self.detail, "context": self.context}


def ffmpeg_error(detail: str, **context: Any) -> VideoError:
    """Convenience constructor for FFmpeg failures."""
    return VideoError(TypedErrorCode.FFMPEG_ERROR, detail, context=context or None)


def invalid_mp4(detail: str, **context: Any) -> VideoError:
    """Convenience constructor for unreadable/invalid MP4 output."""
    return VideoError(TypedErrorCode.INVALID_MP4, detail, context=context or None)

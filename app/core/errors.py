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

    # Output / media integrity
    INVALID_MP4 = "INVALID_MP4"
    FFMPEG_ERROR = "FFMPEG_ERROR"


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

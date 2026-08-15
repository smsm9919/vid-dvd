"""VideoProvider abstraction.

A provider is anything that can turn a text prompt (and optionally reference
images) into a real video clip on disk. Concrete providers live in this package
(e.g. :mod:`app.providers.comfyui`). The :mod:`app.providers.registry` selects
the first healthy provider at runtime and raises a typed ``NO_PROVIDER`` error if
none is available.

Contract guarantees:
- ``generate`` MUST return a real, existing, non-empty file path. It MUST NOT
  return a path that does not exist. If generation fails for any reason, it
  raises a :class:`~app.core.errors.VideoError` with a typed code.
- Providers never simulate success.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..core.errors import VideoError


@dataclass
class GenerationRequest:
    """A single clip generation request.

    Attributes:
        prompt: positive visual prompt for the clip.
        negative_prompt: negative prompt (provider may ignore if unsupported).
        duration: desired clip duration in seconds.
        width / height: target resolution (provider may scale/clamp).
        frames: desired frame count (some providers prefer frames over duration).
        fps: target frames per second.
        seed: deterministic seed; ``None`` lets the provider choose.
        reference_images: optional list of image paths for image-to-video / consistency.
        workflow_path: optional override of the provider's default workflow file.
        extra: provider-specific knobs.
    """

    prompt: str
    negative_prompt: str = ""
    duration: float = 4.0
    width: int = 1080
    height: int = 1920
    frames: Optional[int] = None
    fps: float = 24.0
    seed: Optional[int] = None
    reference_images: list[Path] = field(default_factory=list)
    workflow_path: Optional[Path] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderInfo:
    """Static metadata describing a provider."""

    name: str
    kind: str  # e.g. "local", "cloud"
    description: str = ""


@dataclass
class GenerationResult:
    """Result of a successful single-clip generation.

    ``path`` is guaranteed to exist and be non-empty when returned; the provider
    is responsible for verifying the output before returning it.
    """

    path: Path
    prompt_id: Optional[str] = None
    model: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)


class VideoProvider(abc.ABC):
    """Abstract base class for all video generation providers."""

    @property
    @abc.abstractmethod
    def info(self) -> ProviderInfo:
        """Static metadata for this provider."""

    @abc.abstractmethod
    async def detect(self) -> bool:
        """Return True if the provider is reachable and usable right now.

        Must be cheap and side-effect free. Should NOT raise; return False on
        any transport failure.
        """

    @abc.abstractmethod
    async def health(self) -> dict[str, Any]:
        """Return a structured health report.

        Keys commonly include ``ok`` (bool) and ``data`` (provider-specific).
        Raises :class:`VideoError` only if detection itself is ambiguous.
        """

    @abc.abstractmethod
    async def list_models(self) -> list[str]:
        """Return the list of model identifiers the provider can see.

        Raises a typed :class:`VideoError` (e.g. ``COMFYUI_UNREACHABLE``) if the
        provider cannot be queried.
        """

    @abc.abstractmethod
    async def validate(self) -> dict[str, Any]:
        """Validate provider prerequisites (endpoint, workflow, required models).

        Returns a dict with at least ``ok`` (bool) and ``issues`` (list[str]).
        Raises a typed :class:`VideoError` for hard transport failures.
        """

    @abc.abstractmethod
    async def generate(self, request: GenerationRequest, destination: Path) -> GenerationResult:
        """Generate a single video clip and write it to ``destination``.

        On success, ``destination`` MUST exist and be non-empty and the returned
        ``GenerationResult.path`` MUST equal ``destination``. On any failure,
        raise a :class:`VideoError` with the appropriate typed code. Never
        simulate success.
        """

    # -- optional capabilities -------------------------------------------------
    def supports_image_to_video(self) -> bool:
        return False

    def supports_image_sequence(self) -> bool:
        return False


class ProviderUnavailableError(VideoError):
    """Raised by the registry when no provider can satisfy a request."""

    def __init__(self, detail: str = "No video provider available.", **context: Any) -> None:
        from ..core.errors import TypedErrorCode

        super().__init__(TypedErrorCode.NO_PROVIDER, detail, context=context or None)

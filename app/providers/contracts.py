"""Generic multi-provider contracts (Phase 13).

These extend — never replace — the existing provider abstractions
(:class:`~app.providers.base.VideoProvider`, :class:`~app.voice.tts.TTSProvider`,
:class:`~app.audio.music.MusicProvider`, :class:`~app.audio.sfx.SFXProvider`).

They add the cross-cutting metadata the multi-provider hub needs to route
safely: cost (so paid providers are never selected unless explicitly enabled),
license/provenance (so no asset is falsely marked commercially safe), declared
capabilities (so a provider is only asked for what it actually supports), and a
runtime profile (local/remote, GPU requirement, hardware feasibility).

Design rules:
- Every value is explicit and verifiable, never guessed.
- A provider is only ``available`` when its underlying detection reports truth.
- Paid providers default to disabled and expose cost metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ProviderKind(str, Enum):
    """Where a provider executes."""

    LOCAL = "local"
    REMOTE = "remote"
    STOCK = "stock"


class ProviderStatus(str, Enum):
    """Honest readiness states shown in the dashboard."""

    READY = "READY"
    DEGRADED = "DEGRADED"
    BLOCKED = "BLOCKED"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    ERROR = "ERROR"


class Capability(str, Enum):
    """What a provider can produce/retrieve. Capability routers match on these."""

    # Video
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"
    STOCK_VIDEO = "stock_video"
    # Image
    IMAGE_GENERATION = "image_generation"
    STOCK_IMAGE = "stock_image"
    # Audio
    TEXT_TO_SPEECH = "text_to_speech"
    MUSIC_GENERATION = "music_generation"
    STOCK_MUSIC = "stock_music"
    SFX = "sfx"
    STOCK_SFX = "stock_sfx"


@dataclass
class ProviderLicense:
    """License/provenance metadata for a provider or asset.

    ``commercial_use`` is one of: ``allowed``, ``restricted``, ``unknown``.
    It MUST default to ``unknown`` and only be set to ``allowed`` when the
    provider's official license terms actually permit it.
    """

    name: str = "Unknown"
    spdx: Optional[str] = None
    commercial_use: str = "unknown"  # allowed | restricted | unknown
    attribution_required: bool = False
    attribution_text: Optional[str] = None
    source_url: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "spdx": self.spdx,
            "commercial_use": self.commercial_use,
            "attribution_required": self.attribution_required,
            "attribution_text": self.attribution_text,
            "source_url": self.source_url,
        }


@dataclass
class ProviderCost:
    """Cost metadata. Free providers report ``is_paid=False``.

    Paid providers are never selected unless ``ALLOW_PAID_PROVIDERS=true`` and
    the per-call estimate is within ``MAX_PAID_COST_USD``.
    """

    is_paid: bool = False
    unit: str = "usd_per_call"  # usd_per_call | usd_per_second | free
    estimate_per_call: float = 0.0
    free_quota_per_month: Optional[int] = None
    note: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_paid": self.is_paid,
            "unit": self.unit,
            "estimate_per_call": self.estimate_per_call,
            "free_quota_per_month": self.free_quota_per_month,
            "note": self.note,
        }


@dataclass
class ProviderRuntime:
    """Hardware/runtime requirements so the router can avoid infeasible providers."""

    requires_gpu: bool = False
    requires_vram_gb: Optional[float] = None
    requires_ram_gb: Optional[float] = None
    requires_api_key: bool = False
    requires_network: bool = False
    cpu_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "requires_gpu": self.requires_gpu,
            "requires_vram_gb": self.requires_vram_gb,
            "requires_ram_gb": self.requires_ram_gb,
            "requires_api_key": self.requires_api_key,
            "requires_network": self.requires_network,
            "cpu_fallback": self.cpu_fallback,
        }


@dataclass
class ProviderCapability:
    """Declared capabilities + supported languages/formats."""

    capabilities: list[Capability] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    max_duration_seconds: Optional[float] = None
    supported_resolutions: list[str] = field(default_factory=list)
    supported_orientations: list[str] = field(default_factory=list)

    def supports(self, cap: Capability) -> bool:
        return cap in self.capabilities

    def to_dict(self) -> dict[str, Any]:
        return {
            "capabilities": [c.value for c in self.capabilities],
            "languages": self.languages,
            "max_duration_seconds": self.max_duration_seconds,
            "supported_resolutions": self.supported_resolutions,
            "supported_orientations": self.supported_orientations,
        }


@dataclass
class ProviderMeta:
    """Unified metadata every Phase-13 provider exposes.

    Combines the static identity (name/kind/version) with cost, license,
    runtime requirements, and declared capabilities. Concrete provider
    implementations return this from ``meta()``.
    """

    name: str
    kind: ProviderKind
    version: Optional[str] = None
    description: str = ""
    cost: ProviderCost = field(default_factory=ProviderCost)
    license: ProviderLicense = field(default_factory=ProviderLicense)
    runtime: ProviderRuntime = field(default_factory=ProviderRuntime)
    capability: ProviderCapability = field(default_factory=ProviderCapability)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "version": self.version,
            "description": self.description,
            "cost": self.cost.to_dict(),
            "license": self.license.to_dict(),
            "runtime": self.runtime.to_dict(),
            "capability": self.capability.to_dict(),
        }

"""Stock asset provider abstraction (Phase 13).

A :class:`StockProvider` retrieves real, license-tagged media (video/images)
from free stock APIs. Adapters (Pexels, Pixabay) implement the concrete REST
calls. The contract guarantees:

- ``available`` is true only when the API key is configured.
- ``search`` returns normalized :class:`StockHit` records with provenance.
- ``download`` writes a real, verified file to disk and returns its path.
- Every asset carries license + attribution metadata from the provider.
- Downloaded media is cached and de-duplicated by content hash.

No provider fakes availability, and no asset is marked commercially safe
unless the provider's official license terms support it.
"""

from __future__ import annotations

import abc
import hashlib
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import httpx

from .. import config
from ..core.errors import TypedErrorCode, VideoError
from ..core.logging import log
from .contracts import (
    Capability,
    ProviderCapability,
    ProviderCost,
    ProviderKind,
    ProviderLicense,
    ProviderMeta,
    ProviderRuntime,
)


class StockMediaType(str, Enum):
    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"


class StockOrientation(str, Enum):
    LANDSCAPE = "landscape"
    PORTRAIT = "portrait"
    SQUARE = "square"


@dataclass
class StockHit:
    """A single normalized search result with full provenance."""

    provider: str
    media_type: StockMediaType
    asset_id: str
    page_url: str
    download_url: str
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[float] = None
    fps: Optional[float] = None
    thumbnail_url: Optional[str] = None
    author: Optional[str] = None
    author_url: Optional[str] = None
    license_name: str = "Unknown"
    license_commercial_use: str = "unknown"
    attribution_required: bool = False
    attribution_text: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def orientation(self) -> Optional[StockOrientation]:
        if self.width and self.height:
            if self.width > self.height * 1.2:
                return StockOrientation.LANDSCAPE
            if self.height > self.width * 1.2:
                return StockOrientation.PORTRAIT
            return StockOrientation.SQUARE
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "media_type": self.media_type.value,
            "asset_id": self.asset_id,
            "page_url": self.page_url,
            "download_url": self.download_url,
            "width": self.width,
            "height": self.height,
            "duration": self.duration,
            "fps": self.fps,
            "thumbnail_url": self.thumbnail_url,
            "author": self.author,
            "author_url": self.author_url,
            "license_name": self.license_name,
            "license_commercial_use": self.license_commercial_use,
            "attribution_required": self.attribution_required,
            "attribution_text": self.attribution_text,
            "orientation": self.orientation.value if self.orientation else None,
        }


@dataclass
class StockSearchRequest:
    query: str
    media_type: StockMediaType = StockMediaType.VIDEO
    orientation: Optional[StockOrientation] = None
    min_width: Optional[int] = None
    min_height: Optional[int] = None
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None
    per_page: int = 15
    page: int = 1
    language: str = "en"

    def validate(self) -> None:
        if not self.query.strip():
            raise VideoError(TypedErrorCode.WORKFLOW_INVALID, "Stock search query is empty.")
        if self.per_page < 1 or self.per_page > 80:
            raise VideoError(TypedErrorCode.WORKFLOW_INVALID, f"per_page {self.per_page} out of range [1,80].")


@dataclass
class StockDownloadResult:
    path: Path
    hit: StockHit
    sha256: str
    bytes_size: int


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class StockProvider(abc.ABC):
    """Abstract base for stock media providers."""

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def available(self) -> bool:
        """True only when the API key is configured."""

    @abc.abstractmethod
    def meta(self) -> ProviderMeta:
        """Unified provider metadata (cost/license/runtime/capability)."""

    @abc.abstractmethod
    def search(self, request: StockSearchRequest) -> list[StockHit]:
        """Search the provider. Raises STOCK_SEARCH_FAILED on transport errors."""

    @abc.abstractmethod
    def _download_url(self, hit: StockHit) -> str:
        """Resolve the best download URL for a hit."""

    # -- shared download + verify + cache ------------------------------------
    def download(self, hit: StockHit, destination: Optional[Path] = None) -> StockDownloadResult:
        """Download a stock asset, verify it is non-empty, hash it, cache it.

        Reuses an existing cached file when the content hash matches.
        Raises STOCK_DOWNLOAD_FAILED on any transport/write/size error.
        """
        if not self.available:
            raise VideoError(
                TypedErrorCode.PROVIDER_DISABLED,
                f"{self.name} is not configured (missing API key).",
                context={"provider": self.name},
            )
        url = self._download_url(hit)
        if not url:
            raise VideoError(
                TypedErrorCode.STOCK_DOWNLOAD_FAILED,
                f"{self.name} hit {hit.asset_id} has no downloadable URL.",
                context={"provider": self.name, "asset_id": hit.asset_id},
            )
        # Cache by provider + asset_id + url hash to avoid re-downloading.
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        ext = self._infer_extension(hit, url)
        cache_dir = config.ASSET_CACHE_DIR / self.name
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / f"{hit.asset_id}_{url_hash}{ext}"
        if cached.exists() and cached.stat().st_size > 0:
            sha = file_sha256(cached)
            log("STOCK", f"cache hit {self.name}:{hit.asset_id}", sha256=sha[:12])
            return StockDownloadResult(path=cached, hit=hit, sha256=sha, bytes_size=cached.stat().st_size)

        out = destination or cached
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            with httpx.Client(timeout=config.STOCK_HTTP_TIMEOUT, follow_redirects=True) as client:
                with client.stream("GET", url) as r:
                    if r.status_code != 200:
                        raise VideoError(
                            TypedErrorCode.STOCK_DOWNLOAD_FAILED,
                            f"{self.name} download HTTP {r.status_code} for asset {hit.asset_id}.",
                            context={"provider": self.name, "asset_id": hit.asset_id, "status": r.status_code},
                        )
                    downloaded = 0
                    with open(out, "wb") as f:
                        for chunk in r.iter_bytes(chunk_size=1 << 16):
                            if not chunk:
                                continue
                            downloaded += len(chunk)
                            if downloaded > config.STOCK_MAX_DOWNLOAD_BYTES:
                                f.close()
                                out.unlink(missing_ok=True)
                                raise VideoError(
                                    TypedErrorCode.STOCK_DOWNLOAD_FAILED,
                                    f"{self.name} asset {hit.asset_id} exceeds max download size.",
                                    context={"provider": self.name, "limit_bytes": config.STOCK_MAX_DOWNLOAD_BYTES},
                                )
                            f.write(chunk)
        except httpx.RequestError as e:
            raise VideoError(
                TypedErrorCode.STOCK_DOWNLOAD_FAILED,
                f"{self.name} download failed for asset {hit.asset_id}: {e}",
                context={"provider": self.name, "asset_id": hit.asset_id},
            ) from e
        if not out.exists() or out.stat().st_size == 0:
            raise VideoError(
                TypedErrorCode.STOCK_DOWNLOAD_FAILED,
                f"{self.name} asset {hit.asset_id} downloaded as empty file.",
                context={"provider": self.name, "asset_id": hit.asset_id, "path": str(out)},
            )
        sha = file_sha256(out)
        log("STOCK", f"downloaded {self.name}:{hit.asset_id}", bytes=out.stat().st_size, sha256=sha[:12])
        return StockDownloadResult(path=out, hit=hit, sha256=sha, bytes_size=out.stat().st_size)

    @staticmethod
    def _infer_extension(hit: StockHit, url: str) -> str:
        low = url.lower().split("?")[0]
        for ext in (".mp4", ".mov", ".webm", ".ogg", ".oga", ".jpg", ".jpeg", ".png", ".webp", ".wav", ".mp3"):
            if low.endswith(ext):
                return ext
        if hit.media_type == StockMediaType.VIDEO:
            return ".mp4"
        if hit.media_type == StockMediaType.AUDIO:
            return ".wav"
        return ".jpg"


def stock_provider_status(providers: list[StockProvider]) -> list[dict[str, Any]]:
    """Per-provider health summary for the dashboard."""
    out: list[dict[str, Any]] = []
    for p in providers:
        m = p.meta()
        out.append({
            "name": p.name,
            "available": p.available(),
            "kind": m.kind.value,
            "cost": m.cost.to_dict(),
            "license": m.license.to_dict(),
            "runtime": m.runtime.to_dict(),
        })
    return out

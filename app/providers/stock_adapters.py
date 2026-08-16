"""Concrete stock provider adapters (Phase 13).

Pexels and Pixabay — both free tiers, both with official REST APIs, both with
licenses that permit commercial use (verified against official sources, see
PROVIDER_FACTS in AGENTS.md). API keys are read only from environment variables
and never logged.
"""

from __future__ import annotations

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
from .stock import (
    StockHit,
    StockMediaType,
    StockOrientation,
    StockProvider,
    StockSearchRequest,
)

_PEXELS_LICENSE = ProviderLicense(
    name="Pexels License",
    commercial_use="allowed",
    attribution_required=False,
    attribution_text="Photos and videos provided by Pexels",
    source_url="https://www.pexels.com/terms-of-service/",
)
_PIXABAY_LICENSE = ProviderLicense(
    name="Pixabay Content License",
    commercial_use="allowed",
    attribution_required=False,
    attribution_text="Content from Pixabay",
    source_url="https://pixabay.com/service/license-summary/",
)


class PexelsStockProvider(StockProvider):
    """Pexels stock provider (free, commercial-use-permitted).

    API: GET https://api.pexels.com/videos/search  (videos)
         GET https://api.pexels.com/v1/search      (photos)
    Auth: Authorization header with the API key.
    """

    BASE = "https://api.pexels.com"

    @property
    def name(self) -> str:
        return "pexels"

    @property
    def available(self) -> bool:
        return bool(config.PEXELS_API_KEY)

    def meta(self) -> ProviderMeta:
        return ProviderMeta(
            name=self.name,
            kind=ProviderKind.STOCK,
            version="1",
            description="Pexels free stock video/photo API",
            cost=ProviderCost(is_paid=False, unit="free", free_quota_per_month=20000,
                              note="Default rate limit; higher on request with attribution."),
            license=_PEXELS_LICENSE,
            runtime=ProviderRuntime(requires_api_key=True, requires_network=True, cpu_fallback=True),
            capability=ProviderCapability(
                capabilities=[Capability.STOCK_VIDEO, Capability.STOCK_IMAGE],
                supported_orientations=["landscape", "portrait", "square"],
            ),
        )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": config.PEXELS_API_KEY}

    def search(self, request: StockSearchRequest) -> list[StockHit]:
        request.validate()
        if not self.available:
            raise VideoError(
                TypedErrorCode.PROVIDER_DISABLED,
                "Pexels API key not configured (PEXELS_API_KEY).",
                context={"provider": self.name},
            )
        path = "/videos/search" if request.media_type == StockMediaType.VIDEO else "/v1/search"
        params: dict[str, Any] = {"query": request.query, "per_page": request.per_page, "page": request.page}
        if request.orientation:
            params["orientation"] = request.orientation.value
        if request.min_duration and request.media_type == StockMediaType.VIDEO:
            params["min_duration"] = int(request.min_duration)
        if request.max_duration and request.media_type == StockMediaType.VIDEO:
            params["max_duration"] = int(request.max_duration)
        try:
            r = httpx.get(f"{self.BASE}{path}", params=params, headers=self._headers(),
                          timeout=config.STOCK_HTTP_TIMEOUT)
        except httpx.RequestError as e:
            raise VideoError(
                TypedErrorCode.STOCK_SEARCH_FAILED,
                f"Pexels search failed: {e}",
                context={"provider": self.name, "query": request.query},
            ) from e
        if r.status_code == 429:
            raise VideoError(
                TypedErrorCode.QUOTA_EXCEEDED,
                "Pexels API rate limit exceeded (HTTP 429).",
                context={"provider": self.name},
            )
        if r.status_code != 200:
            raise VideoError(
                TypedErrorCode.STOCK_SEARCH_FAILED,
                f"Pexels search HTTP {r.status_code}: {r.text[:200]}",
                context={"provider": self.name, "status": r.status_code},
            )
        data = r.json()
        items = data.get("videos", []) if request.media_type == StockMediaType.VIDEO else data.get("photos", [])
        return [self._normalize(item, request.media_type) for item in items]

    def _normalize(self, item: dict[str, Any], media_type: StockMediaType) -> StockHit:
        if media_type == StockMediaType.VIDEO:
            files = item.get("video_files", [])
            # Prefer an HD mp4 file.
            best = next((f for f in files if f.get("file_type") == "video/mp4"
                         and f.get("quality") in ("hd", "sd")), None)
            if best is None and files:
                best = next((f for f in files if f.get("file_type") == "video/mp4"), files[0])
            dl = best.get("link") if best else None
            user = item.get("user", {}) or {}
            return StockHit(
                provider=self.name, media_type=media_type, asset_id=str(item.get("id")),
                page_url=item.get("url", ""), download_url=dl or "",
                width=best.get("width") if best else item.get("width"),
                height=best.get("height") if best else item.get("height"),
                duration=float(item.get("duration", 0)) if item.get("duration") else None,
                fps=best.get("fps") if best else None,
                thumbnail_url=item.get("image"), author=user.get("name"),
                author_url=user.get("url"), license_name=_PEXELS_LICENSE.name,
                license_commercial_use=_PEXELS_LICENSE.commercial_use,
                attribution_required=_PEXELS_LICENSE.attribution_required,
                attribution_text=_PEXELS_LICENSE.attribution_text, raw=item,
            )
        # photo
        user = item.get("user", {}) or {}
        return StockHit(
            provider=self.name, media_type=media_type, asset_id=str(item.get("id")),
            page_url=item.get("url", ""), download_url=item.get("src", {}).get("original", ""),
            width=item.get("width"), height=item.get("height"),
            thumbnail_url=item.get("src", {}).get("medium"), author=user.get("name"),
            author_url=user.get("url"), license_name=_PEXELS_LICENSE.name,
            license_commercial_use=_PEXELS_LICENSE.commercial_use,
            attribution_required=_PEXELS_LICENSE.attribution_required,
            attribution_text=_PEXELS_LICENSE.attribution_text, raw=item,
        )

    def _download_url(self, hit: StockHit) -> str:
        return hit.download_url


class PixabayStockProvider(StockProvider):
    """Pixabay stock provider (free, commercial-use-permitted).

    API: GET https://pixabay.com/api/videos/  (videos)
         GET https://pixabay.com/api/        (images)
    Auth: ``key`` query parameter.
    """

    BASE = "https://pixabay.com/api"

    @property
    def name(self) -> str:
        return "pixabay"

    @property
    def available(self) -> bool:
        return bool(config.PIXABAY_API_KEY)

    def meta(self) -> ProviderMeta:
        return ProviderMeta(
            name=self.name,
            kind=ProviderKind.STOCK,
            version="1",
            description="Pixabay free stock video/image API",
            cost=ProviderCost(is_paid=False, unit="free", free_quota_per_month=100,
                              note="~100 requests/60s; results capped at 500."),
            license=_PIXABAY_LICENSE,
            runtime=ProviderRuntime(requires_api_key=True, requires_network=True, cpu_fallback=True),
            capability=ProviderCapability(
                capabilities=[Capability.STOCK_VIDEO, Capability.STOCK_IMAGE],
                supported_orientations=["horizontal", "vertical", "all"],
            ),
        )

    def search(self, request: StockSearchRequest) -> list[StockHit]:
        request.validate()
        if not self.available:
            raise VideoError(
                TypedErrorCode.PROVIDER_DISABLED,
                "Pixabay API key not configured (PIXABAY_API_KEY).",
                context={"provider": self.name},
            )
        path = "/videos/" if request.media_type == StockMediaType.VIDEO else "/"
        params: dict[str, Any] = {
            "key": config.PIXABAY_API_KEY, "q": request.query,
            "per_page": min(request.per_page, 200), "page": request.page,
            "safesearch": "true",
        }
        if request.orientation:
            # Pixabay uses horizontal/vertical for video, all/horizontal/vertical for images.
            orient = "horizontal" if request.orientation == StockOrientation.LANDSCAPE else \
                     "vertical" if request.orientation == StockOrientation.PORTRAIT else "all"
            params["orientation"] = orient
        if request.media_type == StockMediaType.VIDEO:
            params["video_type"] = "film"
        try:
            r = httpx.get(f"{self.BASE}{path}", params=params, timeout=config.STOCK_HTTP_TIMEOUT)
        except httpx.RequestError as e:
            raise VideoError(
                TypedErrorCode.STOCK_SEARCH_FAILED,
                f"Pixabay search failed: {e}",
                context={"provider": self.name, "query": request.query},
            ) from e
        if r.status_code == 429:
            raise VideoError(
                TypedErrorCode.QUOTA_EXCEEDED,
                "Pixabay API rate limit exceeded (HTTP 429).",
                context={"provider": self.name},
            )
        if r.status_code != 200:
            raise VideoError(
                TypedErrorCode.STOCK_SEARCH_FAILED,
                f"Pixabay search HTTP {r.status_code}: {r.text[:200]}",
                context={"provider": self.name, "status": r.status_code},
            )
        data = r.json()
        hits = data.get("hits", [])
        return [self._normalize(h, request.media_type) for h in hits]

    def _normalize(self, item: dict[str, Any], media_type: StockMediaType) -> StockHit:
        if media_type == StockMediaType.VIDEO:
            vids = item.get("videos", {})
            # Prefer large, then medium, then small, then tiny.
            chosen = vids.get("large") or vids.get("medium") or vids.get("small") or vids.get("tiny") or {}
            return StockHit(
                provider=self.name, media_type=media_type, asset_id=str(item.get("id")),
                page_url=item.get("pageURL", ""), download_url=chosen.get("url", ""),
                width=chosen.get("width"), height=chosen.get("height"),
                duration=float(item.get("duration", 0)) if item.get("duration") else None,
                thumbnail_url=item.get("videos", {}).get("tiny", {}).get("thumbnail"),
                author=item.get("user"), author_url=None,
                license_name=_PIXABAY_LICENSE.name,
                license_commercial_use=_PIXABAY_LICENSE.commercial_use,
                attribution_required=_PIXABAY_LICENSE.attribution_required,
                attribution_text=_PIXABAY_LICENSE.attribution_text, raw=item,
            )
        return StockHit(
            provider=self.name, media_type=media_type, asset_id=str(item.get("id")),
            page_url=item.get("pageURL", ""),
            download_url=item.get("largeImageURL") or item.get("webformatURL", ""),
            width=item.get("imageWidth"), height=item.get("imageHeight"),
            thumbnail_url=item.get("previewURL"), author=item.get("user"), author_url=None,
            license_name=_PIXABAY_LICENSE.name,
            license_commercial_use=_PIXABAY_LICENSE.commercial_use,
            attribution_required=_PIXABAY_LICENSE.attribution_required,
            attribution_text=_PIXABAY_LICENSE.attribution_text, raw=item,
        )

    def _download_url(self, hit: StockHit) -> str:
        return hit.download_url


def build_stock_providers() -> list[StockProvider]:
    """Instantiate all configured stock providers (preference order)."""
    return [PexelsStockProvider(), PixabayStockProvider()]


async def select_stock_provider(providers: Optional[list[StockProvider]] = None) -> StockProvider:
    """Return the first available stock provider; NO_PROVIDER if none."""
    providers = providers if providers is not None else build_stock_providers()
    for p in providers:
        if p.available:
            log("STOCK", f"selected {p.name}")
            return p
    from ..core.errors import VideoError, TypedErrorCode
    raise VideoError(
        TypedErrorCode.NO_PROVIDER,
        "No stock provider available. Configure PEXELS_API_KEY or PIXABAY_API_KEY.",
        context={"configured": [p.name for p in providers]},
    )

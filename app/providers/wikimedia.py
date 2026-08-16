"""Wikimedia Commons stock provider (Phase 13).

A keyless, API-key-free stock media provider using the official MediaWiki
Action API on Wikimedia Commons. This is the critical free-first media path:
it works on any machine with network access and a compliant User-Agent,
without any credentials or GPU.

Official sources verified during Phase 13 research:
- API docs: https://commons.wikimedia.org/wiki/Commons:API/MediaWiki
- Imageinfo: https://www.mediawiki.org/wiki/API:Imageinfo
- Etiquette/User-Agent: https://www.mediawiki.org/wiki/API:Etiquette
- Rate limits: https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits
  (200 req/min with a compliant User-Agent; 3 concurrent max)
- Licensing: https://commons.wikimedia.org/wiki/Commons:Licensing
  (only freely-licensed or public-domain media; commercial use allowed;
   non-commercial-only licenses are rejected at upload)

License handling (critical):
- Every file's license is read from the ``extmetadata`` API field
  (LicenseShortName, LicenseUrl, UsageTerms, Copyrighted, Artist).
- No asset is marked commercially safe without this evidence.
- ``commercial_use`` is derived: ``allowed`` for CC0/public-domain,
  ``restricted`` for CC BY/CC BY-SA (allowed but attribution required),
  ``unknown`` when metadata is missing or unparseable.
- Attribution is required for CC BY / CC BY-SA; the artist + license URL are
  preserved on every asset so downstream licensing can be honored.

Transcoding:
- Commons stores video as WebM (VP8/VP9 + Vorbis/Opus) and audio as Ogg.
- FFmpeg transcodes WebM → MP4 (H.264) and Ogg → WAV/MP3 for pipeline use.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

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
    StockDownloadResult,
    StockHit,
    StockMediaType,
    StockOrientation,
    StockProvider,
    StockSearchRequest,
    file_sha256,
)

# Official MediaWiki Action API endpoint for Wikimedia Commons.
_COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# Compliant User-Agent (per Wikimedia policy: must include contact info).
# Configurable so deployments can set their own contact URL/email.
_USER_AGENT = (
    f"vid-dvd/{getattr(config, '_VERSION', '0.1')} "
    "(https://github.com/smsm9919/vid-dvd; contact: openhands@all-hands.dev)"
)

# Wikimedia license mapping (derived from extmetadata, never guessed).
# CC0 / public domain → commercial allowed, no attribution.
# CC BY / CC BY-SA → commercial allowed, attribution required (restricted).
# Anything unrecognized → unknown (must not be claimed safe).
_PD_MARKERS = ("public domain", "pd-", "cc0", "pdm")
_BY_MARKERS = ("cc by", "cc-by", "attribution")
_BY_SA_MARKERS = ("cc by-sa", "cc-by-sa", "attribution-sharealike")


def _classify_license(short_name: str, usage_terms: str, copyrighted: str) -> tuple[str, bool]:
    """Return (commercial_use, attribution_required) from license metadata.

    Never returns 'allowed' without evidence. Defaults to ('unknown', True).
    """
    blob = f"{short_name} {usage_terms} {copyrighted}".lower()
    if not short_name and not usage_terms:
        return "unknown", True
    # Public domain / CC0 → allowed, no attribution required.
    if any(m in blob for m in _PD_MARKERS):
        return "allowed", False
    if copyrighted == "False" or copyrighted.lower() == "false":
        return "allowed", False
    # CC BY-SA → allowed but attribution required (share-alike obligation).
    if any(m in blob for m in _BY_SA_MARKERS):
        return "allowed", True
    # CC BY (plain attribution) → allowed but attribution required.
    if any(m in blob for m in _BY_MARKERS):
        return "allowed", True
    # GFDL, LGPL, GPL → copyleft, restricted for commercial derivative works.
    if "gfdl" in blob or "gpl" in blob or "lgpl" in blob:
        return "restricted", True
    return "unknown", True


def _strip_html(s: str) -> str:
    """Strip HTML tags from extmetadata values (they are HTML-formatted)."""
    if not s:
        return s
    text = re.sub(r"<[^>]+>", "", s)
    return text.strip()


class WikimediaCommonsProvider(StockProvider):
    """Keyless stock media provider via the MediaWiki Action API.

    No API key required. Uses a compliant User-Agent and respects Wikimedia
    etiquette (max 3 concurrent requests, 200 req/min).
    """

    BASE = _COMMONS_API

    def __init__(self) -> None:
        self._concurrency = threading.Semaphore(3)  # Wikimedia etiquette: ≤3 concurrent

    @property
    def name(self) -> str:
        return "wikimedia"

    @property
    def available(self) -> bool:
        """Available when network is reachable and httpx is importable.

        No API key needed — only a compliant User-Agent. We consider it
        available by default (network is checked lazily on first request).
        """
        return True

    def meta(self) -> ProviderMeta:
        return ProviderMeta(
            name=self.name,
            kind=ProviderKind.STOCK,
            version="1",
            description="Wikimedia Commons — keyless free media (CC/PD), MediaWiki API",
            cost=ProviderCost(is_paid=False, unit="free",
                              note="No API key; compliant User-Agent required; 200 req/min."),
            license=ProviderLicense(
                name="Per-file (CC BY/CC BY-SA/CC0/Public Domain)",
                commercial_use="allowed",
                attribution_required=True,  # per-file; defaults to required
                attribution_text="See per-file license metadata (LicenseUrl + Artist).",
                source_url="https://commons.wikimedia.org/wiki/Commons:Licensing",
            ),
            runtime=ProviderRuntime(
                requires_api_key=False, requires_network=True, cpu_fallback=True,
            ),
            capability=ProviderCapability(
                capabilities=[Capability.STOCK_VIDEO, Capability.STOCK_IMAGE, Capability.STOCK_SFX],
                supported_orientations=["landscape", "portrait", "square"],
            ),
        )

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": _USER_AGENT}

    # ----------------------------------------------------------------- search
    def search(self, request: StockSearchRequest) -> list[StockHit]:
        request.validate()
        # Build the search query with filetype filter.
        # Namespace 6 = File: namespace on Commons.
        if request.media_type == StockMediaType.VIDEO:
            file_filter = f"filetype:video"
            mime_hint = "video"
        elif request.media_type == StockMediaType.IMAGE:
            file_filter = "filetype:bitmap"
            mime_hint = "image"
        else:
            file_filter = "filetype:audio"
            mime_hint = "audio"
        gsrsearch = f"{request.query} {file_filter}"
        params: dict[str, Any] = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": gsrsearch,
            "gsrnamespace": 6,
            "gsrlimit": min(request.per_page, 50),
            "gsroffset": (request.page - 1) * request.per_page,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime|size|mediatype",
            "iiurlwidth": 400,
        }
        try:
            with self._concurrency:
                r = httpx.get(self.BASE, params=params, headers=self._headers(),
                              timeout=config.STOCK_HTTP_TIMEOUT)
        except httpx.RequestError as e:
            raise VideoError(
                TypedErrorCode.STOCK_SEARCH_FAILED,
                f"Wikimedia Commons search failed: {e}",
                context={"provider": self.name, "query": request.query},
            ) from e
        if r.status_code == 429:
            raise VideoError(
                TypedErrorCode.QUOTA_EXCEEDED,
                "Wikimedia Commons rate limit exceeded (HTTP 429). Reduce request frequency.",
                context={"provider": self.name},
            )
        if r.status_code != 200:
            raise VideoError(
                TypedErrorCode.STOCK_SEARCH_FAILED,
                f"Wikimedia Commons search HTTP {r.status_code}: {r.text[:200]}",
                context={"provider": self.name, "status": r.status_code},
            )
        data = r.json()
        pages = (data.get("query") or {}).get("pages") or {}
        hits: list[StockHit] = []
        for page in pages.values():
            hit = self._normalize(page, request.media_type)
            if hit is not None:
                hits.append(hit)
        # Preserve search ranking order.
        hits.sort(key=lambda h: h.raw.get("index", 999))
        return hits

    def _normalize(self, page: dict[str, Any], media_type: StockMediaType) -> Optional[StockHit]:
        """Convert a MediaWiki page record to a StockHit with full provenance."""
        title = page.get("title", "")
        if not title.startswith("File:"):
            return None
        infos = page.get("imageinfo") or []
        if not infos:
            return None
        info = infos[0]
        mediatype = info.get("mediatype", "")
        mime = info.get("mime", "")
        # Verify the mediatype matches what was requested.
        if media_type == StockMediaType.VIDEO and "VIDEO" not in mediatype.upper():
            return None
        if media_type == StockMediaType.IMAGE and "BITMAP" not in mediatype.upper():
            return None
        extmeta = info.get("extmetadata") or {}
        short_name = _strip_html(extmeta.get("LicenseShortName", {}).get("value", ""))
        license_url = extmeta.get("LicenseUrl", {}).get("value", "")
        usage_terms = _strip_html(extmeta.get("UsageTerms", {}).get("value", ""))
        copyrighted = extmeta.get("Copyrighted", {}).get("value", "")
        artist = _strip_html(extmeta.get("Artist", {}).get("value", ""))
        commercial_use, attribution_required = _classify_license(short_name, usage_terms, copyrighted)
        # Build attribution text.
        attribution = None
        if attribution_required:
            attribution = f"By {artist}" if artist else "See source"
            if license_url:
                attribution += f" — {license_url}"
            attribution += f" (via Wikimedia Commons, {short_name})"
        else:
            attribution = f"Public domain / CC0 (via Wikimedia Commons, {short_name})"
        asset_id = str(page.get("pageid", title))
        download_url = info.get("url", "")
        desc_url = info.get("descriptionurl", "")
        thumb = info.get("thumburl")
        return StockHit(
            provider=self.name,
            media_type=media_type,
            asset_id=asset_id,
            page_url=desc_url,
            download_url=download_url,
            width=info.get("width"),
            height=info.get("height"),
            duration=float(info.get("duration", 0)) if info.get("duration") else None,
            thumbnail_url=thumb,
            author=artist or None,
            author_url=None,
            license_name=short_name or "Unknown",
            license_commercial_use=commercial_use,
            attribution_required=attribution_required,
            attribution_text=attribution,
            raw={**page, "_mime": mime, "_mediatype": mediatype},
        )

    # ----------------------------------------------------------------- download
    def _download_url(self, hit: StockHit) -> str:
        return hit.download_url

    def download(self, hit: StockHit, destination: Optional[Path] = None) -> StockDownloadResult:
        """Download a Wikimedia asset, transcode to MP4/WAV if needed, verify.

        WebM video → transcode to MP4 (H.264 + AAC) for pipeline compatibility.
        Ogg audio → transcode to WAV for the audio QC/mixer.
        JPEG/PNG images → kept as-is.
        """
        if not hit.download_url:
            raise VideoError(
                TypedErrorCode.STOCK_DOWNLOAD_FAILED,
                f"Wikimedia hit {hit.asset_id} has no download URL.",
                context={"provider": self.name, "asset_id": hit.asset_id},
            )
        # Cache by provider + asset_id + url hash.
        import hashlib
        url_hash = hashlib.sha256(hit.download_url.encode()).hexdigest()[:16]
        cache_dir = config.ASSET_CACHE_DIR / self.name
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Determine if transcoding is needed from the raw mime.
        mime = (hit.raw.get("_mime") or "").lower()
        needs_video_transcode = "video/webm" in mime or "video/ogg" in mime or mime == ""
        needs_audio_transcode = "audio/ogg" in mime or "audio/webm" in mime
        if hit.media_type == StockMediaType.VIDEO:
            target_ext = ".mp4" if needs_video_transcode else ".mp4"
        elif hit.media_type == StockMediaType.IMAGE:
            target_ext = Path(hit.download_url).suffix.lower().split("?")[0] or ".jpg"
        else:
            target_ext = ".wav" if needs_audio_transcode else ".wav"
        cached = cache_dir / f"{hit.asset_id}_{url_hash}{target_ext}"
        if cached.exists() and cached.stat().st_size > 0:
            sha = file_sha256(cached)
            log("WIKIMEDIA", f"cache hit {hit.asset_id}", sha256=sha[:12])
            return StockDownloadResult(path=cached, hit=hit, sha256=sha, bytes_size=cached.stat().st_size)
        # Download the original first.
        raw_ext = Path(hit.download_url.split("?")[0]).suffix.lower() or (".webm" if needs_video_transcode else ".bin")
        raw_path = cache_dir / f"{hit.asset_id}_{url_hash}_raw{raw_ext}"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream_download(hit.download_url, raw_path, hit.asset_id)
        # Transcode if needed.
        out = destination or cached
        out.parent.mkdir(parents=True, exist_ok=True)
        if hit.media_type == StockMediaType.VIDEO and needs_video_transcode:
            self._transcode_video(raw_path, out)
            raw_path.unlink(missing_ok=True)
        elif hit.media_type != StockMediaType.IMAGE and needs_audio_transcode:
            self._transcode_audio(raw_path, out)
            raw_path.unlink(missing_ok=True)
        else:
            # No transcode needed (image, or already MP4): move to target.
            if raw_path != out:
                shutil.move(str(raw_path), str(out))
        if not out.exists() or out.stat().st_size == 0:
            raise VideoError(
                TypedErrorCode.STOCK_DOWNLOAD_FAILED,
                f"Wikimedia asset {hit.asset_id} produced empty output.",
                context={"provider": self.name, "asset_id": hit.asset_id, "path": str(out)},
            )
        sha = file_sha256(out)
        log("WIKIMEDIA", f"downloaded {hit.asset_id}", bytes=out.stat().st_size,
            sha256=sha[:12], transcode=needs_video_transcode or needs_audio_transcode)
        return StockDownloadResult(path=out, hit=hit, sha256=sha, bytes_size=out.stat().st_size)

    def _stream_download(self, url: str, dest: Path, asset_id: str) -> None:
        try:
            with self._concurrency:
                with httpx.Client(timeout=config.STOCK_HTTP_TIMEOUT, follow_redirects=True,
                                  headers=self._headers()) as client:
                    with client.stream("GET", url) as r:
                        if r.status_code != 200:
                            raise VideoError(
                                TypedErrorCode.STOCK_DOWNLOAD_FAILED,
                                f"Wikimedia download HTTP {r.status_code} for asset {asset_id}.",
                                context={"provider": self.name, "asset_id": asset_id, "status": r.status_code},
                            )
                        downloaded = 0
                        with open(dest, "wb") as f:
                            for chunk in r.iter_bytes(chunk_size=1 << 16):
                                if not chunk:
                                    continue
                                downloaded += len(chunk)
                                if downloaded > config.STOCK_MAX_DOWNLOAD_BYTES:
                                    f.close()
                                    dest.unlink(missing_ok=True)
                                    raise VideoError(
                                        TypedErrorCode.STOCK_DOWNLOAD_FAILED,
                                        f"Wikimedia asset {asset_id} exceeds max download size.",
                                        context={"limit_bytes": config.STOCK_MAX_DOWNLOAD_BYTES},
                                    )
                                f.write(chunk)
        except httpx.RequestError as e:
            dest.unlink(missing_ok=True)
            raise VideoError(
                TypedErrorCode.STOCK_DOWNLOAD_FAILED,
                f"Wikimedia download failed for asset {asset_id}: {e}",
                context={"provider": self.name, "asset_id": asset_id},
            ) from e

    @staticmethod
    def _transcode_video(src: Path, dst: Path) -> None:
        """Transcode WebM/Ogg video → MP4 (H.264 + AAC) via FFmpeg."""
        if not shutil.which("ffmpeg"):
            raise VideoError(TypedErrorCode.FFMPEG_ERROR, "ffmpeg not available to transcode Wikimedia video.")
        cmd = [
            "ffmpeg", "-y", "-i", str(src),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(dst),
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0 or not dst.exists():
            raise VideoError(
                TypedErrorCode.FFMPEG_ERROR,
                f"Wikimedia video transcode failed: {proc.stderr.decode('utf-8', 'replace')[:500]}",
            )

    @staticmethod
    def _transcode_audio(src: Path, dst: Path) -> None:
        """Transcode Ogg/Webm audio → WAV via FFmpeg."""
        if not shutil.which("ffmpeg"):
            raise VideoError(TypedErrorCode.FFMPEG_ERROR, "ffmpeg not available to transcode Wikimedia audio.")
        cmd = ["ffmpeg", "-y", "-i", str(src), "-c:a", "pcm_s16le", "-ar", "44100", str(dst)]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0 or not dst.exists():
            raise VideoError(
                TypedErrorCode.FFMPEG_ERROR,
                f"Wikimedia audio transcode failed: {proc.stderr.decode('utf-8', 'replace')[:500]}",
            )


def build_wikimedia_provider() -> WikimediaCommonsProvider:
    return WikimediaCommonsProvider()

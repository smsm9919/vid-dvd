"""Tests for the Wikimedia Commons provider (Phase 13).

Two tiers:
1. Unit tests — license classification, normalization, metadata. Always run.
2. Live runtime tests — real search/download/transcode against Wikimedia.
   Skipped when network is unavailable. These download REAL media files and
   verify them with FFmpeg/ffprobe (no mocks).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import httpx
import pytest

from app import config
from app.providers.wikimedia import (
    WikimediaCommonsProvider,
    _classify_license,
    _strip_html,
    build_wikimedia_provider,
)
from app.providers.stock import StockHit, StockMediaType, StockSearchRequest
from app.providers.stock_adapters import build_stock_providers
from app.providers.router import build_default_router
from app.core.errors import TypedErrorCode, VideoError


def _ffmpeg_ok() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _network_ok() -> bool:
    try:
        r = httpx.get("https://commons.wikimedia.org/w/api.php",
                      params={"action": "query", "format": "json", "meta": "siteinfo"},
                      headers={"User-Agent": "vid-dvd-test/0.1 (test)"},
                      timeout=10)
        return r.status_code == 200 and "query" in r.text
    except Exception:
        return False


LIVE = _network_ok() and _ffmpeg_ok()


# ---------------------------------------------------------- license classification
def test_classify_public_domain():
    cu, attr = _classify_license("Public domain", "Public domain", "False")
    assert cu == "allowed"
    assert attr is False


def test_classify_cc0():
    cu, attr = _classify_license("CC0", "CC0 1.0 Universal Public Domain Dedication", "False")
    assert cu == "allowed"
    assert attr is False


def test_classify_cc_by():
    cu, attr = _classify_license("CC BY 4.0", "Creative Commons Attribution 4.0", "True")
    assert cu == "allowed"
    assert attr is True


def test_classify_cc_by_sa():
    cu, attr = _classify_license("CC BY-SA 4.0", "Creative Commons Attribution-Share Alike 4.0", "True")
    assert cu == "allowed"
    assert attr is True


def test_classify_gpl_restricted():
    cu, attr = _classify_license("GNU GPL", "GNU General Public License", "True")
    assert cu == "restricted"
    assert attr is True


def test_classify_unknown_defaults_to_unknown():
    """Never claim commercial safety without evidence."""
    cu, attr = _classify_license("", "", "")
    assert cu == "unknown"
    assert attr is True


def test_classify_unrecognized_license_unknown():
    cu, attr = _classify_license("Some Custom License", "Custom terms", "True")
    assert cu == "unknown"


def test_strip_html():
    assert _strip_html('<a href="x">John Doe</a>') == "John Doe"
    assert _strip_html("") == ""
    assert _strip_html("plain text") == "plain text"


# ---------------------------------------------------------- provider metadata
def test_provider_name():
    assert WikimediaCommonsProvider().name == "wikimedia"


def test_provider_available_without_api_key():
    """Wikimedia needs NO API key — it is the keyless path."""
    assert WikimediaCommonsProvider().available is True


def test_provider_meta_no_api_key_required():
    m = WikimediaCommonsProvider().meta()
    assert m.runtime.requires_api_key is False
    assert m.cost.is_paid is False
    assert m.runtime.requires_network is True


def test_provider_meta_capabilities():
    m = WikimediaCommonsProvider().meta()
    from app.providers.contracts import Capability
    assert Capability.STOCK_VIDEO in m.capability.capabilities
    assert Capability.STOCK_IMAGE in m.capability.capabilities
    assert Capability.STOCK_SFX in m.capability.capabilities


def test_provider_meta_license_is_per_file():
    """License is per-file (variable), never blanket-claimed safe."""
    m = WikimediaCommonsProvider().meta()
    assert "Per-file" in m.license.name or "per-file" in m.license.name.lower()


def test_provider_user_agent_includes_contact():
    """Wikimedia etiquette requires a descriptive User-Agent with contact."""
    from app.providers.wikimedia import _USER_AGENT
    assert "vid-dvd" in _USER_AGENT
    assert "github.com" in _USER_AGENT or "@" in _USER_AGENT


def test_build_stock_providers_includes_wikimedia_first():
    """Wikimedia is the keyless first-choice stock provider."""
    providers = build_stock_providers()
    names = [p.name for p in providers]
    assert "wikimedia" in names
    assert names[0] == "wikimedia"


def test_router_includes_wikimedia():
    r = build_default_router()
    names = [mp.name for mp, _ in r.providers]
    assert "wikimedia" in names


# ---------------------------------------------------------- search validation
def test_search_empty_query_rejected():
    p = WikimediaCommonsProvider()
    with pytest.raises(VideoError) as ei:
        p.search(StockSearchRequest(query="", media_type=StockMediaType.VIDEO))
    assert ei.value.code == TypedErrorCode.WORKFLOW_INVALID


# ---------------------------------------------------------- LIVE RUNTIME TESTS
@pytest.mark.skipif(not LIVE, reason="Network or FFmpeg unavailable")
def test_live_search_returns_real_video_hits():
    """REAL search against Wikimedia Commons — returns real video results."""
    p = WikimediaCommonsProvider()
    hits = p.search(StockSearchRequest(query="sunset beach", media_type=StockMediaType.VIDEO, per_page=5))
    assert len(hits) > 0
    hit = hits[0]
    assert hit.provider == "wikimedia"
    assert hit.media_type == StockMediaType.VIDEO
    assert hit.download_url.startswith("http")
    assert hit.page_url.startswith("http")
    # Every hit MUST carry license metadata (never blank/unknown silently).
    assert hit.license_name  # must be present (even if "Unknown")
    assert hit.license_commercial_use in ("allowed", "restricted", "unknown")


@pytest.mark.skipif(not LIVE, reason="Network or FFmpeg unavailable")
def test_live_search_license_metadata_extracted():
    """REAL license metadata is extracted from extmetadata."""
    p = WikimediaCommonsProvider()
    hits = p.search(StockSearchRequest(query="cat", media_type=StockMediaType.IMAGE, per_page=5))
    assert len(hits) > 0
    # At least one hit should have a recognizable license (CC/PD).
    recognized = [h for h in hits if h.license_name and h.license_name != "Unknown"]
    assert len(recognized) > 0, "No hits with recognizable license metadata"
    hit = recognized[0]
    assert hit.license_name
    assert hit.license_commercial_use in ("allowed", "restricted")


@pytest.mark.skipif(not LIVE, reason="Network or FFmpeg unavailable")
def test_live_download_and_transcode_webm_to_mp4(tmp_path, monkeypatch):
    """REAL download + WebM→MP4 transcode + FFmpeg QC verification."""
    monkeypatch.setattr(config, "ASSET_CACHE_DIR", tmp_path / "cache")
    p = WikimediaCommonsProvider()
    hits = p.search(StockSearchRequest(query="sunset", media_type=StockMediaType.VIDEO, per_page=10))
    assert len(hits) > 0
    # Pick a hit that is a video with a download URL.
    hit = next((h for h in hits if h.download_url), None)
    assert hit is not None, "No downloadable video hit"
    result = p.download(hit)
    assert result.path.exists()
    assert result.path.stat().st_size > 1000
    assert len(result.sha256) == 64
    # Verify the transcoded MP4 is real with ffprobe.
    from app.media import verify_mp4
    report = verify_mp4(result.path)
    assert report["ok"] is True
    assert report["width"] > 0
    assert report["height"] > 0
    assert report["duration"] > 0
    print(f"REAL Wikimedia video: {report['width']}x{report['height']}, "
          f"{report['duration']:.1f}s, codec={report['video_codec']}")


@pytest.mark.skipif(not LIVE, reason="Network or FFmpeg unavailable")
def test_live_download_caches_on_second_call(tmp_path, monkeypatch):
    """Downloaded assets are cached and de-duplicated (no re-download)."""
    monkeypatch.setattr(config, "ASSET_CACHE_DIR", tmp_path / "cache")
    p = WikimediaCommonsProvider()
    hits = p.search(StockSearchRequest(query="ocean", media_type=StockMediaType.VIDEO, per_page=5))
    assert len(hits) > 0
    hit = next((h for h in hits if h.download_url), None)
    assert hit is not None
    r1 = p.download(hit)
    first_mtime = r1.path.stat().st_mtime
    r2 = p.download(hit)
    # Same path (cache hit), not re-downloaded.
    assert r1.path == r2.path
    assert r2.path.stat().st_mtime == first_mtime
    assert r1.sha256 == r2.sha256


@pytest.mark.skipif(not LIVE, reason="Network or FFmpeg unavailable")
def test_live_download_image(tmp_path, monkeypatch):
    """REAL image download (no transcode needed)."""
    monkeypatch.setattr(config, "ASSET_CACHE_DIR", tmp_path / "cache")
    p = WikimediaCommonsProvider()
    hits = p.search(StockSearchRequest(query="mountain landscape", media_type=StockMediaType.IMAGE, per_page=5))
    assert len(hits) > 0
    hit = next((h for h in hits if h.download_url), None)
    assert hit is not None
    result = p.download(hit)
    assert result.path.exists()
    assert result.path.stat().st_size > 1000


@pytest.mark.skipif(not LIVE, reason="Network or FFmpeg unavailable")
def test_live_attribution_preserved_for_cc_licensed_asset(tmp_path, monkeypatch):
    """CC BY/CC BY-SA assets carry attribution text with artist + license URL."""
    monkeypatch.setattr(config, "ASSET_CACHE_DIR", tmp_path / "cache")
    p = WikimediaCommonsProvider()
    hits = p.search(StockSearchRequest(query="city", media_type=StockMediaType.IMAGE, per_page=10))
    # Only hits where the license actually requires attribution (CC BY / CC BY-SA).
    attr_hits = [h for h in hits if h.attribution_required and h.license_name != "Unknown"]
    if not attr_hits:
        pytest.skip("No attribution-required CC-licensed hits in this search")
    hit = attr_hits[0]
    assert hit.attribution_required is True
    assert hit.attribution_text
    assert "Wikimedia" in hit.attribution_text

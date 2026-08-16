"""Unit tests for stock providers (Phase 13).

These use httpx MockTransport to exercise the real search-normalization,
download, caching, hashing, and license-metadata logic against fixture
responses that mirror the official Pexels/Pixabay API shapes.

NOTE: These are fake-server tests. They verify adapter correctness, NOT real
network generation. Real runtime readiness requires PEXELS_API_KEY /
PIXABAY_API_KEY and is reported as STOCK_RUNTIME=BLOCKED without them.
"""

import httpx
import pytest

from app import config
from app.providers import stock_adapters
from app.providers.stock import (
    StockHit,
    StockMediaType,
    StockOrientation,
    StockSearchRequest,
    file_sha256,
)
from app.providers.stock_adapters import (
    PexelsStockProvider,
    PixabayStockProvider,
)
from app.core.errors import TypedErrorCode, VideoError


# --------------------------------------------------------------------- helpers
def _patch_key(monkeypatch, attr, value):
    monkeypatch.setattr(config, attr, value)


def _mount(monkeypatch, fn, responses):
    """Route httpx requests through a MockTransport dispatcher.

    Patches both httpx.get (search) and httpx.Client (streaming download) so
    each call gets a fresh client backed by the same transport.
    """
    transport = httpx.MockTransport(fn)
    real_client = httpx.Client  # capture before patching

    def make_client(*a, **k):
        k.pop("transport", None)
        allowed = {kk: vv for kk, vv in k.items() if kk in ("timeout", "follow_redirects", "base_url")}
        return real_client(transport=transport, **allowed)

    monkeypatch.setattr(httpx, "Client", make_client)
    monkeypatch.setattr(httpx, "get", lambda url, params=None, headers=None, timeout=None: make_client().get(url, params=params, headers=headers))


# --------------------------------------------------------------------- availability
def test_pexels_unavailable_without_key(monkeypatch):
    _patch_key(monkeypatch, "PEXELS_API_KEY", "")
    assert PexelsStockProvider().available is False


def test_pixabay_unavailable_without_key(monkeypatch):
    _patch_key(monkeypatch, "PIXABAY_API_KEY", "")
    assert PixabayStockProvider().available is False


def test_pexels_available_with_key(monkeypatch):
    _patch_key(monkeypatch, "PEXELS_API_KEY", "fake-key")
    assert PexelsStockProvider().available is True


# --------------------------------------------------------------------- license metadata
def test_pexels_license_commercial_allowed():
    lic = PexelsStockProvider().meta().license
    assert lic.commercial_use == "allowed"
    assert lic.name == "Pexels License"


def test_pixabay_license_commercial_allowed():
    lic = PixabayStockProvider().meta().license
    assert lic.commercial_use == "allowed"
    assert lic.name == "Pixabay Content License"


def test_both_stock_providers_are_free():
    for p in (PexelsStockProvider(), PixabayStockProvider()):
        assert p.meta().cost.is_paid is False


# --------------------------------------------------------------------- search request validation
def test_search_request_empty_query_rejected():
    with pytest.raises(VideoError) as ei:
        StockSearchRequest(query="").validate()
    assert ei.value.code == TypedErrorCode.WORKFLOW_INVALID


def test_search_request_per_page_bounds():
    with pytest.raises(VideoError):
        StockSearchRequest(query="x", per_page=0).validate()
    with pytest.raises(VideoError):
        StockSearchRequest(query="x", per_page=81).validate()


# --------------------------------------------------------------------- search normalization (fake API)
_PEXELS_VIDEO_RESP = {
    "videos": [{
        "id": 123, "url": "https://www.pexels.com/video/123/",
        "image": "https://img/pexels-123.jpg", "duration": 8, "width": 1920, "height": 1080,
        "user": {"id": 1, "name": "Jane", "url": "https://www.pexels.com/@jane"},
        "video_files": [
            {"id": 1, "file_type": "video/mp4", "quality": "hd", "width": 1280, "height": 720, "fps": 24, "link": "https://cdn/pexels-123-hd.mp4"},
            {"id": 2, "file_type": "video/mp4", "quality": "sd", "width": 640, "height": 360, "fps": 24, "link": "https://cdn/pexels-123-sd.mp4"},
        ],
    }]
}

_PIXABAY_VIDEO_RESP = {
    "total": 1, "totalHits": 1,
    "hits": [{
        "id": 456, "pageURL": "https://pixabay.com/videos/456/", "duration": 12,
        "user": "Coverr", "videos": {
            "large": {"url": "https://cdn/pixabay-456-large.mp4", "width": 1920, "height": 1080, "size": 6000000, "thumbnail": "https://cdn/t456.jpg"},
            "medium": {"url": "https://cdn/pixabay-456-med.mp4", "width": 1280, "height": 720, "size": 3000000},
        },
    }],
}


def test_pexels_search_normalizes_video(monkeypatch):
    _patch_key(monkeypatch, "PEXELS_API_KEY", "fake-key")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/videos/search"
        assert request.headers["Authorization"] == "fake-key"
        return httpx.Response(200, json=_PEXELS_VIDEO_RESP)

    _mount(monkeypatch, handler, None)
    hits = PexelsStockProvider().search(StockSearchRequest(query="ocean", media_type=StockMediaType.VIDEO,
                                                           orientation=StockOrientation.LANDSCAPE))
    assert len(hits) == 1
    h = hits[0]
    assert h.provider == "pexels"
    assert h.media_type == StockMediaType.VIDEO
    assert h.asset_id == "123"
    assert h.download_url == "https://cdn/pexels-123-hd.mp4"  # HD preferred
    assert h.author == "Jane"
    assert h.license_commercial_use == "allowed"
    assert h.orientation == StockOrientation.LANDSCAPE


def test_pixabay_search_normalizes_video(monkeypatch):
    _patch_key(monkeypatch, "PIXABAY_API_KEY", "fake-key")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/videos/"
        assert "key=fake-key" in str(request.url)
        return httpx.Response(200, json=_PIXABAY_VIDEO_RESP)

    _mount(monkeypatch, handler, None)
    hits = PixabayStockProvider().search(StockSearchRequest(query="city", media_type=StockMediaType.VIDEO))
    assert len(hits) == 1
    h = hits[0]
    assert h.provider == "pixabay"
    assert h.asset_id == "456"
    assert h.download_url == "https://cdn/pixabay-456-large.mp4"  # large preferred
    assert h.duration == 12.0
    assert h.license_name == "Pixabay Content License"


def test_pexels_search_disabled_without_key(monkeypatch):
    _patch_key(monkeypatch, "PEXELS_API_KEY", "")
    with pytest.raises(VideoError) as ei:
        PexelsStockProvider().search(StockSearchRequest(query="x"))
    assert ei.value.code == TypedErrorCode.PROVIDER_DISABLED


def test_pexels_search_rate_limit(monkeypatch):
    _patch_key(monkeypatch, "PEXELS_API_KEY", "fake-key")
    _mount(monkeypatch, lambda r: httpx.Response(429, text="rate"), None)
    with pytest.raises(VideoError) as ei:
        PexelsStockProvider().search(StockSearchRequest(query="x"))
    assert ei.value.code == TypedErrorCode.QUOTA_EXCEEDED


def test_pixabay_search_http_error(monkeypatch):
    _patch_key(monkeypatch, "PIXABAY_API_KEY", "fake-key")
    _mount(monkeypatch, lambda r: httpx.Response(500, text="err"), None)
    with pytest.raises(VideoError) as ei:
        PixabayStockProvider().search(StockSearchRequest(query="x"))
    assert ei.value.code == TypedErrorCode.STOCK_SEARCH_FAILED


# --------------------------------------------------------------------- download + cache
def _binary(n: int) -> bytes:
    return bytes(i % 256 for i in range(n))


def test_download_writes_and_hashes(monkeypatch, tmp_path):
    _patch_key(monkeypatch, "PEXELS_API_KEY", "fake-key")
    monkeypatch.setattr(config, "ASSET_CACHE_DIR", tmp_path / "cache")
    payload = _binary(2048)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/videos/search":
            return httpx.Response(200, json=_PEXELS_VIDEO_RESP)
        if request.url.host == "cdn":
            return httpx.Response(200, content=payload)
        return httpx.Response(404)

    _mount(monkeypatch, handler, None)
    p = PexelsStockProvider()
    hits = p.search(StockSearchRequest(query="x"))
    dest = tmp_path / "v.mp4"
    res = p.download(hits[0], destination=dest)
    assert res.path == dest
    assert dest.exists() and dest.stat().st_size == 2048
    assert res.sha256 == file_sha256(dest)
    assert res.bytes_size == 2048


def test_download_cache_reuse(monkeypatch, tmp_path):
    _patch_key(monkeypatch, "PEXELS_API_KEY", "fake-key")
    monkeypatch.setattr(config, "ASSET_CACHE_DIR", tmp_path / "cache")
    calls = {"n": 0}
    payload = _binary(1024)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/videos/search":
            return httpx.Response(200, json=_PEXELS_VIDEO_RESP)
        if request.url.host == "cdn":
            calls["n"] += 1
            return httpx.Response(200, content=payload)
        return httpx.Response(404)

    _mount(monkeypatch, handler, None)
    p = PexelsStockProvider()
    hit = p.search(StockSearchRequest(query="x"))[0]
    r1 = p.download(hit)
    r2 = p.download(hit)  # second call should hit cache, no re-download
    assert r1.sha256 == r2.sha256
    assert calls["n"] == 1  # downloaded only once


def test_download_rejects_empty(monkeypatch, tmp_path):
    _patch_key(monkeypatch, "PEXELS_API_KEY", "fake-key")
    monkeypatch.setattr(config, "ASSET_CACHE_DIR", tmp_path / "cache")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/videos/search":
            return httpx.Response(200, json=_PEXELS_VIDEO_RESP)
        return httpx.Response(200, content=b"")

    _mount(monkeypatch, handler, None)
    p = PexelsStockProvider()
    hit = p.search(StockSearchRequest(query="x"))[0]
    with pytest.raises(VideoError) as ei:
        p.download(hit, destination=tmp_path / "e.mp4")
    assert ei.value.code == TypedErrorCode.STOCK_DOWNLOAD_FAILED


def test_download_disabled_without_key(monkeypatch, tmp_path):
    _patch_key(monkeypatch, "PEXELS_API_KEY", "")
    hit = StockHit(provider="pexels", media_type=StockMediaType.VIDEO, asset_id="1",
                   page_url="", download_url="https://cdn/x.mp4")
    with pytest.raises(VideoError) as ei:
        PexelsStockProvider().download(hit, destination=tmp_path / "x.mp4")
    assert ei.value.code == TypedErrorCode.PROVIDER_DISABLED


def test_file_sha256_deterministic(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    assert file_sha256(f) == file_sha256(f)

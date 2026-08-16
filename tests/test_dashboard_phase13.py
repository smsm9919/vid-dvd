"""Tests for Phase 13 dashboard routes (free-first hub)."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import config


client = TestClient(app)


# --------------------------------------------------------------------- router status
def test_dashboard_router_status_lists_all_providers():
    r = client.get("/api/dashboard/providers/router")
    assert r.status_code == 200
    data = r.json()
    names = [p["name"] for p in data["providers"]]
    assert "wikimedia" in names  # keyless, always available
    assert "pexels" in names
    assert "pixabay" in names
    assert "piper" in names
    assert data["count"] == 4


def test_dashboard_router_status_reports_paid_block():
    r = client.get("/api/dashboard/providers/router")
    data = r.json()
    # All three providers are free, so none should be blocked as paid.
    for p in data["providers"]:
        assert p["cost"]["is_paid"] is False


# --------------------------------------------------------------------- stock search
def test_stock_search_keyless_wikimedia_works_without_keys(monkeypatch):
    """Wikimedia is keyless — stock search works even with no Pexels/Pixabay keys."""
    monkeypatch.setattr(config, "PEXELS_API_KEY", "")
    monkeypatch.setattr(config, "PIXABAY_API_KEY", "")
    r = client.get("/api/dashboard/stock/search", params={"query": "ocean"})
    assert r.status_code == 200
    data = r.json()
    # Wikimedia is available keyless, so status is OK (not NOT_CONFIGURED).
    assert data["status"] == "OK"
    assert data["provider"] == "wikimedia"


def test_stock_search_not_configured_when_no_provider(monkeypatch):
    """When ALL stock providers are unavailable, returns NOT_CONFIGURED honestly."""
    # Make Wikimedia unavailable by patching its availability.
    from app.providers.stock_adapters import build_stock_providers
    import app.providers.wikimedia as wm
    monkeypatch.setattr(config, "PEXELS_API_KEY", "")
    monkeypatch.setattr(config, "PIXABAY_API_KEY", "")
    monkeypatch.setattr(wm.WikimediaCommonsProvider, "available", property(lambda self: False))
    r = client.get("/api/dashboard/stock/search", params={"query": "ocean"})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "NOT_CONFIGURED"
    assert data["hits"] == []


def test_stock_search_invalid_media_type():
    r = client.get("/api/dashboard/stock/search", params={"query": "x", "media_type": "bogus"})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "INVALID_MEDIA_TYPE"


def test_stock_search_invalid_orientation():
    r = client.get("/api/dashboard/stock/search", params={"query": "x", "orientation": "diagonal"})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "INVALID_ORIENTATION"


# --------------------------------------------------------------------- provider panel
def test_provider_panel_includes_stock_and_router():
    r = client.get("/api/dashboard/providers")
    assert r.status_code == 200
    data = r.json()
    assert "stock" in data
    assert "router" in data
    assert "free_first" in data
    assert isinstance(data["stock"], list)
    stock_names = [s["name"] for s in data["stock"]]
    assert "wikimedia" in stock_names  # keyless provider
    assert "pexels" in stock_names
    assert "pixabay" in stock_names


def test_provider_panel_free_first_policy():
    r = client.get("/api/dashboard/providers")
    data = r.json()
    ff = data["free_first"]
    assert ff["free_first"] is True
    assert ff["allow_paid_providers"] is False
    assert "blocked" in ff["policy"].lower()


# --------------------------------------------------------------------- readiness
def test_readiness_includes_stock_stage():
    r = client.get("/api/dashboard/readiness")
    assert r.status_code == 200
    data = r.json()
    assert "stock" in data
    # Wikimedia is keyless and always available, so stock is READY.
    assert data["stock"] == "READY"
    assert "providers" in data
    assert "stock" in data["providers"]


# --------------------------------------------------------------------- asset library + license
def test_asset_library_empty_for_new_project():
    # Use a non-existent project id — should return empty, not error.
    r = client.get("/api/dashboard/projects/nonexistent_pid/asset-library")
    assert r.status_code == 200
    data = r.json()
    assert data["assets"] == []
    assert data["total"] == 0


def test_license_report_empty_for_new_project():
    r = client.get("/api/dashboard/projects/nonexistent_pid/license-report")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 0
    assert data["assets"] == []


def test_asset_library_invalid_type_rejected():
    r = client.get("/api/dashboard/projects/somepid/asset-library", params={"type": "bogus"})
    assert r.status_code == 400


# --------------------------------------------------------------------- tts status diagnostic
def test_tts_status_not_configured_when_piper_disabled(monkeypatch):
    monkeypatch.setattr(config, "PIPER_ENABLED", False)
    r = client.get("/api/dashboard/providers")
    data = r.json()
    assert data["tts"]["status"] == "NOT_CONFIGURED"
    assert "Piper" in data["tts"]["reason"] or "TTS" in data["tts"]["reason"]


def test_stock_download_missing_project_id():
    r = client.post("/api/dashboard/stock/download", json={"hit": {}})
    assert r.status_code == 400

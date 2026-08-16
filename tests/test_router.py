"""Tests for the free-first provider router (Phase 13)."""

import pytest

from app import config
from app.providers.contracts import (
    Capability,
    ProviderCapability,
    ProviderCost,
    ProviderKind,
    ProviderLicense,
    ProviderMeta,
    ProviderRuntime,
)
from app.providers.router import ProviderRouter, build_default_router, build_tts_providers
from app.core.errors import TypedErrorCode, VideoError


class FakeProvider:
    """Minimal provider exposing meta() + available + name."""

    def __init__(self, name, *, available, paid=False, caps=None, langs=None,
                 cpu_fallback=True, commercial="allowed", cost_estimate=0.0):
        self._name = name
        self._available = available
        self._caps = caps or []
        self._langs = langs or []
        self._paid = paid
        self._cpu = cpu_fallback
        self._commercial = commercial
        self._cost = cost_estimate

    @property
    def name(self):
        return self._name

    @property
    def available(self):
        return self._available

    def meta(self):
        return ProviderMeta(
            name=self._name, kind=ProviderKind.STOCK,
            cost=ProviderCost(is_paid=self._paid, estimate_per_call=self._cost),
            license=ProviderLicense(commercial_use=self._commercial),
            runtime=ProviderRuntime(cpu_fallback=self._cpu),
            capability=ProviderCapability(capabilities=self._caps, languages=self._langs),
        )


# --------------------------------------------------------------------- free-first
def test_free_preferred_over_paid(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_PAID_PROVIDERS", True)
    monkeypatch.setattr(config, "MAX_PAID_COST_USD", 1.0)
    r = ProviderRouter()
    paid = FakeProvider("paid", available=True, paid=True, caps=[Capability.STOCK_VIDEO], cost_estimate=0.05)
    free = FakeProvider("free", available=True, paid=False, caps=[Capability.STOCK_VIDEO])
    r.register(paid, paid)
    r.register(free, free)
    mp, inst = r.select(capability=Capability.STOCK_VIDEO)
    assert inst.name == "free"


def test_paid_blocked_when_free_first(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_PAID_PROVIDERS", False)
    r = ProviderRouter()
    paid = FakeProvider("paid", available=True, paid=True, caps=[Capability.STOCK_VIDEO], cost_estimate=0.05)
    r.register(paid, paid)
    assert r.select(capability=Capability.STOCK_VIDEO) is None


def test_paid_allowed_within_budget(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_PAID_PROVIDERS", True)
    monkeypatch.setattr(config, "MAX_PAID_COST_USD", 0.10)
    r = ProviderRouter()
    paid = FakeProvider("paid", available=True, paid=True, caps=[Capability.STOCK_VIDEO], cost_estimate=0.05)
    r.register(paid, paid)
    mp, inst = r.select(capability=Capability.STOCK_VIDEO)
    assert inst.name == "paid"


def test_paid_blocked_over_budget(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_PAID_PROVIDERS", True)
    monkeypatch.setattr(config, "MAX_PAID_COST_USD", 0.01)
    r = ProviderRouter()
    paid = FakeProvider("paid", available=True, paid=True, caps=[Capability.STOCK_VIDEO], cost_estimate=0.05)
    r.register(paid, paid)
    assert r.select(capability=Capability.STOCK_VIDEO) is None


# --------------------------------------------------------------------- eligibility
def test_unavailable_provider_skipped():
    r = ProviderRouter()
    r.register(FakeProvider("off", available=False, caps=[Capability.STOCK_VIDEO]),
               FakeProvider("off", available=False, caps=[Capability.STOCK_VIDEO]))
    assert r.select(capability=Capability.STOCK_VIDEO) is None


def test_capability_must_match():
    r = ProviderRouter()
    r.register(FakeProvider("img", available=True, caps=[Capability.STOCK_IMAGE]),
               FakeProvider("img", available=True, caps=[Capability.STOCK_IMAGE]))
    assert r.select(capability=Capability.STOCK_VIDEO) is None
    mp, inst = r.select(capability=Capability.STOCK_IMAGE)
    assert inst.name == "img"


def test_language_must_match_when_declared():
    r = ProviderRouter()
    r.register(FakeProvider("p", available=True, caps=[Capability.TEXT_TO_SPEECH], langs=["en", "de"]),
               FakeProvider("p", available=True, caps=[Capability.TEXT_TO_SPEECH], langs=["en", "de"]))
    assert r.select(capability=Capability.TEXT_TO_SPEECH, language="ar") is None
    mp, inst = r.select(capability=Capability.TEXT_TO_SPEECH, language="de")
    assert inst.name == "p"


def test_no_language_filter_when_provider_has_no_languages():
    r = ProviderRouter()
    r.register(FakeProvider("p", available=True, caps=[Capability.STOCK_VIDEO], langs=[]),
               FakeProvider("p", available=True, caps=[Capability.STOCK_VIDEO], langs=[]))
    mp, inst = r.select(capability=Capability.STOCK_VIDEO, language="anything")
    assert inst is not None


# --------------------------------------------------------------------- select_or_raise
def test_select_or_raise_no_provider():
    r = ProviderRouter()
    with pytest.raises(VideoError) as ei:
        r.select_or_raise(capability=Capability.STOCK_VIDEO)
    assert ei.value.code == TypedErrorCode.NO_PROVIDER


# --------------------------------------------------------------------- status
def test_status_reports_eligibility(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_PAID_PROVIDERS", False)
    r = ProviderRouter()
    r.register(FakeProvider("free", available=True, caps=[Capability.STOCK_VIDEO]),
               FakeProvider("free", available=True, caps=[Capability.STOCK_VIDEO]))
    r.register(FakeProvider("paid", available=True, paid=True, caps=[Capability.STOCK_VIDEO]),
               FakeProvider("paid", available=True, paid=True, caps=[Capability.STOCK_VIDEO]))
    status = r.status()
    assert len(status) == 2
    free_s = next(s for s in status if s["name"] == "free")
    paid_s = next(s for s in status if s["name"] == "paid")
    assert free_s["eligible"] is True
    assert paid_s["eligible"] is False
    assert "blocked" in paid_s["reason"].lower()


# --------------------------------------------------------------------- tier ordering
def test_cpu_fallback_preferred_over_network_only_free(monkeypatch):
    r = ProviderRouter()
    # Both free, but one has cpu_fallback=True (tier 0), other False (tier 1).
    net = FakeProvider("net", available=True, caps=[Capability.STOCK_VIDEO], cpu_fallback=False)
    local = FakeProvider("local", available=True, caps=[Capability.STOCK_VIDEO], cpu_fallback=True)
    r.register(net, net)
    r.register(local, local)
    mp, inst = r.select(capability=Capability.STOCK_VIDEO)
    assert inst.name == "local"


# --------------------------------------------------------------------- factory
def test_build_default_router_has_stock_and_piper():
    r = build_default_router()
    names = [mp.name for mp, _ in r.providers]
    assert "pexels" in names
    assert "pixabay" in names
    assert "piper" in names


def test_build_tts_providers_empty_when_piper_disabled():
    # PIPER_ENABLED defaults false → no TTS providers.
    providers = build_tts_providers()
    assert providers == []


def test_build_tts_providers_piper_when_enabled(monkeypatch):
    import shutil
    monkeypatch.setattr(config, "PIPER_ENABLED", True)
    if not shutil.which("piper"):
        pytest.skip("piper binary not installed")
    providers = build_tts_providers()
    assert len(providers) == 1
    assert providers[0].name == "piper"

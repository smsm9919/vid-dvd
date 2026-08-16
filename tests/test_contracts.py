"""Unit tests for the Phase 13 generic provider contracts.

These verify the cross-cutting metadata (cost/license/capability/runtime) that
the multi-provider hub relies on for safe routing. No provider behaviour is
mocked here — only the data contracts.
"""

from app.providers.contracts import (
    Capability,
    ProviderCapability,
    ProviderCost,
    ProviderKind,
    ProviderLicense,
    ProviderMeta,
    ProviderRuntime,
    ProviderStatus,
)


def test_provider_kind_values():
    assert ProviderKind.LOCAL.value == "local"
    assert ProviderKind.REMOTE.value == "remote"
    assert ProviderKind.STOCK.value == "stock"


def test_provider_status_has_honest_states():
    states = {s.value for s in ProviderStatus}
    assert {"READY", "BLOCKED", "NOT_CONFIGURED", "DEGRADED", "ERROR"} <= states


def test_license_defaults_to_unknown_commercial_use():
    lic = ProviderLicense()
    assert lic.commercial_use == "unknown"
    assert lic.attribution_required is False
    assert lic.name == "Unknown"


def test_license_allows_explicit_commercial_allowed_only_when_set():
    lic = ProviderLicense(name="Pexels License", commercial_use="allowed")
    assert lic.commercial_use == "allowed"
    assert lic.to_dict()["commercial_use"] == "allowed"


def test_cost_defaults_to_free():
    cost = ProviderCost()
    assert cost.is_paid is False
    assert cost.estimate_per_call == 0.0


def test_cost_paid_metadata_preserved():
    cost = ProviderCost(is_paid=True, unit="usd_per_second", estimate_per_call=0.02)
    d = cost.to_dict()
    assert d["is_paid"] is True
    assert d["unit"] == "usd_per_second"


def test_capability_supports_check():
    cap = ProviderCapability(capabilities=[Capability.STOCK_VIDEO, Capability.STOCK_IMAGE])
    assert cap.supports(Capability.STOCK_VIDEO)
    assert not cap.supports(Capability.TEXT_TO_VIDEO)


def test_runtime_defaults_no_gpu():
    rt = ProviderRuntime()
    assert rt.requires_gpu is False
    assert rt.cpu_fallback is False


def test_runtime_gpu_required_flag():
    rt = ProviderRuntime(requires_gpu=True, requires_vram_gb=8.0, cpu_fallback=False)
    assert rt.to_dict()["requires_vram_gb"] == 8.0


def test_provider_meta_full_round_trip():
    meta = ProviderMeta(
        name="pexels",
        kind=ProviderKind.STOCK,
        version="1",
        cost=ProviderCost(is_paid=False, unit="free"),
        license=ProviderLicense(name="Pexels License", commercial_use="allowed"),
        runtime=ProviderRuntime(requires_api_key=True, requires_network=True),
        capability=ProviderCapability(
            capabilities=[Capability.STOCK_VIDEO],
            supported_orientations=["portrait", "landscape"],
        ),
    )
    d = meta.to_dict()
    assert d["name"] == "pexels"
    assert d["kind"] == "stock"
    assert d["license"]["commercial_use"] == "allowed"
    assert d["cost"]["is_paid"] is False
    assert d["runtime"]["requires_api_key"] is True
    assert "stock_video" in d["capability"]["capabilities"]


def test_provider_meta_paid_provider_still_represented():
    """Paid providers are not hidden — they are disabled by the router, not erased."""
    meta = ProviderMeta(
        name="paid-video",
        kind=ProviderKind.REMOTE,
        cost=ProviderCost(is_paid=True, estimate_per_call=0.05),
    )
    assert meta.cost.is_paid is True
    assert meta.to_dict()["cost"]["estimate_per_call"] == 0.05

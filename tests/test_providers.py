"""Tests for the VideoProvider base contract and registry (Phase 4)."""

import asyncio
from pathlib import Path

import pytest

from app.core.errors import TypedErrorCode, VideoError
from app.providers.base import (
    GenerationRequest,
    GenerationResult,
    ProviderInfo,
    ProviderUnavailableError,
    VideoProvider,
)
from app.providers.registry import build_providers, provider_status, select_provider


class _FakeProvider(VideoProvider):
    """A controllable fake provider for registry/contract tests."""

    def __init__(self, name: str, healthy: bool):
        self._name = name
        self._healthy = healthy

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(name=self._name, kind="test", description="fake")

    async def detect(self) -> bool:
        return self._healthy

    async def health(self) -> dict:
        return {"ok": self._healthy, "data": {"name": self._name}}

    async def list_models(self) -> list[str]:
        return ["m1", "m2"] if self._healthy else []

    async def validate(self) -> dict:
        return {"ok": self._healthy, "issues": [] if self._healthy else ["down"]}

    async def generate(self, request, destination):
        destination = Path(destination)
        destination.write_bytes(b"\x00\x00\x00\x00")  # NOT a real mp4; only for shape tests
        return GenerationResult(path=destination, prompt_id="fake")


def test_generation_request_defaults():
    r = GenerationRequest(prompt="a lion")
    assert r.prompt == "a lion"
    assert r.negative_prompt == ""
    assert r.duration == 4.0
    assert r.width == 1080 and r.height == 1920
    assert r.reference_images == []


def test_base_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        VideoProvider()  # type: ignore[abstract]


def test_provider_unavailable_error_is_no_provider():
    e = ProviderUnavailableError()
    assert e.code is TypedErrorCode.NO_PROVIDER
    assert isinstance(e, VideoError)


def test_select_provider_returns_first_healthy():
    providers = [_FakeProvider("a", False), _FakeProvider("b", True), _FakeProvider("c", True)]
    chosen = asyncio.run(select_provider(providers))
    assert chosen.info.name == "b"


def test_select_provider_raises_no_provider_when_all_down():
    providers = [_FakeProvider("a", False), _FakeProvider("b", False)]
    with pytest.raises(VideoError) as exc:
        asyncio.run(select_provider(providers))
    assert exc.value.code is TypedErrorCode.NO_PROVIDER


def test_provider_status_reports_all():
    providers = [_FakeProvider("a", True), _FakeProvider("b", False)]
    status = asyncio.run(provider_status(providers))
    assert len(status) == 2
    assert status[0]["ok"] is True and status[0]["name"] == "a"
    assert status[1]["ok"] is False and status[1]["name"] == "b"


def test_build_providers_includes_comfyui():
    providers = build_providers()
    assert any(p.info.name == "comfyui" for p in providers)


def test_select_provider_real_comfyui_unreachable_returns_no_provider():
    # In this sandbox ComfyUI is NOT running. select_provider must NOT fake a
    # provider; it must raise NO_PROVIDER rather than returning something usable.
    try:
        p = asyncio.run(select_provider(build_providers()))
        # If a provider is somehow selected, it must actually be healthy.
        assert asyncio.run(p.detect()) is True
    except VideoError as e:
        assert e.code is TypedErrorCode.NO_PROVIDER

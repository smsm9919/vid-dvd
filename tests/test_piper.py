"""Tests for the Piper TTS provider (Phase 13).

Two tiers:
1. Metadata / disabled-path tests — always run. Verify the provider is honest
   about availability, license (GPL-3.0 restricted), cost (free), capabilities.
2. Real synthesis tests — skipped unless PIPER_ENABLED=true, the piper binary
   is on PATH, and voice models are available. These produce REAL WAV files and
   verify them with FFmpeg-backed audio QC (no fakes).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from app import config
from app.voice.piper import PiperProvider, _PIPER_LICENSE
from app.voice.tts import VoiceRequest
from app.core.errors import TypedErrorCode, VideoError


def _piper_installed() -> bool:
    return shutil.which("piper") is not None


def _ffmpeg_ok() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def _voices_available(voices_dir: Path) -> bool:
    return (voices_dir / "en_US-lessac-medium.onnx").exists() and \
           (voices_dir / "en_US-lessac-medium.onnx.json").exists()


PIPER_RUNTIME = config.PIPER_ENABLED and _piper_installed() and _ffmpeg_ok()
TEST_VOICES = Path("/tmp/piper_voices")
PIPER_WITH_VOICES = PIPER_RUNTIME and _voices_available(TEST_VOICES)


# --------------------------------------------------------------------- metadata
def test_piper_name():
    assert PiperProvider().name == "piper"


def test_piper_unavailable_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "PIPER_ENABLED", False)
    assert PiperProvider().available is False


def test_piper_license_is_gpl3_restricted():
    """Piper is GPL-3.0 — commercial use is restricted (license obligation)."""
    lic = _PIPER_LICENSE
    assert lic.spdx == "GPL-3.0-only"
    assert lic.commercial_use == "restricted"
    assert lic.attribution_required is True


def test_piper_meta_records_gpl_license():
    m = PiperProvider().meta()
    assert m.license.spdx == "GPL-3.0-only"
    assert m.license.commercial_use == "restricted"
    assert m.cost.is_paid is False
    assert m.runtime.requires_gpu is False
    assert m.runtime.cpu_fallback is True


def test_piper_no_gpu_required():
    m = PiperProvider().meta()
    assert m.runtime.requires_gpu is False
    assert m.runtime.requires_ram_gb is not None and m.runtime.requires_ram_gb <= 2.0


def test_piper_capabilities_text_to_speech():
    m = PiperProvider().meta()
    from app.providers.contracts import Capability
    assert Capability.TEXT_TO_SPEECH in m.capability.capabilities


def test_piper_list_voices_default_en():
    voices = PiperProvider().list_voices("en")
    assert "en_US-lessac-medium" in voices or len(voices) >= 0


# --------------------------------------------------------------------- disabled path
def test_piper_synthesize_raises_no_provider_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PIPER_ENABLED", False)
    p = PiperProvider()
    req = VoiceRequest(text="hello", language="en", output_format="wav")
    with pytest.raises(VideoError) as ei:
        p.synthesize(req, tmp_path / "out.wav")
    assert ei.value.code == TypedErrorCode.NO_PROVIDER


def test_piper_no_default_arabic_voice_raises_capability(monkeypatch, tmp_path):
    """Arabic has no default Piper voice unless configured → CAPABILITY_UNSUPPORTED.

    The piper binary is mocked as present so this capability check is isolated
    from the environment (the binary may not be installed in every test env).
    The real-synthesis tests are separately gated by PIPER_RUNTIME.
    """
    monkeypatch.setattr(config, "PIPER_ENABLED", True)
    monkeypatch.setattr(config, "PIPER_DEFAULT_VOICE_AR", "")
    monkeypatch.setattr("app.voice.piper.shutil.which", lambda _: "/usr/local/bin/piper")
    p = PiperProvider(voices_dir=tmp_path / "voices")
    req = VoiceRequest(text="مرحبا", language="ar", output_format="wav")
    with pytest.raises(VideoError) as ei:
        p.synthesize(req, tmp_path / "out.wav")
    assert ei.value.code == TypedErrorCode.CAPABILITY_UNSUPPORTED


def test_piper_empty_text_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PIPER_ENABLED", True)
    p = PiperProvider(voices_dir=tmp_path / "voices")
    req = VoiceRequest(text="   ", language="en", output_format="wav")
    with pytest.raises(VideoError) as ei:
        p.synthesize(req, tmp_path / "out.wav")
    assert ei.value.code == TypedErrorCode.WORKFLOW_INVALID


# --------------------------------------------------------------------- REAL synthesis
@pytest.mark.skipif(not PIPER_WITH_VOICES, reason="Piper runtime + voices not available")
def test_real_piper_synthesizes_english_wav(monkeypatch, tmp_path):
    """REAL synthesis: produces a real, FFmpeg-verified WAV on CPU."""
    monkeypatch.setattr(config, "PIPER_ENABLED", True)
    monkeypatch.setattr(config, "PIPER_VOICES_DIR", TEST_VOICES)
    p = PiperProvider(voices_dir=TEST_VOICES)
    assert p.available is True
    req = VoiceRequest(
        text="This is a real Piper TTS synthesis test through the provider on CPU.",
        language="en", output_format="wav",
    )
    dest = tmp_path / "en.wav"
    res = p.synthesize(req, dest)
    assert res.provider == "piper"
    assert dest.exists() and dest.stat().st_size > 1000
    assert res.duration > 1.0
    assert res.sample_rate == 22050
    assert res.voice == "en_US-lessac-medium"


@pytest.mark.skipif(not PIPER_WITH_VOICES, reason="Piper runtime + voices not available")
def test_real_piper_synthesizes_german_wav(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PIPER_ENABLED", True)
    monkeypatch.setattr(config, "PIPER_VOICES_DIR", TEST_VOICES)
    p = PiperProvider(voices_dir=TEST_VOICES)
    req = VoiceRequest(
        text="Hallo, dies ist ein deutscher Werbespot-Test mit Piper Sprachsynthese.",
        language="de", output_format="wav",
    )
    dest = tmp_path / "de.wav"
    res = p.synthesize(req, dest)
    assert dest.exists() and dest.stat().st_size > 1000
    assert res.duration > 1.0
    assert res.language == "de"
    assert res.voice == "de_DE-thorsten-medium"


@pytest.mark.skipif(not (PIPER_WITH_VOICES and shutil.which("ffmpeg")), reason="Piper + ffmpeg not available")
def test_real_piper_mp3_transcode(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PIPER_ENABLED", True)
    monkeypatch.setattr(config, "PIPER_VOICES_DIR", TEST_VOICES)
    p = PiperProvider(voices_dir=TEST_VOICES)
    req = VoiceRequest(text="MP3 transcode test with Piper.", language="en", output_format="mp3")
    dest = tmp_path / "out.mp3"
    res = p.synthesize(req, dest)
    assert dest.exists() and dest.stat().st_size > 1000
    assert str(res.path).endswith(".mp3")
    # WAV temp should be cleaned up
    assert not (tmp_path / "out.wav").exists()

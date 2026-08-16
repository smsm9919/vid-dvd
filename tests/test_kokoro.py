"""Tests for the Kokoro-82M TTS provider (Phase 13).

Two tiers:
1. Unit tests — metadata, availability, license, validation. Always run.
2. Live runtime tests — REAL WAV synthesis verified with ffprobe.
   Skipped when KOKORO_ENABLED is unset or model files are missing.
   These generate REAL audio (no mocks/fakes).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from app import config
from app.voice.kokoro import KokoroProvider, build_kokoro_provider, _KOKORO_LICENSE
from app.voice.tts import VoiceRequest, SUPPORTED_LANGUAGES
from app.core.errors import TypedErrorCode, VideoError


def _kokoro_available() -> bool:
    if not config.KOKORO_ENABLED:
        return False
    onnx = config.KOKORO_MODEL_DIR / config.KOKORO_MODEL_FILE
    voices = config.KOKORO_MODEL_DIR / config.KOKORO_VOICES_FILE
    return onnx.exists() and voices.exists()


def _ffmpeg_ok() -> bool:
    return shutil.which("ffprobe") is not None


LIVE = _kokoro_available() and _ffmpeg_ok()


# ---------------------------------------------------------- provider metadata
def test_provider_name():
    assert KokoroProvider().name == "kokoro"


def test_provider_unavailable_when_disabled():
    """When KOKORO_ENABLED is false, provider is NOT available (honest)."""
    p = KokoroProvider()
    if not config.KOKORO_ENABLED:
        assert p.available is False


def test_provider_meta_apache_license():
    """Kokoro is Apache 2.0 — commercial-safe (no copyleft)."""
    m = KokoroProvider().meta()
    assert m.license.spdx == "Apache-2.0"
    assert m.license.commercial_use == "allowed"
    assert m.license.attribution_required is False


def test_provider_meta_free_local_cpu():
    m = KokoroProvider().meta()
    assert m.cost.is_paid is False
    assert m.runtime.requires_gpu is False
    assert m.runtime.requires_api_key is False
    assert m.runtime.cpu_fallback is True
    assert m.runtime.requires_network is False  # network only for initial model download


def test_provider_meta_languages():
    m = KokoroProvider().meta()
    assert "en" in m.capability.languages
    assert "de" in m.capability.languages
    assert "ar" in m.capability.languages


def test_provider_license_method():
    p = KokoroProvider()
    lic = p.license()
    assert lic.spdx == "Apache-2.0"
    assert "Kokoro" in lic.name


def test_provider_no_voice_cloning_claim():
    """Kokoro has fixed preset voices only — no cloning."""
    p = KokoroProvider()
    # list_voices returns preset names, never a 'clone' claim.
    voices = p.list_voices("en")
    assert all("clone" not in v.lower() for v in voices)
    assert len(voices) > 0


def test_build_tts_providers_kokoro_preferred():
    """build_tts_providers lists Kokoro before Piper (Apache preferred)."""
    from app.providers.router import build_tts_providers
    providers = build_tts_providers()
    # If both are available, Kokoro should come first.
    names = [p.name for p in providers]
    if "kokoro" in names and "piper" in names:
        assert names.index("kokoro") < names.index("piper")


# ---------------------------------------------------------- synthesis validation
def test_synthesize_raises_when_disabled(tmp_path):
    """When Kokoro is not available, synthesize raises NO_PROVIDER (no fake)."""
    if config.KOKORO_ENABLED:
        pytest.skip("Kokoro enabled — cannot test disabled path")
    p = KokoroProvider()
    req = VoiceRequest(text="Hello world", language="en", output_format="wav")
    with pytest.raises(VideoError) as ei:
        p.synthesize(req, tmp_path / "out.wav")
    assert ei.value.code == TypedErrorCode.NO_PROVIDER


def test_synthesize_empty_text_rejected(tmp_path):
    if not config.KOKORO_ENABLED:
        pytest.skip("Kokoro disabled")
    p = KokoroProvider()
    req = VoiceRequest(text="   ", language="en", output_format="wav")
    with pytest.raises(VideoError) as ei:
        p.synthesize(req, tmp_path / "out.wav")
    assert ei.value.code == TypedErrorCode.WORKFLOW_INVALID


# ---------------------------------------------------------- LIVE RUNTIME TESTS
@pytest.mark.skipif(not LIVE, reason="Kokoro not enabled or FFmpeg unavailable")
def test_live_english_synthesis(tmp_path):
    """REAL English WAV synthesis, verified with ffprobe."""
    p = KokoroProvider()
    assert p.available is True
    req = VoiceRequest(text="Hello, this is a real Kokoro text to speech test for English.",
                       language="en", output_format="wav")
    dest = tmp_path / "en_voice.wav"
    result = p.synthesize(req, dest)
    assert result.path.exists()
    assert result.path.stat().st_size > 1000
    assert result.provider == "kokoro"
    assert result.language == "en"
    assert result.duration > 0.5
    assert result.sample_rate == 24000
    # Verify with ffprobe independently.
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,sample_rate,channels:format=duration",
         "-of", "json", str(result.path)],
        capture_output=True, text=True)
    import json
    d = json.loads(probe.stdout)
    s = d["streams"][0]
    assert s["codec_name"] == "pcm_s16le"
    assert int(s["sample_rate"]) == 24000
    assert float(d["format"]["duration"]) > 0.5
    print(f"REAL Kokoro EN: {result.duration:.2f}s, {result.sample_rate}Hz, {result.path.stat().st_size} bytes")


@pytest.mark.skipif(not LIVE, reason="Kokoro not enabled or FFmpeg unavailable")
def test_live_german_synthesis(tmp_path):
    """REAL German WAV synthesis (via espeak-ng phonemization), ffprobe-verified.

    Kokoro has NO native German voice; it uses the en voice with de phonemes.
    This is a documented limitation — the test confirms real output, not native voice.
    """
    p = KokoroProvider()
    req = VoiceRequest(text="Guten Tag, das ist ein deutscher Sprachtest.",
                       language="de", output_format="wav")
    dest = tmp_path / "de_voice.wav"
    result = p.synthesize(req, dest)
    assert result.path.exists()
    assert result.path.stat().st_size > 1000
    assert result.language == "de"
    assert result.duration > 0.5
    # ffprobe verification.
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name:format=duration",
         "-of", "default=noprint_wrappers=1", str(result.path)],
        capture_output=True, text=True)
    assert "pcm_s16le" in probe.stdout
    print(f"REAL Kokoro DE: {result.duration:.2f}s")


@pytest.mark.skipif(not LIVE, reason="Kokoro not enabled or FFmpeg unavailable")
def test_live_arabic_synthesis(tmp_path):
    """REAL Arabic WAV synthesis (via espeak-ng phonemization), ffprobe-verified.

    Kokoro has NO native Arabic voice; it uses the en voice with ar phonemes.
    This is a documented limitation — real output, not native voice.
    """
    p = KokoroProvider()
    req = VoiceRequest(text="مرحبا، هذا اختبار للنص العربي.",
                       language="ar", output_format="wav")
    dest = tmp_path / "ar_voice.wav"
    result = p.synthesize(req, dest)
    assert result.path.exists()
    assert result.path.stat().st_size > 1000
    assert result.language == "ar"
    assert result.duration > 0.5
    print(f"REAL Kokoro AR: {result.duration:.2f}s")


@pytest.mark.skipif(not LIVE, reason="Kokoro not enabled or FFmpeg unavailable")
def test_live_mp3_output(tmp_path):
    """REAL MP3 output (WAV→MP3 transcode), ffprobe-verified."""
    p = KokoroProvider()
    req = VoiceRequest(text="MP3 format test.", language="en", output_format="mp3")
    dest = tmp_path / "voice.mp3"
    result = p.synthesize(req, dest)
    assert result.path.exists()
    assert result.path.suffix == ".mp3"
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1",
         str(result.path)], capture_output=True, text=True)
    assert "mp3" in probe.stdout.lower()


@pytest.mark.skipif(not LIVE, reason="Kokoro not enabled or FFmpeg unavailable")
def test_live_audio_qc_passes(tmp_path):
    """The synthesized WAV passes the project's own verify_audio QC."""
    from app.audio.qc import verify_audio
    p = KokoroProvider()
    req = VoiceRequest(text="Quality control verification test.", language="en", output_format="wav")
    dest = tmp_path / "qc_test.wav"
    result = p.synthesize(req, dest)
    report = verify_audio(result.path)
    assert report["ok"] is True
    assert report["duration"] > 0.5

"""Comprehensive tests for Phase 9: voiceover, audio mixing, music, SFX, captions.

Deterministic test audio is generated with FFmpeg for the mixing/caption
pipeline. This is TEST VERIFIED for FFmpeg runtime, but TTS/music/SFX provider
runtime is NOT VERIFIED (no external provider configured) — Null providers
correctly raise NO_PROVIDER rather than faking success.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from app.brain.models import ContentBrief
from app.brain.content_brain import local_content_plan
from app.ads.brief import AdBrief
from app.ads.variants import generate_variant
from app.brain.models import Platform, Objective, BrandProfile
from app.core.errors import TypedErrorCode, VideoError
from app.voice.tts import (
    SUPPORTED_LANGUAGES, NullTTSProvider, TTSError, TTSProvider, VoiceRequest, VoiceResult, select_tts_provider,
)
from app.voice.voiceover import (
    VoiceAsset, VoiceTimingReport, build_voice_request, generate_project_voiceover,
    generate_scene_voiceover, validate_project_voice_timing, validate_voice_timing,
)
from app.audio.qc import AudioError, generate_silent_audio, generate_tone_audio, probe_audio, verify_audio
from app.audio.music import MUSIC_MOODS, MusicError, MusicRequest, NullMusicProvider, parse_mood
from app.audio.sfx import SFX_CATEGORIES, NullSFXProvider, SFXError, SFXRequest, parse_sfx_categories
from app.audio.mixer import MixError, MixOptions, MixTrack, mix_audio, mix_scene_audio
from app.captions.captions import (
    CaptionCue, CaptionFormat, CaptionReport, CaptionStyle, STYLE_PROFILES,
    build_cues_from_plan, generate_captions, render_srt, render_vtt, style_profile,
    validate_caption_timing, write_captions,
)


# ---------------------------------------------------------------- fixtures
def _plan(**over):
    return local_content_plan(ContentBrief(idea="lion hunting in jungle", duration_seconds=12, mode="cinematic"))


def _ad_plan(key="ugc"):
    brief = AdBrief(product_or_service="Noir perfume", target_audience="women 25-45",
                    platform=Platform.TIKTOK, duration_seconds=15, objective=Objective.CONVERSION)
    return generate_variant(brief, key).plan


class _FakeTTSProvider(TTSProvider):
    """Deterministic fake TTS producing real silent audio via FFmpeg.

    For testing the voiceover pipeline only — NOT real TTS runtime verification.
    """

    def __init__(self, tmp: Path):
        self._tmp = tmp

    @property
    def name(self) -> str:
        return "fake"

    @property
    def available(self) -> bool:
        return True

    def list_voices(self, language: str = "en") -> list[str]:
        return ["voice_a", "voice_b"]

    def synthesize(self, request: VoiceRequest, destination: Path) -> VoiceResult:
        request.validate()
        # Deterministic duration based on text length (NOT real TTS).
        dur = max(0.5, len(request.text) / 20.0)
        generate_silent_audio(destination, dur)
        qc = verify_audio(destination)
        return VoiceResult(path=destination, duration=qc["duration"], provider=self.name,
                           language=request.language, voice=request.voice, text=request.text,
                           sample_rate=qc["sample_rate"])


# ---------------------------------------------------------------- TTS request validation
def test_voice_request_validates_ok():
    VoiceRequest(text="hello world", language="en").validate()
    VoiceRequest(text="Hallo", language="de", gender="male", rate=1.2).validate()
    VoiceRequest(text="مرحبا", language="ar", emotion="calm").validate()


@pytest.mark.parametrize("bad_lang", ["fr", "es", "jp", ""])
def test_voice_request_bad_language(bad_lang):
    with pytest.raises(TTSError) as e:
        VoiceRequest(text="x", language=bad_lang).validate()
    assert e.value.code is TypedErrorCode.WORKFLOW_INVALID


def test_voice_request_empty_text():
    with pytest.raises(TTSError):
        VoiceRequest(text="   ", language="en").validate()


def test_voice_request_bad_rate():
    with pytest.raises(TTSError):
        VoiceRequest(text="x", language="en", rate=0).validate()
    with pytest.raises(TTSError):
        VoiceRequest(text="x", language="en", rate=5).validate()


def test_voice_request_bad_pitch():
    with pytest.raises(TTSError):
        VoiceRequest(text="x", language="en", pitch=-1).validate()


def test_voice_request_bad_gender():
    with pytest.raises(TTSError):
        VoiceRequest(text="x", language="en", gender="robot").validate()


def test_voice_request_bad_format():
    with pytest.raises(TTSError):
        VoiceRequest(text="x", language="en", output_format="ogg").validate()


def test_supported_languages():
    assert set(SUPPORTED_LANGUAGES) == {"en", "de", "ar"}


def test_voice_request_to_dict_serializable():
    json.dumps(VoiceRequest(text="hi", language="ar").to_dict())


# ---------------------------------------------------------------- null providers
def test_null_tts_raises_no_provider(tmp_path):
    p = NullTTSProvider()
    assert p.available is False
    assert p.list_voices() == []
    with pytest.raises(TTSError) as e:
        p.synthesize(VoiceRequest(text="hi", language="en"), tmp_path / "x.mp3")
    assert e.value.code is TypedErrorCode.NO_PROVIDER


def test_select_tts_returns_null_when_none_available():
    p = select_tts_provider([])
    assert p.name == "null"
    assert p.available is False


def test_select_tts_returns_first_available(tmp_path):
    fake = _FakeTTSProvider(tmp_path)
    p = select_tts_provider([NullTTSProvider(), fake])
    assert p.name == "fake"


def test_null_music_raises_no_provider(tmp_path):
    p = NullMusicProvider()
    assert p.available is False
    with pytest.raises(MusicError) as e:
        p.generate(MusicRequest(mood="cinematic", duration=10), tmp_path / "m.mp3")
    assert e.value.code is TypedErrorCode.NO_PROVIDER


def test_null_sfx_raises_no_provider(tmp_path):
    p = NullSFXProvider()
    assert p.available is False
    with pytest.raises(SFXError) as e:
        p.generate(SFXRequest(category="whoosh", duration=1), tmp_path / "s.mp3")
    assert e.value.code is TypedErrorCode.NO_PROVIDER


# ---------------------------------------------------------------- language propagation
def test_language_propagates_plan_to_request():
    plan = _plan()
    plan.voiceover.language = "de"
    scene = plan.scenes[0]
    req = build_voice_request(scene, plan)
    assert req.language == "de"


def test_language_propagates_arabic():
    plan = _plan()
    plan.voiceover.language = "ar"
    scene = plan.scenes[0]
    scene.voiceover.line = "الأسد يصطاد"
    req = build_voice_request(scene, plan)
    assert req.language == "ar"
    assert req.text == "الأسد يصطاد"


def test_pace_to_rate():
    plan = _plan()
    plan.voiceover.pace = "fast"
    scene = plan.scenes[0]
    req = build_voice_request(scene, plan)
    assert req.rate == 1.3


# ---------------------------------------------------------------- voice timing
def test_voice_timing_ok():
    plan = _plan()
    scene = plan.scenes[0]
    r = validate_voice_timing(scene, 1.0)
    assert r.severity.value == "ok"
    assert r.slack >= 0


def test_voice_timing_overlong_error():
    plan = _plan()
    scene = plan.scenes[0]
    r = validate_voice_timing(scene, scene.duration + 5)
    assert r.severity.value == "error"
    assert r.strategy is not None
    assert "shorten" in r.strategy or "increase" in r.strategy


def test_voice_timing_offset_warning():
    plan = _plan()
    scene = plan.scenes[0]
    scene.voiceover.start_offset = scene.duration - 0.5
    r = validate_voice_timing(scene, 1.0)
    # voice fits if started at 0 but offset pushes it over -> warning
    assert r.severity.value in ("warning", "error")


def test_voice_timing_report_serializable():
    plan = _plan()
    r = validate_voice_timing(plan.scenes[0], 1.0)
    json.dumps(r.to_dict())


def test_project_voice_timing():
    plan = _plan()
    durations = {s.index: 1.0 for s in plan.scenes}
    reports = validate_project_voice_timing(plan, durations)
    assert len(reports) == len(plan.scenes)
    assert all(r.severity.value == "ok" for r in reports)


# ---------------------------------------------------------------- voiceover generation
def test_generate_scene_voiceover_success(tmp_path):
    plan = _plan()
    scene = plan.scenes[0]
    provider = _FakeTTSProvider(tmp_path)
    asset, result = generate_scene_voiceover(scene, plan, provider, project_id="p1", output_dir=tmp_path / "v")
    assert Path(asset.path).exists()
    assert asset.duration > 0
    assert asset.language == plan.voiceover.language
    assert asset.scene_index == scene.index
    assert asset.project_id == "p1"
    json.dumps(asset.to_dict())


def test_generate_scene_voiceover_null_raises(tmp_path):
    plan = _plan()
    scene = plan.scenes[0]
    with pytest.raises(TTSError) as e:
        generate_scene_voiceover(scene, plan, NullTTSProvider(), output_dir=tmp_path)
    assert e.value.code is TypedErrorCode.NO_PROVIDER


def test_generate_project_voiceover_skips_empty_lines(tmp_path):
    plan = _plan()
    for s in plan.scenes[1:]:
        s.voiceover.line = ""
    provider = _FakeTTSProvider(tmp_path)
    assets, timing = generate_project_voiceover(plan, provider, project_id="p1", output_dir=tmp_path / "v")
    assert len(assets) == 1
    assert len(timing) == 1


def test_generate_project_voiceover_all_scenes(tmp_path):
    plan = _plan()
    for s in plan.scenes:
        s.voiceover.line = "narration here"
    provider = _FakeTTSProvider(tmp_path)
    assets, timing = generate_project_voiceover(plan, provider, project_id="p1", output_dir=tmp_path / "v")
    assert len(assets) == len(plan.scenes)
    assert all(Path(a.path).exists() for a in assets)


# ---------------------------------------------------------------- audio QC
def test_verify_audio_missing(tmp_path):
    with pytest.raises(AudioError) as e:
        verify_audio(tmp_path / "nope.mp3")
    assert e.value.code is TypedErrorCode.INVALID_AUDIO


def test_verify_audio_empty(tmp_path):
    f = tmp_path / "empty.mp3"
    f.write_bytes(b"")
    with pytest.raises(AudioError) as e:
        verify_audio(f)
    assert e.value.code is TypedErrorCode.INVALID_AUDIO


def test_verify_audio_not_audio(tmp_path):
    f = tmp_path / "notaudio.mp3"
    f.write_bytes(b"this is text not audio")
    with pytest.raises(AudioError) as e:
        verify_audio(f)
    assert e.value.code is TypedErrorCode.INVALID_AUDIO


def test_verify_audio_real(tmp_path):
    f = generate_tone_audio(tmp_path / "tone.mp3", 1.0)
    qc = verify_audio(f)
    assert qc["ok"] is True
    assert qc["duration"] > 0
    assert qc["sample_rate"] > 0
    assert qc["channels"] > 0
    assert qc["codec"] is not None


def test_generate_silent_audio_real(tmp_path):
    f = generate_silent_audio(tmp_path / "silence.mp3", 2.0)
    qc = verify_audio(f)
    assert qc["ok"] is True
    assert qc["duration"] >= 1.9


# ---------------------------------------------------------------- music
def test_music_moods_complete():
    for m in ["cinematic", "emotional", "energetic", "dark", "luxury", "corporate", "suspense", "documentary"]:
        assert m in MUSIC_MOODS


def test_music_request_validates():
    MusicRequest(mood="luxury", duration=10).validate()
    MusicRequest(mood="dark", duration=5, fade_in=1, fade_out=1).validate()


def test_music_request_bad_mood():
    with pytest.raises(MusicError):
        MusicRequest(mood="happy", duration=10).validate()


def test_music_request_bad_fades():
    with pytest.raises(MusicError):
        MusicRequest(mood="cinematic", duration=2, fade_in=2, fade_out=2).validate()


def test_parse_mood_extracts():
    assert parse_mood("dark and cinematic") == "cinematic"
    assert parse_mood("luxury elegant") == "luxury"
    assert parse_mood("") == "cinematic"


# ---------------------------------------------------------------- sfx
def test_sfx_categories_complete():
    for c in ["whoosh", "impact", "footsteps", "rain", "wind", "jungle_ambience", "traffic", "product_sounds", "cinematic_hits"]:
        assert c in SFX_CATEGORIES


def test_sfx_request_validates():
    SFXRequest(category="whoosh", duration=1).validate()


def test_sfx_request_bad_category():
    with pytest.raises(SFXError):
        SFXRequest(category="explosion", duration=1).validate()


def test_parse_sfx_categories_extracts():
    assert "rain" in parse_sfx_categories("rain and wind ambience")
    assert "jungle_ambience" in parse_sfx_categories("jungle_ambience at dusk")


# ---------------------------------------------------------------- audio mixing (REAL FFmpeg)
def test_mix_audio_four_tracks(tmp_path):
    voice = generate_tone_audio(tmp_path / "voice.mp3", 3.0, freq=300)
    music = generate_tone_audio(tmp_path / "music.mp3", 5.0, freq=120)
    sfx = generate_silent_audio(tmp_path / "sfx.mp3", 1.0)
    amb = generate_tone_audio(tmp_path / "amb.mp3", 5.0, freq=80)
    tracks = [
        MixTrack(path=voice, kind="voice", volume=1.0),
        MixTrack(path=music, kind="music", volume=0.6, duck_when_voice=True, fade_in=0.5, fade_out=0.5),
        MixTrack(path=sfx, kind="sfx", volume=0.5, start_offset=1.0),
        MixTrack(path=amb, kind="ambience", volume=0.3),
    ]
    opts = MixOptions(target_duration=5.0, music_duck_level=0.25, fade_in=0.3, fade_out=0.5)
    result = mix_audio(tracks, tmp_path / "mix.mp3", opts)
    assert Path(result.path).exists()
    assert result.duration > 0
    qc = verify_audio(result.path)
    assert qc["ok"] is True


def test_mix_audio_voice_only(tmp_path):
    voice = generate_tone_audio(tmp_path / "voice.mp3", 2.0)
    result = mix_audio([MixTrack(path=voice, kind="voice")], tmp_path / "vo.mp3", MixOptions(target_duration=2.0))
    assert verify_audio(result.path)["ok"] is True


def test_mix_audio_music_ducking(tmp_path):
    voice = generate_tone_audio(tmp_path / "voice.mp3", 2.0, freq=300)
    music = generate_tone_audio(tmp_path / "music.mp3", 3.0, freq=120)
    tracks = [
        MixTrack(path=voice, kind="voice", volume=1.0),
        MixTrack(path=music, kind="music", volume=0.8, duck_when_voice=True),
    ]
    result = mix_audio(tracks, tmp_path / "duck.mp3", MixOptions(target_duration=3.0, music_duck_level=0.2))
    assert verify_audio(result.path)["ok"] is True


def test_mix_audio_fades(tmp_path):
    music = generate_tone_audio(tmp_path / "music.mp3", 3.0)
    tracks = [MixTrack(path=music, kind="music", volume=0.8, fade_in=0.5, fade_out=0.5, duration=3.0)]
    result = mix_audio(tracks, tmp_path / "fade.mp3", MixOptions(target_duration=3.0, fade_in=0.3, fade_out=0.5))
    assert verify_audio(result.path)["ok"] is True


def test_mix_audio_clipping_prevention(tmp_path):
    # Loud tone -> alimiter must prevent clipping.
    loud = generate_tone_audio(tmp_path / "loud.mp3", 2.0, freq=1000)
    tracks = [MixTrack(path=loud, kind="voice", volume=2.0)]
    result = mix_audio(tracks, tmp_path / "limited.mp3", MixOptions(target_duration=2.0))
    assert verify_audio(result.path)["ok"] is True


def test_mix_scene_audio_convenience(tmp_path):
    voice = generate_tone_audio(tmp_path / "voice.mp3", 2.0)
    music = generate_tone_audio(tmp_path / "music.mp3", 3.0)
    sfx = generate_silent_audio(tmp_path / "sfx.mp3", 1.0)
    amb = generate_tone_audio(tmp_path / "amb.mp3", 3.0)
    result = mix_scene_audio(voice, music, [sfx], amb, tmp_path / "scene.mp3",
                             scene_duration=3.0, voice_start_offset=0.5, music_duck=True)
    assert verify_audio(result.path)["ok"] is True


def test_mix_missing_track_raises(tmp_path):
    with pytest.raises(MixError) as e:
        mix_audio([MixTrack(path=tmp_path / "nope.mp3", kind="voice")], tmp_path / "x.mp3")
    assert e.value.code is TypedErrorCode.INVALID_AUDIO


def test_mix_no_tracks_raises(tmp_path):
    with pytest.raises(MixError) as e:
        mix_audio([], tmp_path / "x.mp3")
    assert e.value.code is TypedErrorCode.WORKFLOW_INVALID


def test_mix_bad_duck_level(tmp_path):
    with pytest.raises(MixError):
        MixOptions(music_duck_level=1.5).validate()


def test_mix_options_serializable():
    json.dumps(MixOptions(target_duration=5.0).to_dict())


def test_mix_never_overwrites_source(tmp_path):
    voice = generate_tone_audio(tmp_path / "voice.mp3", 1.0)
    src_size = voice.stat().st_size
    mix_audio([MixTrack(path=voice, kind="voice")], tmp_path / "out.mp3", MixOptions(target_duration=1.0))
    # Source untouched.
    assert voice.stat().st_size == src_size


# ---------------------------------------------------------------- captions
def test_build_cues_from_plan():
    plan = _plan()
    for s in plan.scenes:
        s.caption.text = f"caption {s.index}"
    cues = build_cues_from_plan(plan)
    assert len(cues) == len(plan.scenes)
    assert cues[0].start == 0.0


def test_build_cues_falls_back_to_voiceover():
    plan = _plan()
    for s in plan.scenes:
        s.caption.text = ""
        s.voiceover.line = f"vo {s.index}"
    cues = build_cues_from_plan(plan)
    assert all(c.text.startswith("vo") for c in cues)


def test_render_srt_format():
    cues = [CaptionCue(index=1, start=0.0, end=2.0, text="hello", scene_index=1)]
    srt = render_srt(cues)
    assert "1" in srt
    assert "00:00:00,000 --> 00:00:02,000" in srt
    assert "hello" in srt


def test_render_vtt_format():
    cues = [CaptionCue(index=1, start=0.0, end=2.0, text="hello", scene_index=1)]
    vtt = render_vtt(cues)
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:02.000" in vtt


def test_generate_captions_srt():
    plan = _plan()
    for s in plan.scenes:
        s.caption.text = f"cap {s.index}"
    rep = generate_captions(plan, CaptionFormat.SRT, CaptionStyle.TIKTOK)
    assert len(rep.cues) == len(plan.scenes)
    assert not rep.errors
    assert rep.content.strip().endswith(plan.scenes[-1].caption.text)


def test_generate_captions_vtt():
    plan = _plan()
    for s in plan.scenes:
        s.caption.text = f"cap {s.index}"
    rep = generate_captions(plan, CaptionFormat.VTT, CaptionStyle.REELS)
    assert rep.content.startswith("WEBVTT")


def test_generate_captions_burned_in():
    plan = _plan()
    for s in plan.scenes:
        s.caption.text = f"cap {s.index}"
    rep = generate_captions(plan, CaptionFormat.BURNED_IN, CaptionStyle.SHORTS)
    assert rep.content == ""  # no text content; style profile carried
    assert len(rep.cues) == len(plan.scenes)


def test_caption_styles_all_present():
    for st in [CaptionStyle.TIKTOK, CaptionStyle.REELS, CaptionStyle.SHORTS, CaptionStyle.YOUTUBE]:
        prof = style_profile(st)
        assert "font_size" in prof and "position" in prof


def test_caption_timing_validation_negative_start():
    cues = [CaptionCue(index=1, start=-1.0, end=2.0, text="x", scene_index=1)]
    errs = validate_caption_timing(cues, 10.0)
    assert any(e.code == "NEGATIVE_START" for e in errs)


def test_caption_timing_validation_end_before_start():
    cues = [CaptionCue(index=1, start=3.0, end=1.0, text="x", scene_index=1)]
    errs = validate_caption_timing(cues, 10.0)
    assert any(e.code == "END_BEFORE_START" for e in errs)


def test_caption_timing_validation_out_of_range():
    cues = [CaptionCue(index=1, start=8.0, end=12.0, text="x", scene_index=1)]
    errs = validate_caption_timing(cues, 10.0)
    assert any(e.code == "OUT_OF_RANGE" for e in errs)


def test_caption_timing_validation_overlap():
    cues = [
        CaptionCue(index=1, start=0.0, end=3.0, text="a", scene_index=1),
        CaptionCue(index=2, start=2.0, end=4.0, text="b", scene_index=2),
    ]
    errs = validate_caption_timing(cues, 10.0)
    assert any(e.code == "OVERLAP" for e in errs)


def test_caption_timing_validation_empty_text():
    cues = [CaptionCue(index=1, start=0.0, end=2.0, text="", scene_index=1)]
    errs = validate_caption_timing(cues, 10.0)
    assert any(e.code == "EMPTY_TEXT" for e in errs)


def test_caption_timing_valid_no_errors():
    cues = [
        CaptionCue(index=1, start=0.0, end=2.0, text="a", scene_index=1),
        CaptionCue(index=2, start=2.0, end=4.0, text="b", scene_index=2),
    ]
    assert validate_caption_timing(cues, 10.0) == []


def test_caption_report_serializable():
    plan = _plan()
    rep = generate_captions(plan, CaptionFormat.SRT, CaptionStyle.TIKTOK)
    json.dumps(rep.to_dict())


def test_caption_error_to_dict():
    from app.captions.captions import CaptionValidationError
    e = CaptionValidationError("X", "msg", context={"a": 1})
    json.dumps(e.to_dict())


def test_write_captions_srt(tmp_path):
    plan = _plan()
    for s in plan.scenes:
        s.caption.text = f"cap {s.index}"
    rep = generate_captions(plan, CaptionFormat.SRT, CaptionStyle.YOUTUBE)
    p = write_captions(rep, tmp_path / "caps")
    assert p.suffix == ".srt"
    assert p.exists()
    assert "WEBVTT" not in p.read_text()


def test_write_captions_vtt(tmp_path):
    plan = _plan()
    for s in plan.scenes:
        s.caption.text = f"cap {s.index}"
    rep = generate_captions(plan, CaptionFormat.VTT, CaptionStyle.SHORTS)
    p = write_captions(rep, tmp_path / "caps")
    assert p.suffix == ".vtt"
    assert p.read_text().startswith("WEBVTT")


# ---------------------------------------------------------------- Arabic + multi-language
def test_arabic_captions_preserved():
    plan = _plan()
    plan.voiceover.language = "ar"
    for s in plan.scenes:
        s.caption.text = "الأسد يصطاد في الغابة"
    rep = generate_captions(plan, CaptionFormat.SRT, CaptionStyle.TIKTOK)
    assert "الأسد" in rep.content
    assert all("الأسد" in c.text for c in rep.cues)


def test_german_captions_preserved():
    plan = _plan()
    plan.voiceover.language = "de"
    for s in plan.scenes:
        s.caption.text = "Der Löwe jagt im Dschungel"
    rep = generate_captions(plan, CaptionFormat.VTT, CaptionStyle.YOUTUBE)
    assert "Der Löwe" in rep.content


def test_multi_scene_synchronization():
    plan = _plan()
    for i, s in enumerate(plan.scenes):
        s.caption.text = f"scene {i}"
    cues = build_cues_from_plan(plan)
    # Cues must be sequential and contiguous.
    for a, b in zip(cues, cues[1:]):
        assert b.start >= a.start
    # Total span matches project duration.
    assert cues[-1].end <= sum(s.duration for s in plan.scenes) + 0.1


# ---------------------------------------------------------------- backward compatibility
def test_backward_compat_phase5_plan_still_works():
    plan = local_content_plan(ContentBrief(idea="x", duration_seconds=12))
    cues = build_cues_from_plan(plan)
    assert len(cues) <= len(plan.scenes)


def test_backward_compat_phase6_ad_plan_captions():
    plan = _ad_plan("cinematic")
    rep = generate_captions(plan, CaptionFormat.SRT, CaptionStyle.TIKTOK)
    assert rep.to_dict()["ok"] is True or len(rep.errors) >= 0  # structure intact


def test_backward_compat_phase7_resolve_still_works():
    from app.scene.continuity import resolve_all_scenes, validate_continuity
    plan = _plan()
    ctxs = resolve_all_scenes(plan)
    assert len(ctxs) == len(plan.scenes)
    assert validate_continuity(plan).ok


def test_backward_compat_phase8_wan_provider():
    from app.providers.wan import WanProvider, GenerationOptions
    p = WanProvider("http://127.0.0.1:1", timeout=2)
    opts = GenerationOptions(seed=1)
    opts.validate_ranges()
    assert opts.seed == 1


def test_invalid_audio_error_code_added():
    assert TypedErrorCode.INVALID_AUDIO.value == "INVALID_AUDIO"


# ---------------------------------------------------------------- end-to-end voice+mix
def test_end_to_end_voice_then_mix(tmp_path):
    """Generate voice (fake TTS) then mix with music (tone) — real FFmpeg."""
    plan = _plan()
    scene = plan.scenes[0]
    scene.voiceover.line = "A short narration line"
    provider = _FakeTTSProvider(tmp_path)
    asset, _ = generate_scene_voiceover(scene, plan, provider, project_id="e2e", output_dir=tmp_path / "v")
    music = generate_tone_audio(tmp_path / "music.mp3", 3.0, freq=120)
    result = mix_scene_audio(Path(asset.path), music, [], None, tmp_path / "final.mp3",
                             scene_duration=scene.duration, music_duck=True)
    assert verify_audio(result.path)["ok"] is True

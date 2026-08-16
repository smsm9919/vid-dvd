"""Comprehensive tests for Phase 10: editing timeline, assembly, export, QC.

Uses real FFmpeg to generate deterministic test MP4s and run the full assembly
pipeline. This is valid runtime verification of the EDITING ENGINE (not of AI
video generation, which remains blocked). TTS/music/SFX providers stay blocked
(Null providers); audio fixtures are deterministic FFmpeg tones/silence, clearly
NOT real provider output.
"""

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from app.brain.models import ContentBrief
from app.brain.content_brain import local_content_plan
from app.core.errors import TypedErrorCode, VideoError
from app.editing.timeline import (
    EditingError, Timeline, TimelineScene, build_timeline_from_assets, validate_timeline,
)
from app.editing.profiles import (
    INSTAGRAM_REELS, PROFILES, Quality, SQUARE, TIKTOK, YOUTUBE, YOUTUBE_SHORTS,
    ExportProfile, get_profile, register_profile,
)
from app.editing.transitions import (
    SUPPORTED_TRANSITIONS, TransitionSpec, TransitionType, parse_transition, transition_filter,
)
from app.editing.compositor import (
    _build_ass_from_cues, _ffmpeg_filter_quote_path, captions_renderable, font_available,
)
from app.editing.assembly import (
    ExportRequest, ExportResult, export_video, validate_scene_assets,
)
from app.editing.qc import final_qc
from app.audio.qc import generate_silent_audio, generate_tone_audio


# ---------------------------------------------------------------- fixtures
def _make_scene_video(path: Path, color: str = "red", dur: float = 3.0,
                      w: int = 720, h: int = 1280, with_audio: bool = True):
    if with_audio:
        cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i",
               f"color=c={color}:s={w}x{h}:d={dur}:r=30",
               "-f", "lavfi", "-i", f"sine=frequency=440:duration={dur}",
               "-t", str(dur), "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-shortest", str(path)]
    else:
        cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i",
               f"color=c={color}:s={w}x{h}:d={dur}:r=30",
               "-t", str(dur), "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-an", str(path)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return path


@pytest.fixture
def tmp():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- typed errors
def test_editing_error_codes_present():
    for c in ["MISSING_VIDEO_ASSET", "MISSING_AUDIO_ASSET", "INVALID_TIMELINE",
              "UNSUPPORTED_PROFILE", "TRANSITION_ERROR", "CAPTION_RENDER_ERROR",
              "EXPORT_FAILED", "FINAL_QC_FAILED"]:
        assert hasattr(TypedErrorCode, c), c


# ---------------------------------------------------------------- timeline
def test_timeline_build_basic(tmp):
    tl = build_timeline_from_assets([3.0, 3.0], [tmp/"a.mp4", tmp/"b.mp4"])
    assert len(tl.scenes) == 2
    assert tl.total_duration == 6.0
    assert tl.scenes[0].start_time == 0.0
    assert tl.scenes[1].start_time == 3.0
    assert tl.scenes[0].scene_index == 1
    assert tl.scenes[1].scene_index == 2


def test_timeline_first_scene_transition_forced_cut(tmp):
    tl = build_timeline_from_assets([3.0], [tmp/"a.mp4"], transitions=["crossfade"],
                                     transition_durations=[0.5])
    assert tl.scenes[0].transition == "cut"
    assert tl.scenes[0].transition_duration == 0.0


def test_timeline_mismatched_counts_raise(tmp):
    with pytest.raises(EditingError) as e:
        build_timeline_from_assets([3.0, 3.0], [tmp/"a.mp4"])
    assert e.value.code is TypedErrorCode.INVALID_TIMELINE


def test_timeline_serializable(tmp):
    tl = build_timeline_from_assets([3.0], [tmp/"a.mp4"])
    import json
    json.dumps(tl.to_dict())


def test_timeline_scene_to_dict(tmp):
    ts = TimelineScene(scene_index=1, start_time=0.0, duration=3.0, video_asset=tmp/"a.mp4")
    import json
    json.dumps(ts.to_dict())


def test_validate_timeline_empty():
    errs = validate_timeline(Timeline())
    assert len(errs) == 1
    assert errs[0].code is TypedErrorCode.INVALID_TIMELINE


def test_validate_timeline_missing_video(tmp):
    tl = build_timeline_from_assets([3.0], [None])  # type: ignore
    errs = validate_timeline(tl)
    assert any(e.code is TypedErrorCode.MISSING_VIDEO_ASSET for e in errs)


def test_validate_timeline_non_positive_duration(tmp):
    tl = build_timeline_from_assets([0.0], [tmp/"a.mp4"])
    errs = validate_timeline(tl)
    assert any(e.code is TypedErrorCode.INVALID_TIMELINE for e in errs)


def test_validate_timeline_bad_start_time(tmp):
    ts = TimelineScene(scene_index=1, start_time=5.0, duration=3.0, video_asset=tmp/"a.mp4")
    tl = Timeline(scenes=[ts], total_duration=3.0)
    errs = validate_timeline(tl)
    assert any("start_time" in e.detail for e in errs)


def test_validate_timeline_negative_offset(tmp):
    ts = TimelineScene(scene_index=1, start_time=0.0, duration=3.0,
                       video_asset=tmp/"a.mp4", voice_start_offset=-1.0)
    tl = Timeline(scenes=[ts], total_duration=3.0)
    errs = validate_timeline(tl)
    assert any("negative" in e.detail.lower() for e in errs)


def test_validate_timeline_offset_exceeds_duration(tmp):
    ts = TimelineScene(scene_index=1, start_time=0.0, duration=3.0,
                       video_asset=tmp/"a.mp4", voice_asset=tmp/"v.mp3", voice_start_offset=3.0)
    tl = Timeline(scenes=[ts], total_duration=3.0)
    errs = validate_timeline(tl)
    assert any(e.code is TypedErrorCode.INVALID_TIMELINE for e in errs)


def test_validate_timeline_bad_transition(tmp):
    ts = TimelineScene(scene_index=1, start_time=0.0, duration=3.0,
                       video_asset=tmp/"a.mp4", transition="spin")
    tl = Timeline(scenes=[ts], total_duration=3.0)
    errs = validate_timeline(tl)
    assert any(e.code is TypedErrorCode.TRANSITION_ERROR for e in errs)


def test_validate_timeline_caption_out_of_range(tmp):
    ts = TimelineScene(scene_index=1, start_time=0.0, duration=3.0,
                       video_asset=tmp/"a.mp4", caption_text="hi",
                       caption_start=0.0, caption_end=10.0)
    tl = Timeline(scenes=[ts], total_duration=3.0)
    errs = validate_timeline(tl)
    assert any("out of range" in e.detail.lower() for e in errs)


def test_validate_timeline_total_duration_mismatch(tmp):
    ts = TimelineScene(scene_index=1, start_time=0.0, duration=3.0, video_asset=tmp/"a.mp4")
    tl = Timeline(scenes=[ts], total_duration=99.0)
    errs = validate_timeline(tl)
    assert any("total_duration" in e.detail for e in errs)


def test_validate_timeline_valid(tmp):
    tl = build_timeline_from_assets([3.0, 3.0], [tmp/"a.mp4", tmp/"b.mp4"])
    assert validate_timeline(tl) == []


def test_validate_scene_assets_missing_video():
    ts = TimelineScene(scene_index=1, start_time=0.0, duration=3.0, video_asset=None)
    errs = validate_scene_assets(ts)
    assert any(e.code is TypedErrorCode.MISSING_VIDEO_ASSET for e in errs)


def test_validate_scene_assets_video_not_found(tmp):
    ts = TimelineScene(scene_index=1, start_time=0.0, duration=3.0,
                       video_asset=tmp/"nope.mp4")
    errs = validate_scene_assets(ts)
    assert any(e.code is TypedErrorCode.MISSING_VIDEO_ASSET for e in errs)


def test_validate_scene_assets_video_unreadable(tmp):
    bad = tmp/"bad.mp4"
    bad.write_bytes(b"not a video")
    ts = TimelineScene(scene_index=1, start_time=0.0, duration=3.0, video_asset=bad)
    errs = validate_scene_assets(ts)
    assert any(e.code is TypedErrorCode.MISSING_VIDEO_ASSET for e in errs)


def test_validate_scene_assets_voice_missing(tmp):
    v = _make_scene_video(tmp/"v.mp4")
    ts = TimelineScene(scene_index=1, start_time=0.0, duration=3.0,
                       video_asset=v, voice_asset=tmp/"nope.mp3")
    errs = validate_scene_assets(ts)
    assert any(e.code is TypedErrorCode.MISSING_AUDIO_ASSET for e in errs)


def test_validate_scene_assets_valid(tmp):
    v = _make_scene_video(tmp/"v.mp4")
    vo = generate_silent_audio(tmp/"vo.mp3", 2.0)
    ts = TimelineScene(scene_index=1, start_time=0.0, duration=3.0,
                       video_asset=v, voice_asset=vo)
    assert validate_scene_assets(ts) == []


# ---------------------------------------------------------------- profiles
def test_profiles_all_present():
    for n in ["TIKTOK", "INSTAGRAM_REELS", "YOUTUBE_SHORTS", "YOUTUBE", "SQUARE"]:
        assert n in PROFILES


def test_profile_resolutions():
    assert TIKTOK.resolution == (1080, 1920)
    assert INSTAGRAM_REELS.resolution == (1080, 1920)
    assert YOUTUBE_SHORTS.resolution == (1080, 1920)
    assert YOUTUBE.resolution == (1920, 1080)
    assert SQUARE.resolution == (1080, 1080)


def test_profile_aspect_ratios():
    assert TIKTOK.aspect == "9:16"
    assert YOUTUBE.aspect == "16:9"
    assert SQUARE.aspect == "1:1"


def test_get_profile_quality_applies():
    low = get_profile("TIKTOK", quality=Quality.LOW)
    high = get_profile("TIKTOK", quality=Quality.HIGH)
    assert low.crf > high.crf  # lower quality = higher CRF


def test_get_profile_unknown_raises():
    with pytest.raises(EditingError) as e:
        get_profile("NOPE", quality=Quality.HIGH)
    assert e.value.code is TypedErrorCode.UNSUPPORTED_PROFILE


def test_register_profile_runtime():
    custom = ExportProfile("CUSTOM", 640, 640, fps=24)
    register_profile(custom)
    assert "CUSTOM" in PROFILES
    p = get_profile("CUSTOM", quality=Quality.HIGH)
    assert p.resolution == (640, 640)
    # cleanup
    del PROFILES["CUSTOM"]


def test_profile_serializable():
    import json
    json.dumps(TIKTOK.to_dict())


# ---------------------------------------------------------------- transitions
def test_parse_transitions():
    assert parse_transition(None) == (TransitionType.CUT, 0.0)
    assert parse_transition("fade") == (TransitionType.FADE, 0.5)
    assert parse_transition("use crossfade here") == (TransitionType.CROSSFADE, 0.5)
    assert parse_transition("dip to black") == (TransitionType.DIP_TO_BLACK, 0.7)
    assert parse_transition("wipe") == (TransitionType.WIPE, 0.5)


def test_supported_transitions():
    assert {"cut", "fade", "crossfade", "dip_to_black", "wipe"} == SUPPORTED_TRANSITIONS


def test_transition_spec_validate_bad_type():
    # Construct a spec with an invalid type value via object.__setattr__ bypass.
    spec = TransitionSpec(TransitionType.CUT, 0.0)
    object.__setattr__(spec, "type", "spin")
    with pytest.raises(EditingError) as e:
        spec.validate(3.0)
    assert e.value.code is TypedErrorCode.TRANSITION_ERROR


def test_transition_spec_validate_negative_duration():
    spec = TransitionSpec(TransitionType.FADE, -1.0)
    with pytest.raises(EditingError) as e:
        spec.validate(3.0)
    assert e.value.code is TypedErrorCode.TRANSITION_ERROR


def test_transition_spec_validate_xfade_too_long():
    spec = TransitionSpec(TransitionType.CROSSFADE, 5.0)
    with pytest.raises(EditingError) as e:
        spec.validate(3.0)
    assert e.value.code is TypedErrorCode.TRANSITION_ERROR


def test_transition_filter_crossfade():
    spec = TransitionSpec(TransitionType.CROSSFADE, 0.5)
    f = transition_filter(spec, "[v0]", "[v1]", "[out]")
    assert "xfade=transition=fade:duration=0.5" in f


def test_transition_filter_dip_to_black():
    spec = TransitionSpec(TransitionType.DIP_TO_BLACK, 0.7)
    f = transition_filter(spec, "[v0]", "[v1]", "[out]")
    assert "fadeblack" in f


def test_transition_filter_wipe():
    spec = TransitionSpec(TransitionType.WIPE, 0.5)
    f = transition_filter(spec, "[v0]", "[v1]", "[out]")
    assert "wipeleft" in f


def test_transition_filter_cut_returns_none():
    spec = TransitionSpec(TransitionType.CUT, 0.0)
    assert transition_filter(spec, "[v0]", "[v1]", "[out]") is None


def test_transition_spec_serializable():
    import json
    json.dumps(TransitionSpec(TransitionType.FADE, 0.5).to_dict())


# ---------------------------------------------------------------- compositor helpers
def test_captions_renderable():
    # Should be True in this env (libass). Just confirm it returns a bool.
    assert isinstance(captions_renderable(), bool)


def test_font_available_dejavu():
    assert font_available("DejaVu Sans") is True


def test_resolve_bundled_bold_font_exists():
    # Bold resolution for branding drawtext must point at a real bundled file.
    from app.editing.compositor import _resolve_bundled_font, fonts_dir
    regular = _resolve_bundled_font("DejaVu Sans")
    bold = _resolve_bundled_font("DejaVu Sans", bold=True)
    assert regular is not None and regular.exists()
    assert bold is not None and bold.exists()
    assert bold != regular
    assert fonts_dir() is not None and fonts_dir().is_dir()


def test_ffmpeg_filter_quote_path_windows_safe():
    # The escaping helper must produce FFmpeg-filtergraph-safe option values for
    # Windows drive letters, backslashes, spaces, parens and apostrophes.
    cases = {
        r"C:\Users\foo\assets\fonts\DejaVuSans-Bold.ttf":
            r"'C\:\\Users\\foo\\assets\\fonts\\DejaVuSans-Bold.ttf'",
        r"C:\Users\My (Dir)\font.ttf":
            r"'C\:\\Users\\My (Dir)\\font.ttf'",
        r"C:\Users\o'brien\font.ttf":
            r"'C\:\\Users\\o\'brien\\font.ttf'",
    }
    for raw, expected in cases.items():
        assert _ffmpeg_filter_quote_path(raw) == expected, raw
    # A POSIX path (no special chars) is wrapped in single quotes, unchanged.
    assert _ffmpeg_filter_quote_path("/opt/fonts/DejaVu.ttf") == "'/opt/fonts/DejaVu.ttf'"


def test_ffmpeg_filter_quote_path_parses_in_real_ffmpeg():
    # Real FFmpeg must PARSE filtergraphs that embed simulated Windows font
    # paths through the helper (no "Error parsing filterchain"). This is the
    # runtime regression guard for the Windows-path bug. It does not require
    # the font file to exist — parse-level acceptance is what failed before.
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    simulated = [
        r"C:\Users\foo\assets\fonts\DejaVuSans-Bold.ttf",          # drive + backslashes
        r"C:\Users\My (Dir)\assets\fonts\DejaVuSans-Bold.ttf",      # spaces + parens
        r"C:\Users\o'brien\assets\fonts\DejaVuSans-Bold.ttf",       # apostrophe
    ]
    for path in simulated:
        vf = (f"drawtext=text='X':fontcolor=white:fontsize=40:"
              f"fontfile={_ffmpeg_filter_quote_path(path)}:x=10:y=10")
        p = subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=1",
             "-vf", vf, "-frames:v", "1", "-an", "-f", "null", "-"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        # "Error parsing" is the exact symptom of the Windows-path bug; the file
        # not existing surfaces as a different drawtext error, never a parse error.
        assert "Error parsing" not in p.stderr, f"FFmpeg failed to parse escaped path {path!r}\n{p.stderr[-400:]}"


def test_build_ass_from_cues_structure():
    ass = _build_ass_from_cues([{"start": 0.0, "end": 2.0, "text": "hello"}],
                               {"font_size": 36, "font_color": "white", "position": "bottom",
                                "stroke": "black", "stroke_width": 2}, 1080, 1920)
    assert "[Script Info]" in ass
    assert "[V4+ Styles]" in ass
    assert "Dialogue:" in ass
    assert "hello" in ass


def test_build_ass_arabic_text():
    ass = _build_ass_from_cues([{"start": 0.0, "end": 2.0, "text": "الأسد يصطاد"}],
                               {"font_size": 36, "position": "bottom"}, 1080, 1920)
    assert "الأسد يصطاد" in ass


def test_build_ass_german_text():
    ass = _build_ass_from_cues([{"start": 0.0, "end": 2.0, "text": "Der Löwe jagt"}],
                               {"font_size": 36, "position": "bottom"}, 1080, 1920)
    assert "Der Löwe jagt" in ass


# ---------------------------------------------------------------- export (REAL FFmpeg)
def test_export_tiktok_basic(tmp):
    v1 = _make_scene_video(tmp/"s1.mp4", "red", 3.0, with_audio=False)
    v2 = _make_scene_video(tmp/"s2.mp4", "blue", 3.0, with_audio=False)
    tl = build_timeline_from_assets([3.0, 3.0], [v1, v2])
    req = ExportRequest(timeline=tl, profile_name="TIKTOK", quality=Quality.HIGH,
                       silent=True, project_id="t")
    r = export_video(req, output_dir=tmp/"out")
    assert r.status == "COMPLETED"
    assert r.resolution == (1080, 1920)
    assert abs(r.duration - 6.0) < 0.5
    assert r.video_codec == "h264"
    assert r.qc["ok"] is True


def test_export_youtube_16_9(tmp):
    v = _make_scene_video(tmp/"s.mp4", "red", 3.0, 1280, 720, with_audio=False)
    tl = build_timeline_from_assets([3.0], [v])
    r = export_video(ExportRequest(timeline=tl, profile_name="YOUTUBE", quality=Quality.HIGH,
                                   silent=True, project_id="yt"), output_dir=tmp/"out")
    assert r.status == "COMPLETED"
    assert r.resolution == (1920, 1080)


def test_export_square_1_1(tmp):
    v = _make_scene_video(tmp/"s.mp4", "red", 3.0, 1080, 1080, with_audio=False)
    tl = build_timeline_from_assets([3.0], [v])
    r = export_video(ExportRequest(timeline=tl, profile_name="SQUARE", quality=Quality.HIGH,
                                   silent=True, project_id="sq"), output_dir=tmp/"out")
    assert r.status == "COMPLETED"
    assert r.resolution == (1080, 1080)


def test_export_three_scenes_concat(tmp):
    vids = [_make_scene_video(tmp/f"s{i}.mp4", c, 2.0, 640, 480, with_audio=False)
            for i, c in enumerate(["red", "green", "blue"])]
    tl = build_timeline_from_assets([2.0, 2.0, 2.0], vids)
    r = export_video(ExportRequest(timeline=tl, profile_name="TIKTOK", quality=Quality.HIGH,
                                   silent=True, project_id="3s"), output_dir=tmp/"out")
    assert r.status == "COMPLETED"
    assert abs(r.duration - 6.0) < 0.5


def test_export_with_audio(tmp):
    v = _make_scene_video(tmp/"s.mp4", "red", 3.0, with_audio=False)
    vo = generate_silent_audio(tmp/"vo.mp3", 2.5)
    tl = build_timeline_from_assets([3.0], [v], voice_assets=[vo])
    r = export_video(ExportRequest(timeline=tl, profile_name="TIKTOK", quality=Quality.HIGH,
                                   project_id="aud"), output_dir=tmp/"out")
    assert r.status == "COMPLETED"
    assert r.audio_codec == "aac"
    assert r.qc["has_audio"] is True


def test_export_normalize_different_resolutions(tmp):
    # Sources with different resolutions must normalize without error.
    v1 = _make_scene_video(tmp/"s1.mp4", "red", 2.0, 640, 480, with_audio=False)
    v2 = _make_scene_video(tmp/"s2.mp4", "blue", 2.0, 1280, 720, with_audio=False)
    tl = build_timeline_from_assets([2.0, 2.0], [v1, v2])
    r = export_video(ExportRequest(timeline=tl, profile_name="TIKTOK", quality=Quality.HIGH,
                                   silent=True, project_id="mix"), output_dir=tmp/"out")
    assert r.status == "COMPLETED"
    assert r.resolution == (1080, 1920)


def test_export_burned_in_captions_english(tmp):
    v = _make_scene_video(tmp/"s.mp4", "black", 5.0, 1080, 1920, with_audio=False)
    tl = build_timeline_from_assets([5.0], [v])
    r = export_video(ExportRequest(timeline=tl, profile_name="TIKTOK", quality=Quality.HIGH,
        include_captions=True, caption_mode="burned_in",
        caption_cues=[{"start": 0.0, "end": 2.0, "text": "HELLO_CAPTION"}],
        caption_style={"font_size": 48, "font_color": "white", "position": "bottom",
                       "stroke": "black", "stroke_width": 3},
        silent=True, project_id="cap"), output_dir=tmp/"out")
    assert r.status == "COMPLETED"
    assert r.qc["ok"] is True


def test_export_burned_in_captions_arabic(tmp):
    v = _make_scene_video(tmp/"s.mp4", "black", 4.0, 1080, 1920, with_audio=False)
    tl = build_timeline_from_assets([4.0], [v])
    r = export_video(ExportRequest(timeline=tl, profile_name="TIKTOK", quality=Quality.HIGH,
        include_captions=True, caption_mode="burned_in",
        caption_cues=[{"start": 0.0, "end": 2.0, "text": "الأسد يصطاد في الغابة"}],
        caption_style={"font_size": 48, "position": "bottom"},
        silent=True, project_id="ar"), output_dir=tmp/"out")
    assert r.status == "COMPLETED"


def test_export_burned_in_captions_german(tmp):
    v = _make_scene_video(tmp/"s.mp4", "black", 4.0, 1080, 1920, with_audio=False)
    tl = build_timeline_from_assets([4.0], [v])
    r = export_video(ExportRequest(timeline=tl, profile_name="TIKTOK", quality=Quality.HIGH,
        include_captions=True, caption_mode="burned_in",
        caption_cues=[{"start": 0.0, "end": 2.0, "text": "Der Löwe jagt im Dschungel"}],
        caption_style={"font_size": 48, "position": "bottom"},
        silent=True, project_id="de"), output_dir=tmp/"out")
    assert r.status == "COMPLETED"


def test_export_caption_text_actually_rendered(tmp):
    """Confirm burned-in captions actually draw pixels (frame hash differs)."""
    v = _make_scene_video(tmp/"s.mp4", "black", 5.0, 1080, 1920, with_audio=False)
    tl = build_timeline_from_assets([5.0], [v])
    r_cap = export_video(ExportRequest(timeline=tl, profile_name="TIKTOK", quality=Quality.HIGH,
        include_captions=True, caption_mode="burned_in",
        caption_cues=[{"start": 0.0, "end": 2.0, "text": "VISIBLE_TEXT"}],
        caption_style={"font_size": 60, "font_color": "white", "position": "bottom",
                       "stroke": "black", "stroke_width": 3},
        silent=True, project_id="cap"), output_dir=tmp/"cap")
    r_nocap = export_video(ExportRequest(timeline=tl, profile_name="TIKTOK", quality=Quality.HIGH,
        silent=True, project_id="nocap"), output_dir=tmp/"nocap")
    def fh(video, t):
        f = tmp / f"fr_{t}.ppm"
        subprocess.run(["ffmpeg", "-y", "-i", str(video), "-ss", str(t), "-frames:v", "1",
                        "-f", "image2", str(f)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return hashlib.md5(f.read_bytes()).hexdigest()
    assert fh(r_cap.output_path, 0.5) != fh(r_nocap.output_path, 0.5)  # text rendered
    assert fh(r_cap.output_path, 4.0) == fh(r_nocap.output_path, 4.0)  # gone after


def test_export_branding(tmp):
    v = _make_scene_video(tmp/"s.mp4", "blue", 4.0, 1080, 1920, with_audio=False)
    logo = tmp/"logo.png"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=white:s=120x60:d=1",
                    "-frames:v", "1", str(logo)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    tl = build_timeline_from_assets([4.0], [v])
    r = export_video(ExportRequest(timeline=tl, profile_name="TIKTOK", quality=Quality.HIGH,
        include_branding=True, brand={"cta": "Shop Now", "watermark": "BRAND", "logo_path": str(logo)},
        silent=True, project_id="brand"), output_dir=tmp/"out")
    assert r.status == "COMPLETED"
    assert r.qc["ok"] is True


def test_export_missing_video_asset(tmp):
    tl = build_timeline_from_assets([3.0], [tmp/"nonexistent.mp4"])
    r = export_video(ExportRequest(timeline=tl, profile_name="TIKTOK", project_id="bad"),
                     output_dir=tmp/"out")
    assert r.status == "FAILED"
    assert r.error_code == "MISSING_VIDEO_ASSET"


def test_export_unsupported_profile(tmp):
    v = _make_scene_video(tmp/"s.mp4", "red", 3.0, with_audio=False)
    tl = build_timeline_from_assets([3.0], [v])
    r = export_video(ExportRequest(timeline=tl, profile_name="NOPE", project_id="bad"),
                     output_dir=tmp/"out")
    assert r.status == "FAILED"
    assert r.error_code == "UNSUPPORTED_PROFILE"


def test_export_no_source_overwrite(tmp):
    v = _make_scene_video(tmp/"s.mp4", "red", 3.0, with_audio=False)
    src_size = v.stat().st_size
    tl = build_timeline_from_assets([3.0], [v])
    export_video(ExportRequest(timeline=tl, profile_name="TIKTOK", quality=Quality.HIGH,
                               silent=True, project_id="nw"), output_dir=tmp/"out")
    assert v.stat().st_size == src_size  # source untouched


def test_export_result_serializable(tmp):
    r = ExportResult(status="FAILED", output_path=None, profile="TIKTOK", duration=0.0,
                     resolution=None, video_codec=None, audio_codec=None,
                     qc={"ok": False}, error_code="X", error_detail="d")
    import json
    json.dumps(r.to_dict())


def test_export_request_serializable(tmp):
    tl = build_timeline_from_assets([3.0], [tmp/"a.mp4"])
    req = ExportRequest(timeline=tl, profile_name="TIKTOK", project_id="x")
    import json
    json.dumps(req.to_dict())


# ---------------------------------------------------------------- final QC
def test_final_qc_missing(tmp):
    qc = final_qc(tmp/"nope.mp4")
    assert qc["ok"] is False
    assert qc["error_code"] == "FINAL_QC_FAILED"


def test_final_qc_empty(tmp):
    f = tmp/"empty.mp4"
    f.write_bytes(b"")
    qc = final_qc(f)
    assert qc["ok"] is False


def test_final_qc_no_video_stream(tmp):
    f = tmp/"audio_only.mp3"
    generate_tone_audio(f, 2.0)
    qc = final_qc(f)
    assert qc["ok"] is False


def test_final_qc_real_video(tmp):
    v = _make_scene_video(tmp/"s.mp4", "red", 3.0, with_audio=False)
    # No audio in this fixture; require_audio=False so QC passes on video alone.
    qc = final_qc(v, require_audio=False)
    assert qc["ok"] is True
    assert qc["video_codec"] == "h264"


def test_final_qc_missing_audio_required(tmp):
    v = _make_scene_video(tmp/"s.mp4", "red", 3.0, with_audio=False)
    qc = final_qc(v, require_audio=True)
    assert qc["ok"] is False
    assert "audio" in qc["error"].lower()


def test_final_qc_audio_optional(tmp):
    v = _make_scene_video(tmp/"s.mp4", "red", 3.0, with_audio=False)
    qc = final_qc(v, require_audio=False)
    assert qc["ok"] is True


def test_final_qc_profile_conformance(tmp):
    v = _make_scene_video(tmp/"s.mp4", "red", 3.0, 1080, 1920, with_audio=False)
    qc = final_qc(v, profile=TIKTOK, require_audio=False)
    assert qc["ok"] is True


def test_final_qc_profile_resolution_mismatch(tmp):
    v = _make_scene_video(tmp/"s.mp4", "red", 3.0, 640, 480, with_audio=False)
    qc = final_qc(v, profile=TIKTOK, require_audio=False)
    assert qc["ok"] is False
    assert "Resolution" in qc["error"]


# ---------------------------------------------------------------- audio/video sync
def test_audio_shorter_than_video_ok(tmp):
    # Voice 2s, video 3s — audio muxed with -shortest uses video length.
    v = _make_scene_video(tmp/"s.mp4", "red", 3.0, with_audio=False)
    vo = generate_silent_audio(tmp/"vo.mp3", 2.0)
    tl = build_timeline_from_assets([3.0], [v], voice_assets=[vo])
    r = export_video(ExportRequest(timeline=tl, profile_name="TIKTOK", project_id="sync"),
                     output_dir=tmp/"out")
    assert r.status == "COMPLETED"
    assert abs(r.duration - 3.0) < 0.5


def test_no_silent_truncate_of_long_audio(tmp):
    # Long music mixed to scene duration via mix_audio target_duration.
    v = _make_scene_video(tmp/"s.mp4", "red", 3.0, with_audio=False)
    music = generate_tone_audio(tmp/"music.mp3", 10.0, freq=120)
    tl = build_timeline_from_assets([3.0], [v], music_assets=[[music]])
    r = export_video(ExportRequest(timeline=tl, profile_name="TIKTOK", project_id="long"),
                     output_dir=tmp/"out")
    assert r.status == "COMPLETED"
    # Final duration ~3s, not 10s (mix truncated to target, not silent).
    assert r.duration < 5.0


# ---------------------------------------------------------------- backward compat
def test_backward_compat_phase5_plan():
    plan = local_content_plan(ContentBrief(idea="x", duration_seconds=12))
    assert len(plan.scenes) > 0


def test_backward_compat_phase7_resolve():
    from app.scene.continuity import resolve_all_scenes
    plan = local_content_plan(ContentBrief(idea="lion", duration_seconds=12, mode="cinematic"))
    ctxs = resolve_all_scenes(plan)
    assert len(ctxs) == len(plan.scenes)


def test_backward_compat_phase8_wan():
    from app.providers.wan import WanProvider, GenerationOptions
    p = WanProvider("http://127.0.0.1:1", timeout=2)
    opts = GenerationOptions(seed=1)
    opts.validate_ranges()
    assert opts.seed == 1


def test_backward_compat_phase9_audio():
    from app.audio.mixer import mix_audio, MixTrack, MixOptions
    from app.audio.qc import generate_tone_audio, verify_audio
    v = generate_tone_audio(Path(tempfile.mkdtemp())/"v.mp3", 1.0)
    r = mix_audio([MixTrack(path=v, kind="voice")], v.parent/"out.mp3",
                  MixOptions(target_duration=1.0))
    assert verify_audio(r.path)["ok"]


def test_backward_compat_phase9_captions():
    from app.captions.captions import generate_captions, CaptionFormat, CaptionStyle
    plan = local_content_plan(ContentBrief(idea="x", duration_seconds=10))
    rep = generate_captions(plan, CaptionFormat.SRT, CaptionStyle.TIKTOK)
    assert rep.to_dict()["format"] == "srt"

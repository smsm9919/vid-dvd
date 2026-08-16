"""Tests for the provenance-aware asset library (Phase 13).

These exercise real file-backed persistence, hash de-duplication, and real
media QC (FFmpeg-backed verify_mp4 / verify_audio). A valid MP4 and a valid
WAV are generated with real ffmpeg at test setup.
"""

import json
import subprocess
from pathlib import Path

import pytest

from app.assets.library import (
    AssetLibrary,
    AssetOrigin,
    AssetRecord,
    AssetType,
    LicenseRecord,
    _sha256,
)
from app.core.errors import TypedErrorCode, VideoError


def _ffmpeg_ok() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


FFMPEG = _ffmpeg_ok()
pytestmark = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not available")


def _make_valid_mp4(path: Path, duration: float = 2.0) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=320x240:rate=24",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ], capture_output=True, check=True)


def _make_valid_wav(path: Path, duration: float = 1.0) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-ac", "1", "-ar", "22050", str(path),
    ], capture_output=True, check=True)


def _stock_record(asset_id: str, path: Path, *, sha: str, size: int,
                  commercial: str = "allowed") -> AssetRecord:
    return AssetRecord(
        asset_id=asset_id, type=AssetType.VIDEO, path=str(path), origin=AssetOrigin.STOCK,
        provider="pexels", source_url="https://cdn/x.mp4", source_asset_id="123",
        page_url="https://www.pexels.com/video/123/",
        license=LicenseRecord(name="Pexels License", commercial_use=commercial,
                              attribution_required=False, provider="pexels",
                              author="Jane", source_url="https://www.pexels.com/terms-of-service/"),
        hash=sha, bytes_size=size, width=1280, height=720, duration=8.0,
        tags=["ocean", "nature"], scene_usage=[0],
    )


# --------------------------------------------------------------------- persistence
def test_library_creates_index_on_init(tmp_path):
    lib = AssetLibrary(tmp_path / "lib")
    assert (tmp_path / "lib" / "index.json").exists()
    assert lib.list() == []


def test_add_and_get_round_trip(tmp_path):
    lib = AssetLibrary(tmp_path / "lib")
    rec = _stock_record("a1", tmp_path / "v.mp4", sha="abc", size=100)
    lib.add(rec)
    got = lib.get("a1")
    assert got is not None
    assert got.provider == "pexels"
    assert got.license.commercial_use == "allowed"


def test_persistence_survives_reload(tmp_path):
    lib = AssetLibrary(tmp_path / "lib")
    lib.add(_stock_record("a1", tmp_path / "v.mp4", sha="abc", size=100))
    lib2 = AssetLibrary(tmp_path / "lib")
    assert lib2.get("a1") is not None
    assert lib2.get("a1").license.name == "Pexels License"


def test_corrupt_index_starts_fresh(tmp_path):
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    (lib_dir / "index.json").write_text("{not valid json")
    lib = AssetLibrary(lib_dir)
    assert lib.list() == []
    assert (lib_dir / "index.json").exists()  # recreated


# --------------------------------------------------------------------- dedupe by hash
def test_dedupe_by_hash_merges_scene_usage(tmp_path):
    lib = AssetLibrary(tmp_path / "lib")
    rec1 = _stock_record("a1", tmp_path / "v.mp4", sha="HHH", size=100)
    rec1.scene_usage = [0]
    rec2 = _stock_record("a2", tmp_path / "v.mp4", sha="HHH", size=100)  # same hash
    rec2.scene_usage = [1, 2]
    lib.add(rec1)
    result = lib.add(rec2)
    assert result.asset_id == "a1"  # existing wins
    assert set(result.scene_usage) == {0, 1, 2}
    assert len(lib.list()) == 1


def test_find_by_hash(tmp_path):
    lib = AssetLibrary(tmp_path / "lib")
    lib.add(_stock_record("a1", tmp_path / "v.mp4", sha="HHH", size=100))
    assert lib.find_by_hash("HHH") is not None
    assert lib.find_by_hash("other") is None


# --------------------------------------------------------------------- file validity
def test_get_marks_fail_when_file_missing(tmp_path):
    lib = AssetLibrary(tmp_path / "lib")
    lib.add(_stock_record("a1", tmp_path / "gone.mp4", sha="abc", size=100))
    got = lib.get("a1")
    assert got.qc_state == "fail"
    assert "missing" in got.qc_detail


def test_get_marks_fail_when_hash_mismatch(tmp_path):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"real content here")
    lib = AssetLibrary(tmp_path / "lib")
    lib.add(_stock_record("a1", f, sha="wronghash", size=len(b"real content here")))
    got = lib.get("a1")
    assert got.qc_state == "fail"
    assert "mismatch" in got.qc_detail


def test_get_passes_when_hash_matches(tmp_path):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"real content here")
    sha = _sha256(f)
    lib = AssetLibrary(tmp_path / "lib")
    lib.add(_stock_record("a1", f, sha=sha, size=f.stat().st_size))
    got = lib.get("a1")
    # qc_state stays unknown until verify() runs, but file is valid
    assert got.qc_state in ("unknown", "pass")


# --------------------------------------------------------------------- real QC verification
def test_verify_valid_mp4_pass(tmp_path):
    mp4 = tmp_path / "v.mp4"
    _make_valid_mp4(mp4)
    sha = _sha256(mp4)
    lib = AssetLibrary(tmp_path / "lib")
    lib.add(AssetRecord(
        asset_id="v1", type=AssetType.VIDEO, path=str(mp4), origin=AssetOrigin.STOCK,
        provider="pexels", hash=sha, bytes_size=mp4.stat().st_size,
    ))
    rec = lib.verify("v1")
    assert rec.qc_state == "pass"
    assert rec.width == 320
    assert rec.height == 240
    assert rec.duration is not None and rec.duration > 0


def test_verify_corrupt_mp4_fails(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not an mp4")
    sha = _sha256(bad)
    lib = AssetLibrary(tmp_path / "lib")
    lib.add(AssetRecord(
        asset_id="b1", type=AssetType.VIDEO, path=str(bad), origin=AssetOrigin.STOCK,
        provider="pexels", hash=sha, bytes_size=bad.stat().st_size,
    ))
    rec = lib.verify("b1")
    assert rec.qc_state == "fail"
    assert rec.qc_detail is not None


def test_verify_valid_wav_pass(tmp_path):
    wav = tmp_path / "a.wav"
    _make_valid_wav(wav)
    sha = _sha256(wav)
    lib = AssetLibrary(tmp_path / "lib")
    lib.add(AssetRecord(
        asset_id="a1", type=AssetType.VOICE, path=str(wav), origin=AssetOrigin.GENERATED,
        provider="piper", hash=sha, bytes_size=wav.stat().st_size,
    ))
    rec = lib.verify("a1")
    assert rec.qc_state == "pass"
    assert rec.sample_rate == 22050
    assert rec.duration is not None


def test_verify_unknown_asset_raises(tmp_path):
    lib = AssetLibrary(tmp_path / "lib")
    with pytest.raises(VideoError) as ei:
        lib.verify("nope")
    assert ei.value.code == TypedErrorCode.ASSET_NOT_FOUND


# --------------------------------------------------------------------- license report
def test_license_report_all_commercial_allowed(tmp_path):
    lib = AssetLibrary(tmp_path / "lib")
    lib.add(_stock_record("a1", tmp_path / "v.mp4", sha="h1", size=100, commercial="allowed"))
    lib.add(_stock_record("a2", tmp_path / "v2.mp4", sha="h2", size=100, commercial="allowed"))
    rep = lib.license_report()
    assert rep["total"] == 2
    assert rep["commercial_use_unknown_or_restricted"] == 0
    assert "commercial-use-permitted" in rep["note"].lower() or "All assets" in rep["note"]


def test_license_report_flags_unknown(tmp_path):
    lib = AssetLibrary(tmp_path / "lib")
    lib.add(_stock_record("a1", tmp_path / "v.mp4", sha="h1", size=100, commercial="allowed"))
    lib.add(_stock_record("a2", tmp_path / "v2.mp4", sha="h2", size=100, commercial="unknown"))
    rep = lib.license_report()
    assert rep["commercial_use_unknown_or_restricted"] == 1
    assert "verify" in rep["note"].lower()


# --------------------------------------------------------------------- delete
def test_delete_removes_from_index(tmp_path):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"x")
    lib = AssetLibrary(tmp_path / "lib")
    lib.add(_stock_record("a1", f, sha=_sha256(f), size=1))
    assert lib.delete("a1", remove_file=True) is True
    assert lib.get("a1") is None
    assert not f.exists()
    assert lib.delete("nope") is False

"""Provenance-aware asset library (Phase 13).

A persistent, file-backed catalog of every external asset used by a production
(video clips, images, audio, voice, music, SFX, references). Each asset carries
full provenance (provider, source URL, author, license, retrieval timestamp)
and is verified against the real file on disk before it is reported as usable.

Guarantees:
- The library NEVER reports an asset as PASS without real file verification.
- License status defaults to ``unknown`` and is only ``allowed`` when the
  provider's official terms support it.
- Assets are de-duplicated by content hash (sha256).
- The index survives restart (atomic JSON write via temp+rename).
- Downloaded media lives outside Git (under ASSET_LIBRARY_DIR / project assets).
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from ..core.errors import TypedErrorCode, VideoError
from ..core.logging import log
from ..media import probe, verify_mp4
from ..audio.qc import probe_audio, verify_audio


class AssetType(str, Enum):
    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"
    VOICE = "voice"
    MUSIC = "music"
    SFX = "sfx"
    REFERENCE = "reference"

    @classmethod
    def from_media_type(cls, value: str) -> "AssetType":
        v = value.lower()
        if v in ("video", "stock_video"):
            return cls.VIDEO
        if v in ("image", "stock_image", "generated_image", "reference_image", "user_image"):
            return cls.IMAGE
        return cls.REFERENCE


class AssetOrigin(str, Enum):
    STOCK = "stock"
    GENERATED = "generated"
    USER = "user"
    REFERENCE = "reference"


@dataclass
class LicenseRecord:
    name: str = "Unknown"
    commercial_use: str = "unknown"
    attribution_required: bool = False
    attribution_text: Optional[str] = None
    source_url: Optional[str] = None
    provider: Optional[str] = None
    author: Optional[str] = None
    retrieved_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AssetRecord:
    asset_id: str
    type: AssetType
    path: str
    origin: AssetOrigin
    provider: Optional[str] = None
    source_url: Optional[str] = None
    source_asset_id: Optional[str] = None
    page_url: Optional[str] = None
    license: LicenseRecord = field(default_factory=LicenseRecord)
    hash: Optional[str] = None
    mime_type: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[float] = None
    fps: Optional[float] = None
    codec: Optional[str] = None
    sample_rate: Optional[int] = None
    bytes_size: Optional[int] = None
    quality_score: Optional[float] = None
    tags: list[str] = field(default_factory=list)
    scene_usage: list[int] = field(default_factory=list)
    project_id: Optional[str] = None
    qc_state: str = "unknown"
    qc_detail: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        d["origin"] = self.origin.value
        d["license"] = self.license.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AssetRecord":
        lic = d.get("license") or {}
        return cls(
            asset_id=d["asset_id"],
            type=AssetType(d["type"]),
            path=d["path"],
            origin=AssetOrigin(d.get("origin", "stock")),
            provider=d.get("provider"),
            source_url=d.get("source_url"),
            source_asset_id=d.get("source_asset_id"),
            page_url=d.get("page_url"),
            license=LicenseRecord(**{k: lic.get(k) for k in (
                "name", "commercial_use", "attribution_required", "attribution_text",
                "source_url", "provider", "author", "retrieved_at")}),
            hash=d.get("hash"),
            mime_type=d.get("mime_type"),
            width=d.get("width"),
            height=d.get("height"),
            duration=d.get("duration"),
            fps=d.get("fps"),
            codec=d.get("codec"),
            sample_rate=d.get("sample_rate"),
            bytes_size=d.get("bytes_size"),
            quality_score=d.get("quality_score"),
            tags=d.get("tags", []),
            scene_usage=d.get("scene_usage", []),
            project_id=d.get("project_id"),
            qc_state=d.get("qc_state", "unknown"),
            qc_detail=d.get("qc_detail"),
            created_at=d.get("created_at", datetime.now(timezone.utc).isoformat()),
        )


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class AssetLibrary:
    """Persistent provenance-aware asset catalog.

    The index file is a JSON document under ``library_dir/index.json``. Reads
    re-validate the real file (exists, non-empty, hash match) before reporting
    an asset as usable; invalid entries are marked qc_state=fail.
    """

    def __init__(self, library_dir: Path) -> None:
        self.library_dir = Path(library_dir)
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.library_dir / "index.json"
        self._assets: dict[str, AssetRecord] = {}
        self._by_hash: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.index_path.exists():
            self._assets = {}
            self._by_hash = {}
            self._save()
            return
        try:
            data = json.loads(self.index_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log("ASSET", f"corrupt index, starting fresh: {e}")
            self._assets = {}
            self._by_hash = {}
            self._save()
            return
        self._assets = {}
        self._by_hash = {}
        for d in data.get("assets", []):
            try:
                rec = AssetRecord.from_dict(d)
            except Exception:
                continue
            self._assets[rec.asset_id] = rec
            if rec.hash:
                self._by_hash[rec.hash] = rec.asset_id

    def _save(self) -> None:
        data = {"assets": [r.to_dict() for r in self._assets.values()]}
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.index_path)

    def add(self, record: AssetRecord) -> AssetRecord:
        """Add or update an asset. De-duplicates by content hash."""
        if record.hash and record.hash in self._by_hash:
            existing_id = self._by_hash[record.hash]
            existing = self._assets[existing_id]
            for s in record.scene_usage:
                if s not in existing.scene_usage:
                    existing.scene_usage.append(s)
            for t in record.tags:
                if t not in existing.tags:
                    existing.tags.append(t)
            self._save()
            return existing
        self._assets[record.asset_id] = record
        if record.hash:
            self._by_hash[record.hash] = record.asset_id
        self._save()
        log("ASSET", f"added {record.asset_id}", type=record.type.value, provider=record.provider)
        return record

    def get(self, asset_id: str) -> Optional[AssetRecord]:
        rec = self._assets.get(asset_id)
        if rec is None:
            return None
        if not self._file_valid(rec):
            rec.qc_state = "fail"
            rec.qc_detail = "file missing/empty/hash mismatch"
            self._save()
        return rec

    def list(self, *, type: Optional[AssetType] = None,
             project_id: Optional[str] = None) -> list[AssetRecord]:
        out = []
        for rec in list(self._assets.values()):
            if type and rec.type != type:
                continue
            if project_id and rec.project_id != project_id:
                continue
            out.append(rec)
        return out

    def search(self, query: str) -> list[AssetRecord]:
        q = query.lower()
        return [r for r in self._assets.values()
                if q in " ".join(r.tags).lower() or (r.provider and q in r.provider.lower())]

    def find_by_hash(self, sha256: str) -> Optional[AssetRecord]:
        aid = self._by_hash.get(sha256)
        return self._assets.get(aid) if aid else None

    def invalidate(self, asset_id: str) -> None:
        rec = self._assets.get(asset_id)
        if rec:
            rec.qc_state = "fail"
            rec.qc_detail = "manually invalidated"
            self._save()

    def delete(self, asset_id: str, *, remove_file: bool = False) -> bool:
        rec = self._assets.pop(asset_id, None)
        if rec and rec.hash:
            self._by_hash.pop(rec.hash, None)
        if rec and remove_file:
            try:
                Path(rec.path).unlink(missing_ok=True)
            except OSError:
                pass
        self._save()
        return rec is not None

    def verify(self, asset_id: str) -> AssetRecord:
        """Run real media QC on the asset file. Updates qc_state honestly."""
        rec = self._assets.get(asset_id)
        if rec is None:
            raise VideoError(TypedErrorCode.ASSET_NOT_FOUND, f"Asset {asset_id} not in library.")
        if not self._file_valid(rec):
            rec.qc_state = "fail"
            rec.qc_detail = "file missing/empty/hash mismatch"
            self._save()
            return rec
        try:
            if rec.type == AssetType.VIDEO:
                report = verify_mp4(Path(rec.path))
                if not report.get("ok"):
                    rec.qc_state = "fail"
                    rec.qc_detail = f"verify_mp4: {report.get('error', 'failed')}"
                else:
                    rec.qc_state = "pass"
                    rec.qc_detail = None
                    rec.duration = report.get("duration", rec.duration)
                    rec.width = report.get("width", rec.width)
                    rec.height = report.get("height", rec.height)
                    rec.fps = report.get("fps", rec.fps)
                    rec.codec = report.get("codec", rec.codec)
            elif rec.type in (AssetType.AUDIO, AssetType.VOICE, AssetType.MUSIC, AssetType.SFX):
                report = verify_audio(Path(rec.path))
                if not report.get("ok"):
                    rec.qc_state = "fail"
                    rec.qc_detail = f"verify_audio: {report.get('error', 'failed')}"
                else:
                    rec.qc_state = "pass"
                    rec.qc_detail = None
                    rec.duration = report.get("duration", rec.duration)
                    rec.sample_rate = report.get("sample_rate", rec.sample_rate)
            else:
                rec.qc_state = "pass"
                rec.qc_detail = None
        except VideoError as e:
            rec.qc_state = "fail"
            rec.qc_detail = e.detail
        except Exception as e:  # noqa: BLE001
            rec.qc_state = "fail"
            rec.qc_detail = f"unexpected: {e}"
        self._save()
        return rec

    def _file_valid(self, rec: AssetRecord) -> bool:
        p = Path(rec.path)
        if not p.exists() or p.stat().st_size == 0:
            return False
        if rec.hash:
            try:
                return _sha256(p) == rec.hash
            except OSError:
                return False
        return True

    def license_report(self, *, project_id: Optional[str] = None) -> dict[str, Any]:
        records = self.list(project_id=project_id)
        items = []
        commercial_unknown = 0
        for r in records:
            items.append({
                "asset_id": r.asset_id, "type": r.type.value, "provider": r.provider,
                "source_url": r.source_url, "author": r.license.author,
                "license": r.license.name, "commercial_use": r.license.commercial_use,
                "attribution_required": r.license.attribution_required,
                "attribution_text": r.license.attribution_text,
            })
            if r.license.commercial_use != "allowed":
                commercial_unknown += 1
        return {
            "assets": items,
            "total": len(items),
            "commercial_use_unknown_or_restricted": commercial_unknown,
            "note": ("All assets have commercial-use-permitted licenses."
                     if commercial_unknown == 0 and items
                     else "Some assets have unknown/restricted commercial use — verify before commercial use."),
        }


def library_for_project(project_id: str) -> AssetLibrary:
    from .. import config
    return AssetLibrary(config.ASSET_LIBRARY_DIR / project_id)

"""Cache + idempotency store (Phase 11).

Safe caching for successful deterministic/intermediate outputs. A cached asset
is valid only if: file exists, file size > 0, metadata matches, and QC passes
where applicable. Never trusts a cache entry blindly — re-validates on lookup.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from ..core.errors import TypedErrorCode, VideoError
from ..core.logging import log
from ..media import verify_mp4
from ..audio.qc import verify_audio
from .models import fingerprint, file_fingerprint


@dataclass
class CacheEntry:
    key: str  # input fingerprint
    job_type: str
    output_path: str
    output_fingerprint: str
    size: int
    created_at: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CacheEntry":
        return cls(**d)


class CacheStore:
    """File-backed cache for deterministic/intermediate outputs.

    The store is a JSON index mapping cache_key -> CacheEntry. On lookup, the
    referenced file is re-validated (exists, size>0, fingerprint match, QC).
    An entry whose file is missing/corrupted is treated as a miss and evicted.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, CacheEntry] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                for k, v in data.items():
                    self._entries[k] = CacheEntry.from_dict(v)
            except (json.JSONDecodeError, TypeError, ValueError):
                log("CACHE", "index corrupted; starting fresh", path=str(self.path))
                self._entries = {}

    def _save(self) -> None:
        data = {k: v.to_dict() for k, v in self._entries.items()}
        self.path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    @staticmethod
    def _key(job_type: str, input_fingerprint: str) -> str:
        return f"{job_type}:{input_fingerprint}"

    def lookup(self, job_type: str, input_fingerprint: str, *,
               expected_metadata: Optional[dict[str, Any]] = None,
               validate: Optional[str] = None) -> Optional[CacheEntry]:
        """Look up a cached entry by (job_type, fingerprint).

        validate: 'video' | 'audio' | None — applies the matching QC.
        Re-validates the file on every lookup. Returns None on miss/invalid.
        """
        key = self._key(job_type, input_fingerprint)
        entry = self._entries.get(key)
        if entry is None:
            return None
        if not self._validate(entry, expected_metadata, validate):
            log("CACHE", "evicting invalid entry", key=key)
            del self._entries[key]
            self._save()
            return None
        log("CACHE", "hit", key=key, path=entry.output_path)
        return entry

    def _validate(self, entry: CacheEntry,
                  expected_metadata: Optional[dict[str, Any]],
                  validate: Optional[str]) -> bool:
        p = Path(entry.output_path)
        if not p.exists() or not p.is_file():
            return False
        if p.stat().st_size <= 0:
            return False
        # Fingerprint must match (content integrity).
        fp = file_fingerprint(p)
        if fp != entry.output_fingerprint:
            return False
        # Metadata match (e.g. expected duration/resolution).
        if expected_metadata:
            for k, v in expected_metadata.items():
                if entry.metadata.get(k) != v:
                    return False
        # QC where applicable.
        if validate == "video":
            try:
                verify_mp4(p)
            except VideoError:
                return False
        elif validate == "audio":
            try:
                verify_audio(p)
            except VideoError:
                return False
        return True

    def store(self, job_type: str, input_fingerprint: str, output_path: Path,
              *, metadata: Optional[dict[str, Any]] = None,
              validate: Optional[str] = None) -> CacheEntry:
        """Store a successful output. Validates before storing (never blindly)."""
        p = Path(output_path)
        if not p.exists() or p.is_file() is False:
            raise VideoError(
                TypedErrorCode.CACHE_INVALID,
                f"Cannot cache missing output: {p}",
                context={"path": str(p)})
        if p.stat().st_size <= 0:
            raise VideoError(
                TypedErrorCode.CACHE_INVALID,
                f"Cannot cache empty output: {p}",
                context={"path": str(p)})
        if validate == "video":
            verify_mp4(p)  # raises INVALID_MP4
        elif validate == "audio":
            verify_audio(p)  # raises INVALID_AUDIO
        fp = file_fingerprint(p)
        if fp is None:
            raise VideoError(TypedErrorCode.CACHE_INVALID, f"Cannot fingerprint {p}.")
        entry = CacheEntry(
            key=self._key(job_type, input_fingerprint), job_type=job_type,
            output_path=str(p), output_fingerprint=fp, size=p.stat().st_size,
            created_at=time.time(), metadata=metadata or {},
        )
        self._entries[entry.key] = entry
        self._save()
        log("CACHE", "stored", key=entry.key, path=str(p))
        return entry

    def invalidate(self, job_type: str, input_fingerprint: str) -> bool:
        key = self._key(job_type, input_fingerprint)
        if key in self._entries:
            del self._entries[key]
            self._save()
            return True
        return False

    def __len__(self) -> int:
        return len(self._entries)

    def all_entries(self) -> list[CacheEntry]:
        return list(self._entries.values())

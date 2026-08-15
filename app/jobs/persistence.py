"""File-based JSON persistence (Phase 11).

Survives application restart. No heavyweight infrastructure dependency — each
project's jobs are stored as a single JSON file. Atomic writes via temp + rename
to avoid corruption on crash mid-write.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from ..core.errors import TypedErrorCode, VideoError
from ..core.logging import log
from ..config import PROJECTS_DIR
from .models import Job


class JobStore:
    """Persists jobs per project as JSON files.

    Layout: PROJECTS_DIR/<project_id>/jobs.json
    Contains: {"project_id": ..., "jobs": {job_id: job_dict, ...}, "updated_at": ...}
    """

    def __init__(self, project_id: str, root: Optional[Path] = None) -> None:
        self.project_id = project_id
        self.root = Path(root) if root else PROJECTS_DIR
        self.dir = self.root / project_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "jobs.json"
        self._jobs: dict[str, Job] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._jobs = {}
            # Persist an empty index so the project is discoverable for listing/recovery.
            self._save()
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for jid, jd in data.get("jobs", {}).items():
                self._jobs[jid] = Job.from_dict(jd)
        except (json.JSONDecodeError, TypeError, ValueError, KeyError) as e:
            log("PERSIST", "corrupt jobs.json; starting fresh",
                project=self.project_id, error=str(e))
            self._jobs = {}

    def _save(self) -> None:
        data = {
            "project_id": self.project_id,
            "updated_at": time.time(),
            "jobs": {jid: j.to_dict() for jid, j in self._jobs.items()},
        }
        # Atomic write: temp file + rename.
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp, self.path)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise

    def put(self, job: Job) -> None:
        self._jobs[job.job_id] = job
        self._save()

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def require(self, job_id: str) -> Job:
        j = self._jobs.get(job_id)
        if j is None:
            raise VideoError(
                TypedErrorCode.JOB_NOT_FOUND,
                f"Job {job_id} not found in project {self.project_id}.",
                context={"job_id": job_id, "project_id": self.project_id})
        return j

    def all(self) -> list[Job]:
        return list(self._jobs.values())

    def by_type(self, job_type: str) -> list[Job]:
        return [j for j in self._jobs.values() if j.job_type.value == job_type]

    def by_scene(self, scene_index: int) -> list[Job]:
        return [j for j in self._jobs.values() if j.scene_index == scene_index]

    def remove(self, job_id: str) -> bool:
        if job_id in self._jobs:
            del self._jobs[job_id]
            self._save()
            return True
        return False

    def __len__(self) -> int:
        return len(self._jobs)


class ProjectRegistry:
    """Tracks all known project IDs (for listing + recovery)."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root else PROJECTS_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def list_projects(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted([p.name for p in self.root.iterdir()
                       if p.is_dir() and (p / "jobs.json").exists()])

    def project_dir(self, project_id: str) -> Path:
        return self.root / project_id

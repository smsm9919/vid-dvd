"""Production orchestrator (Phase 11).

Connects the existing components — Content Brain, Advertising Variants, Scene
Continuity, Wan/ComfyUI, Voice, Music, SFX, Captions, Editing, Final QC — under
a single job-driven state machine with dependencies, retries, caching,
resumability, and recovery. Does NOT rewrite those components; it calls them.

The orchestrator is provider-aware and never fakes provider success. When a
real provider is unavailable, the job FAILS with a typed error (NO_PROVIDER /
COMFYUI_UNREACHABLE) and may be retried per policy if the error is transient.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..audio.qc import generate_silent_audio, generate_tone_audio, verify_audio
from ..brain.content_brain import local_content_plan
from ..brain.models import ContentBrief, ProductionPlan, Scene
from ..config import OUTPUT_DIR, PROJECTS_DIR
from ..core.errors import TypedErrorCode, VideoError
from ..core.logging import log
from ..editing.assembly import ExportRequest, ExportResult, export_video
from ..editing.profiles import Quality, get_profile
from ..editing.timeline import build_timeline_from_assets
from ..media import verify_mp4
from ..scene.continuity import resolve_all_scenes, resolve_scene_context
from .cache import CacheStore
from .dependencies import (
    ConcurrencyLimiter, assert_dependencies, check_dependencies, compute_progress,
)
from .models import Job, JobType, file_fingerprint, fingerprint, make_job, new_project_id
from .persistence import JobStore, ProjectRegistry
from .retry import DEFAULT_POLICY, RetryPolicy, record_retry, should_retry
from .state import (
    JobState, assert_transition, can_cancel, can_retry, is_active, is_terminal,
    is_valid_transition,
)


class JobError(VideoError):
    """Typed job-orchestration failure."""

    def __init__(self, code: TypedErrorCode, detail: str = "", *,
                 context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(code, detail, context=context)


@dataclass
class Project:
    """An orchestrator project: a store + cache + plan."""

    project_id: str
    store: JobStore
    cache: CacheStore
    plan: Optional[ProductionPlan] = None
    scene_contexts: dict[int, Any] = field(default_factory=dict)
    assets_dir: Path = field(default_factory=lambda: Path("."))
    created_at: float = field(default_factory=time.time)

    def to_summary(self) -> dict[str, Any]:
        jobs = self.store.all()
        prog = compute_progress(jobs)
        return {
            "project_id": self.project_id,
            "created_at": self.created_at,
            "plan_present": self.plan is not None,
            "scene_count": len(self.plan.scenes) if self.plan else 0,
            "job_count": len(jobs),
            "progress": prog.to_dict(),
        }


class Orchestrator:
    """The production job engine.

    Synchronous execution of deterministic stages (planning, scene resolution,
    captions, editing, QC) and provider-aware execution of generation stages
    (video, voice, music, sfx) that FAIL with NO_PROVIDER when no real provider
    is configured. Independent scene jobs run with bounded concurrency.
    """

    def __init__(self, root: Optional[Path] = None,
                 policy: RetryPolicy = DEFAULT_POLICY,
                 limiter: Optional[ConcurrencyLimiter] = None) -> None:
        self.root = Path(root) if root else PROJECTS_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry = ProjectRegistry(self.root)
        self.policy = policy
        self.limiter = limiter or ConcurrencyLimiter()
        self.projects: dict[str, Project] = {}

    # --------------------------------------------------------------- project mgmt
    def create_project(self, project_id: Optional[str] = None) -> Project:
        pid = project_id or new_project_id()
        store = JobStore(pid, root=self.root)
        cache = CacheStore(self.root / pid / "cache.json")
        assets = self.root / pid / "assets"
        assets.mkdir(parents=True, exist_ok=True)
        proj = Project(project_id=pid, store=store, cache=cache, assets_dir=assets)
        self.projects[pid] = proj
        log("PROJECT", "created", project_id=pid)
        return proj

    def get_project(self, project_id: str) -> Project:
        if project_id not in self.projects:
            # Rehydrate from disk (resumability across restart).
            if (self.root / project_id / "jobs.json").exists():
                store = JobStore(project_id, root=self.root)
                cache = CacheStore(self.root / project_id / "cache.json")
                assets = self.root / project_id / "assets"
                proj = Project(project_id=project_id, store=store, cache=cache, assets_dir=assets)
                self.projects[project_id] = proj
                return proj
            raise JobError(TypedErrorCode.JOB_NOT_FOUND,
                           f"Project {project_id} not found.",
                           context={"project_id": project_id})
        return self.projects[project_id]

    def list_projects(self) -> list[str]:
        return self.registry.list_projects()

    # --------------------------------------------------------------- job mgmt
    def create_job(self, project_id: str, job_type: JobType, **kwargs) -> Job:
        proj = self.get_project(project_id)
        job = make_job(project_id, job_type, **kwargs)
        proj.store.put(job)
        log("JOB", "created", job_id=job.job_id, project=project_id,
            type=job_type.value, state=job.state.value)
        return job

    def get_job(self, project_id: str, job_id: str) -> Job:
        proj = self.get_project(project_id)
        return proj.store.require(job_id)

    def list_jobs(self, project_id: str) -> list[Job]:
        return self.get_project(project_id).store.all()

    def transition(self, project_id: str, job_id: str, to_state: JobState,
                   *, output_path: Optional[str] = None,
                   error_code: Optional[str] = None,
                   error_detail: Optional[str] = None,
                   outputs: Optional[dict] = None) -> Job:
        proj = self.get_project(project_id)
        job = proj.store.require(job_id)
        assert_transition(job.state, to_state)
        if to_state is JobState.RUNNING and job.started_at is None:
            job.started_at = time.time()
        if is_terminal(to_state) and to_state is not JobState.FAILED:
            job.completed_at = time.time()
        if to_state is JobState.FAILED:
            job.error_code = error_code
            job.error_detail = error_detail
        if output_path:
            job.output_path = output_path
        if outputs:
            job.outputs = outputs
        job.state = to_state
        proj.store.put(job)
        log("JOB", to_state.value.lower(), job_id=job.job_id, project=project_id,
            type=job.job_type.value)
        return job

    def retry_job(self, project_id: str, job_id: str) -> Job:
        proj = self.get_project(project_id)
        job = proj.store.require(job_id)
        if not can_retry(job.state):
            raise JobError(TypedErrorCode.NON_RETRYABLE,
                           f"Job {job_id} is not in a retryable state ({job.state.value}).",
                           context={"job_id": job_id, "state": job.state.value})
        delay = record_retry(job, self.policy)
        assert_transition(job.state, JobState.RETRYING)
        job.state = JobState.RETRYING
        proj.store.put(job)
        log("JOB", "retry", job_id=job_id, attempt=job.retry_count, backoff=delay)
        # Transition to QUEUED for re-execution.
        assert_transition(job.state, JobState.QUEUED)
        job.state = JobState.QUEUED
        job.error_code = None
        job.error_detail = None
        proj.store.put(job)
        return job

    def cancel_job(self, project_id: str, job_id: str) -> Job:
        proj = self.get_project(project_id)
        job = proj.store.require(job_id)
        if not can_cancel(job.state):
            raise JobError(TypedErrorCode.CANCEL_NOT_ALLOWED,
                           f"Job {job_id} cannot be canceled from state {job.state.value}.",
                           context={"job_id": job_id, "state": job.state.value})
        assert_transition(job.state, JobState.CANCEL_REQUESTED)
        job.cancel_requested = True
        job.state = JobState.CANCEL_REQUESTED
        # If an external provider cannot actually be stopped, record that honestly.
        # Here we do not invoke any provider cancel (no running provider call in
        # the synchronous path), so provider_canceled stays None (unknown).
        proj.store.put(job)
        log("JOB", "cancel_requested", job_id=job_id, provider_canceled=job.provider_canceled)
        # Final transition to CANCELED.
        assert_transition(job.state, JobState.CANCELED)
        job.state = JobState.CANCELED
        proj.store.put(job)
        return job

    def get_progress(self, project_id: str) -> dict[str, Any]:
        proj = self.get_project(project_id)
        return compute_progress(proj.store.all()).to_dict()

    # --------------------------------------------------------------- execution
    def _fail(self, proj: Project, job: Job, code: str, detail: str,
              context: Optional[dict] = None) -> Job:
        job.error_code = code
        job.error_detail = detail
        if context:
            job.outputs = {**job.outputs, **context}
        assert_transition(job.state, JobState.FAILED)
        job.state = JobState.FAILED
        proj.store.put(job)
        log("JOB", "failed", job_id=job.job_id, code=code, detail=detail[:100])
        return job

    def _complete(self, proj: Project, job: Job, output_path: Optional[str] = None,
                  outputs: Optional[dict] = None) -> Job:
        if output_path:
            job.output_path = output_path
        if outputs:
            job.outputs = {**job.outputs, **outputs}
        assert_transition(job.state, JobState.COMPLETED)
        job.state = JobState.COMPLETED
        job.completed_at = time.time()
        proj.store.put(job)
        log("JOB", "completed", job_id=job.job_id, output=output_path)
        return job

    def _run_with_retry(self, proj: Project, job: Job, fn) -> Job:
        """Execute a job function with retry policy. fn() returns (path, outputs)
        or raises VideoError. Caching: if a valid cached result exists for the
        job's fingerprint, reuse it (idempotency)."""
        # Idempotency: check cache first.
        cached = proj.cache.lookup(job.job_type.value, job.input_fingerprint or "",
                                   validate=self._cache_validate_kind(job.job_type))
        if cached is not None:
            # Transition through QUEUED -> RUNNING -> COMPLETED (valid path).
            if is_valid_transition(job.state, JobState.QUEUED) and job.state is not JobState.RUNNING:
                assert_transition(job.state, JobState.QUEUED)
                job.state = JobState.QUEUED
                proj.store.put(job)
            if job.state is JobState.QUEUED:
                assert_transition(job.state, JobState.RUNNING)
                job.state = JobState.RUNNING
                if job.started_at is None:
                    job.started_at = time.time()
                proj.store.put(job)
            job.output_path = cached.output_path
            job.output_fingerprint = cached.output_fingerprint
            log("JOB", "cache_reuse", job_id=job.job_id, path=cached.output_path)
            return self._complete(proj, job, output_path=cached.output_path)
        # Check dependencies before running.
        try:
            assert_dependencies(job, {j.job_id: j for j in proj.store.all()})
        except VideoError as e:
            return self._fail(proj, job, e.code.value, e.detail, e.context)
        # Transition to QUEUED then RUNNING (valid path: DRAFT/PLANNED/etc -> QUEUED -> RUNNING).
        if job.state in (JobState.DRAFT, JobState.PLANNED, JobState.SCENES_READY,
                         JobState.VIDEO_GENERATED, JobState.AUDIO_READY, JobState.RETRYING):
            if job.state is not JobState.QUEUED:
                if is_valid_transition(job.state, JobState.QUEUED):
                    assert_transition(job.state, JobState.QUEUED)
                    job.state = JobState.QUEUED
                    proj.store.put(job)
        if job.state is JobState.QUEUED:
            assert_transition(job.state, JobState.RUNNING)
            job.state = JobState.RUNNING
            if job.started_at is None:
                job.started_at = time.time()
            proj.store.put(job)
        while True:
            try:
                result = fn()
                path = result[0] if isinstance(result, tuple) else result
                outs = result[1] if isinstance(result, tuple) and len(result) > 1 else {}
                # Validate output integrity before completing.
                self._validate_output(job, path)
                # Cache the successful output.
                if path:
                    try:
                        entry = proj.cache.store(
                            job.job_type.value, job.input_fingerprint or "", Path(path),
                            validate=self._cache_validate_kind(job.job_type))
                        job.output_fingerprint = entry.output_fingerprint
                    except VideoError:
                        pass  # caching is best-effort; completion still proceeds
                return self._complete(proj, job, output_path=str(path) if path else None,
                                      outputs=outs)
            except VideoError as e:
                if should_retry(job, self.policy):
                    delay = record_retry(job, self.policy)
                    log("JOB", "retry", job_id=job.job_id, attempt=job.retry_count,
                        backoff=delay, code=e.code.value)
                    proj.store.put(job)
                    continue
                return self._fail(proj, job, e.code.value, e.detail, e.context)

    def _cache_validate_kind(self, job_type: JobType) -> Optional[str]:
        if job_type is JobType.VIDEO_SCENE:
            return "video"
        if job_type in (JobType.VOICEOVER, JobType.MUSIC, JobType.SFX):
            return "audio"
        return None

    def _validate_output(self, job: Job, path: Optional[str]) -> None:
        if not path:
            return
        p = Path(path)
        if job.job_type is JobType.VIDEO_SCENE:
            verify_mp4(p)  # raises INVALID_MP4
        elif job.job_type in (JobType.VOICEOVER, JobType.MUSIC, JobType.SFX):
            verify_audio(p)  # raises INVALID_AUDIO

    # --------------------------------------------------------------- stage executors
    def run_content_plan(self, project_id: str, job_id: str) -> Job:
        proj = self.get_project(project_id)
        job = proj.store.require(job_id)

        def fn():
            brief_dict = job.inputs.get("brief", {})
            brief = ContentBrief(**brief_dict) if isinstance(brief_dict, dict) else brief_dict
            plan = local_content_plan(brief)
            proj.plan = plan
            plan_path = proj.assets_dir / "plan.json"
            from ..ads.variants import json_safe
            plan_path.write_text(json.dumps(json_safe(plan), default=str, indent=2,
                                            ensure_ascii=False), encoding="utf-8")
            return str(plan_path), {"scene_count": len(plan.scenes),
                                    "duration": sum(s.duration for s in plan.scenes)}
        return self._run_with_retry(proj, job, fn)

    def run_scene_resolution(self, project_id: str, job_id: str) -> Job:
        proj = self.get_project(project_id)
        job = proj.store.require(job_id)
        # Lazy-load the plan from plan.json if not in memory (e.g. after a
        # cache-reuse of CONTENT_PLAN, which bypasses fn() that sets proj.plan).
        if proj.plan is None:
            plan_path = proj.assets_dir / "plan.json"
            if plan_path.exists():
                from ..brain.models import ProductionPlan
                try:
                    proj.plan = ProductionPlan.model_validate_json(plan_path.read_text("utf-8"))
                except Exception:
                    pass
        # Lazy-load scene contexts by re-running resolution if not in memory
        # (e.g. after a cache-reuse of SCENE_RESOLUTION, which bypasses fn()).
        # resolve_all_scenes is deterministic from the plan, so this is safe.
        if not proj.scene_contexts and proj.plan is not None:
            try:
                ctxs = resolve_all_scenes(proj.plan)
                for i, ctx in enumerate(ctxs, start=1):
                    proj.scene_contexts[i] = ctx
            except Exception:
                pass
        if proj.plan is None:
            return self._fail(proj, job, TypedErrorCode.DEPENDENCY_MISSING.value,
                              "ProductionPlan not available; run CONTENT_PLAN first.")

        def fn():
            ctxs = resolve_all_scenes(proj.plan)
            for i, ctx in enumerate(ctxs, start=1):
                proj.scene_contexts[i] = ctx
            ctx_path = proj.assets_dir / "scene_contexts.json"
            ctx_path.write_text(json.dumps([c.to_dict() for c in ctxs], default=str,
                                           indent=2, ensure_ascii=False), encoding="utf-8")
            return str(ctx_path), {"resolved_scenes": len(ctxs)}
        return self._run_with_retry(proj, job, fn)

    def run_video_scene(self, project_id: str, job_id: str) -> Job:
        """Provider-aware video generation. Tries AI generation (ComfyUI/Wan)
        first; on NO_PROVIDER, falls back to real stock footage (Wikimedia/
        Pexels/Pixabay) so a real video always flows into assembly when any
        stock provider is available. Never fakes a video."""
        proj = self.get_project(project_id)
        job = proj.store.require(job_id)
        if not self.limiter.try_acquire(JobType.VIDEO_SCENE):
            return self._fail(proj, job, TypedErrorCode.CONCURRENCY_LIMIT.value,
                              f"Video concurrency limit reached ({self.limiter.max_video}).")
        try:
            def fn():
                scene_index = job.inputs.get("scene_index")
                ctx = proj.scene_contexts.get(scene_index)
                prompt = job.inputs.get("prompt") or (
                    ctx.visual_prompt if ctx else job.inputs.get("text", ""))
                width = job.inputs.get("width", 1080)
                height = job.inputs.get("height", 1920)
                duration = job.inputs.get("duration", 4.0)
                # 1) Try AI video generation (ComfyUI/Wan).
                try:
                    from ..providers.registry import select_provider
                    from ..providers.base import GenerationRequest
                    req = GenerationRequest(
                        prompt=prompt,
                        negative_prompt=job.inputs.get("negative_prompt", ""),
                        duration=duration,
                        width=width,
                        height=height,
                        seed=job.seed,
                    )
                    loop = asyncio.new_event_loop()
                    try:
                        provider = loop.run_until_complete(select_provider())
                        job.provider = provider.info.name
                        result = loop.run_until_complete(
                            provider.generate(req, output_dir=proj.assets_dir))
                    finally:
                        loop.close()
                    return str(result.path), {"prompt_id": getattr(result, "prompt_id", None),
                                              "source": "ai_generation"}
                except VideoError as e:
                    if e.code is not TypedErrorCode.NO_PROVIDER:
                        raise
                    # 2) Stock footage fallback (Wikimedia/Pexels/Pixabay).
                    log("STOCK_FALLBACK", f"AI video provider unavailable for scene {scene_index}; "
                        "using real stock footage", provider_hint="stock")
                    return self._stock_video_fallback(proj, job, prompt, width, height, duration)
            return self._run_with_retry(proj, job, fn)
        finally:
            self.limiter.release(JobType.VIDEO_SCENE)

    def _stock_video_fallback(self, proj, job, prompt: str,
                              width: int, height: int, duration: float) -> tuple[str, dict]:
        """Download a real stock video for the scene. Verifies the MP4 and
        records full license provenance. Raises NO_PROVIDER when no stock
        provider is available; raises NO_OUTPUT when search yields no hits."""
        from ..providers.stock import StockMediaType, StockOrientation, StockSearchRequest
        from ..providers.stock_adapters import build_stock_providers
        providers = [p for p in build_stock_providers() if p.available]
        if not providers:
            raise VideoError(
                TypedErrorCode.NO_PROVIDER,
                "No video provider available: AI generation not configured AND no stock "
                "provider available. Configure ComfyUI/Wan or a stock provider "
                "(Wikimedia is keyless).",
                context={"ai_provider": "NO_PROVIDER", "stock_providers": []},
            )
        # Infer orientation from target dimensions.
        if height > width * 1.2:
            orient = StockOrientation.PORTRAIT
        elif width > height * 1.2:
            orient = StockOrientation.LANDSCAPE
        else:
            orient = None
        # Extract candidate search queries from the structured visual prompt.
        # Stock search works best with short visual noun phrases; the prompt may
        # be a long [IDENTITY]...[ACTION]... string with ad copy that stock
        # search can't match. Build a list of candidate queries (best first).
        candidates = self._extract_stock_queries(prompt)
        # Always include a generic fallback last.
        if not candidates or candidates[-1] != "nature landscape":
            candidates.append("nature landscape")
        last_err = None
        for provider in providers:
            for query in candidates:
                req = StockSearchRequest(
                    query=query,
                    media_type=StockMediaType.VIDEO,
                    orientation=orient,
                    min_duration=max(1.0, duration - 1.0),
                    per_page=10, page=1, language="en",
                )
                try:
                    hits = provider.search(req)
                except VideoError as e:
                    last_err = e
                    continue
                # Filter to hits that allow commercial use (free-first policy).
                usable = [h for h in hits
                          if h.license_commercial_use in ("allowed", "unknown")]
                if not usable:
                    continue
                hit = usable[0]
                dest = proj.assets_dir / f"scene_{job.inputs.get('scene_index', 0)}_{hit.asset_id}.mp4"
                result = provider.download(hit, destination=dest)
                # Verify the downloaded file is a real, decodable MP4.
                verify_mp4(result.path)
                job.provider = provider.name
                log("STOCK_VIDEO", f"downloaded {result.path.name}",
                    provider=provider.name, license=hit.license_name,
                    bytes_size=result.bytes_size, query=query)
                return str(result.path), {
                    "source": "stock_footage",
                    "stock_provider": provider.name,
                    "stock_asset_id": hit.asset_id,
                    "stock_page_url": hit.page_url,
                    "license_name": hit.license_name,
                    "license_commercial_use": hit.license_commercial_use,
                    "attribution_required": hit.attribution_required,
                    "attribution_text": hit.attribution_text,
                    "author": hit.author,
                    "sha256": result.sha256,
                    "query": query,
                }
        # No provider/query yielded usable hits.
        raise VideoError(
            TypedErrorCode.NO_OUTPUT,
            f"Stock video search yielded no usable hits (tried {len(candidates)} queries). "
            f"Last error: {last_err}" if last_err else
            f"Stock video search yielded no usable hits (tried {len(candidates)} queries).",
            context={"queries": candidates, "original_prompt": prompt[:200],
                     "orientation": orient.value if orient else None},
        )

    @staticmethod
    def _extract_stock_queries(prompt: str) -> list[str]:
        """Extract candidate stock-search queries from a structured visual prompt.

        Returns a list of short noun phrases ordered by visual relevance.
        Stock search (Wikimedia/Pexels) needs simple terms, but scene continuity
        prompts are long structured strings with ad copy. We extract from
        ENVIRONMENT/DETAILS/ACTION sections and also try the raw idea keywords.
        """
        import re
        # Non-visual filler words to strip.
        filler = {
            "consistent", "environment", "for", "the", "a", "an", "image",
            "introducing", "showing", "shown", "stable", "packaging", "colors",
            "signature", "across", "all", "scenes", "unless", "narrative",
            "explicitly", "changes", "it", "product", "character", "subject",
            "same", "facial", "features", "build", "wardrobe", "clothing",
            "human", "striking", "curiosity", "driving", "powerful", "hook",
            "app", "relaxation",
        }
        queries: list[str] = []

        def _clean(text: str) -> str:
            text = re.sub(r"\([^)]*\)", "", text)
            text = re.sub(r"^(?:Powerful hook|Hook|CTA|Call to action|Reveal|"
                          r"Climax|Payoff|Button|Logo|Tagline|motion|transition)\s*:?\s*",
                          "", text, flags=re.IGNORECASE)
            words = [w.strip(" ,.;:()'\"") for w in text.split()]
            words = [w for w in words if w and w.lower() not in filler
                     and not w.startswith("'") and len(w) > 1]
            return " ".join(words[:6]).strip()

        for tag in ("ENVIRONMENT", "DETAILS", "ACTION"):
            m = re.search(rf"\[{tag}\]\s*(.+?)(?:\[|$)", prompt, re.IGNORECASE | re.DOTALL)
            if m:
                q = _clean(m.group(1))
                if len(q) >= 3 and q not in queries:
                    queries.append(q)
        # Also try extracting noun-like keywords from the whole prompt.
        clean = re.sub(r"\[[A-Z]+\]", "", prompt)
        q = _clean(clean)
        if len(q) >= 3 and q not in queries:
            queries.append(q)
        return queries

    def run_voiceover(self, project_id: str, job_id: str) -> Job:
        """Provider-aware TTS. Uses the free-first router to select a real TTS
        provider (Piper when enabled). FAILS with NO_PROVIDER when none is
        configured — never fakes success with deterministic test audio."""
        proj = self.get_project(project_id)
        job = proj.store.require(job_id)
        if not self.limiter.try_acquire(JobType.VOICEOVER):
            return self._fail(proj, job, TypedErrorCode.CONCURRENCY_LIMIT.value,
                              f"Audio concurrency limit reached ({self.limiter.max_audio}).")
        try:
            def fn():
                from ..providers.router import build_tts_providers
                from ..voice.tts import NullTTSProvider, select_tts_provider
                providers = build_tts_providers()
                provider = select_tts_provider(providers)
                if isinstance(provider, NullTTSProvider):
                    raise VideoError(
                        TypedErrorCode.NO_PROVIDER,
                        "No real TTS provider configured. Enable Piper "
                        "(PIPER_ENABLED=true, GPL-3.0) for free local TTS.",
                        context={"language": job.inputs.get("language", "en")})
                # A real provider is available: synthesize the scene voiceover.
                scene_index = int(job.inputs.get("scene_index", 0))
                language = job.inputs.get("language", "en")
                text = job.inputs.get("text", "").strip()
                if not text:
                    raise VideoError(
                        TypedErrorCode.WORKFLOW_INVALID,
                        "Voiceover job has no text to synthesize.",
                        context={"scene_index": scene_index})
                from ..voice.tts import VoiceRequest
                from ..audio.qc import verify_audio
                out_dir = OUTPUT_DIR / project_id / f"scene_{scene_index}" / "voice"
                out_dir.mkdir(parents=True, exist_ok=True)
                fmt = job.inputs.get("output_format", "wav")
                dest = out_dir / f"scene_{scene_index}_voice.{fmt}"
                request = VoiceRequest(text=text, language=language, output_format=fmt)
                result = provider.synthesize(request, dest)
                # QC: verify the audio is real and decodable before reporting success.
                qc = verify_audio(result.path)
                if not qc.get("ok"):
                    raise VideoError(
                        TypedErrorCode.INVALID_AUDIO,
                        f"TTS output failed QC: {qc.get('error', 'failed')}",
                        context={"path": str(result.path)})
                job.provider = provider.name
                outs = {
                    "path": str(result.path), "duration": result.duration,
                    "sample_rate": result.sample_rate, "voice": result.voice,
                    "language": result.language, "provider": provider.name,
                }
                return str(result.path), outs
            return self._run_with_retry(proj, job, fn)
        finally:
            self.limiter.release(JobType.VOICEOVER)

    def run_assembly(self, project_id: str, job_id: str) -> Job:
        """Real FFmpeg final assembly using the Phase 10 engine."""
        proj = self.get_project(project_id)
        job = proj.store.require(job_id)

        def fn():
            # Collect completed VIDEO_SCENE jobs in scene order (inside fn so the
            # dependency check in _run_with_retry runs first, surfacing
            # DEPENDENCY_FAILED before DEPENDENCY_MISSING).
            video_jobs = sorted(
                [j for j in proj.store.by_type(JobType.VIDEO_SCENE.value)
                 if j.state is JobState.COMPLETED and j.output_path],
                key=lambda j: j.scene_index or 0)
            if not video_jobs:
                raise VideoError(TypedErrorCode.DEPENDENCY_MISSING.value,
                                 "No completed VIDEO_SCENE jobs to assemble.")
            durations = [j.inputs.get("duration", 4.0) for j in video_jobs]
            videos = [Path(j.output_path) for j in video_jobs]
            # Collect completed VOICEOVER jobs (optional — only when not silent).
            silent = job.inputs.get("silent", True)
            voice_assets = None
            caption_texts = None
            if not silent:
                voice_jobs = {int(j.scene_index or 0): j for j in
                              proj.store.by_type(JobType.VOICEOVER.value)
                              if j.state is JobState.COMPLETED and j.output_path}
                voice_assets = []
                caption_texts = []
                for vj in video_jobs:
                    si = int(vj.scene_index or 0)
                    vj_voice = voice_jobs.get(si)
                    voice_assets.append(Path(vj_voice.output_path) if vj_voice else None)
                    caption_texts.append(vj.inputs.get("text", ""))
            tl = build_timeline_from_assets(durations, videos,
                                            voice_assets=voice_assets,
                                            caption_texts=caption_texts)
            profile_name = job.inputs.get("profile_name", "TIKTOK")
            req = ExportRequest(timeline=tl, profile_name=profile_name,
                                quality=Quality(job.inputs.get("quality", "high")),
                                silent=silent,
                                project_id=project_id)
            out_dir = proj.assets_dir / "final"
            result = export_video(req, output_dir=out_dir)
            if result.status != "COMPLETED":
                raise VideoError(TypedErrorCode.FINAL_QC_FAILED,
                                 result.error_detail or "Assembly failed.",
                                 context={"error_code": result.error_code})
            return result.output_path, {"qc": result.qc, "resolution": result.resolution,
                                         "duration": result.duration}
        return self._run_with_retry(proj, job, fn)

    # --------------------------------------------------------------- recovery
    def recover_project(self, project_id: str) -> list[Job]:
        """Startup recovery: detect jobs left RUNNING/QUEUED/RETRYING after a
        crash. Do NOT blindly mark them successful. Re-validate outputs; if a
        valid output exists, mark COMPLETED; otherwise mark FAILED (needs retry).
        """
        proj = self.get_project(project_id)
        recovered: list[Job] = []
        for job in proj.store.all():
            if job.state in (JobState.RUNNING, JobState.QUEUED, JobState.RETRYING):
                if job.output_path and Path(job.output_path).exists() and \
                   Path(job.output_path).stat().st_size > 0:
                    # Re-validate the output integrity.
                    try:
                        self._validate_output(job, job.output_path)
                        # Re-fingerprint to ensure content matches.
                        fp = file_fingerprint(Path(job.output_path))
                        if job.output_fingerprint and fp != job.output_fingerprint:
                            raise VideoError(TypedErrorCode.CACHE_INVALID,
                                             "Output fingerprint mismatch on recovery.")
                        job.state = JobState.COMPLETED
                        job.completed_at = time.time()
                        log("JOB", "recovered_completed", job_id=job.job_id)
                    except VideoError as e:
                        job.state = JobState.FAILED
                        job.error_code = e.code.value
                        job.error_detail = f"Recovery validation failed: {e.detail}"
                        log("JOB", "recovered_failed", job_id=job.job_id, code=e.code.value)
                else:
                    job.state = JobState.FAILED
                    job.error_code = TypedErrorCode.NO_OUTPUT.value
                    job.error_detail = "Job was interrupted and produced no valid output."
                    log("JOB", "recovered_failed", job_id=job.job_id, code="NO_OUTPUT")
                proj.store.put(job)
                recovered.append(job)
        return recovered

    def recover_all(self) -> dict[str, list[Job]]:
        """Recover all known projects on startup."""
        out: dict[str, list[Job]] = {}
        for pid in self.list_projects():
            out[pid] = self.recover_project(pid)
        return out

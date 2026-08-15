# vid-dvd — Agent Memory

## Project
AI Video & Advertising Factory. Repository: `smsm9919/vid-dvd`, branch `main`.

## Development Workflow (MANDATORY)
TEST → VERIFY → COMMIT → PUSH for every phase/step. Only commit+push when all
tests pass. Push to `main` at `smsm9919/vid-dvd`. Never push broken/untested code.
Distinguish CODE VERIFIED / TEST VERIFIED / RUNTIME VERIFIED. No fake runtime
verification (no mocks/fake MP4/placeholders as runtime success).

## Environment
- Python 3.13, pydantic 2.13, pytest 9.1, httpx. FFmpeg 7.1.5 + ffprobe installed.
- GITHUB_TOKEN available for pushes/API. Git identity: openhands / openhands@all-hands.dev.
- Commit messages: conventional `feat(scope): ...`, include `Co-authored-by: openhands <openhands@all-hands.dev>`.
- Commits in small logical units; never one giant commit.

## Architecture (current)
- `app/core/` — errors.py (TypedErrorCode 9 codes + VideoError), logging.py
- `app/providers/` — base.py (ABC), comfyui.py (14-step verify), registry.py (NO_PROVIDER when none)
- `app/comfy.py` — backwards-compat shim re-exporting ComfyClient/ComfyError
- `app/media.py` — re-encoding concat (no -c copy), verify_mp4, probe, typed MediaError
- `app/brain/` — models.py (ProductionPlan schema, 4 layers), content_brain.py (local + Gemini fallback)
- `app/ads/` — brief.py (AdBrief), claims.py (claim safety, never fabricates proof), variants.py (7 variants), scoring.py (9-dim heuristic)
- `app/main.py` — FastAPI routes: /, /api/health, /api/plan, /api/projects, /api/ads/variants[/{key}], generate, download

## Key Invariants
- ProductionPlan validator requires hook scene + cta scene for advertisement mode.
- Claim safety: proof only from approved facts; else "requires verification, no claim fabricated".
- Phase 6 scoring is a heuristic, NOT a conversion predictor.

## Phases Status
- Phase 3-4 (typed errors, providers, media): DONE, pushed
- Phase 5 (content brain): DONE, pushed
- Phase 6 (ads + variants): DONE, pushed (commit 0f2024e)
- Phase 7 (scene continuity + references): DONE, pushed
- Phase 8 (Wan 2.2 T2V/I2V + reference workflows): DONE, pushed
- Phase 9 (voice, audio, music, SFX, captions): DONE, pushed
- Phase 10 (professional video editing + final assembly): DONE, pushed
- Phase 11 (production job orchestration + recovery): DONE, pushed
- Phase 12+ : NOT STARTED (awaiting approval)

## Test Suite
- 503 tests passing. Run: `python -m pytest tests/ -q`
- FFmpeg editing/assembly runtime-verified (real FFmpeg 7.1.5 + libass).
- Job orchestrator runtime-verified (real multi-scene project, state transitions,
  caching/idempotency, dependency failure, recovery/resume, real FFmpeg assembly).
- TTS/music/SFX/Wan provider runtime NOT VERIFIED (no external provider
  configured; jobs FAIL with NO_PROVIDER, never fake success).

## Phase 11 Architecture
- `app/jobs/state.py`: JobState enum (18 states: DRAFT→PLANNING→PLANNED→
  SCENE_RESOLUTION→SCENES_READY→VIDEO_GENERATION→VIDEO_GENERATED→AUDIO_GENERATION
  →AUDIO_READY→ASSEMBLY→QUALITY_CONTROL→COMPLETED; FAILED, RETRYING, QUEUED,
  RUNNING, CANCEL_REQUESTED→CANCELED). FORWARD map validates every transition.
  assert_transition raises INVALID_TRANSITION. is_terminal/is_active/can_retry/
  can_cancel helpers.
- `app/jobs/models.py`: Job (job_id, project_id, job_type, state, parent_job_id,
  scene_index, timestamps, retry_count, max_retries, provider/model/workflow/seed,
  input_fingerprint, output_fingerprint, output_path, error_code/detail,
  cancel_requested, provider_canceled, inputs, outputs, depends_on, last_backoff).
  JobType (10 types: CONTENT_PLAN, AD_VARIANTS, SCENE_RESOLUTION, VIDEO_SCENE,
  VOICEOVER, MUSIC, SFX, CAPTIONS, ASSEMBLY, FINAL_QC). fingerprint() (SHA-256,
  order-stable), file_fingerprint() (size+content hash), is_retryable/
  is_non_retryable (RETRYABLE_CODES/NON_RETRYABLE_CODES sets; unknown defaults
  non-retryable). to_dict/from_dict serialization.
- `app/jobs/retry.py`: RetryPolicy (base_delay, max_delay, max_retries, jitter;
  compute_delay = bounded 2^n + jitter). should_retry (retryable + retries left),
  record_retry (increments count, raises NON_RETRYABLE for deterministic errors,
  RETRY_EXHAUSTED when exceeded), classify_error.
- `app/jobs/cache.py`: CacheStore (file-backed JSON index; lookup re-validates:
  exists/size>0/fingerprint match/metadata match/QC via verify_mp4/verify_audio;
  evicts invalid entries; never trusts blindly). store() validates before caching.
- `app/jobs/persistence.py`: JobStore (PROJECTS_DIR/<pid>/jobs.json, atomic write
  via temp+rename, survives restart, rehydrates on load, creates empty index on
  init so projects are discoverable). ProjectRegistry (lists projects with
  jobs.json). Corrupt JSON starts fresh (no crash).
- `app/jobs/dependencies.py`: check_dependencies (COMPLETED=ready, FAILED=blocked,
  non-terminal=pending, missing=error), assert_dependencies (raises
  DEPENDENCY_FAILED/DEPENDENCY_MISSING). compute_progress (from actual job states,
  no fake %; overall=completed/total; stage=first in-progress; scene="N/M").
  ConcurrencyLimiter (MAX_CONCURRENT_VIDEO_JOBS=2, MAX_CONCURRENT_AUDIO_JOBS=4,
  conservative defaults; try_acquire/release).
- `app/jobs/orchestrator.py`: Orchestrator (connects Content Brain→scene
  continuity→Wan provider→voice→editing→QC without rewriting them). _run_with_retry
  (cache check→QUEUED→RUNNING→fn()→validate output→cache→COMPLETED; retry on
  retryable; fail on non-retryable). run_content_plan (local_content_plan, real
  plan.json), run_scene_resolution (resolve_all_scenes), run_video_scene
  (provider-aware: select_provider raises NO_PROVIDER — never fakes), run_voiceover
  (NullTTSProvider → NO_PROVIDER), run_assembly (real FFmpeg export_video +
  final_qc). recover_project (RUNNING/QUEUED/RETRYING → re-validate output →
  COMPLETED or FAILED+NO_OUTPUT; never blindly marks abandoned jobs successful).
  cancel_job (CANCEL_REQUESTED→CANCELED; provider_canceled=None when external
  provider not actually stopped). retry_job (FAILED→RETRYING→QUEUED).
- `app/core/errors.py`: added 10 Phase 11 codes: INVALID_TRANSITION,
  DEPENDENCY_FAILED, DEPENDENCY_MISSING, JOB_NOT_FOUND, CACHE_INVALID,
  RETRY_EXHAUSTED, NON_RETRYABLE, CANCEL_NOT_ALLOWED, CONCURRENCY_LIMIT,
  PERSISTENCE_ERROR (additive).
- API: 12 routes under /api/jobs/ (create/list/get projects, create/list/get/run/
  retry/cancel jobs, progress, output, recover). Existing routes preserved.
- Runtime verified: real 3-scene project (CONTENT_PLAN→SCENE_RESOLUTION→3×VIDEO_SCENE
  →ASSEMBLY COMPLETED, 1080×1920, 12.0s, QC True); idempotency (cache reuse);
  dependency failure (DEPENDENCY_FAILED); recovery (RUNNING→FAILED+NO_OUTPUT);
  resumability (assembly after restart reuses cached videos); retry (retryable
  QUEUED, non-retryable raises); cancel (CANCELED); NO_PROVIDER for voice/video
  (never fake). Live API: all 12 endpoints verified.

## Phase 10 Architecture
- `app/editing/timeline.py`: TimelineScene (scene_index, start, duration,
  video/voice/music/sfx/ambience/caption/transition/brand), Timeline,
  validate_timeline (sequential/contiguous/non-negative/no asset beyond scene/
  transition fits), build_timeline_from_assets (first scene forced cut).
- `app/editing/profiles.py`: ExportProfile (configurable width/height/fps/codec/
  pixfmt/audio/bitrate/crf/preset/aspect), TIKTOK/INSTAGRAM_REELS/YOUTUBE_SHORTS/
  YOUTUBE/SQUARE, Quality (low/medium/high → crf/preset), get_profile/register.
- `app/editing/transitions.py`: TransitionType (cut/fade/crossfade/dip_to_black/
  wipe), parse_transition (keyword → type+default_duration), TransitionSpec.validate
  (bad type/negative/xfade-too-long), transition_filter (xfade/fadeblack/wipeleft).
- `app/editing/compositor.py`: _normalize_scene_video (scale+pad preserve aspect,
  fps, pixfmt, fade-in), _concat_scenes (re-encode concat), _build_ass_from_cues
  (ASS script, Arabic/DE preserved), _burn_captions (subtitles filter; CAPTION_RENDER_ERROR
  if libass unavailable), _apply_branding (logo overlay + drawtext CTA/watermark),
  _mux_audio (apad+atrim to video duration — no silent video truncation), _silent_audio.
- `app/editing/assembly.py`: ExportRequest, ExportResult, export_video (validate
  timeline+assets → normalize → concat → burn captions → branding → audio → final QC;
  never COMPLETED without QC-verified MP4; never overwrites source), validate_scene_assets.
- `app/editing/qc.py`: final_qc (exists/size/duration/res/fps/codec/pixfmt/video+
  audio stream/audio duration sync/profile conformance; FINAL_QC_FAILED).
- `app/core/errors.py`: added MISSING_VIDEO_ASSET, MISSING_AUDIO_ASSET,
  INVALID_TIMELINE, UNSUPPORTED_PROFILE, TRANSITION_ERROR, CAPTION_RENDER_ERROR,
  EXPORT_FAILED, FINAL_QC_FAILED (additive).
- Routes: /api/assembly/export, /api/assembly/profiles (existing routes preserved).
- FFmpeg 7.1.5 with libass/fontconfig/freetype — burned-in captions render (incl.
  Arabic RTL + German umlauts), frame-hash-verified.
- A/V sync: short audio padded (apad), long audio trimmed (atrim) to video length.

## Phase 9 Architecture
- `app/voice/tts.py`: TTSProvider ABC, VoiceRequest (validated), VoiceResult,
  NullTTSProvider (NO_PROVIDER), select_tts; en/de/ar.
- `app/voice/voiceover.py`: build_voice_request (language propagation),
  validate_voice_timing (ok/warning/error + strategy, no silent truncation),
  generate_scene/project_voiceover (QC-verified).
- `app/audio/qc.py`: verify_audio/probe_audio (INVALID_AUDIO), deterministic
  test audio generators (silent/tone — NOT real provider output).
- `app/audio/music.py`: MusicProvider ABC, MusicRequest, 8 moods, parse_mood.
- `app/audio/sfx.py`: SFXProvider ABC, SFXRequest, 10 categories, parse_sfx.
- `app/audio/mixer.py`: FFmpeg mix_audio (per-track volume, ducking, fades,
  loudnorm, alimiter clipping prevention), mix_scene_audio convenience.
- `app/captions/captions.py`: SRT/VTT/burned_in, TikTok/Reels/Shorts/YouTube
  styles, validate_caption_timing (negative/overlap/out-of-range), Arabic-safe.
- `app/core/errors.py`: added INVALID_AUDIO (additive).
- Routes: /api/voice/generate, /api/audio/mix, /api/captions/generate, /api/audio/qc.
- FFmpeg filter chain rule: input label + first filter no comma; [mixed]chain[out].

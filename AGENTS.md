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
- Phase 9+ : NOT STARTED (awaiting approval)

## Test Suite
- 252 tests passing. Run: `python -m pytest tests/ -q`
- Tests are CODE/TEST verified. No real ComfyUI + Wan + GPU runtime in this env
  (COMFYUI_RUNTIME_BLOCKED). Real runtime verification requires real ComfyUI +
  Wan 2.2 weights + GPU.

## Phase 8 Architecture
- `app/providers/wan.py`: WanProvider(VideoProvider) composing ComfyUIProvider for
  transport; separates WAN LOGIC (workflow/model/reference/option validation,
  I2V image handling) from COMFYUI TRANSPORT (HTTP queue/poll/download/verify).
  GenerationOptions (range-validated), GenerationMetadata, ReadinessReport.
- `workflows/wan22_t2v_api.json`, `workflows/wan22_i2v_api.json`: documented
  adapter TEMPLATES (NOT fake workflows); validate required node classes; fail
  WORKFLOW_NOT_FOUND/WORKFLOW_INVALID. See workflows/README.md.
- `app/core/errors.py`: added WORKFLOW_NOT_FOUND + INVALID_REFERENCE (additive).
- `app/providers/registry.py`: supports wan + comfyui; reads VIDEO_PROVIDERS dynamically.
- `app/config.py`: WAN_T2V_WORKFLOW, WAN_I2V_WORKFLOW, WAN_REQUIRED_MODEL.
- Routes: /api/diagnose (READY/NOT_READY + blockers), /api/generate (T2V/I2V,
  never fake success, returns FAILED + error_code).
- Generate path consumes ResolvedSceneContext (Phase 7) — never a bare prompt.
- I2V reference validation is fail-fast (before any network call).

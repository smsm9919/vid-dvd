# VideoFactory Local

A free-first AI Video & Advertising Factory. Produces real, QC-verified MP4
videos using only free providers (no GPU, no paid API keys required for the
default path).

**Two production paths:**

1. **Free-first hub (default, Phase 13)** — produces real MP4s with voiceover
   using Wikimedia Commons stock video + Kokoro-82M TTS + FFmpeg assembly.
   Works on any machine with network access and FFmpeg. No GPU, no API keys.
2. **AI generation path (ComfyUI/Wan 2.2)** — generates original video clips
   via ComfyUI when a GPU is available. Falls back to the free-first path
   automatically when no AI provider is configured.

Pipeline: Topic/script → content plan → scene resolution → video (stock or AI)
→ voiceover (Kokoro) → FFmpeg assembly (concat + captions + audio mux + QC)
→ final MP4.

It does NOT auto-publish to TikTok or YouTube.

## Important
This package is the orchestration layer. AI model weights are intentionally NOT
bundled because they are several GB and have separate licenses. For the free-first
path, only the Kokoro model files (~100MB) are needed; for the AI generation path,
install ComfyUI and the chosen model separately.

## Requirements
- Python 3.11+ (3.13 tested)
- FFmpeg 7.x on PATH (with libass for burned-in captions)
- Network access (for Wikimedia Commons stock video)

**Free-first providers (no API key needed):**
- Wikimedia Commons stock video — keyless, CC0/CC BY/CC BY-SA licensed
- Kokoro-82M TTS — local CPU, Apache 2.0 weights (download ~100MB model)
- FFmpeg assembly — local, real h264/aac MP4 output with QC

**Optional paid/AI providers:**
- ComfyUI + Wan 2.2 5B (GPU, original AI video generation)
- Pexels/Pixabay stock (free API key, higher-quality stock)
- Gemini (stronger scene planning)

## Setup (free-first path)
1. Install Python 3.11+ and FFmpeg (with libass).
2. `pip install -r requirements.txt`
3. `pip install kokoro-onnx soundfile`  (TTS runtime)
4. Download Kokoro model files:
   ```bash
   mkdir -p models/kokoro
   curl -L -o models/kokoro/kokoro-v1.0.int8.onnx \
     https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx
   curl -L -o models/kokoro/voices-v1.0.bin \
     https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
   ```
5. `KOKORO_ENABLED=true uvicorn app.main:app --port 8090`
6. Open http://127.0.0.1:8090/dashboard

## Setup (AI generation path)
1. Install/run ComfyUI at http://127.0.0.1:8188 with a GPU.
2. In ComfyUI: Workflow → Browse Templates → Video → "Wan2.2 5B video generation".
3. Install/download the model files requested by the official workflow.
4. Export that workflow as **API format** and save it as `workflows/wan22_ti2v_api.json`.
5. Run the app as above; the orchestrator will use AI generation and fall back
   to stock when the AI provider is unavailable.

Copy `.env.example` to `.env` for custom settings.

## Honest verification status
All Python source files were syntax-compiled with `py_compile`; no secrets are
hard-coded; the ComfyUI client follows the documented `/prompt`,
`/history/{prompt_id}` and `/view` API pattern; and scene planning works without
any external API. The free-first path was runtime-verified end-to-end: real
Wikimedia stock video → real Kokoro TTS → real FFmpeg assembly → QC-verified
MP4 (h264/aac, 1080×1920). No mocks, fakes, or synthetic assets were used in
the verification run.

A literal "100% works on every machine" claim would be false without the user's
exact GPU, ComfyUI version, installed model files and exported workflow. This
package is designed to fail clearly when those prerequisites are missing.

## Licensing
This project does not redistribute third-party model weights. Check every model
and media license before commercial use.

**Free-first provider licenses (verified):**
- **Wikimedia Commons** — per-file (CC0/Public Domain/CC BY/CC BY-SA).
  Commercial use allowed; attribution required for CC BY/CC BY-SA. Per-file
  license metadata is preserved on every downloaded asset.
  Source: https://commons.wikimedia.org/wiki/Commons:Licensing
- **Kokoro-82M** — Apache 2.0 model weights (commercial-safe, no copyleft).
  Source: https://huggingface.co/hexgrad/Kokoro-82M
- **kokoro-onnx runtime** — MIT.
  Source: https://github.com/thewh1teagle/kokoro-onnx
- **FFmpeg** — GPL/LGPL (configure flags determine exact license; the Debian
  build enables GPL components). Ensure your FFmpeg build's license is
  compatible with your distribution target.
- **Pexels** (optional) — Pexels License (commercial use allowed, no
  attribution required). Source: https://www.pexels.com/terms-of-service/
- **Pixabay** (optional) — Pixabay Content License (commercial use allowed).
  Source: https://pixabay.com/service/license-summary/
- **Piper TTS** (optional, opt-in) — GPL-3.0. Only enable if your project can
  comply with GPL-3.0. Source: https://github.com/OHF-Voice/piper1-gpl

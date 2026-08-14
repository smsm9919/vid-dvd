# VideoFactory Local

Windows-first local orchestrator for cinematic short videos.

Pipeline: Topic/script -> scene plan -> ComfyUI video clips -> FFmpeg assembly -> final MP4.

It does NOT auto-publish to TikTok or YouTube.

## Important
This package is the orchestration layer. AI model weights are intentionally NOT bundled because they are several GB and have separate licenses. You install ComfyUI and the chosen model separately.

Recommended first local target: **Wan 2.2 TI2V 5B in ComfyUI**. Current official ComfyUI documentation provides a native Wan 2.2 5B workflow and states it should fit around 8GB VRAM with native offloading. LTX-2 is also supported by ComfyUI, but its official LTXVideo extension currently documents 32GB+ VRAM for its LTX-2 workflows.

## Requirements
- Windows 10/11
- Python 3.11+
- FFmpeg on PATH
- ComfyUI running at http://127.0.0.1:8188
- NVIDIA GPU recommended for local video generation
- Wan 2.2 model files installed in ComfyUI

Optional: Gemini API key for stronger scene planning.

## Setup
1. Install Python 3.11+ and FFmpeg.
2. Install/run ComfyUI.
3. In ComfyUI: Workflow -> Browse Templates -> Video -> "Wan2.2 5B video generation".
4. Install/download the model files requested by the official workflow.
5. Export that workflow as **API format** and save it as `workflows/wan22_ti2v_api.json`.
6. Run `setup.bat`, then `run.bat`.
7. Open http://127.0.0.1:8090

Copy `.env.example` to `.env` if you need custom settings.

## Honest verification status
Before packaging, all Python source files were syntax-compiled with `py_compile`; no secrets are hard-coded; the ComfyUI client follows the documented `/prompt`, `/history/{prompt_id}` and `/view` API pattern; and scene planning works without any external API.

A literal "100% works on every machine" claim would be false without the user's exact GPU, ComfyUI version, installed model files and exported workflow. This package is designed to fail clearly when those prerequisites are missing.

## Licensing
This project does not redistribute third-party model weights. Check every model and media license before commercial use.

"""ComfyUI video provider.

Implements the :class:`~app.providers.base.VideoProvider` contract against a
local (or remote) ComfyUI instance. The generation path performs the full
verification sequence required by the production pipeline:

    1. Detect ComfyUI (reachability)
    2. Query system information
    3. Detect available models
    4. Validate required model weights
    5. Validate workflow JSON
    6. Submit the actual prompt
    7. Receive the real prompt/job id
    8. Poll the real job
    9. Detect completion
   10. Locate the generated video
   11. Verify the MP4 exists
   12. Verify file size > 0
   13. Verify the video can be read with FFmpeg
   14. Only then mark the job COMPLETED (return a GenerationResult)

Every failure maps to a typed :class:`~app.core.errors.VideoError`:

    COMFYUI_UNREACHABLE   - endpoint not reachable / not ComfyUI
    MODEL_NOT_FOUND       - required model weights absent
    WORKFLOW_INVALID      - workflow file missing or not API-format JSON
    WORKFLOW_REJECTED     - ComfyUI /prompt rejected the workflow
    GENERATION_TIMEOUT    - job did not finish in time
    NO_OUTPUT             - finished but produced no media
    INVALID_MP4           - output exists but is not a valid MP4
    FFMPEG_ERROR          - FFmpeg cannot read / probe the output

This provider NEVER simulates success. If any verification step fails, a typed
error is raised and no fake path is returned.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx

from ..core.errors import TypedErrorCode, VideoError
from ..core.logging import log
from .base import (
    GenerationRequest,
    GenerationResult,
    ProviderInfo,
    VideoProvider,
)


class ComfyError(VideoError):
    """Backwards-compatible ComfyUI error.

    Historically ``app.comfy.ComfyError`` subclassed ``RuntimeError``. To keep
    the typed-error guarantees while preserving the old public name, it now
    subclasses :class:`VideoError` (which itself subclasses ``RuntimeError``).
    Existing ``except ComfyError`` blocks keep working.
    """

    def __init__(self, code: TypedErrorCode, detail: str = "", *, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(code, detail, context=context)


class ComfyUIProvider(VideoProvider):
    """ComfyUI implementation of the VideoProvider contract."""

    def __init__(
        self,
        base_url: str,
        timeout: int = 1800,
        *,
        workflow_path: Optional[Path] = None,
        prompt_node_id: str = "",
        prompt_field: str = "text",
        required_models: Optional[list[str]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = int(timeout)
        self.workflow_path = workflow_path
        self.prompt_node_id = prompt_node_id.strip()
        self.prompt_field = prompt_field.strip() or "text"
        # Substrings of model filenames that must be present. Empty = skip the
        # required-models check (step 4) and only report what is available.
        self.required_models = list(required_models or [])

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="comfyui",
            kind="local",
            description="Local ComfyUI video generation (Wan / T2V / I2V workflows).",
        )

    # ------------------------------------------------------------------ transport
    def _client(self, timeout: Optional[float] = None) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout if timeout is not None else 30)

    async def _get(self, path: str, *, timeout: Optional[float] = None) -> httpx.Response:
        try:
            async with self._client(timeout) as c:
                return await c.get(f"{self.base_url}{path}")
        except httpx.RequestError as e:
            raise ComfyError(
                TypedErrorCode.COMFYUI_UNREACHABLE,
                f"Cannot reach ComfyUI at {self.base_url}: {e}",
                context={"url": self.base_url, "path": path},
            )

    # ------------------------------------------------------------------ step 1+2: detect + system info
    async def detect(self) -> bool:
        """Step 1: cheap reachability check. Never raises."""
        try:
            r = await self._get("/system_stats", timeout=10)
            return r.status_code == 200
        except VideoError:
            return False

    async def system_stats(self) -> dict[str, Any]:
        """Step 2: query system information (GPU/VRAM/system)."""
        r = await self._get("/system_stats", timeout=10)
        if r.status_code != 200:
            raise ComfyError(
                TypedErrorCode.COMFYUI_UNREACHABLE,
                f"/system_stats returned HTTP {r.status_code}",
                context={"url": self.base_url},
            )
        try:
            return r.json()
        except ValueError as e:
            raise ComfyError(
                TypedErrorCode.COMFYUI_UNREACHABLE,
                f"/system_stats returned non-JSON: {e}",
                context={"url": self.base_url},
            )

    async def health(self) -> dict[str, Any]:
        """Structured health report used by the registry and dashboard."""
        try:
            data = await self.system_stats()
            return {"ok": True, "data": data, "url": self.base_url}
        except ComfyError as e:
            return {"ok": False, "error": e.to_dict(), "url": self.base_url}

    # ------------------------------------------------------------------ step 3+4: models
    async def list_models(self) -> list[str]:
        """Step 3: detect available models via /object_info."""
        r = await self._get("/object_info", timeout=30)
        if r.status_code != 200:
            raise ComfyError(
                TypedErrorCode.COMFYUI_UNREACHABLE,
                f"/object_info returned HTTP {r.status_code}",
                context={"url": self.base_url},
            )
        try:
            data = r.json()
        except ValueError as e:
            raise ComfyError(
                TypedErrorCode.COMFYUI_UNREACHABLE,
                f"/object_info returned non-JSON: {e}",
            )
        models: list[str] = []
        # Collect every model-like input field across all nodes.
        for node in data.values():
            if not isinstance(node, dict):
                continue
            for field_name in ("unet", "checkpoint", "vae", "lora", "model"):
                inp = node.get("input", {}).get(field_name) if isinstance(node.get("input"), dict) else None
                if isinstance(inp, list):
                    models.extend(str(m) for m in inp if isinstance(m, str) and m)
        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique: list[str] = []
        for m in models:
            if m not in seen:
                seen.add(m)
                unique.append(m)
        return unique

    def _check_required_models(self, available: list[str]) -> list[str]:
        """Step 4: validate required model weights. Returns list of missing names."""
        avail_lower = [a.lower() for a in available]
        missing: list[str] = []
        for req in self.required_models:
            needle = req.lower()
            if not any(needle in a for a in avail_lower):
                missing.append(req)
        return missing

    # ------------------------------------------------------------------ step 5: workflow validation
    def load_workflow(self, path: Optional[Path] = None) -> dict[str, Any]:
        """Step 5: validate workflow JSON is ComfyUI API-format (a dict of nodes)."""
        path = path or self.workflow_path
        if path is None or not Path(path).exists():
            raise ComfyError(
                TypedErrorCode.WORKFLOW_INVALID,
                f"Workflow not found: {path}",
                context={"path": str(path) if path else None},
            )
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise ComfyError(
                TypedErrorCode.WORKFLOW_INVALID,
                f"Workflow file is not valid JSON: {e}",
                context={"path": str(path)},
            )
        if not isinstance(data, dict) or not data:
            raise ComfyError(
                TypedErrorCode.WORKFLOW_INVALID,
                "Workflow must be non-empty ComfyUI API-format JSON (dict of node_id -> node).",
                context={"path": str(path)},
            )
        return data

    def inject_prompt(
        self,
        workflow: dict[str, Any],
        prompt: str,
        node_id: str = "",
        field: str = "text",
    ) -> dict[str, Any]:
        """Inject the positive prompt into a copy of the workflow.

        Preserves the original auto-detection heuristic from the legacy client.
        Raises WORKFLOW_INVALID if the prompt node/field cannot be found.
        """
        wf = json.loads(json.dumps(workflow))
        if node_id:
            node = wf.get(str(node_id))
            if not node:
                raise ComfyError(
                    TypedErrorCode.WORKFLOW_INVALID,
                    f"Prompt node {node_id} not found in workflow.",
                    context={"node_id": node_id},
                )
            node.setdefault("inputs", {})[field] = prompt
            return wf
        candidates: list[tuple[int, str, str]] = []
        for nid, node in wf.items():
            if not isinstance(node, dict):
                continue
            cls = str(node.get("class_type", "")).lower()
            for key, val in node.get("inputs", {}).items():
                if isinstance(val, str) and key.lower() in {"text", "prompt", "positive", "positive_prompt"}:
                    score = 0 if ("positive" in key.lower() or "cliptext" in cls) else 1
                    candidates.append((score, nid, key))
        if not candidates:
            raise ComfyError(
                TypedErrorCode.WORKFLOW_INVALID,
                "Could not auto-detect a positive prompt field. "
                "Set COMFYUI_PROMPT_NODE_ID and COMFYUI_PROMPT_FIELD.",
            )
        candidates.sort()
        _, nid, key = candidates[0]
        wf[nid]["inputs"][key] = prompt
        return wf

    # ------------------------------------------------------------------ step 6+7: submit
    async def queue(self, workflow: dict[str, Any]) -> str:
        """Step 6+7: submit the prompt and return the real prompt_id."""
        client_id = str(uuid.uuid4())
        try:
            async with self._client(30) as c:
                r = await c.post(
                    f"{self.base_url}/prompt",
                    json={"prompt": workflow, "client_id": client_id},
                )
        except httpx.RequestError as e:
            raise ComfyError(
                TypedErrorCode.COMFYUI_UNREACHABLE,
                f"Cannot submit to ComfyUI: {e}",
                context={"url": self.base_url},
            )
        if r.status_code != 200:
            raise ComfyError(
                TypedErrorCode.WORKFLOW_REJECTED,
                f"ComfyUI /prompt rejected workflow (HTTP {r.status_code}): {r.text[:1000]}",
                context={"status": r.status_code},
            )
        try:
            data = r.json()
        except ValueError as e:
            raise ComfyError(
                TypedErrorCode.WORKFLOW_REJECTED,
                f"ComfyUI /prompt returned non-JSON: {e}",
            )
        if "prompt_id" not in data:
            raise ComfyError(
                TypedErrorCode.WORKFLOW_REJECTED,
                f"ComfyUI rejected workflow (no prompt_id): {data}",
                context={"response": data},
            )
        return data["prompt_id"]

    # ------------------------------------------------------------------ step 8+9: poll
    async def wait(self, prompt_id: str) -> dict[str, Any]:
        """Step 8+9: poll the real job until it completes or times out."""
        deadline = time.monotonic() + self.timeout
        async with self._client(30) as c:
            while time.monotonic() < deadline:
                try:
                    r = await c.get(f"{self.base_url}/history/{prompt_id}")
                except httpx.RequestError as e:
                    raise ComfyError(
                        TypedErrorCode.COMFYUI_UNREACHABLE,
                        f"Lost connection to ComfyUI while polling: {e}",
                        context={"prompt_id": prompt_id},
                    )
                if r.status_code != 200:
                    raise ComfyError(
                        TypedErrorCode.COMFYUI_UNREACHABLE,
                        f"/history/{prompt_id} returned HTTP {r.status_code}",
                        context={"prompt_id": prompt_id},
                    )
                item = r.json().get(prompt_id)
                if item:
                    status = item.get("status", {})
                    if status.get("status_str") == "error":
                        raise ComfyError(
                            TypedErrorCode.WORKFLOW_REJECTED,
                            f"ComfyUI job errored: {status}",
                            context={"prompt_id": prompt_id, "status": status},
                        )
                    outputs = item.get("outputs", {})
                    if outputs:
                        return outputs
                await asyncio.sleep(2)
        raise ComfyError(
            TypedErrorCode.GENERATION_TIMEOUT,
            f"Timed out after {self.timeout}s waiting for ComfyUI job {prompt_id}.",
            context={"prompt_id": prompt_id, "timeout": self.timeout},
        )

    # ------------------------------------------------------------------ step 10: locate output
    def collect_outputs(self, outputs: dict[str, Any]) -> list[dict[str, Any]]:
        """Step 10: locate generated media items from the job outputs."""
        candidates: list[dict[str, Any]] = []
        for out in outputs.values():
            if not isinstance(out, dict):
                continue
            for key in ("gifs", "videos", "images"):
                for item in out.get(key, []) or []:
                    if isinstance(item, dict) and item.get("filename"):
                        candidates.append(item)
        return candidates

    async def download_first_media(self, outputs: dict[str, Any], destination: Path) -> Path:
        """Download the first media item to ``destination``.

        Raises NO_OUTPUT if the job produced no media. (Steps 10–12 of existence
        + size verification are completed by :meth:`generate`'s verify step.)
        """
        candidates = self.collect_outputs(outputs)
        if not candidates:
            raise ComfyError(
                TypedErrorCode.NO_OUTPUT,
                "ComfyUI finished but returned no media output.",
                context={"outputs": outputs},
            )
        item = candidates[0]
        params = {
            "filename": item["filename"],
            "subfolder": item.get("subfolder", ""),
            "type": item.get("type", "output"),
        }
        try:
            async with self._client(120) as c:
                r = await c.get(f"{self.base_url}/view", params=params)
        except httpx.RequestError as e:
            raise ComfyError(
                TypedErrorCode.COMFYUI_UNREACHABLE,
                f"Cannot download output from ComfyUI: {e}",
                context={"params": params},
            )
        if r.status_code != 200:
            raise ComfyError(
                TypedErrorCode.NO_OUTPUT,
                f"/view returned HTTP {r.status_code} for {params}",
                context={"params": params, "status": r.status_code},
            )
        destination = Path(destination)
        destination.write_bytes(r.content)
        return destination

    # ------------------------------------------------------------------ step 11-13: verify
    def verify_output(self, path: Path) -> None:
        """Steps 11–13: verify the MP4 exists, is non-empty, and FFmpeg-readable.

        Raises INVALID_MP4 / FFMPEG_ERROR as appropriate. Requires FFmpeg on PATH.
        """
        path = Path(path)
        # Step 11: exists
        if not path.exists():
            raise ComfyError(
                TypedErrorCode.INVALID_MP4,
                f"Output file does not exist: {path}",
                context={"path": str(path)},
            )
        # Step 12: size > 0
        size = path.stat().st_size
        if size <= 0:
            raise ComfyError(
                TypedErrorCode.INVALID_MP4,
                f"Output file is empty (0 bytes): {path}",
                context={"path": str(path), "size": size},
            )
        # Step 13: FFmpeg readability
        if not shutil.which("ffprobe"):
            # FFmpeg missing is an environment problem, not a fake success.
            raise ComfyError(
                TypedErrorCode.FFMPEG_ERROR,
                "ffprobe not found on PATH; cannot verify MP4 readability.",
                context={"path": str(path)},
            )
        import subprocess

        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,width,height", "-of", "json", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        if probe.returncode != 0:
            raise ComfyError(
                TypedErrorCode.INVALID_MP4,
                f"ffprobe cannot read output as a valid video: {probe.stderr[-1000:].strip()}",
                context={"path": str(path), "stderr": probe.stderr[-500:]},
            )
        try:
            streams = json.loads(probe.stdout or "{}").get("streams", [])
        except json.JSONDecodeError:
            streams = []
        if not streams:
            raise ComfyError(
                TypedErrorCode.INVALID_MP4,
                "ffprobe found no video stream in the output.",
                context={"path": str(path)},
            )

    # ------------------------------------------------------------------ step 14: validate (full)
    async def validate(self) -> dict[str, Any]:
        """Validate endpoint + workflow + required models. Returns a report dict."""
        issues: list[str] = []
        reachable = await self.detect()
        if not reachable:
            issues.append(f"ComfyUI unreachable at {self.base_url}")
            return {"ok": False, "issues": issues, "reachable": False}

        models: list[str] = []
        try:
            models = await self.list_models()
        except ComfyError as e:
            issues.append(f"Could not list models: {e.code.value}")

        missing = self._check_required_models(models) if self.required_models else []
        if missing:
            issues.append(f"Missing required models: {missing}")

        workflow_ok = False
        if self.workflow_path is not None:
            try:
                self.load_workflow()
                workflow_ok = True
            except ComfyError as e:
                issues.append(f"Workflow invalid: {e.code.value} — {e.detail}")
        else:
            issues.append("No workflow path configured")

        return {
            "ok": not issues,
            "issues": issues,
            "reachable": reachable,
            "models_count": len(models),
            "missing_models": missing,
            "workflow_ok": workflow_ok,
        }

    # ------------------------------------------------------------------ full generation (steps 1-14)
    async def generate(self, request: GenerationRequest, destination: Path) -> GenerationResult:
        """Run the full 14-step generation + verification for one clip.

        On success ``destination`` is a real, verified MP4. On failure a typed
        :class:`ComfyError` is raised. Never simulates success.
        """
        destination = Path(destination)
        workflow_path = request.workflow_path or self.workflow_path

        # Step 1: detect
        log("PROVIDER", f"ComfyUI detect at {self.base_url}")
        if not await self.detect():
            raise ComfyError(
                TypedErrorCode.COMFYUI_UNREACHABLE,
                f"ComfyUI not reachable at {self.base_url}",
                context={"url": self.base_url},
            )
        # Step 2: system info
        stats = await self.system_stats()
        log("PROVIDER", "ComfyUI detected", vram=_extract_vram(stats))
        # Step 3: models
        models = await self.list_models()
        log("MODEL", f"{len(models)} models visible")
        # Step 4: required models
        missing = self._check_required_models(models) if self.required_models else []
        if missing:
            raise ComfyError(
                TypedErrorCode.MODEL_NOT_FOUND,
                f"Required model weights missing: {missing}",
                context={"missing": missing, "available_sample": models[:20]},
            )
        # Step 5: workflow
        workflow = self.load_workflow(workflow_path)
        log("WORKFLOW", "validated")
        # Step 6: inject + submit
        full_prompt = request.prompt + ("\nNegative prompt: " + request.negative_prompt if request.negative_prompt else "")
        wf = self.inject_prompt(workflow, full_prompt, self.prompt_node_id, self.prompt_field)
        prompt_id = await self.queue(wf)
        # Step 7: real prompt_id received
        log("JOB", "submitted", prompt_id=prompt_id)
        # Step 8+9: poll to completion
        outputs = await self.wait(prompt_id)
        log("JOB", "completed", prompt_id=prompt_id)
        # Step 10–12: download
        await self.download_first_media(outputs, destination)
        log("OUTPUT", str(destination))
        # Step 13: verify
        self.verify_output(destination)
        # Step 14: COMPLETED
        log("QC", "passed", path=str(destination))
        return GenerationResult(path=destination, prompt_id=prompt_id, raw={"outputs": outputs})


def _extract_vram(stats: dict[str, Any]) -> Optional[str]:
    try:
        dev = stats.get("devices", [{}])[0]
        vram = dev.get("vram_total") or dev.get("vram_total") or dev.get("memory_total")
        return f"{int(vram) // (1024 ** 3)}GB" if vram else None
    except Exception:
        return None


# --------------------------------------------------------------------------- compat shim
class ComfyClient(ComfyUIProvider):
    """Backwards-compatible client.

    Preserves the exact legacy constructor and method signatures used by
    ``app.main`` and ``app.comfy`` so existing call sites keep working while the
    new typed-error path is used under the hood.
    """

    def __init__(self, base_url: str, timeout: int = 1800) -> None:
        # Legacy callers did not pass workflow/node/field; keep them None/defaults
        # so that generate()/load_workflow() require an explicit path, matching
        # the old behavior where main.py passed WORKFLOW_PATH explicitly.
        super().__init__(
            base_url,
            timeout,
            workflow_path=None,
            prompt_node_id="",
            prompt_field="text",
        )

"""Wan 2.2 video generation provider (Phase 8).

Implements the Wan-specific generation layer ON TOP of the existing ComfyUI
transport (:class:`~app.providers.comfyui.ComfyUIProvider`). It deliberately
separates WAN LOGIC (workflow selection, model detection, reference-image
validation, generation-option validation, I2V image handling) from COMFYUI
TRANSPORT (HTTP queue/poll/download/verify), which stays in ComfyUIProvider.

The provider consumes Phase 7's :class:`~app.scene.continuity.ResolvedSceneContext`
and :class:`~app.scene.references.ReferenceRegistry` so the downstream generation
request ALWAYS carries the resolved continuity context — it never receives a
bare prompt.

This provider NEVER simulates success. Every failure maps to a typed
:class:`~app.core.errors.VideoError`:

    COMFYUI_UNREACHABLE   - endpoint not reachable
    WORKFLOW_NOT_FOUND    - workflow file missing
    WORKFLOW_INVALID      - workflow malformed or missing required Wan nodes
    WORKFLOW_REJECTED     - ComfyUI /prompt rejected the workflow
    MODEL_NOT_FOUND       - Wan model not detected in ComfyUI
    INVALID_REFERENCE     - reference image missing/unreadable/wrong format
    GENERATION_TIMEOUT    - job did not finish in time
    NO_OUTPUT             - finished but produced no media
    INVALID_MP4           - output not a valid MP4
    FFMPEG_ERROR          - FFmpeg cannot probe the output
"""

from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import httpx

from ..config import COMFYUI_TIMEOUT_SECONDS, COMFYUI_URL
from ..core.errors import TypedErrorCode, VideoError
from ..core.logging import log
from ..scene.continuity import ResolvedSceneContext
from ..scene.references import ReferenceImage, ReferenceRegistry
from .base import GenerationRequest, GenerationResult, ProviderInfo, VideoProvider
from .comfyui import ComfyError, ComfyUIProvider


# ---------------------------------------------------------------- options

class WanMode(str, Enum):
    T2V = "t2v"
    I2V = "i2v"


_SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Required node class_types that must appear in a Wan workflow.
_T2V_REQUIRED = {
    "loader": ("CheckpointLoaderSimple", "UnetLoaderGGUF", "WanModelLoader"),
    "positive_prompt": ("CLIPTextEncode",),
    "negative_prompt": ("CLIPTextEncode",),
    "sampler": ("KSampler",),
    "vae_decode": ("VAEDecode",),
    "save_video": ("SaveAnimatedWEBP", "SaveVideo", "VHS_VideoCombine"),
}
_I2V_REQUIRED = {**_T2V_REQUIRED, "load_image": ("LoadImage",)}


@dataclass
class GenerationOptions:
    """Structured, range-validated generation parameters for Wan.

    Ranges are validated before anything is sent to ComfyUI.
    """

    width: int = 832
    height: int = 480
    frames: int = 81
    fps: float = 24.0
    steps: int = 20
    cfg: float = 6.0
    seed: Optional[int] = None
    negative_prompt: str = ""
    mode: WanMode = WanMode.T2V
    workflow_path: Optional[Path] = None
    reference_image_ids: list[str] = field(default_factory=list)

    def validate_ranges(self) -> None:
        if not (64 <= self.width <= 4096):
            raise VideoError(TypedErrorCode.WORKFLOW_INVALID, f"width {self.width} out of range [64,4096].")
        if not (64 <= self.height <= 4096):
            raise VideoError(TypedErrorCode.WORKFLOW_INVALID, f"height {self.height} out of range [64,4096].")
        if not (1 <= self.frames <= 1000):
            raise VideoError(TypedErrorCode.WORKFLOW_INVALID, f"frames {self.frames} out of range [1,1000].")
        if not (1 <= self.fps <= 120):
            raise VideoError(TypedErrorCode.WORKFLOW_INVALID, f"fps {self.fps} out of range [1,120].")
        if not (1 <= self.steps <= 200):
            raise VideoError(TypedErrorCode.WORKFLOW_INVALID, f"steps {self.steps} out of range [1,200].")
        if not (0.0 <= self.cfg <= 30.0):
            raise VideoError(TypedErrorCode.WORKFLOW_INVALID, f"cfg {self.cfg} out of range [0,30].")
        if self.seed is not None and not (0 <= self.seed < 2**31):
            raise VideoError(TypedErrorCode.WORKFLOW_INVALID, f"seed {self.seed} out of range [0,2^31).")

    def resolve_seed(self) -> int:
        """Return an explicit seed, generating+recording one if omitted."""
        if self.seed is None:
            self.seed = random.randint(0, 2**31 - 1)
            log("WAN", f"seed generated: {self.seed}")
        return self.seed

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mode"] = self.mode.value
        d["workflow_path"] = str(self.workflow_path) if self.workflow_path else None
        return d


# ---------------------------------------------------------------- output metadata

@dataclass
class GenerationMetadata:
    """Per-generation metadata persisted alongside the output."""

    project_id: Optional[str]
    scene_index: Optional[int]
    provider: str
    model: Optional[str]
    workflow: str
    mode: str
    seed: int
    prompt: str
    negative_prompt: str
    reference_ids: list[str]
    timestamp: str
    output_path: str
    duration: float
    width: int
    height: int
    prompt_id: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


# ---------------------------------------------------------------- readiness

@dataclass
class ReadinessReport:
    """Honest readiness state with exact blockers."""

    ready: bool
    comfyui_reachable: bool
    comfyui_version: Optional[str] = None
    vram: Optional[str] = None
    wan_detected: bool = False
    detected_models: list[str] = field(default_factory=list)
    workflows_available: list[str] = field(default_factory=list)
    workflow_valid: bool = False
    references_available: bool = True
    gpu_available: bool = False
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- Wan provider

class WanProvider(VideoProvider):
    """Wan 2.2 T2V/I2V provider composing ComfyUIProvider for transport."""

    def __init__(
        self,
        base_url: str = COMFYUI_URL,
        timeout: int = COMFYUI_TIMEOUT_SECONDS,
        *,
        t2v_workflow: Optional[Path] = None,
        i2v_workflow: Optional[Path] = None,
        required_model_substring: str = "wan2.2",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = int(timeout)
        self.t2v_workflow = t2v_workflow
        self.i2v_workflow = i2v_workflow
        self.required_model_substring = required_model_substring.lower()
        # ComfyUI transport delegate (shared HTTP client + queue/poll/download/verify).
        self._transport = ComfyUIProvider(self.base_url, self.timeout)

    @property
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            name="wan",
            kind="local",
            description="Wan 2.2 T2V + I2V video generation via ComfyUI.",
        )

    def supports_image_to_video(self) -> bool:
        return True

    # ------------------------------------------------------------------ transport delegation
    async def detect(self) -> bool:
        return await self._transport.detect()

    async def health(self) -> dict[str, Any]:
        return await self._transport.health()

    async def list_models(self) -> list[str]:
        return await self._transport.list_models()

    # ------------------------------------------------------------------ Wan model detection
    def detect_wan_models(self, available: list[str]) -> list[str]:
        """Return models whose filename contains the Wan substring (e.g. wan2.2)."""
        return [m for m in available if self.required_model_substring in m.lower()]

    async def require_wan_model(self) -> list[str]:
        """Detect Wan models; raise MODEL_NOT_FOUND with diagnostics if absent."""
        models = await self.list_models()
        wan = self.detect_wan_models(models)
        if not wan:
            raise ComfyError(
                TypedErrorCode.MODEL_NOT_FOUND,
                f"No Wan model detected (looking for '{self.required_model_substring}').",
                context={
                    "expected_substring": self.required_model_substring,
                    "detected_models": models[:30],
                    "endpoint": self.base_url,
                },
            )
        return wan

    # ------------------------------------------------------------------ workflow validation
    def _required_nodes(self, mode: WanMode) -> dict[str, tuple[str, ...]]:
        return _I2V_REQUIRED if mode == WanMode.I2V else _T2V_REQUIRED

    def workflow_path_for(self, mode: WanMode, override: Optional[Path] = None) -> Optional[Path]:
        if override is not None:
            return override
        return self.t2v_workflow if mode == WanMode.T2V else self.i2v_workflow

    def load_and_validate_workflow(self, mode: WanMode, override: Optional[Path] = None) -> dict[str, Any]:
        """Load a Wan workflow and validate required node classes exist.

        Raises WORKFLOW_NOT_FOUND if the file is missing, WORKFLOW_INVALID if
        malformed or missing required Wan nodes.
        """
        path = self.workflow_path_for(mode, override)
        if path is None or not Path(path).exists():
            raise ComfyError(
                TypedErrorCode.WORKFLOW_NOT_FOUND,
                f"Wan {mode.value.upper()} workflow not found: {path}",
                context={"mode": mode.value, "path": str(path) if path else None},
            )
        workflow = self._transport.load_workflow(Path(path))
        # Validate required node classes are present.
        required = self._required_nodes(mode)
        present_classes = {
            str(node.get("class_type", "")) for node in workflow.values()
            if isinstance(node, dict) and not str(node.get("class_type", "")).startswith("_")
        }
        missing_roles: list[str] = []
        for role, classes in required.items():
            if not any(c in present_classes for c in classes):
                missing_roles.append(role)
        if missing_roles:
            raise ComfyError(
                TypedErrorCode.WORKFLOW_INVALID,
                f"Wan {mode.value.upper()} workflow missing required node roles: {missing_roles}. "
                f"Export a real API-format Wan workflow from ComfyUI (see workflows/README.md).",
                context={"mode": mode.value, "missing_roles": missing_roles,
                         "present_classes": sorted(present_classes)},
            )
        return workflow

    # ------------------------------------------------------------------ reference validation
    def validate_reference_image(self, path: Path) -> None:
        """Validate a reference image file. Raises INVALID_REFERENCE on any problem."""
        path = Path(path)
        if not path.exists():
            raise VideoError(
                TypedErrorCode.INVALID_REFERENCE,
                f"Reference image does not exist: {path}",
                context={"path": str(path)},
            )
        if path.suffix.lower() not in _SUPPORTED_IMAGE_EXTS:
            raise VideoError(
                TypedErrorCode.INVALID_REFERENCE,
                f"Reference image unsupported format '{path.suffix}'. Supported: {sorted(_SUPPORTED_IMAGE_EXTS)}.",
                context={"path": str(path), "ext": path.suffix},
            )
        if path.stat().st_size <= 0:
            raise VideoError(
                TypedErrorCode.INVALID_REFERENCE,
                f"Reference image is empty (0 bytes): {path}",
                context={"path": str(path)},
            )
        try:
            with path.open("rb") as f:
                head = f.read(16)
            if not head:
                raise VideoError(
                    TypedErrorCode.INVALID_REFERENCE,
                    f"Reference image unreadable: {path}",
                    context={"path": str(path)},
                )
        except OSError as e:
            raise VideoError(
                TypedErrorCode.INVALID_REFERENCE,
                f"Reference image cannot be read: {e}",
                context={"path": str(path)},
            )

    def resolve_references(
        self, context: ResolvedSceneContext, registry: ReferenceRegistry,
    ) -> list[ReferenceImage]:
        """Resolve references by stable IDs from the resolved context.

        Only references whose target_id is in the context's continuity_ids are
        returned. The provider never guesses which image belongs to which
        character/product.
        """
        wanted = set(context.continuity_ids)
        resolved: list[ReferenceImage] = []
        for img in registry.images:
            if img.target_id in wanted:
                resolved.append(img)
        return resolved

    def require_references_available(
        self, context: ResolvedSceneContext, registry: ReferenceRegistry,
    ) -> list[ReferenceImage]:
        """For I2V: validate that at least one available reference exists.

        Raises INVALID_REFERENCE if a reference is required but none of the
        bound references have a real backing file.
        """
        refs = self.resolve_references(context, registry)
        if not refs:
            raise VideoError(
                TypedErrorCode.INVALID_REFERENCE,
                "I2V mode requires at least one reference image bound to a "
                "continuity ID in this scene, but none were found.",
                context={"continuity_ids": context.continuity_ids},
            )
        available = [r for r in refs if r.is_available()]
        if not available:
            raise VideoError(
                TypedErrorCode.INVALID_REFERENCE,
                "I2V references are bound but none have a real backing file on disk.",
                context={"bound_targets": [r.target_id for r in refs]},
            )
        return available

    # ------------------------------------------------------------------ reference upload
    async def upload_reference(self, image_path: Path) -> str:
        """Upload a reference image to ComfyUI and return the server filename.

        Raises INVALID_REFERENCE / COMFYUI_UNREACHABLE on failure.
        """
        self.validate_reference_image(image_path)
        try:
            async with httpx.AsyncClient(timeout=60) as c:
                with Path(image_path).open("rb") as f:
                    r = await c.post(
                        f"{self.base_url}/upload/image",
                        files={"image": (Path(image_path).name, f, "image/png")},
                        data={"overwrite": "true"},
                    )
        except httpx.RequestError as e:
            raise ComfyError(
                TypedErrorCode.COMFYUI_UNREACHABLE,
                f"Cannot upload reference image to ComfyUI: {e}",
                context={"path": str(image_path)},
            )
        if r.status_code != 200:
            raise VideoError(
                TypedErrorCode.INVALID_REFERENCE,
                f"ComfyUI /upload/image returned HTTP {r.status_code}: {r.text[:200]}",
                context={"path": str(image_path), "status": r.status_code},
            )
        try:
            data = r.json()
        except ValueError as e:
            raise VideoError(
                TypedErrorCode.INVALID_REFERENCE,
                f"ComfyUI /upload/image returned non-JSON: {e}",
            )
        name = data.get("name") or data.get("filename")
        if not name:
            raise VideoError(
                TypedErrorCode.INVALID_REFERENCE,
                f"ComfyUI /upload/image returned no filename: {data}",
                context={"response": data},
            )
        return name

    # ------------------------------------------------------------------ prompt building
    def build_request_from_context(
        self, context: ResolvedSceneContext, options: GenerationOptions,
    ) -> GenerationRequest:
        """Build a GenerationRequest carrying the resolved continuity context.

        The positive prompt is the structured visual_prompt from the resolved
        context — the provider never receives a bare prompt.
        """
        options.validate_ranges()
        return GenerationRequest(
            prompt=context.visual_prompt,
            negative_prompt=options.negative_prompt or context.negative_prompt,
            duration=context.duration,
            width=options.width,
            height=options.height,
            frames=options.frames,
            fps=options.fps,
            seed=options.resolve_seed(),
            reference_images=[],  # filled for I2V
            workflow_path=options.workflow_path,
            extra={"mode": options.mode.value, "steps": options.steps, "cfg": options.cfg},
        )

    # ------------------------------------------------------------------ output management
    def _output_dir(self, project_id: Optional[str], scene_index: Optional[int]) -> Path:
        from ..config import OUTPUT_DIR
        run_id = uuid.uuid4().hex[:12]
        sub = Path(OUTPUT_DIR) / (project_id or "standalone") / f"scene_{scene_index or 0}" / run_id
        sub.mkdir(parents=True, exist_ok=True)
        return sub

    # ------------------------------------------------------------------ full generation
    async def generate(
        self, request: GenerationRequest, destination: Path,
    ) -> GenerationResult:
        """Generate one clip via ComfyUI transport (delegates full 14-step verify)."""
        return await self._transport.generate(request, Path(destination))

    async def generate_from_context(
        self,
        context: ResolvedSceneContext,
        options: GenerationOptions,
        registry: Optional[ReferenceRegistry] = None,
        *,
        project_id: Optional[str] = None,
        destination_dir: Optional[Path] = None,
    ) -> tuple[GenerationResult, GenerationMetadata]:
        """Full Wan generation from a resolved scene context.

        T2V: text-to-video using the context's structured prompt.
        I2V: validates + uploads a reference image, then conditions generation.

        Returns (result, metadata). The result.path is a verified MP4.
        Never returns without a verified output.
        """
        # Step 7: continuity context resolved (precondition).
        if not context.visual_prompt:
            raise VideoError(
                TypedErrorCode.WORKFLOW_INVALID,
                "Resolved scene context has no visual_prompt; continuity not resolved.",
                context={"scene_index": context.scene_index},
            )
        options.validate_ranges()
        mode = options.mode

        # I2V reference validation happens FIRST (local, fail-fast) before any
        # network call: a missing reference must never reach ComfyUI.
        ref_ids: list[str] = []
        i2v_refs: list[ReferenceImage] = []
        if mode == WanMode.I2V:
            if registry is None:
                raise VideoError(
                    TypedErrorCode.INVALID_REFERENCE,
                    "I2V mode requires a ReferenceRegistry; none provided.",
                )
            i2v_refs = self.require_references_available(context, registry)
            ref_ids = [r.target_id for r in i2v_refs]

        # Step 1: detect ComfyUI
        log("WAN", f"mode={mode.value} scene={context.scene_index}")
        if not await self.detect():
            raise ComfyError(
                TypedErrorCode.COMFYUI_UNREACHABLE,
                f"ComfyUI not reachable at {self.base_url}",
                context={"url": self.base_url},
            )
        # Step 2-3: system stats + Wan model detection
        stats = await self._transport.system_stats()
        wan_models = await self.require_wan_model()
        model_name = wan_models[0]
        log("MODEL", f"Wan detected: {model_name}")
        # Step 4: workflow exists + valid
        workflow = self.load_and_validate_workflow(mode, options.workflow_path)
        log("WORKFLOW", f"Wan {mode.value.upper()} validated")

        # I2V: reference image upload (validation already done above).
        if mode == WanMode.I2V:
            ref = next(r for r in i2v_refs if r.is_available() and r.path is not None)
            server_name = await self.upload_reference(Path(ref.path))
            log("WAN", f"I2V reference uploaded: {server_name} (targets={ref_ids})")
            workflow = self._inject_load_image(workflow, server_name)

        # Build the generation request (continuity prompt preserved).
        request = self.build_request_from_context(context, options)
        request.workflow_path = options.workflow_path
        # Step 5-6: inject prompts + sampler seed via transport
        workflow = self._transport.inject_prompt(
            workflow, request.prompt, "", "text",
        )
        workflow = self._inject_negative_prompt(workflow, options.negative_prompt or context.negative_prompt)
        workflow = self._inject_sampler(workflow, options)

        # Output management
        out_dir = Path(destination_dir) if destination_dir else self._output_dir(project_id, context.scene_index)
        out_dir.mkdir(parents=True, exist_ok=True)
        destination = out_dir / f"scene_{context.scene_index}_wan.mp4"

        # Step 7-14: submit, poll, download, verify (transport)
        prompt_id = await self._transport.queue(workflow)
        log("JOB", "submitted", prompt_id=prompt_id)
        outputs = await self._transport.wait(prompt_id)
        log("JOB", "completed", prompt_id=prompt_id)
        await self._transport.download_first_media(outputs, destination)
        log("OUTPUT", str(destination))
        self._transport.verify_output(destination)
        log("QC", "passed", path=str(destination))

        result = GenerationResult(path=destination, prompt_id=prompt_id, model=model_name, raw={"outputs": outputs})
        meta = GenerationMetadata(
            project_id=project_id, scene_index=context.scene_index,
            provider="wan", model=model_name,
            workflow=str(options.workflow_path or self.workflow_path_for(mode)),
            mode=mode.value, seed=options.seed or 0,
            prompt=request.prompt, negative_prompt=request.negative_prompt,
            reference_ids=ref_ids, timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            output_path=str(destination), duration=context.duration,
            width=options.width, height=options.height, prompt_id=prompt_id,
        )
        meta.save(out_dir / "metadata.json")
        return result, meta

    # ------------------------------------------------------------------ workflow injection helpers
    def _inject_negative_prompt(self, workflow: dict[str, Any], negative: str) -> dict[str, Any]:
        """Inject the negative prompt into the negative CLIPTextEncode node."""
        if not negative:
            return workflow
        wf = json.loads(json.dumps(workflow))
        # Find a CLIPTextEncode whose positive counterpart is the main prompt;
        # the other CLIPTextEncode is the negative one. Heuristic: a CLIPTextEncode
        # whose inputs.text does NOT already equal the positive prompt.
        positive = wf
        clip_nodes = [(nid, n) for nid, n in wf.items()
                      if isinstance(n, dict) and n.get("class_type") == "CLIPTextEncode"
                      and not str(nid).startswith("_")]
        if not clip_nodes:
            return wf
        # If two CLIPTextEncode nodes, the second is conventionally negative.
        if len(clip_nodes) >= 2:
            nid, _ = clip_nodes[1]
            wf[nid].setdefault("inputs", {})["text"] = negative
        else:
            nid, _ = clip_nodes[0]
            wf[nid].setdefault("inputs", {})["text"] = negative
        return wf

    def _inject_load_image(self, workflow: dict[str, Any], server_name: str) -> dict[str, Any]:
        """Inject the uploaded reference filename into the LoadImage node."""
        wf = json.loads(json.dumps(workflow))
        for nid, node in wf.items():
            if isinstance(node, dict) and node.get("class_type") == "LoadImage":
                node.setdefault("inputs", {})["image"] = server_name
                return wf
        raise ComfyError(
            TypedErrorCode.WORKFLOW_INVALID,
            "I2V workflow has no LoadImage node to receive the reference image.",
        )

    def _inject_sampler(self, workflow: dict[str, Any], options: GenerationOptions) -> dict[str, Any]:
        """Inject seed/steps/cfg into the KSampler node."""
        wf = json.loads(json.dumps(workflow))
        seed = options.seed if options.seed is not None else 0
        for nid, node in wf.items():
            if isinstance(node, dict) and node.get("class_type") == "KSampler":
                inp = node.setdefault("inputs", {})
                inp["seed"] = seed
                inp["steps"] = options.steps
                inp["cfg"] = options.cfg
                return wf
        # No KSampler found is not fatal (some workflows use ModelSamplingSD3);
        # the seed is still recorded in metadata for reproducibility.
        return wf

    # ------------------------------------------------------------------ validate (contract)
    async def validate(self) -> dict[str, Any]:
        report = await self.diagnose()
        return {"ok": report.ready, "issues": report.blockers,
                "reachable": report.comfyui_reachable, "wan_detected": report.wan_detected}

    # ------------------------------------------------------------------ diagnostics
    async def diagnose(
        self,
        *,
        context: Optional[ResolvedSceneContext] = None,
        registry: Optional[ReferenceRegistry] = None,
    ) -> ReadinessReport:
        """Return a single honest READY/NOT_READY state with exact blockers."""
        blockers: list[str] = []
        reachable = await self.detect()
        if not reachable:
            blockers.append(f"ComfyUI unreachable at {self.base_url}")
            return ReadinessReport(ready=False, comfyui_reachable=False, blockers=blockers)

        version: Optional[str] = None
        vram: Optional[str] = None
        gpu = False
        try:
            stats = await self._transport.system_stats()
            version = stats.get("system", {}).get("comfyui_version")
            devs = stats.get("devices", [])
            if devs:
                dev = devs[0]
                vram_total = dev.get("vram_total") or dev.get("vram_free")
                if vram_total:
                    vram = f"{int(vram_total) // (1024 ** 3)}GB"
                gpu = bool(dev.get("name")) and dev.get("type", "").lower() != "cpu"
        except ComfyError as e:
            blockers.append(f"system_stats failed: {e.code.value}")

        wan_models: list[str] = []
        detected: list[str] = []
        try:
            detected = await self.list_models()
            wan_models = self.detect_wan_models(detected)
            if not wan_models:
                blockers.append(f"No Wan model detected (looking for '{self.required_model_substring}').")
        except ComfyError as e:
            blockers.append(f"Could not list models: {e.code.value}")

        workflows_available: list[str] = []
        workflow_valid = True
        for mode, path in (("t2v", self.t2v_workflow), ("i2v", self.i2v_workflow)):
            if path and Path(path).exists():
                workflows_available.append(f"{mode}:{path.name}")
                try:
                    self.load_and_validate_workflow(WanMode(mode), path)
                except ComfyError as e:
                    workflow_valid = False
                    blockers.append(f"{mode} workflow invalid: {e.code.value} — {e.detail[:80]}")
            else:
                blockers.append(f"{mode} workflow not found: {path}")
                workflow_valid = False

        references_available = True
        if context is not None and registry is not None:
            refs = self.resolve_references(context, registry)
            available = [r for r in refs if r.is_available()]
            if context.continuity_ids and not refs:
                references_available = False
                blockers.append("Scene requires continuity references but none are bound.")
            elif refs and not available:
                references_available = False
                blockers.append("References bound but none have a real backing file.")

        if not gpu:
            blockers.append("No GPU detected; Wan generation may be impractical (high VRAM requirement).")

        return ReadinessReport(
            ready=(not blockers),
            comfyui_reachable=reachable,
            comfyui_version=version,
            vram=vram,
            wan_detected=bool(wan_models),
            detected_models=wan_models or detected[:20],
            workflows_available=workflows_available,
            workflow_valid=workflow_valid,
            references_available=references_available,
            gpu_available=gpu,
            blockers=blockers,
        )

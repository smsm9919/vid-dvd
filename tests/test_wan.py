"""Comprehensive tests for the Wan 2.2 T2V/I2V provider (Phase 8).

These tests exercise the Wan logic (workflow validation, model detection,
reference validation, option validation, prompt injection, I2V image handling,
typed failure propagation) using a small in-process fake ComfyUI HTTP server.

They are CODE/TEST VERIFIED only. They do NOT constitute real video runtime
verification — a fake server is used, and no real MP4 is generated. Real runtime
verification requires a real ComfyUI + Wan 2.2 + GPU (reported as
COMFYUI_RUNTIME_BLOCKED in this environment).
"""

import asyncio
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pytest

from app.brain.models import ContentBrief
from app.brain.content_brain import local_content_plan
from app.scene.continuity import resolve_scene_context
from app.scene.references import ReferenceImage, ReferenceKind, ReferenceRegistry, registry_from_plan
from app.core.errors import TypedErrorCode, VideoError
from app.providers.base import GenerationRequest
from app.providers.comfyui import ComfyError
from app.providers.wan import (
    GenerationMetadata,
    GenerationOptions,
    ReadinessReport,
    WanMode,
    WanProvider,
)


# ---------------------------------------------------------------- fake server
def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class _FakeHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body=b"", ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        routes = self.server.routes  # type: ignore[attr-defined]
        path = urlparse(self.path).path
        if path in routes:
            kind, payload = routes[path]
            if kind == "json":
                self._send(200, json.dumps(payload).encode())
            elif kind == "bytes":
                self._send(200, payload, ctype="application/octet-stream")
            elif kind == "status":
                self._send(payload, b"")
            return
        self._send(404, b"not found")

    def do_POST(self):
        routes = self.server.routes  # type: ignore[attr-defined]
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        if path in routes:
            kind, payload = routes[path]
            if kind == "json":
                self._send(200, json.dumps(payload).encode())
            elif callable(payload):
                self._send(200, json.dumps(payload(body)).encode())
            elif kind == "status":
                self._send(payload, b"")
            return
        self._send(404, b"not found")


class _FakeComfy:
    def __init__(self, routes):
        self.port = _free_port()
        self.routes = routes
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), _FakeHandler)
        self.server.routes = routes  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *a):
        self.server.shutdown()
        self.server.server_close()


# ---------------------------------------------------------------- fixtures
def _real_t2v_workflow():
    return Path("workflows/wan22_t2v_api.json")


def _real_i2v_workflow():
    return Path("workflows/wan22_i2v_api.json")


def _ctx():
    plan = local_content_plan(ContentBrief(idea="lion hunting in jungle", duration_seconds=8, mode="cinematic"))
    ctx = resolve_scene_context(plan, plan.scenes[0].index)
    return ctx, plan


def _wan_routes(system_stats=None, models=None, prompt_id="pid-1", media=None):
    return {
        "/system_stats": ("json", system_stats or {"system": {"comfyui_version": "0.3.0"}, "devices": [{"name": "cuda", "type": "cuda", "vram_total": 8 * (1024**3)}]}),
        "/object_info": ("json", {"Loader": {"input": {"unet": models or ["wan2.2_5b.safetensors"]}}}),
        "/prompt": ("json", {"prompt_id": prompt_id}),
        "/history/pid-1": ("json", {"pid-1": {"status": {"status_str": "success"},
            "outputs": {"9": {"videos": [{"filename": "out.mp4", "subfolder": "", "type": "output"}]}}}}),
        "/view": ("bytes", media or b"FAKEVIDEO"),
        "/upload/image": ("json", {"name": "uploaded_ref.png"}),
    }


# ---------------------------------------------------------------- options
def test_options_validate_ranges_ok():
    opts = GenerationOptions(width=832, height=480, frames=81, steps=20, cfg=6.0, seed=42)
    opts.validate_ranges()  # no raise


@pytest.mark.parametrize("field,bad", [("width", 10), ("width", 99999), ("height", 0), ("frames", 0), ("frames", 5000), ("fps", 0), ("steps", 0), ("cfg", -1), ("cfg", 99)])
def test_options_bad_ranges(field, bad):
    opts = GenerationOptions()
    setattr(opts, field, bad)
    with pytest.raises(VideoError) as e:
        opts.validate_ranges()
    assert e.value.code is TypedErrorCode.WORKFLOW_INVALID


def test_options_seed_generated_when_omitted():
    opts = GenerationOptions()
    assert opts.seed is None
    s = opts.resolve_seed()
    assert s == opts.seed
    assert 0 <= s < 2**31


def test_options_seed_preserved_when_set():
    opts = GenerationOptions(seed=12345)
    assert opts.resolve_seed() == 12345


def test_options_bad_seed_raises():
    opts = GenerationOptions(seed=-5)
    with pytest.raises(VideoError):
        opts.validate_ranges()


def test_options_to_dict_serializable():
    d = GenerationOptions(seed=1).to_dict()
    json.dumps(d)
    assert d["mode"] == "t2v"


# ---------------------------------------------------------------- model detection
def test_detect_wan_models_finds_substring():
    p = WanProvider("http://0.0.0.0:1")
    models = ["wan2.2_5b.safetensors", "vae.safetensors", "sdxl.safetensors"]
    assert p.detect_wan_models(models) == ["wan2.2_5b.safetensors"]


def test_detect_wan_models_empty():
    p = WanProvider("http://0.0.0.0:1")
    assert p.detect_wan_models(["sdxl.safetensors"]) == []


def test_require_wan_model_raises_when_absent():
    routes = {"/system_stats": ("json", {}), "/object_info": ("json", {"Loader": {"input": {"unet": ["sdxl.safetensors"]}}})}
    with _FakeComfy(routes) as base:
        p = WanProvider(base, timeout=5)
        with pytest.raises(ComfyError) as e:
            asyncio.run(p.require_wan_model())
        assert e.value.code is TypedErrorCode.MODEL_NOT_FOUND
        assert "detected_models" in e.value.context
        assert "endpoint" in e.value.context


def test_require_wan_model_returns_when_present():
    routes = {"/system_stats": ("json", {}), "/object_info": ("json", {"Loader": {"input": {"unet": ["wan2.2_5b.safetensors"]}}})}
    with _FakeComfy(routes) as base:
        p = WanProvider(base, timeout=5)
        wan = asyncio.run(p.require_wan_model())
        assert wan == ["wan2.2_5b.safetensors"]


# ---------------------------------------------------------------- workflow validation
def test_workflow_not_found(tmp_path):
    p = WanProvider("http://0.0.0.0:1")
    with pytest.raises(ComfyError) as e:
        p.load_and_validate_workflow(WanMode.T2V, tmp_path / "nope.json")
    assert e.value.code is TypedErrorCode.WORKFLOW_NOT_FOUND


def test_workflow_none_configured():
    p = WanProvider("http://0.0.0.0:1", t2v_workflow=None)
    with pytest.raises(ComfyError) as e:
        p.load_and_validate_workflow(WanMode.T2V)
    assert e.value.code is TypedErrorCode.WORKFLOW_NOT_FOUND


def test_workflow_validates_template_t2v():
    p = WanProvider("http://0.0.0.0:1", t2v_workflow=_real_t2v_workflow())
    wf = p.load_and_validate_workflow(WanMode.T2V)
    assert isinstance(wf, dict) and len(wf) > 0


def test_workflow_validates_template_i2v():
    p = WanProvider("http://0.0.0.0:1", t2v_workflow=_real_t2v_workflow(), i2v_workflow=_real_i2v_workflow())
    wf = p.load_and_validate_workflow(WanMode.I2V)
    assert isinstance(wf, dict) and len(wf) > 0


def test_workflow_missing_required_node(tmp_path):
    # Workflow missing a VAEDecode node.
    bad = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
           "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "x"}},
           "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "x"}},
           "4": {"class_type": "KSampler", "inputs": {}},
           "6": {"class_type": "SaveVideo", "inputs": {}}}
    f = tmp_path / "bad_t2v.json"
    f.write_text(json.dumps(bad))
    p = WanProvider("http://0.0.0.0:1", t2v_workflow=f)
    with pytest.raises(ComfyError) as e:
        p.load_and_validate_workflow(WanMode.T2V)
    assert e.value.code is TypedErrorCode.WORKFLOW_INVALID
    assert "vae_decode" in str(e.value.context.get("missing_roles"))


def test_i2v_workflow_requires_load_image(tmp_path):
    bad = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
           "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "x"}},
           "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "x"}},
           "4": {"class_type": "KSampler", "inputs": {}},
           "5": {"class_type": "VAEDecode", "inputs": {}},
           "6": {"class_type": "SaveVideo", "inputs": {}}}  # no LoadImage
    f = tmp_path / "bad_i2v.json"
    f.write_text(json.dumps(bad))
    p = WanProvider("http://0.0.0.0:1", i2v_workflow=f)
    with pytest.raises(ComfyError) as e:
        p.load_and_validate_workflow(WanMode.I2V)
    assert e.value.code is TypedErrorCode.WORKFLOW_INVALID
    assert "load_image" in str(e.value.context.get("missing_roles"))


# ---------------------------------------------------------------- reference validation
def test_reference_missing_file(tmp_path):
    p = WanProvider("http://0.0.0.0:1")
    with pytest.raises(VideoError) as e:
        p.validate_reference_image(tmp_path / "nope.png")
    assert e.value.code is TypedErrorCode.INVALID_REFERENCE


def test_reference_bad_format(tmp_path):
    f = tmp_path / "ref.gif"
    f.write_bytes(b"x")
    p = WanProvider("http://0.0.0.0:1")
    with pytest.raises(VideoError) as e:
        p.validate_reference_image(f)
    assert e.value.code is TypedErrorCode.INVALID_REFERENCE


def test_reference_empty_file(tmp_path):
    f = tmp_path / "ref.png"
    f.write_bytes(b"")
    p = WanProvider("http://0.0.0.0:1")
    with pytest.raises(VideoError) as e:
        p.validate_reference_image(f)
    assert e.value.code is TypedErrorCode.INVALID_REFERENCE


def test_reference_valid(tmp_path):
    f = tmp_path / "ref.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"fakedata")
    p = WanProvider("http://0.0.0.0:1")
    p.validate_reference_image(f)  # no raise


def test_resolve_references_by_stable_id():
    ctx, plan = _ctx()
    reg = registry_from_plan(plan)
    p = WanProvider("http://0.0.0.0:1")
    refs = p.resolve_references(ctx, reg)
    # Only references whose target_id is in continuity_ids.
    assert all(r.target_id in ctx.continuity_ids for r in refs)


def test_require_references_available_none_bound():
    ctx, plan = _ctx()
    empty_reg = ReferenceRegistry()
    p = WanProvider("http://0.0.0.0:1")
    with pytest.raises(VideoError) as e:
        p.require_references_available(ctx, empty_reg)
    assert e.value.code is TypedErrorCode.INVALID_REFERENCE


def test_require_references_available_no_backing_file(tmp_path):
    ctx, plan = _ctx()
    # Bind a reference with no real file.
    reg = ReferenceRegistry(images=[ReferenceImage(ReferenceKind.CHARACTER, ctx.continuity_ids[0], path=tmp_path / "nope.png")])
    p = WanProvider("http://0.0.0.0:1")
    with pytest.raises(VideoError) as e:
        p.require_references_available(ctx, reg)
    assert e.value.code is TypedErrorCode.INVALID_REFERENCE


# ---------------------------------------------------------------- request building
def test_build_request_from_context_carries_continuity_prompt():
    ctx, plan = _ctx()
    opts = GenerationOptions(seed=7)
    p = WanProvider("http://0.0.0.0:1")
    req = p.build_request_from_context(ctx, opts)
    assert req.prompt == ctx.visual_prompt  # structured continuity prompt, not bare
    assert "[IDENTITY]" in req.prompt
    assert req.seed == 7
    assert req.width == opts.width


def test_build_request_uses_context_negative_when_options_empty():
    ctx, plan = _ctx()
    opts = GenerationOptions(seed=1, negative_prompt="")
    p = WanProvider("http://0.0.0.0:1")
    req = p.build_request_from_context(ctx, opts)
    assert req.negative_prompt == ctx.negative_prompt


# ---------------------------------------------------------------- prompt injection
def test_inject_negative_prompt_into_second_clipnode():
    p = WanProvider("http://0.0.0.0:1")
    wf = {"2": {"class_type": "CLIPTextEncode", "inputs": {"text": "pos"}},
          "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "old"}}}
    out = p._inject_negative_prompt(wf, "blurry, distorted")
    assert out["3"]["inputs"]["text"] == "blurry, distorted"
    # original untouched
    assert wf["3"]["inputs"]["text"] == "old"


def test_inject_load_image():
    p = WanProvider("http://0.0.0.0:1")
    wf = {"2": {"class_type": "LoadImage", "inputs": {"image": "old.png"}}}
    out = p._inject_load_image(wf, "uploaded_ref.png")
    assert out["2"]["inputs"]["image"] == "uploaded_ref.png"


def test_inject_load_image_missing_node(tmp_path):
    p = WanProvider("http://0.0.0.0:1")
    wf = {"1": {"class_type": "X", "inputs": {}}}
    with pytest.raises(ComfyError) as e:
        p._inject_load_image(wf, "x.png")
    assert e.value.code is TypedErrorCode.WORKFLOW_INVALID


def test_inject_sampler_sets_seed_steps_cfg():
    p = WanProvider("http://0.0.0.0:1")
    wf = {"4": {"class_type": "KSampler", "inputs": {"seed": 0, "steps": 1, "cfg": 1.0}}}
    out = p._inject_sampler(wf, GenerationOptions(seed=99, steps=25, cfg=7.0))
    assert out["4"]["inputs"]["seed"] == 99
    assert out["4"]["inputs"]["steps"] == 25
    assert out["4"]["inputs"]["cfg"] == 7.0


# ---------------------------------------------------------------- output metadata
def test_metadata_save_and_serialize(tmp_path):
    meta = GenerationMetadata(
        project_id="p1", scene_index=1, provider="wan", model="wan2.2_5b",
        workflow="wf.json", mode="t2v", seed=42, prompt="p", negative_prompt="n",
        reference_ids=[], timestamp="t", output_path="o.mp4", duration=4.0,
        width=832, height=480, prompt_id="pid",
    )
    d = meta.to_dict()
    json.dumps(d)
    meta.save(tmp_path / "meta.json")
    assert (tmp_path / "meta.json").exists()
    back = json.loads((tmp_path / "meta.json").read_text())
    assert back["seed"] == 42


# ---------------------------------------------------------------- diagnostics
def test_diagnose_unreachable_reports_not_ready():
    p = WanProvider("http://127.0.0.1:1", timeout=3, t2v_workflow=_real_t2v_workflow(), i2v_workflow=_real_i2v_workflow())
    rep = asyncio.run(p.diagnose())
    assert rep.ready is False
    assert rep.comfyui_reachable is False
    assert any("unreachable" in b for b in rep.blockers)


def test_diagnose_no_wan_model_reports_blocker():
    routes = {
        "/system_stats": ("json", {"system": {"comfyui_version": "0.3.0"}, "devices": [{"name": "cuda", "type": "cuda", "vram_total": 8*(1024**3)}]}),
        "/object_info": ("json", {"Loader": {"input": {"unet": ["sdxl.safetensors"]}}}),
    }
    with _FakeComfy(routes) as base:
        p = WanProvider(base, timeout=5, t2v_workflow=_real_t2v_workflow(), i2v_workflow=_real_i2v_workflow())
        rep = asyncio.run(p.diagnose())
        assert rep.ready is False
        assert rep.wan_detected is False
        assert any("Wan model" in b for b in rep.blockers)


def test_diagnose_no_gpu_reports_blocker():
    routes = {
        "/system_stats": ("json", {"system": {"comfyui_version": "0.3.0"}, "devices": [{"name": "cpu", "type": "cpu", "vram_total": 0}]}),
        "/object_info": ("json", {"Loader": {"input": {"unet": ["wan2.2_5b.safetensors"]}}}),
    }
    with _FakeComfy(routes) as base:
        p = WanProvider(base, timeout=5, t2v_workflow=_real_t2v_workflow(), i2v_workflow=_real_i2v_workflow())
        rep = asyncio.run(p.diagnose())
        assert rep.ready is False
        assert any("GPU" in b for b in rep.blockers)


def test_diagnose_ready_when_all_present():
    routes = _wan_routes()
    with _FakeComfy(routes) as base:
        p = WanProvider(base, timeout=5, t2v_workflow=_real_t2v_workflow(), i2v_workflow=_real_i2v_workflow())
        rep = asyncio.run(p.diagnose())
        assert rep.comfyui_reachable is True
        assert rep.wan_detected is True
        assert rep.gpu_available is True


# ---------------------------------------------------------------- full generate (typed failures)
def test_generate_from_context_unreachable():
    ctx, plan = _ctx()
    reg = registry_from_plan(plan)
    p = WanProvider("http://127.0.0.1:1", timeout=3, t2v_workflow=_real_t2v_workflow())
    opts = GenerationOptions(seed=1, mode=WanMode.T2V)
    with pytest.raises(ComfyError) as e:
        asyncio.run(p.generate_from_context(ctx, opts, registry=reg, project_id="t"))
    assert e.value.code is TypedErrorCode.COMFYUI_UNREACHABLE


def test_generate_from_context_model_not_found():
    routes = {
        "/system_stats": ("json", {"devices": [{"name": "cuda", "type": "cuda", "vram_total": 8*(1024**3)}]}),
        "/object_info": ("json", {"Loader": {"input": {"unet": ["sdxl.safetensors"]}}}),
    }
    ctx, plan = _ctx()
    reg = registry_from_plan(plan)
    with _FakeComfy(routes) as base:
        p = WanProvider(base, timeout=5, t2v_workflow=_real_t2v_workflow())
        opts = GenerationOptions(seed=1, mode=WanMode.T2V)
        with pytest.raises(ComfyError) as e:
            asyncio.run(p.generate_from_context(ctx, opts, registry=reg, project_id="t"))
        assert e.value.code is TypedErrorCode.MODEL_NOT_FOUND


def test_generate_from_context_workflow_not_found():
    routes = _wan_routes()
    ctx, plan = _ctx()
    reg = registry_from_plan(plan)
    with _FakeComfy(routes) as base:
        p = WanProvider(base, timeout=5, t2v_workflow=None)  # no workflow
        opts = GenerationOptions(seed=1, mode=WanMode.T2V)
        with pytest.raises(ComfyError) as e:
            asyncio.run(p.generate_from_context(ctx, opts, registry=reg, project_id="t"))
        assert e.value.code is TypedErrorCode.WORKFLOW_NOT_FOUND


def test_generate_from_context_no_output():
    routes = {
        "/system_stats": ("json", {"devices": [{"name": "cuda", "type": "cuda", "vram_total": 8*(1024**3)}]}),
        "/object_info": ("json", {"Loader": {"input": {"unet": ["wan2.2_5b.safetensors"]}}}),
        "/prompt": ("json", {"prompt_id": "pid-1"}),
        "/history/pid-1": ("json", {"pid-1": {"status": {"status_str": "success"}, "outputs": {"9": {}}}}),
    }
    ctx, plan = _ctx()
    reg = registry_from_plan(plan)
    with _FakeComfy(routes) as base:
        p = WanProvider(base, timeout=5, t2v_workflow=_real_t2v_workflow())
        opts = GenerationOptions(seed=1, mode=WanMode.T2V)
        with pytest.raises(ComfyError) as e:
            asyncio.run(p.generate_from_context(ctx, opts, registry=reg, project_id="t"))
        assert e.value.code is TypedErrorCode.NO_OUTPUT


def test_generate_from_context_invalid_mp4(tmp_path):
    # /view returns bytes that are not a valid MP4 -> verify_output raises.
    routes = _wan_routes(media=b"NOTAMP4")
    ctx, plan = _ctx()
    reg = registry_from_plan(plan)
    with _FakeComfy(routes) as base:
        p = WanProvider(base, timeout=5, t2v_workflow=_real_t2v_workflow())
        opts = GenerationOptions(seed=1, mode=WanMode.T2V)
        with pytest.raises(ComfyError) as e:
            asyncio.run(p.generate_from_context(ctx, opts, registry=reg, project_id="t", destination_dir=tmp_path))
        assert e.value.code is TypedErrorCode.INVALID_MP4


def test_generate_from_context_missing_context_prompt():
    ctx, plan = _ctx()
    ctx.visual_prompt = ""  # break continuity resolution precondition
    reg = registry_from_plan(plan)
    p = WanProvider("http://0.0.0.0:1", timeout=3, t2v_workflow=_real_t2v_workflow())
    opts = GenerationOptions(seed=1, mode=WanMode.T2V)
    with pytest.raises(VideoError) as e:
        asyncio.run(p.generate_from_context(ctx, opts, registry=reg, project_id="t"))
    assert e.value.code is TypedErrorCode.WORKFLOW_INVALID


def test_generate_i2v_requires_registry():
    ctx, plan = _ctx()
    p = WanProvider("http://0.0.0.0:1", timeout=3, i2v_workflow=_real_i2v_workflow())
    opts = GenerationOptions(seed=1, mode=WanMode.I2V)
    with pytest.raises(VideoError) as e:
        asyncio.run(p.generate_from_context(ctx, opts, registry=None, project_id="t"))
    assert e.value.code is TypedErrorCode.INVALID_REFERENCE


def test_generate_i2v_no_available_reference():
    ctx, plan = _ctx()
    reg = ReferenceRegistry()  # empty -> no refs bound to scene
    p = WanProvider("http://0.0.0.0:1", timeout=3, i2v_workflow=_real_i2v_workflow())
    opts = GenerationOptions(seed=1, mode=WanMode.I2V)
    with pytest.raises(VideoError) as e:
        asyncio.run(p.generate_from_context(ctx, opts, registry=reg, project_id="t"))
    assert e.value.code is TypedErrorCode.INVALID_REFERENCE


# ---------------------------------------------------------------- full generate SUCCESS (fake server)
def test_generate_t2v_success_fake(tmp_path):
    """CODE/TEST only: fake server returns bytes; not real runtime verification.

    Uses a real tiny MP4 so verify_output (ffprobe) actually passes.
    """
    import subprocess
    # Generate a real 1-frame MP4 with ffmpeg so ffprobe passes.
    mp4 = tmp_path / "real.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1",
         "-pix_fmt", "yuv420p", str(mp4)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    media = mp4.read_bytes()
    routes = _wan_routes(media=media)
    ctx, plan = _ctx()
    reg = registry_from_plan(plan)
    with _FakeComfy(routes) as base:
        p = WanProvider(base, timeout=5, t2v_workflow=_real_t2v_workflow())
        opts = GenerationOptions(seed=42, mode=WanMode.T2V)
        result, meta = asyncio.run(p.generate_from_context(
            ctx, opts, registry=reg, project_id="fakeproj", destination_dir=tmp_path / "out"))
        assert result.path.exists()
        assert result.path.stat().st_size > 0
        assert meta.seed == 42
        assert meta.mode == "t2v"
        assert meta.model == "wan2.2_5b.safetensors"
        assert meta.output_path == str(result.path)


def test_generate_i2v_success_fake(tmp_path):
    import subprocess
    mp4 = tmp_path / "real.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=64x64:d=0.1",
         "-pix_fmt", "yuv420p", str(mp4)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    media = mp4.read_bytes()
    routes = _wan_routes(media=media)
    ctx, plan = _ctx()
    reg = registry_from_plan(plan)
    # Make a reference image available on disk.
    ref_path = tmp_path / "ref.png"
    ref_path.write_bytes(b"\x89PNG\r\n\x1a\nfakedata")
    for img in reg.images:
        if img.kind == ReferenceKind.CHARACTER:
            img.path = ref_path
    with _FakeComfy(routes) as base:
        p = WanProvider(base, timeout=5, i2v_workflow=_real_i2v_workflow())
        opts = GenerationOptions(seed=7, mode=WanMode.I2V)
        result, meta = asyncio.run(p.generate_from_context(
            ctx, opts, registry=reg, project_id="fakeproj", destination_dir=tmp_path / "out"))
        assert result.path.exists()
        assert meta.mode == "i2v"
        assert meta.reference_ids  # references recorded


# ---------------------------------------------------------------- backward compatibility
def test_comfyui_provider_still_works():
    from app.providers.comfyui import ComfyUIProvider
    p = ComfyUIProvider("http://127.0.0.1:1", timeout=2)
    assert asyncio.run(p.detect()) is False


def test_registry_still_supports_comfyui():
    from app.providers.registry import build_providers
    import app.config as cfg
    cfg.VIDEO_PROVIDERS = "comfyui"
    providers = build_providers()
    assert providers[0].info.name == "comfyui"


def test_registry_supports_wan():
    from app.providers.registry import build_providers
    import app.config as cfg
    cfg.VIDEO_PROVIDERS = "wan"
    providers = build_providers()
    assert providers[0].info.name == "wan"


def test_registry_supports_both():
    from app.providers.registry import build_providers
    import app.config as cfg
    cfg.VIDEO_PROVIDERS = "wan,comfyui"
    providers = build_providers()
    assert [p.info.name for p in providers] == ["wan", "comfyui"]


# ---------------------------------------------------------------- provider capabilities
def test_wan_supports_i2v():
    p = WanProvider("http://0.0.0.0:1")
    assert p.supports_image_to_video() is True


def test_readiness_report_serializable():
    rep = ReadinessReport(ready=False, comfyui_reachable=False, blockers=["x"])
    json.dumps(rep.to_dict())


def test_generate_options_default_mode_t2v():
    assert GenerationOptions().mode == WanMode.T2V

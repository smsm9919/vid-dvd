"""Tests for the ComfyUI provider typed-error behavior (Phase 4).

These tests exercise the typed failure states without a real ComfyUI by using a
small in-process fake HTTP server. They verify that the provider NEVER simulates
success and raises the correct typed code for each failure mode.
"""

import asyncio
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pytest

from app.core.errors import TypedErrorCode, VideoError
from app.providers.comfyui import ComfyError, ComfyUIProvider
from app.providers.base import GenerationRequest


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _FakeHandler(BaseHTTPRequestHandler):
    """Routes are configured per-test via the server's `routes` dict."""

    def log_message(self, *a):  # silence
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
        self.rfile.read(length)
        if path in routes:
            kind, payload = routes[path]
            if kind == "json":
                self._send(200, json.dumps(payload).encode())
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


# --------------------------------------------------------------------- detection
def test_detect_unreachable_returns_false():
    p = ComfyUIProvider("http://127.0.0.1:1", timeout=5)
    assert asyncio.run(p.detect()) is False


def test_health_unreachable_reports_not_ok():
    p = ComfyUIProvider("http://127.0.0.1:1", timeout=5)
    h = asyncio.run(p.health())
    assert h["ok"] is False
    assert h["error"]["code"] == "COMFYUI_UNREACHABLE"


# --------------------------------------------------------------------- workflow
def test_load_workflow_missing_file(tmp_path):
    p = ComfyUIProvider("http://127.0.0.1:1")
    with pytest.raises(ComfyError) as e:
        p.load_workflow(tmp_path / "nope.json")
    assert e.value.code is TypedErrorCode.WORKFLOW_INVALID


def test_load_workflow_invalid_json(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("not json{", encoding="utf-8")
    p = ComfyUIProvider("http://127.0.0.1:1")
    with pytest.raises(ComfyError) as e:
        p.load_workflow(f)
    assert e.value.code is TypedErrorCode.WORKFLOW_INVALID


def test_load_workflow_not_dict(tmp_path):
    f = tmp_path / "list.json"
    f.write_text("[1,2,3]", encoding="utf-8")
    p = ComfyUIProvider("http://0.0.0.0:1")
    with pytest.raises(ComfyError) as e:
        p.load_workflow(f)
    assert e.value.code is TypedErrorCode.WORKFLOW_INVALID


def test_load_workflow_valid(tmp_path):
    f = tmp_path / "wf.json"
    f.write_text(json.dumps({"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "x"}}}), encoding="utf-8")
    p = ComfyUIProvider("http://0.0.0.0:1")
    wf = p.load_workflow(f)
    assert "1" in wf


# --------------------------------------------------------------------- inject
def test_inject_prompt_auto_detect():
    p = ComfyUIProvider("http://0.0.0.0:1")
    wf = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "old"}}}
    out = p.inject_prompt(wf, "new prompt")
    assert out["1"]["inputs"]["text"] == "new prompt"
    # original not mutated
    assert wf["1"]["inputs"]["text"] == "old"


def test_inject_prompt_manual_node_missing():
    p = ComfyUIProvider("http://0.0.0.0:1")
    wf = {"1": {"class_type": "X", "inputs": {"text": "old"}}}
    with pytest.raises(ComfyError) as e:
        p.inject_prompt(wf, "p", node_id="99")
    assert e.value.code is TypedErrorCode.WORKFLOW_INVALID


def test_inject_prompt_no_candidate():
    p = ComfyUIProvider("http://0.0.0.0:1")
    wf = {"1": {"class_type": "X", "inputs": {"foo": "bar"}}}
    with pytest.raises(ComfyError) as e:
        p.inject_prompt(wf, "p")
    assert e.value.code is TypedErrorCode.WORKFLOW_INVALID


# --------------------------------------------------------------------- queue
def test_queue_rejected_returns_workflow_rejected():
    routes = {"/prompt": ("json", {"error": "bad node"})}
    with _FakeComfy(routes) as base:
        p = ComfyUIProvider(base)
        with pytest.raises(ComfyError) as e:
            asyncio.run(p.queue({"1": {}}))
        # Fake server returns 200 with no prompt_id -> WORKFLOW_REJECTED
        assert e.value.code is TypedErrorCode.WORKFLOW_REJECTED


def test_queue_http_error():
    routes = {"/prompt": ("status", 500)}
    with _FakeComfy(routes) as base:
        p = ComfyUIProvider(base)
        with pytest.raises(ComfyError) as e:
            asyncio.run(p.queue({"1": {}}))
        assert e.value.code is TypedErrorCode.WORKFLOW_REJECTED


def test_queue_success_returns_prompt_id():
    routes = {"/prompt": ("json", {"prompt_id": "pid-123"})}
    with _FakeComfy(routes) as base:
        p = ComfyUIProvider(base)
        pid = asyncio.run(p.queue({"1": {}}))
        assert pid == "pid-123"


# --------------------------------------------------------------------- wait
def test_wait_timeout():
    routes = {"/history/pid": ("json", {})}  # never completes
    with _FakeComfy(routes) as base:
        p = ComfyUIProvider(base, timeout=1)
        with pytest.raises(ComfyError) as e:
            asyncio.run(p.wait("pid"))
        assert e.value.code is TypedErrorCode.GENERATION_TIMEOUT


def test_wait_job_error():
    routes = {"/history/pid": ("json", {"pid": {"status": {"status_str": "error", "messages": "boom"}}})}
    with _FakeComfy(routes) as base:
        p = ComfyUIProvider(base, timeout=5)
        with pytest.raises(ComfyError) as e:
            asyncio.run(p.wait("pid"))
        assert e.value.code is TypedErrorCode.WORKFLOW_REJECTED


def test_wait_success_returns_outputs():
    routes = {"/history/pid": ("json", {"pid": {"status": {"status_str": "success"}, "outputs": {"9": {"videos": [{"filename": "out.mp4"}]}}}})}
    with _FakeComfy(routes) as base:
        p = ComfyUIProvider(base, timeout=5)
        outputs = asyncio.run(p.wait("pid"))
        assert "9" in outputs


# --------------------------------------------------------------------- outputs
def test_collect_outputs_empty():
    p = ComfyUIProvider("http://0.0.0.0:1")
    assert p.collect_outputs({}) == []
    assert p.collect_outputs({"9": {}}) == []


def test_download_first_media_no_output():
    p = ComfyUIProvider("http://0.0.0.0:1")
    with pytest.raises(ComfyError) as e:
        asyncio.run(p.download_first_media({}, Path("/tmp/x.mp4")))
    assert e.value.code is TypedErrorCode.NO_OUTPUT


def test_download_first_media_success(tmp_path):
    payload = b"FAKEVIDEOBYTES"
    routes = {
        "/history/pid": ("json", {"pid": {"status": {"status_str": "success"}, "outputs": {"9": {"videos": [{"filename": "out.mp4", "subfolder": "", "type": "output"}]}}}}),
        "/view": ("bytes", payload),
    }
    dest = tmp_path / "clip.mp4"
    with _FakeComfy(routes) as base:
        p = ComfyUIProvider(base, timeout=5)
        outputs = asyncio.run(p.wait("pid"))
        asyncio.run(p.download_first_media(outputs, dest))
        assert dest.exists() and dest.read_bytes() == payload


# --------------------------------------------------------------------- verify
def test_verify_output_missing(tmp_path):
    p = ComfyUIProvider("http://0.0.0.0:1")
    with pytest.raises(ComfyError) as e:
        p.verify_output(tmp_path / "nope.mp4")
    assert e.value.code is TypedErrorCode.INVALID_MP4


def test_verify_output_empty(tmp_path):
    f = tmp_path / "empty.mp4"
    f.write_bytes(b"")
    p = ComfyUIProvider("http://0.0.0.0:1")
    with pytest.raises(ComfyError) as e:
        p.verify_output(f)
    assert e.value.code is TypedErrorCode.INVALID_MP4


def test_verify_output_not_a_video(tmp_path):
    f = tmp_path / "notvideo.mp4"
    f.write_bytes(b"this is not an mp4 at all")
    p = ComfyUIProvider("http://0.0.0.0:1")
    with pytest.raises(ComfyError) as e:
        p.verify_output(f)
    assert e.value.code is TypedErrorCode.INVALID_MP4


# --------------------------------------------------------------------- models
def test_required_models_missing():
    p = ComfyUIProvider("http://0.0.0.0:1", required_models=["wan2.2_5b", "does_not_exist"])
    missing = p._check_required_models(["wan2.2_5b.safetensors", "vae.safetensors"])
    assert "does_not_exist" in missing
    assert "wan2.2_5b" not in missing


def test_validate_reports_unreachable():
    p = ComfyUIProvider("http://127.0.0.1:1", workflow_path=Path("workflows/wan22_ti2v_api.json"))
    report = asyncio.run(p.validate())
    assert report["ok"] is False
    assert report["reachable"] is False


# --------------------------------------------------------------------- full generate
def test_generate_unreachable_raises_typed(tmp_path):
    p = ComfyUIProvider("http://127.0.0.1:1", timeout=2)
    req = GenerationRequest(prompt="a lion", negative_prompt="blurry")
    with pytest.raises(ComfyError) as e:
        asyncio.run(p.generate(req, tmp_path / "out.mp4"))
    assert e.value.code is TypedErrorCode.COMFYUI_UNREACHABLE
    # NEVER simulates success: no output file created
    assert not (tmp_path / "out.mp4").exists()


def test_generate_no_output_raises_typed(tmp_path):
    # Reachable, but job returns no media -> NO_OUTPUT, never a fake file.
    routes = {
        "/system_stats": ("json", {"devices": [{"name": "cpu"}]}),
        "/object_info": ("json", {"Loader": {"input": {"unet": ["wan2.2_5b"]}}}),
        "/prompt": ("json", {"prompt_id": "pid"}),
        "/history/pid": ("json", {"pid": {"status": {"status_str": "success"}, "outputs": {"9": {}}}}),
    }
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps({"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "x"}}}), encoding="utf-8")
    with _FakeComfy(routes) as base:
        p = ComfyUIProvider(base, timeout=5, workflow_path=wf)
        req = GenerationRequest(prompt="a lion", negative_prompt="blurry")
        with pytest.raises(ComfyError) as e:
            asyncio.run(p.generate(req, tmp_path / "out.mp4"))
        assert e.value.code is TypedErrorCode.NO_OUTPUT
        assert not (tmp_path / "out.mp4").exists()

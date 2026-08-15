"""Provider detection and registry.

Builds the list of configured providers, selects the first healthy one, and
raises a typed ``NO_PROVIDER`` error when none is available. The registry never
fabricates a provider: if detection fails for every candidate, generation is
rejected with :class:`~app.core.errors.TypedErrorCode.NO_PROVIDER`.
"""

from __future__ import annotations

from typing import Optional

from .. import config
from ..config import (
    COMFYUI_PROMPT_FIELD,
    COMFYUI_PROMPT_NODE_ID,
    COMFYUI_TIMEOUT_SECONDS,
    COMFYUI_URL,
    WAN_I2V_WORKFLOW,
    WAN_REQUIRED_MODEL,
    WAN_T2V_WORKFLOW,
    WORKFLOW_PATH,
)
from ..core.errors import TypedErrorCode, VideoError
from ..core.logging import log
from .base import VideoProvider
from .comfyui import ComfyUIProvider
from .wan import WanProvider


def build_providers() -> list[VideoProvider]:
    """Instantiate all configured providers in preference order.

    Reads ``config.VIDEO_PROVIDERS`` dynamically so test/runtime overrides of
    the module attribute take effect.
    """
    providers: list[VideoProvider] = []
    raw = getattr(config, "VIDEO_PROVIDERS", "")
    names = [n.strip().lower() for n in raw.split(",") if n.strip()] or ["comfyui"]
    for name in names:
        if name == "comfyui":
            providers.append(
                ComfyUIProvider(
                    config.COMFYUI_URL,
                    config.COMFYUI_TIMEOUT_SECONDS,
                    workflow_path=WORKFLOW_PATH,
                    prompt_node_id=COMFYUI_PROMPT_NODE_ID,
                    prompt_field=COMFYUI_PROMPT_FIELD,
                )
            )
        elif name == "wan":
            providers.append(
                WanProvider(
                    config.COMFYUI_URL,
                    config.COMFYUI_TIMEOUT_SECONDS,
                    t2v_workflow=WAN_T2V_WORKFLOW,
                    i2v_workflow=WAN_I2V_WORKFLOW,
                    required_model_substring=WAN_REQUIRED_MODEL,
                )
            )
        else:
            log("REGISTRY", f"Unknown provider '{name}' in VIDEO_PROVIDERS; ignored")
    return providers


async def select_provider(providers: Optional[list[VideoProvider]] = None) -> VideoProvider:
    """Return the first provider that reports healthy via :meth:`detect`.

    Raises ``NO_PROVIDER`` if none is available. Detection is real: each
    provider's ``detect()`` is actually invoked.
    """
    providers = providers if providers is not None else build_providers()
    for p in providers:
        if await p.detect():
            log("PROVIDER", f"selected {p.info.name}")
            return p
    raise VideoError(
        TypedErrorCode.NO_PROVIDER,
        "No video provider available. Configure ComfyUI (COMFYUI_URL) or another provider.",
        context={"configured": [p.info.name for p in providers]},
    )


async def provider_status(providers: Optional[list[VideoProvider]] = None) -> list[dict]:
    """Return a per-provider health summary for the dashboard / health endpoint."""
    providers = providers if providers is not None else build_providers()
    out: list[dict] = []
    for p in providers:
        try:
            h = await p.health()
        except VideoError as e:
            h = {"ok": False, "error": e.to_dict()}
        out.append({"name": p.info.name, "kind": p.info.kind, **h})
    return out

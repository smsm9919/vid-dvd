"""Structured observability logging for the production pipeline.

Emits concise, prefixed log lines suitable for the dashboard and diagnostics,
e.g.::

    [PROVIDER] ComfyUI detected vram=8GB
    [JOB] submitted prompt_id=...
    [QC] passed path=...

This is intentionally lightweight (stdlib ``logging``) and does not fake any
metrics — it only reports events that actually happened.
"""

from __future__ import annotations

import logging
from typing import Any

_LOGGER = logging.getLogger("videofactory")
if not _LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)


def log(stage: str, message: str, **context: Any) -> None:
    """Emit a ``[STAGE] message key=value ...`` log line.

    Only real events should be logged; never log a success that did not happen.
    """
    parts = [f"{k}={_fmt(v)}" for k, v in context.items() if v is not None]
    suffix = (" " + " ".join(parts)) if parts else ""
    _LOGGER.info("[%s] %s%s", stage.upper(), message, suffix)


def _fmt(v: Any) -> str:
    if isinstance(v, str):
        return v
    return repr(v)

"""Backwards-compatible shim for the legacy ``app.comfy`` public API.

The ComfyUI implementation now lives in :mod:`app.providers.comfyui` and exposes
typed :class:`~app.core.errors.VideoError` failures. This module re-exports the
two names the rest of the codebase historically depended on — ``ComfyError`` and
``ComfyClient`` — so existing imports and call sites keep working unchanged.

``ComfyError`` remains a ``RuntimeError`` subclass (via ``VideoError``) and
``ComfyClient`` exposes the same methods (``health``, ``load_workflow``,
``inject_prompt``, ``queue``, ``wait``, ``download_first_media``) with the same
signatures. The only behavioral change is that failures now carry typed error
codes instead of bare strings.
"""

from .providers.comfyui import ComfyClient, ComfyError

__all__ = ["ComfyClient", "ComfyError"]

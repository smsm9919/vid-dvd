"""Free-first provider router (Phase 13).

Selects the best provider for a given capability + language, enforcing the
free-first policy:

1. Only providers that are ``available`` (real detection) are considered.
2. Paid providers are excluded unless ``ALLOW_PAID_PROVIDERS=true`` AND the
   per-call estimate is within ``MAX_PAID_COST_USD``.
3. Among eligible providers, free + CPU-fallback providers are preferred
   (free-first), then ordered by a stable preference list.
4. If no provider matches the requested capability/language, raises
   ``NO_PROVIDER`` (never fakes a provider).

The router is the single place where the "free-first" guarantee is enforced.
"""

from __future__ import annotations

from typing import Any, Optional

from .. import config
from ..core.errors import TypedErrorCode, VideoError
from ..core.logging import log
from .contracts import Capability, ProviderKind, ProviderMeta


class ProviderRouter:
    """Routes a capability request to the best eligible provider.

    Providers are registered as ``(meta_provider, instance)`` pairs where
    ``meta_provider`` exposes ``meta() -> ProviderMeta`` and ``available``.
    """

    def __init__(self) -> None:
        self._providers: list[tuple[Any, Any]] = []

    def register(self, meta_provider: Any, instance: Any) -> None:
        """Register a provider. ``meta_provider`` must expose meta() + available."""
        self._providers.append((meta_provider, instance))

    def register_all(self, items: list[tuple[Any, Any]]) -> None:
        for mp, inst in items:
            self.register(mp, inst)

    @property
    def providers(self) -> list[tuple[Any, Any]]:
        return list(self._providers)

    def _eligible(self, meta: ProviderMeta, available: bool, *,
                  capability: Optional[Capability], language: Optional[str],
                  allow_paid: bool, max_cost: float) -> tuple[bool, str]:
        """Return (eligible, reason). reason is empty when eligible."""
        if not available:
            return False, "not available"
        if capability is not None and not meta.capability.supports(capability):
            return False, f"missing capability {capability.value}"
        if language is not None and meta.capability.languages and language not in meta.capability.languages:
            return False, f"language {language} not supported"
        if meta.cost.is_paid:
            if not allow_paid:
                return False, "paid provider blocked (FREE_FIRST)"
            if meta.cost.estimate_per_call > max_cost:
                return False, (f"paid cost {meta.cost.estimate_per_call} exceeds "
                               f"max {max_cost}")
        return True, ""

    def select(self, *, capability: Optional[Capability] = None,
               language: Optional[str] = None) -> Optional[tuple[Any, Any]]:
        """Select the best eligible provider or None.

        Preference order: free + CPU-fallback > free > paid (if allowed).
        Within a tier, registration order is preserved (stable).
        """
        allow_paid = config.ALLOW_PAID_PROVIDERS
        max_cost = config.MAX_PAID_COST_USD
        candidates: list[tuple[int, int, Any, Any]] = []
        for idx, (mp, inst) in enumerate(self._providers):
            try:
                meta = mp.meta() if hasattr(mp, "meta") else inst.meta()
                available = getattr(mp, "available", getattr(inst, "available", False))
            except Exception:
                continue
            eligible, reason = self._eligible(meta, available, capability=capability,
                                               language=language, allow_paid=allow_paid,
                                               max_cost=max_cost)
            if not eligible:
                continue
            # Tier: 0 = free + cpu_fallback, 1 = free (no cpu fallback), 2 = paid
            if meta.cost.is_paid:
                tier = 2
            elif meta.runtime.cpu_fallback:
                tier = 0
            else:
                tier = 1
            candidates.append((tier, idx, mp, inst))
        if not candidates:
            return None
        candidates.sort(key=lambda c: (c[0], c[1]))
        chosen = candidates[0]
        log("ROUTER", f"selected {chosen[2].name if hasattr(chosen[2], 'name') else chosen[3].name}",
            capability=capability.value if capability else None,
            language=language, tier=chosen[0])
        return (chosen[2], chosen[3])

    def select_or_raise(self, *, capability: Optional[Capability] = None,
                        language: Optional[str] = None) -> tuple[Any, Any]:
        result = self.select(capability=capability, language=language)
        if result is None:
            raise VideoError(
                TypedErrorCode.NO_PROVIDER,
                "No eligible provider for the requested capability/language.",
                context={"capability": capability.value if capability else None,
                         "language": language,
                         "free_first": config.FREE_FIRST,
                         "allow_paid": config.ALLOW_PAID_PROVIDERS},
            )
        return result

    def status(self) -> list[dict[str, Any]]:
        """Per-provider routing status for the dashboard."""
        out = []
        allow_paid = config.ALLOW_PAID_PROVIDERS
        max_cost = config.MAX_PAID_COST_USD
        for mp, inst in self._providers:
            try:
                meta = mp.meta() if hasattr(mp, "meta") else inst.meta()
                available = getattr(mp, "available", getattr(inst, "available", False))
            except Exception as e:
                out.append({"name": "?", "available": False, "error": str(e)})
                continue
            eligible, reason = self._eligible(meta, available, capability=None,
                                               language=None, allow_paid=allow_paid,
                                               max_cost=max_cost)
            out.append({
                "name": meta.name, "kind": meta.kind.value, "available": available,
                "eligible": eligible, "reason": reason or "eligible",
                "cost": meta.cost.to_dict(), "license": meta.license.to_dict(),
                "capabilities": [c.value for c in meta.capability.capabilities],
                "languages": meta.capability.languages,
                "requires_gpu": meta.runtime.requires_gpu,
                "cpu_fallback": meta.runtime.cpu_fallback,
            })
        return out


# --------------------------------------------------------------------- factories
def build_default_router() -> ProviderRouter:
    """Build the default free-first router with all configured providers."""
    router = ProviderRouter()
    # Stock providers (free, network) — Wikimedia first (keyless).
    try:
        from .stock_adapters import build_stock_providers
        for p in build_stock_providers():
            router.register(p, p)
    except Exception:
        pass
    # Kokoro TTS (free, local CPU, Apache 2.0 — preferred over Piper).
    try:
        from ..voice.kokoro import build_kokoro_provider
        k = build_kokoro_provider()
        router.register(k, k)
    except Exception:
        pass
    # Piper TTS (free, local CPU, GPL-3.0 opt-in — fallback).
    try:
        from ..voice.piper import build_piper_provider
        p = build_piper_provider()
        router.register(p, p)
    except Exception:
        pass
    return router


def build_tts_providers() -> list:
    """Build the list of real TTS providers for select_tts_provider.

    Kokoro (Apache 2.0) is preferred over Piper (GPL-3.0) for commercial use.
    """
    providers = []
    try:
        from ..voice.kokoro import build_kokoro_provider
        k = build_kokoro_provider()
        if k.available:
            providers.append(k)
    except Exception:
        pass
    try:
        from ..voice.piper import build_piper_provider
        p = build_piper_provider()
        if p.available:
            providers.append(p)
    except Exception:
        pass
    return providers

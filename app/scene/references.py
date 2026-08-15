"""Reference-image data model and resolution (Phase 7).

Establishes the architecture for future image-to-video / reference-image
conditioning (character / product / environment reference images). Phase 7 only
defines the data model and resolution mechanism — it does NOT execute any
Wan/ComfyUI I2V generation (that is Phase 8).

A :class:`ReferenceRegistry` maps stable identity IDs to reference images so the
scene resolver can attach the correct references to each scene's generation
context. References are resolved by ID, never guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from ..brain.models import ProductionPlan


class ReferenceKind(str, Enum):
    """What a reference image conditions."""

    CHARACTER = "character"
    PRODUCT = "product"
    ENVIRONMENT = "environment"
    STYLE = "style"


@dataclass
class ReferenceImage:
    """A reference image bound to a stable identity ID.

    ``path``/``url`` may both be empty during planning; Phase 8 resolves them to
    real on-disk files before generation. The ``target_id`` ties the image to a
    :class:`~app.brain.models.CharacterIdentity` / ``ProductIdentity`` id or an
    environment key, so references are never ambiguous.
    """

    kind: ReferenceKind
    target_id: str
    path: Optional[Path] = None
    url: Optional[str] = None
    description: Optional[str] = None
    weight: float = 1.0  # conditioning strength hint for Phase 8

    def is_available(self) -> bool:
        """True if the reference points to a real, non-empty file on disk."""
        if self.path is None:
            return False
        p = Path(self.path)
        return p.exists() and p.stat().st_size > 0

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "target_id": self.target_id,
            "path": str(self.path) if self.path else None,
            "url": self.url,
            "description": self.description,
            "weight": self.weight,
            "available": self.is_available(),
        }


@dataclass
class ReferenceRegistry:
    """Maps stable identity IDs -> reference images.

    Build from a plan's continuity memory so every character/product id can be
    looked up. Resolution is deterministic and ID-based.
    """

    images: list[ReferenceImage] = field(default_factory=list)

    def add(self, image: ReferenceImage) -> None:
        self.images.append(image)

    def for_target(self, target_id: str, kind: Optional[ReferenceKind] = None) -> list[ReferenceImage]:
        if kind is None:
            return [img for img in self.images if img.target_id == target_id]
        return [img for img in self.images if img.target_id == target_id and img.kind == kind]

    def for_character(self, character_id: str) -> list[ReferenceImage]:
        return self.for_target(character_id, ReferenceKind.CHARACTER)

    def for_product(self, product_id: str) -> list[ReferenceImage]:
        return self.for_target(product_id, ReferenceKind.PRODUCT)

    def environment_references(self) -> list[ReferenceImage]:
        return [img for img in self.images if img.kind == ReferenceKind.ENVIRONMENT]

    def available(self) -> list[ReferenceImage]:
        """Only references whose backing file actually exists (Phase 8 readiness)."""
        return [img for img in self.images if img.is_available()]

    def to_dict(self) -> dict:
        return {"images": [img.to_dict() for img in self.images]}


def registry_from_plan(plan: ProductionPlan) -> ReferenceRegistry:
    """Build a registry keyed by the plan's stable identity IDs.

    Ensures stable IDs are assigned first so the registry and the plan agree on
    IDs. Reference image files are NOT created here — only the binding slots are
    prepared so Phase 8 can populate ``path``/``url``.
    """
    from .continuity import ensure_stable_ids
    ensure_stable_ids(plan)
    reg = ReferenceRegistry()
    cm = plan.continuity
    for ch in cm.characters:
        cid = ch.id or "character_1"
        reg.add(ReferenceImage(
            kind=ReferenceKind.CHARACTER, target_id=cid,
            description=f"Reference for character {ch.name or ch.role or cid}.",
        ))
    if cm.product is not None:
        pid = cm.product.id or "product_1"
        reg.add(ReferenceImage(
            kind=ReferenceKind.PRODUCT, target_id=pid,
            description=f"Reference for product {cm.product.name or pid}.",
        ))
    if cm.environment or cm.location:
        reg.add(ReferenceImage(
            kind=ReferenceKind.ENVIRONMENT, target_id="environment",
            description=f"Reference for environment: {cm.location or cm.environment}.",
        ))
    return reg

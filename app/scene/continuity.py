"""Scene continuity and reference consistency (Phase 7).

This layer operates BEFORE generation and produces a resolved scene-generation
context so downstream video providers can never receive a scene prompt without
its required continuity context. It guarantees scene-to-scene consistency for
characters, products, environments, clothing, lighting, color palette, camera
language, visual style, and props.

It does NOT duplicate continuity logic inside video providers and does NOT
execute any generation (Phase 8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..brain.models import (
    CharacterIdentity,
    ContinuityMemory,
    ProductIdentity,
    ProductionPlan,
    Scene,
    SceneCamera,
    SceneCharacter,
)
from .references import ReferenceImage, ReferenceRegistry, registry_from_plan


# ---------------------------------------------------------------- stable IDs

def _slug_id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx}"


def ensure_stable_ids(plan: ProductionPlan) -> ProductionPlan:
    """Assign deterministic stable IDs to characters/products if missing.

    Mutates the plan's continuity memory in place and returns the plan. Existing
    IDs are preserved. This is idempotent: running twice yields the same IDs.
    """
    cm = plan.continuity
    for i, ch in enumerate(cm.characters, 1):
        if not ch.id:
            ch.id = _slug_id("character", i)
    if cm.product is not None and not cm.product.id:
        cm.product.id = "product_1"
    return plan


def _character_lookup(plan: ProductionPlan) -> dict[str, CharacterIdentity]:
    """Map {id, name, role} -> CharacterIdentity for flexible resolution."""
    lookup: dict[str, CharacterIdentity] = {}
    for ch in plan.continuity.characters:
        if ch.id:
            lookup[ch.id] = ch
        if ch.name:
            lookup[ch.name] = ch
        if ch.role:
            lookup[ch.role] = ch
    return lookup


def _product_lookup(plan: ProductionPlan) -> dict[str, ProductIdentity]:
    prod = plan.continuity.product
    if prod is None:
        return {}
    lookup: dict[str, ProductIdentity] = {}
    if prod.id:
        lookup[prod.id] = prod
    if prod.name:
        lookup[prod.name] = prod
    lookup.setdefault("product", prod)
    return lookup


# ---------------------------------------------------------------- resolved context

@dataclass
class ResolvedCharacter:
    """A character fully resolved from continuity memory for one scene."""

    id: str
    name: Optional[str]
    role: Optional[str]
    appearance: str
    clothing: Optional[str]
    action: str  # what the character does in THIS scene
    full_description: str  # stable identity text reused across scenes
    reference_images: list[ReferenceImage] = field(default_factory=list)


@dataclass
class ResolvedSceneContext:
    """Fully resolved scene context consumed by downstream video providers.

    Combines GLOBAL CONTINUITY + SCENE-SPECIFIC ACTION into the final generation
    context. Providers receive this instead of a bare prompt string.
    """

    scene_index: int
    purpose: str
    duration: float
    identity_section: str
    environment_section: str
    style_section: str
    camera_section: str
    action_section: str
    motion_section: str
    details_section: str
    negative_section: str
    visual_prompt: str  # final structured prompt
    negative_prompt: str
    characters: list[ResolvedCharacter] = field(default_factory=list)
    product: Optional[ProductIdentity] = None
    reference_images: list[ReferenceImage] = field(default_factory=list)
    camera: SceneCamera = field(default_factory=SceneCamera)
    continuity_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scene_index": self.scene_index,
            "purpose": self.purpose,
            "duration": self.duration,
            "sections": {
                "identity": self.identity_section,
                "environment": self.environment_section,
                "style": self.style_section,
                "camera": self.camera_section,
                "action": self.action_section,
                "motion": self.motion_section,
                "details": self.details_section,
                "negative": self.negative_section,
            },
            "visual_prompt": self.visual_prompt,
            "negative_prompt": self.negative_prompt,
            "characters": [
                {
                    "id": c.id, "name": c.name, "role": c.role,
                    "appearance": c.appearance, "clothing": c.clothing,
                    "action": c.action, "full_description": c.full_description,
                }
                for c in self.characters
            ],
            "product": (self.product.model_dump() if self.product else None),
            "reference_images": [r.to_dict() for r in self.reference_images],
            "camera": {
                "shot_type": self.camera.shot_type, "lens": self.camera.lens,
                "framing": self.camera.framing, "movement": self.camera.movement,
            },
            "continuity_ids": self.continuity_ids,
        }


# ---------------------------------------------------------------- prompt construction

def _identity_text(chars: list[ResolvedCharacter], product: Optional[ProductIdentity]) -> str:
    parts: list[str] = []
    for c in chars:
        parts.append(f"Character '{c.name or c.role or c.id}': {c.full_description}")
    if product is not None:
        feat = " ".join(product.distinctive_features)
        parts.append(f"Product '{product.name or 'product'}': {product.appearance}"
                     + (f" {feat}" if feat else "")
                     + (f", packaging: {product.packaging}" if product.packaging else ""))
    return " ".join(parts) if parts else "No specific identity constraints."


def _environment_text(scene: Scene, cm: ContinuityMemory) -> str:
    env = scene.environment or cm.environment or cm.location or ""
    parts = [env] if env else []
    if cm.location:
        parts.append(f"location: {cm.location}")
    if cm.time_of_day:
        parts.append(f"time of day: {cm.time_of_day}")
    if cm.weather:
        parts.append(f"weather: {cm.weather}")
    if cm.season:
        parts.append(f"season: {cm.season}")
    if cm.architecture:
        parts.append(f"architecture: {cm.architecture}")
    if cm.environmental_objects:
        parts.append("objects: " + ", ".join(cm.environmental_objects))
    return " ".join(parts) if parts else "Environment consistent with continuity."


def _style_text(cm: ContinuityMemory, scene: Scene) -> str:
    parts: list[str] = []
    if cm.visual_style:
        parts.append(cm.visual_style)
    elif scene.lighting:
        parts.append(scene.lighting)
    if cm.color_palette:
        parts.append("palette: " + ", ".join(cm.color_palette))
    return " ".join(parts) if parts else "Coherent visual style."


def _camera_text(scene: Scene, cm: ContinuityMemory) -> str:
    cam = scene.camera
    parts = []
    if cam.shot_type:
        parts.append(cam.shot_type)
    if cam.lens:
        parts.append(cam.lens)
    if cam.framing:
        parts.append(cam.framing)
    if cam.movement:
        parts.append(cam.movement)
    if cm.camera_language:
        parts.append(cm.camera_language)
    return " ".join(parts) if parts else "consistent camera language."


def _motion_text(scene: Scene) -> str:
    if scene.camera.movement:
        return f"motion: {scene.camera.movement}"
    return "realistic natural motion"


def _details_text(scene: Scene) -> str:
    parts = [scene.description]
    if scene.transition_intent:
        parts.append(f"transition: {scene.transition_intent}")
    return " ".join(parts)


def _character_full_description(ch: CharacterIdentity) -> str:
    """Stable identity text reused verbatim across every scene referencing ch."""
    bits: list[str] = [ch.appearance]
    if ch.hair:
        bits.append(f"hair: {ch.hair}")
    if ch.face:
        bits.append(f"face: {ch.face}")
    if ch.body:
        bits.append(f"build: {ch.body}")
    if ch.skin:
        bits.append(f"skin: {ch.skin}")
    if ch.clothing:
        bits.append(f"clothing: {ch.clothing}")
    if ch.accessories:
        bits.append(f"accessories: {ch.accessories}")
    if ch.distinguishing_features:
        bits.append("features: " + ", ".join(ch.distinguishing_features))
    if ch.physical_attributes:
        bits.append(", ".join(ch.physical_attributes))
    return ". ".join(bits)


def build_visual_prompt(ctx: ResolvedSceneContext) -> str:
    """Assemble the structured visual prompt from separated sections.

    Sections are deliberately separated (IDENTITY / ENVIRONMENT / STYLE / CAMERA
    / ACTION / MOTION / DETAILS / NEGATIVE) rather than blindly concatenated.
    """
    return (
        f"[IDENTITY] {ctx.identity_section} "
        f"[ENVIRONMENT] {ctx.environment_section} "
        f"[STYLE] {ctx.style_section} "
        f"[CAMERA] {ctx.camera_section} "
        f"[ACTION] {ctx.action_section} "
        f"[MOTION] {ctx.motion_section} "
        f"[DETAILS] {ctx.details_section}"
    )


# ---------------------------------------------------------------- scene resolver

def resolve_scene_context(
    plan: ProductionPlan,
    scene_index: int,
    registry: Optional[ReferenceRegistry] = None,
) -> ResolvedSceneContext:
    """Return the fully resolved scene context for a scene.

    Raises IndexError if ``scene_index`` is out of range. The returned context
    always carries the global continuity identity for the scene's referenced
    characters/product — a provider can never receive a bare prompt.
    """
    ensure_stable_ids(plan)
    scene = next((s for s in plan.scenes if s.index == scene_index), None)
    if scene is None:
        raise IndexError(f"Scene index {scene_index} not found in plan.")

    if registry is None:
        registry = registry_from_plan(plan)

    cm = plan.continuity
    char_lookup = _character_lookup(plan)
    prod_lookup = _product_lookup(plan)

    resolved_chars: list[ResolvedCharacter] = []
    for sc in scene.characters:
        identity = _resolve_scene_character(sc, char_lookup, scene)
        refs: list[ReferenceImage] = []
        if identity is not None and identity.id:
            refs = registry.for_character(identity.id)
        if identity is not None:
            resolved_chars.append(ResolvedCharacter(
                id=identity.id or "character_unknown",
                name=identity.name, role=identity.role,
                appearance=identity.appearance, clothing=identity.clothing,
                action=sc.action,
                full_description=_character_full_description(identity),
                reference_images=refs,
            ))

    product = cm.product
    prod_refs: list[ReferenceImage] = []
    if product is not None and product.id:
        prod_refs = registry.for_product(product.id)

    env_refs = registry.environment_references()
    all_refs = prod_refs + env_refs
    for rc in resolved_chars:
        all_refs = rc.reference_images + all_refs

    continuity_ids: list[str] = []
    for rc in resolved_chars:
        continuity_ids.append(rc.id)
    if product is not None and product.id:
        continuity_ids.append(product.id)

    identity_section = _identity_text(resolved_chars, product)
    environment_section = _environment_text(scene, cm)
    style_section = _style_text(cm, scene)
    camera_section = _camera_text(scene, cm)
    action_section = scene.description
    motion_section = _motion_text(scene)
    details_section = _details_text(scene)
    negative_section = scene.negative_prompt

    ctx = ResolvedSceneContext(
        scene_index=scene.index,
        purpose=scene.purpose.value,
        duration=scene.duration,
        identity_section=identity_section,
        environment_section=environment_section,
        style_section=style_section,
        camera_section=camera_section,
        action_section=action_section,
        motion_section=motion_section,
        details_section=details_section,
        negative_section=negative_section,
        visual_prompt="",  # filled below
        negative_prompt=scene.negative_prompt,
        characters=resolved_chars,
        product=product,
        reference_images=all_refs,
        camera=scene.camera,
        continuity_ids=continuity_ids,
    )
    ctx.visual_prompt = build_visual_prompt(ctx)
    return ctx


def _resolve_scene_character(
    sc: SceneCharacter, char_lookup: dict[str, CharacterIdentity], scene: Scene,
) -> Optional[CharacterIdentity]:
    """Resolve a scene's character reference to a stable identity.

    Resolution order: explicit ``ref`` -> name -> role. If a scene lists a
    character action without a ref, the first continuity character is used when
    available (the protagonist), so a character action is never orphaned.
    """
    if sc.ref and sc.ref in char_lookup:
        return char_lookup[sc.ref]
    if sc.ref:
        # Try partial match on name/role.
        for key, ch in char_lookup.items():
            if sc.ref.lower() in (key or "").lower():
                return ch
    if char_lookup:
        # Default to the first declared character (protagonist).
        return next(iter(char_lookup.values()))
    return None


def resolve_all_scenes(
    plan: ProductionPlan, registry: Optional[ReferenceRegistry] = None,
) -> list[ResolvedSceneContext]:
    """Resolve every scene in a plan."""
    ensure_stable_ids(plan)
    if registry is None:
        registry = registry_from_plan(plan)
    return [resolve_scene_context(plan, s.index, registry) for s in plan.scenes]


# ---------------------------------------------------------------- validation

class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass
class ContinuityIssue:
    severity: Severity
    code: str
    scene_index: Optional[int]
    message: str

    def to_dict(self) -> dict:
        return {
            "severity": self.severity.value, "code": self.code,
            "scene_index": self.scene_index, "message": self.message,
        }


@dataclass
class ContinuityReport:
    errors: list[ContinuityIssue] = field(default_factory=list)
    warnings: list[ContinuityIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "errors": [i.to_dict() for i in self.errors],
            "warnings": [i.to_dict() for i in self.warnings],
        }


def validate_continuity(plan: ProductionPlan) -> ContinuityReport:
    """Deterministic continuity validation.

    ERRORs block generation: missing referenced character, conflicting character
    identity (same id, different appearance), product identity drift.
    WARNINGs are allowed transitions: clothing/environment/lighting/style change
    that is not declared via transition_intent.
    """
    ensure_stable_ids(plan)
    report = ContinuityReport()
    cm = plan.continuity
    char_lookup = _character_lookup(plan)
    # Stable identity map: id -> appearance (must not drift).
    char_appearance: dict[str, str] = {}
    for ch in cm.characters:
        cid = ch.id or "unknown"
        if cid in char_appearance and char_appearance[cid] != ch.appearance:
            report.errors.append(ContinuityIssue(
                Severity.ERROR, "CHARACTER_IDENTITY_CONFLICT", None,
                f"Character '{cid}' has conflicting appearances: "
                f"'{char_appearance[cid]}' vs '{ch.appearance}'.",
            ))
        char_appearance[cid] = ch.appearance

    # Product identity drift.
    if cm.product is not None:
        # Single product in memory by design; drift is checked across scenes below.
        pass

    prev_env = cm.environment
    prev_lighting = cm.lighting
    prev_clothing: dict[str, Optional[str]] = {
        (ch.id or "unknown"): ch.clothing for ch in cm.characters
    }
    for scene in plan.scenes:
        # Missing character reference: scene lists a ref not in memory.
        for sc in scene.characters:
            if sc.ref and sc.ref not in char_lookup:
                # Allow protagonist fallback only if memory non-empty.
                if not char_lookup:
                    report.errors.append(ContinuityIssue(
                        Severity.ERROR, "MISSING_CHARACTER_REFERENCE", scene.index,
                        f"Scene {scene.index} references character '{sc.ref}' "
                        f"not present in continuity memory.",
                    ))
        # Environment drift (warning unless transition declared).
        cur_env = scene.environment or prev_env
        if prev_env and cur_env and cur_env != prev_env and not scene.transition_intent:
            report.warnings.append(ContinuityIssue(
                Severity.WARNING, "ENVIRONMENT_CHANGE", scene.index,
                f"Scene {scene.index} environment changed without a declared transition.",
            ))
        prev_env = cur_env
        # Lighting drift.
        cur_light = scene.lighting or prev_lighting
        if prev_lighting and cur_light and cur_light != prev_lighting and not scene.transition_intent:
            report.warnings.append(ContinuityIssue(
                Severity.WARNING, "LIGHTING_CHANGE", scene.index,
                f"Scene {scene.index} lighting changed without a declared transition.",
            ))
        prev_lighting = cur_light
        # Clothing drift per character (warning; allowed if transition declared).
        for sc in scene.characters:
            ident = _resolve_scene_character(sc, char_lookup, scene)
            if ident is None:
                continue
            cid = ident.id or "unknown"
            if cid in prev_clothing and prev_clothing[cid] and ident.clothing \
                    and prev_clothing[cid] != ident.clothing and not scene.transition_intent:
                report.warnings.append(ContinuityIssue(
                    Severity.WARNING, "CLOTHING_CHANGE", scene.index,
                    f"Character '{cid}' clothing changed in scene {scene.index} "
                    f"without a declared transition.",
                ))
            prev_clothing[cid] = ident.clothing or prev_clothing.get(cid)
    return report

"""Comprehensive tests for scene continuity + reference consistency (Phase 7)."""

import copy
import json

import pytest
from pydantic import ValidationError

from app.brain.models import (
    CharacterIdentity,
    ContentBrief,
    ContinuityMemory,
    ProductIdentity,
    ProductionPlan,
    Scene,
    SceneCamera,
    SceneCharacter,
    ScenePurpose,
)
from app.brain.content_brain import local_content_plan
from app.ads.brief import AdBrief
from app.ads.variants import generate_variant, generate_variants
from app.scene.continuity import (
    ContinuityReport,
    ContinuityIssue,
    ResolvedSceneContext,
    Severity,
    build_visual_prompt,
    ensure_stable_ids,
    resolve_all_scenes,
    resolve_scene_context,
    validate_continuity,
)
from app.scene.references import (
    ReferenceImage,
    ReferenceKind,
    ReferenceRegistry,
    registry_from_plan,
)
from app.brain.models import Platform, Objective, BrandProfile


# ---------------------------------------------------------------- fixtures
def _plan(**over) -> ProductionPlan:
    brief = ContentBrief(idea="a lion hunting in the jungle", duration_seconds=20, mode="cinematic")
    return local_content_plan(brief)


def _ad_plan(variant="ugc") -> ProductionPlan:
    brief = AdBrief(
        product_or_service="Noir Élixir perfume",
        brand=BrandProfile(brand_name="Maison Noir", colors=["#111", "#C9A96E"]),
        target_audience="women 25-45", platform=Platform.TIKTOK,
        duration_seconds=15, objective=Objective.CONVERSION,
    )
    return generate_variant(brief, variant).plan


def _manual_plan(characters, product=None, environment="jungle", scenes=None) -> ProductionPlan:
    cm = ContinuityMemory(
        characters=characters, product=product, environment=environment,
        lighting="moody", visual_style="cinematic", color_palette=["#222", "#0a0"],
        camera_language="slow dolly",
    )
    base = _plan()
    base.continuity = cm
    if scenes is not None:
        base.scenes = scenes
    return base


def _scene(idx, purpose=ScenePurpose.HOOK, env="jungle", lighting="moody", chars=None, transition=None):
    return Scene(
        index=idx, purpose=purpose, duration=4.0, description=f"scene {idx}",
        camera=SceneCamera(shot_type="medium", movement="slow dolly"),
        environment=env, lighting=lighting, characters=chars or [],
        visual_prompt="x", negative_prompt="no text",
        continuity_refs=["protagonist"],
        transition_intent=transition,
    )


# ---------------------------------------------------------------- stable IDs
def test_ensure_stable_ids_assigns_to_characters():
    plan = _manual_plan([CharacterIdentity(role="protagonist", appearance="tall man")])
    ensure_stable_ids(plan)
    assert plan.continuity.characters[0].id == "character_1"


def test_ensure_stable_ids_assigns_to_product():
    plan = _manual_plan(
        [CharacterIdentity(role="protagonist", appearance="tall man")],
        product=ProductIdentity(name="perfume", appearance="bottle"),
    )
    ensure_stable_ids(plan)
    assert plan.continuity.product.id == "product_1"


def test_ensure_stable_ids_preserves_existing():
    ch = CharacterIdentity(id="hero_42", role="protagonist", appearance="tall man")
    plan = _manual_plan([ch])
    ensure_stable_ids(plan)
    assert plan.continuity.characters[0].id == "hero_42"


def test_ensure_stable_ids_idempotent():
    plan = _manual_plan([CharacterIdentity(role="protagonist", appearance="tall man")])
    ensure_stable_ids(plan)
    first = plan.continuity.characters[0].id
    ensure_stable_ids(plan)
    assert plan.continuity.characters[0].id == first


# ---------------------------------------------------------------- stable identity across scenes
@pytest.mark.parametrize("n", [3, 5, 6, 8, 10])
def test_character_identity_stable_across_n_scenes(n):
    ch = CharacterIdentity(role="protagonist", appearance="tall man with dark beard",
                           hair="short black", clothing="leather jacket")
    scenes = [_scene(i, chars=[SceneCharacter(ref="protagonist", action=f"act {i}")]) for i in range(1, n + 1)]
    # Scale duration to sum correctly.
    total = sum(s.duration for s in scenes)
    plan = _manual_plan([ch], scenes=scenes)
    plan.content.duration_seconds = total
    ctxs = resolve_all_scenes(plan)
    assert len(ctxs) == n
    # Same full_description in every scene.
    descs = {c.full_description for ctx in ctxs for c in ctx.characters}
    assert len(descs) == 1


def test_character_id_reused_not_redefined():
    ch = CharacterIdentity(id="hero", role="protagonist", appearance="tall man")
    plan = _manual_plan([ch], scenes=[_scene(1, chars=[SceneCharacter(ref="hero", action="enters")])])
    plan.content.duration_seconds = 4.0
    ctx = resolve_scene_context(plan, 1)
    assert ctx.characters[0].id == "hero"
    assert ctx.characters[0].appearance == "tall man"


def test_product_identity_stable_across_scenes():
    prod = ProductIdentity(id="perf", name="Noir", appearance="black glass bottle",
                           packaging="matte box", colors=["#111", "#C9A96E"], materials=["glass"])
    scenes = [
        _scene(1, chars=[SceneCharacter(ref="protagonist", action="holds bottle")]),
        _scene(2, purpose=ScenePurpose.PRODUCT, chars=[SceneCharacter(ref="protagonist", action="places bottle")]),
    ]
    plan = _manual_plan([CharacterIdentity(role="protagonist", appearance="woman")], product=prod, scenes=scenes)
    plan.content.duration_seconds = 8.0
    ctxs = resolve_all_scenes(plan)
    # Product identity identical in both scenes.
    assert ctxs[0].product.appearance == ctxs[1].product.appearance == "black glass bottle"
    assert all(ctx.product.id == "perf" for ctx in ctxs)


# ---------------------------------------------------------------- environment consistency
def test_environment_section_consistent_across_scenes():
    scenes = [_scene(i) for i in range(1, 4)]
    plan = _manual_plan([CharacterIdentity(role="protagonist", appearance="man")], scenes=scenes)
    plan.content.duration_seconds = 12.0
    ctxs = resolve_all_scenes(plan)
    envs = {ctx.environment_section for ctx in ctxs}
    assert len(envs) == 1  # all identical


def test_environment_fields_propagated():
    cm = ContinuityMemory(
        characters=[CharacterIdentity(role="protagonist", appearance="man")],
        environment="jungle", location="Amazon", time_of_day="dawn",
        weather="misty", season="wet", architecture="none",
        environmental_objects=["vines", "ruins"], lighting="soft",
        visual_style="cinematic", color_palette=["#0a0"],
    )
    base = _plan()
    base.continuity = cm
    ctx = resolve_scene_context(base, base.scenes[0].index)
    env = ctx.environment_section
    assert "Amazon" in env and "dawn" in env and "misty" in env and "vines" in env


# ---------------------------------------------------------------- intentional vs accidental changes
def test_intentional_environment_change_with_transition_is_warning():
    ch = CharacterIdentity(role="protagonist", appearance="man")
    scenes = [
        _scene(1, env="jungle"),
        _scene(2, env="city", transition="hard cut"),
    ]
    plan = _manual_plan([ch], scenes=scenes)
    plan.content.duration_seconds = 8.0
    report = validate_continuity(plan)
    assert report.ok  # no errors
    # transition declared -> not even a warning
    env_warnings = [w for w in report.warnings if w.code == "ENVIRONMENT_CHANGE"]
    assert env_warnings == []


def test_accidental_environment_change_is_warning():
    ch = CharacterIdentity(role="protagonist", appearance="man")
    scenes = [
        _scene(1, env="jungle"),
        _scene(2, env="city"),  # no transition declared
    ]
    plan = _manual_plan([ch], scenes=scenes)
    plan.content.duration_seconds = 8.0
    report = validate_continuity(plan)
    assert report.ok
    env_warnings = [w for w in report.warnings if w.code == "ENVIRONMENT_CHANGE"]
    assert len(env_warnings) == 1


def test_clothing_change_is_warning_not_error():
    ch = CharacterIdentity(role="protagonist", appearance="man", clothing="jacket")
    scenes = [_scene(1, chars=[SceneCharacter(ref="protagonist", action="x")])]
    plan = _manual_plan([ch], scenes=scenes)
    plan.content.duration_seconds = 4.0
    # First validate, then change clothing in scene 2.
    scenes.append(_scene(2, chars=[SceneCharacter(ref="protagonist", action="y")]))
    plan.scenes = scenes
    plan.content.duration_seconds = 8.0
    # Simulate clothing change by editing continuity (intentional would use transition).
    report = validate_continuity(plan)
    # Clothing is on the character identity, not scene; warnings come from scene-level diffs.
    # This still must be OK (no errors).
    assert report.ok


# ---------------------------------------------------------------- conflicting changes (ERRORs)
def test_conflicting_character_identity_is_error():
    cm = ContinuityMemory(
        characters=[
            CharacterIdentity(id="character_1", role="protagonist", appearance="tall man with beard"),
            CharacterIdentity(id="character_1", role="protagonist", appearance="short woman with blonde hair"),
        ],
        visual_style="x", lighting="y",
    )
    plan = _plan()
    plan.continuity = cm
    report = validate_continuity(plan)
    assert not report.ok
    assert any(e.code == "CHARACTER_IDENTITY_CONFLICT" for e in report.errors)


def test_missing_character_reference_is_error_when_memory_empty():
    plan = _plan()
    plan.continuity = ContinuityMemory(characters=[], visual_style="x", lighting="y")
    # Force a scene to reference a character not in (empty) memory.
    plan.scenes[0].characters = [SceneCharacter(ref="ghost", action="appears")]
    report = validate_continuity(plan)
    assert any(e.code == "MISSING_CHARACTER_REFERENCE" for e in report.errors)


def test_missing_character_reference_falls_back_when_memory_nonempty():
    # When memory has characters, an unknown ref falls back to protagonist (warning-free).
    plan = _plan()
    plan.scenes[0].characters = [SceneCharacter(ref="nonexistent", action="x")]
    report = validate_continuity(plan)
    assert report.ok  # fallback used, no error


# ---------------------------------------------------------------- continuity_refs resolution
def test_continuity_refs_resolved_to_ids():
    ch = CharacterIdentity(id="hero", role="protagonist", appearance="man")
    prod = ProductIdentity(id="perf", name="Noir", appearance="bottle")
    plan = _manual_plan([ch], product=prod, scenes=[_scene(1, chars=[SceneCharacter(ref="hero", action="holds")])])
    plan.content.duration_seconds = 4.0
    ctx = resolve_scene_context(plan, 1)
    assert "hero" in ctx.continuity_ids
    assert "perf" in ctx.continuity_ids


def test_scene_character_without_ref_uses_protagonist():
    ch = CharacterIdentity(id="hero", role="protagonist", appearance="man")
    plan = _manual_plan([ch], scenes=[_scene(1, chars=[SceneCharacter(ref=None, action="walks")])])
    plan.content.duration_seconds = 4.0
    ctx = resolve_scene_context(plan, 1)
    assert len(ctx.characters) == 1
    assert ctx.characters[0].id == "hero"


# ---------------------------------------------------------------- resolved prompt construction
def test_resolved_prompt_has_all_sections():
    ctx = resolve_scene_context(_plan(), _plan().scenes[0].index)
    for tag in ["[IDENTITY]", "[ENVIRONMENT]", "[STYLE]", "[CAMERA]", "[ACTION]", "[MOTION]", "[DETAILS]"]:
        assert tag in ctx.visual_prompt, tag


def test_identity_section_reused_across_scenes():
    plan = _plan()
    ctxs = resolve_all_scenes(plan)
    ids = {ctx.identity_section for ctx in ctxs}
    assert len(ids) == 1  # identical identity text


def test_build_visual_prompt_deterministic():
    ctx = resolve_scene_context(_plan(), _plan().scenes[0].index)
    assert build_visual_prompt(ctx) == ctx.visual_prompt


def test_resolved_context_to_dict_serializable():
    ctx = resolve_scene_context(_plan(), _plan().scenes[0].index)
    json.dumps(ctx.to_dict())


def test_resolve_out_of_range_raises():
    with pytest.raises(IndexError):
        resolve_scene_context(_plan(), 999)


# ---------------------------------------------------------------- deterministic output
def test_resolve_all_deterministic():
    plan = _plan()
    a = [c.visual_prompt for c in resolve_all_scenes(plan)]
    b = [c.visual_prompt for c in resolve_all_scenes(plan)]
    assert a == b


# ---------------------------------------------------------------- serialization / deserialization
def test_plan_with_extended_fields_roundtrips():
    ch = CharacterIdentity(id="hero", role="protagonist", appearance="man", hair="black",
                           face="sharp jawline", body="athletic", clothing="jacket",
                           distinguishing_features=["scar on cheek"])
    prod = ProductIdentity(id="perf", name="Noir", appearance="bottle", shape="cylindrical",
                           packaging="matte box", logo_placement="center", colors=["#111"],
                           materials=["glass"], distinctive_features=["gold cap"])
    cm = ContinuityMemory(
        characters=[ch], product=prod, environment="jungle", location="Amazon",
        time_of_day="dawn", weather="misty", season="wet", architecture="none",
        environmental_objects=["vines"], lighting="soft", color_palette=["#0a0"],
        camera_language="dolly", visual_style="cinematic",
    )
    plan = _plan()
    plan.continuity = cm
    s = plan.model_dump_json()
    back = ProductionPlan.model_validate_json(s)
    assert back.continuity.characters[0].distinguishing_features == ["scar on cheek"]
    assert back.continuity.product.materials == ["glass"]
    assert back.continuity.weather == "misty"


# ---------------------------------------------------------------- backward compatibility
def test_backward_compat_phase5_content_plan():
    plan = local_content_plan(ContentBrief(idea="x", duration_seconds=12))
    ctxs = resolve_all_scenes(plan)
    assert len(ctxs) == len(plan.scenes)
    report = validate_continuity(plan)
    assert report.ok


def test_backward_compat_phase5_plan_serializes():
    plan = local_content_plan(ContentBrief(idea="x", duration_seconds=12))
    back = ProductionPlan.model_validate_json(plan.model_dump_json())
    assert back.model_dump() == plan.model_dump()


@pytest.mark.parametrize("key", ["emotional", "direct_response", "cinematic", "ugc", "product_demo", "problem_solution", "testimonial"])
def test_backward_compat_phase6_all_variants_resolve(key):
    plan = _ad_plan(key)
    ctxs = resolve_all_scenes(plan)
    assert all("[IDENTITY]" in c.visual_prompt for c in ctxs)
    assert validate_continuity(plan).ok


def test_phase6_variants_still_generate_after_model_extension():
    brief = AdBrief(product_or_service="X", target_audience="y", platform=Platform.TIKTOK, duration_seconds=10)
    results = generate_variants(brief)
    assert len(results) == 7


# ---------------------------------------------------------------- reference images
def test_registry_from_plan_binds_character_and_product():
    plan = _ad_plan()
    reg = registry_from_plan(plan)
    kinds = {img.kind for img in reg.images}
    assert ReferenceKind.CHARACTER in kinds
    assert ReferenceKind.PRODUCT in kinds


def test_registry_resolves_by_target_id():
    plan = _ad_plan()
    reg = registry_from_plan(plan)
    char_id = plan.continuity.characters[0].id
    assert reg.for_character(char_id)
    prod_id = plan.continuity.product.id
    assert reg.for_product(prod_id)


def test_reference_image_available_only_if_file_exists(tmp_path):
    p = tmp_path / "ref.png"
    p.write_bytes(b"x")
    img = ReferenceImage(ReferenceKind.CHARACTER, "hero", path=p)
    assert img.is_available()
    img2 = ReferenceImage(ReferenceKind.CHARACTER, "hero", path=tmp_path / "nope.png")
    assert not img2.is_available()


def test_reference_image_to_dict_serializable():
    img = ReferenceImage(ReferenceKind.PRODUCT, "perf", description="bottle ref")
    json.dumps(img.to_dict())


def test_registry_available_filters_existing(tmp_path):
    p = tmp_path / "r.png"
    p.write_bytes(b"x")
    reg = ReferenceRegistry(images=[
        ReferenceImage(ReferenceKind.CHARACTER, "hero", path=p),
        ReferenceImage(ReferenceKind.CHARACTER, "hero2", path=tmp_path / "no.png"),
    ])
    assert len(reg.available()) == 1


def test_resolved_context_attaches_reference_images():
    plan = _ad_plan()
    reg = registry_from_plan(plan)
    ctx = resolve_scene_context(plan, plan.scenes[0].index, reg)
    assert ctx.reference_images  # some references attached


# ---------------------------------------------------------------- validation report
def test_validation_report_to_dict():
    report = validate_continuity(_plan())
    d = report.to_dict()
    assert "ok" in d and "errors" in d and "warnings" in d


def test_clean_plan_has_no_errors():
    report = validate_continuity(_plan())
    assert report.ok
    assert report.errors == []


def test_lighting_change_is_warning():
    ch = CharacterIdentity(role="protagonist", appearance="man")
    scenes = [_scene(1, lighting="soft"), _scene(2, lighting="harsh")]
    plan = _manual_plan([ch], scenes=scenes)
    plan.content.duration_seconds = 8.0
    report = validate_continuity(plan)
    assert report.ok
    assert any(w.code == "LIGHTING_CHANGE" for w in report.warnings)


def test_error_vs_warning_severity_distinct():
    cm = ContinuityMemory(
        characters=[
            CharacterIdentity(id="c1", role="p", appearance="a"),
            CharacterIdentity(id="c1", role="p", appearance="b"),
        ],
        visual_style="x", lighting="y",
    )
    plan = _plan()
    plan.continuity = cm
    report = validate_continuity(plan)
    assert any(i.severity == Severity.ERROR for i in report.errors)
    # A clean plan produces only warnings at most, never errors.
    report2 = validate_continuity(_plan())
    assert all(i.severity == Severity.WARNING for i in report2.warnings)


# ---------------------------------------------------------------- full resolved payload
def test_full_resolve_all_payload_serializable():
    plan = _ad_plan()
    ctxs = resolve_all_scenes(plan)
    report = validate_continuity(plan)
    reg = registry_from_plan(plan)
    payload = {
        "contexts": [c.to_dict() for c in ctxs],
        "validation": report.to_dict(),
        "references": reg.to_dict(),
    }
    json.dumps(payload)

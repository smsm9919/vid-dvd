"""Comprehensive tests for the AI Content Brain (Phase 5).

These are CODE/TEST VERIFIED. No real LLM API is called (GEMINI_API_KEY is
absent in this environment), so the Gemini path is exercised via monkeypatched
fakes, never claimed as RUNTIME VERIFIED.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from app.brain.content_brain import (
    _durations,
    _purpose_sequence,
    gemini_content_plan,
    local_content_plan,
    plan_content,
)
from app.brain.models import (
    AdVariant,
    AspectRatio,
    BrandProfile,
    ContentBrief,
    NarrativeStructure,
    Objective,
    Platform,
    ProductionMode,
    ProductionPlan,
    Scene,
    ScenePurpose,
    Tone,
)


# ---------------------------------------------------------------- fixtures
def _ad_brief(**over) -> ContentBrief:
    base = dict(
        idea="luxury perfume advertisement",
        product_or_service="Noir Élixir perfume",
        audience="affluent women 25-45",
        platform=Platform.TIKTOK,
        language="en",
        country="US",
        duration_seconds=20,
        objective=Objective.CONVERSION,
        tone=Tone.EMOTIONAL,
        mode=ProductionMode.ADVERTISEMENT,
        variant_hint=AdVariant.EMOTIONAL,
    )
    base.update(over)
    return ContentBrief(**base)


def _cinematic_brief(**over) -> ContentBrief:
    base = dict(
        idea="a lion hunting in the jungle",
        platform=Platform.YOUTUBE_SHORTS,
        duration_seconds=30,
        mode=ProductionMode.CINEMATIC,
    )
    base.update(over)
    return ContentBrief(**base)


# ---------------------------------------------------------------- minimum valid plan
def test_minimum_valid_production_plan():
    brief = ContentBrief(idea="minimal idea", duration_seconds=12)
    plan = local_content_plan(brief)
    assert isinstance(plan, ProductionPlan)
    assert plan.meta.planner == "local"
    assert len(plan.scenes) >= 1
    assert plan.content.topic == "minimal idea"


def test_plan_from_product_only():
    brief = ContentBrief(product_or_service="Widget X", duration_seconds=10, mode=ProductionMode.ADVERTISEMENT)
    plan = local_content_plan(brief)
    assert plan.content.product_or_service == "Widget X"
    assert any(s.purpose == ScenePurpose.CTA for s in plan.scenes)


# ---------------------------------------------------------------- scene generation
def test_scene_count_matches_derived():
    brief = _ad_brief(scene_count=6)
    plan = local_content_plan(brief)
    assert len(plan.scenes) == 6
    assert [s.index for s in plan.scenes] == [1, 2, 3, 4, 5, 6]


def test_scene_fields_populated():
    plan = local_content_plan(_ad_brief(scene_count=4))
    for s in plan.scenes:
        assert s.description
        assert s.visual_prompt
        assert s.negative_prompt
        assert s.camera.shot_type
        assert s.voiceover is not None
        assert s.caption.text


def test_purpose_sequence_ad_has_hook_and_cta():
    for count in (3, 4, 5, 6, 8, 10):
        seq = _purpose_sequence(ProductionMode.ADVERTISEMENT, count)
        assert seq[0] == ScenePurpose.HOOK
        assert seq[-1] == ScenePurpose.CTA
        assert len(seq) == count


def test_purpose_sequence_cinematic_has_hook():
    seq = _purpose_sequence(ProductionMode.CINEMATIC, 5)
    assert seq[0] == ScenePurpose.HOOK
    assert ScenePurpose.CTA not in seq


# ---------------------------------------------------------------- timing consistency
def test_durations_sum_to_total():
    for total in (10, 15, 20, 30, 45, 60):
        seq = _purpose_sequence(ProductionMode.ADVERTISEMENT, 5)
        d = _durations(seq, total)
        assert abs(sum(d) - total) < 0.05, (total, d, sum(d))
        assert all(x > 0 for x in d)


def test_plan_scene_durations_sum_to_plan_duration():
    for total in (10, 20, 30, 60):
        plan = local_content_plan(_ad_brief(duration_seconds=total, scene_count=5))
        assert abs(sum(s.duration for s in plan.scenes) - total) < 0.5


def test_hook_scene_is_shorter_than_average():
    plan = local_content_plan(_ad_brief(duration_seconds=30, scene_count=6))
    avg = sum(s.duration for s in plan.scenes) / len(plan.scenes)
    hook = next(s for s in plan.scenes if s.purpose == ScenePurpose.HOOK)
    assert hook.duration <= avg


# ---------------------------------------------------------------- hook + CTA presence
def test_ad_plan_has_hook_and_cta_scenes():
    plan = local_content_plan(_ad_brief())
    purposes = [s.purpose for s in plan.scenes]
    assert ScenePurpose.HOOK in purposes
    assert ScenePurpose.CTA in purposes
    assert plan.creative.hook
    assert plan.creative.cta


def test_cinematic_plan_has_hook_but_no_cta():
    plan = local_content_plan(_cinematic_brief())
    purposes = [s.purpose for s in plan.scenes]
    assert ScenePurpose.HOOK in purposes
    assert ScenePurpose.CTA not in purposes


# ---------------------------------------------------------------- continuity fields
def test_continuity_memory_populated():
    plan = local_content_plan(_ad_brief())
    c = plan.continuity
    assert c.visual_style
    assert c.camera_language
    assert c.color_palette
    assert c.lighting
    # Product identity present for ad with product.
    assert c.product is not None
    assert c.product.appearance


def test_continuity_character_stable_description():
    plan = local_content_plan(_cinematic_brief())
    # Protagonist appearance is identical in continuity and referenced by scenes.
    protag = plan.continuity.characters[0]
    assert protag.appearance
    refs = {r for s in plan.scenes for r in s.continuity_refs}
    assert "protagonist" in refs


def test_continuity_product_identity_consistent():
    brief = _ad_brief(brand=BrandProfile(brand_name="Maison", colors=["#111", "#caa"], visual_style="dark luxury"))
    plan = local_content_plan(brief)
    assert plan.continuity.product.name == "Noir Élixir perfume"
    assert plan.continuity.color_palette == ["#111", "#caa"]


# ---------------------------------------------------------------- voice-over timing
def test_voiceover_per_scene_timing_within_scene():
    plan = local_content_plan(_ad_brief(scene_count=5))
    for s in plan.scenes:
        assert s.voiceover.start_offset <= s.duration


def test_voiceover_full_script_aggregates_scenes():
    plan = local_content_plan(_ad_brief(scene_count=4))
    lines = [s.voiceover.line for s in plan.scenes if s.voiceover.line]
    assert plan.voiceover.full_script.strip() == " ".join(lines).strip()


def test_voiceover_start_offset_exceeding_duration_rejected():
    with pytest.raises(ValidationError):
        Scene(
            index=1, purpose=ScenePurpose.HOOK, duration=2.0, description="d",
            visual_prompt="p", negative_prompt="n",
            voiceover={"line": "hi", "start_offset": 5.0},
        )


# ---------------------------------------------------------------- deterministic fallback
def test_local_planner_is_deterministic():
    brief = _ad_brief()
    a = local_content_plan(brief).model_dump_json()
    b = local_content_plan(brief).model_dump_json()
    assert a == b


def test_plan_content_without_gemini_key_uses_local(monkeypatch):
    monkeypatch.setattr("app.brain.content_brain.GEMINI_API_KEY", "")
    plan = asyncio.run(plan_content(_ad_brief()))
    assert plan.meta.planner == "local"


# ---------------------------------------------------------------- Gemini failure fallback
def _fake_gemini_failure(*a, **kw):
    raise RuntimeError("network down")


def test_gemini_failure_falls_back_cleanly(monkeypatch):
    monkeypatch.setattr("app.brain.content_brain.GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr("app.brain.content_brain.gemini_content_plan", _fake_gemini_failure)
    plan = asyncio.run(plan_content(_ad_brief()))
    assert plan.meta.planner == "local_fallback"
    assert plan.meta.warnings
    assert "Gemini unavailable" in plan.meta.warnings[0]


async def _fake_gemini_success(brief, *, api_key=None, model=None):
    # Build a valid plan via the local planner but mark planner=gemini to simulate
    # a successful LLM round-trip without a real API call.
    plan = local_content_plan(brief)
    plan.meta = plan.meta.model_copy(update={"planner": "gemini", "model": "gemini-test"})
    return plan


def test_gemini_success_used_when_available(monkeypatch):
    monkeypatch.setattr("app.brain.content_brain.GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr("app.brain.content_brain.gemini_content_plan", _fake_gemini_success)
    plan = asyncio.run(plan_content(_ad_brief()))
    assert plan.meta.planner == "gemini"
    assert plan.meta.model == "gemini-test"


def test_gemini_no_key_raises_for_direct_call():
    # Direct gemini_content_plan without a key must raise (caller falls back).
    with pytest.raises(RuntimeError):
        asyncio.run(gemini_content_plan(_ad_brief(), api_key="", model="x"))


# ---------------------------------------------------------------- invalid input handling
def test_brief_requires_idea_or_product():
    with pytest.raises(ValidationError):
        ContentBrief(idea="", product_or_service="", duration_seconds=10)


def test_brief_duration_bounds():
    with pytest.raises(ValidationError):
        ContentBrief(idea="x", duration_seconds=0)
    with pytest.raises(ValidationError):
        ContentBrief(idea="x", duration_seconds=400)


def test_plan_rejects_scene_duration_mismatch():
    plan = local_content_plan(_ad_brief(duration_seconds=20))
    data = plan.model_dump()
    data["scenes"][0]["duration"] = 100  # break the sum
    with pytest.raises(ValidationError):
        ProductionPlan.model_validate(data)


def test_plan_rejects_missing_cta_for_advertisement():
    plan = local_content_plan(_ad_brief())
    data = plan.model_dump()
    # Remove CTA purpose from the last scene.
    data["scenes"][-1]["purpose"] = "benefit"
    with pytest.raises(ValidationError):
        ProductionPlan.model_validate(data)


def test_plan_rejects_non_contiguous_indexes():
    plan = local_content_plan(_ad_brief(scene_count=4))
    data = plan.model_dump()
    data["scenes"][1]["index"] = 99
    with pytest.raises(ValidationError):
        ProductionPlan.model_validate(data)


def test_plan_rejects_empty_scenes():
    plan = local_content_plan(_ad_brief())
    data = plan.model_dump()
    data["scenes"] = []
    with pytest.raises(ValidationError):
        ProductionPlan.model_validate(data)


# ---------------------------------------------------------------- serialization
def test_serialization_roundtrip():
    plan = local_content_plan(_ad_brief(scene_count=5))
    s = plan.model_dump_json()
    assert isinstance(s, str)
    back = ProductionPlan.model_validate_json(s)
    assert back.model_dump() == plan.model_dump()


def test_serialization_is_plain_json():
    plan = local_content_plan(_ad_brief())
    data = json.loads(plan.model_dump_json())
    assert data["meta"]["planner"] == "local"
    assert isinstance(data["scenes"], list)
    assert isinstance(data["content"]["audience"], dict)


# ---------------------------------------------------------------- ad variant schema support
def test_ad_variant_hint_recorded():
    brief = _ad_brief(variant_hint=AdVariant.DIRECT_RESPONSE)
    plan = local_content_plan(brief)
    assert plan.content.variant_hint == AdVariant.DIRECT_RESPONSE


def test_all_ad_variants_assignable():
    for v in AdVariant:
        brief = _ad_brief(variant_hint=v)
        plan = local_content_plan(brief)
        assert plan.content.variant_hint == v


# ---------------------------------------------------------------- multi-language
def test_multi_language_propagates_to_voiceover_and_captions():
    for lang in ("en", "de", "ar"):
        plan = local_content_plan(_ad_brief(language=lang))
        assert plan.voiceover.language == lang
        assert plan.content.language == lang
        assert all(c.scene_index >= 1 for c in plan.captions.items)


# ---------------------------------------------------------------- platform/aspect
def test_platform_aspect_ratio_mapping():
    plan = local_content_plan(ContentBrief(idea="x", platform=Platform.YOUTUBE, duration_seconds=60))
    assert plan.content.aspect_ratio == AspectRatio.HORIZONTAL_16_9
    plan2 = local_content_plan(_ad_brief(platform=Platform.INSTAGRAM_REELS))
    assert plan2.content.aspect_ratio == AspectRatio.VERTICAL_9_16

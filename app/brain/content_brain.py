"""AI Content Brain — converts a brief into a :class:`ProductionPlan`.

Two planners, same contract:

* :func:`local_content_plan` — deterministic, no network, no API key. Always
  works. Produces a complete, valid plan from the brief alone.
* :func:`gemini_content_plan` — optional, uses the Gemini API when
  ``GEMINI_API_KEY`` is configured. On ANY failure it raises so the caller can
  fall back to the local planner; the application never fails wholesale.

The top-level entry point :func:`plan_content` picks Gemini when available and
falls back cleanly, recording the planner source in ``PlanMetadata``.

This module produces PLANS ONLY. It never generates video/audio/captions and
never touches the ComfyUI provider or FFmpeg.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from ..config import GEMINI_API_KEY, GEMINI_MODEL
from ..core.logging import log
from .models import (
    AdVariant,
    AspectRatio,
    AudioPlan,
    Audience,
    BrandProfile,
    CaptionItem,
    CaptionPlan,
    CharacterIdentity,
    ContentBrief,
    ContentPlan,
    ContinuityMemory,
    CreativeDecision,
    NarrativeStructure,
    Objective,
    PlanMetadata,
    ProductionMode,
    ProductionPlan,
    ProductIdentity,
    Scene,
    SceneCamera,
    SceneCaption,
    SceneCharacter,
    ScenePurpose,
    SceneVoiceover,
    Tone,
    VoiceOverPlan,
)

# Shared style vocabulary reused by deterministic prompt construction.
_NEG = (
    "blurry, distorted anatomy, extra limbs, duplicate subjects, deformed face, "
    "text, logo, watermark, low quality, jitter, flicker, inconsistent identity"
)

PLATFORM_DEFAULTS = {
    "tiktok": (AspectRatio.VERTICAL_9_16, 15.0),
    "instagram_reels": (AspectRatio.VERTICAL_9_16, 15.0),
    "youtube_shorts": (AspectRatio.VERTICAL_9_16, 30.0),
    "youtube": (AspectRatio.HORIZONTAL_16_9, 60.0),
    "other": (AspectRatio.VERTICAL_9_16, 30.0),
}


def _resolve_duration(brief: ContentBrief) -> float:
    if brief.duration_seconds:
        return brief.duration_seconds
    return PLATFORM_DEFAULTS[brief.platform.value][1]


def _aspect_for(platform) -> AspectRatio:
    return PLATFORM_DEFAULTS[platform.value][0]


def _derive_scene_count(brief: ContentBrief, duration: float) -> int:
    if brief.scene_count:
        return brief.scene_count
    # ~1 scene per 4-6s, clamped to [3, 10].
    return max(3, min(10, round(duration / 5)))


def _visual_style(brief: ContentBrief) -> str:
    if brief.visual_style:
        return brief.visual_style
    if brief.brand and brief.brand.visual_style:
        return brief.brand.visual_style
    return (
        "ultra-realistic cinematic, dramatic lighting, volumetric atmosphere, "
        "shallow depth of field, high dynamic range, cinematic color grading"
    )


def _audience(brief: ContentBrief) -> Audience:
    return Audience(
        description=brief.audience or "general audience interested in the topic",
        market=brief.country,
    )


def _core_message(brief: ContentBrief) -> str:
    subject = brief.product_or_service or brief.idea
    if brief.mode == ProductionMode.ADVERTISEMENT:
        return f"{subject} solves a real need for the target audience."
    return f"A cinematic exploration of {subject}."


def _narrative_for(mode: ProductionMode) -> NarrativeStructure:
    if mode == ProductionMode.ADVERTISEMENT:
        return NarrativeStructure.HOOK_PROBLEM_SOLUTION_BENEFIT_PROOF_CTA
    if mode == ProductionMode.DOCUMENTARY:
        return NarrativeStructure.DOCUMENTARY
    if mode == ProductionMode.UGC:
        return NarrativeStructure.BEFORE_AFTER
    return NarrativeStructure.THREE_ACT


def _purpose_sequence(mode: ProductionMode, count: int) -> list[ScenePurpose]:
    """Distribute scene purposes deterministically by mode and count.

    Guarantees: hook is always first (for non-documentary), and CTA is always
    last for advertisement/promotional modes, regardless of count.
    """
    needs_cta = mode in (ProductionMode.ADVERTISEMENT, ProductionMode.PROMOTIONAL)
    # Reserve first (hook) and last (cta) slots, then fill the middle.
    middle_slots = count - 1 - (1 if needs_cta else 0)

    if mode == ProductionMode.ADVERTISEMENT:
        middle_pool = [ScenePurpose.PROBLEM, ScenePurpose.SOLUTION, ScenePurpose.BENEFIT, ScenePurpose.PROOF, ScenePurpose.PRODUCT]
    elif mode == ProductionMode.DOCUMENTARY:
        middle_pool = [ScenePurpose.PRODUCT, ScenePurpose.EMOTION, ScenePurpose.ESTABLISH]
    elif mode == ProductionMode.UGC:
        middle_pool = [ScenePurpose.PROBLEM, ScenePurpose.PRODUCT, ScenePurpose.PROOF, ScenePurpose.BENEFIT]
    else:
        middle_pool = [ScenePurpose.ESTABLISH, ScenePurpose.EMOTION, ScenePurpose.PRODUCT]

    middle = []
    i = 0
    while len(middle) < middle_slots and middle_pool:
        middle.append(middle_pool[i % len(middle_pool)])
        i += 1
    # Fallback if middle_slots somehow exceeds pool variety (rare); keep valid purposes.
    while len(middle) < middle_slots:
        middle.append(ScenePurpose.EMOTION if mode != ProductionMode.ADVERTISEMENT else ScenePurpose.BENEFIT)

    if mode == ProductionMode.DOCUMENTARY:
        seq = [ScenePurpose.ESTABLISH] + middle + ([ScenePurpose.CLOSING] if not needs_cta else [ScenePurpose.CTA])
    else:
        seq = [ScenePurpose.HOOK] + middle + ([ScenePurpose.CTA] if needs_cta else [ScenePurpose.CLOSING])
    return seq[:count]


def _durations(purposes: list[ScenePurpose], total: float) -> list[float]:
    """Allocate per-scene durations that sum to ``total``.

    Hook and CTA get shorter, punchier durations; mid scenes share the rest.
    """
    n = len(purposes)
    if n == 1:
        return [round(total, 2)]
    weights = []
    for p in purposes:
        if p == ScenePurpose.HOOK:
            weights.append(0.6)
        elif p == ScenePurpose.CTA:
            weights.append(0.7)
        elif p in (ScenePurpose.TRANSITION,):
            weights.append(0.4)
        else:
            weights.append(1.0)
    raw_total = sum(weights)
    durations = [round(total * w / raw_total, 2) for w in weights]
    # Correct rounding drift onto the largest mid scene.
    drift = round(total - sum(durations), 2)
    if abs(drift) > 0:
        mid = max(range(n), key=lambda i: durations[i])
        durations[mid] = round(durations[mid] + drift, 2)
    return durations


def _hook_text(brief: ContentBrief) -> str:
    subject = brief.product_or_service or brief.idea
    if brief.mode == ProductionMode.ADVERTISEMENT:
        return f"Stop scrolling — this changes how you think about {subject}."
    return f"A breathtaking opening image that pulls you into the world of {subject}."


def _cta_text(brief: ContentBrief) -> str:
    if brief.brand and brief.brand.cta:
        return brief.brand.cta
    if brief.mode == ProductionMode.ADVERTISEMENT:
        return "Try it today — link in bio."
    return "Watch till the end."


def _build_continuity(brief: ContentBrief, style: str) -> ContinuityMemory:
    product = None
    if brief.product_or_service:
        product = ProductIdentity(
            name=brief.product_or_service,
            appearance=f"the {brief.product_or_service}, shown consistently with stable packaging and signature colors",
            signature_colors=brief.brand.colors if brief.brand else [],
        )
    return ContinuityMemory(
        characters=[
            CharacterIdentity(
                role="protagonist",
                appearance="consistent human subject, same facial features and build across all scenes",
                clothing="stable wardrobe across scenes unless the narrative explicitly changes it",
            )
        ] if brief.mode != ProductionMode.DOCUMENTARY else [],
        product=product,
        environment=f"coherent environment for {brief.idea or brief.product_or_service}",
        lighting="consistent motivated lighting matching the time of day",
        color_palette=brief.brand.colors if brief.brand else ["teal", "orange", "deep shadow"],
        camera_language="consistent cinematic camera language",
        visual_style=style,
    )


def _scene_description(purpose: ScenePurpose, brief: ContentBrief, idx: int, total: int) -> str:
    subject = brief.product_or_service or brief.idea
    m = {
        ScenePurpose.HOOK: f"Powerful hook: a striking, curiosity-driving image introducing {subject}.",
        ScenePurpose.ESTABLISH: f"Establish the world and atmosphere around {subject}.",
        ScenePurpose.PROBLEM: f"Show the frustration or problem the audience faces before {subject}.",
        ScenePurpose.SOLUTION: f"Introduce {subject} as the solution, clearly and attractively.",
        ScenePurpose.BENEFIT: f"Show a concrete benefit of {subject} in action.",
        ScenePurpose.PROOF: f"Credibility/proof moment for {subject} (demonstration or result).",
        ScenePurpose.PRODUCT: f"Hero product shot of {subject} with stable identity.",
        ScenePurpose.EMOTION: f"Emotional beat that connects the audience to {subject}.",
        ScenePurpose.TRANSITION: "Quick transition maintaining visual continuity.",
        ScenePurpose.CTA: f"Call to action: direct the viewer toward the next step for {subject}.",
        ScenePurpose.CLOSING: f"Memorable closing image tied to {subject}.",
    }
    return m.get(purpose, f"Scene {idx}/{total}: continue the story about {subject}.")


def _camera_for(purpose: ScenePurpose) -> SceneCamera:
    table = {
        ScenePurpose.HOOK: SceneCamera(shot_type="close-up", lens="35mm", framing="centered", movement="slow dolly in"),
        ScenePurpose.PRODUCT: SceneCamera(shot_type="macro", lens="50mm", framing="product hero", movement="slow orbit"),
        ScenePurpose.CTA: SceneCamera(shot_type="medium", lens="35mm", framing="subject + text space", movement="static"),
        ScenePurpose.ESTABLISH: SceneCamera(shot_type="wide", lens="24mm", framing="environment", movement="slow pan"),
    }
    return table.get(purpose, SceneCamera(shot_type="medium", lens="35mm", framing="rule of thirds", movement="handheld"))


def _vo_line(purpose: ScenePurpose, brief: ContentBrief) -> str:
    subject = brief.product_or_service or brief.idea
    if brief.mode != ProductionMode.ADVERTISEMENT:
        return {
            ScenePurpose.HOOK: f"In the world of {subject}, everything is about to change.",
            ScenePurpose.CLOSING: f"This is {subject}, like you've never seen it.",
        }.get(purpose, f"Continuing the story of {subject}.")
    return {
        ScenePurpose.HOOK: _hook_text(brief),
        ScenePurpose.PROBLEM: "You know the feeling — nothing seems to work.",
        ScenePurpose.SOLUTION: f"Meet {subject}. Built to change that.",
        ScenePurpose.BENEFIT: "Real results, less effort.",
        ScenePurpose.PROOF: "Trusted by people who demand more.",
        ScenePurpose.CTA: _cta_text(brief),
    }.get(purpose, "")


def _build_scene(
    idx: int,
    purpose: ScenePurpose,
    duration: float,
    brief: ContentBrief,
    style: str,
    total: int,
) -> Scene:
    subject = brief.product_or_service or brief.idea
    desc = _scene_description(purpose, brief, idx, total)
    cam = _camera_for(purpose)
    visual = (
        f"{desc} Subject: {subject}. Camera: {cam.shot_type} {cam.lens or ''} {cam.movement or ''}. "
        f"Environment and lighting consistent with continuity. {style}. "
        f"Coherent subject identity, realistic motion, no text on screen."
    )
    vo_line = _vo_line(purpose, brief)
    return Scene(
        index=idx,
        purpose=purpose,
        duration=duration,
        description=desc,
        camera=cam,
        environment=f"consistent environment for {subject}",
        lighting="consistent motivated lighting",
        characters=[SceneCharacter(ref="protagonist", action=desc)] if brief.mode != ProductionMode.DOCUMENTARY else [],
        visual_prompt=visual,
        negative_prompt=_NEG,
        voiceover=SceneVoiceover(
            line=vo_line,
            direction=f"{brief.tone.value} tone, clear pacing",
            start_offset=0.0,
        ),
        sfx=[] if purpose != ScenePurpose.HOOK else ["subtle whoosh"],
        music="building cinematic bed" if purpose in (ScenePurpose.HOOK, ScenePurpose.EMOTION) else "ambient underscore",
        caption=SceneCaption(text=vo_line or desc, timing_intent="first_half"),
        transition_intent="hard cut" if purpose != ScenePurpose.TRANSITION else "quick wipe",
        continuity_refs=["protagonist"] + ([subject] if brief.product_or_service else []),
    )


def local_content_plan(brief: ContentBrief) -> ProductionPlan:
    """Deterministic local Content Brain. No network, no API key, always works."""
    duration = _resolve_duration(brief)
    count = _derive_scene_count(brief, duration)
    purposes = _purpose_sequence(brief.mode, count)
    durations = _durations(purposes, duration)
    style = _visual_style(brief)

    scenes = [
        _build_scene(i + 1, p, durations[i], brief, style, count)
        for i, p in enumerate(purposes)
    ]

    content = ContentPlan(
        topic=brief.idea or brief.product_or_service or "",
        product_or_service=brief.product_or_service,
        audience=_audience(brief),
        platform=brief.platform,
        objective=brief.objective,
        core_message=_core_message(brief),
        language=brief.language,
        country=brief.country,
        duration_seconds=duration,
        tone=brief.tone,
        visual_style=style,
        mode=brief.mode,
        aspect_ratio=_aspect_for(brief.platform),
        variant_hint=brief.variant_hint,
    )
    creative = CreativeDecision(
        hook=_hook_text(brief),
        problem="The audience faces a real, relatable frustration." if brief.mode == ProductionMode.ADVERTISEMENT else None,
        solution=brief.product_or_service if brief.mode == ProductionMode.ADVERTISEMENT else None,
        benefits=["saves time", "feels effortless", "looks premium"] if brief.mode == ProductionMode.ADVERTISEMENT else [],
        proof="visible demonstration of results" if brief.mode == ProductionMode.ADVERTISEMENT else None,
        cta=_cta_text(brief),
        narrative_structure=_narrative_for(brief.mode),
    )
    continuity = _build_continuity(brief, style)
    voiceover = VoiceOverPlan(
        full_script=" ".join(s.voiceover.line for s in scenes if s.voiceover.line),
        direction=f"{brief.tone.value} tone, {brief.language}",
        language=brief.language,
        voice_gender="female" if brief.tone in (Tone.EMOTIONAL, Tone.INSPIRATIONAL) else "male",
        pace="measured",
    )
    audio = AudioPlan(
        music_direction=f"{brief.tone.value} cinematic score matching the narrative arc",
        sfx_direction="subtle, motivated sound effects; punch on hook and CTA",
        ambience="environmental ambience consistent with the setting",
    )
    captions = CaptionPlan(
        items=[CaptionItem(scene_index=s.index, text=s.caption.text, timing_intent=s.caption.timing_intent) for s in scenes],
        style_preset=brief.platform.value,
        format_intent="burned_in",
    )

    return ProductionPlan(
        title=brief.idea or brief.product_or_service or "Untitled Production",
        content=content,
        creative=creative,
        continuity=continuity,
        scenes=scenes,
        voiceover=voiceover,
        audio=audio,
        captions=captions,
        brand=brief.brand,
        meta=PlanMetadata(planner="local", warnings=[]),
    )


# ---------------------------------------------------------------- Gemini

_GEMINI_INSTRUCTION = """You are a production-grade AI Content Brain for a video & advertising factory.
Convert the user brief into a STRICT JSON object matching this schema (no markdown, no commentary):
{{
  "title": string,
  "content": {{"topic","product_or_service","audience":{{"description","age_range?","interests","market?"}},"platform","objective","core_message","language","country?","duration_seconds","tone","visual_style","mode","aspect_ratio","variant_hint?"}},
  "creative": {{"hook","problem?","solution?","benefits":[],"proof?","cta","narrative_structure"}},
  "continuity": {{"characters":[{{"name?","role?","appearance","clothing?","physical_attributes":[]}}],"product?":{{"name?","appearance","packaging?","signature_colors":[]}},"environment","time_of_day?","lighting","color_palette":[],"camera_language","visual_style"}},
  "scenes": [{{"index","purpose","duration","description","camera":{{"shot_type?","lens?","framing?","movement?"}},"environment","lighting","characters":[{{"ref?","action"}}],"visual_prompt","negative_prompt","voiceover":{{"line","direction?","start_offset"}},"sfx":[],"music?","caption":{{"text","timing_intent?"}},"transition_intent?","continuity_refs":[]}}],
  "voiceover": {{"full_script","direction","language","voice_gender?","pace?"}},
  "audio": {{"music_direction","sfx_direction","ambience?"}},
  "captions": {{"items":[{{"scene_index","text","timing_intent?"}}],"style_preset?","format_intent?"}}
}}
Rules:
- Scene durations MUST sum to content.duration_seconds within 0.5s.
- Scenes MUST be contiguous from index 1.
- Include at least one scene with purpose "hook".
- If mode is "advertisement" or "promotional", include a scene with purpose "cta".
- voiceover.start_offset must be <= scene.duration.
- Use the provided enum values for platform, objective, tone, mode, aspect_ratio, variant_hint, narrative_structure, purpose.
"""


def _gemini_payload(brief: ContentBrief) -> dict:
    return {
        "contents": [{"parts": [{"text": _GEMINI_INSTRUCTION + "\nBRIEF:\n" + brief.model_dump_json(indent=2)}]}],
        "generationConfig": {"response_mime_type": "application/json"},
    }


async def gemini_content_plan(brief: ContentBrief, *, api_key: Optional[str] = None, model: Optional[str] = None) -> ProductionPlan:
    """Optional Gemini-backed Content Brain.

    Raises on any failure (network, parse, validation) so the caller can fall
    back to :func:`local_content_plan`. Never partially succeeds.
    """
    import httpx

    key = api_key or GEMINI_API_KEY
    mdl = model or GEMINI_MODEL
    if not key:
        raise RuntimeError("No GEMINI_API_KEY configured.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{mdl}:generateContent"
    log("BRAIN", "gemini request", model=mdl)
    async with httpx.AsyncClient(timeout=90) as c:
        r = await c.post(url, params={"key": key}, json=_gemini_payload(brief))
        r.raise_for_status()
        data = r.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Gemini returned no parseable content: {e}") from e
    text = re.sub(r"^```json\s*|\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Gemini returned non-JSON: {e}") from e
    plan = ProductionPlan.model_validate(payload)
    plan.meta = PlanMetadata(planner="gemini", model=mdl)
    return plan


async def plan_content(brief: ContentBrief) -> ProductionPlan:
    """Top-level entry: use Gemini if configured, else local. Never fails.

    Records the planner source. On Gemini failure, falls back cleanly to the
    deterministic local planner with a recorded warning.
    """
    if GEMINI_API_KEY:
        try:
            return await gemini_content_plan(brief)
        except Exception as e:  # noqa: BLE001 - fallback is the whole point
            log("BRAIN", f"gemini failed, falling back to local: {e}")
            plan = local_content_plan(brief)
            plan.meta = PlanMetadata(planner="local_fallback", warnings=[f"Gemini unavailable: {e}"])
            return plan
    return local_content_plan(brief)

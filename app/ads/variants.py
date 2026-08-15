"""Creative variant engine (Phase 6).

Generates seven genuinely different advertising strategies from one
:class:`AdBrief`. Each variant differs in hook, narrative structure, pacing,
scene purposes, visual strategy, voice-over strategy, CTA, emotional angle,
camera language, and editing intent — not just word substitutions.

Every variant produces a :class:`~app.brain.models.ProductionPlan` so it is
fully compatible with the downstream pipeline. Claim safety is enforced via
:mod:`app.ads.claims`: proof is only ever drawn from approved facts; otherwise
the proof scene is marked ``requires_verification`` and a warning is recorded.
No claims are fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..brain.models import (
    AdVariant,
    AspectRatio,
    AudioPlan,
    Audience,
    BrandProfile,
    CaptionItem,
    CaptionPlan,
    CharacterIdentity,
    ContentPlan,
    ContinuityMemory,
    CreativeDecision,
    NarrativeStructure,
    Objective,
    PlanMetadata,
    Platform,
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
from ..brain.content_brain import _durations
from .brief import AdBrief
from .claims import ClaimAssessment, ClaimStatus, assess_lines, classify_claim, safe_proof_for


# ---------------------------------------------------------------- strategy spec

@dataclass
class VariantStrategy:
    """Declarative description of one creative variant."""

    key: str
    name: str
    label: str  # A / B / C ...
    strategic_objective: str
    target_emotional_response: str
    narrative: NarrativeStructure
    tone: Tone
    visual_style: str
    camera_language: str
    vo_strategy: str
    cta_strategy: str
    pacing: str  # fast / medium / slow
    purposes: list[ScenePurpose]
    risks: list[str] = field(default_factory=list)
    claim_requirements: str = ""


def _scene_count(brief: AdBrief) -> int:
    # ~1 scene / 3-4s for fast pacing variants; 4-6s for slow.
    return max(3, min(8, round(brief.duration_seconds / 4)))


def _aspect(platform: Platform) -> AspectRatio:
    return AspectRatio.HORIZONTAL_16_9 if platform == Platform.YOUTUBE else AspectRatio.VERTICAL_9_16


# ---------------------------------------------------------------- the 7 strategies

def _strategies(brief: AdBrief) -> list[VariantStrategy]:
    """Return all seven variant strategies (genuinely different)."""
    p = brief.platform
    short_form = p in (Platform.TIKTOK, Platform.INSTAGRAM_REELS, Platform.YOUTUBE_SHORTS)
    base_count = _scene_count(brief)

    return [
        VariantStrategy(
            key="emotional", name="Emotional", label="A",
            strategic_objective="Move the viewer to feel something personal about the product.",
            target_emotional_response="Warmth, aspiration, emotional resonance.",
            narrative=NarrativeStructure.THREE_ACT,
            tone=Tone.EMOTIONAL,
            visual_style="warm golden-hour cinematography, soft focus, intimate close-ups, muted filmic grade",
            camera_language="slow dolly & gentle handheld, shallow depth of field, human-scale framing",
            vo_strategy="First-person reflective voiceover, slow cadence, emotional pauses.",
            cta_strategy="Soft, invitation-style CTA; feeling over urgency.",
            pacing="slow",
            purposes=[ScenePurpose.HOOK, ScenePurpose.ESTABLISH, ScenePurpose.EMOTION, ScenePurpose.PRODUCT, ScenePurpose.CTA],
            risks=["May under-perform on direct-response metrics.", "Needs strong casting/performer."],
            claim_requirements="Avoid performance claims; lean on feeling.",
        ),
        VariantStrategy(
            key="direct_response", name="Direct Response", label="B",
            strategic_objective="Drive immediate action with urgency and a clear offer.",
            target_emotional_response="Urgency, FOMO, decisiveness.",
            narrative=NarrativeStructure.BEFORE_AFTER,
            tone=Tone.AGGRESSIVE,
            visual_style="high-contrast, punchy color, bold text-on-screen space, fast cuts",
            camera_language="snappy static & quick push-ins, product-centric framing",
            vo_strategy="Direct, imperative voiceover; problem-agitate-solution cadence; repeat CTA.",
            cta_strategy="Hard CTA with offer + deadline; repeated.",
            pacing="fast",
            purposes=[ScenePurpose.HOOK, ScenePurpose.PROBLEM, ScenePurpose.SOLUTION, ScenePurpose.BENEFIT, ScenePurpose.CTA, ScenePurpose.CTA],
            risks=["Can feel salesy; claim risk on offer/performance language."],
            claim_requirements="Offer must match brief; no fabricated discounts or guarantees.",
        ),
        VariantStrategy(
            key="cinematic", name="Cinematic", label="C",
            strategic_objective="Position the product as premium through filmic craft.",
            target_emotional_response="Awe, aspiration, prestige.",
            narrative=NarrativeStructure.THREE_ACT,
            tone=Tone.CINEMATIC,
            visual_style="anamorphic lens flares, moody lighting, rich color grade, epic compositions",
            camera_language="slow crane, dolly, orbit; wide establishing shots; macro hero shots",
            vo_strategy="Minimal voiceover; let the score and imagery carry the story.",
            cta_strategy="Restrained, brand-led CTA at the end.",
            pacing="slow",
            purposes=[ScenePurpose.HOOK, ScenePurpose.ESTABLISH, ScenePurpose.PRODUCT, ScenePurpose.EMOTION, ScenePurpose.CTA],
            risks=["Weak direct response; long setup hurts short-form if not cut tightly."],
            claim_requirements="No claims; pure brand/image building.",
        ),
        VariantStrategy(
            key="ugc", name="UGC", label="D",
            strategic_objective="Feel authentic and native to the feed.",
            target_emotional_response="Relatability, trust, curiosity.",
            narrative=NarrativeStructure.BEFORE_AFTER,
            tone=Tone.PLAYFUL,
            visual_style="phone-shot, vertical, natural lighting, minimal grading, real environment",
            camera_language="selfie handheld, direct-to-camera, quick whip pans",
            vo_strategy="Conversational first-person; 'POV' framing; casual pacing.",
            cta_strategy="Native, low-friction CTA ('comment / link in bio').",
            pacing="fast",
            purposes=[ScenePurpose.HOOK, ScenePurpose.PROBLEM, ScenePurpose.PRODUCT, ScenePurpose.PROOF, ScenePurpose.CTA],
            risks=["Authenticity can suffer if claims are over-stated."],
            claim_requirements="No fabricated testimonials; only approved ones.",
        ),
        VariantStrategy(
            key="product_demo", name="Product Demonstration", label="E",
            strategic_objective="Show exactly how the product works and delivers value.",
            target_emotional_response="Clarity, confidence, 'I see how it works'.",
            narrative=NarrativeStructure.BEFORE_AFTER,
            tone=Tone.PROFESSIONAL,
            visual_style="clean studio, macro detail, controlled lighting, crisp focus",
            camera_language="macro, top-down, slow orbit; step-by-step coverage",
            vo_strategy="Instructional voiceover explaining each step clearly.",
            cta_strategy="Feature-led CTA ('see it in action — try it').",
            pacing="medium",
            purposes=[ScenePurpose.HOOK, ScenePurpose.PRODUCT, ScenePurpose.PROOF, ScenePurpose.BENEFIT, ScenePurpose.CTA],
            risks=["Claim risk if demo implies results not supplied as facts."],
            claim_requirements="Demo must only show approved capabilities; no result claims.",
        ),
        VariantStrategy(
            key="problem_solution", name="Problem/Solution", label="F",
            strategic_objective="Make the pain vivid, then resolve it with the product.",
            target_emotional_response="Relief, recognition, satisfaction.",
            narrative=NarrativeStructure.BEFORE_AFTER,
            tone=Tone.URGENT,
            visual_style="contrast: desaturated problem beat -> vibrant solution beat",
            camera_language="shaky/tight on problem; smooth/wide on solution",
            vo_strategy="Question-led VO ('Tired of...?') then resolved answer.",
            cta_strategy="Solution-led CTA tied to the resolved pain.",
            pacing="medium",
            purposes=[ScenePurpose.HOOK, ScenePurpose.PROBLEM, ScenePurpose.SOLUTION, ScenePurpose.BENEFIT, ScenePurpose.CTA],
            risks=["Over-agitating the problem can feel manipulative."],
            claim_requirements="Problem must be realistic; solution claims must be approved.",
        ),
        VariantStrategy(
            key="testimonial", name="Testimonial", label="G",
            strategic_objective="Build credibility through a real customer voice.",
            target_emotional_response="Trust, social proof, reassurance.",
            narrative=NarrativeStructure.TESTIMONIAL,
            tone=Tone.INSPIRATIONAL,
            visual_style="documentary interview styling, soft key light, shallow background",
            camera_language="locked-off interview framing, slow push-in, b-roll cutaways",
            vo_strategy="Quoted customer voiceover (approved testimonial only).",
            cta_strategy="Trust-led CTA ('join them — try it').",
            pacing="medium",
            purposes=[ScenePurpose.HOOK, ScenePurpose.PROOF, ScenePurpose.BENEFIT, ScenePurpose.PRODUCT, ScenePurpose.CTA],
            risks=["Fabricated testimonials are prohibited; requires an approved one.",
                   "If no approved testimonial exists, proof scene is marked requires_verification."],
            claim_requirements="Only approved testimonials may be quoted verbatim.",
        ),
    ]


# ---------------------------------------------------------------- hook builders (genuinely different)

def _hook(variant: VariantStrategy, brief: AdBrief) -> str:
    p = brief.product_or_service
    aud = brief.target_audience
    if variant.key == "emotional":
        return f"You know that feeling — the moment {p} finally fits into your life."
    if variant.key == "direct_response":
        return f"Stop scrolling. If you're {aud}, this 15 seconds changes everything."
    if variant.key == "cinematic":
        return f"Every great story begins with a single detail. This is {p}."
    if variant.key == "ugc":
        return f"POV: you just found the {p} everyone's been talking about."
    if variant.key == "product_demo":
        return f"Watch exactly how {p} works — in 10 seconds."
    if variant.key == "problem_solution":
        return f"Tired of settling for less? Here's how {p} fixes it."
    if variant.key == "testimonial":
        return f"'I didn't believe it either' — a real take on {p}."
    return f"Introducing {p}."


def _vo_line(purpose: ScenePurpose, variant: VariantStrategy, brief: AdBrief, proof: Optional[str]) -> str:
    p = brief.product_or_service
    cta = brief.effective_cta
    if variant.key == "emotional":
        return {
            ScenePurpose.ESTABLISH: f"There's a version of your day that feels lighter.",
            ScenePurpose.EMOTION: f"That's the feeling {p} was made for.",
            ScenePurpose.PRODUCT: f"Crafted for {brief.target_audience}.",
            ScenePurpose.CTA: cta,
        }.get(purpose, f"{p}, for the moments that matter.")
    if variant.key == "direct_response":
        return {
            ScenePurpose.PROBLEM: f"You're losing time every single day without {p}.",
            ScenePurpose.SOLUTION: f"{p} is the fix. Here's why.",
            ScenePurpose.BENEFIT: "Less effort. Real difference.",
            ScenePurpose.CTA: cta,
        }.get(purpose, f"{p}. Act now.")
    if variant.key == "cinematic":
        return {
            ScenePurpose.PRODUCT: f"{p}.",
            ScenePurpose.CLOSING: cta,
            ScenePurpose.EMOTION: "",
        }.get(purpose, "")
    if variant.key == "ugc":
        return {
            ScenePurpose.PROBLEM: f"Okay so I used to struggle with this too.",
            ScenePurpose.PRODUCT: f"Then I tried {p}.",
            ScenePurpose.PROOF: proof or "Here's what happened.",
            ScenePurpose.CTA: cta,
        }.get(purpose, f"So yeah — {p}.")
    if variant.key == "product_demo":
        return {
            ScenePurpose.PRODUCT: f"Step one — this is {p}.",
            ScenePurpose.PROOF: proof or "Watch it do exactly what it's designed to.",
            ScenePurpose.BENEFIT: "And that's the result.",
            ScenePurpose.CTA: cta,
        }.get(purpose, f"{p}, step by step.")
    if variant.key == "problem_solution":
        return {
            ScenePurpose.PROBLEM: f"Sound familiar? It doesn't have to be this way.",
            ScenePurpose.SOLUTION: f"Enter {p}.",
            ScenePurpose.BENEFIT: "Problem, solved.",
            ScenePurpose.CTA: cta,
        }.get(purpose, f"{p}.")
    if variant.key == "testimonial":
        return {
            ScenePurpose.PROOF: (f'"{proof}"' if proof else '"I was skeptical at first."'),
            ScenePurpose.BENEFIT: "And it genuinely made a difference.",
            ScenePurpose.PRODUCT: f"This is {p}.",
            ScenePurpose.CTA: cta,
        }.get(purpose, "")
    return ""


def _camera(purpose: ScenePurpose, variant: VariantStrategy) -> SceneCamera:
    if variant.key == "emotional":
        return SceneCamera(shot_type="close-up", lens="50mm", framing="intimate", movement="slow dolly in")
    if variant.key == "direct_response":
        return SceneCamera(shot_type="medium", lens="35mm", framing="product + text space", movement="quick push-in")
    if variant.key == "cinematic":
        return SceneCamera(shot_type="wide", lens="anamorphic 40mm", framing="epic composition", movement="slow crane")
    if variant.key == "ugc":
        return SceneCamera(shot_type="selfie", lens="phone wide", framing="direct to camera", movement="handheld")
    if variant.key == "product_demo":
        return SceneCamera(shot_type="macro", lens="50mm", framing="top-down / detail", movement="slow orbit")
    if variant.key == "problem_solution":
        if purpose == ScenePurpose.PROBLEM:
            return SceneCamera(shot_type="close-up", lens="35mm", framing="tight & shaky", movement="handheld")
        return SceneCamera(shot_type="medium", lens="35mm", framing="smooth & open", movement="slow dolly")
    if variant.key == "testimonial":
        return SceneCamera(shot_type="medium", lens="50mm", framing="interview", movement="locked off / slow push-in")
    return SceneCamera(shot_type="medium", lens="35mm", framing="centered", movement="static")


# ---------------------------------------------------------------- variant result

@dataclass
class VariantResult:
    """A generated variant: strategy metadata + ProductionPlan + claim checks."""

    strategy: VariantStrategy
    plan: ProductionPlan
    claim_checks: list[ClaimAssessment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "variant": self.strategy.key,
            "label": self.strategy.label,
            "name": self.strategy.name,
            "strategic_objective": self.strategy.strategic_objective,
            "target_emotional_response": self.strategy.target_emotional_response,
            "narrative": self.strategy.narrative.value,
            "tone": self.strategy.tone.value,
            "visual_style": self.strategy.visual_style,
            "camera_language": self.strategy.camera_language,
            "vo_strategy": self.strategy.vo_strategy,
            "cta_strategy": self.strategy.cta_strategy,
            "pacing": self.strategy.pacing,
            "risks": self.strategy.risks,
            "claim_requirements": self.strategy.claim_requirements,
            "warnings": self.warnings,
            "claim_checks": [c.to_dict() for c in self.claim_checks],
            "plan": json_safe(self.plan),
        }


def json_safe(plan: ProductionPlan) -> dict:
    return plan.model_dump()


# ---------------------------------------------------------------- plan builder

def _build_scene(
    idx: int, purpose: ScenePurpose, duration: float, variant: VariantStrategy,
    brief: AdBrief, proof: Optional[str],
) -> Scene:
    p = brief.product_or_service
    vo = _vo_line(purpose, variant, brief, proof)
    cam = _camera(purpose, variant)
    desc_map = {
        ScenePurpose.HOOK: f"{variant.name} hook: {_hook(variant, brief)}",
        ScenePurpose.PROBLEM: f"Vivid problem beat for {brief.target_audience}.",
        ScenePurpose.SOLUTION: f"{p} introduced as the {variant.name.lower()} resolution.",
        ScenePurpose.BENEFIT: f"A concrete, on-brand benefit of {p}.",
        ScenePurpose.PROOF: (f"Approved proof: {proof}." if proof else "Proof scene — requires verification, no claim fabricated."),
        ScenePurpose.PRODUCT: f"Hero {variant.name.lower()} product moment for {p}.",
        ScenePurpose.EMOTION: f"Emotional beat connecting {brief.target_audience} to {p}.",
        ScenePurpose.ESTABLISH: f"{variant.name} establishing atmosphere for {p}.",
        ScenePurpose.CLOSING: f"Cinematic closing image for {p}.",
        ScenePurpose.CTA: brief.effective_cta,
    }
    desc = desc_map.get(purpose, f"{variant.name} scene for {p}.")
    visual = (
        f"{desc} Subject: {p}. Camera: {cam.shot_type} {cam.lens or ''} {cam.movement or ''}. "
        f"{variant.visual_style}. Coherent identity, {variant.pacing} pacing, no text on screen."
    )
    caption_text = vo or desc
    return Scene(
        index=idx,
        purpose=purpose,
        duration=duration,
        description=desc,
        camera=cam,
        environment=f"{variant.name.lower()} environment consistent with continuity",
        lighting=variant.visual_style.split(",")[0],
        characters=[SceneCharacter(ref="protagonist", action=desc)] if variant.key != "cinematic" else [],
        visual_prompt=visual,
        negative_prompt="blurry, distorted anatomy, extra limbs, duplicate subjects, deformed face, text, logo, watermark, low quality, inconsistent identity",
        voiceover=SceneVoiceover(line=vo, direction=variant.vo_strategy, start_offset=0.0),
        sfx=(["subtle whoosh"] if purpose == ScenePurpose.HOOK else []),
        music=("emotional score" if variant.key == "emotional" else
               "punchy beat" if variant.key == "direct_response" else
               "epic cinematic score" if variant.key == "cinematic" else
               "lo-fi bed" if variant.key == "ugc" else
               "clean minimal bed" if variant.key == "product_demo" else
               "tension->release" if variant.key == "problem_solution" else
               "warm inspirational"),
        caption=SceneCaption(text=caption_text, timing_intent="first_half"),
        transition_intent=("fast cut" if variant.pacing == "fast" else "hard cut" if variant.pacing == "medium" else "slow dissolve"),
        continuity_refs=(["protagonist"] + ([p] if p else [])),
    )


def _purposes_for(variant: VariantStrategy, brief: AdBrief) -> list[ScenePurpose]:
    """Adapt the variant's purpose sequence to the requested duration/count."""
    count = max(3, min(len(variant.purposes), _scene_count(brief)))
    seq = variant.purposes[:count]
    # Guarantee hook first and cta last for all ad variants (advertisement mode requires both).
    seq[0] = ScenePurpose.HOOK
    seq[-1] = ScenePurpose.CTA
    return seq


def build_variant_plan(variant: VariantStrategy, brief: AdBrief) -> VariantResult:
    """Build one ProductionPlan for a variant, with claim safety enforced."""
    proof, warnings = safe_proof_for(brief)
    purposes = _purposes_for(variant, brief)
    durations = _durations(purposes, brief.duration_seconds)
    scenes = [
        _build_scene(i + 1, purposes[i], durations[i], variant, brief, proof)
        for i in range(len(purposes))
    ]

    content_plan = ContentPlan(
        topic=brief.product_or_service,
        product_or_service=brief.product_or_service,
        audience=Audience(description=brief.target_audience, market=brief.market),
        platform=brief.platform,
        objective=brief.objective,
        core_message=f"{brief.product_or_service} for {brief.target_audience}.",
        language=brief.language,
        country=brief.market,
        duration_seconds=brief.duration_seconds,
        tone=variant.tone,
        visual_style=variant.visual_style,
        mode=ProductionMode.ADVERTISEMENT,
        aspect_ratio=_aspect(brief.platform),
        variant_hint=AdVariant(variant.key) if variant.key in {v.value for v in AdVariant} else None,
    )

    creative = CreativeDecision(
        hook=_hook(variant, brief),
        problem=("A relatable pain for the audience." if ScenePurpose.PROBLEM in purposes else None),
        solution=(brief.product_or_service if ScenePurpose.SOLUTION in purposes else None),
        benefits=["on-brand benefit"] if ScenePurpose.BENEFIT in purposes else [],
        proof=(proof if (ScenePurpose.PROOF in purposes and proof) else None),
        cta=brief.effective_cta,
        narrative_structure=variant.narrative,
    )
    continuity = ContinuityMemory(
        characters=[CharacterIdentity(role="protagonist", appearance="consistent subject across scenes", clothing="stable wardrobe")]
        if variant.key != "cinematic" else [],
        product=ProductIdentity(name=brief.product_or_service, appearance=f"the {brief.product_or_service}, stable identity",
                                signature_colors=brief.brand.colors if brief.brand else []),
        environment=f"{variant.name.lower()} environment",
        lighting=variant.visual_style.split(",")[0],
        color_palette=brief.brand.colors if brief.brand else [],
        camera_language=variant.camera_language,
        visual_style=variant.visual_style,
    )
    voiceover = VoiceOverPlan(
        full_script=" ".join(s.voiceover.line for s in scenes if s.voiceover.line),
        direction=variant.vo_strategy,
        language=brief.language,
        voice_gender=("female" if variant.tone in (Tone.EMOTIONAL, Tone.INSPIRATIONAL) else "male"),
        pace=variant.pacing,
    )
    audio = AudioPlan(
        music_direction=f"{variant.tone.value} {variant.name.lower()} score",
        sfx_direction=f"{variant.pacing} pacing; punch on hook and CTA",
        ambience=f"{variant.name.lower()} ambience",
    )
    captions = CaptionPlan(
        items=[CaptionItem(scene_index=s.index, text=s.caption.text, timing_intent=s.caption.timing_intent) for s in scenes],
        style_preset=brief.platform.value,
        format_intent="burned_in",
    )

    # Claim safety: assess every VO/caption/proof line emitted.
    lines = [s.voiceover.line for s in scenes if s.voiceover.line] + [s.caption.text for s in scenes]
    if proof:
        lines.append(proof)
    checks = assess_lines(lines, brief)
    # Surface any unverified/prohibited checks as warnings.
    for c in checks:
        if c.status in (ClaimStatus.UNVERIFIED_CLAIM, ClaimStatus.PROHIBITED):
            warnings.append(f"Claim safety: '{c.text[:60]}…' -> {c.status.value}: {c.reason}")

    plan = ProductionPlan(
        title=f"{brief.product_or_service} — {variant.name} ({variant.label})",
        content=content_plan,
        creative=creative,
        continuity=continuity,
        scenes=scenes,
        voiceover=voiceover,
        audio=audio,
        captions=captions,
        brand=brief.brand,
        meta=PlanMetadata(planner="local", warnings=warnings, model=None),
    )
    return VariantResult(strategy=variant, plan=plan, claim_checks=checks, warnings=warnings)


# ---------------------------------------------------------------- public API

_VARIANT_KEYS: list[str] = ["emotional", "direct_response", "cinematic", "ugc", "product_demo", "problem_solution", "testimonial"]


def all_variant_keys() -> list[str]:
    """Return the seven variant keys."""
    return list(_VARIANT_KEYS)


def _strategy_by_key(key: str, brief: AdBrief) -> VariantStrategy:
    for v in _strategies(brief):
        if v.key == key:
            return v
    raise KeyError(f"Unknown variant: {key}")


def generate_variants(brief: AdBrief) -> list[VariantResult]:
    """Generate all seven variant plans for an AdBrief."""
    return [build_variant_plan(v, brief) for v in _strategies(brief)]


def generate_variant(brief: AdBrief, key: str) -> VariantResult:
    """Generate a single variant by key."""
    return build_variant_plan(_strategy_by_key(key, brief), brief)

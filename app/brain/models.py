"""AI Content Brain — strongly typed production plan models (Phase 5).

The schema deliberately separates four concerns so they never get tangled:

    CONTENT PLAN     → who/what/why (audience, platform, objective, message)
    CREATIVE DECISION → narrative & persuasion (hook, problem, solution, CTA)
    SCENE PLAN        → shot-level staging (camera, lighting, characters, VO, SFX)
    GENERATION PROMPT → the actual text-to-video prompt strings (visual_prompt,
                        negative_prompt) consumed by downstream providers

Phase 5 builds the intelligence/plan only. It does NOT generate video, audio,
or captions, and does NOT implement the Phase 6 variant engine — but the schema
reserves fields (``AdVariant``, ``variant_hint``) so Phase 6 can plug in later
without a schema rewrite.

A :class:`ContinuityMemory` block is carried through the plan so future Phase 7
scene continuity can keep character/product/environment identity stable across
scenes.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------- enums

class Platform(str, Enum):
    TIKTOK = "tiktok"
    INSTAGRAM_REELS = "instagram_reels"
    YOUTUBE_SHORTS = "youtube_shorts"
    YOUTUBE = "youtube"
    OTHER = "other"


class Objective(str, Enum):
    AWARENESS = "awareness"
    CONSIDERATION = "consideration"
    CONVERSION = "conversion"
    ENGAGEMENT = "engagement"
    STORYTELLING = "storytelling"


class Tone(str, Enum):
    EMOTIONAL = "emotional"
    AGGRESSIVE = "aggressive"
    CINEMATIC = "cinematic"
    PLAYFUL = "playful"
    PROFESSIONAL = "professional"
    URGENT = "urgent"
    INSPIRATIONAL = "inspirational"


class ProductionMode(str, Enum):
    """High-level content kind. ``advertisement`` enables ad creative fields."""

    ADVERTISEMENT = "advertisement"
    CINEMATIC = "cinematic"
    DOCUMENTARY = "documentary"
    UGC = "ugc"
    PROMOTIONAL = "promotional"


class AdVariant(str, Enum):
    """Reserved for Phase 6. Phase 5 only records a hint, never runs variants."""

    EMOTIONAL = "emotional"
    DIRECT_RESPONSE = "direct_response"
    CINEMATIC = "cinematic"
    UGC = "ugc"
    PRODUCT_DEMO = "product_demo"
    PROBLEM_SOLUTION = "problem_solution"
    TESTIMONIAL = "testimonial"


class NarrativeStructure(str, Enum):
    HOOK_PROBLEM_SOLUTION_BENEFIT_PROOF_CTA = "hook_problem_solution_benefit_proof_cta"
    THREE_ACT = "three_act"
    BEFORE_AFTER = "before_after"
    TESTIMONIAL = "testimonial"
    MONTAGE = "montage"
    DOCUMENTARY = "documentary"


class ScenePurpose(str, Enum):
    HOOK = "hook"
    ESTABLISH = "establish"
    PROBLEM = "problem"
    SOLUTION = "solution"
    BENEFIT = "benefit"
    PROOF = "proof"
    EMOTION = "emotion"
    PRODUCT = "product"
    TRANSITION = "transition"
    CTA = "cta"
    CLOSING = "closing"


class AspectRatio(str, Enum):
    VERTICAL_9_16 = "9:16"
    HORIZONTAL_16_9 = "16:9"
    SQUARE_1_1 = "1:1"


# ---------------------------------------------------------------- layer 1: content plan

class Audience(BaseModel):
    description: str = Field(..., description="Primary target audience description.")
    age_range: Optional[str] = None
    interests: list[str] = Field(default_factory=list)
    market: Optional[str] = Field(None, description="Country/market, e.g. DE, US, AE.")


class ContentPlan(BaseModel):
    """Layer 1 — CONTENT PLAN: the strategic who/what/why."""

    topic: str
    product_or_service: Optional[str] = None
    audience: Audience
    platform: Platform
    objective: Objective
    core_message: str
    language: str = "en"
    country: Optional[str] = None
    duration_seconds: float = Field(..., gt=0)
    tone: Tone
    visual_style: str
    mode: ProductionMode = ProductionMode.CINEMATIC
    aspect_ratio: AspectRatio = AspectRatio.VERTICAL_9_16
    variant_hint: Optional[AdVariant] = None


# ---------------------------------------------------------------- layer 2: creative decision

class CreativeDecision(BaseModel):
    """Layer 2 — CREATIVE DECISION: narrative & persuasion structure."""

    hook: str = Field(..., description="First-seconds attention grabber.")
    problem: Optional[str] = None
    solution: Optional[str] = None
    benefits: list[str] = Field(default_factory=list)
    proof: Optional[str] = None
    cta: str = Field(..., description="Call to action.")
    narrative_structure: NarrativeStructure


# ---------------------------------------------------------------- continuity memory

class CharacterIdentity(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    appearance: str = Field(..., description="Stable physical appearance description.")
    clothing: Optional[str] = None
    physical_attributes: list[str] = Field(default_factory=list)


class ProductIdentity(BaseModel):
    name: Optional[str] = None
    appearance: str = Field(..., description="Stable product visual description.")
    packaging: Optional[str] = None
    signature_colors: list[str] = Field(default_factory=list)


class ContinuityMemory(BaseModel):
    """Carried across scenes so identity does not randomly change."""

    characters: list[CharacterIdentity] = Field(default_factory=list)
    product: Optional[ProductIdentity] = None
    environment: str = ""
    time_of_day: Optional[str] = None
    lighting: str = ""
    color_palette: list[str] = Field(default_factory=list)
    camera_language: str = ""
    visual_style: str = ""


# ---------------------------------------------------------------- layer 3: scene plan

class SceneCamera(BaseModel):
    shot_type: Optional[str] = Field(None, description="e.g. wide, medium, close-up, macro.")
    lens: Optional[str] = None
    framing: Optional[str] = None
    movement: Optional[str] = Field(None, description="e.g. dolly in, handheld, static, orbit.")


class SceneCharacter(BaseModel):
    ref: Optional[str] = Field(None, description="Reference to a ContinuityMemory character name/role.")
    action: str = ""


class SceneVoiceover(BaseModel):
    line: str = ""
    direction: Optional[str] = Field(None, description="Tone/pace/emotion direction for TTS.")
    start_offset: float = Field(0.0, ge=0.0, description="Seconds into the scene the line starts.")
    # Timing consistency: end must not exceed scene duration (validated in Scene).


class SceneCaption(BaseModel):
    text: str = ""
    timing_intent: Optional[str] = Field(None, description="e.g. full_scene, first_half, punchline.")


class Scene(BaseModel):
    """Layer 3 — SCENE PLAN for one shot, plus Layer 4 generation prompts.

    The visual_prompt / negative_prompt fields are the GENERATION PROMPT layer
    consumed by downstream video providers. They are kept on the Scene (rather
    than a separate object) because they are shot-specific, but are clearly
    named to distinguish them from the staging fields above.
    """

    index: int = Field(..., ge=1)
    purpose: ScenePurpose
    duration: float = Field(..., gt=0)
    description: str
    camera: SceneCamera = Field(default_factory=SceneCamera)
    environment: str = ""
    lighting: str = ""
    characters: list[SceneCharacter] = Field(default_factory=list)
    # Layer 4: generation prompt
    visual_prompt: str
    negative_prompt: str
    # scene-level audio/caption intent (rendering happens in later phases)
    voiceover: SceneVoiceover = Field(default_factory=SceneVoiceover)
    sfx: list[str] = Field(default_factory=list)
    music: Optional[str] = None
    caption: SceneCaption = Field(default_factory=SceneCaption)
    transition_intent: Optional[str] = None
    continuity_refs: list[str] = Field(
        default_factory=list,
        description="Names/roles this scene must reuse from ContinuityMemory.",
    )

    @model_validator(mode="after")
    def _voiceover_within_scene(self) -> "Scene":
        vo = self.voiceover
        if vo.line and vo.start_offset > self.duration:
            raise ValueError(
                f"Scene {self.index}: voiceover start_offset {vo.start_offset} "
                f"exceeds scene duration {self.duration}."
            )
        return self


# ---------------------------------------------------------------- audio / caption plans

class VoiceOverPlan(BaseModel):
    full_script: str
    direction: str
    language: str = "en"
    # Per-scene timing is encoded on each Scene.voiceover; this is the holistic view.
    voice_gender: Optional[str] = None
    pace: Optional[str] = None


class AudioPlan(BaseModel):
    music_direction: str
    sfx_direction: str
    ambience: Optional[str] = None


class CaptionItem(BaseModel):
    scene_index: int
    text: str
    timing_intent: Optional[str] = None


class CaptionPlan(BaseModel):
    items: list[CaptionItem] = Field(default_factory=list)
    style_preset: Optional[str] = Field(None, description="e.g. tiktok, reels, shorts.")
    format_intent: Optional[str] = Field(None, description="burned_in, srt, vtt — rendered later.")


# ---------------------------------------------------------------- brand

class BrandProfile(BaseModel):
    brand_name: Optional[str] = None
    logo_note: Optional[str] = None
    colors: list[str] = Field(default_factory=list)
    typography: Optional[str] = None
    tone_of_voice: Optional[str] = None
    preferred_language: Optional[str] = None
    cta: Optional[str] = None
    visual_style: Optional[str] = None


# ---------------------------------------------------------------- top-level plan

class PlanMetadata(BaseModel):
    planner: str = Field(..., description="local | gemini | local_fallback")
    warnings: list[str] = Field(default_factory=list)
    model: Optional[str] = None


class ProductionPlan(BaseModel):
    """The complete, JSON-serializable production intelligence plan.

    Downstream phases (scene generation, video, audio, captions, editing) read
    from this plan. Phase 5 only produces it; it never generates media.
    """

    title: str
    content: ContentPlan
    creative: CreativeDecision
    continuity: ContinuityMemory
    scenes: list[Scene]
    voiceover: VoiceOverPlan
    audio: AudioPlan
    captions: CaptionPlan
    brand: Optional[BrandProfile] = None
    meta: PlanMetadata

    @model_validator(mode="after")
    def _timing_consistency(self) -> "ProductionPlan":
        if not self.scenes:
            raise ValueError("ProductionPlan must contain at least one scene.")
        total = sum(s.duration for s in self.scenes)
        target = self.content.duration_seconds
        # Allow a small float tolerance for rounding.
        if abs(total - target) > 0.5:
            raise ValueError(
                f"Scene durations sum to {total:.2f}s but target duration is {target}s."
            )
        # Hook must be present (first or among first scenes) for social/ad content.
        purposes = [s.purpose for s in self.scenes]
        if ScenePurpose.HOOK not in purposes and self.content.mode != ProductionMode.DOCUMENTARY:
            raise ValueError("ProductionPlan must include at least one hook scene.")
        if ScenePurpose.CTA not in purposes and self.content.mode in (
            ProductionMode.ADVERTISEMENT,
            ProductionMode.PROMOTIONAL,
        ):
            raise ValueError("Advertisement/Promotional plans must include a CTA scene.")
        # Indexes must be contiguous starting at 1.
        indexes = [s.index for s in self.scenes]
        if indexes != list(range(1, len(self.scenes) + 1)):
            raise ValueError(f"Scene indexes must be contiguous from 1; got {indexes}.")
        return self


# ---------------------------------------------------------------- input brief

class ContentBrief(BaseModel):
    """User-facing input to the Content Brain."""

    idea: str = Field("", description="Raw idea / topic.")
    product_or_service: Optional[str] = None
    audience: Optional[str] = None
    platform: Platform = Platform.YOUTUBE_SHORTS
    language: str = "en"
    country: Optional[str] = None
    duration_seconds: float = Field(30.0, gt=0, le=300)
    objective: Objective = Objective.STORYTELLING
    tone: Tone = Tone.CINEMATIC
    visual_style: Optional[str] = None
    mode: ProductionMode = ProductionMode.CINEMATIC
    variant_hint: Optional[AdVariant] = None
    brand: Optional[BrandProfile] = None
    script: Optional[str] = None
    scene_count: Optional[int] = Field(None, ge=1, le=12)

    @model_validator(mode="after")
    def _need_idea_or_product(self) -> "ContentBrief":
        if not (self.idea.strip() or (self.product_or_service or "").strip()):
            raise ValueError("Provide at least an idea or a product/service.")
        return self

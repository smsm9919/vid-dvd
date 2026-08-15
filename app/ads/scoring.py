"""Creative scoring and variant comparison (Phase 6).

A deterministic heuristic that evaluates each variant across nine dimensions.
This is a *creative-quality heuristic only* — it does NOT predict actual
conversions or marketing performance, and it never invents performance metrics.

Dimensions (each 0-10):
    hook_strength, clarity, product_visibility, differentiation, pacing,
    cta_clarity, audience_relevance, claim_safety, platform_fit

Scores are derived from the variant's strategy + ProductionPlan + claim checks,
not from random or runtime signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..brain.models import Platform, ScenePurpose
from .brief import AdBrief, ClaimStatus
from .variants import VariantResult, VariantStrategy


@dataclass
class DimensionScore:
    dimension: str
    score: float
    reason: str


@dataclass
class CreativeScore:
    """Heuristic creative-quality score for one variant."""

    variant: str
    label: str
    dimensions: list[DimensionScore] = field(default_factory=list)
    total: float = 0.0
    max_total: float = 90.0  # 9 dims * 10
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.total = round(sum(d.score for d in self.dimensions), 2)

    @property
    def normalized(self) -> float:
        return round(self.total / self.max_total, 3) if self.max_total else 0.0

    def to_dict(self) -> dict:
        return {
            "variant": self.variant,
            "label": self.label,
            "total": self.total,
            "max_total": self.max_total,
            "normalized": self.normalized,
            "dimensions": [{"dimension": d.dimension, "score": d.score, "reason": d.reason} for d in self.dimensions],
            "notes": self.notes,
        }


_DIMENSIONS = [
    "hook_strength", "clarity", "product_visibility", "differentiation", "pacing",
    "cta_clarity", "audience_relevance", "claim_safety", "platform_fit",
]


def _clamp(x: float) -> float:
    return max(0.0, min(10.0, round(x, 1)))


def score_variant(result: VariantResult, brief: AdBrief) -> CreativeScore:
    """Compute a deterministic heuristic score for a variant."""
    s: VariantStrategy = result.strategy
    plan = result.plan
    checks = result.claim_checks
    purposes = [sc.purpose for sc in plan.scenes]

    # --- hook_strength: hook present + short-form punchiness
    has_hook = ScenePurpose.HOOK in purposes
    hook_len = len(plan.creative.hook)
    hook_score = 6.0 if has_hook else 2.0
    if has_hook and brief.platform in (Platform.TIKTOK, Platform.INSTAGRAM_REELS, Platform.YOUTUBE_SHORTS):
        hook_score += 2.0 if hook_len <= 90 else 0.5
    if s.key in ("direct_response", "ugc", "problem_solution"):
        hook_score += 1.5  # pattern-interrupt style hooks

    # --- clarity: direct_response/product_demo highest; cinematic lower (image-led)
    clarity = {"direct_response": 9.0, "product_demo": 9.0, "problem_solution": 8.5,
               "ugc": 7.5, "testimonial": 7.0, "emotional": 6.5, "cinematic": 5.5}.get(s.key, 6.0)

    # --- product_visibility: product_demo/problem_solution highest
    product_vis = {"product_demo": 9.5, "direct_response": 8.5, "problem_solution": 8.0,
                   "ugc": 7.5, "testimonial": 6.5, "emotional": 6.0, "cinematic": 7.0}.get(s.key, 6.0)
    if ScenePurpose.PRODUCT in purposes:
        product_vis += 1.0

    # --- differentiation: distinctiveness of visual/camera approach
    diff = {"cinematic": 9.0, "ugc": 8.5, "testimonial": 8.0, "emotional": 7.0,
            "product_demo": 6.5, "problem_solution": 6.0, "direct_response": 5.5}.get(s.key, 6.0)

    # --- pacing fit to platform/duration
    short_form = brief.platform in (Platform.TIKTOK, Platform.INSTAGRAM_REELS, Platform.YOUTUBE_SHORTS)
    if short_form:
        pacing = {"direct_response": 9.0, "ugc": 8.5, "product_demo": 7.5, "problem_solution": 7.5,
                  "testimonial": 6.0, "emotional": 5.5, "cinematic": 5.0}.get(s.key, 6.0)
    else:
        pacing = {"cinematic": 9.0, "emotional": 8.5, "testimonial": 8.0, "product_demo": 7.5,
                  "problem_solution": 7.0, "direct_response": 6.5, "ugc": 6.0}.get(s.key, 6.0)

    # --- cta_clarity: direct_response highest; cinematic lowest
    cta_score = {"direct_response": 9.5, "problem_solution": 8.5, "product_demo": 8.0,
                 "ugc": 7.5, "testimonial": 7.0, "emotional": 6.0, "cinematic": 4.5}.get(s.key, 6.0)
    if ScenePurpose.CTA in purposes:
        cta_score += 1.0
    # repeated CTA bonus (direct_response has two CTA scenes)
    cta_count = purposes.count(ScenePurpose.CTA)
    if cta_count > 1:
        cta_score += 1.0

    # --- audience_relevance: UGC/testimonial high relatability
    aud = {"ugc": 9.0, "testimonial": 8.5, "emotional": 8.0, "problem_solution": 8.0,
           "direct_response": 7.5, "product_demo": 7.0, "cinematic": 6.0}.get(s.key, 6.0)

    # --- claim_safety: based on claim checks (never invents safety)
    unverified = sum(1 for c in checks if c.status == ClaimStatus.UNVERIFIED_CLAIM)
    prohibited = sum(1 for c in checks if c.status == ClaimStatus.PROHIBITED)
    claim_safety = 10.0 - 2.0 * unverified - 4.0 * prohibited
    if not checks:
        claim_safety = 9.0  # no claims made -> safe but unproven

    # --- platform_fit
    if short_form:
        pf = {"ugc": 9.5, "direct_response": 9.0, "problem_solution": 8.5, "product_demo": 8.0,
              "testimonial": 7.0, "emotional": 6.5, "cinematic": 6.0}.get(s.key, 6.0)
    elif brief.platform == Platform.YOUTUBE:
        pf = {"cinematic": 9.5, "emotional": 8.5, "testimonial": 8.5, "product_demo": 8.0,
              "problem_solution": 7.5, "direct_response": 7.0, "ugc": 6.5}.get(s.key, 6.0)
    else:
        pf = 7.0

    dims = [
        DimensionScore("hook_strength", _clamp(hook_score), "Hook presence + short-form punchiness."),
        DimensionScore("clarity", _clamp(clarity), "How clearly the message lands."),
        DimensionScore("product_visibility", _clamp(product_vis), "How visibly the product is shown."),
        DimensionScore("differentiation", _clamp(diff), "Visual/camera distinctiveness vs other variants."),
        DimensionScore("pacing", _clamp(pacing), "Pacing fit to platform & duration."),
        DimensionScore("cta_clarity", _clamp(cta_score), "Clarity & prominence of the call to action."),
        DimensionScore("audience_relevance", _clamp(aud), "Relatability to the target audience."),
        DimensionScore("claim_safety", _clamp(claim_safety), f"{unverified} unverified, {prohibited} prohibited claims."),
        DimensionScore("platform_fit", _clamp(pf), "Fit to the requested platform."),
    ]
    notes: list[str] = []
    if unverified:
        notes.append(f"{unverified} unverified claim(s) detected — verify before publishing.")
    if prohibited:
        notes.append(f"{prohibited} prohibited claim(s) detected — must be removed.")
    return CreativeScore(variant=s.key, label=s.label, dimensions=dims, notes=notes)


# ---------------------------------------------------------------- comparison

@dataclass
class VariantComparison:
    """Heuristic comparison across generated variants."""

    strongest_hook: str
    strongest_cta: str
    most_cinematic: str
    most_direct_response: str
    highest_claim_risk: str
    most_platform_appropriate: str
    most_differentiated: str
    ranking: list[dict] = field(default_factory=list)  # sorted by total

    def to_dict(self) -> dict:
        return {
            "strongest_hook": self.strongest_hook,
            "strongest_cta": self.strongest_cta,
            "most_cinematic": self.most_cinematic,
            "most_direct_response": self.most_direct_response,
            "highest_claim_risk": self.highest_claim_risk,
            "most_platform_appropriate": self.most_platform_appropriate,
            "most_differentiated": self.most_differentiated,
            "ranking": self.ranking,
        }


def _dim(score: CreativeScore, name: str) -> float:
    return next((d.score for d in score.dimensions if d.dimension == name), 0.0)


def compare_variants(results: list[VariantResult], brief: AdBrief) -> VariantComparison:
    """Compare variants heuristically. All judgments are heuristic, not performance predictions."""
    scores = [score_variant(r, brief) for r in results]
    by_var = {s.variant: s for s in scores}

    def _argmax(dim_name):
        best, best_val = None, -1.0
        for s in scores:
            val = _dim(s, dim_name)
            if val > best_val:
                best, best_val = s.variant, val
        return best

    strongest_hook = _argmax("hook_strength")
    strongest_cta = _argmax("cta_clarity")
    # most cinematic is explicitly the cinematic variant if present
    most_cinematic = next((r.strategy.key for r in results if r.strategy.key == "cinematic"), _argmax("differentiation"))
    most_direct_response = next((r.strategy.key for r in results if r.strategy.key == "direct_response"), None)

    # highest claim risk = lowest claim_safety
    highest_claim_risk = min(scores, key=lambda s: _dim(s, "claim_safety")).variant if scores else None

    most_platform_appropriate = _argmax("platform_fit")
    most_differentiated = _argmax("differentiation")

    ranking = sorted(
        [{"variant": s.variant, "label": s.label, "total": s.total, "normalized": s.normalized} for s in scores],
        key=lambda x: x["total"], reverse=True,
    )
    return VariantComparison(
        strongest_hook=strongest_hook,
        strongest_cta=strongest_cta,
        most_cinematic=most_cinematic,
        most_direct_response=most_direct_response,
        highest_claim_risk=highest_claim_risk,
        most_platform_appropriate=most_platform_appropriate,
        most_differentiated=most_differentiated,
        ranking=ranking,
    )

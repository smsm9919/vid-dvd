"""Advertising brief and claim-safety schema (Phase 6).

An :class:`AdBrief` is the strongly-typed input to the advertising creative
engine. It deliberately separates what the brand has *approved* (facts, claims)
from what is *prohibited*, so the variant engine can never fabricate proof.

Claim safety is enforced by :mod:`app.ads.claims`: every creative idea that
implies proof, statistics, guarantees, testimonials, medical/financial/performance
claims must be traceable to an approved fact, or it is flagged
``UNVERIFIED_CLAIM`` / ``PROHIBITED`` — never silently emitted as fact.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from ..brain.models import BrandProfile, Objective, Platform


class ClaimStatus(str, Enum):
    """Safety classification of a creative statement about a product."""

    SUPPORTED_FACT = "supported_fact"          # traceable to an approved claim/fact
    CREATIVE_INTERPRETATION = "creative_interpretation"  # subjective framing, no proof implied
    UNVERIFIED_CLAIM = "unverified_claim"      # implies proof not supplied; needs verification
    PROHIBITED = "prohibited"                  # explicitly prohibited or illegal claim type


class FactType(str, Enum):
    """Kind of evidence a product fact represents."""

    FEATURE = "feature"
    BENEFIT = "benefit"
    STATISTIC = "statistic"
    TESTIMONIAL = "testimonial"
    ENDORSEMENT = "endorsement"
    DEMONSTRATION = "demonstration"
    OTHER = "other"


class ProductFact(BaseModel):
    """A user-supplied, approved piece of product evidence.

    Only facts supplied here may be used as proof in variants. The engine never
    invents new facts.
    """

    text: str
    fact_type: FactType = FactType.FEATURE
    source: Optional[str] = Field(None, description="Where this fact comes from (study, review, spec).")


class AdBrief(BaseModel):
    """The advertising brief — input to the creative variant engine."""

    product_or_service: str
    brand: Optional[BrandProfile] = None
    target_audience: str
    market: Optional[str] = None
    language: str = "en"
    platform: Platform = Platform.TIKTOK
    duration_seconds: float = Field(15.0, gt=0, le=300)
    objective: Objective = Objective.CONVERSION
    offer: Optional[str] = None
    cta: Optional[str] = None
    product_facts: list[ProductFact] = Field(default_factory=list)
    approved_claims: list[str] = Field(
        default_factory=list,
        description="Exact claim strings the brand has approved for use as proof.",
    )
    prohibited_claims: list[str] = Field(
        default_factory=list,
        description="Claims the brand explicitly forbids (legal/brand-safety).",
    )
    reference_material: Optional[str] = None

    @model_validator(mode="after")
    def _product_required(self) -> "AdBrief":
        if not self.product_or_service.strip():
            raise ValueError("AdBrief requires a product_or_service.")
        if not self.target_audience.strip():
            raise ValueError("AdBrief requires a target_audience.")
        return self

    @property
    def effective_cta(self) -> str:
        if self.cta and self.cta.strip():
            return self.cta.strip()
        if self.brand and self.brand.cta:
            return self.brand.cta.strip()
        # Platform-appropriate default CTA (never a fabricated offer).
        if self.platform in (Platform.TIKTOK, Platform.INSTAGRAM_REELS):
            return "Shop now — link in bio."
        return "Learn more today."

    @property
    def all_approved_claims(self) -> list[str]:
        """Approved claims from both the explicit list and product facts."""
        claims = list(self.approved_claims)
        for f in self.product_facts:
            claims.append(f.text)
        return claims

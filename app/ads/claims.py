"""Claim-safety engine (Phase 6).

Classifies creative statements about a product against the approved/prohibited
claims in an :class:`AdBrief`. The engine NEVER fabricates proof: if a statement
implies evidence that was not supplied, it is flagged
``UNVERIFIED_CLAIM`` (or ``PROHIBITED`` for illegal/forbidden categories) and a
safer alternative is suggested.

Forbidden claim patterns (never emitted as fact unless explicitly approved):
    - clinical/medical proof ("clinically proven", "cures", "treats")
    - guarantees ("guaranteed results", "money-back", "risk-free")
    - absolutes/superlatives ("best in the market", "#1", "world's leading")
    - instant/performance ("works instantly", "results in 24 hours")
    - fabricated social proof ("customers love it", "millions sold")
    - financial claims ("double your money", "risk-free investment")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .brief import AdBrief, ClaimStatus


# Each pattern maps to a human-readable reason and a safer framing suggestion.
_FORBIDDEN: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"clinically proven|clinically-tested|fda[- ]?approved|cures?|treats?|heals?", re.I),
     "Medical/clinical claim", "Show the product in use; avoid medical efficacy language unless approved."),
    (re.compile(r"guaranteed|money[- ]?back|risk[- ]?free|no risk", re.I),
     "Guarantee claim", "Use 'designed to' / 'made to' framing instead of guarantees."),
    (re.compile(r"best (in|seller|selling)|#1|world'?s leading|top[- ]?rated|unbeatable|nothing beats", re.I),
     "Absolute/superlative claim", "Describe a specific feature rather than claiming market supremacy."),
    (re.compile(r"works instantly|instant results|results? in \d+|overnight|immediate results", re.I),
     "Instant/performance claim", "Show a realistic demonstration timeline instead of instant results."),
    (re.compile(r"customers love it|millions sold|everyone (loves|uses)|loved by (thousands|millions)", re.I),
     "Fabricated social proof", "Quote a specific approved testimonial, or omit social proof."),
    (re.compile(r"double your money|risk[- ]?free investment|get rich|guaranteed return", re.I),
     "Financial claim", "Avoid financial-outcome promises entirely."),
]


@dataclass
class ClaimAssessment:
    """Result of classifying one creative statement."""

    text: str
    status: ClaimStatus
    reason: str = ""
    matched_approved: Optional[str] = None
    safer_alternative: Optional[str] = None
    requires_verification: bool = False

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "status": self.status.value,
            "reason": self.reason,
            "matched_approved": self.matched_approved,
            "safer_alternative": self.safer_alternative,
            "requires_verification": self.requires_verification,
        }


def _is_approved(text: str, approved: list[str]) -> Optional[str]:
    """Return the approved claim if it matches the text in either direction.

    Matches when the approved claim is a substring of the text OR the text is a
    substring of the approved claim (so a shorter creative line still counts as
    covered by a longer approved claim).
    """
    t = text.lower()
    for a in approved:
        if not a:
            continue
        al = a.lower()
        if al in t or t in al:
            return a
    return None


def _is_prohibited(text: str, prohibited: list[str]) -> Optional[str]:
    t = text.lower()
    for p in prohibited:
        if p and p.lower() in t:
            return p
    return None


def classify_claim(text: str, brief: AdBrief) -> ClaimAssessment:
    """Classify a single creative statement for claim safety."""
    if not text or not text.strip():
        return ClaimAssessment(text=text, status=ClaimStatus.CREATIVE_INTERPRETATION, reason="Empty/non-claim text.")

    # 1. Explicitly prohibited by the brand.
    hit = _is_prohibited(text, brief.prohibited_claims)
    if hit:
        return ClaimAssessment(
            text=text, status=ClaimStatus.PROHIBITED,
            reason=f"Matches a prohibited claim: '{hit}'.",
            safer_alternative="Remove this claim; it is explicitly prohibited by the brand.",
        )

    # 2. Forbidden category pattern.
    for pat, reason, safer in _FORBIDDEN:
        if pat.search(text):
            # Even forbidden patterns are allowed ONLY if explicitly approved verbatim.
            approved_hit = _is_approved(text, brief.all_approved_claims)
            if approved_hit:
                return ClaimAssessment(
                    text=text, status=ClaimStatus.SUPPORTED_FACT,
                    reason=f"Approved claim covers a normally-restricted phrase: '{approved_hit}'.",
                    matched_approved=approved_hit,
                )
            return ClaimAssessment(
                text=text, status=ClaimStatus.UNVERIFIED_CLAIM,
                reason=f"{reason} — no approved evidence supplied.",
                safer_alternative=safer,
                requires_verification=True,
            )

    # 3. Traceable to an approved claim/fact.
    approved_hit = _is_approved(text, brief.all_approved_claims)
    if approved_hit:
        return ClaimAssessment(
            text=text, status=ClaimStatus.SUPPORTED_FACT,
            reason=f"Traceable to approved claim: '{approved_hit}'.",
            matched_approved=approved_hit,
        )

    # 4. Proof-implying keywords without supplied evidence.
    if re.search(r"\b(proven|evidence|studies? show|data shows|results show|tested|certified)\b", text, re.I):
        return ClaimAssessment(
            text=text, status=ClaimStatus.UNVERIFIED_CLAIM,
            reason="Implies proof/evidence not present in approved claims.",
            safer_alternative="Cite a specific approved fact, or soften to 'designed to'.",
            requires_verification=True,
        )

    # 5. Default: subjective creative framing with no proof implied.
    return ClaimAssessment(
        text=text, status=ClaimStatus.CREATIVE_INTERPRETATION,
        reason="Subjective/creative framing; no proof implied.",
    )


def assess_lines(lines: list[str], brief: AdBrief) -> list[ClaimAssessment]:
    """Classify a list of creative lines (VO/captions/proof text)."""
    return [classify_claim(line, brief) for line in lines if line and line.strip()]


def safe_proof_for(brief: AdBrief) -> tuple[Optional[str], list[str]]:
    """Return (proof_text_or_None, warnings).

    If the brief supplies an approved fact suitable as proof, return it.
    Otherwise return None and a warning explaining no proof was fabricated.
    """
    warnings: list[str] = []
    # Prefer a statistic/testimonial/demonstration fact, else any fact.
    preferred = [f for f in brief.product_facts if f.fact_type.value in ("statistic", "testimonial", "demonstration")]
    pool = preferred or brief.product_facts
    if pool:
        return pool[0].text, warnings
    warnings.append(
        "No approved proof supplied. Proof scene marked 'requires verification' — no claim fabricated."
    )
    return None, warnings

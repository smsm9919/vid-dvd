"""Voiceover generation and timing (Phase 9).

Generates voice-over from :class:`~app.brain.models.ProductionPlan.voiceover`
and each :class:`~app.brain.models.Scene.voiceover`, maintaining exact scene
association. Each voice asset carries structured metadata.

Timing validation ensures voice duration fits within its scene (considering
``start_offset``), detects overlong voice without silently truncating, and
returns a safe adjustment strategy.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from ..brain.models import ProductionPlan, Scene
from ..config import OUTPUT_DIR
from ..core.errors import TypedErrorCode, VideoError
from ..audio.qc import verify_audio
from .tts import TTSError, TTSProvider, VoiceRequest, VoiceResult


class TimingSeverity(str, Enum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class VoiceTimingReport:
    scene_index: int
    scene_duration: float
    start_offset: float
    voice_duration: float
    end_time: float
    severity: TimingSeverity
    slack: float  # scene_duration - end_time
    message: str
    strategy: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_index": self.scene_index, "scene_duration": self.scene_duration,
            "start_offset": self.start_offset, "voice_duration": self.voice_duration,
            "end_time": self.end_time, "severity": self.severity.value,
            "slack": self.slack, "message": self.message, "strategy": self.strategy,
        }


@dataclass
class VoiceAsset:
    """A generated voice asset with full metadata."""

    project_id: Optional[str]
    scene_index: int
    language: str
    voice: Optional[str]
    text: str
    duration: float
    provider: str
    path: str
    sample_rate: int = 44100
    start_offset: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id, "scene_index": self.scene_index,
            "language": self.language, "voice": self.voice, "text": self.text,
            "duration": self.duration, "provider": self.provider, "path": self.path,
            "sample_rate": self.sample_rate, "start_offset": self.start_offset,
        }


def validate_voice_timing(scene: Scene, voice_duration: float) -> VoiceTimingReport:
    """Validate voice duration vs scene duration + start_offset.

    Does NOT silently truncate. Returns a structured report with severity and a
    safe adjustment strategy when the voice is too long.
    """
    sd = scene.duration
    off = scene.voiceover.start_offset
    end = off + voice_duration
    slack = sd - end
    if end <= sd:
        sev = TimingSeverity.OK
        msg = f"Voice fits within scene ({voice_duration:.2f}s + {off:.2f}s offset <= {sd:.2f}s)."
        strategy = None
    elif off > 0 and voice_duration <= sd:
        # Voice fits in scene if started at 0; offset is the problem.
        sev = TimingSeverity.WARNING
        msg = (f"Voice ({voice_duration:.2f}s) fits scene ({sd:.2f}s) only if start_offset "
               f"is reduced from {off:.2f}s to 0.")
        strategy = "reduce start_offset to 0.0"
    elif voice_duration > sd:
        sev = TimingSeverity.ERROR
        over = voice_duration - sd
        msg = (f"Voice ({voice_duration:.2f}s) exceeds scene ({sd:.2f}s) by {over:.2f}s. "
               f"Do NOT silently truncate.")
        strategy = (f"shorten the script by ~{over:.2f}s, increase scene duration to "
                    f"{voice_duration:.2f}s, or raise TTS rate to "
                    f"{min(voice_duration / sd, 4.0):.2f}x")
    else:
        sev = TimingSeverity.WARNING
        msg = f"Voice end time {end:.2f}s slightly exceeds scene {sd:.2f}s."
        strategy = "reduce start_offset or shorten script"
    return VoiceTimingReport(
        scene_index=scene.index, scene_duration=sd, start_offset=off,
        voice_duration=voice_duration, end_time=end, severity=sev, slack=slack,
        message=msg, strategy=strategy,
    )


def validate_project_voice_timing(plan: ProductionPlan,
                                  durations: dict[int, float]) -> list[VoiceTimingReport]:
    """Validate timing across all scenes. ``durations`` maps scene_index -> voice duration."""
    reports: list[VoiceTimingReport] = []
    for scene in plan.scenes:
        vd = durations.get(scene.index, 0.0)
        reports.append(validate_voice_timing(scene, vd))
    return reports


def build_voice_request(scene: Scene, plan: ProductionPlan) -> VoiceRequest:
    """Build a TTS request from the scene + plan voiceover, propagating language."""
    vo = scene.voiceover
    vplan = plan.voiceover
    language = vplan.language or "en"
    gender = vplan.voice_gender
    # rate from pace hint (non-numeric pace ignored).
    rate = 1.0
    if vplan.pace:
        pace_map = {"slow": 0.8, "normal": 1.0, "fast": 1.3, "fast_slow": 0.8}
        rate = pace_map.get(vplan.pace.lower(), 1.0)
    return VoiceRequest(
        text=vo.line, language=language, gender=gender,
        rate=rate, pitch=1.0, emotion=vo.direction, direction=vplan.direction,
        output_format="mp3",
    )


def generate_scene_voiceover(
    scene: Scene, plan: ProductionPlan, provider: TTSProvider,
    *, project_id: Optional[str] = None, output_dir: Optional[Path] = None,
) -> tuple[VoiceAsset, VoiceResult]:
    """Generate voiceover for one scene.

    Returns (asset, result). The result.path is a real, verified audio file.
    Raises TTSError / VideoError on failure — never fake success.
    """
    request = build_voice_request(scene, plan)
    request.validate()
    out_dir = Path(output_dir) if output_dir else _voice_dir(project_id, scene.index)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"scene_{scene.index}_voice.{request.output_format}"
    result = provider.synthesize(request, dest)
    # QC: verify the audio is real and decodable before reporting success.
    qc = verify_audio(result.path)
    asset = VoiceAsset(
        project_id=project_id, scene_index=scene.index, language=result.language,
        voice=result.voice, text=result.text, duration=qc["duration"],
        provider=result.provider, path=str(result.path), sample_rate=qc["sample_rate"],
        start_offset=scene.voiceover.start_offset,
    )
    return asset, result


def generate_project_voiceover(
    plan: ProductionPlan, provider: TTSProvider,
    *, project_id: Optional[str] = None, output_dir: Optional[Path] = None,
) -> tuple[list[VoiceAsset], list[VoiceTimingReport]]:
    """Generate voiceover for all scenes in a plan.

    Scenes with empty voiceover lines are skipped. Returns (assets, timing).
    """
    assets: list[VoiceAsset] = []
    timing: list[VoiceTimingReport] = []
    for scene in plan.scenes:
        if not scene.voiceover.line.strip():
            continue
        asset, result = generate_scene_voiceover(
            scene, plan, provider, project_id=project_id, output_dir=output_dir,
        )
        assets.append(asset)
        timing.append(validate_voice_timing(scene, asset.duration))
    return assets, timing


def _voice_dir(project_id: Optional[str], scene_index: int) -> Path:
    run_id = uuid.uuid4().hex[:12]
    return Path(OUTPUT_DIR) / (project_id or "standalone") / f"scene_{scene_index}" / run_id / "voice"

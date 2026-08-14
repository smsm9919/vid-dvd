import json, re
from .models import Scene

STYLE = (
    "ultra-realistic cinematic wildlife film, professional cinema camera, "
    "physically plausible movement, realistic animal anatomy, dramatic lighting, "
    "volumetric atmosphere, shallow depth of field, detailed fur and scales, "
    "high dynamic range, cinematic color grading, vertical 9:16 composition"
)

def local_scene_plan(topic, script, duration, scene_count):
    text = re.sub(r"\s+", " ", (script or "").strip())
    if text:
        parts = re.split(r"(?<=[.!?])\s+", text)
    else:
        parts = [
            f"Establish the environment and introduce the subjects: {topic}.",
            f"Build suspense around {topic} with a dramatic reveal.",
            f"Escalate the action connected to {topic}.",
            f"Show the visual climax of {topic} with dynamic movement.",
            f"End with a memorable cinematic final image related to {topic}."
        ]
    parts = parts[:scene_count]
    if len(parts) < scene_count:
        parts += [f"Continue the cinematic story about {topic}."] * (scene_count-len(parts))
    each = round(duration/scene_count, 2)
    return [Scene(index=i+1, duration=each, description=desc,
                  prompt=f"{desc} {STYLE}. Dynamic camera, coherent subject identity, realistic motion, no text.",
                  negative_prompt="blurry, distorted anatomy, extra limbs, duplicate subjects, deformed face, text, logo, watermark, low quality")
            for i, desc in enumerate(parts)]

async def gemini_scene_plan(topic, script, duration, scene_count, api_key, model):
    import httpx
    url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    prompt=f'''Create a production-ready shot list for a {duration}-second vertical 9:16 cinematic video.
Topic: {topic}
Script: {script or "(write a short original script)"}
Return ONLY valid JSON:
{{"scenes":[{{"duration":number,"description":"...","prompt":"...","negative_prompt":"..."}}]}}
Need exactly {scene_count} scenes. Make prompts concrete and cinematic: subject, action, environment, camera, lighting and motion.'''
    async with httpx.AsyncClient(timeout=90) as c:
        r=await c.post(url, params={"key":api_key}, json={"contents":[{"parts":[{"text":prompt}]}]})
        r.raise_for_status()
        data=r.json()
    text=data["candidates"][0]["content"]["parts"][0]["text"].strip()
    text=re.sub(r"^```json\s*|\s*```$", "", text)
    payload=json.loads(text)
    return [Scene(index=i+1, duration=float(s["duration"]), description=s["description"],
                  prompt=s["prompt"], negative_prompt=s.get("negative_prompt","blurry, distorted anatomy, text, watermark, logo"))
            for i,s in enumerate(payload["scenes"])]

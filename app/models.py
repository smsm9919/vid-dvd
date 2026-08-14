from pydantic import BaseModel, Field

class Scene(BaseModel):
    index: int
    duration: float
    description: str
    prompt: str
    negative_prompt: str

class ProjectCreate(BaseModel):
    title: str = "Cinematic Short"
    topic: str = ""
    script: str = ""
    language: str = "en"
    duration: int = Field(default=30, ge=5, le=120)
    scene_count: int = Field(default=5, ge=1, le=12)

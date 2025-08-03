from typing import Optional
from pydantic import BaseModel
from app.be.core.config import settings

class AIModelParameters(BaseModel):
    temperature: Optional[float] = settings.temperature
    max_tokens: Optional[int] = settings.max_tokens
    top_p: Optional[float] = settings.top_p
    frequency_penalty: Optional[float] = settings.frequency_penalty
    presence_penalty: Optional[float] = settings.presence_penalty
    
class InferencePayload(BaseModel):
    query: str 
    history: Optional[str] = None
    ai_model_parameters: Optional[AIModelParameters] = None

class InferenceResponse(BaseModel):
    response: str


from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class QueryRequest(BaseModel):
    question: str
    max_retries: Optional[int] = 2

class QueryResponse(BaseModel):
    answer: str
    quality: str
    metrics: Dict
    history: List[Dict]
    sources: Optional[List[str]] = []

class HealthResponse(BaseModel):
    status: str
    engine: str
    model: str
    features: List[str]
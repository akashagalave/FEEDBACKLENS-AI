from pydantic import BaseModel
from typing import Optional


class UnderstandRequest(BaseModel):
    query: str
    company: Optional[str] = None


class UnderstandResponse(BaseModel):
    company: str
    intent: str
    focus: Optional[str] = None
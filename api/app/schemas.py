from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    texts: List[str] = Field(..., description="Lista de textos EN (ej. secciones del PDF)")
    engine: Optional[Literal["opus", "nllb"]] = Field(default=None, description="Motor a usar")
    batch_size: int = Field(default=8, ge=1, le=64)


class TranslateResponse(BaseModel):
    engine: str
    translations: List[str]

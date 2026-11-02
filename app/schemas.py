from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)


class DocumentOut(BaseModel):
    id: int
    title: str
    chunks: int


class AskIn(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=4, ge=1, le=20)


class SourceOut(BaseModel):
    document_id: int
    position: int
    text: str
    distance: float


class AskOut(BaseModel):
    answer: str
    sources: list[SourceOut]

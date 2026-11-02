from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Split by characters with overlap, so a sentence cut at the border appears whole in one chunk."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


async def retrieve(session: AsyncSession, question_vector: list[float], k: int):
    """Nearest chunks by cosine distance (pgvector). Smaller distance means closer."""
    distance = Chunk.embedding.cosine_distance(question_vector).label("distance")
    result = await session.execute(select(Chunk, distance).order_by(distance).limit(k))
    return result.all()


def build_prompt(question: str, sources) -> str:
    context = "\n\n".join(f"[{i + 1}] {chunk.text}" for i, (chunk, _) in enumerate(sources))
    return (
        "Answer the question using only the context below. "
        "If the context does not contain the answer, say you do not know. "
        "Cite the sources like [1].\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

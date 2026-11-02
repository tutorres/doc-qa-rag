from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_session
from app.embeddings import Embedder, SentenceTransformerEmbedder
from app.llm import LLM, OpenAICompatibleLLM, LLMError
from app.models import Chunk, Document
from app.rag import build_prompt, chunk_text, retrieve
from app.schemas import AskIn, AskOut, DocumentIn, DocumentOut, SourceOut

app = FastAPI(title="doc-qa")


@lru_cache
def get_embedder() -> Embedder:
    return SentenceTransformerEmbedder(settings.embedding_model)


@lru_cache
def get_llm() -> LLM:
    return OpenAICompatibleLLM(settings.llm_api_key, settings.llm_base_url, settings.llm_model)


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentOut, status_code=201)
async def create_document(
    body: DocumentIn,
    session: AsyncSession = Depends(get_session),
    embedder: Embedder = Depends(get_embedder),
):
    pieces = chunk_text(body.content)
    vectors = await embedder.embed(pieces)
    document = Document(
        title=body.title,
        content=body.content,
        chunks=[Chunk(position=i, text=t, embedding=v) for i, (t, v) in enumerate(zip(pieces, vectors))],
    )
    session.add(document)
    await session.commit()
    return DocumentOut(id=document.id, title=document.title, chunks=len(pieces))


@app.post("/ask", response_model=AskOut)
async def ask(
    body: AskIn,
    session: AsyncSession = Depends(get_session),
    embedder: Embedder = Depends(get_embedder),
    llm: LLM = Depends(get_llm),
):
    question_vector = (await embedder.embed([body.question]))[0]
    sources = await retrieve(session, question_vector, body.k)
    if not sources:
        raise HTTPException(status_code=404, detail="No documents indexed yet")
    try:
        answer = await llm.complete(build_prompt(body.question, sources))
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return AskOut(
        answer=answer,
        sources=[
            SourceOut(document_id=c.document_id, position=c.position, text=c.text, distance=float(d))
            for c, d in sources
        ],
    )

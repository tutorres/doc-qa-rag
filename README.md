# doc-qa-rag

Small document Q&A service, built as a study project: async FastAPI, SQLAlchemy 2.0 (async) with Alembic migrations, Postgres with pgvector, and a simple RAG flow (chunk, embed, retrieve, generate with sources).

## How it works

- `POST /documents` receives plain text, splits it into overlapping chunks, embeds each chunk and stores everything in one transaction.
- `POST /ask` embeds the question, fetches the k nearest chunks with pgvector (cosine distance), builds a prompt with that context and asks an LLM. The answer comes back with the sources used.

## Stack

- FastAPI + Pydantic
- SQLAlchemy 2.0 async (asyncpg) and Alembic
- Postgres 16 with pgvector
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384 dims), run in a thread so the event loop is not blocked
- LLM: any provider with an OpenAI compatible chat API (Groq by default, xAI or OpenAI work by changing the URL and model), behind an `LLM` interface

## Run

```
docker compose up -d
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        # then fill LLM_API_KEY and LLM_MODEL
alembic upgrade head
uvicorn app.main:app --reload
```

## Tests

Tests use a real Postgres (`docqa_test` database) with a fake embedder and a fake LLM, so they need no network or API key.

```
docker compose exec db psql -U docqa -d docqa -c "CREATE DATABASE docqa_test"
pytest
```

## Design notes

- Embedder and LLM sit behind small interfaces and are injected with FastAPI dependencies, so tests swap them.
- A document and its chunks are saved in a single transaction: either everything is stored or nothing.
- The `vector` extension is enabled in the first Alembic migration.

## Known limits

- Chunking is by characters, not by sentences or tokens.
- No auth, no pagination, no reranking, no evaluation of answer quality.
- Documents are plain text only (no PDF extraction).
- No background queue: indexing happens inside the request.

import asyncio
import hashlib
import math
from typing import Protocol

DIM = 384


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Real embedder. The model runs on CPU, so it goes to a thread to not block the event loop."""

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.get_running_loop().run_in_executor(None, self._encode, texts)


class FakeEmbedder:
    """Deterministic bag-of-words vectors for tests: texts sharing words end up close."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        vec = [0.0] * DIM
        for word in text.lower().split():
            word = word.strip(".,?!;:")
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

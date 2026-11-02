from typing import Protocol

import httpx


class LLMError(Exception):
    pass


class LLM(Protocol):
    async def complete(self, prompt: str) -> str: ...


class OpenAICompatibleLLM:
    """Any provider with an OpenAI compatible chat completions endpoint (Groq, xAI, OpenAI...)."""

    def __init__(self, api_key: str, base_url: str, model: str):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model

    async def complete(self, prompt: str) -> str:
        if not self._api_key or not self._model:
            raise LLMError("LLM_API_KEY and LLM_MODEL must be set in .env")
        payload = {"model": self._model, "messages": [{"role": "user", "content": prompt}]}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(f"{self._base_url}/chat/completions", json=payload, headers=headers)
        if response.status_code != 200:
            raise LLMError(f"LLM returned status {response.status_code}")
        return response.json()["choices"][0]["message"]["content"]


class FakeLLM:
    """Echoes the prompt so tests can check that the retrieved context was sent."""

    async def complete(self, prompt: str) -> str:
        return f"FAKE ANSWER based on: {prompt}"

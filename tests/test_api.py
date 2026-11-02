async def test_create_document_returns_chunk_count(client):
    body = {"title": "manual", "content": "word " * 400}  # 2000 chars
    response = await client.post("/documents", json=body)
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "manual"
    assert data["chunks"] == 3


async def test_empty_content_is_rejected(client):
    response = await client.post("/documents", json={"title": "x", "content": ""})
    assert response.status_code == 422


async def test_ask_without_documents_is_404(client):
    response = await client.post("/ask", json={"question": "anything"})
    assert response.status_code == 404


async def test_ask_retrieves_the_most_relevant_chunk(client):
    await client.post("/documents", json={"title": "it", "content": "To reset your password open the settings page and click reset."})
    await client.post("/documents", json={"title": "pets", "content": "The cat sleeps on the sofa all afternoon."})

    response = await client.post("/ask", json={"question": "How do I reset my password?", "k": 1})
    assert response.status_code == 200
    data = response.json()
    assert len(data["sources"]) == 1
    assert "reset your password" in data["sources"][0]["text"]
    assert "reset your password" in data["answer"]  # the context reached the LLM

from fastapi import FastAPI

app = FastAPI(title="doc-qa")


@app.get("/health")
async def health():
    return {"status": "ok"}

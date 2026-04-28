from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes.welfare import router
from src.embedding.kosimcse import KoSimCSEEmbedder


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.embedder = KoSimCSEEmbedder()
    yield


app = FastAPI(title="BokjiDream RAG API", version="0.1.0", lifespan=lifespan)
app.include_router(router)

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.config import resolve_api_config
from src.api.routes.welfare import router
from src.db.chroma import validate_existing_collection
from src.embedding.kosimcse import KoSimCSEEmbedder


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    api_config = resolve_api_config()
    app.state.api_config = api_config
    await validate_existing_collection(
        api_config.collection_name,
        require_section_metadata=api_config.requires_section_metadata,
    )
    app.state.embedder = KoSimCSEEmbedder()
    yield


app = FastAPI(title="BokjiDream RAG API", version="0.1.0", lifespan=lifespan)
app.include_router(router)

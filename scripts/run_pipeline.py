#!/usr/bin/env python3
"""전체 수집 → 인덱싱 파이프라인 실행.

Usage:
    PUBLIC_DATA_API_KEY=<key> python scripts/run_pipeline.py
    CHROMA_MODE=ephemeral PUBLIC_DATA_API_KEY=dummy python scripts/run_pipeline.py  # 스모크 테스트
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 — `python scripts/run_pipeline.py` 직접 실행 지원
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.crawler.collect import collect_all  # noqa: E402
from src.embedding.kosimcse import KoSimCSEEmbedder  # noqa: E402
from src.pipeline.index import index_welfare_items  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    embedder = KoSimCSEEmbedder()
    logger.info("수집 시작")
    items = await collect_all()
    logger.info("수집 완료: %d건", len(items))
    count = await index_welfare_items(items, embedder, tokenizer=embedder.tokenizer)
    logger.info("인덱싱 완료: %d 청크", count)


if __name__ == "__main__":
    asyncio.run(main())

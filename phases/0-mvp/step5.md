# Step 5: pipeline

## 읽어야 할 파일

먼저 아래 파일들을 읽고 설계 의도를 파악하라:

- `docs/ARCHITECTURE.md`
- `docs/ADR.md` (ADR-001, ADR-007)
- `src/models/welfare.py` (WelfareRaw 확인)
- `src/db/chroma.py` (컬렉션 이름, get_collection 확인)
- `src/embedding/__init__.py` (EmbedderProtocol 확인)
- `phases/0-mvp/index.json` (step 4 summary 확인)

## 작업

`WelfareRaw` 리스트를 받아 청크 분할 → 임베딩 → ChromaDB upsert하는 파이프라인을 구현한다.

### 구현할 파일

#### 1. `src/pipeline/chunker.py` — 문서 청크 분할

```python
from __future__ import annotations
from src.models.welfare import WelfareRaw

CHUNK_SIZE = 1000      # 글자 수
CHUNK_OVERLAP = 200    # 겹침 글자 수

def make_document_text(item: WelfareRaw) -> str:
    """WelfareRaw의 주요 텍스트 필드를 하나의 문서로 조합."""

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """텍스트를 청크 리스트로 분할."""
```

- `make_document_text`는 아래 순서로 `\n\n`을 구분자로 이어붙임:
  ```python
  return "\n\n".join(filter(None, [
      item.serv_nm,
      item.serv_dgst,
      item.tgtr_dtl_cn,
      item.slct_crit_cn,
      item.alw_serv_cn,
  ]))
  ```
  `\n\n` 구분자를 쓰는 이유: 필드 경계가 청크 분할 경계로 자연스럽게 작동하고,
  임베딩 시 각 필드의 의미가 유지된다. 단순 이어붙이면 "...수급자입니다선정기준은..."
  처럼 의미가 끊겨 검색 품질이 저하된다.
- 청크 분할은 **순수 글자 수 기준 슬라이딩 윈도우** 방식으로 구현한다:
  ```python
  def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
      if len(text) <= chunk_size:
          return [text] if text else []
      chunks = []
      start = 0
      while start < len(text):
          end = start + chunk_size
          chunks.append(text[start:end])
          start += chunk_size - overlap
      return chunks
  ```
  한국어 복지 문서는 `。`, `.\n` 빈도가 낮고 문장 경계 탐지가 불안정하므로
  MVP에서는 순수 글자 수 기준을 사용한다. 오버랩(200자, 20%)이 문맥 단절을 보완한다.

#### 2. `src/pipeline/index.py` — ChromaDB 인덱싱 진입점

```python
from __future__ import annotations
from src.models.welfare import WelfareRaw
from src.db.chroma import get_collection, WELFARE_COLLECTION
from src.embedding.protocol import EmbedderProtocol

async def index_welfare_items(
    items: list[WelfareRaw],
    embedder: EmbedderProtocol,
) -> int:
    """WelfareRaw 리스트를 청크 분할 → 임베딩 → ChromaDB upsert.
    
    Returns:
        upsert된 청크 수
    """
```

- 청크별 ID: `{serv_id}_chunk_{n}` 형식
- 메타데이터: `serv_id`, `serv_nm`, `serv_dgst`, `jur_mnof_nm`, `trgter_indvdl`(JSON), `intrs_thema`(JSON),
  `sprt_cyc_nm`, `srv_pvsn_nm`, `serv_dtl_link`, `tgtr_dtl_cn`, `slct_crit_cn`, `alw_serv_cn`
  (상세 텍스트 필드가 반드시 포함되어야 함 — 아래 메타데이터 스키마 참고)
- `embedder`를 DI로 받아 `EmbedderProtocol` 인터페이스만 사용 (구체 구현 분리)
- ChromaDB `upsert` 사용 (중복 실행 시 덮어쓰기)

### 메타데이터 스키마 (ChromaDB에 저장)

```python
{
    "serv_id": "WLF00000035",
    "serv_nm": "서비스명",
    "jur_mnof_nm": "국토교통부",
    "trgter_indvdl": '["저소득", "노인"]',   # JSON 문자열 (ChromaDB는 list 미지원)
    "intrs_thema": '["주거", "생활지원"]',    # JSON 문자열
    "sprt_cyc_nm": "년",
    "srv_pvsn_nm": "현금지급",
    "serv_dtl_link": "https://...",
    # 상세 텍스트 필드 — WelfareDetail 복원에 사용 (retriever/search.py 참고)
    "tgtr_dtl_cn": "수급 대상 상세 텍스트...",
    "slct_crit_cn": "선정 기준 텍스트...",
    "alw_serv_cn": "서비스 내용 텍스트...",
    "serv_dgst": "서비스 개요...",
}
```

- `trgter_indvdl`, `intrs_thema`는 `json.dumps()`로 직렬화하여 저장
- retriever에서 `json.loads()`로 역직렬화
- **`tgtr_dtl_cn`, `slct_crit_cn`, `alw_serv_cn`, `serv_dgst`를 메타데이터에 반드시 저장**:
  `make_document_text()`가 이 필드들을 이어붙인 청크를 생성하기 때문에
  나중에 청크 텍스트만으로 개별 필드를 복원하는 것은 불가능하다.
  retriever의 `get_welfare_detail()`은 메타데이터에서 직접 이 값을 읽는다.

### 주의사항

- `embedder`는 DI로 주입받는다 — `index.py`에서 `KoSimCSEEmbedder`를 직접 import하지 마라
- `embedding/` 레이어의 `EmbedderProtocol`만 참조
- 배치 처리: `embedder.embed()` 호출 시 청크를 한 번에 넘겨 효율화
- **chromadb 호출(`collection.upsert` 등)은 반드시 `asyncio.to_thread()`로 래핑** (ADR-008):
  ```python
  await asyncio.to_thread(collection.upsert, ids=ids, documents=docs, embeddings=embeds, metadatas=metas)
  ```

## Acceptance Criteria

```bash
mypy src/pipeline/                             # 타입 오류 없음
pytest tests/unit/test_chunker.py tests/unit/test_index.py -v  # 모든 테스트 통과
ruff check src/pipeline/                      # 린트 오류 없음
```

테스트 파일 두 개를 함께 작성한다:
- `tests/unit/test_chunker.py` — chunker 전용
- `tests/unit/test_index.py` — index 전용

최소한 아래 케이스를 커버해야 한다:

- `chunk_text()` CHUNK_SIZE 초과 텍스트 → 여러 청크 반환
- `chunk_text()` CHUNK_OVERLAP 크기만큼 겹침 검증
- `make_document_text()` 빈 상세 필드 있을 때 정상 동작
- `index_welfare_items()` mock embedder + mock ChromaDB로 upsert 호출 검증

### 파이프라인 진입점 스크립트 (`scripts/run_pipeline.py`)

step 5와 함께 아래 스크립트도 작성한다. ChromaDB를 실제로 채울 수 있는
유일한 진입점이며, 이 파일이 없으면 API 서버가 기동해도 검색 결과가 항상 빈 배열이다.

```python
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

from src.crawler.collect import collect_all
from src.embedding.kosimcse import KoSimCSEEmbedder
from src.pipeline.index import index_welfare_items

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    embedder = KoSimCSEEmbedder()
    logger.info("수집 시작")
    items = await collect_all()
    logger.info("수집 완료: %d건", len(items))
    count = await index_welfare_items(items, embedder)
    logger.info("인덱싱 완료: %d 청크", count)


if __name__ == "__main__":
    asyncio.run(main())
```

이 파일은 `src/pipeline/` 구현이 완료된 직후에 작성하며,
`scripts/execute.py` 하네스는 이 파일을 AC 커맨드로 실행하지 않는다
(외부 API 키 필요). 배포 후 별도로 실행한다.

## 검증 절차

1. 위 AC 커맨드를 순서대로 실행한다.
2. 아키텍처 체크리스트:
   - `index.py`가 `EmbedderProtocol`만 참조하고 `KoSimCSEEmbedder`를 직접 import하지 않는가?
   - 메타데이터에 `trgter_indvdl`, `intrs_thema`가 JSON 문자열로 저장되는가?
   - 청크 ID 형식이 `{serv_id}_chunk_{n}`인가?
   - `scripts/run_pipeline.py`가 작성되었는가?
3. 결과에 따라 `phases/0-mvp/index.json`의 step 5를 업데이트한다:
   - 성공 → `"status": "completed"`, `"summary": "src/pipeline/ 구현 완료 — chunker, index 함수 완료, scripts/run_pipeline.py 진입점 추가"`
   - 수정 3회 시도 후에도 실패 → `"status": "error"`, `"error_message": "구체적 에러 내용"`
   - 사용자 개입 필요 → `"status": "blocked"`, `"blocked_reason": "구체적 사유"` 후 즉시 중단

## 금지사항

- `pipeline/`에서 `api/`, `retriever/`, `crawler/`를 임포트하지 마라
- `index.py`에서 `KoSimCSEEmbedder`를 직접 인스턴스화하지 마라 (DI 원칙)
- ChromaDB `add()` 대신 반드시 `upsert()` 사용 (재실행 안전성)

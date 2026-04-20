# Step 3: embedding-layer

## 읽어야 할 파일

먼저 아래 파일들을 읽고 설계 의도를 파악하라:

- `docs/ARCHITECTURE.md`
- `docs/ADR.md` (ADR-002, ADR-007)
- `phases/0-mvp/index.json` (step 2 summary 확인)

## 작업

`src/embedding/` 레이어를 구현한다. `pipeline/`(인덱싱)과 `retriever/`(쿼리 임베딩) 양쪽에서
공유하는 **최하위 인프라 레이어**다. 다른 `src/` 모듈을 절대 임포트하지 않는다.

### 구현할 파일

#### 1. `src/embedding/protocol.py` — EmbedderProtocol

```python
from __future__ import annotations
from typing import Protocol, runtime_checkable

@runtime_checkable
class EmbedderProtocol(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """텍스트 리스트를 임베딩 벡터 리스트로 변환."""
        ...
```

- `runtime_checkable`로 선언하여 `isinstance()` 체크 가능하게 한다

#### 2. `src/embedding/kosimcse.py` — KoSimCSE 구체 구현체

```python
from __future__ import annotations
from sentence_transformers import SentenceTransformer
from src.embedding.protocol import EmbedderProtocol

MODEL_NAME = "jhgan/ko-sroberta-multitask"

class KoSimCSEEmbedder:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, convert_to_tensor=False)
        return [e.tolist() for e in embeddings]  # numpy array → list[float] (float32 아님)
```

- `EmbedderProtocol`을 명시적으로 `implements`하지 않아도 구조적 서브타이핑으로 동작
- 모델은 처음 사용 시 자동 다운로드 (~400MB). 이는 정상 동작

#### 3. `src/embedding/__init__.py` — 공개 인터페이스

```python
from src.embedding.protocol import EmbedderProtocol
from src.embedding.kosimcse import KoSimCSEEmbedder

__all__ = ["EmbedderProtocol", "KoSimCSEEmbedder"]
```

### 주의사항

- `embedding/` 레이어는 다른 `src/` 모듈을 임포트하지 않는다 (최하위 인프라)
- `sentence_transformers`가 `pyproject.toml`에 의존성으로 있는지 확인한다
- 테스트에서 실제 모델 다운로드를 피하려면 `EmbedderProtocol`을 구현하는 mock을 사용한다

## Acceptance Criteria

```bash
mypy src/embedding/                          # 타입 오류 없음
pytest tests/unit/test_embedder.py -v       # 모든 테스트 통과
ruff check src/embedding/                    # 린트 오류 없음
```

테스트 파일 `tests/unit/test_embedder.py`도 함께 작성한다.
최소한 아래 케이스를 커버해야 한다:

- `KoSimCSEEmbedder`가 `EmbedderProtocol`을 만족하는지 `isinstance()` 검증
- mock embedder로 `embed()` 호출 시 올바른 shape 반환 (`list[list[float]]`)
- 빈 리스트 입력 시 빈 리스트 반환

> 실제 모델 로드가 필요한 통합 테스트는 `tests/integration/`에 분리하고, unit 테스트는 mock만 사용한다.

## 검증 절차

1. 위 AC 커맨드를 순서대로 실행한다.
2. 아키텍처 체크리스트:
   - `EmbedderProtocol`이 `src/embedding/protocol.py`에 정의되었는가?
   - `KoSimCSEEmbedder`가 Protocol을 구조적으로 만족하는가?
   - `src/embedding/`이 다른 `src/` 모듈을 임포트하지 않는가?
3. 결과에 따라 `phases/0-mvp/index.json`의 step 3을 업데이트한다:
   - 성공 → `"status": "completed"`, `"summary": "src/embedding/ 생성 — EmbedderProtocol, KoSimCSEEmbedder 정의 완료"`
   - 수정 3회 시도 후에도 실패 → `"status": "error"`, `"error_message": "구체적 에러 내용"`
   - 사용자 개입 필요 → `"status": "blocked"`, `"blocked_reason": "구체적 사유"` 후 즉시 중단

## 금지사항

- `src/embedding/`에서 `src/crawler/`, `src/pipeline/`, `src/retriever/`, `src/db/`, `src/api/`를 임포트하지 마라
- `sentence_transformers` 외 외부 임베딩 API(OpenAI 등)를 호출하지 마라 (ADR-002)

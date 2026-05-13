.PHONY: setup run-mvp run-mvp-push lint typecheck test test-unit test-integration test-search-quality test-all

# 개발 환경 초기 설정
setup:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

# 하네스 실행 (venv 활성화 불필요 — execute.py가 .venv/bin 자동 주입)
run-mvp:
	python3 scripts/execute.py 0-mvp

run-mvp-push:
	python3 scripts/execute.py 0-mvp --push

# 코드 품질
lint:
	.venv/bin/ruff check src/ tests/ scripts/

typecheck:
	.venv/bin/mypy src/

test: test-unit

test-unit:
	.venv/bin/pytest tests/unit --cov=src --cov-report=term-missing

test-integration:
	.venv/bin/pytest tests/integration/test_api.py tests/integration/test_crawler.py --cov=src --cov-report=term-missing

test-search-quality:
	.venv/bin/pytest tests/integration/test_search_quality.py --cov=src --cov-report=term-missing

test-all:
	.venv/bin/pytest tests --cov=src --cov-report=term-missing

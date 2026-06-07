#!/usr/bin/env python3
"""Evaluate search quality against the current persistent Chroma index.

The production search API receives a profile-shaped SearchRequest. This
evaluation harness adds an audit-only answer contract and query fragment so
ranking changes are not confused with evaluation-set semantics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.db.chroma import WELFARE_COLLECTION, get_collection  # noqa: E402
from src.embedding.kosimcse import KoSimCSEEmbedder  # noqa: E402
from src.models.welfare import SearchRequest, SearchResponse  # noqa: E402
from src.retriever.intent import build_query_intent  # noqa: E402
from src.retriever.rerank import rank_service_candidates  # noqa: E402
from src.retriever.search import (  # noqa: E402
    ADAPTIVE_MAX_CANDIDATES,
    DEFAULT_MAX_CANDIDATES,
    _initial_candidate_count,
    _query_collection,
    _response_results_from_raw,
    build_query_text,
)

DEFAULT_CHROMA_PERSIST_DIR = "data/chroma"
DEFAULT_SECTION_AWARE_CHROMA_PERSIST_DIR = "data/chroma-section-aware"


@contextmanager
def _evaluation_persist_dir(requested_path: Path) -> Iterator[Path]:
    """Evaluate tracked DBs through a temp copy so Chroma cannot dirty them."""
    protected_paths = {
        Path(DEFAULT_CHROMA_PERSIST_DIR).resolve(),
        Path(DEFAULT_SECTION_AWARE_CHROMA_PERSIST_DIR).resolve(),
    }
    if requested_path.resolve() not in protected_paths:
        yield requested_path
        return

    with tempfile.TemporaryDirectory(prefix="bokjidream-chroma-eval-") as tmp_dir:
        copied_path = Path(tmp_dir) / requested_path.name
        shutil.copytree(requested_path.resolve(), copied_path)
        yield copied_path


EVIDENCE_FIELDS = frozenset(
    {
        "serv_nm",
        "serv_dgst",
        "tgtr_dtl_cn",
        "slct_crit_cn",
        "trgter_indvdl",
        "intrs_thema",
    }
)

DEFAULT_EXPECTATION_REASON = (
    "기존 expected_ids 계약을 구조화해 보존한 후보다. 현재 입력만으로 must 확정은 보류한다."
)
DEFAULT_EXCLUSION_REASON = "기존 excluded_ids 회귀 계약을 구조화해 보존한 제외 후보다."


@dataclass(frozen=True)
class ServiceExpectation:
    serv_id: str
    reason: str
    evidence_field: str | None = None
    condition: str | None = None


def svc(
    serv_id: str,
    reason: str = DEFAULT_EXPECTATION_REASON,
    evidence_field: str | None = "serv_nm",
    condition: str | None = None,
) -> ServiceExpectation:
    return ServiceExpectation(
        serv_id=serv_id,
        reason=reason,
        evidence_field=evidence_field,
        condition=condition,
    )


def excluded_svc(serv_id: str, reason: str = DEFAULT_EXCLUSION_REASON) -> ServiceExpectation:
    return ServiceExpectation(serv_id=serv_id, reason=reason, evidence_field="serv_nm")


def _expectation_ids(expectations: Iterable[ServiceExpectation]) -> tuple[str, ...]:
    return tuple(expectation.serv_id for expectation in expectations)


@dataclass(frozen=True)
class EvalCase:
    name: str
    request: SearchRequest
    acceptable_ids: tuple[ServiceExpectation, ...]
    notes: str
    excluded_ids: tuple[ServiceExpectation, ...] = ()
    must_ids: tuple[ServiceExpectation, ...] = ()
    conditional_ids: tuple[ServiceExpectation, ...] = ()
    ambiguous: bool = False
    ambiguity_reason: str | None = None
    query: str | None = None
    intent_theme: str | None = None

    @property
    def expected_ids(self) -> tuple[str, ...]:
        return _expectation_ids((*self.must_ids, *self.acceptable_ids, *self.conditional_ids))


EXPECTED_EVAL_CASE_COUNT = 100


EVAL_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        "노인_저소득",
        SearchRequest(age=67, income_level="저소득", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00003191",
                "노인_저소득: 노인맞춤돌봄서비스은 평가 메모의 후보군(노인 돌봄/안전/소득/일자리 관련 서비스 중 하나)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001115",
                "노인_저소득: 노인 개안수술비 지원은 평가 메모의 후보군(노인 돌봄/안전/소득/일자리 관련 서비스 중 하나)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001155",
                "노인_저소득: 노인일자리 및 사회활동 지원사업은 평가 메모의 후보군(노인 돌봄/안전/소득/일자리 관련 서비스 중 하나)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001093",
                "노인_저소득: 독거노인·장애인 응급안전안심서비스은 평가 메모의 후보군(노인 돌봄/안전/소득/일자리 관련 서비스 중 하나)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001087",
                "노인_저소득: 가사·간병 방문 지원사업은 평가 메모의 후보군(노인 돌봄/안전/소득/일자리 관련 서비스 중 하나)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "노인 돌봄/안전/소득/일자리 관련 서비스 중 하나",
        must_ids=(
            svc(
                "WLF00001164",
                "67세 저소득 사용자는 기초연금의 노인 소득지원 대상성과 직접 맞닿는다.",
                "tgtr_dtl_cn",
            ),
        ),
    ),
    EvalCase(
        "노인_기초수급",
        SearchRequest(
            age=72, income_level="기초생활수급자", disability=False, pregnant=False, top_k=10
        ),
        (
            svc(
                "WLF00003191",
                "노인_기초수급: 노인맞춤돌봄서비스은 평가 메모의 후보군(노인/기초생활 관련 서비스)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001155",
                "노인_기초수급: 노인일자리 및 사회활동 지원사업은 평가 메모의 후보군(노인/기초생활 관련 서비스)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001093",
                "노인_기초수급: 독거노인·장애인 응급안전안심서비스은 평가 메모의 후보군(노인/기초생활 관련 서비스)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001087",
                "노인_기초수급: 가사·간병 방문 지원사업은 평가 메모의 후보군(노인/기초생활 관련 서비스)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "노인/기초생활 관련 서비스",
        must_ids=(
            svc(
                "WLF00001164",
                "72세 기초생활수급자는 기초연금의 고령자 소득지원 조건과 직접 관련된다.",
                "tgtr_dtl_cn",
            ),
            svc(
                "WLF00001132",
                "기초생활수급자 입력은 생계급여의 수급자 대상 조건과 직접 관련된다.",
                "tgtr_dtl_cn",
            ),
        ),
    ),
    EvalCase(
        "독거노인_저소득_1인가구",
        SearchRequest(
            age=78,
            income_level="저소득",
            household_size=1,
            disability=False,
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00003191",
                "독거노인_저소득_1인가구: 노인맞춤돌봄서비스은 평가 메모의 후보군(노인맞춤돌봄, 응급안전안심, 기초연금)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001093",
                "독거노인_저소득_1인가구: 독거노인·장애인 응급안전안심서비스은 평가 메모의 후보군(노인맞춤돌봄, 응급안전안심, 기초연금)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001164",
                "독거노인_저소득_1인가구: 기초연금은 평가 메모의 후보군(노인맞춤돌봄, 응급안전안심, 기초연금)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "노인맞춤돌봄, 응급안전안심, 기초연금",
    ),
    EvalCase(
        "청년_저소득",
        SearchRequest(age=24, income_level="저소득", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00004661",
                "청년_저소득: 청년월세 지원사업은 평가 메모의 후보군(청년월세, 청년내일저축, 국민취업지원, 직업훈련생계비)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000060",
                "청년_저소득: 청년내일저축계좌은 평가 메모의 후보군(청년월세, 청년내일저축, 국민취업지원, 직업훈련생계비)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003245",
                "청년_저소득: 국민취업지원제도은 평가 메모의 후보군(청년월세, 청년내일저축, 국민취업지원, 직업훈련생계비)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003266",
                "청년_저소득: 직업훈련생계비대부은 평가 메모의 후보군(청년월세, 청년내일저축, 국민취업지원, 직업훈련생계비)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "청년월세, 청년내일저축, 국민취업지원, 직업훈련생계비",
    ),
    EvalCase(
        "청년_일반_실업",
        SearchRequest(
            age=25,
            income_level="일반",
            disability=False,
            employment_status="실업",
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00003239",
                "청년_일반_실업: 고용복지플러스센터은 평가 메모의 후보군(국민취업지원, 고용복지플러스, 취업취약계층 지원, 청년내일채움공제)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001172",
                "청년_일반_실업: 취업취약계층 고용지원 사업은 평가 메모의 후보군(국민취업지원, 고용복지플러스, 취업취약계층 지원, 청년내일채움공제)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006215",
                "청년_일반_실업: 청년내일채움공제은 평가 메모의 후보군(국민취업지원, 고용복지플러스, 취업취약계층 지원, 청년내일채움공제)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "국민취업지원, 고용복지플러스, 취업취약계층 지원, 청년내일채움공제",
        must_ids=(
            svc(
                "WLF00003245",
                "실업 상태의 25세 사용자는 국민취업지원제도의 구직·취업지원 대상성과 직접 관련된다.",
                "serv_dgst",
            ),
        ),
        query="취업 구직 직업훈련 고용센터 자활근로",
        intent_theme="employment",
    ),
    EvalCase(
        "청년_저소득_실업",
        SearchRequest(
            age=27,
            income_level="저소득",
            disability=False,
            employment_status="실업",
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00003245",
                "청년_저소득_실업: 국민취업지원제도은 평가 메모의 후보군(저소득 청년 고용/자산/주거 관련 서비스)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003266",
                "청년_저소득_실업: 직업훈련생계비대부은 평가 메모의 후보군(저소득 청년 고용/자산/주거 관련 서비스)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000060",
                "청년_저소득_실업: 청년내일저축계좌은 평가 메모의 후보군(저소득 청년 고용/자산/주거 관련 서비스)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00004661",
                "청년_저소득_실업: 청년월세 지원사업은 평가 메모의 후보군(저소득 청년 고용/자산/주거 관련 서비스)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "저소득 청년 고용/자산/주거 관련 서비스",
        query="취업 구직 직업훈련 고용센터 자활근로",
        intent_theme="employment",
    ),
    EvalCase(
        "중장년_일반_실업",
        SearchRequest(
            age=45,
            income_level="일반",
            disability=False,
            employment_status="실업",
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00003245",
                "중장년_일반_실업: 국민취업지원제도은 평가 메모의 후보군(고용/직업훈련 관련 서비스)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003239",
                "중장년_일반_실업: 고용복지플러스센터은 평가 메모의 후보군(고용/직업훈련 관련 서비스)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001172",
                "중장년_일반_실업: 취업취약계층 고용지원 사업은 평가 메모의 후보군(고용/직업훈련 관련 서비스)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003266",
                "중장년_일반_실업: 직업훈련생계비대부은 평가 메모의 후보군(고용/직업훈련 관련 서비스)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "고용/직업훈련 관련 서비스",
        query="취업 구직 직업훈련 고용센터 자활근로",
        intent_theme="employment",
    ),
    EvalCase(
        "기초수급_실업_근로능력",
        SearchRequest(
            age=42,
            income_level="기초생활수급자",
            disability=False,
            employment_status="실업",
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00001138",
                "기초수급_실업_근로능력: 자활근로(기초, 차상위)은 평가 메모의 후보군(자활근로, 지역자활센터, 국민취업지원, 생계급여)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001157",
                "기초수급_실업_근로능력: 지역자활센터 운영은 평가 메모의 후보군(자활근로, 지역자활센터, 국민취업지원, 생계급여)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003245",
                "기초수급_실업_근로능력: 국민취업지원제도은 평가 메모의 후보군(자활근로, 지역자활센터, 국민취업지원, 생계급여)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001132",
                "기초수급_실업_근로능력: 생계급여(맞춤형 급여)은 평가 메모의 후보군(자활근로, 지역자활센터, 국민취업지원, 생계급여)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "자활근로, 지역자활센터, 국민취업지원, 생계급여",
        query="취업 구직 직업훈련 고용센터 자활근로",
        intent_theme="employment",
    ),
    EvalCase(
        "중증장애인_저소득",
        SearchRequest(
            age=40,
            income_level="저소득",
            disability=True,
            disability_severity="중증",
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00003260",
                "중증장애인_저소득: 장애인활동지원은 평가 메모의 후보군(장애인활동지원, 장애인연금, 장애수당, 보조기기)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003265",
                "중증장애인_저소득: 장애수당은 평가 메모의 후보군(장애인활동지원, 장애인연금, 장애수당, 보조기기)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003211",
                "중증장애인_저소득: 장애인보조기기 교부은 평가 메모의 후보군(장애인활동지원, 장애인연금, 장애수당, 보조기기)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "장애인활동지원, 장애인연금, 장애수당, 보조기기",
        must_ids=(
            svc(
                "WLF00003249",
                "중증 장애와 저소득 입력은 장애인연금의 중증 장애인 소득지원 대상성과 직접 관련된다.",
                "tgtr_dtl_cn",
            ),
        ),
    ),
    EvalCase(
        "중증장애인_일자리",
        SearchRequest(
            age=35,
            income_level="저소득",
            disability=True,
            disability_severity="중증",
            employment_status="실업",
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00000025",
                "중증장애인_일자리: 장애인일자리지원은 평가 메모의 후보군(장애인일자리, 직업재활, 고용장려금, 활동지원)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003189",
                "중증장애인_일자리: 중증장애인직업재활지원은 평가 메모의 후보군(장애인일자리, 직업재활, 고용장려금, 활동지원)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001090",
                "중증장애인_일자리: 장애인고용장려금은 평가 메모의 후보군(장애인일자리, 직업재활, 고용장려금, 활동지원)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003260",
                "중증장애인_일자리: 장애인활동지원은 평가 메모의 후보군(장애인일자리, 직업재활, 고용장려금, 활동지원)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "장애인일자리, 직업재활, 고용장려금, 활동지원",
    ),
    EvalCase(
        "한부모_이혼_저소득_자녀",
        SearchRequest(
            age=35,
            income_level="저소득",
            marital_status="이혼",
            has_children=True,
            disability=False,
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00001104",
                "한부모_이혼_저소득_자녀: 한부모가족자녀 교육비 지원은 평가 메모의 후보군(아동양육비, 자녀교육비, 아이돌봄, 초중고 교육비)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000024",
                "한부모_이혼_저소득_자녀: 아이돌봄서비스은 평가 메모의 후보군(아동양육비, 자녀교육비, 아이돌봄, 초중고 교육비)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001103",
                "한부모_이혼_저소득_자녀: 초중고 교육비 지원사업(고교학비 지원)은 평가 메모의 후보군(아동양육비, 자녀교육비, 아이돌봄, 초중고 교육비)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "아동양육비, 자녀교육비, 아이돌봄, 초중고 교육비",
        must_ids=(
            svc(
                "WLF00001068",
                "이혼·저소득·자녀 있음 입력은 한부모가족 아동양육비 지원 대상성과 직접 관련된다.",
                "tgtr_dtl_cn",
            ),
        ),
    ),
    EvalCase(
        "한부모_사별_저소득_자녀",
        SearchRequest(
            age=42,
            income_level="저소득",
            marital_status="사별",
            has_children=True,
            disability=False,
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00001068",
                "한부모_사별_저소득_자녀: 한부모가족 아동양육비 지원은 평가 메모의 후보군(아동양육비, 자녀교육비, 아이돌봄, 초중고 교육비)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001104",
                "한부모_사별_저소득_자녀: 한부모가족자녀 교육비 지원은 평가 메모의 후보군(아동양육비, 자녀교육비, 아이돌봄, 초중고 교육비)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000024",
                "한부모_사별_저소득_자녀: 아이돌봄서비스은 평가 메모의 후보군(아동양육비, 자녀교육비, 아이돌봄, 초중고 교육비)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001103",
                "한부모_사별_저소득_자녀: 초중고 교육비 지원사업(고교학비 지원)은 평가 메모의 후보군(아동양육비, 자녀교육비, 아이돌봄, 초중고 교육비)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "아동양육비, 자녀교육비, 아이돌봄, 초중고 교육비",
    ),
    EvalCase(
        "영유아_자녀_일반",
        SearchRequest(
            age=32,
            income_level="일반",
            has_children=True,
            disability=False,
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00000024",
                "영유아_자녀_일반: 아이돌봄서비스은 평가 메모의 후보군(아이돌봄, 영유아보육료, 가정양육수당, 유아학비)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003250",
                "영유아_자녀_일반: 영유아보육료 지원은 평가 메모의 후보군(아이돌봄, 영유아보육료, 가정양육수당, 유아학비)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003253",
                "영유아_자녀_일반: 가정양육수당 지원사업은 평가 메모의 후보군(아이돌봄, 영유아보육료, 가정양육수당, 유아학비)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000969",
                "영유아_자녀_일반: 유아학비 지원(3~5세 누리과정 지원)은 평가 메모의 후보군(아이돌봄, 영유아보육료, 가정양육수당, 유아학비)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "아이돌봄, 영유아보육료, 가정양육수당, 유아학비",
    ),
    EvalCase(
        "기초수급_중년",
        SearchRequest(
            age=50, income_level="기초생활수급자", disability=False, pregnant=False, top_k=10
        ),
        (
            svc(
                "WLF00003201",
                "기초수급_중년: 주거급여(맞춤형 급여)은 평가 메모의 후보군(생계급여, 주거급여, 교육급여, 양곡할인)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001089",
                "기초수급_중년: 교육급여(맞춤형 급여)은 평가 메모의 후보군(생계급여, 주거급여, 교육급여, 양곡할인)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000074",
                "기초수급_중년: 양곡할인은 평가 메모의 후보군(생계급여, 주거급여, 교육급여, 양곡할인)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "생계급여, 주거급여, 교육급여, 양곡할인",
        must_ids=(
            svc(
                "WLF00001132",
                "기초생활수급자 입력은 생계급여의 수급자 대상 조건과 직접 관련된다.",
                "tgtr_dtl_cn",
            ),
        ),
        query="생계 생활지원 주거 감면 양곡 긴급복지",
        intent_theme="basic-living",
    ),
    EvalCase(
        "저소득_중년",
        SearchRequest(age=50, income_level="저소득", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00001132",
                "저소득_중년: 생계급여(맞춤형 급여)은 평가 메모의 후보군(생활지원/주거/에너지/문화/양곡 관련 서비스)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003201",
                "저소득_중년: 주거급여(맞춤형 급여)은 평가 메모의 후보군(생활지원/주거/에너지/문화/양곡 관련 서비스)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000072",
                "저소득_중년: 에너지바우처은 평가 메모의 후보군(생활지원/주거/에너지/문화/양곡 관련 서비스)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000055",
                "저소득_중년: 통합문화이용권은 평가 메모의 후보군(생활지원/주거/에너지/문화/양곡 관련 서비스)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000074",
                "저소득_중년: 양곡할인은 평가 메모의 후보군(생활지원/주거/에너지/문화/양곡 관련 서비스)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003257",
                "저소득_중년: 이동통신요금감면은 평가 메모의 후보군(생활지원/주거/에너지/문화/양곡 관련 서비스)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "생활지원/주거/에너지/문화/양곡 관련 서비스",
        query="생계 생활지원 주거 감면 양곡 긴급복지",
        intent_theme="basic-living",
    ),
    EvalCase(
        "차상위_중년",
        SearchRequest(
            age=50, income_level="차상위계층", disability=False, pregnant=False, top_k=10
        ),
        (
            svc(
                "WLF00001087",
                "차상위_중년: 가사·간병 방문 지원사업은 평가 메모의 후보군(차상위 조건이 포함된 생활/돌봄/자활/교육 서비스)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001093",
                "차상위_중년: 독거노인·장애인 응급안전안심서비스은 평가 메모의 후보군(차상위 조건이 포함된 생활/돌봄/자활/교육 서비스)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001138",
                "차상위_중년: 자활근로(기초, 차상위)은 평가 메모의 후보군(차상위 조건이 포함된 생활/돌봄/자활/교육 서비스)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001089",
                "차상위_중년: 교육급여(맞춤형 급여)은 평가 메모의 후보군(차상위 조건이 포함된 생활/돌봄/자활/교육 서비스)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "차상위 조건이 포함된 생활/돌봄/자활/교육 서비스",
        query="생계 생활지원 주거 감면 양곡 긴급복지",
        intent_theme="basic-living",
    ),
    EvalCase(
        "저소득_주거취약",
        SearchRequest(
            age=45,
            income_level="저소득",
            household_size=1,
            disability=False,
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00003201",
                "저소득_주거취약: 주거급여(맞춤형 급여)은 평가 메모의 후보군(주거급여, 긴급복지 주거, 주거상향, 매입/전세임대)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000917",
                "저소득_주거취약: 긴급복지 주거지원은 평가 메모의 후보군(주거급여, 긴급복지 주거, 주거상향, 매입/전세임대)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000036",
                "저소득_주거취약: 주거취약계층 주거상향 지원사업은 평가 메모의 후보군(주거급여, 긴급복지 주거, 주거상향, 매입/전세임대)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000062",
                "저소득_주거취약: 기존주택등 매입임대주택 지원사업은 평가 메모의 후보군(주거급여, 긴급복지 주거, 주거상향, 매입/전세임대)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003269",
                "저소득_주거취약: 기존주택 전세임대주택 지원사업은 평가 메모의 후보군(주거급여, 긴급복지 주거, 주거상향, 매입/전세임대)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "주거급여, 긴급복지 주거, 주거상향, 매입/전세임대",
        query="주거 주거급여 공공임대 전세 월세 주거비",
        intent_theme="housing",
    ),
    EvalCase(
        "저소득_임산부",
        SearchRequest(
            age=30,
            income_level="저소득",
            marital_status="기혼",
            disability=False,
            pregnant=True,
            top_k=10,
        ),
        (
            svc(
                "WLF00000061",
                "저소득_임산부: 의료급여임신.출산진료비지원은 평가 메모의 후보군(고위험 임산부, 임신출산 진료비, 해산급여, 긴급복지 해산비)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "고위험 임산부, 임신출산 진료비, 해산급여, 긴급복지 해산비",
        conditional_ids=(
            svc(
                "WLF00001088",
                "고위험 임신 여부가 추가로 확인되어야 강한 정답이 된다.",
                "slct_crit_cn",
                "고위험 임산부 진단 또는 관련 의료 조건 확인",
            ),
            svc(
                "WLF00001135",
                "해산급여는 실제 출산·해산 상황 확인이 추가로 필요하다.",
                "slct_crit_cn",
                "출산 또는 해산급여 지급 사유 확인",
            ),
            svc(
                "WLF00003178",
                "긴급복지 해산비는 긴급지원 사유와 해산 상황 확인이 추가로 필요하다.",
                "slct_crit_cn",
                "긴급복지 지원 사유와 해산비 지급 사유 확인",
            ),
        ),
        query="임신 출산 산모 건강관리 모자보건",
        intent_theme="maternity",
    ),
    EvalCase(
        "저소득_청소년",
        SearchRequest(age=16, income_level="저소득", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00000078",
                "저소득_청소년: 청소년특별지원은 평가 메모의 후보군(청소년특별지원, 학교밖청소년, 생리용품, 지역아동센터)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000948",
                "저소득_청소년: 학교 밖 청소년 지원은 평가 메모의 후보군(청소년특별지원, 학교밖청소년, 생리용품, 지역아동센터)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000781",
                "저소득_청소년: 여성청소년 생리용품 지원은 평가 메모의 후보군(청소년특별지원, 학교밖청소년, 생리용품, 지역아동센터)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001107",
                "저소득_청소년: 지역아동센터 지원은 평가 메모의 후보군(청소년특별지원, 학교밖청소년, 생리용품, 지역아동센터)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "청소년특별지원, 학교밖청소년, 생리용품, 지역아동센터",
    ),
    EvalCase(
        "아동_저소득_자녀",
        SearchRequest(
            age=8,
            income_level="저소득",
            has_children=True,
            disability=False,
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00001103",
                "아동_저소득_자녀: 초중고 교육비 지원사업(고교학비 지원)은 평가 메모의 후보군(교육비, 교육급여, 지역아동센터, 아이돌봄, 아동발달지원계좌)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001089",
                "아동_저소득_자녀: 교육급여(맞춤형 급여)은 평가 메모의 후보군(교육비, 교육급여, 지역아동센터, 아이돌봄, 아동발달지원계좌)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001107",
                "아동_저소득_자녀: 지역아동센터 지원은 평가 메모의 후보군(교육비, 교육급여, 지역아동센터, 아이돌봄, 아동발달지원계좌)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000024",
                "아동_저소득_자녀: 아이돌봄서비스은 평가 메모의 후보군(교육비, 교육급여, 지역아동센터, 아이돌봄, 아동발달지원계좌)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003258",
                "아동_저소득_자녀: 아동발달지원계좌(디딤씨앗통장)지원은 평가 메모의 후보군(교육비, 교육급여, 지역아동센터, 아이돌봄, 아동발달지원계좌)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "교육비, 교육급여, 지역아동센터, 아이돌봄, 아동발달지원계좌",
    ),
    EvalCase(
        "61세_저소득_비장애_자녀없음",
        SearchRequest(
            age=61,
            income_level="저소득",
            household_size=1,
            has_children=False,
            disability=False,
            employment_status="비경제활동",
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00001132",
                "61세_저소득_비장애_자녀없음: 생계급여(맞춤형 급여)은 평가 메모의 후보군(61세 저소득 1인 가구에는 생활/주거/돌봄/감면 중심 결과가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003201",
                "61세_저소득_비장애_자녀없음: 주거급여(맞춤형 급여)은 평가 메모의 후보군(61세 저소득 1인 가구에는 생활/주거/돌봄/감면 중심 결과가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001087",
                "61세_저소득_비장애_자녀없음: 가사·간병 방문 지원사업은 평가 메모의 후보군(61세 저소득 1인 가구에는 생활/주거/돌봄/감면 중심 결과가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003180",
                "61세_저소득_비장애_자녀없음: 긴급복지 생계지원은 평가 메모의 후보군(61세 저소득 1인 가구에는 생활/주거/돌봄/감면 중심 결과가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003257",
                "61세_저소득_비장애_자녀없음: 이동통신요금감면은 평가 메모의 후보군(61세 저소득 1인 가구에는 생활/주거/돌봄/감면 중심 결과가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "61세 저소득 1인 가구에는 생활/주거/돌봄/감면 중심 결과가 적절",
        excluded_ids=(
            excluded_svc(
                "WLF00001093",
                "61세_저소득_비장애_자녀없음: 독거노인·장애인 응급안전안심서비스은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00001164",
                "61세_저소득_비장애_자녀없음: 기초연금은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003191",
                "61세_저소득_비장애_자녀없음: 노인맞춤돌봄서비스은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00001155",
                "61세_저소득_비장애_자녀없음: 노인일자리 및 사회활동 지원사업은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003249",
                "61세_저소득_비장애_자녀없음: 장애인연금은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003265",
                "61세_저소득_비장애_자녀없음: 장애수당은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00001068",
                "61세_저소득_비장애_자녀없음: 한부모가족 아동양육비 지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00000024",
                "61세_저소득_비장애_자녀없음: 아이돌봄서비스은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
        ),
    ),
    EvalCase(
        "40대_저소득_비장애",
        SearchRequest(age=40, income_level="저소득", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00001132",
                "40대_저소득_비장애: 생계급여(맞춤형 급여)은 평가 메모의 후보군(비장애 저소득 중장년에는 일반 생활지원/주거/감면 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003201",
                "40대_저소득_비장애: 주거급여(맞춤형 급여)은 평가 메모의 후보군(비장애 저소득 중장년에는 일반 생활지원/주거/감면 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000072",
                "40대_저소득_비장애: 에너지바우처은 평가 메모의 후보군(비장애 저소득 중장년에는 일반 생활지원/주거/감면 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003257",
                "40대_저소득_비장애: 이동통신요금감면은 평가 메모의 후보군(비장애 저소득 중장년에는 일반 생활지원/주거/감면 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003180",
                "40대_저소득_비장애: 긴급복지 생계지원은 평가 메모의 후보군(비장애 저소득 중장년에는 일반 생활지원/주거/감면 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "비장애 저소득 중장년에는 일반 생활지원/주거/감면 서비스가 적절",
        excluded_ids=(
            excluded_svc(
                "WLF00003211",
                "40대_저소득_비장애: 장애인보조기기 교부은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003249",
                "40대_저소득_비장애: 장애인연금은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003265",
                "40대_저소득_비장애: 장애수당은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003181",
                "40대_저소득_비장애: 장애인의료비지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003260",
                "40대_저소득_비장애: 장애인활동지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
        ),
    ),
    EvalCase(
        "40대_저소득_자녀없음",
        SearchRequest(
            age=40,
            income_level="저소득",
            has_children=False,
            disability=False,
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00001132",
                "40대_저소득_자녀없음: 생계급여(맞춤형 급여)은 평가 메모의 후보군(자녀가 없는 저소득 중장년에는 자녀/보육 전용 서비스가 상위에 오면 안 됨)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003201",
                "40대_저소득_자녀없음: 주거급여(맞춤형 급여)은 평가 메모의 후보군(자녀가 없는 저소득 중장년에는 자녀/보육 전용 서비스가 상위에 오면 안 됨)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000072",
                "40대_저소득_자녀없음: 에너지바우처은 평가 메모의 후보군(자녀가 없는 저소득 중장년에는 자녀/보육 전용 서비스가 상위에 오면 안 됨)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003257",
                "40대_저소득_자녀없음: 이동통신요금감면은 평가 메모의 후보군(자녀가 없는 저소득 중장년에는 자녀/보육 전용 서비스가 상위에 오면 안 됨)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003180",
                "40대_저소득_자녀없음: 긴급복지 생계지원은 평가 메모의 후보군(자녀가 없는 저소득 중장년에는 자녀/보육 전용 서비스가 상위에 오면 안 됨)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "자녀가 없는 저소득 중장년에는 자녀/보육 전용 서비스가 상위에 오면 안 됨",
        excluded_ids=(
            excluded_svc(
                "WLF00001068",
                "40대_저소득_자녀없음: 한부모가족 아동양육비 지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00000024",
                "40대_저소득_자녀없음: 아이돌봄서비스은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003253",
                "40대_저소득_자녀없음: 가정양육수당 지원사업은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00001171",
                "40대_저소득_자녀없음: 아동수당 지급은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00005023",
                "40대_저소득_자녀없음: 청소년부모 아동양육비 지원사업은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
        ),
    ),
    EvalCase(
        "30대_저소득_임신아님",
        SearchRequest(age=30, income_level="저소득", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00004661",
                "30대_저소득_임신아님: 청년월세 지원사업은 평가 메모의 후보군(임신 정보가 없는 청년 저소득 사용자는 주거/생활지원 중심 결과가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003201",
                "30대_저소득_임신아님: 주거급여(맞춤형 급여)은 평가 메모의 후보군(임신 정보가 없는 청년 저소득 사용자는 주거/생활지원 중심 결과가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001132",
                "30대_저소득_임신아님: 생계급여(맞춤형 급여)은 평가 메모의 후보군(임신 정보가 없는 청년 저소득 사용자는 주거/생활지원 중심 결과가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003257",
                "30대_저소득_임신아님: 이동통신요금감면은 평가 메모의 후보군(임신 정보가 없는 청년 저소득 사용자는 주거/생활지원 중심 결과가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003180",
                "30대_저소득_임신아님: 긴급복지 생계지원은 평가 메모의 후보군(임신 정보가 없는 청년 저소득 사용자는 주거/생활지원 중심 결과가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "임신 정보가 없는 청년 저소득 사용자는 주거/생활지원 중심 결과가 적절",
        excluded_ids=(
            excluded_svc(
                "WLF00001088",
                "30대_저소득_임신아님: 고위험 임산부 의료비 지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00000061",
                "30대_저소득_임신아님: 의료급여임신.출산진료비지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00001135",
                "30대_저소득_임신아님: 해산급여은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003178",
                "30대_저소득_임신아님: 긴급복지 해산비지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
        ),
    ),
    EvalCase(
        "5세_일반_영유아",
        SearchRequest(age=5, income_level="일반", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00003250",
                "5세_일반_영유아: 영유아보육료 지원은 평가 메모의 후보군(5세 영유아에게는 보육료, 유아학비, 아동수당, 시간제보육 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003253",
                "5세_일반_영유아: 가정양육수당 지원사업은 평가 메모의 후보군(5세 영유아에게는 보육료, 유아학비, 아동수당, 시간제보육 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000969",
                "5세_일반_영유아: 유아학비 지원(3~5세 누리과정 지원)은 평가 메모의 후보군(5세 영유아에게는 보육료, 유아학비, 아동수당, 시간제보육 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000037",
                "5세_일반_영유아: 시간제보육 지원은 평가 메모의 후보군(5세 영유아에게는 보육료, 유아학비, 아동수당, 시간제보육 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000030",
                "5세_일반_영유아: 육아종합지원서비스 제공은 평가 메모의 후보군(5세 영유아에게는 보육료, 유아학비, 아동수당, 시간제보육 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "5세 영유아에게는 보육료, 유아학비, 아동수당, 시간제보육 계열이 적절",
        must_ids=(
            svc(
                "WLF00001171",
                "5세 아동 입력은 아동수당 지급의 아동 대상 조건과 직접 관련된다.",
                "tgtr_dtl_cn",
            ),
        ),
    ),
    EvalCase(
        "6세_저소득_아동돌봄",
        SearchRequest(age=6, income_level="저소득", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00001171",
                "6세_저소득_아동돌봄: 아동수당 지급은 평가 메모의 후보군(6세 저소득 아동에게는 아동수당, 지역 돌봄, 방과후 돌봄 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001107",
                "6세_저소득_아동돌봄: 지역아동센터 지원은 평가 메모의 후보군(6세 저소득 아동에게는 아동수당, 지역 돌봄, 방과후 돌봄 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000089",
                "6세_저소득_아동돌봄: 다함께 돌봄 사업은 평가 메모의 후보군(6세 저소득 아동에게는 아동수당, 지역 돌봄, 방과후 돌봄 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001140",
                "6세_저소득_아동돌봄: 방과후보육료지원은 평가 메모의 후보군(6세 저소득 아동에게는 아동수당, 지역 돌봄, 방과후 돌봄 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003223",
                "6세_저소득_아동돌봄: 고난도 보호대상아동 맞춤형 사례관리서비스은 평가 메모의 후보군(6세 저소득 아동에게는 아동수당, 지역 돌봄, 방과후 돌봄 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "6세 저소득 아동에게는 아동수당, 지역 돌봄, 방과후 돌봄 계열이 적절",
        query="돌봄 안부 안전 장기요양 방문 재가",
        intent_theme="care",
    ),
    EvalCase(
        "18세_저소득_청소년",
        SearchRequest(age=18, income_level="저소득", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00000078",
                "18세_저소득_청소년: 청소년특별지원은 평가 메모의 후보군(18세 저소득 청소년에게는 청소년 보호, 상담, 생활지원 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000948",
                "18세_저소득_청소년: 학교 밖 청소년 지원은 평가 메모의 후보군(18세 저소득 청소년에게는 청소년 보호, 상담, 생활지원 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000781",
                "18세_저소득_청소년: 여성청소년 생리용품 지원은 평가 메모의 후보군(18세 저소득 청소년에게는 청소년 보호, 상담, 생활지원 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003254",
                "18세_저소득_청소년: 청소년방과후아카데미운영지원은 평가 메모의 후보군(18세 저소득 청소년에게는 청소년 보호, 상담, 생활지원 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003200",
                "18세_저소득_청소년: 청소년동반자프로그램 운영은 평가 메모의 후보군(18세 저소득 청소년에게는 청소년 보호, 상담, 생활지원 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003236",
                "18세_저소득_청소년: 청소년상담1388 전화상담은 평가 메모의 후보군(18세 저소득 청소년에게는 청소년 보호, 상담, 생활지원 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "18세 저소득 청소년에게는 청소년 보호, 상담, 생활지원 서비스가 적절",
    ),
    EvalCase(
        "19세_저소득_청년",
        SearchRequest(age=19, income_level="저소득", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00004661",
                "19세_저소득_청년: 청년월세 지원사업은 평가 메모의 후보군(19세 저소득 청년에게는 청년 주거, 자산형성, 취업, 학자금 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000060",
                "19세_저소득_청년: 청년내일저축계좌은 평가 메모의 후보군(19세 저소득 청년에게는 청년 주거, 자산형성, 취업, 학자금 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003245",
                "19세_저소득_청년: 국민취업지원제도은 평가 메모의 후보군(19세 저소득 청년에게는 청년 주거, 자산형성, 취업, 학자금 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003266",
                "19세_저소득_청년: 직업훈련생계비대부은 평가 메모의 후보군(19세 저소득 청년에게는 청년 주거, 자산형성, 취업, 학자금 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001076",
                "19세_저소득_청년: 서민금융 활성화 지원(햇살론youth 보증사업)은 평가 메모의 후보군(19세 저소득 청년에게는 청년 주거, 자산형성, 취업, 학자금 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003277",
                "19세_저소득_청년: 취업 후 상환 학자금대출은 평가 메모의 후보군(19세 저소득 청년에게는 청년 주거, 자산형성, 취업, 학자금 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "19세 저소득 청년에게는 청년 주거, 자산형성, 취업, 학자금 계열이 적절",
    ),
    EvalCase(
        "34세_저소득_청년",
        SearchRequest(
            age=34,
            income_level="저소득",
            disability=False,
            employment_status="실업",
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00004661",
                "34세_저소득_청년: 청년월세 지원사업은 평가 메모의 후보군(34세 저소득 실업 청년에게는 청년/저소득 주거와 고용 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000060",
                "34세_저소득_청년: 청년내일저축계좌은 평가 메모의 후보군(34세 저소득 실업 청년에게는 청년/저소득 주거와 고용 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003245",
                "34세_저소득_청년: 국민취업지원제도은 평가 메모의 후보군(34세 저소득 실업 청년에게는 청년/저소득 주거와 고용 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003266",
                "34세_저소득_청년: 직업훈련생계비대부은 평가 메모의 후보군(34세 저소득 실업 청년에게는 청년/저소득 주거와 고용 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006215",
                "34세_저소득_청년: 청년내일채움공제은 평가 메모의 후보군(34세 저소득 실업 청년에게는 청년/저소득 주거와 고용 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00004649",
                "34세_저소득_청년: 행복주택 공급은 평가 메모의 후보군(34세 저소득 실업 청년에게는 청년/저소득 주거와 고용 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "34세 저소득 실업 청년에게는 청년/저소득 주거와 고용 지원이 적절",
    ),
    EvalCase(
        "35세_일반_실업",
        SearchRequest(
            age=35,
            income_level="일반",
            disability=False,
            employment_status="실업",
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00003245",
                "35세_일반_실업: 국민취업지원제도은 평가 메모의 후보군(35세 일반 실업자는 청년 전용보다 일반 고용/재취업 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003239",
                "35세_일반_실업: 고용복지플러스센터은 평가 메모의 후보군(35세 일반 실업자는 청년 전용보다 일반 고용/재취업 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001172",
                "35세_일반_실업: 취업취약계층 고용지원 사업은 평가 메모의 후보군(35세 일반 실업자는 청년 전용보다 일반 고용/재취업 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006229",
                "35세_일반_실업: 국민내일배움카드제 직업훈련지원(훈련비, 훈련장려금)은 평가 메모의 후보군(35세 일반 실업자는 청년 전용보다 일반 고용/재취업 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00005852",
                "35세_일반_실업: 중장년 경력지원제은 평가 메모의 후보군(35세 일반 실업자는 청년 전용보다 일반 고용/재취업 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001156",
                "35세_일반_실업: 재취업지원서비스 시행지원은 평가 메모의 후보군(35세 일반 실업자는 청년 전용보다 일반 고용/재취업 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "35세 일반 실업자는 청년 전용보다 일반 고용/재취업 서비스가 적절",
        excluded_ids=(
            excluded_svc(
                "WLF00004661",
                "35세_일반_실업: 청년월세 지원사업은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00000060",
                "35세_일반_실업: 청년내일저축계좌은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00006215",
                "35세_일반_실업: 청년내일채움공제은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00001076",
                "35세_일반_실업: 서민금융 활성화 지원(햇살론youth 보증사업)은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00000812",
                "35세_일반_실업: 청년창업농장학금 지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
        ),
        query="취업 구직 직업훈련 고용센터 자활근로",
        intent_theme="employment",
    ),
    EvalCase(
        "64세_저소득_비장애_1인가구",
        SearchRequest(
            age=64,
            income_level="저소득",
            household_size=1,
            disability=False,
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00001132",
                "64세_저소득_비장애_1인가구: 생계급여(맞춤형 급여)은 평가 메모의 후보군(64세 저소득 비장애 1인 가구는 일반 저소득 생활/주거/감면 중심이어야 함)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003201",
                "64세_저소득_비장애_1인가구: 주거급여(맞춤형 급여)은 평가 메모의 후보군(64세 저소득 비장애 1인 가구는 일반 저소득 생활/주거/감면 중심이어야 함)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003180",
                "64세_저소득_비장애_1인가구: 긴급복지 생계지원은 평가 메모의 후보군(64세 저소득 비장애 1인 가구는 일반 저소득 생활/주거/감면 중심이어야 함)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003257",
                "64세_저소득_비장애_1인가구: 이동통신요금감면은 평가 메모의 후보군(64세 저소득 비장애 1인 가구는 일반 저소득 생활/주거/감면 중심이어야 함)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000072",
                "64세_저소득_비장애_1인가구: 에너지바우처은 평가 메모의 후보군(64세 저소득 비장애 1인 가구는 일반 저소득 생활/주거/감면 중심이어야 함)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001087",
                "64세_저소득_비장애_1인가구: 가사·간병 방문 지원사업은 평가 메모의 후보군(64세 저소득 비장애 1인 가구는 일반 저소득 생활/주거/감면 중심이어야 함)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "64세 저소득 비장애 1인 가구는 일반 저소득 생활/주거/감면 중심이어야 함",
        excluded_ids=(
            excluded_svc(
                "WLF00001164",
                "64세_저소득_비장애_1인가구: 기초연금은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003191",
                "64세_저소득_비장애_1인가구: 노인맞춤돌봄서비스은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00001155",
                "64세_저소득_비장애_1인가구: 노인일자리 및 사회활동 지원사업은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00001093",
                "64세_저소득_비장애_1인가구: 독거노인·장애인 응급안전안심서비스은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003249",
                "64세_저소득_비장애_1인가구: 장애인연금은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003265",
                "64세_저소득_비장애_1인가구: 장애수당은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
        ),
    ),
    EvalCase(
        "65세_저소득_1인가구",
        SearchRequest(
            age=65,
            income_level="저소득",
            household_size=1,
            disability=False,
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00001164",
                "65세_저소득_1인가구: 기초연금은 평가 메모의 후보군(65세 이상 저소득 1인 가구에는 노인 소득, 돌봄, 안전, 의료 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003191",
                "65세_저소득_1인가구: 노인맞춤돌봄서비스은 평가 메모의 후보군(65세 이상 저소득 1인 가구에는 노인 소득, 돌봄, 안전, 의료 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001093",
                "65세_저소득_1인가구: 독거노인·장애인 응급안전안심서비스은 평가 메모의 후보군(65세 이상 저소득 1인 가구에는 노인 소득, 돌봄, 안전, 의료 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001155",
                "65세_저소득_1인가구: 노인일자리 및 사회활동 지원사업은 평가 메모의 후보군(65세 이상 저소득 1인 가구에는 노인 소득, 돌봄, 안전, 의료 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001115",
                "65세_저소득_1인가구: 노인 개안수술비 지원은 평가 메모의 후보군(65세 이상 저소득 1인 가구에는 노인 소득, 돌봄, 안전, 의료 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001179",
                "65세_저소득_1인가구: 노인 무릎인공관절 수술 지원 사업은 평가 메모의 후보군(65세 이상 저소득 1인 가구에는 노인 소득, 돌봄, 안전, 의료 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "65세 이상 저소득 1인 가구에는 노인 소득, 돌봄, 안전, 의료 서비스가 적절",
    ),
    EvalCase(
        "65세_일반_노후준비",
        SearchRequest(age=65, income_level="일반", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00000031",
                "65세_일반_노후준비: 노후준비서비스은 평가 메모의 후보군(65세 일반 노인에게는 노후준비, 장기요양, 노인성 질환 예방 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001108",
                "65세_일반_노후준비: 주택담보노후연금보증은 평가 메모의 후보군(65세 일반 노인에게는 노후준비, 장기요양, 노인성 질환 예방 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000103",
                "65세_일반_노후준비: 노인장기요양보험 복지용구 급여은 평가 메모의 후보군(65세 일반 노인에게는 노후준비, 장기요양, 노인성 질환 예방 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001177",
                "65세_일반_노후준비: 장기요양 본인부담금 감경은 평가 메모의 후보군(65세 일반 노인에게는 노후준비, 장기요양, 노인성 질환 예방 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003241",
                "65세_일반_노후준비: 노후긴급자금 대부사업은 평가 메모의 후보군(65세 일반 노인에게는 노후준비, 장기요양, 노인성 질환 예방 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003280",
                "65세_일반_노후준비: 전립선등 노인성질환 예방관리은 평가 메모의 후보군(65세 일반 노인에게는 노후준비, 장기요양, 노인성 질환 예방 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "65세 일반 노인에게는 노후준비, 장기요양, 노인성 질환 예방 서비스가 적절",
    ),
    EvalCase(
        "임산부_일반",
        SearchRequest(
            age=30,
            income_level="일반",
            marital_status="기혼",
            disability=False,
            pregnant=True,
            top_k=10,
        ),
        (
            svc(
                "WLF00001161",
                "임산부_일반: 표준모자보건수첩 제공은 평가 메모의 후보군(일반 임산부에게는 임신·출산 건강관리와 출산 초기 지원 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003213",
                "임산부_일반: 인플루엔자 국가예방접종 지원사업은 평가 메모의 후보군(일반 임산부에게는 임신·출산 건강관리와 출산 초기 지원 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00004656",
                "임산부_일반: 첫만남이용권은 평가 메모의 후보군(일반 임산부에게는 임신·출산 건강관리와 출산 초기 지원 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00005631",
                "임산부_일반: 아가와 엄마를 위한 무료 공익보험(우체국대한민국 엄마보험)은 평가 메모의 후보군(일반 임산부에게는 임신·출산 건강관리와 출산 초기 지원 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00005858",
                "임산부_일반: 임신 사전건강관리 지원사업은 평가 메모의 후보군(일반 임산부에게는 임신·출산 건강관리와 출산 초기 지원 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00004647",
                "임산부_일반: 국민연금 출산크레딧은 평가 메모의 후보군(일반 임산부에게는 임신·출산 건강관리와 출산 초기 지원 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "일반 임산부에게는 임신·출산 건강관리와 출산 초기 지원 서비스가 적절",
        query="임신 출산 산모 건강관리 모자보건",
        intent_theme="maternity",
    ),
    EvalCase(
        "임산부_저소득_출산지원",
        SearchRequest(
            age=30,
            income_level="저소득",
            marital_status="기혼",
            disability=False,
            pregnant=True,
            top_k=10,
        ),
        (
            svc(
                "WLF00001088",
                "임산부_저소득_출산지원: 고위험 임산부 의료비 지원은 평가 메모의 후보군(저소득 임산부에게는 의료비, 해산급여, 기저귀·조제분유, 영양 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000061",
                "임산부_저소득_출산지원: 의료급여임신.출산진료비지원은 평가 메모의 후보군(저소득 임산부에게는 의료비, 해산급여, 기저귀·조제분유, 영양 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001135",
                "임산부_저소득_출산지원: 해산급여은 평가 메모의 후보군(저소득 임산부에게는 의료비, 해산급여, 기저귀·조제분유, 영양 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003178",
                "임산부_저소득_출산지원: 긴급복지 해산비지원은 평가 메모의 후보군(저소득 임산부에게는 의료비, 해산급여, 기저귀·조제분유, 영양 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000092",
                "임산부_저소득_출산지원: 저소득층 기저귀·조제분유 지원은 평가 메모의 후보군(저소득 임산부에게는 의료비, 해산급여, 기저귀·조제분유, 영양 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006239",
                "임산부_저소득_출산지원: 영양플러스 사업은 평가 메모의 후보군(저소득 임산부에게는 의료비, 해산급여, 기저귀·조제분유, 영양 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "저소득 임산부에게는 의료비, 해산급여, 기저귀·조제분유, 영양 지원이 적절",
        query="임신 출산 의료비 해산급여 해산비",
        intent_theme="maternity:birth-cost",
    ),
    EvalCase(
        "산모_저소득_신생아",
        SearchRequest(
            age=31,
            income_level="저소득",
            marital_status="기혼",
            disability=False,
            pregnant=True,
            top_k=10,
        ),
        (
            svc(
                "WLF00001188",
                "산모_저소득_신생아: 산모·신생아 건강관리 지원사업은 평가 메모의 후보군(출산 전후 산모에게는 산모·신생아 건강관리와 출산·육아기 급여가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003226",
                "산모_저소득_신생아: 모성보호육아지원(출산전후휴가(유산ㆍ사산휴가 포함) 급여, 육아휴직등 급여)은 평가 메모의 후보군(출산 전후 산모에게는 산모·신생아 건강관리와 출산·육아기 급여가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003224",
                "산모_저소득_신생아: 출산육아기 고용안정장려금은 평가 메모의 후보군(출산 전후 산모에게는 산모·신생아 건강관리와 출산·육아기 급여가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000838",
                "산모_저소득_신생아: 고용보험 미적용자 출산급여 지원은 평가 메모의 후보군(출산 전후 산모에게는 산모·신생아 건강관리와 출산·육아기 급여가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00004656",
                "산모_저소득_신생아: 첫만남이용권은 평가 메모의 후보군(출산 전후 산모에게는 산모·신생아 건강관리와 출산·육아기 급여가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00004657",
                "산모_저소득_신생아: 부모급여 지원은 평가 메모의 후보군(출산 전후 산모에게는 산모·신생아 건강관리와 출산·육아기 급여가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "출산 전후 산모에게는 산모·신생아 건강관리와 출산·육아기 급여가 적절",
        query="임신 출산 산모 건강관리 모자보건",
        intent_theme="maternity",
    ),
    EvalCase(
        "청소년산모_저소득",
        SearchRequest(age=18, income_level="저소득", disability=False, pregnant=True, top_k=10),
        (
            svc(
                "WLF00003246",
                "청소년산모_저소득: 청소년산모 임신·출산 의료비 지원은 평가 메모의 후보군(저소득 청소년 산모에게는 청소년산모 의료비와 임신·출산 급여가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001088",
                "청소년산모_저소득: 고위험 임산부 의료비 지원은 평가 메모의 후보군(저소득 청소년 산모에게는 청소년산모 의료비와 임신·출산 급여가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000061",
                "청소년산모_저소득: 의료급여임신.출산진료비지원은 평가 메모의 후보군(저소득 청소년 산모에게는 청소년산모 의료비와 임신·출산 급여가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001135",
                "청소년산모_저소득: 해산급여은 평가 메모의 후보군(저소득 청소년 산모에게는 청소년산모 의료비와 임신·출산 급여가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003178",
                "청소년산모_저소득: 긴급복지 해산비지원은 평가 메모의 후보군(저소득 청소년 산모에게는 청소년산모 의료비와 임신·출산 급여가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "저소득 청소년 산모에게는 청소년산모 의료비와 임신·출산 급여가 적절",
        query="임신 출산 산모 건강관리 모자보건",
        intent_theme="maternity",
    ),
    EvalCase(
        "한부모_저소득_주거",
        SearchRequest(
            age=35,
            income_level="저소득",
            marital_status="이혼",
            has_children=True,
            disability=False,
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00000091",
                "한부모_저소득_주거: 한부모가족복지시설 지원은 평가 메모의 후보군(저소득 한부모 가구에는 한부모 시설과 공공임대/전세임대 주거 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000062",
                "한부모_저소득_주거: 기존주택등 매입임대주택 지원사업은 평가 메모의 후보군(저소득 한부모 가구에는 한부모 시설과 공공임대/전세임대 주거 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003269",
                "한부모_저소득_주거: 기존주택 전세임대주택 지원사업은 평가 메모의 후보군(저소득 한부모 가구에는 한부모 시설과 공공임대/전세임대 주거 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00004649",
                "한부모_저소득_주거: 행복주택 공급은 평가 메모의 후보군(저소득 한부모 가구에는 한부모 시설과 공공임대/전세임대 주거 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003222",
                "한부모_저소득_주거: 버팀목전세자금대출은 평가 메모의 후보군(저소득 한부모 가구에는 한부모 시설과 공공임대/전세임대 주거 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003244",
                "한부모_저소득_주거: 국민임대주택공급은 평가 메모의 후보군(저소득 한부모 가구에는 한부모 시설과 공공임대/전세임대 주거 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "저소득 한부모 가구에는 한부모 시설과 공공임대/전세임대 주거 지원이 적절",
        query="주거 주거급여 공공임대 전세 월세 주거비",
        intent_theme="housing",
    ),
    EvalCase(
        "한부모_저소득_양육비",
        SearchRequest(
            age=36,
            income_level="저소득",
            marital_status="이혼",
            has_children=True,
            disability=False,
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00001068",
                "한부모_저소득_양육비: 한부모가족 아동양육비 지원은 평가 메모의 후보군(저소득 한부모 가구에는 아동양육비, 양육비 이행, 한부모 보험이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00005856",
                "한부모_저소득_양육비: 양육비 선지급은 평가 메모의 후보군(저소득 한부모 가구에는 아동양육비, 양육비 이행, 한부모 보험이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006278",
                "한부모_저소득_양육비: 서민금융진흥원 소액보험(한부모가정의료보험)은 평가 메모의 후보군(저소득 한부모 가구에는 아동양육비, 양육비 이행, 한부모 보험이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003186",
                "한부모_저소득_양육비: 양육비 이행 원스톱 종합서비스은 평가 메모의 후보군(저소득 한부모 가구에는 아동양육비, 양육비 이행, 한부모 보험이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001109",
                "한부모_저소득_양육비: 청소년한부모 아동양육 및 자립지원은 평가 메모의 후보군(저소득 한부모 가구에는 아동양육비, 양육비 이행, 한부모 보험이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "저소득 한부모 가구에는 아동양육비, 양육비 이행, 한부모 보험이 적절",
        query="한부모 양육비 자녀 양육 양육비 이행",
        intent_theme="child-support",
    ),
    EvalCase(
        "한부모_저소득_교육",
        SearchRequest(
            age=40,
            income_level="저소득",
            marital_status="사별",
            has_children=True,
            disability=False,
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00001104",
                "한부모_저소득_교육: 한부모가족자녀 교육비 지원은 평가 메모의 후보군(저소득 한부모 자녀에게는 교육비, 방과후, 장학, 교육정보화 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001103",
                "한부모_저소득_교육: 초중고 교육비 지원사업(고교학비 지원)은 평가 메모의 후보군(저소득 한부모 자녀에게는 교육비, 방과후, 장학, 교육정보화 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000867",
                "한부모_저소득_교육: 방과후학교 자유수강권은 평가 메모의 후보군(저소득 한부모 자녀에게는 교육비, 방과후, 장학, 교육정보화 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003177",
                "한부모_저소득_교육: 복권기금 꿈사다리 장학사업은 평가 메모의 후보군(저소득 한부모 자녀에게는 교육비, 방과후, 장학, 교육정보화 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003227",
                "한부모_저소득_교육: 초.중.고 학생 교육정보화 지원은 평가 메모의 후보군(저소득 한부모 자녀에게는 교육비, 방과후, 장학, 교육정보화 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001120",
                "한부모_저소득_교육: 교육복지우선지원사업은 평가 메모의 후보군(저소득 한부모 자녀에게는 교육비, 방과후, 장학, 교육정보화 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "저소득 한부모 자녀에게는 교육비, 방과후, 장학, 교육정보화 지원이 적절",
        query="교육비 장학 방과후 교육정보화 급식",
        intent_theme="education",
    ),
    EvalCase(
        "한부모_일반_양육비",
        SearchRequest(
            age=36,
            income_level="일반",
            marital_status="이혼",
            has_children=True,
            disability=False,
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00001068",
                "한부모_일반_양육비: 한부모가족 아동양육비 지원은 평가 메모의 후보군(일반 한부모에게도 한부모 자격 기반 양육비/양육비 이행 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003186",
                "한부모_일반_양육비: 양육비 이행 원스톱 종합서비스은 평가 메모의 후보군(일반 한부모에게도 한부모 자격 기반 양육비/양육비 이행 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00005856",
                "한부모_일반_양육비: 양육비 선지급은 평가 메모의 후보군(일반 한부모에게도 한부모 자격 기반 양육비/양육비 이행 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000091",
                "한부모_일반_양육비: 한부모가족복지시설 지원은 평가 메모의 후보군(일반 한부모에게도 한부모 자격 기반 양육비/양육비 이행 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000024",
                "한부모_일반_양육비: 아이돌봄서비스은 평가 메모의 후보군(일반 한부모에게도 한부모 자격 기반 양육비/양육비 이행 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "일반 한부모에게도 한부모 자격 기반 양육비/양육비 이행 서비스가 적절",
        query="한부모 양육비 자녀 양육 양육비 이행",
        intent_theme="child-support",
    ),
    EvalCase(
        "장애아동_저소득_돌봄",
        SearchRequest(
            age=10,
            income_level="저소득",
            disability=True,
            disability_severity="중증",
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00001067",
                "장애아동_저소득_돌봄: 장애아보육료지원은 평가 메모의 후보군(저소득 장애아동에게는 장애아 보육, 수당, 재활, 가족양육 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003198",
                "장애아동_저소득_돌봄: 장애아동수당은 평가 메모의 후보군(저소득 장애아동에게는 장애아 보육, 수당, 재활, 가족양육 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003195",
                "장애아동_저소득_돌봄: 발달재활서비스은 평가 메모의 후보군(저소득 장애아동에게는 장애아 보육, 수당, 재활, 가족양육 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003262",
                "장애아동_저소득_돌봄: 장애아가족양육지원은 평가 메모의 후보군(저소득 장애아동에게는 장애아 보육, 수당, 재활, 가족양육 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001113",
                "장애아동_저소득_돌봄: (특수교육대상자) 치료지원서비스은 평가 메모의 후보군(저소득 장애아동에게는 장애아 보육, 수당, 재활, 가족양육 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001178",
                "장애아동_저소득_돌봄: 청소년 발달장애인 방과후활동서비스은 평가 메모의 후보군(저소득 장애아동에게는 장애아 보육, 수당, 재활, 가족양육 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "저소득 장애아동에게는 장애아 보육, 수당, 재활, 가족양육 지원이 적절",
        query="돌봄 안부 안전 장기요양 방문 재가",
        intent_theme="care",
    ),
    EvalCase(
        "경증장애인_일반_생활",
        SearchRequest(
            age=30,
            income_level="일반",
            disability=True,
            disability_severity="경증",
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00000027",
                "경증장애인_일반_생활: 장애인 운전교육 사업은 평가 메모의 후보군(일반 경증 장애인에게는 정보화, 문화, 통신, 이동 관련 생활 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000066",
                "경증장애인_일반_생활: 장애인 집합 정보화교육은 평가 메모의 후보군(일반 경증 장애인에게는 정보화, 문화, 통신, 이동 관련 생활 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001117",
                "경증장애인_일반_생활: 장애인문화·예술 지원은 평가 메모의 후보군(일반 경증 장애인에게는 정보화, 문화, 통신, 이동 관련 생활 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003219",
                "경증장애인_일반_생활: 통신중계서비스 제공은 평가 메모의 후보군(일반 경증 장애인에게는 정보화, 문화, 통신, 이동 관련 생활 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000104",
                "경증장애인_일반_생활: 시각·청각장애인용 TV 보급사업은 평가 메모의 후보군(일반 경증 장애인에게는 정보화, 문화, 통신, 이동 관련 생활 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003173",
                "경증장애인_일반_생활: 장애인스포츠강좌이용권 지원은 평가 메모의 후보군(일반 경증 장애인에게는 정보화, 문화, 통신, 이동 관련 생활 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "일반 경증 장애인에게는 정보화, 문화, 통신, 이동 관련 생활 지원이 적절",
    ),
    EvalCase(
        "경증장애인_저소득_보조기기",
        SearchRequest(
            age=45,
            income_level="저소득",
            disability=True,
            disability_severity="경증",
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00003211",
                "경증장애인_저소득_보조기기: 장애인보조기기 교부은 평가 메모의 후보군(저소득 경증 장애인에게는 보조기기, 의료비, 장애수당 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000067",
                "경증장애인_저소득_보조기기: 의료급여 장애인보조기기 지원은 평가 메모의 후보군(저소득 경증 장애인에게는 보조기기, 의료비, 장애수당 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003181",
                "경증장애인_저소득_보조기기: 장애인의료비지원은 평가 메모의 후보군(저소득 경증 장애인에게는 보조기기, 의료비, 장애수당 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000099",
                "경증장애인_저소득_보조기기: 저소득장애인 진단서 발급비 및 검사비 지원사업은 평가 메모의 후보군(저소득 경증 장애인에게는 보조기기, 의료비, 장애수당 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003265",
                "경증장애인_저소득_보조기기: 장애수당은 평가 메모의 후보군(저소득 경증 장애인에게는 보조기기, 의료비, 장애수당 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001062",
                "경증장애인_저소득_보조기기: 정보통신보조기기 보급은 평가 메모의 후보군(저소득 경증 장애인에게는 보조기기, 의료비, 장애수당 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "저소득 경증 장애인에게는 보조기기, 의료비, 장애수당 지원이 적절",
    ),
    EvalCase(
        "중증장애인_주거",
        SearchRequest(
            age=40,
            income_level="저소득",
            household_size=1,
            disability=True,
            disability_severity="중증",
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00003260",
                "중증장애인_주거: 장애인활동지원은 평가 메모의 후보군(중증 장애 1인 가구에는 활동지원, 자립, 주택개조, 안전안심 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00005024",
                "중증장애인_주거: 장애인 자립지원 시범사업은 평가 메모의 후보군(중증 장애 1인 가구에는 활동지원, 자립, 주택개조, 안전안심 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006260",
                "중증장애인_주거: 장애인 주택개조사업은 평가 메모의 후보군(중증 장애 1인 가구에는 활동지원, 자립, 주택개조, 안전안심 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003249",
                "중증장애인_주거: 장애인연금은 평가 메모의 후보군(중증 장애 1인 가구에는 활동지원, 자립, 주택개조, 안전안심 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001093",
                "중증장애인_주거: 독거노인·장애인 응급안전안심서비스은 평가 메모의 후보군(중증 장애 1인 가구에는 활동지원, 자립, 주택개조, 안전안심 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "중증 장애 1인 가구에는 활동지원, 자립, 주택개조, 안전안심 서비스가 적절",
        query="주거 주거급여 공공임대 전세 월세 주거비",
        intent_theme="housing",
    ),
    EvalCase(
        "장애인_청년_취업",
        SearchRequest(
            age=25,
            income_level="저소득",
            disability=True,
            disability_severity="중증",
            employment_status="실업",
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00000025",
                "장애인_청년_취업: 장애인일자리지원은 평가 메모의 후보군(실업 상태의 청년 중증 장애인에게는 장애인 취업/직업재활 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003189",
                "장애인_청년_취업: 중증장애인직업재활지원은 평가 메모의 후보군(실업 상태의 청년 중증 장애인에게는 장애인 취업/직업재활 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00004655",
                "장애인_청년_취업: 장애인취업성공패키지은 평가 메모의 후보군(실업 상태의 청년 중증 장애인에게는 장애인 취업/직업재활 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003259",
                "장애인_청년_취업: 중증장애인지원고용은 평가 메모의 후보군(실업 상태의 청년 중증 장애인에게는 장애인 취업/직업재활 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00004651",
                "장애인_청년_취업: 직업능력개발운영(훈련수당)은 평가 메모의 후보군(실업 상태의 청년 중증 장애인에게는 장애인 취업/직업재활 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001154",
                "장애인_청년_취업: 중증장애인근로자 출퇴근비용 지원 사업은 평가 메모의 후보군(실업 상태의 청년 중증 장애인에게는 장애인 취업/직업재활 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "실업 상태의 청년 중증 장애인에게는 장애인 취업/직업재활 서비스가 적절",
    ),
    EvalCase(
        "장애인_중장년_창업",
        SearchRequest(
            age=45,
            income_level="저소득",
            disability=True,
            disability_severity="경증",
            employment_status="실업",
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00003170",
                "장애인_중장년_창업: 장애인 창업점포 지원사업은 평가 메모의 후보군(중장년 장애인 실업자에게는 장애인 창업, 기업, 인턴, 고용 융자 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003174",
                "장애인_중장년_창업: 장애인창업육성은 평가 메모의 후보군(중장년 장애인 실업자에게는 장애인 창업, 기업, 인턴, 고용 융자 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001167",
                "장애인_중장년_창업: 장애인기업종합지원센터운영은 평가 메모의 후보군(중장년 장애인 실업자에게는 장애인 창업, 기업, 인턴, 고용 융자 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003210",
                "장애인_중장년_창업: 장애인기업 성장기반구축은 평가 메모의 후보군(중장년 장애인 실업자에게는 장애인 창업, 기업, 인턴, 고용 융자 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000068",
                "장애인_중장년_창업: 장애인고용증진융자은 평가 메모의 후보군(중장년 장애인 실업자에게는 장애인 창업, 기업, 인턴, 고용 융자 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00004654",
                "장애인_중장년_창업: 장애인인턴제은 평가 메모의 후보군(중장년 장애인 실업자에게는 장애인 창업, 기업, 인턴, 고용 융자 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "중장년 장애인 실업자에게는 장애인 창업, 기업, 인턴, 고용 융자 서비스가 적절",
        query="창업 점포 기업 인턴 고용 융자",
        intent_theme="startup",
    ),
    EvalCase(
        "비장애_청년_저소득_장애오탐방지",
        SearchRequest(age=24, income_level="저소득", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00004661",
                "비장애_청년_저소득_장애오탐방지: 청년월세 지원사업은 평가 메모의 후보군(비장애 청년에게 장애인 전용 급여/활동지원이 상위 노출되면 안 됨)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000060",
                "비장애_청년_저소득_장애오탐방지: 청년내일저축계좌은 평가 메모의 후보군(비장애 청년에게 장애인 전용 급여/활동지원이 상위 노출되면 안 됨)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003245",
                "비장애_청년_저소득_장애오탐방지: 국민취업지원제도은 평가 메모의 후보군(비장애 청년에게 장애인 전용 급여/활동지원이 상위 노출되면 안 됨)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003277",
                "비장애_청년_저소득_장애오탐방지: 취업 후 상환 학자금대출은 평가 메모의 후보군(비장애 청년에게 장애인 전용 급여/활동지원이 상위 노출되면 안 됨)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001076",
                "비장애_청년_저소득_장애오탐방지: 서민금융 활성화 지원(햇살론youth 보증사업)은 평가 메모의 후보군(비장애 청년에게 장애인 전용 급여/활동지원이 상위 노출되면 안 됨)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "비장애 청년에게 장애인 전용 급여/활동지원이 상위 노출되면 안 됨",
        excluded_ids=(
            excluded_svc(
                "WLF00003260",
                "비장애_청년_저소득_장애오탐방지: 장애인활동지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003249",
                "비장애_청년_저소득_장애오탐방지: 장애인연금은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003265",
                "비장애_청년_저소득_장애오탐방지: 장애수당은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003211",
                "비장애_청년_저소득_장애오탐방지: 장애인보조기기 교부은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003181",
                "비장애_청년_저소득_장애오탐방지: 장애인의료비지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00000025",
                "비장애_청년_저소득_장애오탐방지: 장애인일자리지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
        ),
        intent_theme="guardrail:not_disabled",
    ),
    EvalCase(
        "비장애_중장년_일반_장애오탐방지",
        SearchRequest(age=45, income_level="일반", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00003239",
                "비장애_중장년_일반_장애오탐방지: 고용복지플러스센터은 평가 메모의 후보군(일반 비장애 중장년에게 장애인 전용 서비스가 상위 노출되면 안 됨)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001172",
                "비장애_중장년_일반_장애오탐방지: 취업취약계층 고용지원 사업은 평가 메모의 후보군(일반 비장애 중장년에게 장애인 전용 서비스가 상위 노출되면 안 됨)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001083",
                "비장애_중장년_일반_장애오탐방지: 디지털배움터은 평가 메모의 후보군(일반 비장애 중장년에게 장애인 전용 서비스가 상위 노출되면 안 됨)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003264",
                "비장애_중장년_일반_장애오탐방지: 통합건강증진사업은 평가 메모의 후보군(일반 비장애 중장년에게 장애인 전용 서비스가 상위 노출되면 안 됨)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00005036",
                "비장애_중장년_일반_장애오탐방지: 개인채무조정은 평가 메모의 후보군(일반 비장애 중장년에게 장애인 전용 서비스가 상위 노출되면 안 됨)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "일반 비장애 중장년에게 장애인 전용 서비스가 상위 노출되면 안 됨",
        excluded_ids=(
            excluded_svc(
                "WLF00003260",
                "비장애_중장년_일반_장애오탐방지: 장애인활동지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003249",
                "비장애_중장년_일반_장애오탐방지: 장애인연금은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003265",
                "비장애_중장년_일반_장애오탐방지: 장애수당은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003189",
                "비장애_중장년_일반_장애오탐방지: 중증장애인직업재활지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00005031",
                "비장애_중장년_일반_장애오탐방지: 발달장애인 긴급돌봄사업은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
        ),
        intent_theme="guardrail:not_disabled",
    ),
    EvalCase(
        "기초수급_주거",
        SearchRequest(
            age=50,
            income_level="기초생활수급자",
            household_size=1,
            disability=False,
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00000888",
                "기초수급_주거: 영구임대주택공급은 평가 메모의 후보군(기초생활수급 1인 가구에는 주거급여와 공공임대/긴급 주거 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000062",
                "기초수급_주거: 기존주택등 매입임대주택 지원사업은 평가 메모의 후보군(기초생활수급 1인 가구에는 주거급여와 공공임대/긴급 주거 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003269",
                "기초수급_주거: 기존주택 전세임대주택 지원사업은 평가 메모의 후보군(기초생활수급 1인 가구에는 주거급여와 공공임대/긴급 주거 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003244",
                "기초수급_주거: 국민임대주택공급은 평가 메모의 후보군(기초생활수급 1인 가구에는 주거급여와 공공임대/긴급 주거 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000917",
                "기초수급_주거: 긴급복지 주거지원은 평가 메모의 후보군(기초생활수급 1인 가구에는 주거급여와 공공임대/긴급 주거 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "기초생활수급 1인 가구에는 주거급여와 공공임대/긴급 주거 지원이 적절",
        must_ids=(
            svc(
                "WLF00003201",
                "기초생활수급자와 1인 가구 입력은 주거급여의 주거 지원 대상성과 직접 관련된다.",
                "tgtr_dtl_cn",
            ),
        ),
        query="주거 주거급여 공공임대 전세 월세 주거비",
        intent_theme="housing",
    ),
    EvalCase(
        "기초수급_의료",
        SearchRequest(
            age=50, income_level="기초생활수급자", disability=False, pregnant=False, top_k=10
        ),
        (
            svc(
                "WLF00000102",
                "기초수급_의료: 의료급여(의료급여)은 평가 메모의 후보군(기초생활수급자는 의료급여와 의료급여 본인부담/건강검진 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003202",
                "기초수급_의료: 의료급여(본인부담 상한금)은 평가 메모의 후보군(기초생활수급자는 의료급여와 의료급여 본인부담/건강검진 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003218",
                "기초수급_의료: 의료급여(의료급여건강생활유지비)은 평가 메모의 후보군(기초생활수급자는 의료급여와 의료급여 본인부담/건강검진 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003228",
                "기초수급_의료: 의료급여 선택의료급여기관제은 평가 메모의 후보군(기초생활수급자는 의료급여와 의료급여 본인부담/건강검진 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003273",
                "기초수급_의료: 의료급여(본인부담 보상금)은 평가 메모의 후보군(기초생활수급자는 의료급여와 의료급여 본인부담/건강검진 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003238",
                "기초수급_의료: 의료급여수급권자 일반건강검진비 지원은 평가 메모의 후보군(기초생활수급자는 의료급여와 의료급여 본인부담/건강검진 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "기초생활수급자는 의료급여와 의료급여 본인부담/건강검진 지원이 적절",
        query="의료비 의료급여 건강검진 질환 치료비",
        intent_theme="medical",
    ),
    EvalCase(
        "기초수급_교육_아동",
        SearchRequest(
            age=12, income_level="기초생활수급자", disability=False, pregnant=False, top_k=10
        ),
        (
            svc(
                "WLF00001089",
                "기초수급_교육_아동: 교육급여(맞춤형 급여)은 평가 메모의 후보군(기초생활수급 아동에게는 교육급여, 교육비, 급식, 방과후 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001103",
                "기초수급_교육_아동: 초중고 교육비 지원사업(고교학비 지원)은 평가 메모의 후보군(기초생활수급 아동에게는 교육급여, 교육비, 급식, 방과후 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003227",
                "기초수급_교육_아동: 초.중.고 학생 교육정보화 지원은 평가 메모의 후보군(기초생활수급 아동에게는 교육급여, 교육비, 급식, 방과후 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001120",
                "기초수급_교육_아동: 교육복지우선지원사업은 평가 메모의 후보군(기초생활수급 아동에게는 교육급여, 교육비, 급식, 방과후 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001182",
                "기초수급_교육_아동: 학교우유급식은 평가 메모의 후보군(기초생활수급 아동에게는 교육급여, 교육비, 급식, 방과후 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000867",
                "기초수급_교육_아동: 방과후학교 자유수강권은 평가 메모의 후보군(기초생활수급 아동에게는 교육급여, 교육비, 급식, 방과후 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "기초생활수급 아동에게는 교육급여, 교육비, 급식, 방과후 지원이 적절",
        query="교육비 장학 방과후 교육정보화 급식",
        intent_theme="education",
    ),
    EvalCase(
        "기초수급_생계위기",
        SearchRequest(
            age=48,
            income_level="기초생활수급자",
            household_size=1,
            disability=False,
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00003180",
                "기초수급_생계위기: 긴급복지 생계지원은 평가 메모의 후보군(기초생활수급 생계위기 가구에는 생계급여와 긴급복지 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001132",
                "기초수급_생계위기: 생계급여(맞춤형 급여)은 평가 메모의 후보군(기초생활수급 생계위기 가구에는 생계급여와 긴급복지 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000892",
                "기초수급_생계위기: 긴급복지 장제비지원은 평가 메모의 후보군(기초생활수급 생계위기 가구에는 생계급여와 긴급복지 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000917",
                "기초수급_생계위기: 긴급복지 주거지원은 평가 메모의 후보군(기초생활수급 생계위기 가구에는 생계급여와 긴급복지 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001112",
                "기초수급_생계위기: 긴급복지 교육지원은 평가 메모의 후보군(기초생활수급 생계위기 가구에는 생계급여와 긴급복지 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003229",
                "기초수급_생계위기: 긴급복지 사회복지시설이용지원은 평가 메모의 후보군(기초생활수급 생계위기 가구에는 생계급여와 긴급복지 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003267",
                "기초수급_생계위기: 장제급여은 평가 메모의 후보군(기초생활수급 생계위기 가구에는 생계급여와 긴급복지 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "기초생활수급 생계위기 가구에는 생계급여와 긴급복지 계열이 적절",
        query="생계 생활지원 주거 감면 양곡 긴급복지",
        intent_theme="basic-living",
    ),
    EvalCase(
        "차상위_의료",
        SearchRequest(
            age=50, income_level="차상위계층", disability=False, pregnant=False, top_k=10
        ),
        (
            svc(
                "WLF00001119",
                "차상위_의료: 차상위본인부담경감대상자지원은 평가 메모의 후보군(차상위계층에는 본인부담 경감과 의료급여/재난적 의료비 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003202",
                "차상위_의료: 의료급여(본인부담 상한금)은 평가 메모의 후보군(차상위계층에는 본인부담 경감과 의료급여/재난적 의료비 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000102",
                "차상위_의료: 의료급여(의료급여)은 평가 메모의 후보군(차상위계층에는 본인부담 경감과 의료급여/재난적 의료비 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003171",
                "차상위_의료: 의료급여본인부담면제은 평가 메모의 후보군(차상위계층에는 본인부담 경감과 의료급여/재난적 의료비 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000923",
                "차상위_의료: 의료급여 중증질환, 희귀질환 및 중증난치질환자 지원은 평가 메모의 후보군(차상위계층에는 본인부담 경감과 의료급여/재난적 의료비 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003247",
                "차상위_의료: 재난적의료비 지원 사업은 평가 메모의 후보군(차상위계층에는 본인부담 경감과 의료급여/재난적 의료비 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "차상위계층에는 본인부담 경감과 의료급여/재난적 의료비 지원이 적절",
        query="의료비 의료급여 건강검진 질환 치료비",
        intent_theme="medical",
    ),
    EvalCase(
        "차상위_자활",
        SearchRequest(
            age=42,
            income_level="차상위계층",
            disability=False,
            employment_status="실업",
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00001138",
                "차상위_자활: 자활근로(기초, 차상위)은 평가 메모의 후보군(차상위 실업자는 자활근로, 지역자활센터, 취업지원 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001157",
                "차상위_자활: 지역자활센터 운영은 평가 메모의 후보군(차상위 실업자는 자활근로, 지역자활센터, 취업지원 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006196",
                "차상위_자활: 자활성공지원금 지급·관리은 평가 메모의 후보군(차상위 실업자는 자활근로, 지역자활센터, 취업지원 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003245",
                "차상위_자활: 국민취업지원제도은 평가 메모의 후보군(차상위 실업자는 자활근로, 지역자활센터, 취업지원 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003266",
                "차상위_자활: 직업훈련생계비대부은 평가 메모의 후보군(차상위 실업자는 자활근로, 지역자활센터, 취업지원 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "차상위 실업자는 자활근로, 지역자활센터, 취업지원 서비스가 적절",
        query="취업 구직 직업훈련 고용센터 자활근로",
        intent_theme="employment",
    ),
    EvalCase(
        "차상위_교육_청소년",
        SearchRequest(
            age=16, income_level="차상위계층", disability=False, pregnant=False, top_k=10
        ),
        (
            svc(
                "WLF00001103",
                "차상위_교육_청소년: 초중고 교육비 지원사업(고교학비 지원)은 평가 메모의 후보군(차상위 청소년에게는 교육비, 방과후, 생리용품, 특별지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001089",
                "차상위_교육_청소년: 교육급여(맞춤형 급여)은 평가 메모의 후보군(차상위 청소년에게는 교육비, 방과후, 생리용품, 특별지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000867",
                "차상위_교육_청소년: 방과후학교 자유수강권은 평가 메모의 후보군(차상위 청소년에게는 교육비, 방과후, 생리용품, 특별지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001120",
                "차상위_교육_청소년: 교육복지우선지원사업은 평가 메모의 후보군(차상위 청소년에게는 교육비, 방과후, 생리용품, 특별지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000781",
                "차상위_교육_청소년: 여성청소년 생리용품 지원은 평가 메모의 후보군(차상위 청소년에게는 교육비, 방과후, 생리용품, 특별지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000078",
                "차상위_교육_청소년: 청소년특별지원은 평가 메모의 후보군(차상위 청소년에게는 교육비, 방과후, 생리용품, 특별지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "차상위 청소년에게는 교육비, 방과후, 생리용품, 특별지원이 적절",
        query="교육비 장학 방과후 교육정보화 급식",
        intent_theme="education",
    ),
    EvalCase(
        "저소득_에너지",
        SearchRequest(
            age=50,
            income_level="저소득",
            household_size=1,
            disability=False,
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00000072",
                "저소득_에너지: 에너지바우처은 평가 메모의 후보군(저소득 1인 가구에는 에너지바우처, 에너지효율개선, 요금감면이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001128",
                "저소득_에너지: 저소득층에너지효율개선은 평가 메모의 후보군(저소득 1인 가구에는 에너지바우처, 에너지효율개선, 요금감면이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000049",
                "저소득_에너지: 에너지 취약계층 고효율조명기기 무상교체 지원(취약계층 에너지복지사업)은 평가 메모의 후보군(저소득 1인 가구에는 에너지바우처, 에너지효율개선, 요금감면이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00004639",
                "저소득_에너지: 전기요금 복지할인은 평가 메모의 후보군(저소득 1인 가구에는 에너지바우처, 에너지효율개선, 요금감면이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00005444",
                "저소득_에너지: 취약계층 고효율가전 구매지원(취약계층 에너지복지사업)은 평가 메모의 후보군(저소득 1인 가구에는 에너지바우처, 에너지효율개선, 요금감면이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006265",
                "저소득_에너지: 연탄쿠폰은 평가 메모의 후보군(저소득 1인 가구에는 에너지바우처, 에너지효율개선, 요금감면이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006224",
                "저소득_에너지: 저소득층 수도요금감면은 평가 메모의 후보군(저소득 1인 가구에는 에너지바우처, 에너지효율개선, 요금감면이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "저소득 1인 가구에는 에너지바우처, 에너지효율개선, 요금감면이 적절",
        query="에너지 바우처 효율개선 전기요금 난방 수도요금",
        intent_theme="energy",
    ),
    EvalCase(
        "저소득_문화",
        SearchRequest(age=30, income_level="저소득", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00000055",
                "저소득_문화: 통합문화이용권은 평가 메모의 후보군(저소득 성인에게는 문화·여가·교육 바우처 계열 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000076",
                "저소득_문화: 스포츠강좌이용권은 평가 메모의 후보군(저소득 성인에게는 문화·여가·교육 바우처 계열 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000946",
                "저소득_문화: 산림복지서비스이용권은 평가 메모의 후보군(저소득 성인에게는 문화·여가·교육 바우처 계열 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003199",
                "저소득_문화: 예술활동준비금 지원은 평가 메모의 후보군(저소득 성인에게는 문화·여가·교육 바우처 계열 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00004658",
                "저소득_문화: 과학문화 바우처 지원은 평가 메모의 후보군(저소득 성인에게는 문화·여가·교육 바우처 계열 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003240",
                "저소득_문화: 평생교육이용권 지원은 평가 메모의 후보군(저소득 성인에게는 문화·여가·교육 바우처 계열 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "저소득 성인에게는 문화·여가·교육 바우처 계열 서비스가 적절",
        query="문화 여가 스포츠 교육 바우처 이용권",
        intent_theme="culture",
    ),
    EvalCase(
        "저소득_금융",
        SearchRequest(age=30, income_level="저소득", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00000100",
                "저소득_금융: 자산형성지원사업(희망저축계좌Ⅰ, Ⅱ)은 평가 메모의 후보군(저소득 성인에게는 자산형성, 학자금, 서민금융, 신용·부채관리 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000060",
                "저소득_금융: 청년내일저축계좌은 평가 메모의 후보군(저소득 성인에게는 자산형성, 학자금, 서민금융, 신용·부채관리 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003277",
                "저소득_금융: 취업 후 상환 학자금대출은 평가 메모의 후보군(저소득 성인에게는 자산형성, 학자금, 서민금융, 신용·부채관리 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006268",
                "저소득_금융: 불법사금융예방대출은 평가 메모의 후보군(저소득 성인에게는 자산형성, 학자금, 서민금융, 신용·부채관리 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006273",
                "저소득_금융: 햇살론특례은 평가 메모의 후보군(저소득 성인에게는 자산형성, 학자금, 서민금융, 신용·부채관리 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006274",
                "저소득_금융: 햇살론일반은 평가 메모의 후보군(저소득 성인에게는 자산형성, 학자금, 서민금융, 신용·부채관리 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006277",
                "저소득_금융: 서민금융진흥원 신용･부채관리 컨설팅은 평가 메모의 후보군(저소득 성인에게는 자산형성, 학자금, 서민금융, 신용·부채관리 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "저소득 성인에게는 자산형성, 학자금, 서민금융, 신용·부채관리 서비스가 적절",
        query="자산형성 학자금 서민금융 대출 신용 부채관리",
        intent_theme="finance",
    ),
    EvalCase(
        "저소득_의료비",
        SearchRequest(age=50, income_level="저소득", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00000864",
                "저소득_의료비: 희귀질환자 의료비 지원사업은 평가 메모의 후보군(저소득 성인에게는 의료급여, 희귀질환, 재난적 의료비 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003247",
                "저소득_의료비: 재난적의료비 지원 사업은 평가 메모의 후보군(저소득 성인에게는 의료급여, 희귀질환, 재난적 의료비 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000102",
                "저소득_의료비: 의료급여(의료급여)은 평가 메모의 후보군(저소득 성인에게는 의료급여, 희귀질환, 재난적 의료비 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000923",
                "저소득_의료비: 의료급여 중증질환, 희귀질환 및 중증난치질환자 지원은 평가 메모의 후보군(저소득 성인에게는 의료급여, 희귀질환, 재난적 의료비 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003202",
                "저소득_의료비: 의료급여(본인부담 상한금)은 평가 메모의 후보군(저소득 성인에게는 의료급여, 희귀질환, 재난적 의료비 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003238",
                "저소득_의료비: 의료급여수급권자 일반건강검진비 지원은 평가 메모의 후보군(저소득 성인에게는 의료급여, 희귀질환, 재난적 의료비 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "저소득 성인에게는 의료급여, 희귀질환, 재난적 의료비 지원이 적절",
        query="의료비 의료급여 건강검진 질환 치료비",
        intent_theme="medical",
    ),
    EvalCase(
        "저소득_1인가구_안부안전",
        SearchRequest(
            age=50,
            income_level="저소득",
            household_size=1,
            disability=False,
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00001087",
                "저소득_1인가구_안부안전: 가사·간병 방문 지원사업은 평가 메모의 후보군(저소득 1인 가구에는 방문돌봄, 안부확인, 긴급돌봄, 통합사례관리가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001093",
                "저소득_1인가구_안부안전: 독거노인·장애인 응급안전안심서비스은 평가 메모의 후보군(저소득 1인 가구에는 방문돌봄, 안부확인, 긴급돌봄, 통합사례관리가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006237",
                "저소득_1인가구_안부안전: 안부살핌 우편서비스 사업은 평가 메모의 후보군(저소득 1인 가구에는 방문돌봄, 안부확인, 긴급돌봄, 통합사례관리가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00005442",
                "저소득_1인가구_안부안전: 긴급돌봄 지원사업은 평가 메모의 후보군(저소득 1인 가구에는 방문돌봄, 안부확인, 긴급돌봄, 통합사례관리가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000059",
                "저소득_1인가구_안부안전: 희망복지지원단 통합사례관리은 평가 메모의 후보군(저소득 1인 가구에는 방문돌봄, 안부확인, 긴급돌봄, 통합사례관리가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003229",
                "저소득_1인가구_안부안전: 긴급복지 사회복지시설이용지원은 평가 메모의 후보군(저소득 1인 가구에는 방문돌봄, 안부확인, 긴급돌봄, 통합사례관리가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "저소득 1인 가구에는 방문돌봄, 안부확인, 긴급돌봄, 통합사례관리가 적절",
        query="돌봄 안부 안전 장기요양 방문 재가",
        intent_theme="care",
    ),
    EvalCase(
        "일반_중장년_특수대상오탐방지",
        SearchRequest(age=45, income_level="일반", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00001083",
                "일반_중장년_특수대상오탐방지: 디지털배움터은 평가 메모의 후보군(일반 중장년에게 보훈/북한이탈/다문화/입양·위탁/농어업/청소년부모 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003239",
                "일반_중장년_특수대상오탐방지: 고용복지플러스센터은 평가 메모의 후보군(일반 중장년에게 보훈/북한이탈/다문화/입양·위탁/농어업/청소년부모 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003264",
                "일반_중장년_특수대상오탐방지: 통합건강증진사업은 평가 메모의 후보군(일반 중장년에게 보훈/북한이탈/다문화/입양·위탁/농어업/청소년부모 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00005036",
                "일반_중장년_특수대상오탐방지: 개인채무조정은 평가 메모의 후보군(일반 중장년에게 보훈/북한이탈/다문화/입양·위탁/농어업/청소년부모 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00004997",
                "일반_중장년_특수대상오탐방지: 한국형 상병수당 시범사업은 평가 메모의 후보군(일반 중장년에게 보훈/북한이탈/다문화/입양·위탁/농어업/청소년부모 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "일반 중장년에게 보훈/북한이탈/다문화/입양·위탁/농어업/청소년부모 전용 서비스는 오탐",
        excluded_ids=(
            excluded_svc(
                "WLF00000048",
                "일반_중장년_특수대상오탐방지: 국가유공자등대부지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00000063",
                "일반_중장년_특수대상오탐방지: (북한이탈주민)교육비 지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00000837",
                "일반_중장년_특수대상오탐방지: 다문화가족 지원사업은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00001116",
                "일반_중장년_특수대상오탐방지: 입양비용지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00000023",
                "일반_중장년_특수대상오탐방지: 농어가목돈마련저축 저축장려금 지급은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00005023",
                "일반_중장년_특수대상오탐방지: 청소년부모 아동양육비 지원사업은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
        ),
        intent_theme="guardrail:special_target",
    ),
    EvalCase(
        "일반_청년_특수대상오탐방지",
        SearchRequest(age=25, income_level="일반", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00001076",
                "일반_청년_특수대상오탐방지: 서민금융 활성화 지원(햇살론youth 보증사업)은 평가 메모의 후보군(일반 청년에게 특수대상 전용 서비스가 상위 노출되면 안 됨)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003277",
                "일반_청년_특수대상오탐방지: 취업 후 상환 학자금대출은 평가 메모의 후보군(일반 청년에게 특수대상 전용 서비스가 상위 노출되면 안 됨)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003239",
                "일반_청년_특수대상오탐방지: 고용복지플러스센터은 평가 메모의 후보군(일반 청년에게 특수대상 전용 서비스가 상위 노출되면 안 됨)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006229",
                "일반_청년_특수대상오탐방지: 국민내일배움카드제 직업훈련지원(훈련비, 훈련장려금)은 평가 메모의 후보군(일반 청년에게 특수대상 전용 서비스가 상위 노출되면 안 됨)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006249",
                "일반_청년_특수대상오탐방지: 해외취업 지원은 평가 메모의 후보군(일반 청년에게 특수대상 전용 서비스가 상위 노출되면 안 됨)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006267",
                "일반_청년_특수대상오탐방지: 서민금융진흥원 금융교육은 평가 메모의 후보군(일반 청년에게 특수대상 전용 서비스가 상위 노출되면 안 됨)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "일반 청년에게 특수대상 전용 서비스가 상위 노출되면 안 됨",
        excluded_ids=(
            excluded_svc(
                "WLF00000054",
                "일반_청년_특수대상오탐방지: 국가보훈대상자 보훈장학금은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00000086",
                "일반_청년_특수대상오탐방지: (북한이탈주민) 탈북청소년 교육지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003192",
                "일반_청년_특수대상오탐방지: 다문화가족 방문교육 서비스은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003235",
                "일반_청년_특수대상오탐방지: 입양아동 양육수당 지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00001096",
                "일반_청년_특수대상오탐방지: 농촌출신대학생학자금융자은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00005023",
                "일반_청년_특수대상오탐방지: 청소년부모 아동양육비 지원사업은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
        ),
        intent_theme="guardrail:special_target",
    ),
    EvalCase(
        "일반_아동_입양위탁오탐방지",
        SearchRequest(age=10, income_level="일반", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00001171",
                "일반_아동_입양위탁오탐방지: 아동수당 지급은 평가 메모의 후보군(입양·위탁 정보가 없는 일반 아동에게 입양·위탁 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000089",
                "일반_아동_입양위탁오탐방지: 다함께 돌봄 사업은 평가 메모의 후보군(입양·위탁 정보가 없는 일반 아동에게 입양·위탁 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003223",
                "일반_아동_입양위탁오탐방지: 고난도 보호대상아동 맞춤형 사례관리서비스은 평가 메모의 후보군(입양·위탁 정보가 없는 일반 아동에게 입양·위탁 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003242",
                "일반_아동_입양위탁오탐방지: 국가예방접종 사업은 평가 메모의 후보군(입양·위탁 정보가 없는 일반 아동에게 입양·위탁 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001141",
                "일반_아동_입양위탁오탐방지: 지역사회 청소년통합지원체계(청소년안전망)은 평가 메모의 후보군(입양·위탁 정보가 없는 일반 아동에게 입양·위탁 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "입양·위탁 정보가 없는 일반 아동에게 입양·위탁 전용 서비스는 오탐",
        excluded_ids=(
            excluded_svc(
                "WLF00001116",
                "일반_아동_입양위탁오탐방지: 입양비용지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003235",
                "일반_아동_입양위탁오탐방지: 입양아동 양육수당 지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003268",
                "일반_아동_입양위탁오탐방지: 가정위탁아동 상해보험료 지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003270",
                "일반_아동_입양위탁오탐방지: 입양·가정위탁아동 심리치료 지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00005032",
                "일반_아동_입양위탁오탐방지: 전문아동보호비 지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00005033",
                "일반_아동_입양위탁오탐방지: 아동용품구입비 지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00005445",
                "일반_아동_입양위탁오탐방지: 자립준비청년(보호종료아동) 자립정착금 지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
        ),
        intent_theme="guardrail:special_target",
    ),
    EvalCase(
        "일반_임신아님_출산오탐방지",
        SearchRequest(age=30, income_level="일반", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00001083",
                "일반_임신아님_출산오탐방지: 디지털배움터은 평가 메모의 후보군(임신 정보가 없는 일반 성인에게 임신·출산 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003264",
                "일반_임신아님_출산오탐방지: 통합건강증진사업은 평가 메모의 후보군(임신 정보가 없는 일반 성인에게 임신·출산 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00005036",
                "일반_임신아님_출산오탐방지: 개인채무조정은 평가 메모의 후보군(임신 정보가 없는 일반 성인에게 임신·출산 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003239",
                "일반_임신아님_출산오탐방지: 고용복지플러스센터은 평가 메모의 후보군(임신 정보가 없는 일반 성인에게 임신·출산 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006267",
                "일반_임신아님_출산오탐방지: 서민금융진흥원 금융교육은 평가 메모의 후보군(임신 정보가 없는 일반 성인에게 임신·출산 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "임신 정보가 없는 일반 성인에게 임신·출산 전용 서비스는 오탐",
        excluded_ids=(
            excluded_svc(
                "WLF00001088",
                "일반_임신아님_출산오탐방지: 고위험 임산부 의료비 지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00000061",
                "일반_임신아님_출산오탐방지: 의료급여임신.출산진료비지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00001135",
                "일반_임신아님_출산오탐방지: 해산급여은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003178",
                "일반_임신아님_출산오탐방지: 긴급복지 해산비지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003246",
                "일반_임신아님_출산오탐방지: 청소년산모 임신·출산 의료비 지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003278",
                "일반_임신아님_출산오탐방지: 여성장애인 출산비용지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
        ),
        intent_theme="guardrail:not_pregnant",
    ),
    EvalCase(
        "자녀없음_중장년_보육오탐방지",
        SearchRequest(
            age=45,
            income_level="저소득",
            has_children=False,
            disability=False,
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00001132",
                "자녀없음_중장년_보육오탐방지: 생계급여(맞춤형 급여)은 평가 메모의 후보군(자녀 없음 중장년에게 보육/아동 전용 서비스가 상위 노출되면 안 됨)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003201",
                "자녀없음_중장년_보육오탐방지: 주거급여(맞춤형 급여)은 평가 메모의 후보군(자녀 없음 중장년에게 보육/아동 전용 서비스가 상위 노출되면 안 됨)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000072",
                "자녀없음_중장년_보육오탐방지: 에너지바우처은 평가 메모의 후보군(자녀 없음 중장년에게 보육/아동 전용 서비스가 상위 노출되면 안 됨)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003257",
                "자녀없음_중장년_보육오탐방지: 이동통신요금감면은 평가 메모의 후보군(자녀 없음 중장년에게 보육/아동 전용 서비스가 상위 노출되면 안 됨)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003180",
                "자녀없음_중장년_보육오탐방지: 긴급복지 생계지원은 평가 메모의 후보군(자녀 없음 중장년에게 보육/아동 전용 서비스가 상위 노출되면 안 됨)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "자녀 없음 중장년에게 보육/아동 전용 서비스가 상위 노출되면 안 됨",
        excluded_ids=(
            excluded_svc(
                "WLF00001068",
                "자녀없음_중장년_보육오탐방지: 한부모가족 아동양육비 지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00000024",
                "자녀없음_중장년_보육오탐방지: 아이돌봄서비스은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003250",
                "자녀없음_중장년_보육오탐방지: 영유아보육료 지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003253",
                "자녀없음_중장년_보육오탐방지: 가정양육수당 지원사업은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00001171",
                "자녀없음_중장년_보육오탐방지: 아동수당 지급은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00005023",
                "자녀없음_중장년_보육오탐방지: 청소년부모 아동양육비 지원사업은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
        ),
        intent_theme="guardrail:no_children_or_parenting",
    ),
    EvalCase(
        "미혼_청년_한부모오탐방지",
        SearchRequest(
            age=27,
            income_level="일반",
            marital_status="미혼",
            has_children=False,
            disability=False,
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00001076",
                "미혼_청년_한부모오탐방지: 서민금융 활성화 지원(햇살론youth 보증사업)은 평가 메모의 후보군(자녀가 없는 미혼 청년에게 한부모·청소년부모 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003277",
                "미혼_청년_한부모오탐방지: 취업 후 상환 학자금대출은 평가 메모의 후보군(자녀가 없는 미혼 청년에게 한부모·청소년부모 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003239",
                "미혼_청년_한부모오탐방지: 고용복지플러스센터은 평가 메모의 후보군(자녀가 없는 미혼 청년에게 한부모·청소년부모 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006229",
                "미혼_청년_한부모오탐방지: 국민내일배움카드제 직업훈련지원(훈련비, 훈련장려금)은 평가 메모의 후보군(자녀가 없는 미혼 청년에게 한부모·청소년부모 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006267",
                "미혼_청년_한부모오탐방지: 서민금융진흥원 금융교육은 평가 메모의 후보군(자녀가 없는 미혼 청년에게 한부모·청소년부모 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "자녀가 없는 미혼 청년에게 한부모·청소년부모 전용 서비스는 오탐",
        excluded_ids=(
            excluded_svc(
                "WLF00001068",
                "미혼_청년_한부모오탐방지: 한부모가족 아동양육비 지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00000091",
                "미혼_청년_한부모오탐방지: 한부모가족복지시설 지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00001109",
                "미혼_청년_한부모오탐방지: 청소년한부모 아동양육 및 자립지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00005856",
                "미혼_청년_한부모오탐방지: 양육비 선지급은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00006278",
                "미혼_청년_한부모오탐방지: 서민금융진흥원 소액보험(한부모가정의료보험)은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00005023",
                "미혼_청년_한부모오탐방지: 청소년부모 아동양육비 지원사업은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
        ),
        intent_theme="guardrail:no_children_or_parenting",
    ),
    EvalCase(
        "저소득_실업_직업훈련",
        SearchRequest(
            age=30,
            income_level="저소득",
            disability=False,
            employment_status="실업",
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00003245",
                "저소득_실업_직업훈련: 국민취업지원제도은 평가 메모의 후보군(저소득 실업자는 국민취업지원, 직업훈련, 자활근로 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003266",
                "저소득_실업_직업훈련: 직업훈련생계비대부은 평가 메모의 후보군(저소득 실업자는 국민취업지원, 직업훈련, 자활근로 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001138",
                "저소득_실업_직업훈련: 자활근로(기초, 차상위)은 평가 메모의 후보군(저소득 실업자는 국민취업지원, 직업훈련, 자활근로 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001157",
                "저소득_실업_직업훈련: 지역자활센터 운영은 평가 메모의 후보군(저소득 실업자는 국민취업지원, 직업훈련, 자활근로 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006229",
                "저소득_실업_직업훈련: 국민내일배움카드제 직업훈련지원(훈련비, 훈련장려금)은 평가 메모의 후보군(저소득 실업자는 국민취업지원, 직업훈련, 자활근로 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006196",
                "저소득_실업_직업훈련: 자활성공지원금 지급·관리은 평가 메모의 후보군(저소득 실업자는 국민취업지원, 직업훈련, 자활근로 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "저소득 실업자는 국민취업지원, 직업훈련, 자활근로 서비스가 적절",
        query="취업 구직 직업훈련 고용센터 자활근로",
        intent_theme="employment",
    ),
    EvalCase(
        "일반_실업_고용센터",
        SearchRequest(
            age=30,
            income_level="일반",
            disability=False,
            employment_status="실업",
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00003239",
                "일반_실업_고용센터: 고용복지플러스센터은 평가 메모의 후보군(일반 실업자에게는 고용센터, 취업취약계층, 직업훈련, 재취업 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001172",
                "일반_실업_고용센터: 취업취약계층 고용지원 사업은 평가 메모의 후보군(일반 실업자에게는 고용센터, 취업취약계층, 직업훈련, 재취업 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006229",
                "일반_실업_고용센터: 국민내일배움카드제 직업훈련지원(훈련비, 훈련장려금)은 평가 메모의 후보군(일반 실업자에게는 고용센터, 취업취약계층, 직업훈련, 재취업 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001156",
                "일반_실업_고용센터: 재취업지원서비스 시행지원은 평가 메모의 후보군(일반 실업자에게는 고용센터, 취업취약계층, 직업훈련, 재취업 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00005852",
                "일반_실업_고용센터: 중장년 경력지원제은 평가 메모의 후보군(일반 실업자에게는 고용센터, 취업취약계층, 직업훈련, 재취업 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006249",
                "일반_실업_고용센터: 해외취업 지원은 평가 메모의 후보군(일반 실업자에게는 고용센터, 취업취약계층, 직업훈련, 재취업 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "일반 실업자에게는 고용센터, 취업취약계층, 직업훈련, 재취업 서비스가 적절",
        query="취업 구직 직업훈련 고용센터 자활근로",
        intent_theme="employment",
    ),
    EvalCase(
        "중장년_실업_재취업",
        SearchRequest(
            age=52,
            income_level="일반",
            disability=False,
            employment_status="실업",
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00005852",
                "중장년_실업_재취업: 중장년 경력지원제은 평가 메모의 후보군(중장년 실업자에게는 재취업, 경력지원, 기술창업, 직업훈련 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003205",
                "중장년_실업_재취업: 중장년 기술창업센터 지원사업은 평가 메모의 후보군(중장년 실업자에게는 재취업, 경력지원, 기술창업, 직업훈련 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001156",
                "중장년_실업_재취업: 재취업지원서비스 시행지원은 평가 메모의 후보군(중장년 실업자에게는 재취업, 경력지원, 기술창업, 직업훈련 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003239",
                "중장년_실업_재취업: 고용복지플러스센터은 평가 메모의 후보군(중장년 실업자에게는 재취업, 경력지원, 기술창업, 직업훈련 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001172",
                "중장년_실업_재취업: 취업취약계층 고용지원 사업은 평가 메모의 후보군(중장년 실업자에게는 재취업, 경력지원, 기술창업, 직업훈련 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006229",
                "중장년_실업_재취업: 국민내일배움카드제 직업훈련지원(훈련비, 훈련장려금)은 평가 메모의 후보군(중장년 실업자에게는 재취업, 경력지원, 기술창업, 직업훈련 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "중장년 실업자에게는 재취업, 경력지원, 기술창업, 직업훈련 서비스가 적절",
        query="취업 구직 직업훈련 고용센터 자활근로",
        intent_theme="employment",
    ),
    EvalCase(
        "청년_취업_금융",
        SearchRequest(
            age=28,
            income_level="일반",
            disability=False,
            employment_status="취업",
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00006215",
                "청년_취업_금융: 청년내일채움공제은 평가 메모의 후보군(취업 중인 청년에게는 청년 자산형성, 학자금, 금융교육 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003277",
                "청년_취업_금융: 취업 후 상환 학자금대출은 평가 메모의 후보군(취업 중인 청년에게는 청년 자산형성, 학자금, 금융교육 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001076",
                "청년_취업_금융: 서민금융 활성화 지원(햇살론youth 보증사업)은 평가 메모의 후보군(취업 중인 청년에게는 청년 자산형성, 학자금, 금융교육 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006267",
                "청년_취업_금융: 서민금융진흥원 금융교육은 평가 메모의 후보군(취업 중인 청년에게는 청년 자산형성, 학자금, 금융교육 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006277",
                "청년_취업_금융: 서민금융진흥원 신용･부채관리 컨설팅은 평가 메모의 후보군(취업 중인 청년에게는 청년 자산형성, 학자금, 금융교육 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "취업 중인 청년에게는 청년 자산형성, 학자금, 금융교육 서비스가 적절",
        query="자산형성 학자금 서민금융 대출 신용 부채관리",
        intent_theme="finance",
    ),
    EvalCase(
        "취업_저소득_근로장려",
        SearchRequest(
            age=45,
            income_level="저소득",
            disability=False,
            employment_status="취업",
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00001148",
                "취업_저소득_근로장려: 근로·자녀장려금은 평가 메모의 후보군(취업 중인 저소득 중장년에게는 근로장려, 사회보험, 생활안정자금이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000095",
                "취업_저소득_근로장려: 사회보험사각지대해소 사업(두루누리 사회보험료 지원사업)은 평가 메모의 후보군(취업 중인 저소득 중장년에게는 근로장려, 사회보험, 생활안정자금이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000044",
                "취업_저소득_근로장려: 생활안정자금(융자)은 평가 메모의 후보군(취업 중인 저소득 중장년에게는 근로장려, 사회보험, 생활안정자금이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006264",
                "취업_저소득_근로장려: 생활안정자금(이차보전)은 평가 메모의 후보군(취업 중인 저소득 중장년에게는 근로장려, 사회보험, 생활안정자금이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006258",
                "취업_저소득_근로장려: 건강보험 임의계속가입제도은 평가 메모의 후보군(취업 중인 저소득 중장년에게는 근로장려, 사회보험, 생활안정자금이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "취업 중인 저소득 중장년에게는 근로장려, 사회보험, 생활안정자금이 적절",
    ),
    EvalCase(
        "비경제활동_저소득_생활지원",
        SearchRequest(
            age=45,
            income_level="저소득",
            disability=False,
            employment_status="비경제활동",
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00001132",
                "비경제활동_저소득_생활지원: 생계급여(맞춤형 급여)은 평가 메모의 후보군(비경제활동 저소득 성인에게는 생활/주거/에너지/문화/양곡 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003201",
                "비경제활동_저소득_생활지원: 주거급여(맞춤형 급여)은 평가 메모의 후보군(비경제활동 저소득 성인에게는 생활/주거/에너지/문화/양곡 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000072",
                "비경제활동_저소득_생활지원: 에너지바우처은 평가 메모의 후보군(비경제활동 저소득 성인에게는 생활/주거/에너지/문화/양곡 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000055",
                "비경제활동_저소득_생활지원: 통합문화이용권은 평가 메모의 후보군(비경제활동 저소득 성인에게는 생활/주거/에너지/문화/양곡 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000074",
                "비경제활동_저소득_생활지원: 양곡할인은 평가 메모의 후보군(비경제활동 저소득 성인에게는 생활/주거/에너지/문화/양곡 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003180",
                "비경제활동_저소득_생활지원: 긴급복지 생계지원은 평가 메모의 후보군(비경제활동 저소득 성인에게는 생활/주거/에너지/문화/양곡 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "비경제활동 저소득 성인에게는 생활/주거/에너지/문화/양곡 지원이 적절",
        query="생계 생활지원 주거 감면 양곡 긴급복지",
        intent_theme="basic-living",
    ),
    EvalCase(
        "노인_저소득_의료",
        SearchRequest(age=70, income_level="저소득", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00001115",
                "노인_저소득_의료: 노인 개안수술비 지원은 평가 메모의 후보군(저소득 노인에게는 노인성 질환, 치과, 치매, 재난적 의료비 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001169",
                "노인_저소득_의료: 의료급여 틀니·치과임플란트은 평가 메모의 후보군(저소득 노인에게는 노인성 질환, 치과, 치매, 재난적 의료비 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001179",
                "노인_저소득_의료: 노인 무릎인공관절 수술 지원 사업은 평가 메모의 후보군(저소득 노인에게는 노인성 질환, 치과, 치매, 재난적 의료비 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00005004",
                "노인_저소득_의료: 치매검사비 지원은 평가 메모의 후보군(저소득 노인에게는 노인성 질환, 치과, 치매, 재난적 의료비 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003280",
                "노인_저소득_의료: 전립선등 노인성질환 예방관리은 평가 메모의 후보군(저소득 노인에게는 노인성 질환, 치과, 치매, 재난적 의료비 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003247",
                "노인_저소득_의료: 재난적의료비 지원 사업은 평가 메모의 후보군(저소득 노인에게는 노인성 질환, 치과, 치매, 재난적 의료비 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "저소득 노인에게는 노인성 질환, 치과, 치매, 재난적 의료비 지원이 적절",
        query="의료비 의료급여 건강검진 질환 치료비",
        intent_theme="medical",
    ),
    EvalCase(
        "노인_저소득_돌봄",
        SearchRequest(
            age=75,
            income_level="저소득",
            household_size=1,
            disability=False,
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00003191",
                "노인_저소득_돌봄: 노인맞춤돌봄서비스은 평가 메모의 후보군(저소득 독거 노인에게는 맞춤돌봄, 응급안전, 재가/시설 돌봄이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001093",
                "노인_저소득_돌봄: 독거노인·장애인 응급안전안심서비스은 평가 메모의 후보군(저소득 독거 노인에게는 맞춤돌봄, 응급안전, 재가/시설 돌봄이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001086",
                "노인_저소득_돌봄: 특별현금급여(가족요양비)은 평가 메모의 후보군(저소득 독거 노인에게는 맞춤돌봄, 응급안전, 재가/시설 돌봄이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003194",
                "노인_저소득_돌봄: 시설급여은 평가 메모의 후보군(저소득 독거 노인에게는 맞춤돌봄, 응급안전, 재가/시설 돌봄이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003248",
                "노인_저소득_돌봄: 재가급여은 평가 메모의 후보군(저소득 독거 노인에게는 맞춤돌봄, 응급안전, 재가/시설 돌봄이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00005442",
                "노인_저소득_돌봄: 긴급돌봄 지원사업은 평가 메모의 후보군(저소득 독거 노인에게는 맞춤돌봄, 응급안전, 재가/시설 돌봄이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "저소득 독거 노인에게는 맞춤돌봄, 응급안전, 재가/시설 돌봄이 적절",
        query="돌봄 안부 안전 장기요양 방문 재가",
        intent_theme="care",
    ),
    EvalCase(
        "노인_일반_장기요양",
        SearchRequest(age=78, income_level="일반", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00000103",
                "노인_일반_장기요양: 노인장기요양보험 복지용구 급여은 평가 메모의 후보군(일반 노인에게는 장기요양 복지용구, 시설급여, 재가급여 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001177",
                "노인_일반_장기요양: 장기요양 본인부담금 감경은 평가 메모의 후보군(일반 노인에게는 장기요양 복지용구, 시설급여, 재가급여 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003194",
                "노인_일반_장기요양: 시설급여은 평가 메모의 후보군(일반 노인에게는 장기요양 복지용구, 시설급여, 재가급여 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003248",
                "노인_일반_장기요양: 재가급여은 평가 메모의 후보군(일반 노인에게는 장기요양 복지용구, 시설급여, 재가급여 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001086",
                "노인_일반_장기요양: 특별현금급여(가족요양비)은 평가 메모의 후보군(일반 노인에게는 장기요양 복지용구, 시설급여, 재가급여 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001181",
                "노인_일반_장기요양: 중앙노인돌봄지원기관 운영지원은 평가 메모의 후보군(일반 노인에게는 장기요양 복지용구, 시설급여, 재가급여 계열이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "일반 노인에게는 장기요양 복지용구, 시설급여, 재가급여 계열이 적절",
        query="돌봄 안부 안전 장기요양 방문 재가",
        intent_theme="care",
    ),
    EvalCase(
        "노인_보훈아님_보훈오탐방지",
        SearchRequest(age=70, income_level="일반", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00000031",
                "노인_보훈아님_보훈오탐방지: 노후준비서비스은 평가 메모의 후보군(보훈 정보가 없는 일반 노인에게 보훈대상자 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001108",
                "노인_보훈아님_보훈오탐방지: 주택담보노후연금보증은 평가 메모의 후보군(보훈 정보가 없는 일반 노인에게 보훈대상자 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000103",
                "노인_보훈아님_보훈오탐방지: 노인장기요양보험 복지용구 급여은 평가 메모의 후보군(보훈 정보가 없는 일반 노인에게 보훈대상자 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001177",
                "노인_보훈아님_보훈오탐방지: 장기요양 본인부담금 감경은 평가 메모의 후보군(보훈 정보가 없는 일반 노인에게 보훈대상자 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003280",
                "노인_보훈아님_보훈오탐방지: 전립선등 노인성질환 예방관리은 평가 메모의 후보군(보훈 정보가 없는 일반 노인에게 보훈대상자 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "보훈 정보가 없는 일반 노인에게 보훈대상자 전용 서비스는 오탐",
        excluded_ids=(
            excluded_svc(
                "WLF00000048",
                "노인_보훈아님_보훈오탐방지: 국가유공자등대부지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00000054",
                "노인_보훈아님_보훈오탐방지: 국가보훈대상자 보훈장학금은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00000069",
                "노인_보훈아님_보훈오탐방지: 보훈원 양로지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00000097",
                "노인_보훈아님_보훈오탐방지: 국가유공자등생활조정수당은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00000098",
                "노인_보훈아님_보훈오탐방지: 국가유공자재가복지지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003243",
                "노인_보훈아님_보훈오탐방지: 보훈요양원 이용지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00004662",
                "노인_보훈아님_보훈오탐방지: 보훈대상자 생계지원금 지급은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
        ),
        intent_theme="guardrail:unknown_veteran",
    ),
    EvalCase(
        "북한이탈정보없음_일반_오탐방지",
        SearchRequest(age=42, income_level="일반", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00001083",
                "북한이탈정보없음_일반_오탐방지: 디지털배움터은 평가 메모의 후보군(북한이탈 정보가 없는 일반 성인에게 북한이탈주민 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003239",
                "북한이탈정보없음_일반_오탐방지: 고용복지플러스센터은 평가 메모의 후보군(북한이탈 정보가 없는 일반 성인에게 북한이탈주민 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003264",
                "북한이탈정보없음_일반_오탐방지: 통합건강증진사업은 평가 메모의 후보군(북한이탈 정보가 없는 일반 성인에게 북한이탈주민 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00005036",
                "북한이탈정보없음_일반_오탐방지: 개인채무조정은 평가 메모의 후보군(북한이탈 정보가 없는 일반 성인에게 북한이탈주민 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00004997",
                "북한이탈정보없음_일반_오탐방지: 한국형 상병수당 시범사업은 평가 메모의 후보군(북한이탈 정보가 없는 일반 성인에게 북한이탈주민 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "북한이탈 정보가 없는 일반 성인에게 북한이탈주민 전용 서비스는 오탐",
        excluded_ids=(
            excluded_svc(
                "WLF00000063",
                "북한이탈정보없음_일반_오탐방지: (북한이탈주민)교육비 지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00000782",
                "북한이탈정보없음_일반_오탐방지: (북한이탈주민)사회보장 지원(수급권자 범위의 특례)은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00000815",
                "북한이탈정보없음_일반_오탐방지: (북한이탈주민)주택알선 지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00001020",
                "북한이탈정보없음_일반_오탐방지: 북한이탈주민 자산형성지원제도(미래행복통장)은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00001168",
                "북한이탈정보없음_일반_오탐방지: (북한이탈주민)자립자활지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003168",
                "북한이탈정보없음_일반_오탐방지: (북한이탈주민)취업 지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003172",
                "북한이탈정보없음_일반_오탐방지: (북한이탈주민)정착금 지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003206",
                "북한이탈정보없음_일반_오탐방지: (북한이탈주민) 의료비 지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
        ),
        ambiguous=True,
        ambiguity_reason="일반 성인 입력만으로 긍정 정답을 확정하기 어렵고 북한이탈주민 전용 서비스 오탐 회귀를 주로 확인하는 케이스다.",
        intent_theme="guardrail:unknown_north_korean_defector",
    ),
    EvalCase(
        "다문화정보없음_일반_오탐방지",
        SearchRequest(age=42, income_level="일반", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00001083",
                "다문화정보없음_일반_오탐방지: 디지털배움터은 평가 메모의 후보군(다문화 정보가 없는 일반 성인에게 다문화가족 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003239",
                "다문화정보없음_일반_오탐방지: 고용복지플러스센터은 평가 메모의 후보군(다문화 정보가 없는 일반 성인에게 다문화가족 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003264",
                "다문화정보없음_일반_오탐방지: 통합건강증진사업은 평가 메모의 후보군(다문화 정보가 없는 일반 성인에게 다문화가족 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00005036",
                "다문화정보없음_일반_오탐방지: 개인채무조정은 평가 메모의 후보군(다문화 정보가 없는 일반 성인에게 다문화가족 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00004997",
                "다문화정보없음_일반_오탐방지: 한국형 상병수당 시범사업은 평가 메모의 후보군(다문화 정보가 없는 일반 성인에게 다문화가족 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "다문화 정보가 없는 일반 성인에게 다문화가족 전용 서비스는 오탐",
        excluded_ids=(
            excluded_svc(
                "WLF00000837",
                "다문화정보없음_일반_오탐방지: 다문화가족 지원사업은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00001114",
                "다문화정보없음_일반_오탐방지: 이주배경 청소년 지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00001183",
                "다문화정보없음_일반_오탐방지: 다문화보육료지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003175",
                "다문화정보없음_일반_오탐방지: 결혼이민자 통번역 서비스은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003188",
                "다문화정보없음_일반_오탐방지: 다문화·탈북학생 멘토링은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003192",
                "다문화정보없음_일반_오탐방지: 다문화가족 방문교육 서비스은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003282",
                "다문화정보없음_일반_오탐방지: 다문화가족 자녀 언어발달지원서비스은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00005448",
                "다문화정보없음_일반_오탐방지: 다문화가족 자녀 교육활동비 지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
        ),
        ambiguous=True,
        ambiguity_reason="일반 성인 입력만으로 긍정 정답을 확정하기 어렵고 다문화가족 전용 서비스 오탐 회귀를 주로 확인하는 케이스다.",
        intent_theme="guardrail:unknown_multicultural",
    ),
    EvalCase(
        "농어업정보없음_일반_오탐방지",
        SearchRequest(age=42, income_level="일반", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00001083",
                "농어업정보없음_일반_오탐방지: 디지털배움터은 평가 메모의 후보군(농어업 종사 정보가 없는 일반 성인에게 농어업 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003239",
                "농어업정보없음_일반_오탐방지: 고용복지플러스센터은 평가 메모의 후보군(농어업 종사 정보가 없는 일반 성인에게 농어업 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003264",
                "농어업정보없음_일반_오탐방지: 통합건강증진사업은 평가 메모의 후보군(농어업 종사 정보가 없는 일반 성인에게 농어업 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00005036",
                "농어업정보없음_일반_오탐방지: 개인채무조정은 평가 메모의 후보군(농어업 종사 정보가 없는 일반 성인에게 농어업 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00004997",
                "농어업정보없음_일반_오탐방지: 한국형 상병수당 시범사업은 평가 메모의 후보군(농어업 종사 정보가 없는 일반 성인에게 농어업 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "농어업 종사 정보가 없는 일반 성인에게 농어업 전용 서비스는 오탐",
        excluded_ids=(
            excluded_svc(
                "WLF00000023",
                "농어업정보없음_일반_오탐방지: 농어가목돈마련저축 저축장려금 지급은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00001096",
                "농어업정보없음_일반_오탐방지: 농촌출신대학생학자금융자은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00001097",
                "농어업정보없음_일반_오탐방지: 농업인연금보험료지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00001099",
                "농어업정보없음_일반_오탐방지: 농업인건강보험료지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003207",
                "농어업정보없음_일반_오탐방지: 영농도우미 지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003232",
                "농어업정보없음_일반_오탐방지: 농업인안전보험은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00005026",
                "농어업정보없음_일반_오탐방지: 여성어업인 특화건강검진사업은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00005446",
                "농어업정보없음_일반_오탐방지: 어업인 안전보험은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00006245",
                "농어업정보없음_일반_오탐방지: 어업활동지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00006250",
                "농어업정보없음_일반_오탐방지: 농어촌 기본소득 시범사업은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
        ),
        ambiguous=True,
        ambiguity_reason="일반 성인 입력만으로 긍정 정답을 확정하기 어렵고 농어업 전용 서비스 오탐 회귀를 주로 확인하는 케이스다.",
        intent_theme="guardrail:unknown_agriculture_or_fishery",
    ),
    EvalCase(
        "청소년부모정보없음_성인_오탐방지",
        SearchRequest(
            age=42,
            income_level="일반",
            has_children=False,
            disability=False,
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00001083",
                "청소년부모정보없음_성인_오탐방지: 디지털배움터은 평가 메모의 후보군(청소년부모 정보가 없는 일반 성인에게 청소년부모 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003239",
                "청소년부모정보없음_성인_오탐방지: 고용복지플러스센터은 평가 메모의 후보군(청소년부모 정보가 없는 일반 성인에게 청소년부모 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003264",
                "청소년부모정보없음_성인_오탐방지: 통합건강증진사업은 평가 메모의 후보군(청소년부모 정보가 없는 일반 성인에게 청소년부모 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00005036",
                "청소년부모정보없음_성인_오탐방지: 개인채무조정은 평가 메모의 후보군(청소년부모 정보가 없는 일반 성인에게 청소년부모 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00004997",
                "청소년부모정보없음_성인_오탐방지: 한국형 상병수당 시범사업은 평가 메모의 후보군(청소년부모 정보가 없는 일반 성인에게 청소년부모 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "청소년부모 정보가 없는 일반 성인에게 청소년부모 전용 서비스는 오탐",
        excluded_ids=(
            excluded_svc(
                "WLF00005023",
                "청소년부모정보없음_성인_오탐방지: 청소년부모 아동양육비 지원사업은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00001109",
                "청소년부모정보없음_성인_오탐방지: 청소년한부모 아동양육 및 자립지원은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
        ),
        intent_theme="guardrail:no_children_or_parenting",
    ),
    EvalCase(
        "5세_저소득_영유아건강",
        SearchRequest(age=5, income_level="저소득", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00000040",
                "5세_저소득_영유아건강: 선천성대사이상 검사 및 환아관리은 평가 메모의 후보군(5세 저소득 영유아에게는 검진, 예방접종, 영양, 선천성 질환 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001130",
                "5세_저소득_영유아건강: 선천성 난청검사 및 보청기 지원은 평가 메모의 후보군(5세 저소득 영유아에게는 검진, 예방접종, 영양, 선천성 질환 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001185",
                "5세_저소득_영유아건강: 의료급여수급권자 영유아건강검진비 지원은 평가 메모의 후보군(5세 저소득 영유아에게는 검진, 예방접종, 영양, 선천성 질환 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003237",
                "5세_저소득_영유아건강: 미숙아 및 선천성이상아 의료비 지원은 평가 메모의 후보군(5세 저소득 영유아에게는 검진, 예방접종, 영양, 선천성 질환 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003242",
                "5세_저소득_영유아건강: 국가예방접종 사업은 평가 메모의 후보군(5세 저소득 영유아에게는 검진, 예방접종, 영양, 선천성 질환 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006239",
                "5세_저소득_영유아건강: 영양플러스 사업은 평가 메모의 후보군(5세 저소득 영유아에게는 검진, 예방접종, 영양, 선천성 질환 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "5세 저소득 영유아에게는 검진, 예방접종, 영양, 선천성 질환 지원이 적절",
        query="의료비 의료급여 건강검진 질환 치료비",
        intent_theme="medical",
    ),
    EvalCase(
        "6세_일반_초등전아동",
        SearchRequest(age=6, income_level="일반", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00001171",
                "6세_일반_초등전아동: 아동수당 지급은 평가 메모의 후보군(6세 일반 아동에게는 아동수당, 다함께돌봄, 예방접종, 안전망 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000089",
                "6세_일반_초등전아동: 다함께 돌봄 사업은 평가 메모의 후보군(6세 일반 아동에게는 아동수당, 다함께돌봄, 예방접종, 안전망 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003242",
                "6세_일반_초등전아동: 국가예방접종 사업은 평가 메모의 후보군(6세 일반 아동에게는 아동수당, 다함께돌봄, 예방접종, 안전망 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001141",
                "6세_일반_초등전아동: 지역사회 청소년통합지원체계(청소년안전망)은 평가 메모의 후보군(6세 일반 아동에게는 아동수당, 다함께돌봄, 예방접종, 안전망 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003223",
                "6세_일반_초등전아동: 고난도 보호대상아동 맞춤형 사례관리서비스은 평가 메모의 후보군(6세 일반 아동에게는 아동수당, 다함께돌봄, 예방접종, 안전망 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "6세 일반 아동에게는 아동수당, 다함께돌봄, 예방접종, 안전망 서비스가 적절",
    ),
    EvalCase(
        "18세_일반_청소년상담",
        SearchRequest(age=18, income_level="일반", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00003200",
                "18세_일반_청소년상담: 청소년동반자프로그램 운영은 평가 메모의 후보군(18세 일반 청소년에게는 상담, 청소년안전망, 성문화센터, 복지시설 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003204",
                "18세_일반_청소년상담: 청소년상담1388 온라인상담은 평가 메모의 후보군(18세 일반 청소년에게는 상담, 청소년안전망, 성문화센터, 복지시설 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003236",
                "18세_일반_청소년상담: 청소년상담1388 전화상담은 평가 메모의 후보군(18세 일반 청소년에게는 상담, 청소년안전망, 성문화센터, 복지시설 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001141",
                "18세_일반_청소년상담: 지역사회 청소년통합지원체계(청소년안전망)은 평가 메모의 후보군(18세 일반 청소년에게는 상담, 청소년안전망, 성문화센터, 복지시설 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00004650",
                "18세_일반_청소년상담: 청소년성문화센터설치운영은 평가 메모의 후보군(18세 일반 청소년에게는 상담, 청소년안전망, 성문화센터, 복지시설 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000751",
                "18세_일반_청소년상담: 청소년복지시설 운영 지원은 평가 메모의 후보군(18세 일반 청소년에게는 상담, 청소년안전망, 성문화센터, 복지시설 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "18세 일반 청소년에게는 상담, 청소년안전망, 성문화센터, 복지시설 지원이 적절",
    ),
    EvalCase(
        "19세_일반_청년금융",
        SearchRequest(age=19, income_level="일반", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00001076",
                "19세_일반_청년금융: 서민금융 활성화 지원(햇살론youth 보증사업)은 평가 메모의 후보군(19세 일반 청년에게는 학자금, 청년 금융, 고용/훈련 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003277",
                "19세_일반_청년금융: 취업 후 상환 학자금대출은 평가 메모의 후보군(19세 일반 청년에게는 학자금, 청년 금융, 고용/훈련 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003239",
                "19세_일반_청년금융: 고용복지플러스센터은 평가 메모의 후보군(19세 일반 청년에게는 학자금, 청년 금융, 고용/훈련 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006229",
                "19세_일반_청년금융: 국민내일배움카드제 직업훈련지원(훈련비, 훈련장려금)은 평가 메모의 후보군(19세 일반 청년에게는 학자금, 청년 금융, 고용/훈련 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006267",
                "19세_일반_청년금융: 서민금융진흥원 금융교육은 평가 메모의 후보군(19세 일반 청년에게는 학자금, 청년 금융, 고용/훈련 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006249",
                "19세_일반_청년금융: 해외취업 지원은 평가 메모의 후보군(19세 일반 청년에게는 학자금, 청년 금융, 고용/훈련 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "19세 일반 청년에게는 학자금, 청년 금융, 고용/훈련 서비스가 적절",
        query="자산형성 학자금 서민금융 대출 신용 부채관리",
        intent_theme="finance",
    ),
    EvalCase(
        "34세_일반_청년고용",
        SearchRequest(
            age=34,
            income_level="일반",
            disability=False,
            employment_status="실업",
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00003245",
                "34세_일반_청년고용: 국민취업지원제도은 평가 메모의 후보군(34세 일반 실업 청년에게는 청년/일반 고용과 직업훈련 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003239",
                "34세_일반_청년고용: 고용복지플러스센터은 평가 메모의 후보군(34세 일반 실업 청년에게는 청년/일반 고용과 직업훈련 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006229",
                "34세_일반_청년고용: 국민내일배움카드제 직업훈련지원(훈련비, 훈련장려금)은 평가 메모의 후보군(34세 일반 실업 청년에게는 청년/일반 고용과 직업훈련 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006215",
                "34세_일반_청년고용: 청년내일채움공제은 평가 메모의 후보군(34세 일반 실업 청년에게는 청년/일반 고용과 직업훈련 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006249",
                "34세_일반_청년고용: 해외취업 지원은 평가 메모의 후보군(34세 일반 실업 청년에게는 청년/일반 고용과 직업훈련 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001076",
                "34세_일반_청년고용: 서민금융 활성화 지원(햇살론youth 보증사업)은 평가 메모의 후보군(34세 일반 실업 청년에게는 청년/일반 고용과 직업훈련 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "34세 일반 실업 청년에게는 청년/일반 고용과 직업훈련 서비스가 적절",
        query="취업 구직 직업훈련 고용센터 자활근로",
        intent_theme="employment",
    ),
    EvalCase(
        "35세_저소득_중장년전환",
        SearchRequest(
            age=35,
            income_level="저소득",
            disability=False,
            employment_status="실업",
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00003245",
                "35세_저소득_중장년전환: 국민취업지원제도은 평가 메모의 후보군(35세 저소득 실업자는 청년 경계 이후 일반/중장년 고용지원도 함께 고려되어야 함)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003266",
                "35세_저소득_중장년전환: 직업훈련생계비대부은 평가 메모의 후보군(35세 저소득 실업자는 청년 경계 이후 일반/중장년 고용지원도 함께 고려되어야 함)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00005852",
                "35세_저소득_중장년전환: 중장년 경력지원제은 평가 메모의 후보군(35세 저소득 실업자는 청년 경계 이후 일반/중장년 고용지원도 함께 고려되어야 함)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003205",
                "35세_저소득_중장년전환: 중장년 기술창업센터 지원사업은 평가 메모의 후보군(35세 저소득 실업자는 청년 경계 이후 일반/중장년 고용지원도 함께 고려되어야 함)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001138",
                "35세_저소득_중장년전환: 자활근로(기초, 차상위)은 평가 메모의 후보군(35세 저소득 실업자는 청년 경계 이후 일반/중장년 고용지원도 함께 고려되어야 함)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001157",
                "35세_저소득_중장년전환: 지역자활센터 운영은 평가 메모의 후보군(35세 저소득 실업자는 청년 경계 이후 일반/중장년 고용지원도 함께 고려되어야 함)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "35세 저소득 실업자는 청년 경계 이후 일반/중장년 고용지원도 함께 고려되어야 함",
        excluded_ids=(
            excluded_svc(
                "WLF00004661",
                "35세_저소득_중장년전환: 청년월세 지원사업은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00000060",
                "35세_저소득_중장년전환: 청년내일저축계좌은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00006215",
                "35세_저소득_중장년전환: 청년내일채움공제은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
        ),
    ),
    EvalCase(
        "61세_일반_노인전용오탐방지",
        SearchRequest(
            age=61,
            income_level="일반",
            household_size=1,
            has_children=False,
            disability=False,
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00000031",
                "61세_일반_노인전용오탐방지: 노후준비서비스은 평가 메모의 후보군(61세 일반 비장애 1인 가구에게 65세 이상 노인 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001083",
                "61세_일반_노인전용오탐방지: 디지털배움터은 평가 메모의 후보군(61세 일반 비장애 1인 가구에게 65세 이상 노인 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003264",
                "61세_일반_노인전용오탐방지: 통합건강증진사업은 평가 메모의 후보군(61세 일반 비장애 1인 가구에게 65세 이상 노인 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00005036",
                "61세_일반_노인전용오탐방지: 개인채무조정은 평가 메모의 후보군(61세 일반 비장애 1인 가구에게 65세 이상 노인 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00004997",
                "61세_일반_노인전용오탐방지: 한국형 상병수당 시범사업은 평가 메모의 후보군(61세 일반 비장애 1인 가구에게 65세 이상 노인 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "61세 일반 비장애 1인 가구에게 65세 이상 노인 전용 서비스는 오탐",
        excluded_ids=(
            excluded_svc(
                "WLF00001164",
                "61세_일반_노인전용오탐방지: 기초연금은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003191",
                "61세_일반_노인전용오탐방지: 노인맞춤돌봄서비스은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00001155",
                "61세_일반_노인전용오탐방지: 노인일자리 및 사회활동 지원사업은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00001093",
                "61세_일반_노인전용오탐방지: 독거노인·장애인 응급안전안심서비스은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
        ),
        intent_theme="guardrail:under_senior_age",
    ),
    EvalCase(
        "64세_일반_노후전환오탐방지",
        SearchRequest(
            age=64,
            income_level="일반",
            household_size=1,
            disability=False,
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00000031",
                "64세_일반_노후전환오탐방지: 노후준비서비스은 평가 메모의 후보군(64세 일반 사용자에게 65세 이상 저소득 노인 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001108",
                "64세_일반_노후전환오탐방지: 주택담보노후연금보증은 평가 메모의 후보군(64세 일반 사용자에게 65세 이상 저소득 노인 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001083",
                "64세_일반_노후전환오탐방지: 디지털배움터은 평가 메모의 후보군(64세 일반 사용자에게 65세 이상 저소득 노인 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003264",
                "64세_일반_노후전환오탐방지: 통합건강증진사업은 평가 메모의 후보군(64세 일반 사용자에게 65세 이상 저소득 노인 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00005036",
                "64세_일반_노후전환오탐방지: 개인채무조정은 평가 메모의 후보군(64세 일반 사용자에게 65세 이상 저소득 노인 전용 서비스는 오탐)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "64세 일반 사용자에게 65세 이상 저소득 노인 전용 서비스는 오탐",
        excluded_ids=(
            excluded_svc(
                "WLF00001164",
                "64세_일반_노후전환오탐방지: 기초연금은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00003191",
                "64세_일반_노후전환오탐방지: 노인맞춤돌봄서비스은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00001155",
                "64세_일반_노후전환오탐방지: 노인일자리 및 사회활동 지원사업은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
            excluded_svc(
                "WLF00001093",
                "64세_일반_노후전환오탐방지: 독거노인·장애인 응급안전안심서비스은 명시 조건과 충돌하거나 입력에 없는 특수 조건에 의존해 top-5 제외 회귀 후보로 유지한다.",
            ),
        ),
        intent_theme="guardrail:under_senior_age",
    ),
    EvalCase(
        "65세_저소득_노인소득의료",
        SearchRequest(age=66, income_level="저소득", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00001164",
                "65세_저소득_노인소득의료: 기초연금은 평가 메모의 후보군(65세 이상 저소득 노인에게는 기초연금과 노인 의료/돌봄 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001115",
                "65세_저소득_노인소득의료: 노인 개안수술비 지원은 평가 메모의 후보군(65세 이상 저소득 노인에게는 기초연금과 노인 의료/돌봄 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001169",
                "65세_저소득_노인소득의료: 의료급여 틀니·치과임플란트은 평가 메모의 후보군(65세 이상 저소득 노인에게는 기초연금과 노인 의료/돌봄 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001179",
                "65세_저소득_노인소득의료: 노인 무릎인공관절 수술 지원 사업은 평가 메모의 후보군(65세 이상 저소득 노인에게는 기초연금과 노인 의료/돌봄 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00005004",
                "65세_저소득_노인소득의료: 치매검사비 지원은 평가 메모의 후보군(65세 이상 저소득 노인에게는 기초연금과 노인 의료/돌봄 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003191",
                "65세_저소득_노인소득의료: 노인맞춤돌봄서비스은 평가 메모의 후보군(65세 이상 저소득 노인에게는 기초연금과 노인 의료/돌봄 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "65세 이상 저소득 노인에게는 기초연금과 노인 의료/돌봄 서비스가 적절",
        query="의료비 의료급여 건강검진 질환 치료비",
        intent_theme="medical",
    ),
    EvalCase(
        "아동_일반_돌봄",
        SearchRequest(age=9, income_level="일반", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00001171",
                "아동_일반_돌봄: 아동수당 지급은 평가 메모의 후보군(일반 아동에게는 아동수당, 다함께돌봄, 예방접종, 청소년안전망이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000089",
                "아동_일반_돌봄: 다함께 돌봄 사업은 평가 메모의 후보군(일반 아동에게는 아동수당, 다함께돌봄, 예방접종, 청소년안전망이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001141",
                "아동_일반_돌봄: 지역사회 청소년통합지원체계(청소년안전망)은 평가 메모의 후보군(일반 아동에게는 아동수당, 다함께돌봄, 예방접종, 청소년안전망이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003242",
                "아동_일반_돌봄: 국가예방접종 사업은 평가 메모의 후보군(일반 아동에게는 아동수당, 다함께돌봄, 예방접종, 청소년안전망이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003223",
                "아동_일반_돌봄: 고난도 보호대상아동 맞춤형 사례관리서비스은 평가 메모의 후보군(일반 아동에게는 아동수당, 다함께돌봄, 예방접종, 청소년안전망이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "일반 아동에게는 아동수당, 다함께돌봄, 예방접종, 청소년안전망이 적절",
        query="돌봄 안부 안전 장기요양 방문 재가",
        intent_theme="care",
    ),
    EvalCase(
        "아동_저소득_교육문화",
        SearchRequest(age=12, income_level="저소득", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00001103",
                "아동_저소득_교육문화: 초중고 교육비 지원사업(고교학비 지원)은 평가 메모의 후보군(저소득 아동에게는 교육비, 교육급여, 지역아동센터, 문화·체육 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001089",
                "아동_저소득_교육문화: 교육급여(맞춤형 급여)은 평가 메모의 후보군(저소득 아동에게는 교육비, 교육급여, 지역아동센터, 문화·체육 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001107",
                "아동_저소득_교육문화: 지역아동센터 지원은 평가 메모의 후보군(저소득 아동에게는 교육비, 교육급여, 지역아동센터, 문화·체육 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000076",
                "아동_저소득_교육문화: 스포츠강좌이용권은 평가 메모의 후보군(저소득 아동에게는 교육비, 교육급여, 지역아동센터, 문화·체육 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001120",
                "아동_저소득_교육문화: 교육복지우선지원사업은 평가 메모의 후보군(저소득 아동에게는 교육비, 교육급여, 지역아동센터, 문화·체육 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000867",
                "아동_저소득_교육문화: 방과후학교 자유수강권은 평가 메모의 후보군(저소득 아동에게는 교육비, 교육급여, 지역아동센터, 문화·체육 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "저소득 아동에게는 교육비, 교육급여, 지역아동센터, 문화·체육 지원이 적절",
        query="문화 여가 스포츠 교육 바우처 이용권",
        intent_theme="culture",
    ),
    EvalCase(
        "청소년_일반_상담보호",
        SearchRequest(age=16, income_level="일반", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00003200",
                "청소년_일반_상담보호: 청소년동반자프로그램 운영은 평가 메모의 후보군(일반 청소년에게는 상담, 보호, 안전망 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003204",
                "청소년_일반_상담보호: 청소년상담1388 온라인상담은 평가 메모의 후보군(일반 청소년에게는 상담, 보호, 안전망 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003236",
                "청소년_일반_상담보호: 청소년상담1388 전화상담은 평가 메모의 후보군(일반 청소년에게는 상담, 보호, 안전망 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001141",
                "청소년_일반_상담보호: 지역사회 청소년통합지원체계(청소년안전망)은 평가 메모의 후보군(일반 청소년에게는 상담, 보호, 안전망 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00004650",
                "청소년_일반_상담보호: 청소년성문화센터설치운영은 평가 메모의 후보군(일반 청소년에게는 상담, 보호, 안전망 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000751",
                "청소년_일반_상담보호: 청소년복지시설 운영 지원은 평가 메모의 후보군(일반 청소년에게는 상담, 보호, 안전망 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "일반 청소년에게는 상담, 보호, 안전망 서비스가 적절",
    ),
    EvalCase(
        "청소년_저소득_자립교육",
        SearchRequest(age=17, income_level="저소득", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00000078",
                "청소년_저소득_자립교육: 청소년특별지원은 평가 메모의 후보군(저소득 청소년에게는 특별지원, 생리용품, 학교밖/방과후/교육비 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000781",
                "청소년_저소득_자립교육: 여성청소년 생리용품 지원은 평가 메모의 후보군(저소득 청소년에게는 특별지원, 생리용품, 학교밖/방과후/교육비 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000948",
                "청소년_저소득_자립교육: 학교 밖 청소년 지원은 평가 메모의 후보군(저소득 청소년에게는 특별지원, 생리용품, 학교밖/방과후/교육비 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003254",
                "청소년_저소득_자립교육: 청소년방과후아카데미운영지원은 평가 메모의 후보군(저소득 청소년에게는 특별지원, 생리용품, 학교밖/방과후/교육비 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001103",
                "청소년_저소득_자립교육: 초중고 교육비 지원사업(고교학비 지원)은 평가 메모의 후보군(저소득 청소년에게는 특별지원, 생리용품, 학교밖/방과후/교육비 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001120",
                "청소년_저소득_자립교육: 교육복지우선지원사업은 평가 메모의 후보군(저소득 청소년에게는 특별지원, 생리용품, 학교밖/방과후/교육비 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "저소득 청소년에게는 특별지원, 생리용품, 학교밖/방과후/교육비 지원이 적절",
        query="교육비 장학 방과후 교육정보화 급식",
        intent_theme="education",
    ),
    EvalCase(
        "청년_저소득_주거",
        SearchRequest(
            age=29,
            income_level="저소득",
            household_size=1,
            disability=False,
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00004661",
                "청년_저소득_주거: 청년월세 지원사업은 평가 메모의 후보군(저소득 청년 1인 가구에는 청년월세, 행복주택, 전월세 주거금융이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00004649",
                "청년_저소득_주거: 행복주택 공급은 평가 메모의 후보군(저소득 청년 1인 가구에는 청년월세, 행복주택, 전월세 주거금융이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003222",
                "청년_저소득_주거: 버팀목전세자금대출은 평가 메모의 후보군(저소득 청년 1인 가구에는 청년월세, 행복주택, 전월세 주거금융이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001063",
                "청년_저소득_주거: 주거안정 월세대출은 평가 메모의 후보군(저소득 청년 1인 가구에는 청년월세, 행복주택, 전월세 주거금융이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001079",
                "청년_저소득_주거: 주거안정 월세대출 보증은 평가 메모의 후보군(저소득 청년 1인 가구에는 청년월세, 행복주택, 전월세 주거금융이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003269",
                "청년_저소득_주거: 기존주택 전세임대주택 지원사업은 평가 메모의 후보군(저소득 청년 1인 가구에는 청년월세, 행복주택, 전월세 주거금융이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "저소득 청년 1인 가구에는 청년월세, 행복주택, 전월세 주거금융이 적절",
        query="주거 주거급여 공공임대 전세 월세 주거비",
        intent_theme="housing",
    ),
    EvalCase(
        "청년_일반_교육대출",
        SearchRequest(age=22, income_level="일반", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00003277",
                "청년_일반_교육대출: 취업 후 상환 학자금대출은 평가 메모의 후보군(일반 청년 학생에게는 학자금대출, 장학금, 취업연계 장학/현장실습 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003276",
                "청년_일반_교육대출: 일반 상환 학자금대출은 평가 메모의 후보군(일반 청년 학생에게는 학자금대출, 장학금, 취업연계 장학/현장실습 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001071",
                "청년_일반_교육대출: 우수학생 국가장학금 지원은 평가 메모의 후보군(일반 청년 학생에게는 학자금대출, 장학금, 취업연계 장학/현장실습 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000052",
                "청년_일반_교육대출: 인문100년장학금은 평가 메모의 후보군(일반 청년 학생에게는 학자금대출, 장학금, 취업연계 장학/현장실습 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006270",
                "청년_일반_교육대출: 중소기업 취업연계 장학사업 (희망사다리 I유형)은 평가 메모의 후보군(일반 청년 학생에게는 학자금대출, 장학금, 취업연계 장학/현장실습 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00006242",
                "청년_일반_교육대출: 현장실습 지원금은 평가 메모의 후보군(일반 청년 학생에게는 학자금대출, 장학금, 취업연계 장학/현장실습 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "일반 청년 학생에게는 학자금대출, 장학금, 취업연계 장학/현장실습 지원이 적절",
        query="자산형성 학자금 서민금융 대출 신용 부채관리",
        intent_theme="finance",
    ),
    EvalCase(
        "중장년_저소득_주거에너지",
        SearchRequest(
            age=48,
            income_level="저소득",
            household_size=1,
            disability=False,
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00003201",
                "중장년_저소득_주거에너지: 주거급여(맞춤형 급여)은 평가 메모의 후보군(저소득 중장년 1인 가구에는 주거급여, 주거상향, 에너지 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000036",
                "중장년_저소득_주거에너지: 주거취약계층 주거상향 지원사업은 평가 메모의 후보군(저소득 중장년 1인 가구에는 주거급여, 주거상향, 에너지 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000917",
                "중장년_저소득_주거에너지: 긴급복지 주거지원은 평가 메모의 후보군(저소득 중장년 1인 가구에는 주거급여, 주거상향, 에너지 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000072",
                "중장년_저소득_주거에너지: 에너지바우처은 평가 메모의 후보군(저소득 중장년 1인 가구에는 주거급여, 주거상향, 에너지 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001128",
                "중장년_저소득_주거에너지: 저소득층에너지효율개선은 평가 메모의 후보군(저소득 중장년 1인 가구에는 주거급여, 주거상향, 에너지 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00004639",
                "중장년_저소득_주거에너지: 전기요금 복지할인은 평가 메모의 후보군(저소득 중장년 1인 가구에는 주거급여, 주거상향, 에너지 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "저소득 중장년 1인 가구에는 주거급여, 주거상향, 에너지 지원이 적절",
        query="주거 주거급여 공공임대 전세 월세 주거비",
        intent_theme="housing",
    ),
    EvalCase(
        "중장년_일반_건강생활",
        SearchRequest(age=48, income_level="일반", disability=False, pregnant=False, top_k=10),
        (
            svc(
                "WLF00003264",
                "중장년_일반_건강생활: 통합건강증진사업은 평가 메모의 후보군(일반 중장년에게는 건강증진, 암검진, 만성질환 관리, 생활금융 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001176",
                "중장년_일반_건강생활: 암검진사업은 평가 메모의 후보군(일반 중장년에게는 건강증진, 암검진, 만성질환 관리, 생활금융 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000047",
                "중장년_일반_건강생활: 고혈압·당뇨병 등록관리사업은 평가 메모의 후보군(일반 중장년에게는 건강증진, 암검진, 만성질환 관리, 생활금융 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00004997",
                "중장년_일반_건강생활: 한국형 상병수당 시범사업은 평가 메모의 후보군(일반 중장년에게는 건강증진, 암검진, 만성질환 관리, 생활금융 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001083",
                "중장년_일반_건강생활: 디지털배움터은 평가 메모의 후보군(일반 중장년에게는 건강증진, 암검진, 만성질환 관리, 생활금융 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00005036",
                "중장년_일반_건강생활: 개인채무조정은 평가 메모의 후보군(일반 중장년에게는 건강증진, 암검진, 만성질환 관리, 생활금융 서비스가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "일반 중장년에게는 건강증진, 암검진, 만성질환 관리, 생활금융 서비스가 적절",
        query="의료비 의료급여 건강검진 질환 치료비",
        intent_theme="medical",
    ),
    EvalCase(
        "저소득_한부모_자녀_복합",
        SearchRequest(
            age=36,
            income_level="저소득",
            household_size=2,
            marital_status="이혼",
            has_children=True,
            disability=False,
            pregnant=False,
            top_k=10,
        ),
        (
            svc(
                "WLF00001068",
                "저소득_한부모_자녀_복합: 한부모가족 아동양육비 지원은 평가 메모의 후보군(저소득+한부모+자녀 복합 조건에는 양육비, 교육비, 주거/돌봄 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001104",
                "저소득_한부모_자녀_복합: 한부모가족자녀 교육비 지원은 평가 메모의 후보군(저소득+한부모+자녀 복합 조건에는 양육비, 교육비, 주거/돌봄 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00005856",
                "저소득_한부모_자녀_복합: 양육비 선지급은 평가 메모의 후보군(저소득+한부모+자녀 복합 조건에는 양육비, 교육비, 주거/돌봄 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003186",
                "저소득_한부모_자녀_복합: 양육비 이행 원스톱 종합서비스은 평가 메모의 후보군(저소득+한부모+자녀 복합 조건에는 양육비, 교육비, 주거/돌봄 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000091",
                "저소득_한부모_자녀_복합: 한부모가족복지시설 지원은 평가 메모의 후보군(저소득+한부모+자녀 복합 조건에는 양육비, 교육비, 주거/돌봄 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000024",
                "저소득_한부모_자녀_복합: 아이돌봄서비스은 평가 메모의 후보군(저소득+한부모+자녀 복합 조건에는 양육비, 교육비, 주거/돌봄 지원이 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "저소득+한부모+자녀 복합 조건에는 양육비, 교육비, 주거/돌봄 지원이 적절",
    ),
    EvalCase(
        "임산부_저소득_영양_복합",
        SearchRequest(
            age=30,
            income_level="저소득",
            marital_status="기혼",
            disability=False,
            pregnant=True,
            top_k=10,
        ),
        (
            svc(
                "WLF00006239",
                "임산부_저소득_영양_복합: 영양플러스 사업은 평가 메모의 후보군(임산부+저소득 복합 조건에는 영양, 기저귀·조제분유, 의료비, 산모 건강관리가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000092",
                "임산부_저소득_영양_복합: 저소득층 기저귀·조제분유 지원은 평가 메모의 후보군(임산부+저소득 복합 조건에는 영양, 기저귀·조제분유, 의료비, 산모 건강관리가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001088",
                "임산부_저소득_영양_복합: 고위험 임산부 의료비 지원은 평가 메모의 후보군(임산부+저소득 복합 조건에는 영양, 기저귀·조제분유, 의료비, 산모 건강관리가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00001188",
                "임산부_저소득_영양_복합: 산모·신생아 건강관리 지원사업은 평가 메모의 후보군(임산부+저소득 복합 조건에는 영양, 기저귀·조제분유, 의료비, 산모 건강관리가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00000061",
                "임산부_저소득_영양_복합: 의료급여임신.출산진료비지원은 평가 메모의 후보군(임산부+저소득 복합 조건에는 영양, 기저귀·조제분유, 의료비, 산모 건강관리가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
            svc(
                "WLF00003178",
                "임산부_저소득_영양_복합: 긴급복지 해산비지원은 평가 메모의 후보군(임산부+저소득 복합 조건에는 영양, 기저귀·조제분유, 의료비, 산모 건강관리가 적절)에 포함된다. 현재 입력만으로 must 확정은 보류한다.",
            ),
        ),
        "임산부+저소득 복합 조건에는 영양, 기저귀·조제분유, 의료비, 산모 건강관리가 적절",
        query="임산부 영양 기저귀 조제분유 산모 건강관리",
        intent_theme="maternity:nutrition",
    ),
)


async def _diagnose_case(
    case: EvalCase,
    embedder: KoSimCSEEmbedder,
    collection_name: str,
    adaptive_fetch: bool,
) -> None:
    """실패 케이스의 raw distance rank → boost → final rank 흐름 출력."""
    query_text = _evaluation_query_text(case)
    vec: list[float] = embedder.embed([query_text])[0]

    collection = await get_collection(collection_name)
    n_results = (
        min(max(case.request.top_k * 20, 100), 500)
        if adaptive_fetch
        else min(case.request.top_k * 5, 100)
    )

    def _query() -> Any:
        return collection.query(
            query_embeddings=[vec],  # type: ignore[arg-type]
            n_results=n_results,
        )

    raw: Any = await asyncio.to_thread(_query)
    distances: list[float] = raw["distances"][0]
    metadatas: list[dict[str, str]] = raw["metadatas"][0]

    # raw 순위별 chunk를 service 단위로 모은 뒤 실제 rerank helper로 재계산한다.
    raw_by_serv_id: dict[str, list[tuple[int, dict[str, str], float]]] = {}
    for raw_rank, (meta, dist) in enumerate(zip(metadatas, distances), start=1):
        raw_by_serv_id.setdefault(meta["serv_id"], []).append((raw_rank, meta, dist))

    intent = build_query_intent(
        case.request,
        query_text=query_text,
        intent_theme=_infer_intent_theme(case),
    )
    enable_section_rerank = collection_name != WELFARE_COLLECTION
    ranked = sorted(
        (
            (
                min(raw_rank for raw_rank, _, _ in chunks),
                rank_service_candidates(
                    case.request,
                    intent,
                    [(meta, dist) for _, meta, dist in chunks],
                    enable_section_rerank=enable_section_rerank,
                ),
            )
            for chunks in raw_by_serv_id.values()
        ),
        key=lambda x: x[1].score,
        reverse=True,
    )
    expected = set(case.expected_ids)

    print(f"\nDIAGNOSE {case.name}  query={query_text}")
    print(
        f"  {'#':>3}  {'raw':>4}  {'dist':>6}  {'raw_score':>9}  {'boost':>6}  "
        f"{'weighted':>8}  {'ev':>5}  {'theme':>6}  {'pen':>5}  {'final':>6}  "
        f"{'sections':<28}  serv_id          serv_nm"
    )
    for final_rank, (raw_rank, service) in enumerate(
        ranked[:15],
        start=1,
    ):
        marker = "* " if service.metadata["serv_id"] in expected else "  "
        section_text = _format_section_scores(service.section_scores)
        reason_text = ",".join(service.reasons)
        print(
            f"  {marker}{final_rank:>2}.  raw={raw_rank:>3}  "
            f"dist={service.distance:.3f}  raw_score={service.raw_score:.3f}  "
            f"boost={service.profile_boost:+.3f}  "
            f"weighted={service.section_weighted_score:.3f}  "
            f"ev={service.section_evidence_boost:+.3f}  "
            f"theme={service.theme_adjustment:+.3f}  "
            f"pen={service.negative_penalty:.3f}  "
            f"final={service.score:.3f}  {section_text:<28}  "
            f"{service.metadata['serv_id']}  {service.metadata['serv_nm'][:28]}  {reason_text}"
        )


def _rank_of(result_ids: Sequence[str], expectations: Sequence[ServiceExpectation]) -> int | None:
    expected = {expectation.serv_id for expectation in expectations}
    for idx, serv_id in enumerate(result_ids, start=1):
        if serv_id in expected:
            return idx
    return None


def _hit_at(
    result_ids: Sequence[str],
    expectations: Sequence[ServiceExpectation],
    k: int,
) -> bool:
    expected = {expectation.serv_id for expectation in expectations}
    return any(serv_id in expected for serv_id in result_ids[:k])


def _format_section_scores(section_scores: dict[str, float]) -> str:
    text = ",".join(f"{section}:{score:.3f}" for section, score in sorted(section_scores.items()))
    return text[:28]


async def _index_snapshot(collection_name: str) -> tuple[int, int, dict[str, dict[str, str]]]:
    collection = await get_collection(collection_name)

    def _get() -> Any:
        return collection.get(include=["metadatas"])

    raw: Any = await asyncio.to_thread(_get)
    service_metadata: dict[str, dict[str, str]] = {}
    for meta in raw["metadatas"]:
        if meta is None or not isinstance(meta.get("serv_id"), str):
            continue
        serv_id = meta["serv_id"]
        current = service_metadata.get(serv_id)
        if current is None or _metadata_richness(meta) > _metadata_richness(current):
            service_metadata[serv_id] = meta
    return len(raw["ids"]), len(service_metadata), service_metadata


def _metadata_richness(metadata: dict[str, str]) -> int:
    return sum(
        1
        for field in (
            "serv_nm",
            "serv_dgst",
            "tgtr_dtl_cn",
            "slct_crit_cn",
            "trgter_indvdl",
            "intrs_thema",
        )
        if metadata.get(field)
    )


def _request_key(request: SearchRequest) -> str:
    return json.dumps(request.model_dump(), ensure_ascii=False, sort_keys=True)


def _infer_intent_theme(case: EvalCase) -> str | None:
    return case.intent_theme


def _infer_query(case: EvalCase) -> str | None:
    return case.query


def _effective_query_key(case: EvalCase) -> str:
    return json.dumps(
        {
            "request": case.request.model_dump(),
            "query": _infer_query(case),
            "intent_theme": _infer_intent_theme(case),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _evaluation_query_text(case: EvalCase) -> str:
    parts = [build_query_text(case.request)]
    query = _infer_query(case)
    intent_theme = _infer_intent_theme(case)
    if query:
        parts.append(query)
    if intent_theme and not intent_theme.startswith("guardrail:"):
        parts.append(intent_theme)
    return " ".join(parts)


@dataclass(frozen=True)
class CollisionReport:
    key: str
    case_names: tuple[str, ...]
    conflict: bool
    reason: str


def raw_request_collision_reports(cases: Sequence[EvalCase]) -> tuple[CollisionReport, ...]:
    groups: dict[str, list[EvalCase]] = {}
    for case in cases:
        groups.setdefault(_request_key(case.request), []).append(case)
    return tuple(
        CollisionReport(
            key=key,
            case_names=tuple(case.name for case in group),
            conflict=False,
            reason="SearchRequest만 같은 평가 케이스 그룹",
        )
        for key, group in sorted(groups.items())
        if len(group) > 1
    )


def effective_query_collision_reports(cases: Sequence[EvalCase]) -> tuple[CollisionReport, ...]:
    groups: dict[str, list[EvalCase]] = {}
    for case in cases:
        groups.setdefault(_effective_query_key(case), []).append(case)

    reports: list[CollisionReport] = []
    for key, group in sorted(groups.items()):
        if len(group) <= 1:
            continue
        conflict_reasons: list[str] = []
        for case in group:
            excluded = {expectation.serv_id for expectation in case.excluded_ids}
            if not excluded:
                continue
            for other in group:
                if other is case:
                    continue
                other_positive = {
                    expectation.serv_id
                    for expectation in (
                        *other.must_ids,
                        *other.acceptable_ids,
                        *other.conditional_ids,
                    )
                }
                overlap = sorted(excluded & other_positive)
                if overlap:
                    conflict_reasons.append(
                        f"{case.name} excluded overlaps {other.name}: {','.join(overlap)}"
                    )

        must_sets = {
            case.name: {expectation.serv_id for expectation in case.must_ids} for case in group
        }
        distinct_must_sets = list({frozenset(ids) for ids in must_sets.values() if ids})
        if len(distinct_must_sets) > 1:
            if not _all_sets_are_nested(distinct_must_sets):
                conflict_reasons.append("must_ids sets differ without subset relationship")

        positive_sets = {
            case.name: {
                expectation.serv_id
                for expectation in (
                    *case.must_ids,
                    *case.acceptable_ids,
                    *case.conditional_ids,
                )
            }
            for case in group
        }
        distinct_positive_sets = list({frozenset(ids) for ids in positive_sets.values() if ids})
        if len(distinct_positive_sets) > 1 and not _all_sets_are_nested(distinct_positive_sets):
            conflict_reasons.append("positive expectation sets are mutually different")

        reports.append(
            CollisionReport(
                key=key,
                case_names=tuple(case.name for case in group),
                conflict=bool(conflict_reasons),
                reason="; ".join(conflict_reasons)
                if conflict_reasons
                else "documented subset/equal group",
            )
        )
    return tuple(reports)


def _all_sets_are_nested(sets: Sequence[frozenset[str]]) -> bool:
    for left in sets:
        for right in sets:
            if not (left <= right or right <= left):
                return False
    return True


def _expectation_counts(cases: Sequence[EvalCase]) -> dict[str, int]:
    return {
        "must": sum(len(case.must_ids) for case in cases),
        "acceptable": sum(len(case.acceptable_ids) for case in cases),
        "conditional": sum(len(case.conditional_ids) for case in cases),
        "excluded": sum(len(case.excluded_ids) for case in cases),
        "ambiguous": sum(1 for case in cases if case.ambiguous),
    }


def _print_contract_audit(cases: Sequence[EvalCase]) -> None:
    counts = _expectation_counts(cases)
    raw_reports = raw_request_collision_reports(cases)
    effective_reports = effective_query_collision_reports(cases)

    print("Evaluation Contract Audit")
    print(f"cases: {len(cases)}")
    print(
        "expectations: "
        f"must={counts['must']} acceptable={counts['acceptable']} "
        f"conditional={counts['conditional']} excluded={counts['excluded']} "
        f"ambiguous_cases={counts['ambiguous']}"
    )
    print(f"raw_request_collision_groups: {len(raw_reports)}")
    for report in raw_reports:
        print(f"  - {', '.join(report.case_names)}")
    conflict_reports = [report for report in effective_reports if report.conflict]
    print(f"effective_query_collision_groups: {len(effective_reports)}")
    print(f"effective_query_conflicts: {len(conflict_reports)}")
    for report in effective_reports:
        status = "CONFLICT" if report.conflict else "OK"
        print(f"  - {status}: {', '.join(report.case_names)} ({report.reason})")
    print()


def _validation_error(case_name: str, field: str, serv_id: str, detail: str) -> str:
    return (
        "EVAL_CASE_VALIDATION_ERROR "
        f"case={case_name!r} field={field!r} id={serv_id!r} detail={detail}"
    )


def _validate_eval_cases(
    indexed_service_metadata: dict[str, dict[str, str]],
    cases: Sequence[EvalCase] = EVAL_CASES,
    *,
    expected_case_count: int | None = EXPECTED_EVAL_CASE_COUNT,
) -> list[str]:
    errors: list[str] = []
    indexed_service_ids = set(indexed_service_metadata)

    if expected_case_count is not None and len(cases) != expected_case_count:
        errors.append(
            _validation_error(
                "EVAL_CASES",
                "length",
                "-",
                f"expected {expected_case_count}, got {len(cases)}",
            )
        )

    names = [case.name for case in cases]
    for name, count in Counter(names).items():
        if count > 1:
            errors.append(
                _validation_error(name, "name", name, f"duplicate name appears {count} times")
            )

    for case in cases:
        case_name = case.name or "<empty>"
        if not case.name.strip():
            errors.append(_validation_error(case_name, "name", "-", "name must not be empty"))
        if not case.notes.strip():
            errors.append(_validation_error(case_name, "notes", "-", "notes must not be empty"))
        if case.ambiguous and not (case.ambiguity_reason or "").strip():
            errors.append(
                _validation_error(
                    case_name,
                    "ambiguity_reason",
                    "-",
                    "ambiguous=True requires ambiguity_reason",
                )
            )
        if (
            not case.ambiguous
            and not case.must_ids
            and not case.acceptable_ids
            and not case.excluded_ids
        ):
            errors.append(
                _validation_error(
                    case_name,
                    "expectations",
                    "-",
                    "non-ambiguous cases need must/acceptable expectations or explicit excluded-only contract",
                )
            )
        if case.request.top_k != 10:
            errors.append(
                _validation_error(
                    case_name,
                    "request.top_k",
                    str(case.request.top_k),
                    "top_k must be exactly 10",
                )
            )

        seen_by_field: dict[str, str] = {}
        for field_name, expectations in (
            ("must_ids", case.must_ids),
            ("acceptable_ids", case.acceptable_ids),
            ("conditional_ids", case.conditional_ids),
            ("excluded_ids", case.excluded_ids),
        ):
            for expectation in expectations:
                if not isinstance(expectation, ServiceExpectation):
                    errors.append(
                        _validation_error(
                            case_name,
                            field_name,
                            repr(expectation),
                            "expectation must be a ServiceExpectation",
                        )
                    )
                    continue
                serv_id = expectation.serv_id
                if not serv_id.strip():
                    errors.append(
                        _validation_error(case_name, field_name, "-", "serv_id must not be empty")
                    )
                if not expectation.reason.strip():
                    errors.append(
                        _validation_error(case_name, field_name, serv_id, "reason is required")
                    )
                if expectation.reason in {DEFAULT_EXPECTATION_REASON, DEFAULT_EXCLUSION_REASON}:
                    errors.append(
                        _validation_error(
                            case_name,
                            field_name,
                            serv_id,
                            "default reason is not sufficiently auditable",
                        )
                    )
                if expectation.evidence_field is not None:
                    if expectation.evidence_field not in EVIDENCE_FIELDS:
                        errors.append(
                            _validation_error(
                                case_name,
                                field_name,
                                serv_id,
                                f"unsupported evidence_field {expectation.evidence_field!r}",
                            )
                        )
                    elif serv_id in indexed_service_metadata and not indexed_service_metadata[
                        serv_id
                    ].get(expectation.evidence_field):
                        errors.append(
                            _validation_error(
                                case_name,
                                field_name,
                                serv_id,
                                f"evidence_field {expectation.evidence_field!r} is empty",
                            )
                        )
                if field_name == "must_ids" and expectation.evidence_field is None:
                    errors.append(
                        _validation_error(
                            case_name,
                            field_name,
                            serv_id,
                            "must_ids require evidence_field",
                        )
                    )
                if field_name == "must_ids" and "평가 메모" in expectation.reason:
                    errors.append(
                        _validation_error(
                            case_name,
                            field_name,
                            serv_id,
                            "must_ids reason cannot rely on evaluation notes",
                        )
                    )
                if field_name == "conditional_ids" and not (expectation.condition or "").strip():
                    errors.append(
                        _validation_error(
                            case_name,
                            field_name,
                            serv_id,
                            "conditional_ids require condition",
                        )
                    )
                previous_field = seen_by_field.get(serv_id)
                if previous_field is not None:
                    errors.append(
                        _validation_error(
                            case_name,
                            field_name,
                            serv_id,
                            f"serv_id also appears in {previous_field}",
                        )
                    )
                seen_by_field[serv_id] = field_name
                if serv_id not in indexed_service_ids:
                    errors.append(
                        _validation_error(
                            case_name,
                            field_name,
                            serv_id,
                            "serv_id is not present in the current Chroma index",
                        )
                    )

    for report in effective_query_collision_reports(cases):
        if report.conflict:
            errors.append(
                _validation_error(
                    "EVAL_CASES",
                    "effective_query_collision",
                    "-",
                    f"{','.join(report.case_names)}: {report.reason}",
                )
            )

    return errors


@dataclass(frozen=True)
class EvaluationRow:
    case: EvalCase
    result_ids: tuple[str, ...]
    results: tuple[tuple[str, str, float], ...]
    must_rank: int | None
    acceptable_rank: int | None
    conditional_rank: int | None
    excluded_hits_at5: tuple[str, ...]


@dataclass(frozen=True)
class RatioMetric:
    numerator: int
    denominator: int

    @property
    def value(self) -> float:
        if self.denominator == 0:
            return 0.0
        return self.numerator / self.denominator


@dataclass(frozen=True)
class EvaluationMetrics:
    must_hit_at: dict[int, RatioMetric]
    acceptable_hit_at5: RatioMetric
    must_or_acceptable_hit_at5: RatioMetric
    exclusion_pass_at5: RatioMetric
    exclusion_pass_at5_on_excluded_cases: RatioMetric
    conditional_hit_at5: RatioMetric
    mrr_must: RatioMetric
    exclusion_violations_at5: tuple[tuple[str, tuple[str, ...]], ...]


def compute_metrics(rows: Sequence[EvaluationRow]) -> EvaluationMetrics:
    eligible_must_rows = [row for row in rows if not row.case.ambiguous and row.case.must_ids]
    must_hit_at = {
        k: RatioMetric(
            sum(_hit_at(row.result_ids, row.case.must_ids, k) for row in eligible_must_rows),
            len(eligible_must_rows),
        )
        for k in (1, 3, 5)
    }

    acceptable_rows = [row for row in rows if not row.case.ambiguous and row.case.acceptable_ids]
    eligible_positive_rows = [
        row
        for row in rows
        if not row.case.ambiguous and (row.case.must_ids or row.case.acceptable_ids)
    ]
    excluded_rows = [row for row in rows if row.case.excluded_ids]
    conditional_rows = [row for row in rows if row.case.conditional_ids]
    exclusion_violations = tuple(
        (row.case.name, row.excluded_hits_at5) for row in rows if row.excluded_hits_at5
    )

    mrr_numerator = sum(
        0.0 if row.must_rank is None else 1.0 / row.must_rank for row in eligible_must_rows
    )

    return EvaluationMetrics(
        must_hit_at=must_hit_at,
        acceptable_hit_at5=RatioMetric(
            sum(_hit_at(row.result_ids, row.case.acceptable_ids, 5) for row in acceptable_rows),
            len(acceptable_rows),
        ),
        must_or_acceptable_hit_at5=RatioMetric(
            sum(
                _hit_at(row.result_ids, (*row.case.must_ids, *row.case.acceptable_ids), 5)
                for row in eligible_positive_rows
            ),
            len(eligible_positive_rows),
        ),
        exclusion_pass_at5=RatioMetric(
            sum(1 for row in rows if not row.excluded_hits_at5),
            len(rows),
        ),
        exclusion_pass_at5_on_excluded_cases=RatioMetric(
            sum(1 for row in excluded_rows if not row.excluded_hits_at5),
            len(excluded_rows),
        ),
        conditional_hit_at5=RatioMetric(
            sum(_hit_at(row.result_ids, row.case.conditional_ids, 5) for row in conditional_rows),
            len(conditional_rows),
        ),
        mrr_must=RatioMetric(round(mrr_numerator * 1_000_000), len(eligible_must_rows) * 1_000_000),
        exclusion_violations_at5=exclusion_violations,
    )


def _format_ratio(metric: RatioMetric) -> str:
    return f"{metric.value:.3f} ({metric.numerator}/{metric.denominator})"


async def _search_eval_case(
    case: EvalCase,
    embedder: KoSimCSEEmbedder,
    *,
    collection_name: str,
    adaptive_fetch: bool,
) -> SearchResponse:
    query_text = _evaluation_query_text(case)
    vec: list[float] = embedder.embed([query_text])[0]
    collection = await get_collection(collection_name)
    candidate_limit = ADAPTIVE_MAX_CANDIDATES if adaptive_fetch else DEFAULT_MAX_CANDIDATES
    n_results = _initial_candidate_count(case.request.top_k, adaptive_fetch, candidate_limit)

    while True:
        raw = await _query_collection(collection, vec, n_results)
        intent = build_query_intent(
            case.request,
            query_text=query_text,
            intent_theme=_infer_intent_theme(case),
        )
        search_results = _response_results_from_raw(
            case.request,
            raw,
            enable_section_rerank=collection_name != WELFARE_COLLECTION,
            intent=intent,
        )
        if (
            not adaptive_fetch
            or len(search_results) >= case.request.top_k
            or n_results >= candidate_limit
        ):
            break
        next_n_results = min(candidate_limit, n_results * 2)
        if next_n_results == n_results:
            break
        n_results = next_n_results

    return SearchResponse(results=search_results)


async def evaluate(
    verbose: bool,
    diagnose: bool = False,
    collection_name: str = WELFARE_COLLECTION,
    adaptive_fetch: bool = False,
) -> int:
    chunk_count, service_count, indexed_service_metadata = await _index_snapshot(collection_name)
    _print_contract_audit(EVAL_CASES)
    validation_errors = _validate_eval_cases(indexed_service_metadata)
    if validation_errors:
        for error in validation_errors:
            print(error, file=sys.stderr)
        return 2

    embedder = KoSimCSEEmbedder()

    rows: list[EvaluationRow] = []
    latencies_ms: list[float] = []
    eligibility_status_counts: Counter[str] = Counter()

    for case in EVAL_CASES:
        started_at = time.perf_counter()
        response = await _search_eval_case(
            case,
            embedder,
            collection_name=collection_name,
            adaptive_fetch=adaptive_fetch,
        )
        latencies_ms.append((time.perf_counter() - started_at) * 1000)
        eligibility_status_counts.update(result.eligibility_status for result in response.results)
        results = [(r.serv_id, r.serv_nm, r.score) for r in response.results]
        result_ids = [serv_id for serv_id, _, _ in results]
        excluded_ids = {expectation.serv_id for expectation in case.excluded_ids}
        rows.append(
            EvaluationRow(
                case=case,
                result_ids=tuple(result_ids),
                results=tuple(results),
                must_rank=_rank_of(result_ids, case.must_ids),
                acceptable_rank=_rank_of(result_ids, case.acceptable_ids),
                conditional_rank=_rank_of(result_ids, case.conditional_ids),
                excluded_hits_at5=tuple(
                    serv_id for serv_id in result_ids[:5] if serv_id in excluded_ids
                ),
            )
        )

    metrics = compute_metrics(rows)
    avg_latency_ms = sum(latencies_ms) / len(latencies_ms)
    p95_index = min(len(latencies_ms) - 1, math.ceil(len(latencies_ms) * 0.95) - 1)
    p95_latency_ms = sorted(latencies_ms)[p95_index]
    returned_result_count = sum(eligibility_status_counts.values())

    print("Search Quality Evaluation")
    print(f"collection: {collection_name}")
    print(f"index: {chunk_count} chunks / {service_count} services")
    print(f"cases: {len(rows)}")
    for k in (1, 3, 5):
        print(f"must_hit@{k}: {_format_ratio(metrics.must_hit_at[k])}")
    print(f"acceptable_hit@5: {_format_ratio(metrics.acceptable_hit_at5)}")
    print(f"must_or_acceptable_hit@5: {_format_ratio(metrics.must_or_acceptable_hit_at5)}")
    print(f"exclusion_pass@5: {_format_ratio(metrics.exclusion_pass_at5)}")
    print(
        "exclusion_pass@5_on_excluded_cases: "
        f"{_format_ratio(metrics.exclusion_pass_at5_on_excluded_cases)}"
    )
    print(f"conditional_hit@5: {_format_ratio(metrics.conditional_hit_at5)}")
    print(
        f"mrr_must: {metrics.mrr_must.value:.3f} ({len([row for row in rows if not row.case.ambiguous and row.case.must_ids])} cases)"
    )
    print(f"exclusion_violations@5: {len(metrics.exclusion_violations_at5)}")
    for case_name, hits in metrics.exclusion_violations_at5:
        print(f"  {case_name}: {','.join(hits)}")
    print(f"avg_latency_ms: {avg_latency_ms:.1f}")
    print(f"p95_latency_ms: {p95_latency_ms:.1f}")
    print("eligibility_status_distribution:")
    for status in ("likely", "needs_more_info", "unlikely"):
        count = eligibility_status_counts[status]
        ratio = 0.0 if returned_result_count == 0 else count / returned_result_count
        print(f"  {status}: {count} ({ratio:.3f})")
    print()

    for row in rows:
        case = row.case
        positive_hit = (
            case.ambiguous
            or _hit_at(row.result_ids, (*case.must_ids, *case.acceptable_ids), 5)
            or not (case.must_ids or case.acceptable_ids)
        )
        status = "PASS" if positive_hit and not row.excluded_hits_at5 else "FAIL"
        must_rank_text = "-" if row.must_rank is None else str(row.must_rank)
        acceptable_rank_text = "-" if row.acceptable_rank is None else str(row.acceptable_rank)
        conditional_rank_text = "-" if row.conditional_rank is None else str(row.conditional_rank)
        print(
            f"{status} {case.name}: "
            f"must_rank={must_rank_text} acceptable_rank={acceptable_rank_text} "
            f"conditional_rank={conditional_rank_text}"
        )
        if case.excluded_ids:
            excluded_text = ",".join(expectation.serv_id for expectation in case.excluded_ids)
            hit_text = "-" if not row.excluded_hits_at5 else ",".join(row.excluded_hits_at5)
            print(f"  excluded: {excluded_text}")
            print(f"  excluded_hits@5: {hit_text}")
        if verbose or status == "FAIL":
            print(f"  query: {_evaluation_query_text(case)}")
            if case.ambiguous:
                print(f"  ambiguous: {case.ambiguity_reason}")
            print(f"  notes: {case.notes}")
            must_ids = {expectation.serv_id for expectation in case.must_ids}
            acceptable_ids = {expectation.serv_id for expectation in case.acceptable_ids}
            conditional_ids = {expectation.serv_id for expectation in case.conditional_ids}
            excluded_ids = {expectation.serv_id for expectation in case.excluded_ids}
            for pos, (serv_id, serv_nm, score) in enumerate(row.results[:10], start=1):
                if serv_id in must_ids:
                    marker = "M"
                elif serv_id in acceptable_ids:
                    marker = "A"
                elif serv_id in conditional_ids:
                    marker = "C"
                elif serv_id in excluded_ids:
                    marker = "X"
                else:
                    marker = " "
                print(f"  {marker}{pos:>2}. {serv_id} {serv_nm} score={score:.3f}")
        if diagnose and status == "FAIL":
            await _diagnose_case(case, embedder, collection_name, adaptive_fetch)
    return (
        0
        if metrics.must_or_acceptable_hit_at5.value >= 0.8
        and len(metrics.exclusion_violations_at5) == 0
        else 1
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Print top results for every case")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="For FAIL cases: show raw distance rank, boost, and final rank side-by-side",
    )
    parser.add_argument(
        "--chroma-persist-dir",
        default=DEFAULT_CHROMA_PERSIST_DIR,
        help="Chroma persist directory to evaluate. Defaults to the baseline data/chroma.",
    )
    parser.add_argument(
        "--collection",
        default=WELFARE_COLLECTION,
        help="Chroma collection to evaluate. Defaults to the baseline welfare_services.",
    )
    args = parser.parse_args()

    requested_path = Path(args.chroma_persist_dir)
    baseline_path = Path(DEFAULT_CHROMA_PERSIST_DIR).resolve()
    adaptive_fetch = (
        args.collection != WELFARE_COLLECTION or requested_path.resolve() != baseline_path
    )

    with _evaluation_persist_dir(requested_path) as evaluation_path:
        os.environ["CHROMA_PERSIST_DIR"] = str(evaluation_path)
        raise SystemExit(
            asyncio.run(
                evaluate(
                    verbose=args.verbose,
                    diagnose=args.diagnose,
                    collection_name=args.collection,
                    adaptive_fetch=adaptive_fetch,
                )
            )
        )


if __name__ == "__main__":
    main()

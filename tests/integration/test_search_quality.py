from __future__ import annotations

import os
from collections.abc import Generator

import pytest

import src.db.chroma as _chroma_module
from scripts.evaluate_search import ServiceExpectation, svc
from src.embedding.kosimcse import KoSimCSEEmbedder
from src.models.welfare import SearchRequest, WelfareRaw
from src.pipeline.index import index_welfare_items
from src.retriever.search import search_welfare

# ── 테스트 데이터 (12개, 6개 도메인 × 2) ─────────────────────────────────────
# 실제 운영 데이터와 유사하게 도메인 내 경쟁 항목을 포함한다.

_WELFARE_ITEMS = [
    # ── 노인 ──────────────────────────────────────────────────────────────────
    WelfareRaw(
        serv_id="T001",
        serv_nm="기초연금",
        serv_dgst="65세 이상 저소득 어르신에게 매월 연금을 지급합니다.",
        jur_mnof_nm="보건복지부",
        trgter_indvdl=["노인", "저소득"],
        intrs_thema=["소득지원"],
        tgtr_dtl_cn="만 65세 이상이고 소득인정액이 기준액 이하인 어르신",
        slct_crit_cn="소득인정액 하위 70% 이하 노인",
        alw_serv_cn="매월 최대 32만원 현금 지급",
    ),
    WelfareRaw(
        serv_id="T002",
        serv_nm="독거노인 돌봄 서비스",
        serv_dgst="혼자 사는 노인에게 안전 확인 및 생활 지원을 제공합니다.",
        jur_mnof_nm="보건복지부",
        trgter_indvdl=["노인"],
        intrs_thema=["돌봄"],
        tgtr_dtl_cn="65세 이상 독거노인 중 돌봄이 필요한 분",
        slct_crit_cn="기초생활수급자 또는 차상위계층인 독거노인",
        alw_serv_cn="정기 안전 확인, 생활 교육, 서비스 연계",
    ),
    # ── 청년 ──────────────────────────────────────────────────────────────────
    WelfareRaw(
        serv_id="T003",
        serv_nm="청년 취업 지원사업",
        serv_dgst="구직 중인 청년에게 취업 훈련과 일자리를 연계합니다.",
        jur_mnof_nm="고용노동부",
        trgter_indvdl=["청년"],
        intrs_thema=["일자리"],
        tgtr_dtl_cn="만 18세 이상 34세 이하 구직 활동 중인 청년",
        slct_crit_cn="고용보험 미가입 실업 상태의 청년",
        alw_serv_cn="직업훈련 프로그램 참여 및 취업 알선",
    ),
    WelfareRaw(
        serv_id="T004",
        serv_nm="청년 월세 특별지원",
        serv_dgst="저소득 청년의 주거비 부담을 줄이기 위해 월세를 지원합니다.",
        jur_mnof_nm="국토교통부",
        trgter_indvdl=["청년"],
        intrs_thema=["주거"],
        tgtr_dtl_cn="만 19세 이상 34세 이하 독립 거주 무주택 청년",
        slct_crit_cn="청년 독립 가구 소득이 기준 중위소득 60% 이하",
        alw_serv_cn="월 최대 20만원 월세 지원",
    ),
    # ── 장애인 ────────────────────────────────────────────────────────────────
    WelfareRaw(
        serv_id="T005",
        serv_nm="장애인 활동지원 서비스",
        serv_dgst="혼자 일상생활이 어려운 중증 장애인에게 활동보조를 지원합니다.",
        jur_mnof_nm="보건복지부",
        trgter_indvdl=["장애인"],
        intrs_thema=["장애인복지"],
        tgtr_dtl_cn="만 6세 이상 65세 미만 등록 장애인 중 혼자 일상생활이 어려운 중증 장애인",
        slct_crit_cn="장애 정도가 심한 중증 장애인",
        alw_serv_cn="활동보조인 파견, 방문목욕, 방문간호",
    ),
    WelfareRaw(
        serv_id="T006",
        serv_nm="장애인 연금",
        serv_dgst="중증 장애인의 안정적 생활을 위해 연금을 지급합니다.",
        jur_mnof_nm="보건복지부",
        trgter_indvdl=["장애인", "저소득"],
        intrs_thema=["소득지원"],
        tgtr_dtl_cn="만 18세 이상 중증 장애인 중 소득인정액이 기준액 이하인 자",
        slct_crit_cn="소득인정액이 선정기준액 이하인 18세 이상 중증 장애인",
        alw_serv_cn="매월 장애인 연금 지급",
    ),
    # ── 한부모·아동 ────────────────────────────────────────────────────────────
    WelfareRaw(
        serv_id="T007",
        serv_nm="한부모 가족 아동양육비 지원",
        serv_dgst="한부모 가족의 아동 양육을 지원합니다.",
        jur_mnof_nm="여성가족부",
        trgter_indvdl=["한부모가족", "아동"],
        intrs_thema=["가족지원"],
        tgtr_dtl_cn="이혼, 사별 등으로 홀로 자녀를 양육하는 한부모 가족",
        slct_crit_cn="소득인정액이 기준 중위소득 60% 이하인 한부모의 만 18세 미만 자녀",
        alw_serv_cn="아동 1인당 월 아동양육비 지급",
    ),
    WelfareRaw(
        serv_id="T008",
        serv_nm="영유아 보육료 지원",
        serv_dgst="어린이집을 이용하는 영유아 가정에 보육료를 지원합니다.",
        jur_mnof_nm="보건복지부",
        trgter_indvdl=["영유아", "아동"],
        intrs_thema=["보육"],
        tgtr_dtl_cn="만 0~5세 영유아로 어린이집을 이용하는 아동의 부모",
        slct_crit_cn="소득에 관계없이 어린이집 이용 영유아 가구",
        alw_serv_cn="보육료 전액 또는 일부 지원",
    ),
    # ── 기초생활수급 ──────────────────────────────────────────────────────────
    WelfareRaw(
        serv_id="T009",
        serv_nm="기초생활수급자 생계급여",
        serv_dgst="최저 생활을 보장하기 위해 생계급여를 지급합니다.",
        jur_mnof_nm="보건복지부",
        trgter_indvdl=["기초생활수급자"],
        intrs_thema=["생활지원"],
        tgtr_dtl_cn="소득인정액이 기준 중위소득 30% 이하인 기초생활수급자 가구",
        slct_crit_cn="기초생활수급자 선정기준을 충족하는 가구",
        alw_serv_cn="매월 생계급여 현금 지급",
    ),
    WelfareRaw(
        serv_id="T010",
        serv_nm="의료급여",
        serv_dgst="저소득층의 의료비 부담을 줄이기 위해 의료급여를 지원합니다.",
        jur_mnof_nm="보건복지부",
        trgter_indvdl=["기초생활수급자", "차상위계층"],
        intrs_thema=["의료"],
        tgtr_dtl_cn="국민기초생활보장법에 따른 의료급여 수급권자",
        slct_crit_cn="기초생활수급자 또는 차상위계층 중 의료급여 조건을 충족하는 자",
        alw_serv_cn="외래, 입원, 약제 등 의료비 지원",
    ),
    # ── 실업·고용 ──────────────────────────────────────────────────────────────
    WelfareRaw(
        serv_id="T011",
        serv_nm="실업급여",
        serv_dgst="고용보험 가입자가 비자발적 실직 시 구직 활동을 지원합니다.",
        jur_mnof_nm="고용노동부",
        trgter_indvdl=["실업자"],
        intrs_thema=["일자리"],
        tgtr_dtl_cn="고용보험 피보험자로서 이직 전 18개월 중 180일 이상 근무한 실업자",
        slct_crit_cn="비자발적 이직 후 적극적으로 구직 활동 중인 자",
        alw_serv_cn="매월 구직급여 지급, 취업 알선 서비스",
    ),
    WelfareRaw(
        serv_id="T012",
        serv_nm="자활 지원사업",
        serv_dgst="기초생활수급자의 경제적 자립을 위한 자활 근로 기회를 제공합니다.",
        jur_mnof_nm="보건복지부",
        trgter_indvdl=["기초생활수급자"],
        intrs_thema=["일자리"],
        tgtr_dtl_cn="근로 능력이 있는 기초생활수급자 및 차상위계층",
        slct_crit_cn="자활 근로 참여를 희망하는 기초생활수급자",
        alw_serv_cn="자활 근로 참여 기회 및 자활급여 지급",
    ),
]

# ── 검색 케이스 (name, request, expected_serv_id, top_k) ───────────────────────
# top_k: 해당 서비스가 상위 N위 안에 들어야 통과

_SEARCH_CASES = [
    # ── 노인 ──────────────────────────────────────────────────────────────────
    (
        "노인_저소득_기초연금_top3",
        SearchRequest(age=67, income_level="저소득", top_k=5),
        "T001",
        3,
    ),
    (
        "노인_기초수급_기초연금_top3",
        SearchRequest(age=72, income_level="기초생활수급자", top_k=5),
        "T001",
        3,
    ),
    (
        "노인_기초수급_독거돌봄_top5",
        SearchRequest(age=70, income_level="기초생활수급자", top_k=5),
        "T002",
        5,
    ),
    # ── 청년 ──────────────────────────────────────────────────────────────────
    (
        "청년_실업_취업지원_top3",
        SearchRequest(age=25, income_level="일반", employment_status="실업", top_k=5),
        "T003",
        3,
    ),
    (
        "청년_저소득_실업_취업지원_top3",
        SearchRequest(age=28, income_level="저소득", employment_status="실업", top_k=5),
        "T003",
        3,
    ),
    (
        "청년_저소득_월세지원_top5",
        SearchRequest(age=23, income_level="저소득", top_k=5),
        "T004",
        5,
    ),
    # ── 장애인 ────────────────────────────────────────────────────────────────
    (
        "중증장애인_40대_활동지원_top3",
        SearchRequest(
            age=40, income_level="저소득", disability=True, disability_severity="중증", top_k=5
        ),
        "T005",
        3,
    ),
    (
        "중증장애인_50대_장애연금_top3",
        SearchRequest(
            age=52, income_level="저소득", disability=True, disability_severity="중증", top_k=5
        ),
        "T006",
        3,
    ),
    # ── 한부모 ────────────────────────────────────────────────────────────────
    (
        "한부모_이혼_자녀있음_양육비_top3",
        SearchRequest(
            age=35, income_level="저소득", marital_status="이혼", has_children=True, top_k=5
        ),
        "T007",
        3,
    ),
    (
        "한부모_사별_자녀있음_양육비_top3",
        SearchRequest(
            age=40, income_level="저소득", marital_status="사별", has_children=True, top_k=5
        ),
        "T007",
        3,
    ),
    # ── 기초수급 ──────────────────────────────────────────────────────────────
    (
        "기초수급_중년_생계급여_top3",
        SearchRequest(age=45, income_level="기초생활수급자", top_k=5),
        "T009",
        3,
    ),
    (
        "기초수급_중년_의료급여_top5",
        SearchRequest(age=50, income_level="기초생활수급자", top_k=5),
        "T010",
        5,
    ),
    # ── 실업·고용 ──────────────────────────────────────────────────────────────
    (
        "중년_실업_실업급여_top3",
        SearchRequest(age=38, income_level="일반", employment_status="실업", top_k=5),
        "T011",
        3,
    ),
    (
        "기초수급_근로능력_자활_top5",
        SearchRequest(age=42, income_level="기초생활수급자", employment_status="실업", top_k=5),
        "T012",
        5,
    ),
    (
        "청년_기초수급_실업_취업지원_top3",
        SearchRequest(age=27, income_level="기초생활수급자", employment_status="실업", top_k=5),
        "T003",
        3,
    ),
]

# 파라미터 이름을 pytest 예약어 'request'와 충돌하지 않도록 'req'로 사용
_SEARCH_CASES_PARAMS = [
    pytest.param(
        name,
        req,
        svc(expected_id, "integration search quality must-hit contract", "serv_nm"),
        top_k,
        id=name,
    )
    for name, req, expected_id, top_k in _SEARCH_CASES
]


_ELIGIBILITY_ITEMS = [
    WelfareRaw(
        serv_id="E001",
        serv_nm="독거노인·장애인 응급안전안심서비스",
        serv_dgst="독거노인과 장애인 가구의 응급 상황 대응을 지원합니다.",
        jur_mnof_nm="보건복지부",
        trgter_indvdl=["노인", "장애인"],
        intrs_thema=["돌봄"],
        slct_crit_cn=(
            "독거노인은 실제로 혼자 살고있는 만 65세 이상의 노인입니다. "
            "장애인 가구는 장애인활동지원 수급자이면서 독거 또는 취약가구에 "
            "해당하는 장애인입니다."
        ),
    ),
    WelfareRaw(
        serv_id="E002",
        serv_nm="가사·간병 방문 지원사업",
        serv_dgst="가사·간병 서비스가 필요한 저소득 가구를 지원합니다.",
        jur_mnof_nm="보건복지부",
        trgter_indvdl=["저소득"],
        intrs_thema=["돌봄"],
        tgtr_dtl_cn=(
            "만 65세 미만의 기준중위소득 70% 이하 계층 중 가사·간병 서비스가 "
            "필요한 자를 지원합니다."
        ),
        slct_crit_cn=(
            "장애의 정도가 심한 장애인, 6개월 이상 치료를 요하는 중증질환자, "
            "희귀난치성 질환자, 한부모가정, 만 65세 미만의 의료급여 수급자 중 "
            "장기입원 사례관리 퇴원자 등이 신청할 수 있습니다."
        ),
    ),
    WelfareRaw(
        serv_id="E003",
        serv_nm="근로·자녀장려금",
        serv_dgst="저소득 근로 가구와 자녀 양육 가구에 장려금을 지급합니다.",
        jur_mnof_nm="국세청",
        trgter_indvdl=["저소득"],
        intrs_thema=["소득지원"],
        slct_crit_cn=(
            "근로소득 또는 사업소득 또는 종교인소득이 있는 거주자로서 요건을 "
            "갖춘 경우 신청 가능합니다. 자녀장려금은 부양자녀가 있는 경우 "
            "적용됩니다."
        ),
    ),
    WelfareRaw(
        serv_id="E004",
        serv_nm="생계급여(맞춤형 급여)",
        serv_dgst="저소득 가구의 최저 생활을 보장하기 위해 생계급여를 지급합니다.",
        jur_mnof_nm="보건복지부",
        trgter_indvdl=["기초생활수급자"],
        intrs_thema=["생활지원"],
        tgtr_dtl_cn=(
            "가구의 소득인정액이 생계급여 선정기준 이하로서 생계급여 수급자로 "
            "결정된 수급자에게 지급합니다."
        ),
        slct_crit_cn="생계급여 기준 중위소득 32% 이하입니다.",
    ),
    WelfareRaw(
        serv_id="E005",
        serv_nm="자활근로(기초, 차상위)",
        serv_dgst="저소득층의 자립을 위해 자활근로 참여 기회를 제공합니다.",
        jur_mnof_nm="보건복지부",
        trgter_indvdl=["기초생활수급자", "차상위계층"],
        intrs_thema=["일자리"],
        tgtr_dtl_cn="국민기초생활보장법에 따른 수급자 및 차상위 계층을 지원합니다.",
        slct_crit_cn=(
            "조건부수급자, 자활급여특례자, 일반수급자, 차상위자 중 자활사업 참여 "
            "자격이 있는 사람을 대상으로 합니다."
        ),
    ),
]


@pytest.fixture(scope="session")
def embedder() -> KoSimCSEEmbedder:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return KoSimCSEEmbedder()


@pytest.fixture(autouse=True)
def reset_chroma(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    monkeypatch.setenv("CHROMA_MODE", "ephemeral")
    _chroma_module._client = None
    _chroma_module._lock = None
    yield
    _chroma_module._client = None
    _chroma_module._lock = None


@pytest.mark.integration
class TestSearchQuality:
    """실제 KoSimCSE 임베딩 + in-memory ChromaDB 로 검색 관련도를 검증한다."""

    @pytest.mark.parametrize("name,req,expected,top_k", _SEARCH_CASES_PARAMS)
    async def test_search_accuracy(
        self,
        name: str,
        req: SearchRequest,
        expected: ServiceExpectation,
        top_k: int,
        embedder: KoSimCSEEmbedder,
    ) -> None:
        await index_welfare_items(_WELFARE_ITEMS, embedder)
        response = await search_welfare(req, embedder)
        result_ids = [r.serv_id for r in response.results]
        assert expected.serv_id in result_ids[:top_k], (
            f"[{name}] 기대 서비스 {expected.serv_id}가 top{top_k} 안에 없음. 실제 순위: {result_ids}"
        )


@pytest.mark.integration
class TestEligibilityGuardrail:
    async def test_filters_unlikely_and_marks_needs_more_info(
        self,
        embedder: KoSimCSEEmbedder,
    ) -> None:
        await index_welfare_items(_ELIGIBILITY_ITEMS, embedder)

        response = await search_welfare(
            SearchRequest(
                age=61,
                income_level="저소득",
                household_size=1,
                marital_status="사별",
                has_children=False,
                disability=False,
                employment_status="비경제활동",
                region="대구",
                top_k=5,
            ),
            embedder,
        )

        results_by_id = {result.serv_id: result for result in response.results}
        result_ids = list(results_by_id)

        assert "E001" not in result_ids
        assert results_by_id["E002"].eligibility_status == "needs_more_info"
        assert results_by_id["E003"].eligibility_status == "needs_more_info"
        assert results_by_id["E004"].eligibility_status in {"likely", "needs_more_info"}

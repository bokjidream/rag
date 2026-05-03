from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.models.welfare import (
    SearchRequest,
    SearchResponse,
    SearchResult,
    WelfareDetail,
    WelfareRaw,
)


class TestSearchRequest:
    def test_required_fields_only(self) -> None:
        req = SearchRequest(age=30, income_level="저소득")
        assert req.age == 30
        assert req.income_level == "저소득"
        assert req.disability is False
        assert req.top_k == 5

    def test_missing_age_raises(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(income_level="저소득")  # type: ignore[call-arg]

    def test_missing_income_level_raises(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(age=30)  # type: ignore[call-arg]

    def test_invalid_income_level_raises(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(age=30, income_level="부자")  # type: ignore[arg-type]

    def test_invalid_marital_status_raises(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(age=30, income_level="저소득", marital_status="동거")  # type: ignore[arg-type]

    def test_invalid_employment_status_raises(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(age=30, income_level="저소득", employment_status="무직")  # type: ignore[arg-type]

    def test_all_optional_fields(self) -> None:
        req = SearchRequest(
            age=65,
            income_level="기초생활수급자",
            household_size=1,
            marital_status="사별",
            has_children=False,
            disability=True,
            disability_severity="중증",
            employment_status="비경제활동",
            region="서울",
            top_k=10,
        )
        assert req.household_size == 1
        assert req.disability_severity == "중증"
        assert req.region == "서울"

    @pytest.mark.parametrize("level", ["기초생활수급자", "차상위계층", "저소득", "일반"])
    def test_all_income_levels_valid(self, level: str) -> None:
        req = SearchRequest(age=30, income_level=level)  # type: ignore[arg-type]
        assert req.income_level == level

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("age", -1),
            ("age", 131),
            ("household_size", 0),
            ("household_size", 21),
            ("top_k", 0),
            ("top_k", 51),
            ("region", ""),
        ],
    )
    def test_invalid_bounds_raise(self, field: str, value: object) -> None:
        payload: dict[str, object] = {"age": 30, "income_level": "저소득", field: value}
        with pytest.raises(ValidationError):
            SearchRequest(**payload)  # type: ignore[arg-type]

    def test_disability_requires_severity(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(age=30, income_level="저소득", disability=True)

    def test_disability_severity_requires_disability_true(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(age=30, income_level="저소득", disability_severity="중증")

    def test_pregnant_defaults_false(self) -> None:
        req = SearchRequest(age=30, income_level="저소득")
        assert req.pregnant is False

    def test_pregnant_true(self) -> None:
        req = SearchRequest(age=30, income_level="저소득", pregnant=True)
        assert req.pregnant is True

    def test_strips_whitespace(self) -> None:
        req = SearchRequest(age=30, income_level="저소득", region=" 서울 ")
        assert req.region == "서울"


class TestSearchResult:
    def test_includes_department_field(self) -> None:
        result = SearchResult(
            serv_id="WLF00000001",
            serv_nm="복지서비스명",
            serv_dgst="서비스 개요",
            department="국토교통부",
            score=0.87,
            trgter_indvdl=["저소득"],
            intrs_thema=["주거"],
        )
        assert result.department == "국토교통부"
        assert result.score == 0.87
        assert result.trgter_indvdl == ["저소득"]

    def test_missing_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            SearchResult(  # type: ignore[call-arg]
                serv_id="WLF00000001",
                serv_nm="복지서비스명",
                serv_dgst="서비스 개요",
                score=0.87,
                trgter_indvdl=[],
                intrs_thema=[],
                # department 누락
            )

    def test_from_metadata_maps_fields(self) -> None:
        result = SearchResult.from_metadata(
            {
                "serv_id": "WLF00000001",
                "serv_nm": "복지서비스명",
                "serv_dgst": "서비스 개요",
                "jur_mnof_nm": "국토교통부",
                "trgter_indvdl": json.dumps(["저소득"], ensure_ascii=False),
                "intrs_thema": json.dumps(["주거"], ensure_ascii=False),
            },
            distance=0.2,
        )
        assert result.department == "국토교통부"
        assert result.score == pytest.approx(0.8)
        assert result.trgter_indvdl == ["저소득"]


class TestSearchResponse:
    def test_empty_results(self) -> None:
        resp = SearchResponse(results=[])
        assert resp.results == []

    def test_with_results(self) -> None:
        result = SearchResult(
            serv_id="WLF00000001",
            serv_nm="복지서비스명",
            serv_dgst="서비스 개요",
            department="복지부",
            score=0.9,
            trgter_indvdl=[],
            intrs_thema=[],
        )
        resp = SearchResponse(results=[result])
        assert len(resp.results) == 1


class TestWelfareDetail:
    def test_includes_application_fields(self) -> None:
        detail = WelfareDetail(
            serv_id="WLF00000035",
            serv_nm="서비스명",
            serv_dgst="서비스 개요",
            tgtr_dtl_cn="수급 대상 상세",
            slct_crit_cn="선정 기준",
            alw_serv_cn="서비스 내용",
            sprt_cyc_nm="년",
            srv_pvsn_nm="현금지급",
            trgter_indvdl=["저소득"],
            intrs_thema=["주거", "생활지원"],
            application_url="https://bokjiro.go.kr/",
            required_documents=[],
            application_fields=[],
        )
        assert detail.application_url == "https://bokjiro.go.kr/"
        assert detail.required_documents == []
        assert detail.application_fields == []

    def test_default_empty_lists(self) -> None:
        detail = WelfareDetail(
            serv_id="WLF00000035",
            serv_nm="서비스명",
            serv_dgst="서비스 개요",
            tgtr_dtl_cn="수급 대상",
            slct_crit_cn="선정 기준",
            alw_serv_cn="서비스 내용",
            sprt_cyc_nm="월",
            srv_pvsn_nm="서비스제공",
            trgter_indvdl=[],
            intrs_thema=[],
            application_url="https://bokjiro.go.kr/",
        )
        assert detail.required_documents == []
        assert detail.application_fields == []

    def test_from_metadata_maps_fields(self) -> None:
        detail = WelfareDetail.from_metadata(
            {
                "serv_id": "WLF00000035",
                "serv_nm": "서비스명",
                "serv_dgst": "서비스 개요",
                "tgtr_dtl_cn": "수급 대상",
                "slct_crit_cn": "선정 기준",
                "alw_serv_cn": "서비스 내용",
                "sprt_cyc_nm": "월",
                "srv_pvsn_nm": "서비스제공",
                "trgter_indvdl": json.dumps(["저소득"], ensure_ascii=False),
                "intrs_thema": json.dumps(["주거"], ensure_ascii=False),
                "serv_dtl_link": "https://bokjiro.go.kr/",
            }
        )
        assert detail.application_url == "https://bokjiro.go.kr/"
        assert detail.trgter_indvdl == ["저소득"]


class TestWelfareRaw:
    def test_normal_creation(self) -> None:
        raw = WelfareRaw(
            serv_id="WLF00000001",
            serv_nm="복지서비스",
            serv_dgst="개요",
            jur_mnof_nm="보건복지부",
            trgter_indvdl=["저소득", "노인"],
            intrs_thema=["생활지원"],
            sprt_cyc_nm="월",
            srv_pvsn_nm="현금지급",
            serv_dtl_link="https://example.com",
        )
        assert raw.tgtr_dtl_cn == ""
        assert raw.slct_crit_cn == ""
        assert raw.alw_serv_cn == ""

    def test_with_detail_fields(self) -> None:
        raw = WelfareRaw(
            serv_id="WLF00000001",
            serv_nm="복지서비스",
            serv_dgst="개요",
            jur_mnof_nm="보건복지부",
            trgter_indvdl=["저소득"],
            intrs_thema=["주거"],
            sprt_cyc_nm="년",
            srv_pvsn_nm="서비스제공",
            serv_dtl_link="https://example.com",
            tgtr_dtl_cn="수급 대상 상세",
            slct_crit_cn="선정 기준",
            alw_serv_cn="서비스 내용",
        )
        assert raw.tgtr_dtl_cn == "수급 대상 상세"

    @pytest.mark.parametrize("field", ["serv_id", "serv_nm"])
    def test_required_text_fields_reject_blank(self, field: str) -> None:
        payload: dict[str, object] = {
            "serv_id": "WLF00000001",
            "serv_nm": "복지서비스",
            "serv_dgst": "개요",
            "trgter_indvdl": [],
            "intrs_thema": [],
        }
        payload[field] = " "
        with pytest.raises(ValidationError):
            WelfareRaw(**payload)  # type: ignore[arg-type]

    def test_to_metadata_serializes_array_fields(self) -> None:
        raw = WelfareRaw(
            serv_id="WLF00000001",
            serv_nm="복지서비스",
            serv_dgst="개요",
            jur_mnof_nm="보건복지부",
            trgter_indvdl=["저소득", "노인"],
            intrs_thema=["생활지원"],
        )
        meta = raw.to_metadata()
        assert meta["serv_id"] == "WLF00000001"
        assert json.loads(meta["trgter_indvdl"]) == ["저소득", "노인"]
        assert json.loads(meta["intrs_thema"]) == ["생활지원"]

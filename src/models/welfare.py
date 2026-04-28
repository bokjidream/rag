from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class SearchRequest(BaseModel):
    age: int
    income_level: Literal["기초생활수급자", "차상위계층", "저소득", "일반"]
    household_size: int | None = None
    marital_status: Literal["미혼", "기혼", "이혼", "사별"] | None = None
    has_children: bool | None = None
    disability: bool = False
    disability_severity: Literal["경증", "중증"] | None = None
    employment_status: Literal["취업", "실업", "비경제활동"] | None = None
    region: str | None = None
    top_k: int = 5


class SearchResult(BaseModel):
    serv_id: str
    serv_nm: str
    serv_dgst: str
    department: str
    score: float
    trgter_indvdl: list[str]
    intrs_thema: list[str]


class SearchResponse(BaseModel):
    results: list[SearchResult]


class WelfareDetail(BaseModel):
    serv_id: str
    serv_nm: str
    serv_dgst: str
    tgtr_dtl_cn: str
    slct_crit_cn: str
    alw_serv_cn: str
    sprt_cyc_nm: str
    srv_pvsn_nm: str
    trgter_indvdl: list[str]
    intrs_thema: list[str]
    application_url: str
    required_documents: list[str] = []
    application_fields: list[str] = []


class WelfareRaw(BaseModel):
    serv_id: str
    serv_nm: str
    serv_dgst: str
    jur_mnof_nm: str
    trgter_indvdl: list[str]
    intrs_thema: list[str]
    sprt_cyc_nm: str
    srv_pvsn_nm: str
    serv_dtl_link: str
    tgtr_dtl_cn: str = ""
    slct_crit_cn: str = ""
    alw_serv_cn: str = ""

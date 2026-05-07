from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    age: int = Field(ge=0, le=130)
    income_level: Literal["기초생활수급자", "차상위계층", "저소득", "일반"]
    household_size: int | None = Field(default=None, ge=1, le=20)
    marital_status: Literal["미혼", "기혼", "이혼", "사별"] | None = None
    has_children: bool | None = None
    disability: bool = False
    disability_severity: Literal["경증", "중증"] | None = None
    employment_status: Literal["취업", "실업", "비경제활동"] | None = None
    pregnant: bool = False
    region: str | None = Field(default=None, min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)

    @model_validator(mode="after")
    def validate_disability_fields(self) -> SearchRequest:
        if self.disability and self.disability_severity is None:
            raise ValueError("disability=True이면 disability_severity를 지정해야 합니다.")
        if not self.disability and self.disability_severity is not None:
            raise ValueError("disability=False이면 disability_severity를 지정할 수 없습니다.")
        return self


class SearchResult(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    serv_id: str = Field(min_length=1)
    serv_nm: str = Field(min_length=1)
    serv_dgst: str
    department: str
    score: float
    trgter_indvdl: list[str]
    intrs_thema: list[str]

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, str],
        distance: float,
        score: float | None = None,
    ) -> SearchResult:
        return cls(
            serv_id=metadata["serv_id"],
            serv_nm=metadata["serv_nm"],
            serv_dgst=metadata["serv_dgst"],
            department=metadata["jur_mnof_nm"],
            score=max(0.0, 1.0 - distance) if score is None else score,
            trgter_indvdl=json.loads(metadata["trgter_indvdl"]),
            intrs_thema=json.loads(metadata["intrs_thema"]),
        )


class SearchResponse(BaseModel):
    results: list[SearchResult]


class ApplicationForm(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str
    url: str = ""
    file_type: Literal["pdf", "hwp", "hwpx", "etc"] = "etc"


class WelfareDetail(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    serv_id: str = Field(min_length=1)
    serv_nm: str = Field(min_length=1)
    serv_dgst: str
    tgtr_dtl_cn: str = ""
    slct_crit_cn: str = ""
    alw_serv_cn: str = ""
    sprt_cyc_nm: str = ""
    srv_pvsn_nm: str = ""
    trgter_indvdl: list[str]
    intrs_thema: list[str]
    application_url: str = ""
    application_method: str = ""
    application_forms: list[ApplicationForm] = Field(default_factory=list)
    required_documents: list[str] = Field(default_factory=list)

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, str]) -> WelfareDetail:
        return cls(
            serv_id=metadata["serv_id"],
            serv_nm=metadata["serv_nm"],
            serv_dgst=metadata["serv_dgst"],
            tgtr_dtl_cn=metadata["tgtr_dtl_cn"],
            slct_crit_cn=metadata["slct_crit_cn"],
            alw_serv_cn=metadata["alw_serv_cn"],
            sprt_cyc_nm=metadata["sprt_cyc_nm"],
            srv_pvsn_nm=metadata["srv_pvsn_nm"],
            trgter_indvdl=json.loads(metadata["trgter_indvdl"]),
            intrs_thema=json.loads(metadata["intrs_thema"]),
            application_url=metadata["serv_dtl_link"],
            application_method=metadata.get("application_method", ""),
            application_forms=[
                ApplicationForm(**item)
                for item in json.loads(metadata.get("application_forms", "[]"))
            ],
            required_documents=json.loads(metadata.get("required_documents", "[]")),
        )


class WelfareRaw(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    serv_id: str = Field(min_length=1)
    serv_nm: str = Field(min_length=1)
    serv_dgst: str
    jur_mnof_nm: str = ""
    trgter_indvdl: list[str]
    intrs_thema: list[str]
    sprt_cyc_nm: str = ""
    srv_pvsn_nm: str = ""
    serv_dtl_link: str = ""
    tgtr_dtl_cn: str = ""
    slct_crit_cn: str = ""
    alw_serv_cn: str = ""
    application_method: str = ""
    application_forms: list[ApplicationForm] = Field(default_factory=list)
    required_documents: list[str] = Field(default_factory=list)

    def to_metadata(self) -> dict[str, str]:
        return {
            "serv_id": self.serv_id,
            "serv_nm": self.serv_nm,
            "serv_dgst": self.serv_dgst,
            "jur_mnof_nm": self.jur_mnof_nm,
            "trgter_indvdl": json.dumps(self.trgter_indvdl, ensure_ascii=False),
            "intrs_thema": json.dumps(self.intrs_thema, ensure_ascii=False),
            "sprt_cyc_nm": self.sprt_cyc_nm,
            "srv_pvsn_nm": self.srv_pvsn_nm,
            "serv_dtl_link": self.serv_dtl_link,
            "tgtr_dtl_cn": self.tgtr_dtl_cn,
            "slct_crit_cn": self.slct_crit_cn,
            "alw_serv_cn": self.alw_serv_cn,
            "application_method": self.application_method,
            "application_forms": json.dumps(
                [form.model_dump() for form in self.application_forms],
                ensure_ascii=False,
            ),
            "required_documents": json.dumps(self.required_documents, ensure_ascii=False),
        }

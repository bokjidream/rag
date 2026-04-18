"""크롤러 테스트용 fixture — 실제 API 응답을 본뜬 샘플 데이터."""

SAMPLE_LIST_PAGE1 = {
    "wantedList": {
        "totCnt": "2",
        "servList": [
            {"servId": "WLF00000001", "servNm": "기초연금"},
            {"servId": "WLF00000002", "servNm": "장애인연금"},
        ],
    }
}

SAMPLE_DETAIL_001: dict = {
    "wantedDtl": {
        "servId": "WLF00000001",
        "servNm": "기초연금",
        "trgterIndvdlArray": "노인",
        "tgtrDtlCn": "만 65세 이상 어르신",
        "slctCritCn": "소득 하위 70%",
        "alwServCn": "월 최대 334,000원",
        "wlfareInfoOutlCn": "노인 기초연금 지원 서비스",
        "onapPsbltYn": "Y",
        "rprsCtadr": "129",
        "servDtlLink": "https://bokjiro.go.kr/001",
    }
}

SAMPLE_DETAIL_002: dict = {
    "wantedDtl": {
        "servId": "WLF00000002",
        "servNm": "장애인연금",
        "trgterIndvdlArray": "장애인",
        "tgtrDtlCn": "18세 이상 중증장애인",
        "slctCritCn": "소득 하위 70%",
        "alwServCn": "월 최대 403,180원",
        "wlfareInfoOutlCn": "중증장애인 연금 지원 서비스",
        "onapPsbltYn": "Y",
        "rprsCtadr": "129",
        "servDtlLink": "https://bokjiro.go.kr/002",
    }
}

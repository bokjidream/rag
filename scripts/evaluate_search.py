#!/usr/bin/env python3
"""Evaluate search quality against the current persistent Chroma index.

This script uses real indexed welfare services in data/chroma. It intentionally
keeps expected answers as sets because the search API receives a user profile,
not a user intent such as "housing" or "medical benefit".
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
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
from src.models.welfare import SearchRequest  # noqa: E402
from src.retriever.search import (  # noqa: E402
    _profile_boost,
    _rank_score,
    build_query_text,
    search_welfare,
)


@dataclass(frozen=True)
class EvalCase:
    name: str
    request: SearchRequest
    expected_ids: tuple[str, ...]
    notes: str


EVAL_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        "노인_저소득",
        SearchRequest(age=67, income_level="저소득", top_k=10),
        (
            "WLF00001164",  # 기초연금
            "WLF00003191",  # 노인맞춤돌봄
            "WLF00001115",  # 노인 개안수술
            "WLF00001155",  # 노인일자리
            "WLF00001093",  # 독거노인·장애인 응급안전안심서비스
            "WLF00001087",  # 가사·간병 방문 지원사업
        ),
        "노인 돌봄/안전/소득/일자리 관련 서비스 중 하나",
    ),
    EvalCase(
        "노인_기초수급",
        SearchRequest(age=72, income_level="기초생활수급자", top_k=10),
        (
            "WLF00001164",  # 기초연금
            "WLF00003191",  # 노인맞춤돌봄
            "WLF00001155",  # 노인일자리
            "WLF00001132",  # 생계급여
            "WLF00001093",  # 독거노인·장애인 응급안전안심서비스
            "WLF00001087",  # 가사·간병 방문 지원사업
        ),
        "노인/기초생활 관련 서비스",
    ),
    EvalCase(
        "독거노인_저소득_1인가구",
        SearchRequest(age=78, income_level="저소득", household_size=1, top_k=10),
        ("WLF00003191", "WLF00001093", "WLF00001164"),
        "노인맞춤돌봄, 응급안전안심, 기초연금",
    ),
    EvalCase(
        "청년_저소득",
        SearchRequest(age=24, income_level="저소득", top_k=10),
        ("WLF00004661", "WLF00000060", "WLF00003245", "WLF00003266"),
        "청년월세, 청년내일저축, 국민취업지원, 직업훈련생계비",
    ),
    EvalCase(
        "청년_일반_실업",
        SearchRequest(age=25, income_level="일반", employment_status="실업", top_k=10),
        ("WLF00003245", "WLF00003239", "WLF00001172", "WLF00006215"),
        "국민취업지원, 고용복지플러스, 취업취약계층 지원, 청년내일채움공제",
    ),
    EvalCase(
        "청년_저소득_실업",
        SearchRequest(age=27, income_level="저소득", employment_status="실업", top_k=10),
        ("WLF00003245", "WLF00003266", "WLF00000060", "WLF00004661"),
        "저소득 청년 고용/자산/주거 관련 서비스",
    ),
    EvalCase(
        "중장년_일반_실업",
        SearchRequest(age=45, income_level="일반", employment_status="실업", top_k=10),
        ("WLF00003245", "WLF00003239", "WLF00001172", "WLF00003266"),
        "고용/직업훈련 관련 서비스",
    ),
    EvalCase(
        "기초수급_실업_근로능력",
        SearchRequest(age=42, income_level="기초생활수급자", employment_status="실업", top_k=10),
        ("WLF00001138", "WLF00001157", "WLF00003245", "WLF00001132"),
        "자활근로, 지역자활센터, 국민취업지원, 생계급여",
    ),
    EvalCase(
        "중증장애인_저소득",
        SearchRequest(
            age=40,
            income_level="저소득",
            disability=True,
            disability_severity="중증",
            top_k=10,
        ),
        ("WLF00003260", "WLF00003249", "WLF00003265", "WLF00003211"),
        "장애인활동지원, 장애인연금, 장애수당, 보조기기",
    ),
    EvalCase(
        "중증장애인_일자리",
        SearchRequest(
            age=35,
            income_level="저소득",
            disability=True,
            disability_severity="중증",
            employment_status="실업",
            top_k=10,
        ),
        ("WLF00000025", "WLF00003189", "WLF00001090", "WLF00003260"),
        "장애인일자리, 직업재활, 고용장려금, 활동지원",
    ),
    EvalCase(
        "한부모_이혼_저소득_자녀",
        SearchRequest(
            age=35,
            income_level="저소득",
            marital_status="이혼",
            has_children=True,
            top_k=10,
        ),
        ("WLF00001068", "WLF00001104", "WLF00000024", "WLF00001103"),
        "아동양육비, 자녀교육비, 아이돌봄, 초중고 교육비",
    ),
    EvalCase(
        "한부모_사별_저소득_자녀",
        SearchRequest(
            age=42,
            income_level="저소득",
            marital_status="사별",
            has_children=True,
            top_k=10,
        ),
        ("WLF00001068", "WLF00001104", "WLF00000024", "WLF00001103"),
        "아동양육비, 자녀교육비, 아이돌봄, 초중고 교육비",
    ),
    EvalCase(
        "영유아_자녀_일반",
        SearchRequest(age=32, income_level="일반", has_children=True, top_k=10),
        ("WLF00000024", "WLF00003250", "WLF00003253", "WLF00000969"),
        "아이돌봄, 영유아보육료, 가정양육수당, 유아학비",
    ),
    EvalCase(
        "기초수급_중년",
        SearchRequest(age=50, income_level="기초생활수급자", top_k=10),
        ("WLF00001132", "WLF00003201", "WLF00001089", "WLF00000074"),
        "생계급여, 주거급여, 교육급여, 양곡할인",
    ),
    EvalCase(
        "저소득_중년",
        SearchRequest(age=50, income_level="저소득", top_k=10),
        (
            "WLF00001132",
            "WLF00003201",
            "WLF00000072",
            "WLF00000055",
            "WLF00000074",
            "WLF00003257",
        ),
        "생활지원/주거/에너지/문화/양곡 관련 서비스",
    ),
    EvalCase(
        "차상위_중년",
        SearchRequest(age=50, income_level="차상위계층", top_k=10),
        ("WLF00001087", "WLF00001093", "WLF00001138", "WLF00001089"),
        "차상위 조건이 포함된 생활/돌봄/자활/교육 서비스",
    ),
    EvalCase(
        "저소득_주거취약",
        SearchRequest(age=45, income_level="저소득", household_size=1, top_k=10),
        ("WLF00003201", "WLF00000917", "WLF00000036", "WLF00000062", "WLF00003269"),
        "주거급여, 긴급복지 주거, 주거상향, 매입/전세임대",
    ),
    EvalCase(
        "저소득_임산부",
        SearchRequest(age=30, income_level="저소득", marital_status="기혼", pregnant=True, top_k=10),
        ("WLF00001088", "WLF00000061", "WLF00001135", "WLF00003178"),
        "고위험 임산부, 임신출산 진료비, 해산급여, 긴급복지 해산비",
    ),
    EvalCase(
        "저소득_청소년",
        SearchRequest(age=16, income_level="저소득", top_k=10),
        ("WLF00000078", "WLF00000948", "WLF00000781", "WLF00001107"),
        "청소년특별지원, 학교밖청소년, 생리용품, 지역아동센터",
    ),
    EvalCase(
        "아동_저소득_자녀",
        SearchRequest(age=8, income_level="저소득", has_children=True, top_k=10),
        ("WLF00001103", "WLF00001089", "WLF00001107", "WLF00000024", "WLF00003258"),
        "교육비, 교육급여, 지역아동센터, 아이돌봄, 아동발달지원계좌",
    ),
)


async def _diagnose_case(case: EvalCase, embedder: KoSimCSEEmbedder) -> None:
    """실패 케이스의 raw distance rank → boost → final rank 흐름 출력."""
    query_text = build_query_text(case.request)
    vec: list[float] = embedder.embed([query_text])[0]

    collection = await get_collection(WELFARE_COLLECTION)

    def _query() -> Any:
        return collection.query(
            query_embeddings=[vec],  # type: ignore[arg-type]
            n_results=min(case.request.top_k * 5, 100),
        )

    raw: Any = await asyncio.to_thread(_query)
    distances: list[float] = raw["distances"][0]
    metadatas: list[dict[str, str]] = raw["metadatas"][0]

    # raw 순위별 (serv_id당 최고 섹션 청크) 집계
    raw_best: dict[str, tuple[int, dict[str, str], float, float, float]] = {}
    for raw_rank, (meta, dist) in enumerate(zip(metadatas, distances), start=1):
        serv_id = meta["serv_id"]
        boost = _profile_boost(case.request, meta)
        score = _rank_score(case.request, meta, dist)
        current = raw_best.get(serv_id)
        if current is None or score > current[4]:
            raw_best[serv_id] = (raw_rank, meta, dist, boost, score)

    ranked = sorted(raw_best.values(), key=lambda x: x[4], reverse=True)
    expected = set(case.expected_ids)

    print(f"\nDIAGNOSE {case.name}  query={query_text}")
    print(f"  {'#':>3}  {'raw':>4}  {'dist':>6}  {'boost':>6}  {'score':>6}  serv_id          serv_nm")
    for final_rank, (raw_rank, meta, dist, boost, score) in enumerate(ranked[:15], start=1):
        marker = "* " if meta["serv_id"] in expected else "  "
        print(
            f"  {marker}{final_rank:>2}.  raw={raw_rank:>3}  "
            f"dist={dist:.3f}  boost={boost:+.3f}  score={score:.3f}  "
            f"{meta['serv_id']}  {meta['serv_nm'][:28]}"
        )


def _rank_of(result_ids: list[str], expected_ids: tuple[str, ...]) -> int | None:
    expected = set(expected_ids)
    for idx, serv_id in enumerate(result_ids, start=1):
        if serv_id in expected:
            return idx
    return None


async def _service_count() -> tuple[int, int]:
    collection = await get_collection(WELFARE_COLLECTION)
    raw = collection.get(include=["metadatas"])
    service_ids = {
        meta["serv_id"]
        for meta in raw["metadatas"]
        if meta is not None and isinstance(meta.get("serv_id"), str)
    }
    return len(raw["ids"]), len(service_ids)


async def evaluate(verbose: bool, diagnose: bool = False) -> int:
    chunk_count, service_count = await _service_count()
    embedder = KoSimCSEEmbedder()

    ranks: list[int | None] = []
    rows: list[tuple[EvalCase, int | None, list[tuple[str, str, float]]]] = []

    for case in EVAL_CASES:
        response = await search_welfare(case.request, embedder)
        results = [(r.serv_id, r.serv_nm, r.score) for r in response.results]
        rank = _rank_of([serv_id for serv_id, _, _ in results], case.expected_ids)
        ranks.append(rank)
        rows.append((case, rank, results))

    total = len(ranks)

    def recall_at(k: int) -> float:
        return sum(rank is not None and rank <= k for rank in ranks) / total

    mrr = sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / total

    print("Search Quality Evaluation")
    print(f"index: {chunk_count} chunks / {service_count} services")
    print(f"cases: {total}")
    print(f"recall@1: {recall_at(1):.3f}")
    print(f"recall@3: {recall_at(3):.3f}")
    print(f"recall@5: {recall_at(5):.3f}")
    print(f"recall@10: {recall_at(10):.3f}")
    print(f"mrr: {mrr:.3f}")
    print()

    for case, rank, results in rows:
        status = "PASS" if rank is not None and rank <= 5 else "FAIL"
        rank_text = "-" if rank is None else str(rank)
        print(f"{status} {case.name}: rank={rank_text} expected={','.join(case.expected_ids)}")
        if verbose or status == "FAIL":
            print(f"  query: {build_query_text(case.request)}")
            print(f"  notes: {case.notes}")
            for pos, (serv_id, serv_nm, score) in enumerate(results[:10], start=1):
                marker = "*" if serv_id in case.expected_ids else " "
                print(f"  {marker}{pos:>2}. {serv_id} {serv_nm} score={score:.3f}")
        if diagnose and status == "FAIL":
            await _diagnose_case(case, embedder)
    return 0 if recall_at(5) >= 0.8 else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Print top results for every case")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="For FAIL cases: show raw distance rank, boost, and final rank side-by-side",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(evaluate(verbose=args.verbose, diagnose=args.diagnose)))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a section-aware Chroma index from the existing baseline corpus."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import chromadb

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.chroma import WELFARE_COLLECTION  # noqa: E402
from src.embedding.kosimcse import KoSimCSEEmbedder  # noqa: E402
from src.pipeline.chunker import chunk_metadata_sections  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_PERSIST_DIR = REPO_ROOT / "data/chroma"
DEFAULT_TARGET_PERSIST_DIR = REPO_ROOT / "data/chroma-section-aware"
SECTION_AWARE_COLLECTION = "welfare_services_section_aware"

PRESERVED_METADATA_FIELDS = (
    "serv_id",
    "serv_nm",
    "serv_dgst",
    "jur_mnof_nm",
    "trgter_indvdl",
    "intrs_thema",
    "tgtr_dtl_cn",
    "slct_crit_cn",
    "alw_serv_cn",
    "sprt_cyc_nm",
    "srv_pvsn_nm",
    "serv_dtl_link",
    "application_method",
    "application_forms",
    "required_documents",
)
JSON_ARRAY_FIELDS = {
    "trgter_indvdl",
    "intrs_thema",
    "application_forms",
    "required_documents",
}
BATCH_SIZE = 128


def _metadata_value(
    string_value: str | None,
    int_value: int | None,
    float_value: float | None,
    bool_value: int | None,
) -> str:
    if string_value is not None:
        return string_value
    if int_value is not None:
        return str(int_value)
    if float_value is not None:
        return str(float_value)
    if bool_value is not None:
        return "true" if bool_value else "false"
    return ""


def _read_baseline_metadata(
    persist_dir: Path,
    collection_name: str,
) -> tuple[int, dict[str, dict[str, str]]]:
    db_path = persist_dir / "chroma.sqlite3"
    if not db_path.exists():
        raise FileNotFoundError(f"baseline sqlite DB not found: {db_path}")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            """
            SELECT
                e.embedding_id,
                m.key,
                m.string_value,
                m.int_value,
                m.float_value,
                m.bool_value
            FROM embeddings e
            JOIN embedding_metadata m ON m.id = e.id
            JOIN segments s ON s.id = e.segment_id
            JOIN collections c ON c.id = s.collection
            WHERE c.name = ?
            """,
            (collection_name,),
        ).fetchall()
    finally:
        conn.close()

    by_embedding_id: dict[str, dict[str, str]] = {}
    for embedding_id, key, string_value, int_value, float_value, bool_value in rows:
        if key == "chroma:document":
            continue
        by_embedding_id.setdefault(embedding_id, {})[key] = _metadata_value(
            string_value,
            int_value,
            float_value,
            bool_value,
        )

    by_serv_id: dict[str, dict[str, str]] = {}
    for metadata in by_embedding_id.values():
        serv_id = metadata.get("serv_id", "")
        if not serv_id:
            continue
        current = by_serv_id.get(serv_id)
        if current is None or _metadata_richness(metadata) > _metadata_richness(current):
            by_serv_id[serv_id] = _normalize_metadata(metadata)

    return len(by_embedding_id), by_serv_id


def _metadata_richness(metadata: dict[str, str]) -> int:
    return sum(1 for field in PRESERVED_METADATA_FIELDS if metadata.get(field))


def _normalize_metadata(metadata: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for field in PRESERVED_METADATA_FIELDS:
        default = "[]" if field in JSON_ARRAY_FIELDS else ""
        normalized[field] = metadata.get(field, default)
    return normalized


def _assert_safe_section_target(persist_dir: Path, collection_name: str) -> None:
    resolved_target = persist_dir.resolve()
    if resolved_target == DEFAULT_SOURCE_PERSIST_DIR.resolve():
        raise SystemExit(
            "Refusing to write section-aware index into baseline data/chroma directory."
        )
    if collection_name == WELFARE_COLLECTION:
        raise SystemExit("Refusing to write section-aware index into welfare_services collection.")


def _get_existing_ids(collection: chromadb.Collection) -> list[str]:
    raw = collection.get(include=[])
    ids = raw.get("ids", [])
    return list(ids)


def build_section_aware_index(
    source_persist_dir: Path,
    source_collection: str,
    target_persist_dir: Path,
    target_collection: str,
) -> int:
    _assert_safe_section_target(target_persist_dir, target_collection)

    baseline_chunk_count, services = _read_baseline_metadata(source_persist_dir, source_collection)
    if not services:
        raise SystemExit(
            f"No services found in {source_persist_dir} collection {source_collection!r}."
        )

    embedder = KoSimCSEEmbedder()

    all_ids: list[str] = []
    all_docs: list[str] = []
    all_metas: list[dict[str, str | int]] = []
    section_counts: Counter[str] = Counter()

    for serv_id in sorted(services):
        metadata = services[serv_id]
        chunks = chunk_metadata_sections(metadata, tokenizer=embedder.tokenizer)
        for n, chunk in enumerate(chunks):
            all_ids.append(f"{serv_id}_{chunk.section}_{n}")
            all_docs.append(chunk.text)
            all_metas.append(
                {
                    **metadata,
                    "chunk_section": chunk.section,
                    "chunk_token_count": chunk.token_count,
                }
            )
            section_counts[chunk.section] += 1

    client = chromadb.PersistentClient(str(target_persist_dir))
    collection = client.get_or_create_collection(
        target_collection,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids = _get_existing_ids(collection)
    next_id_set = set(all_ids)
    stale_ids = sorted(item_id for item_id in existing_ids if item_id not in next_id_set)
    if stale_ids:
        collection.delete(ids=stale_ids)

    for start in range(0, len(all_ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_docs = all_docs[start:end]
        embeddings = embedder.embed(batch_docs)
        collection.upsert(
            ids=all_ids[start:end],
            documents=batch_docs,
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=all_metas[start:end],  # type: ignore[arg-type]
        )

    print("Section-Aware Index Build")
    print(f"source: {source_persist_dir} / {source_collection}")
    print(f"target: {target_persist_dir} / {target_collection}")
    print(f"baseline: {baseline_chunk_count} chunks / {len(services)} services")
    print(f"section_aware: {len(all_ids)} chunks / {len(services)} services")
    print("section_counts:")
    for section, count in sorted(section_counts.items()):
        print(f"  {section}: {count}")

    return len(all_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-chroma-persist-dir",
        default=str(DEFAULT_SOURCE_PERSIST_DIR),
        help="Baseline Chroma persist directory to read in sqlite read-only mode.",
    )
    parser.add_argument(
        "--source-collection",
        default=WELFARE_COLLECTION,
        help="Baseline collection name to read.",
    )
    parser.add_argument(
        "--chroma-persist-dir",
        default=str(DEFAULT_TARGET_PERSIST_DIR),
        help="Target Chroma persist directory for the section-aware index.",
    )
    parser.add_argument(
        "--collection",
        default=SECTION_AWARE_COLLECTION,
        help="Target section-aware collection name.",
    )
    args = parser.parse_args()

    build_section_aware_index(
        source_persist_dir=Path(args.source_chroma_persist_dir),
        source_collection=args.source_collection,
        target_persist_dir=Path(args.chroma_persist_dir),
        target_collection=args.collection,
    )


if __name__ == "__main__":
    main()

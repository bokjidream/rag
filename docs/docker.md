# Docker deployment

The API container can use the section-aware Chroma collection through runtime
environment variables. The section-aware database is not copied into the image
and should not be committed. Mount the database directory from the host or
server at compose runtime.

## Compose defaults

`docker-compose.yml` sets:

- `CHROMA_MODE=persistent`
- `CHROMA_PERSIST_DIR=/app/data/chroma-section-aware`
- `WELFARE_COLLECTION_NAME=welfare_services_section_aware`
- `${CHROMA_HOST_DIR:-./data/chroma-section-aware}:/app/data/chroma-section-aware`

The API resolves `WELFARE_COLLECTION_NAME` once at startup, validates that the
collection exists and is not empty, and uses the same collection name for both
`/welfare/search` and `/welfare/{serv_id}`. For non-baseline collections, startup
validation also checks sampled metadata for `chunk_section`.

`WELFARE_ADAPTIVE_FETCH` is optional. If it is unset, adaptive fetch is enabled
when `WELFARE_COLLECTION_NAME` is not the baseline `welfare_services`
collection. If set, accepted values are `true` and `false` after trimming and
lowercasing. Any other value fails startup configuration.

## Run

Build and run with the default local section-aware DB mount:

```sh
docker compose build
docker compose up
```

To mount a different database directory:

```sh
CHROMA_HOST_DIR=/srv/bokjidream/chroma-section-aware docker compose up
```

## Smoke test with a DB copy

Chroma can update SQLite runtime metadata while the container is running. Use a
temporary copy for smoke tests instead of the original DB directory:

```sh
tmp_dir="$(mktemp -d)"
cp -a data/chroma-section-aware "$tmp_dir/chroma-section-aware"

CHROMA_HOST_DIR="$tmp_dir/chroma-section-aware" docker compose up -d

search_body="$(curl -fsS -X POST "http://localhost:8000/welfare/search" \
  -H "Content-Type: application/json" \
  -d '{"age":65,"income_level":"저소득","top_k":3}')"

serv_id="$(printf '%s' "$search_body" | python3 -c 'import json,sys; body=json.load(sys.stdin); assert len(body["results"]) > 0; print(body["results"][0]["serv_id"])')"

curl -fsS "http://localhost:8000/welfare/$serv_id" >/dev/null

CHROMA_HOST_DIR="$tmp_dir/chroma-section-aware" docker compose down
```

The search response should return HTTP 200 with at least one result, and the
detail request should return HTTP 200 for the returned `serv_id`.

## Embedding model download

The Dockerfile installs CPU-only PyTorch wheels during build and defaults to
runtime embedding-model download. On the first container start,
`sentence-transformers` downloads `jhgan/ko-sroberta-multitask` into
`/app/.cache`. This requires outbound network access to Hugging Face or a
pre-populated cache mounted at `/app/.cache`.

For deployments without runtime network access, prefetch the model during image
build:

```sh
docker compose build --build-arg PRELOAD_EMBEDDING_MODEL=true
```

That build needs network access, increases the image size, and stores the model
cache inside the image. If you instead mount a cache volume at `/app/.cache`,
populate that volume before starting an offline container.

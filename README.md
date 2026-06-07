# rag

## ChromaDB seed data

`data/chroma` is committed as the local seed database so the RAG API can run with
prebuilt welfare-service embeddings.

- Commit `data/chroma` changes only in intentional data-update PRs.
- Do not commit local `data/chroma/chroma.sqlite3` changes caused by ordinary
  development or test runs.
- If only ChromaDB internal runtime state changed, such as the `acquire_write`
  table, treat it as a local artifact and restore it before committing.
- SQLite runtime sidecar files (`*.sqlite3-wal`, `*.sqlite3-shm`,
  `*.sqlite3-journal`) are ignored.

## Docker

See [docs/docker.md](docs/docker.md) for Docker Compose deployment, section-aware
collection configuration, and smoke-test commands.

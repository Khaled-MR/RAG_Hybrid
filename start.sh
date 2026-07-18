#!/usr/bin/env bash
# One-command start for the full RAG stack (Linux / macOS / Git-Bash on Windows).
#   ./start.sh            # build (if needed), start everything, then ingest data/
#   ./start.sh --no-ingest
set -e
cd "$(dirname "$0")"

[ -f .env ] || cp .env.example .env

echo "==> Building & starting the stack (first build downloads ~torch/CUDA, be patient)..."
# BuildKit can drop large downloads on some networks (WSL2 MTU). If the build
# fails on an SSL/network error, retry with the legacy builder:
#   DOCKER_BUILDKIT=0 docker compose build
docker compose up -d --build

echo "==> Waiting for the backend to be healthy..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:8000/api/health >/dev/null 2>&1; then echo "backend up."; break; fi
  sleep 5
done

if [ "$1" != "--no-ingest" ]; then
  echo "==> Ingesting everything under backend/data/ ..."
  docker compose exec -T backend python ingest.py || \
    echo "(ingest skipped/failed — add files to backend/data/ and run: docker compose exec backend python ingest.py)"
fi

echo ""
echo "Done. Open  http://localhost:3000"

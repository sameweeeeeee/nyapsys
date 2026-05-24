#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_DIR/.env"

mkdir -p ~/volumes/models ~/volumes/chromadb ~/volumes/sqlite "$PROJECT_DIR/logs"

echo "Starting llama-server..."
llama-server \
  --model ~/volumes/models/Nyapsys-2B-MoE.Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 4096 \
  --n-predict 2048 \
  --threads 4 \
  --parallel 2 \
  --flash-attn \
  --mlock \
  --cont-batching &

LLAMA_PID=$!
echo "Waiting for llama-server..."
until curl -sf http://127.0.0.1:8080/health > /dev/null 2>&1; do sleep 2; done
echo "llama-server ready"

sleep 2

echo "Starting ChromaDB..."
python3.11 -m chromadb.cli.cli run \
  --path ~/volumes/chromadb \
  --host 127.0.0.1 \
  --port 8001 &

CHROMA_PID=$!
sleep 3
echo "ChromaDB ready"

echo "Starting FastAPI..."
cd "$PROJECT_DIR/backend"
python3.11 -m uvicorn app.main:app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1

kill $LLAMA_PID $CHROMA_PID 2>/dev/null || true

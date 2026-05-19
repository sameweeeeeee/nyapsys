#!/bin/bash
set -e
source ~/nyapsys/.env

mkdir -p ~/volumes/models ~/volumes/chromadb ~/volumes/sqlite ~/nyapsys/logs

llama-server --model ~/volumes/models/Nyapsys-1B.Q4_K_M.gguf --host 127.0.0.1 --port 8080 --ctx-size 4096 --n-predict 2048 --threads 4 --parallel 2 --cont-batching &
LLAMA_PID=$!

echo "Waiting for llama-server..."
until curl -sf http://127.0.0.1:8080/health > /dev/null 2>&1; do sleep 2; done

python3.11 -m chromadb.cli.cli run --path ~/volumes/chromadb --host 127.0.0.1 --port 8001 &
CHROMA_PID=$!
sleep 3

cd ~/nyapsys/backend
python3.11 -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1

kill $LLAMA_PID $CHROMA_PID 2>/dev/null || true
#!/bin/bash
set -e
source ~/nyapsys/.env

MODEL_DIR=~/volumes/models

echo "=== Downloading model ==="
[ ! -f "$MODEL_DIR/Nyapsys-1B.Q4_K_M.gguf" ] && gsutil cp gs://$GCS_BUCKET/Nyapsys-1B.Q4_K_M.gguf $MODEL_DIR/ || echo "Already present"
ls -lh $MODEL_DIR/
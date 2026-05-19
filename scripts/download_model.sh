#!/bin/bash
set -e
source /opt/nyapsys/.env

MODEL_DIR=/volumes/models

echo "=== Downloading model files ==="

if [ ! -f "$MODEL_DIR/Nyapsys-11B-Vision.Q4_K_M.gguf" ]; then
  echo "Downloading Nyapsys GGUF from DO Spaces..."
  AWS_ACCESS_KEY_ID=$DO_SPACES_KEY \
  AWS_SECRET_ACCESS_KEY=$DO_SPACES_SECRET \
  aws s3 cp s3://$DO_SPACES_BUCKET/Nyapsys-11B-Vision.Q4_K_M.gguf \
    $MODEL_DIR/ \
    --endpoint-url $DO_SPACES_ENDPOINT
else
  echo "Main GGUF already present — skipping"
fi

if [ ! -f "$MODEL_DIR/mmproj-model-f16.gguf" ]; then
  echo "Downloading multimodal projector weights..."
  wget -q --show-progress \
    -O $MODEL_DIR/mmproj-model-f16.gguf \
    "https://huggingface.co/leafspark/Llama-3.2-11B-Vision-Instruct-GGUF/resolve/main/mmproj-model-f16.gguf"
else
  echo "mmproj already present — skipping"
fi

echo "=== Model files ready ==="
ls -lh $MODEL_DIR/
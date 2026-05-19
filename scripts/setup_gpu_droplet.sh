#!/bin/bash
set -e

echo "=== Nyapsys GPU droplet setup ==="

pip install unsloth trl transformers peft datasets boto3 bitsandbytes accelerate

echo "Cloning llama.cpp..."
git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp
cd /opt/llama.cpp && make -j$(nproc)

echo "Pulling dataset from DO Spaces..."
aws s3 sync s3://nyapsys-training/dataset/ ./data/prepared/ \
  --endpoint-url https://nyc3.digitaloceanspaces.com

echo "Cloning nyapsys repo..."
git clone https://github.com/sameweeeeeee/nyapsys /opt/nyapsys

echo "Verifying GPU..."
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo "=== GPU droplet ready ==="
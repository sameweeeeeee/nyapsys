#!/bin/bash
set -e
echo "=== GCP L4 GPU setup ==="

apt-get update && apt-get install -y git curl

pip install google-cloud-sdk

pip install torch transformers trl peft datasets tokenizers accelerate bitsandbytes scipy

git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp
cd /opt/llama.cpp && make -j$(nproc)

git clone https://github.com/YOUR_USER/nyapsys /opt/nyapsys

nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

echo "=== GPU ready ==="
#!/bin/bash
set -e
echo "=== GCP L4 GPU setup ==="

apt-get update && apt-get install -y git curl

git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp
cd /opt/llama.cpp && make -j$(nproc)

git clone https://github.com/YOUR_USERNAME/nyapsys /opt/nyapsys

cd /opt/nyapsys
pip install -r training/requirements.txt

nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

echo "=== GPU ready ==="
# Nyapsys Fine-Tuning Pipeline

This pipeline fine-tunes Llama 3.2 11B Vision Instruct into Nyapsys.

## Prerequisites

- Mac M3 (or any Mac) for data preparation
- DigitalOcean GPU droplet (RTX 6000 Ada) for training
- DO Spaces bucket for dataset/model storage

## Step 1: Prepare Dataset (Mac M3)

```bash
cd training
pip install -r requirements.txt
python prepare_dataset.py
```

Output:
- `training/data/prepared/train.jsonl` (~45k examples)
- `training/data/prepared/eval.jsonl` (~5k examples)

Upload to DO Spaces:
```bash
python upload_to_spaces.py --source data/prepared/ --prefix dataset/
```

## Step 2: Create GPU Droplet

```bash
doctl compute droplet create nyapsys-gpu-training \
  --size gpu-rtx6000ada-1 \
  --image gpu-h100x1-base \
  --region nyc2 \
  --ssh-keys YOUR_SSH_KEY_ID \
  --wait
```

Get IP:
```bash
doctl compute droplet get nyapsys-gpu-training --format PublicIPv4
```

SSH in:
```bash
ssh root@GPU_DROPLET_IP
```

## Step 3: Setup GPU Droplet

On the GPU droplet, run:
```bash
bash /opt/nyapsys/scripts/setup_gpu_droplet.sh
```

Or manually:
```bash
# Install training deps
pip install unsloth trl transformers peft datasets boto3 bitsandbytes accelerate

# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp
cd /opt/llama.cpp && make -j$(nproc)

# Pull dataset from Spaces
aws s3 sync s3://nyapsys-training/dataset/ ./data/prepared/ \
  --endpoint-url https://nyc3.digitaloceanspaces.com

# Clone repo
git clone https://github.com/YOUR_USERNAME/nyapsys /opt/nyapsys

# Verify GPU
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Step 4: Run Training

```bash
cd /opt/nyapsys
python training/train.py
```

Expected:
- Training time: 10-15 hours
- Final eval loss: < 1.2

## Step 5: Merge and Export

```bash
python training/merge_and_export.py
```

This creates `output/Nyapsys-11B-Vision.Q4_K_M.gguf` (~5.96 GB).

## Step 6: Upload Model

```bash
python training/upload_to_spaces.py \
  --file ./output/Nyapsys-11B-Vision.Q4_K_M.gguf \
  --prefix models/
```

## Step 7: Destroy GPU Droplet

```bash
doctl compute droplet delete nyapsys-gpu-training --force
echo "Billing stopped"
```

## Dataset Composition

| Dataset | Purpose | Mix |
|---------|---------|-----|
| Alpaca Cleaned | Instruction following | 30% |
| SQuAD v2 | Reading comprehension | 25% |
| LLaVA Instruct | Vision + image Q&A | 25% |
| LMSYS Chat 1M | Multi-turn conversations | 15% |
| Custom JSONL | Nyapsys persona | 5% |

## Troubleshooting

- **OOM**: Reduce batch size, enable gradient checkpointing
- **Slow training**: Check GPU utilization with `nvidia-smi`
- **Loss too high**: Dataset may need cleaning, try reducing learning rate
# Nyapsys Training Pipeline

2B MoE from-scratch model (4 experts × 500M, top-2 routing = 1B active per token).

## Step 1: Prepare Data (Mac M3)

```bash
cd training
pip install -r requirements.txt
python prepare_dataset.py
gsutil -m cp -r training/data/tokenized/ gs://nyapsys-training/tokenized/
```

## Step 2: GCP GPU Setup

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Request L4 GPU quota first (IAM & Admin → Quotas)

gcloud compute instances create nyapsys-training \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=pytorch-llatest-gpu \
  --image-project=deeplearning-platform-release \
  --provisioning-model=SPOT

gcloud compute ssh nyapsys-training --zone=us-central1-a

git clone https://github.com/YOUR_USER/nyapsys /opt/nyapsys
pip install -r training/requirements.txt
gsutil -m cp -r gs://nyapsys-training/tokenized/ training/data/tokenized/
```

## Step 3: Pretrain

```bash
cd /opt/nyapsys
python training/train.py \
  --config 2b_moe \
  --data_path gs://nyapsys-training/tokenized/ \
  --output_dir gs://nyapsys-training/checkpoints/
```

~70-90 hrs on L4 (~$28-36 spot).

## Step 4: Instruction Tune

```bash
python training/instruction_tune.py \
  --base_model gs://nyapsys-training/checkpoints/final \
  --output_dir gs://nyapsys-training/instruct-output \
  --lora_r 32 --lora_alpha 64 --epochs 3
```

## Step 5: Export GGUF

```bash
python training/merge_and_export.py \
  --model_path gs://nyapsys-training/instruct-output \
  --output_path ./Nyapsys-2B-MoE.Q4_K_M.gguf

python training/upload_to_gcs.py --file ./Nyapsys-2B-MoE.Q4_K_M.gguf --bucket nyapsys-models
```

## Step 6: Destroy Instance

```bash
gcloud compute instances delete nyapsys-training --zone=us-central1-a --quiet
```

Total: ~$37-48 (covered by $300 GCP credit, ~$252-263 remaining for retrains)